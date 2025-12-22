/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fused_transform_aggregate.hpp"
#include "fused_transform_aggregate_kernels.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <thrust/sequence.h>

#include <chrono>
#include <vector>

namespace spark_rapids_jni {

namespace {

constexpr size_t BUFFER_ALIGNMENT = 8;
constexpr int BLOCK_SIZE = 256;

/**
 * @brief Round up to the nearest multiple of alignment
 */
constexpr size_t align_size(size_t size, size_t alignment = BUFFER_ALIGNMENT)
{
  return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Custom memory resource that allocates from a pre-existing buffer
 */
class contiguous_buffer_resource final : public rmm::mr::device_memory_resource {
 public:
  contiguous_buffer_resource(void* base, std::size_t size)
      : base_(static_cast<uint8_t*>(base)), size_(size), offset_(0) {}

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override {
    std::size_t aligned_offset = align_size(offset_, BUFFER_ALIGNMENT);
    CUDF_EXPECTS(aligned_offset + bytes <= size_, 
                 "Contiguous buffer exhausted");
    void* ptr = base_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
  }

  void do_deallocate(void*, std::size_t, rmm::cuda_stream_view) noexcept override {
    // No-op: memory is freed when the parent buffer is destroyed
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override {
    return this == &other;
  }

 private:
  uint8_t* base_;
  std::size_t size_;
  std::size_t offset_;
};

/**
 * @brief Convert FusedExprSpec to DeviceExprSpec
 */
detail::DeviceExprSpec to_device_spec(FusedExprSpec const& spec) {
  detail::DeviceExprSpec d_spec{};
  d_spec.transform_op = static_cast<int32_t>(spec.transform_op);
  d_spec.value_col_idx = spec.value_col_idx;
  d_spec.cond_col_idx = spec.cond_col_idx;
  d_spec.other_col_idx = spec.other_col_idx;
  d_spec.default_val = spec.default_val;
  d_spec.threshold = spec.threshold;
  d_spec.else_val = spec.else_val;
  d_spec.agg_op = static_cast<int32_t>(spec.agg_op);
  d_spec.output_type_id = static_cast<int32_t>(spec.output_type.id());
  d_spec.output_idx = spec.output_idx;
  return d_spec;
}

}  // namespace

// ============================================================================
// STRING HELPERS
// ============================================================================

std::string transform_op_name(TransformOp op) {
  switch (op) {
    case TransformOp::IDENTITY: return "IDENTITY";
    case TransformOp::COALESCE: return "COALESCE";
    case TransformOp::COALESCE_MUL_SELF: return "COALESCE_MUL_SELF";
    case TransformOp::COALESCE_MUL_OTHER: return "COALESCE_MUL_OTHER";
    case TransformOp::CONDITIONAL: return "CONDITIONAL";
    case TransformOp::CONDITIONAL_COALESCE: return "CONDITIONAL_COALESCE";
    default: return "UNKNOWN";
  }
}

std::string agg_op_name(AggOp op) {
  switch (op) {
    case AggOp::SUM: return "SUM";
    case AggOp::COUNT: return "COUNT";
    case AggOp::AVG: return "AVG";
    case AggOp::MIN: return "MIN";
    case AggOp::MAX: return "MAX";
    default: return "UNKNOWN";
  }
}

// ============================================================================
// VALIDATION
// ============================================================================

bool can_fuse(FusedExecutionPlan const& plan, std::string* reason) {
  if (plan.expressions.empty()) {
    if (reason) *reason = "No expressions to fuse";
    return false;
  }
  
  if (plan.group_by_col_indices.empty()) {
    if (reason) *reason = "No group-by columns specified";
    return false;
  }
  
  for (size_t i = 0; i < plan.expressions.size(); ++i) {
    auto const& expr = plan.expressions[i];
    if (expr.value_col_idx < 0) {
      if (reason) *reason = "Expression " + std::to_string(i) + " has invalid value column index";
      return false;
    }
    
    // Validate CONDITIONAL expressions have condition column
    if (expr.transform_op == TransformOp::CONDITIONAL || 
        expr.transform_op == TransformOp::CONDITIONAL_COALESCE) {
      if (expr.cond_col_idx < 0) {
        if (reason) *reason = "Expression " + std::to_string(i) + " (CONDITIONAL) missing condition column";
        return false;
      }
    }
    
    // Validate COALESCE_MUL_OTHER has other column
    if (expr.transform_op == TransformOp::COALESCE_MUL_OTHER) {
      if (expr.other_col_idx < 0) {
        if (reason) *reason = "Expression " + std::to_string(i) + " (COALESCE_MUL_OTHER) missing other column";
        return false;
      }
    }
  }
  
  return true;
}

// ============================================================================
// BUFFER SIZE ESTIMATION
// ============================================================================

size_t estimate_output_buffer_size(
    FusedExecutionPlan const& plan,
    cudf::size_type estimated_num_groups)
{
  size_t total = 0;
  
  for (auto const& expr : plan.expressions) {
    // Data buffer (int64_t or double)
    size_t data_size = align_size(sizeof(int64_t) * estimated_num_groups);
    
    // Count buffer (for AVG)
    size_t count_size = 0;
    if (expr.agg_op == AggOp::AVG) {
      count_size = align_size(sizeof(int64_t) * estimated_num_groups);
    }
    
    // Validity buffer
    size_t validity_size = align_size(cudf::bitmask_allocation_size_bytes(estimated_num_groups));
    
    total += data_size + count_size + validity_size;
  }
  
  // Group-by key columns (rough estimate)
  total += align_size(sizeof(int64_t) * estimated_num_groups) * plan.group_by_col_indices.size();
  
  return total;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

FusedExecutionResult execute_fused_transform_aggregate(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Validate plan
  std::string validation_error;
  CUDF_EXPECTS(can_fuse(plan, &validation_error), 
               "Invalid execution plan: " + validation_error);
  
  FusedExecutionResult result;
  result.stats.num_expressions_fused = plan.expressions.size();
  result.stats.execution_strategy = plan.enable_warp_reduction ? "warp_reduction" : "basic";
  
  // Handle empty input
  if (input.num_rows() == 0) {
    result.stats.kernel_time_ns = 0;
    result.stats.total_time_ns = 0;
    result.stats.num_groups = 0;
    result.output_keys = std::make_unique<cudf::table>();
    result.output_values = std::make_unique<cudf::table>();
    return result;
  }
  
  cudf::size_type num_rows = input.num_rows();
  size_t num_exprs = plan.expressions.size();
  
  // ============================================================================
  // Step 1: Build group-by keys table and get groups
  // ============================================================================
  
  std::vector<cudf::column_view> key_columns;
  for (auto idx : plan.group_by_col_indices) {
    CUDF_EXPECTS(idx >= 0 && idx < input.num_columns(),
                 "Group-by column index out of range");
    key_columns.push_back(input.column(idx));
  }
  cudf::table_view keys_table(key_columns);
  
  // Use cudf::groupby to get group information
  cudf::groupby::groupby groupby_obj(keys_table, cudf::null_policy::EXCLUDE);
  auto groups = groupby_obj.get_groups(input, stream);
  
  cudf::size_type num_groups = groups.offsets.size() - 1;
  result.stats.num_groups = num_groups;
  
  // Convert offsets to per-row group IDs
  // offsets: [0, 3, 5, 8] -> group_ids: [0,0,0,1,1,2,2,2]
  rmm::device_uvector<cudf::size_type> d_group_ids(num_rows, stream, mr);
  
  // Simple kernel to expand offsets to group IDs
  auto const& offsets = groups.offsets;
  rmm::device_uvector<cudf::size_type> d_offsets(offsets.size(), stream, mr);
  cudaMemcpyAsync(d_offsets.data(), offsets.data(), 
                  offsets.size() * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice, stream.value());
  
  // Fill group IDs
  {
    // Use thrust to expand offsets to labels
    thrust::fill(rmm::exec_policy_nosync(stream, mr), 
                 d_group_ids.begin(), d_group_ids.end(), 
                 cudf::size_type(0));
    
    // Mark group boundaries
    for (size_t g = 0; g < offsets.size() - 1; ++g) {
      if (offsets[g] < num_rows) {
        cudaMemcpyAsync(d_group_ids.data() + offsets[g], &g,
                        sizeof(cudf::size_type), cudaMemcpyHostToDevice, stream.value());
      }
    }
    
    // Prefix sum to propagate group IDs
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr),
                           d_group_ids.begin(), d_group_ids.end(),
                           d_group_ids.begin(),
                           thrust::maximum<cudf::size_type>());
  }
  
  // ============================================================================
  // Step 2: Allocate shared output buffer
  // ============================================================================
  
  // Calculate required buffer size
  size_t sum_buffer_size = num_exprs * num_groups * sizeof(int64_t);
  size_t count_buffer_size = num_exprs * num_groups * sizeof(int64_t);  // For AVG
  size_t total_buffer_size = align_size(sum_buffer_size) + align_size(count_buffer_size);
  
  // Add space for output columns (final results)
  for (auto const& expr : plan.expressions) {
    total_buffer_size += align_size(sizeof(int64_t) * num_groups);  // Data
    total_buffer_size += align_size(cudf::bitmask_allocation_size_bytes(num_groups));  // Validity
  }
  
  result.shared_buffer = rmm::device_buffer(total_buffer_size, stream, mr);
  result.buffer_mr = std::make_unique<contiguous_buffer_resource>(
      result.shared_buffer.data(), result.shared_buffer.size());
  
  // ============================================================================
  // Step 3: Prepare device data structures
  // ============================================================================
  
  // Convert expressions to device format
  std::vector<detail::DeviceExprSpec> h_specs;
  h_specs.reserve(num_exprs);
  for (auto const& expr : plan.expressions) {
    h_specs.push_back(to_device_spec(expr));
  }
  
  rmm::device_uvector<detail::DeviceExprSpec> d_specs(num_exprs, stream, mr);
  cudaMemcpyAsync(d_specs.data(), h_specs.data(),
                  num_exprs * sizeof(detail::DeviceExprSpec),
                  cudaMemcpyHostToDevice, stream.value());
  
  // Prepare input column pointers
  std::vector<void const*> h_value_ptrs(input.num_columns());
  std::vector<cudf::bitmask_type const*> h_validity_ptrs(input.num_columns());
  
  for (int i = 0; i < input.num_columns(); ++i) {
    h_value_ptrs[i] = input.column(i).head();
    h_validity_ptrs[i] = input.column(i).null_mask();
  }
  
  rmm::device_uvector<void const*> d_value_ptrs(input.num_columns(), stream, mr);
  rmm::device_uvector<cudf::bitmask_type const*> d_validity_ptrs(input.num_columns(), stream, mr);
  
  cudaMemcpyAsync(d_value_ptrs.data(), h_value_ptrs.data(),
                  input.num_columns() * sizeof(void*),
                  cudaMemcpyHostToDevice, stream.value());
  cudaMemcpyAsync(d_validity_ptrs.data(), h_validity_ptrs.data(),
                  input.num_columns() * sizeof(cudf::bitmask_type*),
                  cudaMemcpyHostToDevice, stream.value());
  
  // Allocate sum and count output buffers
  rmm::device_uvector<int64_t> d_sum_buffer(num_exprs * num_groups, stream, mr);
  rmm::device_uvector<int64_t> d_count_buffer(num_exprs * num_groups, stream, mr);
  
  // Create pointer arrays for kernel
  std::vector<int64_t*> h_sum_ptrs(num_exprs);
  std::vector<int64_t*> h_count_ptrs(num_exprs);
  for (size_t i = 0; i < num_exprs; ++i) {
    h_sum_ptrs[i] = d_sum_buffer.data() + i * num_groups;
    h_count_ptrs[i] = d_count_buffer.data() + i * num_groups;
  }
  
  rmm::device_uvector<int64_t*> d_sum_ptrs(num_exprs, stream, mr);
  rmm::device_uvector<int64_t*> d_count_ptrs(num_exprs, stream, mr);
  
  cudaMemcpyAsync(d_sum_ptrs.data(), h_sum_ptrs.data(),
                  num_exprs * sizeof(int64_t*),
                  cudaMemcpyHostToDevice, stream.value());
  cudaMemcpyAsync(d_count_ptrs.data(), h_count_ptrs.data(),
                  num_exprs * sizeof(int64_t*),
                  cudaMemcpyHostToDevice, stream.value());
  
  // ============================================================================
  // Step 4: Initialize output buffers
  // ============================================================================
  
  {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE, num_exprs);
    
    detail::init_output_buffers_kernel<<<grid, block, 0, stream.value()>>>(
        d_sum_ptrs.data(),
        d_count_ptrs.data(),
        d_specs.data(),
        static_cast<int32_t>(num_exprs),
        num_groups);
  }
  
  // ============================================================================
  // Step 5: Execute fused kernel
  // ============================================================================
  
  auto kernel_start = std::chrono::high_resolution_clock::now();
  
  {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, num_exprs);
    
    if (plan.enable_warp_reduction) {
      detail::fused_transform_aggregate_kernel_warp_int64<<<grid, block, 0, stream.value()>>>(
          d_value_ptrs.data(),
          d_validity_ptrs.data(),
          num_rows,
          d_group_ids.data(),
          num_groups,
          d_specs.data(),
          static_cast<int32_t>(num_exprs),
          d_sum_ptrs.data(),
          d_count_ptrs.data());
    } else {
      detail::fused_transform_aggregate_kernel_int64<<<grid, block, 0, stream.value()>>>(
          d_value_ptrs.data(),
          d_validity_ptrs.data(),
          num_rows,
          d_group_ids.data(),
          num_groups,
          d_specs.data(),
          static_cast<int32_t>(num_exprs),
          d_sum_ptrs.data(),
          d_count_ptrs.data());
    }
  }
  
  stream.synchronize();
  
  auto kernel_end = std::chrono::high_resolution_clock::now();
  result.stats.kernel_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      kernel_end - kernel_start).count();
  
  // ============================================================================
  // Step 6: Build output columns
  // ============================================================================
  
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.reserve(num_exprs);
  
  for (size_t i = 0; i < num_exprs; ++i) {
    auto const& expr = plan.expressions[i];
    
    if (expr.agg_op == AggOp::AVG) {
      // AVG needs sum/count, output as double
      rmm::device_uvector<double> d_avg_output(num_groups, stream, mr);
      
      // Compute avg = sum / count on device using thrust
      thrust::transform(
          rmm::exec_policy_nosync(stream, mr),
          thrust::make_counting_iterator<cudf::size_type>(0),
          thrust::make_counting_iterator<cudf::size_type>(num_groups),
          d_avg_output.begin(),
          [sum_ptr = h_sum_ptrs[i], count_ptr = h_count_ptrs[i]] __device__ (cudf::size_type idx) {
            int64_t count = count_ptr[idx];
            return (count > 0) ? (static_cast<double>(sum_ptr[idx]) / count) : 0.0;
          });
      
      // Create column with double type
      auto col = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::FLOAT64},
          num_groups,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      
      cudaMemcpyAsync(col->mutable_view().data<double>(),
                      d_avg_output.data(),
                      num_groups * sizeof(double),
                      cudaMemcpyDeviceToDevice, stream.value());
      
      output_columns.push_back(std::move(col));
    } else if (expr.agg_op == AggOp::COUNT) {
      // COUNT uses count buffer
      auto col = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT64},
          num_groups,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      
      cudaMemcpyAsync(col->mutable_view().data<int64_t>(),
                      h_count_ptrs[i],
                      num_groups * sizeof(int64_t),
                      cudaMemcpyDeviceToDevice, stream.value());
      
      output_columns.push_back(std::move(col));
    } else {
      // SUM/MIN/MAX use sum buffer
      auto col = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT64},
          num_groups,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      
      cudaMemcpyAsync(col->mutable_view().data<int64_t>(),
                      h_sum_ptrs[i],
                      num_groups * sizeof(int64_t),
                      cudaMemcpyDeviceToDevice, stream.value());
      
      output_columns.push_back(std::move(col));
    }
  }
  
  result.output_values = std::make_unique<cudf::table>(std::move(output_columns));
  
  // Extract unique keys (one per group) using group offsets
  // groups.offsets: [0, 3, 5, 8] means groups start at indices 0, 3, 5
  // We need to gather keys at these start positions
  {
    std::vector<cudf::size_type> unique_key_indices;
    unique_key_indices.reserve(num_groups);
    for (cudf::size_type g = 0; g < num_groups; ++g) {
      unique_key_indices.push_back(groups.offsets[g]);
    }
    
    // Create device vector of gather indices
    rmm::device_uvector<cudf::size_type> d_gather_map(num_groups, stream, mr);
    cudaMemcpyAsync(d_gather_map.data(), unique_key_indices.data(),
                    num_groups * sizeof(cudf::size_type),
                    cudaMemcpyHostToDevice, stream.value());
    
    // Gather unique keys
    auto gather_map_col = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32},
        num_groups,
        d_gather_map.data(),
        nullptr,
        0);
    
    result.output_keys = cudf::gather(
        groups.keys->view(),
        gather_map_col,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);
  }
  
  stream.synchronize();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  result.stats.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time).count();
  
  return result;
}

}  // namespace spark_rapids_jni


