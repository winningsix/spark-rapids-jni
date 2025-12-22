/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "batched_transform_kernels.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <vector>

namespace spark_rapids_jni {
namespace transform {

// ============================================================================
// Batched Transform Operations for Pre-Project Optimization
//
// These functions provide host-side wrappers for batched transform kernels
// that process multiple columns in a single kernel launch, reducing the
// overhead of numerous kernel launches during the pre-project stage.
//
// Key patterns optimized:
// 1. COALESCE(col, scalar) - replace nulls with scalar value
// 2. COALESCE(col1, col2) - replace nulls with another column's value
// 3. IF_ELSE(cond, true_val, false_val) - ternary conditional
// 4. COALESCE(a, 0) * COALESCE(b, 0) - fused coalesce-multiply pattern
// ============================================================================

/**
 * @brief Batch COALESCE with scalar default values
 *
 * Replaces NULL values in multiple columns with scalar defaults in a single
 * kernel launch. This is the most common pattern in multi-column aggregation workloads:
 *   output[i] = COALESCE(input[i], default_scalar)
 *
 * Performance benefit: Reduces N kernel launches to 1 launch for N columns.
 *
 * @tparam T Data type (int64_t, double, etc.)
 * @param input_columns Vector of input columns to process
 * @param default_values Vector of scalar default values (one per column)
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns with nulls replaced
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar(
    std::vector<cudf::column_view> const& input_columns,
    std::vector<T> const& default_values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  auto const num_cols = input_columns.size();
  CUDF_EXPECTS(num_cols > 0, "At least one column required");
  CUDF_EXPECTS(default_values.size() == num_cols, "Default values size must match columns");
  
  auto const num_rows = input_columns[0].size();
  
  // Verify all columns have same size
  for (auto const& col : input_columns) {
    CUDF_EXPECTS(col.size() == num_rows, "All columns must have the same number of rows");
  }
  
  // Handle empty input
  if (num_rows == 0 || num_cols == 0) {
    std::vector<std::unique_ptr<cudf::column>> results;
    for (size_t i = 0; i < num_cols; ++i) {
      results.push_back(cudf::make_empty_column(input_columns[i].type()));
    }
    return results;
  }
  
  // Build host arrays
  std::vector<T const*> h_input_ptrs(num_cols);
  std::vector<cudf::bitmask_type const*> h_null_masks(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    h_input_ptrs[i] = input_columns[i].data<T>();
    h_null_masks[i] = input_columns[i].null_mask();
  }
  
  // Allocate output columns
  std::vector<std::unique_ptr<cudf::column>> results;
  results.reserve(num_cols);
  std::vector<T*> h_output_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    // Output has no nulls since COALESCE fills them in
    auto output = cudf::make_fixed_width_column(
        input_columns[i].type(),
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    h_output_ptrs[i] = output->mutable_view().data<T>();
    results.push_back(std::move(output));
  }
  
  // Copy to device
  auto d_input_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_input_ptrs.data(), h_input_ptrs.size()),
      stream, mr);
  auto d_null_masks = cudf::detail::make_device_uvector_async(
      cudf::host_span<cudf::bitmask_type const* const>(h_null_masks.data(), h_null_masks.size()),
      stream, mr);
  auto d_defaults = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const>(default_values.data(), default_values.size()),
      stream, mr);
  auto d_output_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T* const>(h_output_ptrs.data(), h_output_ptrs.size()),
      stream, mr);
  
  // Launch batched kernel
  constexpr int block_size = kernels::BATCH_AGG_BLOCK_SIZE;
  auto const num_blocks_x = (num_rows + block_size - 1) / block_size;
  dim3 grid(num_blocks_x, num_cols);  // 2D grid: x=rows, y=columns
  
  kernels::batch_coalesce_scalar_kernel<T><<<grid, block_size, 0, stream.value()>>>(
      d_input_ptrs.data(),
      d_null_masks.data(),
      d_defaults.data(),
      num_rows,
      num_cols,
      d_output_ptrs.data());
  
  return results;
}

/**
 * @brief Batch COALESCE with column default values
 *
 * Replaces NULL values in multiple columns with corresponding values from
 * default columns:
 *   output[i] = COALESCE(input[i], default_column[i])
 *
 * @tparam T Data type
 * @param input_columns Primary input columns
 * @param default_columns Default value columns (same size as input)
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column(
    std::vector<cudf::column_view> const& input_columns,
    std::vector<cudf::column_view> const& default_columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  auto const num_cols = input_columns.size();
  CUDF_EXPECTS(num_cols > 0, "At least one column required");
  CUDF_EXPECTS(default_columns.size() == num_cols, "Default columns size must match input");
  
  auto const num_rows = input_columns[0].size();
  
  // Verify sizes
  for (size_t i = 0; i < num_cols; ++i) {
    CUDF_EXPECTS(input_columns[i].size() == num_rows, "All columns must have same size");
    CUDF_EXPECTS(default_columns[i].size() == num_rows, "Default columns must have same size");
  }
  
  if (num_rows == 0 || num_cols == 0) {
    std::vector<std::unique_ptr<cudf::column>> results;
    for (size_t i = 0; i < num_cols; ++i) {
      results.push_back(cudf::make_empty_column(input_columns[i].type()));
    }
    return results;
  }
  
  // Build host arrays
  std::vector<T const*> h_input_ptrs(num_cols);
  std::vector<cudf::bitmask_type const*> h_null_masks(num_cols);
  std::vector<T const*> h_default_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    h_input_ptrs[i] = input_columns[i].data<T>();
    h_null_masks[i] = input_columns[i].null_mask();
    h_default_ptrs[i] = default_columns[i].data<T>();
  }
  
  // Allocate outputs
  std::vector<std::unique_ptr<cudf::column>> results;
  results.reserve(num_cols);
  std::vector<T*> h_output_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    auto output = cudf::make_fixed_width_column(
        input_columns[i].type(),
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    h_output_ptrs[i] = output->mutable_view().data<T>();
    results.push_back(std::move(output));
  }
  
  // Copy to device
  auto d_input_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_input_ptrs.data(), h_input_ptrs.size()),
      stream, mr);
  auto d_null_masks = cudf::detail::make_device_uvector_async(
      cudf::host_span<cudf::bitmask_type const* const>(h_null_masks.data(), h_null_masks.size()),
      stream, mr);
  auto d_default_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_default_ptrs.data(), h_default_ptrs.size()),
      stream, mr);
  auto d_output_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T* const>(h_output_ptrs.data(), h_output_ptrs.size()),
      stream, mr);
  
  // Launch kernel
  constexpr int block_size = kernels::BATCH_AGG_BLOCK_SIZE;
  auto const num_blocks_x = (num_rows + block_size - 1) / block_size;
  dim3 grid(num_blocks_x, num_cols);
  
  kernels::batch_coalesce_column_kernel<T><<<grid, block_size, 0, stream.value()>>>(
      d_input_ptrs.data(),
      d_null_masks.data(),
      d_default_ptrs.data(),
      num_rows,
      num_cols,
      d_output_ptrs.data());
  
  return results;
}

/**
 * @brief Batch IF_ELSE (ternary conditional)
 *
 * Computes: output = condition ? true_val : false_val
 * for multiple columns in a single kernel launch.
 *
 * @tparam T Data type
 * @param conditions Boolean condition columns
 * @param true_columns True-case value columns
 * @param false_columns False-case value columns
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_if_else(
    std::vector<cudf::column_view> const& conditions,
    std::vector<cudf::column_view> const& true_columns,
    std::vector<cudf::column_view> const& false_columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  auto const num_cols = conditions.size();
  CUDF_EXPECTS(num_cols > 0, "At least one column required");
  CUDF_EXPECTS(true_columns.size() == num_cols, "True columns size must match conditions");
  CUDF_EXPECTS(false_columns.size() == num_cols, "False columns size must match conditions");
  
  auto const num_rows = conditions[0].size();
  
  if (num_rows == 0 || num_cols == 0) {
    std::vector<std::unique_ptr<cudf::column>> results;
    for (size_t i = 0; i < num_cols; ++i) {
      results.push_back(cudf::make_empty_column(true_columns[i].type()));
    }
    return results;
  }
  
  // Build host arrays
  std::vector<bool const*> h_cond_ptrs(num_cols);
  std::vector<T const*> h_true_ptrs(num_cols);
  std::vector<T const*> h_false_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    h_cond_ptrs[i] = conditions[i].data<bool>();
    h_true_ptrs[i] = true_columns[i].data<T>();
    h_false_ptrs[i] = false_columns[i].data<T>();
  }
  
  // Allocate outputs
  std::vector<std::unique_ptr<cudf::column>> results;
  results.reserve(num_cols);
  std::vector<T*> h_output_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    auto output = cudf::make_fixed_width_column(
        true_columns[i].type(),
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    h_output_ptrs[i] = output->mutable_view().data<T>();
    results.push_back(std::move(output));
  }
  
  // Copy to device
  auto d_cond_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<bool const* const>(h_cond_ptrs.data(), h_cond_ptrs.size()),
      stream, mr);
  auto d_true_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_true_ptrs.data(), h_true_ptrs.size()),
      stream, mr);
  auto d_false_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_false_ptrs.data(), h_false_ptrs.size()),
      stream, mr);
  auto d_output_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T* const>(h_output_ptrs.data(), h_output_ptrs.size()),
      stream, mr);
  
  // Launch kernel
  constexpr int block_size = kernels::BATCH_AGG_BLOCK_SIZE;
  auto const num_blocks_x = (num_rows + block_size - 1) / block_size;
  dim3 grid(num_blocks_x, num_cols);
  
  kernels::batch_if_else_kernel<T><<<grid, block_size, 0, stream.value()>>>(
      d_cond_ptrs.data(),
      d_true_ptrs.data(),
      d_false_ptrs.data(),
      num_rows,
      num_cols,
      d_output_ptrs.data());
  
  return results;
}

/**
 * @brief Fused COALESCE + MULTIPLY batch operation
 *
 * Computes: output = COALESCE(a, default_a) * COALESCE(b, default_b)
 * for multiple columns in a single kernel launch.
 *
 * This is the most common pattern in multi-column aggregation workloads for variance
 * computation: (gpucoalesce(x, 0) * gpucoalesce(y, 0))
 *
 * Performance benefit:
 * - Without fusion: 3 kernel launches per column-pair (2 coalesce + 1 multiply)
 * - With fusion: 1 kernel launch for ALL column-pairs
 *
 * @tparam T Data type
 * @param a_columns First operand columns
 * @param a_defaults Scalar defaults for first operands
 * @param b_columns Second operand columns
 * @param b_defaults Scalar defaults for second operands
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply(
    std::vector<cudf::column_view> const& a_columns,
    std::vector<T> const& a_defaults,
    std::vector<cudf::column_view> const& b_columns,
    std::vector<T> const& b_defaults,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  auto const num_cols = a_columns.size();
  CUDF_EXPECTS(num_cols > 0, "At least one column required");
  CUDF_EXPECTS(b_columns.size() == num_cols, "b_columns size must match a_columns");
  CUDF_EXPECTS(a_defaults.size() == num_cols, "a_defaults size must match columns");
  CUDF_EXPECTS(b_defaults.size() == num_cols, "b_defaults size must match columns");
  
  auto const num_rows = a_columns[0].size();
  
  if (num_rows == 0 || num_cols == 0) {
    std::vector<std::unique_ptr<cudf::column>> results;
    for (size_t i = 0; i < num_cols; ++i) {
      results.push_back(cudf::make_empty_column(a_columns[i].type()));
    }
    return results;
  }
  
  // Build host arrays
  std::vector<T const*> h_a_ptrs(num_cols);
  std::vector<cudf::bitmask_type const*> h_a_masks(num_cols);
  std::vector<T const*> h_b_ptrs(num_cols);
  std::vector<cudf::bitmask_type const*> h_b_masks(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    h_a_ptrs[i] = a_columns[i].data<T>();
    h_a_masks[i] = a_columns[i].null_mask();
    h_b_ptrs[i] = b_columns[i].data<T>();
    h_b_masks[i] = b_columns[i].null_mask();
  }
  
  // Allocate outputs
  std::vector<std::unique_ptr<cudf::column>> results;
  results.reserve(num_cols);
  std::vector<T*> h_output_ptrs(num_cols);
  
  for (size_t i = 0; i < num_cols; ++i) {
    auto output = cudf::make_fixed_width_column(
        a_columns[i].type(),
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    h_output_ptrs[i] = output->mutable_view().data<T>();
    results.push_back(std::move(output));
  }
  
  // Copy to device
  auto d_a_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_a_ptrs.data(), h_a_ptrs.size()),
      stream, mr);
  auto d_a_masks = cudf::detail::make_device_uvector_async(
      cudf::host_span<cudf::bitmask_type const* const>(h_a_masks.data(), h_a_masks.size()),
      stream, mr);
  auto d_a_defaults = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const>(a_defaults.data(), a_defaults.size()),
      stream, mr);
  auto d_b_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const* const>(h_b_ptrs.data(), h_b_ptrs.size()),
      stream, mr);
  auto d_b_masks = cudf::detail::make_device_uvector_async(
      cudf::host_span<cudf::bitmask_type const* const>(h_b_masks.data(), h_b_masks.size()),
      stream, mr);
  auto d_b_defaults = cudf::detail::make_device_uvector_async(
      cudf::host_span<T const>(b_defaults.data(), b_defaults.size()),
      stream, mr);
  auto d_output_ptrs = cudf::detail::make_device_uvector_async(
      cudf::host_span<T* const>(h_output_ptrs.data(), h_output_ptrs.size()),
      stream, mr);
  
  // Launch FUSED kernel - this is where the magic happens!
  // Instead of: coalesce(a) -> coalesce(b) -> multiply (3 launches per column)
  // We do: coalesce_multiply for ALL columns (1 launch total)
  constexpr int block_size = kernels::BATCH_AGG_BLOCK_SIZE;
  auto const num_blocks_x = (num_rows + block_size - 1) / block_size;
  dim3 grid(num_blocks_x, num_cols);
  
  kernels::batch_coalesce_multiply_kernel<T><<<grid, block_size, 0, stream.value()>>>(
      d_a_ptrs.data(),
      d_a_masks.data(),
      d_a_defaults.data(),
      d_b_ptrs.data(),
      d_b_masks.data(),
      d_b_defaults.data(),
      num_rows,
      num_cols,
      d_output_ptrs.data());
  
  return results;
}

// Explicit template instantiations for common types
template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<double>(
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<int32_t>(
    std::vector<cudf::column_view> const&, std::vector<int32_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column<double>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

template std::vector<std::unique_ptr<cudf::column>> batch_if_else<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    std::vector<cudf::column_view> const&, rmm::cuda_stream_view, rmm::device_async_resource_ref);
template std::vector<std::unique_ptr<cudf::column>> batch_if_else<double>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    std::vector<cudf::column_view> const&, rmm::cuda_stream_view, rmm::device_async_resource_ref);

template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply<double>(
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

}  // namespace transform
}  // namespace spark_rapids_jni

// ============================================================================
// JNI Bindings for Batched Transform Operations
// ============================================================================

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

extern "C" {

/**
 * @brief JNI: Batch COALESCE with scalar defaults (int64)
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_BatchedTransform_batchCoalesceScalarInt64(
  JNIEnv* env,
  jclass,
  jlongArray j_column_handles,
  jlongArray j_defaults)
{
  JNI_NULL_CHECK(env, j_column_handles, "column handles is null", nullptr);
  JNI_NULL_CHECK(env, j_defaults, "defaults is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    cudf::jni::native_jlongArray col_handles(env, j_column_handles);
    cudf::jni::native_jlongArray defaults(env, j_defaults);
    
    auto const num_cols = col_handles.size();
    CUDF_EXPECTS(defaults.size() == num_cols, "Defaults size must match columns");
    
    std::vector<cudf::column_view> columns;
    columns.reserve(num_cols);
    std::vector<int64_t> default_vals(num_cols);
    
    for (int i = 0; i < num_cols; ++i) {
      auto* col = reinterpret_cast<cudf::column_view*>(col_handles[i]);
      columns.push_back(*col);
      default_vals[i] = static_cast<int64_t>(defaults[i]);
    }
    
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    
    auto results = spark_rapids_jni::transform::batch_coalesce_scalar<int64_t>(
        columns, default_vals, stream, mr);
    
    cudf::jni::native_jlongArray ret(env, num_cols);
    for (size_t i = 0; i < results.size(); ++i) {
      ret[i] = reinterpret_cast<jlong>(results[i].release());
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief JNI: Batch COALESCE with scalar defaults (double)
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_BatchedTransform_batchCoalesceScalarDouble(
  JNIEnv* env,
  jclass,
  jlongArray j_column_handles,
  jdoubleArray j_defaults)
{
  JNI_NULL_CHECK(env, j_column_handles, "column handles is null", nullptr);
  JNI_NULL_CHECK(env, j_defaults, "defaults is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    cudf::jni::native_jlongArray col_handles(env, j_column_handles);
    cudf::jni::native_jdoubleArray defaults(env, j_defaults);
    
    auto const num_cols = col_handles.size();
    CUDF_EXPECTS(defaults.size() == num_cols, "Defaults size must match columns");
    
    std::vector<cudf::column_view> columns;
    columns.reserve(num_cols);
    std::vector<double> default_vals(num_cols);
    
    for (int i = 0; i < num_cols; ++i) {
      auto* col = reinterpret_cast<cudf::column_view*>(col_handles[i]);
      columns.push_back(*col);
      default_vals[i] = defaults[i];
    }
    
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    
    auto results = spark_rapids_jni::transform::batch_coalesce_scalar<double>(
        columns, default_vals, stream, mr);
    
    cudf::jni::native_jlongArray ret(env, num_cols);
    for (size_t i = 0; i < results.size(); ++i) {
      ret[i] = reinterpret_cast<jlong>(results[i].release());
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief JNI: Fused COALESCE + MULTIPLY (double)
 *
 * Computes: output[i] = COALESCE(a[i], default_a[i]) * COALESCE(b[i], default_b[i])
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_BatchedTransform_batchCoalesceMultiplyDouble(
  JNIEnv* env,
  jclass,
  jlongArray j_a_handles,
  jdoubleArray j_a_defaults,
  jlongArray j_b_handles,
  jdoubleArray j_b_defaults)
{
  JNI_NULL_CHECK(env, j_a_handles, "a handles is null", nullptr);
  JNI_NULL_CHECK(env, j_a_defaults, "a defaults is null", nullptr);
  JNI_NULL_CHECK(env, j_b_handles, "b handles is null", nullptr);
  JNI_NULL_CHECK(env, j_b_defaults, "b defaults is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    cudf::jni::native_jlongArray a_handles(env, j_a_handles);
    cudf::jni::native_jdoubleArray a_defaults(env, j_a_defaults);
    cudf::jni::native_jlongArray b_handles(env, j_b_handles);
    cudf::jni::native_jdoubleArray b_defaults(env, j_b_defaults);
    
    auto const num_cols = a_handles.size();
    CUDF_EXPECTS(b_handles.size() == num_cols, "b columns must match a columns");
    CUDF_EXPECTS(a_defaults.size() == num_cols, "a defaults must match columns");
    CUDF_EXPECTS(b_defaults.size() == num_cols, "b defaults must match columns");
    
    std::vector<cudf::column_view> a_cols, b_cols;
    std::vector<double> a_def_vals(num_cols), b_def_vals(num_cols);
    
    for (int i = 0; i < num_cols; ++i) {
      a_cols.push_back(*reinterpret_cast<cudf::column_view*>(a_handles[i]));
      b_cols.push_back(*reinterpret_cast<cudf::column_view*>(b_handles[i]));
      a_def_vals[i] = a_defaults[i];
      b_def_vals[i] = b_defaults[i];
    }
    
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    
    auto results = spark_rapids_jni::transform::batch_coalesce_multiply<double>(
        a_cols, a_def_vals, b_cols, b_def_vals, stream, mr);
    
    cudf::jni::native_jlongArray ret(env, num_cols);
    for (size_t i = 0; i < results.size(); ++i) {
      ret[i] = reinterpret_cast<jlong>(results[i].release());
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
