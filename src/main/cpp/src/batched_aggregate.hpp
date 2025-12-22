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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Enum representing the type of aggregation
 */
enum class BatchedAggType : int32_t {
  SUM = 0,
  AVG,
  COUNT,
  MIN,
  MAX,
  COUNT_DISTINCT,
  // Add more as needed
};

/**
 * @brief Specification for a single aggregation request
 */
struct batched_agg_spec {
  cudf::column_view values;         // Input column to aggregate
  BatchedAggType agg_type;          // Type of aggregation
  cudf::data_type output_type;      // Expected output data type
  cudf::null_policy null_handling;  // How to handle nulls in aggregation
};

/**
 * @brief A slice of a shared buffer, representing one column's data
 *
 * This is similar to the buffer_slice concept in shuffle_assemble.cu
 */
struct agg_buffer_slice {
  uint8_t* data;           // Pointer to start of this slice
  size_t size;             // Size of this slice in bytes
  size_t offset_in_buffer; // Offset from base of shared buffer

  agg_buffer_slice() : data(nullptr), size(0), offset_in_buffer(0) {}
  agg_buffer_slice(uint8_t* d, size_t s, size_t o) : data(d), size(s), offset_in_buffer(o) {}
};

/**
 * @brief Result of batched aggregation with shared buffer ownership
 * 
 * IMPORTANT: The buffer_mr member keeps the memory resource alive. This is critical
 * because columns created with the custom memory resource store a reference to it.
 * When columns are garbage collected, they call deallocate() through that reference.
 * If buffer_mr is destroyed before the columns, this causes a SIGSEGV crash.
 * 
 * The destruction order must be: columns first, then buffer_mr, then shared_buffer.
 * C++ struct members are destroyed in reverse declaration order, so we declare:
 * 1. shared_buffer (destroyed last - columns' data lives here)
 * 2. buffer_mr (destroyed second - columns call deallocate through this)
 * 3. values/keys (destroyed first - triggers column cleanup)
 */
struct batched_agg_result {
  // Declared first, destroyed last: the actual memory that columns point to
  rmm::device_buffer shared_buffer;
  
  // Declared second, destroyed second: memory resource with no-op deallocate
  // Must outlive columns that were created using it
  std::unique_ptr<rmm::mr::device_memory_resource> buffer_mr;
  
  // Declared last, destroyed first: the columns that reference shared_buffer
  std::unique_ptr<cudf::table> keys;
  std::unique_ptr<cudf::table> values;

  // Number of groups (rows in output)
  [[nodiscard]] cudf::size_type num_groups() const {
    return keys ? keys->num_rows() : 0;
  }
};

/**
 * @brief Metadata about the shared buffer allocation
 */
struct shared_buffer_info {
  size_t total_size;                         // Total buffer size
  std::vector<agg_buffer_slice> data_slices; // Slices for data buffers
  std::vector<agg_buffer_slice> validity_slices; // Slices for validity buffers
  size_t num_output_columns;                 // Number of output columns
};

/**
 * @brief Estimate the output buffer size needed for batched aggregation
 *
 * This function estimates the total memory needed for the output columns
 * based on the input specifications and estimated number of output groups.
 *
 * @param specs Vector of aggregation specifications
 * @param estimated_num_groups Estimated number of output groups
 * @return Total estimated buffer size in bytes
 */
size_t estimate_output_buffer_size(
    std::vector<batched_agg_spec> const& specs,
    cudf::size_type estimated_num_groups);

/**
 * @brief Calculate the exact buffer layout for output columns
 *
 * Given the exact number of groups, calculate the precise memory layout
 * for all output columns.
 *
 * @param specs Vector of aggregation specifications
 * @param num_groups Exact number of output groups
 * @param stream CUDA stream for operations
 * @return shared_buffer_info with buffer layout details
 */
shared_buffer_info calculate_buffer_layout(
    std::vector<batched_agg_spec> const& specs,
    cudf::size_type num_groups,
    rmm::cuda_stream_view stream);

/**
 * @brief Perform batched groupby aggregation with shared output buffer
 *
 * This function performs multiple aggregations in a single pass, allocating
 * a single large buffer for all output columns to reduce memory fragmentation
 * and allocation overhead.
 *
 * Key benefits:
 * - Single memory allocation for all output columns
 * - Reduced RMM allocation overhead
 * - Better memory locality for subsequent operations
 * - Lower peak memory usage due to reduced fragmentation
 *
 * @param keys Table view of grouping columns
 * @param specs Vector of aggregation specifications (value columns + agg types)
 * @param stream CUDA stream for operations
 * @param mr Memory resource for device allocations
 * @return batched_agg_result containing output keys, values, and shared buffer
 *
 * @throws cudf::logic_error if any specification is invalid
 */
batched_agg_result batched_groupby_aggregate(
    cudf::table_view const& keys,
    std::vector<batched_agg_spec> const& specs,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Group aggregation specifications by type for batched processing
 *
 * This helper function groups specifications by their aggregation type
 * to enable vectorized processing of similar operations.
 *
 * @param specs Vector of aggregation specifications
 * @return Map from aggregation type to vector of specs with that type
 */
std::map<BatchedAggType, std::vector<size_t>> group_specs_by_type(
    std::vector<batched_agg_spec> const& specs);

/**
 * @brief Convert BatchedAggType to cudf aggregation
 *
 * @param type The batched aggregation type
 * @return Unique pointer to cudf aggregation object
 */
std::unique_ptr<cudf::groupby_aggregation> to_cudf_aggregation(BatchedAggType type);

namespace detail {

/**
 * @brief Internal function to allocate and partition the shared buffer
 *
 * Allocates a single contiguous buffer and creates slices for each
 * output column's data and validity buffers.
 *
 * @param layout Buffer layout information
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Pair of device buffer and vector of slices
 */
std::pair<rmm::device_buffer, std::vector<agg_buffer_slice>>
allocate_shared_buffer(
    shared_buffer_info const& layout,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Build output columns from buffer slices
 *
 * Creates cudf column objects that reference the pre-allocated buffer slices.
 *
 * @param specs Original aggregation specs
 * @param num_groups Number of output rows
 * @param slices Buffer slices for each column
 * @param shared_buffer The underlying shared buffer
 * @param stream CUDA stream
 * @return Vector of column unique pointers
 */
std::vector<std::unique_ptr<cudf::column>>
build_output_columns_from_slices(
    std::vector<batched_agg_spec> const& specs,
    cudf::size_type num_groups,
    std::vector<agg_buffer_slice> const& slices,
    rmm::device_buffer const& shared_buffer,
    rmm::cuda_stream_view stream);

}  // namespace detail

}  // namespace spark_rapids_jni

