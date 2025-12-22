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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {
namespace transform {

/**
 * @brief Batched Transform Operations for Pre-Project Optimization
 *
 * These functions provide batch processing of common transform operations
 * that appear in Spark's pre-project stage. By processing multiple columns
 * in a single kernel launch (using 2D grid), we significantly reduce kernel
 * launch overhead.
 *
 * Performance characteristics:
 * - Traditional approach: N columns × M operations = N×M kernel launches
 * - Batched approach: N columns × M operations ≈ M kernel launches
 *
 * For a workload with 100 columns and COALESCE+MULTIPLY pattern:
 * - Traditional: ~300 kernel launches (coalesce a, coalesce b, multiply)
 * - Batched:     ~1 kernel launch (fused coalesce-multiply)
 */

/**
 * @brief Batch COALESCE with scalar default values
 *
 * Replaces NULL values in multiple columns with scalar defaults:
 *   output[i] = input[i] IS NULL ? default_values[i] : input[i]
 *
 * @tparam T Data type (int64_t, double, int32_t)
 * @param input_columns Input columns to process
 * @param default_values Scalar default for each column
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns (no nulls)
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar(
    std::vector<cudf::column_view> const& input_columns,
    std::vector<T> const& default_values,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Batch COALESCE with column default values
 *
 * Replaces NULL values with corresponding values from default columns:
 *   output[row,col] = input[row,col] IS NULL ? default[row,col] : input[row,col]
 *
 * @tparam T Data type
 * @param input_columns Primary input columns
 * @param default_columns Default value columns
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column(
    std::vector<cudf::column_view> const& input_columns,
    std::vector<cudf::column_view> const& default_columns,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Batch IF_ELSE (ternary conditional)
 *
 * Computes conditional selection for multiple columns:
 *   output = condition ? true_value : false_value
 *
 * @tparam T Data type for values
 * @param conditions Boolean condition columns
 * @param true_columns Values when condition is true
 * @param false_columns Values when condition is false
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_if_else(
    std::vector<cudf::column_view> const& conditions,
    std::vector<cudf::column_view> const& true_columns,
    std::vector<cudf::column_view> const& false_columns,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Fused COALESCE + MULTIPLY batch operation
 *
 * Computes: output = COALESCE(a, default_a) * COALESCE(b, default_b)
 * for multiple column pairs in a single kernel launch.
 *
 * This is the most common pattern in multi-column aggregation workloads:
 *   (gpucoalesce(pre_col, 0) * gpucoalesce(post_col, 0))
 *
 * Without fusion: 3 kernels per column-pair × N pairs = 3N kernels
 * With fusion: 1 kernel for all N pairs
 *
 * @tparam T Data type
 * @param a_columns First operand columns
 * @param a_defaults Scalar defaults for first operands
 * @param b_columns Second operand columns
 * @param b_defaults Scalar defaults for second operands
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of output columns (a * b with nulls handled)
 */
template <typename T>
std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply(
    std::vector<cudf::column_view> const& a_columns,
    std::vector<T> const& a_defaults,
    std::vector<cudf::column_view> const& b_columns,
    std::vector<T> const& b_defaults,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// Extern declarations for common instantiations
extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<double>(
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_scalar<int32_t>(
    std::vector<cudf::column_view> const&, std::vector<int32_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_column<double>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

extern template std::vector<std::unique_ptr<cudf::column>> batch_if_else<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    std::vector<cudf::column_view> const&, rmm::cuda_stream_view, rmm::device_async_resource_ref);
extern template std::vector<std::unique_ptr<cudf::column>> batch_if_else<double>(
    std::vector<cudf::column_view> const&, std::vector<cudf::column_view> const&,
    std::vector<cudf::column_view> const&, rmm::cuda_stream_view, rmm::device_async_resource_ref);

extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply<int64_t>(
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    std::vector<cudf::column_view> const&, std::vector<int64_t> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);
extern template std::vector<std::unique_ptr<cudf::column>> batch_coalesce_multiply<double>(
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    std::vector<cudf::column_view> const&, std::vector<double> const&,
    rmm::cuda_stream_view, rmm::device_async_resource_ref);

}  // namespace transform
}  // namespace spark_rapids_jni
