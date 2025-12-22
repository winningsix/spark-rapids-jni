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

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

namespace spark_rapids_jni {
namespace kernels {

// Block size for batched kernels
constexpr int BATCH_AGG_BLOCK_SIZE = 256;

// ============================================================================
// Batched Transform Kernels
// ============================================================================

/**
 * @brief Batched COALESCE kernel with scalar defaults
 */
template <typename T>
__global__ void batch_coalesce_scalar_kernel(
    T const* const* input_ptrs,
    cudf::bitmask_type const* const* null_masks,
    T const* default_vals,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  bool const is_null = (null_mask != nullptr) && !cudf::bit_is_set(null_mask, row_idx);
  
  output_ptrs[col_idx][row_idx] = is_null ? default_vals[col_idx] : input_ptrs[col_idx][row_idx];
}

/**
 * @brief Batched COALESCE kernel with column defaults
 */
template <typename T>
__global__ void batch_coalesce_column_kernel(
    T const* const* input_ptrs,
    cudf::bitmask_type const* const* input_null_masks,
    T const* const* default_ptrs,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = input_null_masks[col_idx];
  bool const is_null = (null_mask != nullptr) && !cudf::bit_is_set(null_mask, row_idx);
  
  output_ptrs[col_idx][row_idx] = is_null ? default_ptrs[col_idx][row_idx] : input_ptrs[col_idx][row_idx];
}

/**
 * @brief Batched IF_ELSE kernel (ternary conditional)
 */
template <typename T>
__global__ void batch_if_else_kernel(
    bool const* const* condition_ptrs,
    T const* const* true_ptrs,
    T const* const* false_ptrs,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  bool const cond = condition_ptrs[col_idx][row_idx];
  output_ptrs[col_idx][row_idx] = cond ? true_ptrs[col_idx][row_idx] : false_ptrs[col_idx][row_idx];
}

/**
 * @brief Fused COALESCE + MULTIPLY kernel
 */
template <typename T>
__global__ void batch_coalesce_multiply_kernel(
    T const* const* a_ptrs,
    cudf::bitmask_type const* const* a_null_masks,
    T const* default_a,
    T const* const* b_ptrs,
    cudf::bitmask_type const* const* b_null_masks,
    T const* default_b,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  // Coalesce A
  auto const a_mask = a_null_masks[col_idx];
  bool const a_null = (a_mask != nullptr) && !cudf::bit_is_set(a_mask, row_idx);
  T const a_val = a_null ? default_a[col_idx] : a_ptrs[col_idx][row_idx];
  
  // Coalesce B
  auto const b_mask = b_null_masks[col_idx];
  bool const b_null = (b_mask != nullptr) && !cudf::bit_is_set(b_mask, row_idx);
  T const b_val = b_null ? default_b[col_idx] : b_ptrs[col_idx][row_idx];
  
  // Multiply
  output_ptrs[col_idx][row_idx] = a_val * b_val;
}

}  // namespace kernels
}  // namespace spark_rapids_jni
