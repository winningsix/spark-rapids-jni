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

#include <cuda/atomic>
#include <cuda/std/limits>

namespace spark_rapids_jni {
namespace kernels {

// Block size for aggregation kernels
constexpr int BATCH_AGG_BLOCK_SIZE = 256;

// ============================================================================
// Helper for atomic operations on int64_t
// CUDA's built-in atomicAdd doesn't support int64_t directly, only unsigned long long.
// We use cuda::atomic_ref from libcu++ for proper int64_t atomic operations.
// ============================================================================
__device__ __forceinline__ void atomicAdd_int64(int64_t* address, int64_t val)
{
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref(*address);
  ref.fetch_add(val, cuda::memory_order_relaxed);
}

__device__ __forceinline__ void atomicAdd_size_type(cudf::size_type* address, cudf::size_type val)
{
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref(*address);
  ref.fetch_add(val, cuda::memory_order_relaxed);
}

// ============================================================================
// Batched Aggregation Kernels
// 
// These kernels process multiple value columns in a SINGLE kernel launch using
// a 2D grid, where:
//   - gridDim.x * blockDim.x = threads to process rows
//   - gridDim.y = number of value columns
//
// This reduces kernel launch overhead by ~N times for N columns, following
// the same pattern as cudf's PR 20872 (batch_null_count).
// ============================================================================

/**
 * @brief Batched SUM kernel for int64 output
 *
 * Processes multiple value columns simultaneously using a 2D grid.
 * Each column is processed by a separate "row" of thread blocks (gridDim.y).
 *
 * @tparam T Input value type
 * @param value_ptrs Array of pointers to value columns (device memory)
 * @param null_masks Array of pointers to null masks (device memory, can be nullptr)
 * @param group_labels Group label for each input row
 * @param num_rows Number of input rows
 * @param num_cols Number of value columns
 * @param num_groups Number of output groups
 * @param output_ptrs Array of pointers to output arrays (one per column)
 */
template <typename T>
__global__ void batch_sum_kernel_int64(
    T const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  // Check if this row is valid (null handling)
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    return;  // Skip null values
  }
  
  auto const value = static_cast<int64_t>(value_ptrs[col_idx][row_idx]);
  auto const group = group_labels[row_idx];
  auto* output = output_ptrs[col_idx];
  
  atomicAdd_int64(&output[group], value);
}

/**
 * @brief Batched SUM kernel for double output
 */
template <typename T>
__global__ void batch_sum_kernel_double(
    T const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    double* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    return;
  }
  
  auto const value = static_cast<double>(value_ptrs[col_idx][row_idx]);
  auto const group = group_labels[row_idx];
  auto* output = output_ptrs[col_idx];
  
  atomicAdd(&output[group], value);
}

// =============================================================================
// GPU-SPECIFIC OPTIMIZATIONS (Different from CPU/Velox)
// =============================================================================
// Key differences from CPU optimization:
// 1. Atomic contention is the #1 bottleneck (not memory allocation)
// 2. Warp-level reduction can reduce atomics by 32x
// 3. Shared memory for block-level partial sums
// 4. Coalesced memory access patterns are critical
// 5. Perfect hash for small integer keys (no hash computation)
// =============================================================================

// =============================================================================
// PERFECT HASH OPTIMIZATION
// =============================================================================
// When keys are small integers (0 to max_key), we can skip the entire
// groupby/hash process and use the key value directly as the output index.
//
// Benefits:
// - No hash computation
// - No sorting (get_groups)
// - No label generation
// - Direct scatter to output
//
// Applicable when:
// - Single column integer key
// - Key values are dense (0, 1, 2, ..., N-1) or nearly dense
// - max(key) is reasonable (< 10M typically)
// =============================================================================

/**
 * @brief Perfect hash SUM kernel for small integer keys
 *
 * When keys are small integers, use key value directly as output index.
 * This completely bypasses groupby/hash/sort!
 *
 * @param values Input values to sum
 * @param keys Integer keys (used directly as output indices)
 * @param null_mask Null mask for values (can be nullptr)
 * @param num_rows Number of input rows
 * @param max_key Maximum key value (output array size)
 * @param output Output array indexed by key value
 */
template <typename KeyType, typename ValueType, typename OutputType>
__global__ void perfect_hash_sum_kernel(
    ValueType const* values,
    KeyType const* keys,
    cudf::bitmask_type const* null_mask,
    cudf::size_type num_rows,
    KeyType max_key,
    OutputType* output)
{
  auto const idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_rows) return;
  
  // Check null
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, idx)) return;
  
  auto const key = keys[idx];
  if (key < 0 || key > max_key) return;  // Bounds check
  
  auto const value = static_cast<OutputType>(values[idx]);
  
  // Direct atomic to output[key] - no hash, no lookup!
  if constexpr (std::is_same_v<OutputType, int64_t>) {
    atomicAdd_int64(&output[key], value);
  } else if constexpr (std::is_same_v<OutputType, double>) {
    atomicAdd(&output[key], value);
  } else {
    atomicAdd(&output[key], value);
  }
}

/**
 * @brief Perfect hash COUNT kernel for small integer keys
 */
template <typename KeyType>
__global__ void perfect_hash_count_kernel(
    KeyType const* keys,
    cudf::bitmask_type const* null_mask,
    cudf::size_type num_rows,
    KeyType max_key,
    cudf::size_type* output)
{
  auto const idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_rows) return;
  
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, idx)) return;
  
  auto const key = keys[idx];
  if (key < 0 || key > max_key) return;
  
  atomicAdd_size_type(&output[key], 1);
}

/**
 * @brief Batched perfect hash SUM for multiple columns
 *
 * When keys are small integers, process all value columns in one kernel
 * using the key value directly as the output index.
 *
 * 2D grid: (rows, columns)
 */
template <typename KeyType>
__global__ void batch_perfect_hash_sum_kernel(
    int64_t const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    KeyType const* keys,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    KeyType max_key,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) return;
  
  auto const key = keys[row_idx];
  if (key < 0 || key > max_key) return;
  
  auto const value = value_ptrs[col_idx][row_idx];
  
  // Direct scatter: output[col][key] += value
  atomicAdd_int64(&output_ptrs[col_idx][key], value);
}

/**
 * @brief Batched perfect hash COUNT for multiple columns
 */
template <typename KeyType>
__global__ void batch_perfect_hash_count_kernel(
    cudf::bitmask_type const* const* null_masks,
    KeyType const* keys,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    KeyType max_key,
    cudf::size_type* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) return;
  
  auto const key = keys[row_idx];
  if (key < 0 || key > max_key) return;
  
  atomicAdd_size_type(&output_ptrs[col_idx][key], 1);
}

/**
 * @brief Warp-level reduction using shuffle instructions
 * 
 * Reduces 32 values within a warp to a single value, eliminating
 * 31 atomic operations per warp. This is a GPU-specific optimization
 * that has no CPU equivalent.
 */
__device__ __forceinline__ int64_t warp_reduce_sum(int64_t val)
{
  // Use warp shuffle to reduce within the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ cudf::size_type warp_reduce_sum_int(cudf::size_type val)
{
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/**
 * @brief Batched SUM kernel with GPU-specific optimizations
 *
 * GPU optimizations applied:
 * 1. WARP-LEVEL REDUCTION: Reduce 32 values before atomic → 32x fewer atomics
 * 2. COALESCED ACCESS: Threads in warp read consecutive memory addresses
 * 3. SHARED MEMORY: Block-level partial sums (optional, for very high contention)
 * 
 * This differs from CPU (Velox) where the focus is on:
 * - Memory locality (cache lines)
 * - Branch prediction
 * - SIMD vectorization
 */
__global__ void batch_sum_kernel(
    int64_t const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  
  // Get value (0 if out of bounds or null)
  int64_t my_value = 0;
  cudf::size_type my_group = 0;
  bool is_valid = false;
  
  if (row_idx < num_rows) {
    auto const null_mask = null_masks[col_idx];
    is_valid = (null_mask == nullptr || cudf::bit_is_set(null_mask, row_idx));
    if (is_valid) {
      my_value = value_ptrs[col_idx][row_idx];
      my_group = group_labels[row_idx];
    }
  }
  
  // =========================================================================
  // GPU OPTIMIZATION: Warp-level aggregation to reduce atomic contention
  // =========================================================================
  // Check if all threads in warp have the same group (common case for sorted data)
  // If so, we can do warp reduction and only ONE atomic per warp (32x reduction)
  
  unsigned int warp_mask = __ballot_sync(0xffffffff, is_valid);
  int lane_id = threadIdx.x & 31;
  
  // Broadcast first valid thread's group to all threads in warp
  cudf::size_type warp_group = __shfl_sync(0xffffffff, my_group, __ffs(warp_mask) - 1);
  
  // Check if all valid threads in warp have the same group
  bool same_group = __all_sync(warp_mask, !is_valid || my_group == warp_group);
  
  if (same_group && warp_mask != 0) {
    // FAST PATH: All threads in warp belong to same group
    // Do warp reduction (32 values → 1), then single atomic
    int64_t warp_sum = warp_reduce_sum(is_valid ? my_value : 0);
    if (lane_id == 0) {
      atomicAdd_int64(&output_ptrs[col_idx][warp_group], warp_sum);
    }
  } else {
    // SLOW PATH: Different groups in warp, fall back to per-thread atomics
    if (is_valid) {
      atomicAdd_int64(&output_ptrs[col_idx][my_group], my_value);
    }
  }
}

/**
 * @brief Batched COUNT kernel (counts non-null values)
 *
 * Counts valid (non-null) values for each group across multiple columns.
 */
/**
 * @brief Fused COALESCE + SUM kernel
 *
 * CRITICAL OPTIMIZATION: Eliminates the separate Project phase for COALESCE!
 *
 * Instead of:
 *   Project: COALESCE(col, 0) → temp_col   (35+ hours!)
 *   Aggregate: SUM(temp_col)
 *
 * We do:
 *   Aggregate: SUM(col) with null→0 replacement (in one pass)
 *
 * This saves:
 * - Entire Project phase time
 * - Memory bandwidth (no intermediate materialization)
 * - N kernel launches for N COALESCE operations
 */
__global__ void batch_sum_coalesce_kernel(
    int64_t const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    int64_t null_replacement,  // The default value (usually 0)
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  auto const group = group_labels[row_idx];
  
  // COALESCE logic fused into SUM
  int64_t value;
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    // NULL → use replacement value (e.g., 0)
    value = null_replacement;
  } else {
    value = value_ptrs[col_idx][row_idx];
  }
  
  atomicAdd_int64(&output_ptrs[col_idx][group], value);
}

/**
 * @brief Fused COALESCE + SUM kernel with warp reduction
 *
 * Combines:
 * 1. COALESCE(col, 0) - handles NULL values
 * 2. Warp-level reduction - reduces atomic contention
 * 3. Shared groupby - uses pre-computed group labels
 */
__global__ void batch_sum_coalesce_warp_kernel(
    int64_t const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    int64_t null_replacement,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  
  // Get value with COALESCE logic
  int64_t my_value = null_replacement;  // Default for out-of-bounds or NULL
  cudf::size_type my_group = 0;
  bool is_active = false;
  
  if (row_idx < num_rows) {
    is_active = true;
    my_group = group_labels[row_idx];
    
    auto const null_mask = null_masks[col_idx];
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
      my_value = null_replacement;  // COALESCE: NULL → 0
    } else {
      my_value = value_ptrs[col_idx][row_idx];
    }
  }
  
  // Warp-level optimization for sorted data
  unsigned int warp_mask = __ballot_sync(0xffffffff, is_active);
  int lane_id = threadIdx.x & 31;
  
  if (warp_mask == 0) return;
  
  cudf::size_type warp_group = __shfl_sync(0xffffffff, my_group, __ffs(warp_mask) - 1);
  bool same_group = __all_sync(warp_mask, !is_active || my_group == warp_group);
  
  if (same_group) {
    // FAST PATH: Warp reduction + 1 atomic
    int64_t warp_sum = warp_reduce_sum(is_active ? my_value : 0);
    if (lane_id == 0) {
      atomicAdd_int64(&output_ptrs[col_idx][warp_group], warp_sum);
    }
  } else {
    // SLOW PATH: Per-thread atomic
    if (is_active) {
      atomicAdd_int64(&output_ptrs[col_idx][my_group], my_value);
    }
  }
}

// =============================================================================
// VARIANCE/COVARIANCE SPECIALIZED KERNELS
// =============================================================================
// Optimized for the pattern: SUM(COALESCE(x,0) * COALESCE(y,0))
// This is extremely common in statistical computations.
// =============================================================================

/**
 * @brief Fused variance kernel: SUM(COALESCE(x,0)²) in one pass
 *
 * Instead of:
 *   Project: temp = COALESCE(x, 0) * COALESCE(x, 0)  // 2x COALESCE + 1x MUL
 *   Aggregate: SUM(temp)
 *
 * We do:
 *   value = is_null ? 0 : x
 *   SUM(value * value)  // One pass, no intermediate
 *
 * Saves: 2 COALESCE kernels + 1 MUL kernel + memory for temp column
 */
__global__ void batch_sum_variance_kernel(
    int64_t const* const* value_ptrs,       // x values
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)            // SUM(x²)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  auto const group = group_labels[row_idx];
  
  // COALESCE(x, 0) - inline
  int64_t x = 0;
  if (null_mask == nullptr || cudf::bit_is_set(null_mask, row_idx)) {
    x = value_ptrs[col_idx][row_idx];
  }
  
  // x² for variance
  int64_t x_squared = x * x;
  
  atomicAdd_int64(&output_ptrs[col_idx][group], x_squared);
}

/**
 * @brief Fused covariance kernel: SUM(COALESCE(x,0) * COALESCE(y,0)) in one pass
 *
 * For computing covariance between pairs of columns.
 * Input: pairs of columns [(x1,y1), (x2,y2), ...]
 * Output: SUM(xi * yi) for each pair
 */
__global__ void batch_sum_covariance_kernel(
    int64_t const* const* x_ptrs,           // First column of each pair
    int64_t const* const* y_ptrs,           // Second column of each pair
    cudf::bitmask_type const* const* x_null_masks,
    cudf::bitmask_type const* const* y_null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_pairs,              // Number of (x,y) pairs
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)            // SUM(x*y)
{
  auto const pair_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (pair_idx >= num_pairs) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const group = group_labels[row_idx];
  
  // COALESCE(x, 0)
  int64_t x = 0;
  auto const x_mask = x_null_masks[pair_idx];
  if (x_mask == nullptr || cudf::bit_is_set(x_mask, row_idx)) {
    x = x_ptrs[pair_idx][row_idx];
  }
  
  // COALESCE(y, 0)
  int64_t y = 0;
  auto const y_mask = y_null_masks[pair_idx];
  if (y_mask == nullptr || cudf::bit_is_set(y_mask, row_idx)) {
    y = y_ptrs[pair_idx][row_idx];
  }
  
  // x * y for covariance
  int64_t xy = x * y;
  
  atomicAdd_int64(&output_ptrs[pair_idx][group], xy);
}

/**
 * @brief Fused conditional variance: SUM(COALESCE(IF(cond, x, 0), 0)²)
 *
 * For pattern: SUM(COALESCE(IF(cond > 0, x, 0), 0) * COALESCE(IF(cond > 0, x, 0), 0))
 * 
 * Instead of:
 *   temp1 = IF(cond > 0, x, 0)
 *   temp2 = COALESCE(temp1, 0)
 *   temp3 = temp2 * temp2
 *   SUM(temp3)
 *
 * We do everything in one kernel!
 */
__global__ void batch_sum_conditional_variance_kernel(
    int64_t const* const* value_ptrs,       // x values
    int64_t const* const* cond_ptrs,        // condition columns
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    int64_t* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const group = group_labels[row_idx];
  
  // IF(cond > 0, x, 0) then COALESCE(result, 0)
  int64_t value = 0;
  int64_t cond = cond_ptrs[col_idx][row_idx];
  
  if (cond > 0) {
    auto const null_mask = null_masks[col_idx];
    if (null_mask == nullptr || cudf::bit_is_set(null_mask, row_idx)) {
      value = value_ptrs[col_idx][row_idx];
    }
  }
  
  // Square for variance
  int64_t value_squared = value * value;
  
  atomicAdd_int64(&output_ptrs[col_idx][group], value_squared);
}

/**
 * @brief Batched COUNT kernel with GPU-specific optimizations
 *
 * Same warp-level optimization as SUM kernel:
 * - For sorted data (from get_groups), threads in same warp often have same group
 * - Warp reduction: 32 counts → 1 atomic (32x fewer atomics)
 */
__global__ void batch_count_kernel(
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    cudf::size_type* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  
  // Determine if this thread contributes a count
  bool is_valid = false;
  cudf::size_type my_group = 0;
  
  if (row_idx < num_rows) {
    auto const null_mask = null_masks[col_idx];
    is_valid = (null_mask == nullptr || cudf::bit_is_set(null_mask, row_idx));
    if (is_valid) {
      my_group = group_labels[row_idx];
    }
  }
  
  // =========================================================================
  // GPU OPTIMIZATION: Warp-level count aggregation
  // =========================================================================
  unsigned int warp_mask = __ballot_sync(0xffffffff, is_valid);
  int lane_id = threadIdx.x & 31;
  
  if (warp_mask == 0) return;  // No valid rows in this warp
  
  // Broadcast first valid thread's group
  cudf::size_type warp_group = __shfl_sync(0xffffffff, my_group, __ffs(warp_mask) - 1);
  
  // Check if all valid threads have same group
  bool same_group = __all_sync(warp_mask, !is_valid || my_group == warp_group);
  
  if (same_group) {
    // FAST PATH: Count valid bits in warp, single atomic
    cudf::size_type warp_count = __popc(warp_mask);
    if (lane_id == 0) {
      atomicAdd_size_type(&output_ptrs[col_idx][warp_group], warp_count);
    }
  } else {
    // SLOW PATH: Per-thread atomics
    if (is_valid) {
      atomicAdd_size_type(&output_ptrs[col_idx][my_group], 1);
    }
  }
}

/**
 * @brief Batched MIN kernel
 *
 * Finds minimum value for each group across multiple columns.
 */
template <typename T>
__global__ void batch_min_kernel(
    T const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    return;
  }
  
  auto const value = value_ptrs[col_idx][row_idx];
  auto const group = group_labels[row_idx];
  auto* output = output_ptrs[col_idx];
  
  // Atomic min for the group
  atomicMin(&output[group], value);
}

/**
 * @brief Batched MAX kernel
 *
 * Finds maximum value for each group across multiple columns.
 */
template <typename T>
__global__ void batch_max_kernel(
    T const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    return;
  }
  
  auto const value = value_ptrs[col_idx][row_idx];
  auto const group = group_labels[row_idx];
  auto* output = output_ptrs[col_idx];
  
  // Atomic max for the group
  atomicMax(&output[group], value);
}

/**
 * @brief Batched aggregation for SUM + COUNT (used to compute AVG)
 *
 * Computes both sum and count in a single kernel pass, which is used to
 * derive AVG = SUM / COUNT afterward.
 */
template <typename T>
__global__ void batch_sum_count_kernel(
    T const* const* value_ptrs,
    cudf::bitmask_type const* const* null_masks,
    cudf::size_type const* group_labels,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    cudf::size_type num_groups,
    double* const* sum_output_ptrs,
    int64_t* const* count_output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  auto const null_mask = null_masks[col_idx];
  if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) {
    return;
  }
  
  auto const value = static_cast<double>(value_ptrs[col_idx][row_idx]);
  auto const group = group_labels[row_idx];
  
  atomicAdd(&sum_output_ptrs[col_idx][group], value);
  atomicAdd_int64(&count_output_ptrs[col_idx][group], static_cast<int64_t>(1));
}

/**
 * @brief Compute AVG from precomputed SUM and COUNT
 */
__global__ void batch_compute_avg_kernel(
    double const* const* sum_ptrs,
    int64_t const* const* count_ptrs,
    cudf::size_type num_groups,
    cudf::size_type num_cols,
    double* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const group_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (group_idx >= num_groups) return;
  
  auto const sum = sum_ptrs[col_idx][group_idx];
  auto const count = count_ptrs[col_idx][group_idx];
  
  output_ptrs[col_idx][group_idx] = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
}

/**
 * @brief Batched null count kernel (like PR 20872)
 *
 * Counts null values for multiple bitmasks in a single kernel call.
 */
__global__ void batch_null_count_kernel(
    cudf::bitmask_type const* const* bitmasks,
    cudf::size_type start,
    cudf::size_type stop,
    cudf::size_type num_bitmasks,
    cudf::size_type* output_counts)
{
  auto const bitmask_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (bitmask_idx >= num_bitmasks) return;
  
  auto const bitmask = bitmasks[bitmask_idx];
  if (bitmask == nullptr) return;  // All valid, count stays 0
  
  auto const bit_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x) + start;
  if (bit_idx >= stop) return;
  
  // Count unset bit (null)
  if (!cudf::bit_is_set(bitmask, bit_idx)) {
    atomicAdd_size_type(&output_counts[bitmask_idx], static_cast<cudf::size_type>(1));
  }
}

/**
 * @brief Initialize output buffers with identity values
 *
 * - SUM: 0
 * - COUNT: 0
 * - MIN: type max
 * - MAX: type min
 */
template <typename T>
__global__ void init_identity_kernel(
    T* const* output_ptrs,
    cudf::size_type num_groups,
    cudf::size_type num_cols,
    T identity_value)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const group_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (group_idx >= num_groups) return;
  
  output_ptrs[col_idx][group_idx] = identity_value;
}

// ============================================================================
// Batched Transform Kernels
// 
// These kernels process multiple transform operations in a SINGLE kernel launch
// using a 2D grid, reducing kernel launch overhead significantly for operations
// like COALESCE, IF_ELSE, etc. that are common in pre-project stages.
// ============================================================================

/**
 * @brief Batched COALESCE kernel with scalar defaults
 *
 * Replaces NULL values with a scalar default value. Processes multiple columns
 * in a single kernel launch using a 2D grid:
 *   - gridDim.y = number of columns
 *   - gridDim.x * blockDim.x = threads to process rows
 *
 * This significantly reduces kernel launch overhead compared to calling
 * individual coalesce operations for each column.
 *
 * @tparam T Data type (int64_t, double, etc.)
 * @param input_ptrs Array of pointers to input columns
 * @param null_masks Array of pointers to null masks (can be nullptr = all valid)
 * @param default_vals Array of default values (one per column)
 * @param num_rows Number of input rows
 * @param num_cols Number of columns to process
 * @param output_ptrs Array of pointers to output columns
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
  
  // COALESCE: return default if null, otherwise return input value
  output_ptrs[col_idx][row_idx] = is_null ? default_vals[col_idx] : input_ptrs[col_idx][row_idx];
}

/**
 * @brief Batched COALESCE kernel with column defaults
 *
 * Replaces NULL values with values from another column. Each input column
 * has a corresponding default column.
 *
 * @tparam T Data type
 * @param input_ptrs Array of pointers to primary input columns
 * @param input_null_masks Array of pointers to null masks for primary inputs
 * @param default_ptrs Array of pointers to default value columns
 * @param num_rows Number of input rows
 * @param num_cols Number of columns to process
 * @param output_ptrs Array of pointers to output columns
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
  
  // COALESCE: return default column value if null, otherwise return input
  output_ptrs[col_idx][row_idx] = is_null ? default_ptrs[col_idx][row_idx] : input_ptrs[col_idx][row_idx];
}

/**
 * @brief Batched IF_ELSE kernel (ternary conditional)
 *
 * Implements: output = condition ? true_val : false_val
 * for multiple columns in a single kernel launch.
 *
 * @tparam T Data type for true/false/output values
 * @param condition_ptrs Array of pointers to boolean condition columns
 * @param true_ptrs Array of pointers to true-case value columns
 * @param false_ptrs Array of pointers to false-case value columns
 * @param num_rows Number of rows
 * @param num_cols Number of column-sets to process
 * @param output_ptrs Array of pointers to output columns
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
 * @brief Batched IF_ELSE kernel with scalar false value
 *
 * Optimized for common pattern: if (cond) col else 0
 *
 * @tparam T Data type
 * @param condition_ptrs Array of pointers to boolean condition columns
 * @param true_ptrs Array of pointers to true-case value columns
 * @param false_vals Array of scalar false values (one per column)
 * @param num_rows Number of rows
 * @param num_cols Number of column-sets
 * @param output_ptrs Array of pointers to output columns
 */
template <typename T>
__global__ void batch_if_else_scalar_false_kernel(
    bool const* const* condition_ptrs,
    T const* const* true_ptrs,
    T const* false_vals,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  bool const cond = condition_ptrs[col_idx][row_idx];
  output_ptrs[col_idx][row_idx] = cond ? true_ptrs[col_idx][row_idx] : false_vals[col_idx];
}

/**
 * @brief Batched multiply kernel: out = a * b
 *
 * Common pattern in pre-project: coalesce(x,0) * coalesce(y,0)
 * This kernel can be fused with coalesce for even better performance.
 *
 * @tparam T Data type
 * @param a_ptrs Array of pointers to first operand columns
 * @param b_ptrs Array of pointers to second operand columns
 * @param num_rows Number of rows
 * @param num_cols Number of columns
 * @param output_ptrs Array of pointers to output columns
 */
template <typename T>
__global__ void batch_multiply_kernel(
    T const* const* a_ptrs,
    T const* const* b_ptrs,
    cudf::size_type num_rows,
    cudf::size_type num_cols,
    T* const* output_ptrs)
{
  auto const col_idx = static_cast<cudf::size_type>(blockIdx.y);
  if (col_idx >= num_cols) return;
  
  auto const row_idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;
  
  output_ptrs[col_idx][row_idx] = a_ptrs[col_idx][row_idx] * b_ptrs[col_idx][row_idx];
}

/**
 * @brief Fused COALESCE + MULTIPLY kernel
 *
 * Computes: output = coalesce(a, default_a) * coalesce(b, default_b)
 * in a single pass. This is a very common pattern in multi-column aggregation workloads.
 *
 * @tparam T Data type
 * @param a_ptrs First operand columns
 * @param a_null_masks Null masks for first operands
 * @param default_a Default values for first operands
 * @param b_ptrs Second operand columns
 * @param b_null_masks Null masks for second operands
 * @param default_b Default values for second operands
 * @param num_rows Number of rows
 * @param num_cols Number of columns
 * @param output_ptrs Output columns
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

