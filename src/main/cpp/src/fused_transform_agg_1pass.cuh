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

#pragma once

#include "fused_transform_aggregate.hpp"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <cstdint>

namespace spark_rapids_jni {
namespace fused_1pass {

// ============================================================================
// CONFIGURATION
// ============================================================================

// Shared memory hash table size per block
constexpr int SHARED_HT_SIZE = 1024;  // Number of slots in shared memory hash table
constexpr int BLOCK_SIZE = 256;
constexpr int ROWS_PER_THREAD = 4;

// ============================================================================
// HASH TABLE SLOT
// ============================================================================

struct HashSlot {
  int64_t key;           // Group key (-1 = empty)
  int64_t sum;           // Sum accumulator
  int64_t count;         // Count accumulator
  int64_t sum_sq;        // Sum of squares (for variance)
  int64_t min_val;       // Min value
  int64_t max_val;       // Max value
};

// ============================================================================
// TRANSFORM FUNCTORS
// ============================================================================

template <TransformOp Op>
struct TransformFunctor {
  __device__ __forceinline__ int64_t operator()(
      int64_t val, int64_t val2, int64_t default_val, bool is_null) const;
};

template <>
struct TransformFunctor<TransformOp::IDENTITY> {
  __device__ __forceinline__ int64_t operator()(
      int64_t val, int64_t /*val2*/, int64_t default_val, bool is_null) const {
    return is_null ? default_val : val;
  }
};

template <>
struct TransformFunctor<TransformOp::COALESCE> {
  __device__ __forceinline__ int64_t operator()(
      int64_t val, int64_t /*val2*/, int64_t default_val, bool is_null) const {
    return is_null ? default_val : val;
  }
};

template <>
struct TransformFunctor<TransformOp::COALESCE_MUL_SELF> {
  __device__ __forceinline__ int64_t operator()(
      int64_t val, int64_t /*val2*/, int64_t default_val, bool is_null) const {
    int64_t v = is_null ? default_val : val;
    return v * v;
  }
};

template <>
struct TransformFunctor<TransformOp::COALESCE_MUL_OTHER> {
  __device__ __forceinline__ int64_t operator()(
      int64_t val, int64_t val2, int64_t default_val, bool is_null) const {
    int64_t v1 = is_null ? default_val : val;
    // Assume val2 uses same null handling for simplicity
    return v1 * val2;
  }
};

// ============================================================================
// FUSED TRANSFORM + 1ST PASS AGGREGATION KERNEL
// ============================================================================

/**
 * @brief Fused kernel that combines transform and partial aggregation
 * 
 * Each block:
 * 1. Reads a chunk of input rows
 * 2. Applies transform in registers (no intermediate column!)
 * 3. Aggregates into shared memory hash table
 * 4. Flushes partial results to global memory
 * 
 * This eliminates the intermediate column write/read!
 */
template <TransformOp transform_op>
__global__ void fused_transform_agg_1pass_kernel(
    // Input
    int64_t const* __restrict__ keys,
    int64_t const* __restrict__ values,
    int64_t const* __restrict__ values2,     // For MUL_OTHER
    uint32_t const* __restrict__ null_mask,  // Null bitmask
    int64_t num_rows,
    int64_t default_val,
    
    // Output: partial aggregates (one row per block per unique key in block)
    int64_t* __restrict__ out_keys,
    int64_t* __restrict__ out_sums,
    int64_t* __restrict__ out_counts,
    int64_t* __restrict__ out_sum_sqs,
    int32_t* __restrict__ out_num_entries,  // Number of entries per block
    
    int64_t num_groups_hint)  // Hint for total number of groups
{
  namespace cg = cooperative_groups;
  
  // Shared memory hash table
  __shared__ HashSlot shared_ht[SHARED_HT_SIZE];
  __shared__ int32_t num_entries;
  
  // Initialize shared memory
  for (int i = threadIdx.x; i < SHARED_HT_SIZE; i += BLOCK_SIZE) {
    shared_ht[i].key = -1;  // Empty marker
    shared_ht[i].sum = 0;
    shared_ht[i].count = 0;
    shared_ht[i].sum_sq = 0;
    shared_ht[i].min_val = INT64_MAX;
    shared_ht[i].max_val = INT64_MIN;
  }
  if (threadIdx.x == 0) {
    num_entries = 0;
  }
  __syncthreads();
  
  // Calculate row range for this block
  int64_t rows_per_block = BLOCK_SIZE * ROWS_PER_THREAD;
  int64_t start_row = blockIdx.x * rows_per_block;
  int64_t end_row = min(start_row + rows_per_block, num_rows);
  
  // Transform functor
  TransformFunctor<transform_op> transform;
  
  // Process rows
  for (int64_t row = start_row + threadIdx.x; row < end_row; row += BLOCK_SIZE) {
    // Read key
    int64_t key = keys[row];
    
    // Read value and check null
    int64_t val = values[row];
    int64_t val2 = values2 ? values2[row] : 0;
    bool is_null = null_mask && !cudf::bit_is_set(null_mask, row);
    
    // Apply transform IN REGISTERS (no intermediate column!)
    int64_t transformed = transform(val, val2, default_val, is_null);
    
    // Hash into shared memory table
    uint32_t hash = static_cast<uint32_t>(key) % SHARED_HT_SIZE;
    
    // Linear probing
    for (int probe = 0; probe < SHARED_HT_SIZE; ++probe) {
      uint32_t slot = (hash + probe) % SHARED_HT_SIZE;
      
      int64_t old_key = atomicCAS(
          reinterpret_cast<unsigned long long*>(&shared_ht[slot].key),
          static_cast<unsigned long long>(-1),
          static_cast<unsigned long long>(key));
      
      if (old_key == -1 || old_key == key) {
        // Found slot - aggregate
        atomicAdd(reinterpret_cast<unsigned long long*>(&shared_ht[slot].sum),
                  static_cast<unsigned long long>(transformed));
        atomicAdd(reinterpret_cast<unsigned long long*>(&shared_ht[slot].count),
                  1ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(&shared_ht[slot].sum_sq),
                  static_cast<unsigned long long>(transformed * transformed));
        
        if (old_key == -1) {
          // New entry
          atomicAdd(&num_entries, 1);
        }
        break;
      }
      // Collision - continue probing
    }
  }
  
  __syncthreads();
  
  // Flush shared memory to global memory
  int64_t block_output_base = blockIdx.x * SHARED_HT_SIZE;
  
  for (int i = threadIdx.x; i < SHARED_HT_SIZE; i += BLOCK_SIZE) {
    if (shared_ht[i].key != -1) {
      int64_t out_idx = block_output_base + i;
      out_keys[out_idx] = shared_ht[i].key;
      out_sums[out_idx] = shared_ht[i].sum;
      out_counts[out_idx] = shared_ht[i].count;
      out_sum_sqs[out_idx] = shared_ht[i].sum_sq;
    }
  }
  
  if (threadIdx.x == 0) {
    out_num_entries[blockIdx.x] = num_entries;
  }
}

// ============================================================================
// SIMPLE VERSION: Direct atomic to global memory
// For when number of groups is small enough to fit in L2 cache
// ============================================================================

template <TransformOp transform_op>
__global__ void fused_transform_agg_direct_kernel(
    // Input
    int64_t const* __restrict__ keys,
    int64_t const* __restrict__ values,
    int64_t const* __restrict__ values2,
    uint32_t const* __restrict__ null_mask,
    int64_t num_rows,
    int64_t default_val,
    
    // Output: direct aggregation (pre-allocated for all groups)
    int64_t* __restrict__ out_sums,
    int64_t* __restrict__ out_counts,
    int64_t* __restrict__ out_sum_sqs)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  
  TransformFunctor<transform_op> transform;
  
  for (int64_t row = tid; row < num_rows; row += stride) {
    // Read key
    int64_t key = keys[row];
    
    // Read value and check null
    int64_t val = values[row];
    int64_t val2 = values2 ? values2[row] : 0;
    bool is_null = null_mask && !cudf::bit_is_set(null_mask, row);
    
    // Apply transform IN REGISTERS
    int64_t transformed = transform(val, val2, default_val, is_null);
    
    // Direct atomic to output (assumes key IS the index, i.e., dense keys 0..N-1)
    atomicAdd(reinterpret_cast<unsigned long long*>(&out_sums[key]),
              static_cast<unsigned long long>(transformed));
    atomicAdd(reinterpret_cast<unsigned long long*>(&out_counts[key]), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(&out_sum_sqs[key]),
              static_cast<unsigned long long>(transformed * transformed));
  }
}

// ============================================================================
// WARP-LEVEL REDUCTION VERSION
// Uses warp shuffle for intra-warp reduction before atomic
// ============================================================================

template <TransformOp transform_op>
__global__ void fused_transform_agg_warp_kernel(
    int64_t const* __restrict__ keys,
    int64_t const* __restrict__ values,
    int64_t const* __restrict__ values2,
    uint32_t const* __restrict__ null_mask,
    int64_t num_rows,
    int64_t default_val,
    int64_t* __restrict__ out_sums,
    int64_t* __restrict__ out_counts)
{
  namespace cg = cooperative_groups;
  
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  
  TransformFunctor<transform_op> transform;
  
  // Process multiple rows per thread for better occupancy
  for (int64_t row = tid; row < num_rows; row += stride) {
    int64_t key = keys[row];
    int64_t val = values[row];
    int64_t val2 = values2 ? values2[row] : 0;
    bool is_null = null_mask && !cudf::bit_is_set(null_mask, row);
    
    // Transform in registers
    int64_t transformed = transform(val, val2, default_val, is_null);
    
    // Warp-level reduction for threads with same key
    auto warp = cg::coalesced_threads();
    unsigned int warp_mask = __match_any_sync(0xFFFFFFFF, key);
    
    // Find threads in warp with same key
    auto group = cg::labeled_partition(warp, key);
    
    // Reduce within group
    int64_t local_sum = transformed;
    int64_t local_count = 1;
    
    for (int offset = group.size() / 2; offset > 0; offset /= 2) {
      local_sum += group.shfl_down(local_sum, offset);
      local_count += group.shfl_down(local_count, offset);
    }
    
    // Only leader writes
    if (group.thread_rank() == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&out_sums[key]),
                static_cast<unsigned long long>(local_sum));
      atomicAdd(reinterpret_cast<unsigned long long*>(&out_counts[key]),
                static_cast<unsigned long long>(local_count));
    }
  }
}

}  // namespace fused_1pass
}  // namespace spark_rapids_jni


