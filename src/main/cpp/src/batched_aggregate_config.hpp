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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <string>

namespace spark_rapids_jni {

// =============================================================================
// BATCHED AGGREGATION CONFIGURATION
// =============================================================================
// Each optimization can be independently enabled/disabled for verification.
// Spark-Rapids exposes these via:
//   spark.rapids.sql.batchedAggregate.xxx
// =============================================================================

/**
 * @brief Configuration for batched aggregation optimizations
 * 
 * Each flag corresponds to a Spark config:
 *   spark.rapids.sql.batchedAggregate.enabled              - Master switch
 *   spark.rapids.sql.batchedAggregate.warpReduction        - Warp-level reduction
 *   spark.rapids.sql.batchedAggregate.contiguousOutput     - Single buffer allocation
 *   spark.rapids.sql.batchedAggregate.sharedGroupby        - Reuse get_groups() result
 *   spark.rapids.sql.batchedAggregate.perfectHash          - Direct key-as-index
 *   spark.rapids.sql.batchedAggregate.prefetch             - GPU memory prefetch
 *   spark.rapids.sql.batchedAggregate.adaptive             - Auto-select best strategy
 */
struct BatchedAggConfig {
  // Master switch
  bool enabled = true;
  
  // Individual optimizations (can be toggled independently)
  // 
  // Benchmark Results (10M rows):
  //   Shared Memory:  2.69x speedup ONLY for < 128 groups, SLOWER otherwise
  //   Perfect Hash:   0.76x (no benefit, extra key read overhead)
  //   L2 Cache:       1.0x  (no benefit, GPU already optimizes)
  //   Warp Reduction: No benefit for random groups (only helps sorted data)
  //   Multi-Column:   1.42x speedup for low cardinality
  //
  bool warp_reduction_enabled = false;     // Warp-level reduction - disabled by default (no benefit)
  bool contiguous_output_enabled = true;   // Single buffer for all outputs - MAIN BENEFIT
  bool shared_groupby_enabled = true;      // Reuse get_groups() results - MAIN BENEFIT
  bool perfect_hash_enabled = false;       // Direct key-as-index - disabled (no benefit)
  bool prefetch_enabled = false;           // GPU memory prefetch - disabled (no benefit)
  bool adaptive_enabled = true;            // Auto-select best strategy
  bool shmem_agg_enabled = false;          // Shared memory aggregation - only for low cardinality
  
  // Thresholds for adaptive strategy
  int32_t min_columns_for_batching = 4;    // Min columns to enable batching
  int32_t perfect_hash_max_keys = 1000000; // Max key range for perfect hash (unused)
  int32_t warp_reduction_min_rows = 10000; // Min rows for warp reduction benefit (unused)
  int32_t prefetch_min_rows = 100000;      // Min rows for prefetch benefit (unused)
  int32_t shmem_max_groups = 128;          // Max groups for shared memory benefit
  
  // Debug/logging
  bool debug_logging = false;
  
  /**
   * @brief Create config from environment variables (for testing)
   */
  static BatchedAggConfig from_env();
  
  /**
   * @brief Create config from Spark conf map
   */
  static BatchedAggConfig from_spark_conf(
      bool enabled,
      bool warp_reduction,
      bool contiguous_output,
      bool shared_groupby,
      bool perfect_hash,
      bool prefetch,
      bool adaptive,
      int32_t min_columns,
      int32_t perfect_hash_max_keys,
      int32_t warp_reduction_min_rows);
      
  /**
   * @brief Get default configuration
   */
  static BatchedAggConfig defaults() { return BatchedAggConfig{}; }
};

// =============================================================================
// ADAPTIVE STRATEGY SELECTION
// =============================================================================
// Automatically selects the best execution strategy based on:
// - Number of aggregation columns
// - Key cardinality and type
// - Data size
// - GPU characteristics
// =============================================================================

/**
 * @brief Execution strategy for batched aggregation
 */
enum class AggStrategy : int32_t {
  CUDF_DEFAULT = 0,    // Use standard cudf::groupby::aggregate
  BATCHED_KERNEL,      // Batched kernels with warp reduction
  PERFECT_HASH,        // Direct key-as-index (for small integer keys)
  HYBRID,              // Mix of strategies based on column types
};

/**
 * @brief Analysis results for adaptive strategy selection
 */
struct DataCharacteristics {
  cudf::size_type num_rows;
  cudf::size_type num_columns;
  cudf::size_type num_groups_estimate;
  bool single_integer_key;
  int64_t key_min;
  int64_t key_max;
  double key_density;  // (num_groups / (max - min + 1))
  bool has_nulls;
  
  /**
   * @brief Analyze input data characteristics
   */
  static DataCharacteristics analyze(
      cudf::table_view const& keys,
      cudf::size_type num_value_columns);
};

/**
 * @brief Select optimal execution strategy based on data and config
 */
AggStrategy select_strategy(
    DataCharacteristics const& data,
    BatchedAggConfig const& config);

/**
 * @brief Get human-readable strategy name (for logging)
 */
std::string strategy_name(AggStrategy strategy);

// =============================================================================
// GPU PREFETCH OPTIMIZATION
// =============================================================================
// Prefetch data to GPU for better memory access patterns.
// Most effective when:
// - Data is large (> 100K rows)
// - Multiple passes over data
// - Unified memory is used
// =============================================================================

/**
 * @brief Prefetch configuration
 */
struct PrefetchConfig {
  bool enabled = false;
  size_t prefetch_size = 0;  // 0 = auto-calculate
  int device_id = 0;
};

/**
 * @brief Prefetch input columns to GPU
 * 
 * Uses cudaMemPrefetchAsync for unified memory, or async copy for pinned memory.
 * For pure device memory, this is a no-op.
 */
void prefetch_input_data(
    cudf::table_view const& keys,
    cudf::table_view const& values,
    PrefetchConfig const& config,
    rmm::cuda_stream_view stream);

// =============================================================================
// SPARK-RAPIDS INTEGRATION
// =============================================================================
// JNI entry points for Spark-Rapids integration.
// These are called from GpuAggregateExec via BatchedAggregate.java
// =============================================================================

/**
 * @brief Check if batched aggregation should be used
 * 
 * Called from Java to determine if batched path is beneficial.
 * 
 * @param num_agg_columns Number of aggregation columns
 * @param num_rows Estimated number of input rows
 * @param key_type_id cudf type ID of the groupby key
 * @param config Configuration from Spark conf
 * @return true if batched aggregation should be used
 */
bool should_use_batched_aggregation(
    int32_t num_agg_columns,
    int64_t num_rows,
    int32_t key_type_id,
    BatchedAggConfig const& config);

/**
 * @brief Get recommended strategy as integer (for JNI)
 */
int32_t get_recommended_strategy(
    int32_t num_agg_columns,
    int64_t num_rows,
    int32_t key_type_id,
    int64_t key_min,
    int64_t key_max,
    BatchedAggConfig const& config);

}  // namespace spark_rapids_jni

