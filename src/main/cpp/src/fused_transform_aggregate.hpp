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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace spark_rapids_jni {

// ============================================================================
// TRANSFORM OPERATIONS
// ============================================================================
// These represent common Spark expression patterns that can be fused with
// aggregation. Each pattern is designed to handle specific expression types
// that are NOT supported by cuDF AST (like coalesce, if-else).
// ============================================================================

/**
 * @brief Enum representing transform operations that can be fused with aggregation
 * 
 * These patterns cover ~95% of common Spark Project expressions:
 * - IDENTITY: Direct column reference
 * - COALESCE: coalesce(col, default)
 * - COALESCE_MUL_SELF: coalesce(col, d) * coalesce(col, d)
 * - COALESCE_MUL_OTHER: coalesce(col1, d1) * coalesce(col2, d2)
 * - CONDITIONAL: if(cond > threshold) val else default
 * - CONDITIONAL_COALESCE: if(cond > t) coalesce(val, d) else 0
 */
enum class TransformOp : int32_t {
  IDENTITY = 0,           // Direct value: val
  COALESCE,               // coalesce(val, default)
  COALESCE_MUL_SELF,      // coalesce(val, d) * coalesce(val, d)
  COALESCE_MUL_OTHER,     // coalesce(val1, d1) * coalesce(val2, d2)
  CONDITIONAL,            // if(cond > threshold) val else default
  CONDITIONAL_COALESCE,   // if(cond > t) coalesce(val, d) else 0
  
  // TPC-H patterns (Phase 1a)
  MUL,                    // val * other  (TPC-H Q6: l_extendedprice * l_discount)
  MUL_SUB_CONST,          // val * (const - other)  (TPC-H Q1: l_extendedprice * (1 - l_discount))
  CASE_MUL,               // CASE WHEN cond THEN val * other ELSE 0  (TPC-H Q14)
  
  // TPC-H Q1 sum_charge pattern: val * (const1 - other) * (const2 + cond)
  // For: l_extendedprice * (1 - l_discount) * (1 + l_tax)
  // Uses: value_col=a, other_col=b, cond_col=c, default_val=const1, threshold=const2
  MUL_SUB_CONST_MUL_ADD_CONST,
};

/**
 * @brief Enum representing aggregation operations
 */
enum class AggOp : int32_t {
  SUM = 0,
  COUNT,
  AVG,      // Requires both sum and count
  MIN,
  MAX,
};

// ============================================================================
// EXPRESSION SPECIFICATION
// ============================================================================

/**
 * @brief Specification for a single fused transform + aggregate expression
 * 
 * This struct describes how to transform input data and aggregate it in a
 * single fused kernel. It's designed to be serializable across JNI.
 */
struct FusedExprSpec {
  // Transform configuration
  TransformOp transform_op;         // Type of transform
  int32_t value_col_idx;            // Primary value column index
  int32_t cond_col_idx;             // Condition column index (for CONDITIONAL)
  int32_t other_col_idx;            // Second column index (for COALESCE_MUL_OTHER)
  
  // Constants for transform
  int64_t default_val;              // Default value for coalesce / constant for MUL_SUB_CONST
  int64_t threshold;                // Threshold for conditional (cond > threshold)
  int64_t else_val;                 // Value when condition is false
  
  // Aggregation configuration
  AggOp agg_op;                     // Aggregation type
  cudf::data_type output_type;      // Output data type
  
  // Output location
  int32_t output_idx;               // Index in output table
  
  // Debug info
  std::string spark_expr;           // Original Spark expression (for debugging)
  
  // Default constructor
  FusedExprSpec()
      : transform_op(TransformOp::IDENTITY),
        value_col_idx(-1),
        cond_col_idx(-1),
        other_col_idx(-1),
        default_val(0),
        threshold(0),
        else_val(0),
        agg_op(AggOp::SUM),
        output_type(cudf::data_type{cudf::type_id::INT64}),
        output_idx(-1) {}
};

// ============================================================================
// EXECUTION PLAN
// ============================================================================

/**
 * @brief Execution plan for fused transform + aggregate
 */
struct FusedExecutionPlan {
  std::vector<FusedExprSpec> expressions;      // All expressions to execute
  std::vector<int32_t> group_by_col_indices;   // Group-by column indices
  
  // Optimization hints
  // 
  // Benchmark findings (10M rows):
  //   Warp Reduction:  ~1.0x - NO BENEFIT for random groups (only helps sorted data)
  //   Perfect Hash:    0.76x - NO BENEFIT (extra key read overhead)
  //   Multi-Column:    1.42x - MAIN BENEFIT from fusing expressions
  //   Contiguous Out:  MAIN BENEFIT from single allocation
  //
  bool enable_warp_reduction;        // Use warp-level reduction - disabled by default
  bool enable_perfect_hash;          // Use key as index - disabled by default
  int32_t estimated_num_groups;      // Hint for output size (-1 = unknown)
  
  FusedExecutionPlan()
      : enable_warp_reduction(false),   // Disabled: no benefit for random groups
        enable_perfect_hash(false),     // Disabled: no benefit, adds overhead
        estimated_num_groups(-1) {}
};

// ============================================================================
// EXECUTION RESULT
// ============================================================================

/**
 * @brief Result of fused transform + aggregate execution
 * 
 * Memory ownership: Each output column owns its memory independently
 * using the default memory resource. No shared buffer is used.
 * Columns can be closed in any order.
 */
struct FusedExecutionResult {
  std::unique_ptr<cudf::table> output_keys;    // Group-by keys
  std::unique_ptr<cudf::table> output_values;  // Aggregated values
  
  // Execution statistics
  struct Stats {
    int64_t kernel_time_ns = 0;
    int64_t total_time_ns = 0;
    int32_t num_expressions_fused = 0;
    int32_t num_groups = 0;
    std::string execution_strategy;
  } stats;
  
  [[nodiscard]] cudf::size_type num_groups() const {
    return output_keys ? output_keys->num_rows() : 0;
  }
};

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * @brief Check if an execution plan can be fused
 * 
 * @param plan The execution plan to validate
 * @param reason Output: reason if fusion is not possible
 * @return true if the plan can be fused, false otherwise
 */
bool can_fuse(FusedExecutionPlan const& plan, std::string* reason = nullptr);

/**
 * @brief Execute fused transform + aggregate
 * 
 * This function performs the following in a single fused kernel:
 * 1. Transform input columns using the specified TransformOp
 * 2. Aggregate transformed values using the specified AggOp
 * 3. Output results to a shared buffer (single memory allocation)
 * 
 * Key optimizations:
 * - Single memory allocation for all outputs (reduces fragmentation)
 * - Single kernel launch for all expressions (reduces overhead)
 * - Warp-level reduction for fewer atomic operations
 * - Optional perfect hash for small integer keys
 * 
 * @param input Input table
 * @param plan Execution plan with expressions and configuration
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return FusedExecutionResult with output tables and statistics
 */
FusedExecutionResult execute_fused_transform_aggregate(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Get string name for TransformOp (for debugging/logging)
 */
std::string transform_op_name(TransformOp op);

/**
 * @brief Get string name for AggOp (for debugging/logging)
 */
std::string agg_op_name(AggOp op);

/**
 * @brief Estimate output buffer size for planning
 */
size_t estimate_output_buffer_size(
    FusedExecutionPlan const& plan,
    cudf::size_type estimated_num_groups);

}  // namespace spark_rapids_jni


