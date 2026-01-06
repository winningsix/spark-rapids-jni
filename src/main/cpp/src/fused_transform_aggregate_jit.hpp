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

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string>
#include <vector>

namespace spark_rapids_jni {
namespace jit {

// ============================================================================
// JIT-BASED TRANSFORM EXPRESSION BUILDER
// ============================================================================

/**
 * @brief Builds a cudf AST expression from a FusedExprSpec
 * 
 * This class converts our FusedExprSpec transform operations into cudf AST
 * expressions that can be JIT-compiled for efficient execution.
 * 
 * Supported transforms:
 * - IDENTITY: column[col_idx]
 * - COALESCE: coalesce(column[col_idx], default_val)  
 * - COALESCE_MUL_SELF: coalesce(col, d) * coalesce(col, d)
 * - COALESCE_MUL_OTHER: coalesce(col1, d1) * coalesce(col2, d2)
 * - CONDITIONAL: if(cond > threshold, val, else_val)
 * - CONDITIONAL_COALESCE: if(cond > t, coalesce(val, d), 0)
 */
class JitExpressionBuilder {
public:
  /**
   * @brief Build a cudf AST expression for the given FusedExprSpec
   * 
   * @param spec The expression specification
   * @param input Input table for type information
   * @return A pair of (expression tree, root expression reference)
   */
  static std::pair<cudf::ast::tree, cudf::ast::expression const*>
  build_expression(FusedExprSpec const& spec, cudf::table_view const& input);

  /**
   * @brief Generate CUDA UDF string for the transform operation
   * 
   * This method generates a CUDA kernel string that can be passed to
   * cudf::transform() for JIT compilation.
   * 
   * @param spec The expression specification
   * @param input Input table for type information
   * @return CUDA UDF string
   */
  static std::string generate_cuda_udf(FusedExprSpec const& spec,
                                       cudf::table_view const& input);

  /**
   * @brief Check if the transform can be expressed as a cudf AST
   * 
   * Some complex transforms might require falling back to custom CUDA UDF.
   */
  static bool can_use_ast(FusedExprSpec const& spec);
};

// ============================================================================
// JIT-BASED EXECUTION
// ============================================================================

/**
 * @brief Execute fused transform + aggregate using cudf JIT
 * 
 * This implementation uses cudf's JIT infrastructure instead of hand-written
 * CUDA kernels. The approach:
 * 
 * 1. For each expression, build a cudf AST or CUDA UDF
 * 2. Use cudf::compute_column_jit() or cudf::transform() for the transform step
 * 3. Use cudf::groupby for aggregation
 * 
 * Benefits:
 * - Leverages cudf's optimized JIT compilation and caching
 * - More flexible expression support
 * - Easier to extend and maintain
 * - Automatic type handling
 * 
 * Trade-offs:
 * - Transform and aggregate are not fused into single kernel
 * - May have higher memory usage for intermediate results
 * - JIT compilation overhead for first run (cached thereafter)
 * 
 * @param input Input table
 * @param plan Execution plan with expressions and configuration
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return FusedExecutionResult with output tables and statistics
 */
FusedExecutionResult execute_fused_transform_aggregate_jit(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Execute using hybrid approach: AST for simple transforms, UDF for complex
 * 
 * This method automatically chooses the best execution path:
 * - Simple transforms (IDENTITY, basic math) -> cudf AST
 * - Complex transforms (COALESCE, CONDITIONAL) -> Custom CUDA UDF
 * 
 * @param input Input table
 * @param plan Execution plan with expressions and configuration
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return FusedExecutionResult with output tables and statistics
 */
FusedExecutionResult execute_fused_transform_aggregate_hybrid(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ============================================================================
// BENCHMARKING AND COMPARISON
// ============================================================================

/**
 * @brief Execution mode for comparison
 */
enum class ExecutionMode {
  HAND_WRITTEN_KERNEL,  // Original hand-written CUDA kernel
  JIT_AST,              // cudf AST-based JIT
  JIT_UDF,              // cudf CUDA UDF-based JIT  
  HYBRID,               // Automatic selection
};

/**
 * @brief Execute with specified mode for benchmarking
 */
FusedExecutionResult execute_with_mode(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    ExecutionMode mode,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Benchmark result for comparison
 */
struct BenchmarkResult {
  ExecutionMode mode;
  double transform_time_ms;
  double aggregate_time_ms;
  double total_time_ms;
  size_t peak_memory_bytes;
  bool jit_cache_hit;
};

/**
 * @brief Run benchmark comparing different execution modes
 */
std::vector<BenchmarkResult> benchmark_execution_modes(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    int warmup_iterations = 3,
    int benchmark_iterations = 10,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace jit
}  // namespace spark_rapids_jni


