/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * @file fused_ast_aggregate.hpp
 * @brief Fused AST Expression Evaluation + Aggregation
 * 
 * This is the LONG-TERM solution for fusing arbitrary expressions with aggregation.
 * It replaces the fixed-pattern TransformOp approach with a general AST evaluator.
 * 
 * Architecture:
 * 
 *   Spark SQL Expression
 *         ↓
 *   GpuExpression.convertToAst()
 *         ↓
 *   Serialized AST (bytes)
 *         ↓ (JNI)
 *   cudf::ast::expression
 *         ↓
 *   ┌─────────────────────────────────────────┐
 *   │  Fused Kernel                           │
 *   │                                         │
 *   │  for each row:                          │
 *   │    1. ast_evaluator.evaluate(row)       │
 *   │       → result in register              │
 *   │    2. aggregator.accumulate(result)     │
 *   │       → no global memory write!         │
 *   │                                         │
 *   └─────────────────────────────────────────┘
 *         ↓
 *   Aggregated Result
 * 
 * Benefits over Handwrite Fuse:
 * - Supports ANY cudf AST-compatible expression
 * - No need to add new TransformOp for new patterns
 * - Leverages cudf's existing AST infrastructure
 * - Automatic type handling and null propagation
 * 
 * Performance Target:
 * - Should match Handwrite Fuse performance (within 5%)
 * - Much better than separate AST + Aggregate (currently 7% slower)
 */

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {
namespace ast_fused {

// ============================================================================
// AST EXPRESSION SPEC FOR JNI SERIALIZATION
// ============================================================================

/**
 * @brief Specification for a single fused AST expression + aggregation
 * 
 * This struct carries the serialized AST expression and aggregation config
 * across the JNI boundary.
 */
struct AstAggSpec {
  // Serialized cudf AST expression (from CompiledExpression.serialize())
  std::vector<uint8_t> serialized_ast;
  
  // Aggregation type
  enum class AggType : int32_t {
    SUM = 0,
    COUNT,
    MIN,
    MAX,
    // AVG is implemented as SUM + COUNT
  };
  AggType agg_type;
  
  // Output column index
  int32_t output_idx;
  
  // Data type info
  cudf::data_type output_dtype;
};

/**
 * @brief Execution plan for fused AST + aggregation
 */
struct AstFusedPlan {
  // Group-by key column indices
  std::vector<int32_t> key_indices;
  
  // AST expression + aggregation specs
  std::vector<AstAggSpec> agg_specs;
  
  // Execution hints
  bool enable_warp_reduction = true;
  int32_t target_rows_per_group = 32;
};

/**
 * @brief Result of fused AST + aggregation execution
 */
struct AstFusedResult {
  std::unique_ptr<cudf::table> keys;
  std::unique_ptr<cudf::table> values;
  
  // Performance metrics
  double kernel_time_ms;
  size_t rows_processed;
  size_t groups_produced;
};

// ============================================================================
// AST EVALUATOR ADAPTOR
// ============================================================================

/**
 * @brief Adaptor that wraps cudf AST evaluator for use in fused kernels
 * 
 * This class provides a simple interface to evaluate AST expressions
 * row-by-row within a CUDA kernel, without writing intermediate results
 * to global memory.
 * 
 * Usage in kernel:
 * @code
 * __global__ void fused_kernel(...) {
 *   AstEvaluatorAdaptor evaluator(table, ast_expr);
 *   
 *   for (row_index : my_rows) {
 *     // Evaluate AST expression - result in register
 *     auto value = evaluator.evaluate<OutputType>(row_index);
 *     
 *     // Directly accumulate without memory write
 *     local_sum += value;
 *   }
 * }
 * @endcode
 */
template <bool has_nulls>
class AstEvaluatorAdaptor {
public:
  __device__ AstEvaluatorAdaptor(
      cudf::table_device_view const& table,
      cudf::ast::detail::expression_device_view const& expr_data,
      void* intermediate_storage)
    : evaluator_(table, expr_data)
    , intermediate_storage_(intermediate_storage)
  {}
  
  /**
   * @brief Evaluate AST expression for a single row
   * 
   * @tparam T Output type
   * @param row_index Row to evaluate
   * @return Evaluation result (in register, no memory write)
   */
  template <typename T>
  __device__ T evaluate(cudf::size_type row_index) {
    // Use cudf's expression evaluator internally
    // The result stays in register, not written to global memory
    using IntermediateType = cudf::ast::detail::IntermediateDataType<has_nulls>;
    auto* typed_storage = static_cast<IntermediateType*>(intermediate_storage_);
    
    // Create a temporary result holder
    T result{};
    // TODO: Implement actual evaluation using cudf's evaluator
    // This requires modifying evaluate() to return value instead of writing to column
    
    return result;
  }
  
  /**
   * @brief Check if row has null result
   */
  __device__ bool is_null(cudf::size_type row_index) {
    // TODO: Implement null check
    return false;
  }
  
private:
  cudf::ast::detail::expression_evaluator<has_nulls, false> evaluator_;
  void* intermediate_storage_;
};

// ============================================================================
// FUSED AST + AGGREGATE EXECUTION
// ============================================================================

/**
 * @brief Execute fused AST expression evaluation + aggregation
 * 
 * This is the main entry point for the long-term solution.
 * It compiles AST expressions and executes them fused with aggregation
 * in a single kernel pass.
 * 
 * @param input Input table
 * @param plan Execution plan with AST expressions and aggregation config
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Aggregation result
 */
AstFusedResult execute_ast_fused_aggregate(
    cudf::table_view const& input,
    AstFusedPlan const& plan,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Build execution plan from serialized AST expressions
 * 
 * This method is called from JNI to construct the execution plan
 * from serialized data passed from Scala.
 */
AstFusedPlan build_plan_from_jni(
    std::vector<int32_t> const& key_indices,
    std::vector<std::vector<uint8_t>> const& serialized_asts,
    std::vector<int32_t> const& agg_types,
    std::vector<int32_t> const& output_dtypes);

}  // namespace ast_fused
}  // namespace spark_rapids_jni

