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

#include "fused_transform_aggregate_jit.hpp"
#include "fused_transform_aggregate.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <chrono>
#include <sstream>
#include <vector>

namespace spark_rapids_jni {
namespace jit {

namespace {

// ============================================================================
// EXPRESSION IR (INTERMEDIATE REPRESENTATION)
// ============================================================================

/**
 * @brief Abstract base for expression IR nodes
 * 
 * This allows building complex expressions programmatically and then
 * generating CUDA code from the IR tree.
 */
struct ExprIR {
  virtual ~ExprIR() = default;
  virtual std::string to_cuda(std::string const& out_type) const = 0;
};

struct LiteralIR : ExprIR {
  int64_t value;
  explicit LiteralIR(int64_t v) : value(v) {}
  std::string to_cuda(std::string const& out_type) const override {
    return "static_cast<" + out_type + ">(" + std::to_string(value) + ")";
  }
};

struct InputRefIR : ExprIR {
  int index;
  explicit InputRefIR(int idx) : index(idx) {}
  std::string to_cuda(std::string const& out_type) const override {
    // Direct value access - types must match or be implicitly convertible
    // For timestamp types, use auto to preserve the type
    return "*in" + std::to_string(index);
  }
};

struct CoalesceIR : ExprIR {
  std::unique_ptr<ExprIR> value;
  std::unique_ptr<ExprIR> default_value;
  int input_index;
  
  CoalesceIR(int idx, std::unique_ptr<ExprIR> def) 
    : input_index(idx), default_value(std::move(def)) {}
  
  std::string to_cuda(std::string const& out_type) const override {
    return "in" + std::to_string(input_index) + ".has_value() ? "
           "static_cast<" + out_type + ">(*in" + std::to_string(input_index) + ") : " +
           default_value->to_cuda(out_type);
  }
};

struct BinaryOpIR : ExprIR {
  enum class Op { ADD, SUB, MUL, DIV };
  Op op;
  std::unique_ptr<ExprIR> left;
  std::unique_ptr<ExprIR> right;
  
  BinaryOpIR(Op o, std::unique_ptr<ExprIR> l, std::unique_ptr<ExprIR> r)
    : op(o), left(std::move(l)), right(std::move(r)) {}
  
  std::string to_cuda(std::string const& out_type) const override {
    std::string op_str;
    switch (op) {
      case Op::ADD: op_str = " + "; break;
      case Op::SUB: op_str = " - "; break;
      case Op::MUL: op_str = " * "; break;
      case Op::DIV: op_str = " / "; break;
    }
    return "(" + left->to_cuda(out_type) + op_str + right->to_cuda(out_type) + ")";
  }
};

struct ConditionalIR : ExprIR {
  int cond_input_index;
  int64_t threshold;
  std::unique_ptr<ExprIR> then_expr;
  std::unique_ptr<ExprIR> else_expr;
  
  ConditionalIR(int cond_idx, int64_t thresh, 
                std::unique_ptr<ExprIR> then_e, std::unique_ptr<ExprIR> else_e)
    : cond_input_index(cond_idx), threshold(thresh), 
      then_expr(std::move(then_e)), else_expr(std::move(else_e)) {}
  
  std::string to_cuda(std::string const& out_type) const override {
    return "(in" + std::to_string(cond_input_index) + ".has_value() && "
           "*in" + std::to_string(cond_input_index) + " > " + std::to_string(threshold) + ") ? " 
           "(" + then_expr->to_cuda(out_type) + ") : (" + else_expr->to_cuda(out_type) + ")";
  }
};

// ============================================================================
// IR BUILDER FROM FUSEDEXPRSPEC
// ============================================================================

/**
 * @brief Build expression IR from FusedExprSpec
 * 
 * This builds a tree representation of the expression, which can then
 * be serialized to CUDA code. This approach is more flexible than
 * string templates.
 */
class ExprIRBuilder {
public:
  static std::unique_ptr<ExprIR> build(FusedExprSpec const& spec) {
    switch (spec.transform_op) {
      case TransformOp::IDENTITY:
        return std::make_unique<InputRefIR>(0);
        
      case TransformOp::COALESCE:
        return std::make_unique<CoalesceIR>(0, 
            std::make_unique<LiteralIR>(spec.default_val));
        
      case TransformOp::COALESCE_MUL_SELF: {
        auto coalesced = std::make_unique<CoalesceIR>(0,
            std::make_unique<LiteralIR>(spec.default_val));
        // self * self - need to duplicate the expression
        auto left = std::make_unique<CoalesceIR>(0,
            std::make_unique<LiteralIR>(spec.default_val));
        auto right = std::make_unique<CoalesceIR>(0,
            std::make_unique<LiteralIR>(spec.default_val));
        return std::make_unique<BinaryOpIR>(BinaryOpIR::Op::MUL,
            std::move(left), std::move(right));
      }
        
      case TransformOp::COALESCE_MUL_OTHER: {
        auto left = std::make_unique<CoalesceIR>(0,
            std::make_unique<LiteralIR>(spec.default_val));
        auto right = std::make_unique<CoalesceIR>(1,
            std::make_unique<LiteralIR>(spec.default_val));
        return std::make_unique<BinaryOpIR>(BinaryOpIR::Op::MUL,
            std::move(left), std::move(right));
      }
        
      case TransformOp::CONDITIONAL:
        return std::make_unique<ConditionalIR>(0, spec.threshold,
            std::make_unique<InputRefIR>(1),
            std::make_unique<LiteralIR>(spec.else_val));
        
      case TransformOp::CONDITIONAL_COALESCE:
        return std::make_unique<ConditionalIR>(0, spec.threshold,
            std::make_unique<CoalesceIR>(1, 
                std::make_unique<LiteralIR>(spec.default_val)),
            std::make_unique<LiteralIR>(0));
        
      default:
        CUDF_FAIL("Unknown transform type");
    }
  }
};

// ============================================================================
// ADVANCED CODE GENERATOR USING IR
// ============================================================================

/**
 * @brief Generate CUDA UDF from expression IR
 */
class IRCodeGenerator {
public:
  IRCodeGenerator(cudf::table_view const& input) : input_(input) {}
  
  std::string generate(FusedExprSpec const& spec) {
    std::ostringstream code;
    
    // Build IR tree
    auto ir = ExprIRBuilder::build(spec);
    
    // Collect input columns
    auto inputs = collect_inputs(spec);
    
    // Generate signature using cuda::std::optional for null-aware mode
    // Format: __device__ void func(Out* out, cuda::std::optional<In0> in0, ...)
    code << "template <typename Out";
    for (size_t i = 0; i < inputs.size(); ++i) {
      code << ", typename In" << i;
    }
    code << ">\n";
    code << "__device__ void GENERIC_TRANSFORM_OP(\n";
    code << "    Out* out";
    for (size_t i = 0; i < inputs.size(); ++i) {
      code << ",\n    cuda::std::optional<In" << i << "> in" << i;
    }
    code << ") {\n";
    
    // Generate body from IR
    code << "  *out = " << ir->to_cuda("Out") << ";\n";
    code << "}\n";
    
    return code.str();
  }
  
private:
  cudf::table_view const& input_;
  
  std::vector<int32_t> collect_inputs(FusedExprSpec const& spec) {
    std::vector<int32_t> inputs;
    inputs.push_back(spec.value_col_idx);
    if (spec.transform_op == TransformOp::CONDITIONAL ||
        spec.transform_op == TransformOp::CONDITIONAL_COALESCE) {
      inputs.insert(inputs.begin(), spec.cond_col_idx);
    }
    if (spec.transform_op == TransformOp::COALESCE_MUL_OTHER) {
      inputs.push_back(spec.other_col_idx);
    }
    return inputs;
  }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Get input columns for a transform expression
 */
std::vector<cudf::column_view> get_input_columns(FusedExprSpec const& spec,
                                                  cudf::table_view const& input) {
  std::vector<cudf::column_view> cols;
  
  switch (spec.transform_op) {
    case TransformOp::IDENTITY:
    case TransformOp::COALESCE:
    case TransformOp::COALESCE_MUL_SELF:
      cols.push_back(input.column(spec.value_col_idx));
      break;
      
    case TransformOp::COALESCE_MUL_OTHER:
      cols.push_back(input.column(spec.value_col_idx));
      cols.push_back(input.column(spec.other_col_idx));
      break;
      
    case TransformOp::CONDITIONAL:
    case TransformOp::CONDITIONAL_COALESCE:
      cols.push_back(input.column(spec.cond_col_idx));
      cols.push_back(input.column(spec.value_col_idx));
      break;
      
    default:
      CUDF_FAIL("Unknown transform type");
  }
  
  return cols;
}

/**
 * @brief Execute single transform using dynamically generated JIT code
 */
std::unique_ptr<cudf::column> execute_transform_jit(
    FusedExprSpec const& spec,
    cudf::table_view const& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  
  // Get input columns
  auto input_cols = get_input_columns(spec, input);
  
  // Generate CUDA UDF dynamically using IR
  IRCodeGenerator gen(input);
  std::string udf = gen.generate(spec);
  
  // Execute transform using cudf JIT (null-aware mode)
  return cudf::transform(input_cols, udf, spec.output_type, false, std::nullopt, 
                         cudf::null_aware::YES, stream, mr);
}

/**
 * @brief Convert AggOp to cudf aggregation
 */
std::unique_ptr<cudf::groupby_aggregation> make_aggregation(AggOp op) {
  switch (op) {
    case AggOp::SUM:   return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case AggOp::COUNT: return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    case AggOp::AVG:   return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    case AggOp::MIN:   return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case AggOp::MAX:   return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    default:
      CUDF_FAIL("Unknown aggregation type");
  }
}

}  // namespace

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

bool JitExpressionBuilder::can_use_ast(FusedExprSpec const& spec) {
  // Currently only IDENTITY can use pure AST
  // All others need the IR-based code generator
  return spec.transform_op == TransformOp::IDENTITY;
}

std::pair<cudf::ast::tree, cudf::ast::expression const*>
JitExpressionBuilder::build_expression(FusedExprSpec const& spec,
                                       cudf::table_view const& input) {
  if (spec.transform_op == TransformOp::IDENTITY) {
    cudf::ast::tree tree;
    auto const& col_ref = tree.emplace<cudf::ast::column_reference>(spec.value_col_idx);
    return {std::move(tree), &col_ref};
  }
  CUDF_FAIL("Transform type " + transform_op_name(spec.transform_op) + 
            " requires IR-based code generation");
}

std::string JitExpressionBuilder::generate_cuda_udf(FusedExprSpec const& spec,
                                                    cudf::table_view const& input) {
  IRCodeGenerator gen(input);
  return gen.generate(spec);
}

// ============================================================================
// MAIN JIT EXECUTION
// ============================================================================

FusedExecutionResult execute_fused_transform_aggregate_jit(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  
  CUDF_FUNC_RANGE();
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Validate plan
  std::string validation_error;
  CUDF_EXPECTS(can_fuse(plan, &validation_error), 
               "Invalid execution plan: " + validation_error);
  
  FusedExecutionResult result;
  result.stats.num_expressions_fused = plan.expressions.size();
  result.stats.execution_strategy = "jit_ir_codegen";
  
  // Handle empty input
  if (input.num_rows() == 0) {
    result.stats.kernel_time_ns = 0;
    result.stats.total_time_ns = 0;
    result.stats.num_groups = 0;
    result.output_keys = std::make_unique<cudf::table>();
    result.output_values = std::make_unique<cudf::table>();
    return result;
  }
  
  // ============================================================================
  // Step 1: Execute transforms using dynamically generated JIT code
  // ============================================================================
  
  std::vector<std::unique_ptr<cudf::column>> transformed_columns;
  transformed_columns.reserve(plan.expressions.size());
  
  for (auto const& expr : plan.expressions) {
    auto transformed = execute_transform_jit(expr, input, stream, mr);
    transformed_columns.push_back(std::move(transformed));
  }
  
  // ============================================================================
  // Step 2: Build groupby keys
  // ============================================================================
  
  std::vector<cudf::column_view> key_columns;
  for (auto idx : plan.group_by_col_indices) {
    key_columns.push_back(input.column(idx));
  }
  cudf::table_view keys_table(key_columns);
  
  // ============================================================================
  // Step 3: Execute aggregations using cudf groupby
  // ============================================================================
  
  cudf::groupby::groupby groupby_obj(keys_table, cudf::null_policy::EXCLUDE);
  
  // Build aggregation requests
  std::vector<cudf::groupby::aggregation_request> requests;
  for (size_t i = 0; i < plan.expressions.size(); ++i) {
    cudf::groupby::aggregation_request req;
    req.values = transformed_columns[i]->view();
    req.aggregations.push_back(make_aggregation(plan.expressions[i].agg_op));
    requests.push_back(std::move(req));
  }
  
  auto agg_start = std::chrono::high_resolution_clock::now();
  
  // Execute groupby
  auto [keys_result, values_result] = groupby_obj.aggregate(requests, stream, mr);
  
  auto agg_end = std::chrono::high_resolution_clock::now();
  
  result.stats.kernel_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      agg_end - agg_start).count();
  
  // ============================================================================
  // Step 4: Extract results
  // ============================================================================
  
  result.output_keys = std::move(keys_result);
  
  // Flatten aggregation results
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  for (auto& agg_result : values_result) {
    for (auto& col : agg_result.results) {
      output_columns.push_back(std::move(col));
    }
  }
  
  result.output_values = std::make_unique<cudf::table>(std::move(output_columns));
  result.stats.num_groups = result.output_keys->num_rows();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  result.stats.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time).count();
  
  return result;
}

// ============================================================================
// HYBRID AND MODE SELECTION
// ============================================================================

FusedExecutionResult execute_fused_transform_aggregate_hybrid(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return execute_fused_transform_aggregate_jit(input, plan, stream, mr);
}

FusedExecutionResult execute_with_mode(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    ExecutionMode mode,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  
  switch (mode) {
    case ExecutionMode::HAND_WRITTEN_KERNEL:
      return execute_fused_transform_aggregate(input, plan, stream, mr);
      
    case ExecutionMode::JIT_AST:
    case ExecutionMode::JIT_UDF:
      return execute_fused_transform_aggregate_jit(input, plan, stream, mr);
      
    case ExecutionMode::HYBRID:
      return execute_fused_transform_aggregate_hybrid(input, plan, stream, mr);
      
    default:
      CUDF_FAIL("Unknown execution mode");
  }
}

// ============================================================================
// BENCHMARKING
// ============================================================================

std::vector<BenchmarkResult> benchmark_execution_modes(
    cudf::table_view const& input,
    FusedExecutionPlan const& plan,
    int warmup_iterations,
    int benchmark_iterations,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  
  std::vector<BenchmarkResult> results;
  
  std::vector<ExecutionMode> modes = {
    ExecutionMode::HAND_WRITTEN_KERNEL,
    ExecutionMode::JIT_UDF
  };
  
  for (auto mode : modes) {
    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
      auto _ = execute_with_mode(input, plan, mode, stream, mr);
      stream.synchronize();
    }
    
    // Benchmark
    double total_time = 0.0;
    for (int i = 0; i < benchmark_iterations; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      auto result = execute_with_mode(input, plan, mode, stream, mr);
      stream.synchronize();
      auto end = std::chrono::high_resolution_clock::now();
      
      total_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    BenchmarkResult br;
    br.mode = mode;
    br.total_time_ms = total_time / benchmark_iterations;
    br.transform_time_ms = 0;
    br.aggregate_time_ms = 0;
    br.peak_memory_bytes = 0;
    br.jit_cache_hit = (mode != ExecutionMode::HAND_WRITTEN_KERNEL);
    
    results.push_back(br);
  }
  
  return results;
}

}  // namespace jit
}  // namespace spark_rapids_jni
