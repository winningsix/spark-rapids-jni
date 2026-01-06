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

#include "fused_transform_aggregate.hpp"
#include "fused_transform_aggregate_jit.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <random>
#include <vector>

namespace spark_rapids_jni {
namespace benchmark {

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

std::unique_ptr<cudf::table> create_test_data(cudf::size_type num_rows,
                                               cudf::size_type num_groups,
                                               rmm::cuda_stream_view stream) {
  std::vector<int64_t> h_keys(num_rows);
  std::vector<int64_t> h_values(num_rows);
  
  std::mt19937 gen(42);
  std::uniform_int_distribution<int64_t> key_dist(0, num_groups - 1);
  std::uniform_int_distribution<int64_t> val_dist(1, 100);
  
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    h_keys[i] = key_dist(gen);
    h_values[i] = val_dist(gen);
  }
  
  rmm::device_uvector<int64_t> d_keys(num_rows, stream);
  rmm::device_uvector<int64_t> d_values(num_rows, stream);
  
  cudaMemcpyAsync(d_keys.data(), h_keys.data(), 
                  num_rows * sizeof(int64_t), cudaMemcpyHostToDevice, stream.value());
  cudaMemcpyAsync(d_values.data(), h_values.data(),
                  num_rows * sizeof(int64_t), cudaMemcpyHostToDevice, stream.value());
  
  auto key_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows,
      cudf::mask_state::UNALLOCATED, stream);
  auto val_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows,
      cudf::mask_state::UNALLOCATED, stream);
  
  cudaMemcpyAsync(key_col->mutable_view().data<int64_t>(), d_keys.data(),
                  num_rows * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream.value());
  cudaMemcpyAsync(val_col->mutable_view().data<int64_t>(), d_values.data(),
                  num_rows * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream.value());
  
  stream.synchronize();
  
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_col));
  columns.push_back(std::move(val_col));
  
  return std::make_unique<cudf::table>(std::move(columns));
}

FusedExecutionPlan create_simple_plan() {
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  
  FusedExprSpec expr;
  expr.transform_op = TransformOp::IDENTITY;
  expr.value_col_idx = 1;
  expr.agg_op = AggOp::SUM;
  expr.output_type = cudf::data_type{cudf::type_id::INT64};
  expr.output_idx = 0;
  plan.expressions.push_back(expr);
  
  return plan;
}

FusedExecutionPlan create_multi_expr_plan() {
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  
  // SUM
  FusedExprSpec sum_expr;
  sum_expr.transform_op = TransformOp::IDENTITY;
  sum_expr.value_col_idx = 1;
  sum_expr.agg_op = AggOp::SUM;
  sum_expr.output_type = cudf::data_type{cudf::type_id::INT64};
  sum_expr.output_idx = 0;
  plan.expressions.push_back(sum_expr);
  
  // COUNT
  FusedExprSpec count_expr;
  count_expr.transform_op = TransformOp::IDENTITY;
  count_expr.value_col_idx = 1;
  count_expr.agg_op = AggOp::COUNT;
  count_expr.output_type = cudf::data_type{cudf::type_id::INT64};
  count_expr.output_idx = 1;
  plan.expressions.push_back(count_expr);
  
  // AVG
  FusedExprSpec avg_expr;
  avg_expr.transform_op = TransformOp::IDENTITY;
  avg_expr.value_col_idx = 1;
  avg_expr.agg_op = AggOp::AVG;
  avg_expr.output_type = cudf::data_type{cudf::type_id::FLOAT64};
  avg_expr.output_idx = 2;
  plan.expressions.push_back(avg_expr);
  
  return plan;
}

FusedExecutionPlan create_coalesce_plan() {
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  
  // SUM(COALESCE(col, 0))
  FusedExprSpec expr;
  expr.transform_op = TransformOp::COALESCE;
  expr.value_col_idx = 1;
  expr.default_val = 0;
  expr.agg_op = AggOp::SUM;
  expr.output_type = cudf::data_type{cudf::type_id::INT64};
  expr.output_idx = 0;
  plan.expressions.push_back(expr);
  
  return plan;
}

// ============================================================================
// NVBENCH BENCHMARKS
// ============================================================================

void BM_HandWrittenKernel(nvbench::state& state) {
  auto num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto stream = cudf::get_default_stream();
  
  auto data = create_test_data(num_rows, num_groups, stream);
  auto plan = create_multi_expr_plan();
  
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = execute_fused_transform_aggregate(data->view(), plan, stream);
  });
}

void BM_JitIRKernel(nvbench::state& state) {
  auto num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto stream = cudf::get_default_stream();
  
  auto data = create_test_data(num_rows, num_groups, stream);
  auto plan = create_multi_expr_plan();
  
  // Warmup to trigger JIT compilation
  auto _ = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  stream.synchronize();
  
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  });
}

void BM_CoalesceHandWritten(nvbench::state& state) {
  auto num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto stream = cudf::get_default_stream();
  
  auto data = create_test_data(num_rows, num_groups, stream);
  auto plan = create_coalesce_plan();
  
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = execute_fused_transform_aggregate(data->view(), plan, stream);
  });
}

void BM_CoalesceJit(nvbench::state& state) {
  auto num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto stream = cudf::get_default_stream();
  
  auto data = create_test_data(num_rows, num_groups, stream);
  auto plan = create_coalesce_plan();
  
  // Warmup
  auto _ = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  stream.synchronize();
  
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  });
}

// Register benchmarks
NVBENCH_BENCH(BM_HandWrittenKernel)
  .set_name("HandWritten_MultiExpr")
  .add_int64_axis("num_rows", {10000, 100000, 1000000, 10000000})
  .add_int64_axis("num_groups", {100, 1000, 10000, 100000});

NVBENCH_BENCH(BM_JitIRKernel)
  .set_name("JIT_IR_MultiExpr")
  .add_int64_axis("num_rows", {10000, 100000, 1000000, 10000000})
  .add_int64_axis("num_groups", {100, 1000, 10000, 100000});

NVBENCH_BENCH(BM_CoalesceHandWritten)
  .set_name("HandWritten_Coalesce")
  .add_int64_axis("num_rows", {10000, 100000, 1000000, 10000000})
  .add_int64_axis("num_groups", {100, 1000, 10000, 100000});

NVBENCH_BENCH(BM_CoalesceJit)
  .set_name("JIT_IR_Coalesce")
  .add_int64_axis("num_rows", {10000, 100000, 1000000, 10000000})
  .add_int64_axis("num_groups", {100, 1000, 10000, 100000});

}  // namespace benchmark
}  // namespace spark_rapids_jni


