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
#include <cudf/filling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace spark_rapids_jni {
namespace jit {
namespace test {

// ============================================================================
// TEST FIXTURE
// ============================================================================

class FusedTransformJitTest : public ::testing::Test {
protected:
  rmm::cuda_stream_view stream = cudf::get_default_stream();
  
  /**
   * @brief Create test data with specified size and number of groups
   */
  std::unique_ptr<cudf::table> create_test_data(cudf::size_type num_rows,
                                                 cudf::size_type num_groups) {
    // Generate group keys (values 0 to num_groups-1)
    std::vector<int64_t> h_keys(num_rows);
    std::vector<int64_t> h_values(num_rows);
    
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int64_t> key_dist(0, num_groups - 1);
    std::uniform_int_distribution<int64_t> val_dist(1, 100);
    
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      h_keys[i] = key_dist(gen);
      h_values[i] = val_dist(gen);
    }
    
    // Create columns
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
  
  /**
   * @brief Create a simple execution plan
   */
  FusedExecutionPlan create_simple_plan() {
    FusedExecutionPlan plan;
    plan.group_by_col_indices = {0};  // Group by first column
    
    // Add SUM(col1) expression
    FusedExprSpec expr;
    expr.transform_op = TransformOp::IDENTITY;
    expr.value_col_idx = 1;
    expr.agg_op = AggOp::SUM;
    expr.output_type = cudf::data_type{cudf::type_id::INT64};
    expr.output_idx = 0;
    
    plan.expressions.push_back(expr);
    return plan;
  }
  
  /**
   * @brief Create a plan with multiple expressions
   */
  FusedExecutionPlan create_multi_expr_plan() {
    FusedExecutionPlan plan;
    plan.group_by_col_indices = {0};
    
    // SUM(col1)
    FusedExprSpec sum_expr;
    sum_expr.transform_op = TransformOp::IDENTITY;
    sum_expr.value_col_idx = 1;
    sum_expr.agg_op = AggOp::SUM;
    sum_expr.output_type = cudf::data_type{cudf::type_id::INT64};
    sum_expr.output_idx = 0;
    plan.expressions.push_back(sum_expr);
    
    // COUNT(col1)
    FusedExprSpec count_expr;
    count_expr.transform_op = TransformOp::IDENTITY;
    count_expr.value_col_idx = 1;
    count_expr.agg_op = AggOp::COUNT;
    count_expr.output_type = cudf::data_type{cudf::type_id::INT64};
    count_expr.output_idx = 1;
    plan.expressions.push_back(count_expr);
    
    // AVG(col1)
    FusedExprSpec avg_expr;
    avg_expr.transform_op = TransformOp::IDENTITY;
    avg_expr.value_col_idx = 1;
    avg_expr.agg_op = AggOp::AVG;
    avg_expr.output_type = cudf::data_type{cudf::type_id::FLOAT64};
    avg_expr.output_idx = 2;
    plan.expressions.push_back(avg_expr);
    
    return plan;
  }
  
  /**
   * @brief Create a plan with coalesce transform
   */
  FusedExecutionPlan create_coalesce_plan() {
    FusedExecutionPlan plan;
    plan.group_by_col_indices = {0};
    
    // SUM(COALESCE(col1, 0))
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
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(FusedTransformJitTest, SimpleSum) {
  auto data = create_test_data(1000, 10);
  auto plan = create_simple_plan();
  
  auto result = execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  
  EXPECT_GT(result.num_groups(), 0);
  EXPECT_EQ(result.output_values->num_columns(), 1);
  EXPECT_EQ(result.stats.execution_strategy, "jit_transform");
  
  std::cout << "SimpleSum: " << result.num_groups() << " groups, "
            << result.stats.total_time_ns / 1e6 << " ms" << std::endl;
}

TEST_F(FusedTransformJitTest, MultipleExpressions) {
  auto data = create_test_data(1000, 10);
  auto plan = create_multi_expr_plan();
  
  auto result = execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  
  EXPECT_GT(result.num_groups(), 0);
  EXPECT_EQ(result.output_values->num_columns(), 3);  // SUM, COUNT, AVG
  
  std::cout << "MultipleExpressions: " << result.num_groups() << " groups, "
            << result.stats.total_time_ns / 1e6 << " ms" << std::endl;
}

TEST_F(FusedTransformJitTest, CoalesceTransform) {
  auto data = create_test_data(1000, 10);
  auto plan = create_coalesce_plan();
  
  auto result = execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  
  EXPECT_GT(result.num_groups(), 0);
  EXPECT_EQ(result.output_values->num_columns(), 1);
  
  std::cout << "CoalesceTransform: " << result.num_groups() << " groups, "
            << result.stats.total_time_ns / 1e6 << " ms" << std::endl;
}

TEST_F(FusedTransformJitTest, EmptyInput) {
  auto data = create_test_data(0, 0);
  auto plan = create_simple_plan();
  
  auto result = execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  
  EXPECT_EQ(result.num_groups(), 0);
}

// ============================================================================
// CORRECTNESS COMPARISON TESTS
// ============================================================================

TEST_F(FusedTransformJitTest, CompareWithHandWritten) {
  auto data = create_test_data(10000, 100);
  auto plan = create_simple_plan();
  
  // Execute with hand-written kernel
  auto hw_result = execute_fused_transform_aggregate(data->view(), plan, stream);
  
  // Execute with JIT
  auto jit_result = execute_fused_transform_aggregate_jit(data->view(), plan, stream);
  
  // Compare results
  EXPECT_EQ(hw_result.num_groups(), jit_result.num_groups());
  
  std::cout << "Hand-written: " << hw_result.stats.total_time_ns / 1e6 << " ms" << std::endl;
  std::cout << "JIT:          " << jit_result.stats.total_time_ns / 1e6 << " ms" << std::endl;
}

// ============================================================================
// PERFORMANCE BENCHMARK TESTS
// ============================================================================

TEST_F(FusedTransformJitTest, BenchmarkSmallData) {
  auto data = create_test_data(10000, 100);
  auto plan = create_multi_expr_plan();
  
  auto results = benchmark_execution_modes(data->view(), plan, 3, 5, stream);
  
  std::cout << "\n=== Benchmark Results (10K rows, 100 groups) ===" << std::endl;
  for (auto const& r : results) {
    std::string mode_name;
    switch (r.mode) {
      case ExecutionMode::HAND_WRITTEN_KERNEL: mode_name = "Hand-written"; break;
      case ExecutionMode::JIT_UDF: mode_name = "JIT UDF"; break;
      default: mode_name = "Unknown"; break;
    }
    std::cout << mode_name << ": " << r.total_time_ms << " ms" << std::endl;
  }
}

TEST_F(FusedTransformJitTest, BenchmarkMediumData) {
  auto data = create_test_data(100000, 1000);
  auto plan = create_multi_expr_plan();
  
  auto results = benchmark_execution_modes(data->view(), plan, 3, 5, stream);
  
  std::cout << "\n=== Benchmark Results (100K rows, 1K groups) ===" << std::endl;
  for (auto const& r : results) {
    std::string mode_name;
    switch (r.mode) {
      case ExecutionMode::HAND_WRITTEN_KERNEL: mode_name = "Hand-written"; break;
      case ExecutionMode::JIT_UDF: mode_name = "JIT UDF"; break;
      default: mode_name = "Unknown"; break;
    }
    std::cout << mode_name << ": " << r.total_time_ms << " ms" << std::endl;
  }
}

TEST_F(FusedTransformJitTest, BenchmarkLargeData) {
  auto data = create_test_data(1000000, 10000);
  auto plan = create_multi_expr_plan();
  
  auto results = benchmark_execution_modes(data->view(), plan, 2, 3, stream);
  
  std::cout << "\n=== Benchmark Results (1M rows, 10K groups) ===" << std::endl;
  for (auto const& r : results) {
    std::string mode_name;
    switch (r.mode) {
      case ExecutionMode::HAND_WRITTEN_KERNEL: mode_name = "Hand-written"; break;
      case ExecutionMode::JIT_UDF: mode_name = "JIT UDF"; break;
      default: mode_name = "Unknown"; break;
    }
    std::cout << mode_name << ": " << r.total_time_ms << " ms" << std::endl;
  }
}

// ============================================================================
// EXECUTION MODE TESTS
// ============================================================================

TEST_F(FusedTransformJitTest, ExecutionModeSelection) {
  auto data = create_test_data(10000, 100);
  auto plan = create_simple_plan();
  
  // Test each execution mode
  auto hw = execute_with_mode(data->view(), plan, ExecutionMode::HAND_WRITTEN_KERNEL, stream);
  auto jit = execute_with_mode(data->view(), plan, ExecutionMode::JIT_UDF, stream);
  auto hybrid = execute_with_mode(data->view(), plan, ExecutionMode::HYBRID, stream);
  
  EXPECT_EQ(hw.num_groups(), jit.num_groups());
  EXPECT_EQ(hw.num_groups(), hybrid.num_groups());
}

}  // namespace test
}  // namespace jit
}  // namespace spark_rapids_jni

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

