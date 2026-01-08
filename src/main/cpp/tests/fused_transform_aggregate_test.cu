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

#include "../src/fused_transform_aggregate.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace spark_rapids_jni {
namespace test {

class FusedTransformAggregateTest : public cudf::test::BaseFixture {
 protected:
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
  rmm::device_async_resource_ref mr() { return cudf::get_current_device_resource_ref(); }
};

/**
 * @brief Create a simple int64 column for testing
 */
std::unique_ptr<cudf::column> make_int64_column(
    std::vector<int64_t> const& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  
  cudaMemcpyAsync(col->mutable_view().data<int64_t>(),
                  values.data(),
                  values.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  
  return col;
}

// ============================================================================
// Basic Tests
// ============================================================================

TEST_F(FusedTransformAggregateTest, CanFuseValidPlan) {
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::IDENTITY;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  plan.expressions.push_back(spec);
  
  std::string reason;
  EXPECT_TRUE(can_fuse(plan, &reason));
}

TEST_F(FusedTransformAggregateTest, CannotFuseEmptyExpressions) {
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  // No expressions
  
  std::string reason;
  EXPECT_FALSE(can_fuse(plan, &reason));
  EXPECT_EQ(reason, "No expressions to fuse");
}

TEST_F(FusedTransformAggregateTest, CannotFuseNoGroupBy) {
  FusedExecutionPlan plan;
  // No group-by columns is now allowed for scalar aggregation
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::IDENTITY;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  plan.expressions.push_back(spec);
  
  std::string reason;
  EXPECT_TRUE(can_fuse(plan, &reason)) << "Scalar aggregation should be supported";
}

// ============================================================================
// Simple Sum Test
// ============================================================================

TEST_F(FusedTransformAggregateTest, SimpleSumSingleGroup) {
  // Input: key=[0,0,0,0], val=[1,2,3,4]
  // Expected: key=[0], sum=[10]
  
  auto key_col = make_int64_column({0, 0, 0, 0}, stream(), mr());
  auto val_col = make_int64_column({1, 2, 3, 4}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  plan.enable_warp_reduction = false;
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::IDENTITY;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  spec.output_idx = 0;
  plan.expressions.push_back(spec);
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 1);
  EXPECT_EQ(result.output_values->num_columns(), 1);
  
  // Check result
  std::vector<int64_t> output(1);
  cudaMemcpy(output.data(), 
             result.output_values->get_column(0).view().data<int64_t>(),
             sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  
  EXPECT_EQ(output[0], 10);
}

TEST_F(FusedTransformAggregateTest, SimpleSumMultipleGroups) {
  // Input: key=[0,0,1,1,2], val=[1,2,3,4,5]
  // Expected: key=[0,1,2], sum=[3,7,5]
  
  auto key_col = make_int64_column({0, 0, 1, 1, 2}, stream(), mr());
  auto val_col = make_int64_column({1, 2, 3, 4, 5}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  plan.enable_warp_reduction = false;
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::IDENTITY;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  plan.expressions.push_back(spec);
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 3);
  
  // Get results (note: order may vary based on cudf::groupby implementation)
  std::vector<int64_t> output(3);
  cudaMemcpy(output.data(), 
             result.output_values->get_column(0).view().data<int64_t>(),
             3 * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  
  // Sort for comparison (groupby doesn't guarantee order)
  std::sort(output.begin(), output.end());
  EXPECT_EQ(output[0], 3);   // group 0: 1+2
  EXPECT_EQ(output[1], 5);   // group 2: 5
  EXPECT_EQ(output[2], 7);   // group 1: 3+4
}

// ============================================================================
// Coalesce Tests
// ============================================================================

TEST_F(FusedTransformAggregateTest, CoalesceMulSelf) {
  // Input: key=[0,0], val=[2,3]
  // Expression: SUM(COALESCE(val, 0) * COALESCE(val, 0)) = SUM(val^2)
  // Expected: 2^2 + 3^2 = 4 + 9 = 13
  
  auto key_col = make_int64_column({0, 0}, stream(), mr());
  auto val_col = make_int64_column({2, 3}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  plan.enable_warp_reduction = false;
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::COALESCE_MUL_SELF;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  spec.default_val = 0;
  plan.expressions.push_back(spec);
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 1);
  
  std::vector<int64_t> output(1);
  cudaMemcpy(output.data(), 
             result.output_values->get_column(0).view().data<int64_t>(),
             sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  
  EXPECT_EQ(output[0], 13);  // 4 + 9
}

// ============================================================================
// Multiple Expressions Test
// ============================================================================

TEST_F(FusedTransformAggregateTest, MultipleExpressions) {
  // Input: key=[0,0,0], val=[1,2,3]
  // Expressions:
  //   1. SUM(val) = 6
  //   2. SUM(val * val) = 1 + 4 + 9 = 14
  //   3. COUNT = 3
  
  auto key_col = make_int64_column({0, 0, 0}, stream(), mr());
  auto val_col = make_int64_column({1, 2, 3}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  plan.enable_warp_reduction = true;  // Test with warp reduction
  
  // Expression 1: SUM(val)
  {
    FusedExprSpec spec;
    spec.transform_op = TransformOp::IDENTITY;
    spec.agg_op = AggOp::SUM;
    spec.value_col_idx = 1;
    plan.expressions.push_back(spec);
  }
  
  // Expression 2: SUM(val^2)
  {
    FusedExprSpec spec;
    spec.transform_op = TransformOp::COALESCE_MUL_SELF;
    spec.agg_op = AggOp::SUM;
    spec.value_col_idx = 1;
    spec.default_val = 0;
    plan.expressions.push_back(spec);
  }
  
  // Expression 3: COUNT
  {
    FusedExprSpec spec;
    spec.transform_op = TransformOp::IDENTITY;
    spec.agg_op = AggOp::COUNT;
    spec.value_col_idx = 1;
    plan.expressions.push_back(spec);
  }
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 1);
  EXPECT_EQ(result.output_values->num_columns(), 3);
  EXPECT_EQ(result.stats.num_expressions_fused, 3);
  
  // Check results
  std::vector<int64_t> sum_val(1);
  std::vector<int64_t> sum_sq(1);
  std::vector<int64_t> count(1);
  
  cudaMemcpy(sum_val.data(), 
             result.output_values->get_column(0).view().data<int64_t>(),
             sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_sq.data(), 
             result.output_values->get_column(1).view().data<int64_t>(),
             sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(count.data(), 
             result.output_values->get_column(2).view().data<int64_t>(),
             sizeof(int64_t), cudaMemcpyDeviceToHost);
  
  EXPECT_EQ(sum_val[0], 6);   // 1 + 2 + 3
  EXPECT_EQ(sum_sq[0], 14);   // 1 + 4 + 9
  EXPECT_EQ(count[0], 3);
}

// ============================================================================
// Empty Input Test
// ============================================================================

TEST_F(FusedTransformAggregateTest, EmptyInput) {
  auto key_col = make_int64_column({}, stream(), mr());
  auto val_col = make_int64_column({}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::IDENTITY;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 1;
  plan.expressions.push_back(spec);
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 0);
}

// ============================================================================
// Conditional Test
// ============================================================================

TEST_F(FusedTransformAggregateTest, ConditionalSum) {
  // Input: key=[0,0,0,0], cond=[1,0,2,0], val=[10,20,30,40]
  // Expression: SUM(IF(cond > 0, val, 0))
  // Expected: 10 + 0 + 30 + 0 = 40
  
  auto key_col = make_int64_column({0, 0, 0, 0}, stream(), mr());
  auto cond_col = make_int64_column({1, 0, 2, 0}, stream(), mr());
  auto val_col = make_int64_column({10, 20, 30, 40}, stream(), mr());
  
  std::vector<cudf::column_view> columns = {key_col->view(), cond_col->view(), val_col->view()};
  cudf::table_view input(columns);
  
  FusedExecutionPlan plan;
  plan.group_by_col_indices = {0};
  plan.enable_warp_reduction = false;
  
  FusedExprSpec spec;
  spec.transform_op = TransformOp::CONDITIONAL;
  spec.agg_op = AggOp::SUM;
  spec.value_col_idx = 2;    // val column
  spec.cond_col_idx = 1;     // cond column
  spec.threshold = 0;        // cond > 0
  spec.else_val = 0;         // else 0
  plan.expressions.push_back(spec);
  
  auto result = execute_fused_transform_aggregate(input, plan, stream(), mr());
  
  EXPECT_EQ(result.num_groups(), 1);
  
  std::vector<int64_t> output(1);
  cudaMemcpy(output.data(), 
             result.output_values->get_column(0).view().data<int64_t>(),
             sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  
  EXPECT_EQ(output[0], 40);  // 10 + 30
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(FusedTransformAggregateTest, TransformOpName) {
  EXPECT_EQ(transform_op_name(TransformOp::IDENTITY), "IDENTITY");
  EXPECT_EQ(transform_op_name(TransformOp::COALESCE), "COALESCE");
  EXPECT_EQ(transform_op_name(TransformOp::COALESCE_MUL_SELF), "COALESCE_MUL_SELF");
  EXPECT_EQ(transform_op_name(TransformOp::CONDITIONAL), "CONDITIONAL");
}

TEST_F(FusedTransformAggregateTest, AggOpName) {
  EXPECT_EQ(agg_op_name(AggOp::SUM), "SUM");
  EXPECT_EQ(agg_op_name(AggOp::COUNT), "COUNT");
  EXPECT_EQ(agg_op_name(AggOp::AVG), "AVG");
  EXPECT_EQ(agg_op_name(AggOp::MIN), "MIN");
  EXPECT_EQ(agg_op_name(AggOp::MAX), "MAX");
}

}  // namespace test
}  // namespace spark_rapids_jni

// Main function for running tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


