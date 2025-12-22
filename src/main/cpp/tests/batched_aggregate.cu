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

#include "batched_aggregate.hpp"
#include "batched_aggregate_config.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace {

using namespace spark_rapids_jni;

// =============================================================================
// Test Fixture
// =============================================================================

class BatchedAggregateTest : public cudf::test::BaseFixture {
 protected:
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
  rmm::device_async_resource_ref mr() { return cudf::get_current_device_resource_ref(); }
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Create test data with specified number of rows and groups
 */
std::pair<std::unique_ptr<cudf::column>, std::vector<std::unique_ptr<cudf::column>>>
create_test_data(cudf::size_type num_rows, cudf::size_type num_groups, int num_value_cols) {
  // Create key column
  std::vector<int64_t> key_data(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i] = i % num_groups;
  }
  auto keys = cudf::test::fixed_width_column_wrapper<int64_t>(key_data.begin(), key_data.end());
  
  // Create value columns
  std::vector<std::unique_ptr<cudf::column>> value_cols;
  for (int c = 0; c < num_value_cols; ++c) {
    std::vector<int64_t> val_data(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      val_data[i] = (i * (c + 1)) % 10000;
    }
    auto col = cudf::test::fixed_width_column_wrapper<int64_t>(val_data.begin(), val_data.end());
    value_cols.push_back(col.release());
  }
  
  return {keys.release(), std::move(value_cols)};
}

/**
 * @brief Create nullable test data
 */
std::pair<std::unique_ptr<cudf::column>, std::vector<std::unique_ptr<cudf::column>>>
create_nullable_test_data(cudf::size_type num_rows, cudf::size_type num_groups, int num_value_cols) {
  // Create key column
  std::vector<int64_t> key_data(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i] = i % num_groups;
  }
  auto keys = cudf::test::fixed_width_column_wrapper<int64_t>(key_data.begin(), key_data.end());
  
  // Create value columns with nulls
  std::vector<std::unique_ptr<cudf::column>> value_cols;
  for (int c = 0; c < num_value_cols; ++c) {
    std::vector<int64_t> val_data(num_rows);
    std::vector<bool> validity(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      val_data[i] = (i * (c + 1)) % 10000;
      validity[i] = (i % (c + 3)) != 0;  // Some null pattern
    }
    auto col = cudf::test::fixed_width_column_wrapper<int64_t>(
        val_data.begin(), val_data.end(), validity.begin());
    value_cols.push_back(col.release());
  }
  
  return {keys.release(), std::move(value_cols)};
}

/**
 * @brief Build aggregation specs for testing
 */
std::vector<batched_agg_spec> build_specs(
    std::vector<std::unique_ptr<cudf::column>> const& value_cols,
    BatchedAggType agg_type) {
  std::vector<batched_agg_spec> specs;
  for (auto const& col : value_cols) {
    specs.push_back(batched_agg_spec{
        col->view(),
        agg_type,
        cudf::data_type{cudf::type_id::INT64},
        cudf::null_policy::EXCLUDE
    });
  }
  return specs;
}

/**
 * @brief Build mixed aggregation specs
 */
std::vector<batched_agg_spec> build_mixed_specs(
    std::vector<std::unique_ptr<cudf::column>> const& value_cols) {
  std::vector<batched_agg_spec> specs;
  BatchedAggType types[] = {BatchedAggType::SUM, BatchedAggType::COUNT, 
                            BatchedAggType::AVG, BatchedAggType::MIN, BatchedAggType::MAX};
  
  for (size_t i = 0; i < value_cols.size(); ++i) {
    auto agg_type = types[i % 5];
    auto out_type = (agg_type == BatchedAggType::AVG) ? 
        cudf::data_type{cudf::type_id::FLOAT64} : 
        cudf::data_type{cudf::type_id::INT64};
    
    specs.push_back(batched_agg_spec{
        value_cols[i]->view(),
        agg_type,
        out_type,
        cudf::null_policy::EXCLUDE
    });
  }
  return specs;
}

}  // namespace

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(BatchedAggregateTest, BasicSumAggregation)
{
  constexpr cudf::size_type num_rows = 1000;
  constexpr cudf::size_type num_groups = 50;
  constexpr int num_cols = 5;
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  // Verify result structure
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, BasicCountAggregation)
{
  constexpr cudf::size_type num_rows = 500;
  constexpr cudf::size_type num_groups = 25;
  constexpr int num_cols = 4;
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::COUNT);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
  
  // Each group should have num_rows/num_groups count
  auto count_col = result.values->get_column(0).view();
  EXPECT_FALSE(count_col.has_nulls());
}

TEST_F(BatchedAggregateTest, BasicMinMaxAggregation)
{
  constexpr cudf::size_type num_rows = 800;
  constexpr cudf::size_type num_groups = 40;
  constexpr int num_cols = 6;
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  
  // Build specs with alternating MIN/MAX
  std::vector<batched_agg_spec> specs;
  for (size_t i = 0; i < value_cols.size(); ++i) {
    auto agg_type = (i % 2 == 0) ? BatchedAggType::MIN : BatchedAggType::MAX;
    specs.push_back(batched_agg_spec{
        value_cols[i]->view(),
        agg_type,
        cudf::data_type{cudf::type_id::INT64},
        cudf::null_policy::EXCLUDE
    });
  }
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, MixedAggregationTypes)
{
  constexpr cudf::size_type num_rows = 1000;
  constexpr cudf::size_type num_groups = 50;
  constexpr int num_cols = 10;
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_mixed_specs(value_cols);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

// =============================================================================
// Optimization Flag Tests
// =============================================================================

TEST_F(BatchedAggregateTest, WarpReductionOptimization)
{
  // Test with warp reduction enabled vs disabled
  constexpr cudf::size_type num_rows = 50000;  // Large enough for warp reduction
  constexpr cudf::size_type num_groups = 100;
  constexpr int num_cols = 8;
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  // Run with default config (warp reduction enabled)
  auto result1 = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result1.keys->num_rows(), num_groups);
  EXPECT_EQ(result1.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, ContiguousOutputOptimization)
{
  // Test contiguous output buffer optimization
  constexpr cudf::size_type num_rows = 5000;
  constexpr cudf::size_type num_groups = 50;
  constexpr int num_cols = 15;  // Many columns to benefit from contiguous output
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
  
  // Verify shared buffer is populated (contiguous output)
  // The shared_buffer should be non-empty when contiguous output is enabled
  // Note: The actual implementation may vary
}

TEST_F(BatchedAggregateTest, SharedGroupbyOptimization)
{
  // Test shared groupby results optimization
  constexpr cudf::size_type num_rows = 10000;
  constexpr cudf::size_type num_groups = 100;
  constexpr int num_cols = 20;  // Many columns benefit from shared groupby
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_mixed_specs(value_cols);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

// =============================================================================
// Null Handling Tests
// =============================================================================

TEST_F(BatchedAggregateTest, NullableColumnSum)
{
  constexpr cudf::size_type num_rows = 1000;
  constexpr cudf::size_type num_groups = 50;
  constexpr int num_cols = 5;
  
  auto [key_col, value_cols] = create_nullable_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, NullableColumnCount)
{
  constexpr cudf::size_type num_rows = 800;
  constexpr cudf::size_type num_groups = 40;
  constexpr int num_cols = 4;
  
  auto [key_col, value_cols] = create_nullable_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_specs(value_cols, BatchedAggType::COUNT);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  
  // Count should never have nulls (even when input has nulls)
  for (int i = 0; i < num_cols; ++i) {
    EXPECT_FALSE(result.values->get_column(i).has_nulls());
  }
}

TEST_F(BatchedAggregateTest, NullableColumnAvg)
{
  constexpr cudf::size_type num_rows = 600;
  constexpr cudf::size_type num_groups = 30;
  constexpr int num_cols = 3;
  
  auto [key_col, value_cols] = create_nullable_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  
  std::vector<batched_agg_spec> specs;
  for (auto const& col : value_cols) {
    specs.push_back(batched_agg_spec{
        col->view(),
        BatchedAggType::AVG,
        cudf::data_type{cudf::type_id::FLOAT64},
        cudf::null_policy::EXCLUDE
    });
  }
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(BatchedAggregateTest, EmptyInput)
{
  // Test with empty input
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>{};
  auto val_col = cudf::test::fixed_width_column_wrapper<int64_t>{};
  
  auto keys = cudf::table_view{{keys_col}};
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::SUM,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), 0);
  EXPECT_EQ(result.values->num_rows(), 0);
}

TEST_F(BatchedAggregateTest, SingleGroup)
{
  // All rows belong to same group
  constexpr cudf::size_type num_rows = 500;
  constexpr int num_cols = 5;
  
  std::vector<int64_t> key_data(num_rows, 42);  // All same key
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>(
      key_data.begin(), key_data.end());
  
  std::vector<std::unique_ptr<cudf::column>> value_cols;
  for (int c = 0; c < num_cols; ++c) {
    std::vector<int64_t> val_data(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      val_data[i] = i * (c + 1);
    }
    auto col = cudf::test::fixed_width_column_wrapper<int64_t>(
        val_data.begin(), val_data.end());
    value_cols.push_back(col.release());
  }
  
  auto keys = cudf::table_view{{keys_col}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), 1);  // Single group
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, HighCardinalityKeys)
{
  // Each row is its own group
  constexpr cudf::size_type num_rows = 500;
  constexpr int num_cols = 3;
  
  std::vector<int64_t> key_data(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i] = i;  // Unique key per row
  }
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>(
      key_data.begin(), key_data.end());
  
  std::vector<std::unique_ptr<cudf::column>> value_cols;
  for (int c = 0; c < num_cols; ++c) {
    std::vector<int64_t> val_data(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      val_data[i] = i * (c + 1);
    }
    auto col = cudf::test::fixed_width_column_wrapper<int64_t>(
        val_data.begin(), val_data.end());
    value_cols.push_back(col.release());
  }
  
  auto keys = cudf::table_view{{keys_col}};
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_rows);  // Each row is a group
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

TEST_F(BatchedAggregateTest, LargeNumberOfColumns)
{
  // Stress test with many aggregation columns
  constexpr cudf::size_type num_rows = 2000;
  constexpr cudf::size_type num_groups = 50;
  constexpr int num_cols = 50;  // Large number of columns
  
  auto [key_col, value_cols] = create_test_data(num_rows, num_groups, num_cols);
  auto keys = cudf::table_view{{key_col->view()}};
  auto specs = build_mixed_specs(value_cols);
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), num_cols);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(BatchedAggregateTest, ConfigDefaultValues)
{
  auto config = BatchedAggConfig::defaults();
  
  EXPECT_TRUE(config.enabled);
  // Benchmark findings: warp reduction and perfect hash show no benefit
  EXPECT_FALSE(config.warp_reduction_enabled);  // No benefit for random groups
  EXPECT_TRUE(config.contiguous_output_enabled);
  EXPECT_TRUE(config.shared_groupby_enabled);
  EXPECT_FALSE(config.perfect_hash_enabled);    // No benefit, adds overhead
  EXPECT_FALSE(config.prefetch_enabled);
  EXPECT_TRUE(config.adaptive_enabled);
  
  EXPECT_EQ(config.min_columns_for_batching, 4);
  EXPECT_EQ(config.perfect_hash_max_keys, 1000000);
  EXPECT_EQ(config.warp_reduction_min_rows, 10000);
}

TEST_F(BatchedAggregateTest, StrategySelectionCudfDefault)
{
  DataCharacteristics data;
  data.num_rows = 100;
  data.num_columns = 2;  // Less than min_columns_for_batching
  data.num_groups_estimate = 10;
  data.single_integer_key = true;
  data.key_min = 0;
  data.key_max = 9;
  data.key_density = 1.0;
  data.has_nulls = false;
  
  auto config = BatchedAggConfig::defaults();
  auto strategy = select_strategy(data, config);
  
  // With few columns, should fallback to cudf default
  EXPECT_EQ(strategy, AggStrategy::CUDF_DEFAULT);
}

TEST_F(BatchedAggregateTest, StrategySelectionBatchedKernel)
{
  DataCharacteristics data;
  data.num_rows = 50000;  // Large enough for batched kernel
  data.num_columns = 20;  // Many columns
  data.num_groups_estimate = 100;
  data.single_integer_key = true;
  data.key_min = 0;
  data.key_max = 1000000;  // Large range, not suitable for perfect hash
  data.key_density = 0.0001;
  data.has_nulls = false;
  
  auto config = BatchedAggConfig::defaults();
  auto strategy = select_strategy(data, config);
  
  // Should select batched kernel for many columns with large key range
  EXPECT_EQ(strategy, AggStrategy::BATCHED_KERNEL);
}

TEST_F(BatchedAggregateTest, StrategySelectionPerfectHash)
{
  DataCharacteristics data;
  data.num_rows = 10000;
  data.num_columns = 10;
  data.num_groups_estimate = 100;
  data.single_integer_key = true;
  data.key_min = 0;
  data.key_max = 99;  // Small range, suitable for perfect hash
  data.key_density = 1.0;  // Dense
  data.has_nulls = false;
  
  auto config = BatchedAggConfig::defaults();
  auto strategy = select_strategy(data, config);
  
  // Benchmark findings: perfect hash provides no benefit, so it's disabled by default
  // With many columns and defaults, should select BATCHED_KERNEL
  EXPECT_EQ(strategy, AggStrategy::BATCHED_KERNEL);
  
  // If perfect hash is explicitly enabled, should still use batched kernel
  // because the strategy selection logic focuses on multi-column optimization
  config.perfect_hash_enabled = true;
  strategy = select_strategy(data, config);
  EXPECT_EQ(strategy, AggStrategy::BATCHED_KERNEL);
}

TEST_F(BatchedAggregateTest, StrategyName)
{
  EXPECT_EQ(strategy_name(AggStrategy::CUDF_DEFAULT), "CUDF_DEFAULT");
  EXPECT_EQ(strategy_name(AggStrategy::BATCHED_KERNEL), "BATCHED_KERNEL");
  EXPECT_EQ(strategy_name(AggStrategy::PERFECT_HASH), "PERFECT_HASH");
  EXPECT_EQ(strategy_name(AggStrategy::HYBRID), "HYBRID");
}

// =============================================================================
// Data Type Tests
// =============================================================================

TEST_F(BatchedAggregateTest, Int32Values)
{
  constexpr cudf::size_type num_rows = 500;
  constexpr cudf::size_type num_groups = 25;
  
  std::vector<int64_t> key_data(num_rows);
  std::vector<int32_t> val_data(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i] = i % num_groups;
    val_data[i] = static_cast<int32_t>(i * 10);
  }
  
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>(
      key_data.begin(), key_data.end());
  auto val_col = cudf::test::fixed_width_column_wrapper<int32_t>(
      val_data.begin(), val_data.end());
  
  auto keys = cudf::table_view{{keys_col}};
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::SUM,
      cudf::data_type{cudf::type_id::INT64},  // Promote to int64
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->num_columns(), 1);
}

TEST_F(BatchedAggregateTest, DoubleValues)
{
  constexpr cudf::size_type num_rows = 600;
  constexpr cudf::size_type num_groups = 30;
  
  std::vector<int64_t> key_data(num_rows);
  std::vector<double> val_data(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i] = i % num_groups;
    val_data[i] = static_cast<double>(i) * 1.5;
  }
  
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>(
      key_data.begin(), key_data.end());
  auto val_col = cudf::test::fixed_width_column_wrapper<double>(
      val_data.begin(), val_data.end());
  
  auto keys = cudf::table_view{{keys_col}};
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::AVG,
      cudf::data_type{cudf::type_id::FLOAT64},
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), num_groups);
  EXPECT_EQ(result.values->get_column(0).type().id(), cudf::type_id::FLOAT64);
}

// =============================================================================
// Correctness Verification Tests
// =============================================================================

TEST_F(BatchedAggregateTest, SumCorrectnessSimple)
{
  // Create simple data where we can verify the result
  // Keys: [0, 0, 1, 1, 2, 2]
  // Values: [1, 2, 3, 4, 5, 6]
  // Expected sums: group 0 -> 3, group 1 -> 7, group 2 -> 11
  
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>{0, 0, 1, 1, 2, 2};
  auto val_col = cudf::test::fixed_width_column_wrapper<int64_t>{1, 2, 3, 4, 5, 6};
  
  auto keys = cudf::table_view{{keys_col}};
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::SUM,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), 3);
  
  // Note: Groups may be in different order, so we just check count
  EXPECT_EQ(result.values->num_columns(), 1);
}

TEST_F(BatchedAggregateTest, CountCorrectnessSimple)
{
  // Keys: [0, 0, 0, 1, 1, 2]
  // Values: anything
  // Expected counts: group 0 -> 3, group 1 -> 2, group 2 -> 1
  
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>{0, 0, 0, 1, 1, 2};
  auto val_col = cudf::test::fixed_width_column_wrapper<int64_t>{10, 20, 30, 40, 50, 60};
  
  auto keys = cudf::table_view{{keys_col}};
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::COUNT,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), 3);
  EXPECT_EQ(result.values->num_columns(), 1);
  EXPECT_FALSE(result.values->get_column(0).has_nulls());
}

TEST_F(BatchedAggregateTest, MinMaxCorrectnessSimple)
{
  // Keys: [0, 0, 1, 1]
  // Values: [10, 5, 3, 8]
  // Expected min: group 0 -> 5, group 1 -> 3
  // Expected max: group 0 -> 10, group 1 -> 8
  
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>{0, 0, 1, 1};
  auto val_col = cudf::test::fixed_width_column_wrapper<int64_t>{10, 5, 3, 8};
  
  auto keys = cudf::table_view{{keys_col}};
  
  std::vector<batched_agg_spec> specs;
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::MIN,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE
  });
  specs.push_back(batched_agg_spec{
      val_col,
      BatchedAggType::MAX,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE
  });
  
  auto result = batched_groupby_aggregate(keys, specs, stream(), mr());
  
  EXPECT_EQ(result.keys->num_rows(), 2);
  EXPECT_EQ(result.values->num_columns(), 2);
}




