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

#include <benchmarks/common/generate_input.hpp>

#include <batched_aggregate.hpp>
#include <batched_aggregate_config.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

using namespace spark_rapids_jni;

namespace {

/**
 * @brief Generate test data for benchmarking
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<std::unique_ptr<cudf::column>>>
generate_benchmark_data(cudf::size_type num_rows, cudf::size_type num_groups, int num_value_cols) {
  // Generate key column using random data generator
  data_profile_builder key_builder;
  key_builder.no_validity();
  key_builder.cardinality(num_groups);
  auto key_table = create_random_table({{cudf::type_id::INT64}}, row_count{num_rows}, key_builder);
  
  // Generate value columns
  std::vector<std::unique_ptr<cudf::column>> value_cols;
  data_profile_builder value_builder;
  value_builder.no_validity();
  
  for (int c = 0; c < num_value_cols; ++c) {
    auto val_table = create_random_table({{cudf::type_id::INT64}}, row_count{num_rows}, value_builder);
    value_cols.push_back(std::move(val_table->release()[0]));
  }
  
  return {std::move(key_table), std::move(value_cols)};
}

/**
 * @brief Build aggregation specs for benchmarking
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
 * @brief Build aggregation specs for benchmarking
 */
std::vector<batched_agg_spec> build_benchmark_specs(
    std::vector<std::unique_ptr<cudf::column>> const& value_cols,
    BatchedAggType agg_type) {
  std::vector<batched_agg_spec> specs;
  for (auto const& col : value_cols) {
    auto out_type = (agg_type == BatchedAggType::AVG) ? 
        cudf::data_type{cudf::type_id::FLOAT64} : 
        cudf::data_type{cudf::type_id::INT64};
    specs.push_back(batched_agg_spec{
        col->view(),
        agg_type,
        out_type,
        cudf::null_policy::EXCLUDE
    });
  }
  return specs;
}

/**
 * @brief Build mixed aggregation specs (SUM, COUNT, AVG, MIN, MAX)
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
// Benchmark: Batched vs Standard cudf groupby
// =============================================================================

static void bench_batched_vs_standard(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_cols = static_cast<int>(state.get_int64("num_cols"));
  auto const use_batched = state.get_int64("use_batched") != 0;
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  
  if (use_batched) {
    // Batched aggregate path
    state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
               [&](nvbench::launch& launch, auto& timer) {
                 timer.start();
                 auto result = batched_groupby_aggregate(keys, specs, stream, mr);
                 stream.synchronize();
                 timer.stop();
               });
  } else {
    // Standard cudf groupby path
    cudf::groupby::groupby groupby_obj(keys, cudf::null_policy::EXCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests;
    for (auto const& col : value_cols) {
      cudf::groupby::aggregation_request req;
      req.values = col->view();
      req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      requests.push_back(std::move(req));
    }
    
    state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
               [&](nvbench::launch& launch, auto& timer) {
                 timer.start();
                 auto result = groupby_obj.aggregate(requests, stream, mr);
                 stream.synchronize();
                 timer.stop();
               });
  }
  
  // Calculate throughput
  size_t const bytes_read = num_rows * sizeof(int64_t) * (1 + num_cols);
  size_t const bytes_written = num_groups * sizeof(int64_t) * (1 + num_cols);
  state.add_element_count(static_cast<std::size_t>(num_rows), "Rows");
  state.add_global_memory_reads(bytes_read, "Read");
  state.add_global_memory_writes(bytes_written, "Write");
}

NVBENCH_BENCH(bench_batched_vs_standard)
  .set_name("Batched vs Standard Aggregate")
  .add_int64_axis("num_rows", {100000, 500000, 1000000, 5000000})
  .add_int64_axis("num_groups", {100, 1000, 10000})
  .add_int64_axis("num_cols", {10, 20, 50, 100})
  .add_int64_axis("use_batched", {0, 1});

// =============================================================================
// Benchmark: Different aggregation types
// =============================================================================

static void bench_aggregation_types(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_cols = static_cast<int>(state.get_int64("num_cols"));
  auto const agg_type_id = static_cast<int>(state.get_int64("agg_type"));
  
  BatchedAggType agg_type;
  switch (agg_type_id) {
    case 0: agg_type = BatchedAggType::SUM; break;
    case 1: agg_type = BatchedAggType::COUNT; break;
    case 2: agg_type = BatchedAggType::AVG; break;
    case 3: agg_type = BatchedAggType::MIN; break;
    case 4: agg_type = BatchedAggType::MAX; break;
    default: agg_type = BatchedAggType::SUM;
  }
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, agg_type);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  state.add_element_count(static_cast<std::size_t>(num_rows), "Rows");
}

NVBENCH_BENCH(bench_aggregation_types)
  .set_name("Aggregation Types Performance")
  .add_int64_axis("num_rows", {1000000})
  .add_int64_axis("num_groups", {1000})
  .add_int64_axis("num_cols", {20})
  .add_int64_axis("agg_type", {0, 1, 2, 3, 4});  // SUM, COUNT, AVG, MIN, MAX

// =============================================================================
// Benchmark: Mixed aggregation types
// =============================================================================

static void bench_mixed_aggregations(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_cols = static_cast<int>(state.get_int64("num_cols"));
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_mixed_specs(value_cols);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  size_t const bytes_read = num_rows * sizeof(int64_t) * (1 + num_cols);
  state.add_element_count(static_cast<std::size_t>(num_rows), "Rows");
  state.add_global_memory_reads(bytes_read, "Read");
}

NVBENCH_BENCH(bench_mixed_aggregations)
  .set_name("Mixed Aggregations (SUM/COUNT/AVG/MIN/MAX)")
  .add_int64_axis("num_rows", {500000, 1000000, 5000000})
  .add_int64_axis("num_groups", {100, 1000, 10000})
  .add_int64_axis("num_cols", {10, 25, 50, 100});

// =============================================================================
// Benchmark: Column count scaling
// =============================================================================

static void bench_column_scaling(nvbench::state& state)
{
  auto const num_rows = 1000000;
  auto const num_groups = 1000;
  auto const num_cols = static_cast<int>(state.get_int64("num_cols"));
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  state.add_element_count(static_cast<std::size_t>(num_cols), "Columns");
  state.add_element_count(static_cast<std::size_t>(num_rows * num_cols), "Elements");
}

NVBENCH_BENCH(bench_column_scaling)
  .set_name("Column Count Scaling")
  .add_int64_axis("num_cols", {5, 10, 20, 30, 50, 75, 100, 150, 200});

// =============================================================================
// Benchmark: Group count scaling
// =============================================================================

static void bench_group_scaling(nvbench::state& state)
{
  auto const num_rows = 2000000;
  auto const num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_cols = 30;
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  state.add_element_count(static_cast<std::size_t>(num_groups), "Groups");
}

NVBENCH_BENCH(bench_group_scaling)
  .set_name("Group Count Scaling")
  .add_int64_axis("num_groups", {10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000});

// =============================================================================
// Benchmark: Row count scaling
// =============================================================================

static void bench_row_scaling(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_groups = 1000;
  auto const num_cols = 30;
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  size_t const bytes_processed = num_rows * sizeof(int64_t) * (1 + num_cols);
  state.add_element_count(static_cast<std::size_t>(num_rows), "Rows");
  state.add_global_memory_reads(bytes_processed, "Bytes");
}

NVBENCH_BENCH(bench_row_scaling)
  .set_name("Row Count Scaling")
  .add_int64_axis("num_rows", {100000, 250000, 500000, 1000000, 2000000, 5000000, 10000000});

// =============================================================================
// Benchmark: Memory allocation comparison
// =============================================================================

static void bench_allocation_overhead(nvbench::state& state)
{
  // This benchmark measures the benefit of contiguous output buffer
  // by comparing allocation patterns
  auto const num_rows = 1000000;
  auto const num_groups = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_cols = static_cast<int>(state.get_int64("num_cols"));
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  // Report allocation count reduction
  // Standard: num_cols * 2 allocations (data + validity per column)
  // Batched: 1 allocation (contiguous buffer)
  state.add_element_count(static_cast<std::size_t>(num_cols * 2), "Standard allocs");
  state.add_element_count(1, "Batched allocs");
}

NVBENCH_BENCH(bench_allocation_overhead)
  .set_name("Allocation Overhead (Contiguous Output)")
  .add_int64_axis("num_groups", {100, 1000})
  .add_int64_axis("num_cols", {20, 50, 100, 200});

// =============================================================================
// Benchmark: Warp reduction benefit
// =============================================================================

static void bench_warp_reduction(nvbench::state& state)
{
  // Warp reduction is most beneficial with many rows per group
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const rows_per_group = static_cast<cudf::size_type>(state.get_int64("rows_per_group"));
  auto const num_groups = num_rows / rows_per_group;
  auto const num_cols = 30;
  
  auto [key_table, value_cols] = generate_benchmark_data(num_rows, num_groups, num_cols);
  auto keys = key_table->view();
  auto specs = build_benchmark_specs(value_cols, BatchedAggType::SUM);
  
  auto const stream = cudf::get_default_stream();
  auto const mr = cudf::get_current_device_resource_ref();
  
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = batched_groupby_aggregate(keys, specs, stream, mr);
               stream.synchronize();
               timer.stop();
             });
  
  state.add_element_count(static_cast<std::size_t>(rows_per_group), "Rows/Group");
  state.add_element_count(static_cast<std::size_t>(num_groups), "Groups");
}

NVBENCH_BENCH(bench_warp_reduction)
  .set_name("Warp Reduction Benefit")
  .add_int64_axis("num_rows", {2000000})
  .add_int64_axis("rows_per_group", {10, 32, 64, 128, 256, 512, 1024, 2048});



