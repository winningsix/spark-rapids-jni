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

/**
 * @file jit_perf_benchmark.cu
 * @brief Performance comparison between hand-written kernel and JIT implementation
 * 
 * Compile: nvcc -std=c++20 -O3 jit_perf_benchmark.cu -o jit_perf_benchmark
 * Run: ./jit_perf_benchmark
 */

#include "fused_transform_aggregate.hpp"
#include "fused_transform_aggregate_jit.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

namespace spark_rapids_jni {
namespace benchmark {

/**
 * @brief Create test data with specified size and number of groups
 */
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

/**
 * @brief Create a plan with multiple expressions (SUM, COUNT, AVG)
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
 * @brief Run benchmark for a specific data size
 */
void run_benchmark(cudf::size_type num_rows, 
                   cudf::size_type num_groups,
                   int warmup_iterations,
                   int benchmark_iterations,
                   rmm::cuda_stream_view stream) {
  
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "Benchmark: " << num_rows << " rows, " << num_groups << " groups" << std::endl;
  std::cout << "Expressions: SUM, COUNT, AVG" << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  
  auto data = create_test_data(num_rows, num_groups, stream);
  auto plan = create_multi_expr_plan();
  
  // ============================================================================
  // Hand-written kernel benchmark
  // ============================================================================
  std::cout << "\n[Hand-written Kernel]" << std::endl;
  
  // Warmup
  for (int i = 0; i < warmup_iterations; ++i) {
    auto result = execute_fused_transform_aggregate(data->view(), plan, stream);
    stream.synchronize();
  }
  
  // Benchmark
  double hw_total_time = 0.0;
  int hw_groups = 0;
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = execute_fused_transform_aggregate(data->view(), plan, stream);
    stream.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    hw_total_time += std::chrono::duration<double, std::milli>(end - start).count();
    hw_groups = result.num_groups();
  }
  double hw_avg_time = hw_total_time / benchmark_iterations;
  
  std::cout << "  Average time: " << std::fixed << std::setprecision(3) 
            << hw_avg_time << " ms" << std::endl;
  std::cout << "  Output groups: " << hw_groups << std::endl;
  
  // ============================================================================
  // JIT kernel benchmark
  // ============================================================================
  std::cout << "\n[JIT Transform Kernel]" << std::endl;
  
  // Warmup (includes JIT compilation)
  for (int i = 0; i < warmup_iterations; ++i) {
    auto result = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
    stream.synchronize();
  }
  
  // Benchmark
  double jit_total_time = 0.0;
  int jit_groups = 0;
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = jit::execute_fused_transform_aggregate_jit(data->view(), plan, stream);
    stream.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    jit_total_time += std::chrono::duration<double, std::milli>(end - start).count();
    jit_groups = result.num_groups();
  }
  double jit_avg_time = jit_total_time / benchmark_iterations;
  
  std::cout << "  Average time: " << std::fixed << std::setprecision(3)
            << jit_avg_time << " ms" << std::endl;
  std::cout << "  Output groups: " << jit_groups << std::endl;
  
  // ============================================================================
  // Comparison
  // ============================================================================
  std::cout << "\n[Comparison]" << std::endl;
  double speedup = jit_avg_time / hw_avg_time;
  std::cout << "  Hand-written / JIT speedup: " << std::fixed << std::setprecision(2)
            << (1.0 / speedup) << "x" << std::endl;
  
  if (speedup > 1.0) {
    std::cout << "  -> Hand-written kernel is " << std::fixed << std::setprecision(1)
              << ((speedup - 1.0) * 100) << "% faster" << std::endl;
  } else {
    std::cout << "  -> JIT kernel is " << std::fixed << std::setprecision(1)
              << ((1.0/speedup - 1.0) * 100) << "% faster" << std::endl;
  }
}

}  // namespace benchmark
}  // namespace spark_rapids_jni

// ============================================================================
// Exported function for calling from JNI or other code
// ============================================================================

extern "C" {

void run_jit_performance_benchmark() {
  auto stream = cudf::get_default_stream();
  
  std::cout << "\n" << std::string(60, '#') << std::endl;
  std::cout << "# Fused Transform Aggregate: JIT vs Hand-written Benchmark" << std::endl;
  std::cout << std::string(60, '#') << std::endl;
  
  // Small data
  spark_rapids_jni::benchmark::run_benchmark(10000, 100, 3, 10, stream);
  
  // Medium data
  spark_rapids_jni::benchmark::run_benchmark(100000, 1000, 3, 10, stream);
  
  // Large data
  spark_rapids_jni::benchmark::run_benchmark(1000000, 10000, 2, 5, stream);
  
  // Very large data
  spark_rapids_jni::benchmark::run_benchmark(10000000, 100000, 2, 3, stream);
  
  std::cout << "\n" << std::string(60, '#') << std::endl;
  std::cout << "# Benchmark Complete" << std::endl;
  std::cout << std::string(60, '#') << std::endl;
}

}  // extern "C"


