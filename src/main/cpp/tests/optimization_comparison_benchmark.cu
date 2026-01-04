/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Optimization Comparison Benchmark (Simplified)
 * Tests key optimizations for aggregation kernels
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/random.h>

#include <chrono>
#include <iostream>
#include <iomanip>

namespace spark_rapids_jni {
namespace opt_benchmark {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t SHMEM_MAX_GROUPS = 128;

// Helper
__device__ __forceinline__ cudf::size_type atomicCAS_sizetype(
    cudf::size_type* addr, cudf::size_type compare, cudf::size_type val) {
    return atomicCAS(reinterpret_cast<unsigned int*>(addr),
                     static_cast<unsigned int>(compare),
                     static_cast<unsigned int>(val));
}

// ============================================================================
// 1. Baseline: Naive Global Atomic
// ============================================================================
__global__ void kernel_baseline(
    int64_t const* __restrict__ values,
    cudf::size_type const* __restrict__ group_ids,
    cudf::size_type num_rows,
    unsigned long long* __restrict__ output)
{
    cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    atomicAdd(&output[group_ids[idx]], static_cast<unsigned long long>(values[idx]));
}

// ============================================================================
// 2. Perfect Hash: Key as direct index (no hash computation)
// ============================================================================
__global__ void kernel_perfect_hash(
    int64_t const* __restrict__ values,
    int64_t const* __restrict__ keys,
    cudf::size_type num_rows,
    unsigned long long* __restrict__ output)
{
    cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    // Key is directly the index - no hash needed
    atomicAdd(&output[keys[idx]], static_cast<unsigned long long>(values[idx]));
}

// ============================================================================
// 3. L2 Cache Hint (__ldg)
// ============================================================================
__global__ void kernel_ldg_cached(
    int64_t const* __restrict__ values,
    cudf::size_type const* __restrict__ group_ids,
    cudf::size_type num_rows,
    unsigned long long* __restrict__ output)
{
    cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    // Use __ldg for L2 cached reads
    atomicAdd(&output[__ldg(&group_ids[idx])], 
              static_cast<unsigned long long>(__ldg(&values[idx])));
}

// ============================================================================
// 4. Shared Memory Aggregation
// ============================================================================
__global__ void kernel_shared_memory(
    int64_t const* __restrict__ values,
    cudf::size_type const* __restrict__ group_ids,
    cudf::size_type num_rows,
    unsigned long long* __restrict__ output)
{
    extern __shared__ char shared_mem[];
    auto* shmem_map = reinterpret_cast<cudf::size_type*>(shared_mem);
    auto* shmem_accum = reinterpret_cast<unsigned long long*>(&shmem_map[SHMEM_MAX_GROUPS]);
    
    // Init shared memory
    for (int i = threadIdx.x; i < SHMEM_MAX_GROUPS; i += blockDim.x) {
        shmem_map[i] = static_cast<cudf::size_type>(-1);
        shmem_accum[i] = 0;
    }
    __syncthreads();
    
    cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_rows) {
        cudf::size_type group = group_ids[idx];
        int64_t val = values[idx];
        
        // Linear probe in shared memory
        int slot = group % SHMEM_MAX_GROUPS;
        bool found = false;
        
        for (int i = 0; i < 16 && !found; ++i) {  // Limit probes
            int probe = (slot + i) % SHMEM_MAX_GROUPS;
            cudf::size_type existing = atomicCAS_sizetype(&shmem_map[probe],
                                                          static_cast<cudf::size_type>(-1),
                                                          group);
            if (existing == static_cast<cudf::size_type>(-1) || existing == group) {
                atomicAdd(&shmem_accum[probe], static_cast<unsigned long long>(val));
                found = true;
            }
        }
        
        if (!found) {
            // Fallback to global
            atomicAdd(&output[group], static_cast<unsigned long long>(val));
        }
    }
    __syncthreads();
    
    // Flush to global
    for (int i = threadIdx.x; i < SHMEM_MAX_GROUPS; i += blockDim.x) {
        if (shmem_map[i] != static_cast<cudf::size_type>(-1) && shmem_accum[i] != 0) {
            atomicAdd(&output[shmem_map[i]], shmem_accum[i]);
        }
    }
}

// ============================================================================
// 5. Multi-Column (amortize group lookup)
// ============================================================================
__global__ void kernel_multi_column(
    int64_t const* const* __restrict__ value_cols,
    int32_t num_cols,
    cudf::size_type const* __restrict__ group_ids,
    cudf::size_type num_rows,
    unsigned long long* const* __restrict__ outputs)
{
    cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    cudf::size_type group = group_ids[idx];  // Single lookup for all columns
    
    for (int c = 0; c < num_cols; ++c) {
        atomicAdd(&outputs[c][group], static_cast<unsigned long long>(value_cols[c][idx]));
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

void run_benchmark(cudf::size_type num_rows, cudf::size_type num_groups,
                   rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
    std::cout << "\n================================================================\n";
    std::cout << "Rows: " << num_rows << ", Groups: " << num_groups << "\n";
    std::cout << "================================================================\n";
    
    // Allocate data
    rmm::device_uvector<int64_t> values(num_rows, stream, mr);
    rmm::device_uvector<int64_t> keys(num_rows, stream, mr);
    rmm::device_uvector<cudf::size_type> group_ids(num_rows, stream, mr);
    rmm::device_uvector<unsigned long long> output(num_groups, stream, mr);
    
    // Initialize
    thrust::transform(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<int64_t>(0),
                     thrust::make_counting_iterator<int64_t>(num_rows),
                     values.begin(),
                     [] __device__ (int64_t i) { return (i % 100) + 1; });
    
    thrust::transform(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(num_rows),
                     group_ids.begin(),
                     [num_groups] __device__ (cudf::size_type i) { return i % num_groups; });
    
    thrust::transform(rmm::exec_policy(stream),
                     group_ids.begin(), group_ids.end(), keys.begin(),
                     [] __device__ (cudf::size_type g) { return static_cast<int64_t>(g); });
    
    stream.synchronize();
    
    int num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shmem_size = SHMEM_MAX_GROUPS * (sizeof(cudf::size_type) + sizeof(unsigned long long));
    
    constexpr int WARMUP = 2;
    constexpr int RUNS = 5;
    
    auto benchmark = [&](const char* name, auto kernel_fn) {
        for (int i = 0; i < WARMUP; ++i) {
            thrust::fill(rmm::exec_policy(stream), output.begin(), output.end(), 0ULL);
            kernel_fn();
        }
        stream.synchronize();
        
        auto start = Clock::now();
        for (int i = 0; i < RUNS; ++i) {
            thrust::fill(rmm::exec_policy(stream), output.begin(), output.end(), 0ULL);
            kernel_fn();
        }
        stream.synchronize();
        auto end = Clock::now();
        
        double avg_ms = Duration(end - start).count() / RUNS;
        std::cout << "  " << std::setw(25) << std::left << name 
                  << std::fixed << std::setprecision(3) << std::setw(10) << avg_ms << " ms\n";
        return avg_ms;
    };
    
    std::cout << "\n  Optimization                   Time\n";
    std::cout << "  -----------------------------------------\n";
    
    double t_baseline = benchmark("Baseline (Global Atomic)", [&]() {
        kernel_baseline<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
            values.data(), group_ids.data(), num_rows, output.data());
    });
    
    double t_perfect = benchmark("Perfect Hash", [&]() {
        kernel_perfect_hash<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
            values.data(), keys.data(), num_rows, output.data());
    });
    
    double t_ldg = benchmark("L2 Cache (__ldg)", [&]() {
        kernel_ldg_cached<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
            values.data(), group_ids.data(), num_rows, output.data());
    });
    
    double t_shmem = benchmark("Shared Memory", [&]() {
        kernel_shared_memory<<<num_blocks, BLOCK_SIZE, shmem_size, stream.value()>>>(
            values.data(), group_ids.data(), num_rows, output.data());
    });
    
    std::cout << "\n  Speedup vs Baseline:\n";
    std::cout << "  -----------------------------------------\n";
    std::cout << "  Perfect Hash:     " << std::setprecision(2) << t_baseline / t_perfect << "x\n";
    std::cout << "  L2 Cache:         " << t_baseline / t_ldg << "x\n";
    std::cout << "  Shared Memory:    " << t_baseline / t_shmem << "x\n";
}

void run_multi_col_benchmark(cudf::size_type num_rows, cudf::size_type num_groups,
                              rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
    std::cout << "\n================================================================\n";
    std::cout << "Multi-Column Test: " << num_rows << " rows, " << num_groups << " groups\n";
    std::cout << "================================================================\n";
    
    constexpr int NUM_COLS = 10;
    
    std::vector<rmm::device_uvector<int64_t>> value_cols;
    std::vector<int64_t*> value_ptrs_h(NUM_COLS);
    std::vector<rmm::device_uvector<unsigned long long>> out_cols;
    std::vector<unsigned long long*> out_ptrs_h(NUM_COLS);
    
    for (int i = 0; i < NUM_COLS; ++i) {
        value_cols.emplace_back(num_rows, stream, mr);
        thrust::fill(rmm::exec_policy(stream), value_cols[i].begin(), value_cols[i].end(), int64_t(i + 1));
        value_ptrs_h[i] = value_cols[i].data();
        
        out_cols.emplace_back(num_groups, stream, mr);
        out_ptrs_h[i] = out_cols[i].data();
    }
    
    rmm::device_uvector<int64_t*> value_ptrs(NUM_COLS, stream, mr);
    rmm::device_uvector<unsigned long long*> out_ptrs(NUM_COLS, stream, mr);
    cudaMemcpyAsync(value_ptrs.data(), value_ptrs_h.data(), NUM_COLS * sizeof(int64_t*), cudaMemcpyHostToDevice, stream.value());
    cudaMemcpyAsync(out_ptrs.data(), out_ptrs_h.data(), NUM_COLS * sizeof(unsigned long long*), cudaMemcpyHostToDevice, stream.value());
    
    rmm::device_uvector<cudf::size_type> group_ids(num_rows, stream, mr);
    thrust::transform(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(num_rows),
                     group_ids.begin(),
                     [num_groups] __device__ (cudf::size_type i) { return i % num_groups; });
    stream.synchronize();
    
    int num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int WARMUP = 2;
    constexpr int RUNS = 5;
    
    auto reset = [&]() {
        for (auto& c : out_cols) thrust::fill(rmm::exec_policy(stream), c.begin(), c.end(), 0ULL);
    };
    
    // Separate kernels
    for (int i = 0; i < WARMUP; ++i) {
        reset();
        for (int c = 0; c < NUM_COLS; ++c) {
            kernel_baseline<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
                value_cols[c].data(), group_ids.data(), num_rows, out_cols[c].data());
        }
    }
    stream.synchronize();
    
    auto start1 = Clock::now();
    for (int i = 0; i < RUNS; ++i) {
        reset();
        for (int c = 0; c < NUM_COLS; ++c) {
            kernel_baseline<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
                value_cols[c].data(), group_ids.data(), num_rows, out_cols[c].data());
        }
    }
    stream.synchronize();
    double t_sep = Duration(Clock::now() - start1).count() / RUNS;
    
    // Multi-column kernel
    for (int i = 0; i < WARMUP; ++i) {
        reset();
        kernel_multi_column<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
            value_ptrs.data(), NUM_COLS, group_ids.data(), num_rows, out_ptrs.data());
    }
    stream.synchronize();
    
    auto start2 = Clock::now();
    for (int i = 0; i < RUNS; ++i) {
        reset();
        kernel_multi_column<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
            value_ptrs.data(), NUM_COLS, group_ids.data(), num_rows, out_ptrs.data());
    }
    stream.synchronize();
    double t_multi = Duration(Clock::now() - start2).count() / RUNS;
    
    std::cout << "\n  " << NUM_COLS << " columns:\n";
    std::cout << "  Separate kernels: " << std::fixed << std::setprecision(3) << t_sep << " ms\n";
    std::cout << "  Multi-column:     " << t_multi << " ms\n";
    std::cout << "  Speedup:          " << std::setprecision(2) << t_sep / t_multi << "x\n";
}

}  // namespace
}  // namespace

int main() {
    std::cout << "============================================================\n";
    std::cout << "  Optimization Comparison Benchmark\n";
    std::cout << "============================================================\n";
    
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    
    using namespace spark_rapids_jni::opt_benchmark;
    
    // Customer scenario: ~2.5M rows per batch (GpuProject input), 186K groups
    std::cout << "\n=== Customer Scenario: 2.5M rows, 186K groups ===\n";
    std::cout << "  (8,163,225,211,914 / 3,242,253 = ~2.5M rows/batch)\n";
    std::cout << "  (~13.5 rows per group - still sparse)\n\n";
    run_benchmark(2517620, 186112, stream, mr);
    
    // Multi-column test matching customer scenario
    run_multi_col_benchmark(2517620, 186112, stream, mr);
    
    std::cout << "\n============================================================\n";
    std::cout << "  Complete!\n";
    std::cout << "============================================================\n";
    
    return 0;
}
