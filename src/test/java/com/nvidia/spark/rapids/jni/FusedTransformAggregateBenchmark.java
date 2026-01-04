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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Test;

/**
 * Micro-benchmark for FusedTransformAggregate JNI performance.
 * 
 * Tests fused Project + Aggregate kernel vs standard cudf groupby.
 */
public class FusedTransformAggregateBenchmark {
    
    private static final int WARMUP_ITERATIONS = 3;
    private static final int BENCHMARK_ITERATIONS = 5;
    
    @Test
    public void benchmarkFusedVsStandard() {
        System.out.println("\n=== FusedTransformAggregate JNI Benchmark ===\n");
        
        // Test different data sizes (reduced for stability)
        int[] rowCounts = {10_000, 100_000, 500_000};
        int[] groupCounts = {100, 1000};
        
        for (int numRows : rowCounts) {
            for (int numGroups : groupCounts) {
                if (numGroups > numRows / 10) continue; // Skip if too many groups
                
                System.out.printf("--- Rows: %,d, Groups: %,d ---\n", numRows, numGroups);
                runBenchmark(numRows, numGroups);
                System.out.println();
            }
        }
    }
    
    @Test
    public void benchmarkSimpleSumAgg() {
        System.out.println("\n=== Simple SUM Aggregation Benchmark ===\n");
        
        int numRows = 100_000;
        int numGroups = 1000;
        
        System.out.printf("Configuration: %,d rows, %,d groups\n\n", numRows, numGroups);
        runBenchmark(numRows, numGroups);
    }
    
    @Test
    public void benchmarkMultipleExpressions() {
        System.out.println("\n=== Multiple Expressions Benchmark ===\n");
        
        int numRows = 100_000;
        int numGroups = 1000;
        
        System.out.printf("Configuration: %,d rows, %,d groups, multiple aggregations\n\n", numRows, numGroups);
        runMultiExprBenchmark(numRows, numGroups);
    }
    
    private void runBenchmark(int numRows, int numGroups) {
        // Create test data: groupKey (int), value1 (long), value2 (long)
        try (ColumnVector groupKey = generateGroupKeys(numRows, numGroups);
             ColumnVector value1 = generateValues(numRows);
             ColumnVector value2 = generateValues(numRows);
             Table inputTable = new Table(groupKey, value1, value2)) {
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                runFused(inputTable);
                runStandardGroupBy(inputTable);
            }
            
            // Benchmark Fused
            long fusedTotalNs = 0;
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                runFused(inputTable);
                fusedTotalNs += System.nanoTime() - start;
            }
            double fusedAvgMs = (fusedTotalNs / BENCHMARK_ITERATIONS) / 1_000_000.0;
            
            // Benchmark Standard cudf groupby
            long stdTotalNs = 0;
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                runStandardGroupBy(inputTable);
                stdTotalNs += System.nanoTime() - start;
            }
            double stdAvgMs = (stdTotalNs / BENCHMARK_ITERATIONS) / 1_000_000.0;
            
            // Print results
            double speedup = stdAvgMs / fusedAvgMs;
            System.out.printf("  Fused:    %.2f ms (avg over %d runs)\n", fusedAvgMs, BENCHMARK_ITERATIONS);
            System.out.printf("  Standard: %.2f ms (avg over %d runs)\n", stdAvgMs, BENCHMARK_ITERATIONS);
            System.out.printf("  Speedup:  %.2fx %s\n", speedup, speedup > 1 ? "FASTER" : "SLOWER");
        }
    }
    
    private void runMultiExprBenchmark(int numRows, int numGroups) {
        // Create test data with more columns for complex expressions
        try (ColumnVector groupKey = generateGroupKeys(numRows, numGroups);
             ColumnVector value1 = generateValues(numRows);
             ColumnVector value2 = generateValues(numRows);
             ColumnVector value3 = generateValues(numRows);
             ColumnVector condCol = generateBooleans(numRows);
             Table inputTable = new Table(groupKey, value1, value2, value3, condCol)) {
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                runFusedMulti(inputTable);
                runStandardGroupByMulti(inputTable);
            }
            
            // Benchmark Fused
            long fusedTotalNs = 0;
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                runFusedMulti(inputTable);
                fusedTotalNs += System.nanoTime() - start;
            }
            double fusedAvgMs = (fusedTotalNs / BENCHMARK_ITERATIONS) / 1_000_000.0;
            
            // Benchmark Standard cudf groupby
            long stdTotalNs = 0;
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                runStandardGroupByMulti(inputTable);
                stdTotalNs += System.nanoTime() - start;
            }
            double stdAvgMs = (stdTotalNs / BENCHMARK_ITERATIONS) / 1_000_000.0;
            
            // Print results
            double speedup = stdAvgMs / fusedAvgMs;
            System.out.printf("  Fused (multi-expr):    %.2f ms (avg over %d runs)\n", fusedAvgMs, BENCHMARK_ITERATIONS);
            System.out.printf("  Standard (multi-expr): %.2f ms (avg over %d runs)\n", stdAvgMs, BENCHMARK_ITERATIONS);
            System.out.printf("  Speedup:               %.2fx %s\n", speedup, speedup > 1 ? "FASTER" : "SLOWER");
        }
    }
    
    private void runFused(Table inputTable) {
        // Group by column 0, SUM column 1, SUM column 2
        FusedTransformAggregate.ExpressionBuilder builder = 
            new FusedTransformAggregate.ExpressionBuilder()
                .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                .addIdentity(2, FusedTransformAggregate.AGG_SUM);
        
        try (FusedTransformAggregate.FusedResult result = 
                 FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
            // Consume result to ensure kernel completes
            if (result.getKeys() != null) {
                result.getKeys().getRowCount();
            }
        }
    }
    
    private void runFusedMulti(Table inputTable) {
        // Group by column 0
        // SUM(value1), SUM(value2), SUM(value3), COUNT(*)
        FusedTransformAggregate.ExpressionBuilder builder = 
            new FusedTransformAggregate.ExpressionBuilder()
                .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                .addIdentity(2, FusedTransformAggregate.AGG_SUM)
                .addIdentity(3, FusedTransformAggregate.AGG_SUM)
                .addCountAll(FusedTransformAggregate.AGG_COUNT);
        
        try (FusedTransformAggregate.FusedResult result = 
                 FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
            if (result.getKeys() != null) {
                result.getKeys().getRowCount();
            }
        }
    }
    
    private void runStandardGroupBy(Table inputTable) {
        // Standard cudf groupby: group by column 0, SUM columns 1 and 2
        try (Table result = inputTable.groupBy(0).aggregate(
                GroupByAggregation.sum().onColumn(1),
                GroupByAggregation.sum().onColumn(2))) {
            // Consume result to ensure kernel completes
            result.getRowCount();
        }
    }
    
    private void runStandardGroupByMulti(Table inputTable) {
        // Standard cudf groupby with multiple aggregations
        try (Table result = inputTable.groupBy(0).aggregate(
                GroupByAggregation.sum().onColumn(1),
                GroupByAggregation.sum().onColumn(2),
                GroupByAggregation.sum().onColumn(3),
                GroupByAggregation.count().onColumn(1))) {
            result.getRowCount();
        }
    }
    
    private ColumnVector generateGroupKeys(int numRows, int numGroups) {
        // Generate group keys in range [0, numGroups)
        int[] keys = new int[numRows];
        for (int i = 0; i < numRows; i++) {
            keys[i] = i % numGroups;
        }
        return ColumnVector.fromInts(keys);
    }
    
    private ColumnVector generateValues(int numRows) {
        // Generate random-ish long values
        long[] values = new long[numRows];
        for (int i = 0; i < numRows; i++) {
            values[i] = (i * 17 + 42) % 10000;
        }
        return ColumnVector.fromLongs(values);
    }
    
    private ColumnVector generateBooleans(int numRows) {
        // Generate alternating boolean values
        boolean[] values = new boolean[numRows];
        for (int i = 0; i < numRows; i++) {
            values[i] = (i % 2 == 0);
        }
        return ColumnVector.fromBooleans(values);
    }
}

