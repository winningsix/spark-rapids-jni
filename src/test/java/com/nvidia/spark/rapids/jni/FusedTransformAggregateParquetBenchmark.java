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

import java.io.File;

/**
 * JNI-level benchmark for FusedTransformAggregate.
 * 
 * Tests performance across multiple dimensions:
 * 1. Rows per group
 * 2. Number of expressions
 * 3. With/without explode (repeat) to simulate real workloads
 */
public class FusedTransformAggregateParquetBenchmark {

    private static final int WARMUP = 2;
    private static final int ITERATIONS = 3;
    
    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.append(s);
        return sb.toString();
    }

    // =========================================================================
    // Main Comprehensive Benchmark
    // =========================================================================
    
    @Test
    public void benchmarkComprehensive() {
        System.out.println("\n" + repeat("=", 80));
        System.out.println("FusedTransformAggregate Comprehensive Benchmark");
        System.out.println(repeat("=", 80));
        
        // Test dimensions
        int[] rowsPerGroupOptions = {100, 500, 1000};      // Rows per group
        int[] expressionCounts = {10, 30, 50};             // Number of expressions (kernel limit ~64)
        int[] repeatFactors = {1, 10};                     // Explode factor (1 = no explode)
        
        int baseRows = 100000;  // Base row count before repeat
        
        System.out.println("\nBase rows: " + String.format("%,d", baseRows));
        System.out.println("Warmup: " + WARMUP + ", Iterations: " + ITERATIONS);
        System.out.println();
        
        // Print header
        System.out.println(String.format("%-12s %-10s %-10s %-12s %-12s %-12s %-10s",
            "Rows/Group", "Exprs", "Repeat", "Fused(ms)", "Std(ms)", "Speedup", "Rows"));
        System.out.println(repeat("-", 80));
        
        for (int repeatFactor : repeatFactors) {
            for (int rowsPerGroup : rowsPerGroupOptions) {
                for (int numExprs : expressionCounts) {
                    runSingleBenchmark(baseRows, rowsPerGroup, numExprs, repeatFactor);
                }
            }
            System.out.println();  // Blank line between repeat factors
        }
        
        System.out.println(repeat("=", 80));
    }
    
    private void runSingleBenchmark(int baseRows, int rowsPerGroup, int numExprs, int repeatFactor) {
        int numGroups = baseRows / rowsPerGroup;
        if (numGroups < 10) numGroups = 10;  // Minimum groups
        
        // Generate base data
        Table baseTable = generateTestData(baseRows, numGroups, numExprs);
        Table inputTable = null;
        
        try {
            // Apply repeat (simulate explode)
            if (repeatFactor > 1) {
                inputTable = baseTable.repeat(repeatFactor);
                baseTable.close();
            } else {
                inputTable = baseTable;
            }
            
            long totalRows = inputTable.getRowCount();
            
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                runFused(inputTable, numExprs);
                runStandard(inputTable, numExprs);
            }
            
            // Benchmark Fused
            long fusedNs = 0;
            for (int i = 0; i < ITERATIONS; i++) {
                long start = System.nanoTime();
                runFused(inputTable, numExprs);
                fusedNs += System.nanoTime() - start;
            }
            double fusedMs = (fusedNs / ITERATIONS) / 1_000_000.0;
            
            // Benchmark Standard
            long stdNs = 0;
            for (int i = 0; i < ITERATIONS; i++) {
                long start = System.nanoTime();
                runStandard(inputTable, numExprs);
                stdNs += System.nanoTime() - start;
            }
            double stdMs = (stdNs / ITERATIONS) / 1_000_000.0;
            
            double speedup = stdMs / fusedMs;
            
            System.out.println(String.format("%-12d %-10d %-10d %-12.2f %-12.2f %-10.2fx %,d",
                rowsPerGroup, numExprs, repeatFactor, fusedMs, stdMs, speedup, totalRows));
            
        } finally {
            if (inputTable != null) inputTable.close();
        }
    }
    
    // =========================================================================
    // Expression Count Scaling Benchmark
    // =========================================================================
    
    @Test
    public void benchmarkExpressionScaling() {
        System.out.println("\n" + repeat("=", 80));
        System.out.println("Expression Count Scaling Benchmark");
        System.out.println(repeat("=", 80));
        
        int baseRows = 50000;
        int numGroups = 1000;
        int repeatFactor = 10;  // Simulate explode
        
        int[] exprCounts = {10, 20, 30, 40, 50, 60};  // Kernel limit ~64 expressions
        
        System.out.println("Base rows: " + String.format("%,d", baseRows));
        System.out.println("Groups: " + String.format("%,d", numGroups));
        System.out.println("Repeat factor: " + repeatFactor);
        System.out.println("Total rows: " + String.format("%,d", baseRows * repeatFactor));
        System.out.println();
        
        System.out.println(String.format("%-15s %-12s %-12s %-10s",
            "Expressions", "Fused(ms)", "Std(ms)", "Speedup"));
        System.out.println(repeat("-", 50));
        
        for (int numExprs : exprCounts) {
            Table baseTable = generateTestData(baseRows, numGroups, numExprs);
            try (Table inputTable = baseTable.repeat(repeatFactor)) {
                baseTable.close();
                
                // Warmup
                for (int i = 0; i < WARMUP; i++) {
                    runFused(inputTable, numExprs);
                    runStandard(inputTable, numExprs);
                }
                
                // Benchmark
                long fusedNs = 0, stdNs = 0;
                for (int i = 0; i < ITERATIONS; i++) {
                    long start = System.nanoTime();
                    runFused(inputTable, numExprs);
                    fusedNs += System.nanoTime() - start;
                    
                    start = System.nanoTime();
                    runStandard(inputTable, numExprs);
                    stdNs += System.nanoTime() - start;
                }
                
                double fusedMs = (fusedNs / ITERATIONS) / 1_000_000.0;
                double stdMs = (stdNs / ITERATIONS) / 1_000_000.0;
                double speedup = stdMs / fusedMs;
                
                System.out.println(String.format("%-15d %-12.2f %-12.2f %.2fx",
                    numExprs, fusedMs, stdMs, speedup));
            }
        }
        
        System.out.println(repeat("=", 80));
    }
    
    // =========================================================================
    // Rows Per Group Scaling Benchmark
    // =========================================================================
    
    @Test
    public void benchmarkRowsPerGroupScaling() {
        System.out.println("\n" + repeat("=", 80));
        System.out.println("Rows Per Group Scaling Benchmark");
        System.out.println(repeat("=", 80));
        
        int totalRows = 100000;  // Fixed total rows (reduced)
        int numExprs = 30;       // Reduced to avoid kernel limit
        int repeatFactor = 5;    // Reduced
        
        int[] rowsPerGroupOptions = {50, 100, 250, 500, 1000, 2000};
        
        System.out.println("Total rows (after repeat): " + String.format("%,d", totalRows * repeatFactor));
        System.out.println("Expressions: " + numExprs);
        System.out.println();
        
        System.out.println(String.format("%-15s %-10s %-12s %-12s %-10s",
            "Rows/Group", "Groups", "Fused(ms)", "Std(ms)", "Speedup"));
        System.out.println(repeat("-", 60));
        
        for (int rowsPerGroup : rowsPerGroupOptions) {
            int numGroups = totalRows / rowsPerGroup;
            if (numGroups < 10) continue;
            
            Table baseTable = generateTestData(totalRows, numGroups, numExprs);
            try (Table inputTable = baseTable.repeat(repeatFactor)) {
                baseTable.close();
                
                // Warmup
                for (int i = 0; i < WARMUP; i++) {
                    runFused(inputTable, numExprs);
                    runStandard(inputTable, numExprs);
                }
                
                // Benchmark
                long fusedNs = 0, stdNs = 0;
                for (int i = 0; i < ITERATIONS; i++) {
                    long start = System.nanoTime();
                    runFused(inputTable, numExprs);
                    fusedNs += System.nanoTime() - start;
                    
                    start = System.nanoTime();
                    runStandard(inputTable, numExprs);
                    stdNs += System.nanoTime() - start;
                }
                
                double fusedMs = (fusedNs / ITERATIONS) / 1_000_000.0;
                double stdMs = (stdNs / ITERATIONS) / 1_000_000.0;
                double speedup = stdMs / fusedMs;
                
                System.out.println(String.format("%-15d %-10d %-12.2f %-12.2f %.2fx",
                    rowsPerGroup * repeatFactor, numGroups, fusedMs, stdMs, speedup));
            }
        }
        
        System.out.println(repeat("=", 80));
    }
    
    // =========================================================================
    // Explode (Repeat) Factor Benchmark
    // =========================================================================
    
    @Test
    public void benchmarkExplodeScaling() {
        System.out.println("\n" + repeat("=", 80));
        System.out.println("Explode (Repeat) Factor Scaling Benchmark");
        System.out.println(repeat("=", 80));
        
        int baseRows = 50000;
        int numGroups = 1000;
        int numExprs = 50;
        
        int[] repeatFactors = {1, 5, 10, 15, 20};
        
        System.out.println("Base rows: " + String.format("%,d", baseRows));
        System.out.println("Groups: " + numGroups);
        System.out.println("Expressions: " + numExprs);
        System.out.println();
        
        System.out.println(String.format("%-15s %-15s %-12s %-12s %-10s",
            "Repeat", "Total Rows", "Fused(ms)", "Std(ms)", "Speedup"));
        System.out.println(repeat("-", 65));
        
        Table baseTable = generateTestData(baseRows, numGroups, numExprs);
        
        try {
            for (int repeatFactor : repeatFactors) {
                Table inputTable;
                boolean needsClose = false;
                if (repeatFactor > 1) {
                    inputTable = baseTable.repeat(repeatFactor);
                    needsClose = true;  // Only close if we created a new table
                } else {
                    inputTable = baseTable;
                    // Don't close - baseTable will be closed in outer finally
                }
                
                try {
                    // Warmup
                    for (int i = 0; i < WARMUP; i++) {
                        runFused(inputTable, numExprs);
                        runStandard(inputTable, numExprs);
                    }
                    
                    // Benchmark
                    long fusedNs = 0, stdNs = 0;
                    for (int i = 0; i < ITERATIONS; i++) {
                        long start = System.nanoTime();
                        runFused(inputTable, numExprs);
                        fusedNs += System.nanoTime() - start;
                        
                        start = System.nanoTime();
                        runStandard(inputTable, numExprs);
                        stdNs += System.nanoTime() - start;
                    }
                    
                    double fusedMs = (fusedNs / ITERATIONS) / 1_000_000.0;
                    double stdMs = (stdNs / ITERATIONS) / 1_000_000.0;
                    double speedup = stdMs / fusedMs;
                    
                    System.out.println(String.format("%-15d %-15s %-12.2f %-12.2f %.2fx",
                        repeatFactor, String.format("%,d", inputTable.getRowCount()), 
                        fusedMs, stdMs, speedup));
                } finally {
                    if (needsClose) {
                    inputTable.close();
                    }
                }
            }
        } finally {
            baseTable.close();
        }
        
        System.out.println(repeat("=", 80));
    }
    
    // =========================================================================
    // Helper Methods
    // =========================================================================
    
    /**
     * Generate test data with specified number of columns.
     * Note: The returned Table takes ownership of all columns.
     */
    private Table generateTestData(int numRows, int numGroups, int numMetricCols) {
        int[] groupKeys = new int[numRows];
        for (int i = 0; i < numRows; i++) {
            groupKeys[i] = i % numGroups;
        }
        
        // Limit metric columns to avoid excessive memory
        int actualMetricCols = Math.min(numMetricCols, 60);
        
        ColumnVector[] cols = new ColumnVector[actualMetricCols + 1];
        boolean success = false;
        try {
            cols[0] = ColumnVector.fromInts(groupKeys);
            
            for (int c = 0; c < actualMetricCols; c++) {
                Long[] values = new Long[numRows];
                int nullMod = 10 + c;  // Vary null pattern
                for (int i = 0; i < numRows; i++) {
                    values[i] = (i % nullMod == 0) ? null : (long)((i * (17 + c)) % 10000);
                }
                cols[c + 1] = ColumnVector.fromBoxedLongs(values);
            }
            
            Table result = new Table(cols);
            success = true;
            return result;
        } finally {
            if (!success) {
                // Clean up on failure
                for (ColumnVector col : cols) {
                    if (col != null) col.close();
                }
            }
        }
    }
    
    /**
     * Run fused transform + aggregate.
     */
    private long runFused(Table inputTable, int numExprs) {
        FusedTransformAggregate.ExpressionBuilder builder = 
            new FusedTransformAggregate.ExpressionBuilder();
        
        int numCols = inputTable.getNumberOfColumns();
        int numMetrics = Math.min(numExprs, numCols - 1);
        
        for (int i = 0; i < numMetrics; i++) {
            int colIdx = 1 + (i % (numCols - 1));  // Cycle through available columns
            
            // coalesce(col, 0) * coalesce(col, 0) -> SUM
            builder.addCoalesceMulSelf(colIdx, 0L, FusedTransformAggregate.AGG_SUM);
        }
        
        // COUNT(*)
        builder.addCountAll(FusedTransformAggregate.AGG_COUNT);
        
        try (FusedTransformAggregate.FusedResult result = 
                 FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
            return result.getNumGroups();
        }
    }
    
    /**
     * Run standard cudf operations: transform columns then aggregate.
     * Carefully manages resources to avoid leaks.
     */
    private long runStandard(Table inputTable, int numExprs) {
        int numCols = inputTable.getNumberOfColumns();
        int numMetrics = Math.min(numExprs, numCols - 1);
        
        // Use ArrayList for easier resource management
        java.util.List<ColumnVector> toClose = new java.util.ArrayList<>();
        
        try {
            // Step 1: Transform columns - coalesce(col, 0) * coalesce(col, 0)
            ColumnVector[] transformedCols = new ColumnVector[numMetrics + 1];
            
            // Group key column
            transformedCols[0] = inputTable.getColumn(0);
            transformedCols[0].incRefCount();
            toClose.add(transformedCols[0]);
            
            try (Scalar zero = Scalar.fromLong(0)) {
                for (int i = 0; i < numMetrics; i++) {
                    int colIdx = 1 + (i % (numCols - 1));
                    ColumnVector col = inputTable.getColumn(colIdx);
                    ColumnVector coalesced = col.replaceNulls(zero);
                    toClose.add(coalesced);
                    ColumnVector multiplied = coalesced.mul(coalesced);
                    toClose.add(multiplied);
                    transformedCols[i + 1] = multiplied;
                }
            }
            
            // Step 2: Aggregate
            // Note: Table constructor does NOT take ownership, we still own the columns
            try (Table transformedTable = new Table(transformedCols)) {
                GroupByAggregationOnColumn[] aggs = new GroupByAggregationOnColumn[numMetrics + 1];
                for (int i = 0; i < numMetrics; i++) {
                    aggs[i] = GroupByAggregation.sum().onColumn(1 + i);
                }
                aggs[numMetrics] = GroupByAggregation.count().onColumn(1);
                
                try (Table result = transformedTable.groupBy(0).aggregate(aggs)) {
                    return result.getRowCount();
                }
            }
        } finally {
            // Close all intermediate columns in reverse order
            for (int i = toClose.size() - 1; i >= 0; i--) {
                try {
                    toClose.get(i).close();
                } catch (Exception e) {
                    // Ignore close errors during cleanup
                }
            }
        }
    }
    
    // =========================================================================
    // Legacy Simple Benchmark (for quick testing)
    // =========================================================================
    
    @Test
    public void benchmarkSimple() {
        System.out.println("\n" + repeat("=", 70));
        System.out.println("Simple Benchmark (500K rows, 1000 groups, 50 exprs, repeat=10)");
        System.out.println(repeat("=", 70));
        
        int baseRows = 50000;
        int numGroups = 1000;
        int numExprs = 50;
        int repeatFactor = 10;
        
        Table baseTable = generateTestData(baseRows, numGroups, numExprs);
        try (Table inputTable = baseTable.repeat(repeatFactor)) {
            baseTable.close();
            
            System.out.println("Total rows: " + String.format("%,d", inputTable.getRowCount()));
            System.out.println();
            
            // Warmup
            for (int i = 0; i < 3; i++) {
                runFused(inputTable, numExprs);
                runStandard(inputTable, numExprs);
            }
            
            // Benchmark
            System.out.println("--- Fused ---");
            long fusedNs = 0;
            for (int i = 0; i < 5; i++) {
                long start = System.nanoTime();
                runFused(inputTable, numExprs);
                long elapsed = System.nanoTime() - start;
                fusedNs += elapsed;
                System.out.println("  Run " + (i+1) + ": " + String.format("%.2f", elapsed/1e6) + " ms");
            }
            
            System.out.println("\n--- Standard ---");
            long stdNs = 0;
            for (int i = 0; i < 5; i++) {
                long start = System.nanoTime();
                runStandard(inputTable, numExprs);
                long elapsed = System.nanoTime() - start;
                stdNs += elapsed;
                System.out.println("  Run " + (i+1) + ": " + String.format("%.2f", elapsed/1e6) + " ms");
            }
            
            double fusedMs = (fusedNs / 5) / 1e6;
            double stdMs = (stdNs / 5) / 1e6;
            
            System.out.println("\n" + repeat("=", 70));
            System.out.println("Fused:    " + String.format("%.2f", fusedMs) + " ms");
            System.out.println("Standard: " + String.format("%.2f", stdMs) + " ms");
            System.out.println("Speedup:  " + String.format("%.2f", stdMs/fusedMs) + "x");
            System.out.println(repeat("=", 70));
        }
    }
}
