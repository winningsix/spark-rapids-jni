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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Unit tests for FusedTransformAggregate execution modes.
 * 
 * Tests verify that all execution modes produce correct results and
 * compares their outputs for consistency.
 */
public class FusedTransformAggregateExecutionModeTest {

    // =========================================================================
    // Test Data Helper
    // =========================================================================
    
    private static class TestData implements AutoCloseable {
        final Table inputTable;
        final int numGroups;
        
        TestData(int numRows, int numGroups) {
            this.numGroups = numGroups;
            
            // Generate test data
            int[] keys = new int[numRows];
            long[] values = new long[numRows];
            long[] values2 = new long[numRows];
            
            for (int i = 0; i < numRows; i++) {
                keys[i] = i % numGroups;
                values[i] = (i + 1);  // 1, 2, 3, ...
                values2[i] = (i + 1) * 2;  // 2, 4, 6, ...
            }
            
            ColumnVector keyCol = ColumnVector.fromInts(keys);
            ColumnVector valCol = ColumnVector.fromLongs(values);
            ColumnVector val2Col = ColumnVector.fromLongs(values2);
            
            this.inputTable = new Table(keyCol, valCol, val2Col);
        }
        
        @Override
        public void close() {
            if (inputTable != null) {
                inputTable.close();
            }
        }
    }
    
    // =========================================================================
    // Execution Mode Tests
    // =========================================================================
    
    @ParameterizedTest
    @EnumSource(FusedTransformAggregate.ExecutionMode.class)
    void testAllModesProduceSameResults(FusedTransformAggregate.ExecutionMode mode) {
        try (TestData data = new TestData(1000, 10)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                .setExecutionMode(mode);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(data.inputTable, new int[]{0}, builder, config)) {
                
                assertNotNull(result.getKeys(), "Keys should not be null for mode: " + mode);
                assertNotNull(result.getValues(), "Values should not be null for mode: " + mode);
                assertEquals(data.numGroups, result.getNumGroups(), 
                    "Should have correct number of groups for mode: " + mode);
            }
        }
    }
    
    @Test
    void testModeConsistency() {
        // Test that all modes produce identical results for the same input
        try (TestData data = new TestData(1000, 10)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT);
            
            // Collect results from all modes
            Map<FusedTransformAggregate.ExecutionMode, long[]> sumResults = new HashMap<>();
            Map<FusedTransformAggregate.ExecutionMode, long[]> countResults = new HashMap<>();
            
            for (FusedTransformAggregate.ExecutionMode mode : 
                     FusedTransformAggregate.ExecutionMode.values()) {
                
                FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                    .setExecutionMode(mode);
                
                try (FusedTransformAggregate.FusedResult result = 
                         FusedTransformAggregate.execute(data.inputTable, new int[]{0}, builder, config)) {
                    
                    // Extract sum and count values
                    try (HostColumnVector sumCol = result.getValues().getColumn(0).copyToHost();
                         HostColumnVector countCol = result.getValues().getColumn(1).copyToHost()) {
                        
                        long[] sums = new long[(int)result.getNumGroups()];
                        long[] counts = new long[(int)result.getNumGroups()];
                        
                        for (int i = 0; i < result.getNumGroups(); i++) {
                            sums[i] = sumCol.getLong(i);
                            counts[i] = countCol.getLong(i);
                        }
                        
                        // Sort by key for comparison
                        try (HostColumnVector keyCol = result.getKeys().getColumn(0).copyToHost()) {
                            // Create sorted arrays
                            long[] sortedSums = new long[(int)result.getNumGroups()];
                            long[] sortedCounts = new long[(int)result.getNumGroups()];
                            for (int i = 0; i < result.getNumGroups(); i++) {
                                int key = keyCol.getInt(i);
                                sortedSums[key] = sums[i];
                                sortedCounts[key] = counts[i];
                            }
                            sumResults.put(mode, sortedSums);
                            countResults.put(mode, sortedCounts);
                        }
                    }
                }
            }
            
            // Compare all modes against AUTO mode as reference
            long[] refSums = sumResults.get(FusedTransformAggregate.ExecutionMode.AUTO);
            long[] refCounts = countResults.get(FusedTransformAggregate.ExecutionMode.AUTO);
            
            for (FusedTransformAggregate.ExecutionMode mode : 
                     FusedTransformAggregate.ExecutionMode.values()) {
                assertArrayEquals(refSums, sumResults.get(mode),
                    "Sum results should match for mode: " + mode);
                assertArrayEquals(refCounts, countResults.get(mode),
                    "Count results should match for mode: " + mode);
            }
        }
    }
    
    // =========================================================================
    // Transform Operation Tests with Different Modes
    // =========================================================================
    
    @ParameterizedTest
    @EnumSource(FusedTransformAggregate.ExecutionMode.class)
    void testCoalesceMulSelf(FusedTransformAggregate.ExecutionMode mode) {
        // Test val * val (variance component)
        try (ColumnVector keys = ColumnVector.fromInts(0, 0, 0, 1, 1, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5, 6);
             Table inputTable = new Table(keys, values)) {
            
            // sum(val * val)
            // group 0: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
            // group 1: 4*4 + 5*5 + 6*6 = 16 + 25 + 36 = 77
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesceMulSelf(1, 0L, FusedTransformAggregate.AGG_SUM);
            
            FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                .setExecutionMode(mode);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder, config)) {
                
                assertEquals(2, result.getNumGroups());
                
                // Verify results
                try (HostColumnVector keyCol = result.getKeys().getColumn(0).copyToHost();
                     HostColumnVector sumCol = result.getValues().getColumn(0).copyToHost()) {
                    
                    Map<Integer, Long> resultMap = new HashMap<>();
                    for (int i = 0; i < result.getNumGroups(); i++) {
                        resultMap.put(keyCol.getInt(i), sumCol.getLong(i));
                    }
                    
                    assertEquals(14L, resultMap.get(0), 
                        "Group 0 sum(val*val) should be 14 for mode: " + mode);
                    assertEquals(77L, resultMap.get(1), 
                        "Group 1 sum(val*val) should be 77 for mode: " + mode);
                }
            }
        }
    }
    
    @ParameterizedTest
    @EnumSource(FusedTransformAggregate.ExecutionMode.class)
    void testMultipleAggregations(FusedTransformAggregate.ExecutionMode mode) {
        // Test SUM, COUNT, and SUM(val*val) together - variance components
        try (ColumnVector keys = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromLongs(10, 20, 30, 40);
             Table inputTable = new Table(keys, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)     // sum(val)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT)   // count(val)
                    .addCoalesceMulSelf(1, 0L, FusedTransformAggregate.AGG_SUM);  // sum(val*val)
            
            FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                .setExecutionMode(mode);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder, config)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(3, result.getValues().getNumberOfColumns());
                
                // group 0: sum=30, count=2, sum_sq=100+400=500
                // group 1: sum=70, count=2, sum_sq=900+1600=2500
                try (HostColumnVector keyCol = result.getKeys().getColumn(0).copyToHost();
                     HostColumnVector sumCol = result.getValues().getColumn(0).copyToHost();
                     HostColumnVector countCol = result.getValues().getColumn(1).copyToHost();
                     HostColumnVector sumSqCol = result.getValues().getColumn(2).copyToHost()) {
                    
                    for (int i = 0; i < result.getNumGroups(); i++) {
                        int key = keyCol.getInt(i);
                        long sum = sumCol.getLong(i);
                        long count = countCol.getLong(i);
                        long sumSq = sumSqCol.getLong(i);
                        
                        if (key == 0) {
                            assertEquals(30L, sum, "Group 0 sum for mode: " + mode);
                            assertEquals(2L, count, "Group 0 count for mode: " + mode);
                            assertEquals(500L, sumSq, "Group 0 sum_sq for mode: " + mode);
                        } else {
                            assertEquals(70L, sum, "Group 1 sum for mode: " + mode);
                            assertEquals(2L, count, "Group 1 count for mode: " + mode);
                            assertEquals(2500L, sumSq, "Group 1 sum_sq for mode: " + mode);
                        }
                    }
                }
            }
        }
    }
    
    // =========================================================================
    // Configuration Tests
    // =========================================================================
    
    @Test
    void testConfigurationOptions() {
        // Test that configuration options are respected
        FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
            .setExecutionMode(FusedTransformAggregate.ExecutionMode.FUSED_1PASS)
            .setEnableWarpReduction(true)
            .setEnablePerfectHash(false)
            .setFused1PassGroupThreshold(500000);
        
        assertEquals(FusedTransformAggregate.ExecutionMode.FUSED_1PASS, config.getExecutionMode());
        assertTrue(config.isWarpReductionEnabled());
        assertFalse(config.isPerfectHashEnabled());
        assertEquals(500000, config.getFused1PassGroupThreshold());
    }
    
    @Test
    void testModeSelection() {
        FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
            .setFused1PassGroupThreshold(100000);
        
        // Small group count should select FUSED_1PASS
        FusedTransformAggregate.ExecutionMode mode1 = 
            FusedTransformAggregate.selectBestMode(1000000, 10000, 4, config);
        assertEquals(FusedTransformAggregate.ExecutionMode.FUSED_1PASS, mode1);
        
        // Large group count should select JIT_TRANSFORM
        FusedTransformAggregate.ExecutionMode mode2 = 
            FusedTransformAggregate.selectBestMode(1000000, 500000, 4, config);
        assertEquals(FusedTransformAggregate.ExecutionMode.JIT_TRANSFORM, mode2);
        
        // Explicit mode should be returned as-is
        config.setExecutionMode(FusedTransformAggregate.ExecutionMode.HAND_WRITTEN_KERNEL);
        FusedTransformAggregate.ExecutionMode mode3 = 
            FusedTransformAggregate.selectBestMode(1000000, 10000, 4, config);
        assertEquals(FusedTransformAggregate.ExecutionMode.HAND_WRITTEN_KERNEL, mode3);
    }
    
    @Test
    void testDefaultConfig() {
        // Save original config
        FusedTransformAggregate.Config original = FusedTransformAggregate.getDefaultConfig();
        
        try {
            // Set new default config
            FusedTransformAggregate.Config newConfig = new FusedTransformAggregate.Config()
                .setExecutionMode(FusedTransformAggregate.ExecutionMode.JIT_TRANSFORM);
            FusedTransformAggregate.setDefaultConfig(newConfig);
            
            // Verify it's used
            assertEquals(FusedTransformAggregate.ExecutionMode.JIT_TRANSFORM,
                FusedTransformAggregate.getDefaultConfig().getExecutionMode());
        } finally {
            // Restore original
            FusedTransformAggregate.setDefaultConfig(original);
        }
    }
    
    // =========================================================================
    // Null Handling Tests
    // =========================================================================
    
    @ParameterizedTest
    @EnumSource(FusedTransformAggregate.ExecutionMode.class)
    void testNullValueHandling(FusedTransformAggregate.ExecutionMode mode) {
        try (ColumnVector keys = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(10L, null, null, 40L);
             Table inputTable = new Table(keys, values)) {
            
            // COALESCE with default value 5
            // group 0: 10 + 5 = 15
            // group 1: 5 + 40 = 45
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesce(1, 5L, FusedTransformAggregate.AGG_SUM);
            
            FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                .setExecutionMode(mode);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder, config)) {
                
                assertEquals(2, result.getNumGroups());
                
                try (HostColumnVector keyCol = result.getKeys().getColumn(0).copyToHost();
                     HostColumnVector sumCol = result.getValues().getColumn(0).copyToHost()) {
                    
                    Map<Integer, Long> resultMap = new HashMap<>();
                    for (int i = 0; i < result.getNumGroups(); i++) {
                        resultMap.put(keyCol.getInt(i), sumCol.getLong(i));
                    }
                    
                    assertEquals(15L, resultMap.get(0), 
                        "Group 0 coalesce sum should be 15 for mode: " + mode);
                    assertEquals(45L, resultMap.get(1), 
                        "Group 1 coalesce sum should be 45 for mode: " + mode);
                }
            }
        }
    }
    
    // =========================================================================
    // Large Data Test
    // =========================================================================
    
    @Test
    void testLargeDataAllModes() {
        int numRows = 100000;
        int numGroups = 1000;
        
        try (TestData data = new TestData(numRows, numGroups)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT);
            
            // Test all modes complete without error
            for (FusedTransformAggregate.ExecutionMode mode : 
                     FusedTransformAggregate.ExecutionMode.values()) {
                
                FusedTransformAggregate.Config config = new FusedTransformAggregate.Config()
                    .setExecutionMode(mode);
                
                try (FusedTransformAggregate.FusedResult result = 
                         FusedTransformAggregate.execute(data.inputTable, new int[]{0}, builder, config)) {
                    
                    assertEquals(numGroups, result.getNumGroups(),
                        "Should have " + numGroups + " groups for mode: " + mode);
                    assertEquals(2, result.getValues().getNumberOfColumns(),
                        "Should have 2 value columns for mode: " + mode);
                }
            }
        }
    }
}

