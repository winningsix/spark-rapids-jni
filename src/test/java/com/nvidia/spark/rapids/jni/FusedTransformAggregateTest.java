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
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for FusedTransformAggregate.
 * 
 * Tests cover:
 * - All transform operations (IDENTITY, COALESCE, COALESCE_MUL_SELF, etc.)
 * - All aggregation operations (SUM, COUNT, AVG, MIN, MAX)
 * - Null value handling
 * - Multiple group-by columns
 * - Multiple expressions
 * - Error cases
 */
public class FusedTransformAggregateTest {

    // =========================================================================
    // Basic Transform Operations
    // =========================================================================

    @Test
    public void testIdentitySum() {
        // 10 rows, 2 groups: group 0 has values 1,3,5,7,9 (sum=25), group 1 has 2,4,6,8,10 (sum=30)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertNotNull(result.getKeys());
                assertNotNull(result.getValues());
                assertEquals(2, result.getNumGroups());
                assertEquals(2, result.getKeys().getRowCount());
                assertEquals(2, result.getValues().getRowCount());
                assertEquals(1, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testCoalesceSum() {
        // Test COALESCE with null values: coalesce(col, defaultVal)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(10L, null, null, 20L);
             Table inputTable = new Table(groupKey, values)) {
            
            // coalesce(values, 5) then sum
            // group 0: 10 + 5 = 15
            // group 1: 5 + 20 = 25
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesce(1, 5L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(1, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testCoalesceMulSelf() {
        // Test coalesce(col, default) * coalesce(col, default)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(2L, null, 3L, null);
             Table inputTable = new Table(groupKey, values)) {
            
            // coalesce(val, 1) * coalesce(val, 1) then sum
            // group 0: 2*2 + 1*1 = 5
            // group 1: 3*3 + 1*1 = 10
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesceMulSelf(1, 1L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testCoalesceMulOther() {
        // Test coalesce(col1, default) * coalesce(col2, default)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector col1 = ColumnVector.fromBoxedLongs(2L, null, 3L, null);
             ColumnVector col2 = ColumnVector.fromBoxedLongs(3L, null, 4L, null);
             Table inputTable = new Table(groupKey, col1, col2)) {
            
            // coalesce(col1, 1) * coalesce(col2, 1) then sum
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesceMulOther(1, 2, 1L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testConditional() {
        // Test conditional: if(cond > threshold, value, elseVal)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromLongs(10, 20, 30, 40);
             ColumnVector conditions = ColumnVector.fromLongs(5, 15, 25, 5);
             Table inputTable = new Table(groupKey, values, conditions)) {
            
            // if(cond > 10, value, 0) then sum
            // group 0: 0 + 20 = 20 (only second row passes cond > 10)
            // group 1: 30 + 0 = 30 (only first row passes cond > 10)
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addConditional(1, 2, 10L, 0L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testConditionalCoalesce() {
        // Test conditional with coalesce: if(cond > threshold, coalesce(value, default), elseVal)
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(10L, null, null, 40L);
             ColumnVector conditions = ColumnVector.fromLongs(15, 15, 25, 5);
             Table inputTable = new Table(groupKey, values, conditions)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addConditionalCoalesce(1, 2, 99L, 10L, 0L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    // =========================================================================
    // All Aggregation Operations
    // =========================================================================

    @Test
    public void testAggCount() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1, 0);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5);
             Table inputTable = new Table(groupKey, values)) {
            
            // count(*) per group: group 0 has 3, group 1 has 2
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCountAll(FusedTransformAggregate.AGG_COUNT);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testAggMin() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1, 0);
             ColumnVector values = ColumnVector.fromLongs(5, 2, 3, 8, 1);
             Table inputTable = new Table(groupKey, values)) {
            
            // min per group: group 0 min=1, group 1 min=2
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testAggMax() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1, 0);
             ColumnVector values = ColumnVector.fromLongs(5, 2, 3, 8, 1);
             Table inputTable = new Table(groupKey, values)) {
            
            // max per group: group 0 max=5, group 1 max=8
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testAggAvg() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromLongs(10, 20, 30, 50);
             Table inputTable = new Table(groupKey, values)) {
            
            // avg per group: group 0 avg=15, group 1 avg=40
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_AVG);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    // =========================================================================
    // Multiple Group-By Columns
    // =========================================================================

    @Test
    public void testMultipleGroupByColumns() {
        try (ColumnVector key1 = ColumnVector.fromInts(0, 0, 1, 1, 0, 1);
             ColumnVector key2 = ColumnVector.fromStrings("a", "b", "a", "b", "a", "a");
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5, 6);
             Table inputTable = new Table(key1, key2, values)) {
            
            // Groups: (0,a), (0,b), (1,a), (1,b)
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(2, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0, 1}, builder)) {
                
                assertEquals(4, result.getNumGroups());
                assertEquals(2, result.getKeys().getNumberOfColumns());
            }
        }
    }

    // =========================================================================
    // Multiple Expressions
    // =========================================================================

    @Test
    public void testMultipleExpressions() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromLongs(10, 20, 30, 40);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT)
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN)
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX);
            
            assertEquals(4, builder.size());
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(4, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testMixedTransformExpressions() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector col1 = ColumnVector.fromBoxedLongs(10L, null, 30L, null);
             ColumnVector col2 = ColumnVector.fromLongs(2, 3, 4, 5);
             Table inputTable = new Table(groupKey, col1, col2)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addCoalesce(1, 0L, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(2, FusedTransformAggregate.AGG_SUM)
                    .addCoalesceMulSelf(1, 1L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(4, result.getValues().getNumberOfColumns());
            }
        }
    }

    // =========================================================================
    // Null Value Handling
    // =========================================================================

    @Test
    public void testAllNullValues() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(null, null, null, null);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesce(1, 100L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                // Each group should have sum = 200 (2 rows * 100 default)
            }
        }
    }

    @Test
    public void testMixedNullValues() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 0, 1, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedLongs(1L, null, 3L, null, 5L, null);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addCoalesce(1, 0L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(2, result.getValues().getNumberOfColumns());
            }
        }
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    @Test
    public void testSingleGroup() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 0, 0);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(1, result.getNumGroups());
            }
        }
    }

    @Test
    public void testManyGroups() {
        // 100 unique groups
        int[] keys = new int[100];
        long[] vals = new long[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = i;
            vals[i] = i * 10L;
        }
        
        try (ColumnVector groupKey = ColumnVector.fromInts(keys);
             ColumnVector values = ColumnVector.fromLongs(vals);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(100, result.getNumGroups());
            }
        }
    }
    
    @Test 
    public void testLargeInput() {
        // 10000 rows, 100 groups
        int numRows = 10000;
        int numGroups = 100;
        int[] keys = new int[numRows];
        long[] vals = new long[numRows];
        for (int i = 0; i < numRows; i++) {
            keys[i] = i % numGroups;
            vals[i] = i;
        }
        
        try (ColumnVector groupKey = ColumnVector.fromInts(keys);
             ColumnVector values = ColumnVector.fromLongs(vals);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT)
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN)
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(numGroups, result.getNumGroups());
                assertEquals(4, result.getValues().getNumberOfColumns());
            }
        }
    }

    // =========================================================================
    // Error Cases
    // =========================================================================

    @Test
    public void testEmptyExpressions() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder();
            
            assertThrows(IllegalArgumentException.class, () -> {
                FusedTransformAggregate.execute(inputTable, new int[]{0}, builder);
            });
        }
    }

    @Test
    public void testEmptyGroupBy() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            assertThrows(IllegalArgumentException.class, () -> {
                FusedTransformAggregate.execute(inputTable, new int[]{}, builder);
            });
        }
    }

    @Test
    public void testNullTable() {
        FusedTransformAggregate.ExpressionBuilder builder = 
            new FusedTransformAggregate.ExpressionBuilder()
                .addIdentity(1, FusedTransformAggregate.AGG_SUM);
        
        assertThrows(IllegalArgumentException.class, () -> {
            FusedTransformAggregate.execute(null, new int[]{0}, builder);
        });
    }

    @Test
    public void testNullGroupByIndices() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            assertThrows(IllegalArgumentException.class, () -> {
                FusedTransformAggregate.execute(inputTable, null, builder);
            });
        }
    }

    // =========================================================================
    // Warp Reduction Toggle
    // =========================================================================

    @Test
    public void testWithWarpReductionDisabled() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            // Test with warp reduction disabled
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder, false)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    // =========================================================================
    // API: shouldUseFused / canFuse
    // =========================================================================

    @Test
    public void testShouldUseFused() {
        // canFuse returns true if numExpressions >= 4 and numGroupByCols > 0
        assertTrue(FusedTransformAggregate.shouldUseFused(4, 1));
        assertTrue(FusedTransformAggregate.shouldUseFused(10, 2));
        assertFalse(FusedTransformAggregate.shouldUseFused(3, 1));
        assertFalse(FusedTransformAggregate.shouldUseFused(4, 0));
    }

    // =========================================================================
    // Standard cudf GroupBy Comparison
    // =========================================================================

    @Test 
    public void testStandardGroupByComparison() {
        // Verify our fused impl produces same group count as standard cudf
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
             Table inputTable = new Table(groupKey, values)) {
            
            // Standard cudf groupby
            try (Table stdResult = inputTable.groupBy(0).aggregate(
                    GroupByAggregation.sum().onColumn(1))) {
                
                // Fused groupby
                FusedTransformAggregate.ExpressionBuilder builder = 
                    new FusedTransformAggregate.ExpressionBuilder()
                        .addIdentity(1, FusedTransformAggregate.AGG_SUM);
                
                try (FusedTransformAggregate.FusedResult fusedResult = 
                         FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                    
                    // Both should have 2 groups
                    assertEquals(stdResult.getRowCount(), fusedResult.getNumGroups());
                }
            }
        }
    }

    // =========================================================================
    // Multiple Data Types
    // =========================================================================

    @Test
    public void testInt32Values() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromInts(10, 20, 30, 40);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testInt64Values() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromLongs(
                 Long.MAX_VALUE / 2, Long.MIN_VALUE / 2, 100L, -100L);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN)
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(3, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testFloat32Values() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromFloats(1.5f, 2.5f, 3.5f, 4.5f);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_AVG);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testFloat64Values() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromDoubles(1.111, 2.222, 3.333, 4.444);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN)
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX)
                    .addIdentity(1, FusedTransformAggregate.AGG_AVG);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(4, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testStringGroupKey() {
        // String as group-by key
        try (ColumnVector groupKey = ColumnVector.fromStrings("apple", "banana", "apple", "banana", "apple");
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4, 5);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                // apple: sum=9 (1+3+5), count=3
                // banana: sum=6 (2+4), count=2
            }
        }
    }

    @Test
    public void testInt64GroupKey() {
        // INT64 as group-by key
        try (ColumnVector groupKey = ColumnVector.fromLongs(100L, 200L, 100L, 200L);
             ColumnVector values = ColumnVector.fromLongs(1, 2, 3, 4);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testShortValues() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromShorts((short)10, (short)20, (short)30, (short)40);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
        }
    }
}

    @Test
    public void testByteValues() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.fromBytes((byte)1, (byte)2, (byte)3, (byte)4);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testMixedDataTypes() {
        // Multiple columns with different data types
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector intCol = ColumnVector.fromInts(10, 20, 30, 40);
             ColumnVector longCol = ColumnVector.fromLongs(100, 200, 300, 400);
             ColumnVector floatCol = ColumnVector.fromFloats(1.1f, 2.2f, 3.3f, 4.4f);
             ColumnVector doubleCol = ColumnVector.fromDoubles(1.11, 2.22, 3.33, 4.44);
             Table inputTable = new Table(groupKey, intCol, longCol, floatCol, doubleCol)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_SUM)   // int
                    .addIdentity(2, FusedTransformAggregate.AGG_SUM)   // long
                    .addIdentity(3, FusedTransformAggregate.AGG_SUM)   // float
                    .addIdentity(4, FusedTransformAggregate.AGG_SUM);  // double
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
                assertEquals(4, result.getValues().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testNullableFloat64() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedDoubles(1.5, null, null, 4.5);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesce(1, 0L, FusedTransformAggregate.AGG_SUM);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testNullableInt32() {
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 0, 1, 1);
             ColumnVector values = ColumnVector.fromBoxedInts(10, null, null, 40);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addCoalesce(1, 0L, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(1, FusedTransformAggregate.AGG_COUNT);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }

    @Test
    public void testMultiTypeGroupByKeys() {
        // Group by (int, string) composite key
        try (ColumnVector intKey = ColumnVector.fromInts(1, 1, 2, 2, 1, 2);
             ColumnVector strKey = ColumnVector.fromStrings("a", "b", "a", "b", "a", "a");
             ColumnVector values = ColumnVector.fromLongs(10, 20, 30, 40, 50, 60);
             Table inputTable = new Table(intKey, strKey, values)) {
            
            // Groups: (1,a), (1,b), (2,a), (2,b)
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(2, FusedTransformAggregate.AGG_SUM)
                    .addIdentity(2, FusedTransformAggregate.AGG_COUNT);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0, 1}, builder)) {
                
                assertEquals(4, result.getNumGroups());
                assertEquals(2, result.getKeys().getNumberOfColumns());
            }
        }
    }

    @Test
    public void testTimestampValues() {
        // Timestamp (INT64 microseconds) values
        long now = System.currentTimeMillis() * 1000; // microseconds
        try (ColumnVector groupKey = ColumnVector.fromInts(0, 1, 0, 1);
             ColumnVector values = ColumnVector.timestampMicroSecondsFromLongs(
                 now, now + 1000, now + 2000, now + 3000);
             Table inputTable = new Table(groupKey, values)) {
            
            FusedTransformAggregate.ExpressionBuilder builder = 
                new FusedTransformAggregate.ExpressionBuilder()
                    .addIdentity(1, FusedTransformAggregate.AGG_MIN)
                    .addIdentity(1, FusedTransformAggregate.AGG_MAX);
            
            try (FusedTransformAggregate.FusedResult result = 
                     FusedTransformAggregate.execute(inputTable, new int[]{0}, builder)) {
                
                assertEquals(2, result.getNumGroups());
            }
        }
    }
}
