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

import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.CudfException;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JNI bindings for fused transform + aggregate operations.
 * 
 * This class provides highly optimized operations that fuse Project expressions
 * with Aggregate operations in a single GPU kernel, eliminating intermediate
 * memory allocations and reducing memory bandwidth.
 * 
 * Memory Management:
 * - Returns Tables with columns that use cudf's default memory resource
 * - Columns have proper reference counting via cudf
 * - SpillableColumnarBatch can pack on-demand when spilling is needed
 */
public class FusedTransformAggregate {
    private static final Logger LOG = LoggerFactory.getLogger(FusedTransformAggregate.class);
    
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    // =========================================================================
    // Transform Operations (must match C++ TransformOp enum)
    // =========================================================================
    
    public static final int TRANSFORM_IDENTITY = 0;
    public static final int TRANSFORM_COALESCE = 1;
    public static final int TRANSFORM_COALESCE_MUL_SELF = 2;
    public static final int TRANSFORM_COALESCE_MUL_OTHER = 3;
    public static final int TRANSFORM_CONDITIONAL = 4;
    public static final int TRANSFORM_CONDITIONAL_COALESCE = 5;

    // =========================================================================
    // Aggregation Operations (must match C++ AggOp enum)
    // =========================================================================
    
    public static final int AGG_SUM = 0;
    public static final int AGG_COUNT = 1;
    public static final int AGG_AVG = 2;
    public static final int AGG_MIN = 3;
    public static final int AGG_MAX = 4;

    // =========================================================================
    // Expression Builder
    // =========================================================================

    public static class ExpressionBuilder {
        private final List<Integer> transformOps = new ArrayList<>();
        private final List<Integer> aggOps = new ArrayList<>();
        private final List<Integer> valueColIndices = new ArrayList<>();
        private final List<Integer> condColIndices = new ArrayList<>();
        private final List<Integer> otherColIndices = new ArrayList<>();
        private final List<Long> defaultVals = new ArrayList<>();
        private final List<Long> thresholds = new ArrayList<>();
        private final List<Long> elseVals = new ArrayList<>();

        public ExpressionBuilder addIdentity(int colIdx, int aggOp) {
            transformOps.add(TRANSFORM_IDENTITY);
            aggOps.add(aggOp);
            valueColIndices.add(colIdx);
            condColIndices.add(-1);
            otherColIndices.add(-1);
            defaultVals.add(0L);
            thresholds.add(0L);
            elseVals.add(0L);
            return this;
        }

        public ExpressionBuilder addCountAll(int aggOp) {
            transformOps.add(TRANSFORM_IDENTITY);
            aggOps.add(AGG_COUNT);
            valueColIndices.add(0);
            condColIndices.add(-1);
            otherColIndices.add(-1);
            defaultVals.add(0L);
            thresholds.add(0L);
            elseVals.add(0L);
            return this;
        }

        public ExpressionBuilder addCoalesce(int colIdx, long defaultVal, int aggOp) {
            transformOps.add(TRANSFORM_COALESCE);
            aggOps.add(aggOp);
            valueColIndices.add(colIdx);
            condColIndices.add(-1);
            otherColIndices.add(-1);
            defaultVals.add(defaultVal);
            thresholds.add(0L);
            elseVals.add(0L);
            return this;
        }

        public ExpressionBuilder addCoalesceMulSelf(int colIdx, long defaultVal, int aggOp) {
            transformOps.add(TRANSFORM_COALESCE_MUL_SELF);
            aggOps.add(aggOp);
            valueColIndices.add(colIdx);
            condColIndices.add(-1);
            otherColIndices.add(-1);
            defaultVals.add(defaultVal);
            thresholds.add(0L);
            elseVals.add(0L);
            return this;
        }

        public ExpressionBuilder addCoalesceMulOther(int colIdx1, int colIdx2, long defaultVal, int aggOp) {
            transformOps.add(TRANSFORM_COALESCE_MUL_OTHER);
            aggOps.add(aggOp);
            valueColIndices.add(colIdx1);
            condColIndices.add(-1);
            otherColIndices.add(colIdx2);
            defaultVals.add(defaultVal);
            thresholds.add(0L);
            elseVals.add(0L);
            return this;
        }

        public ExpressionBuilder addConditional(int valColIdx, int condColIdx, 
                                                 long threshold, long elseVal, int aggOp) {
            transformOps.add(TRANSFORM_CONDITIONAL);
            aggOps.add(aggOp);
            valueColIndices.add(valColIdx);
            condColIndices.add(condColIdx);
            otherColIndices.add(-1);
            defaultVals.add(0L);
            thresholds.add(threshold);
            elseVals.add(elseVal);
            return this;
        }

        public ExpressionBuilder addConditionalCoalesce(int valColIdx, int condColIdx,
                                                         long defaultVal, long threshold, 
                                                         long elseVal, int aggOp) {
            transformOps.add(TRANSFORM_CONDITIONAL_COALESCE);
            aggOps.add(aggOp);
            valueColIndices.add(valColIdx);
            condColIndices.add(condColIdx);
            otherColIndices.add(-1);
            defaultVals.add(defaultVal);
            thresholds.add(threshold);
            elseVals.add(elseVal);
            return this;
        }

        public int size() {
            return transformOps.size();
        }

        int[] getTransformOps() {
            return transformOps.stream().mapToInt(Integer::intValue).toArray();
        }

        int[] getAggOps() {
            return aggOps.stream().mapToInt(Integer::intValue).toArray();
        }

        int[] getValueColIndices() {
            return valueColIndices.stream().mapToInt(Integer::intValue).toArray();
        }

        int[] getCondColIndices() {
            return condColIndices.stream().mapToInt(Integer::intValue).toArray();
        }

        int[] getOtherColIndices() {
            return otherColIndices.stream().mapToInt(Integer::intValue).toArray();
        }

        long[] getDefaultVals() {
            return defaultVals.stream().mapToLong(Long::longValue).toArray();
        }

        long[] getThresholds() {
            return thresholds.stream().mapToLong(Long::longValue).toArray();
        }

        long[] getElseVals() {
            return elseVals.stream().mapToLong(Long::longValue).toArray();
        }
    }

    // =========================================================================
    // Result Class - Simple Tables with proper ref counting
    // =========================================================================

    /**
     * Result of fused transform + aggregate operation.
     * 
     * Contains separate Tables for keys and values. Each Table owns its columns
     * with proper reference counting. The caller is responsible for closing both
     * tables when done.
     */
    public static class FusedResult implements AutoCloseable {
        private Table keys;
        private Table values;
        private boolean closed = false;

        FusedResult(Table keys, Table values) {
            this.keys = keys;
            this.values = values;
        }

        /** Get the group-by keys table. */
        public Table getKeys() {
            return keys;
        }

        /** Get the aggregated values table. */
        public Table getValues() {
            return values;
        }

        /** Get number of output groups. */
        public long getNumGroups() {
            return keys != null ? keys.getRowCount() : 0;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                if (keys != null) {
                    keys.close();
                    keys = null;
                }
                if (values != null) {
                    values.close();
                    values = null;
                }
            }
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    public static boolean shouldUseFused(int numExpressions, int numGroupByCols) {
        return canFuse(numExpressions, numGroupByCols);
    }

    /**
     * Execute fused transform + aggregate.
     * 
     * @param inputTable Input table
     * @param groupByIndices Indices of group-by columns
     * @param expressions Expression builder with all expressions
     * @param enableWarpReduction Whether to use warp-level reduction optimization
     * @return FusedResult containing keys and values Tables
     */
    public static FusedResult execute(Table inputTable, int[] groupByIndices,
                                       ExpressionBuilder expressions,
                                       boolean enableWarpReduction) {
        if (expressions.size() == 0) {
            throw new IllegalArgumentException("No expressions specified");
        }
        if (groupByIndices == null || groupByIndices.length == 0) {
            throw new IllegalArgumentException("No group-by columns specified");
        }
        if (inputTable == null) {
            throw new IllegalArgumentException("Input table cannot be null");
        }

        LOG.debug("Executing fused transform+aggregate: {} rows, {} expressions, {} groups",
            inputTable.getRowCount(), expressions.size(), groupByIndices.length);
        
        long startTime = System.nanoTime();
        
        long[] handles = executeFused(
            inputTable.getNativeView(),
            groupByIndices,
            expressions.getTransformOps(),
            expressions.getAggOps(),
            expressions.getValueColIndices(),
            expressions.getCondColIndices(),
            expressions.getOtherColIndices(),
            expressions.getDefaultVals(),
            expressions.getThresholds(),
            expressions.getElseVals(),
            enableWarpReduction
        );
        
        long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;
        LOG.debug("Fused transform+aggregate completed in {} ms", elapsedMs);

        // Parse returned handles: [numKeyCols, keyCol0, ..., numValCols, valCol0, ...]
        return parseResult(handles);
    }
    
    /**
     * Parse JNI result handles into FusedResult.
     * Format: [numKeyCols, keyCol0, ..., numValCols, valCol0, ...]
     */
    private static FusedResult parseResult(long[] handles) {
        int idx = 0;
        int numKeyCols = (int) handles[idx++];
        
        // Extract key columns
        long[] keyColHandles = new long[numKeyCols];
        System.arraycopy(handles, idx, keyColHandles, 0, numKeyCols);
        idx += numKeyCols;
        
        // Extract value columns
        int numValCols = (int) handles[idx++];
        long[] valColHandles = new long[numValCols];
        System.arraycopy(handles, idx, valColHandles, 0, numValCols);
        
        // Build Tables from native handles (each column owns its memory independently)
        Table keys = numKeyCols > 0 ? new Table(keyColHandles) : null;
        Table values = numValCols > 0 ? new Table(valColHandles) : null;
        
        return new FusedResult(keys, values);
    }

    public static FusedResult execute(Table inputTable, int[] groupByIndices,
                                       ExpressionBuilder expressions) {
        return execute(inputTable, groupByIndices, expressions, true);
    }

    // =========================================================================
    // Native Methods
    // =========================================================================

    private static native long[] executeFused(
        long inputTableHandle,
        int[] groupByIndices,
        int[] transformOps,
        int[] aggOps,
        int[] valueColIndices,
        int[] condColIndices,
        int[] otherColIndices,
        long[] defaultVals,
        long[] thresholds,
        long[] elseVals,
        boolean enableWarpReduction
    );

    private static native boolean canFuse(int numExpressions, int numGroupByCols);
}
