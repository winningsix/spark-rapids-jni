/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * JNI bindings for batched aggregation operations.
 * 
 * This class provides optimized multi-column aggregation that processes
 * multiple aggregation columns in a single kernel launch, reducing overhead
 * and improving GPU utilization.
 * 
 * Supported patterns:
 * - SIMPLE_SUM: SUM(col)
 * - COALESCE_SUM: SUM(COALESCE(col, 0))
 * - VARIANCE_SUM: SUM(COALESCE(x,0) * COALESCE(x,0))
 * - COVARIANCE_SUM: SUM(COALESCE(x,0) * COALESCE(y,0))
 * - CONDITIONAL_SUM: SUM(IF(cond > 0, col, 0))
 * - CONDITIONAL_VARIANCE_SUM: SUM(COALESCE(IF(cond,x,0),0)^2)
 */
public class BatchedAggregate {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    /**
     * Aggregation pattern types.
     * Must match the C++ enum in batched_aggregate_kernels.cuh
     */
    public static final int PATTERN_SIMPLE_SUM = 0;
    public static final int PATTERN_COALESCE_SUM = 1;
    public static final int PATTERN_VARIANCE_SUM = 2;
    public static final int PATTERN_COVARIANCE_SUM = 3;
    public static final int PATTERN_CONDITIONAL_SUM = 4;
    public static final int PATTERN_CONDITIONAL_VARIANCE_SUM = 5;
    public static final int PATTERN_COUNT = 6;

    /**
     * Configuration for batched aggregation optimizations.
     */
    public static class BatchedAggConfig {
        public final boolean enableWarpReduction;
        public final boolean enableContiguousOutput;
        public final boolean enableSharedGroupby;
        public final boolean enablePerfectHash;
        public final boolean enableAdaptive;
        public final int perfectHashMaxKeys;

        public BatchedAggConfig(
                boolean enableWarpReduction,
                boolean enableContiguousOutput,
                boolean enableSharedGroupby,
                boolean enablePerfectHash,
                boolean enableAdaptive,
                int perfectHashMaxKeys) {
            this.enableWarpReduction = enableWarpReduction;
            this.enableContiguousOutput = enableContiguousOutput;
            this.enableSharedGroupby = enableSharedGroupby;
            this.enablePerfectHash = enablePerfectHash;
            this.enableAdaptive = enableAdaptive;
            this.perfectHashMaxKeys = perfectHashMaxKeys;
        }

        /**
         * Default configuration with all optimizations enabled.
         */
        public static BatchedAggConfig defaultConfig() {
            return new BatchedAggConfig(
                true,   // warpReduction
                true,   // contiguousOutput
                true,   // sharedGroupby
                true,   // perfectHash
                true,   // adaptive
                1000000 // perfectHashMaxKeys
            );
        }
    }

    /**
     * Aggregation type enum for simplified Spark Rapids integration.
     */
    public enum AggregationType {
        SUM(0),
        COUNT(1),
        MIN(2),
        MAX(3),
        AVG(4),
        COUNT_DISTINCT(5);

        private final int value;

        AggregationType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * Simple aggregation specification for Spark Rapids integration.
     * Maps a column index to an aggregation type.
     */
    public static class AggregationSpec {
        public final int columnIndex;
        public final AggregationType aggType;

        public AggregationSpec(int columnIndex, AggregationType aggType) {
            this.columnIndex = columnIndex;
            this.aggType = aggType;
        }

        /**
         * Convert to internal AggSpec format.
         */
        public AggSpec toAggSpec() {
            int patternType;
            switch (aggType) {
                case SUM:
                    patternType = PATTERN_SIMPLE_SUM;
                    break;
                case COUNT:
                    patternType = PATTERN_COUNT;
                    break;
                default:
                    patternType = PATTERN_SIMPLE_SUM;
            }
            return new AggSpec(patternType, new int[]{columnIndex}, new long[]{});
        }
    }

    /**
     * Specification for a single aggregation operation.
     */
    public static class AggSpec {
        public final int patternType;
        public final int[] inputColumnIndices;
        public final long[] parameters;

        /**
         * Create an aggregation specification.
         * 
         * @param patternType The pattern type (PATTERN_SIMPLE_SUM, etc.)
         * @param inputColumnIndices Column indices to use as input
         * @param parameters Additional parameters (e.g., default value, threshold)
         */
        public AggSpec(int patternType, int[] inputColumnIndices, long[] parameters) {
            this.patternType = patternType;
            this.inputColumnIndices = inputColumnIndices;
            this.parameters = parameters;
        }

        /**
         * Create a simple SUM specification.
         */
        public static AggSpec simpleSum(int colIndex) {
            return new AggSpec(PATTERN_SIMPLE_SUM, new int[]{colIndex}, new long[]{});
        }

        /**
         * Create a COALESCE+SUM specification.
         */
        public static AggSpec coalesceSum(int colIndex, long defaultValue) {
            return new AggSpec(PATTERN_COALESCE_SUM, new int[]{colIndex}, new long[]{defaultValue});
        }

        /**
         * Create a variance SUM specification: SUM(COALESCE(x,def)^2)
         */
        public static AggSpec varianceSum(int colIndex, long defaultValue) {
            return new AggSpec(PATTERN_VARIANCE_SUM, new int[]{colIndex}, new long[]{defaultValue});
        }

        /**
         * Create a covariance SUM specification: SUM(COALESCE(x,def) * COALESCE(y,def))
         */
        public static AggSpec covarianceSum(int colIndexX, int colIndexY, long defaultValue) {
            return new AggSpec(PATTERN_COVARIANCE_SUM, 
                new int[]{colIndexX, colIndexY}, new long[]{defaultValue});
        }

        /**
         * Create a conditional SUM specification: SUM(IF(cond > threshold, val, 0))
         */
        public static AggSpec conditionalSum(int condColIndex, int valueColIndex, long threshold) {
            return new AggSpec(PATTERN_CONDITIONAL_SUM,
                new int[]{condColIndex, valueColIndex}, new long[]{threshold});
        }

        /**
         * Create a conditional variance SUM specification.
         */
        public static AggSpec conditionalVarianceSum(
                int condColIndex, int valueColIndex, long threshold, long defaultValue) {
            return new AggSpec(PATTERN_CONDITIONAL_VARIANCE_SUM,
                new int[]{condColIndex, valueColIndex}, new long[]{threshold, defaultValue});
        }
    }

    /**
     * Result of batched aggregation.
     */
    public static class BatchedAggResult implements AutoCloseable {
        private Table keys;
        private Table values;
        private long bufferMrHandle;       // Native handle to memory resource (must outlive columns)
        private long sharedBufferHandle;   // Native handle to shared buffer

        public BatchedAggResult(Table keys, Table values, long bufferMrHandle, long sharedBufferHandle) {
            this.keys = keys;
            this.values = values;
            this.bufferMrHandle = bufferMrHandle;
            this.sharedBufferHandle = sharedBufferHandle;
        }

        public Table getKeys() { return keys; }
        public Table getValues() { return values; }

        /**
         * Get the combined result as a single table (keys + values).
         */
        public Table getCombinedTable() {
            int numKeyCols = (int) keys.getNumberOfColumns();
            int numValCols = (int) values.getNumberOfColumns();
            ColumnVector[] allCols = new ColumnVector[numKeyCols + numValCols];
            
            for (int i = 0; i < numKeyCols; i++) {
                allCols[i] = keys.getColumn(i);
            }
            for (int i = 0; i < numValCols; i++) {
                allCols[numKeyCols + i] = values.getColumn(i);
            }
            
            return new Table(allCols);
        }

        @Override
        public void close() {
            // IMPORTANT: Destruction order matters!
            // 1. Close tables FIRST (columns call buffer_mr->deallocate())
            // 2. Then release buffer_mr (safe because columns are gone)
            // 3. Finally release shared_buffer (safe because nothing references it)
            if (keys != null) {
                keys.close();
                keys = null;
            }
            if (values != null) {
                values.close();
                values = null;
            }
            // Release buffer_mr (after columns are closed)
            if (bufferMrHandle != 0) {
                releaseBufferMr(bufferMrHandle);
                bufferMrHandle = 0;
            }
            // Release shared buffer (after buffer_mr is released)
            if (sharedBufferHandle != 0) {
                releaseSharedBuffer(sharedBufferHandle);
                sharedBufferHandle = 0;
            }
        }
    }

    /**
     * Execute batched aggregation on input data.
     * 
     * @param inputTable The input table containing all columns
     * @param keyColumnIndices Indices of grouping key columns
     * @param aggSpecs Array of aggregation specifications
     * @param config Configuration for optimizations
     * @return BatchedAggResult containing aggregated keys and values
     */
    public static BatchedAggResult execute(
            Table inputTable,
            int[] keyColumnIndices,
            AggSpec[] aggSpecs,
            BatchedAggConfig config) {
        
        // Validate inputs
        if (inputTable == null) {
            throw new IllegalArgumentException("Input table cannot be null");
        }
        if (keyColumnIndices == null || keyColumnIndices.length == 0) {
            throw new IllegalArgumentException("Key column indices cannot be null or empty");
        }
        if (aggSpecs == null || aggSpecs.length == 0) {
            throw new IllegalArgumentException("Aggregation specs cannot be null or empty");
        }

        // Flatten aggSpecs to arrays for JNI call
        int numSpecs = aggSpecs.length;
        int[] patternTypes = new int[numSpecs];
        int[] numInputCols = new int[numSpecs];
        int totalInputCols = 0;
        int totalParams = 0;
        
        for (int i = 0; i < numSpecs; i++) {
            patternTypes[i] = aggSpecs[i].patternType;
            numInputCols[i] = aggSpecs[i].inputColumnIndices.length;
            totalInputCols += aggSpecs[i].inputColumnIndices.length;
            totalParams += aggSpecs[i].parameters.length;
        }

        int[] allInputCols = new int[totalInputCols];
        long[] allParams = new long[totalParams];
        int colOffset = 0;
        int paramOffset = 0;
        
        for (int i = 0; i < numSpecs; i++) {
            System.arraycopy(aggSpecs[i].inputColumnIndices, 0, 
                allInputCols, colOffset, aggSpecs[i].inputColumnIndices.length);
            colOffset += aggSpecs[i].inputColumnIndices.length;
            
            System.arraycopy(aggSpecs[i].parameters, 0,
                allParams, paramOffset, aggSpecs[i].parameters.length);
            paramOffset += aggSpecs[i].parameters.length;
        }

        // Call native method
        // Returns: [numKeyCols, keyCol0, keyCol1, ..., numValCols, valCol0, valCol1, ...]
        long[] resultHandles = batchedGroupbyAggregate(
            inputTable.getNativeView(),
            keyColumnIndices,
            patternTypes,
            numInputCols,
            allInputCols,
            allParams,
            config.enableWarpReduction,
            config.enableContiguousOutput,
            config.enableSharedGroupby,
            config.enablePerfectHash,
            config.enableAdaptive,
            config.perfectHashMaxKeys
        );

        // Parse result: [numKeyCols, keyCol0, ..., numValCols, valCol0, ..., bufferMrPtr, sharedBufferPtr]
        int numKeyCols = (int) resultHandles[0];
        long[] keyColHandles = new long[numKeyCols];
        System.arraycopy(resultHandles, 1, keyColHandles, 0, numKeyCols);
        
        int numValCols = (int) resultHandles[1 + numKeyCols];
        long[] valColHandles = new long[numValCols];
        System.arraycopy(resultHandles, 2 + numKeyCols, valColHandles, 0, numValCols);
        
        // Get buffer handles (last two elements)
        long bufferMrHandle = resultHandles[resultHandles.length - 2];
        long sharedBufferHandle = resultHandles[resultHandles.length - 1];
        
        return new BatchedAggResult(
            new Table(keyColHandles),
            new Table(valColHandles),
            bufferMrHandle,
            sharedBufferHandle
        );
    }

    /**
     * Simplified execute method using default configuration.
     */
    public static BatchedAggResult execute(
            Table inputTable,
            int[] keyColumnIndices,
            AggSpec[] aggSpecs) {
        return execute(inputTable, keyColumnIndices, aggSpecs, BatchedAggConfig.defaultConfig());
    }

    /**
     * Execute batched groupby aggregation with configuration.
     * This is the main entry point for Spark Rapids integration.
     * 
     * @param inputTable The input table containing all columns
     * @param keyColumnIndices Indices of grouping key columns
     * @param aggSpecs Array of aggregation specifications
     * @param config Configuration for optimizations
     * @return BatchedAggResult containing aggregated keys and values
     */
    public static BatchedAggResult groupByAggregate(
            Table inputTable,
            int[] keyColumnIndices,
            AggSpec[] aggSpecs,
            BatchedAggConfig config) {
        return execute(inputTable, keyColumnIndices, aggSpecs, config);
    }

    /**
     * Execute batched groupby aggregation with default configuration.
     * 
     * @param inputTable The input table containing all columns
     * @param keyColumnIndices Indices of grouping key columns
     * @param aggSpecs Array of aggregation specifications
     * @return BatchedAggResult containing aggregated keys and values
     */
    public static BatchedAggResult groupByAggregate(
            Table inputTable,
            int[] keyColumnIndices,
            AggSpec[] aggSpecs) {
        return execute(inputTable, keyColumnIndices, aggSpecs, BatchedAggConfig.defaultConfig());
    }

    /**
     * Execute batched groupby aggregation using AggregationSpec (simplified API).
     * 
     * @param inputTable The input table containing all columns
     * @param keyColumnIndices Indices of grouping key columns
     * @param aggSpecs Array of simplified aggregation specifications
     * @param config Configuration for optimizations
     * @return BatchedAggResult containing aggregated keys and values
     */
    public static BatchedAggResult groupByAggregate(
            Table inputTable,
            int[] keyColumnIndices,
            AggregationSpec[] aggSpecs,
            BatchedAggConfig config) {
        // Convert AggregationSpec[] to AggSpec[]
        AggSpec[] internalSpecs = new AggSpec[aggSpecs.length];
        for (int i = 0; i < aggSpecs.length; i++) {
            internalSpecs[i] = aggSpecs[i].toAggSpec();
        }
        return execute(inputTable, keyColumnIndices, internalSpecs, config);
    }

    /**
     * Execute batched groupby aggregation using AggregationSpec with default config.
     * 
     * @param inputTable The input table containing all columns
     * @param keyColumnIndices Indices of grouping key columns
     * @param aggSpecs Array of simplified aggregation specifications
     * @return BatchedAggResult containing aggregated keys and values
     */
    public static BatchedAggResult groupByAggregate(
            Table inputTable,
            int[] keyColumnIndices,
            AggregationSpec[] aggSpecs) {
        return groupByAggregate(inputTable, keyColumnIndices, aggSpecs, BatchedAggConfig.defaultConfig());
    }

    // Native method declarations
    
    /**
     * Native batched groupby aggregation.
     * 
     * @param inputTableHandle Native handle to input table
     * @param keyColumnIndices Indices of key columns
     * @param patternTypes Pattern type for each aggregation
     * @param numInputCols Number of input columns for each aggregation
     * @param allInputCols Flattened array of all input column indices
     * @param allParams Flattened array of all parameters
     * @param enableWarpReduction Enable warp-level reduction optimization
     * @param enableContiguousOutput Enable contiguous output buffer
     * @param enableSharedGroupby Enable shared groupby results
     * @param enablePerfectHash Enable perfect hash optimization
     * @param enableAdaptive Enable adaptive strategy selection
     * @param perfectHashMaxKeys Max keys for perfect hash
     * @return Array: [numKeyCols, keyCols..., numValCols, valCols..., sharedBufferHandle]
     */
    private static native long[] batchedGroupbyAggregate(
        long inputTableHandle,
        int[] keyColumnIndices,
        int[] patternTypes,
        int[] numInputCols,
        int[] allInputCols,
        long[] allParams,
        boolean enableWarpReduction,
        boolean enableContiguousOutput,
        boolean enableSharedGroupby,
        boolean enablePerfectHash,
        boolean enableAdaptive,
        int perfectHashMaxKeys
    );

    /**
     * Release the buffer memory resource.
     * MUST be called after all result columns are closed but BEFORE releaseSharedBuffer.
     * 
     * @param handle Native handle to the buffer memory resource
     */
    private static native void releaseBufferMr(long handle);

    /**
     * Release the shared buffer allocated during batched aggregation.
     * MUST be called after releaseBufferMr.
     * 
     * @param handle Native handle to the shared buffer
     */
    private static native void releaseSharedBuffer(long handle);
}



