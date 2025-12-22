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

/**
 * Batched Transform Operations for Pre-Project Optimization
 *
 * This class provides batched processing of common transform operations that
 * appear in Spark's pre-project stage. By processing multiple columns in a
 * single kernel launch (using 2D GPU grid), we significantly reduce kernel
 * launch overhead.
 *
 * <h2>Performance Characteristics</h2>
 * <ul>
 *   <li>Traditional approach: N columns × M operations = N×M kernel launches</li>
 *   <li>Batched approach: N columns × M operations ≈ M kernel launches</li>
 * </ul>
 *
 * <h2>Supported Operations</h2>
 * <ul>
 *   <li>COALESCE(column, scalar) - Replace nulls with scalar default</li>
 *   <li>COALESCE(column1, column2) - Replace nulls with another column's value</li>
 *   <li>IF_ELSE(condition, true_val, false_val) - Ternary conditional</li>
 *   <li>COALESCE(a, 0) * COALESCE(b, 0) - Fused coalesce-multiply pattern</li>
 * </ul>
 *
 * <h2>Example: Multi-Column Aggregation Pattern</h2>
 * <pre>
 * // Common pre-project pattern:
 * // (gpucoalesce(pre_col, 0) * gpucoalesce(post_col, 0))
 *
 * // Traditional approach: 3 kernel launches per column pair
 * // Batched approach: 1 kernel launch for ALL column pairs
 *
 * ColumnVector[] results = BatchedTransform.batchCoalesceMultiply(
 *     preCols, new double[]{0.0, 0.0, ...},
 *     postCols, new double[]{0.0, 0.0, ...}
 * );
 * </pre>
 */
public class BatchedTransform {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Batch COALESCE with scalar defaults for INT64 columns.
   *
   * Replaces NULL values in multiple columns with scalar defaults:
   * output[i] = input[i] IS NULL ? defaults[i] : input[i]
   *
   * @param columns Input columns to process
   * @param defaults Scalar default value for each column
   * @return Array of output columns (no nulls)
   */
  public static ColumnVector[] batchCoalesceScalar(ColumnVector[] columns, long[] defaults) {
    if (columns == null || columns.length == 0) {
      return new ColumnVector[0];
    }
    if (columns.length != defaults.length) {
      throw new IllegalArgumentException("Number of defaults must match number of columns");
    }

    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      handles[i] = columns[i].getNativeView();
    }

    long[] resultHandles = batchCoalesceScalarInt64(handles, defaults);
    ColumnVector[] results = new ColumnVector[resultHandles.length];
    for (int i = 0; i < resultHandles.length; i++) {
      results[i] = new ColumnVector(resultHandles[i]);
    }
    return results;
  }

  /**
   * Batch COALESCE with scalar defaults for FLOAT64 columns.
   *
   * @param columns Input columns to process
   * @param defaults Scalar default value for each column
   * @return Array of output columns (no nulls)
   */
  public static ColumnVector[] batchCoalesceScalar(ColumnVector[] columns, double[] defaults) {
    if (columns == null || columns.length == 0) {
      return new ColumnVector[0];
    }
    if (columns.length != defaults.length) {
      throw new IllegalArgumentException("Number of defaults must match number of columns");
    }

    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      handles[i] = columns[i].getNativeView();
    }

    long[] resultHandles = batchCoalesceScalarDouble(handles, defaults);
    ColumnVector[] results = new ColumnVector[resultHandles.length];
    for (int i = 0; i < resultHandles.length; i++) {
      results[i] = new ColumnVector(resultHandles[i]);
    }
    return results;
  }

  /**
   * Fused COALESCE + MULTIPLY batch operation for FLOAT64.
   *
   * Computes: output = COALESCE(a, default_a) * COALESCE(b, default_b)
   * for multiple column pairs in a single kernel launch.
   *
   * This is the most common pattern in multi-aggregation workloads:
   * (gpucoalesce(pre_col, 0) * gpucoalesce(post_col, 0))
   *
   * <h3>Performance</h3>
   * <ul>
   *   <li>Without fusion: 3 kernels per column-pair × N pairs = 3N kernels</li>
   *   <li>With fusion: 1 kernel for all N pairs</li>
   * </ul>
   *
   * @param aColumns First operand columns
   * @param aDefaults Scalar defaults for first operands
   * @param bColumns Second operand columns
   * @param bDefaults Scalar defaults for second operands
   * @return Array of output columns (a * b with nulls handled)
   */
  public static ColumnVector[] batchCoalesceMultiply(
      ColumnVector[] aColumns, double[] aDefaults,
      ColumnVector[] bColumns, double[] bDefaults) {
    if (aColumns == null || aColumns.length == 0) {
      return new ColumnVector[0];
    }
    if (aColumns.length != bColumns.length) {
      throw new IllegalArgumentException("Number of a and b columns must match");
    }
    if (aColumns.length != aDefaults.length || bColumns.length != bDefaults.length) {
      throw new IllegalArgumentException("Number of defaults must match number of columns");
    }

    long[] aHandles = new long[aColumns.length];
    long[] bHandles = new long[bColumns.length];
    for (int i = 0; i < aColumns.length; i++) {
      aHandles[i] = aColumns[i].getNativeView();
      bHandles[i] = bColumns[i].getNativeView();
    }

    long[] resultHandles = batchCoalesceMultiplyDouble(aHandles, aDefaults, bHandles, bDefaults);
    ColumnVector[] results = new ColumnVector[resultHandles.length];
    for (int i = 0; i < resultHandles.length; i++) {
      results[i] = new ColumnVector(resultHandles[i]);
    }
    return results;
  }

  // Native method declarations
  private static native long[] batchCoalesceScalarInt64(long[] columnHandles, long[] defaults);
  private static native long[] batchCoalesceScalarDouble(long[] columnHandles, double[] defaults);
  private static native long[] batchCoalesceMultiplyDouble(
      long[] aHandles, double[] aDefaults, long[] bHandles, double[] bDefaults);
}
