/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * Dictionary-aware optimizations for Parquet processing.
 * 
 * This class provides optimized operations on dictionary-encoded columns:
 * 1. Dictionary Filter: Evaluate filter on dictionary keys (small), then apply to indices (large)
 * 2. Dictionary Aggregation: Aggregate using dictionary indices as bucket keys
 * 
 * Configuration:
 * - spark.rapids.sql.parquet.dictionaryOptimization.enabled (default: true)
 * - spark.rapids.sql.parquet.dictionaryOptimization.minRowsRatio (default: 100)
 *   Only use dictionary optimization if rows/keys ratio >= this value
 * 
 * Example speedup for TPC-H lineitem l_shipdate (2555 keys, 600M rows):
 * - Traditional filter: 600M comparisons
 * - Dictionary filter: 2555 comparisons + 600M gathers
 * - Speedup: 235,000x fewer comparisons
 */
public class DictionaryOptimization {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Configuration
  private static volatile boolean enabled = true;
  private static volatile int minRowsRatio = 100;

  /**
   * Enable or disable dictionary optimization globally.
   * @param enable true to enable, false to disable
   */
  public static void setEnabled(boolean enable) {
    enabled = enable;
  }

  /**
   * Check if dictionary optimization is enabled.
   * @return true if enabled
   */
  public static boolean isEnabled() {
    return enabled;
  }

  /**
   * Set minimum rows/keys ratio for dictionary optimization.
   * Only use optimization if numRows/numKeys >= this value.
   * @param ratio minimum ratio (default: 100)
   */
  public static void setMinRowsRatio(int ratio) {
    minRowsRatio = ratio;
  }

  /**
   * Get minimum rows/keys ratio.
   * @return minimum ratio
   */
  public static int getMinRowsRatio() {
    return minRowsRatio;
  }

  /**
   * Check if dictionary optimization should be used for given column stats.
   * @param numKeys number of dictionary keys
   * @param numRows number of rows
   * @return true if dictionary optimization is recommended
   */
  public static boolean shouldUseDictionaryOptimization(long numKeys, long numRows) {
    if (!enabled) return false;
    if (numKeys <= 0 || numRows <= 0) return false;
    return (numRows / numKeys) >= minRowsRatio;
  }

  /**
   * Filter a dictionary column using comparison operator.
   * Evaluates filter on keys (small), then applies to indices (large).
   * 
   * @param dictionaryColumn The dictionary-encoded column (DICTIONARY32 type)
   * @param op Comparison operator (LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, EQUAL, NOT_EQUAL)
   * @param compareValue Scalar value to compare against
   * @return Boolean column with filter result for each row
   */
  public static ColumnVector filterDictionaryColumn(ColumnView dictionaryColumn,
                                                     BinaryOp op,
                                                     Scalar compareValue) {
    if (!enabled) {
      throw new IllegalStateException("Dictionary optimization is disabled");
    }
    // Use ordinal() which matches the nativeId values in BinaryOp
    return new ColumnVector(filterDictionaryColumnNative(
        dictionaryColumn.getNativeView(),
        op.ordinal(),
        compareValue.getScalarHandle()));
  }

  /**
   * Count rows grouped by dictionary key.
   * Uses dictionary indices as direct bucket keys (no hash table).
   * 
   * @param dictionaryColumn The dictionary-encoded column to group by
   * @return Table with two columns: [keys, counts]
   */
  public static Table countByDictionary(ColumnView dictionaryColumn) {
    if (!enabled) {
      throw new IllegalStateException("Dictionary optimization is disabled");
    }
    long[] columnPointers = countByDictionaryNative(dictionaryColumn.getNativeView());
    return new Table(columnPointers);
  }

  /**
   * Sum values grouped by dictionary key.
   * Uses dictionary indices as direct bucket keys (no hash table).
   * 
   * @param dictionaryColumn The dictionary-encoded column to group by
   * @param valuesColumn The numeric column to sum
   * @return Table with two columns: [keys, sums]
   */
  public static Table sumByDictionary(ColumnView dictionaryColumn, ColumnView valuesColumn) {
    if (!enabled) {
      throw new IllegalStateException("Dictionary optimization is disabled");
    }
    long[] columnPointers = sumByDictionaryNative(
        dictionaryColumn.getNativeView(),
        valuesColumn.getNativeView());
    return new Table(columnPointers);
  }

  /**
   * Average values grouped by dictionary key.
   * Uses dictionary indices as direct bucket keys (no hash table).
   * 
   * @param dictionaryColumn The dictionary-encoded column to group by
   * @param valuesColumn The numeric column to average
   * @return Table with two columns: [keys, averages]
   */
  public static Table avgByDictionary(ColumnView dictionaryColumn, ColumnView valuesColumn) {
    if (!enabled) {
      throw new IllegalStateException("Dictionary optimization is disabled");
    }
    long[] columnPointers = avgByDictionaryNative(
        dictionaryColumn.getNativeView(),
        valuesColumn.getNativeView());
    return new Table(columnPointers);
  }

  /**
   * Get dictionary statistics for a column.
   * @param dictionaryColumn The dictionary-encoded column
   * @return Array of [numKeys, numRows]
   */
  public static long[] getDictionaryStats(ColumnView dictionaryColumn) {
    return getDictionaryStatsNative(dictionaryColumn.getNativeView());
  }

  // Native methods
  private static native long filterDictionaryColumnNative(long columnHandle, int opId, long scalarHandle);
  private static native long[] countByDictionaryNative(long columnHandle);
  private static native long[] sumByDictionaryNative(long columnHandle, long valuesHandle);
  private static native long[] avgByDictionaryNative(long columnHandle, long valuesHandle);
  private static native long[] getDictionaryStatsNative(long columnHandle);
}
