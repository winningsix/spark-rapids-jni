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
 * A class to perform hybrid scan operations on Parquet files.
 * 
 * Hybrid Scan is optimized for highly selective filters by reading the file in two passes:
 * 1. First pass: Read filter columns and build a row mask
 * 2. Second pass: Read payload columns using the row mask to skip unnecessary data
 * 
 * This is particularly effective when:
 * - The filter has high selectivity (filters out most rows)
 * - There's a clear separation between filter columns and payload columns
 * - Payload columns are large (strings, arrays, etc.)
 */
public class ParquetHybridScan implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private final String[] filterColumns;
  private final String[] payloadColumns;

  /**
   * Create a ParquetHybridScan instance from footer bytes without AST filter.
   * This constructor is for backward compatibility and when filters cannot be 
   * converted to AST format.
   * 
   * @param footerBuffer The host memory buffer containing the Parquet footer
   * @param footerOffset The offset in the buffer where the footer starts
   * @param footerLength The length of the footer
   * @param filterColumns The names of columns used in filter predicates
   * @param payloadColumns The names of columns not used in filters (payload)
   * @param timestampTypeId The type ID for timestamp columns (e.g., TIMESTAMP_MICROSECONDS)
   */
  public ParquetHybridScan(HostMemoryBuffer footerBuffer, 
                           long footerOffset,
                           long footerLength,
                           String[] filterColumns, 
                           String[] payloadColumns,
                           int timestampTypeId) {
    this(footerBuffer, footerOffset, footerLength, filterColumns, payloadColumns, 
         timestampTypeId, 0);
  }

  /**
   * Create a ParquetHybridScan instance from footer bytes with an AST filter expression.
   * The AST filter is required for the hybrid scan to perform filter pushdown optimizations
   * like row group pruning, page index filtering, and row mask building.
   * 
   * @param footerBuffer The host memory buffer containing the Parquet footer
   * @param footerOffset The offset in the buffer where the footer starts
   * @param footerLength The length of the footer
   * @param filterColumns The names of columns used in filter predicates
   * @param payloadColumns The names of columns not used in filters (payload)
   * @param timestampTypeId The type ID for timestamp columns (e.g., TIMESTAMP_MICROSECONDS)
   * @param astFilterHandle The native handle to a compiled AST filter expression (from 
   *                        ai.rapids.cudf.ast.CompiledExpression), or 0 for no filter
   */
  public ParquetHybridScan(HostMemoryBuffer footerBuffer, 
                           long footerOffset,
                           long footerLength,
                           String[] filterColumns, 
                           String[] payloadColumns,
                           int timestampTypeId,
                           long astFilterHandle) {
    this.filterColumns = filterColumns;
    this.payloadColumns = payloadColumns;
    this.nativeHandle = createNativeWithFilter(footerBuffer.getAddress() + footerOffset, 
                                               footerLength,
                                               filterColumns, 
                                               payloadColumns,
                                               timestampTypeId,
                                               astFilterHandle);
  }

  /**
   * Get all row group indices in the file.
   * @return Array of row group indices (0-based)
   */
  public int[] getAllRowGroups() {
    return getAllRowGroups(nativeHandle);
  }

  /**
   * Filter row groups using column statistics (min/max values).
   * This can eliminate row groups that cannot contain matching rows.
   * 
   * @param rowGroupIndices The indices of row groups to filter
   * @return Filtered array of row group indices
   */
  public int[] filterRowGroupsWithStats(int[] rowGroupIndices) {
    return filterRowGroupsWithStats(nativeHandle, rowGroupIndices);
  }

  /**
   * Get the byte range for reading page index data.
   * @return Array of two longs: [offset, length]
   */
  public long[] getPageIndexByteRange() {
    return getPageIndexByteRange(nativeHandle);
  }

  /**
   * Setup page index data for more precise row-level filtering.
   * 
   * @param pageIndexBuffer The buffer containing page index data
   * @param offset The offset in the buffer
   * @param length The length of page index data
   */
  public void setupPageIndex(HostMemoryBuffer pageIndexBuffer, long offset, long length) {
    setupPageIndex(nativeHandle, pageIndexBuffer.getAddress() + offset, length);
  }

  /**
   * Get the total number of rows in the specified row groups.
   * 
   * @param rowGroupIndices The row group indices
   * @return Total number of rows
   */
  public long getTotalRowsInRowGroups(int[] rowGroupIndices) {
    return getTotalRowsInRowGroups(nativeHandle, rowGroupIndices);
  }

  /**
   * Get the number of rows in a specific row group.
   * 
   * @param rowGroupIndex The row group index
   * @return Number of rows in the row group
   */
  public long getRowsInRowGroup(int rowGroupIndex) {
    return getRowsInRowGroup(nativeHandle, rowGroupIndex);
  }

  /**
   * Build a row mask using page index statistics.
   * The mask indicates which rows might match the filter predicates.
   * 
   * @param rowGroupIndices The row group indices to consider
   * @return A ColumnVector of boolean values (true = might match, false = definitely doesn't match)
   */
  public ColumnVector buildRowMaskWithPageIndex(int[] rowGroupIndices) {
    return new ColumnVector(buildRowMaskWithPageIndex(nativeHandle, rowGroupIndices));
  }

  /**
   * Get byte ranges for reading filter column chunks.
   * Returns pairs of (offset, length) for each column chunk.
   * 
   * @param rowGroupIndices The row groups to read
   * @return Array of longs: [offset1, length1, offset2, length2, ...]
   */
  public long[] getFilterColumnChunkRanges(int[] rowGroupIndices) {
    return getFilterColumnChunkRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Get byte ranges for reading payload column chunks.
   * Returns pairs of (offset, length) for each column chunk.
   * 
   * @param rowGroupIndices The row groups to read
   * @return Array of longs: [offset1, length1, offset2, length2, ...]
   */
  public long[] getPayloadColumnChunkRanges(int[] rowGroupIndices) {
    return getPayloadColumnChunkRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Materialize filter columns from device buffers into a Table.
   * 
   * @param rowGroupIndices The row groups being read
   * @param bufferAddresses Array of device buffer addresses
   * @param bufferSizes Array of buffer sizes
   * @param rowMask Native handle of the row mask ColumnVector
   * @param usePageIndex Whether page index was used
   * @return A Table containing the filter columns
   */
  public Table materializeFilterColumns(int[] rowGroupIndices,
                                        long[] bufferAddresses,
                                        long[] bufferSizes,
                                        long rowMask,
                                        boolean usePageIndex) {
    long[] columnPointers = materializeFilterColumns(nativeHandle, rowGroupIndices,
                                                     bufferAddresses, bufferSizes,
                                                     rowMask, usePageIndex);
    return columnPointers != null ? new Table(columnPointers) : null;
  }

  /**
   * Materialize payload columns from device buffers into a Table.
   * 
   * @param rowGroupIndices The row groups being read
   * @param bufferAddresses Array of device buffer addresses
   * @param bufferSizes Array of buffer sizes
   * @param rowMask Native handle of the row mask ColumnVector
   * @param usePageIndex Whether page index was used
   * @return A Table containing the payload columns
   */
  public Table materializePayloadColumns(int[] rowGroupIndices,
                                         long[] bufferAddresses,
                                         long[] bufferSizes,
                                         long rowMask,
                                         boolean usePageIndex) {
    long[] columnPointers = materializePayloadColumns(nativeHandle, rowGroupIndices,
                                                      bufferAddresses, bufferSizes,
                                                      rowMask, usePageIndex);
    return columnPointers != null ? new Table(columnPointers) : null;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      closeNative(nativeHandle);
      nativeHandle = 0;
    }
  }

  // Native methods
  private static native long createNativeWithFilter(long footerAddress, long footerLength,
                                          String[] filterColumns, String[] payloadColumns,
                                          int timestampTypeId, long astFilterHandle);
  private static native void closeNative(long handle);
  private static native int[] getAllRowGroups(long handle);
  private static native int[] filterRowGroupsWithStats(long handle, int[] rowGroupIndices);
  private static native long[] getPageIndexByteRange(long handle);
  private static native void setupPageIndex(long handle, long bufferAddress, long length);
  private static native long getTotalRowsInRowGroups(long handle, int[] rowGroupIndices);
  private static native long getRowsInRowGroup(long handle, int rowGroupIndex);
  private static native long buildRowMaskWithPageIndex(long handle, int[] rowGroupIndices);
  private static native long[] getFilterColumnChunkRanges(long handle, int[] rowGroupIndices);
  private static native long[] getPayloadColumnChunkRanges(long handle, int[] rowGroupIndices);
  private static native long[] materializeFilterColumns(long handle, int[] rowGroupIndices,
                                                        long[] bufferAddresses, long[] bufferSizes,
                                                        long rowMask, boolean usePageIndex);
  private static native long[] materializePayloadColumns(long handle, int[] rowGroupIndices,
                                                         long[] bufferAddresses, long[] bufferSizes,
                                                         long rowMask, boolean usePageIndex);
}

