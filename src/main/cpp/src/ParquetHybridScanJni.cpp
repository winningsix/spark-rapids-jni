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

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "jni_compiled_expr.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/ast/expressions.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace rapids {
namespace jni {

/**
 * Wrapper class that holds the cuDF hybrid_scan_reader and associated state.
 */
class ParquetHybridScanState {
public:
  std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> reader;
  cudf::io::parquet_reader_options options;
  std::vector<cudf::size_type> current_row_groups;
  std::unique_ptr<cudf::column> row_mask;
  bool page_index_setup = false;
  // Keep a reference to the AST filter to ensure it stays alive
  cudf::jni::ast::compiled_expr const* ast_filter = nullptr;
  
  ParquetHybridScanState(cudf::host_span<uint8_t const> footer_bytes,
                         cudf::io::parquet_reader_options const& opts,
                         cudf::jni::ast::compiled_expr const* filter = nullptr)
    : options(opts), ast_filter(filter) {
    reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      footer_bytes, options);
  }
  
  std::vector<cudf::size_type> getAllRowGroups() {
    return reader->all_row_groups(options);
  }
  
  std::vector<cudf::size_type> filterRowGroupsWithStats(
      std::vector<cudf::size_type> const& rg_indices,
      rmm::cuda_stream_view stream) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    return reader->filter_row_groups_with_stats(span, options, stream);
  }
  
  cudf::io::text::byte_range_info getPageIndexByteRange() {
    return reader->page_index_byte_range();
  }
  
  void setupPageIndex(cudf::host_span<uint8_t const> page_index_bytes) {
    reader->setup_page_index(page_index_bytes);
    page_index_setup = true;
  }
  
  cudf::size_type getTotalRowsInRowGroups(std::vector<cudf::size_type> const& rg_indices) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    return reader->total_rows_in_row_groups(span);
  }
  
  std::unique_ptr<cudf::column> buildRowMaskWithPageIndex(
      std::vector<cudf::size_type> const& rg_indices,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    return reader->build_row_mask_with_page_index_stats(span, options, stream, mr);
  }
  
  std::vector<cudf::io::text::byte_range_info> getFilterColumnChunkRanges(
      std::vector<cudf::size_type> const& rg_indices) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    return reader->filter_column_chunks_byte_ranges(span, options);
  }
  
  std::vector<cudf::io::text::byte_range_info> getPayloadColumnChunkRanges(
      std::vector<cudf::size_type> const& rg_indices) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    return reader->payload_column_chunks_byte_ranges(span, options);
  }
  
  cudf::io::table_with_metadata materializeFilterColumns(
      std::vector<cudf::size_type> const& rg_indices,
      std::vector<rmm::device_buffer>&& column_chunk_buffers,
      cudf::mutable_column_view& row_mask_view,
      bool use_page_mask,
      rmm::cuda_stream_view stream) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    auto mask_mode = use_page_mask 
      ? cudf::io::parquet::experimental::use_data_page_mask::YES
      : cudf::io::parquet::experimental::use_data_page_mask::NO;
    return reader->materialize_filter_columns(
      span, std::move(column_chunk_buffers), row_mask_view, mask_mode, options, stream);
  }
  
  cudf::io::table_with_metadata materializePayloadColumns(
      std::vector<cudf::size_type> const& rg_indices,
      std::vector<rmm::device_buffer>&& column_chunk_buffers,
      cudf::column_view const& row_mask_view,
      bool use_page_mask,
      rmm::cuda_stream_view stream) {
    cudf::host_span<cudf::size_type const> span(rg_indices);
    auto mask_mode = use_page_mask 
      ? cudf::io::parquet::experimental::use_data_page_mask::YES
      : cudf::io::parquet::experimental::use_data_page_mask::NO;
    return reader->materialize_payload_columns(
      span, std::move(column_chunk_buffers), row_mask_view, mask_mode, options, stream);
  }
};

}  // namespace jni
}  // namespace rapids

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_createNativeWithFilter(JNIEnv* env,
                                                                 jclass,
                                                                 jlong footer_address,
                                                                 jlong footer_length,
                                                                 jobjectArray filter_columns,
                                                                 jobjectArray payload_columns,
                                                                 jint timestamp_type_id,
                                                                 jlong ast_filter_handle)
{
  CUDF_FUNC_RANGE();
  JNI_TRY
  {
    cudf::jni::native_jstringArray n_filter_cols(env, filter_columns);
    cudf::jni::native_jstringArray n_payload_cols(env, payload_columns);
    
    // Build the list of all columns to read
    auto filter_col_names = n_filter_cols.as_cpp_vector();
    auto payload_col_names = n_payload_cols.as_cpp_vector();
    
    std::vector<std::string> all_columns;
    all_columns.insert(all_columns.end(), filter_col_names.begin(), filter_col_names.end());
    all_columns.insert(all_columns.end(), payload_col_names.begin(), payload_col_names.end());
    
    // Create parquet reader options with empty source (footer-only mode)
    auto options_builder = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info());
    
    if (!all_columns.empty()) {
      options_builder.columns(all_columns);
    }
    
    // Set explicit filter column names for hybrid scan
    // This is needed because Spark-Rapids generates AST with column_reference (indices)
    // instead of column_name_reference (names), and cuDF needs column names to select filter columns
    if (!filter_col_names.empty()) {
      options_builder.filter_columns(filter_col_names);
    }
    
    // Set timestamp type if specified
    if (timestamp_type_id >= 0) {
      options_builder.timestamp_type(cudf::data_type{static_cast<cudf::type_id>(timestamp_type_id)});
    }
    
    // Set AST filter if provided
    cudf::jni::ast::compiled_expr const* ast_filter = nullptr;
    if (ast_filter_handle != 0) {
      ast_filter = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(ast_filter_handle);
      options_builder.filter(ast_filter->get_top_expression());
    }
    
    auto options = options_builder.build();
    
    // Create footer span
    cudf::host_span<uint8_t const> footer_span(
      reinterpret_cast<uint8_t const*>(footer_address),
      static_cast<size_t>(footer_length));
    
    auto state = std::make_unique<rapids::jni::ParquetHybridScanState>(footer_span, options, ast_filter);
    
    return cudf::jni::release_as_jlong(state);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_closeNative(JNIEnv* env,
                                                               jclass,
                                                               jlong handle)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    delete state;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jintArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getAllRowGroups(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    auto indices = state->getAllRowGroups();
    
    cudf::jni::native_jintArray result(env, indices.size());
    std::copy(indices.begin(), indices.end(), result.begin());
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_filterRowGroupsWithStats(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle,
                                                                            jintArray row_group_indices)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    
    std::vector<cudf::size_type> indices(n_indices.begin(), n_indices.end());
    auto filtered = state->filterRowGroupsWithStats(indices, rmm::cuda_stream_default);
    
    cudf::jni::native_jintArray result(env, filtered.size());
    std::copy(filtered.begin(), filtered.end(), result.begin());
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getPageIndexByteRange(JNIEnv* env,
                                                                         jclass,
                                                                         jlong handle)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    auto range = state->getPageIndexByteRange();
    
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = range.offset();
    result[1] = range.size();
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_setupPageIndex(JNIEnv* env,
                                                                  jclass,
                                                                  jlong handle,
                                                                  jlong buffer_address,
                                                                  jlong length)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    
    cudf::host_span<uint8_t const> page_index_span(
      reinterpret_cast<uint8_t const*>(buffer_address),
      static_cast<size_t>(length));
    
    state->setupPageIndex(page_index_span);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getTotalRowsInRowGroups(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jintArray row_group_indices)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    
    std::vector<cudf::size_type> indices(n_indices.begin(), n_indices.end());
    return state->getTotalRowsInRowGroups(indices);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getRowsInRowGroup(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jint row_group_index)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    std::vector<cudf::size_type> indices = {row_group_index};
    return state->getTotalRowsInRowGroups(indices);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_buildRowMaskWithPageIndex(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle,
                                                                              jintArray row_group_indices)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    
    std::vector<cudf::size_type> indices(n_indices.begin(), n_indices.end());
    
    auto col = state->buildRowMaskWithPageIndex(
      indices, 
      rmm::cuda_stream_default,
      rmm::mr::get_current_device_resource());
    
    return cudf::jni::release_as_jlong(col);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getFilterColumnChunkRanges(JNIEnv* env,
                                                                               jclass,
                                                                               jlong handle,
                                                                               jintArray row_group_indices)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    
    std::vector<cudf::size_type> rg_indices(n_indices.begin(), n_indices.end());
    auto ranges = state->getFilterColumnChunkRanges(rg_indices);
    
    // Convert to offset/length pairs
    std::vector<jlong> result_vec;
    for (const auto& range : ranges) {
      result_vec.push_back(range.offset());
      result_vec.push_back(range.size());
    }
    
    cudf::jni::native_jlongArray result(env, result_vec.size());
    std::copy(result_vec.begin(), result_vec.end(), result.begin());
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_getPayloadColumnChunkRanges(JNIEnv* env,
                                                                                jclass,
                                                                                jlong handle,
                                                                                jintArray row_group_indices)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    
    std::vector<cudf::size_type> rg_indices(n_indices.begin(), n_indices.end());
    auto ranges = state->getPayloadColumnChunkRanges(rg_indices);
    
    // Convert to offset/length pairs
    std::vector<jlong> result_vec;
    for (const auto& range : ranges) {
      result_vec.push_back(range.offset());
      result_vec.push_back(range.size());
    }
    
    cudf::jni::native_jlongArray result(env, result_vec.size());
    std::copy(result_vec.begin(), result_vec.end(), result.begin());
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_materializeFilterColumns(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jintArray row_group_indices,
                                                                             jlongArray buffer_addresses,
                                                                             jlongArray buffer_sizes,
                                                                             jlong row_mask_handle,
                                                                             jboolean use_page_index)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    cudf::jni::native_jlongArray n_addresses(env, buffer_addresses);
    cudf::jni::native_jlongArray n_sizes(env, buffer_sizes);
    
    std::vector<cudf::size_type> rg_indices(n_indices.begin(), n_indices.end());
    
    // Create device buffers by copying from HOST memory
    // This approach lets cuDF/RMM own the device memory, avoiding ownership conflicts
    // with Java's DeviceMemoryBuffer
    std::vector<rmm::device_buffer> buffers;
    auto stream = rmm::cuda_stream_default;
    auto mr = rmm::mr::get_current_device_resource();
    
    for (int i = 0; i < n_addresses.size(); ++i) {
      auto host_ptr = reinterpret_cast<void const*>(n_addresses[i]);
      auto size = static_cast<size_t>(n_sizes[i]);
      
      // Allocate device buffer (owned by RMM)
      rmm::device_buffer dev_buf(size, stream, mr);
      
      // Copy from host to device
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        dev_buf.data(),
        host_ptr,
        size,
        cudaMemcpyHostToDevice,
        stream.value()));
      
      buffers.push_back(std::move(dev_buf));
    }
    
    // Synchronize to ensure copies are complete before proceeding
    stream.synchronize();
    
    // Get row mask column view
    // NOTE: row_mask_handle is a column_view pointer from ColumnVector.getNativeView(),
    // not a column pointer.
    auto const* row_mask_view_ptr = reinterpret_cast<cudf::column_view const*>(row_mask_handle);
    
    // Create a copy of the row mask as a new column so we can get a mutable view.
    // This is necessary because cuDF's materialize_filter_columns updates the row mask in place.
    // We store this in the state so materializePayloadColumns can use the updated mask.
    state->row_mask = std::make_unique<cudf::column>(*row_mask_view_ptr, stream);
    auto row_mask_mutable_view = state->row_mask->mutable_view();
    
    auto result = state->materializeFilterColumns(
      rg_indices,
      std::move(buffers),
      row_mask_mutable_view,
      use_page_index,
      stream);
    
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetHybridScan_materializePayloadColumns(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle,
                                                                              jintArray row_group_indices,
                                                                              jlongArray buffer_addresses,
                                                                              jlongArray buffer_sizes,
                                                                              jlong row_mask_handle,
                                                                              jboolean use_page_index)
{
  JNI_TRY
  {
    auto* state = reinterpret_cast<rapids::jni::ParquetHybridScanState*>(handle);
    cudf::jni::native_jintArray n_indices(env, row_group_indices);
    cudf::jni::native_jlongArray n_addresses(env, buffer_addresses);
    cudf::jni::native_jlongArray n_sizes(env, buffer_sizes);
    
    std::vector<cudf::size_type> rg_indices(n_indices.begin(), n_indices.end());
    
    // Create device buffers by copying from HOST memory
    // This approach lets cuDF/RMM own the device memory, avoiding ownership conflicts
    std::vector<rmm::device_buffer> buffers;
    auto stream = rmm::cuda_stream_default;
    auto mr = rmm::mr::get_current_device_resource();
    
    for (int i = 0; i < n_addresses.size(); ++i) {
      auto host_ptr = reinterpret_cast<void const*>(n_addresses[i]);
      auto size = static_cast<size_t>(n_sizes[i]);
      
      // Allocate device buffer (owned by RMM)
      rmm::device_buffer dev_buf(size, stream, mr);
      
      // Copy from host to device
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        dev_buf.data(),
        host_ptr,
        size,
        cudaMemcpyHostToDevice,
        stream.value()));
      
      buffers.push_back(std::move(dev_buf));
    }
    
    // Synchronize to ensure copies are complete
    stream.synchronize();
    
    // Use the updated row mask from materializeFilterColumns if available,
    // otherwise use the provided row_mask_handle (for fallback mode without filter columns)
    cudf::column_view row_mask_view;
    if (state->row_mask) {
      // Use the updated row mask stored in state (from materializeFilterColumns)
      row_mask_view = state->row_mask->view();
    } else {
      // Fallback: use the provided row mask handle directly
      auto const* row_mask_view_ptr = reinterpret_cast<cudf::column_view const*>(row_mask_handle);
      row_mask_view = *row_mask_view_ptr;
    }
    
    auto result = state->materializePayloadColumns(
      rg_indices,
      std::move(buffers),
      row_mask_view,
      use_page_index,
      stream);
    
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
