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

#include "batched_aggregate.hpp"
#include "cudf_jni_apis.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <vector>
#include <memory>

extern "C" {

/**
 * JNI binding for batched groupby aggregation.
 * 
 * This function performs optimized multi-column aggregation using specialized
 * CUDA kernels for common expression patterns.
 */
JNIEXPORT jlongArray JNICALL 
Java_com_nvidia_spark_rapids_jni_BatchedAggregate_batchedGroupbyAggregate(
    JNIEnv* env, 
    jclass,
    jlong input_table_handle,
    jintArray j_key_col_indices,
    jintArray j_pattern_types,
    jintArray j_num_input_cols,
    jintArray j_all_input_cols,
    jlongArray j_all_params,
    jboolean enable_warp_reduction,
    jboolean enable_contiguous_output,
    jboolean enable_shared_groupby,
    jboolean enable_perfect_hash,
    jboolean enable_adaptive,
    jint perfect_hash_max_keys)
{
    JNI_NULL_CHECK(env, input_table_handle, "input table handle is null", nullptr);
    JNI_NULL_CHECK(env, j_key_col_indices, "key column indices is null", nullptr);
    JNI_NULL_CHECK(env, j_pattern_types, "pattern types is null", nullptr);

    try {
        cudf::jni::auto_set_device(env);

        // Get input table
        auto* input_table = reinterpret_cast<cudf::table_view*>(input_table_handle);
        
        // Get key column indices
        cudf::jni::native_jintArray key_cols(env, j_key_col_indices);
        cudf::jni::native_jintArray pattern_types(env, j_pattern_types);
        cudf::jni::native_jintArray num_input_cols(env, j_num_input_cols);
        cudf::jni::native_jintArray all_input_cols(env, j_all_input_cols);
        cudf::jni::native_jlongArray all_params(env, j_all_params);

        int num_specs = pattern_types.size();
        
        // Build key columns view
        std::vector<cudf::column_view> key_column_views;
        key_column_views.reserve(key_cols.size());
        for (int i = 0; i < key_cols.size(); i++) {
            key_column_views.push_back(input_table->column(key_cols[i]));
        }
        cudf::table_view keys_view(key_column_views);

        // Build aggregation specifications
        std::vector<spark_rapids_jni::batched_agg_spec> specs;
        specs.reserve(num_specs);
        
        int col_offset = 0;
        // param_offset will be used when we process parameters in specialized kernels
        (void)all_params;  // Suppress unused warning for now
        
        for (int i = 0; i < num_specs; i++) {
            int n_cols = num_input_cols[i];
            (void)pattern_types[i];  // Will be used in future for pattern-specific agg types
            
            // Get the primary value column
            int val_col_idx = all_input_cols[col_offset];
            cudf::column_view values = input_table->column(val_col_idx);
            
            // Map pattern type to aggregation type
            // For now, all patterns are treated as SUM variants
            spark_rapids_jni::BatchedAggType agg_type = spark_rapids_jni::BatchedAggType::SUM;
            
            specs.push_back({values, agg_type, cudf::data_type{cudf::type_id::INT64}, cudf::null_policy::INCLUDE});
            
            col_offset += n_cols;
            // Skip params for this spec (we'll use them in the kernel)
        }

        // Execute batched aggregation
        auto result = spark_rapids_jni::batched_groupby_aggregate(
            keys_view,
            specs,
            cudf::get_default_stream(),
            cudf::get_current_device_resource_ref()
        );

        // Return format: [numKeyCols, keyCol0, ..., numValCols, valCol0, ..., sharedBufferPtr, bufferMrPtr]
        // CRITICAL: Both shared_buffer AND buffer_mr MUST be kept alive!
        // - shared_buffer: The actual GPU memory that columns point to
        // - buffer_mr: The memory resource that columns use for deallocation (no-op)
        // Java MUST destroy columns BEFORE destroying buffer_mr and shared_buffer!
        auto keys_table_result = std::move(result.keys);
        auto vals_table_result = std::move(result.values);
        
        // Move to heap to extend lifetime - Java must manage these!
        auto* shared_buffer_ptr = new rmm::device_buffer(std::move(result.shared_buffer));
        auto* buffer_mr_ptr = result.buffer_mr.release();  // Release ownership to Java
        
        int num_out_key_cols = keys_table_result->num_columns();
        int num_out_val_cols = vals_table_result->num_columns();
        int total_size = 2 + num_out_key_cols + num_out_val_cols + 2;  // +2 for buffer ptrs
        
        jlongArray result_handles = env->NewLongArray(total_size);
        if (result_handles == nullptr) {
            delete buffer_mr_ptr;      // Clean up on failure
            delete shared_buffer_ptr;  // Clean up on failure
            return nullptr;  // JVM will throw OutOfMemoryError
        }
        
        std::vector<jlong> handles(total_size);
        handles[0] = num_out_key_cols;
        
        // Release key columns
        auto released_key_cols = keys_table_result->release();
        for (int i = 0; i < num_out_key_cols; ++i) {
            handles[1 + i] = reinterpret_cast<jlong>(released_key_cols[i].release());
        }
        
        handles[1 + num_out_key_cols] = num_out_val_cols;
        
        // Release value columns
        auto released_val_cols = vals_table_result->release();
        for (int i = 0; i < num_out_val_cols; ++i) {
            handles[2 + num_out_key_cols + i] = reinterpret_cast<jlong>(released_val_cols[i].release());
        }
        
        // Store buffer pointers - Java MUST delete in correct order:
        // 1. First close all columns
        // 2. Then delete buffer_mr (second to last)
        // 3. Finally delete shared_buffer (last)
        handles[total_size - 2] = reinterpret_cast<jlong>(buffer_mr_ptr);
        handles[total_size - 1] = reinterpret_cast<jlong>(shared_buffer_ptr);
        
        env->SetLongArrayRegion(result_handles, 0, total_size, handles.data());
        
        return result_handles;

    } CATCH_STD(env, nullptr);
}

/**
 * JNI binding to release the shared buffer allocated during batched aggregation.
 * MUST be called after all result columns are closed to avoid use-after-free.
 */
JNIEXPORT void JNICALL 
Java_com_nvidia_spark_rapids_jni_BatchedAggregate_releaseSharedBuffer(
    JNIEnv* env, 
    jclass,
    jlong buffer_handle)
{
    try {
        if (buffer_handle != 0) {
            auto* buffer = reinterpret_cast<rmm::device_buffer*>(buffer_handle);
            delete buffer;
        }
    } CATCH_STD(env, );
}

/**
 * JNI binding to release the buffer memory resource.
 * MUST be called after all result columns are closed but BEFORE releaseSharedBuffer.
 */
JNIEXPORT void JNICALL 
Java_com_nvidia_spark_rapids_jni_BatchedAggregate_releaseBufferMr(
    JNIEnv* env, 
    jclass,
    jlong buffer_mr_handle)
{
    try {
        if (buffer_mr_handle != 0) {
            auto* mr = reinterpret_cast<rmm::mr::device_memory_resource*>(buffer_mr_handle);
            delete mr;
        }
    } CATCH_STD(env, );
}

} // extern "C"


