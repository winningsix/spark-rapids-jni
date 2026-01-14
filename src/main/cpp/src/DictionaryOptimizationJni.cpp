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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/filter.hpp>
#include <cudf/dictionary/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

extern "C" {

/**
 * Filter a dictionary column using comparison operator.
 */
JNIEXPORT jlong JNICALL 
Java_com_nvidia_spark_rapids_jni_DictionaryOptimization_filterDictionaryColumnNative(
    JNIEnv* env, 
    jclass, 
    jlong columnHandle, 
    jint opId, 
    jlong scalarHandle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const& column = *reinterpret_cast<cudf::column_view const*>(columnHandle);
    auto const& scalar = *reinterpret_cast<cudf::scalar const*>(scalarHandle);
    
    // Create dictionary column view
    cudf::dictionary_column_view dict_view(column);
    
    // Convert JNI op ID to cudf AST operator
    auto op = static_cast<cudf::ast::ast_operator>(opId);
    
    // Call cudf dictionary filter
    auto result = cudf::dictionary::filter_dictionary_column(dict_view, op, scalar);
    
    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

/**
 * Count rows grouped by dictionary key.
 */
JNIEXPORT jlongArray JNICALL 
Java_com_nvidia_spark_rapids_jni_DictionaryOptimization_countByDictionaryNative(
    JNIEnv* env, 
    jclass, 
    jlong columnHandle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const& column = *reinterpret_cast<cudf::column_view const*>(columnHandle);
    cudf::dictionary_column_view dict_view(column);
    
    auto result = cudf::dictionary::count_by_dictionary(dict_view);
    
    // Convert table to array of column pointers
    auto const num_columns = result->num_columns();
    cudf::jni::native_jlongArray result_handles(env, num_columns);
    
    auto columns = result->release();
    for (size_t i = 0; i < columns.size(); ++i) {
      result_handles[i] = reinterpret_cast<jlong>(columns[i].release());
    }
    
    return result_handles.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * Sum values grouped by dictionary key.
 */
JNIEXPORT jlongArray JNICALL 
Java_com_nvidia_spark_rapids_jni_DictionaryOptimization_sumByDictionaryNative(
    JNIEnv* env, 
    jclass, 
    jlong columnHandle,
    jlong valuesHandle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const& column = *reinterpret_cast<cudf::column_view const*>(columnHandle);
    auto const& values = *reinterpret_cast<cudf::column_view const*>(valuesHandle);
    cudf::dictionary_column_view dict_view(column);
    
    auto result = cudf::dictionary::sum_by_dictionary(dict_view, values);
    
    // Convert table to array of column pointers
    auto const num_columns = result->num_columns();
    cudf::jni::native_jlongArray result_handles(env, num_columns);
    
    auto columns = result->release();
    for (size_t i = 0; i < columns.size(); ++i) {
      result_handles[i] = reinterpret_cast<jlong>(columns[i].release());
    }
    
    return result_handles.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * Average values grouped by dictionary key.
 */
JNIEXPORT jlongArray JNICALL 
Java_com_nvidia_spark_rapids_jni_DictionaryOptimization_avgByDictionaryNative(
    JNIEnv* env, 
    jclass, 
    jlong columnHandle,
    jlong valuesHandle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const& column = *reinterpret_cast<cudf::column_view const*>(columnHandle);
    auto const& values = *reinterpret_cast<cudf::column_view const*>(valuesHandle);
    cudf::dictionary_column_view dict_view(column);
    
    auto result = cudf::dictionary::avg_by_dictionary(dict_view, values);
    
    // Convert table to array of column pointers
    auto const num_columns = result->num_columns();
    cudf::jni::native_jlongArray result_handles(env, num_columns);
    
    auto columns = result->release();
    for (size_t i = 0; i < columns.size(); ++i) {
      result_handles[i] = reinterpret_cast<jlong>(columns[i].release());
    }
    
    return result_handles.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * Get dictionary statistics (num keys, num rows).
 */
JNIEXPORT jlongArray JNICALL 
Java_com_nvidia_spark_rapids_jni_DictionaryOptimization_getDictionaryStatsNative(
    JNIEnv* env, 
    jclass, 
    jlong columnHandle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const& column = *reinterpret_cast<cudf::column_view const*>(columnHandle);
    cudf::dictionary_column_view dict_view(column);
    
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = dict_view.keys().size();
    result[1] = dict_view.size();
    
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

} // extern "C"
