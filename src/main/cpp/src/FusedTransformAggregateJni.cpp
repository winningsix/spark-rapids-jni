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

#include "fused_transform_aggregate.hpp"
#include "cudf_jni_apis.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

extern "C" {

/**
 * JNI method to execute fused transform + aggregate.
 * 
 * Returns two Tables (keys and values) as separate native handles.
 * Each Table owns its columns, which use the default MR with proper ref counting.
 * 
 * @return Array of handles: [numKeyCols, keyCol0, ..., numValCols, valCol0, ...]
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_FusedTransformAggregate_executeFused(
    JNIEnv* env,
    jclass j_class,
    jlong input_table_handle,
    jintArray group_by_indices,
    jintArray transform_ops,
    jintArray agg_ops,
    jintArray value_col_indices,
    jintArray cond_col_indices,
    jintArray other_col_indices,
    jlongArray default_vals,
    jlongArray thresholds,
    jlongArray else_vals,
    jboolean enable_warp_reduction)
{
  JNI_NULL_CHECK(env, input_table_handle, "Input table is null", nullptr);
  JNI_NULL_CHECK(env, group_by_indices, "Group-by indices are null", nullptr);
  JNI_NULL_CHECK(env, transform_ops, "Transform ops are null", nullptr);
  JNI_NULL_CHECK(env, agg_ops, "Agg ops are null", nullptr);
  JNI_NULL_CHECK(env, value_col_indices, "Value column indices are null", nullptr);
  
  try {
    cudf::jni::auto_set_device(env);
    
    auto const& input_table = *reinterpret_cast<cudf::table_view const*>(input_table_handle);
    
    // Get array lengths
    jsize num_exprs = env->GetArrayLength(transform_ops);
    jsize num_group_by_cols = env->GetArrayLength(group_by_indices);
    
    // Get array elements
    jint* j_group_by = env->GetIntArrayElements(group_by_indices, nullptr);
    jint* j_transform_ops = env->GetIntArrayElements(transform_ops, nullptr);
    jint* j_agg_ops = env->GetIntArrayElements(agg_ops, nullptr);
    jint* j_value_cols = env->GetIntArrayElements(value_col_indices, nullptr);
    jint* j_cond_cols = cond_col_indices ? env->GetIntArrayElements(cond_col_indices, nullptr) : nullptr;
    jint* j_other_cols = other_col_indices ? env->GetIntArrayElements(other_col_indices, nullptr) : nullptr;
    jlong* j_defaults = default_vals ? env->GetLongArrayElements(default_vals, nullptr) : nullptr;
    jlong* j_thresholds = thresholds ? env->GetLongArrayElements(thresholds, nullptr) : nullptr;
    jlong* j_else_vals = else_vals ? env->GetLongArrayElements(else_vals, nullptr) : nullptr;
    
    // Build execution plan
    spark_rapids_jni::FusedExecutionPlan plan;
    plan.enable_warp_reduction = enable_warp_reduction;
    plan.enable_perfect_hash = true;
    
    // Add group-by columns
    for (jsize i = 0; i < num_group_by_cols; ++i) {
      plan.group_by_col_indices.push_back(j_group_by[i]);
    }
    
    // Add expressions
    for (jsize i = 0; i < num_exprs; ++i) {
      spark_rapids_jni::FusedExprSpec spec;
      spec.transform_op = static_cast<spark_rapids_jni::TransformOp>(j_transform_ops[i]);
      spec.agg_op = static_cast<spark_rapids_jni::AggOp>(j_agg_ops[i]);
      spec.value_col_idx = j_value_cols[i];
      spec.cond_col_idx = j_cond_cols ? j_cond_cols[i] : -1;
      spec.other_col_idx = j_other_cols ? j_other_cols[i] : -1;
      spec.default_val = j_defaults ? j_defaults[i] : 0;
      spec.threshold = j_thresholds ? j_thresholds[i] : 0;
      spec.else_val = j_else_vals ? j_else_vals[i] : 0;
      spec.output_type = cudf::data_type{cudf::type_id::INT64};
      spec.output_idx = i;
      
      plan.expressions.push_back(spec);
    }
    
    // Release arrays
    env->ReleaseIntArrayElements(group_by_indices, j_group_by, JNI_ABORT);
    env->ReleaseIntArrayElements(transform_ops, j_transform_ops, JNI_ABORT);
    env->ReleaseIntArrayElements(agg_ops, j_agg_ops, JNI_ABORT);
    env->ReleaseIntArrayElements(value_col_indices, j_value_cols, JNI_ABORT);
    if (j_cond_cols) env->ReleaseIntArrayElements(cond_col_indices, j_cond_cols, JNI_ABORT);
    if (j_other_cols) env->ReleaseIntArrayElements(other_col_indices, j_other_cols, JNI_ABORT);
    if (j_defaults) env->ReleaseLongArrayElements(default_vals, j_defaults, JNI_ABORT);
    if (j_thresholds) env->ReleaseLongArrayElements(thresholds, j_thresholds, JNI_ABORT);
    if (j_else_vals) env->ReleaseLongArrayElements(else_vals, j_else_vals, JNI_ABORT);
    
    // Execute
    auto result = spark_rapids_jni::execute_fused_transform_aggregate(
        input_table,
        plan,
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());
    
    // Build return array with column handles
    // Format: [numKeyCols, keyCol0, ..., numValCols, valCol0, ...]
    int num_key_cols = result.output_keys ? result.output_keys->num_columns() : 0;
    int num_val_cols = result.output_values ? result.output_values->num_columns() : 0;
    int total_handles = 2 + num_key_cols + num_val_cols;  // +2 for counts
    
    std::vector<jlong> handles(total_handles);
    int idx = 0;
    
    // Key columns - release ownership to Java
    handles[idx++] = num_key_cols;
    if (result.output_keys) {
      auto key_cols = result.output_keys->release();
      for (auto& col : key_cols) {
        handles[idx++] = reinterpret_cast<jlong>(col.release());
      }
    }
    
    // Value columns - release ownership to Java
    handles[idx++] = num_val_cols;
    if (result.output_values) {
      auto val_cols = result.output_values->release();
      for (auto& col : val_cols) {
        handles[idx++] = reinterpret_cast<jlong>(col.release());
      }
    }
    
    // NOTE: shared_buffer and buffer_mr are not used since columns use default MR
    // They will be destroyed when result goes out of scope
    
    // Create and return Java array
    jlongArray ret = env->NewLongArray(total_handles);
    env->SetLongArrayRegion(ret, 0, total_handles, handles.data());
    
    return ret;
  }
  CATCH_STD(env, nullptr);
}

/**
 * JNI method to check if a plan can be fused
 */
JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_FusedTransformAggregate_canFuse(
    JNIEnv* env,
    jclass j_class,
    jint num_expressions,
    jint num_group_by_cols)
{
  // Simple heuristic: fuse if we have at least 4 expressions
  return num_expressions >= 4 && num_group_by_cols > 0;
}

}  // extern "C"
