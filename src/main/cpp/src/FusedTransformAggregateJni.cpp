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

#include <memory>

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
    jclass,
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
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    // Get input table view
    auto const& input_table = *reinterpret_cast<cudf::table_view const*>(input_table_handle);
    
    // Use native array wrappers for automatic resource management
    auto const j_group_by = cudf::jni::native_jintArray(env, group_by_indices);
    auto const j_transform_ops = cudf::jni::native_jintArray(env, transform_ops);
    auto const j_agg_ops = cudf::jni::native_jintArray(env, agg_ops);
    auto const j_value_cols = cudf::jni::native_jintArray(env, value_col_indices);
    
    // Optional arrays - use unique_ptr for RAII with optional construction
    std::unique_ptr<cudf::jni::native_jintArray> j_cond_cols_ptr;
    std::unique_ptr<cudf::jni::native_jintArray> j_other_cols_ptr;
    std::unique_ptr<cudf::jni::native_jlongArray> j_defaults_ptr;
    std::unique_ptr<cudf::jni::native_jlongArray> j_thresholds_ptr;
    std::unique_ptr<cudf::jni::native_jlongArray> j_else_vals_ptr;
    
    if (cond_col_indices) {
      j_cond_cols_ptr = std::make_unique<cudf::jni::native_jintArray>(env, cond_col_indices);
    }
    if (other_col_indices) {
      j_other_cols_ptr = std::make_unique<cudf::jni::native_jintArray>(env, other_col_indices);
    }
    if (default_vals) {
      j_defaults_ptr = std::make_unique<cudf::jni::native_jlongArray>(env, default_vals);
    }
    if (thresholds) {
      j_thresholds_ptr = std::make_unique<cudf::jni::native_jlongArray>(env, thresholds);
    }
    if (else_vals) {
      j_else_vals_ptr = std::make_unique<cudf::jni::native_jlongArray>(env, else_vals);
    }
    
    // Build execution plan
    spark_rapids_jni::FusedExecutionPlan plan;
    plan.enable_warp_reduction = enable_warp_reduction;
    plan.enable_perfect_hash = true;
    
    // Add group-by columns
    plan.group_by_col_indices.reserve(j_group_by.size());
    for (int i = 0; i < j_group_by.size(); ++i) {
      plan.group_by_col_indices.push_back(j_group_by[i]);
    }
    
    // Add expressions
    auto const num_exprs = j_transform_ops.size();
    plan.expressions.reserve(num_exprs);
    for (int i = 0; i < num_exprs; ++i) {
      spark_rapids_jni::FusedExprSpec spec;
      spec.transform_op = static_cast<spark_rapids_jni::TransformOp>(j_transform_ops[i]);
      spec.agg_op = static_cast<spark_rapids_jni::AggOp>(j_agg_ops[i]);
      spec.value_col_idx = j_value_cols[i];
      spec.cond_col_idx = j_cond_cols_ptr ? (*j_cond_cols_ptr)[i] : -1;
      spec.other_col_idx = j_other_cols_ptr ? (*j_other_cols_ptr)[i] : -1;
      spec.default_val = j_defaults_ptr ? (*j_defaults_ptr)[i] : 0;
      spec.threshold = j_thresholds_ptr ? (*j_thresholds_ptr)[i] : 0;
      spec.else_val = j_else_vals_ptr ? (*j_else_vals_ptr)[i] : 0;
      spec.output_type = cudf::data_type{cudf::type_id::INT64};
      spec.output_idx = i;
      
      plan.expressions.push_back(spec);
    }
    
    // Execute fused transform + aggregate
    auto result = spark_rapids_jni::execute_fused_transform_aggregate(
        input_table,
        plan,
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());
    
    // Build output: [numKeyCols, keyCol0, ..., numValCols, valCol0, ...]
    auto const num_key_cols = result.output_keys ? result.output_keys->num_columns() : 0;
    auto const num_val_cols = result.output_values ? result.output_values->num_columns() : 0;
    auto const total_handles = 2 + num_key_cols + num_val_cols;
    
    auto out_handles = cudf::jni::native_jlongArray(env, total_handles);
    int idx = 0;
    
    // Key columns
    out_handles[idx++] = num_key_cols;
    if (result.output_keys) {
      for (auto& col : result.output_keys->release()) {
        out_handles[idx++] = cudf::jni::release_as_jlong(col);
      }
    }
    
    // Value columns
    out_handles[idx++] = num_val_cols;
    if (result.output_values) {
      for (auto& col : result.output_values->release()) {
        out_handles[idx++] = cudf::jni::release_as_jlong(col);
      }
    }
    
    return out_handles.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * JNI method to check if a plan can be fused
 */
JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_FusedTransformAggregate_canFuse(
    JNIEnv* env,
    jclass,
    jint num_expressions,
    jint num_group_by_cols)
{
  // Simple heuristic: fuse if we have at least 4 expressions
  return num_expressions >= 4 && num_group_by_cols > 0;
}

}  // extern "C"
