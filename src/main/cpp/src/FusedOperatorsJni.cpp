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
#include "fused_operators.hpp"
#include "jni_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>

extern "C" {

/**
 * JNI method to execute fused filter + project operation.
 * 
 * @param env JNI environment
 * @param tableHandle Native handle to the input table
 * @param filterExprHandle Native handle to the compiled filter expression
 * @param projectExprHandles Array of native handles to project expressions
 * @return Native handle to the result table
 */
JNIEXPORT jlong JNICALL 
Java_com_nvidia_spark_rapids_jni_FusedOperators_fusedFilterProjectNative(
    JNIEnv* env,
    jclass,
    jlong tableHandle,
    jlong filterExprHandle,
    jlongArray projectExprHandles)
{
    JNI_NULL_CHECK(env, tableHandle, "table handle is null", 0);
    JNI_NULL_CHECK(env, filterExprHandle, "filter expression handle is null", 0);
    JNI_NULL_CHECK(env, projectExprHandles, "project expressions handle is null", 0);
    
    JNI_TRY
    {
        cudf::jni::auto_set_device(env);
        
        // Get the input table
        auto const& table = *reinterpret_cast<cudf::table_view const*>(tableHandle);
        
        // Get the filter expression
        auto const& filter_expr = *reinterpret_cast<cudf::ast::expression const*>(filterExprHandle);
        
        // Get project expressions
        cudf::jni::native_jlongArray project_handles(env, projectExprHandles);
        std::vector<std::reference_wrapper<cudf::ast::expression const>> project_exprs;
        project_exprs.reserve(project_handles.size());
        
        for (int i = 0; i < project_handles.size(); ++i) {
            auto const* expr = reinterpret_cast<cudf::ast::expression const*>(project_handles[i]);
            project_exprs.push_back(std::cref(*expr));
        }
        
        // Execute fused operation
        auto result = spark_rapids_jni::fused_filter_project(
            table,
            filter_expr,
            project_exprs,
            cudf::get_default_stream(),
            cudf::get_current_device_resource_ref()
        );
        
        return cudf::jni::release_as_jlong(std::move(result));
    }
    JNI_CATCH(env, 0);
}

/**
 * JNI method to execute fused filter + project + aggregate operation.
 */
JNIEXPORT jlong JNICALL 
Java_com_nvidia_spark_rapids_jni_FusedOperators_fusedFilterProjectAggregateNative(
    JNIEnv* env,
    jclass,
    jlong tableHandle,
    jlong filterExprHandle,
    jlongArray projectExprHandles,
    jlongArray aggExprHandles,
    jintArray groupByColumns)
{
    JNI_NULL_CHECK(env, tableHandle, "table handle is null", 0);
    JNI_NULL_CHECK(env, filterExprHandle, "filter expression handle is null", 0);
    JNI_NULL_CHECK(env, projectExprHandles, "project expressions handle is null", 0);
    JNI_NULL_CHECK(env, aggExprHandles, "aggregate expressions handle is null", 0);
    
    JNI_TRY
    {
        cudf::jni::auto_set_device(env);
        
        // Get the input table
        auto const& table = *reinterpret_cast<cudf::table_view const*>(tableHandle);
        
        // Get the filter expression
        auto const& filter_expr = *reinterpret_cast<cudf::ast::expression const*>(filterExprHandle);
        
        // Get project expressions
        cudf::jni::native_jlongArray project_handles(env, projectExprHandles);
        std::vector<std::reference_wrapper<cudf::ast::expression const>> project_exprs;
        project_exprs.reserve(project_handles.size());
        
        for (int i = 0; i < project_handles.size(); ++i) {
            auto const* expr = reinterpret_cast<cudf::ast::expression const*>(project_handles[i]);
            project_exprs.push_back(std::cref(*expr));
        }
        
        // Get aggregate expressions
        cudf::jni::native_jlongArray agg_handles(env, aggExprHandles);
        std::vector<std::reference_wrapper<cudf::ast::expression const>> agg_exprs;
        agg_exprs.reserve(agg_handles.size());
        
        for (int i = 0; i < agg_handles.size(); ++i) {
            auto const* expr = reinterpret_cast<cudf::ast::expression const*>(agg_handles[i]);
            agg_exprs.push_back(std::cref(*expr));
        }
        
        // Get group by columns
        std::vector<cudf::size_type> group_by_cols;
        if (groupByColumns != nullptr) {
            cudf::jni::native_jintArray group_cols(env, groupByColumns);
            group_by_cols.reserve(group_cols.size());
            for (int i = 0; i < group_cols.size(); ++i) {
                group_by_cols.push_back(static_cast<cudf::size_type>(group_cols[i]));
            }
        }
        
        // Execute fused operation
        auto result = spark_rapids_jni::fused_filter_project_aggregate(
            table,
            filter_expr,
            project_exprs,
            agg_exprs,
            group_by_cols,
            cudf::get_default_stream(),
            cudf::get_current_device_resource_ref()
        );
        
        return cudf::jni::release_as_jlong(std::move(result));
    }
    JNI_CATCH(env, 0);
}

/**
 * JNI method to compile an AST expression from JSON.
 * 
 * Note: This is a placeholder. Full implementation would parse JSON
 * and build the AST expression tree.
 */
JNIEXPORT jlong JNICALL 
Java_com_nvidia_spark_rapids_jni_FusedOperators_compileExpressionNative(
    JNIEnv* env,
    jclass,
    jstring astJsonString,
    jintArray inputColumnTypes)
{
    JNI_NULL_CHECK(env, astJsonString, "AST JSON string is null", 0);
    
    JNI_TRY
    {
        cudf::jni::auto_set_device(env);
        
        // TODO: Implement JSON parsing and AST construction
        // For now, return a placeholder
        // This would parse the JSON representation of the expression
        // and build a cudf::ast::expression tree
        
        // Placeholder: return 0 to indicate not implemented
        JNI_THROW_NEW(env, "java/lang/UnsupportedOperationException",
                      "Expression compilation from JSON not yet implemented", 0);
        
        return 0;
    }
    JNI_CATCH(env, 0);
}

/**
 * JNI method to free a compiled expression.
 */
JNIEXPORT void JNICALL 
Java_com_nvidia_spark_rapids_jni_FusedOperators_freeExpressionNative(
    JNIEnv* env,
    jclass,
    jlong exprHandle)
{
    JNI_TRY
    {
        cudf::jni::auto_set_device(env);
        
        if (exprHandle != 0) {
            auto* expr = reinterpret_cast<cudf::ast::expression*>(exprHandle);
            delete expr;
        }
    }
    JNI_CATCH(env, );
}

}  // extern "C"

