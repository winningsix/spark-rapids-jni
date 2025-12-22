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

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Execute a fused Filter + Project operation using cuDF AST.
 *
 * This function combines filter evaluation and projection into an optimized
 * execution path that minimizes intermediate data materialization.
 *
 * The execution flow:
 * 1. Evaluate the filter expression to create a boolean mask
 * 2. Apply the mask to filter rows
 * 3. Evaluate project expressions on filtered data
 *
 * For maximum performance, expressions are compiled and can be cached.
 *
 * @param input Input table to filter and project
 * @param filter_expr Filter expression (evaluates to boolean)
 * @param project_exprs Vector of project expressions
 * @param stream CUDA stream for async execution
 * @param mr Memory resource for allocations
 * @return Result table with filtered and projected data
 */
std::unique_ptr<cudf::table> fused_filter_project(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Execute a fused Filter + Project + Aggregate operation.
 *
 * This function combines filtering, projection, and aggregation into
 * an optimized execution path.
 *
 * @param input Input table
 * @param filter_expr Filter expression
 * @param project_exprs Project expressions
 * @param agg_exprs Aggregation expressions
 * @param group_by_cols Column indices for GROUP BY
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Aggregated result table
 */
std::unique_ptr<cudf::table> fused_filter_project_aggregate(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& agg_exprs,
    std::vector<cudf::size_type> const& group_by_cols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Evaluate filter expression and return mask.
 *
 * @param input Input table
 * @param filter_expr Filter expression
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Boolean column representing the filter mask
 */
std::unique_ptr<cudf::column> evaluate_filter(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Apply multiple project expressions on a table.
 *
 * @param input Input table
 * @param project_exprs Vector of project expressions
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Table with projected columns
 */
std::unique_ptr<cudf::table> apply_projections(
    cudf::table_view const& input,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

}  // namespace spark_rapids_jni

