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

#include "fused_operators.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> evaluate_filter(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    CUDF_FUNC_RANGE();
    
    // Use cuDF AST compute_column to evaluate the filter expression
    // This produces a boolean column
    return cudf::compute_column(input, filter_expr, stream, mr);
}

std::unique_ptr<cudf::table> apply_projections(
    cudf::table_view const& input,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    CUDF_FUNC_RANGE();
    
    std::vector<std::unique_ptr<cudf::column>> result_columns;
    result_columns.reserve(project_exprs.size());
    
    // Evaluate each project expression using cuDF AST
    for (auto const& expr : project_exprs) {
        auto col = cudf::compute_column(input, expr.get(), stream, mr);
        result_columns.push_back(std::move(col));
    }
    
    return std::make_unique<cudf::table>(std::move(result_columns));
}

std::unique_ptr<cudf::table> fused_filter_project(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    CUDF_FUNC_RANGE();
    
    if (input.num_rows() == 0) {
        // Handle empty input
        std::vector<std::unique_ptr<cudf::column>> empty_columns;
        for (size_t i = 0; i < project_exprs.size(); ++i) {
            empty_columns.push_back(cudf::make_empty_column(cudf::type_id::INT64));
        }
        return std::make_unique<cudf::table>(std::move(empty_columns));
    }
    
    // Step 1: Evaluate filter expression to get boolean mask
    auto filter_mask = evaluate_filter(input, filter_expr, stream, mr);
    
    // Step 2: Apply boolean mask to filter the input table
    // This uses cudf::apply_boolean_mask which is optimized for GPU
    auto filtered_table = cudf::apply_boolean_mask(input, filter_mask->view(), stream, mr);
    
    // Step 3: Evaluate project expressions on filtered data
    // This is where we get the benefit - only computing projections on filtered rows
    return apply_projections(filtered_table->view(), project_exprs, stream, mr);
}

std::unique_ptr<cudf::table> fused_filter_project_aggregate(
    cudf::table_view const& input,
    cudf::ast::expression const& filter_expr,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& project_exprs,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& agg_exprs,
    std::vector<cudf::size_type> const& group_by_cols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    CUDF_FUNC_RANGE();
    
    // Step 1 & 2: Fused filter + project
    auto projected = fused_filter_project(input, filter_expr, project_exprs, stream, mr);
    
    // Step 3: Apply aggregation
    // For now, we return the projected result
    // Full aggregation fusion requires more complex AST handling
    // TODO: Implement aggregation fusion in Phase 2
    
    return projected;
}

}  // namespace spark_rapids_jni

