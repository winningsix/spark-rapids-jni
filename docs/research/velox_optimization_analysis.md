# Velox 优化技术对 AST Fused Aggregate 框架的启示

## 1. 研究目标

分析 Velox 的关键优化技术，探讨其如何帮助改进 Spark-RAPIDS 的 AST Fused Aggregate 框架。

以 **TPC-H Q6** 和 **Multi-Agg Workload** 作为分析入口。

---

## 2. TPC-H Q6 查询特征分析

### 2.1 SQL 定义

```sql
SELECT SUM(l_extendedprice * l_discount) as revenue
FROM lineitem
WHERE l_shipdate >= '1994-01-01' 
  AND l_shipdate < '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
```

### 2.2 计算特点

| 特征 | 描述 |
|------|------|
| **数据规模** | lineitem 表通常是 TPC-H 中最大的表 (SF=1 时约 600 万行) |
| **Filter 密集** | 4 个过滤条件，需要高效的条件评估 |
| **Transform** | 简单乘法 `l_extendedprice * l_discount` |
| **Aggregate** | 全局 SUM (无 GROUP BY) |
| **计算模式** | Filter → Project → Global Aggregate |

### 2.3 Velox Q6 实现

```cpp
// velox/exec/tests/utils/TpchQueryBuilder.cpp:745
TpchPlan TpchQueryBuilder::getQ6Plan() const {
  auto plan = PlanBuilder(pool_.get())
      .tableScan(kLineitem, selectedRowType, fileColumnNames,
                 {shipDateFilter,
                  "l_discount between 0.05 and 0.07",
                  "l_quantity < 24.0"})
      .project({"l_extendedprice * l_discount"})
      .partialAggregation({}, {"sum(p0)"})
      .localPartition(std::vector<std::string>{})
      .finalAggregation()
      .planNode();
}
```

---

## 3. Multi-Agg Workload 查询特征分析

### 3.1 SQL 模式

```sql
SELECT 
  vid,
  SUM(coalesce(pre_capping2000, 0)),
  SUM(coalesce(pre_capping2000, 0) * coalesce(pre_capping2000, 0)),  -- variance
  AVG(CASE WHEN cond > 0 THEN pre_capping2000 ELSE 0 END),
  ...  -- 100+ aggregations
FROM source_table
GROUP BY vid
```

### 3.2 计算特点

| 特征 | 描述 |
|------|------|
| **聚合数量** | 100-200 个聚合函数 |
| **表达式复杂度** | 每个聚合包含嵌套的 coalesce/CASE WHEN |
| **数据规模** | 百万级行，百级分组键 |
| **计算模式** | Project (多表达式) → Grouped Aggregate |
| **瓶颈** | 大量中间列内存分配 |

---

## 4. Velox 关键优化技术分析

### 4.1 强制内联 (FOLLY_ALWAYS_INLINE, INLINE_LAMBDA)

#### 原理
```cpp
// Velox 使用 FOLLY_ALWAYS_INLINE 强制编译器内联热点函数
template <typename T>
struct FuncOne {
  FOLLY_ALWAYS_INLINE bool call(out_type<Varchar>& result, 
                                 const arg_type<Varchar>& arg1) {
    // 函数体被直接内联到调用处
  }
};

// INLINE_LAMBDA 用于内联 lambda 表达式
rows.applyToSelected([&](auto row) INLINE_LAMBDA {
  // Lambda 体被内联
});
```

#### 效果
- 消除函数调用开销 (栈操作、寄存器保存/恢复)
- 允许跨函数边界优化 (常量折叠、死代码消除)

#### 对 AST Fused Aggregate 的启示

**当前问题**：cuDF AST evaluator 使用虚函数表进行表达式分发
```cpp
// cudf/ast/detail/expression_evaluator.cuh
template <typename LHS, typename RHS>
auto operator()(LHS lhs, RHS rhs) const {
  // 虚函数调用开销
  return cudf::ast::detail::operator_dispatcher{}(op, lhs, rhs);
}
```

**改进方向**：
1. 使用模板展开替代运行时分发
2. 为常见表达式模式 (乘法+SUM、coalesce+SUM) 生成特化 kernel
3. 利用 CUDA __forceinline__ 内联关键函数

### 4.2 编译时展开 (模板递归 + constexpr)

#### Velox 实现
```cpp
// velox/expression/SimpleFunctionAdapter.h
template <size_t... Is>
bool allPrimitiveArgsFlatConstantImpl(
    const std::vector<VectorPtr>& args,
    std::index_sequence<Is...>) const {
  return ([&]() {
    if constexpr (isVariadicType<arg_at<Is>>::value) {
      return true;
    } else {
      return args[Is]->isFlatEncoding() || args[Is]->isConstantEncoding();
    }
  }() && ...);  // C++17 fold expression
}
```

#### 效果
- 编译时确定参数数量和类型
- 消除运行时循环遍历参数

#### 对 AST Fused Aggregate 的启示

**当前问题**：JNI 层需要运行时解析表达式树
```cpp
// fused_transform_aggregate.hpp
struct FusedExprSpec {
  TransformOp transform_op;
  int64_t literal_value;  // 运行时参数
};
```

**改进方向**：
1. 使用 CUDA template 特化常见表达式组合
2. 编译时生成 kernel 配置 (类似 nvJitLink)
3. 预定义常用模式的特化版本：
   - `SUM(a * b)`
   - `SUM(coalesce(a, 0))`
   - `SUM(CASE WHEN c > 0 THEN a ELSE 0 END)`

### 4.3 连续内存访问 (FlatVector + rawValues_)

#### Velox 实现
```cpp
// velox/vector/FlatVector.h
template <typename T>
class FlatVector final : public SimpleVector<T> {
 private:
  BufferPtr values_;
  T* rawValues_;  // 直接指针访问

 public:
  const T* rawValues() const { return rawValues_; }
  
  T valueAtFast(vector_size_t idx) const {
    return rawValues_[idx];  // 直接索引，无虚函数
  }
};
```

#### 效果
- Cache 友好的内存布局
- 编译器可自动向量化
- 避免间接寻址开销

#### 对 AST Fused Aggregate 的启示

**当前 cuDF 布局**：
```cpp
// cudf 使用 column_view，可能涉及间接访问
auto data = column.data<T>();  // 可能需要解引用
```

**改进方向**：
1. 确保 Fused kernel 输入使用连续内存布局
2. 预处理将 dictionary encoding 展开为 flat
3. 使用 shared memory 预取数据块

### 4.4 AllSelected 快速路径

#### Velox 实现
```cpp
// velox/vector/SelectivityVector.h:438
template <typename Callable>
inline void SelectivityVector::applyToSelected(Callable func) const {
  if (isAllSelected()) {
    // 快速路径：简单 for 循环，编译器易于向量化
    const auto end = end_;
    for (vector_size_t row = begin_; row < end; ++row) {
      func(row);
    }
  } else {
    // 慢速路径：遍历位图
    bits::forEachSetBit(bits_.data(), begin_, end_, func);
  }
}
```

#### 效果
- 无过滤时避免位图检查
- 简单循环更容易被编译器优化

#### 对 AST Fused Aggregate 的启示

**当前问题**：每行都需要检查 null mask
```cpp
// cudf 聚合需要处理 null
if (!column.nullable() || column.is_valid(row)) {
  accumulator += column.element<T>(row);
}
```

**改进方向**：
1. 添加 "all-valid" 快速路径
2. Null-free batch 使用无条件累加 kernel
3. 分离 null 处理为独立 pass (适用于低 null 率场景)

### 4.5 显式 SIMD (xsimd 库)

#### Velox 实现
```cpp
// velox/common/base/SimdUtil.h
#include <xsimd/xsimd.hpp>

template <typename A = xsimd::default_arch>
constexpr int32_t batchByteSize(const A& = {}) {
  return sizeof(xsimd::types::simd_register<int8_t, A>);
}

// FlatVector 中的 SIMD 加载
xsimd::batch<T> loadSIMDValueBufferAt(size_t index) const;
```

#### 效果
- 利用 AVX/AVX-512 指令
- 单指令处理多个元素

#### 对 AST Fused Aggregate 的启示

**GPU 对应**：warp-level 并行
```cpp
// CUDA warp-level reduction
__device__ float warpReduce(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}
```

**改进方向**：
1. 使用 warp shuffle 进行 per-group reduction
2. 合并多个 transform 结果的 reduction
3. 利用 tensor cores 进行批量乘加

### 4.6 批处理 (SelectivityVector 位图)

#### Velox 实现
```cpp
// SelectivityVector 用于跟踪活跃行
class SelectivityVector {
  std::vector<uint64_t> bits_;  // 位图存储
  vector_size_t begin_, end_;    // 活跃范围
  
  // 批量操作
  void intersect(const SelectivityVector& other);
  void deselect(const SelectivityVector& other);
};
```

#### 效果
- 减少分支预测失败
- 批量处理减少函数调用开销

#### 对 AST Fused Aggregate 的启示

**当前问题**：每行独立处理
```cpp
// 当前 per-row 处理
for (size_t row = 0; row < num_rows; ++row) {
  if (valid_mask[row]) {
    result[group_id[row]] += transform(input[row]);
  }
}
```

**改进方向**：
1. 使用 bitmask 批量过滤
2. 将连续有效行打包处理
3. 利用 CUDA CUB 库的批量操作

---

## 5. TPC-H Q6 优化路径

### 5.1 当前 cuDF 执行流程

```
TableScan → Filter → Project → HashAggregate
   ↓           ↓        ↓           ↓
 (IO)     (4 passes) (1 alloc)  (groupby)
```

### 5.2 Velox 优化后的流程

```
TableScan + Pushdown Filters → Fused Project+Aggregate
   ↓                                    ↓
 (IO + Filter)                    (Single Pass)
```

### 5.3 AST Fused Aggregate 优化建议

```cpp
// 理想的 Fused Kernel for Q6
__global__ void q6_fused_kernel(
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ quantity,
    const int32_t* __restrict__ shipdate,
    double* __restrict__ result,
    size_t num_rows) {
  
  // Shared memory for block-level reduction
  __shared__ double block_sum;
  if (threadIdx.x == 0) block_sum = 0;
  __syncthreads();
  
  double thread_sum = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 
       i += blockDim.x * gridDim.x) {
    
    // Fused filter + transform
    bool valid = (shipdate[i] >= 19940101 && shipdate[i] < 19950101) &&
                 (discount[i] >= 0.05 && discount[i] <= 0.07) &&
                 (quantity[i] < 24.0);
    
    if (valid) {
      thread_sum += extendedprice[i] * discount[i];
    }
  }
  
  // Warp reduction
  thread_sum = warpReduce(thread_sum);
  
  // Block reduction
  if (threadIdx.x % 32 == 0) {
    atomicAdd(&block_sum, thread_sum);
  }
  __syncthreads();
  
  // Global reduction
  if (threadIdx.x == 0) {
    atomicAdd(result, block_sum);
  }
}
```

---

## 6. Multi-Agg Workload 优化路径

### 6.1 当前执行流程

```
TableScan → Project (100+ expressions) → HashAggregate (100+ aggs)
               ↓                              ↓
         (100+ temp columns)           (100+ accumulator updates)
```

### 6.2 核心问题

1. **内存压力**：100+ 中间列导致 GPU OOM
2. **Kernel Launch 开销**：每个表达式一次 launch
3. **低计算强度**：简单表达式无法隐藏内存延迟

### 6.3 AST Fused Aggregate 优化建议

#### A. 表达式分组

```cpp
// 将相似表达式分组
struct ExpressionGroup {
  enum Type { SUM_PRODUCT, AVG_CONDITIONAL, VARIANCE };
  std::vector<int> column_indices;
  std::vector<int64_t> literals;
};

// 每组使用一个特化 kernel
void dispatchKernel(ExpressionGroup group, Table input, Accumulators output);
```

#### B. 多表达式融合 Kernel

```cpp
__global__ void multi_agg_fused_kernel(
    const int64_t* __restrict__ pre_capping,
    const int64_t* __restrict__ condition,
    const int32_t* __restrict__ group_keys,
    // 多个累加器
    int64_t* __restrict__ sum_result,
    int64_t* __restrict__ sum_sq_result,
    int64_t* __restrict__ cond_sum_result,
    size_t num_rows) {
  
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 
       i += blockDim.x * gridDim.x) {
    
    int64_t val = pre_capping[i] ? pre_capping[i] : 0;  // coalesce
    int32_t key = group_keys[i];
    
    // 一次读取，多次累加
    atomicAdd(&sum_result[key], val);
    atomicAdd(&sum_sq_result[key], val * val);  // variance prep
    
    if (condition[i] > 0) {
      atomicAdd(&cond_sum_result[key], val);
    }
  }
}
```

---

## 7. 优化技术适用性对比

| Velox 技术 | TPC-H Q6 适用性 | Multi-Agg 适用性 | GPU 实现难度 |
|------------|----------------|-----------------|-------------|
| FOLLY_ALWAYS_INLINE | ⭐⭐⭐ | ⭐⭐⭐ | 中 (CUDA __forceinline__) |
| 编译时展开 | ⭐⭐ | ⭐⭐⭐ | 高 (需要 JIT 或模板) |
| 连续内存访问 | ⭐⭐⭐ | ⭐⭐⭐ | 低 (cuDF 已支持) |
| AllSelected 快速路径 | ⭐⭐⭐ | ⭐⭐ | 低 |
| 显式 SIMD | ⭐⭐ (→ warp shuffle) | ⭐⭐ | 中 |
| SelectivityVector 批处理 | ⭐⭐⭐ | ⭐⭐ | 中 (bitmask 操作) |

---

## 8. 结论与建议

### 8.1 短期改进 (1-2 周)

1. **Q6 型查询**：
   - 实现 Filter + Project + Global SUM 的 fused kernel
   - 利用 warp shuffle 进行高效 reduction

2. **Multi-Agg 查询**：
   - 识别相同 transform 模式，合并为单次计算
   - 使用 shared memory 缓存 group keys

### 8.2 中期改进 (1-2 月)

1. 引入表达式模式识别，自动选择最优 kernel
2. 实现 "AllValid" 快速路径
3. 添加 coalesce/CASE WHEN 的特化处理

### 8.3 长期方向 (季度级)

1. 开发 GPU AST JIT 编译器 (类似 nvJitLink)
2. 实现自适应 kernel 选择 (根据数据特征)
3. 探索 tensor cores 在聚合场景的应用

---

## 附录 A: Velox 代码参考

```
velox/vector/FlatVector.h          - 连续内存访问
velox/vector/SelectivityVector.h   - 批处理/位图操作
velox/expression/Expr.h            - 表达式评估框架
velox/expression/SimpleFunctionAdapter.h - 内联优化
velox/common/base/SimdUtil.h       - SIMD 工具
velox/functions/lib/aggregates/SumAggregateBase.h - SUM 实现
velox/exec/tests/utils/TpchQueryBuilder.cpp - TPC-H Q6 计划
```

## 附录 B: 相关 cuDF/Spark-Rapids 代码

```
spark-rapids-jni/src/main/cpp/src/fused_transform_aggregate.hpp
spark-rapids/sql-plugin/src/main/scala/com/nvidia/spark/rapids/GpuFusedProjectAggregate.scala
spark-rapids/sql-plugin/src/main/scala/com/nvidia/spark/rapids/AstBatchProject.scala
```

