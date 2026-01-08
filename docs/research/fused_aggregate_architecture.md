# Fused Aggregate 长期架构设计

## 1. 整体架构

### 1.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Expression Input (Spark Plan)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Tier 0: Pattern Matcher                           │
│  识别完整表达式模式 (SUM(coalesce(x,0)*y), VARIANCE(x), etc.)        │
│  命中率目标: 70-80%                                                  │
└─────────────────────────────────────────────────────────────────────┘
        │                                          │
        │ 命中                                      │ 未命中
        ▼                                          ▼
┌──────────────────────┐              ┌────────────────────────────────┐
│  Tier 1: Handwrite   │              │  Tier 1.5: AST Decomposer      │
│  Fused Kernels       │              │  分解为基础操作组合             │
│  (~50 种模式)        │              │                                │
│  • 零延迟            │              │  Transform: [COALESCE, MUL]    │
│  • 最优性能          │              │  Aggregate: SUM                │
└──────────────────────┘              └────────────────────────────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────────────┐
                                      │  Tier 2: Type-Erased Fused     │
                                      │  Generic Kernel                │
                                      │                                │
                                      │  • 3 计算类型 (INT64/F64/DEC)  │
                                      │  • 5 操作组 (Arith/Cmp/Logic)  │
                                      │  • ~45 种 kernel               │
                                      │  命中率: 15-25%                │
                                      └────────────────────────────────┘
                                                   │
                                                   │ 未覆盖
                                                   ▼
                                      ┌────────────────────────────────┐
                                      │  Tier 3: JIT Compiler (可选)   │
                                      │  Runtime 生成特化 kernel       │
                                      │                                │
                                      │  • LRU 缓存 1000+              │
                                      │  • 首次延迟 100-500ms          │
                                      │  命中率: 5-10%                 │
                                      └────────────────────────────────┘
                                                   │
                                                   │ Fallback
                                                   ▼
                                      ┌────────────────────────────────┐
                                      │  Tier 4: cuDF Separate Ops     │
                                      │  (String/Regex/JSON/UDF)       │
                                      │                                │
                                      │  • 非融合执行                   │
                                      │  • 兜底方案                     │
                                      └────────────────────────────────┘
```

### 1.2 与 cuDF 集成架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Spark Rapids Plugin (Scala)                      │
│  GpuFusedProjectAggregate.scala                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │ JNI
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     spark-rapids-jni (C++)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FusedAggregateDispatcher                                    │   │
│  │  • Pattern Matching                                          │   │
│  │  • Tier Selection                                            │   │
│  │  • Statistics Collection                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │              │               │              │             │
│         ▼              ▼               ▼              ▼             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │  Tier 1   │  │ Tier 1.5  │  │  Tier 2   │  │  Tier 3   │        │
│  │ Handwrite │  │   AST     │  │  Generic  │  │   JIT     │        │
│  │  Kernels  │  │ Decompose │  │  Kernel   │  │ Compiler  │        │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
│         │              │               │              │             │
└─────────┼──────────────┼───────────────┼──────────────┼─────────────┘
          │              │               │              │
          │              ▼               ▼              ▼
          │    ┌─────────────────────────────────────────────┐
          │    │              libcudf                         │
          │    ├─────────────────────────────────────────────┤
          │    │  cudf::ast::expression                      │
          │    │  cudf::groupby::aggregate                   │
          │    │  cudf::hash (for group keys)                │
          │    └─────────────────────────────────────────────┘
          │
          └──────────────────────┐
                                 ▼
                    ┌─────────────────────────┐
                    │  Custom CUDA Kernels    │
                    │  (Tier 1 独立于 cuDF)   │
                    └─────────────────────────┘
```

---

## 2. Tier 1: Handwrite Fused Kernels

### 2.1 设计目标

- **覆盖率**: 70-80% 的常见聚合模式
- **性能**: 最优，2-3x 相比非融合执行
- **延迟**: 零编译延迟

### 2.2 可组合的基础操作

采用可组合设计，而非硬编码特定模式，提高可扩展性。

#### 2.2.1 Transform 操作

```cpp
// Transform 操作类型
// 设计原则: 每个 TransformOp 对应一个特定模式，而非原子操作组合
// 这样可以针对每个模式做最优的 kernel 实现
enum class TransformOp {
  // === 基础 (已实现 ✅) ===
  IDENTITY,            // x
  COALESCE,            // coalesce(x, default)
  COALESCE_MUL_SELF,   // coalesce(x,d) * coalesce(x,d)
  COALESCE_MUL_OTHER,  // coalesce(x,d1) * coalesce(y,d2)
  CONDITIONAL,         // if(cond > t) val else default
  CONDITIONAL_COALESCE,// if(cond > t) coalesce(val,d) else 0
  
  // === TPC-H 需要 (Phase 1a 实现) ===
  MUL,                 // a * b                    -- Q6
  MUL_SUB_CONST,       // a * (const - b)         -- Q1: a * (1 - b)
  CASE_MUL,            // CASE WHEN c THEN a*b ELSE 0  -- Q14
  
  // === 扩展 (Phase 2) ===
  ABS,                 // abs(x)
  NEGATE,              // -x
  DIV,                 // a / b
  ADD,                 // a + b
  SUB,                 // a - b
};
```

#### 2.2.2 Aggregation 操作

```cpp
enum class AggOp {
  SUM,
  AVG,
  COUNT,
  MIN,
  MAX,
  // 统计类
  SUM_SQ,       // SUM(x²) - for variance calculation
  COUNT_IF,     // COUNT with condition
};
```

#### 2.2.3 通过组合实现常见模式

| 模式 | 组合表示 | 示例 SQL |
|------|----------|----------|
| 基本聚合 | `AGG(IDENTITY(col))` | `SUM(x)` |
| 空值替换 | `AGG(COALESCE(col, const))` | `SUM(coalesce(x, 0))` |
| 乘法 | `AGG(MUL(col1, col2))` | `SUM(a * b)` |
| 乘常量 | `AGG(MUL(col, LITERAL))` | `SUM(a * 0.5)` |
| 减法乘 | `AGG(MUL(col1, SUB(LITERAL, col2)))` | `SUM(a * (1 - b))` |
| 条件聚合 | `AGG(CASE_WHEN(cond, col, 0))` | `SUM(CASE WHEN c THEN x ELSE 0)` |
| 两列空值乘 | `AGG(MUL(COALESCE(col1,0), COALESCE(col2,0)))` | `SUM(coalesce(x,0)*coalesce(y,0))` |

### 2.3 Phase 1 实现状态

```
========== 已实现 (Multi-Agg Workload) ✅ ==========

Transform                    | Agg   | SQL 示例
-----------------------------|-------|----------------------------------
IDENTITY                     | SUM   | SUM(x)
IDENTITY                     | AVG   | AVG(x)
IDENTITY                     | COUNT | COUNT(x)
IDENTITY                     | MIN   | MIN(x)
IDENTITY                     | MAX   | MAX(x)
COALESCE                     | SUM   | SUM(coalesce(x, 0))
COALESCE                     | AVG   | AVG(coalesce(x, 0))
COALESCE_MUL_SELF            | SUM   | SUM(coalesce(x,0) * coalesce(x,0))
COALESCE_MUL_OTHER           | SUM   | SUM(coalesce(x,0) * coalesce(y,0))
CONDITIONAL                  | SUM   | SUM(IF(cond>t, val, 0))
CONDITIONAL_COALESCE         | SUM   | SUM(IF(cond>t, coalesce(val,d), 0))

========== 待实现 (TPC-H) 🔲 ==========

Transform                    | Agg   | SQL 示例                    | TPC-H
-----------------------------|-------|-----------------------------|---------
MUL                          | SUM   | SUM(a * b)                  | Q6
MUL_SUB_CONST                | SUM   | SUM(a * (1 - b))            | Q1
CASE_MUL                     | SUM   | SUM(CASE WHEN c THEN a*b)   | Q14
```

### 2.4 Grouped Aggregation 实现

#### 2.4.1 核心数据结构

```cpp
// Group Key Hash Table
struct GroupHashTable {
  int32_t* keys;           // Group key values (or key hash)
  int32_t* group_ids;      // Mapping: row_idx → group_id
  int32_t num_groups;      // Number of unique groups
  
  // Accumulator storage (per aggregation)
  void** accumulators;     // Array of accumulator arrays, one per agg
};
```

#### 2.4.2 两阶段聚合流程

```
Phase 1: Hash + Partial Aggregate (Per Block)
┌─────────────────────────────────────────────────────────────────────┐
│  Input Data (N rows)                                                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │
│  │ k0  │ k1  │ k0  │ k2  │ k1  │ k0  │ k2  │ k1  │  Keys            │
│  │ v0  │ v1  │ v2  │ v3  │ v4  │ v5  │ v6  │ v7  │  Values          │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │
│                           │                                          │
│                           ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Block-local Hash Table (Shared Memory)                        │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Key  │  Count  │  Sum(transformed_value)               │  │  │
│  │  ├───────┼─────────┼───────────────────────────────────────┤  │  │
│  │  │  k0   │    3    │  transform(v0)+transform(v2)+...      │  │  │
│  │  │  k1   │    3    │  transform(v1)+transform(v4)+...      │  │  │
│  │  │  k2   │    2    │  transform(v3)+transform(v6)          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Phase 2: Global Merge (Across Blocks)
┌─────────────────────────────────────────────────────────────────────┐
│  Global Hash Table (Device Memory)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Key  │  Count  │  Sum  │  Sum_Sq (for variance)  │  ...       ││
│  ├───────┼─────────┼───────┼─────────────────────────┼────────────┤│
│  │  k0   │   N0    │  S0   │  SS0                    │            ││
│  │  k1   │   N1    │  S1   │  SS1                    │            ││
│  │  k2   │   N2    │  S2   │  SS2                    │            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 Kernel 实现示例: SUM(coalesce(x,0) * y) GROUP BY key

```cpp
// Phase 1: Block-local aggregation with transform fusion
template <typename KeyT, typename ValT, typename AccT>
__global__ void sum_coalesce_mul_grouped_phase1(
    const KeyT* __restrict__ keys,
    const ValT* __restrict__ x,
    const ValT* __restrict__ y,
    const ValT default_x,
    // Output: partial results per block
    KeyT* partial_keys,
    AccT* partial_sums,
    int32_t* partial_counts,
    int n,
    int max_groups_per_block) {
  
  // Shared memory hash table
  extern __shared__ char smem[];
  auto* local_keys = reinterpret_cast<KeyT*>(smem);
  auto* local_sums = reinterpret_cast<AccT*>(smem + max_groups_per_block * sizeof(KeyT));
  auto* local_counts = reinterpret_cast<int32_t*>(local_sums + max_groups_per_block);
  
  // Initialize shared memory
  for (int i = threadIdx.x; i < max_groups_per_block; i += blockDim.x) {
    local_keys[i] = KeyT(-1);  // Invalid key marker
    local_sums[i] = AccT(0);
    local_counts[i] = 0;
  }
  __syncthreads();
  
  // Process rows assigned to this block
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    KeyT key = keys[i];
    
    // ==== Fused Transform ====
    ValT vx = (x[i] != 0) ? x[i] : default_x;  // coalesce(x, default_x)
    ValT vy = y[i];
    AccT transformed = static_cast<AccT>(vx) * static_cast<AccT>(vy);  // multiply
    
    // ==== Local Hash Insert + Aggregate ====
    int slot = hashKey(key) % max_groups_per_block;
    while (true) {
      KeyT existing = atomicCAS(&local_keys[slot], KeyT(-1), key);
      if (existing == KeyT(-1) || existing == key) {
        // Found slot, accumulate
        atomicAdd(&local_sums[slot], transformed);
        atomicAdd(&local_counts[slot], 1);
        break;
      }
      slot = (slot + 1) % max_groups_per_block;  // Linear probing
    }
  }
  __syncthreads();
  
  // Write partial results to global memory
  int block_offset = blockIdx.x * max_groups_per_block;
  for (int i = threadIdx.x; i < max_groups_per_block; i += blockDim.x) {
    if (local_keys[i] != KeyT(-1)) {
      partial_keys[block_offset + i] = local_keys[i];
      partial_sums[block_offset + i] = local_sums[i];
      partial_counts[block_offset + i] = local_counts[i];
    }
  }
}

// Phase 2: Global merge
template <typename KeyT, typename AccT>
__global__ void grouped_merge_phase2(
    const KeyT* partial_keys,
    const AccT* partial_sums,
    const int32_t* partial_counts,
    int num_partials,
    // Global hash table (pre-built by cudf::groupby)
    KeyT* global_keys,
    AccT* global_sums,
    int32_t* global_counts,
    int num_groups) {
  
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_partials; 
       i += gridDim.x * blockDim.x) {
    
    KeyT key = partial_keys[i];
    if (key == KeyT(-1)) continue;
    
    // Find group in global table (binary search or hash)
    int group_id = findGroup(global_keys, num_groups, key);
    
    // Merge
    atomicAdd(&global_sums[group_id], partial_sums[i]);
    atomicAdd(&global_counts[group_id], partial_counts[i]);
  }
}
```

#### 2.4.4 与 cuDF Hash Table 集成

```cpp
// 使用 cuDF 的 hash 基础设施构建 group mapping
#include <cudf/detail/groupby/hash/groupby_kernel.cuh>

class Tier1GroupedAggregator {
public:
  void execute(
      cudf::table_view keys,
      cudf::table_view values,
      ExprPattern pattern,
      cudf::mutable_table_view output) {
    
    // Step 1: 使用 cuDF 构建 group hash table
    auto group_info = buildGroupHashTable(keys);
    // group_info.row_to_group: 每行属于哪个 group
    // group_info.num_groups: 唯一 group 数量
    
    // Step 2: 调用手写融合 kernel
    switch (pattern) {
      case ExprPattern::SUM_COALESCE_MUL:
        launchSumCoalesceMulKernel(
            values, 
            group_info.row_to_group,
            group_info.num_groups,
            output);
        break;
      // ...
    }
  }
  
private:
  // 复用 cuDF 的 hash 逻辑
  GroupInfo buildGroupHashTable(cudf::table_view keys) {
    // 调用 cudf::groupby 内部的 hash 构建
    // 或使用 cudf::distinct + row_bitmask
    return cudf::detail::hash::build_group_mapping(keys);
  }
};
```

---

## 3. Tier 1.5: AST Decomposer

### 3.1 设计目标

- 将复杂表达式分解为 cuDF AST 可表示的基础操作
- 作为 Tier 2/3/4 的前处理步骤

### 3.2 分解逻辑

```cpp
class AstDecomposer {
public:
  struct DecomposedExpr {
    // Transform 部分: cuDF AST 操作序列
    std::unique_ptr<cudf::ast::expression> transform;
    
    // Aggregate 部分
    cudf::aggregation::Kind agg_kind;
    
    // 输入列索引
    std::vector<int> input_columns;
  };
  
  DecomposedExpr decompose(const SparkExpression& expr) {
    DecomposedExpr result;
    
    if (auto* agg = expr.as<AggregateExpr>()) {
      // 提取聚合类型
      result.agg_kind = mapSparkAggToCudf(agg->op);
      
      // 递归分解 transform
      result.transform = decomposeTransform(agg->child);
      
      // 收集输入列
      collectInputColumns(agg->child, result.input_columns);
    }
    
    return result;
  }
  
private:
  std::unique_ptr<cudf::ast::expression> decomposeTransform(const Expr& expr) {
    if (auto* col = expr.as<ColumnRef>()) {
      return std::make_unique<cudf::ast::column_reference>(col->index);
    }
    
    if (auto* lit = expr.as<Literal>()) {
      return std::make_unique<cudf::ast::literal>(lit->value);
    }
    
    if (auto* bin = expr.as<BinaryExpr>()) {
      return std::make_unique<cudf::ast::operation>(
          mapSparkOpToCudf(bin->op),
          decomposeTransform(*bin->left),
          decomposeTransform(*bin->right));
    }
    
    if (auto* coalesce = expr.as<Coalesce>()) {
      // coalesce(a, b) → CASE WHEN a IS NOT NULL THEN a ELSE b
      // 或使用 cudf::ast::operation::NULL_EQUALS
      return std::make_unique<cudf::ast::operation>(
          cudf::ast::ast_operator::COALESCE,
          decomposeTransform(*coalesce->child),
          decomposeTransform(*coalesce->default_val));
    }
    
    // CASE WHEN
    if (auto* caseWhen = expr.as<CaseWhen>()) {
      // 转换为嵌套 IF
      return std::make_unique<cudf::ast::operation>(
          cudf::ast::ast_operator::IF,
          decomposeTransform(*caseWhen->condition),
          decomposeTransform(*caseWhen->then_expr),
          decomposeTransform(*caseWhen->else_expr));
    }
    
    throw std::runtime_error("Unsupported expression for AST decomposition");
  }
};
```

---

## 4. Tier 2: Type-Erased Generic Kernel

### 4.1 设计目标

- 支持任意 Transform 组合
- 通过类型擦除减少 kernel 数量
- 性能略低于 Tier 1，但覆盖更广

### 4.2 类型系统

```cpp
// 统一计算类型 (类型擦除)
enum class ComputeType {
  INT64,      // 处理 int8/int16/int32/int64
  FLOAT64,    // 处理 float32/float64
  DECIMAL128  // 处理所有 decimal
};

// 类型映射
ComputeType getComputeType(cudf::data_type dtype) {
  switch (dtype.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
      return ComputeType::INT64;
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
      return ComputeType::FLOAT64;
    default:
      return ComputeType::DECIMAL128;
  }
}
```

### 4.3 操作规格

```cpp
// 运行时操作描述
struct TransformOpSpec {
  enum class OpType {
    COLUMN_REF,   // 引用输入列
    LITERAL,      // 常量
    UNARY_OP,     // 一元操作 (ABS, NEG, NOT, ...)
    BINARY_OP,    // 二元操作 (ADD, MUL, GT, ...)
    COALESCE,     // coalesce(a, b)
    CASE_WHEN     // CASE WHEN c THEN a ELSE b
  };
  
  OpType type;
  int op_code;       // 具体操作符 (ADD=0, SUB=1, ...)
  int left_idx;      // 左操作数索引 (在中间结果数组中)
  int right_idx;     // 右操作数索引
  int64_t literal;   // 字面量值 (如果是 LITERAL)
};

struct AggSpec {
  enum class AggType { SUM, AVG, COUNT, MIN, MAX, VARIANCE };
  AggType type;
  int transform_result_idx;  // Transform 结果索引
};
```

### 4.4 Generic Grouped Aggregation Kernel

```cpp
template <ComputeType CT>
__global__ void generic_fused_grouped_kernel(
    // 输入
    const void* const* input_columns,     // 输入列数组
    const int32_t* row_to_group,          // 行到 group 的映射
    int num_rows,
    int num_groups,
    // Transform 规格
    const TransformOpSpec* transform_ops,
    int num_transform_ops,
    // Aggregate 规格
    const AggSpec* agg_specs,
    int num_aggs,
    // 输出累加器
    void* const* accumulators) {
  
  using T = typename ComputeTypeTraits<CT>::type;
  
  // 线程本地中间结果栈
  T local_stack[MAX_STACK_SIZE];
  
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < num_rows;
       row += gridDim.x * blockDim.x) {
    
    int group_id = row_to_group[row];
    
    // ===== Execute Transform Operations =====
    int stack_top = 0;
    for (int op_idx = 0; op_idx < num_transform_ops; ++op_idx) {
      const auto& op = transform_ops[op_idx];
      
      switch (op.type) {
        case TransformOpSpec::OpType::COLUMN_REF: {
          const T* col = static_cast<const T*>(input_columns[op.op_code]);
          local_stack[stack_top++] = col[row];
          break;
        }
        
        case TransformOpSpec::OpType::LITERAL: {
          local_stack[stack_top++] = static_cast<T>(op.literal);
          break;
        }
        
        case TransformOpSpec::OpType::BINARY_OP: {
          T right = local_stack[--stack_top];
          T left = local_stack[--stack_top];
          T result = applyBinaryOp<CT>(op.op_code, left, right);
          local_stack[stack_top++] = result;
          break;
        }
        
        case TransformOpSpec::OpType::COALESCE: {
          T default_val = local_stack[--stack_top];
          T value = local_stack[--stack_top];
          local_stack[stack_top++] = (value != T(0)) ? value : default_val;
          break;
        }
        
        case TransformOpSpec::OpType::CASE_WHEN: {
          T else_val = local_stack[--stack_top];
          T then_val = local_stack[--stack_top];
          T cond = local_stack[--stack_top];
          local_stack[stack_top++] = (cond != T(0)) ? then_val : else_val;
          break;
        }
      }
    }
    
    // ===== Execute Aggregations =====
    for (int agg_idx = 0; agg_idx < num_aggs; ++agg_idx) {
      const auto& agg = agg_specs[agg_idx];
      T transformed_value = local_stack[agg.transform_result_idx];
      T* acc = static_cast<T*>(accumulators[agg_idx]);
      
      switch (agg.type) {
        case AggSpec::AggType::SUM:
          atomicAdd(&acc[group_id], transformed_value);
          break;
        case AggSpec::AggType::MIN:
          atomicMin(&acc[group_id], transformed_value);
          break;
        case AggSpec::AggType::MAX:
          atomicMax(&acc[group_id], transformed_value);
          break;
        // AVG/VARIANCE 需要额外的 count 累加器
      }
    }
  }
}

// 二元操作分发
template <ComputeType CT>
__device__ __forceinline__ 
typename ComputeTypeTraits<CT>::type applyBinaryOp(
    int op_code,
    typename ComputeTypeTraits<CT>::type left,
    typename ComputeTypeTraits<CT>::type right) {
  
  using T = typename ComputeTypeTraits<CT>::type;
  
  switch (op_code) {
    case 0: return left + right;   // ADD
    case 1: return left - right;   // SUB
    case 2: return left * right;   // MUL
    case 3: return left / right;   // DIV
    case 4: return left > right;   // GT (returns 0 or 1)
    case 5: return left < right;   // LT
    case 6: return left >= right;  // GE
    case 7: return left <= right;  // LE
    case 8: return left == right;  // EQ
    case 9: return left != right;  // NE
    default: return T(0);
  }
}
```

### 4.5 与 cuDF groupby 集成

```cpp
class Tier2GenericExecutor {
public:
  std::unique_ptr<cudf::table> execute(
      cudf::table_view keys,
      cudf::table_view values,
      const std::vector<DecomposedExpr>& exprs) {
    
    // Step 1: 使用 cuDF 构建 group 映射
    cudf::groupby::groupby gb(keys);
    auto group_info = gb.get_groups();
    // group_info.keys: unique keys
    // group_info.offsets: group 边界
    
    // 构建 row_to_group 映射
    auto row_to_group = buildRowToGroupMapping(group_info);
    
    // Step 2: 转换表达式为 GPU 可执行格式
    auto transform_ops = convertToTransformOps(exprs);
    auto agg_specs = convertToAggSpecs(exprs);
    
    // Step 3: 分配累加器
    auto accumulators = allocateAccumulators(
        group_info.keys->num_rows(), exprs);
    
    // Step 4: 执行 kernel
    ComputeType ct = determineComputeType(values);
    switch (ct) {
      case ComputeType::INT64:
        generic_fused_grouped_kernel<ComputeType::INT64><<<...>>>(
            values, row_to_group, ...);
        break;
      case ComputeType::FLOAT64:
        generic_fused_grouped_kernel<ComputeType::FLOAT64><<<...>>>(
            values, row_to_group, ...);
        break;
    }
    
    // Step 5: 组装结果
    return assembleResults(group_info.keys, accumulators, exprs);
  }
};
```

---

## 5. Tier 3: JIT Compiler (可选)

### 5.1 设计目标

- 处理 Tier 2 无法高效处理的复杂表达式
- 运行时生成特化 kernel
- 使用 LRU 缓存避免重复编译

### 5.2 JIT 编译流程

```cpp
class Tier3JitCompiler {
public:
  CUfunction getOrCompile(
      const std::vector<DecomposedExpr>& exprs,
      const std::vector<cudf::data_type>& types) {
    
    // Step 1: 计算表达式签名
    std::string signature = computeSignature(exprs, types);
    
    // Step 2: 检查缓存
    {
      std::shared_lock lock(cache_mutex_);
      auto it = kernel_cache_.find(signature);
      if (it != kernel_cache_.end()) {
        return it->second;
      }
    }
    
    // Step 3: 生成 CUDA 源码
    std::string source = generateKernelSource(exprs, types);
    
    // Step 4: 编译
    CUfunction kernel = compileWithNVRTC(source);
    
    // Step 5: 存入缓存
    {
      std::unique_lock lock(cache_mutex_);
      kernel_cache_[signature] = kernel;
    }
    
    return kernel;
  }
  
private:
  std::string generateKernelSource(
      const std::vector<DecomposedExpr>& exprs,
      const std::vector<cudf::data_type>& types) {
    
    std::stringstream ss;
    
    // 生成内联的 transform 代码
    ss << "__global__ void jit_fused_kernel(\n";
    ss << "    const void* const* inputs,\n";
    ss << "    const int32_t* row_to_group,\n";
    ss << "    int num_rows, int num_groups,\n";
    ss << "    void* const* outputs) {\n";
    ss << "  for (int row = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "       row < num_rows; row += gridDim.x * blockDim.x) {\n";
    ss << "    int group_id = row_to_group[row];\n";
    
    // 生成每个表达式的内联代码
    for (size_t i = 0; i < exprs.size(); ++i) {
      ss << generateExprCode(exprs[i], types, i);
    }
    
    ss << "  }\n";
    ss << "}\n";
    
    return ss.str();
  }
  
  std::string generateExprCode(
      const DecomposedExpr& expr,
      const std::vector<cudf::data_type>& types,
      size_t expr_idx) {
    
    std::stringstream ss;
    std::string type_name = cudfTypeToCuda(types[expr_idx]);
    
    // 生成 transform 表达式
    ss << "    " << type_name << " val_" << expr_idx << " = ";
    ss << generateTransformExpr(expr.transform.get(), types);
    ss << ";\n";
    
    // 生成聚合
    switch (expr.agg_kind) {
      case cudf::aggregation::SUM:
        ss << "    atomicAdd((" << type_name << "*)outputs[" << expr_idx 
           << "] + group_id, val_" << expr_idx << ");\n";
        break;
      // ...
    }
    
    return ss.str();
  }
  
  CUfunction compileWithNVRTC(const std::string& source) {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, source.c_str(), "jit_kernel.cu", 0, nullptr, nullptr);
    
    const char* opts[] = {"--gpu-architecture=compute_70"};
    nvrtcCompileProgram(prog, 1, opts);
    
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    
    CUmodule module;
    CUfunction kernel;
    cuModuleLoadData(&module, ptx.data());
    cuModuleGetFunction(&kernel, module, "jit_fused_kernel");
    
    nvrtcDestroyProgram(&prog);
    return kernel;
  }
  
  std::unordered_map<std::string, CUfunction> kernel_cache_;
  std::shared_mutex cache_mutex_;
};
```

---

## 6. Tier 4: cuDF Fallback

### 6.1 执行逻辑

```cpp
class Tier4Fallback {
public:
  std::unique_ptr<cudf::table> execute(
      cudf::table_view keys,
      cudf::table_view values,
      const std::vector<DecomposedExpr>& exprs) {
    
    // Step 1: 分别计算每个 transform
    std::vector<std::unique_ptr<cudf::column>> transformed;
    for (const auto& expr : exprs) {
      // 使用 cudf::ast::compute_column
      transformed.push_back(
          cudf::ast::compute_column(values, *expr.transform));
    }
    
    // Step 2: 构建 aggregation requests
    cudf::table_view transformed_view{transformed};
    cudf::groupby::groupby gb(keys);
    
    std::vector<cudf::groupby::aggregation_request> requests;
    for (size_t i = 0; i < exprs.size(); ++i) {
      requests.push_back({
          transformed_view.column(i),
          {cudf::make_aggregation(exprs[i].agg_kind)}
      });
    }
    
    // Step 3: 执行 cudf groupby
    auto [result_keys, result_aggs] = gb.aggregate(requests);
    
    // Step 4: 组装结果
    return assembleResults(std::move(result_keys), std::move(result_aggs));
  }
};
```

---

## 7. 统一调度器

```cpp
class FusedAggregateDispatcher {
public:
  std::unique_ptr<cudf::table> execute(
      cudf::table_view keys,
      cudf::table_view values,
      const std::vector<SparkExpression>& exprs) {
    
    // Step 1: 尝试 Tier 1 模式匹配
    if (auto pattern = tier1_matcher_.match(exprs)) {
      stats_.record(ExecutionTier::HANDWRITE);
      return tier1_.execute(keys, values, *pattern);
    }
    
    // Step 2: 分解表达式
    auto decomposed = ast_decomposer_.decompose(exprs);
    
    // Step 3: 检查是否可用 Tier 2
    if (tier2_.canExecute(decomposed)) {
      stats_.record(ExecutionTier::GENERIC);
      return tier2_.execute(keys, values, decomposed);
    }
    
    // Step 4: 尝试 Tier 3 JIT (如果启用)
    if (jit_enabled_ && tier3_.canCompile(decomposed)) {
      stats_.record(ExecutionTier::JIT);
      return tier3_.execute(keys, values, decomposed);
    }
    
    // Step 5: Fallback to Tier 4
    stats_.record(ExecutionTier::FALLBACK);
    return tier4_.execute(keys, values, decomposed);
  }
  
private:
  Tier1PatternMatcher tier1_matcher_;
  Tier1HandwriteExecutor tier1_;
  AstDecomposer ast_decomposer_;
  Tier2GenericExecutor tier2_;
  Tier3JitCompiler tier3_;
  Tier4Fallback tier4_;
  ExecutionStatistics stats_;
  bool jit_enabled_ = false;
};
```

---

## 8. 关键实现细节: Grouped Aggregation

### 8.1 Group Key Hash Table 策略

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Grouped Aggregation 策略选择                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  数据特征               │  推荐策略                                  │
│  ─────────────────────────────────────────────────────────────────  │
│  Groups << Rows         │  Pre-built global hash table              │
│  (少量 group)           │  所有 block 共享，atomic 更新              │
│                         │                                           │
│  Groups ~ Rows / 100    │  Two-phase aggregation                    │
│  (中等 group)           │  Block-local hash → Global merge          │
│                         │                                           │
│  Groups ~ Rows          │  Sort-based aggregation                   │
│  (大量 group)           │  排序后顺序聚合                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 与 cuDF Hash 基础设施集成

```cpp
// 复用 cuDF 的 hash join / groupby 基础设施

#include <cudf/detail/join/hash_join.cuh>
#include <cudf/detail/groupby/hash/groupby_kernel.cuh>

// 方案 1: 使用 cuDF 构建 row_to_group 映射，然后自定义 kernel
rmm::device_uvector<int32_t> buildRowToGroup(cudf::table_view keys) {
  // 使用 cudf::groupby 的内部 API
  auto hash_table = cudf::detail::hash::build_hash_table(keys);
  return cudf::detail::hash::compute_group_ids(keys, hash_table);
}

// 方案 2: 直接在 cuDF hash table 上做融合聚合 (需要修改 cuDF)
// 这是更深度的集成，可能需要贡献到 cuDF 上游
```


---

## 9. 实施路线图

### 9.1 Phase 1a: 功能实现 - Multi-Agg Workload + TPC-H Q6 (当前)

**目标**: 实现核心业务场景所需的 Transform + Agg 融合功能

**现状**: 
- Transform + Agg 融合 ✅ 已完成
- 使用 cudf::groupby 做分组（两次数据扫描）

**需要完成的模式**:

| 场景 | SQL 模式 | Transform | Agg | 状态 |
|------|----------|-----------|-----|------|
| Multi-Agg | `SUM(coalesce(x,0) * coalesce(y,0))` | COALESCE_MUL_OTHER | SUM | ✅ |
| Multi-Agg | `AVG(coalesce(x,0))` | COALESCE | AVG | ✅ |
| TPC-H Q6 | `SUM(a * b)` | MUL | SUM | 🔲 需添加 |
| TPC-H Q1 | `SUM(a * (1 - b))` | MUL_SUB_CONST | SUM | 🔲 需添加 |
| TPC-H Q14 | `SUM(CASE WHEN ... THEN a*b ELSE 0)` | CASE_MUL | SUM | 🔲 需添加 |

**详细计划**:

```
Week 1-2: 补充 TPC-H 所需模式
├── MUL(col, col) → TransformOp::MUL
├── MUL(col, SUB(LITERAL, col)) → TransformOp::MUL_SUB_CONST  
├── CASE_WHEN + MUL → TransformOp::CASE_MUL
└── 单元测试 & E2E 测试

Week 3-4: Spark-Rapids 集成完善
├── Pattern Matcher 完善
├── E2E 测试 (TPC-H Q1/Q6/Q14)
└── 性能基准测试
```

### 9.2 Phase 1b: 性能评估 - 是否需要 1st Pass Fusion (POC)

**背景**: 当前实现使用 cudf::groupby + fused kernel（两次数据扫描）

**评估问题**:
1. 两次扫描 vs 一次扫描的性能差距有多大？
2. 实现 1st Pass Fusion 的复杂度是否值得？

**POC 计划**:

```
评估步骤:
1. 性能 Profiling
   ├── 测量 cudf::groupby::get_groups() 耗时占比
   ├── 测量 fused kernel 耗时占比
   └── 分析内存带宽瓶颈

2. POC 实现 (如果必要)
   ├── 简单场景: SUM(x) GROUP BY key
   ├── 使用 shared memory hash table
   └── 对比两次扫描 vs 一次扫描性能

3. 决策
   ├── 如果性能提升 < 20%: 保持现状
   ├── 如果性能提升 20-50%: 优先级降低
   └── 如果性能提升 > 50%: 进入 Phase 2
```

**预期结论**: 
- 对于 **compute-bound** 场景 (复杂 transform): 两次扫描影响较小
- 对于 **memory-bound** 场景 (简单 transform): 1st Pass Fusion 可能有显著收益

### 9.3 Phase 2: 扩展覆盖率 (计划)

**目标**: 覆盖 90%+ 常见聚合模式

**时间**: 根据 Phase 1b 评估结果决定

**交付物**:
- 扩展 Transform 模式 (~50 种组合)
- 如果 POC 结论需要: 实现 1st Pass Fusion

**扩展模式**:
```
新增 Transform:
├── DIV(col, col), DIV(col, LITERAL)
├── ABS(col)
├── 嵌套 COALESCE
└── 多条件 CASE_WHEN

新增 Aggregation:
├── SUM_SQ (for VARIANCE)
├── VARIANCE, STDDEV
└── COVARIANCE
```

### 9.4 Phase 3: 通用化 (长期)

**目标**: 自动处理任意表达式

**交付物**:
- Tier 2: Type-Erased Generic Kernel (如果需要)
- Tier 3: JIT Compiler (可选)

---

## 附录 A: 文件组织

```
spark-rapids-jni/src/main/cpp/src/
├── fused_aggregate/
│   ├── dispatcher.hpp             # 统一调度器
│   ├── dispatcher.cpp
│   │
│   ├── pattern_matcher.hpp        # Tier 0: 模式识别
│   ├── pattern_matcher.cpp
│   │
│   ├── tier1/                     # Tier 1: 手写 kernel
│   │   ├── handwrite_executor.hpp
│   │   ├── kernels/
│   │   │   ├── basic_agg.cu       # SUM/AVG/COUNT/MIN/MAX + IDENTITY/COALESCE
│   │   │   ├── binary_agg.cu      # MUL/ADD/SUB combinations
│   │   │   └── conditional_agg.cu # CASE_WHEN combinations
│   │   └── grouped_hash_utils.cuh
│   │
│   ├── tier2/                     # Tier 2: 通用框架 (Phase 2)
│   │   ├── ast_decomposer.hpp
│   │   ├── generic_executor.hpp
│   │   ├── type_erasure.hpp
│   │   └── generic_fused.cu
│   │
│   ├── tier3/                     # Tier 3: JIT (Phase 3)
│   │   ├── jit_compiler.hpp
│   │   └── jit_cache.hpp
│   │
│   └── tier4/                     # Tier 4: Fallback
│       └── fallback_executor.hpp
```
