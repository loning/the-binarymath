# D2.1：递归层次

## 定义

**定义 D2.1**：状态s的递归层次定义为：
$$
\text{level}(s) = \min\{n | s \in \Xi^n(s_0)\}
$$
其中s₀是初始状态，Ξⁿ表示n次应用Collapse算子。

## 形式化性质

1. **单调性**：level(Ξ(s)) > level(s)
2. **良定义性**：每个可达状态有唯一层次
3. **离散性**：level(s) ∈ ℕ

## 层次结构

递归展开形成树状结构：
```
s₀ (level 0)
├── Ξ(s₀) (level 1)
│   ├── Ξ²(s₀) (level 2)
│   └── ...
└── ...
```

## 与时间的关系

在简单情况下：
- level(s) ≈ t（时间步数）
- 但一般level(s) ≤ t（可能有横向演化）

## 信息内容

递归层次反映信息复杂度：
$$
H(s) \geq f(\text{level}(s))
$$
其中f是某个递增函数。

## 与其他定义的关系

- 基于[D1.7 Collapse算子](D1-7-collapse-operator.md)
- 与[D1.4 时间度量](D1-4-time-metric.md)相关
- 影响[D2.2 信息增量](D2-2-information-increment.md)

## 在证明中的应用

- 用于[L1.8 递归不终止](L1-8-recursion-non-termination.md)
- 在[T5.6 Kolmogorov复杂度](T5-6-kolmogorov-complexity.md)中出现
- 解释逻辑深度概念

## 计算复杂度

- 判定level(s)可能是计算困难的
- 但存在有效的上界估计
- 与压缩复杂度相关

## 哲学意义

递归层次反映了：
- 意识的深度
- 理解的层次
- 抽象的等级

## 形式化标记

- **类型**：定义（Definition）
- **编号**：D2.1
- **依赖**：D1.7
- **被引用**：L1.8, T5.6等