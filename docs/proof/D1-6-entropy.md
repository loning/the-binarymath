# D1.6：熵

## 定义

**定义 D1.6**：系统的信息熵定义为：
$$H(S_t) = \log_2 |S_t|$$

其中|S_t|是时刻t系统可能状态的数量。

## 性质

1. **非负性**：H(S_t) ≥ 0
2. **单调性**：|S_t| < |S_{t'}| ⟹ H(S_t) < H(S_{t'})
3. **可加性**：对独立子系统，H(S₁ × S₂) = H(S₁) + H(S₂)

## 物理意义

- 度量系统的信息含量
- 反映系统的复杂度
- 刻画不确定性程度

## 与其他定义的关系

- 依赖[D1.1 自指完备性](D1-1-self-referential-completeness.md)（状态集的定义）
- 与[D2.2 信息增量](D2-2-information-increment.md)相关
- 在[D1.4 时间度量](D1-4-time-metric.md)中起作用

## 在定理中的应用

- [L1.3 熵的单调性](L1-3-entropy-monotonicity.md)的主要对象
- [T3.1 熵增定理](T3-1-entropy-increase.md)的核心
- [T5.1 Shannon熵涌现](T5-1-shannon-entropy-emergence.md)的基础

## 计算示例

若系统有n个可能状态：
- n = 1: H = 0（完全确定）
- n = 2: H = 1（1比特信息）
- n = 2^k: H = k（k比特信息）

## 形式化标记

- **类型**：定义（Definition）
- **编号**：D1.6
- **依赖**：D1.1
- **被引用**：L1.3, T3.1, T3.2, T3.3, T5.1等