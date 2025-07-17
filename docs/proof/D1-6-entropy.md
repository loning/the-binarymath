# D1.6：熵

## 定义

**定义 D1.6**：自指完备系统信息熵的构造性定义。

### 基础构造

对于自指完备系统$S$在时刻$t$的状态集合$S_t \subseteq S$，定义信息熵函数$H: \mathcal{P}(S) \to \mathbb{R}^+$：

$$
H(S_t) = \begin{cases}
0 & \text{如果 } |S_t| = 0 \text{ 或 } |S_t| = 1 \\
\log_2 |S_t| + \text{StructuralComplexity}(S_t) & \text{如果 } |S_t| > 1
\end{cases}
$$
### 结构复杂度函数

定义$\text{StructuralComplexity}: \mathcal{P}(S) \to \mathbb{R}^+$：
$$
\text{StructuralComplexity}(S_t) = \frac{1}{|S_t|} \sum_{s \in S_t} \log_2(1 + |s|)
$$
其中$|s|$是状态$s$的二进制长度。

### 良定义性验证

**引理D1.6.1**：函数$H$良定义。

*证明*：
1. **定义域完整性**：$H$在$\mathcal{P}(S)$上处处有定义
2. **值域约束**：对所有$S_t \subseteq S$，$H(S_t) \in \mathbb{R}^+$
3. **计算可行性**：$H(S_t)$对有限集合$S_t$总是可计算的 ∎

## 形式化性质

1. **非负性**：$\forall S_t \subseteq S: H(S_t) \geq 0$
   *证明*：由定义，$\log_2$函数和$\text{StructuralComplexity}$函数均非负 ∎

2. **严格单调性**：$S_t \subsetneq S_{t'} \Rightarrow H(S_t) < H(S_{t'})$
   *证明*：
   - 如果$|S_t| < |S_{t'}|$，则$\log_2 |S_t| < \log_2 |S_{t'}|$
   - 结构复杂度项通常也增加，确保严格不等式 ∎

3. **分解性质**：对不相交的状态集$S_1, S_2$：
   
$$
H(S_1 \cup S_2) = \frac{|S_1|}{|S_1 \cup S_2|} H(S_1) + \frac{|S_2|}{|S_1 \cup S_2|} H(S_2) + \log_2 |S_1 \cup S_2|
$$
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