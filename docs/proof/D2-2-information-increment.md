# D2.2：信息增量

## 定义

**定义 D2.2**：从状态s到Ξ(s)的信息增量的构造性定义。

### 信息度量函数

设H: S → ℝ⁺为信息度量函数，对于状态$s \in S$：
$$
H(s) = \log_2 |s| + \text{Complexity}(s)
$$
其中Complexity(s)是s的结构复杂度。

### 信息增量定义

从状态s到Ξ(s)的信息增量定义为：
$$
\Delta I(s) = H(\Xi(s)) - H(s)
$$

### 正定性证明

由[D1.7 Collapse算子](D1-7-collapse-operator.md)的扩展性质，有$|\Xi(s)| > |s|$，因此：
$$
\Delta I(s) \geq \log_2 |\Xi(s)| - \log_2 |s| = \log_2 \frac{|\Xi(s)|}{|s|} > 0
$$
## 形式化性质

1. **正定性**：∀s ∈ S: ΔI(s) > 0
2. **有界性**：∃c > 0: ΔI(s) ≥ c（正下界）
3. **累积性**：总信息I_n = Σᵢ₌₀ⁿ⁻¹ ΔI(Ξⁱ(s₀))

## 信息增量的来源

1. **新结构**：Ξ(s)包含s中没有的模式
2. **关联信息**：s与Ξ(s)之间的映射关系
3. **元信息**：关于递归过程本身的信息

## 最小增量定理

存在常数c > 0使得：
$$
\Delta I(s) \geq c, \quad \forall s \in S
$$
这保证了真正的信息创生。

## 与熵增的关系

系统总熵增：
$$
H(S_{t+1}) - H(S_t) \geq \sum_{s \in S_t} p(s) \cdot \Delta I(s)
$$
其中p(s)是状态s的概率。

## 与其他定义的关系

- 由[D1.7 Collapse算子](D1-7-collapse-operator.md)产生
- 使用[D1.6 熵](D1-6-entropy.md)度量
- 支撑[D2.1 递归层次](D2-1-recursive-level.md)

## 在证明中的应用

- 在[T3.1 熵增定理](T3-1-entropy-increase.md)中使用
- 支持[T3.2 熵增下界](T3-2-entropy-lower-bound.md)
- 用于[T5.6 Kolmogorov复杂度](T5-6-kolmogorov-complexity.md)

## 计算示例

若s是n位二进制串：
- 简单情况：ΔI(s) ≈ log₂(n+1)
- 复杂情况：ΔI(s)依赖于s的结构

## 物理对应

- **能量耗散**：信息增加需要能量
- **时间代价**：ΔI与时间增量相关
- **不可逆性**：信息只增不减

## 形式化标记

- **类型**：定义（Definition）
- **编号**：D2.2
- **依赖**：D1.6, D1.7
- **被引用**：T3.1, T3.2, T5.6等