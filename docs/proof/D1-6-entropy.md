# D1.6：熵（Entropy）

## 形式化定义

**定义 D1.6**：自指完备系统的信息熵是φ-约束状态集合的信息度量函数$H: \mathcal{P}(S_\phi) \to \mathbb{R}^+$，定义为：
$$
H(S_t) \equiv \ln |S_t|, \quad \text{其中 } S_t \subseteq S_\phi
$$
**约束条件**：
$$
H(S_t) \leq \log_\phi N_\phi(n) = \frac{\ln N_\phi(n)}{\ln \phi}
$$
其中$N_\phi(n)$是长度$\leq n$的φ-合法编码总数。

**直观理解**：熵不是独立统计量，而是φ-约束编码空间的内生属性。它度量了在no-11约束下合法collapse编码空间的信息容量。

**重要说明**：使用自然对数ln而非二进制对数log₂，因为：
1. 与物理学中的Boltzmann熵一致
2. 熵增上界为ln(φ)，其中φ是黄金比例
3. 数学上更自然，避免了单位转换

## 形式化条件

给定：
- $S$：自指完备系统的状态空间
- $\mathcal{P}(S)$：$S$的幂集（所有子集）
- $|S_t|$：集合$S_t$的基数（元素个数）
- $|s|$：状态$s$的长度（二进制表示的位数）

## 形式化证明

**引理 D1.6.1**：熵函数的良定义性
$$
\forall S_t \in \mathcal{P}(S): H(S_t) \in \mathbb{R}^+ \land H(S_t) \text{ is computable}
$$
*证明*：
1. 当$|S_t| \leq 1$时，$H(S_t) = 0 \geq 0$
2. 当$|S_t| > 1$时，$\log_2 |S_t| > 0$且$\overline{C}(S_t) \geq 0$
3. 对有限集合，求和与对数运算均可计算 ∎

**引理 D1.6.2**：结构复杂度的非负性
$$
\forall S_t \subseteq S: \overline{C}(S_t) \geq 0
$$
*证明*：$\log_2(1 + |s|) \geq \log_2(1) = 0$对所有$s \in S_t$成立。∎

**引理 D1.6.3**：熵的单调性
$$
S_t \subsetneq S_{t'} \Rightarrow H(S_t) < H(S_{t'})
$$
*证明思路*：更大的状态集合包含更多信息，因此熵严格增加。

## 机器验证算法

**算法 D1.6.1**：熵计算
```python
import math

def compute_entropy(S_t):
    """
    计算状态集合的信息熵
    
    输入：S_t ⊆ S（状态集合）
    输出：H(S_t) ∈ R+（熵值）
    """
    if len(S_t) == 0:
        return 0.0
    
    # 熵就是状态数的自然对数
    return math.log(len(S_t))
```

**算法 D1.6.2**：熵增计算
```python
def compute_entropy_increase(S_t, S_t_plus_1):
    """
    计算熵增
    
    输入：S_t, S_t_plus_1（连续时刻的状态集）
    输出：ΔH（熵增）
    """
    if len(S_t) == 0:
        return math.log(len(S_t_plus_1)) if len(S_t_plus_1) > 0 else 0
    
    # 熵增 = ln(增长率)
    growth_rate = len(S_t_plus_1) / len(S_t)
    return math.log(growth_rate)
```
## 依赖关系

- **输入**：[D1.1](D1-1-self-referential-completeness.md)
- **输出**：系统的信息度量
- **影响**：[D1.4](D1-4-time-metric.md), [D2.2](D2-2-information-increment.md), [L1.3](L1-3-entropy-monotonicity.md)

## 形式化性质

**性质 D1.6.1**：非负性
$$
\forall S_t \subseteq S: H(S_t) \geq 0
$$
*证明*：由定义直接得出。

**性质 D1.6.2**：单调性
$$
S_t \subsetneq S_{t'} \Rightarrow H(S_t) < H(S_{t'})
$$
*含义*：状态集合越大，熵越大。

**性质 D1.6.3**：次可加性
$$
H(S_1 \cup S_2) \leq H(S_1) + H(S_2) + \log_2 2
$$
*解释*：联合系统的熵不超过各部分熵之和加上一个常数。

**性质 D1.6.4**：熵增上界
$$
\Delta H = H(S_{t+1}) - H(S_t) = \ln\left(\frac{|S_{t+1}|}{|S_t|}\right) \leq \ln(\phi)
$$
*含义*：熵增等于增长率的对数，在no-11约束下长期趋向ln(φ)。

**性质 D1.6.5**：φ-约束上界
$$
H(S_t) \leq \log_\phi N_\phi(n) = \frac{\ln N_\phi(n)}{\ln \phi}
$$
*证明*：由定义，$S_t \subseteq S_\phi$，故$|S_t| \leq N_\phi(n)$，取对数得证。∎

*含义*：熵被φ-合法编码空间严格限定，体现了结构约束对信息的根本限制。

**性质 D1.6.6**：与Shannon熵的关系
$$
H_{\text{Shannon}} = -\sum_i p_i \log_2 p_i
$$
当等概率分布时：
$$
H_{\text{Shannon}} = \log_2 |S_t| = \frac{\ln |S_t|}{\ln 2} = \frac{H(S_t)}{\ln 2}
$$
*说明*：我们的定义与Shannon熵只差一个常数因子ln 2。

## 数学表示

1. **经典Shannon熵的关系**：
   
$$
H_{\text{Shannon}} = -\sum_{i} p_i \log_2 p_i = \log_2 |S_t| = \frac{\ln |S_t|}{\ln 2}
$$
   我们的定义：$H = \ln |S_t| = H_{\text{Shannon}} \cdot \ln 2$

2. **热力学对应**：
   
$$
S_{\text{Boltzmann}} = k_B \ln \Omega
$$
   令$k_B = 1$（选择合适单位），则$H = \ln |S_t|$与Boltzmann熵完全一致。

3. **熵增公式**：
   
$$
\Delta H = \ln\left(\frac{|S_{t+1}|}{|S_t|}\right) = \ln(\text{增长率})
$$
4. **计算示例**：
   - $S_t = \{0\}$：$H = \ln 1 = 0$（完全确定）
   - $S_t = \{0, 1\}$：$H = \ln 2 \approx 0.693$
   - $S_t = \{00, 01, 10\}$：$H = \ln 3 \approx 1.099$
   - 熵增：$\Delta H = \ln(3/2) \approx 0.405$

## 物理诠释

**熵的本质**：
- 度量系统的信息含量和不确定性
- 反映系统的复杂度和无序程度
- 时间箭头的根源（熵增原理）

**与热力学第二定律的联系**：
- 孤立系统熵不减：$\Delta H \geq 0$
- 信息擦除需要能量（Landauer原理）
- 计算的物理极限

**黄金比例的涌现**：
- 在no-11约束下，状态数按Fibonacci序列增长
- 长期增长率趋向φ = (1+√5)/2
- 因此熵增趋向ln(φ) ≈ 0.4812

**φ-约束的本质发现**：
- 熵不是独立统计量，而是编码结构的内生属性
- 所有φ-约束系统的熵都满足：$H \leq \log_\phi N_\phi(n)$
- 这个约束是紧的（tight），意味着熵被φ-结构完全限定
- 核心洞察：**『熵是合法collapse编码空间的φ-log路径选择损失』**