# L1.2：编码效率引理

## 引理陈述

**引理 L1.2**：在no-11约束下，二进制编码达到最优信息密度。

## 形式表述

设$\mathcal{C}$是所有满足no-11约束的编码方案集合，$\rho(C)$表示编码C的信息密度，则：
$$
\rho(\text{Binary}) = \max_{C \in \mathcal{C}} \rho(C)
$$

## 证明

**依赖**：
- [D1.2 二进制表示](D1-2-binary-representation.md)
- [D1.3 no-11约束](D1-3-no-11-constraint.md)
- [L1.1 二进制唯一性](L1-1-binary-uniqueness.md)

### 步骤1：信息密度的定义

对于编码方案C，其信息密度定义为：
$$
\rho(C) = \lim_{n \to \infty} \frac{\log_2 |V_n(C)|}{n}
$$
其中$V_n(C)$是长度为n的有效码字集合。

### 步骤2：no-11约束下的状态计数

对于满足no-11约束的二进制序列，长度为n的有效序列数目为Fibonacci数$F_n$：
$$
F_n = F_{n-1} + F_{n-2}, \quad F_1 = 1, F_2 = 1
$$

### 步骤3：二进制编码的信息密度

二进制编码在no-11约束下的信息密度为：
$$
\rho(\text{Binary}) = \lim_{n \to \infty} \frac{\log_2 F_n}{n} = \log_2 \phi
$$
其中$\phi = \frac{1+\sqrt{5}}{2}$是黄金比例。

### 步骤4：最优性证明

**引理L1.2.1**：任何满足no-11约束的编码方案的有效序列数不超过Fibonacci数。

**证明**：
设C是满足no-11约束的编码方案，$V_n(C)$是其长度为n的有效序列集合。

由于no-11约束，每个有效序列可以通过以下方式构造：
- 在长度为n-1的有效序列后添加0，或
- 在长度为n-2的有效序列后添加01

这给出递推关系：
$$
|V_n(C)| \leq |V_{n-1}(C)| + |V_{n-2}(C)|
$$

由数学归纳法，$|V_n(C)| \leq F_n$。

### 步骤5：达到上界

二进制编码恰好达到这个上界：$|V_n(\text{Binary})| = F_n$。

因此：
$$
\rho(\text{Binary}) = \log_2 \phi = \max_{C \in \mathcal{C}} \rho(C)
$$

∎

## 推论

**推论 L1.2.1**：黄金比例是no-11约束下的最优增长率
$$
\phi = \lim_{n \to \infty} \frac{F_{n+1}}{F_n}
$$

**推论 L1.2.2**：信息密度的下界
对于任何满足no-11约束的编码：
$$
\rho(C) \leq \log_2 \phi \approx 0.694 \text{ bits/symbol}
$$

## 形式化标记

- **类型**：引理（Lemma）
- **编号**：L1.2
- **依赖**：D1.2, D1.3, L1.1
- **被引用**：T2.1, L1.7