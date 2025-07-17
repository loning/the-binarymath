# D1.8：φ-表示

## 定义

**定义 D1.8**：φ-表示系统的构造性定义。

### 基础构造

1. **Fibonacci序列定义**：递归序列$F: \mathbb{N} \to \mathbb{N}$
   
$$
F(n) = \begin{cases}
   1 & \text{如果 } n = 1 \\
   2 & \text{如果 } n = 2 \\
   F(n-1) + F(n-2) & \text{如果 } n > 2
   \end{cases}
$$
2. **黄金比例**：$\phi = \frac{1+\sqrt{5}}{2}$，满足$\phi^2 = \phi + 1$

3. **Zeckendorf编码函数**：$Z: \mathbb{N} \to \{0,1\}^*$
   
   对于$n \in \mathbb{N}$，构造过程：
   - 计算$k = \max\{i : F(i) \leq n\}$
   - 定义$b_i$通过贪婪算法：
   
$$
b_i = \begin{cases}
   1 & \text{如果 } F(i) \leq n \text{ 且前面未选择} F(i+1) \\
   0 & \text{否则}
   \end{cases}
$$
### 表示唯一性

**构造引理D1.8.1**：Zeckendorf表示唯一且满足no-11约束。

*证明*：
1. **唯一性**：贪婪算法的确定性保证唯一性
2. **no-11约束**：算法构造自动避免连续的1
3. **完备性**：每个自然数都可表示 ∎

## 唯一性定理

**Zeckendorf定理**：每个正整数有唯一的Fibonacci表示，且不含连续的1。

## 与黄金比例的关系

Fibonacci数的增长率：
$$
\lim_{n \to \infty} \frac{F_{n+1}}{F_n} = \phi
$$
因此φ-表示与黄金比例深刻相关。

## 编码效率

在no-11约束下：
- 信息密度：log₂φ ≈ 0.694 bit/symbol
- 这是理论最优（见[L1.7](L1-7-phi-optimality.md)）

## 与其他定义的关系

- 基于[D1.2 二进制表示](D1-2-binary-representation.md)
- 自动满足[D1.3 no-11约束](D1-3-no-11-constraint.md)
- 实现最优[D1.6 熵](D1-6-entropy.md)编码

## 在证明中的应用

- [L1.7 φ最优性](L1-7-phi-optimality.md)证明其最优
- [T5.4 最优压缩](T5-4-optimal-compression.md)的基础
- [C2.2 黄金比例推论](C2-2-golden-ratio.md)的核心

## 编码算法

```python
def phi_encode(n):
    """将整数n编码为φ-表示"""
    fibs = [1, 2]  # F1, F2
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    
    result = []
    for f in reversed(fibs[:-1]):
        if n >= f:
            result.append('1')
            n -= f
        else:
            result.append('0')
    
    return ''.join(result)
```

## 数学美

φ-表示展现了：
- 数论与几何的统一
- 离散与连续的桥梁
- 约束产生的优美

## 形式化标记

- **类型**：定义（Definition）
- **编号**：D1.8
- **依赖**：D1.2, D1.3
- **被引用**：L1.7, T5.4, C2.2等