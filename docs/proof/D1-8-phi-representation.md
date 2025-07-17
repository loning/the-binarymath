# D1.8：φ-表示

## 定义

**定义 D1.8**：基于Fibonacci数的表示系统：
- φ = (1+√5)/2（黄金比例）
- 每个自然数n有唯一的φ-表示
- φ-表示自然满足no-11约束

## 形式化定义

对于n ∈ ℕ，其φ-表示（Zeckendorf表示）为：
$$
n = \sum_{i=1}^{k} b_i F_i
$$
其中：
- Fᵢ是第i个Fibonacci数（F₁=1, F₂=2, ...）
- bᵢ ∈ {0,1}
- bᵢbᵢ₊₁ = 0（自动满足no-11）

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