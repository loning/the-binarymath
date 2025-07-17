# D1.2：二进制表示

## 定义

**定义 D1.2**：自指完备系统的二进制表示的构造性定义。

### 基础构造

1. **字母表定义**：$\Sigma = \{0, 1\}$

2. **有限串空间**：$\Sigma^* = \bigcup_{n=0}^{\infty} \Sigma^n$，其中$\Sigma^n$是长度为n的所有二进制串的集合

3. **状态空间构造**：
   
$$
S = \{s \in \Sigma^* | |s| < \infty \wedge \text{Encodable}(s)\}
$$
   其中$\text{Encodable}: \Sigma^* \to \{0,1\}$判断串是否可以编码系统信息

### 编码函数定义

定义编码映射$\text{Encode}: \mathcal{U} \to S$，其中$\mathcal{U}$是原始对象空间：
$$
\text{Encode}(x) = \text{ToBinary}(Hash(x)) \in S
$$
其中：
- $Hash: \mathcal{U} \to \mathbb{N}$是哈希函数
- $\text{ToBinary}: \mathbb{N} \to \Sigma^*$是标准二进制转换
## 语义解释

- **0**：潜在/未实现/虚
- **1**：实现/激活/实
- **01**：从潜在到实现的转换
- **10**：从实现回归潜在

## 必要性质

1. **最小性**：|Σ| = 2是最小的非平凡字母表
2. **完备性**：任何有限信息可用二进制编码
3. **对称性**：0和1地位平等但意义相反

## 与其他定义的关系

- 是[D1.1 自指完备性](D1-1-self-referential-completeness.md)的具体实现
- 受[D1.3 no-11约束](D1-3-no-11-constraint.md)限制
- 支撑[D1.8 φ-表示](D1-8-phi-representation.md)

## 在证明中的应用

- [L1.1 二进制唯一性](L1-1-binary-uniqueness.md)证明其必然性
- [T2.1 二进制必然性定理](T2-1-binary-necessity.md)的核心
- [P3 二进制完备性](P3-binary-completeness.md)的基础

## 编码示例

- 空状态：ε（空串）
- 基态：0
- 激发态：1
- 一次递归：01
- 二次递归：0101
- 复杂模式：0100101001...（满足no-11）

## 形式化标记

- **类型**：定义（Definition）
- **编号**：D1.2
- **依赖**：D1.1
- **被引用**：D1.3, D1.8, L1.1, T2.1, P3等