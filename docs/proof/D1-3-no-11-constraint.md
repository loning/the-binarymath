# D1.3：no-11约束（No-11 Constraint）

## 形式化定义

**定义 D1.3**：二进制序列$s$满足no-11约束，当且仅当：
$$
\text{No11Constraint}(s) \equiv \forall i \in [0, |s|-2]: \neg(s[i] = 1 \land s[i+1] = 1)
$$
其中：
- $s \in \{0,1\}^*$：二进制序列
- $s[i]$：序列$s$的第$i$个元素（从0开始索引）
- $|s|$：序列$s$的长度

**直观理解**：序列中不允许出现连续的两个1。

**示例**：
- $010101$ ✓ （满足约束）
- $011010$ ✗ （包含"11"，不满足）
- $100010$ ✓ （满足约束）

## 形式化条件

给定：
- $V \subseteq \{0,1\}^*$：满足no-11约束的所有序列集合
- $\text{substrings}(s)$：序列$s$的所有子串集合
- $F_n$：第$n$个Fibonacci数（$F_1=1, F_2=2, F_n=F_{n-1}+F_{n-2}$）

## 形式化证明

**引理 D1.3.1**：等价表述
$$
\text{No11Constraint}(s) \equiv \text{"11"} \notin \text{substrings}(s)
$$
*证明*：不存在连续两个1等价于子串"11"不在序列中。

**引理 D1.3.2**：间隔性质
$$
\text{No11Constraint}(s) \equiv \forall i,j: (s[i] = 1 \land s[j] = 1 \land i < j) \Rightarrow j - i \geq 2
$$
*含义*：任意两个1之间至少相隔一个位置。

**引理 D1.3.3**：正则语言表示
$$
V = \{s \in \{0,1\}^* | \text{No11Constraint}(s)\} = L(0^*(10^+)^*0^*)
$$
*解释*：满足约束的序列可以被正则表达式$0^*(10^+)^*0^*$匹配。

## 机器验证算法

**算法 D1.3.1**：no-11约束验证
```python
def verify_no_11_constraint(s):
    """
    验证二进制序列是否满足no-11约束
    
    输入：s ∈ {0,1}*（二进制字符串）
    输出：boolean
    """
    for i in range(len(s) - 1):
        if s[i] == '1' and s[i+1] == '1':
            return False
    return True
```

**算法 D1.3.2**：有效序列计数
```python
def count_valid_sequences(n):
    """
    计算长度为n的满足no-11约束的序列数量
    
    输入：n ∈ N（序列长度）
    输出：|V_n|（有效序列数）
    """
    if n == 0:
        return 1  # 空序列
    if n == 1:
        return 2  # "0" 和 "1"
    if n == 2:
        return 3  # "00", "01", "10"
    
    # 动态规划：V[n] = V[n-1] + V[n-2]
    V = [0] * (n + 1)
    V[1] = 2
    V[2] = 3
    
    for k in range(3, n + 1):
        V[k] = V[k-1] + V[k-2]
    
    return V[n]
```

## 依赖关系

- **输入**：[D1.2](D1-2-binary-representation.md)
- **输出**：约束后的二进制序列空间
- **影响**：[D1.7](D1-7-collapse-operator.md), [D1.8](D1-8-phi-representation.md)

## 形式化性质

**性质 D1.3.1**：计数公式
$$
|V_n| = F_{n+2}
$$
其中$V_n = \{s \in \{0,1\}^n | \text{No11Constraint}(s)\}$

*证明思路*：通过递推关系$|V_n| = |V_{n-1}| + |V_{n-2}|$与Fibonacci数列相同。

**性质 D1.3.2**：渐近密度
$$
\lim_{n \to \infty} \frac{|V_n|}{2^n} = \frac{1}{\phi^2} \approx 0.382
$$
*含义*：在所有$n$位二进制序列中，约38.2%满足no-11约束。

**性质 D1.3.3**：最大1数
$$
\max_{s \in V_n} |\{i: s[i] = 1\}| = \lfloor\frac{n+1}{2}\rfloor
$$
*示例*：长度为5的序列最多可以有3个1，如$10101$。

**性质 D1.3.4**：递推关系
根据最后一位，有效序列可分为两类：
- $V_n^{(0)}$：以0结尾的长度为$n$的有效序列
- $V_n^{(1)}$：以1结尾的长度为$n$的有效序列

递推关系：
$$
V_n^{(0)} = V_{n-1}^{(0)} \cup V_{n-1}^{(1)}
$$
$$
V_n^{(1)} = V_{n-1}^{(0)}
$$
$$
|V_n| = |V_n^{(0)}| + |V_n^{(1)}| = |V_{n-1}| + |V_{n-2}|
$$
**性质 D1.3.5**：闭包性
$$
s_1, s_2 \in V \Rightarrow \text{Concat}(s_1, 0, s_2) \in V
$$
*含义*：两个满足约束的序列通过0连接后仍满足约束。

## 数学表示

1. **满足约束的序列集合**：
   
$$
V = \{s \in \{0,1\}^* | \forall i: s[i]s[i+1] \neq \text{"11"}\}
$$
2. **固定长度的有效序列集合**：
   
$$
\text{Valid}_n = \{b_1b_2...b_n \in \{0,1\}^n | \forall i \in [1,n-1]: \neg(b_i = 1 \land b_{i+1} = 1)\}
$$
3. **计数公式的显式表达式**：
   
$$
|V_n| = F_{n+2} = \frac{1}{\sqrt{5}}\left[\left(\frac{1+\sqrt{5}}{2}\right)^{n+2} - \left(\frac{1-\sqrt{5}}{2}\right)^{n+2}\right]
$$
   其中$\phi = \frac{1+\sqrt{5}}{2}$是黄金比例。