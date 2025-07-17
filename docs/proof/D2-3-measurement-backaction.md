# D2.3：测量反作用

## 定义

**定义 D2.3**：观察者o对状态s的反作用：
$$\text{backaction}(o,s) = d(s, o(s))$$

其中d是状态空间的距离度量。

## 距离度量选择

常用的距离度量：

1. **Hamming距离**：不同位的个数
   $$d_H(s,t) = |\{i : s[i] \neq t[i]\}|$$

2. **信息距离**：熵的变化
   $$d_I(s,t) = |H(s) - H(t)|$$

3. **结构距离**：考虑no-11模式的变化

## 反作用的性质

1. **非零性**：backaction(o,s) > 0（由[D1.5](D1-5-observer.md)）
2. **有界性**：∃M: backaction(o,s) ≤ M
3. **观察者依赖**：不同o产生不同反作用

## 最小反作用原理

存在下界：
$$\text{backaction}(o,s) \geq \text{backaction}_{\min} > 0$$

这反映了测量的基本限制。

## 累积效应

多次观察的累积反作用：
$$\text{total\_backaction} = \sum_{i=1}^n \text{backaction}(o_i, s_i)$$

## 与其他定义的关系

- 由[D1.5 观察者](D1-5-observer.md)定义产生
- 影响[D2.2 信息增量](D2-2-information-increment.md)
- 与[D1.4 时间度量](D1-4-time-metric.md)相关

## 在证明中的应用

- 在[L1.6 测量不可逆性](L1-6-measurement-irreversibility.md)中核心
- 支持[T4.3 测量坍缩](T4-3-measurement-collapse.md)
- 用于[T4.4 不确定性原理](T4-4-uncertainty-principle.md)

## 量子对应

- **波函数坍缩**：|ψ⟩ → |测量本征态⟩
- **退相干**：环境导致的持续测量
- **Zeno效应**：频繁测量抑制演化

## 信息论解释

测量提取信息必然改变系统：
$$I_{\text{gained}} \leq \text{backaction} \cdot \log_2 |S|$$

## 形式化标记

- **类型**：定义（Definition）  
- **编号**：D2.3
- **依赖**：D1.5
- **被引用**：L1.6, T4.3, T4.4等