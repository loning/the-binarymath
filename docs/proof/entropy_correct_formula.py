#!/usr/bin/env python3
"""
推导正确的熵公式：基于no-11约束的本质
"""

import math
from typing import Set, Dict, List


def fibonacci(n: int) -> int:
    """计算第n个Fibonacci数"""
    if n <= 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def max_valid_states(n: int) -> int:
    """长度为n的满足no-11约束的最大状态数"""
    return fibonacci(n + 2)


def compute_entropy_correct(S_t: Set[str]) -> float:
    """
    正确的熵定义：考虑no-11约束
    
    关键洞察：
    1. 状态数被Fibonacci序列限制
    2. 熵应该相对于可能的最大状态数
    3. 结构复杂度应该考虑约束
    """
    if len(S_t) <= 1:
        return 0.0
    
    # 获取最大状态长度
    max_len = max(len(s) for s in S_t)
    
    # 理论最大状态数（同长度下）
    max_possible = max_valid_states(max_len)
    
    # 实际状态数
    actual = len(S_t)
    
    # 相对熵：实际vs理论最大
    if max_possible > 1:
        relative_entropy = math.log2(actual) / math.log2(max_possible)
    else:
        relative_entropy = 0
    
    # 结构复杂度（考虑长度分布）
    length_dist = {}
    for s in S_t:
        l = len(s)
        length_dist[l] = length_dist.get(l, 0) + 1
    
    # 长度分布熵
    length_entropy = 0
    for count in length_dist.values():
        p = count / len(S_t)
        if p > 0:
            length_entropy -= p * math.log2(p)
    
    # 最终熵 = 相对熵 + 长度分布熵
    return relative_entropy + 0.5 * length_entropy


def compute_entropy_variants(S_t: Set[str]) -> Dict[str, float]:
    """计算各种熵变体"""
    if len(S_t) <= 1:
        return {
            'naive': 0.0,
            'correct': 0.0,
            'normalized': 0.0,
            'constrained': 0.0
        }
    
    n = len(S_t)
    phi = (1 + math.sqrt(5)) / 2
    
    # 1. 朴素定义
    avg_len = sum(len(s) for s in S_t) / n
    H_naive = math.log2(n) + math.log2(1 + avg_len)
    
    # 2. 正确定义（考虑约束）
    H_correct = compute_entropy_correct(S_t)
    
    # 3. 归一化定义（用φ归一化）
    H_normalized = math.log2(n) / math.log2(phi) + math.log2(1 + avg_len) / math.log2(phi)
    
    # 4. 约束感知定义
    # 考虑到每步最多增加φ倍状态
    max_len = max(len(s) for s in S_t)
    theoretical_max = phi ** max_len / math.sqrt(5)
    if theoretical_max > 1:
        H_constrained = math.log2(n) / math.log2(theoretical_max)
    else:
        H_constrained = 0
    
    return {
        'naive': H_naive,
        'correct': H_correct,
        'normalized': H_normalized,
        'constrained': H_constrained
    }


def evolve_with_constraint(S_t: Set[str]) -> Set[str]:
    """考虑no-11约束的演化"""
    S_next = set()
    
    for s in S_t:
        # 总是可以追加0
        S_next.add(s + '0')
        
        # 只有当最后不是1时才能追加1
        if not s or s[-1] != '1':
            S_next.add(s + '1')
    
    return S_next


def analyze_entropy_growth():
    """分析熵增长模式"""
    phi = (1 + math.sqrt(5)) / 2
    
    print("=== 熵公式对比分析 ===")
    print(f"黄金比例 φ = {phi:.6f}")
    print(f"理论上界 log₂(φ) = {math.log2(phi):.6f}")
    print(f"理论上界 ln(φ) = {math.log(phi):.6f}")
    print()
    
    # 初始状态
    S_t = {'0', '1'}
    
    # 记录历史
    history = {
        'size': [len(S_t)],
        'naive': [],
        'correct': [],
        'normalized': [],
        'constrained': []
    }
    
    prev_entropies = compute_entropy_variants(S_t)
    
    print("步骤 | 状态数 | Δ朴素   | Δ正确   | Δ归一   | Δ约束   |")
    print("-" * 60)
    
    for step in range(15):
        # 演化
        S_t = evolve_with_constraint(S_t)
        history['size'].append(len(S_t))
        
        # 计算熵
        curr_entropies = compute_entropy_variants(S_t)
        
        # 计算熵增
        increases = {}
        for key in ['naive', 'correct', 'normalized', 'constrained']:
            if prev_entropies[key] > 0:
                increases[key] = curr_entropies[key] - prev_entropies[key]
            else:
                increases[key] = curr_entropies[key]
            
            history[key].append(increases[key])
        
        print(f"{step+1:4d} | {len(S_t):6d} | {increases['naive']:7.4f} | " +
              f"{increases['correct']:7.4f} | {increases['normalized']:7.4f} | " +
              f"{increases['constrained']:7.4f} |")
        
        # 标记违反上界的情况
        if increases['naive'] > math.log2(phi):
            print(f"       朴素熵增违反上界！")
        
        prev_entropies = curr_entropies
    
    # 统计分析
    print("\n=== 统计分析 ===")
    for key in ['naive', 'correct', 'normalized', 'constrained']:
        if history[key]:
            max_inc = max(history[key])
            avg_inc = sum(history[key]) / len(history[key])
            
            print(f"\n{key}熵:")
            print(f"  最大熵增: {max_inc:.4f}")
            print(f"  平均熵增: {avg_inc:.4f}")
            print(f"  是否满足log₂(φ)上界: {'是' if max_inc <= math.log2(phi) + 0.001 else '否'}")
    
    # 验证Fibonacci增长
    print("\n=== 验证状态数增长 ===")
    print("步骤 | 实际状态数 | Fibonacci数 | 比值")
    print("-" * 40)
    for i in range(min(10, len(history['size']))):
        actual = history['size'][i]
        fib = fibonacci(i + 2)
        ratio = actual / fib if fib > 0 else 0
        print(f"{i:4d} | {actual:10d} | {fib:11d} | {ratio:.4f}")


def test_extreme_cases():
    """测试极端情况"""
    print("\n\n=== 极端情况测试 ===")
    
    # 测试1：单一长串
    S1 = {'0' * 100}
    H1 = compute_entropy_correct(S1)
    print(f"单一长串(100个0): H = {H1:.4f}")
    
    # 测试2：最大可能集合（长度5）
    S2 = set()
    for i in range(fibonacci(7)):  # F_7 = 13
        # 生成第i个有效5位串
        s = ""
        remaining = i
        for _ in range(5):
            if remaining % 2 == 0 or (s and s[-1] == '1'):
                s += '0'
            else:
                s += '1'
            remaining //= 2
        if '11' not in s and len(s) == 5:
            S2.add(s)
    
    H2 = compute_entropy_correct(S2)
    print(f"最大5位集合({len(S2)}个): H = {H2:.4f}")
    
    # 测试3：混合长度
    S3 = {'0', '10', '010', '0101', '01010'}
    H3 = compute_entropy_correct(S3)
    print(f"混合长度集合: H = {H3:.4f}")


if __name__ == "__main__":
    analyze_entropy_growth()
    test_extreme_cases()