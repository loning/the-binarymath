#!/usr/bin/env python3
"""
简化版熵公式验证 - 测试不同上界
"""

import math
from typing import Set, List, Dict


def satisfies_no11_constraint(s: str) -> bool:
    """检查是否满足no-11约束"""
    return '11' not in s


def evolve_one_step(S_t: Set[str]) -> Set[str]:
    """简单演化：每个状态追加0或1（满足no-11约束）"""
    S_next = set()
    
    for s in S_t:
        # 总是可以追加0
        S_next.add(s + '0')
        
        # 只有当最后不是1时才能追加1
        if not s or s[-1] != '1':
            S_next.add(s + '1')
    
    return S_next


def compute_entropies(S_t: Set[str]) -> Dict[str, float]:
    """计算各种熵定义"""
    if len(S_t) <= 1:
        return {'log2': 0.0, 'ln': 0.0, 'scaled': 0.0}
    
    n = len(S_t)
    avg_len = sum(len(s) for s in S_t) / n
    
    # 原始定义（使用log2）
    H_log2 = math.log2(n) + math.log2(1 + avg_len)
    
    # 自然对数定义
    H_ln = math.log(n) + math.log(1 + avg_len)
    
    # 缩放定义（用黄金比例缩放）
    phi = (1 + math.sqrt(5)) / 2
    H_scaled = (1/phi) * math.log2(n) + (phi-1) * math.log2(1 + avg_len)
    
    return {
        'log2': H_log2,
        'ln': H_ln,
        'scaled': H_scaled
    }


def main():
    """测试熵增上界"""
    phi = (1 + math.sqrt(5)) / 2
    
    print("=== 熵增上界测试 ===")
    print(f"黄金比例 φ = {phi:.6f}")
    print(f"log₂(φ) = {math.log2(phi):.6f}")
    print(f"ln(φ) = {math.log(phi):.6f}")
    print()
    
    # 初始状态
    S_t = {'0', '1'}
    
    # 记录最大熵增
    max_increases = {'log2': 0, 'ln': 0, 'scaled': 0}
    
    print("步骤 | 状态数 | Δlog2   | Δln     | Δscaled |")
    print("-" * 50)
    
    prev_entropies = compute_entropies(S_t)
    
    for step in range(10):
        # 演化一步
        S_t = evolve_one_step(S_t)
        
        # 计算新熵
        curr_entropies = compute_entropies(S_t)
        
        # 计算熵增
        increases = {}
        for key in ['log2', 'ln', 'scaled']:
            if prev_entropies[key] > 0:
                increases[key] = curr_entropies[key] - prev_entropies[key]
            else:
                increases[key] = curr_entropies[key]
            
            # 更新最大值
            max_increases[key] = max(max_increases[key], increases[key])
        
        print(f"{step+1:4d} | {len(S_t):6d} | {increases['log2']:7.4f} | {increases['ln']:7.4f} | {increases['scaled']:7.4f} |")
        
        prev_entropies = curr_entropies
    
    # 分析结果
    print("\n=== 分析结果 ===")
    print(f"\nlog₂定义:")
    print(f"  最大熵增: {max_increases['log2']:.4f}")
    print(f"  是否超过log₂(φ): {'是' if max_increases['log2'] > math.log2(phi) else '否'}")
    
    print(f"\nln定义:")
    print(f"  最大熵增: {max_increases['ln']:.4f}")
    print(f"  是否超过ln(φ): {'是' if max_increases['ln'] > math.log(phi) else '否'}")
    
    print(f"\n缩放定义:")
    print(f"  最大熵增: {max_increases['scaled']:.4f}")
    print(f"  是否超过log₂(φ): {'是' if max_increases['scaled'] > math.log2(phi) else '否'}")
    
    # 测试更激进的演化
    print("\n\n=== 测试多观察者情况 ===")
    
    def evolve_aggressive(S_t: Set[str]) -> Set[str]:
        """更激进的演化：模拟多个观察者"""
        S_next = set()
        
        for s in S_t:
            # 每个状态可能产生多个新状态
            for suffix in ['0', '00', '01', '10', '000', '001', '010', '100', '101']:
                new_s = s + suffix
                if satisfies_no11_constraint(new_s):
                    S_next.add(new_s)
        
        return S_next
    
    # 重新开始
    S_t = {'0', '1'}
    prev_entropies = compute_entropies(S_t)
    
    print("\n激进演化测试:")
    print("步骤 | 状态数 | Δlog2   | Δln     |")
    print("-" * 40)
    
    for step in range(5):
        S_t = evolve_aggressive(S_t)
        curr_entropies = compute_entropies(S_t)
        
        increases = {}
        for key in ['log2', 'ln']:
            increases[key] = curr_entropies[key] - prev_entropies[key]
        
        print(f"{step+1:4d} | {len(S_t):6d} | {increases['log2']:7.4f} | {increases['ln']:7.4f} |")
        
        prev_entropies = curr_entropies
        
        # 检查是否违反上界
        if increases['log2'] > math.log2(phi):
            print(f"  ⚠️  log₂熵增({increases['log2']:.4f}) > log₂(φ)({math.log2(phi):.4f})")
        if increases['ln'] > math.log(phi):
            print(f"  ⚠️  ln熵增({increases['ln']:.4f}) > ln(φ)({math.log(phi):.4f})")


if __name__ == "__main__":
    main()