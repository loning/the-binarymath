#!/usr/bin/env python3
"""
验证二进制自指完备系统的熵公式
通过模拟系统演化，找出合理的熵增范围，反推熵的正确定义
"""

import math
from typing import Set, List, Tuple, Dict


class BinarySystemEvolution:
    """二进制自指完备系统演化模拟"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比例
        self.epsilon = math.log2(self.phi)  # 最小时间量子
        self.evolution_history = []
        
    def satisfies_no11_constraint(self, s: str) -> bool:
        """检查是否满足no-11约束"""
        return '11' not in s
    
    def apply_observer(self, state: str, observer_id: int) -> str:
        """观察者作用：在状态后追加观察结果"""
        # 简单模型：观察者将其ID的二进制表示追加到状态
        observation = str(observer_id % 2)
        return state + observation
    
    def evolve_system(self, S_t: Set[str], num_observers: int) -> Set[str]:
        """系统演化一步"""
        S_next = set()
        
        # 保留原有状态（系统记忆）
        S_next.update(S_t)
        
        # 每个观察者对每个状态作用
        for state in S_t:
            for obs_id in range(num_observers):
                new_state = self.apply_observer(state, obs_id)
                if self.satisfies_no11_constraint(new_state):
                    S_next.add(new_state)
        
        return S_next
    
    def compute_entropy_v1(self, S_t: Set[str]) -> float:
        """原始熵定义：H = log2|S_t| + 平均复杂度"""
        if len(S_t) <= 1:
            return 0.0
        
        base_entropy = math.log2(len(S_t))
        avg_complexity = sum(math.log2(1 + len(s)) for s in S_t) / len(S_t)
        
        return base_entropy + avg_complexity
    
    def compute_entropy_v2(self, S_t: Set[str]) -> float:
        """修正熵定义：H = α*log2|S_t| + β*平均复杂度"""
        if len(S_t) <= 1:
            return 0.0
        
        alpha = 1 / self.phi  # ≈ 0.618
        beta = self.phi - 1   # ≈ 0.618
        
        base_entropy = alpha * math.log2(len(S_t))
        avg_complexity = beta * sum(math.log2(1 + len(s)) for s in S_t) / len(S_t)
        
        return base_entropy + avg_complexity
    
    def compute_entropy_v3(self, S_t: Set[str]) -> float:
        """考虑关联的熵定义：包含状态间的相关性"""
        if len(S_t) <= 1:
            return 0.0
        
        # 基础熵
        base_entropy = math.log2(len(S_t))
        
        # 结构复杂度
        avg_complexity = sum(math.log2(1 + len(s)) for s in S_t) / len(S_t)
        
        # 关联复杂度（简化版：基于共同前缀长度）
        correlation = 0
        states_list = list(S_t)
        if len(states_list) > 1:
            for i in range(len(states_list)):
                for j in range(i+1, len(states_list)):
                    # 计算共同前缀长度
                    common_prefix_len = 0
                    for k in range(min(len(states_list[i]), len(states_list[j]))):
                        if states_list[i][k] == states_list[j][k]:
                            common_prefix_len += 1
                        else:
                            break
                    correlation += common_prefix_len
            
            correlation = correlation / (len(states_list) * (len(states_list) - 1) / 2)
            correlation = math.log2(1 + correlation) if correlation > 0 else 0
        
        return base_entropy + avg_complexity - 0.5 * correlation
    
    def compute_entropy_v4(self, S_t: Set[str]) -> float:
        """使用自然对数的熵定义"""
        if len(S_t) <= 1:
            return 0.0
        
        # 使用自然对数
        base_entropy = math.log(len(S_t))
        avg_complexity = sum(math.log(1 + len(s)) for s in S_t) / len(S_t)
        
        return base_entropy + avg_complexity
    
    def simulate_evolution(self, initial_states: Set[str], steps: int, 
                         num_observers: int = 2) -> Dict[str, List[float]]:
        """模拟系统演化并记录熵的变化"""
        S_t = initial_states.copy()
        
        results = {
            'states': [S_t],
            'sizes': [len(S_t)],
            'entropy_v1': [self.compute_entropy_v1(S_t)],
            'entropy_v2': [self.compute_entropy_v2(S_t)],
            'entropy_v3': [self.compute_entropy_v3(S_t)],
            'entropy_v4': [self.compute_entropy_v4(S_t)],
            'entropy_increase_v1': [],
            'entropy_increase_v2': [],
            'entropy_increase_v3': [],
            'entropy_increase_v4': []
        }
        
        for step in range(steps):
            # 演化一步
            S_next = self.evolve_system(S_t, num_observers)
            
            # 计算各种熵
            H1_t = results['entropy_v1'][-1]
            H2_t = results['entropy_v2'][-1]
            H3_t = results['entropy_v3'][-1]
            H4_t = results['entropy_v4'][-1]
            
            H1_next = self.compute_entropy_v1(S_next)
            H2_next = self.compute_entropy_v2(S_next)
            H3_next = self.compute_entropy_v3(S_next)
            H4_next = self.compute_entropy_v4(S_next)
            
            # 记录结果
            results['states'].append(S_next)
            results['sizes'].append(len(S_next))
            results['entropy_v1'].append(H1_next)
            results['entropy_v2'].append(H2_next)
            results['entropy_v3'].append(H3_next)
            results['entropy_v4'].append(H4_next)
            
            # 计算熵增
            if H1_t > 0:
                results['entropy_increase_v1'].append(H1_next - H1_t)
            else:
                results['entropy_increase_v1'].append(H1_next)
                
            if H2_t > 0:
                results['entropy_increase_v2'].append(H2_next - H2_t)
            else:
                results['entropy_increase_v2'].append(H2_next)
                
            if H3_t > 0:
                results['entropy_increase_v3'].append(H3_next - H3_t)
            else:
                results['entropy_increase_v3'].append(H3_next)
                
            if H4_t > 0:
                results['entropy_increase_v4'].append(H4_next - H4_t)
            else:
                results['entropy_increase_v4'].append(H4_next)
            
            S_t = S_next
            
            # 打印演化信息
            print(f"Step {step+1}:")
            print(f"  状态数: {len(S_t)}")
            print(f"  熵V1: {H1_next:.4f}, 增量: {results['entropy_increase_v1'][-1]:.4f}")
            print(f"  熵V2: {H2_next:.4f}, 增量: {results['entropy_increase_v2'][-1]:.4f}")
            print(f"  熵V3: {H3_next:.4f}, 增量: {results['entropy_increase_v3'][-1]:.4f}")
            print(f"  熵V4: {H4_next:.4f}, 增量: {results['entropy_increase_v4'][-1]:.4f}")
            print(f"  新状态样例: {list(S_t)[:5]}")
            print()
        
        return results
    
    def analyze_entropy_bounds(self, results: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
        """分析熵增的范围"""
        bounds = {}
        
        for version in ['v1', 'v2', 'v3', 'v4']:
            increases = results[f'entropy_increase_{version}']
            if increases:
                min_increase = min(increases)
                max_increase = max(increases)
                avg_increase = sum(increases) / len(increases)
                
                # 不同版本使用不同上界
                if version == 'v4':
                    theoretical_max = math.log(self.phi)  # 自然对数
                else:
                    theoretical_max = math.log2(self.phi)  # 二进制对数
                
                bounds[version] = {
                    'min': min_increase,
                    'max': max_increase,
                    'avg': avg_increase,
                    'theoretical_max': theoretical_max
                }
                
                print(f"\n熵{version.upper()}增量分析:")
                print(f"  最小增量: {min_increase:.4f}")
                print(f"  最大增量: {max_increase:.4f}")
                print(f"  平均增量: {avg_increase:.4f}")
                print(f"  理论上界: {theoretical_max:.4f}")
                print(f"  是否满足上界: {max_increase <= theoretical_max + 0.001}")
        
        return bounds
    
    def print_evolution_summary(self, results: Dict[str, List[float]]):
        """打印演化结果摘要"""
        print("\n=== 演化结果摘要 ===")
        
        # 状态数增长
        print("\n状态数增长:")
        for i in range(0, len(results['sizes']), 5):
            print(f"  步骤 {i}: {results['sizes'][i]} 个状态")
        
        # 最终状态长度分布
        final_states = results['states'][-1]
        lengths = [len(s) for s in final_states]
        length_dist = {}
        for l in lengths:
            length_dist[l] = length_dist.get(l, 0) + 1
        
        print("\n最终状态长度分布:")
        for length in sorted(length_dist.keys()):
            print(f"  长度 {length}: {length_dist[length]} 个状态")
        
        # 熵演化对比
        print("\n熵演化对比:")
        print("步骤 | 熵V1    | 熵V2    | 熵V3    | 熵V4(ln) |")
        print("-" * 55)
        for i in range(0, len(results['entropy_v1']), 5):
            print(f"{i:4d} | {results['entropy_v1'][i]:7.4f} | {results['entropy_v2'][i]:7.4f} | {results['entropy_v3'][i]:7.4f} | {results['entropy_v4'][i]:8.4f} |")


def main():
    """主程序：验证熵公式"""
    print("=== 二进制自指完备系统熵公式验证 ===\n")
    
    # 创建系统
    system = BinarySystemEvolution()
    
    # 设置初始条件
    initial_states = {'0', '1'}  # 最简初始状态
    
    print(f"初始状态: {initial_states}")
    print(f"黄金比例 φ = {system.phi:.6f}")
    print(f"最小时间量子 ε = log₂(φ) = {system.epsilon:.6f}")
    print(f"理论熵增上界 (log₂) = log₂(φ) = {math.log2(system.phi):.6f}")
    print(f"理论熵增上界 (ln) = ln(φ) = {math.log(system.phi):.6f}\n")
    
    # 模拟演化
    print("开始模拟系统演化...\n")
    results = system.simulate_evolution(
        initial_states=initial_states,
        steps=20,
        num_observers=2
    )
    
    # 分析结果
    print("\n" + "="*50)
    bounds = system.analyze_entropy_bounds(results)
    
    # 验证哪个定义最合理
    print("\n=== 结论 ===")
    print("\n各熵定义的表现:")
    
    for version, bound in bounds.items():
        print(f"\n{version.upper()}:")
        print(f"  平均熵增: {bound['avg']:.4f}")
        print(f"  最大熵增: {bound['max']:.4f}")
        print(f"  违反上界: {'是' if bound['max'] > bound['theoretical_max'] else '否'}")
    
    # 打印结果摘要
    system.print_evolution_summary(results)
    
    # 额外测试：不同观察者数量
    print("\n\n=== 测试不同观察者数量的影响 ===")
    for num_obs in [1, 3, 5]:
        print(f"\n观察者数量: {num_obs}")
        results_obs = system.simulate_evolution(
            initial_states={'0', '1'},
            steps=10,
            num_observers=num_obs
        )
        bounds_obs = system.analyze_entropy_bounds(results_obs)


if __name__ == "__main__":
    main()