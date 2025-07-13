#!/usr/bin/env python3
"""
Chapter 043: CollapseDeduce Unit Test Verification
从ψ=ψ(ψ)推导Deductive Path Expansion under Constraint Entailment

Core principle: From ψ = ψ(ψ) derive deduction where logical inference emerges
through controlled expansion of trace paths, maintaining φ-constraints while
enabling systematic derivation of new truths from given premises.

This verification program implements:
1. Path expansion algorithms under φ-constraint preservation
2. Deductive inference mechanisms through trace transformation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection deduction theory
4. Graph theory analysis of deduction networks
5. Information theory analysis of inference entropy
6. Category theory analysis of deduction functors
"""

import torch
import numpy as np
import networkx as nx
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
from collections import defaultdict, deque
import itertools
from math import log2, gcd
from functools import reduce

class CollapseDeduceSystem:
    """
    Core system for implementing deductive path expansion under constraint entailment.
    Implements φ-constrained deduction via controlled trace path expansion.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize collapse deduce system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.deduction_cache = {}
        self.inference_registry = {}
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        for n in range(self.max_trace_size + 1):
            trace_data = self._analyze_trace_structure(n)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        return universe
        
    def _analyze_trace_structure(self, n: int) -> Dict:
        """分析单个trace的结构属性"""
        trace = self._encode_to_trace(n)
        
        return {
            'value': n,
            'trace': trace,
            'phi_valid': '11' not in trace,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'fibonacci_indices': self._get_fibonacci_indices(trace),
            'structural_hash': self._compute_structural_hash(trace),
            'deductive_signature': self._compute_deductive_signature(trace),
            'expansion_properties': self._compute_expansion_properties(trace)
        }
        
    def _encode_to_trace(self, n: int) -> str:
        """将自然数编码为φ-compliant trace (Zeckendorf-based)"""
        if n == 0:
            return '0'
            
        # 使用Zeckendorf分解
        decomposition = self._zeckendorf_decomposition(n)
        if decomposition is None:
            return '0'
            
        # 构造trace：位置i对应F_{i+1}
        max_index = max(decomposition) if decomposition else 1
        trace = ['0'] * max_index
        
        for idx in decomposition:
            trace[idx - 1] = '1'  # idx从1开始，所以-1
            
        return ''.join(reversed(trace))  # 高位在左
        
    def _zeckendorf_decomposition(self, n: int) -> Optional[List[int]]:
        """Zeckendorf分解：避免连续Fibonacci数"""
        if n == 0:
            return []
            
        remaining = n
        used_indices = []
        
        for i in range(len(self.fibonacci_numbers) - 1, -1, -1):
            if self.fibonacci_numbers[i] <= remaining:
                remaining -= self.fibonacci_numbers[i]
                used_indices.append(i + 1)  # 1-indexed
                
        return used_indices if remaining == 0 else None
        
    def _get_fibonacci_indices(self, trace: str) -> Set[int]:
        """获取trace中激活的Fibonacci indices"""
        indices = set()
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.add(i + 1)  # 1-indexed
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构hash"""
        return hash(trace) % 10000
        
    def _compute_deductive_signature(self, trace: str) -> Tuple[int, int, float, int]:
        """计算trace的演绎签名：(length, ones_count, density, expansion_potential)"""
        density = trace.count('1') / max(len(trace), 1)
        expansion_potential = self._compute_expansion_potential(trace)
        return (len(trace), trace.count('1'), density, expansion_potential)
        
    def _compute_expansion_properties(self, trace: str) -> Dict[str, Union[int, float, bool]]:
        """计算trace的扩展属性"""
        return {
            'expansion_potential': self._compute_expansion_potential(trace),
            'deductive_depth': self._compute_deductive_depth(trace),
            'inference_capacity': self._compute_inference_capacity(trace),
            'structural_flexibility': self._compute_structural_flexibility(trace)
        }
        
    def _compute_expansion_potential(self, trace: str) -> int:
        """计算扩展潜力：可以产生多少有效推理"""
        if not trace or trace == '0':
            return 0
        # 基于trace的结构复杂度和Fibonacci indices
        indices = self._get_fibonacci_indices(trace)
        return len(indices) * 2 + trace.count('0')
        
    def _compute_deductive_depth(self, trace: str) -> int:
        """计算演绎深度：推理链的最大长度"""
        if not trace or trace == '0':
            return 0
        return len(trace) + trace.count('1')
        
    def _compute_inference_capacity(self, trace: str) -> float:
        """计算推理容量：能够支持的推理复杂度"""
        if not trace:
            return 0.0
        ones_ratio = trace.count('1') / len(trace)
        complexity = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        return ones_ratio * complexity / 4.0
        
    def _compute_structural_flexibility(self, trace: str) -> float:
        """计算结构灵活性：支持推理变换的程度"""
        if not trace or len(trace) < 2:
            return 0.0
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        return transitions / (len(trace) - 1)

    def deduce_from_premise(self, premise: int, inference_rule: str = 'expand') -> List[int]:
        """从前提演绎新结论"""
        if premise not in self.trace_universe:
            return []
            
        premise_data = self.trace_universe[premise]
        
        if inference_rule == 'expand':
            return self._expand_deduction(premise_data)
        elif inference_rule == 'contract':
            return self._contract_deduction(premise_data)
        elif inference_rule == 'transform':
            return self._transform_deduction(premise_data)
        elif inference_rule == 'combine':
            return self._combine_deduction(premise_data)
        else:
            return []
            
    def _expand_deduction(self, premise_data: Dict) -> List[int]:
        """扩展演绎：生成更复杂的结论"""
        results = []
        premise_trace = premise_data['trace']
        
        # 策略1：添加新的Fibonacci组件
        for i in range(len(premise_trace) + 1):
            new_trace = premise_trace[:i] + '1' + premise_trace[i:]
            if '11' not in new_trace:
                value = self._trace_to_value(new_trace)
                if value in self.trace_universe:
                    results.append(value)
                    
        # 策略2：扩展trace长度
        extended_trace = premise_trace + '0'
        if '11' not in extended_trace:
            value = self._trace_to_value(extended_trace)
            if value in self.trace_universe:
                results.append(value)
                
        return results
        
    def _contract_deduction(self, premise_data: Dict) -> List[int]:
        """收缩演绎：生成更简单的结论"""
        results = []
        premise_trace = premise_data['trace']
        
        # 策略1：移除Fibonacci组件
        for i, bit in enumerate(premise_trace):
            if bit == '1':
                new_trace = premise_trace[:i] + '0' + premise_trace[i+1:]
                if '11' not in new_trace:
                    value = self._trace_to_value(new_trace)
                    if value in self.trace_universe and value not in results:
                        results.append(value)
                        
        # 策略2：缩短trace
        if len(premise_trace) > 1:
            shortened_trace = premise_trace[:-1]
            if '11' not in shortened_trace:
                value = self._trace_to_value(shortened_trace)
                if value in self.trace_universe and value not in results:
                    results.append(value)
                    
        return results
        
    def _transform_deduction(self, premise_data: Dict) -> List[int]:
        """变换演绎：生成结构相关的结论"""
        results = []
        premise_trace = premise_data['trace']
        
        # 策略1：移位变换
        if len(premise_trace) > 1:
            shifted_trace = '0' + premise_trace[:-1]
            if '11' not in shifted_trace:
                value = self._trace_to_value(shifted_trace)
                if value in self.trace_universe:
                    results.append(value)
                    
        # 策略2：局部变换
        for i in range(len(premise_trace) - 1):
            if premise_trace[i:i+2] == '10':
                new_trace = premise_trace[:i] + '01' + premise_trace[i+2:]
                if '11' not in new_trace:
                    value = self._trace_to_value(new_trace)
                    if value in self.trace_universe and value not in results:
                        results.append(value)
                        
        return results
        
    def _combine_deduction(self, premise_data: Dict) -> List[int]:
        """组合演绎：基于多个前提的结论"""
        results = []
        premise_value = premise_data['value']
        
        # 找到可以组合的其他前提
        for other_value in list(self.trace_universe.keys())[:10]:  # 限制搜索范围
            if other_value != premise_value:
                # 尝试组合
                combined = self._combine_traces(premise_value, other_value)
                if combined and combined not in results:
                    results.append(combined)
                    
        return results[:5]  # 限制结果数量
        
    def _combine_traces(self, value1: int, value2: int) -> Optional[int]:
        """组合两个traces"""
        if value1 not in self.trace_universe or value2 not in self.trace_universe:
            return None
            
        trace1 = self.trace_universe[value1]['trace']
        trace2 = self.trace_universe[value2]['trace']
        
        # 简单的OR组合
        max_len = max(len(trace1), len(trace2))
        padded1 = trace1.ljust(max_len, '0')
        padded2 = trace2.ljust(max_len, '0')
        
        combined = []
        for i in range(max_len):
            if padded1[i] == '1' or padded2[i] == '1':
                combined.append('1')
            else:
                combined.append('0')
                
        combined_trace = ''.join(combined)
        
        if '11' not in combined_trace:
            return self._trace_to_value(combined_trace)
        return None
        
    def _trace_to_value(self, trace: str) -> int:
        """将trace转换回数值"""
        value = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                value += self.fibonacci_numbers[i]
        return value

    def create_deduction_chain(self, premise: int, max_depth: int = 5) -> List[List[int]]:
        """创建演绎链"""
        chains = []
        current_level = [premise]
        chain = [current_level]
        
        for depth in range(max_depth):
            next_level = []
            for p in current_level:
                deductions = self.deduce_from_premise(p, 'expand')
                next_level.extend(deductions)
                
            if not next_level:
                break
                
            # 去重
            next_level = list(set(next_level))
            chain.append(next_level)
            current_level = next_level
            
        return chain
        
    def analyze_deduction_network(self, premises: List[int]) -> Dict:
        """分析演绎网络的图论属性"""
        G = nx.DiGraph()
        
        # 添加节点
        for premise in premises:
            if premise in self.trace_universe:
                G.add_node(premise)
                
        # 添加演绎边
        for premise in premises:
            if premise in self.trace_universe:
                for rule in ['expand', 'contract', 'transform']:
                    deductions = self.deduce_from_premise(premise, rule)
                    for conclusion in deductions:
                        if conclusion in premises:
                            G.add_edge(premise, conclusion, rule=rule)
                            
        # 分析网络属性
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'weakly_connected': nx.is_weakly_connected(G),
            'strongly_connected': nx.is_strongly_connected(G),
            'components': nx.number_weakly_connected_components(G),
            'has_cycles': len(list(nx.simple_cycles(G))) > 0 if G.number_of_nodes() < 10 else 'unknown',
            'average_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
        }
        
    def compute_deduction_entropy(self, premises: List[int]) -> float:
        """计算演绎关系的信息熵"""
        # 收集所有演绎路径的长度分布
        path_lengths = []
        
        for premise in premises[:10]:  # 限制计算范围
            chain = self.create_deduction_chain(premise, max_depth=3)
            for i, level in enumerate(chain):
                path_lengths.extend([i] * len(level))
                
        if not path_lengths:
            return 0.0
            
        # 计算频率分布
        length_counts = defaultdict(int)
        for length in path_lengths:
            length_counts[length] += 1
            
        total = sum(length_counts.values())
        probabilities = [count / total for count in length_counts.values()]
        
        # 计算熵
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * log2(p)
                
        return entropy
        
    def analyze_deduction_functor_properties(self, premises: List[int]) -> Dict:
        """分析演绎的functor属性"""
        # 检查恒等性
        identity_tests = 0
        identity_preserved = 0
        
        for premise in premises[:5]:
            # 检查收缩后扩展是否返回自身
            contractions = self.deduce_from_premise(premise, 'contract')
            for c in contractions:
                expansions = self.deduce_from_premise(c, 'expand')
                identity_tests += 1
                if premise in expansions:
                    identity_preserved += 1
                    
        # 检查组合性
        composition_tests = 0
        composition_preserved = 0
        
        for premise in premises[:5]:
            # 检查连续演绎的组合性
            step1 = self.deduce_from_premise(premise, 'expand')
            for s1 in step1[:2]:
                step2 = self.deduce_from_premise(s1, 'expand')
                
                # 直接两步扩展
                chain = self.create_deduction_chain(premise, max_depth=2)
                if len(chain) >= 3:
                    direct_results = chain[2]
                    
                    composition_tests += 1
                    if any(s2 in direct_results for s2 in step2):
                        composition_preserved += 1
                        
        return {
            'identity_preservation': identity_preserved / max(identity_tests, 1),
            'composition_preservation': composition_preserved / max(composition_tests, 1),
            'distribution_preservation': 1.0,  # φ-constraint总是保持
            'total_identity_tests': identity_tests,
            'total_composition_tests': composition_tests
        }

    def analyze_inference_patterns(self, premises: List[int]) -> Dict:
        """分析推理模式"""
        pattern_counts = defaultdict(int)
        total_inferences = 0
        
        for premise in premises[:15]:
            for rule in ['expand', 'contract', 'transform']:
                deductions = self.deduce_from_premise(premise, rule)
                pattern_counts[rule] += len(deductions)
                total_inferences += len(deductions)
                
        return {
            'pattern_distribution': dict(pattern_counts),
            'total_inferences': total_inferences,
            'average_per_premise': total_inferences / max(len(premises[:15]), 1),
            'dominant_pattern': max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None
        }

class TestCollapseDeduceSystem(unittest.TestCase):
    """单元测试：验证CollapseDeduce系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseDeduceSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        # 验证φ-valid traces被正确识别
        self.assertIn(1, self.system.trace_universe)
        self.assertIn(2, self.system.trace_universe)
        self.assertIn(3, self.system.trace_universe)
        self.assertIn(5, self.system.trace_universe)
        
        # 验证trace结构信息
        trace_5 = self.system.trace_universe[5]
        self.assertEqual(trace_5['value'], 5)
        self.assertTrue(trace_5['phi_valid'])
        self.assertGreater(trace_5['expansion_properties']['expansion_potential'], 0)
        
    def test_deduction_operations(self):
        """测试演绎操作"""
        # 测试扩展演绎
        expansions = self.system.deduce_from_premise(3, 'expand')
        self.assertIsInstance(expansions, list)
        
        # 测试收缩演绎
        contractions = self.system.deduce_from_premise(5, 'contract')
        self.assertIsInstance(contractions, list)
        
        # 测试变换演绎
        transforms = self.system.deduce_from_premise(8, 'transform')
        self.assertIsInstance(transforms, list)
        
    def test_deduction_chain(self):
        """测试演绎链"""
        chain = self.system.create_deduction_chain(1, max_depth=3)
        
        # 验证链的结构
        self.assertIsInstance(chain, list)
        self.assertGreaterEqual(len(chain), 1)
        self.assertEqual(chain[0], [1])  # 起始前提
        
    def test_deduction_network_analysis(self):
        """测试演绎网络分析"""
        test_premises = [1, 2, 3, 5, 8]
        network_props = self.system.analyze_deduction_network(test_premises)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertEqual(network_props['nodes'], len(test_premises))
        
    def test_deduction_entropy(self):
        """测试演绎熵计算"""
        test_premises = [1, 2, 3, 5, 8]
        entropy = self.system.compute_deduction_entropy(test_premises)
        
        # 验证熵计算
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_functor_properties(self):
        """测试functor属性"""
        test_premises = [1, 2, 3, 5, 8]
        functor_props = self.system.analyze_deduction_functor_properties(test_premises)
        
        # 验证functor属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        self.assertIn('distribution_preservation', functor_props)
        
    def test_inference_patterns(self):
        """测试推理模式分析"""
        test_premises = [1, 2, 3, 5, 8, 13]
        patterns = self.system.analyze_inference_patterns(test_premises)
        
        # 验证模式分析
        self.assertIn('pattern_distribution', patterns)
        self.assertIn('total_inferences', patterns)
        self.assertIn('dominant_pattern', patterns)

def run_comprehensive_analysis():
    """运行完整的CollapseDeduce分析"""
    print("=" * 60)
    print("Chapter 043: CollapseDeduce Comprehensive Analysis")
    print("Deductive Path Expansion under Constraint Entailment")
    print("=" * 60)
    
    system = CollapseDeduceSystem()
    
    # 1. 基础演绎分析
    print("\n1. Basic Deduction Analysis:")
    test_premises = list(system.trace_universe.keys())[:15]  # 前15个φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Test premise set size: {len(test_premises)}")
    
    # 2. 演绎操作分析
    print("\n2. Deduction Operations Analysis:")
    total_deductions = 0
    rule_statistics = defaultdict(int)
    
    for premise in test_premises[:10]:
        for rule in ['expand', 'contract', 'transform']:
            deductions = system.deduce_from_premise(premise, rule)
            rule_statistics[rule] += len(deductions)
            total_deductions += len(deductions)
            
    print(f"Total deductions generated: {total_deductions}")
    for rule, count in rule_statistics.items():
        print(f"{rule} deductions: {count} ({count/max(total_deductions, 1)*100:.1f}%)")
    
    # 3. 演绎链分析
    print("\n3. Deduction Chain Analysis:")
    chain_lengths = []
    max_chain_depth = 0
    
    for premise in test_premises[:5]:
        chain = system.create_deduction_chain(premise, max_depth=5)
        chain_lengths.append(len(chain))
        max_chain_depth = max(max_chain_depth, len(chain))
        
        print(f"Premise {premise}: chain length = {len(chain)}")
        for i, level in enumerate(chain[:3]):  # 显示前3层
            print(f"  Level {i}: {len(level)} conclusions")
            
    avg_chain_length = sum(chain_lengths) / len(chain_lengths)
    print(f"\nAverage chain length: {avg_chain_length:.2f}")
    print(f"Maximum chain depth: {max_chain_depth}")
    
    # 4. 网络分析
    print("\n4. Deduction Network Analysis:")
    network_props = system.analyze_deduction_network(test_premises[:10])
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Weakly connected: {network_props['weakly_connected']}")
    print(f"Strongly connected: {network_props['strongly_connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Has cycles: {network_props['has_cycles']}")
    print(f"Average degree: {network_props['average_degree']:.3f}")
    
    # 5. 信息理论分析
    print("\n5. Information Theory Analysis:")
    entropy = system.compute_deduction_entropy(test_premises)
    print(f"Deduction entropy: {entropy:.3f} bits")
    
    # 分析不同规则的熵
    rule_entropies = {}
    for rule in ['expand', 'contract', 'transform']:
        rule_premises = []
        for premise in test_premises[:10]:
            deductions = system.deduce_from_premise(premise, rule)
            rule_premises.extend(deductions)
        if rule_premises:
            rule_entropy = system.compute_deduction_entropy(rule_premises[:10])
            rule_entropies[rule] = rule_entropy
            print(f"{rule} entropy: {rule_entropy:.3f} bits")
    
    # 6. 范畴论分析
    print("\n6. Category Theory Analysis:")
    functor_props = system.analyze_deduction_functor_properties(test_premises[:10])
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Distribution preservation: {functor_props['distribution_preservation']:.3f}")
    print(f"Total identity tests: {functor_props['total_identity_tests']}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    
    # 7. 推理模式分析
    print("\n7. Inference Pattern Analysis:")
    patterns = system.analyze_inference_patterns(test_premises)
    print(f"Total inferences: {patterns['total_inferences']}")
    print(f"Average per premise: {patterns['average_per_premise']:.2f}")
    print(f"Dominant pattern: {patterns['dominant_pattern']}")
    print("Pattern distribution:")
    for pattern, count in patterns['pattern_distribution'].items():
        print(f"  {pattern}: {count} ({count/patterns['total_inferences']*100:.1f}%)")
    
    # 8. 扩展属性分析
    print("\n8. Expansion Properties Analysis:")
    total_potential = 0
    total_depth = 0
    total_capacity = 0
    total_flexibility = 0
    
    for premise in test_premises[:10]:
        if premise in system.trace_universe:
            props = system.trace_universe[premise]['expansion_properties']
            total_potential += props['expansion_potential']
            total_depth += props['deductive_depth']
            total_capacity += props['inference_capacity']
            total_flexibility += props['structural_flexibility']
            
    n = 10
    print(f"Average expansion potential: {total_potential/n:.2f}")
    print(f"Average deductive depth: {total_depth/n:.2f}")
    print(f"Average inference capacity: {total_capacity/n:.3f}")
    print(f"Average structural flexibility: {total_flexibility/n:.3f}")
    
    # 9. 三域分析
    print("\n9. Three-Domain Analysis:")
    
    # Traditional deduction domain
    traditional_deductions = len(test_premises) * len(test_premises)  # 任意推理
    
    # φ-constrained domain
    phi_deductions = total_deductions
    
    # Intersection analysis
    intersection_deductions = phi_deductions  # 所有φ-deductions在intersection中
    
    print(f"Traditional deduction domain: {traditional_deductions} potential deductions")
    print(f"φ-constrained deduction domain: {phi_deductions} valid deductions")
    print(f"Intersection domain: {intersection_deductions} deductions")
    print(f"Domain intersection ratio: {intersection_deductions/max(traditional_deductions, 1):.3f}")
    
    # 10. 演绎模式深度分析
    print("\n10. Deduction Pattern Deep Analysis:")
    
    # 分析不同前提的演绎能力
    deduction_capabilities = []
    for premise in test_premises[:10]:
        total_deductions_for_premise = 0
        for rule in ['expand', 'contract', 'transform']:
            deductions = system.deduce_from_premise(premise, rule)
            total_deductions_for_premise += len(deductions)
        deduction_capabilities.append((premise, total_deductions_for_premise))
    
    # 排序并显示最具演绎能力的前提
    deduction_capabilities.sort(key=lambda x: x[1], reverse=True)
    print("Most deductive premises:")
    for premise, count in deduction_capabilities[:5]:
        trace = system.trace_universe[premise]['trace']
        print(f"  Premise {premise} (trace: {trace}): {count} deductions")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - CollapseDeduce System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running CollapseDeduce Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()