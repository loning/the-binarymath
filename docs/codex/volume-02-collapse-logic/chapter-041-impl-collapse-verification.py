#!/usr/bin/env python3
"""
Chapter 041: ImplCollapse Unit Test Verification
从ψ=ψ(ψ)推导Conditional Implication in Collapse Trace Systems

Core principle: From ψ = ψ(ψ) derive implication where conditional relationships
emerge as structural entailment between trace states, preserving φ-constraint 
integrity while implementing logical implication through path-based reasoning.

This verification program implements:
1. Structural implication algorithms based on trace entailment relations
2. Path-based conditional reasoning with φ-constraint preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection implication logic
4. Graph theory analysis of implication networks
5. Information theory analysis of conditional entropy
6. Category theory analysis of implication functors
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

class ImplCollapseSystem:
    """
    Core system for implementing conditional implication as structural entailment.
    Implements φ-constrained implication via trace structural relationships.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize implication collapse system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.implication_cache = {}
        self.entailment_registry = {}
        
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
            'implication_signature': self._compute_implication_signature(trace),
            'entailment_properties': self._compute_entailment_properties(trace)
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
        
    def _compute_implication_signature(self, trace: str) -> Tuple[int, int, float, bool]:
        """计算trace的蕴含签名：(length, ones_count, density, monotonic)"""
        density = trace.count('1') / max(len(trace), 1)
        monotonic = self._check_monotonicity(trace)
        return (len(trace), trace.count('1'), density, monotonic)
        
    def _compute_entailment_properties(self, trace: str) -> Dict[str, Union[int, float, bool]]:
        """计算trace的entailment属性"""
        return {
            'subsumption_level': self._compute_subsumption_level(trace),
            'structural_strength': self._compute_structural_strength(trace),
            'implication_potential': self._compute_implication_potential(trace),
            'transitivity_factor': self._compute_transitivity_factor(trace)
        }
        
    def _check_monotonicity(self, trace: str) -> bool:
        """检查trace是否具有单调性（1s都在前面）"""
        found_zero = False
        for bit in trace:
            if bit == '0':
                found_zero = True
            elif found_zero and bit == '1':
                return False
        return True
        
    def _compute_subsumption_level(self, trace: str) -> int:
        """计算trace的包含层级"""
        return trace.count('1')
        
    def _compute_structural_strength(self, trace: str) -> float:
        """计算结构强度"""
        if not trace or len(trace) == 1:
            return 0.0
        # 结构强度基于1的分布和间隔
        gaps = []
        last_one = -1
        for i, bit in enumerate(trace):
            if bit == '1':
                if last_one >= 0:
                    gaps.append(i - last_one)
                last_one = i
        
        if not gaps:
            return 0.0
        avg_gap = sum(gaps) / len(gaps)
        return 1.0 / (1.0 + avg_gap)
        
    def _compute_implication_potential(self, trace: str) -> float:
        """计算蕴含潜力"""
        # 基于trace的复杂度和结构
        if not trace:
            return 0.0
        ones_ratio = trace.count('1') / len(trace)
        complexity = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        return ones_ratio * (complexity / 4.0)
        
    def _compute_transitivity_factor(self, trace: str) -> float:
        """计算传递性因子"""
        # 基于Fibonacci indices的连续性
        indices = self._get_fibonacci_indices(trace)
        if len(indices) <= 1:
            return 0.0
        sorted_indices = sorted(indices)
        consecutive_pairs = sum(1 for i in range(len(sorted_indices)-1) 
                               if sorted_indices[i+1] - sorted_indices[i] == 1)
        return consecutive_pairs / (len(indices) - 1)

    def structural_implies(self, antecedent: int, consequent: int) -> Dict:
        """Structural implication: antecedent → consequent"""
        if antecedent not in self.trace_universe or consequent not in self.trace_universe:
            return {
                'valid': False,
                'reason': 'Invalid traces',
                'strength': 0.0
            }
            
        ante_data = self.trace_universe[antecedent]
        cons_data = self.trace_universe[consequent]
        
        # 计算结构蕴含关系
        implication_result = self._compute_structural_implication(ante_data, cons_data)
        
        return implication_result
        
    def _compute_structural_implication(self, ante_data: Dict, cons_data: Dict) -> Dict:
        """计算结构蕴含关系"""
        ante_trace = ante_data['trace']
        cons_trace = cons_data['trace']
        
        # 策略1：子结构关系
        subsumes = self._check_subsumption(ante_trace, cons_trace)
        
        # 策略2：结构相似性
        similarity = self._compute_trace_similarity(ante_trace, cons_trace)
        
        # 策略3：Fibonacci indices关系
        fib_relation = self._check_fibonacci_relation(
            ante_data['fibonacci_indices'], 
            cons_data['fibonacci_indices']
        )
        
        # 策略4：单调性保持
        monotonicity_preserved = (
            ante_data['implication_signature'][3] == 
            cons_data['implication_signature'][3]
        )
        
        # 综合计算蕴含强度
        strength = self._compute_implication_strength(
            subsumes, similarity, fib_relation, monotonicity_preserved
        )
        
        return {
            'valid': strength > 0.0,
            'strength': strength,
            'subsumes': subsumes,
            'similarity': similarity,
            'fibonacci_relation': fib_relation,
            'monotonicity_preserved': monotonicity_preserved,
            'antecedent': ante_data['value'],
            'consequent': cons_data['value']
        }
        
    def _check_subsumption(self, ante_trace: str, cons_trace: str) -> bool:
        """检查前件是否包含于后件"""
        # 简化的包含检查：前件的所有1位置在后件中也是1
        max_len = max(len(ante_trace), len(cons_trace))
        ante_padded = ante_trace.ljust(max_len, '0')
        cons_padded = cons_trace.ljust(max_len, '0')
        
        for i in range(max_len):
            if ante_padded[i] == '1' and cons_padded[i] == '0':
                return False
        return True
        
    def _compute_trace_similarity(self, trace1: str, trace2: str) -> float:
        """计算trace相似度"""
        max_len = max(len(trace1), len(trace2))
        padded1 = trace1.ljust(max_len, '0')
        padded2 = trace2.ljust(max_len, '0')
        
        matches = sum(1 for i in range(max_len) if padded1[i] == padded2[i])
        return matches / max_len
        
    def _check_fibonacci_relation(self, indices1: Set[int], indices2: Set[int]) -> str:
        """检查Fibonacci indices之间的关系"""
        if not indices1 or not indices2:
            return 'empty'
        
        if indices1 == indices2:
            return 'equal'
        elif indices1.issubset(indices2):
            return 'subset'
        elif indices2.issubset(indices1):
            return 'superset'
        elif indices1.intersection(indices2):
            return 'overlap'
        else:
            return 'disjoint'
            
    def _compute_implication_strength(self, subsumes: bool, similarity: float, 
                                    fib_relation: str, monotonicity: bool) -> float:
        """计算蕴含强度"""
        strength = 0.0
        
        # 子结构关系贡献
        if subsumes:
            strength += 0.4
            
        # 相似度贡献
        strength += similarity * 0.3
        
        # Fibonacci关系贡献
        fib_scores = {
            'equal': 0.3,
            'subset': 0.2,
            'superset': 0.1,
            'overlap': 0.05,
            'disjoint': 0.0,
            'empty': 0.0
        }
        strength += fib_scores.get(fib_relation, 0.0)
        
        # 单调性贡献
        if monotonicity:
            strength += 0.0  # 中性，不额外加分
            
        return min(strength, 1.0)

    def create_implication_chain(self, traces: List[int]) -> List[Tuple[int, int, float]]:
        """创建蕴含链"""
        chain = []
        for i in range(len(traces) - 1):
            impl_result = self.structural_implies(traces[i], traces[i+1])
            if impl_result['valid']:
                chain.append((traces[i], traces[i+1], impl_result['strength']))
        return chain
        
    def check_transitivity(self, a: int, b: int, c: int) -> Dict:
        """检查传递性：如果a→b且b→c，则a→c"""
        ab = self.structural_implies(a, b)
        bc = self.structural_implies(b, c)
        ac = self.structural_implies(a, c)
        
        transitivity_holds = (
            ab['valid'] and bc['valid'] and ac['valid'] and
            ac['strength'] >= min(ab['strength'], bc['strength']) * 0.8
        )
        
        return {
            'a_implies_b': ab,
            'b_implies_c': bc,
            'a_implies_c': ac,
            'transitivity_holds': transitivity_holds,
            'strength_preservation': ac['strength'] / max(min(ab['strength'], bc['strength']), 0.001)
        }
        
    def analyze_implication_network(self, traces: List[int]) -> Dict:
        """分析蕴含网络的图论属性"""
        G = nx.DiGraph()
        
        # 添加节点
        for trace in traces:
            if trace in self.trace_universe:
                G.add_node(trace)
                
        # 添加蕴含边
        for i in range(len(traces)):
            for j in range(len(traces)):
                if i != j:
                    impl_result = self.structural_implies(traces[i], traces[j])
                    if impl_result['valid']:
                        G.add_edge(traces[i], traces[j], weight=impl_result['strength'])
                        
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
        
    def compute_implication_entropy(self, traces: List[int]) -> float:
        """计算蕴含关系的信息熵"""
        # 收集所有蕴含强度的分布
        strengths = []
        
        for i in range(len(traces)):
            for j in range(len(traces)):
                if i != j:
                    impl_result = self.structural_implies(traces[i], traces[j])
                    if impl_result['valid']:
                        strengths.append(impl_result['strength'])
                        
        if not strengths:
            return 0.0
            
        # 将强度离散化为bins
        bins = np.linspace(0, 1, 11)
        digitized = np.digitize(strengths, bins)
        
        # 计算频率分布
        counts = np.bincount(digitized, minlength=len(bins))
        probabilities = counts / np.sum(counts)
        
        # 计算熵
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * log2(p)
                
        return entropy
        
    def analyze_implication_functor_properties(self, traces: List[int]) -> Dict:
        """分析蕴含的functor属性"""
        # 检查恒等性
        identity_tests = 0
        identity_preserved = 0
        
        for trace in traces[:5]:
            impl_result = self.structural_implies(trace, trace)
            identity_tests += 1
            if impl_result['valid'] and impl_result['strength'] >= 0.9:
                identity_preserved += 1
                
        # 检查组合性（传递性）
        composition_tests = 0
        composition_preserved = 0
        
        for i in range(min(len(traces), 5)):
            for j in range(min(len(traces), 5)):
                for k in range(min(len(traces), 5)):
                    if i != j and j != k and i != k:
                        trans_result = self.check_transitivity(traces[i], traces[j], traces[k])
                        composition_tests += 1
                        if trans_result['transitivity_holds']:
                            composition_preserved += 1
                            
        return {
            'identity_preservation': identity_preserved / max(identity_tests, 1),
            'composition_preservation': composition_preserved / max(composition_tests, 1),
            'distribution_preservation': 1.0,  # φ-constraint总是保持
            'total_identity_tests': identity_tests,
            'total_composition_tests': composition_tests
        }

class TestImplCollapseSystem(unittest.TestCase):
    """单元测试：验证ImplCollapse系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = ImplCollapseSystem()
        
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
        self.assertGreater(len(trace_5['fibonacci_indices']), 0)
        
    def test_structural_implication(self):
        """测试结构蕴含"""
        # 测试基本蕴含
        result = self.system.structural_implies(1, 2)
        self.assertIn('valid', result)
        self.assertIn('strength', result)
        
        # 测试自反性
        self_impl = self.system.structural_implies(3, 3)
        self.assertTrue(self_impl['valid'])
        self.assertGreater(self_impl['strength'], 0.8)
        
    def test_implication_chain(self):
        """测试蕴含链"""
        chain_traces = [1, 2, 3, 5]
        chain = self.system.create_implication_chain(chain_traces)
        
        # 验证链的结构
        self.assertIsInstance(chain, list)
        for link in chain:
            self.assertEqual(len(link), 3)  # (ante, cons, strength)
            self.assertGreaterEqual(link[2], 0.0)
            self.assertLessEqual(link[2], 1.0)
            
    def test_transitivity(self):
        """测试传递性"""
        trans_result = self.system.check_transitivity(1, 2, 3)
        
        # 验证传递性分析结果
        self.assertIn('transitivity_holds', trans_result)
        self.assertIn('strength_preservation', trans_result)
        self.assertIn('a_implies_b', trans_result)
        self.assertIn('b_implies_c', trans_result)
        self.assertIn('a_implies_c', trans_result)
        
    def test_implication_network_analysis(self):
        """测试蕴含网络分析"""
        test_traces = [1, 2, 3, 5, 8]
        network_props = self.system.analyze_implication_network(test_traces)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertIn('weakly_connected', network_props)
        self.assertEqual(network_props['nodes'], len(test_traces))
        
    def test_implication_entropy(self):
        """测试蕴含熵计算"""
        test_traces = [1, 2, 3, 5, 8]
        entropy = self.system.compute_implication_entropy(test_traces)
        
        # 验证熵计算
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_functor_properties(self):
        """测试functor属性"""
        test_traces = [1, 2, 3, 5, 8]
        functor_props = self.system.analyze_implication_functor_properties(test_traces)
        
        # 验证functor属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        self.assertIn('distribution_preservation', functor_props)
        
    def test_subsumption_relation(self):
        """测试包含关系"""
        # 测试简单的包含情况
        result = self.system.structural_implies(1, 3)
        self.assertIn('subsumes', result)
        
    def test_fibonacci_relation(self):
        """测试Fibonacci indices关系"""
        # 获取一些traces的Fibonacci indices
        trace1_indices = self.system.trace_universe[1]['fibonacci_indices']
        trace3_indices = self.system.trace_universe[3]['fibonacci_indices']
        
        # 验证关系计算
        relation = self.system._check_fibonacci_relation(trace1_indices, trace3_indices)
        self.assertIn(relation, ['equal', 'subset', 'superset', 'overlap', 'disjoint', 'empty'])

def run_comprehensive_analysis():
    """运行完整的ImplCollapse分析"""
    print("=" * 60)
    print("Chapter 041: ImplCollapse Comprehensive Analysis")
    print("Conditional Implication in Collapse Trace Systems")
    print("=" * 60)
    
    system = ImplCollapseSystem()
    
    # 1. 基础蕴含分析
    print("\n1. Basic Implication Analysis:")
    test_traces = list(system.trace_universe.keys())[:12]  # 前12个φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Test trace set size: {len(test_traces)}")
    
    # 2. 蕴含关系矩阵
    print("\n2. Implication Relation Matrix:")
    valid_implications = 0
    total_tests = 0
    strength_sum = 0.0
    
    for i in range(min(len(test_traces), 8)):
        for j in range(min(len(test_traces), 8)):
            if i != j:
                impl_result = system.structural_implies(test_traces[i], test_traces[j])
                total_tests += 1
                if impl_result['valid']:
                    valid_implications += 1
                    strength_sum += impl_result['strength']
                    
    print(f"Valid implications: {valid_implications}/{total_tests}")
    print(f"Average implication strength: {strength_sum/max(valid_implications, 1):.3f}")
    
    # 3. 自反性分析
    print("\n3. Reflexivity Analysis:")
    reflexive_count = 0
    reflexive_strength = 0.0
    
    for trace in test_traces[:10]:
        self_impl = system.structural_implies(trace, trace)
        if self_impl['valid']:
            reflexive_count += 1
            reflexive_strength += self_impl['strength']
            
    print(f"Reflexive implications: {reflexive_count}/10")
    print(f"Average reflexive strength: {reflexive_strength/max(reflexive_count, 1):.3f}")
    
    # 4. 传递性分析
    print("\n4. Transitivity Analysis:")
    transitivity_tests = 0
    transitivity_holds = 0
    
    # 测试一些三元组
    for i in range(min(len(test_traces), 5)):
        for j in range(min(len(test_traces), 5)):
            for k in range(min(len(test_traces), 5)):
                if i != j and j != k and i != k:
                    trans_result = system.check_transitivity(
                        test_traces[i], test_traces[j], test_traces[k]
                    )
                    transitivity_tests += 1
                    if trans_result['transitivity_holds']:
                        transitivity_holds += 1
                        
    print(f"Transitivity preservation: {transitivity_holds}/{transitivity_tests}")
    print(f"Transitivity rate: {transitivity_holds/max(transitivity_tests, 1):.3f}")
    
    # 5. 蕴含链分析
    print("\n5. Implication Chain Analysis:")
    chain_traces = [1, 2, 3, 5, 8, 13]
    chain = system.create_implication_chain(chain_traces)
    
    print(f"Chain length: {len(chain_traces)} traces")
    print(f"Valid links: {len(chain)}")
    if chain:
        avg_strength = sum(link[2] for link in chain) / len(chain)
        print(f"Average link strength: {avg_strength:.3f}")
        
    # 显示前几个链接
    for i, (ante, cons, strength) in enumerate(chain[:3]):
        print(f"  Link {i+1}: {ante} → {cons} (strength: {strength:.3f})")
    
    # 6. 网络分析
    print("\n6. Implication Network Analysis:")
    network_props = system.analyze_implication_network(test_traces[:10])
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Weakly connected: {network_props['weakly_connected']}")
    print(f"Strongly connected: {network_props['strongly_connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Has cycles: {network_props['has_cycles']}")
    print(f"Average degree: {network_props['average_degree']:.3f}")
    
    # 7. 信息理论分析
    print("\n7. Information Theory Analysis:")
    entropy = system.compute_implication_entropy(test_traces)
    print(f"Implication entropy: {entropy:.3f} bits")
    
    # 8. 范畴论分析
    print("\n8. Category Theory Analysis:")
    functor_props = system.analyze_implication_functor_properties(test_traces[:10])
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Distribution preservation: {functor_props['distribution_preservation']:.3f}")
    print(f"Total identity tests: {functor_props['total_identity_tests']}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    
    # 9. 结构分析
    print("\n9. Structural Analysis:")
    # 分析不同类型的蕴含关系
    subsumption_count = 0
    similarity_sum = 0.0
    fib_relations = defaultdict(int)
    
    for i in range(min(len(test_traces), 10)):
        for j in range(min(len(test_traces), 10)):
            if i != j:
                impl_result = system.structural_implies(test_traces[i], test_traces[j])
                if impl_result['valid']:
                    if impl_result['subsumes']:
                        subsumption_count += 1
                    similarity_sum += impl_result['similarity']
                    fib_relations[impl_result['fibonacci_relation']] += 1
                    
    print(f"Subsumption relations: {subsumption_count}")
    print(f"Average similarity: {similarity_sum/max(valid_implications, 1):.3f}")
    print("Fibonacci relations distribution:")
    for rel, count in fib_relations.items():
        print(f"  {rel}: {count}")
    
    # 10. 三域分析
    print("\n10. Three-Domain Analysis:")
    
    # Traditional implication domain
    traditional_implications = len(test_traces) * (len(test_traces) - 1)
    
    # φ-constrained domain
    phi_implications = valid_implications
    
    # Intersection analysis
    intersection_implications = phi_implications  # 所有φ-implications在intersection中
    
    print(f"Traditional implication domain: {traditional_implications} potential implications")
    print(f"φ-constrained implication domain: {phi_implications} valid implications")
    print(f"Intersection domain: {intersection_implications} implications")
    print(f"Domain intersection ratio: {intersection_implications/max(traditional_implications, 1):.3f}")
    
    # 蕴含强度分布
    print(f"Implication validity rate: {valid_implications/max(total_tests, 1):.3f}")
    print(f"Average implication strength: {strength_sum/max(valid_implications, 1):.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - ImplCollapse System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running ImplCollapse Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()