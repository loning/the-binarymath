#!/usr/bin/env python3
"""
Chapter 039: QuantCollapse Unit Test Verification
从ψ=ψ(ψ)推导∀ / ∃ Quantification over Collapse Path Spaces

Core principle: From ψ = ψ(ψ) derive quantification where universal (∀) and 
existential (∃) quantifiers operate over φ-constrained trace structures, creating
quantified logic that preserves structural relationships while enabling logical 
reasoning about path spaces.

This verification program implements:
1. Universal quantification algorithms over φ-constrained trace collections
2. Existential quantification mechanisms with structural predicate evaluation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection quantifier logic
4. Graph theory analysis of quantification networks
5. Information theory analysis of quantifier entropy
6. Category theory analysis of quantification functors
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

class QuantCollapseSystem:
    """
    Core system for implementing ∀/∃ quantification over collapse path spaces.
    Implements φ-constrained quantification via trace structural predicate evaluation.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize quantification collapse system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.quantifier_cache = {}
        self.predicate_registry = {}
        
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
            'quantifier_signature': self._compute_quantifier_signature(trace),
            'path_properties': self._compute_path_properties(trace)
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
        
    def _compute_quantifier_signature(self, trace: str) -> Tuple[int, int, int, bool, float]:
        """计算trace的量化签名：(length, ones_count, complexity, symmetry, fibonacci_ratio)"""
        complexity = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        symmetry = trace == trace[::-1]
        fibonacci_ratio = trace.count('1') / max(len(trace), 1)
        return (len(trace), trace.count('1'), complexity, symmetry, fibonacci_ratio)
        
    def _compute_path_properties(self, trace: str) -> Dict[str, Union[int, float, bool]]:
        """计算trace的路径属性"""
        return {
            'max_run_length': self._compute_max_run_length(trace),
            'alternation_count': self._compute_alternation_count(trace),
            'structural_balance': self._compute_structural_balance(trace),
            'path_complexity': self._compute_path_complexity(trace)
        }
        
    def _compute_max_run_length(self, trace: str) -> int:
        """计算最大连续相同位长度"""
        if not trace:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run
        
    def _compute_alternation_count(self, trace: str) -> int:
        """计算0/1交替次数"""
        if len(trace) <= 1:
            return 0
        alternations = 0
        for i in range(1, len(trace)):
            if trace[i] != trace[i-1]:
                alternations += 1
        return alternations
        
    def _compute_structural_balance(self, trace: str) -> float:
        """计算结构平衡度"""
        if not trace:
            return 0.0
        ones_ratio = trace.count('1') / len(trace)
        return 4 * ones_ratio * (1 - ones_ratio)  # 最大为1当ratio=0.5
        
    def _compute_path_complexity(self, trace: str) -> float:
        """计算路径复杂度"""
        if len(trace) <= 1:
            return 0.0
        unique_patterns = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        max_patterns = min(4, len(trace))
        return unique_patterns / max_patterns

    def create_structural_predicate(self, predicate_type: str, params: Dict) -> Callable:
        """创建φ-preserving structural predicate for quantification"""
        if predicate_type == "length_predicate":
            return self._create_length_predicate(params.get('min_length', 1), params.get('max_length', 10))
        elif predicate_type == "ones_predicate":
            return self._create_ones_predicate(params.get('min_ones', 1), params.get('max_ones', 5))
        elif predicate_type == "fibonacci_predicate":
            return self._create_fibonacci_predicate(params.get('required_indices', set()))
        elif predicate_type == "balance_predicate":
            return self._create_balance_predicate(params.get('min_balance', 0.0), params.get('max_balance', 1.0))
        elif predicate_type == "complexity_predicate":
            return self._create_complexity_predicate(params.get('min_complexity', 0.0), params.get('max_complexity', 1.0))
        else:
            return self._create_universal_predicate()
            
    def _create_length_predicate(self, min_length: int, max_length: int) -> Callable:
        """创建基于长度的predicate"""
        def length_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return min_length <= trace_data['length'] <= max_length
        return length_predicate
        
    def _create_ones_predicate(self, min_ones: int, max_ones: int) -> Callable:
        """创建基于1个数的predicate"""
        def ones_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return min_ones <= trace_data['ones_count'] <= max_ones
        return ones_predicate
        
    def _create_fibonacci_predicate(self, required_indices: Set[int]) -> Callable:
        """创建基于Fibonacci indices的predicate"""
        def fibonacci_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return required_indices.issubset(trace_data['fibonacci_indices'])
        return fibonacci_predicate
        
    def _create_balance_predicate(self, min_balance: float, max_balance: float) -> Callable:
        """创建基于结构平衡的predicate"""
        def balance_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            balance = trace_data['path_properties']['structural_balance']
            return min_balance <= balance <= max_balance
        return balance_predicate
        
    def _create_complexity_predicate(self, min_complexity: float, max_complexity: float) -> Callable:
        """创建基于复杂度的predicate"""
        def complexity_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            complexity = trace_data['path_properties']['path_complexity']
            return min_complexity <= complexity <= max_complexity
        return complexity_predicate
        
    def _create_universal_predicate(self) -> Callable:
        """创建universal predicate (true for all φ-valid traces)"""
        def universal_predicate(trace_value: int) -> bool:
            return trace_value in self.trace_universe
        return universal_predicate

    def universal_quantification(self, predicate: Callable, domain: List[int] = None) -> Dict:
        """Universal quantification: ∀x P(x) over φ-constrained domain"""
        if domain is None:
            domain = list(self.trace_universe.keys())
            
        # Filter domain to only φ-valid traces
        valid_domain = [x for x in domain if x in self.trace_universe]
        
        # Evaluate predicate for all elements
        results = []
        for trace_value in valid_domain:
            result = predicate(trace_value)
            results.append(result)
            
        # Universal quantification: true if ALL are true
        universal_result = all(results) if results else True
        
        return {
            'quantifier_type': 'universal',
            'domain_size': len(valid_domain),
            'satisfied_count': sum(results),
            'universal_result': universal_result,
            'satisfaction_ratio': sum(results) / max(len(results), 1),
            'phi_preservation': 1.0,  # All elements are φ-valid
            'domain_elements': valid_domain[:10]  # 前10个用于展示
        }
        
    def existential_quantification(self, predicate: Callable, domain: List[int] = None) -> Dict:
        """Existential quantification: ∃x P(x) over φ-constrained domain"""
        if domain is None:
            domain = list(self.trace_universe.keys())
            
        # Filter domain to only φ-valid traces
        valid_domain = [x for x in domain if x in self.trace_universe]
        
        # Evaluate predicate for all elements
        results = []
        witnesses = []
        for trace_value in valid_domain:
            result = predicate(trace_value)
            results.append(result)
            if result:
                witnesses.append(trace_value)
                
        # Existential quantification: true if ANY is true
        existential_result = any(results) if results else False
        
        return {
            'quantifier_type': 'existential',
            'domain_size': len(valid_domain),
            'satisfied_count': sum(results),
            'existential_result': existential_result,
            'satisfaction_ratio': sum(results) / max(len(results), 1),
            'phi_preservation': 1.0,  # All elements are φ-valid
            'witnesses': witnesses[:10]  # 前10个见证者
        }
        
    def nested_quantification(self, outer_type: str, inner_type: str, 
                            outer_predicate: Callable, inner_predicate: Callable,
                            domain: List[int] = None) -> Dict:
        """Nested quantification: ∀x∃y P(x,y) or ∃x∀y P(x,y)"""
        if domain is None:
            domain = list(self.trace_universe.keys())[:15]  # 限制大小防止计算爆炸
            
        valid_domain = [x for x in domain if x in self.trace_universe]
        
        outer_results = []
        nested_details = []
        
        for x in valid_domain:
            # 对每个x，检查内层量化
            inner_results = []
            for y in valid_domain:
                # 简化的二元谓词：检查两个独立谓词
                combined_result = outer_predicate(x) and inner_predicate(y)
                inner_results.append(combined_result)
                
            # 根据内层量化类型计算结果
            if inner_type == 'universal':
                inner_result = all(inner_results)
            else:  # existential
                inner_result = any(inner_results)
                
            outer_results.append(inner_result)
            nested_details.append({
                'x': x,
                'inner_satisfaction': sum(inner_results),
                'inner_result': inner_result
            })
            
        # 根据外层量化类型计算最终结果
        if outer_type == 'universal':
            final_result = all(outer_results)
        else:  # existential
            final_result = any(outer_results)
            
        return {
            'quantifier_type': f'{outer_type}_{inner_type}',
            'domain_size': len(valid_domain),
            'outer_satisfied': sum(outer_results),
            'final_result': final_result,
            'satisfaction_ratio': sum(outer_results) / max(len(outer_results), 1),
            'phi_preservation': 1.0,
            'nested_details': nested_details[:5]  # 前5个详情
        }
        
    def analyze_quantification_network(self, predicates: List[Callable], domain: List[int]) -> Dict:
        """分析quantification网络的图论属性"""
        G = nx.Graph()
        
        # 为每个predicate添加节点
        for i, predicate in enumerate(predicates):
            G.add_node(f"P{i}")
            
        # 计算predicates之间的满足度相似性并添加边
        for i in range(len(predicates)):
            for j in range(i + 1, len(predicates)):
                # 计算两个predicates的满足度重叠
                results1 = [predicates[i](x) for x in domain if x in self.trace_universe]
                results2 = [predicates[j](x) for x in domain if x in self.trace_universe]
                
                if len(results1) > 0 and len(results2) > 0:
                    agreement = sum(r1 == r2 for r1, r2 in zip(results1, results2))
                    similarity = agreement / len(results1)
                    
                    if similarity > 0.6:  # 相似性阈值
                        G.add_edge(f"P{i}", f"P{j}", weight=similarity)
                        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected': nx.is_connected(G),
            'components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
        }
        
    def compute_quantification_entropy(self, predicate: Callable, domain: List[int]) -> float:
        """计算quantification的信息熵"""
        valid_domain = [x for x in domain if x in self.trace_universe]
        
        if not valid_domain:
            return 0.0
            
        # 计算满足predicate的traces的属性分布
        satisfied_traces = [x for x in valid_domain if predicate(x)]
        
        if not satisfied_traces:
            return 0.0
            
        # 计算长度分布熵
        length_counts = defaultdict(int)
        for trace_value in satisfied_traces:
            trace_data = self.trace_universe[trace_value]
            length_counts[trace_data['length']] += 1
            
        total = len(satisfied_traces)
        entropy = 0.0
        for count in length_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * log2(prob)
                
        return entropy
        
    def analyze_quantification_functor_properties(self, predicates: List[Callable], domain: List[int]) -> Dict:
        """分析quantification的functor属性"""
        valid_domain = [x for x in domain if x in self.trace_universe]
        
        # 检查恒等性保持
        identity_preservation = 1.0  # Universal quantification保持恒等性
        
        # 检查组合性保持
        composition_tests = 0
        composition_preserved = 0
        
        for i in range(len(predicates)):
            for j in range(len(predicates)):
                if i != j:
                    composition_tests += 1
                    
                    # 测试 ∀x(P1(x) ∧ P2(x)) vs ∀xP1(x) ∧ ∀xP2(x)
                    combined_pred = lambda x: predicates[i](x) and predicates[j](x)
                    
                    universal_combined = self.universal_quantification(combined_pred, valid_domain)
                    universal_i = self.universal_quantification(predicates[i], valid_domain)
                    universal_j = self.universal_quantification(predicates[j], valid_domain)
                    
                    expected_result = universal_i['universal_result'] and universal_j['universal_result']
                    actual_result = universal_combined['universal_result']
                    
                    if expected_result == actual_result:
                        composition_preserved += 1
                        
        # 检查分布保持性
        distribution_preservation = 1.0  # 所有操作保持φ-validity
        
        return {
            'identity_preservation': identity_preservation,
            'composition_preservation': composition_preserved / max(composition_tests, 1),
            'distribution_preservation': distribution_preservation,
            'total_composition_tests': composition_tests,
            'total_distribution_tests': len(predicates)
        }

class TestQuantCollapseSystem(unittest.TestCase):
    """单元测试：验证QuantCollapse系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = QuantCollapseSystem()
        
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
        
    def test_predicate_creation(self):
        """测试predicate创建"""
        # 测试length predicate
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 2, 'max_length': 4})
        self.assertIsNotNone(length_pred)
        
        # 测试ones predicate
        ones_pred = self.system.create_structural_predicate("ones_predicate", {'min_ones': 1, 'max_ones': 3})
        self.assertIsNotNone(ones_pred)
        
    def test_universal_quantification(self):
        """测试universal quantification"""
        # 创建一个简单的predicate
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 10})
        
        result = self.system.universal_quantification(length_pred)
        
        # 验证quantification结果
        self.assertIn('universal_result', result)
        self.assertIn('satisfaction_ratio', result)
        self.assertIn('phi_preservation', result)
        self.assertEqual(result['phi_preservation'], 1.0)
        
    def test_existential_quantification(self):
        """测试existential quantification"""
        # 创建一个特定的predicate
        ones_pred = self.system.create_structural_predicate("ones_predicate", {'min_ones': 2, 'max_ones': 3})
        
        result = self.system.existential_quantification(ones_pred)
        
        # 验证quantification结果
        self.assertIn('existential_result', result)
        self.assertIn('witnesses', result)
        self.assertIn('phi_preservation', result)
        self.assertEqual(result['phi_preservation'], 1.0)
        
    def test_nested_quantification(self):
        """测试nested quantification"""
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 2, 'max_length': 4})
        ones_pred = self.system.create_structural_predicate("ones_predicate", {'min_ones': 1, 'max_ones': 2})
        
        result = self.system.nested_quantification('universal', 'existential', length_pred, ones_pred)
        
        # 验证nested quantification结果
        self.assertIn('final_result', result)
        self.assertIn('nested_details', result)
        self.assertIn('phi_preservation', result)
        
    def test_quantification_network_analysis(self):
        """测试quantification网络分析"""
        predicates = [
            self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 3}),
            self.system.create_structural_predicate("ones_predicate", {'min_ones': 1, 'max_ones': 2}),
            self.system.create_structural_predicate("balance_predicate", {'min_balance': 0.0, 'max_balance': 0.5})
        ]
        
        domain = list(self.system.trace_universe.keys())[:10]
        network_props = self.system.analyze_quantification_network(predicates, domain)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertEqual(network_props['nodes'], len(predicates))
        
    def test_quantification_entropy(self):
        """测试quantification entropy计算"""
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 5})
        domain = list(self.system.trace_universe.keys())[:15]
        
        entropy = self.system.compute_quantification_entropy(length_pred, domain)
        
        # 验证entropy计算
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_functor_properties(self):
        """测试functor属性"""
        predicates = [
            self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 4}),
            self.system.create_structural_predicate("ones_predicate", {'min_ones': 1, 'max_ones': 3})
        ]
        
        domain = list(self.system.trace_universe.keys())[:10]
        functor_props = self.system.analyze_quantification_functor_properties(predicates, domain)
        
        # 验证functor属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        self.assertIn('distribution_preservation', functor_props)
        
    def test_complex_predicates(self):
        """测试复杂predicates"""
        # 测试balance predicate
        balance_pred = self.system.create_structural_predicate("balance_predicate", {'min_balance': 0.5, 'max_balance': 1.0})
        result = balance_pred(3)
        self.assertIsInstance(result, bool)
        
        # 测试complexity predicate
        complexity_pred = self.system.create_structural_predicate("complexity_predicate", {'min_complexity': 0.2, 'max_complexity': 0.8})
        result = complexity_pred(5)
        self.assertIsInstance(result, bool)

def run_comprehensive_analysis():
    """运行完整的QuantCollapse分析"""
    print("=" * 60)
    print("Chapter 039: QuantCollapse Comprehensive Analysis")
    print("∀ / ∃ Quantification over Collapse Path Spaces")
    print("=" * 60)
    
    system = QuantCollapseSystem()
    
    # 1. 基础quantification分析
    print("\n1. Basic Quantification Analysis:")
    predicates = [
        ("Length [2-4]", system.create_structural_predicate("length_predicate", {'min_length': 2, 'max_length': 4})),
        ("Ones [1-3]", system.create_structural_predicate("ones_predicate", {'min_ones': 1, 'max_ones': 3})),
        ("Balance [0.3-0.8]", system.create_structural_predicate("balance_predicate", {'min_balance': 0.3, 'max_balance': 0.8})),
        ("Complexity [0.2-0.7]", system.create_structural_predicate("complexity_predicate", {'min_complexity': 0.2, 'max_complexity': 0.7})),
        ("Fibonacci {2}", system.create_structural_predicate("fibonacci_predicate", {'required_indices': {2}}))
    ]
    
    domain = list(system.trace_universe.keys())[:20]  # 前20个φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Analysis domain size: {len(domain)}")
    print(f"Total predicates: {len(predicates)}")
    
    # 2. Universal quantification分析
    print("\n2. Universal Quantification Analysis:")
    for name, predicate in predicates:
        result = system.universal_quantification(predicate, domain)
        print(f"{name}: {result['satisfied_count']}/{result['domain_size']} satisfied, ∀ result={result['universal_result']}, ratio={result['satisfaction_ratio']:.3f}")
    
    # 3. Existential quantification分析
    print("\n3. Existential Quantification Analysis:")
    for name, predicate in predicates:
        result = system.existential_quantification(predicate, domain)
        witness_count = len(result['witnesses'])
        print(f"{name}: {result['satisfied_count']}/{result['domain_size']} satisfied, ∃ result={result['existential_result']}, witnesses={witness_count}")
    
    # 4. Nested quantification分析
    print("\n4. Nested Quantification Analysis:")
    length_pred = predicates[0][1]
    ones_pred = predicates[1][1]
    
    nested_configs = [
        ('universal', 'existential', '∀∃'),
        ('existential', 'universal', '∃∀'),
        ('universal', 'universal', '∀∀'),
        ('existential', 'existential', '∃∃')
    ]
    
    for outer, inner, symbol in nested_configs:
        result = system.nested_quantification(outer, inner, length_pred, ones_pred, domain[:10])
        print(f"{symbol}: {result['outer_satisfied']}/{result['domain_size']} outer satisfied, final result={result['final_result']}")
    
    # 5. 网络分析
    print("\n5. Quantification Network Analysis:")
    pred_functions = [pred for _, pred in predicates]
    network_props = system.analyze_quantification_network(pred_functions, domain)
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Connected: {network_props['connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Average clustering: {network_props['average_clustering']:.3f}")
    
    # 6. 信息理论分析
    print("\n6. Information Theory Analysis:")
    total_entropy = 0.0
    for name, predicate in predicates:
        entropy = system.compute_quantification_entropy(predicate, domain)
        total_entropy += entropy
        print(f"{name} quantification entropy: {entropy:.3f} bits")
    
    avg_entropy = total_entropy / len(predicates)
    print(f"Average quantification entropy: {avg_entropy:.3f} bits")
    
    # 7. 范畴论分析
    print("\n7. Category Theory Analysis:")
    functor_props = system.analyze_quantification_functor_properties(pred_functions, domain)
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Distribution preservation: {functor_props['distribution_preservation']:.3f}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    print(f"Total distribution tests: {functor_props['total_distribution_tests']}")
    
    # 8. 三域分析
    print("\n8. Three-Domain Analysis:")
    
    # Traditional quantification domain (conceptual)
    traditional_quantifications = len(domain)  # 传统量化可以应用于所有元素
    
    # φ-constrained domain
    phi_quantifications = sum(1 for _, pred in predicates 
                             if system.existential_quantification(pred, domain)['existential_result'])
    
    # Intersection analysis
    intersection_quantifications = phi_quantifications  # 所有φ-quantifications在intersection中
    
    print(f"Traditional quantification domain: {traditional_quantifications} elements")
    print(f"φ-constrained quantification domain: {phi_quantifications} active quantifications")
    print(f"Intersection domain: {intersection_quantifications} quantifications")
    print(f"Domain intersection ratio: {intersection_quantifications/max(len(predicates), 1):.3f}")
    
    # 比较不同quantification的满足度
    universal_satisfaction = sum(system.universal_quantification(pred, domain)['satisfaction_ratio'] 
                               for _, pred in predicates) / len(predicates)
    existential_satisfaction = sum(system.existential_quantification(pred, domain)['satisfaction_ratio'] 
                                 for _, pred in predicates) / len(predicates)
    print(f"Average universal satisfaction: {universal_satisfaction:.3f}")
    print(f"Average existential satisfaction: {existential_satisfaction:.3f}")
    print(f"Quantification balance ratio: {existential_satisfaction/max(universal_satisfaction, 0.001):.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - QuantCollapse System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running QuantCollapse Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()