#!/usr/bin/env python3
"""
Chapter 037: PredTrace Unit Test Verification
从ψ=ψ(ψ)推导Logical Predicate as φ-Constrained Structural Selector

Core principle: From ψ = ψ(ψ) derive predicate logic where logical predicates 
emerge as φ-constrained structural selectors that operate on trace structural 
properties while preserving constraint relationships and enabling logical reasoning.

This verification program implements:
1. φ-constrained predicate definition algorithms based on structural properties
2. Trace structural selection mechanisms with constraint preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection predicate logic
4. Graph theory analysis of predicate networks
5. Information theory analysis of selection entropy
6. Category theory analysis of predicate functors
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

class PredTraceSystem:
    """
    Core system for implementing logical predicates as φ-constrained structural selectors.
    Implements φ-constrained predicate logic via trace structural property analysis.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize predicate trace system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.predicate_cache = {}
        self.selection_registry = {}
        
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
            'predicate_signature': self._compute_predicate_signature(trace),
            'complexity_measure': self._compute_complexity_measure(trace)
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
        
    def _compute_predicate_signature(self, trace: str) -> Tuple[int, int, int, bool]:
        """计算trace的谓词签名：(length, ones_count, pattern_complexity, symmetry)"""
        pattern_complexity = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        symmetry = trace == trace[::-1]
        return (len(trace), trace.count('1'), pattern_complexity, symmetry)
        
    def _compute_complexity_measure(self, trace: str) -> float:
        """计算trace的复杂度度量"""
        if len(trace) <= 1:
            return 0.0
            
        # 计算模式复杂度
        pattern_count = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        max_patterns = min(4, len(trace))  # 最大可能的二进制模式数
        
        # 计算分布复杂度
        ones_ratio = trace.count('1') / len(trace)
        distribution_complexity = 4 * ones_ratio * (1 - ones_ratio)  # 最大为1当ratio=0.5
        
        # 结合模式和分布复杂度
        return (pattern_count / max_patterns) * 0.6 + distribution_complexity * 0.4

    def create_structural_predicate(self, predicate_type: str, params: Dict) -> Callable:
        """创建φ-preserving structural predicate"""
        if predicate_type == "length_predicate":
            return self._create_length_predicate(params.get('min_length', 1), params.get('max_length', 10))
        elif predicate_type == "ones_count_predicate":
            return self._create_ones_count_predicate(params.get('min_ones', 1), params.get('max_ones', 5))
        elif predicate_type == "fibonacci_predicate":
            return self._create_fibonacci_predicate(params.get('required_indices', set()))
        elif predicate_type == "complexity_predicate":
            return self._create_complexity_predicate(params.get('min_complexity', 0.0), params.get('max_complexity', 1.0))
        elif predicate_type == "symmetry_predicate":
            return self._create_symmetry_predicate(params.get('require_symmetry', True))
        elif predicate_type == "composite_predicate":
            return self._create_composite_predicate(params.get('predicates', []), params.get('operator', 'AND'))
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
        
    def _create_ones_count_predicate(self, min_ones: int, max_ones: int) -> Callable:
        """创建基于1的个数的predicate"""
        def ones_count_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return min_ones <= trace_data['ones_count'] <= max_ones
            
        return ones_count_predicate
        
    def _create_fibonacci_predicate(self, required_indices: Set[int]) -> Callable:
        """创建基于Fibonacci indices的predicate"""
        def fibonacci_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return required_indices.issubset(trace_data['fibonacci_indices'])
            
        return fibonacci_predicate
        
    def _create_complexity_predicate(self, min_complexity: float, max_complexity: float) -> Callable:
        """创建基于复杂度的predicate"""
        def complexity_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            return min_complexity <= trace_data['complexity_measure'] <= max_complexity
            
        return complexity_predicate
        
    def _create_symmetry_predicate(self, require_symmetry: bool) -> Callable:
        """创建基于对称性的predicate"""
        def symmetry_predicate(trace_value: int) -> bool:
            if trace_value not in self.trace_universe:
                return False
            trace_data = self.trace_universe[trace_value]
            signature = trace_data['predicate_signature']
            is_symmetric = signature[3]
            return is_symmetric == require_symmetry
            
        return symmetry_predicate
        
    def _create_composite_predicate(self, predicates: List[Callable], operator: str) -> Callable:
        """创建复合predicate"""
        def composite_predicate(trace_value: int) -> bool:
            if not predicates:
                return True
                
            results = [pred(trace_value) for pred in predicates]
            
            if operator == 'AND':
                return all(results)
            elif operator == 'OR':
                return any(results)
            elif operator == 'XOR':
                return sum(results) % 2 == 1
            elif operator == 'NAND':
                return not all(results)
            elif operator == 'NOR':
                return not any(results)
            else:
                return all(results)  # 默认AND
                
        return composite_predicate
        
    def _create_universal_predicate(self) -> Callable:
        """创建universal predicate (always true for φ-valid traces)"""
        def universal_predicate(trace_value: int) -> bool:
            return trace_value in self.trace_universe
            
        return universal_predicate

    def apply_predicate_selection(self, predicate: Callable, trace_set: List[int]) -> List[int]:
        """应用predicate进行trace selection"""
        selected = []
        for trace_value in trace_set:
            if predicate(trace_value):
                selected.append(trace_value)
        return selected
        
    def analyze_predicate_selectivity(self, predicate: Callable) -> Dict:
        """分析predicate的选择性"""
        all_traces = list(self.trace_universe.keys())
        selected = self.apply_predicate_selection(predicate, all_traces)
        
        return {
            'total_traces': len(all_traces),
            'selected_traces': len(selected),
            'selectivity': len(selected) / max(len(all_traces), 1),
            'selected_values': selected[:10],  # 前10个选中的值
            'phi_preservation': 1.0  # 所有都是φ-valid
        }
        
    def combine_predicates(self, pred1: Callable, pred2: Callable, operator: str) -> Callable:
        """组合两个predicates"""
        return self._create_composite_predicate([pred1, pred2], operator)
        
    def analyze_predicate_network(self, predicates: List[Callable], trace_set: List[int]) -> Dict:
        """分析predicate网络的图论属性"""
        G = nx.Graph()
        
        # 为每个predicate添加节点
        for i, predicate in enumerate(predicates):
            G.add_node(f"P{i}")
            
        # 计算predicates之间的相似性并添加边
        for i in range(len(predicates)):
            for j in range(i + 1, len(predicates)):
                # 计算两个predicates的选择重叠度
                selection1 = set(self.apply_predicate_selection(predicates[i], trace_set))
                selection2 = set(self.apply_predicate_selection(predicates[j], trace_set))
                
                overlap = len(selection1 & selection2)
                union = len(selection1 | selection2)
                similarity = overlap / max(union, 1)
                
                if similarity > 0.3:  # 阈值
                    G.add_edge(f"P{i}", f"P{j}", weight=similarity)
                    
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected': nx.is_connected(G),
            'components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
        }
        
    def compute_selection_entropy(self, predicate: Callable, trace_set: List[int]) -> float:
        """计算predicate selection的信息熵"""
        selected = self.apply_predicate_selection(predicate, trace_set)
        
        if not selected:
            return 0.0
            
        # 计算选中traces的属性分布
        length_counts = defaultdict(int)
        ones_counts = defaultdict(int)
        
        for trace_value in selected:
            if trace_value in self.trace_universe:
                trace_data = self.trace_universe[trace_value]
                length_counts[trace_data['length']] += 1
                ones_counts[trace_data['ones_count']] += 1
                
        # 计算长度分布熵
        total = len(selected)
        length_entropy = 0.0
        for count in length_counts.values():
            if count > 0:
                prob = count / total
                length_entropy -= prob * log2(prob)
                
        # 计算ones count分布熵
        ones_entropy = 0.0
        for count in ones_counts.values():
            if count > 0:
                prob = count / total
                ones_entropy -= prob * log2(prob)
                
        return (length_entropy + ones_entropy) / 2
        
    def analyze_predicate_functor_properties(self, predicates: List[Callable], trace_set: List[int]) -> Dict:
        """分析predicate的functor属性"""
        # 检查恒等性保持
        universal_pred = self._create_universal_predicate()
        universal_selection = set(self.apply_predicate_selection(universal_pred, trace_set))
        
        # 检查组合性保持
        composition_preserved = 0
        composition_total = 0
        
        for i in range(len(predicates)):
            for j in range(len(predicates)):
                if i != j:
                    composition_total += 1
                    
                    # 测试 (P2 AND P1) vs P2(P1(...))
                    combined_pred = self.combine_predicates(predicates[i], predicates[j], 'AND')
                    combined_selection = set(self.apply_predicate_selection(combined_pred, trace_set))
                    
                    # 逐步应用
                    step1_selection = self.apply_predicate_selection(predicates[i], trace_set)
                    step2_selection = set(self.apply_predicate_selection(predicates[j], step1_selection))
                    
                    if combined_selection == step2_selection:
                        composition_preserved += 1
                        
        # 检查分布保持性
        distribution_preserved = 0
        distribution_total = len(predicates)
        
        for predicate in predicates:
            selection = self.apply_predicate_selection(predicate, trace_set)
            # 检查选择是否保持φ-validity
            if all(val in self.trace_universe for val in selection):
                distribution_preserved += 1
                
        return {
            'identity_preservation': len(universal_selection) / max(len(trace_set), 1),
            'composition_preservation': composition_preserved / max(composition_total, 1),
            'distribution_preservation': distribution_preserved / max(distribution_total, 1),
            'total_composition_tests': composition_total,
            'total_distribution_tests': distribution_total
        }

class TestPredTraceSystem(unittest.TestCase):
    """单元测试：验证PredTrace系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = PredTraceSystem()
        
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
        self.assertTrue(length_pred(3))  # trace长度应该在范围内
        
        # 测试ones count predicate
        ones_pred = self.system.create_structural_predicate("ones_count_predicate", {'min_ones': 1, 'max_ones': 3})
        self.assertIsNotNone(ones_pred)
        self.assertTrue(ones_pred(2))  # 应该满足条件
        
    def test_predicate_selection(self):
        """测试predicate selection"""
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 3, 'max_length': 5})
        test_traces = [1, 2, 3, 5, 8, 13]
        
        selected = self.system.apply_predicate_selection(length_pred, test_traces)
        
        # 验证选择结果
        self.assertIsInstance(selected, list)
        # 验证所有选中的traces都满足predicate
        for trace in selected:
            self.assertTrue(length_pred(trace))
            
    def test_predicate_selectivity(self):
        """测试predicate selectivity分析"""
        ones_pred = self.system.create_structural_predicate("ones_count_predicate", {'min_ones': 2, 'max_ones': 4})
        
        selectivity = self.system.analyze_predicate_selectivity(ones_pred)
        
        # 验证selectivity分析结果
        self.assertIn('total_traces', selectivity)
        self.assertIn('selected_traces', selectivity)
        self.assertIn('selectivity', selectivity)
        self.assertGreaterEqual(selectivity['selectivity'], 0)
        self.assertLessEqual(selectivity['selectivity'], 1)
        
    def test_predicate_combination(self):
        """测试predicate组合"""
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 2, 'max_length': 5})
        ones_pred = self.system.create_structural_predicate("ones_count_predicate", {'min_ones': 1, 'max_ones': 3})
        
        combined = self.system.combine_predicates(length_pred, ones_pred, 'AND')
        self.assertIsNotNone(combined)
        
        # 测试组合predicate
        result = combined(3)
        expected = length_pred(3) and ones_pred(3)
        self.assertEqual(result, expected)
        
    def test_predicate_network_analysis(self):
        """测试predicate网络分析"""
        predicates = [
            self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 3}),
            self.system.create_structural_predicate("ones_count_predicate", {'min_ones': 1, 'max_ones': 2}),
            self.system.create_structural_predicate("complexity_predicate", {'min_complexity': 0.0, 'max_complexity': 0.5})
        ]
        
        test_traces = [1, 2, 3, 5, 8]
        network_props = self.system.analyze_predicate_network(predicates, test_traces)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertEqual(network_props['nodes'], len(predicates))
        
    def test_selection_entropy(self):
        """测试selection entropy计算"""
        length_pred = self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 5})
        test_traces = [1, 2, 3, 5, 8, 13]
        
        entropy = self.system.compute_selection_entropy(length_pred, test_traces)
        
        # 验证entropy计算
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_functor_properties(self):
        """测试functor属性"""
        predicates = [
            self.system.create_structural_predicate("length_predicate", {'min_length': 1, 'max_length': 4}),
            self.system.create_structural_predicate("ones_count_predicate", {'min_ones': 1, 'max_ones': 3})
        ]
        
        test_traces = [1, 2, 3, 5, 8]
        functor_props = self.system.analyze_predicate_functor_properties(predicates, test_traces)
        
        # 验证functor属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        self.assertIn('distribution_preservation', functor_props)
        
    def test_complexity_predicate(self):
        """测试complexity predicate"""
        complexity_pred = self.system.create_structural_predicate("complexity_predicate", {'min_complexity': 0.2, 'max_complexity': 0.8})
        
        # 测试不同traces的complexity
        for trace_val in [1, 2, 3, 5, 8]:
            if trace_val in self.system.trace_universe:
                result = complexity_pred(trace_val)
                self.assertIsInstance(result, bool)
                
    def test_fibonacci_predicate(self):
        """测试fibonacci predicate"""
        fib_pred = self.system.create_structural_predicate("fibonacci_predicate", {'required_indices': {2, 3}})
        
        # 测试fibonacci indices requirements
        for trace_val in [1, 2, 3, 5, 8]:
            if trace_val in self.system.trace_universe:
                result = fib_pred(trace_val)
                self.assertIsInstance(result, bool)

def run_comprehensive_analysis():
    """运行完整的PredTrace分析"""
    print("=" * 60)
    print("Chapter 037: PredTrace Comprehensive Analysis")
    print("Logical Predicate as φ-Constrained Structural Selector")
    print("=" * 60)
    
    system = PredTraceSystem()
    
    # 1. 基础predicate分析
    print("\n1. Basic Predicate Analysis:")
    predicates = [
        ("Length [2-4]", system.create_structural_predicate("length_predicate", {'min_length': 2, 'max_length': 4})),
        ("Ones [1-3]", system.create_structural_predicate("ones_count_predicate", {'min_ones': 1, 'max_ones': 3})),
        ("Complexity [0.2-0.8]", system.create_structural_predicate("complexity_predicate", {'min_complexity': 0.2, 'max_complexity': 0.8})),
        ("Symmetry", system.create_structural_predicate("symmetry_predicate", {'require_symmetry': True})),
        ("Fibonacci {2,3}", system.create_structural_predicate("fibonacci_predicate", {'required_indices': {2, 3}}))
    ]
    
    test_traces = list(system.trace_universe.keys())[:15]  # 前15个φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Test trace set size: {len(test_traces)}")
    print(f"Total predicates: {len(predicates)}")
    
    # 2. Predicate selectivity分析
    print("\n2. Predicate Selectivity Analysis:")
    for name, predicate in predicates:
        selectivity = system.analyze_predicate_selectivity(predicate)
        print(f"{name}: {selectivity['selected_traces']}/{selectivity['total_traces']} traces, selectivity={selectivity['selectivity']:.3f}")
    
    # 3. 网络分析
    print("\n3. Predicate Network Analysis:")
    pred_functions = [pred for _, pred in predicates]
    network_props = system.analyze_predicate_network(pred_functions, test_traces)
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Connected: {network_props['connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Average clustering: {network_props['average_clustering']:.3f}")
    
    # 4. 信息理论分析
    print("\n4. Information Theory Analysis:")
    total_entropy = 0.0
    for name, predicate in predicates:
        entropy = system.compute_selection_entropy(predicate, test_traces)
        total_entropy += entropy
        print(f"{name} selection entropy: {entropy:.3f} bits")
    
    avg_entropy = total_entropy / len(predicates)
    print(f"Average predicate entropy: {avg_entropy:.3f} bits")
    
    # 5. 范畴论分析
    print("\n5. Category Theory Analysis:")
    functor_props = system.analyze_predicate_functor_properties(pred_functions, test_traces)
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Distribution preservation: {functor_props['distribution_preservation']:.3f}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    print(f"Total distribution tests: {functor_props['total_distribution_tests']}")
    
    # 6. Predicate组合分析
    print("\n6. Predicate Combination Analysis:")
    # 创建组合predicates
    length_pred = predicates[0][1]
    ones_pred = predicates[1][1]
    
    operators = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
    for op in operators:
        combined = system.combine_predicates(length_pred, ones_pred, op)
        selectivity = system.analyze_predicate_selectivity(combined)
        print(f"Length {op} Ones: {selectivity['selected_traces']} traces, selectivity={selectivity['selectivity']:.3f}")
    
    # 7. 三域分析
    print("\n7. Three-Domain Analysis:")
    
    # Traditional predicate domain (conceptual)
    traditional_predicates = len(test_traces)  # 传统谓词可以应用于所有元素
    
    # φ-constrained domain
    phi_predicates = sum(1 for _, pred in predicates if system.analyze_predicate_selectivity(pred)['selectivity'] > 0)
    
    # Intersection analysis
    intersection_predicates = phi_predicates  # 所有φ-predicates在intersection中
    
    print(f"Traditional predicate domain: {traditional_predicates} elements")
    print(f"φ-constrained predicate domain: {phi_predicates} active predicates")
    print(f"Intersection domain: {intersection_predicates} predicates")
    print(f"Domain intersection ratio: {intersection_predicates/max(len(predicates), 1):.3f}")
    
    # 比较不同predicate类型的selectivity
    simple_selectivity = system.analyze_predicate_selectivity(predicates[0][1])['selectivity']
    complex_selectivity = system.analyze_predicate_selectivity(predicates[2][1])['selectivity']
    print(f"Simple predicate selectivity: {simple_selectivity:.3f}")
    print(f"Complex predicate selectivity: {complex_selectivity:.3f}")
    print(f"Selectivity enhancement ratio: {complex_selectivity/max(simple_selectivity, 0.001):.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - PredTrace System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running PredTrace Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()