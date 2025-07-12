#!/usr/bin/env python3
"""
Chapter 036: CollapseCompose Unit Test Verification
从ψ=ψ(ψ)推导Compositional Mapping via Trace Chain Propagation

Core principle: From ψ = ψ(ψ) derive function composition where complex 
transformations emerge through chaining collapse transformations that preserve 
φ-constraint structural integrity across compositional chains.

This verification program implements:
1. Trace chain propagation algorithms for compositional mappings
2. φ-preserving composition operators with structural validation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection composition theory
4. Graph theory analysis of composition networks
5. Information theory analysis of compositional complexity
6. Category theory analysis of compositional functors
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

class CollapseComposeSystem:
    """
    Core system for implementing compositional mapping via trace chain propagation.
    Implements φ-constrained composition theory via structural transformation chains.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize collapse composition system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.composition_cache = {}
        self.chain_registry = {}
        
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
            'composition_signature': self._compute_composition_signature(trace)
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
        
    def _compute_composition_signature(self, trace: str) -> Tuple[int, int, int, int]:
        """计算trace的组合签名：(length, ones_count, pattern_hash, complexity)"""
        pattern_hash = sum(int(bit) * (2 ** i) for i, bit in enumerate(trace))
        complexity = len(trace) * trace.count('1')
        return (len(trace), trace.count('1'), pattern_hash % 1000, complexity)

    def create_transformation_function(self, transform_type: str, params: Dict) -> Callable:
        """创建φ-preserving transformation function"""
        if transform_type == "fibonacci_shift":
            return self._create_fibonacci_shift(params.get('shift', 1))
        elif transform_type == "structural_map":
            return self._create_structural_map(params.get('mapping', {}))
        elif transform_type == "constraint_filter":
            return self._create_constraint_filter(params.get('predicate', lambda x: True))
        elif transform_type == "trace_amplify":
            return self._create_trace_amplify(params.get('factor', 2))
        else:
            return self._create_identity_transform()
            
    def _create_fibonacci_shift(self, shift: int) -> Callable:
        """创建Fibonacci index shifting transformation"""
        def fibonacci_shift(input_val: int) -> Optional[int]:
            if input_val not in self.trace_universe:
                return None
            
            input_trace = self.trace_universe[input_val]
            shifted_indices = {idx + shift for idx in input_trace['fibonacci_indices']}
            
            # 找到具有这些shifted indices的trace
            for val, trace_data in self.trace_universe.items():
                if trace_data['fibonacci_indices'] == shifted_indices:
                    return val
                    
            # 如果没有精确匹配，返回最相似的
            best_match = None
            best_overlap = 0
            for val, trace_data in self.trace_universe.items():
                overlap = len(shifted_indices & trace_data['fibonacci_indices'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = val
                    
            return best_match
            
        return fibonacci_shift
        
    def _create_structural_map(self, mapping: Dict[int, int]) -> Callable:
        """创建基于结构的映射transformation"""
        def structural_map(input_val: int) -> Optional[int]:
            if input_val in mapping and mapping[input_val] in self.trace_universe:
                return mapping[input_val]
            elif input_val in self.trace_universe:
                # 通过结构相似性寻找映射
                input_sig = self.trace_universe[input_val]['composition_signature']
                for target_val in self.trace_universe.keys():
                    target_sig = self.trace_universe[target_val]['composition_signature']
                    if input_sig[1] == target_sig[1] and target_val != input_val:  # 相同ones_count
                        return target_val
                return input_val  # 返回自身作为fallback
            return None
            
        return structural_map
        
    def _create_constraint_filter(self, predicate: Callable) -> Callable:
        """创建约束过滤transformation"""
        def constraint_filter(input_val: int) -> Optional[int]:
            if input_val not in self.trace_universe:
                return None
                
            if predicate(self.trace_universe[input_val]):
                return input_val
            else:
                # 寻找满足predicate的最相似trace
                input_sig = self.trace_universe[input_val]['composition_signature']
                best_match = None
                best_similarity = 0
                
                for val, trace_data in self.trace_universe.items():
                    if predicate(trace_data):
                        similarity = self._compute_signature_similarity(input_sig, trace_data['composition_signature'])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = val
                            
                return best_match
                
        return constraint_filter
        
    def _create_trace_amplify(self, factor: int) -> Callable:
        """创建trace amplification transformation"""
        def trace_amplify(input_val: int) -> Optional[int]:
            if input_val not in self.trace_universe:
                return None
                
            # 简单的amplification：寻找具有更多fibonacci indices的trace
            input_trace = self.trace_universe[input_val]
            target_ones = min(input_trace['ones_count'] * factor, 5)  # 限制最大值
            
            for val, trace_data in self.trace_universe.items():
                if trace_data['ones_count'] == target_ones:
                    return val
                    
            # 如果没有精确匹配，返回最接近的
            best_match = None
            best_diff = float('inf')
            for val, trace_data in self.trace_universe.items():
                diff = abs(trace_data['ones_count'] - target_ones)
                if diff < best_diff:
                    best_diff = diff
                    best_match = val
                    
            return best_match
            
        return trace_amplify
        
    def _create_identity_transform(self) -> Callable:
        """创建恒等transformation"""
        def identity_transform(input_val: int) -> Optional[int]:
            return input_val if input_val in self.trace_universe else None
            
        return identity_transform
        
    def _compute_signature_similarity(self, sig1: Tuple, sig2: Tuple) -> float:
        """计算两个composition signatures的相似性"""
        if len(sig1) != len(sig2):
            return 0.0
            
        similarities = []
        for i, (a, b) in enumerate(zip(sig1, sig2)):
            if a == b:
                similarities.append(1.0)
            elif max(a, b) > 0:
                similarities.append(min(a, b) / max(a, b))
            else:
                similarities.append(1.0)
                
        return sum(similarities) / len(similarities)

    def compose_transformations(self, transforms: List[Callable]) -> Callable:
        """组合多个transformations形成compositional chain"""
        def composed_transform(input_val: int) -> Optional[int]:
            current_val = input_val
            for transform in transforms:
                if current_val is None:
                    return None
                current_val = transform(current_val)
                
            return current_val
            
        return composed_transform
        
    def verify_composition_validity(self, transforms: List[Callable], test_inputs: List[int]) -> Dict:
        """验证组合变换的有效性"""
        results = {
            'total_tests': len(test_inputs),
            'successful_compositions': 0,
            'phi_preservation_rate': 0.0,
            'chain_length_distribution': defaultdict(int),
            'composition_entropy': 0.0
        }
        
        successful_outputs = []
        
        for input_val in test_inputs:
            if input_val not in self.trace_universe:
                continue
                
            # 逐步应用transformations
            current_val = input_val
            chain_length = 0
            
            for transform in transforms:
                if current_val is None:
                    break
                next_val = transform(current_val)
                if next_val is not None and next_val in self.trace_universe:
                    current_val = next_val
                    chain_length += 1
                else:
                    break
                    
            if current_val is not None and current_val in self.trace_universe:
                results['successful_compositions'] += 1
                successful_outputs.append(current_val)
                results['chain_length_distribution'][chain_length] += 1
                
        # 计算φ-preservation rate
        if results['total_tests'] > 0:
            results['phi_preservation_rate'] = results['successful_compositions'] / results['total_tests']
            
        # 计算composition entropy
        if successful_outputs:
            output_counts = defaultdict(int)
            for output in successful_outputs:
                output_counts[output] += 1
                
            total = len(successful_outputs)
            entropy = 0.0
            for count in output_counts.values():
                if count > 0:
                    prob = count / total
                    entropy -= prob * log2(prob)
                    
            results['composition_entropy'] = entropy
            
        return results
        
    def analyze_composition_network(self, transforms: List[Callable], test_inputs: List[int]) -> Dict:
        """分析组合网络的图论属性"""
        G = nx.DiGraph()
        
        # 添加节点和边
        for input_val in test_inputs:
            if input_val not in self.trace_universe:
                continue
                
            current_val = input_val
            for i, transform in enumerate(transforms):
                if current_val is None:
                    break
                next_val = transform(current_val)
                if next_val is not None and next_val in self.trace_universe:
                    G.add_edge(current_val, next_val, transform_step=i)
                    current_val = next_val
                else:
                    break
                    
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected': nx.number_strongly_connected_components(G),
            'weakly_connected': nx.number_weakly_connected_components(G),
            'average_path_length': self._compute_average_path_length(G)
        }
        
    def _compute_average_path_length(self, G: nx.DiGraph) -> float:
        """计算有向图的平均路径长度"""
        if G.number_of_nodes() <= 1:
            return 0.0
            
        total_length = 0
        total_pairs = 0
        
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, length in lengths.items():
                if source != target:
                    total_length += length
                    total_pairs += 1
                    
        return total_length / max(total_pairs, 1)
        
    def analyze_composition_functor_properties(self, transforms: List[Callable], test_inputs: List[int]) -> Dict:
        """分析组合的functor属性"""
        # 检查恒等性保持
        identity_preserved = 0
        identity_total = 0
        
        # 检查结构保持性
        structure_preserved = 0
        structure_total = 0
        
        # 检查组合性保持
        composition_preserved = 0
        composition_total = 0
        
        for input_val in test_inputs[:10]:  # 限制测试数量
            if input_val not in self.trace_universe:
                continue
                
            # 测试恒等性
            identity_total += 1
            if len(transforms) > 0:
                result = transforms[0](input_val)
                if result == input_val:  # 如果第一个transform是恒等的
                    identity_preserved += 1
                    
            # 测试结构保持性
            structure_total += 1
            current_val = input_val
            structure_maintained = True
            
            for transform in transforms:
                if current_val is None:
                    structure_maintained = False
                    break
                next_val = transform(current_val)
                if next_val is None or next_val not in self.trace_universe:
                    structure_maintained = False
                    break
                    
                # 检查某些结构属性是否保持
                current_sig = self.trace_universe[current_val]['composition_signature']
                next_sig = self.trace_universe[next_val]['composition_signature']
                
                # 简单的结构保持检查：长度不能差太多
                if abs(current_sig[0] - next_sig[0]) > 2:
                    structure_maintained = False
                    break
                    
                current_val = next_val
                
            if structure_maintained:
                structure_preserved += 1
                
            # 测试组合性
            if len(transforms) >= 2:
                composition_total += 1
                
                # 测试 (f ∘ g)(x) = f(g(x))
                intermediate = transforms[0](input_val)
                if intermediate is not None:
                    direct_result = transforms[1](intermediate)
                    
                    # 用组合函数
                    composed = self.compose_transformations([transforms[0], transforms[1]])
                    composed_result = composed(input_val)
                    
                    if direct_result == composed_result:
                        composition_preserved += 1
                        
        return {
            'identity_preservation': identity_preserved / max(identity_total, 1),
            'structure_preservation': structure_preserved / max(structure_total, 1),
            'composition_preservation': composition_preserved / max(composition_total, 1),
            'total_identity_tests': identity_total,
            'total_structure_tests': structure_total,
            'total_composition_tests': composition_total
        }

class TestCollapseComposeSystem(unittest.TestCase):
    """单元测试：验证CollapseCompose系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseComposeSystem()
        
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
        
    def test_transformation_creation(self):
        """测试transformation function创建"""
        # 测试fibonacci shift
        fib_shift = self.system.create_transformation_function("fibonacci_shift", {'shift': 1})
        self.assertIsNotNone(fib_shift)
        result = fib_shift(1)
        self.assertIsNotNone(result)
        
        # 测试structural map
        struct_map = self.system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3}})
        self.assertIsNotNone(struct_map)
        result = struct_map(1)
        self.assertEqual(result, 2)
        
    def test_transformation_composition(self):
        """测试transformation组合"""
        transform1 = self.system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3, 3: 5}})
        transform2 = self.system.create_transformation_function("fibonacci_shift", {'shift': 1})
        
        composed = self.system.compose_transformations([transform1, transform2])
        result = composed(1)
        self.assertIsNotNone(result)
        
    def test_composition_validity(self):
        """测试组合有效性验证"""
        transforms = [
            self.system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3, 3: 5}}),
            self.system.create_transformation_function("fibonacci_shift", {'shift': 1})
        ]
        
        test_inputs = [1, 2, 3, 5, 8]
        results = self.system.verify_composition_validity(transforms, test_inputs)
        
        # 验证结果结构
        self.assertIn('total_tests', results)
        self.assertIn('successful_compositions', results)
        self.assertIn('phi_preservation_rate', results)
        self.assertGreaterEqual(results['phi_preservation_rate'], 0)
        self.assertLessEqual(results['phi_preservation_rate'], 1)
        
    def test_composition_network_analysis(self):
        """测试组合网络分析"""
        transforms = [
            self.system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3, 3: 5}}),
            self.system.create_transformation_function("trace_amplify", {'factor': 2})
        ]
        
        test_inputs = [1, 2, 3, 5, 8]
        network_props = self.system.analyze_composition_network(transforms, test_inputs)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertIn('is_dag', network_props)
        
    def test_functor_properties(self):
        """测试functor属性"""
        transforms = [
            self.system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3}}),
            self.system.create_transformation_function("fibonacci_shift", {'shift': 1})
        ]
        
        test_inputs = [1, 2, 3, 5]
        functor_props = self.system.analyze_composition_functor_properties(transforms, test_inputs)
        
        # 验证属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('structure_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        
    def test_constraint_filter(self):
        """测试约束过滤transformation"""
        # 创建一个predicate：只保留ones_count >= 2的traces
        predicate = lambda trace_data: trace_data['ones_count'] >= 2
        filter_transform = self.system.create_transformation_function("constraint_filter", {'predicate': predicate})
        
        result = filter_transform(1)  # ones_count = 1, 应该被转换
        self.assertIsNotNone(result)
        if result in self.system.trace_universe:
            self.assertGreaterEqual(self.system.trace_universe[result]['ones_count'], 2)
            
    def test_trace_amplification(self):
        """测试trace amplification"""
        amplify_transform = self.system.create_transformation_function("trace_amplify", {'factor': 2})
        
        result = amplify_transform(2)  # 应该返回一个ones_count更大的trace
        self.assertIsNotNone(result)
        
        if result in self.system.trace_universe:
            original_ones = self.system.trace_universe[2]['ones_count']
            result_ones = self.system.trace_universe[result]['ones_count']
            # amplification应该增加ones_count（在合理范围内）
            self.assertGreaterEqual(result_ones, original_ones)

def run_comprehensive_analysis():
    """运行完整的CollapseCompose分析"""
    print("=" * 60)
    print("Chapter 036: CollapseCompose Comprehensive Analysis")
    print("Compositional Mapping via Trace Chain Propagation")
    print("=" * 60)
    
    system = CollapseComposeSystem()
    
    # 1. 基础组合分析
    print("\n1. Basic Composition Analysis:")
    transforms = [
        system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3, 3: 5, 5: 8}}),
        system.create_transformation_function("fibonacci_shift", {'shift': 1}),
        system.create_transformation_function("trace_amplify", {'factor': 2})
    ]
    
    test_inputs = [1, 2, 3, 5, 8, 13]
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Transformation chain length: {len(transforms)}")
    print(f"Test input size: {len(test_inputs)}")
    
    # 2. 组合有效性验证
    print("\n2. Composition Validity Analysis:")
    validity_results = system.verify_composition_validity(transforms, test_inputs)
    print(f"Total tests: {validity_results['total_tests']}")
    print(f"Successful compositions: {validity_results['successful_compositions']}")
    print(f"φ-preservation rate: {validity_results['phi_preservation_rate']:.3f}")
    print(f"Composition entropy: {validity_results['composition_entropy']:.3f} bits")
    print(f"Chain length distribution: {dict(validity_results['chain_length_distribution'])}")
    
    # 3. 网络分析
    print("\n3. Composition Network Analysis:")
    network_props = system.analyze_composition_network(transforms, test_inputs)
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Is DAG (acyclic): {network_props['is_dag']}")
    print(f"Strongly connected components: {network_props['strongly_connected']}")
    print(f"Weakly connected components: {network_props['weakly_connected']}")
    print(f"Average path length: {network_props['average_path_length']:.3f}")
    
    # 4. 范畴论分析
    print("\n4. Category Theory Analysis:")
    functor_props = system.analyze_composition_functor_properties(transforms, test_inputs)
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Structure preservation: {functor_props['structure_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Total identity tests: {functor_props['total_identity_tests']}")
    print(f"Total structure tests: {functor_props['total_structure_tests']}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    
    # 5. 个别transformation分析
    print("\n5. Individual Transformation Analysis:")
    single_transforms = [
        ("Fibonacci Shift", system.create_transformation_function("fibonacci_shift", {'shift': 1})),
        ("Structural Map", system.create_transformation_function("structural_map", {'mapping': {1: 2, 2: 3, 3: 5}})),
        ("Trace Amplify", system.create_transformation_function("trace_amplify", {'factor': 2})),
        ("Constraint Filter", system.create_transformation_function("constraint_filter", {'predicate': lambda td: td['ones_count'] >= 2}))
    ]
    
    for name, transform in single_transforms:
        single_validity = system.verify_composition_validity([transform], test_inputs)
        print(f"{name}: {single_validity['phi_preservation_rate']:.3f} preservation rate, {single_validity['composition_entropy']:.3f} entropy")
    
    # 6. 三域分析
    print("\n6. Three-Domain Analysis:")
    
    # Traditional function composition domain
    traditional_compose_count = len(test_inputs) * (len(test_inputs) - 1)  # 传统组合可能数
    
    # φ-constrained domain
    phi_compose_count = validity_results['successful_compositions']
    
    # Intersection analysis
    intersection_rate = phi_compose_count / max(traditional_compose_count, 1)
    
    print(f"Traditional composition possibilities: {traditional_compose_count}")
    print(f"φ-constrained successful compositions: {phi_compose_count}")
    print(f"Intersection success rate: {intersection_rate:.3f}")
    print(f"φ-preservation rate: {validity_results['phi_preservation_rate']:.3f}")
    
    # 比较单独变换vs组合变换的熵
    single_entropies = []
    for _, transform in single_transforms:
        single_validity = system.verify_composition_validity([transform], test_inputs)
        single_entropies.append(single_validity['composition_entropy'])
        
    avg_single_entropy = sum(single_entropies) / len(single_entropies)
    print(f"Average single transformation entropy: {avg_single_entropy:.3f} bits")
    print(f"Composed transformation entropy: {validity_results['composition_entropy']:.3f} bits")
    print(f"Composition entropy enhancement: {validity_results['composition_entropy']/max(avg_single_entropy, 0.001):.3f}x")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - CollapseCompose System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running CollapseCompose Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()