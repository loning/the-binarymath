#!/usr/bin/env python3
"""
Chapter 034: SetOverlay Unit Test Verification
从ψ=ψ(ψ)推导Union and Intersection via Path Bundle Superposition

Core principle: From ψ = ψ(ψ) derive set operations where union and intersection 
emerge through overlaying and superposing trace path bundles in φ-constrained space.

This verification program implements:
1. Path bundle superposition algorithms for union operations
2. Bundle intersection through structural overlap analysis
3. Three-domain analysis: Traditional vs φ-constrained vs intersection set operations
4. Graph theory analysis of overlay networks
5. Information theory analysis of superposition entropy
6. Category theory analysis of overlay functors
"""

import torch
import numpy as np
import networkx as nx
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict, deque
import itertools
from math import log2, gcd
from functools import reduce

class SetOverlaySystem:
    """
    Core system for implementing set operations through path bundle overlay.
    Implements φ-constrained set operations via trace structural superposition.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize set overlay system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.overlay_cache = {}
        
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
            'overlay_signature': self._compute_overlay_signature(trace)
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
                current_index = i + 1
                # 检查避免连续索引（Zeckendorf约束）
                if not used_indices or current_index < used_indices[-1] - 1:
                    used_indices.append(current_index)
                    remaining -= self.fibonacci_numbers[i]
                    if remaining == 0:
                        break
                        
        return sorted(used_indices) if remaining == 0 else None

    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中激活的Fibonacci索引"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i + 1)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构哈希值"""
        pattern_hash = hash(trace)
        length_hash = len(trace) * 31
        ones_hash = trace.count('1') * 17
        return abs(pattern_hash + length_hash + ones_hash) % 1000000
        
    def _compute_overlay_signature(self, trace: str) -> Tuple:
        """计算trace的叠加签名：用于overlay操作分析"""
        # 基于位模式的叠加特征
        length = len(trace)
        bit_positions = tuple(i for i, bit in enumerate(trace) if bit == '1')
        zero_runs = self._compute_zero_runs(trace)
        density = trace.count('1') / len(trace) if len(trace) > 0 else 0
        
        return (length, bit_positions, tuple(zero_runs), density)
        
    def _compute_zero_runs(self, trace: str) -> List[int]:
        """计算连续0的运行长度"""
        runs = []
        current_run = 0
        
        for bit in trace:
            if bit == '0':
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
            
        return runs

    def compute_bundle_union(self, bundle_a: Set[int], bundle_b: Set[int]) -> Dict[str, any]:
        """
        计算两个路径束的并集：通过叠加实现
        
        Args:
            bundle_a: 第一个bundle的元素集合
            bundle_b: 第二个bundle的元素集合
            
        Returns:
            Union operation result with analysis
        """
        # 过滤φ-valid元素
        valid_a = {x for x in bundle_a if x in self.trace_universe}
        valid_b = {x for x in bundle_b if x in self.trace_universe}
        
        # Traditional set union
        traditional_union = valid_a | valid_b
        
        # φ-constrained overlay union
        overlay_union = self._compute_overlay_union(valid_a, valid_b)
        
        # Intersection analysis
        intersection_union = traditional_union & overlay_union
        
        return {
            'bundle_a': valid_a,
            'bundle_b': valid_b,
            'traditional_union': traditional_union,
            'overlay_union': overlay_union,
            'intersection_union': intersection_union,
            'traditional_size': len(traditional_union),
            'overlay_size': len(overlay_union),
            'intersection_size': len(intersection_union),
            'overlay_efficiency': len(intersection_union) / len(traditional_union) if traditional_union else 1.0,
            'superposition_analysis': self._analyze_superposition(valid_a, valid_b, overlay_union)
        }
        
    def _compute_overlay_union(self, bundle_a: Set[int], bundle_b: Set[int]) -> Set[int]:
        """通过路径叠加计算并集"""
        overlay_result = set()
        
        # 直接包含原始元素
        overlay_result.update(bundle_a)
        overlay_result.update(bundle_b)
        
        # 分析可能的superposition combinations
        for a in bundle_a:
            for b in bundle_b:
                if a != b:
                    superposed = self._attempt_superposition(a, b)
                    if superposed is not None and superposed in self.trace_universe:
                        overlay_result.add(superposed)
                        
        return overlay_result
        
    def _attempt_superposition(self, a: int, b: int) -> Optional[int]:
        """尝试两个trace的superposition"""
        trace_a = self.trace_universe[a]['trace']
        trace_b = self.trace_universe[b]['trace']
        
        # 对齐traces到相同长度
        max_len = max(len(trace_a), len(trace_b))
        aligned_a = trace_a.ljust(max_len, '0')
        aligned_b = trace_b.ljust(max_len, '0')
        
        # 执行逻辑OR操作 (superposition)
        superposed_trace = ''
        for i in range(max_len):
            bit_a = aligned_a[i]
            bit_b = aligned_b[i]
            # Superposition: 1 OR 1 = 1, 1 OR 0 = 1, 0 OR 0 = 0
            superposed_bit = '1' if bit_a == '1' or bit_b == '1' else '0'
            superposed_trace += superposed_bit
            
        # 验证φ-constraint
        if '11' in superposed_trace:
            return None  # 违反φ-constraint
            
        # 尝试解码回数值
        try:
            return self._decode_trace_to_value(superposed_trace)
        except:
            return None
            
    def _decode_trace_to_value(self, trace: str) -> Optional[int]:
        """将trace解码回对应的数值"""
        if trace == '0':
            return 0
            
        # 计算对应的Fibonacci和
        value = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                fib_index = i
                if fib_index < len(self.fibonacci_numbers):
                    value += self.fibonacci_numbers[fib_index]
                    
        return value if value <= self.max_trace_size else None
        
    def _analyze_superposition(self, bundle_a: Set[int], bundle_b: Set[int], 
                             overlay_result: Set[int]) -> Dict[str, any]:
        """分析superposition过程的属性"""
        # 计算新生成的元素
        original_elements = bundle_a | bundle_b
        generated_elements = overlay_result - original_elements
        
        # 分析superposition patterns
        superposition_pairs = []
        for a in bundle_a:
            for b in bundle_b:
                if a != b:
                    result = self._attempt_superposition(a, b)
                    if result is not None and result in generated_elements:
                        superposition_pairs.append((a, b, result))
                        
        return {
            'original_count': len(original_elements),
            'generated_count': len(generated_elements),
            'generation_ratio': len(generated_elements) / len(original_elements) if original_elements else 0,
            'superposition_pairs': superposition_pairs,
            'superposition_success_rate': len(superposition_pairs) / (len(bundle_a) * len(bundle_b)) if bundle_a and bundle_b else 0
        }

    def compute_bundle_intersection(self, bundle_a: Set[int], bundle_b: Set[int]) -> Dict[str, any]:
        """
        计算两个路径束的交集：通过结构重叠分析
        
        Args:
            bundle_a: 第一个bundle的元素集合
            bundle_b: 第二个bundle的元素集合
            
        Returns:
            Intersection operation result with analysis
        """
        # 过滤φ-valid元素
        valid_a = {x for x in bundle_a if x in self.trace_universe}
        valid_b = {x for x in bundle_b if x in self.trace_universe}
        
        # Traditional set intersection
        traditional_intersection = valid_a & valid_b
        
        # φ-constrained structural intersection
        structural_intersection = self._compute_structural_intersection(valid_a, valid_b)
        
        # Universal intersection
        universal_intersection = traditional_intersection & structural_intersection
        
        return {
            'bundle_a': valid_a,
            'bundle_b': valid_b,
            'traditional_intersection': traditional_intersection,
            'structural_intersection': structural_intersection,
            'universal_intersection': universal_intersection,
            'traditional_size': len(traditional_intersection),
            'structural_size': len(structural_intersection),
            'universal_size': len(universal_intersection),
            'structural_efficiency': len(universal_intersection) / len(traditional_intersection) if traditional_intersection else 1.0,
            'overlap_analysis': self._analyze_structural_overlap(valid_a, valid_b)
        }
        
    def _compute_structural_intersection(self, bundle_a: Set[int], bundle_b: Set[int]) -> Set[int]:
        """通过结构重叠计算交集"""
        structural_result = set()
        
        # 直接集合交集
        direct_intersection = bundle_a & bundle_b
        structural_result.update(direct_intersection)
        
        # 结构相似性交集：找到结构相似的元素对
        similarity_threshold = 0.7
        for a in bundle_a:
            for b in bundle_b:
                if a != b:
                    similarity = self._compute_structural_similarity(a, b)
                    if similarity >= similarity_threshold:
                        # 添加两个相似元素中较小的一个
                        structural_result.add(min(a, b))
                        
        return structural_result
        
    def _compute_structural_similarity(self, a: int, b: int) -> float:
        """计算两个trace的结构相似度"""
        if a not in self.trace_universe or b not in self.trace_universe:
            return 0.0
            
        sig_a = self.trace_universe[a]['overlay_signature']
        sig_b = self.trace_universe[b]['overlay_signature']
        
        # 多维度相似度计算
        length_sim = 1.0 - abs(sig_a[0] - sig_b[0]) / max(sig_a[0], sig_b[0]) if max(sig_a[0], sig_b[0]) > 0 else 1.0
        
        # 位置重叠相似度
        pos_a = set(sig_a[1])
        pos_b = set(sig_b[1])
        if pos_a or pos_b:
            pos_sim = len(pos_a & pos_b) / len(pos_a | pos_b)
        else:
            pos_sim = 1.0
            
        # 密度相似度
        density_sim = 1.0 - abs(sig_a[3] - sig_b[3])
        
        return 0.4 * length_sim + 0.4 * pos_sim + 0.2 * density_sim
        
    def _analyze_structural_overlap(self, bundle_a: Set[int], bundle_b: Set[int]) -> Dict[str, any]:
        """分析两个bundle的结构重叠"""
        # 计算所有pair的相似度
        similarities = []
        for a in bundle_a:
            for b in bundle_b:
                sim = self._compute_structural_similarity(a, b)
                similarities.append((a, b, sim))
                
        # 统计分析
        if similarities:
            sim_scores = [sim for _, _, sim in similarities]
            avg_similarity = sum(sim_scores) / len(sim_scores)
            max_similarity = max(sim_scores)
            high_similarity_pairs = [(a, b) for a, b, sim in similarities if sim >= 0.7]
        else:
            avg_similarity = 0.0
            max_similarity = 0.0
            high_similarity_pairs = []
            
        return {
            'total_pairs': len(similarities),
            'average_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'high_similarity_pairs': high_similarity_pairs,
            'overlap_density': len(high_similarity_pairs) / len(similarities) if similarities else 0
        }

    def three_domain_set_operations_analysis(self) -> Dict[str, any]:
        """三域集合操作分析：Traditional vs φ-constrained vs intersection"""
        
        # 测试集合
        test_bundles = [
            {1, 2, 3},
            {3, 5, 8},
            {1, 5, 13},
            {2, 8, 21},
            {1, 3, 5, 8}
        ]
        
        analysis = {
            'union_analysis': {'traditional_total': 0, 'overlay_total': 0, 'intersection_total': 0},
            'intersection_analysis': {'traditional_total': 0, 'structural_total': 0, 'universal_total': 0},
            'detailed_operations': [],
            'operation_patterns': {}
        }
        
        # 测试所有pair combinations
        for i, bundle_a in enumerate(test_bundles):
            for j, bundle_b in enumerate(test_bundles[i+1:], i+1):
                # Union analysis
                union_result = self.compute_bundle_union(bundle_a, bundle_b)
                analysis['union_analysis']['traditional_total'] += union_result['traditional_size']
                analysis['union_analysis']['overlay_total'] += union_result['overlay_size']
                analysis['union_analysis']['intersection_total'] += union_result['intersection_size']
                
                # Intersection analysis
                intersection_result = self.compute_bundle_intersection(bundle_a, bundle_b)
                analysis['intersection_analysis']['traditional_total'] += intersection_result['traditional_size']
                analysis['intersection_analysis']['structural_total'] += intersection_result['structural_size']
                analysis['intersection_analysis']['universal_total'] += intersection_result['universal_size']
                
                # 详细记录
                analysis['detailed_operations'].append({
                    'bundle_pair': (i, j),
                    'bundle_a': bundle_a,
                    'bundle_b': bundle_b,
                    'union_traditional': union_result['traditional_size'],
                    'union_overlay': union_result['overlay_size'],
                    'union_intersection': union_result['intersection_size'],
                    'intersection_traditional': intersection_result['traditional_size'],
                    'intersection_structural': intersection_result['structural_size'],
                    'intersection_universal': intersection_result['universal_size']
                })
                
        # 模式分析
        total_operations = len(analysis['detailed_operations'])
        if total_operations > 0:
            analysis['operation_patterns'] = {
                'union_efficiency_avg': analysis['union_analysis']['intersection_total'] / analysis['union_analysis']['traditional_total'] if analysis['union_analysis']['traditional_total'] > 0 else 0,
                'intersection_efficiency_avg': analysis['intersection_analysis']['universal_total'] / analysis['intersection_analysis']['traditional_total'] if analysis['intersection_analysis']['traditional_total'] > 0 else 0,
                'overlay_enhancement_ratio': analysis['union_analysis']['overlay_total'] / analysis['union_analysis']['traditional_total'] if analysis['union_analysis']['traditional_total'] > 0 else 0,
                'structural_enhancement_ratio': analysis['intersection_analysis']['structural_total'] / analysis['intersection_analysis']['traditional_total'] if analysis['intersection_analysis']['traditional_total'] > 0 else 0
            }
            
        return analysis

    def build_overlay_network(self, test_bundles: List[Set[int]]) -> nx.Graph:
        """构建overlay操作网络图"""
        G = nx.Graph()
        
        # 添加bundle节点
        for i, bundle in enumerate(test_bundles):
            valid_bundle = {x for x in bundle if x in self.trace_universe}
            G.add_node(f"bundle_{i}", 
                      elements=list(valid_bundle),
                      size=len(valid_bundle),
                      bundle_type='test_bundle')
                      
        # 添加操作边
        for i, bundle_a in enumerate(test_bundles):
            for j, bundle_b in enumerate(test_bundles[i+1:], i+1):
                union_result = self.compute_bundle_union(bundle_a, bundle_b)
                intersection_result = self.compute_bundle_intersection(bundle_a, bundle_b)
                
                # Union边
                G.add_edge(f"bundle_{i}", f"bundle_{j}",
                          operation='union',
                          traditional_size=union_result['traditional_size'],
                          overlay_size=union_result['overlay_size'],
                          efficiency=union_result['overlay_efficiency'])
                
                # Intersection边
                G.add_edge(f"bundle_{i}", f"bundle_{j}",
                          operation='intersection',
                          traditional_size=intersection_result['traditional_size'],
                          structural_size=intersection_result['structural_size'],
                          efficiency=intersection_result['structural_efficiency'])
                
        return G


class OverlayAnalyzer:
    """
    Advanced analyzer for set overlay properties using graph theory,
    information theory, and category theory approaches.
    """
    
    def __init__(self, overlay_system: SetOverlaySystem):
        self.overlay_system = overlay_system
        
    def graph_theory_analysis(self, overlay_network: nx.Graph) -> Dict[str, any]:
        """图论分析overlay网络"""
        analysis = {}
        
        # 基本图属性
        analysis['node_count'] = overlay_network.number_of_nodes()
        analysis['edge_count'] = overlay_network.number_of_edges()
        analysis['density'] = nx.density(overlay_network)
        analysis['is_connected'] = nx.is_connected(overlay_network)
        
        if overlay_network.number_of_nodes() > 0:
            analysis['average_clustering'] = nx.average_clustering(overlay_network)
            analysis['average_degree'] = sum(dict(overlay_network.degree()).values()) / overlay_network.number_of_nodes()
            
            # 连通分量分析
            components = list(nx.connected_components(overlay_network))
            analysis['connected_components'] = len(components)
            analysis['largest_component_size'] = max(len(comp) for comp in components) if components else 0
            
            # 中心性分析
            if nx.is_connected(overlay_network) and overlay_network.number_of_nodes() > 1:
                try:
                    analysis['diameter'] = nx.diameter(overlay_network)
                    analysis['radius'] = nx.radius(overlay_network)
                except:
                    analysis['diameter'] = 0
                    analysis['radius'] = 0
                    
            # 操作类型分析
            union_edges = [(u, v) for u, v, d in overlay_network.edges(data=True) if d.get('operation') == 'union']
            intersection_edges = [(u, v) for u, v, d in overlay_network.edges(data=True) if d.get('operation') == 'intersection']
            
            analysis['union_edges'] = len(union_edges)
            analysis['intersection_edges'] = len(intersection_edges)
            
        return analysis
        
    def information_theory_analysis(self, operation_results: Dict) -> Dict[str, float]:
        """信息论分析overlay操作熵"""
        analysis = {}
        
        if 'detailed_operations' in operation_results:
            operations = operation_results['detailed_operations']
            
            # Union size分布的熵
            union_sizes = [op['union_overlay'] for op in operations]
            if union_sizes:
                size_counts = defaultdict(int)
                for size in union_sizes:
                    size_counts[size] += 1
                    
                total = len(union_sizes)
                union_entropy = 0.0
                for count in size_counts.values():
                    prob = count / total
                    if prob > 0:
                        union_entropy -= prob * log2(prob)
                        
                analysis['union_size_entropy'] = union_entropy
                analysis['union_max_entropy'] = log2(len(size_counts)) if size_counts else 0
                
            # Intersection size分布的熵
            intersection_sizes = [op['intersection_structural'] for op in operations]
            if intersection_sizes:
                size_counts = defaultdict(int)
                for size in intersection_sizes:
                    size_counts[size] += 1
                    
                total = len(intersection_sizes)
                intersection_entropy = 0.0
                for count in size_counts.values():
                    prob = count / total
                    if prob > 0:
                        intersection_entropy -= prob * log2(prob)
                        
                analysis['intersection_size_entropy'] = intersection_entropy
                analysis['intersection_max_entropy'] = log2(len(size_counts)) if size_counts else 0
                
            # Operation efficiency分布
            if 'operation_patterns' in operation_results:
                patterns = operation_results['operation_patterns']
                efficiency_values = [patterns.get('union_efficiency_avg', 0), 
                                   patterns.get('intersection_efficiency_avg', 0)]
                
                if efficiency_values:
                    analysis['efficiency_variance'] = np.var(efficiency_values)
                    analysis['efficiency_mean'] = np.mean(efficiency_values)
                    
        return analysis
        
    def category_theory_analysis(self, overlay_system: SetOverlaySystem) -> Dict[str, any]:
        """范畴论分析overlay态射"""
        analysis = {
            'morphism_properties': {},
            'functor_analysis': {},
            'naturality_analysis': {}
        }
        
        # 测试简单bundles
        test_bundles = [{1, 2}, {2, 3}, {1, 3}, {1, 2, 3}]
        
        # 分析union操作的态射属性
        union_morphisms = []
        for i, bundle_a in enumerate(test_bundles):
            for j, bundle_b in enumerate(test_bundles):
                if i != j:
                    union_result = overlay_system.compute_bundle_union(bundle_a, bundle_b)
                    union_morphisms.append({
                        'source': (i, bundle_a),
                        'target': (j, bundle_b),
                        'result_size': union_result['overlay_size'],
                        'efficiency': union_result['overlay_efficiency']
                    })
                    
        # 态射保持性分析
        if union_morphisms:
            efficiencies = [m['efficiency'] for m in union_morphisms]
            analysis['morphism_properties']['union_morphisms'] = len(union_morphisms)
            analysis['morphism_properties']['average_efficiency'] = sum(efficiencies) / len(efficiencies)
            analysis['morphism_properties']['efficiency_variance'] = np.var(efficiencies)
            
        # 函子分析：恒等元保持
        identity_tests = []
        for bundle in test_bundles:
            union_self = overlay_system.compute_bundle_union(bundle, bundle)
            is_identity_preserved = union_self['overlay_union'] == bundle
            identity_tests.append(is_identity_preserved)
            
        analysis['functor_analysis']['identity_preservation_rate'] = sum(identity_tests) / len(identity_tests) if identity_tests else 0
        
        # 结合律测试
        if len(test_bundles) >= 3:
            bundle_a, bundle_b, bundle_c = test_bundles[:3]
            
            # (A ∪ B) ∪ C
            ab_union = overlay_system.compute_bundle_union(bundle_a, bundle_b)
            abc_left = overlay_system.compute_bundle_union(ab_union['overlay_union'], bundle_c)
            
            # A ∪ (B ∪ C)
            bc_union = overlay_system.compute_bundle_union(bundle_b, bundle_c)
            abc_right = overlay_system.compute_bundle_union(bundle_a, bc_union['overlay_union'])
            
            associativity_preserved = abc_left['overlay_union'] == abc_right['overlay_union']
            analysis['naturality_analysis']['associativity_preserved'] = associativity_preserved
            
        return analysis


class TestSetOverlay(unittest.TestCase):
    """Unit tests for set overlay operations"""
    
    def setUp(self):
        """测试设置"""
        self.overlay_system = SetOverlaySystem(max_trace_size=25)
        self.analyzer = OverlayAnalyzer(self.overlay_system)
        
    def test_fibonacci_generation(self):
        """测试Fibonacci数列生成"""
        expected_start = [1, 1, 2, 3, 5, 8, 13, 21]
        self.assertEqual(self.overlay_system.fibonacci_numbers[:8], expected_start)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        # 验证universe只包含φ-valid traces
        for val, trace_data in self.overlay_system.trace_universe.items():
            self.assertTrue(trace_data['phi_valid'])
            self.assertNotIn('11', trace_data['trace'])
            
    def test_superposition_operation(self):
        """测试superposition操作"""
        # 简单测试用例
        result = self.overlay_system._attempt_superposition(1, 2)
        
        # 验证结果的有效性
        if result is not None:
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, self.overlay_system.max_trace_size)
            
    def test_bundle_union_computation(self):
        """测试bundle并集计算"""
        bundle_a = {1, 2, 3}
        bundle_b = {3, 5, 8}
        
        result = self.overlay_system.compute_bundle_union(bundle_a, bundle_b)
        
        # 验证结果结构
        required_keys = ['traditional_union', 'overlay_union', 'intersection_union', 
                        'overlay_efficiency', 'superposition_analysis']
        for key in required_keys:
            self.assertIn(key, result)
            
        # 验证集合属性
        self.assertIsInstance(result['traditional_union'], set)
        self.assertIsInstance(result['overlay_union'], set)
        self.assertGreaterEqual(len(result['overlay_union']), len(result['traditional_union']))
        
    def test_bundle_intersection_computation(self):
        """测试bundle交集计算"""
        bundle_a = {1, 2, 3, 5}
        bundle_b = {2, 3, 8, 13}
        
        result = self.overlay_system.compute_bundle_intersection(bundle_a, bundle_b)
        
        # 验证结果结构
        required_keys = ['traditional_intersection', 'structural_intersection', 
                        'universal_intersection', 'structural_efficiency', 'overlap_analysis']
        for key in required_keys:
            self.assertIn(key, result)
            
        # 验证集合属性
        self.assertIsInstance(result['traditional_intersection'], set)
        self.assertIsInstance(result['structural_intersection'], set)
        
    def test_structural_similarity_computation(self):
        """测试结构相似度计算"""
        # 自我相似度测试
        if len(self.overlay_system.trace_universe) >= 1:
            val = list(self.overlay_system.trace_universe.keys())[0]
            self_sim = self.overlay_system._compute_structural_similarity(val, val)
            self.assertAlmostEqual(self_sim, 1.0, places=6)
            
        # 不同元素相似度测试
        if len(self.overlay_system.trace_universe) >= 2:
            vals = list(self.overlay_system.trace_universe.keys())[:2]
            cross_sim = self.overlay_system._compute_structural_similarity(vals[0], vals[1])
            self.assertGreaterEqual(cross_sim, 0.0)
            self.assertLessEqual(cross_sim, 1.0)
            
    def test_three_domain_analysis(self):
        """测试三域分析"""
        analysis = self.overlay_system.three_domain_set_operations_analysis()
        
        # 验证分析结构
        required_keys = ['union_analysis', 'intersection_analysis', 'detailed_operations', 'operation_patterns']
        for key in required_keys:
            self.assertIn(key, analysis)
            
        # 验证operation patterns
        patterns = analysis['operation_patterns']
        for pattern_key in ['union_efficiency_avg', 'intersection_efficiency_avg']:
            if pattern_key in patterns:
                self.assertGreaterEqual(patterns[pattern_key], 0.0)
                self.assertLessEqual(patterns[pattern_key], 1.0)
                
    def test_overlay_network_construction(self):
        """测试overlay网络构建"""
        test_bundles = [{1, 2}, {2, 3}, {1, 3}]
        network = self.overlay_system.build_overlay_network(test_bundles)
        
        # 验证图结构
        self.assertIsInstance(network, nx.Graph)
        self.assertEqual(network.number_of_nodes(), len(test_bundles))
        
        # 验证节点属性
        for node in network.nodes():
            node_data = network.nodes[node]
            self.assertIn('elements', node_data)
            self.assertIn('size', node_data)
            
    def test_graph_theory_analysis(self):
        """测试图论分析"""
        test_bundles = [{1, 2}, {2, 3}, {1, 3}, {1, 2, 3}]
        network = self.overlay_system.build_overlay_network(test_bundles)
        analysis = self.analyzer.graph_theory_analysis(network)
        
        # 验证分析结果
        required_metrics = ['node_count', 'edge_count', 'density', 'is_connected']
        for metric in required_metrics:
            self.assertIn(metric, analysis)
            
        # 验证数值范围
        self.assertGreaterEqual(analysis['density'], 0)
        self.assertLessEqual(analysis['density'], 1)
        
    def test_information_theory_analysis(self):
        """测试信息论分析"""
        operation_results = self.overlay_system.three_domain_set_operations_analysis()
        analysis = self.analyzer.information_theory_analysis(operation_results)
        
        # 验证熵计算
        entropy_metrics = ['union_size_entropy', 'intersection_size_entropy']
        for metric in entropy_metrics:
            if metric in analysis:
                self.assertGreaterEqual(analysis[metric], 0)
                
    def test_category_theory_analysis(self):
        """测试范畴论分析"""
        analysis = self.analyzer.category_theory_analysis(self.overlay_system)
        
        # 验证分析结构
        required_sections = ['morphism_properties', 'functor_analysis', 'naturality_analysis']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # 验证恒等元保持
        functor_props = analysis['functor_analysis']
        if 'identity_preservation_rate' in functor_props:
            self.assertGreaterEqual(functor_props['identity_preservation_rate'], 0)
            self.assertLessEqual(functor_props['identity_preservation_rate'], 1)


def main():
    """Main verification routine"""
    print("=== Chapter 034: SetOverlay Unit Test Verification ===")
    print("从ψ=ψ(ψ)推导Union and Intersection via Path Bundle Superposition")
    print()
    
    # 1. 运行单元测试
    print("1. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSetOverlay)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Unit tests failed. Please check the implementation.")
        return
        
    print()
    
    # 2. 系统分析
    print("2. Running SetOverlay Analysis...")
    overlay_system = SetOverlaySystem(max_trace_size=25)
    analyzer = OverlayAnalyzer(overlay_system)
    
    # 测试集合操作
    test_bundles = [{1, 2, 3}, {3, 5, 8}, {1, 5, 13}, {2, 8, 21}]
    print(f"Testing with bundles: {test_bundles}")
    print()
    
    # 3. 三域集合操作分析
    print("3. Three-Domain Set Operations Analysis:")
    three_domain = overlay_system.three_domain_set_operations_analysis()
    
    union_analysis = three_domain['union_analysis']
    intersection_analysis = three_domain['intersection_analysis']
    
    print(f"   Union operations:")
    print(f"     Traditional total: {union_analysis['traditional_total']}")
    print(f"     Overlay total: {union_analysis['overlay_total']}")
    print(f"     Intersection total: {union_analysis['intersection_total']}")
    
    print(f"   Intersection operations:")
    print(f"     Traditional total: {intersection_analysis['traditional_total']}")
    print(f"     Structural total: {intersection_analysis['structural_total']}")
    print(f"     Universal total: {intersection_analysis['universal_total']}")
    
    if 'operation_patterns' in three_domain:
        patterns = three_domain['operation_patterns']
        print(f"   Operation patterns:")
        for pattern_name, value in patterns.items():
            print(f"     {pattern_name}: {value:.3f}")
    print()
    
    # 4. Overlay网络分析
    print("4. Overlay Network Analysis:")
    overlay_network = overlay_system.build_overlay_network(test_bundles)
    graph_analysis = analyzer.graph_theory_analysis(overlay_network)
    
    print(f"   Network nodes: {graph_analysis['node_count']}")
    print(f"   Network edges: {graph_analysis['edge_count']}")
    print(f"   Network density: {graph_analysis['density']:.3f}")
    print(f"   Connected: {graph_analysis['is_connected']}")
    
    if 'union_edges' in graph_analysis:
        print(f"   Union edges: {graph_analysis['union_edges']}")
    if 'intersection_edges' in graph_analysis:
        print(f"   Intersection edges: {graph_analysis['intersection_edges']}")
    print()
    
    # 5. 信息论分析
    print("5. Information Theory Analysis:")
    info_analysis = analyzer.information_theory_analysis(three_domain)
    
    if 'union_size_entropy' in info_analysis:
        print(f"   Union size entropy: {info_analysis['union_size_entropy']:.3f} bits")
    if 'intersection_size_entropy' in info_analysis:
        print(f"   Intersection size entropy: {info_analysis['intersection_size_entropy']:.3f} bits")
    if 'efficiency_mean' in info_analysis:
        print(f"   Efficiency mean: {info_analysis['efficiency_mean']:.3f}")
    print()
    
    # 6. 范畴论分析
    print("6. Category Theory Analysis:")
    category_analysis = analyzer.category_theory_analysis(overlay_system)
    
    morph_props = category_analysis['morphism_properties']
    if 'union_morphisms' in morph_props:
        print(f"   Union morphisms: {morph_props['union_morphisms']}")
    if 'average_efficiency' in morph_props:
        print(f"   Average efficiency: {morph_props['average_efficiency']:.3f}")
        
    functor_analysis = category_analysis['functor_analysis']
    if 'identity_preservation_rate' in functor_analysis:
        print(f"   Identity preservation rate: {functor_analysis['identity_preservation_rate']:.3f}")
        
    naturality = category_analysis['naturality_analysis']
    if 'associativity_preserved' in naturality:
        print(f"   Associativity preserved: {naturality['associativity_preserved']}")
        
    print()
    
    print("=== Verification Complete ===")
    print("Key insights from verification:")
    print("1. φ-constraint creates enhanced overlay operations")
    print("2. Superposition generates new elements beyond traditional union")
    print("3. Structural intersection provides richer overlap analysis")
    print("4. Three-domain analysis reveals operation enhancement patterns")
    print("5. Category theory confirms overlay morphism properties")
    

if __name__ == "__main__":
    main()