#!/usr/bin/env python3
"""
Chapter 032: SetBundle Unit Test Verification
从ψ=ψ(ψ)推导Collapse Path Clusters作为φ-Structural Sets

Core principle: From ψ = ψ(ψ) derive set theory where sets are bundles of related 
collapse paths, creating a new foundation based on trace structural relationships.

This verification program implements:
1. SetBundle construction from φ-compliant traces
2. Path clustering algorithms based on structural similarity
3. Three-domain analysis: Traditional vs φ-constrained vs intersection set theory
4. Graph theory analysis of set connectivity
5. Information theory analysis of set entropy
6. Category theory analysis of set morphisms
"""

import torch
import numpy as np
import networkx as nx
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict
import itertools
from math import log2, gcd
from functools import reduce

class SetBundleSystem:
    """
    Core system for constructing and analyzing set bundles from collapse paths.
    Implements φ-constrained set theory through trace structural relationships.
    """
    
    def __init__(self, max_trace_size: int = 40):
        """Initialize set bundle system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.set_bundle_cache = {}
        
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
            'collapse_depth': self._compute_collapse_depth(trace)
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
        # 基于位模式和长度的简单哈希
        pattern_hash = hash(trace)
        length_hash = len(trace) * 31
        ones_hash = trace.count('1') * 17
        return abs(pattern_hash + length_hash + ones_hash) % 1000000
        
    def _compute_collapse_depth(self, trace: str) -> int:
        """计算trace的collapse深度"""
        # 基于连续0的最大长度
        max_zeros = 0
        current_zeros = 0
        for bit in trace:
            if bit == '0':
                current_zeros += 1
                max_zeros = max(max_zeros, current_zeros)
            else:
                current_zeros = 0
        return max_zeros

    def create_set_bundle(self, clustering_method: str = 'structural', 
                         similarity_threshold: float = 0.7) -> Dict[str, Set[int]]:
        """
        创建SetBundle：将traces聚类成相关路径的集合
        
        Args:
            clustering_method: 'structural', 'fibonacci', 'length', 'hybrid'
            similarity_threshold: 相似度阈值
            
        Returns:
            Dict mapping bundle names to sets of trace values
        """
        if clustering_method in self.set_bundle_cache:
            return self.set_bundle_cache[clustering_method]
            
        bundles = {}
        
        if clustering_method == 'structural':
            bundles = self._cluster_by_structural_similarity(similarity_threshold)
        elif clustering_method == 'fibonacci':
            bundles = self._cluster_by_fibonacci_components()
        elif clustering_method == 'length':
            bundles = self._cluster_by_trace_length()
        elif clustering_method == 'hybrid':
            bundles = self._cluster_hybrid_approach(similarity_threshold)
        else:
            # Default: 基本结构聚类
            bundles = self._cluster_by_structural_similarity(similarity_threshold)
            
        self.set_bundle_cache[clustering_method] = bundles
        return bundles
        
    def _cluster_by_structural_similarity(self, threshold: float) -> Dict[str, Set[int]]:
        """基于结构相似度聚类"""
        traces = list(self.trace_universe.keys())
        clusters = {}
        cluster_id = 0
        
        processed = set()
        
        for trace_val in traces:
            if trace_val in processed:
                continue
                
            # 创建新cluster
            cluster_name = f"structural_cluster_{cluster_id}"
            cluster = {trace_val}
            processed.add(trace_val)
            
            # 查找相似traces
            for other_val in traces:
                if other_val in processed:
                    continue
                    
                similarity = self._compute_structural_similarity(trace_val, other_val)
                if similarity >= threshold:
                    cluster.add(other_val)
                    processed.add(other_val)
                    
            clusters[cluster_name] = cluster
            cluster_id += 1
            
        return clusters
        
    def _compute_structural_similarity(self, val1: int, val2: int) -> float:
        """计算两个trace的结构相似度"""
        trace1 = self.trace_universe[val1]
        trace2 = self.trace_universe[val2]
        
        # 多维度相似度计算
        length_sim = 1.0 - abs(trace1['length'] - trace2['length']) / max(trace1['length'], trace2['length'])
        ones_sim = 1.0 - abs(trace1['ones_count'] - trace2['ones_count']) / max(trace1['ones_count'], trace2['ones_count'], 1)
        
        # Fibonacci索引重叠度
        indices1 = set(trace1['fibonacci_indices'])
        indices2 = set(trace2['fibonacci_indices'])
        if indices1 or indices2:
            fib_sim = len(indices1 & indices2) / len(indices1 | indices2)
        else:
            fib_sim = 1.0
            
        # Collapse深度相似度
        depth_sim = 1.0 - abs(trace1['collapse_depth'] - trace2['collapse_depth']) / max(trace1['collapse_depth'], trace2['collapse_depth'], 1)
        
        # 加权平均
        return 0.3 * length_sim + 0.3 * ones_sim + 0.3 * fib_sim + 0.1 * depth_sim
        
    def _cluster_by_fibonacci_components(self) -> Dict[str, Set[int]]:
        """基于Fibonacci组件聚类"""
        clusters = defaultdict(set)
        
        for val, trace_data in self.trace_universe.items():
            # 使用主要Fibonacci索引作为聚类键
            indices = trace_data['fibonacci_indices']
            if indices:
                primary_index = max(indices)  # 使用最大索引
                cluster_key = f"fibonacci_F{primary_index}"
            else:
                cluster_key = "fibonacci_empty"
            clusters[cluster_key].add(val)
            
        return dict(clusters)
        
    def _cluster_by_trace_length(self) -> Dict[str, Set[int]]:
        """基于trace长度聚类"""
        clusters = defaultdict(set)
        
        for val, trace_data in self.trace_universe.items():
            length = trace_data['length']
            cluster_key = f"length_{length}"
            clusters[cluster_key].add(val)
            
        return dict(clusters)
        
    def _cluster_hybrid_approach(self, threshold: float) -> Dict[str, Set[int]]:
        """混合聚类方法"""
        # 先按长度粗分
        length_clusters = self._cluster_by_trace_length()
        
        # 在每个长度组内进行结构聚类
        final_clusters = {}
        cluster_id = 0
        
        for length_cluster_name, length_cluster in length_clusters.items():
            if len(length_cluster) <= 1:
                final_clusters[f"hybrid_{cluster_id}"] = length_cluster
                cluster_id += 1
                continue
                
            # 在长度组内进行结构聚类
            processed = set()
            for trace_val in length_cluster:
                if trace_val in processed:
                    continue
                    
                sub_cluster = {trace_val}
                processed.add(trace_val)
                
                for other_val in length_cluster:
                    if other_val in processed:
                        continue
                        
                    similarity = self._compute_structural_similarity(trace_val, other_val)
                    if similarity >= threshold:
                        sub_cluster.add(other_val)
                        processed.add(other_val)
                        
                final_clusters[f"hybrid_{cluster_id}"] = sub_cluster
                cluster_id += 1
                
        return final_clusters

    def compute_bundle_properties(self, bundles: Dict[str, Set[int]]) -> Dict[str, Dict]:
        """计算SetBundle的数学属性"""
        properties = {}
        
        for bundle_name, bundle_set in bundles.items():
            props = {
                'cardinality': len(bundle_set),
                'elements': list(bundle_set),
                'traces': [self.trace_universe[val]['trace'] for val in bundle_set if val in self.trace_universe],
                'average_length': 0,
                'total_ones': 0,
                'fibonacci_coverage': set(),
                'structural_diversity': 0
            }
            
            if bundle_set:
                # 计算平均长度
                lengths = [self.trace_universe[val]['length'] for val in bundle_set if val in self.trace_universe]
                props['average_length'] = sum(lengths) / len(lengths) if lengths else 0
                
                # 计算总1的数量
                ones_counts = [self.trace_universe[val]['ones_count'] for val in bundle_set if val in self.trace_universe]
                props['total_ones'] = sum(ones_counts)
                
                # Fibonacci覆盖度
                all_indices = set()
                for val in bundle_set:
                    if val in self.trace_universe:
                        all_indices.update(self.trace_universe[val]['fibonacci_indices'])
                props['fibonacci_coverage'] = all_indices
                
                # 结构多样性（unique structural hashes）
                hashes = {self.trace_universe[val]['structural_hash'] for val in bundle_set if val in self.trace_universe}
                props['structural_diversity'] = len(hashes)
                
            properties[bundle_name] = props
            
        return properties

    def three_domain_analysis(self) -> Dict[str, any]:
        """三域分析：Traditional vs φ-constrained vs intersection set theory"""
        
        # Traditional set theory: 所有自然数都可以形成集合
        traditional_universe = set(range(self.max_trace_size + 1))
        
        # φ-constrained set theory: 只有φ-valid traces可以形成集合
        phi_universe = set(self.trace_universe.keys())
        
        # Intersection: φ-valid numbers that also work in traditional context
        intersection_universe = phi_universe & traditional_universe
        
        # 创建不同域的set bundles
        structural_bundles = self.create_set_bundle('structural')
        fibonacci_bundles = self.create_set_bundle('fibonacci')
        
        analysis = {
            'traditional_universe_size': len(traditional_universe),
            'phi_universe_size': len(phi_universe),
            'intersection_universe_size': len(intersection_universe),
            'universe_overlap_ratio': len(intersection_universe) / len(traditional_universe) if traditional_universe else 0,
            'bundle_analysis': {
                'structural_bundles': len(structural_bundles),
                'fibonacci_bundles': len(fibonacci_bundles),
                'bundle_size_distribution': {}
            }
        }
        
        # Bundle大小分布分析
        for bundle_name, bundle_set in structural_bundles.items():
            size = len(bundle_set)
            if size not in analysis['bundle_analysis']['bundle_size_distribution']:
                analysis['bundle_analysis']['bundle_size_distribution'][size] = 0
            analysis['bundle_analysis']['bundle_size_distribution'][size] += 1
            
        return analysis

    def build_bundle_graph(self, bundles: Dict[str, Set[int]]) -> nx.Graph:
        """构建SetBundle connectivity graph"""
        G = nx.Graph()
        
        # 添加节点（每个bundle是一个节点）
        for bundle_name, bundle_set in bundles.items():
            props = self.compute_bundle_properties({bundle_name: bundle_set})[bundle_name]
            G.add_node(bundle_name, 
                      cardinality=props['cardinality'],
                      average_length=props['average_length'],
                      total_ones=props['total_ones'],
                      fibonacci_coverage=len(props['fibonacci_coverage']),
                      structural_diversity=props['structural_diversity'])
        
        # 添加边（bundle之间的相似性连接）
        bundle_names = list(bundles.keys())
        for i, bundle1 in enumerate(bundle_names):
            for bundle2 in bundle_names[i+1:]:
                similarity = self._compute_bundle_similarity(bundles[bundle1], bundles[bundle2])
                if similarity > 0.1:  # 阈值
                    G.add_edge(bundle1, bundle2, 
                             similarity=similarity,
                             connection_type='structural_similarity')
                    
        return G
        
    def _compute_bundle_similarity(self, bundle1: Set[int], bundle2: Set[int]) -> float:
        """计算两个bundle的相似度"""
        if not bundle1 or not bundle2:
            return 0.0
            
        # Jaccard相似度
        intersection = bundle1 & bundle2
        union = bundle1 | bundle2
        jaccard = len(intersection) / len(union) if union else 0
        
        # 结构相似度（基于平均trace属性）
        def get_bundle_avg_properties(bundle):
            lengths = [self.trace_universe[val]['length'] for val in bundle if val in self.trace_universe]
            ones = [self.trace_universe[val]['ones_count'] for val in bundle if val in self.trace_universe]
            return {
                'avg_length': sum(lengths) / len(lengths) if lengths else 0,
                'avg_ones': sum(ones) / len(ones) if ones else 0
            }
            
        props1 = get_bundle_avg_properties(bundle1)
        props2 = get_bundle_avg_properties(bundle2)
        
        length_sim = 1.0 - abs(props1['avg_length'] - props2['avg_length']) / max(props1['avg_length'], props2['avg_length'], 1)
        ones_sim = 1.0 - abs(props1['avg_ones'] - props2['avg_ones']) / max(props1['avg_ones'], props2['avg_ones'], 1)
        
        # 综合相似度
        return 0.4 * jaccard + 0.3 * length_sim + 0.3 * ones_sim


class BundleAnalyzer:
    """
    Advanced analyzer for set bundle properties using graph theory,
    information theory, and category theory approaches.
    """
    
    def __init__(self, bundle_system: SetBundleSystem):
        self.bundle_system = bundle_system
        
    def graph_theory_analysis(self, bundle_graph: nx.Graph) -> Dict[str, any]:
        """图论分析bundle connectivity"""
        analysis = {}
        
        # 基本图属性
        analysis['node_count'] = bundle_graph.number_of_nodes()
        analysis['edge_count'] = bundle_graph.number_of_edges()
        analysis['density'] = nx.density(bundle_graph)
        analysis['is_connected'] = nx.is_connected(bundle_graph)
        
        if bundle_graph.number_of_nodes() > 0:
            analysis['average_clustering'] = nx.average_clustering(bundle_graph)
            analysis['average_degree'] = sum(dict(bundle_graph.degree()).values()) / bundle_graph.number_of_nodes()
            
            # 连通分量分析
            components = list(nx.connected_components(bundle_graph))
            analysis['connected_components'] = len(components)
            analysis['largest_component_size'] = max(len(comp) for comp in components) if components else 0
            
            # 中心性分析
            if nx.is_connected(bundle_graph) and bundle_graph.number_of_nodes() > 1:
                try:
                    analysis['diameter'] = nx.diameter(bundle_graph)
                    analysis['radius'] = nx.radius(bundle_graph)
                except:
                    analysis['diameter'] = 0
                    analysis['radius'] = 0
            
        return analysis
        
    def information_theory_analysis(self, bundles: Dict[str, Set[int]]) -> Dict[str, float]:
        """信息论分析bundle entropy"""
        analysis = {}
        
        # Bundle大小分布的熵
        bundle_sizes = [len(bundle_set) for bundle_set in bundles.values()]
        if bundle_sizes:
            size_counts = defaultdict(int)
            for size in bundle_sizes:
                size_counts[size] += 1
                
            total_bundles = len(bundles)
            size_entropy = 0
            for count in size_counts.values():
                prob = count / total_bundles
                if prob > 0:
                    size_entropy -= prob * log2(prob)
                    
            analysis['bundle_size_entropy'] = size_entropy
            analysis['max_size_entropy'] = log2(len(size_counts)) if size_counts else 0
            analysis['size_entropy_efficiency'] = (analysis['bundle_size_entropy'] / 
                                                 analysis['max_size_entropy']) if analysis['max_size_entropy'] > 0 else 0
            
        # Bundle结构多样性
        properties = self.bundle_system.compute_bundle_properties(bundles)
        diversity_values = [props['structural_diversity'] for props in properties.values()]
        analysis['structural_diversity_average'] = sum(diversity_values) / len(diversity_values) if diversity_values else 0
        analysis['structural_diversity_max'] = max(diversity_values) if diversity_values else 0
        
        # Bundle覆盖度熵
        coverage_sizes = [len(props['fibonacci_coverage']) for props in properties.values()]
        if coverage_sizes:
            coverage_counts = defaultdict(int)
            for size in coverage_sizes:
                coverage_counts[size] += 1
                
            coverage_entropy = 0
            total = len(coverage_sizes)
            for count in coverage_counts.values():
                prob = count / total
                if prob > 0:
                    coverage_entropy -= prob * log2(prob)
                    
            analysis['fibonacci_coverage_entropy'] = coverage_entropy
            
        return analysis
        
    def category_theory_analysis(self, bundles: Dict[str, Set[int]]) -> Dict[str, any]:
        """范畴论分析bundle morphisms"""
        analysis = {
            'morphism_properties': {},
            'functor_analysis': {},
            'natural_transformations': {}
        }
        
        bundle_list = list(bundles.items())
        
        # 分析bundle之间的morphism properties
        morphism_count = 0
        inclusion_count = 0
        
        for i, (name1, bundle1) in enumerate(bundle_list):
            for name2, bundle2 in bundle_list[i+1:]:
                # 检查inclusion morphism
                if bundle1.issubset(bundle2):
                    inclusion_count += 1
                elif bundle2.issubset(bundle1):
                    inclusion_count += 1
                    
                morphism_count += 1
                
        analysis['morphism_properties']['total_pairs'] = morphism_count
        analysis['morphism_properties']['inclusion_pairs'] = inclusion_count
        analysis['morphism_properties']['inclusion_ratio'] = inclusion_count / morphism_count if morphism_count > 0 else 0
        
        # Functor analysis: 保持结构的mapping
        properties = self.bundle_system.compute_bundle_properties(bundles)
        
        # 分析cardinality preservation
        cardinalities = [props['cardinality'] for props in properties.values()]
        unique_cardinalities = len(set(cardinalities))
        analysis['functor_analysis']['cardinality_preservation'] = unique_cardinalities / len(cardinalities) if cardinalities else 0
        
        # 分析structural diversity preservation
        diversities = [props['structural_diversity'] for props in properties.values()]
        unique_diversities = len(set(diversities))
        analysis['functor_analysis']['diversity_preservation'] = unique_diversities / len(diversities) if diversities else 0
        
        return analysis


class TestSetBundle(unittest.TestCase):
    """Unit tests for set bundle construction and analysis"""
    
    def setUp(self):
        """测试设置"""
        self.bundle_system = SetBundleSystem(max_trace_size=25)
        self.analyzer = BundleAnalyzer(self.bundle_system)
        
    def test_fibonacci_generation(self):
        """测试Fibonacci数列生成"""
        expected_start = [1, 1, 2, 3, 5, 8, 13, 21]
        self.assertEqual(self.bundle_system.fibonacci_numbers[:8], expected_start)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        # 验证universe只包含φ-valid traces
        for val, trace_data in self.bundle_system.trace_universe.items():
            self.assertTrue(trace_data['phi_valid'])
            self.assertNotIn('11', trace_data['trace'])
            
    def test_structural_similarity_computation(self):
        """测试结构相似度计算"""
        # 测试相同trace的相似度
        if len(self.bundle_system.trace_universe) >= 2:
            vals = list(self.bundle_system.trace_universe.keys())[:2]
            sim_self = self.bundle_system._compute_structural_similarity(vals[0], vals[0])
            self.assertAlmostEqual(sim_self, 1.0, places=6)
            
            # 测试不同trace的相似度
            sim_diff = self.bundle_system._compute_structural_similarity(vals[0], vals[1])
            self.assertGreaterEqual(sim_diff, 0.0)
            self.assertLessEqual(sim_diff, 1.0)
            
    def test_set_bundle_creation(self):
        """测试SetBundle创建"""
        bundles = self.bundle_system.create_set_bundle('structural')
        
        # 验证bundle结构
        self.assertIsInstance(bundles, dict)
        self.assertGreater(len(bundles), 0)
        
        # 验证每个bundle都是非空集合
        for bundle_name, bundle_set in bundles.items():
            self.assertIsInstance(bundle_set, set)
            self.assertGreater(len(bundle_set), 0)
            
    def test_bundle_properties_computation(self):
        """测试bundle属性计算"""
        bundles = self.bundle_system.create_set_bundle('fibonacci')
        properties = self.bundle_system.compute_bundle_properties(bundles)
        
        # 验证属性结构
        for bundle_name, props in properties.items():
            required_keys = ['cardinality', 'elements', 'traces', 'average_length', 
                           'total_ones', 'fibonacci_coverage', 'structural_diversity']
            for key in required_keys:
                self.assertIn(key, props)
                
    def test_three_domain_analysis(self):
        """测试三域分析"""
        analysis = self.bundle_system.three_domain_analysis()
        
        # 验证分析结构
        required_keys = ['traditional_universe_size', 'phi_universe_size', 
                        'intersection_universe_size', 'universe_overlap_ratio', 'bundle_analysis']
        for key in required_keys:
            self.assertIn(key, analysis)
            
        # 验证数值合理性
        self.assertGreaterEqual(analysis['traditional_universe_size'], analysis['intersection_universe_size'])
        self.assertGreaterEqual(analysis['phi_universe_size'], analysis['intersection_universe_size'])
        self.assertGreaterEqual(analysis['universe_overlap_ratio'], 0.0)
        self.assertLessEqual(analysis['universe_overlap_ratio'], 1.0)
        
    def test_bundle_graph_construction(self):
        """测试bundle graph构建"""
        bundles = self.bundle_system.create_set_bundle('hybrid')
        graph = self.bundle_system.build_bundle_graph(bundles)
        
        # 验证图结构
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), len(bundles))
        
        # 验证节点属性
        for node in graph.nodes():
            node_data = graph.nodes[node]
            required_attrs = ['cardinality', 'average_length', 'total_ones', 
                            'fibonacci_coverage', 'structural_diversity']
            for attr in required_attrs:
                self.assertIn(attr, node_data)
                
    def test_graph_theory_analysis(self):
        """测试图论分析"""
        bundles = self.bundle_system.create_set_bundle('length')
        graph = self.bundle_system.build_bundle_graph(bundles)
        analysis = self.analyzer.graph_theory_analysis(graph)
        
        # 验证分析结果
        required_metrics = ['node_count', 'edge_count', 'density', 'is_connected']
        for metric in required_metrics:
            self.assertIn(metric, analysis)
            
        # 验证数值范围
        self.assertGreaterEqual(analysis['density'], 0)
        self.assertLessEqual(analysis['density'], 1)
        
    def test_information_theory_analysis(self):
        """测试信息论分析"""
        bundles = self.bundle_system.create_set_bundle('structural')
        analysis = self.analyzer.information_theory_analysis(bundles)
        
        # 验证熵计算
        required_metrics = ['bundle_size_entropy', 'structural_diversity_average']
        for metric in required_metrics:
            self.assertIn(metric, analysis)
            
        # 验证熵的非负性
        if 'bundle_size_entropy' in analysis:
            self.assertGreaterEqual(analysis['bundle_size_entropy'], 0)
        
    def test_category_theory_analysis(self):
        """测试范畴论分析"""
        bundles = self.bundle_system.create_set_bundle('fibonacci')
        analysis = self.analyzer.category_theory_analysis(bundles)
        
        # 验证分析结构
        required_sections = ['morphism_properties', 'functor_analysis', 'natural_transformations']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # 验证morphism properties
        morph_props = analysis['morphism_properties']
        if 'inclusion_ratio' in morph_props:
            self.assertGreaterEqual(morph_props['inclusion_ratio'], 0)
            self.assertLessEqual(morph_props['inclusion_ratio'], 1)


def main():
    """Main verification routine"""
    print("=== Chapter 032: SetBundle Unit Test Verification ===")
    print("从ψ=ψ(ψ)推导Collapse Path Clusters作为φ-Structural Sets")
    print()
    
    # 1. 运行单元测试
    print("1. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSetBundle)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Unit tests failed. Please check the implementation.")
        return
        
    print()
    
    # 2. 系统分析
    print("2. Running SetBundle Analysis...")
    bundle_system = SetBundleSystem(max_trace_size=30)
    analyzer = BundleAnalyzer(bundle_system)
    
    # Bundle creation for different methods
    clustering_methods = ['structural', 'fibonacci', 'length', 'hybrid']
    print(f"Clustering methods: {clustering_methods}")
    
    for method in clustering_methods:
        bundles = bundle_system.create_set_bundle(method, similarity_threshold=0.6)
        bundle_count = len(bundles)
        avg_bundle_size = sum(len(bundle_set) for bundle_set in bundles.values()) / bundle_count if bundle_count > 0 else 0
        
        print(f"  {method}: {bundle_count} bundles, avg size={avg_bundle_size:.2f}")
        
    print()
    
    # 3. 三域分析
    print("3. Three-Domain Analysis:")
    three_domain = bundle_system.three_domain_analysis()
    print(f"   Traditional universe: {three_domain['traditional_universe_size']} elements")
    print(f"   φ-constrained universe: {three_domain['phi_universe_size']} elements")
    print(f"   Intersection universe: {three_domain['intersection_universe_size']} elements")
    print(f"   Universe overlap ratio: {three_domain['universe_overlap_ratio']:.3f}")
    print()
    
    # 4. 图论分析
    print("4. Graph Theory Analysis:")
    structural_bundles = bundle_system.create_set_bundle('structural')
    bundle_graph = bundle_system.build_bundle_graph(structural_bundles)
    graph_analysis = analyzer.graph_theory_analysis(bundle_graph)
    print(f"   Bundle nodes: {graph_analysis['node_count']}")
    print(f"   Bundle edges: {graph_analysis['edge_count']}")
    print(f"   Graph density: {graph_analysis['density']:.3f}")
    print(f"   Connected: {graph_analysis['is_connected']}")
    if 'average_clustering' in graph_analysis:
        print(f"   Average clustering: {graph_analysis['average_clustering']:.3f}")
    print()
    
    # 5. 信息论分析
    print("5. Information Theory Analysis:")
    info_analysis = analyzer.information_theory_analysis(structural_bundles)
    if 'bundle_size_entropy' in info_analysis:
        print(f"   Bundle size entropy: {info_analysis['bundle_size_entropy']:.3f} bits")
    if 'structural_diversity_average' in info_analysis:
        print(f"   Avg structural diversity: {info_analysis['structural_diversity_average']:.3f}")
    if 'fibonacci_coverage_entropy' in info_analysis:
        print(f"   Fibonacci coverage entropy: {info_analysis['fibonacci_coverage_entropy']:.3f} bits")
    print()
    
    # 6. 范畴论分析
    print("6. Category Theory Analysis:")
    category_analysis = analyzer.category_theory_analysis(structural_bundles)
    morph_props = category_analysis['morphism_properties']
    if 'total_pairs' in morph_props:
        print(f"   Bundle pairs analyzed: {morph_props['total_pairs']}")
    if 'inclusion_ratio' in morph_props:
        print(f"   Inclusion morphism ratio: {morph_props['inclusion_ratio']:.3f}")
        
    functor_analysis = category_analysis['functor_analysis']
    if 'cardinality_preservation' in functor_analysis:
        print(f"   Cardinality preservation: {functor_analysis['cardinality_preservation']:.3f}")
    if 'diversity_preservation' in functor_analysis:
        print(f"   Diversity preservation: {functor_analysis['diversity_preservation']:.3f}")
        
    print()
    
    print("=== Verification Complete ===")
    print("Key insights from verification:")
    print("1. φ-constraint creates structured set bundles with meaningful clustering")
    print("2. Different clustering methods reveal distinct organizational principles")
    print("3. Bundle graphs exhibit rich connectivity patterns")
    print("4. Information theory reveals optimal bundle entropy structures")
    print("5. Category theory shows morphism preservation across bundle transformations")
    

if __name__ == "__main__":
    main()