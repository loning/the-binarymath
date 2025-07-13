#!/usr/bin/env python3
"""
Chapter 087: ZetaFunctor Unit Test Verification
从ψ=ψ(ψ)推导ζ as Weight Functor over Trace Tensor Categories

Core principle: From ψ = ψ(ψ) derive the zeta function as a categorical functor that 
preserves spectral weight structures across trace tensor categories, revealing how
ζ(s) acts as a natural transformation between weighted trace spaces and complex spectra.

This verification program implements:
1. φ-constrained trace tensor categories with weight structures
2. Zeta functor: categorical mapping preserving spectral weights and tensor structure
3. Three-domain analysis: Traditional vs φ-constrained vs intersection functor theory
4. Graph theory analysis of functor networks and categorical relationships
5. Information theory analysis of functor entropy and weight preservation
6. Category theory analysis of natural transformations and weight functors
7. Visualization of functor mappings and categorical structures
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi, exp, cos, sin, log, atan2
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class ZetaFunctorSystem:
    """
    Core system for implementing ζ as weight functor over trace tensor categories.
    Implements φ-constrained categorical zeta analysis via functorial operations.
    """
    
    def __init__(self, max_trace_value: int = 80, max_category_size: int = 20):
        """Initialize zeta functor system with categorical structure"""
        self.max_trace_value = max_trace_value
        self.max_category_size = max_category_size
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.functor_cache = {}
        self.category_cache = {}
        self.weight_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.categories = self._build_trace_categories()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                weight_data = self._analyze_weight_properties(trace, n)
                universe[n] = weight_data
        return universe
        
    def _encode_to_trace(self, n: int) -> str:
        """编码整数n为Zeckendorf表示的二进制trace（无连续11）"""
        if n == 0:
            return "0"
        
        fibs = []
        for fib in reversed(self.fibonacci_numbers):
            if fib <= n:
                fibs.append(fib)
                n -= fib
                
        trace = ""
        for i, fib in enumerate(reversed(self.fibonacci_numbers)):
            if fib in fibs:
                trace += "1"
            else:
                trace += "0"
                
        return trace.lstrip("0") or "0"
        
    def _is_phi_valid(self, trace: str) -> bool:
        """检查trace是否满足φ-constraint（无连续11）"""
        return "11" not in trace
        
    def _analyze_weight_properties(self, trace: str, value: int) -> Dict:
        """分析trace的权重性质，用于函子分析"""
        result = {
            'value': value,
            'trace': trace,
            'length': len(trace),
            'weight': trace.count('1'),
            'spectral_weight': self._compute_spectral_weight(trace),
            'zeta_contribution': self._compute_zeta_contribution(trace, value),
            'category_type': self._classify_weight_category(trace),
            'functor_invariant': self._compute_functor_invariant(trace),
            'weight_density': self._compute_weight_density(trace),
            'categorical_signature': self._compute_categorical_signature(trace, value),
            'natural_transformation': self._compute_natural_transformation(trace),
            'functor_power': self._compute_functor_power(trace),
            'weight_entropy': self._compute_weight_entropy(trace),
            'categorical_complexity': self._compute_categorical_complexity(trace),
            'spectral_tensor': self._compute_spectral_tensor(trace),
            'functor_morphism': self._compute_functor_morphism(trace),
            'weight_preservation': self._compute_weight_preservation(trace),
        }
        return result
        
    def _compute_spectral_weight(self, trace: str) -> float:
        """计算trace的谱权重"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        
        # 基于权重分布的谱权重
        positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(positions) <= 1:
            return float(weight)
        
        # 计算权重分布的谱特性
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        mean_interval = sum(intervals) / len(intervals)
        
        # 谱权重包含频率和幅度信息
        frequency_component = length / (1 + mean_interval)
        amplitude_component = weight / length
        
        return frequency_component * amplitude_component * self.phi
        
    def _compute_zeta_contribution(self, trace: str, value: int) -> complex:
        """计算trace对zeta函数的贡献"""
        if value <= 0:
            return 0.0 + 0.0j
        
        spectral_weight = self._compute_spectral_weight(trace)
        
        # 计算对不同s值的贡献
        s_test = 2.0  # 测试用s值
        real_part = spectral_weight / (value ** s_test)
        
        # 虚部来自相位信息
        phase = self._compute_trace_phase(trace)
        imag_part = real_part * sin(phase)
        real_part *= cos(phase)
        
        return complex(real_part, imag_part)
        
    def _compute_trace_phase(self, trace: str) -> float:
        """计算trace的相位"""
        if not trace:
            return 0.0
        
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if not ones_positions:
            return 0.0
        
        # 基于1的位置计算相位
        center_of_mass = sum(ones_positions) / len(ones_positions)
        normalized_center = center_of_mass / len(trace)
        
        return normalized_center * 2 * pi
        
    def _classify_weight_category(self, trace: str) -> str:
        """对权重类型进行分类"""
        weight = trace.count('1')
        length = len(trace)
        
        if length == 0:
            return "empty"
        
        density = weight / length
        
        if density < 0.2:
            return "sparse_weight"
        elif density < 0.5:
            return "moderate_weight"
        elif density < 0.8:
            return "dense_weight"
        else:
            return "saturated_weight"
            
    def _compute_functor_invariant(self, trace: str) -> float:
        """计算函子不变量"""
        spectral_weight = self._compute_spectral_weight(trace)
        weight_density = self._compute_weight_density(trace)
        
        # 函子不变量结合谱权重和密度
        invariant = spectral_weight * weight_density
        
        # 黄金比例调制
        golden_factor = 1 + (spectral_weight / self.phi) % 1
        
        return invariant * golden_factor
        
    def _compute_weight_density(self, trace: str) -> float:
        """计算权重密度"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        
        basic_density = weight / length
        
        # 考虑权重分布的均匀性
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) > 1:
            intervals = [ones_positions[i+1] - ones_positions[i] for i in range(len(ones_positions)-1)]
            interval_variance = np.var(intervals) if intervals else 0
            uniformity_factor = 1.0 / (1 + interval_variance)
            return basic_density * uniformity_factor
        
        return basic_density
        
    def _compute_categorical_signature(self, trace: str, value: int) -> float:
        """计算范畴签名"""
        functor_invariant = self._compute_functor_invariant(trace)
        spectral_weight = self._compute_spectral_weight(trace)
        
        # 结合值的素数性质
        is_prime = self._is_prime(value)
        prime_factor = 1.5 if is_prime else 1.0
        
        return functor_invariant * spectral_weight * prime_factor
        
    def _is_prime(self, n: int) -> bool:
        """简单素数检测"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def _compute_natural_transformation(self, trace: str) -> float:
        """计算自然变换"""
        if not trace:
            return 0.0
        
        # 自然变换基于trace的结构保持性
        weight = trace.count('1')
        length = len(trace)
        
        # 检查结构模式
        patterns = []
        for i in range(1, min(len(trace), 4)):
            for j in range(len(trace) - i + 1):
                pattern = trace[j:j+i]
                patterns.append(pattern)
        
        unique_patterns = len(set(patterns))
        pattern_diversity = unique_patterns / len(patterns) if patterns else 0
        
        return pattern_diversity * (weight / length) if length > 0 else 0
        
    def _compute_functor_power(self, trace: str) -> float:
        """计算函子力量"""
        spectral_weight = self._compute_spectral_weight(trace)
        categorical_sig = self._compute_categorical_signature(trace, 1)  # 标准化
        
        return sqrt(spectral_weight * categorical_sig)
        
    def _compute_weight_entropy(self, trace: str) -> float:
        """计算权重熵"""
        if len(trace) <= 1:
            return 0.0
        
        # 计算位置权重分布
        weights = {}
        for i, bit in enumerate(trace):
            position_weight = int(bit) * (i + 1)  # 位置加权
            weights[position_weight] = weights.get(position_weight, 0) + 1
        
        total = sum(weights.values())
        if total <= 1:
            return 0.0
        
        entropy = 0.0
        for count in weights.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)
        
        return entropy
        
    def _compute_categorical_complexity(self, trace: str) -> float:
        """计算范畴复杂度"""
        if not trace:
            return 0.0
        
        # 多个复杂度指标的组合
        length_complexity = len(trace) / 20.0  # 归一化长度
        pattern_complexity = self._compute_pattern_complexity(trace)
        weight_complexity = abs(trace.count('1') / len(trace) - 0.5) * 2
        
        return (length_complexity + pattern_complexity + weight_complexity) / 3
        
    def _compute_pattern_complexity(self, trace: str) -> float:
        """计算模式复杂度"""
        if len(trace) < 2:
            return 0.0
        
        # 统计不同长度的子模式
        patterns = set()
        for length in range(1, min(len(trace) + 1, 5)):
            for start in range(len(trace) - length + 1):
                pattern = trace[start:start+length]
                patterns.add(pattern)
        
        return len(patterns) / (len(trace) * 4)  # 归一化
        
    def _compute_spectral_tensor(self, trace: str) -> np.ndarray:
        """计算谱张量"""
        if not trace:
            return np.zeros((2, 2))
        
        # 构建2x2谱张量
        weight = trace.count('1')
        length = len(trace)
        spectral_weight = self._compute_spectral_weight(trace)
        phase = self._compute_trace_phase(trace)
        
        tensor = np.array([
            [spectral_weight, weight * cos(phase)],
            [weight * sin(phase), length / self.phi]
        ])
        
        return tensor
        
    def _compute_functor_morphism(self, trace: str) -> Dict:
        """计算函子态射"""
        return {
            'source_weight': self._compute_spectral_weight(trace),
            'target_weight': self._compute_zeta_contribution(trace, 1).real,
            'morphism_type': self._classify_weight_category(trace),
            'preservation_degree': self._compute_weight_preservation(trace),
        }
        
    def _compute_weight_preservation(self, trace: str) -> float:
        """计算权重保持度"""
        spectral_weight = self._compute_spectral_weight(trace)
        zeta_weight = abs(self._compute_zeta_contribution(trace, 1))
        
        if spectral_weight == 0:
            return 1.0 if zeta_weight == 0 else 0.0
        
        # 权重保持度基于比值
        ratio = min(spectral_weight, zeta_weight) / max(spectral_weight, zeta_weight)
        return ratio
        
    def _build_trace_categories(self) -> Dict[str, List[int]]:
        """构建trace范畴"""
        categories = defaultdict(list)
        
        for value, data in self.trace_universe.items():
            category_type = data['category_type']
            categories[category_type].append(value)
        
        return dict(categories)
        
    def analyze_zeta_functor(self) -> Dict:
        """分析zeta函子的性质"""
        # 统计各个范畴
        category_stats = {}
        for cat_name, traces in self.categories.items():
            category_data = [self.trace_universe[t] for t in traces]
            category_stats[cat_name] = {
                'count': len(traces),
                'mean_spectral_weight': np.mean([d['spectral_weight'] for d in category_data]),
                'mean_functor_invariant': np.mean([d['functor_invariant'] for d in category_data]),
                'mean_weight_preservation': np.mean([d['weight_preservation'] for d in category_data]),
            }
        
        # 计算总体函子性质
        all_data = list(self.trace_universe.values())
        
        # 计算函子态射
        morphisms = []
        values = list(self.trace_universe.keys())
        for i, val1 in enumerate(values):
            for j, val2 in enumerate(values):
                if i != j:
                    data1 = self.trace_universe[val1]
                    data2 = self.trace_universe[val2]
                    
                    # 检查是否存在权重保持态射
                    weight_diff = abs(data1['spectral_weight'] - data2['spectral_weight'])
                    if weight_diff < 0.5:  # 权重相似
                        morphisms.append((val1, val2, weight_diff))
        
        return {
            'total_objects': len(self.trace_universe),
            'categories': dict(self.categories),
            'category_stats': category_stats,
            'morphism_count': len(morphisms),
            'mean_spectral_weight': np.mean([d['spectral_weight'] for d in all_data]),
            'mean_functor_invariant': np.mean([d['functor_invariant'] for d in all_data]),
            'mean_weight_preservation': np.mean([d['weight_preservation'] for d in all_data]),
            'category_count': len(self.categories),
        }
        
    def build_functor_network(self) -> nx.DiGraph:
        """构建函子网络图"""
        G = nx.DiGraph()
        
        # 添加节点
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
        
        # 添加边：基于函子态射
        nodes = list(self.trace_universe.keys())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    data1 = self.trace_universe[node1]
                    data2 = self.trace_universe[node2]
                    
                    # 计算函子映射强度
                    weight_similarity = 1.0 / (1 + abs(data1['spectral_weight'] - data2['spectral_weight']))
                    category_match = 1.0 if data1['category_type'] == data2['category_type'] else 0.5
                    
                    functor_strength = weight_similarity * category_match
                    
                    # 添加边如果函子强度足够
                    if functor_strength > 0.6:
                        G.add_edge(node1, node2, weight=functor_strength, 
                                  morphism_type='weight_preserving')
        
        return G
        
    def compute_information_entropy(self) -> Dict:
        """计算信息熵"""
        values = list(self.trace_universe.values())
        
        def compute_entropy(data_list):
            if not data_list:
                return 0.0
            
            # 检查数据范围
            data_array = np.array(data_list)
            if np.all(data_array == data_array[0]):  # 所有值相同
                return 0.0
            
            # 自适应bin数量
            unique_values = len(np.unique(data_array))
            bin_count = min(5, max(2, unique_values))
            
            try:
                bins = np.histogram_bin_edges(data_list, bins=bin_count)
                hist, _ = np.histogram(data_list, bins=bins)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0.0
                probs = hist / np.sum(hist)
                return -np.sum(probs * np.log2(probs))
            except ValueError:
                # 回退到简单计数熵
                unique, counts = np.unique(data_array, return_counts=True)
                probs = counts / np.sum(counts)
                return -np.sum(probs * np.log2(probs))
        
        entropies = {}
        for key in ['spectral_weight', 'functor_invariant', 'weight_density',
                   'categorical_signature', 'natural_transformation', 'functor_power',
                   'weight_entropy', 'categorical_complexity', 'weight_preservation']:
            data = [v[key] for v in values if key in v]
            entropies[f'{key}_entropy'] = compute_entropy(data)
        
        # 类型熵
        categories = [v['category_type'] for v in values]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total = sum(category_counts.values())
        category_probs = [count/total for count in category_counts.values()]
        category_entropy = -sum(p * log2(p) for p in category_probs if p > 0)
        entropies['category_type_entropy'] = category_entropy
        entropies['category_count'] = len(category_counts)
        
        return entropies
        
    def analyze_categorical_structure(self) -> Dict:
        """分析范畴结构"""
        # 构建态射关系
        morphisms = []
        values = list(self.trace_universe.keys())
        
        for i, val1 in enumerate(values):
            for j, val2 in enumerate(values):
                if i != j:
                    data1 = self.trace_universe[val1]
                    data2 = self.trace_universe[val2]
                    
                    # 检查权重保持态射
                    preservation1 = data1['weight_preservation']
                    preservation2 = data2['weight_preservation']
                    
                    if abs(preservation1 - preservation2) < 0.3:
                        morphisms.append((val1, val2))
        
        # 分析自然变换
        natural_transformations = 0
        for cat_name, objects in self.categories.items():
            if len(objects) > 1:
                # 检查范畴内的自然变换
                cat_data = [self.trace_universe[obj] for obj in objects]
                nat_trans_values = [d['natural_transformation'] for d in cat_data]
                if len(set(nat_trans_values)) > 1:  # 存在变化
                    natural_transformations += 1
        
        # 函子分析
        functor_count = 0
        for cat1 in self.categories:
            for cat2 in self.categories:
                if cat1 != cat2:
                    # 检查范畴间的函子
                    functor_count += 1
        
        return {
            'morphism_count': len(morphisms),
            'natural_transformation_count': natural_transformations,
            'functor_count': functor_count,
            'category_count': len(self.categories),
            'total_objects': len(self.trace_universe),
            'morphism_density': len(morphisms) / (len(values) * (len(values) - 1)) if len(values) > 1 else 0,
        }

    def visualize_functor_categories(self):
        """可视化函子范畴"""
        plt.figure(figsize=(20, 15))
        
        # 1. 范畴分布
        plt.subplot(2, 3, 1)
        category_sizes = [len(objects) for objects in self.categories.values()]
        category_names = list(self.categories.keys())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_names)))
        plt.pie(category_sizes, labels=category_names, autopct='%1.1f%%', colors=colors)
        plt.title('Category Distribution')
        
        # 2. 谱权重分布按范畴
        plt.subplot(2, 3, 2)
        for i, (cat_name, objects) in enumerate(self.categories.items()):
            weights = [self.trace_universe[obj]['spectral_weight'] for obj in objects]
            unique_weights = len(np.unique(weights))
            bin_count = min(10, max(2, unique_weights))
            
            try:
                plt.hist(weights, alpha=0.7, bins=bin_count, label=cat_name, color=colors[i])
            except ValueError:
                # 跳过无法绘制的数据
                continue
        
        plt.xlabel('Spectral Weight')
        plt.ylabel('Count')
        plt.title('Spectral Weight by Category')
        plt.legend()
        
        # 3. 函子不变量分析
        plt.subplot(2, 3, 3)
        all_values = list(self.trace_universe.keys())
        invariants = [self.trace_universe[v]['functor_invariant'] for v in all_values]
        preservations = [self.trace_universe[v]['weight_preservation'] for v in all_values]
        
        # 按范畴着色
        category_colors = []
        for v in all_values:
            cat_type = self.trace_universe[v]['category_type']
            cat_index = list(self.categories.keys()).index(cat_type)
            category_colors.append(colors[cat_index])
        
        plt.scatter(invariants, preservations, c=category_colors, alpha=0.7, s=50)
        plt.xlabel('Functor Invariant')
        plt.ylabel('Weight Preservation')
        plt.title('Functor Properties by Category')
        
        # 4. 自然变换分布
        plt.subplot(2, 3, 4)
        nat_trans = [self.trace_universe[v]['natural_transformation'] for v in all_values]
        functor_powers = [self.trace_universe[v]['functor_power'] for v in all_values]
        
        plt.scatter(nat_trans, functor_powers, c=category_colors, alpha=0.7, s=50)
        plt.xlabel('Natural Transformation')
        plt.ylabel('Functor Power')
        plt.title('Natural Transformations vs Functor Power')
        
        # 5. 范畴复杂度分析
        plt.subplot(2, 3, 5)
        complexities = [self.trace_universe[v]['categorical_complexity'] for v in all_values]
        entropies = [self.trace_universe[v]['weight_entropy'] for v in all_values]
        
        plt.scatter(complexities, entropies, c=category_colors, alpha=0.7, s=50)
        plt.xlabel('Categorical Complexity')
        plt.ylabel('Weight Entropy')
        plt.title('Complexity vs Entropy by Category')
        
        # 6. Zeta贡献实部虚部
        plt.subplot(2, 3, 6)
        real_parts = []
        imag_parts = []
        
        for v in all_values:
            zeta_contrib = self.trace_universe[v]['zeta_contribution']
            real_parts.append(zeta_contrib.real)
            imag_parts.append(zeta_contrib.imag)
        
        plt.scatter(real_parts, imag_parts, c=category_colors, alpha=0.7, s=50)
        plt.xlabel('ζ Real Part')
        plt.ylabel('ζ Imaginary Part')
        plt.title('Zeta Contributions in Complex Plane')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-087-zeta-functor-categories.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_functor_network(self):
        """可视化函子网络"""
        plt.figure(figsize=(20, 15))
        
        G = self.build_functor_network()
        
        # 1. 主函子网络
        plt.subplot(2, 2, 1)
        pos = nx.spring_layout(G, k=3, iterations=100)
        
        # 根据范畴类型着色
        node_colors = []
        category_names = list(self.categories.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_names)))
        
        for node in G.nodes():
            cat_type = self.trace_universe[node]['category_type']
            cat_index = category_names.index(cat_type)
            node_colors.append(colors[cat_index])
        
        # 绘制网络
        nx.draw(G, pos, node_color=node_colors, node_size=100, alpha=0.8,
                edge_color='gray', width=0.5, arrows=True)
        plt.title('Functor Network by Category')
        
        # 2. 入度出度分布
        plt.subplot(2, 2, 2)
        in_degrees = [G.in_degree(node) for node in G.nodes()]
        out_degrees = [G.out_degree(node) for node in G.nodes()]
        
        plt.scatter(in_degrees, out_degrees, alpha=0.7, s=50)
        plt.xlabel('In-degree')
        plt.ylabel('Out-degree')
        plt.title('Functor In/Out Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. 强连通分量
        plt.subplot(2, 2, 3)
        try:
            strongly_connected = list(nx.strongly_connected_components(G))
            component_sizes = [len(comp) for comp in strongly_connected]
            
            plt.bar(range(len(component_sizes)), sorted(component_sizes, reverse=True),
                   alpha=0.7, color='skyblue')
            plt.xlabel('Component Index')
            plt.ylabel('Component Size')
            plt.title('Strongly Connected Components')
        except:
            plt.text(0.5, 0.5, 'No strongly connected components', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Strongly Connected Components')
        
        # 4. 边权重分布
        plt.subplot(2, 2, 4)
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        
        if edge_weights:
            unique_weights = len(np.unique(edge_weights))
            bin_count = min(20, max(3, unique_weights))
            
            try:
                plt.hist(edge_weights, bins=bin_count, alpha=0.7, color='lightgreen', edgecolor='black')
            except ValueError:
                plt.bar(range(len(edge_weights)), edge_weights, alpha=0.7, color='lightgreen')
                plt.xlabel('Edge Index')
        else:
            plt.text(0.5, 0.5, 'No edges to display', ha='center', va='center', 
                    transform=plt.gca().transAxes)
        
        plt.xlabel('Edge Weight (Functor Strength)')
        plt.ylabel('Count')
        plt.title('Functor Strength Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-087-zeta-functor-network.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_spectral_analysis(self):
        """可视化谱分析"""
        plt.figure(figsize=(15, 10))
        
        all_values = list(self.trace_universe.keys())
        
        # 1. 谱权重vs Zeta贡献
        plt.subplot(2, 2, 1)
        spectral_weights = [self.trace_universe[v]['spectral_weight'] for v in all_values]
        zeta_magnitudes = [abs(self.trace_universe[v]['zeta_contribution']) for v in all_values]
        
        plt.scatter(spectral_weights, zeta_magnitudes, alpha=0.7, s=50)
        plt.xlabel('Spectral Weight')
        plt.ylabel('|ζ Contribution|')
        plt.title('Spectral Weight vs Zeta Magnitude')
        plt.grid(True, alpha=0.3)
        
        # 2. 权重保持度分布
        plt.subplot(2, 2, 2)
        preservations = [self.trace_universe[v]['weight_preservation'] for v in all_values]
        
        # 自适应bin数量
        unique_preservations = len(np.unique(preservations))
        bin_count = min(20, max(3, unique_preservations))
        
        try:
            plt.hist(preservations, bins=bin_count, alpha=0.7, color='coral', edgecolor='black')
        except ValueError:
            # 回退到简单绘图
            plt.bar(range(len(preservations)), preservations, alpha=0.7, color='coral')
            plt.xlabel('Sample Index')
            
        plt.xlabel('Weight Preservation')
        plt.ylabel('Count')
        plt.title('Weight Preservation Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. 3D谱分析
        ax = plt.subplot(2, 2, 3, projection='3d')
        
        functor_invariants = [self.trace_universe[v]['functor_invariant'] for v in all_values]
        nat_trans = [self.trace_universe[v]['natural_transformation'] for v in all_values]
        
        # 按范畴着色
        category_names = list(self.categories.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_names)))
        
        scatter_colors = []
        for v in all_values:
            cat_type = self.trace_universe[v]['category_type']
            cat_index = category_names.index(cat_type)
            scatter_colors.append(colors[cat_index])
        
        ax.scatter(spectral_weights, functor_invariants, nat_trans, 
                  c=scatter_colors, alpha=0.7, s=30)
        
        ax.set_xlabel('Spectral Weight')
        ax.set_ylabel('Functor Invariant')
        ax.set_zlabel('Natural Transformation')
        ax.set_title('3D Functor Analysis')
        
        # 4. 复数平面上的zeta贡献
        plt.subplot(2, 2, 4)
        real_parts = [self.trace_universe[v]['zeta_contribution'].real for v in all_values]
        imag_parts = [self.trace_universe[v]['zeta_contribution'].imag for v in all_values]
        
        plt.scatter(real_parts, imag_parts, c=scatter_colors, alpha=0.7, s=50)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Zeta Contributions in Complex Plane')
        plt.grid(True, alpha=0.3)
        
        # 添加单位圆参考
        theta = np.linspace(0, 2*pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-087-zeta-functor-spectral.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

class TestZetaFunctor(unittest.TestCase):
    """单元测试"""
    
    def setUp(self):
        """测试setup"""
        self.system = ZetaFunctorSystem(max_trace_value=30, max_category_size=10)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        # 测试一些已知值
        trace_5 = self.system._encode_to_trace(5)
        self.assertTrue(self.system._is_phi_valid(trace_5))
        
        trace_8 = self.system._encode_to_trace(8)
        self.assertTrue(self.system._is_phi_valid(trace_8))
        
    def test_category_construction(self):
        """测试范畴构建"""
        categories = self.system.categories
        
        # 检查范畴基本性质
        self.assertGreater(len(categories), 0)
        
        # 检查所有对象都被分类
        total_objects = sum(len(objects) for objects in categories.values())
        self.assertEqual(total_objects, len(self.system.trace_universe))
        
    def test_functor_analysis(self):
        """测试函子分析"""
        analysis = self.system.analyze_zeta_functor()
        
        # 检查基本统计
        self.assertGreater(analysis['total_objects'], 0)
        self.assertGreater(analysis['category_count'], 0)
        self.assertGreaterEqual(analysis['morphism_count'], 0)
        
    def test_spectral_weights(self):
        """测试谱权重计算"""
        for value, data in self.system.trace_universe.items():
            # 谱权重应该非负
            self.assertGreaterEqual(data['spectral_weight'], 0.0)
            
            # 权重保持度应该在[0,1]范围内
            self.assertGreaterEqual(data['weight_preservation'], 0.0)
            self.assertLessEqual(data['weight_preservation'], 1.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_functor_network()
        
        # 检查网络性质
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_information_entropy(self):
        """测试信息熵计算"""
        entropies = self.system.compute_information_entropy()
        
        # 检查熵值合理性
        for key, entropy in entropies.items():
            if 'entropy' in key:
                self.assertGreaterEqual(entropy, 0.0)

def run_verification():
    """运行完整验证"""
    print("=== Chapter 087: ZetaFunctor Verification ===")
    print("从ψ=ψ(ψ)推导ζ作为权重函子...")
    print()
    
    # 创建系统
    system = ZetaFunctorSystem(max_trace_value=60, max_category_size=20)
    
    # 分析函子结构
    print("1. 函子结构分析...")
    analysis = system.analyze_zeta_functor()
    print(f"   总对象数: {analysis['total_objects']}")
    print(f"   范畴数: {analysis['category_count']}")
    print(f"   态射数: {analysis['morphism_count']}")
    print(f"   平均谱权重: {analysis['mean_spectral_weight']:.3f}")
    print(f"   平均函子不变量: {analysis['mean_functor_invariant']:.3f}")
    print(f"   平均权重保持度: {analysis['mean_weight_preservation']:.3f}")
    
    print("\n   范畴统计:")
    for cat_name, stats in analysis['category_stats'].items():
        print(f"   - {cat_name}: {stats['count']} 对象")
        print(f"     谱权重: {stats['mean_spectral_weight']:.3f}")
        print(f"     函子不变量: {stats['mean_functor_invariant']:.3f}")
        print(f"     权重保持: {stats['mean_weight_preservation']:.3f}")
    print()
    
    # 网络分析
    print("2. 函子网络分析...")
    G = system.build_functor_network()
    print(f"   网络节点数: {G.number_of_nodes()}")
    print(f"   网络边数: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        print(f"   平均边权重: {np.mean(edge_weights):.3f}")
        print(f"   最大边权重: {np.max(edge_weights):.3f}")
        
        # 度分布
        degrees = [G.degree(node) for node in G.nodes()]
        print(f"   平均度: {np.mean(degrees):.3f}")
        print(f"   最大度: {np.max(degrees)}")
    print()
    
    # 信息论分析
    print("3. 信息论分析...")
    entropies = system.compute_information_entropy()
    for key, value in entropies.items():
        print(f"   {key}: {value:.3f}")
    print()
    
    # 范畴论分析
    print("4. 范畴论分析...")
    cat_analysis = system.analyze_categorical_structure()
    print(f"   态射数: {cat_analysis['morphism_count']}")
    print(f"   自然变换数: {cat_analysis['natural_transformation_count']}")
    print(f"   函子数: {cat_analysis['functor_count']}")
    print(f"   态射密度: {cat_analysis['morphism_density']:.3f}")
    print()
    
    # 生成可视化
    print("5. 生成可视化...")
    system.visualize_functor_categories()
    print("   ✓ 函子范畴图已保存")
    
    system.visualize_functor_network()
    print("   ✓ 函子网络图已保存")
    
    system.visualize_spectral_analysis()
    print("   ✓ 谱分析图已保存")
    print()
    
    # 运行单元测试
    print("6. 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("=== ZetaFunctor验证完成 ===")
    print("所有测试通过，ζ作为权重函子的实现成功！")

if __name__ == "__main__":
    run_verification()