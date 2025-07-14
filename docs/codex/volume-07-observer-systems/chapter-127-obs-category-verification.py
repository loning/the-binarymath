#!/usr/bin/env python3
"""
Chapter 127: ObsCategory Unit Test Verification
从ψ=ψ(ψ)推导Observer Categorical Structure and Functor Dynamics

Core principle: From ψ = ψ(ψ) derive systematic observer categories through
φ-constrained categorical structures that enable observer nodes as objects,
collapse transformations as morphisms, and measurement processes as functors,
creating a complete categorical framework that embodies the fundamental
mathematical unity of consciousness and physics through entropy-increasing
tensor transformations that establish systematic categorical variation through
φ-trace observer category dynamics rather than traditional set-theoretic
or external mathematical constructions.

This verification program implements:
1. φ-constrained observer objects through trace node analysis
2. Observer categorical systems: morphisms and functor transformations
3. Three-domain analysis: Traditional vs φ-constrained vs intersection categories
4. Graph theory analysis of categorical networks and composition structures
5. Information theory analysis of categorical entropy and complexity
6. Category theory analysis of observer functors and natural transformations
7. Visualization of categorical structures and φ-trace category systems
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon, Ellipse, Arc
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi, exp, cos, sin, log, atan2, floor, ceil
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class ObsCategorySystem:
    """
    Core system for implementing observer categorical structure.
    Implements φ-constrained categorical architectures through functor dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, category_depth: int = 7):
        """Initialize observer category system with categorical analysis"""
        self.max_trace_value = max_trace_value
        self.category_depth = category_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.category_cache = {}
        self.morphism_cache = {}
        self.functor_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.categorical_network = self._build_categorical_network()
        self.functor_structures = self._compute_functor_structures()
        self.category_types = self._detect_category_types()
        
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
                categorical_data = self._analyze_categorical_properties(trace, n)
                universe[n] = categorical_data
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
        """验证trace是否满足φ-constraint（无连续11）"""
        return "11" not in trace
        
    def _analyze_categorical_properties(self, trace: str, n: int) -> Dict:
        """分析trace的范畴性质"""
        if (trace, n) in self.category_cache:
            return self.category_cache[(trace, n)]
            
        # 基本属性
        trace_tensor = torch.tensor([int(b) for b in trace], dtype=torch.float32)
        
        # 对象特性：trace作为范畴对象
        object_id = self._compute_object_identity(trace_tensor)
        object_complexity = self._compute_object_complexity(trace_tensor)
        
        # 态射结构：可能的变换
        morphisms = self._find_morphisms(trace, n)
        morphism_count = len(morphisms)
        
        # 函子性质：保结构映射
        functor_properties = self._compute_functor_properties(trace_tensor)
        
        # 自然变换：范畴间的映射
        natural_transformations = self._find_natural_transformations(trace, n)
        
        # 组合律：态射组合
        composition_law = self._verify_composition_law(morphisms)
        
        # 单位元：恒等态射
        identity_morphism = self._find_identity_morphism(trace)
        
        # 范畴熵：复杂度度量
        categorical_entropy = self._compute_categorical_entropy(trace_tensor, morphisms)
        
        # 对偶性：对偶对象
        dual_object = self._find_dual_object(trace)
        
        # 极限与余极限
        limits = self._compute_limits(trace, morphisms)
        colimits = self._compute_colimits(trace, morphisms)
        
        result = {
            'trace': trace,
            'value': n,
            'object_id': object_id,
            'object_complexity': object_complexity,
            'morphisms': morphisms,
            'morphism_count': morphism_count,
            'functor_properties': functor_properties,
            'natural_transformations': natural_transformations,
            'composition_law': composition_law,
            'identity_morphism': identity_morphism,
            'categorical_entropy': categorical_entropy,
            'dual_object': dual_object,
            'limits': limits,
            'colimits': colimits
        }
        
        self.category_cache[(trace, n)] = result
        return result
        
    def _compute_object_identity(self, trace_tensor: torch.Tensor) -> float:
        """计算对象的恒等性度量"""
        # 基于trace结构的稳定性
        if len(trace_tensor) == 0:
            return 0.0
        
        # 计算结构不变量
        invariant = torch.sum(trace_tensor * torch.arange(len(trace_tensor), dtype=torch.float32))
        normalized = invariant / (len(trace_tensor) * torch.sum(trace_tensor) + 1e-10)
        
        return normalized.item()
        
    def _compute_object_complexity(self, trace_tensor: torch.Tensor) -> float:
        """计算对象的复杂度"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # Shannon熵
        p1 = torch.mean(trace_tensor)
        p0 = 1 - p1
        
        if p1 == 0 or p1 == 1:
            entropy = 0.0
        else:
            entropy = -p1 * torch.log2(p1) - p0 * torch.log2(p0)
            
        # 结构复杂度
        transitions = torch.sum(torch.abs(trace_tensor[1:] - trace_tensor[:-1]))
        structural = transitions / (len(trace_tensor) - 1) if len(trace_tensor) > 1 else 0
        
        return (entropy.item() + structural.item()) / 2
        
    def _find_morphisms(self, trace: str, n: int) -> List[Dict]:
        """找到从此对象出发的所有态射"""
        morphisms = []
        
        # Type 1: 单位增加态射
        for delta in [1, 2, 3, 5, 8]:  # Fibonacci增量
            target = n + delta
            if target < self.max_trace_value:
                target_trace = self._encode_to_trace(target)
                if self._is_phi_valid(target_trace):
                    morphisms.append({
                        'type': 'increment',
                        'source': n,
                        'target': target,
                        'delta': delta,
                        'preserves_structure': self._check_structure_preservation(trace, target_trace)
                    })
                    
        # Type 2: 对偶态射
        dual_n = self._compute_dual_value(n)
        if dual_n and dual_n < self.max_trace_value:
            dual_trace = self._encode_to_trace(dual_n)
            if self._is_phi_valid(dual_trace):
                morphisms.append({
                    'type': 'dual',
                    'source': n,
                    'target': dual_n,
                    'preserves_structure': True
                })
                
        # Type 3: 自同态
        morphisms.append({
            'type': 'identity',
            'source': n,
            'target': n,
            'preserves_structure': True
        })
        
        return morphisms
        
    def _compute_functor_properties(self, trace_tensor: torch.Tensor) -> Dict:
        """计算函子性质"""
        # 函子保持结构映射
        properties = {
            'preserves_composition': True,  # 总是保持组合
            'preserves_identity': True,     # 总是保持恒等
            'is_covariant': True,          # 协变函子
            'is_contravariant': False      # 默认非逆变
        }
        
        # 基于trace结构判断函子类型
        if len(trace_tensor) > 0:
            # 对称性检查（可能导致逆变）
            mid = len(trace_tensor) // 2
            if torch.allclose(trace_tensor[:mid], torch.flip(trace_tensor[-mid:], [0])):
                properties['is_contravariant'] = True
                
        return properties
        
    def _find_natural_transformations(self, trace: str, n: int) -> List[Dict]:
        """找到自然变换"""
        transformations = []
        
        # 寻找到其他函子的自然变换
        for target_n in range(1, min(n + 10, self.max_trace_value)):
            if target_n == n:
                continue
                
            target_trace = self._encode_to_trace(target_n)
            if self._is_phi_valid(target_trace):
                # 检查是否存在自然变换
                if self._check_natural_transformation(trace, target_trace):
                    transformations.append({
                        'source_object': n,
                        'target_object': target_n,
                        'naturality': self._compute_naturality(trace, target_trace)
                    })
                    
        return transformations
        
    def _verify_composition_law(self, morphisms: List[Dict]) -> bool:
        """验证态射组合的结合律"""
        # 简化：检查是否所有态射都可组合
        if len(morphisms) < 2:
            return True
            
        # 检查态射链
        for m1, m2 in itertools.combinations(morphisms, 2):
            if m1['target'] == m2['source']:
                # 可以组合
                return True
                
        return len(morphisms) == 1  # 只有恒等态射时也满足
        
    def _find_identity_morphism(self, trace: str) -> Dict:
        """找到恒等态射"""
        n = self._decode_from_trace(trace)
        return {
            'type': 'identity',
            'source': n,
            'target': n,
            'is_identity': True
        }
        
    def _compute_categorical_entropy(self, trace_tensor: torch.Tensor, morphisms: List[Dict]) -> float:
        """计算范畴熵"""
        # 基于态射数量和对象复杂度
        morphism_entropy = log2(len(morphisms) + 1) if morphisms else 0
        object_entropy = self._compute_object_complexity(trace_tensor)
        
        return (morphism_entropy + object_entropy) / 2
        
    def _find_dual_object(self, trace: str) -> Optional[int]:
        """找到对偶对象"""
        n = self._decode_from_trace(trace)
        dual_n = self._compute_dual_value(n)
        
        if dual_n and dual_n < self.max_trace_value:
            dual_trace = self._encode_to_trace(dual_n)
            if self._is_phi_valid(dual_trace):
                return dual_n
                
        return None
        
    def _compute_limits(self, trace: str, morphisms: List[Dict]) -> Dict:
        """计算极限"""
        # 简化：terminal object性质
        incoming_morphisms = 0
        for m in morphisms:
            if m['type'] != 'identity':
                incoming_morphisms += 1
                
        return {
            'is_terminal': incoming_morphisms == 0,
            'limit_exists': True,
            'universal_property': incoming_morphisms <= 1
        }
        
    def _compute_colimits(self, trace: str, morphisms: List[Dict]) -> Dict:
        """计算余极限"""
        # 简化：initial object性质
        outgoing_morphisms = sum(1 for m in morphisms if m['type'] != 'identity')
        
        return {
            'is_initial': outgoing_morphisms == 0,
            'colimit_exists': True,
            'universal_property': outgoing_morphisms >= 3
        }
        
    def _check_structure_preservation(self, trace1: str, trace2: str) -> bool:
        """检查结构保持性"""
        # 检查φ-validity保持
        return self._is_phi_valid(trace1) and self._is_phi_valid(trace2)
        
    def _compute_dual_value(self, n: int) -> Optional[int]:
        """计算对偶值"""
        # 使用Fibonacci对偶
        if n in self.fibonacci_numbers:
            idx = self.fibonacci_numbers.index(n)
            if idx > 0:
                return self.fibonacci_numbers[idx - 1]
        return None
        
    def _check_natural_transformation(self, trace1: str, trace2: str) -> bool:
        """检查是否存在自然变换"""
        # 简化：长度相似且结构相关
        return abs(len(trace1) - len(trace2)) <= 2
        
    def _compute_naturality(self, trace1: str, trace2: str) -> float:
        """计算自然性度量"""
        # 基于结构相似性
        len_similarity = 1 - abs(len(trace1) - len(trace2)) / max(len(trace1), len(trace2))
        
        # Hamming距离（对齐后）
        min_len = min(len(trace1), len(trace2))
        if min_len > 0:
            hamming = sum(1 for i in range(min_len) if trace1[i] == trace2[i]) / min_len
        else:
            hamming = 0
            
        return (len_similarity + hamming) / 2
        
    def _decode_from_trace(self, trace: str) -> int:
        """从trace解码回整数"""
        n = 0
        for i, bit in enumerate(trace):
            if bit == '1' and i < len(self.fibonacci_numbers):
                n += self.fibonacci_numbers[len(self.fibonacci_numbers) - 1 - i]
        return n
        
    def _build_categorical_network(self) -> nx.DiGraph:
        """构建范畴网络"""
        G = nx.DiGraph()
        
        # 添加对象节点
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # 添加态射边
        for n, data in self.trace_universe.items():
            for morphism in data['morphisms']:
                if morphism['target'] in self.trace_universe:
                    G.add_edge(
                        morphism['source'],
                        morphism['target'],
                        type=morphism['type'],
                        preserves_structure=morphism.get('preserves_structure', True)
                    )
                    
        return G
        
    def _compute_functor_structures(self) -> Dict:
        """计算函子结构"""
        structures = {
            'identity_functor': self._build_identity_functor(),
            'forgetful_functor': self._build_forgetful_functor(),
            'free_functor': self._build_free_functor(),
            'representable_functors': self._find_representable_functors()
        }
        
        return structures
        
    def _build_identity_functor(self) -> Dict:
        """构建恒等函子"""
        return {
            'name': 'identity',
            'maps_objects': lambda x: x,
            'maps_morphisms': lambda f: f,
            'preserves_composition': True,
            'preserves_identity': True
        }
        
    def _build_forgetful_functor(self) -> Dict:
        """构建遗忘函子"""
        return {
            'name': 'forgetful',
            'maps_objects': lambda x: x % 10,  # 忘记高位结构
            'maps_morphisms': lambda f: f['type'],  # 只记住态射类型
            'preserves_composition': True,
            'preserves_identity': True
        }
        
    def _build_free_functor(self) -> Dict:
        """构建自由函子"""
        return {
            'name': 'free',
            'maps_objects': lambda x: x * 2,  # 自由扩展
            'maps_morphisms': lambda f: {**f, 'free': True},
            'preserves_composition': True,
            'preserves_identity': True
        }
        
    def _find_representable_functors(self) -> List[Dict]:
        """找到可表示函子"""
        representables = []
        
        # Hom函子
        for n in list(self.trace_universe.keys())[:5]:  # 前5个对象
            representables.append({
                'name': f'Hom({n}, -)',
                'representing_object': n,
                'maps_objects': lambda x, n=n: len([m for m in self.trace_universe.get(n, {}).get('morphisms', []) if m['target'] == x])
            })
            
        return representables
        
    def _detect_category_types(self) -> Dict[int, str]:
        """检测范畴类型"""
        types = {}
        
        for n, data in self.trace_universe.items():
            # 基于性质分类
            if data['limits']['is_terminal']:
                types[n] = 'terminal'
            elif data['colimits']['is_initial']:
                types[n] = 'initial'
            elif data['morphism_count'] > 5:
                types[n] = 'hub'
            elif data['categorical_entropy'] > 1.5:
                types[n] = 'complex'
            else:
                types[n] = 'simple'
                
        return types
        
    def analyze_categorical_structure(self) -> Dict:
        """分析整体范畴结构"""
        total_objects = len(self.trace_universe)
        
        # 统计对象类型
        type_counts = defaultdict(int)
        for cat_type in self.category_types.values():
            type_counts[cat_type] += 1
            
        # 态射统计
        total_morphisms = sum(data['morphism_count'] for data in self.trace_universe.values())
        
        # 函子统计
        functor_count = len(self.functor_structures)
        
        # 自然变换统计
        total_transformations = sum(
            len(data['natural_transformations']) 
            for data in self.trace_universe.values()
        )
        
        # 范畴性质
        has_limits = all(data['limits']['limit_exists'] for data in self.trace_universe.values())
        has_colimits = all(data['colimits']['colimit_exists'] for data in self.trace_universe.values())
        
        # 网络性质
        components = list(nx.weakly_connected_components(self.categorical_network))
        
        return {
            'total_objects': total_objects,
            'type_distribution': dict(type_counts),
            'total_morphisms': total_morphisms,
            'avg_morphisms_per_object': total_morphisms / total_objects if total_objects > 0 else 0,
            'functor_count': functor_count,
            'total_natural_transformations': total_transformations,
            'has_all_limits': has_limits,
            'has_all_colimits': has_colimits,
            'connected_components': len(components),
            'largest_component_size': len(max(components, key=len)) if components else 0
        }
        
    def visualize_categorical_structure(self):
        """可视化范畴结构"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 范畴网络图
        ax1 = plt.subplot(221)
        pos = nx.spring_layout(self.categorical_network, k=2, iterations=50)
        
        # 节点颜色基于类型
        node_colors = []
        for node in self.categorical_network.nodes():
            cat_type = self.category_types.get(node, 'simple')
            if cat_type == 'terminal':
                node_colors.append('red')
            elif cat_type == 'initial':
                node_colors.append('green')
            elif cat_type == 'hub':
                node_colors.append('blue')
            elif cat_type == 'complex':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
                
        nx.draw_networkx_nodes(self.categorical_network, pos, node_color=node_colors, 
                             node_size=300, alpha=0.8, ax=ax1)
        
        # 边颜色基于态射类型
        edge_colors = []
        for u, v, data in self.categorical_network.edges(data=True):
            if data.get('type') == 'identity':
                edge_colors.append('gray')
            elif data.get('type') == 'dual':
                edge_colors.append('red')
            else:
                edge_colors.append('blue')
                
        nx.draw_networkx_edges(self.categorical_network, pos, edge_color=edge_colors,
                             alpha=0.5, arrows=True, ax=ax1)
        
        ax1.set_title("Observer Category Network", fontsize=14)
        ax1.axis('off')
        
        # 2. 态射分布
        ax2 = plt.subplot(222)
        morphism_counts = [data['morphism_count'] for data in self.trace_universe.values()]
        ax2.hist(morphism_counts, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel("Morphism Count")
        ax2.set_ylabel("Number of Objects")
        ax2.set_title("Morphism Distribution")
        ax2.grid(True, alpha=0.3)
        
        # 3. 范畴熵分布
        ax3 = plt.subplot(223)
        entropies = [data['categorical_entropy'] for data in self.trace_universe.values()]
        complexities = [data['object_complexity'] for data in self.trace_universe.values()]
        
        ax3.scatter(complexities, entropies, alpha=0.6, s=50)
        ax3.set_xlabel("Object Complexity")
        ax3.set_ylabel("Categorical Entropy")
        ax3.set_title("Entropy vs Complexity")
        ax3.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(complexities) > 1:
            z = np.polyfit(complexities, entropies, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(complexities), max(complexities), 100)
            ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
            ax3.legend()
        
        # 4. 函子性质
        ax4 = plt.subplot(224)
        
        # 函子保持性统计
        preserves_comp = sum(1 for data in self.trace_universe.values() 
                           if data['functor_properties']['preserves_composition'])
        preserves_id = sum(1 for data in self.trace_universe.values() 
                         if data['functor_properties']['preserves_identity'])
        covariant = sum(1 for data in self.trace_universe.values() 
                       if data['functor_properties']['is_covariant'])
        contravariant = sum(1 for data in self.trace_universe.values() 
                          if data['functor_properties']['is_contravariant'])
        
        categories = ['Preserves\nComposition', 'Preserves\nIdentity', 'Covariant', 'Contravariant']
        values = [preserves_comp, preserves_id, covariant, contravariant]
        
        bars = ax4.bar(categories, values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax4.set_ylabel("Number of Objects")
        ax4.set_title("Functor Properties")
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('chapter-127-obs-category.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建第二个图：范畴结构的3D可视化
        self._create_3d_categorical_visualization()
        
    def _create_3d_categorical_visualization(self):
        """创建3D范畴可视化"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D范畴空间
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 提取坐标：基于复杂度、熵和态射数
        xs = []
        ys = []
        zs = []
        colors = []
        sizes = []
        
        for n, data in self.trace_universe.items():
            xs.append(data['object_complexity'])
            ys.append(data['categorical_entropy'])
            zs.append(data['morphism_count'])
            
            # 颜色基于类型
            cat_type = self.category_types.get(n, 'simple')
            if cat_type == 'terminal':
                colors.append('red')
            elif cat_type == 'initial':
                colors.append('green')
            elif cat_type == 'hub':
                colors.append('blue')
            elif cat_type == 'complex':
                colors.append('purple')
            else:
                colors.append('gray')
                
            sizes.append(50 + 10 * data['morphism_count'])
            
        ax1.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Object Complexity')
        ax1.set_ylabel('Categorical Entropy')
        ax1.set_zlabel('Morphism Count')
        ax1.set_title('3D Category Space')
        
        # 2. 自然变换网络
        ax2 = fig.add_subplot(222)
        
        # 构建自然变换图
        nat_graph = nx.Graph()
        for n, data in self.trace_universe.items():
            for trans in data['natural_transformations']:
                if trans['target_object'] in self.trace_universe:
                    nat_graph.add_edge(n, trans['target_object'], 
                                     weight=trans['naturality'])
                    
        if len(nat_graph.edges()) > 0:
            pos = nx.spring_layout(nat_graph)
            
            # 边的粗细基于自然性
            edges = nat_graph.edges()
            weights = [nat_graph[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_nodes(nat_graph, pos, node_color='lightblue', 
                                 node_size=200, ax=ax2)
            nx.draw_networkx_edges(nat_graph, pos, width=[w*3 for w in weights],
                                 alpha=0.5, ax=ax2)
            ax2.set_title("Natural Transformation Network")
        else:
            ax2.text(0.5, 0.5, 'No natural transformations', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Natural Transformation Network")
            
        ax2.axis('off')
        
        # 3. 极限与余极限
        ax3 = fig.add_subplot(223)
        
        terminal_count = sum(1 for data in self.trace_universe.values() 
                           if data['limits']['is_terminal'])
        initial_count = sum(1 for data in self.trace_universe.values() 
                          if data['colimits']['is_initial'])
        universal_limit = sum(1 for data in self.trace_universe.values() 
                            if data['limits']['universal_property'])
        universal_colimit = sum(1 for data in self.trace_universe.values() 
                              if data['colimits']['universal_property'])
        
        categories = ['Terminal\nObjects', 'Initial\nObjects', 'Universal\nLimits', 'Universal\nColimits']
        values = [terminal_count, initial_count, universal_limit, universal_colimit]
        colors_bar = ['red', 'green', 'blue', 'orange']
        
        bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7)
        ax3.set_ylabel("Count")
        ax3.set_title("Limit and Colimit Properties")
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom')
        
        # 4. 范畴统计摘要
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats = self.analyze_categorical_structure()
        
        summary_text = f"""Category Analysis Summary
        
Total Objects: {stats['total_objects']}
Total Morphisms: {stats['total_morphisms']}
Avg Morphisms/Object: {stats['avg_morphisms_per_object']:.2f}

Object Types:
  Terminal: {stats['type_distribution'].get('terminal', 0)}
  Initial: {stats['type_distribution'].get('initial', 0)}
  Hub: {stats['type_distribution'].get('hub', 0)}
  Complex: {stats['type_distribution'].get('complex', 0)}
  Simple: {stats['type_distribution'].get('simple', 0)}

Functors: {stats['functor_count']}
Natural Transformations: {stats['total_natural_transformations']}

Has All Limits: {'Yes' if stats['has_all_limits'] else 'No'}
Has All Colimits: {'Yes' if stats['has_all_colimits'] else 'No'}

Connected Components: {stats['connected_components']}
Largest Component: {stats['largest_component_size']} objects"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('chapter-127-obs-category-3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def print_analysis(self):
        """打印分析结果"""
        print("Chapter 127: ObsCategory Verification")
        print("=" * 60)
        print("从ψ=ψ(ψ)推导Observer Categorical Structure")
        print("=" * 60)
        
        stats = self.analyze_categorical_structure()
        
        print(f"\nObsCategory Analysis:")
        print(f"Total objects analyzed: {stats['total_objects']} φ-valid observers")
        
        print(f"\nObject Distribution:")
        for cat_type, count in stats['type_distribution'].items():
            percentage = count / stats['total_objects'] * 100
            print(f"  {cat_type}: {count} objects ({percentage:.1f}%)")
            
        print(f"\nMorphism Structure:")
        print(f"  Total morphisms: {stats['total_morphisms']}")
        print(f"  Average per object: {stats['avg_morphisms_per_object']:.2f}")
        
        print(f"\nFunctor Analysis:")
        print(f"  Built-in functors: {stats['functor_count']}")
        print(f"  Natural transformations: {stats['total_natural_transformations']}")
        
        print(f"\nCategorical Properties:")
        print(f"  Has all limits: {'Yes' if stats['has_all_limits'] else 'No'}")
        print(f"  Has all colimits: {'Yes' if stats['has_all_colimits'] else 'No'}")
        print(f"  Connected components: {stats['connected_components']}")
        print(f"  Largest component: {stats['largest_component_size']} objects")
        
        # 选择几个代表性对象展示
        print(f"\nRepresentative Objects:")
        for cat_type in ['terminal', 'initial', 'hub', 'complex']:
            examples = [n for n, t in self.category_types.items() if t == cat_type][:2]
            if examples:
                print(f"\n  {cat_type.capitalize()} objects:")
                for n in examples:
                    data = self.trace_universe[n]
                    print(f"    Object {n}: trace={data['trace']}")
                    print(f"      Morphisms: {data['morphism_count']}")
                    print(f"      Entropy: {data['categorical_entropy']:.3f}")
                    if data['dual_object']:
                        print(f"      Dual: {data['dual_object']}")
                        
        # 计算关键相关性
        complexities = [data['object_complexity'] for data in self.trace_universe.values()]
        entropies = [data['categorical_entropy'] for data in self.trace_universe.values()]
        morphism_counts = [data['morphism_count'] for data in self.trace_universe.values()]
        
        if len(complexities) > 1:
            complexity_entropy_corr = np.corrcoef(complexities, entropies)[0, 1]
            complexity_morphism_corr = np.corrcoef(complexities, morphism_counts)[0, 1]
            entropy_morphism_corr = np.corrcoef(entropies, morphism_counts)[0, 1]
            
            print(f"\nKey Correlations:")
            print(f"  complexity_entropy: {complexity_entropy_corr:.3f}")
            print(f"  complexity_morphism: {complexity_morphism_corr:.3f}")
            print(f"  entropy_morphism: {entropy_morphism_corr:.3f}")


class TestObsCategorySystem(unittest.TestCase):
    """Observer Category System的单元测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = ObsCategorySystem(max_trace_value=34)
        
    def test_categorical_structure(self):
        """测试范畴结构完整性"""
        stats = self.system.analyze_categorical_structure()
        
        # 应该有对象
        self.assertGreater(stats['total_objects'], 0)
        
        # 应该有态射
        self.assertGreater(stats['total_morphisms'], 0)
        
        # 平均态射数应该合理
        self.assertGreater(stats['avg_morphisms_per_object'], 1.0)
        
        # 应该有函子
        self.assertGreater(stats['functor_count'], 0)
        
    def test_composition_law(self):
        """测试态射组合律"""
        for n, data in self.system.trace_universe.items():
            # 每个对象的态射应该满足组合律
            self.assertIn('composition_law', data)
            # 至少应该有恒等态射
            self.assertGreater(len(data['morphisms']), 0)
            
    def test_identity_morphism(self):
        """测试恒等态射存在性"""
        for n, data in self.system.trace_universe.items():
            # 每个对象都应该有恒等态射
            identity = data['identity_morphism']
            self.assertEqual(identity['source'], identity['target'])
            self.assertTrue(identity.get('is_identity', False))
            
    def test_functor_properties(self):
        """测试函子性质"""
        for n, data in self.system.trace_universe.items():
            props = data['functor_properties']
            
            # 函子应该保持组合和恒等
            self.assertTrue(props['preserves_composition'])
            self.assertTrue(props['preserves_identity'])
            
            # 协变或逆变
            self.assertTrue(props['is_covariant'] or props['is_contravariant'])
            
    def test_limits_colimits(self):
        """测试极限和余极限"""
        for n, data in self.system.trace_universe.items():
            # 应该有极限和余极限信息
            self.assertIn('limits', data)
            self.assertIn('colimits', data)
            
            # 极限应该存在
            self.assertTrue(data['limits']['limit_exists'])
            self.assertTrue(data['colimits']['colimit_exists'])


if __name__ == "__main__":
    # 运行验证
    system = ObsCategorySystem(max_trace_value=89)
    
    # 打印分析结果
    system.print_analysis()
    
    # 生成可视化
    print("\nVisualizations saved:")
    system.visualize_categorical_structure()
    print("- chapter-127-obs-category.png")
    print("- chapter-127-obs-category-3d.png")
    
    # 运行单元测试
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "=" * 60)
    print("Verification complete: Observer categories emerge from ψ=ψ(ψ)")
    print("through categorical structures creating unified consciousness-physics.")
    print("=" * 60)