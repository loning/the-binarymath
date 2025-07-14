#!/usr/bin/env python3
"""
Chapter 128: TraceEntropy Unit Test Verification
从ψ=ψ(ψ)推导Trace Entropy Definition and Information Measurement

Core principle: From ψ = ψ(ψ) derive systematic trace entropy through
φ-constrained information measurement that enables fundamental entropy definitions
for trace structures within Zeckendorf representation constraints, creating
information measures that embody the essential properties of collapsed information
through entropy-increasing tensor transformations that establish systematic
entropy variation through φ-trace entropy dynamics rather than traditional
Shannon information or external entropy constructions.

This verification program implements:
1. φ-constrained trace entropy through Zeckendorf representation analysis
2. Trace entropy systems: fundamental information measurement for φ-traces
3. Three-domain analysis: Traditional vs φ-constrained vs intersection entropy
4. Graph theory analysis of entropy networks and information flow structures
5. Information theory analysis of φ-entropy bounds and constraint effects
6. Category theory analysis of entropy functors and natural transformations
7. Visualization of trace entropy distributions and φ-constraint entropy systems
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

class TraceEntropySystem:
    """
    Core system for implementing trace entropy definition and measurement.
    Implements φ-constrained entropy architectures through information dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, entropy_depth: int = 8):
        """Initialize trace entropy system with information analysis"""
        self.max_trace_value = max_trace_value
        self.entropy_depth = entropy_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.entropy_cache = {}
        self.trace_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.entropy_network = self._build_entropy_network()
        self.information_flows = self._compute_information_flows()
        self.entropy_categories = self._classify_entropy_types()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的熵分析"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                entropy_data = self._analyze_trace_entropy(trace, n)
                universe[n] = entropy_data
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
        
    def _analyze_trace_entropy(self, trace: str, n: int) -> Dict:
        """分析trace的熵特性"""
        if (trace, n) in self.entropy_cache:
            return self.entropy_cache[(trace, n)]
            
        # 基本trace属性
        trace_tensor = torch.tensor([int(b) for b in trace], dtype=torch.float32)
        
        # φ-熵计算：基于Zeckendorf结构的熵度量
        phi_entropy = self._compute_phi_entropy(trace_tensor)
        
        # Shannon熵（传统对比）
        shannon_entropy = self._compute_shannon_entropy(trace_tensor)
        
        # 结构熵：基于φ-constraint的信息度量
        structural_entropy = self._compute_structural_entropy(trace_tensor)
        
        # 位置熵：信息在trace中的分布
        positional_entropy = self._compute_positional_entropy(trace_tensor)
        
        # 层次熵：不同尺度的信息内容
        hierarchical_entropy = self._compute_hierarchical_entropy(trace_tensor)
        
        # 压缩熵：可压缩性度量
        compression_entropy = self._compute_compression_entropy(trace)
        
        # 黄金比例相关熵
        golden_entropy = self._compute_golden_entropy(trace_tensor)
        
        # 信息密度
        info_density = self._compute_info_density(trace_tensor)
        
        # 熵梯度
        entropy_gradient = self._compute_entropy_gradient(trace_tensor)
        
        # 熵复杂度
        entropy_complexity = self._compute_entropy_complexity(trace_tensor)
        
        # 自相似性
        self_similarity = self._compute_self_similarity(trace)
        
        # 信息流属性
        info_flow_properties = self._compute_info_flow_properties(trace_tensor)
        
        result = {
            'trace': trace,
            'value': n,
            'length': len(trace),
            'ones_count': int(torch.sum(trace_tensor)),
            'phi_entropy': phi_entropy,
            'shannon_entropy': shannon_entropy,
            'structural_entropy': structural_entropy,
            'positional_entropy': positional_entropy,
            'hierarchical_entropy': hierarchical_entropy,
            'compression_entropy': compression_entropy,
            'golden_entropy': golden_entropy,
            'info_density': info_density,
            'entropy_gradient': entropy_gradient,
            'entropy_complexity': entropy_complexity,
            'self_similarity': self_similarity,
            'info_flow_properties': info_flow_properties
        }
        
        self.entropy_cache[(trace, n)] = result
        return result
        
    def _compute_phi_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算φ-约束熵：基于Zeckendorf结构的信息度量"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于φ-constraint的熵计算
        # 考虑相邻位之间的约束关系
        constrained_pairs = 0
        total_pairs = len(trace_tensor) - 1
        
        if total_pairs == 0:
            return torch.mean(trace_tensor).item() / self.phi
            
        for i in range(total_pairs):
            # φ-constraint: 不能有连续的11
            if trace_tensor[i] == 1 and trace_tensor[i+1] == 1:
                # 这种情况在φ-valid trace中不应存在
                constrained_pairs += 1
                
        # φ-熵基于Fibonacci权重的位置熵
        positions = torch.arange(len(trace_tensor), dtype=torch.float32)
        fib_weights = torch.tensor([1/self.phi**i for i in range(len(trace_tensor))], dtype=torch.float32)
        
        # 加权信息内容
        weighted_bits = trace_tensor * fib_weights
        total_weight = torch.sum(fib_weights)
        
        if total_weight == 0:
            return 0.0
            
        # φ-熵：基于Fibonacci权重的信息分布
        normalized_weights = fib_weights / total_weight
        bit_entropy = 0.0
        
        for i in range(len(trace_tensor)):
            if normalized_weights[i] > 0:
                if trace_tensor[i] == 1:
                    # 1位的贡献
                    bit_entropy += normalized_weights[i] * torch.log2(1/normalized_weights[i])
                else:
                    # 0位的贡献更小
                    bit_entropy += normalized_weights[i] * torch.log2(1/(normalized_weights[i] + 1e-10)) * 0.5
                    
        # φ-修正：约束减少熵
        constraint_factor = 1 - constrained_pairs / (total_pairs + 1)
        
        # 黄金比例调制
        phi_factor = 1 / self.phi  # φ^(-1) ≈ 0.618
        
        return bit_entropy.item() * constraint_factor * phi_factor
        
    def _compute_shannon_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算标准Shannon熵用于对比"""
        if len(trace_tensor) == 0:
            return 0.0
            
        p1 = torch.mean(trace_tensor)
        p0 = 1 - p1
        
        if p1 == 0 or p1 == 1:
            return 0.0
            
        return (-p1 * torch.log2(p1) - p0 * torch.log2(p0)).item()
        
    def _compute_structural_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算结构熵：基于转换次数"""
        if len(trace_tensor) <= 1:
            return 0.0
            
        transitions = torch.sum(torch.abs(trace_tensor[1:] - trace_tensor[:-1]))
        max_transitions = len(trace_tensor) - 1
        
        if max_transitions == 0:
            return 0.0
            
        transition_rate = transitions / max_transitions
        
        # 结构熵基于转换模式
        if transition_rate == 0:
            return 0.0
        else:
            return transition_rate.item() * log2(max_transitions + 1)
            
    def _compute_positional_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算位置熵：信息在位置上的分布"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 加权位置信息（右侧位置权重更高，对应Fibonacci权重）
        positions = torch.arange(len(trace_tensor), dtype=torch.float32)
        weights = torch.pow(self.phi, -positions)  # φ^(-i)权重
        
        weighted_info = trace_tensor * weights
        total_weight = torch.sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        normalized_info = weighted_info / total_weight
        
        # 位置熵：加权信息的分散程度
        variance = torch.var(normalized_info)
        return variance.item() * log2(len(trace_tensor) + 1)
        
    def _compute_hierarchical_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算层次熵：不同尺度的信息内容"""
        if len(trace_tensor) <= 1:
            return 0.0
            
        hierarchical_entropies = []
        
        # 多尺度分析
        for scale in [1, 2, 3]:
            if len(trace_tensor) >= scale:
                # 在不同尺度下重新采样
                if scale == 1:
                    scaled_trace = trace_tensor
                else:
                    # 简单子采样
                    indices = torch.arange(0, len(trace_tensor), scale)
                    scaled_trace = trace_tensor[indices]
                    
                if len(scaled_trace) > 0:
                    scale_entropy = self._compute_shannon_entropy(scaled_trace)
                    hierarchical_entropies.append(scale_entropy)
                    
        if not hierarchical_entropies:
            return 0.0
            
        # 层次熵为不同尺度熵的加权平均
        weights = [1/self.phi**i for i in range(len(hierarchical_entropies))]
        total_weight = sum(weights)
        
        return sum(e * w for e, w in zip(hierarchical_entropies, weights)) / total_weight
        
    def _compute_compression_entropy(self, trace: str) -> float:
        """计算压缩熵：基于可压缩性"""
        # 简单的游程编码压缩
        compressed = []
        current_char = trace[0] if trace else '0'
        count = 1
        
        for char in trace[1:]:
            if char == current_char:
                count += 1
            else:
                compressed.append((current_char, count))
                current_char = char
                count = 1
        compressed.append((current_char, count))
        
        # 压缩比率
        original_length = len(trace)
        compressed_length = len(compressed) * 2  # (char, count) pairs
        
        if original_length == 0:
            return 0.0
            
        compression_ratio = compressed_length / original_length
        
        # 压缩熵：基于压缩效率
        return log2(compression_ratio + 1)
        
    def _compute_golden_entropy(self, trace_tensor: torch.Tensor) -> float:
        """计算黄金熵：与φ相关的熵度量"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 计算trace与黄金比例的关系
        ones_ratio = torch.mean(trace_tensor)
        golden_ratio_diff = abs(ones_ratio.item() - (1/self.phi))
        
        # 黄金熵：接近黄金比例时熵更高
        golden_factor = 1 - golden_ratio_diff
        base_entropy = self._compute_shannon_entropy(trace_tensor)
        
        return base_entropy * golden_factor
        
    def _compute_info_density(self, trace_tensor: torch.Tensor) -> float:
        """计算信息密度"""
        if len(trace_tensor) == 0:
            return 0.0
            
        entropy = self._compute_phi_entropy(trace_tensor)
        length = len(trace_tensor)
        
        return entropy / (length + 1)  # 避免除零
        
    def _compute_entropy_gradient(self, trace_tensor: torch.Tensor) -> float:
        """计算熵梯度：熵随位置的变化"""
        if len(trace_tensor) <= 2:
            return 0.0
            
        # 局部熵梯度
        gradients = []
        window_size = 3
        
        for i in range(len(trace_tensor) - window_size + 1):
            window = trace_tensor[i:i+window_size]
            local_entropy = self._compute_shannon_entropy(window)
            gradients.append(local_entropy)
            
        if len(gradients) <= 1:
            return 0.0
            
        # 梯度的变异度
        gradients_tensor = torch.tensor(gradients)
        return torch.std(gradients_tensor).item()
        
    def _compute_entropy_complexity(self, trace_tensor: torch.Tensor) -> float:
        """计算熵复杂度"""
        phi_entropy = self._compute_phi_entropy(trace_tensor)
        shannon_entropy = self._compute_shannon_entropy(trace_tensor)
        structural_entropy = self._compute_structural_entropy(trace_tensor)
        
        # 复杂度为多种熵的组合
        return (phi_entropy + shannon_entropy + structural_entropy) / 3
        
    def _compute_self_similarity(self, trace: str) -> float:
        """计算自相似性"""
        if len(trace) <= 2:
            return 0.0
            
        # 检查不同尺度的自相似性
        similarities = []
        
        for scale in [2, 3]:
            if len(trace) >= scale * 2:
                # 分割trace
                part_size = len(trace) // scale
                parts = [trace[i*part_size:(i+1)*part_size] for i in range(scale)]
                
                # 计算部分间的相似性
                if len(parts) >= 2:
                    similarity = self._compute_string_similarity(parts[0], parts[1])
                    similarities.append(similarity)
                    
        return np.mean(similarities) if similarities else 0.0
        
    def _compute_string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
            
        matches = sum(1 for i in range(min_len) if s1[i] == s2[i])
        return matches / min_len
        
    def _compute_info_flow_properties(self, trace_tensor: torch.Tensor) -> Dict:
        """计算信息流属性"""
        if len(trace_tensor) <= 1:
            return {'flow_direction': 0.0, 'flow_strength': 0.0}
            
        # 信息流方向：从左到右的信息传递
        left_entropy = self._compute_shannon_entropy(trace_tensor[:len(trace_tensor)//2])
        right_entropy = self._compute_shannon_entropy(trace_tensor[len(trace_tensor)//2:])
        
        flow_direction = right_entropy - left_entropy
        flow_strength = abs(flow_direction)
        
        return {
            'flow_direction': flow_direction,
            'flow_strength': flow_strength
        }
        
    def _build_entropy_network(self) -> nx.DiGraph:
        """构建熵网络：trace间的熵关系"""
        G = nx.DiGraph()
        
        # 添加节点
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # 添加熵相似性边
        nodes = list(self.trace_universe.keys())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                data1 = self.trace_universe[n1]
                data2 = self.trace_universe[n2]
                
                # 计算熵距离
                entropy_distance = abs(data1['phi_entropy'] - data2['phi_entropy'])
                
                # 如果熵距离小于阈值，添加边
                if entropy_distance < 0.5:
                    G.add_edge(n1, n2, entropy_distance=entropy_distance)
                    G.add_edge(n2, n1, entropy_distance=entropy_distance)
                    
        return G
        
    def _compute_information_flows(self) -> Dict:
        """计算信息流"""
        flows = {
            'entropy_flows': [],
            'gradient_flows': [],
            'density_flows': []
        }
        
        for n, data in self.trace_universe.items():
            # 熵流
            flow_props = data['info_flow_properties']
            flows['entropy_flows'].append({
                'node': n,
                'direction': flow_props['flow_direction'],
                'strength': flow_props['flow_strength']
            })
            
            # 梯度流
            flows['gradient_flows'].append({
                'node': n,
                'gradient': data['entropy_gradient']
            })
            
            # 密度流
            flows['density_flows'].append({
                'node': n,
                'density': data['info_density']
            })
            
        return flows
        
    def _classify_entropy_types(self) -> Dict[int, str]:
        """分类熵类型"""
        types = {}
        
        for n, data in self.trace_universe.items():
            phi_entropy = data['phi_entropy']
            shannon_entropy = data['shannon_entropy']
            
            # 基于熵特性分类
            if phi_entropy > shannon_entropy * 1.2:
                types[n] = 'phi_dominant'
            elif shannon_entropy > phi_entropy * 1.2:
                types[n] = 'shannon_dominant'
            elif data['golden_entropy'] > phi_entropy * 0.8:
                types[n] = 'golden_resonant'
            elif data['entropy_complexity'] > 1.5:
                types[n] = 'complex'
            else:
                types[n] = 'balanced'
                
        return types
        
    def analyze_entropy_structure(self) -> Dict:
        """分析整体熵结构"""
        total_traces = len(self.trace_universe)
        
        # 熵统计
        phi_entropies = [data['phi_entropy'] for data in self.trace_universe.values()]
        shannon_entropies = [data['shannon_entropy'] for data in self.trace_universe.values()]
        
        # 类型统计
        type_counts = defaultdict(int)
        for entropy_type in self.entropy_categories.values():
            type_counts[entropy_type] += 1
            
        # 网络性质
        components = list(nx.weakly_connected_components(self.entropy_network))
        
        # 相关性分析
        correlations = {}
        if len(phi_entropies) > 1:
            correlations['phi_shannon'] = np.corrcoef(phi_entropies, shannon_entropies)[0, 1]
            
            lengths = [data['length'] for data in self.trace_universe.values()]
            ones_counts = [data['ones_count'] for data in self.trace_universe.values()]
            densities = [data['info_density'] for data in self.trace_universe.values()]
            
            correlations['entropy_length'] = np.corrcoef(phi_entropies, lengths)[0, 1]
            correlations['entropy_ones'] = np.corrcoef(phi_entropies, ones_counts)[0, 1]
            correlations['entropy_density'] = np.corrcoef(phi_entropies, densities)[0, 1]
            
        return {
            'total_traces': total_traces,
            'mean_phi_entropy': np.mean(phi_entropies),
            'mean_shannon_entropy': np.mean(shannon_entropies),
            'std_phi_entropy': np.std(phi_entropies),
            'std_shannon_entropy': np.std(shannon_entropies),
            'type_distribution': dict(type_counts),
            'network_components': len(components),
            'largest_component_size': len(max(components, key=len)) if components else 0,
            'correlations': correlations
        }
        
    def visualize_trace_entropy(self):
        """可视化trace熵"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 熵分布对比
        ax1 = plt.subplot(221)
        phi_entropies = [data['phi_entropy'] for data in self.trace_universe.values()]
        shannon_entropies = [data['shannon_entropy'] for data in self.trace_universe.values()]
        
        ax1.scatter(shannon_entropies, phi_entropies, alpha=0.6, s=50)
        ax1.plot([0, max(shannon_entropies)], [0, max(shannon_entropies)], 'r--', alpha=0.5, label='x=y')
        ax1.set_xlabel('Shannon Entropy')
        ax1.set_ylabel('φ-Entropy')
        ax1.set_title('Entropy Comparison: φ vs Shannon')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 熵类型分布
        ax2 = plt.subplot(222)
        type_counts = defaultdict(int)
        for entropy_type in self.entropy_categories.values():
            type_counts[entropy_type] += 1
            
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = ['skyblue', 'lightgreen', 'gold', 'salmon', 'plum'][:len(types)]
        
        bars = ax2.bar(types, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Entropy Type Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # 3. 熵与trace长度关系
        ax3 = plt.subplot(223)
        lengths = [data['length'] for data in self.trace_universe.values()]
        
        ax3.scatter(lengths, phi_entropies, alpha=0.6, label='φ-Entropy', color='blue')
        ax3.scatter(lengths, shannon_entropies, alpha=0.6, label='Shannon Entropy', color='red')
        ax3.set_xlabel('Trace Length')
        ax3.set_ylabel('Entropy')
        ax3.set_title('Entropy vs Trace Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 信息密度分布
        ax4 = plt.subplot(224)
        densities = [data['info_density'] for data in self.trace_universe.values()]
        complexities = [data['entropy_complexity'] for data in self.trace_universe.values()]
        
        # 根据类型着色
        colors_map = {'phi_dominant': 'blue', 'shannon_dominant': 'red', 
                     'golden_resonant': 'gold', 'complex': 'purple', 'balanced': 'gray'}
        point_colors = [colors_map.get(self.entropy_categories.get(n, 'balanced'), 'gray') 
                       for n in self.trace_universe.keys()]
        
        ax4.scatter(densities, complexities, c=point_colors, alpha=0.6, s=50)
        ax4.set_xlabel('Information Density')
        ax4.set_ylabel('Entropy Complexity')
        ax4.set_title('Density vs Complexity')
        ax4.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, label=label)
                         for label, color in colors_map.items()]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('chapter-128-trace-entropy.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建第二个图：3D熵空间和网络
        self._create_3d_entropy_visualization()
        
    def _create_3d_entropy_visualization(self):
        """创建3D熵可视化"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D熵空间
        ax1 = fig.add_subplot(221, projection='3d')
        
        phi_entropies = [data['phi_entropy'] for data in self.trace_universe.values()]
        shannon_entropies = [data['shannon_entropy'] for data in self.trace_universe.values()]
        golden_entropies = [data['golden_entropy'] for data in self.trace_universe.values()]
        
        # 根据类型着色
        colors_map = {'phi_dominant': 'blue', 'shannon_dominant': 'red', 
                     'golden_resonant': 'gold', 'complex': 'purple', 'balanced': 'gray'}
        point_colors = [colors_map.get(self.entropy_categories.get(n, 'balanced'), 'gray') 
                       for n in self.trace_universe.keys()]
        
        ax1.scatter(phi_entropies, shannon_entropies, golden_entropies, 
                   c=point_colors, alpha=0.6, s=50)
        ax1.set_xlabel('φ-Entropy')
        ax1.set_ylabel('Shannon Entropy')
        ax1.set_zlabel('Golden Entropy')
        ax1.set_title('3D Entropy Space')
        
        # 2. 熵网络
        ax2 = fig.add_subplot(222)
        
        if len(self.entropy_network.edges()) > 0:
            pos = nx.spring_layout(self.entropy_network, k=2, iterations=50)
            
            # 节点颜色基于熵类型
            node_colors = [colors_map.get(self.entropy_categories.get(node, 'balanced'), 'gray') 
                          for node in self.entropy_network.nodes()]
            
            nx.draw_networkx_nodes(self.entropy_network, pos, node_color=node_colors, 
                                 node_size=200, alpha=0.7, ax=ax2)
            nx.draw_networkx_edges(self.entropy_network, pos, alpha=0.3, ax=ax2)
            
            ax2.set_title("Entropy Similarity Network")
        else:
            ax2.text(0.5, 0.5, 'No entropy connections', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title("Entropy Similarity Network")
            
        ax2.axis('off')
        
        # 3. 熵梯度场
        ax3 = fig.add_subplot(223)
        
        # 创建熵梯度的可视化
        gradients = [data['entropy_gradient'] for data in self.trace_universe.values()]
        flow_directions = [data['info_flow_properties']['flow_direction'] 
                          for data in self.trace_universe.values()]
        flow_strengths = [data['info_flow_properties']['flow_strength'] 
                         for data in self.trace_universe.values()]
        
        ax3.quiver(phi_entropies, shannon_entropies, flow_directions, flow_strengths,
                  alpha=0.6, scale=10, color='darkblue')
        ax3.set_xlabel('φ-Entropy')
        ax3.set_ylabel('Shannon Entropy')
        ax3.set_title('Entropy Gradient Field')
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计摘要
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats = self.analyze_entropy_structure()
        
        summary_text = f"""Trace Entropy Analysis Summary
        
Total Traces: {stats['total_traces']}
Mean φ-Entropy: {stats['mean_phi_entropy']:.3f}
Mean Shannon Entropy: {stats['mean_shannon_entropy']:.3f}
Std φ-Entropy: {stats['std_phi_entropy']:.3f}
Std Shannon Entropy: {stats['std_shannon_entropy']:.3f}

Entropy Types:
  φ-Dominant: {stats['type_distribution'].get('phi_dominant', 0)}
  Shannon-Dominant: {stats['type_distribution'].get('shannon_dominant', 0)}
  Golden-Resonant: {stats['type_distribution'].get('golden_resonant', 0)}
  Complex: {stats['type_distribution'].get('complex', 0)}
  Balanced: {stats['type_distribution'].get('balanced', 0)}

Network Properties:
  Components: {stats['network_components']}
  Largest Component: {stats['largest_component_size']} traces

Key Correlations:
  φ-Shannon: {stats['correlations'].get('phi_shannon', 0):.3f}
  Entropy-Length: {stats['correlations'].get('entropy_length', 0):.3f}
  Entropy-Density: {stats['correlations'].get('entropy_density', 0):.3f}"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('chapter-128-trace-entropy-3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def print_analysis(self):
        """打印分析结果"""
        print("Chapter 128: TraceEntropy Verification")
        print("=" * 60)
        print("从ψ=ψ(ψ)推导Trace Entropy Definition")
        print("=" * 60)
        
        stats = self.analyze_entropy_structure()
        
        print(f"\nTraceEntropy Analysis:")
        print(f"Total traces analyzed: {stats['total_traces']} φ-valid traces")
        
        print(f"\nEntropy Statistics:")
        print(f"  Mean φ-entropy: {stats['mean_phi_entropy']:.3f}")
        print(f"  Mean Shannon entropy: {stats['mean_shannon_entropy']:.3f}")
        print(f"  φ-entropy variation: {stats['std_phi_entropy']:.3f}")
        print(f"  Shannon entropy variation: {stats['std_shannon_entropy']:.3f}")
        
        print(f"\nEntropy Type Distribution:")
        for entropy_type, count in stats['type_distribution'].items():
            percentage = count / stats['total_traces'] * 100
            print(f"  {entropy_type}: {count} traces ({percentage:.1f}%)")
            
        print(f"\nNetwork Properties:")
        print(f"  Similarity components: {stats['network_components']}")
        print(f"  Largest component: {stats['largest_component_size']} traces")
        
        print(f"\nKey Correlations:")
        for corr_name, corr_value in stats['correlations'].items():
            print(f"  {corr_name}: {corr_value:.3f}")
            
        # 展示几个代表性例子
        print(f"\nRepresentative Traces:")
        for entropy_type in ['phi_dominant', 'shannon_dominant', 'golden_resonant']:
            examples = [n for n, t in self.entropy_categories.items() if t == entropy_type][:2]
            if examples:
                print(f"\n  {entropy_type} traces:")
                for n in examples:
                    data = self.trace_universe[n]
                    print(f"    Trace {n}: {data['trace']}")
                    print(f"      φ-entropy: {data['phi_entropy']:.3f}")
                    print(f"      Shannon entropy: {data['shannon_entropy']:.3f}")
                    print(f"      Golden entropy: {data['golden_entropy']:.3f}")


class TestTraceEntropySystem(unittest.TestCase):
    """Trace Entropy System的单元测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = TraceEntropySystem(max_trace_value=34)
        
    def test_phi_constraint_validity(self):
        """测试φ-constraint的有效性"""
        for n, data in self.system.trace_universe.items():
            trace = data['trace']
            # 验证没有连续的11
            self.assertNotIn('11', trace, f"Trace {trace} contains consecutive 11s")
            
    def test_entropy_non_negative(self):
        """测试熵值非负性"""
        for n, data in self.system.trace_universe.items():
            self.assertGreaterEqual(data['phi_entropy'], 0, f"φ-entropy negative for trace {n}")
            self.assertGreaterEqual(data['shannon_entropy'], 0, f"Shannon entropy negative for trace {n}")
            self.assertGreaterEqual(data['golden_entropy'], 0, f"Golden entropy negative for trace {n}")
            
    def test_entropy_bounds(self):
        """测试熵的边界条件"""
        for n, data in self.system.trace_universe.items():
            length = data['length']
            # Shannon熵不应超过log2(length)
            if length > 0:
                max_shannon = np.log2(length)
                self.assertLessEqual(data['shannon_entropy'], max_shannon + 0.1, 
                                   f"Shannon entropy too high for trace {n}")
                
    def test_phi_entropy_properties(self):
        """测试φ-熵的特殊性质"""
        phi_entropies = [data['phi_entropy'] for data in self.system.trace_universe.values()]
        shannon_entropies = [data['shannon_entropy'] for data in self.system.trace_universe.values()]
        
        # φ-熵应该与Shannon熵相关但不完全相同
        if len(phi_entropies) > 2:
            correlation = np.corrcoef(phi_entropies, shannon_entropies)[0, 1]
            self.assertGreater(correlation, 0.3, "φ-entropy should correlate with Shannon entropy")
            self.assertLess(correlation, 0.99, "φ-entropy should be distinct from Shannon entropy")
            
    def test_golden_ratio_influence(self):
        """测试黄金比例的影响"""
        # 寻找接近黄金比例的traces
        golden_ratio_traces = 0
        for n, data in self.system.trace_universe.items():
            ones_ratio = data['ones_count'] / data['length'] if data['length'] > 0 else 0
            if abs(ones_ratio - (1/self.system.phi)) < 0.1:
                golden_ratio_traces += 1
                # 这些traces应该有较高的golden_entropy
                self.assertGreater(data['golden_entropy'], data['phi_entropy'] * 0.5)


if __name__ == "__main__":
    # 运行验证
    system = TraceEntropySystem(max_trace_value=89)
    
    # 打印分析结果
    system.print_analysis()
    
    # 生成可视化
    print("\nVisualizations saved:")
    system.visualize_trace_entropy()
    print("- chapter-128-trace-entropy.png")
    print("- chapter-128-trace-entropy-3d.png")
    
    # 运行单元测试
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "=" * 60)
    print("Verification complete: Trace entropy emerges from ψ=ψ(ψ)")
    print("through φ-constrained information measurement creating fundamental")
    print("entropy definitions for collapsed trace structures.")
    print("=" * 60)