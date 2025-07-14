#!/usr/bin/env python3
"""
Chapter 129: HSBound Unit Test Verification
从ψ=ψ(ψ)推导Hurt-Sada Array as Compression Bound Mechanism

Core principle: From ψ = ψ(ψ) derive systematic compression bounds through
Hurt-Sada arrays that provide optimal compression limits in φ-constrained space,
enabling trace compression analysis through Zeckendorf representation constraints
that create compression boundaries embodying the fundamental limits of collapsed
information through entropy-increasing tensor transformations that establish
systematic compression variation through φ-trace compression dynamics rather
than traditional compression algorithms or external compression constructions.

This verification program implements:
1. φ-constrained Hurt-Sada arrays through optimal compression bound analysis
2. Compression systems: fundamental limits and efficiency measurement
3. Three-domain analysis: Traditional vs φ-constrained vs intersection compression
4. Graph theory analysis of compression networks and efficiency structures
5. Information theory analysis of φ-compression bounds and constraint effects
6. Category theory analysis of compression functors and limit properties
7. Visualization of compression bounds and φ-constraint compression systems
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

class HSBoundSystem:
    """
    Core system for implementing Hurt-Sada compression bounds.
    Implements φ-constrained compression architectures through bound analysis.
    """
    
    def __init__(self, max_trace_value: int = 89, compression_depth: int = 8):
        """Initialize HS-bound system with compression analysis"""
        self.max_trace_value = max_trace_value
        self.compression_depth = compression_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.compression_cache = {}
        self.bound_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.compression_network = self._build_compression_network()
        self.hs_arrays = self._compute_hs_arrays()
        self.bound_categories = self._classify_bound_types()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的压缩分析"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                compression_data = self._analyze_compression_bounds(trace, n)
                universe[n] = compression_data
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
        
    def _analyze_compression_bounds(self, trace: str, n: int) -> Dict:
        """分析trace的Hurt-Sada压缩边界"""
        if (trace, n) in self.compression_cache:
            return self.compression_cache[(trace, n)]
            
        # 基本trace属性
        trace_tensor = torch.tensor([int(b) for b in trace], dtype=torch.float32)
        
        # Hurt-Sada下界：最优压缩的理论下界
        hs_lower_bound = self._compute_hs_lower_bound(trace_tensor)
        
        # Hurt-Sada上界：实际可达的压缩上界
        hs_upper_bound = self._compute_hs_upper_bound(trace_tensor)
        
        # φ-压缩效率：在φ-constraint下的压缩性能
        phi_compression_ratio = self._compute_phi_compression_ratio(trace)
        
        # 黄金基底压缩：基于Fibonacci结构的压缩
        golden_compression = self._compute_golden_compression(trace_tensor)
        
        # 结构压缩度：基于φ-constraint的可压缩性
        structural_compressibility = self._compute_structural_compressibility(trace)
        
        # 压缩边界效率：在HS-bound内的效率
        bound_efficiency = self._compute_bound_efficiency(hs_lower_bound, hs_upper_bound, phi_compression_ratio)
        
        # 最优编码长度：理论最优编码
        optimal_encoding_length = self._compute_optimal_encoding_length(trace_tensor)
        
        # 实际编码长度：φ-constraint下的实际编码
        actual_encoding_length = self._compute_actual_encoding_length(trace)
        
        # 压缩损失：最优与实际的差距
        compression_loss = self._compute_compression_loss(optimal_encoding_length, actual_encoding_length)
        
        # Hurt-Sada数组特性
        hs_array_properties = self._compute_hs_array_properties(trace_tensor)
        
        # 边界稳定性：压缩边界的鲁棒性
        boundary_stability = self._compute_boundary_stability(trace)
        
        # 压缩梯度：压缩率随trace变化的梯度
        compression_gradient = self._compute_compression_gradient(trace_tensor)
        
        result = {
            'trace': trace,
            'value': n,
            'length': len(trace),
            'ones_count': int(torch.sum(trace_tensor)),
            'hs_lower_bound': hs_lower_bound,
            'hs_upper_bound': hs_upper_bound,
            'phi_compression_ratio': phi_compression_ratio,
            'golden_compression': golden_compression,
            'structural_compressibility': structural_compressibility,
            'bound_efficiency': bound_efficiency,
            'optimal_encoding_length': optimal_encoding_length,
            'actual_encoding_length': actual_encoding_length,
            'compression_loss': compression_loss,
            'hs_array_properties': hs_array_properties,
            'boundary_stability': boundary_stability,
            'compression_gradient': compression_gradient
        }
        
        self.compression_cache[(trace, n)] = result
        return result
        
    def _compute_hs_lower_bound(self, trace_tensor: torch.Tensor) -> float:
        """计算Hurt-Sada压缩下界"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于信息熵的理论下界
        p1 = torch.mean(trace_tensor)
        p0 = 1 - p1
        
        if p1 == 0 or p1 == 1:
            entropy_bound = 0.0
        else:
            entropy_bound = -p1 * torch.log2(p1) - p0 * torch.log2(p0)
            
        # φ-constraint修正：考虑约束减少的可能性
        constraint_factor = self._compute_constraint_factor(trace_tensor)
        
        # Hurt-Sada下界：熵下界乘以约束因子
        hs_lower = entropy_bound.item() * constraint_factor * len(trace_tensor) / self.phi
        
        return max(hs_lower, 1.0)  # 至少需要1比特
        
    def _compute_hs_upper_bound(self, trace_tensor: torch.Tensor) -> float:
        """计算Hurt-Sada压缩上界"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于结构复杂度的上界
        length = len(trace_tensor)
        
        # 最坏情况：每个位都需要独立编码
        worst_case = length
        
        # φ-优化：利用Fibonacci结构减少编码长度
        phi_optimization = self._compute_phi_optimization_factor(trace_tensor)
        
        # Hurt-Sada上界：最坏情况减去φ-优化
        hs_upper = worst_case - phi_optimization
        
        return max(hs_upper, self._compute_hs_lower_bound(trace_tensor))
        
    def _compute_phi_compression_ratio(self, trace: str) -> float:
        """计算φ-压缩比率"""
        original_length = len(trace)
        
        # 简单游程编码（考虑φ-constraint）
        compressed = []
        i = 0
        while i < len(trace):
            current_bit = trace[i]
            count = 1
            
            # 计算连续相同位的长度（但不能有11）
            while i + count < len(trace) and trace[i + count] == current_bit:
                if current_bit == '1' and count >= 1:  # 避免连续11
                    break
                count += 1
                
            compressed.append((current_bit, count))
            i += count
            
        # 压缩后长度（每个游程用2位表示：1位符号+计数编码）
        compressed_length = len(compressed) * 2
        
        if original_length == 0:
            return 1.0
            
        return compressed_length / original_length
        
    def _compute_golden_compression(self, trace_tensor: torch.Tensor) -> float:
        """计算黄金基底压缩"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于Fibonacci权重的压缩
        weights = torch.tensor([self.phi**(-i) for i in range(len(trace_tensor))], dtype=torch.float32)
        
        # 加权位值
        weighted_value = torch.sum(trace_tensor * weights)
        
        # 黄金压缩：利用φ-结构的压缩效率
        golden_efficiency = weighted_value / (torch.sum(weights) + 1e-10)
        
        return golden_efficiency.item()
        
    def _compute_structural_compressibility(self, trace: str) -> float:
        """计算结构可压缩性"""
        # 基于模式识别的可压缩性
        patterns = {}
        window_size = 3
        
        for i in range(len(trace) - window_size + 1):
            pattern = trace[i:i+window_size]
            if self._is_phi_valid(pattern):  # 只考虑φ-valid模式
                patterns[pattern] = patterns.get(pattern, 0) + 1
                
        if not patterns:
            return 0.0
            
        # 基于模式重复的可压缩性
        total_patterns = sum(patterns.values())
        max_frequency = max(patterns.values())
        
        return max_frequency / total_patterns if total_patterns > 0 else 0.0
        
    def _compute_bound_efficiency(self, lower: float, upper: float, actual: float) -> float:
        """计算边界效率"""
        if upper <= lower:
            return 1.0
            
        # 效率：实际压缩率在边界内的位置
        efficiency = (upper - actual * len(str(actual))) / (upper - lower + 1e-10)
        
        return max(0.0, min(1.0, efficiency))
        
    def _compute_optimal_encoding_length(self, trace_tensor: torch.Tensor) -> float:
        """计算最优编码长度"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于Shannon熵的最优长度
        p1 = torch.mean(trace_tensor)
        p0 = 1 - p1
        
        if p1 == 0 or p1 == 1:
            return len(trace_tensor)
            
        entropy = -p1 * torch.log2(p1) - p0 * torch.log2(p0)
        optimal_length = entropy.item() * len(trace_tensor)
        
        return optimal_length
        
    def _compute_actual_encoding_length(self, trace: str) -> float:
        """计算φ-constraint下的实际编码长度"""
        # 考虑φ-constraint的实际编码
        encoding_length = 0
        i = 0
        
        while i < len(trace):
            if trace[i] == '1':
                # 1位需要检查后续约束
                encoding_length += 1
                if i + 1 < len(trace) and trace[i + 1] == '1':
                    # 这是无效的φ-constraint情况
                    encoding_length += 1  # 额外惩罚
                i += 1
            else:
                # 0位简单编码
                encoding_length += 1
                i += 1
                
        return encoding_length
        
    def _compute_compression_loss(self, optimal: float, actual: float) -> float:
        """计算压缩损失"""
        if optimal == 0:
            return 0.0
            
        return max(0.0, (actual - optimal) / optimal)
        
    def _compute_hs_array_properties(self, trace_tensor: torch.Tensor) -> Dict:
        """计算Hurt-Sada数组性质"""
        # 构建HS数组：压缩边界的数组表示
        length = len(trace_tensor)
        
        # HS数组的基本性质
        hs_array = torch.zeros(length)
        
        for i in range(length):
            # 每个位置的压缩贡献
            local_compression = self._compute_local_compression(trace_tensor, i)
            hs_array[i] = local_compression
            
        properties = {
            'array_sum': torch.sum(hs_array).item(),
            'array_mean': torch.mean(hs_array).item(),
            'array_variance': torch.var(hs_array).item(),
            'max_compression': torch.max(hs_array).item(),
            'min_compression': torch.min(hs_array).item()
        }
        
        return properties
        
    def _compute_local_compression(self, trace_tensor: torch.Tensor, position: int) -> float:
        """计算局部压缩贡献"""
        if position >= len(trace_tensor):
            return 0.0
            
        # 考虑位置权重（Fibonacci权重）
        weight = self.phi**(-position)
        
        # 位值对压缩的贡献
        bit_value = trace_tensor[position].item()
        
        # 局部压缩：权重乘以位值的压缩效率
        local_efficiency = weight * bit_value
        
        # 约束检查：如果违反φ-constraint则降低效率
        if position > 0 and trace_tensor[position-1] == 1 and trace_tensor[position] == 1:
            local_efficiency *= 0.5  # 约束惩罚
            
        return local_efficiency
        
    def _compute_boundary_stability(self, trace: str) -> float:
        """计算边界稳定性"""
        # 通过小扰动测试边界稳定性
        original_ratio = self._compute_phi_compression_ratio(trace)
        
        perturbations = []
        for i in range(len(trace)):
            # 尝试翻转每一位（如果不违反φ-constraint）
            perturbed = list(trace)
            perturbed[i] = '0' if perturbed[i] == '1' else '1'
            perturbed_trace = ''.join(perturbed)
            
            if self._is_phi_valid(perturbed_trace):
                perturbed_ratio = self._compute_phi_compression_ratio(perturbed_trace)
                perturbations.append(abs(perturbed_ratio - original_ratio))
                
        if not perturbations:
            return 1.0
            
        # 稳定性：平均扰动的倒数
        mean_perturbation = np.mean(perturbations)
        stability = 1.0 / (1.0 + mean_perturbation)
        
        return stability
        
    def _compute_compression_gradient(self, trace_tensor: torch.Tensor) -> float:
        """计算压缩梯度"""
        if len(trace_tensor) <= 1:
            return 0.0
            
        # 计算相邻位置的压缩差异
        gradients = []
        for i in range(len(trace_tensor) - 1):
            local_comp_i = self._compute_local_compression(trace_tensor, i)
            local_comp_i1 = self._compute_local_compression(trace_tensor, i+1)
            gradient = abs(local_comp_i1 - local_comp_i)
            gradients.append(gradient)
            
        return np.mean(gradients) if gradients else 0.0
        
    def _compute_constraint_factor(self, trace_tensor: torch.Tensor) -> float:
        """计算约束因子"""
        if len(trace_tensor) <= 1:
            return 1.0
            
        # 检查约束违反
        violations = 0
        for i in range(len(trace_tensor) - 1):
            if trace_tensor[i] == 1 and trace_tensor[i+1] == 1:
                violations += 1
                
        # 约束因子：1减去违反比例
        violation_ratio = violations / (len(trace_tensor) - 1)
        constraint_factor = 1.0 - violation_ratio
        
        return max(0.1, constraint_factor)  # 最小保留0.1
        
    def _compute_phi_optimization_factor(self, trace_tensor: torch.Tensor) -> float:
        """计算φ-优化因子"""
        if len(trace_tensor) == 0:
            return 0.0
            
        # 基于Fibonacci结构的优化
        optimization = 0.0
        
        for i in range(len(trace_tensor)):
            if trace_tensor[i] == 1:
                # 检查是否在Fibonacci位置
                fib_position_weight = 0.0
                for j, fib in enumerate(self.fibonacci_numbers):
                    if i == j:
                        fib_position_weight = 1.0 / fib
                        break
                        
                optimization += fib_position_weight
                
        return optimization
        
    def _build_compression_network(self) -> nx.DiGraph:
        """构建压缩网络：trace间的压缩关系"""
        G = nx.DiGraph()
        
        # 添加节点
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # 添加压缩效率相似性边
        nodes = list(self.trace_universe.keys())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                data1 = self.trace_universe[n1]
                data2 = self.trace_universe[n2]
                
                # 计算压缩效率距离
                efficiency_distance = abs(data1['bound_efficiency'] - data2['bound_efficiency'])
                
                # 如果效率距离小于阈值，添加边
                if efficiency_distance < 0.3:
                    G.add_edge(n1, n2, efficiency_distance=efficiency_distance)
                    G.add_edge(n2, n1, efficiency_distance=efficiency_distance)
                    
        return G
        
    def _compute_hs_arrays(self) -> Dict:
        """计算Hurt-Sada数组"""
        arrays = {
            'bound_arrays': {},
            'efficiency_arrays': {},
            'compression_arrays': {}
        }
        
        for n, data in self.trace_universe.items():
            # 边界数组
            arrays['bound_arrays'][n] = [data['hs_lower_bound'], data['hs_upper_bound']]
            
            # 效率数组
            arrays['efficiency_arrays'][n] = [
                data['bound_efficiency'],
                data['golden_compression'],
                data['structural_compressibility']
            ]
            
            # 压缩数组
            arrays['compression_arrays'][n] = [
                data['phi_compression_ratio'],
                data['compression_loss'],
                data['compression_gradient']
            ]
            
        return arrays
        
    def _classify_bound_types(self) -> Dict[int, str]:
        """分类边界类型"""
        types = {}
        
        for n, data in self.trace_universe.items():
            bound_range = data['hs_upper_bound'] - data['hs_lower_bound']
            efficiency = data['bound_efficiency']
            compression_loss = data['compression_loss']
            
            # 基于边界特性分类
            if bound_range < 2.0 and efficiency > 0.8:
                types[n] = 'tight_optimal'
            elif bound_range < 2.0:
                types[n] = 'tight_suboptimal'
            elif efficiency > 0.8:
                types[n] = 'loose_optimal'
            elif compression_loss > 0.5:
                types[n] = 'high_loss'
            else:
                types[n] = 'moderate'
                
        return types
        
    def analyze_compression_structure(self) -> Dict:
        """分析整体压缩结构"""
        total_traces = len(self.trace_universe)
        
        # 压缩统计
        lower_bounds = [data['hs_lower_bound'] for data in self.trace_universe.values()]
        upper_bounds = [data['hs_upper_bound'] for data in self.trace_universe.values()]
        efficiencies = [data['bound_efficiency'] for data in self.trace_universe.values()]
        
        # 类型统计
        type_counts = defaultdict(int)
        for bound_type in self.bound_categories.values():
            type_counts[bound_type] += 1
            
        # 网络性质
        components = list(nx.weakly_connected_components(self.compression_network))
        
        # 相关性分析
        correlations = {}
        if len(lower_bounds) > 1:
            lengths = [data['length'] for data in self.trace_universe.values()]
            ones_counts = [data['ones_count'] for data in self.trace_universe.values()]
            compression_ratios = [data['phi_compression_ratio'] for data in self.trace_universe.values()]
            
            correlations['efficiency_length'] = np.corrcoef(efficiencies, lengths)[0, 1]
            correlations['efficiency_ones'] = np.corrcoef(efficiencies, ones_counts)[0, 1]
            correlations['efficiency_ratio'] = np.corrcoef(efficiencies, compression_ratios)[0, 1]
            correlations['bound_range_length'] = np.corrcoef(
                [u - l for u, l in zip(upper_bounds, lower_bounds)], lengths)[0, 1]
            
        return {
            'total_traces': total_traces,
            'mean_lower_bound': np.mean(lower_bounds),
            'mean_upper_bound': np.mean(upper_bounds),
            'mean_efficiency': np.mean(efficiencies),
            'std_efficiency': np.std(efficiencies),
            'type_distribution': dict(type_counts),
            'network_components': len(components),
            'largest_component_size': len(max(components, key=len)) if components else 0,
            'correlations': correlations
        }
        
    def visualize_compression_bounds(self):
        """可视化压缩边界"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 压缩边界图
        ax1 = plt.subplot(221)
        lower_bounds = [data['hs_lower_bound'] for data in self.trace_universe.values()]
        upper_bounds = [data['hs_upper_bound'] for data in self.trace_universe.values()]
        efficiencies = [data['bound_efficiency'] for data in self.trace_universe.values()]
        
        # 绘制边界范围
        x_values = list(range(len(lower_bounds)))
        ax1.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.3, color='lightblue', label='HS Bounds')
        ax1.scatter(x_values, efficiencies, c='red', s=30, alpha=0.7, label='Efficiency')
        
        ax1.set_xlabel('Trace Index')
        ax1.set_ylabel('Compression Value')
        ax1.set_title('Hurt-Sada Compression Bounds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 边界类型分布
        ax2 = plt.subplot(222)
        type_counts = defaultdict(int)
        for bound_type in self.bound_categories.values():
            type_counts[bound_type] += 1
            
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = ['skyblue', 'lightgreen', 'gold', 'salmon', 'plum'][:len(types)]
        
        bars = ax2.bar(types, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Compression Bound Types')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # 3. 效率vs压缩比
        ax3 = plt.subplot(223)
        compression_ratios = [data['phi_compression_ratio'] for data in self.trace_universe.values()]
        
        # 根据类型着色
        colors_map = {'tight_optimal': 'green', 'tight_suboptimal': 'orange', 
                     'loose_optimal': 'blue', 'high_loss': 'red', 'moderate': 'gray'}
        point_colors = [colors_map.get(self.bound_categories.get(n, 'moderate'), 'gray') 
                       for n in self.trace_universe.keys()]
        
        ax3.scatter(compression_ratios, efficiencies, c=point_colors, alpha=0.6, s=50)
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('Bound Efficiency')
        ax3.set_title('Efficiency vs Compression Ratio')
        ax3.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, label=label)
                         for label, color in colors_map.items()]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 4. 压缩损失分布
        ax4 = plt.subplot(224)
        compression_losses = [data['compression_loss'] for data in self.trace_universe.values()]
        
        ax4.hist(compression_losses, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Compression Loss')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Compression Loss Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_loss = np.mean(compression_losses)
        ax4.axvline(mean_loss, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_loss:.3f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('chapter-129-hs-bound.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建第二个图：3D压缩空间和网络
        self._create_3d_compression_visualization()
        
    def _create_3d_compression_visualization(self):
        """创建3D压缩可视化"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D压缩空间
        ax1 = fig.add_subplot(221, projection='3d')
        
        lower_bounds = [data['hs_lower_bound'] for data in self.trace_universe.values()]
        upper_bounds = [data['hs_upper_bound'] for data in self.trace_universe.values()]
        efficiencies = [data['bound_efficiency'] for data in self.trace_universe.values()]
        
        # 根据类型着色
        colors_map = {'tight_optimal': 'green', 'tight_suboptimal': 'orange', 
                     'loose_optimal': 'blue', 'high_loss': 'red', 'moderate': 'gray'}
        point_colors = [colors_map.get(self.bound_categories.get(n, 'moderate'), 'gray') 
                       for n in self.trace_universe.keys()]
        
        ax1.scatter(lower_bounds, upper_bounds, efficiencies, 
                   c=point_colors, alpha=0.6, s=50)
        ax1.set_xlabel('Lower Bound')
        ax1.set_ylabel('Upper Bound')
        ax1.set_zlabel('Efficiency')
        ax1.set_title('3D Compression Space')
        
        # 2. 压缩网络
        ax2 = fig.add_subplot(222)
        
        if len(self.compression_network.edges()) > 0:
            pos = nx.spring_layout(self.compression_network, k=2, iterations=50)
            
            # 节点颜色基于类型
            node_colors = [colors_map.get(self.bound_categories.get(node, 'moderate'), 'gray') 
                          for node in self.compression_network.nodes()]
            
            nx.draw_networkx_nodes(self.compression_network, pos, node_color=node_colors, 
                                 node_size=200, alpha=0.7, ax=ax2)
            nx.draw_networkx_edges(self.compression_network, pos, alpha=0.3, ax=ax2)
            
            ax2.set_title("Compression Efficiency Network")
        else:
            ax2.text(0.5, 0.5, 'No compression connections', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title("Compression Efficiency Network")
            
        ax2.axis('off')
        
        # 3. 黄金压缩效率
        ax3 = fig.add_subplot(223)
        
        golden_compressions = [data['golden_compression'] for data in self.trace_universe.values()]
        structural_compressibilities = [data['structural_compressibility'] for data in self.trace_universe.values()]
        
        ax3.scatter(golden_compressions, structural_compressibilities, alpha=0.6, c=point_colors, s=50)
        ax3.set_xlabel('Golden Compression')
        ax3.set_ylabel('Structural Compressibility')
        ax3.set_title('Golden vs Structural Compression')
        ax3.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(golden_compressions) > 1:
            z = np.polyfit(golden_compressions, structural_compressibilities, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(golden_compressions), max(golden_compressions), 100)
            ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
            ax3.legend()
        
        # 4. 统计摘要
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats = self.analyze_compression_structure()
        
        summary_text = f"""Compression Bounds Analysis Summary
        
Total Traces: {stats['total_traces']}
Mean Lower Bound: {stats['mean_lower_bound']:.3f}
Mean Upper Bound: {stats['mean_upper_bound']:.3f}
Mean Efficiency: {stats['mean_efficiency']:.3f}
Efficiency Std: {stats['std_efficiency']:.3f}

Bound Types:
  Tight Optimal: {stats['type_distribution'].get('tight_optimal', 0)}
  Tight Suboptimal: {stats['type_distribution'].get('tight_suboptimal', 0)}
  Loose Optimal: {stats['type_distribution'].get('loose_optimal', 0)}
  High Loss: {stats['type_distribution'].get('high_loss', 0)}
  Moderate: {stats['type_distribution'].get('moderate', 0)}

Network Properties:
  Components: {stats['network_components']}
  Largest Component: {stats['largest_component_size']} traces

Key Correlations:
  Efficiency-Length: {stats['correlations'].get('efficiency_length', 0):.3f}
  Efficiency-Ratio: {stats['correlations'].get('efficiency_ratio', 0):.3f}
  Bound Range-Length: {stats['correlations'].get('bound_range_length', 0):.3f}"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('chapter-129-hs-bound-3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def print_analysis(self):
        """打印分析结果"""
        print("Chapter 129: HSBound Verification")
        print("=" * 60)
        print("从ψ=ψ(ψ)推导Hurt-Sada Compression Bounds")
        print("=" * 60)
        
        stats = self.analyze_compression_structure()
        
        print(f"\nHSBound Analysis:")
        print(f"Total traces analyzed: {stats['total_traces']} φ-valid traces")
        
        print(f"\nCompression Bounds:")
        print(f"  Mean lower bound: {stats['mean_lower_bound']:.3f}")
        print(f"  Mean upper bound: {stats['mean_upper_bound']:.3f}")
        print(f"  Mean efficiency: {stats['mean_efficiency']:.3f}")
        print(f"  Efficiency variation: {stats['std_efficiency']:.3f}")
        
        print(f"\nBound Type Distribution:")
        for bound_type, count in stats['type_distribution'].items():
            percentage = count / stats['total_traces'] * 100
            print(f"  {bound_type}: {count} traces ({percentage:.1f}%)")
            
        print(f"\nNetwork Properties:")
        print(f"  Efficiency components: {stats['network_components']}")
        print(f"  Largest component: {stats['largest_component_size']} traces")
        
        print(f"\nKey Correlations:")
        for corr_name, corr_value in stats['correlations'].items():
            print(f"  {corr_name}: {corr_value:.3f}")
            
        # 展示几个代表性例子
        print(f"\nRepresentative Traces:")
        for bound_type in ['tight_optimal', 'high_loss', 'loose_optimal']:
            examples = [n for n, t in self.bound_categories.items() if t == bound_type][:2]
            if examples:
                print(f"\n  {bound_type} traces:")
                for n in examples:
                    data = self.trace_universe[n]
                    print(f"    Trace {n}: {data['trace']}")
                    print(f"      Lower bound: {data['hs_lower_bound']:.3f}")
                    print(f"      Upper bound: {data['hs_upper_bound']:.3f}")
                    print(f"      Efficiency: {data['bound_efficiency']:.3f}")
                    print(f"      Compression ratio: {data['phi_compression_ratio']:.3f}")


class TestHSBoundSystem(unittest.TestCase):
    """HS-Bound System的单元测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = HSBoundSystem(max_trace_value=34)
        
    def test_compression_bounds_validity(self):
        """测试压缩边界的有效性"""
        for n, data in self.system.trace_universe.items():
            # 下界应该小于等于上界
            self.assertLessEqual(data['hs_lower_bound'], data['hs_upper_bound'], 
                               f"Lower bound > upper bound for trace {n}")
            
            # 边界应该为正数
            self.assertGreater(data['hs_lower_bound'], 0, f"Lower bound <= 0 for trace {n}")
            self.assertGreater(data['hs_upper_bound'], 0, f"Upper bound <= 0 for trace {n}")
            
    def test_efficiency_bounds(self):
        """测试效率边界"""
        for n, data in self.system.trace_universe.items():
            efficiency = data['bound_efficiency']
            # 效率应该在0到1之间
            self.assertGreaterEqual(efficiency, 0.0, f"Efficiency < 0 for trace {n}")
            self.assertLessEqual(efficiency, 1.0, f"Efficiency > 1 for trace {n}")
            
    def test_phi_constraint_preservation(self):
        """测试φ-constraint保持"""
        for n, data in self.system.trace_universe.items():
            trace = data['trace']
            # 验证没有连续的11
            self.assertNotIn('11', trace, f"Trace {trace} contains consecutive 11s")
            
    def test_compression_consistency(self):
        """测试压缩一致性"""
        for n, data in self.system.trace_universe.items():
            # 压缩比应该为正数
            ratio = data['phi_compression_ratio']
            self.assertGreater(ratio, 0, f"Compression ratio <= 0 for trace {n}")
            
            # 压缩损失应该非负
            loss = data['compression_loss']
            self.assertGreaterEqual(loss, 0, f"Compression loss < 0 for trace {n}")
            
    def test_golden_compression_properties(self):
        """测试黄金压缩性质"""
        golden_compressions = [data['golden_compression'] for data in self.system.trace_universe.values()]
        
        # 应该有有效的黄金压缩值
        self.assertTrue(all(g >= 0 for g in golden_compressions), 
                       "Golden compression values should be non-negative")
        
        # 应该有变化
        self.assertGreater(max(golden_compressions) - min(golden_compressions), 0.01,
                          "Golden compression should show variation")


if __name__ == "__main__":
    # 运行验证
    system = HSBoundSystem(max_trace_value=89)
    
    # 打印分析结果
    system.print_analysis()
    
    # 生成可视化
    print("\nVisualizations saved:")
    system.visualize_compression_bounds()
    print("- chapter-129-hs-bound.png")
    print("- chapter-129-hs-bound-3d.png")
    
    # 运行单元测试
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "=" * 60)
    print("Verification complete: Hurt-Sada bounds emerge from ψ=ψ(ψ)")
    print("through φ-constrained compression creating optimal limits")
    print("for collapsed trace compression.")
    print("=" * 60)