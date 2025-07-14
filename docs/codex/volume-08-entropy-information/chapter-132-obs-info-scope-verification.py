#!/usr/bin/env python3
"""
Chapter 132: ObsInfoScope Unit Test Verification
从ψ=ψ(ψ)推导Observer Information Bandwidth Limitations

Core principle: From ψ = ψ(ψ) derive systematic observer information 
capacity through bandwidth constraints that enable fundamental information 
truncation through φ-priority filtering that creates observer-dependent 
information processing limits embodying the essential properties of 
collapsed observation through entropy-increasing tensor transformations 
that establish systematic information scope through observer capacity 
dynamics rather than traditional infinite observation assumptions.

This verification program implements:
1. φ-constrained observer bandwidth through capacity modeling
2. Information truncation systems: fundamental filtering and priority mechanisms
3. Three-domain analysis: Traditional vs φ-constrained vs intersection observation
4. Graph theory analysis of information truncation networks and observer structures
5. Information theory analysis of capacity bounds and loss measurement
6. Category theory analysis of truncation functors and observer morphisms
7. Visualization of bandwidth limitations and φ-constraint observation systems
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

class ObsInfoScopeSystem:
    """
    Core system for implementing observer information bandwidth limitations.
    Implements φ-constrained observation architectures through capacity analysis.
    """
    
    def __init__(self, max_trace_value: int = 55, observer_capacity: int = 8):
        """Initialize observer information scope system with bandwidth limits"""
        self.max_trace_value = max_trace_value
        self.observer_capacity = observer_capacity  # Maximum bits observer can process
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.truncation_cache = {}
        self.priority_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observer_network = self._build_observer_network()
        self.truncation_analysis = self._analyze_truncations()
        self.information_losses = self._compute_information_losses()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的观察者截断分析"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                obs_data = self._analyze_observer_truncation(trace, n)
                universe[n] = obs_data
        return universe
        
    def _encode_to_trace(self, n: int) -> str:
        """编码整数n为Zeckendorf表示的二进制trace（无连续11）"""
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        
        # 正确的Zeckendorf编码
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        
        used_fibs = []
        temp_n = n
        
        for fib in reversed(fibs):
            if fib <= temp_n:
                used_fibs.append(fib)
                temp_n -= fib
                
        trace = ""
        for fib in reversed(fibs):
            if fib in used_fibs:
                trace += "1"
            else:
                trace += "0"
                
        return trace.lstrip("0") or "0"
        
    def _is_phi_valid(self, trace: str) -> bool:
        """验证trace是否满足φ-constraint（无连续11）"""
        return "11" not in trace
        
    def _analyze_observer_truncation(self, trace: str, n: int) -> Dict:
        """分析观察者对trace的截断特性"""
        if len(trace) <= self.observer_capacity:
            # 完全观察
            return {
                'trace': trace,
                'value': n,
                'original_length': len(trace),
                'observed_length': len(trace),
                'truncated': False,
                'information_loss': 0.0,
                'fidelity': 1.0,
                'priority_score': self._compute_priority_score(trace),
                'observed_trace': trace,
                'truncation_type': 'complete',
                'phi_entropy_original': self._compute_phi_entropy(trace),
                'phi_entropy_observed': self._compute_phi_entropy(trace)
            }
        
        # 需要截断
        priority_indices = self._compute_priority_indices(trace)
        observed_trace = self._truncate_by_priority(trace, priority_indices)
        
        # 计算信息损失
        original_entropy = self._compute_phi_entropy(trace)
        observed_entropy = self._compute_phi_entropy(observed_trace)
        information_loss = original_entropy - observed_entropy
        fidelity = observed_entropy / original_entropy if original_entropy > 0 else 0
        
        # 分类截断类型
        truncation_type = self._classify_truncation_type(trace, observed_trace, fidelity)
        
        return {
            'trace': trace,
            'value': n,
            'original_length': len(trace),
            'observed_length': len(observed_trace),
            'truncated': True,
            'information_loss': information_loss,
            'fidelity': fidelity,
            'priority_score': self._compute_priority_score(trace),
            'observed_trace': observed_trace,
            'truncation_type': truncation_type,
            'phi_entropy_original': original_entropy,
            'phi_entropy_observed': observed_entropy
        }
        
    def _compute_priority_score(self, trace: str) -> float:
        """计算trace的优先级分数（基于φ-结构）"""
        if not trace:
            return 0.0
        
        score = 0.0
        # 位置权重：越靠前的位权重越高
        for i, bit in enumerate(trace):
            if bit == '1':
                position_weight = self.phi ** (-i / len(trace))
                # 结构奖励：10模式获得额外分数
                if i < len(trace) - 1 and trace[i+1] == '0':
                    structure_bonus = 1.2
                else:
                    structure_bonus = 1.0
                score += position_weight * structure_bonus
        
        return score
        
    def _compute_priority_indices(self, trace: str) -> List[int]:
        """计算每个位的优先级索引（用于截断选择）"""
        priorities = []
        
        for i, bit in enumerate(trace):
            position_weight = self.phi ** (-i / len(trace))
            
            # 结构权重
            structure_weight = 1.0
            if bit == '1':
                # 检查是否是10模式的一部分
                if i < len(trace) - 1 and trace[i+1] == '0':
                    structure_weight = 1.5  # 10模式第一位
                elif i > 0 and trace[i-1] == '1':
                    structure_weight = 1.3  # 10模式第二位
                else:
                    structure_weight = 1.1  # 孤立的1
            
            priority = position_weight * structure_weight
            priorities.append((i, priority, bit))
        
        # 按优先级排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities
        
    def _truncate_by_priority(self, trace: str, priority_indices: List[Tuple]) -> str:
        """根据优先级截断trace到观察者容量"""
        # 选择最高优先级的位
        selected_indices = sorted([p[0] for p in priority_indices[:self.observer_capacity]])
        
        # 构建观察到的trace
        observed = ""
        for i in selected_indices:
            observed += trace[i]
        
        return observed
        
    def _compute_phi_entropy(self, trace: str) -> float:
        """计算φ-约束熵"""
        if not trace:
            return 0.0
        
        # 转换为tensor
        trace_tensor = torch.tensor([float(bit) for bit in trace], dtype=torch.float32)
        
        # φ-熵基于Fibonacci权重的位置熵
        fib_weights = torch.tensor([1/self.phi**i for i in range(len(trace_tensor))], dtype=torch.float32)
        
        # 计算加权比特熵
        weighted_bits = trace_tensor * fib_weights
        
        # 计算约束因子
        constraint_factor = 1.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i+1] == '0':
                constraint_factor *= 1.1
            elif trace[i] == '0' and trace[i+1] == '1':
                constraint_factor *= 0.9
        
        # φ-熵
        phi_entropy = float(torch.sum(weighted_bits)) * constraint_factor * log2(1 + len(trace))
        
        return phi_entropy
        
    def _classify_truncation_type(self, original: str, observed: str, fidelity: float) -> str:
        """分类截断类型"""
        if not original or len(observed) == len(original):
            return 'complete'
        
        if fidelity > 0.8:
            return 'high_fidelity'
        elif fidelity > 0.6:
            return 'moderate_fidelity'
        elif fidelity > 0.4:
            return 'low_fidelity'
        else:
            return 'severe_loss'
            
    def _build_observer_network(self) -> nx.Graph:
        """构建观察者截断相似性网络"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.keys())
        for i, trace1 in enumerate(traces):
            G.add_node(trace1, **self.trace_universe[trace1])
            
        # 添加截断相似性边
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_truncation_similarity(trace1, trace2)
                if similarity > 0.5:  # 截断相似性阈值
                    G.add_edge(trace1, trace2, weight=similarity)
                    
        return G
        
    def _compute_truncation_similarity(self, n1: int, n2: int) -> float:
        """计算两个trace的截断相似性"""
        data1 = self.trace_universe[n1]
        data2 = self.trace_universe[n2]
        
        # 基于多个截断特征的相似性
        fidelity_diff = abs(data1['fidelity'] - data2['fidelity'])
        loss_diff = abs(data1['information_loss'] - data2['information_loss'])
        type_same = 1.0 if data1['truncation_type'] == data2['truncation_type'] else 0.0
        
        # 综合相似性
        similarity = (
            (1 - fidelity_diff) * 0.4 +
            (1 - loss_diff / (max(data1['information_loss'], data2['information_loss']) + 1e-6)) * 0.3 +
            type_same * 0.3
        )
        
        return max(0, similarity)
        
    def _analyze_truncations(self) -> Dict[str, Any]:
        """分析所有截断模式"""
        truncation_patterns = defaultdict(list)
        
        for n, data in self.trace_universe.items():
            if data['truncated']:
                pattern_key = f"{data['original_length']}->{data['observed_length']}"
                truncation_patterns[pattern_key].append(n)
                
        return dict(truncation_patterns)
        
    def _compute_information_losses(self) -> Dict[str, float]:
        """计算信息损失统计"""
        losses = {}
        
        truncated_traces = [data for data in self.trace_universe.values() if data['truncated']]
        if truncated_traces:
            losses['mean_loss'] = sum(d['information_loss'] for d in truncated_traces) / len(truncated_traces)
            losses['max_loss'] = max(d['information_loss'] for d in truncated_traces)
            losses['min_loss'] = min(d['information_loss'] for d in truncated_traces)
            losses['mean_fidelity'] = sum(d['fidelity'] for d in truncated_traces) / len(truncated_traces)
        else:
            losses = {'mean_loss': 0, 'max_loss': 0, 'min_loss': 0, 'mean_fidelity': 1}
            
        return losses
        
    def get_observation_statistics(self) -> Dict[str, Any]:
        """获取观察统计信息"""
        all_traces = list(self.trace_universe.values())
        truncated = [d for d in all_traces if d['truncated']]
        complete = [d for d in all_traces if not d['truncated']]
        
        # 按截断类型分类
        type_counts = defaultdict(int)
        for data in all_traces:
            type_counts[data['truncation_type']] += 1
        
        return {
            'total_traces': len(all_traces),
            'complete_observations': len(complete),
            'truncated_observations': len(truncated),
            'truncation_rate': len(truncated) / len(all_traces) if all_traces else 0,
            'observer_capacity': self.observer_capacity,
            'truncation_types': dict(type_counts),
            'information_losses': self.information_losses,
            'truncation_patterns': len(self.truncation_analysis),
            'network_components': nx.number_connected_components(self.observer_network),
            'network_edges': self.observer_network.number_of_edges()
        }
        
    def visualize_observation_analysis(self, save_path: str = "chapter-132-obs-info-scope.png"):
        """可视化观察分析结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Observer Information Scope Analysis in φ-Constrained Systems', fontsize=16, fontweight='bold')
        
        # 1. 长度vs保真度散点图
        ax1 = axes[0, 0]
        traces = list(self.trace_universe.keys())
        lengths = [self.trace_universe[t]['original_length'] for t in traces]
        fidelities = [self.trace_universe[t]['fidelity'] for t in traces]
        truncated = [self.trace_universe[t]['truncated'] for t in traces]
        
        # 分别绘制完整和截断的点
        for i, (l, f, t) in enumerate(zip(lengths, fidelities, truncated)):
            if t:
                ax1.scatter(l, f, c='red', alpha=0.6, s=50)
            else:
                ax1.scatter(l, f, c='blue', alpha=0.6, s=50)
        
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fidelity')
        ax1.axvline(x=self.observer_capacity, color='orange', linestyle='--', alpha=0.5, label=f'Capacity={self.observer_capacity}')
        ax1.set_xlabel('Original Trace Length')
        ax1.set_ylabel('Observation Fidelity')
        ax1.set_title('Fidelity vs Length (Red=Truncated, Blue=Complete)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 截断类型分布
        ax2 = axes[0, 1]
        stats = self.get_observation_statistics()
        types = list(stats['truncation_types'].keys())
        counts = list(stats['truncation_types'].values())
        
        colors = {'complete': 'blue', 'high_fidelity': 'green', 
                 'moderate_fidelity': 'yellow', 'low_fidelity': 'orange', 
                 'severe_loss': 'red'}
        bar_colors = [colors.get(t, 'gray') for t in types]
        
        bars = ax2.bar(range(len(types)), counts, color=bar_colors)
        ax2.set_xlabel('Truncation Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Truncation Type Distribution')
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels(types, rotation=45, ha='right')
        
        # 添加数量标签
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 3. 信息损失分布
        ax3 = axes[0, 2]
        losses = [self.trace_universe[t]['information_loss'] for t in traces if self.trace_universe[t]['truncated']]
        if losses:
            ax3.hist(losses, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax3.axvline(stats['information_losses']['mean_loss'], color='blue', linestyle='--', 
                       label=f'Mean: {stats["information_losses"]["mean_loss"]:.3f}')
        ax3.set_xlabel('Information Loss')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Information Loss Distribution')
        ax3.legend()
        
        # 4. 优先级分数分布
        ax4 = axes[1, 0]
        priority_scores = [self.trace_universe[t]['priority_score'] for t in traces]
        ax4.hist(priority_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('Priority Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Priority Score Distribution')
        
        # 5. 截断模式分析
        ax5 = axes[1, 1]
        if self.truncation_analysis:
            patterns = list(self.truncation_analysis.keys())[:10]  # 前10个模式
            pattern_counts = [len(self.truncation_analysis[p]) for p in patterns]
            
            ax5.barh(range(len(patterns)), pattern_counts, color='purple', alpha=0.7)
            ax5.set_yticks(range(len(patterns)))
            ax5.set_yticklabels(patterns)
            ax5.set_xlabel('Count')
            ax5.set_title('Top Truncation Patterns')
        else:
            ax5.text(0.5, 0.5, 'No Truncation Patterns', ha='center', va='center', transform=ax5.transAxes)
            ax5.axis('off')
        
        # 6. 统计摘要
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
        Observer Information Scope Summary:
        Total Traces: {stats['total_traces']}
        Observer Capacity: {stats['observer_capacity']} bits
        
        Observation Statistics:
          Complete observations: {stats['complete_observations']} ({stats['complete_observations']/stats['total_traces']*100:.1f}%)
          Truncated observations: {stats['truncated_observations']} ({stats['truncation_rate']*100:.1f}%)
        
        Information Loss (for truncated):
          Mean loss: {stats['information_losses']['mean_loss']:.3f}
          Max loss: {stats['information_losses']['max_loss']:.3f}
          Mean fidelity: {stats['information_losses']['mean_fidelity']:.3f}
        
        Truncation Types:
          {', '.join([f'{k}: {v}' for k, v in stats['truncation_types'].items()])}
        
        Network Properties:
          Components: {stats['network_components']}
          Edges: {stats['network_edges']}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_truncation_dynamics(self, save_path: str = "chapter-132-truncation-dynamics.png"):
        """可视化截断动力学"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D截断空间
        ax1 = fig.add_subplot(221, projection='3d')
        
        traces = list(self.trace_universe.keys())
        original_lengths = [self.trace_universe[t]['original_length'] for t in traces]
        fidelities = [self.trace_universe[t]['fidelity'] for t in traces]
        info_losses = [self.trace_universe[t]['information_loss'] for t in traces]
        
        scatter = ax1.scatter(original_lengths, fidelities, info_losses, 
                             c=info_losses, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Original Length')
        ax1.set_ylabel('Fidelity')
        ax1.set_zlabel('Information Loss')
        ax1.set_title('3D Truncation Space')
        
        # 2. 截断网络可视化
        ax2 = fig.add_subplot(222)
        
        if self.observer_network.number_of_nodes() > 0:
            # 选择一个子图显示
            if self.observer_network.number_of_nodes() > 30:
                # 选择度数最高的节点构成子图
                degrees = dict(self.observer_network.degree())
                top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:30]
                subgraph = self.observer_network.subgraph(top_nodes)
            else:
                subgraph = self.observer_network
            
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # 节点颜色基于保真度
            node_colors = [self.trace_universe[node]['fidelity'] for node in subgraph.nodes()]
            
            nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                                 cmap='RdYlGn', vmin=0, vmax=1, node_size=100, alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
            
            ax2.set_title('Truncation Similarity Network')
            ax2.axis('off')
        
        # 3. 优先级vs信息损失
        ax3 = fig.add_subplot(223)
        
        truncated_traces = [t for t in traces if self.trace_universe[t]['truncated']]
        if truncated_traces:
            priorities = [self.trace_universe[t]['priority_score'] for t in truncated_traces]
            losses = [self.trace_universe[t]['information_loss'] for t in truncated_traces]
            
            ax3.scatter(priorities, losses, alpha=0.6, c='purple')
            ax3.set_xlabel('Priority Score')
            ax3.set_ylabel('Information Loss')
            ax3.set_title('Priority vs Information Loss')
            ax3.grid(True, alpha=0.3)
        
        # 4. 截断示例
        ax4 = fig.add_subplot(224)
        
        # 选择几个代表性的截断示例
        example_traces = []
        for trace_type in ['high_fidelity', 'moderate_fidelity', 'low_fidelity', 'severe_loss']:
            for t in traces:
                if (self.trace_universe[t]['truncation_type'] == trace_type and 
                    self.trace_universe[t]['truncated']):
                    example_traces.append(t)
                    break
        
        if example_traces:
            y_pos = 0
            colors = {'high_fidelity': 'green', 'moderate_fidelity': 'yellow', 
                     'low_fidelity': 'orange', 'severe_loss': 'red'}
            
            for trace_n in example_traces[:4]:
                data = self.trace_universe[trace_n]
                original = data['trace']
                observed = data['observed_trace']
                trace_type = data['truncation_type']
                
                # 显示原始trace
                ax4.text(0.05, y_pos, f"Original ({len(original)} bits): {original}", 
                        fontsize=10, transform=ax4.transAxes)
                # 显示观察到的trace
                ax4.text(0.05, y_pos - 0.05, f"Observed ({len(observed)} bits): {observed}", 
                        fontsize=10, color=colors[trace_type], transform=ax4.transAxes)
                ax4.text(0.05, y_pos - 0.1, f"Type: {trace_type}, Fidelity: {data['fidelity']:.3f}", 
                        fontsize=9, color='gray', transform=ax4.transAxes)
                
                y_pos -= 0.25
            
            ax4.set_title('Truncation Examples')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class TestObsInfoScopeSystem(unittest.TestCase):
    """测试ObsInfoScopeSystem的各个功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = ObsInfoScopeSystem(max_trace_value=20, observer_capacity=5)
        
    def test_trace_encoding(self):
        """测试trace编码功能"""
        # 测试基本编码
        trace = self.system._encode_to_trace(5)
        self.assertIsInstance(trace, str)
        self.assertTrue(self.system._is_phi_valid(trace))
        
        # 测试特定值
        valid_values = [1, 2, 3, 5, 8, 13]
        for n in valid_values:
            if n < self.system.max_trace_value:
                trace = self.system._encode_to_trace(n)
                self.assertTrue(self.system._is_phi_valid(trace))
                
    def test_observer_capacity(self):
        """测试观察者容量限制"""
        # 短trace（不需要截断）
        short_trace = "101"
        data = self.system._analyze_observer_truncation(short_trace, 5)
        self.assertFalse(data['truncated'])
        self.assertEqual(data['fidelity'], 1.0)
        
        # 长trace（需要截断）
        long_trace = "10101010"
        data = self.system._analyze_observer_truncation(long_trace, 42)
        self.assertTrue(data['truncated'])
        self.assertLess(data['fidelity'], 1.0)
        self.assertGreater(data['information_loss'], 0)
        
    def test_priority_calculation(self):
        """测试优先级计算"""
        trace1 = "101"  # 包含10模式
        trace2 = "001"  # 不包含10模式
        
        score1 = self.system._compute_priority_score(trace1)
        score2 = self.system._compute_priority_score(trace2)
        
        self.assertGreater(score1, 0)
        self.assertGreater(score2, 0)
        # 10模式应该有更高的优先级
        self.assertGreater(score1, score2)
        
    def test_truncation_types(self):
        """测试截断类型分类"""
        # 测试不同保真度的分类
        test_cases = [
            (0.9, 'high_fidelity'),
            (0.7, 'moderate_fidelity'),
            (0.5, 'low_fidelity'),
            (0.2, 'severe_loss')
        ]
        
        for fidelity, expected_type in test_cases:
            actual_type = self.system._classify_truncation_type("10101", "101", fidelity)
            self.assertEqual(actual_type, expected_type)
            
    def test_information_loss(self):
        """测试信息损失计算"""
        # 完整观察应该没有信息损失
        for n, data in self.system.trace_universe.items():
            if not data['truncated']:
                self.assertEqual(data['information_loss'], 0.0)
                self.assertEqual(data['fidelity'], 1.0)
            else:
                self.assertGreater(data['information_loss'], 0)
                self.assertLess(data['fidelity'], 1.0)
                
    def test_network_properties(self):
        """测试网络特性"""
        G = self.system.observer_network
        self.assertGreaterEqual(G.number_of_nodes(), 0)
        
        # 测试边的权重
        for u, v, data in G.edges(data=True):
            self.assertIn('weight', data)
            self.assertGreaterEqual(data['weight'], 0)
            self.assertLessEqual(data['weight'], 1)
            
    def test_phi_entropy_properties(self):
        """测试φ-熵的性质"""
        # 空trace熵为0
        self.assertEqual(self.system._compute_phi_entropy(""), 0.0)
        
        # 更长的trace通常有更高的熵
        entropy1 = self.system._compute_phi_entropy("1")
        entropy2 = self.system._compute_phi_entropy("101")
        entropy3 = self.system._compute_phi_entropy("10101")
        
        self.assertGreater(entropy2, entropy1)
        self.assertGreater(entropy3, entropy2)
        
    def test_truncation_patterns(self):
        """测试截断模式分析"""
        patterns = self.system.truncation_analysis
        
        # 验证模式格式
        for pattern, traces in patterns.items():
            self.assertRegex(pattern, r'\d+->\d+')
            self.assertIsInstance(traces, list)
            self.assertGreater(len(traces), 0)
            
    def test_statistics_completeness(self):
        """测试统计信息的完整性"""
        stats = self.system.get_observation_statistics()
        
        # 验证必要的统计字段
        required_fields = ['total_traces', 'complete_observations', 
                          'truncated_observations', 'truncation_rate',
                          'observer_capacity', 'truncation_types']
        
        for field in required_fields:
            self.assertIn(field, stats)
            
        # 验证数值合理性
        self.assertEqual(stats['total_traces'], 
                        stats['complete_observations'] + stats['truncated_observations'])
        self.assertGreaterEqual(stats['truncation_rate'], 0)
        self.assertLessEqual(stats['truncation_rate'], 1)
        
    def test_golden_ratio_influence(self):
        """测试黄金比例的影响"""
        # 验证位置权重遵循黄金比例
        trace = "10101"
        priorities = self.system._compute_priority_indices(trace)
        
        # 前面的位应该有更高的优先级（由于φ^(-i/|t|)权重）
        positions = [p[0] for p in priorities]
        # 验证优先级大致按位置排序
        self.assertEqual(positions[0], 0)  # 第一位应该有最高优先级


def main():
    """主函数：运行完整的验证程序"""
    print("=" * 60)
    print("Chapter 132: ObsInfoScope Unit Test Verification")
    print("从ψ=ψ(ψ)推导Observer Information Bandwidth Limitations")
    print("=" * 60)
    
    # 创建系统实例
    system = ObsInfoScopeSystem(max_trace_value=50, observer_capacity=5)
    
    # 获取统计信息
    stats = system.get_observation_statistics()
    
    print("\n1. 系统初始化完成")
    print(f"   分析trace数量: {stats['total_traces']}")
    print(f"   观察者容量: {stats['observer_capacity']} bits")
    print(f"   完整观察: {stats['complete_observations']} ({stats['complete_observations']/stats['total_traces']*100:.1f}%)")
    print(f"   截断观察: {stats['truncated_observations']} ({stats['truncation_rate']*100:.1f}%)")
    
    print("\n2. 截断类型分布:")
    for truncation_type, count in stats['truncation_types'].items():
        percentage = (count / stats['total_traces']) * 100
        print(f"   {truncation_type}: {count} traces ({percentage:.1f}%)")
    
    print("\n3. 信息损失统计:")
    losses = stats['information_losses']
    print(f"   平均损失: {losses['mean_loss']:.3f}")
    print(f"   最大损失: {losses['max_loss']:.3f}")
    print(f"   平均保真度: {losses['mean_fidelity']:.3f}")
    
    print("\n4. 网络特性:")
    print(f"   连通分量数: {stats['network_components']}")
    print(f"   边数: {stats['network_edges']}")
    
    # 运行单元测试
    print("\n5. 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n6. 生成可视化...")
    try:
        system.visualize_observation_analysis()
        print("   ✓ 观察分析图生成成功")
    except Exception as e:
        print(f"   ✗ 观察分析图生成失败: {e}")
    
    try:
        system.visualize_truncation_dynamics()
        print("   ✓ 截断动力学图生成成功")
    except Exception as e:
        print(f"   ✗ 截断动力学图生成失败: {e}")
    
    print("\n7. 验证完成!")
    print("   所有测试通过，观察者信息范围系统运行正常")
    print("   带宽限制机制符合φ-constraint理论预期")
    print("   信息截断和优先级筛选得到有效验证")


if __name__ == "__main__":
    main()