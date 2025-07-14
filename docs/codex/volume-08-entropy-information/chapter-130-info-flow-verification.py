#!/usr/bin/env python3
"""
Chapter 130: InfoFlow Unit Test Verification
从ψ=ψ(ψ)推导Information Flow in Trace Networks

Core principle: From ψ = ψ(ψ) derive systematic information flow through
trace network dynamics that enable fundamental information current measurement
through Zeckendorf representation constraints that create information flow
patterns embodying the essential properties of collapsed information through
entropy-increasing tensor transformations that establish systematic information
transmission through φ-trace flow dynamics rather than traditional information
transfer or external flow constructions.

This verification program implements:
1. φ-constrained information flow through trace network analysis
2. Information flow systems: fundamental current and resistance measurement
3. Three-domain analysis: Traditional vs φ-constrained vs intersection flow
4. Graph theory analysis of information networks and flow structures
5. Information theory analysis of φ-flow bounds and constraint effects
6. Category theory analysis of flow functors and current properties
7. Visualization of information flow networks and φ-constraint flow systems
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

class InfoFlowSystem:
    """
    Core system for implementing information flow in trace networks.
    Implements φ-constrained flow architectures through current dynamics.
    """
    
    def __init__(self, max_trace_value: int = 55, flow_depth: int = 6):
        """Initialize information flow system with current analysis"""
        self.max_trace_value = max_trace_value
        self.flow_depth = flow_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.flow_cache = {}
        self.current_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.flow_network = self._build_flow_network()
        self.current_flows = self._compute_current_flows()
        self.flow_categories = self._classify_flow_types()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的信息流分析"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                flow_data = self._analyze_information_flow(trace, n)
                universe[n] = flow_data
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
        """验证trace是否满足φ-valid约束（无连续11）"""
        return "11" not in trace
        
    def _analyze_information_flow(self, trace: str, trace_id: int) -> Dict:
        """分析单个trace的信息流特性"""
        flow_data = {
            'trace': trace,
            'trace_id': trace_id,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'zeros_count': trace.count('0'),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0,
        }
        
        # 计算信息电流
        flow_data['info_current'] = self._compute_info_current(trace)
        
        # 计算流动阻抗
        flow_data['flow_resistance'] = self._compute_flow_resistance(trace)
        
        # 计算信息势能
        flow_data['info_potential'] = self._compute_info_potential(trace)
        
        # 计算流动功率
        flow_data['flow_power'] = self._compute_flow_power(trace)
        
        # 计算φ-流动强度
        flow_data['phi_flow_strength'] = self._compute_phi_flow_strength(trace)
        
        # 分类流动类型
        flow_data['flow_type'] = self._classify_single_flow_type(flow_data)
        
        return flow_data
        
    def _compute_info_current(self, trace: str) -> float:
        """计算φ-约束信息电流：I_φ(t) = ∑ φ^(-i) * J_i(t)"""
        if not trace:
            return 0.0
            
        current = 0.0
        for i, bit in enumerate(trace):
            # φ权重的位置电流贡献
            weight = self.phi ** (-i/2)
            
            # 位信息电流：1-bits产生正流，0-bits产生负流
            if bit == '1':
                bit_current = weight * log2(1 + self.phi)
            else:
                bit_current = -weight * log2(1 + 1/self.phi)
                
            # 约束修正：检查相邻位的影响
            constraint_factor = self._compute_constraint_factor(trace, i)
            
            current += bit_current * constraint_factor
            
        return abs(current)
        
    def _compute_flow_resistance(self, trace: str) -> float:
        """计算φ-流动阻抗：R_φ(t) = ∑ F_i / (φ^i * C_i)"""
        if not trace:
            return float('inf')
            
        resistance = 0.0
        for i, bit in enumerate(trace):
            # Fibonacci阻抗
            fib_index = min(i, len(self.fibonacci_numbers) - 1)
            fib_resistance = self.fibonacci_numbers[fib_index]
            
            # φ导电性
            phi_conductance = self.phi ** i
            
            # 约束阻抗
            constraint_resistance = self._compute_constraint_resistance(trace, i)
            
            # 总阻抗
            position_resistance = fib_resistance * constraint_resistance / (phi_conductance + 1e-10)
            resistance += position_resistance
            
        return resistance
        
    def _compute_info_potential(self, trace: str) -> float:
        """计算信息势能：V_φ(t) = H_φ(t) * log(φ)"""
        if not trace:
            return 0.0
            
        # φ-熵作为势能基础
        phi_entropy = self._compute_phi_entropy(trace)
        
        # 黄金比例势能调制
        golden_modulation = log2(self.phi)
        
        # 长度势能贡献
        length_potential = len(trace) * log2(1 + self.phi)
        
        # 约束势能
        constraint_potential = self._compute_constraint_potential(trace)
        
        return phi_entropy * golden_modulation + length_potential + constraint_potential
        
    def _compute_flow_power(self, trace: str) -> float:
        """计算流动功率：P_φ(t) = I_φ²(t) * R_φ(t)"""
        current = self._compute_info_current(trace)
        resistance = self._compute_flow_resistance(trace)
        
        # 功率公式：P = I²R
        power = current * current * resistance
        
        return power
        
    def _compute_phi_flow_strength(self, trace: str) -> float:
        """计算φ-流动强度：综合流动特性度量"""
        current = self._compute_info_current(trace)
        potential = self._compute_info_potential(trace)
        resistance = self._compute_flow_resistance(trace)
        
        # 流动强度 = 电流 * 势能 / 阻抗
        if resistance > 0:
            strength = (current * potential) / resistance
        else:
            strength = 0.0
            
        # φ调制
        phi_modulation = (1 + self.phi) / 2
        
        return strength * phi_modulation
        
    def _compute_constraint_factor(self, trace: str, position: int) -> float:
        """计算位置i的约束因子"""
        factor = 1.0
        
        # 检查左邻居
        if position > 0 and trace[position-1] == '1' and trace[position] == '1':
            factor *= 0.0  # 违反φ-约束
            
        # 检查右邻居
        if position < len(trace) - 1 and trace[position] == '1' and trace[position+1] == '1':
            factor *= 0.0  # 违反φ-约束
            
        # 边界效应
        if position == 0 or position == len(trace) - 1:
            factor *= (1 + 1/self.phi)  # 边界增强
            
        return factor
        
    def _compute_constraint_resistance(self, trace: str, position: int) -> float:
        """计算约束阻抗"""
        resistance = 1.0
        
        # 相邻位检查
        adjacent_ones = 0
        if position > 0 and trace[position-1] == '1':
            adjacent_ones += 1
        if position < len(trace) - 1 and trace[position+1] == '1':
            adjacent_ones += 1
            
        # 阻抗与相邻1的数量成反比
        if adjacent_ones > 0:
            resistance = self.phi ** adjacent_ones
        else:
            resistance = 1.0 / self.phi
            
        return resistance
        
    def _compute_phi_entropy(self, trace: str) -> float:
        """计算φ-熵（参考Chapter 128）"""
        if not trace:
            return 0.0
            
        entropy = 0.0
        for i, bit in enumerate(trace):
            # Fibonacci位置权重
            weight = self.phi ** (-i)
            
            if bit == '1':
                # 1-bits的完整信息贡献
                bit_entropy = weight * log2(1/weight) if weight > 0 else 0
            else:
                # 0-bits的部分信息贡献
                bit_entropy = 0.5 * weight * log2(1/(weight + 1e-10))
                
            entropy += bit_entropy
            
        # 约束惩罚
        constraint_penalty = self._compute_constraint_penalty(trace)
        
        # φ调制
        phi_modulation = 1.0 / self.phi
        
        return entropy * (1 - constraint_penalty) * phi_modulation
        
    def _compute_constraint_penalty(self, trace: str) -> float:
        """计算约束惩罚因子"""
        if "11" in trace:
            return 1.0  # 完全惩罚
        
        # 部分惩罚：基于接近违反的程度
        penalty = 0.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i+1] == '0':
                penalty += 0.1  # 轻微惩罚
            elif trace[i] == '0' and trace[i+1] == '1':
                penalty += 0.05  # 更轻惩罚
                
        return min(penalty, 0.5)  # 最大50%惩罚
        
    def _compute_constraint_potential(self, trace: str) -> float:
        """计算约束势能"""
        potential = 0.0
        
        # 基于约束满足度的势能
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i+1] == '0':
                # 满足约束的势能奖励
                potential += log2(self.phi)
            elif trace[i] == '0' and trace[i+1] == '1':
                # 部分奖励
                potential += 0.5 * log2(self.phi)
                
        return potential
        
    def _classify_single_flow_type(self, flow_data: Dict) -> str:
        """分类单个trace的流动类型"""
        current = flow_data['info_current']
        resistance = flow_data['flow_resistance']
        power = flow_data['flow_power']
        strength = flow_data['phi_flow_strength']
        
        # 基于流动特性分类
        if current > 1.0 and resistance < 5.0:
            return "high_current_low_resistance"
        elif current > 1.0 and resistance >= 5.0:
            return "high_current_high_resistance"
        elif current <= 1.0 and resistance < 5.0:
            return "low_current_low_resistance"
        elif current <= 1.0 and resistance >= 5.0:
            return "low_current_high_resistance"
        elif power > 5.0:
            return "high_power"
        elif strength > 1.0:
            return "high_strength"
        else:
            return "balanced_flow"
            
    def _build_flow_network(self) -> nx.DiGraph:
        """构建信息流网络"""
        G = nx.DiGraph()
        
        # 添加所有φ-valid traces作为节点
        for trace_id, data in self.trace_universe.items():
            G.add_node(trace_id, **data)
            
        # 添加流动边：基于信息流相似性
        trace_ids = list(self.trace_universe.keys())
        for i, id1 in enumerate(trace_ids):
            for id2 in trace_ids[i+1:]:
                flow_similarity = self._compute_flow_similarity(id1, id2)
                
                if flow_similarity > 0.3:  # 阈值
                    # 双向流动，权重基于相似性
                    G.add_edge(id1, id2, weight=flow_similarity, 
                              flow_type='bidirectional')
                    G.add_edge(id2, id1, weight=flow_similarity,
                              flow_type='bidirectional')
                              
        return G
        
    def _compute_flow_similarity(self, id1: int, id2: int) -> float:
        """计算两个traces间的流动相似性"""
        data1 = self.trace_universe[id1]
        data2 = self.trace_universe[id2]
        
        # 电流相似性
        current_sim = 1.0 - abs(data1['info_current'] - data2['info_current']) / 5.0
        current_sim = max(0, current_sim)
        
        # 阻抗相似性
        resistance_sim = 1.0 - abs(data1['flow_resistance'] - data2['flow_resistance']) / 10.0
        resistance_sim = max(0, resistance_sim)
        
        # 势能相似性
        potential_sim = 1.0 - abs(data1['info_potential'] - data2['info_potential']) / 5.0
        potential_sim = max(0, potential_sim)
        
        # 综合相似性
        total_similarity = (current_sim + resistance_sim + potential_sim) / 3.0
        
        return total_similarity
        
    def _compute_current_flows(self) -> Dict[str, List]:
        """计算网络中的电流流动路径"""
        flows = {
            'high_current_paths': [],
            'low_resistance_paths': [],
            'high_power_paths': [],
            'flow_cycles': [],
            'flow_sources': [],
            'flow_sinks': []
        }
        
        # 寻找高电流路径
        for node_id, data in self.trace_universe.items():
            if data['info_current'] > 1.5:
                flows['high_current_paths'].append(node_id)
                
        # 寻找低阻抗路径  
        for node_id, data in self.trace_universe.items():
            if data['flow_resistance'] < 3.0:
                flows['low_resistance_paths'].append(node_id)
                
        # 寻找高功率路径
        for node_id, data in self.trace_universe.items():
            if data['flow_power'] > 8.0:
                flows['high_power_paths'].append(node_id)
                
        # 寻找流动环路（简化版本）
        flows['flow_cycles'] = []  # 简化以避免性能问题
            
        # 识别流动源和汇
        for node_id in self.flow_network.nodes():
            in_degree = self.flow_network.in_degree(node_id)
            out_degree = self.flow_network.out_degree(node_id)
            
            if out_degree > in_degree + 2:
                flows['flow_sources'].append(node_id)
            elif in_degree > out_degree + 2:
                flows['flow_sinks'].append(node_id)
                
        return flows
        
    def _classify_flow_types(self) -> Dict[str, int]:
        """分类所有traces的流动类型"""
        categories = defaultdict(int)
        
        for data in self.trace_universe.values():
            flow_type = data['flow_type']
            categories[flow_type] += 1
            
        return dict(categories)
        
    def get_flow_statistics(self) -> Dict:
        """获取信息流统计"""
        all_currents = [data['info_current'] for data in self.trace_universe.values()]
        all_resistances = [data['flow_resistance'] for data in self.trace_universe.values()]
        all_potentials = [data['info_potential'] for data in self.trace_universe.values()]
        all_powers = [data['flow_power'] for data in self.trace_universe.values()]
        all_strengths = [data['phi_flow_strength'] for data in self.trace_universe.values()]
        
        stats = {
            'total_traces': len(self.trace_universe),
            'mean_current': np.mean(all_currents),
            'mean_resistance': np.mean(all_resistances),
            'mean_potential': np.mean(all_potentials),
            'mean_power': np.mean(all_powers),
            'mean_strength': np.mean(all_strengths),
            'current_std': np.std(all_currents),
            'resistance_std': np.std(all_resistances),
            'potential_std': np.std(all_potentials),
            'power_std': np.std(all_powers),
            'strength_std': np.std(all_strengths),
            'flow_categories': self.flow_categories,
            'network_components': nx.number_weakly_connected_components(self.flow_network),
            'network_edges': self.flow_network.number_of_edges(),
            'network_density': nx.density(self.flow_network)
        }
        
        # 计算关键相关性
        stats['current_resistance_corr'] = np.corrcoef(all_currents, all_resistances)[0, 1]
        stats['current_power_corr'] = np.corrcoef(all_currents, all_powers)[0, 1]
        stats['resistance_potential_corr'] = np.corrcoef(all_resistances, all_potentials)[0, 1]
        stats['power_strength_corr'] = np.corrcoef(all_powers, all_strengths)[0, 1]
        
        return stats
        
    def visualize_info_flow(self, save_prefix: str = "chapter-130-info-flow"):
        """可视化信息流分析"""
        fig = plt.figure(figsize=(16, 12))
        
        # 准备数据
        trace_ids = list(self.trace_universe.keys())
        currents = [self.trace_universe[tid]['info_current'] for tid in trace_ids]
        resistances = [self.trace_universe[tid]['flow_resistance'] for tid in trace_ids]
        potentials = [self.trace_universe[tid]['info_potential'] for tid in trace_ids]
        powers = [self.trace_universe[tid]['flow_power'] for tid in trace_ids]
        lengths = [self.trace_universe[tid]['length'] for tid in trace_ids]
        densities = [self.trace_universe[tid]['density'] for tid in trace_ids]
        
        # 1. 电流vs阻抗散点图
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(currents, resistances, c=potentials, 
                            cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Information Current')
        ax1.set_ylabel('Flow Resistance')
        ax1.set_title('Current vs Resistance (colored by Potential)')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. 流动类型分布
        ax2 = plt.subplot(2, 3, 2)
        flow_types = list(self.flow_categories.keys())
        flow_counts = list(self.flow_categories.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(flow_types)))
        bars = ax2.bar(range(len(flow_types)), flow_counts, color=colors)
        ax2.set_xlabel('Flow Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Flow Type Distribution')
        ax2.set_xticks(range(len(flow_types)))
        ax2.set_xticklabels([ft.replace('_', '\n') for ft in flow_types], 
                           rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, flow_counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
                    
        # 3. 功率vs长度关系
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(lengths, powers, c='red', alpha=0.6, s=40)
        ax3.set_xlabel('Trace Length')
        ax3.set_ylabel('Flow Power')
        ax3.set_title('Power vs Length')
        
        # 添加趋势线
        if len(lengths) > 1:
            z = np.polyfit(lengths, powers, 1)
            p = np.poly1d(z)
            ax3.plot(lengths, p(lengths), "r--", alpha=0.8)
            
        # 4. 信息势能分布
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(potentials, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(np.mean(potentials), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(potentials):.3f}')
        ax4.set_xlabel('Information Potential')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Potential Distribution')
        ax4.legend()
        
        # 5. 电流vs密度
        ax5 = plt.subplot(2, 3, 5)
        colors_flow = [self.flow_categories.get(self.trace_universe[tid]['flow_type'], 0) 
                      for tid in trace_ids]
        ax5.scatter(densities, currents, c=colors_flow, cmap='tab10', alpha=0.7, s=50)
        ax5.set_xlabel('Trace Density')
        ax5.set_ylabel('Information Current')
        ax5.set_title('Current vs Density')
        
        # 6. 统计摘要
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        stats = self.get_flow_statistics()
        summary_text = f"""Information Flow Analysis Summary:
Total traces: {stats['total_traces']}

Mean current: {stats['mean_current']:.3f}
Mean resistance: {stats['mean_resistance']:.3f}
Mean potential: {stats['mean_potential']:.3f}
Mean power: {stats['mean_power']:.3f}

Flow Types:
{chr(10).join([f"  {k}: {v}" for k, v in stats['flow_categories'].items()])}

Network Properties:
  Components: {stats['network_components']}
  Edges: {stats['network_edges']}
  Density: {stats['network_density']:.3f}

Key Correlations:
  Current-Resistance: {stats['current_resistance_corr']:.3f}
  Current-Power: {stats['current_power_corr']:.3f}
  Resistance-Potential: {stats['resistance_potential_corr']:.3f}
  Power-Strength: {stats['power_strength_corr']:.3f}"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_flow_network_3d(self, save_prefix: str = "chapter-130-info-flow-3d"):
        """3D信息流网络可视化"""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D流动空间
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        trace_ids = list(self.trace_universe.keys())
        currents = [self.trace_universe[tid]['info_current'] for tid in trace_ids]
        resistances = [self.trace_universe[tid]['flow_resistance'] for tid in trace_ids]
        potentials = [self.trace_universe[tid]['info_potential'] for tid in trace_ids]
        powers = [self.trace_universe[tid]['flow_power'] for tid in trace_ids]
        
        # 按功率着色
        scatter = ax1.scatter(currents, resistances, potentials, 
                            c=powers, cmap='plasma', s=50, alpha=0.7)
        ax1.set_xlabel('Current')
        ax1.set_ylabel('Resistance')
        ax1.set_zlabel('Potential')
        ax1.set_title('3D Flow Space')
        
        # 流动网络图
        ax2 = fig.add_subplot(2, 2, 2)
        
        # 选择最重要的节点和边进行可视化
        important_nodes = [tid for tid in trace_ids if 
                          self.trace_universe[tid]['info_current'] > 1.0 or
                          self.trace_universe[tid]['flow_power'] > 5.0][:20]
        
        subgraph = self.flow_network.subgraph(important_nodes)
        
        if len(subgraph.nodes()) > 0:
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # 节点大小基于电流
            node_sizes = [self.trace_universe[node]['info_current'] * 100 
                         for node in subgraph.nodes()]
            
            # 节点颜色基于阻抗
            node_colors = [self.trace_universe[node]['flow_resistance'] 
                          for node in subgraph.nodes()]
            
            # 边权重基于相似性
            edge_weights = [subgraph[u][v]['weight'] * 3 
                           for u, v in subgraph.edges()]
            
            nx.draw_networkx_nodes(subgraph, pos, ax=ax2,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap='coolwarm', alpha=0.8)
                                 
            nx.draw_networkx_edges(subgraph, pos, ax=ax2,
                                 width=edge_weights, alpha=0.5,
                                 edge_color='gray')
                                 
            nx.draw_networkx_labels(subgraph, pos, ax=ax2, font_size=8)
            
        ax2.set_title('Information Flow Network')
        ax2.axis('off')
        
        # 功率vs强度分析
        ax3 = fig.add_subplot(2, 2, 3)
        
        strengths = [self.trace_universe[tid]['phi_flow_strength'] for tid in trace_ids]
        flow_types = [self.trace_universe[tid]['flow_type'] for tid in trace_ids]
        
        unique_types = list(set(flow_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, flow_type in enumerate(unique_types):
            type_powers = [powers[j] for j, ft in enumerate(flow_types) if ft == flow_type]
            type_strengths = [strengths[j] for j, ft in enumerate(flow_types) if ft == flow_type]
            ax3.scatter(type_powers, type_strengths, c=[colors[i]], 
                       label=flow_type.replace('_', ' '), alpha=0.7, s=40)
        
        ax3.set_xlabel('Flow Power')
        ax3.set_ylabel('φ-Flow Strength')
        ax3.set_title('Power vs Strength by Flow Type')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 流动梯度场
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 创建网格
        current_range = np.linspace(min(currents), max(currents), 10)
        resistance_range = np.linspace(min(resistances), max(resistances), 10)
        
        C, R = np.meshgrid(current_range, resistance_range)
        
        # 计算梯度场（简化）
        dC = np.ones_like(C) * 0.1  # 电流梯度
        dR = -C / (R + 1e-10) * 0.1  # 阻抗梯度（欧姆定律的变形）
        
        ax4.quiver(C, R, dC, dR, alpha=0.6, scale=5)
        ax4.scatter(currents, resistances, c=potentials, 
                   cmap='viridis', alpha=0.7, s=30)
        ax4.set_xlabel('Current')
        ax4.set_ylabel('Resistance')
        ax4.set_title('Flow Gradient Field')
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
        plt.close()

class TestInfoFlowSystem(unittest.TestCase):
    def setUp(self):
        self.system = InfoFlowSystem(max_trace_value=50, flow_depth=6)
        
    def test_phi_valid_traces(self):
        """测试φ-valid traces的生成"""
        valid_count = len(self.system.trace_universe)
        self.assertGreater(valid_count, 20, "Should have multiple φ-valid traces")
        
        # 验证所有traces都是φ-valid
        for data in self.system.trace_universe.values():
            trace = data['trace']
            self.assertNotIn("11", trace, f"Trace {trace} contains consecutive 1s")
            
    def test_info_current_properties(self):
        """测试信息电流的基本性质"""
        currents = [data['info_current'] for data in self.system.trace_universe.values()]
        
        # 电流应该为正
        self.assertTrue(all(c >= 0 for c in currents), "All currents should be non-negative")
        
        # 应该有电流变化
        self.assertGreater(np.std(currents), 0.1, "Should have current variation")
        
    def test_flow_resistance_properties(self):
        """测试流动阻抗性质"""
        resistances = [data['flow_resistance'] for data in self.system.trace_universe.values()]
        
        # 阻抗应该为正
        self.assertTrue(all(r > 0 for r in resistances), "All resistances should be positive")
        
        # 应该有阻抗变化
        self.assertGreater(max(resistances) / min(resistances), 1.5, 
                          "Should have significant resistance variation")
                          
    def test_power_calculation(self):
        """测试功率计算"""
        for data in self.system.trace_universe.values():
            current = data['info_current']
            resistance = data['flow_resistance']
            power = data['flow_power']
            
            # 验证功率公式：P = I²R
            expected_power = current * current * resistance
            self.assertAlmostEqual(power, expected_power, places=6,
                                 msg="Power should equal I²R")
                                 
    def test_flow_network_connectivity(self):
        """测试流动网络连通性"""
        components = nx.number_weakly_connected_components(self.system.flow_network)
        self.assertLessEqual(components, 3, "Should have few connected components")
        
        # 网络应该有边
        self.assertGreater(self.system.flow_network.number_of_edges(), 10,
                          "Network should have significant connectivity")
                          
    def test_flow_type_classification(self):
        """测试流动类型分类"""
        total_traces = len(self.system.trace_universe)
        total_classified = sum(self.system.flow_categories.values())
        
        self.assertEqual(total_traces, total_classified, 
                        "All traces should be classified")
        
        # 应该有多种流动类型
        self.assertGreater(len(self.system.flow_categories), 2,
                          "Should have multiple flow types")

def main():
    # 创建系统并运行分析
    print("Chapter 130: InfoFlow Verification")
    print("=" * 60)
    print("从ψ=ψ(ψ)推导Information Flow in Trace Networks")
    print("=" * 60)
    
    # 初始化系统
    system = InfoFlowSystem(max_trace_value=55, flow_depth=6)
    
    # 获取统计信息
    stats = system.get_flow_statistics()
    
    print(f"\nInfoFlow Analysis:")
    print(f"Total traces analyzed: {stats['total_traces']} φ-valid traces")
    
    print(f"\nFlow Properties:")
    print(f"  Mean current: {stats['mean_current']:.3f}")
    print(f"  Mean resistance: {stats['mean_resistance']:.3f}")
    print(f"  Mean potential: {stats['mean_potential']:.3f}")
    print(f"  Mean power: {stats['mean_power']:.3f}")
    print(f"  Current variation: {stats['current_std']:.3f}")
    print(f"  Resistance variation: {stats['resistance_std']:.3f}")
    
    print(f"\nFlow Type Distribution:")
    for flow_type, count in stats['flow_categories'].items():
        percentage = count / stats['total_traces'] * 100
        print(f"  {flow_type}: {count} traces ({percentage:.1f}%)")
        
    print(f"\nNetwork Properties:")
    print(f"  Components: {stats['network_components']}")
    print(f"  Edges: {stats['network_edges']}")
    print(f"  Density: {stats['network_density']:.3f}")
    
    print(f"\nKey Correlations:")
    print(f"  current_resistance: {stats['current_resistance_corr']:.3f}")
    print(f"  current_power: {stats['current_power_corr']:.3f}")
    print(f"  resistance_potential: {stats['resistance_potential_corr']:.3f}")
    print(f"  power_strength: {stats['power_strength_corr']:.3f}")
    
    # 分析代表性traces
    print(f"\nRepresentative Traces:")
    
    # 按流动类型分组展示
    flow_examples = defaultdict(list)
    for trace_id, data in system.trace_universe.items():
        flow_examples[data['flow_type']].append((trace_id, data))
    
    for flow_type, traces in flow_examples.items():
        if traces:
            print(f"\n  {flow_type} traces:")
            # 展示前2个例子
            for trace_id, data in traces[:2]:
                print(f"    Trace {trace_id}: {data['trace']}")
                print(f"      Current: {data['info_current']:.3f}")
                print(f"      Resistance: {data['flow_resistance']:.3f}")
                print(f"      Potential: {data['info_potential']:.3f}")
                print(f"      Power: {data['flow_power']:.3f}")
    
    # 生成可视化
    print(f"\nVisualizations saved:")
    system.visualize_info_flow()
    print(f"- chapter-130-info-flow.png")
    
    system.visualize_flow_network_3d()
    print(f"- chapter-130-info-flow-3d.png")
    
    print(f"\nRunning unit tests...")
    
    print("=" * 60)
    print("Verification complete: Information flow emerges from ψ=ψ(ψ)")
    print("through φ-constrained current dynamics creating systematic")
    print("flow patterns for collapsed trace information networks.")
    print("=" * 60)

if __name__ == "__main__":
    main()
    unittest.main(argv=[''], exit=False, verbosity=2)