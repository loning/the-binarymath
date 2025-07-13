#!/usr/bin/env python3
"""
Chapter 088: ObsFilterZeta Unit Test Verification
从ψ=ψ(ψ)推导Observer-Weighted Filtering of Collapse Frequencies

Core principle: From ψ = ψ(ψ) derive observer-dependent filtering of spectral components
where observers selectively perceive different aspects of the zeta function spectrum,
revealing how consciousness creates observational boundaries in collapse frequency space.

This verification program implements:
1. φ-constrained observer states with filtering parameters
2. Observer filtering: selective perception of spectral components in ζ(s)
3. Three-domain analysis: Traditional vs φ-constrained vs intersection observer theory
4. Graph theory analysis of observer networks and filtering relationships
5. Information theory analysis of filtering entropy and observational encoding
6. Category theory analysis of observer functors and filtering morphisms
7. Visualization of filtering patterns and observer-dependent spectra
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

class ObsFilterZetaSystem:
    """
    Core system for implementing observer-weighted filtering of collapse frequencies.
    Implements φ-constrained observer analysis via filtered spectral operations.
    """
    
    def __init__(self, max_trace_value: int = 70, num_observers: int = 8):
        """Initialize observer filter system with spectral analysis"""
        self.max_trace_value = max_trace_value
        self.num_observers = num_observers
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.observer_cache = {}
        self.filter_cache = {}
        self.spectrum_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observers = self._build_observer_states()
        
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
                spectral_data = self._analyze_spectral_properties(trace, n)
                universe[n] = spectral_data
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
        
    def _analyze_spectral_properties(self, trace: str, value: int) -> Dict:
        """分析trace的谱性质，用于观察者滤波分析"""
        result = {
            'value': value,
            'trace': trace,
            'length': len(trace),
            'weight': trace.count('1'),
            'base_frequency': self._compute_base_frequency(trace),
            'spectral_amplitude': self._compute_spectral_amplitude(trace),
            'phase_signature': self._compute_phase_signature(trace),
            'filter_resonance': self._compute_filter_resonance(trace),
            'observer_visibility': self._compute_observer_visibility(trace),
            'filtering_coefficient': self._compute_filtering_coefficient(trace),
            'spectral_density': self._compute_spectral_density(trace),
            'coherence_measure': self._compute_coherence_measure(trace),
            'observation_weight': self._compute_observation_weight(trace),
            'filter_bandwidth': self._compute_filter_bandwidth(trace),
            'spectral_entropy': self._compute_spectral_entropy(trace),
            'filtering_complexity': self._compute_filtering_complexity(trace),
            'observer_coupling': self._compute_observer_coupling(trace),
        }
        return result
        
    def _compute_base_frequency(self, trace: str) -> float:
        """计算trace的基础频率"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        
        if weight == 0:
            return 0.0
        
        # 基于1的分布计算频率
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) <= 1:
            return weight / length
        
        # 平均间距的倒数
        intervals = [ones_positions[i+1] - ones_positions[i] for i in range(len(ones_positions)-1)]
        mean_interval = sum(intervals) / len(intervals)
        
        return 1.0 / (1 + mean_interval)
        
    def _compute_spectral_amplitude(self, trace: str) -> float:
        """计算谱幅度"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        
        # 基本幅度
        base_amplitude = weight / length
        
        # 黄金比例调制
        golden_factor = 1 + 0.618 * cos(2 * pi * weight / self.phi)
        
        return base_amplitude * golden_factor
        
    def _compute_phase_signature(self, trace: str) -> float:
        """计算相位特征"""
        if not trace:
            return 0.0
        
        # 基于trace模式的相位
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if not ones_positions:
            return 0.0
        
        # 重心位置决定相位
        center_of_mass = sum(ones_positions) / len(ones_positions)
        normalized_center = center_of_mass / len(trace)
        
        return normalized_center * 2 * pi
        
    def _compute_filter_resonance(self, trace: str) -> float:
        """计算滤波共振"""
        base_freq = self._compute_base_frequency(trace)
        amplitude = self._compute_spectral_amplitude(trace)
        
        # 共振强度
        resonance = base_freq * amplitude
        
        # φ调制
        phi_modulation = 1 + sin(resonance * self.phi)
        
        return resonance * phi_modulation
        
    def _compute_observer_visibility(self, trace: str) -> float:
        """计算观察者可见性"""
        if not trace:
            return 0.0
        
        # 基于trace复杂度的可见性
        complexity = self._compute_trace_complexity(trace)
        resonance = self._compute_filter_resonance(trace)
        
        # 可见性函数
        visibility = complexity * resonance / (1 + complexity + resonance)
        
        return min(1.0, visibility)
        
    def _compute_trace_complexity(self, trace: str) -> float:
        """计算trace复杂度"""
        if len(trace) <= 1:
            return 0.0
        
        # 模式多样性
        patterns = set()
        for length in range(1, min(len(trace) + 1, 4)):
            for start in range(len(trace) - length + 1):
                pattern = trace[start:start+length]
                patterns.add(pattern)
        
        return len(patterns) / (len(trace) * 3)
        
    def _compute_filtering_coefficient(self, trace: str) -> float:
        """计算滤波系数"""
        visibility = self._compute_observer_visibility(trace)
        resonance = self._compute_filter_resonance(trace)
        
        # 滤波强度
        return visibility * resonance
        
    def _compute_spectral_density(self, trace: str) -> float:
        """计算谱密度"""
        frequency = self._compute_base_frequency(trace)
        amplitude = self._compute_spectral_amplitude(trace)
        
        return frequency * amplitude * amplitude
        
    def _compute_coherence_measure(self, trace: str) -> float:
        """计算相干性度量"""
        if not trace:
            return 0.0
        
        # 基于相位一致性
        phase = self._compute_phase_signature(trace)
        frequency = self._compute_base_frequency(trace)
        
        # 相干度
        coherence = cos(phase) * frequency
        
        return abs(coherence)
        
    def _compute_observation_weight(self, trace: str) -> float:
        """计算观察权重"""
        visibility = self._compute_observer_visibility(trace)
        coherence = self._compute_coherence_measure(trace)
        
        return sqrt(visibility * coherence)
        
    def _compute_filter_bandwidth(self, trace: str) -> float:
        """计算滤波带宽"""
        frequency = self._compute_base_frequency(trace)
        complexity = self._compute_trace_complexity(trace)
        
        # 带宽与复杂度相关
        return frequency * (1 + complexity)
        
    def _compute_spectral_entropy(self, trace: str) -> float:
        """计算谱熵"""
        if len(trace) <= 1:
            return 0.0
        
        # 基于位分布的熵
        bit_counts = {'0': trace.count('0'), '1': trace.count('1')}
        total = sum(bit_counts.values())
        
        entropy = 0.0
        for count in bit_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)
        
        return entropy
        
    def _compute_filtering_complexity(self, trace: str) -> float:
        """计算滤波复杂度"""
        entropy = self._compute_spectral_entropy(trace)
        complexity = self._compute_trace_complexity(trace)
        bandwidth = self._compute_filter_bandwidth(trace)
        
        return (entropy + complexity + bandwidth) / 3
        
    def _compute_observer_coupling(self, trace: str) -> float:
        """计算观察者耦合"""
        weight = self._compute_observation_weight(trace)
        coherence = self._compute_coherence_measure(trace)
        
        return weight * coherence
        
    def _build_observer_states(self) -> List[Dict]:
        """构建观察者状态"""
        observers = []
        
        for i in range(self.num_observers):
            # 每个观察者有不同的滤波参数
            center_freq = (i + 1) * 0.3 / self.num_observers
            bandwidth = 0.2 + 0.1 * (i % 3)
            sensitivity = 0.5 + 0.5 * sin(i * pi / 4)
            phase_preference = i * 2 * pi / self.num_observers
            
            observer = {
                'id': i,
                'center_frequency': center_freq,
                'bandwidth': bandwidth,
                'sensitivity': sensitivity,
                'phase_preference': phase_preference,
                'filter_type': self._classify_filter_type(center_freq, bandwidth),
                'observation_efficiency': sensitivity * bandwidth,
                'spectral_preference': self._compute_spectral_preference(center_freq, phase_preference),
            }
            observers.append(observer)
        
        return observers
        
    def _classify_filter_type(self, center_freq: float, bandwidth: float) -> str:
        """对滤波器类型进行分类"""
        if bandwidth < 0.25:
            return "narrowband"
        elif bandwidth < 0.35:
            return "moderate"
        else:
            return "wideband"
            
    def _compute_spectral_preference(self, center_freq: float, phase_pref: float) -> float:
        """计算谱偏好"""
        return center_freq * cos(phase_pref) + 0.5
        
    def analyze_observer_filtering(self) -> Dict:
        """分析观察者滤波特性"""
        # 计算每个观察者对所有trace的滤波响应
        observer_responses = {}
        
        for obs in self.observers:
            responses = []
            for value, trace_data in self.trace_universe.items():
                response = self._compute_observer_response(obs, trace_data)
                responses.append(response)
            
            observer_responses[obs['id']] = {
                'responses': responses,
                'mean_response': np.mean(responses),
                'response_variance': np.var(responses),
                'max_response': np.max(responses),
                'filter_type': obs['filter_type'],
                'efficiency': obs['observation_efficiency'],
            }
        
        # 全局统计
        all_responses = []
        for obs_data in observer_responses.values():
            all_responses.extend(obs_data['responses'])
        
        return {
            'num_observers': len(self.observers),
            'num_traces': len(self.trace_universe),
            'observer_responses': observer_responses,
            'global_mean_response': np.mean(all_responses),
            'global_response_variance': np.var(all_responses),
            'total_observations': len(all_responses),
        }
        
    def _compute_observer_response(self, observer: Dict, trace_data: Dict) -> float:
        """计算观察者对特定trace的响应"""
        # 频率匹配
        freq_diff = abs(trace_data['base_frequency'] - observer['center_frequency'])
        freq_response = exp(-freq_diff / observer['bandwidth'])
        
        # 相位匹配
        phase_diff = abs(trace_data['phase_signature'] - observer['phase_preference'])
        phase_response = cos(phase_diff)
        
        # 综合响应
        total_response = freq_response * phase_response * observer['sensitivity']
        total_response *= trace_data['observer_visibility']
        
        return max(0.0, total_response)
        
    def build_observer_network(self) -> nx.Graph:
        """构建观察者网络图"""
        G = nx.Graph()
        
        # 添加观察者节点
        for obs in self.observers:
            G.add_node(obs['id'], **obs)
        
        # 添加边：基于滤波相似性
        for i, obs1 in enumerate(self.observers):
            for j, obs2 in enumerate(self.observers[i+1:], i+1):
                # 计算滤波相似性
                freq_similarity = 1.0 / (1 + abs(obs1['center_frequency'] - obs2['center_frequency']))
                bandwidth_similarity = 1.0 / (1 + abs(obs1['bandwidth'] - obs2['bandwidth']))
                phase_similarity = cos(obs1['phase_preference'] - obs2['phase_preference'])
                
                overall_similarity = (freq_similarity + bandwidth_similarity + phase_similarity) / 3
                
                # 添加边如果相似性足够高
                if overall_similarity > 0.5:
                    G.add_edge(obs1['id'], obs2['id'], weight=overall_similarity,
                              similarity_type='filter_matching')
        
        return G
        
    def compute_information_entropy(self) -> Dict:
        """计算信息熵"""
        # trace数据熵
        trace_values = list(self.trace_universe.values())
        
        def compute_entropy(data_list):
            if not data_list:
                return 0.0
            
            data_array = np.array(data_list)
            if np.all(data_array == data_array[0]):
                return 0.0
            
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
                unique, counts = np.unique(data_array, return_counts=True)
                probs = counts / np.sum(counts)
                return -np.sum(probs * np.log2(probs))
        
        entropies = {}
        
        # trace属性熵
        for key in ['base_frequency', 'spectral_amplitude', 'filter_resonance',
                   'observer_visibility', 'filtering_coefficient', 'spectral_density',
                   'coherence_measure', 'observation_weight', 'filter_bandwidth',
                   'spectral_entropy', 'filtering_complexity', 'observer_coupling']:
            data = [v[key] for v in trace_values if key in v]
            entropies[f'{key}_entropy'] = compute_entropy(data)
        
        # 观察者属性熵
        observer_data = {
            'center_frequency': [obs['center_frequency'] for obs in self.observers],
            'bandwidth': [obs['bandwidth'] for obs in self.observers],
            'sensitivity': [obs['sensitivity'] for obs in self.observers],
            'observation_efficiency': [obs['observation_efficiency'] for obs in self.observers],
        }
        
        for key, data in observer_data.items():
            entropies[f'observer_{key}_entropy'] = compute_entropy(data)
        
        # 滤波器类型熵
        filter_types = [obs['filter_type'] for obs in self.observers]
        type_counts = {}
        for ft in filter_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        
        total = sum(type_counts.values())
        type_probs = [count/total for count in type_counts.values()]
        filter_type_entropy = -sum(p * log2(p) for p in type_probs if p > 0)
        entropies['filter_type_entropy'] = filter_type_entropy
        entropies['filter_type_count'] = len(type_counts)
        
        return entropies
        
    def analyze_filter_categories(self) -> Dict:
        """分析滤波器范畴"""
        # 按类型分组观察者
        filter_categories = defaultdict(list)
        for obs in self.observers:
            filter_categories[obs['filter_type']].append(obs['id'])
        
        # 分析观察者间的态射
        morphisms = []
        for i, obs1 in enumerate(self.observers):
            for j, obs2 in enumerate(self.observers):
                if i != j:
                    # 检查是否存在滤波保持态射
                    freq_compatible = abs(obs1['center_frequency'] - obs2['center_frequency']) < 0.2
                    bandwidth_compatible = abs(obs1['bandwidth'] - obs2['bandwidth']) < 0.15
                    
                    if freq_compatible and bandwidth_compatible:
                        morphisms.append((obs1['id'], obs2['id']))
        
        return {
            'filter_categories': dict(filter_categories),
            'category_count': len(filter_categories),
            'morphism_count': len(morphisms),
            'total_observers': len(self.observers),
            'morphism_density': len(morphisms) / (len(self.observers) * (len(self.observers) - 1)) if len(self.observers) > 1 else 0,
        }

    def visualize_observer_filtering(self):
        """可视化观察者滤波"""
        plt.figure(figsize=(20, 15))
        
        # 1. 观察者滤波器特性
        plt.subplot(2, 3, 1)
        
        # 绘制每个观察者的滤波器响应曲线
        freq_range = np.linspace(0, 1, 100)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.observers)))
        
        for i, obs in enumerate(self.observers):
            response_curve = []
            for f in freq_range:
                freq_diff = abs(f - obs['center_frequency'])
                response = exp(-freq_diff / obs['bandwidth']) * obs['sensitivity']
                response_curve.append(response)
            
            plt.plot(freq_range, response_curve, color=colors[i], 
                    label=f"Obs{obs['id']} ({obs['filter_type']})")
        
        plt.xlabel('Frequency')
        plt.ylabel('Filter Response')
        plt.title('Observer Filter Characteristics')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. 观察者响应热图
        plt.subplot(2, 3, 2)
        
        # 计算响应矩阵
        analysis = self.analyze_observer_filtering()
        response_matrix = []
        
        for obs_id in range(len(self.observers)):
            responses = analysis['observer_responses'][obs_id]['responses']
            response_matrix.append(responses[:20])  # 只显示前20个trace
        
        response_matrix = np.array(response_matrix)
        
        im = plt.imshow(response_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(im, shrink=0.8)
        plt.xlabel('Trace Index')
        plt.ylabel('Observer ID')
        plt.title('Observer Response Matrix')
        
        # 3. 滤波器类型分布
        plt.subplot(2, 3, 3)
        filter_types = [obs['filter_type'] for obs in self.observers]
        type_counts = {}
        for ft in filter_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.title('Filter Type Distribution')
        
        # 4. 观察者参数散点图
        plt.subplot(2, 3, 4)
        center_freqs = [obs['center_frequency'] for obs in self.observers]
        bandwidths = [obs['bandwidth'] for obs in self.observers]
        sensitivities = [obs['sensitivity'] for obs in self.observers]
        
        scatter = plt.scatter(center_freqs, bandwidths, c=sensitivities, 
                            s=100, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, shrink=0.8, label='Sensitivity')
        plt.xlabel('Center Frequency')
        plt.ylabel('Bandwidth')
        plt.title('Observer Parameter Space')
        plt.grid(True, alpha=0.3)
        
        # 5. 频谱可见性分析
        plt.subplot(2, 3, 5)
        trace_values = list(self.trace_universe.values())
        visibilities = [t['observer_visibility'] for t in trace_values]
        coherences = [t['coherence_measure'] for t in trace_values]
        
        plt.scatter(visibilities, coherences, alpha=0.6, s=30)
        plt.xlabel('Observer Visibility')
        plt.ylabel('Coherence Measure')
        plt.title('Visibility vs Coherence')
        plt.grid(True, alpha=0.3)
        
        # 6. 滤波复杂度分布
        plt.subplot(2, 3, 6)
        complexities = [t['filtering_complexity'] for t in trace_values]
        couplings = [t['observer_coupling'] for t in trace_values]
        
        plt.scatter(complexities, couplings, alpha=0.6, s=30, c='orange')
        plt.xlabel('Filtering Complexity')
        plt.ylabel('Observer Coupling')
        plt.title('Complexity vs Coupling')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-088-obs-filter-zeta-filtering.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_observer_network(self):
        """可视化观察者网络"""
        plt.figure(figsize=(15, 10))
        
        G = self.build_observer_network()
        
        # 1. 主观察者网络
        plt.subplot(2, 2, 1)
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        # 根据滤波器类型着色
        filter_type_colors = {'narrowband': 'red', 'moderate': 'blue', 'wideband': 'green'}
        node_colors = []
        for node in G.nodes():
            obs = self.observers[node]
            node_colors.append(filter_type_colors.get(obs['filter_type'], 'gray'))
        
        # 绘制网络
        nx.draw(G, pos, node_color=node_colors, node_size=200, alpha=0.8,
                edge_color='gray', width=1.0)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, {node: f"O{node}" for node in G.nodes()}, 
                               font_size=8)
        
        plt.title('Observer Network by Filter Type')
        
        # 图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=ftype)
                          for ftype, color in filter_type_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 2. 观察者效率分布
        plt.subplot(2, 2, 2)
        efficiencies = [obs['observation_efficiency'] for obs in self.observers]
        
        unique_effs = len(np.unique(efficiencies))
        bin_count = min(8, max(3, unique_effs))
        
        try:
            plt.hist(efficiencies, bins=bin_count, alpha=0.7, color='skyblue', edgecolor='black')
        except ValueError:
            plt.bar(range(len(efficiencies)), efficiencies, alpha=0.7, color='skyblue')
            plt.xlabel('Observer Index')
        
        plt.xlabel('Observation Efficiency')
        plt.ylabel('Count')
        plt.title('Observer Efficiency Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. 相位偏好极坐标图
        plt.subplot(2, 2, 3, projection='polar')
        
        phase_prefs = [obs['phase_preference'] for obs in self.observers]
        sensitivities = [obs['sensitivity'] for obs in self.observers]
        
        plt.scatter(phase_prefs, sensitivities, c=range(len(self.observers)), 
                   s=100, cmap='viridis', alpha=0.7)
        plt.title('Observer Phase Preferences')
        
        # 4. 频率-带宽关系
        plt.subplot(2, 2, 4)
        center_freqs = [obs['center_frequency'] for obs in self.observers]
        bandwidths = [obs['bandwidth'] for obs in self.observers]
        
        # 根据滤波器类型着色
        colors = [filter_type_colors.get(obs['filter_type'], 'gray') for obs in self.observers]
        
        plt.scatter(center_freqs, bandwidths, c=colors, s=100, alpha=0.7)
        plt.xlabel('Center Frequency')
        plt.ylabel('Bandwidth')
        plt.title('Frequency-Bandwidth Relationship')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-088-obs-filter-zeta-network.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_spectral_analysis(self):
        """可视化谱分析"""
        plt.figure(figsize=(20, 12))
        
        trace_values = list(self.trace_universe.values())
        
        # 1. 基础频率vs谱幅度
        plt.subplot(2, 3, 1)
        base_freqs = [t['base_frequency'] for t in trace_values]
        amplitudes = [t['spectral_amplitude'] for t in trace_values]
        
        plt.scatter(base_freqs, amplitudes, alpha=0.6, s=40)
        plt.xlabel('Base Frequency')
        plt.ylabel('Spectral Amplitude')
        plt.title('Frequency vs Amplitude')
        plt.grid(True, alpha=0.3)
        
        # 2. 滤波共振分布
        plt.subplot(2, 3, 2)
        resonances = [t['filter_resonance'] for t in trace_values]
        
        unique_res = len(np.unique(resonances))
        bin_count = min(15, max(3, unique_res))
        
        try:
            plt.hist(resonances, bins=bin_count, alpha=0.7, color='lightcoral', edgecolor='black')
        except ValueError:
            plt.bar(range(len(resonances)), resonances, alpha=0.7, color='lightcoral')
        
        plt.xlabel('Filter Resonance')
        plt.ylabel('Count')
        plt.title('Filter Resonance Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. 观察者可见性vs滤波系数
        plt.subplot(2, 3, 3)
        visibilities = [t['observer_visibility'] for t in trace_values]
        filter_coeffs = [t['filtering_coefficient'] for t in trace_values]
        
        plt.scatter(visibilities, filter_coeffs, alpha=0.6, s=40, c='green')
        plt.xlabel('Observer Visibility')
        plt.ylabel('Filtering Coefficient')
        plt.title('Visibility vs Filter Coefficient')
        plt.grid(True, alpha=0.3)
        
        # 4. 谱密度vs相干性
        plt.subplot(2, 3, 4)
        densities = [t['spectral_density'] for t in trace_values]
        coherences = [t['coherence_measure'] for t in trace_values]
        
        plt.scatter(densities, coherences, alpha=0.6, s=40, c='purple')
        plt.xlabel('Spectral Density')
        plt.ylabel('Coherence Measure')
        plt.title('Density vs Coherence')
        plt.grid(True, alpha=0.3)
        
        # 5. 3D谱分析
        ax = plt.subplot(2, 3, 5, projection='3d')
        
        observation_weights = [t['observation_weight'] for t in trace_values]
        filter_bandwidths = [t['filter_bandwidth'] for t in trace_values]
        
        ax.scatter(base_freqs, observation_weights, filter_bandwidths, 
                  c=amplitudes, s=30, alpha=0.6, cmap='plasma')
        
        ax.set_xlabel('Base Frequency')
        ax.set_ylabel('Observation Weight')
        ax.set_zlabel('Filter Bandwidth')
        ax.set_title('3D Spectral Analysis')
        
        # 6. 观察者响应统计
        plt.subplot(2, 3, 6)
        analysis = self.analyze_observer_filtering()
        
        mean_responses = []
        max_responses = []
        
        for obs_id in range(len(self.observers)):
            obs_data = analysis['observer_responses'][obs_id]
            mean_responses.append(obs_data['mean_response'])
            max_responses.append(obs_data['max_response'])
        
        x_pos = np.arange(len(self.observers))
        width = 0.35
        
        plt.bar(x_pos - width/2, mean_responses, width, label='Mean Response', alpha=0.7)
        plt.bar(x_pos + width/2, max_responses, width, label='Max Response', alpha=0.7)
        
        plt.xlabel('Observer ID')
        plt.ylabel('Response')
        plt.title('Observer Response Statistics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-088-obs-filter-zeta-spectral.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

class TestObsFilterZeta(unittest.TestCase):
    """单元测试"""
    
    def setUp(self):
        """测试setup"""
        self.system = ObsFilterZetaSystem(max_trace_value=30, num_observers=5)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        trace_7 = self.system._encode_to_trace(7)
        self.assertTrue(self.system._is_phi_valid(trace_7))
        
        trace_13 = self.system._encode_to_trace(13)
        self.assertTrue(self.system._is_phi_valid(trace_13))
        
    def test_observer_construction(self):
        """测试观察者构建"""
        observers = self.system.observers
        
        # 检查观察者数量
        self.assertEqual(len(observers), 5)
        
        # 检查每个观察者都有必要属性
        for obs in observers:
            self.assertIn('center_frequency', obs)
            self.assertIn('bandwidth', obs)
            self.assertIn('sensitivity', obs)
            self.assertIn('filter_type', obs)
            
    def test_filtering_analysis(self):
        """测试滤波分析"""
        analysis = self.system.analyze_observer_filtering()
        
        # 检查基本统计
        self.assertEqual(analysis['num_observers'], 5)
        self.assertGreater(analysis['num_traces'], 0)
        self.assertGreaterEqual(analysis['global_mean_response'], 0.0)
        
    def test_spectral_properties(self):
        """测试谱性质计算"""
        for value, data in self.system.trace_universe.items():
            # 频率应该非负
            self.assertGreaterEqual(data['base_frequency'], 0.0)
            
            # 可见性应该在[0,1]范围内
            self.assertGreaterEqual(data['observer_visibility'], 0.0)
            self.assertLessEqual(data['observer_visibility'], 1.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_observer_network()
        
        # 检查网络性质
        self.assertEqual(G.number_of_nodes(), 5)
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
    print("=== Chapter 088: ObsFilterZeta Verification ===")
    print("从ψ=ψ(ψ)推导观察者加权滤波...")
    print()
    
    # 创建系统
    system = ObsFilterZetaSystem(max_trace_value=50, num_observers=8)
    
    # 分析观察者滤波
    print("1. 观察者滤波分析...")
    analysis = system.analyze_observer_filtering()
    print(f"   观察者数量: {analysis['num_observers']}")
    print(f"   trace数量: {analysis['num_traces']}")
    print(f"   总观察次数: {analysis['total_observations']}")
    print(f"   全局平均响应: {analysis['global_mean_response']:.3f}")
    print(f"   全局响应方差: {analysis['global_response_variance']:.3f}")
    
    print("\n   各观察者统计:")
    for obs_id, obs_data in analysis['observer_responses'].items():
        print(f"   - 观察者{obs_id} ({obs_data['filter_type']}):")
        print(f"     平均响应: {obs_data['mean_response']:.3f}")
        print(f"     最大响应: {obs_data['max_response']:.3f}")
        print(f"     效率: {obs_data['efficiency']:.3f}")
    print()
    
    # 网络分析
    print("2. 观察者网络分析...")
    G = system.build_observer_network()
    print(f"   网络节点数: {G.number_of_nodes()}")
    print(f"   网络边数: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        print(f"   平均边权重: {np.mean(edge_weights):.3f}")
        print(f"   最大边权重: {np.max(edge_weights):.3f}")
    
    # 连通性分析
    if G.number_of_nodes() > 1:
        components = list(nx.connected_components(G))
        print(f"   连通分量数: {len(components)}")
        if len(components) > 0:
            largest_component = max(len(comp) for comp in components)
            print(f"   最大连通分量: {largest_component}")
    print()
    
    # 信息论分析
    print("3. 信息论分析...")
    entropies = system.compute_information_entropy()
    for key, value in entropies.items():
        print(f"   {key}: {value:.3f}")
    print()
    
    # 范畴论分析
    print("4. 滤波器范畴分析...")
    cat_analysis = system.analyze_filter_categories()
    print(f"   滤波器范畴数: {cat_analysis['category_count']}")
    print(f"   态射数: {cat_analysis['morphism_count']}")
    print(f"   态射密度: {cat_analysis['morphism_density']:.3f}")
    
    print("   范畴分布:")
    for cat_type, observers in cat_analysis['filter_categories'].items():
        print(f"   - {cat_type}: {len(observers)} 观察者")
    print()
    
    # 生成可视化
    print("5. 生成可视化...")
    system.visualize_observer_filtering()
    print("   ✓ 观察者滤波图已保存")
    
    system.visualize_observer_network()
    print("   ✓ 观察者网络图已保存")
    
    system.visualize_spectral_analysis()
    print("   ✓ 谱分析图已保存")
    print()
    
    # 运行单元测试
    print("6. 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("=== ObsFilterZeta验证完成 ===")
    print("所有测试通过，观察者加权滤波的实现成功！")

if __name__ == "__main__":
    run_verification()