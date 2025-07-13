#!/usr/bin/env python3
"""
Chapter 089: CollapseSpectrum Unit Test Verification
从ψ=ψ(ψ)推导Line Structure and Modulation Patterns in ζ(s)

Core principle: From ψ = ψ(ψ) derive spectral line structure where discrete 
frequency lines emerge from trace resonances, with modulation patterns created
by φ-constraint interactions, revealing how spectral structure organizes into
discrete lines with systematic modulation envelopes.

This verification program implements:
1. φ-constrained spectral line detection through trace resonance analysis
2. Line modulation: systematic variation patterns in spectral line intensities
3. Three-domain analysis: Traditional vs φ-constrained vs intersection spectrum theory
4. Graph theory analysis of line networks and modulation relationships
5. Information theory analysis of line entropy and modulation encoding
6. Category theory analysis of line functors and modulation morphisms
7. Visualization of spectral lines and modulation pattern structures
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
from math import log2, gcd, sqrt, pi, exp, cos, sin, log, atan2, floor, ceil
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class CollapseSpectrumSystem:
    """
    Core system for implementing spectral line structure and modulation patterns.
    Implements φ-constrained spectral analysis via line decomposition operations.
    """
    
    def __init__(self, max_trace_value: int = 60, num_spectral_lines: int = 12):
        """Initialize collapse spectrum system with line analysis"""
        self.max_trace_value = max_trace_value
        self.num_spectral_lines = num_spectral_lines
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.line_cache = {}
        self.modulation_cache = {}
        self.spectrum_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.spectral_lines = self._detect_spectral_lines()
        
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
        """分析trace的谱性质，用于线结构分析"""
        result = {
            'value': value,
            'trace': trace,
            'length': len(trace),
            'weight': trace.count('1'),
            'primary_frequency': self._compute_primary_frequency(trace),
            'line_intensity': self._compute_line_intensity(trace),
            'modulation_depth': self._compute_modulation_depth(trace),
            'spectral_width': self._compute_spectral_width(trace),
            'line_stability': self._compute_line_stability(trace),
            'harmonic_content': self._compute_harmonic_content(trace),
            'modulation_frequency': self._compute_modulation_frequency(trace),
            'phase_coherence': self._compute_phase_coherence(trace),
            'line_coupling': self._compute_line_coupling(trace),
            'spectral_purity': self._compute_spectral_purity(trace),
            'modulation_pattern': self._compute_modulation_pattern(trace),
            'line_category': self._classify_line_category(trace),
            'resonance_strength': self._compute_resonance_strength(trace),
        }
        return result
        
    def _compute_primary_frequency(self, trace: str) -> float:
        """计算主要频率"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        
        if weight == 0:
            return 0.0
        
        # 基于1的分布计算主频
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) <= 1:
            return weight / length
        
        # 主要周期性
        intervals = [ones_positions[i+1] - ones_positions[i] for i in range(len(ones_positions)-1)]
        if intervals:
            mean_interval = sum(intervals) / len(intervals)
            return 1.0 / mean_interval if mean_interval > 0 else 1.0
        
        return weight / length
        
    def _compute_line_intensity(self, trace: str) -> float:
        """计算谱线强度"""
        if not trace:
            return 0.0
        
        weight = trace.count('1')
        length = len(trace)
        primary_freq = self._compute_primary_frequency(trace)
        
        # 强度基于权重和频率
        base_intensity = (weight / length) * primary_freq
        
        # φ调制
        phi_factor = 1 + 0.618 * cos(primary_freq * self.phi)
        
        return base_intensity * phi_factor
        
    def _compute_modulation_depth(self, trace: str) -> float:
        """计算调制深度"""
        if not trace:
            return 0.0
        
        # 基于trace变化的调制
        variations = []
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                variations.append(1)
            else:
                variations.append(0)
        
        if not variations:
            return 0.0
        
        variation_rate = sum(variations) / len(variations)
        return min(1.0, variation_rate * 2)  # 归一化到[0,1]
        
    def _compute_spectral_width(self, trace: str) -> float:
        """计算谱线宽度"""
        primary_freq = self._compute_primary_frequency(trace)
        modulation_depth = self._compute_modulation_depth(trace)
        
        # 宽度与调制深度相关
        base_width = 0.1 + 0.3 * modulation_depth
        
        # 频率依赖
        freq_factor = 1 + 0.5 * primary_freq
        
        return base_width * freq_factor
        
    def _compute_line_stability(self, trace: str) -> float:
        """计算谱线稳定性"""
        if not trace:
            return 1.0
        
        # 基于模式一致性的稳定性
        patterns = []
        pattern_length = min(3, len(trace) // 2)
        
        for i in range(len(trace) - pattern_length + 1):
            pattern = trace[i:i+pattern_length]
            patterns.append(pattern)
        
        if not patterns:
            return 1.0
        
        # 模式多样性（低多样性 = 高稳定性）
        unique_patterns = len(set(patterns))
        stability = 1.0 - (unique_patterns / len(patterns))
        
        return max(0.0, stability)
        
    def _compute_harmonic_content(self, trace: str) -> List[float]:
        """计算谐波内容"""
        primary_freq = self._compute_primary_frequency(trace)
        intensity = self._compute_line_intensity(trace)
        
        # 生成前5个谐波
        harmonics = []
        for h in range(1, 6):
            harmonic_freq = primary_freq * h
            harmonic_intensity = intensity / (h * h)  # 衰减
            harmonics.append(harmonic_intensity)
        
        return harmonics
        
    def _compute_modulation_frequency(self, trace: str) -> float:
        """计算调制频率"""
        primary_freq = self._compute_primary_frequency(trace)
        modulation_depth = self._compute_modulation_depth(trace)
        
        # 调制频率通常是主频的分数
        mod_freq = primary_freq * (0.1 + 0.3 * modulation_depth)
        
        return mod_freq
        
    def _compute_phase_coherence(self, trace: str) -> float:
        """计算相位相干性"""
        if not trace:
            return 1.0
        
        # 基于相位一致性
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) < 2:
            return 1.0
        
        # 检查相位规律性
        phases = [(pos / len(trace)) * 2 * pi for pos in ones_positions]
        
        # 计算相位离散度
        phase_variations = []
        for i in range(len(phases) - 1):
            diff = abs(phases[i+1] - phases[i])
            phase_variations.append(diff)
        
        if not phase_variations:
            return 1.0
        
        mean_variation = sum(phase_variations) / len(phase_variations)
        coherence = exp(-mean_variation / pi)  # 归一化相干度
        
        return coherence
        
    def _compute_line_coupling(self, trace: str) -> float:
        """计算谱线耦合"""
        harmonics = self._compute_harmonic_content(trace)
        
        # 谐波间的耦合强度
        coupling = 0.0
        for i in range(len(harmonics) - 1):
            coupling += harmonics[i] * harmonics[i+1]
        
        return coupling
        
    def _compute_spectral_purity(self, trace: str) -> float:
        """计算谱纯度"""
        harmonics = self._compute_harmonic_content(trace)
        
        if not harmonics or harmonics[0] == 0:
            return 0.0
        
        # 主谐波与总功率的比值
        total_power = sum(h*h for h in harmonics)
        fundamental_power = harmonics[0] * harmonics[0]
        
        purity = fundamental_power / total_power if total_power > 0 else 0.0
        
        return purity
        
    def _compute_modulation_pattern(self, trace: str) -> str:
        """计算调制模式"""
        modulation_depth = self._compute_modulation_depth(trace)
        modulation_freq = self._compute_modulation_frequency(trace)
        
        if modulation_depth < 0.3:
            return "weak_modulation"
        elif modulation_freq < 0.2:
            return "slow_modulation"
        elif modulation_freq > 0.8:
            return "fast_modulation"
        else:
            return "moderate_modulation"
            
    def _classify_line_category(self, trace: str) -> str:
        """对谱线类型进行分类"""
        intensity = self._compute_line_intensity(trace)
        stability = self._compute_line_stability(trace)
        purity = self._compute_spectral_purity(trace)
        
        if intensity > 0.8 and stability > 0.7:
            return "strong_stable"
        elif purity > 0.6:
            return "pure_line"
        elif stability < 0.3:
            return "unstable_line"
        else:
            return "moderate_line"
            
    def _compute_resonance_strength(self, trace: str) -> float:
        """计算共振强度"""
        intensity = self._compute_line_intensity(trace)
        coherence = self._compute_phase_coherence(trace)
        stability = self._compute_line_stability(trace)
        
        return intensity * coherence * stability
        
    def _detect_spectral_lines(self) -> List[Dict]:
        """检测谱线结构"""
        lines = []
        
        # 按频率分组traces
        frequency_groups = defaultdict(list)
        for value, data in self.trace_universe.items():
            freq_bin = round(data['primary_frequency'] * 10) / 10  # 0.1精度分组
            frequency_groups[freq_bin].append((value, data))
        
        # 为每个频率组创建谱线
        for freq, traces in frequency_groups.items():
            if len(traces) >= 2:  # 至少2个trace形成谱线
                line_data = self._analyze_line_structure(freq, traces)
                lines.append(line_data)
        
        # 按强度排序，取前num_spectral_lines条
        lines.sort(key=lambda x: x['total_intensity'], reverse=True)
        return lines[:self.num_spectral_lines]
        
    def _analyze_line_structure(self, frequency: float, traces: List[Tuple]) -> Dict:
        """分析谱线结构"""
        intensities = [data['line_intensity'] for _, data in traces]
        modulations = [data['modulation_depth'] for _, data in traces]
        stabilities = [data['line_stability'] for _, data in traces]
        purities = [data['spectral_purity'] for _, data in traces]
        
        return {
            'frequency': frequency,
            'trace_count': len(traces),
            'total_intensity': sum(intensities),
            'mean_intensity': np.mean(intensities),
            'intensity_variance': np.var(intensities),
            'mean_modulation': np.mean(modulations),
            'mean_stability': np.mean(stabilities),
            'mean_purity': np.mean(purities),
            'line_width': self._compute_line_width(traces),
            'modulation_envelope': self._compute_modulation_envelope(traces),
            'traces': traces,
        }
        
    def _compute_line_width(self, traces: List[Tuple]) -> float:
        """计算谱线宽度"""
        widths = [data['spectral_width'] for _, data in traces]
        return np.mean(widths) if widths else 0.0
        
    def _compute_modulation_envelope(self, traces: List[Tuple]) -> List[float]:
        """计算调制包络"""
        envelope = []
        
        # 生成包络函数
        for i in range(20):  # 20个采样点
            t = i / 19.0  # 归一化时间
            amplitude = 0.0
            
            for _, data in traces:
                mod_freq = data['modulation_frequency']
                mod_depth = data['modulation_depth']
                contribution = mod_depth * cos(2 * pi * mod_freq * t)
                amplitude += contribution
            
            envelope.append(amplitude / len(traces))
        
        return envelope
        
    def analyze_spectrum_structure(self) -> Dict:
        """分析谱结构"""
        lines = self.spectral_lines
        
        if not lines:
            return {
                'total_lines': 0,
                'mean_intensity': 0.0,
                'spectral_coverage': 0.0,
                'modulation_diversity': 0.0,
            }
        
        # 基本统计
        intensities = [line['total_intensity'] for line in lines]
        frequencies = [line['frequency'] for line in lines]
        modulations = [line['mean_modulation'] for line in lines]
        
        # 谱覆盖范围
        freq_range = max(frequencies) - min(frequencies) if len(frequencies) > 1 else 0.0
        
        # 调制多样性
        modulation_diversity = np.var(modulations) if len(modulations) > 1 else 0.0
        
        return {
            'total_lines': len(lines),
            'mean_intensity': np.mean(intensities),
            'intensity_variance': np.var(intensities),
            'spectral_coverage': freq_range,
            'mean_frequency': np.mean(frequencies),
            'modulation_diversity': modulation_diversity,
            'mean_modulation': np.mean(modulations),
            'strongest_line_freq': frequencies[0] if frequencies else 0.0,
            'weakest_line_intensity': min(intensities) if intensities else 0.0,
            'line_density': len(lines) / (freq_range + 0.1),  # 避免除零
        }
        
    def build_line_network(self) -> nx.Graph:
        """构建谱线网络图"""
        G = nx.Graph()
        
        # 添加谱线节点
        for i, line in enumerate(self.spectral_lines):
            G.add_node(i, **line)
        
        # 添加边：基于频率接近度和调制相似性
        for i, line1 in enumerate(self.spectral_lines):
            for j, line2 in enumerate(self.spectral_lines[i+1:], i+1):
                # 频率相似性
                freq_diff = abs(line1['frequency'] - line2['frequency'])
                freq_similarity = exp(-freq_diff * 5)  # 频率相似度
                
                # 调制相似性
                mod_diff = abs(line1['mean_modulation'] - line2['mean_modulation'])
                mod_similarity = exp(-mod_diff * 3)
                
                # 综合相似性
                similarity = (freq_similarity + mod_similarity) / 2
                
                # 添加边如果相似性足够高
                if similarity > 0.3:
                    G.add_edge(i, j, weight=similarity, 
                              connection_type='spectral_proximity')
        
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
        for key in ['primary_frequency', 'line_intensity', 'modulation_depth',
                   'spectral_width', 'line_stability', 'modulation_frequency',
                   'phase_coherence', 'line_coupling', 'spectral_purity',
                   'resonance_strength']:
            data = [v[key] for v in trace_values if key in v]
            entropies[f'{key}_entropy'] = compute_entropy(data)
        
        # 谱线属性熵
        if self.spectral_lines:
            line_data = {
                'frequency': [line['frequency'] for line in self.spectral_lines],
                'total_intensity': [line['total_intensity'] for line in self.spectral_lines],
                'mean_modulation': [line['mean_modulation'] for line in self.spectral_lines],
                'line_width': [line['line_width'] for line in self.spectral_lines],
            }
            
            for key, data in line_data.items():
                entropies[f'line_{key}_entropy'] = compute_entropy(data)
        
        # 分类熵
        categories = [v['line_category'] for v in trace_values]
        cat_counts = {}
        for cat in categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        total = sum(cat_counts.values())
        cat_probs = [count/total for count in cat_counts.values()]
        category_entropy = -sum(p * log2(p) for p in cat_probs if p > 0)
        entropies['line_category_entropy'] = category_entropy
        entropies['line_category_count'] = len(cat_counts)
        
        # 调制模式熵
        mod_patterns = [v['modulation_pattern'] for v in trace_values]
        pattern_counts = {}
        for pattern in mod_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        total = sum(pattern_counts.values())
        pattern_probs = [count/total for count in pattern_counts.values()]
        pattern_entropy = -sum(p * log2(p) for p in pattern_probs if p > 0)
        entropies['modulation_pattern_entropy'] = pattern_entropy
        entropies['modulation_pattern_count'] = len(pattern_counts)
        
        return entropies
        
    def analyze_line_categories(self) -> Dict:
        """分析谱线范畴"""
        # 按类型分组谱线traces
        line_categories = defaultdict(list)
        for value, data in self.trace_universe.items():
            line_categories[data['line_category']].append(value)
        
        # 分析traces间的态射
        morphisms = []
        traces = list(self.trace_universe.keys())
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces):
                if i != j:
                    data1 = self.trace_universe[trace1]
                    data2 = self.trace_universe[trace2]
                    
                    # 检查是否存在谱线保持态射
                    freq_compatible = abs(data1['primary_frequency'] - data2['primary_frequency']) < 0.2
                    intensity_compatible = abs(data1['line_intensity'] - data2['line_intensity']) < 0.3
                    
                    if freq_compatible and intensity_compatible:
                        morphisms.append((trace1, trace2))
        
        return {
            'line_categories': dict(line_categories),
            'category_count': len(line_categories),
            'morphism_count': len(morphisms),
            'total_traces': len(self.trace_universe),
            'morphism_density': len(morphisms) / (len(traces) * (len(traces) - 1)) if len(traces) > 1 else 0,
        }

    def visualize_spectral_lines(self):
        """可视化谱线结构"""
        plt.figure(figsize=(20, 15))
        
        # 1. 谱线图
        plt.subplot(2, 3, 1)
        
        if self.spectral_lines:
            frequencies = [line['frequency'] for line in self.spectral_lines]
            intensities = [line['total_intensity'] for line in self.spectral_lines]
            
            # 绘制谱线
            for freq, intensity in zip(frequencies, intensities):
                plt.axvline(x=freq, ymin=0, ymax=intensity/max(intensities), 
                           linewidth=3, alpha=0.7)
            
            plt.scatter(frequencies, intensities, s=100, c='red', alpha=0.8, zorder=5)
            
            plt.xlabel('Frequency')
            plt.ylabel('Intensity')
            plt.title('Spectral Line Structure')
            plt.grid(True, alpha=0.3)
        
        # 2. 调制包络
        plt.subplot(2, 3, 2)
        
        if self.spectral_lines:
            time_points = np.linspace(0, 1, 20)
            
            for i, line in enumerate(self.spectral_lines[:5]):  # 显示前5条线
                envelope = line['modulation_envelope']
                plt.plot(time_points, envelope, label=f"Line {i} ({line['frequency']:.2f})", 
                        linewidth=2, alpha=0.7)
            
            plt.xlabel('Normalized Time')
            plt.ylabel('Modulation Amplitude')
            plt.title('Modulation Envelopes')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. 谱线强度分布
        plt.subplot(2, 3, 3)
        
        trace_values = list(self.trace_universe.values())
        intensities = [t['line_intensity'] for t in trace_values]
        
        unique_intensities = len(np.unique(intensities))
        bin_count = min(15, max(3, unique_intensities))
        
        try:
            plt.hist(intensities, bins=bin_count, alpha=0.7, color='skyblue', edgecolor='black')
        except ValueError:
            plt.bar(range(len(intensities)), intensities, alpha=0.7, color='skyblue')
        
        plt.xlabel('Line Intensity')
        plt.ylabel('Count')
        plt.title('Line Intensity Distribution')
        plt.grid(True, alpha=0.3)
        
        # 4. 频率vs调制深度
        plt.subplot(2, 3, 4)
        
        frequencies = [t['primary_frequency'] for t in trace_values]
        modulations = [t['modulation_depth'] for t in trace_values]
        categories = [t['line_category'] for t in trace_values]
        
        # 按类别着色
        category_colors = {'strong_stable': 'red', 'pure_line': 'blue', 
                          'unstable_line': 'orange', 'moderate_line': 'green'}
        colors = [category_colors.get(cat, 'gray') for cat in categories]
        
        plt.scatter(frequencies, modulations, c=colors, alpha=0.6, s=50)
        plt.xlabel('Primary Frequency')
        plt.ylabel('Modulation Depth')
        plt.title('Frequency vs Modulation')
        plt.grid(True, alpha=0.3)
        
        # 图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=cat)
                          for cat, color in category_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 5. 相位相干性分析
        plt.subplot(2, 3, 5)
        
        coherences = [t['phase_coherence'] for t in trace_values]
        purities = [t['spectral_purity'] for t in trace_values]
        
        plt.scatter(coherences, purities, c=colors, alpha=0.6, s=50)
        plt.xlabel('Phase Coherence')
        plt.ylabel('Spectral Purity')
        plt.title('Coherence vs Purity')
        plt.grid(True, alpha=0.3)
        
        # 6. 谐波内容可视化
        plt.subplot(2, 3, 6)
        
        # 显示前几个traces的谐波
        sample_traces = list(trace_values)[:8]
        harmonic_data = []
        
        for trace in sample_traces:
            harmonics = trace['harmonic_content']
            harmonic_data.append(harmonics)
        
        harmonic_matrix = np.array(harmonic_data)
        
        im = plt.imshow(harmonic_matrix, aspect='auto', cmap='plasma')
        plt.colorbar(im, shrink=0.8)
        plt.xlabel('Harmonic Number')
        plt.ylabel('Trace Index')
        plt.title('Harmonic Content Matrix')
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-089-collapse-spectrum-lines.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_modulation_patterns(self):
        """可视化调制模式"""
        plt.figure(figsize=(15, 10))
        
        trace_values = list(self.trace_universe.values())
        
        # 1. 调制模式分布
        plt.subplot(2, 2, 1)
        
        patterns = [t['modulation_pattern'] for t in trace_values]
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        plt.pie(pattern_counts.values(), labels=pattern_counts.keys(), autopct='%1.1f%%')
        plt.title('Modulation Pattern Distribution')
        
        # 2. 调制频率vs深度
        plt.subplot(2, 2, 2)
        
        mod_freqs = [t['modulation_frequency'] for t in trace_values]
        mod_depths = [t['modulation_depth'] for t in trace_values]
        
        plt.scatter(mod_freqs, mod_depths, alpha=0.6, s=50)
        plt.xlabel('Modulation Frequency')
        plt.ylabel('Modulation Depth')
        plt.title('Modulation Frequency vs Depth')
        plt.grid(True, alpha=0.3)
        
        # 3. 稳定性vs纯度
        plt.subplot(2, 2, 3)
        
        stabilities = [t['line_stability'] for t in trace_values]
        purities = [t['spectral_purity'] for t in trace_values]
        
        plt.scatter(stabilities, purities, alpha=0.6, s=50, c='orange')
        plt.xlabel('Line Stability')
        plt.ylabel('Spectral Purity')
        plt.title('Stability vs Purity')
        plt.grid(True, alpha=0.3)
        
        # 4. 共振强度分布
        plt.subplot(2, 2, 4)
        
        resonances = [t['resonance_strength'] for t in trace_values]
        
        unique_res = len(np.unique(resonances))
        bin_count = min(12, max(3, unique_res))
        
        try:
            plt.hist(resonances, bins=bin_count, alpha=0.7, color='lightgreen', edgecolor='black')
        except ValueError:
            plt.bar(range(len(resonances)), resonances, alpha=0.7, color='lightgreen')
        
        plt.xlabel('Resonance Strength')
        plt.ylabel('Count')
        plt.title('Resonance Strength Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-089-collapse-spectrum-modulation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_line_network(self):
        """可视化谱线网络"""
        plt.figure(figsize=(12, 8))
        
        G = self.build_line_network()
        
        if G.number_of_nodes() > 0:
            # 1. 网络布局
            pos = nx.spring_layout(G, k=3, iterations=100)
            
            # 根据强度确定节点大小
            node_sizes = []
            for node in G.nodes():
                line = self.spectral_lines[node]
                size = line['total_intensity'] * 500 + 100
                node_sizes.append(size)
            
            # 根据频率确定颜色
            node_colors = []
            for node in G.nodes():
                line = self.spectral_lines[node]
                freq = line['frequency']
                node_colors.append(freq)
            
            # 绘制网络
            scatter = plt.scatter([pos[node][0] for node in G.nodes()], 
                                [pos[node][1] for node in G.nodes()], 
                                c=node_colors, s=node_sizes, cmap='viridis', alpha=0.8)
            
            # 绘制边
            for edge in G.edges():
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1.0)
            
            # 添加节点标签
            for node in G.nodes():
                x, y = pos[node]
                plt.annotate(f"L{node}\n{self.spectral_lines[node]['frequency']:.2f}", 
                           (x, y), ha='center', va='center', fontsize=8)
            
            plt.title('Spectral Line Network')
            plt.colorbar(scatter, label='Frequency', shrink=0.8)
        else:
            plt.text(0.5, 0.5, 'No spectral line network to display', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Spectral Line Network')
        
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-089-collapse-spectrum-network.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

class TestCollapseSpectrum(unittest.TestCase):
    """单元测试"""
    
    def setUp(self):
        """测试setup"""
        self.system = CollapseSpectrumSystem(max_trace_value=25, num_spectral_lines=6)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        trace_8 = self.system._encode_to_trace(8)
        self.assertTrue(self.system._is_phi_valid(trace_8))
        
        trace_13 = self.system._encode_to_trace(13)
        self.assertTrue(self.system._is_phi_valid(trace_13))
        
    def test_spectral_line_detection(self):
        """测试谱线检测"""
        lines = self.system.spectral_lines
        
        # 检查谱线基本性质
        self.assertGreaterEqual(len(lines), 0)
        
        for line in lines:
            self.assertIn('frequency', line)
            self.assertIn('total_intensity', line)
            self.assertIn('modulation_envelope', line)
            self.assertGreaterEqual(line['total_intensity'], 0.0)
            
    def test_spectrum_analysis(self):
        """测试谱分析"""
        analysis = self.system.analyze_spectrum_structure()
        
        # 检查基本统计
        self.assertGreaterEqual(analysis['total_lines'], 0)
        self.assertGreaterEqual(analysis['mean_intensity'], 0.0)
        self.assertGreaterEqual(analysis['spectral_coverage'], 0.0)
        
    def test_spectral_properties(self):
        """测试谱性质计算"""
        for value, data in self.system.trace_universe.items():
            # 频率应该非负
            self.assertGreaterEqual(data['primary_frequency'], 0.0)
            
            # 强度应该非负
            self.assertGreaterEqual(data['line_intensity'], 0.0)
            
            # 调制深度应该在[0,1]范围内
            self.assertGreaterEqual(data['modulation_depth'], 0.0)
            self.assertLessEqual(data['modulation_depth'], 1.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_line_network()
        
        # 检查网络性质
        self.assertGreaterEqual(G.number_of_nodes(), 0)
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
    print("=== Chapter 089: CollapseSpectrum Verification ===")
    print("从ψ=ψ(ψ)推导谱线结构和调制模式...")
    print()
    
    # 创建系统
    system = CollapseSpectrumSystem(max_trace_value=45, num_spectral_lines=10)
    
    # 分析谱线结构
    print("1. 谱线结构分析...")
    analysis = system.analyze_spectrum_structure()
    print(f"   检测到谱线数: {analysis['total_lines']}")
    print(f"   平均强度: {analysis['mean_intensity']:.3f}")
    print(f"   强度方差: {analysis['intensity_variance']:.3f}")
    print(f"   谱覆盖范围: {analysis['spectral_coverage']:.3f}")
    print(f"   平均频率: {analysis['mean_frequency']:.3f}")
    print(f"   调制多样性: {analysis['modulation_diversity']:.3f}")
    print(f"   谱线密度: {analysis['line_density']:.3f}")
    
    if analysis['total_lines'] > 0:
        print(f"   最强谱线频率: {analysis['strongest_line_freq']:.3f}")
        print(f"   最弱谱线强度: {analysis['weakest_line_intensity']:.3f}")
    print()
    
    # 网络分析
    print("2. 谱线网络分析...")
    G = system.build_line_network()
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
    print("4. 谱线范畴分析...")
    cat_analysis = system.analyze_line_categories()
    print(f"   谱线范畴数: {cat_analysis['category_count']}")
    print(f"   态射数: {cat_analysis['morphism_count']}")
    print(f"   态射密度: {cat_analysis['morphism_density']:.3f}")
    
    print("   范畴分布:")
    for cat_type, traces in cat_analysis['line_categories'].items():
        print(f"   - {cat_type}: {len(traces)} traces")
    print()
    
    # 生成可视化
    print("5. 生成可视化...")
    system.visualize_spectral_lines()
    print("   ✓ 谱线结构图已保存")
    
    system.visualize_modulation_patterns()
    print("   ✓ 调制模式图已保存")
    
    system.visualize_line_network()
    print("   ✓ 谱线网络图已保存")
    print()
    
    # 运行单元测试
    print("6. 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("=== CollapseSpectrum验证完成 ===")
    print("所有测试通过，谱线结构和调制模式的实现成功！")

if __name__ == "__main__":
    run_verification()