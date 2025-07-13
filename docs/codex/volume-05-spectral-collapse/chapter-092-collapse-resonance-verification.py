#!/usr/bin/env python3
"""
Chapter 092: CollapseResonance Unit Test Verification
从ψ=ψ(ψ)推导ζ-Aligned Trace Structures in Spectral Collapse Systems

Core principle: From ψ = ψ(ψ) derive resonant mode structures where traces
align with zeta function patterns, creating systematic resonance networks that
amplify specific spectral frequencies while damping others, revealing how
φ-constraints create the fundamental resonance architecture of collapse space.

This verification program implements:
1. φ-constrained resonance detection through ζ-alignment analysis
2. Resonant mode identification: systematic frequency amplification patterns
3. Three-domain analysis: Traditional vs φ-constrained vs intersection resonance theory
4. Graph theory analysis of resonance networks and coupling structures
5. Information theory analysis of resonance entropy and frequency encoding
6. Category theory analysis of resonance functors and mode morphisms
7. Visualization of resonance structures and ζ-aligned patterns
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

class CollapseResonanceSystem:
    """
    Core system for implementing ζ-aligned trace structures in spectral collapse systems.
    Implements φ-constrained resonance via ζ-function alignment operations.
    """
    
    def __init__(self, max_trace_value: int = 100, num_resonance_modes: int = 10):
        """Initialize collapse resonance system with ζ-alignment analysis"""
        self.max_trace_value = max_trace_value
        self.num_resonance_modes = num_resonance_modes
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.resonance_cache = {}
        self.zeta_cache = {}
        self.alignment_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.resonance_modes = self._detect_resonance_modes()
        self.zeta_alignment = self._compute_zeta_alignment()
        
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
                resonance_data = self._analyze_resonance_properties(trace, n)
                universe[n] = resonance_data
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
        
    def _analyze_resonance_properties(self, trace: str, value: int) -> Dict:
        """分析trace的resonance性质，用于ζ-alignment分析"""
        # Core resonance analysis
        zeta_frequency = self._compute_zeta_frequency(trace, value)
        resonance_amplitude = self._compute_resonance_amplitude(trace, value)
        mode_coupling = self._compute_mode_coupling(trace)
        phase_alignment = self._compute_phase_alignment(trace, value)
        
        # ζ-specific properties
        zeta_alignment = self._compute_local_zeta_alignment(trace, value)
        resonance_strength = self._compute_resonance_strength(trace, value)
        damping_factor = self._compute_damping_factor(trace)
        harmonic_content = self._compute_harmonic_content(trace, value)
        
        return {
            'value': value,
            'trace': trace,
            'zeta_frequency': zeta_frequency,
            'resonance_amplitude': resonance_amplitude,
            'mode_coupling': mode_coupling,
            'phase_alignment': phase_alignment,
            'zeta_alignment': zeta_alignment,
            'resonance_strength': resonance_strength,
            'damping_factor': damping_factor,
            'harmonic_content': harmonic_content,
            'resonance_classification': self._classify_resonance_mode(trace, value),
            'mode_index': self._determine_mode_index(zeta_frequency)
        }
        
    def _compute_zeta_frequency(self, trace: str, value: int) -> float:
        """计算ζ-aligned frequency（基于黎曼ζ函数的共振频率）"""
        if value == 0:
            return 0.0
            
        # Frequency aligned with zeta function critical line
        critical_line_real = 0.5  # Re(s) = 1/2 for critical line
        
        # Imaginary part based on trace structure and value
        imaginary_part = log(value + 1) / log(2)
        
        # Modulation based on trace pattern
        trace_modulation = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Zeta-like oscillation
                trace_modulation += cos(pi * (i + 1) / len(trace)) / (i + 1)
                
        zeta_frequency = imaginary_part + trace_modulation / len(trace)
        
        # Normalize to [0, 2π] for consistent frequency analysis
        return (zeta_frequency % (2 * pi))
        
    def _compute_resonance_amplitude(self, trace: str, value: int) -> float:
        """计算resonance amplitude（共振幅度）"""
        if not trace or value == 0:
            return 0.0
            
        # Amplitude based on trace density and structure
        ones_count = trace.count('1')
        trace_length = len(trace)
        
        if trace_length == 0:
            return 0.0
            
        # Base amplitude from density
        density_amplitude = ones_count / trace_length
        
        # Enhancement from ζ-alignment
        zeta_freq = self._compute_zeta_frequency(trace, value)
        zeta_enhancement = abs(sin(zeta_freq)) * abs(cos(zeta_freq / 2))
        
        return density_amplitude * zeta_enhancement
        
    def _compute_mode_coupling(self, trace: str) -> float:
        """计算mode coupling strength（模式耦合强度）"""
        if len(trace) <= 1:
            return 0.0
            
        # Coupling based on transitions and patterns
        coupling = 0.0
        
        # Count pattern variations
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                coupling += 1.0
                
        # Normalize by trace length
        coupling /= (len(trace) - 1)
        
        # Enhance based on φ-constraint satisfaction
        if "11" not in trace:
            coupling *= self.phi  # φ-enhancement
            
        return min(coupling, 1.0)  # Cap at 1.0
        
    def _compute_phase_alignment(self, trace: str, value: int) -> float:
        """计算phase alignment with ζ-function（与ζ函数的相位对齐）"""
        if not trace or value == 0:
            return 0.0
            
        # Phase based on zeta function critical line behavior
        zeta_freq = self._compute_zeta_frequency(trace, value)
        
        # Alignment measure using trace structure
        alignment = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Phase contribution aligned with zeta zeros pattern
                phase_contrib = cos(zeta_freq * (i + 1) / len(trace))
                alignment += phase_contrib
                
        return abs(alignment) / max(trace.count('1'), 1)
        
    def _compute_local_zeta_alignment(self, trace: str, value: int) -> float:
        """计算local ζ-alignment score（局部ζ对齐分数）"""
        if value == 0:
            return 0.0
            
        # Simplified zeta-like function evaluation
        s_real = 0.5  # Critical line
        s_imag = self._compute_zeta_frequency(trace, value)
        
        # Approximate |ζ(s)| behavior for alignment
        zeta_magnitude = 1.0 / sqrt(1 + s_imag * s_imag)
        
        # Alignment based on trace harmonics matching zeta oscillations
        harmonic_sum = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                harmonic_sum += 1.0 / ((i + 1) ** s_real) * cos(s_imag * log(i + 1))
                
        alignment = abs(harmonic_sum) * zeta_magnitude
        
        return min(alignment, 1.0)
        
    def _compute_resonance_strength(self, trace: str, value: int) -> float:
        """计算resonance strength（共振强度）"""
        if not trace:
            return 0.0
            
        # Strength based on amplitude and alignment
        amplitude = self._compute_resonance_amplitude(trace, value)
        alignment = self._compute_local_zeta_alignment(trace, value)
        coupling = self._compute_mode_coupling(trace)
        
        # Combined strength with φ-weighting
        strength = amplitude * alignment * coupling * self.phi
        
        return min(strength, 1.0)
        
    def _compute_damping_factor(self, trace: str) -> float:
        """计算damping factor（阻尼因子）"""
        if not trace:
            return 1.0  # Maximum damping
            
        # Damping inversely related to trace organization
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                transitions += 1
                
        # More transitions = less organization = more damping
        if len(trace) <= 1:
            return 1.0
            
        damping = transitions / (len(trace) - 1)
        
        # φ-constraint reduces damping
        if "11" not in trace:
            damping /= self.phi
            
        return min(damping, 1.0)
        
    def _compute_harmonic_content(self, trace: str, value: int) -> float:
        """计算harmonic content（谐波含量）"""
        if not trace or value == 0:
            return 0.0
            
        # Harmonic content based on trace periodicity and structure
        harmonic_sum = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                # Harmonic contribution
                frequency = 2 * pi * (i + 1) / len(trace)
                harmonic_sum += sin(frequency * value) / (i + 1)
                
        return abs(harmonic_sum) / len(trace)
        
    def _classify_resonance_mode(self, trace: str, value: int) -> str:
        """分类resonance mode类型"""
        strength = self._compute_resonance_strength(trace, value)
        alignment = self._compute_local_zeta_alignment(trace, value)
        damping = self._compute_damping_factor(trace)
        
        if strength > 0.7 and alignment > 0.6:
            return "strong_resonance"
        elif strength > 0.4 and alignment > 0.3:
            return "moderate_resonance"
        elif damping > 0.7:
            return "damped_mode"
        else:
            return "weak_resonance"
            
    def _determine_mode_index(self, frequency: float) -> int:
        """确定frequency所属的mode index"""
        # Divide frequency space into mode regions
        mode_width = (2 * pi) / self.num_resonance_modes
        mode_index = int(frequency / mode_width)
        return min(mode_index, self.num_resonance_modes - 1)
        
    def _detect_resonance_modes(self) -> Dict:
        """检测系统中的resonance modes"""
        modes = {}
        
        # Group traces by mode index
        for value, data in self.trace_universe.items():
            mode_idx = data['mode_index']
            if mode_idx not in modes:
                modes[mode_idx] = []
            modes[mode_idx].append(value)
            
        # Analyze each mode
        mode_analysis = {}
        for mode_idx, traces in modes.items():
            if not traces:
                continue
                
            mode_data = [self.trace_universe[t] for t in traces]
            
            analysis = {
                'mode_index': mode_idx,
                'trace_count': len(traces),
                'mean_strength': np.mean([d['resonance_strength'] for d in mode_data]),
                'mean_alignment': np.mean([d['zeta_alignment'] for d in mode_data]),
                'mean_damping': np.mean([d['damping_factor'] for d in mode_data]),
                'frequency_range': (
                    np.min([d['zeta_frequency'] for d in mode_data]),
                    np.max([d['zeta_frequency'] for d in mode_data])
                ),
                'dominant_classification': self._find_dominant_classification(mode_data)
            }
            
            mode_analysis[mode_idx] = analysis
            
        return mode_analysis
        
    def _find_dominant_classification(self, mode_data: List[Dict]) -> str:
        """找到mode中的主导classification"""
        classifications = [d['resonance_classification'] for d in mode_data]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        return max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else "unknown"
        
    def _compute_zeta_alignment(self) -> Dict:
        """计算全局ζ-alignment patterns"""
        all_traces = list(self.trace_universe.values())
        
        # Global alignment statistics
        alignments = [t['zeta_alignment'] for t in all_traces]
        strengths = [t['resonance_strength'] for t in all_traces]
        
        # Correlation analysis
        alignment_strength_corr = np.corrcoef(alignments, strengths)[0, 1] if len(alignments) > 1 else 0.0
        
        # Distribution analysis
        strong_aligned = sum(1 for t in all_traces if t['zeta_alignment'] > 0.5 and t['resonance_strength'] > 0.5)
        
        return {
            'mean_alignment': np.mean(alignments),
            'alignment_std': np.std(alignments),
            'mean_strength': np.mean(strengths),
            'strength_std': np.std(strengths),
            'alignment_strength_correlation': alignment_strength_corr,
            'strong_aligned_count': strong_aligned,
            'strong_aligned_fraction': strong_aligned / len(all_traces)
        }
        
    def analyze_resonance_properties(self) -> Dict:
        """分析resonance系统的全局性质"""
        all_traces = list(self.trace_universe.values())
        
        # Statistical analysis
        frequencies = [t['zeta_frequency'] for t in all_traces]
        amplitudes = [t['resonance_amplitude'] for t in all_traces]
        strengths = [t['resonance_strength'] for t in all_traces]
        alignments = [t['zeta_alignment'] for t in all_traces]
        dampings = [t['damping_factor'] for t in all_traces]
        couplings = [t['mode_coupling'] for t in all_traces]
        harmonics = [t['harmonic_content'] for t in all_traces]
        
        # Classification distribution
        classifications = [t['resonance_classification'] for t in all_traces]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        # Mode distribution
        mode_indices = [t['mode_index'] for t in all_traces]
        mode_counts = {}
        for mode in mode_indices:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
        return {
            'total_traces': len(all_traces),
            'frequency_stats': {
                'mean': np.mean(frequencies),
                'std': np.std(frequencies),
                'range': np.max(frequencies) - np.min(frequencies)
            },
            'amplitude_stats': {
                'mean': np.mean(amplitudes),
                'std': np.std(amplitudes),
                'max': np.max(amplitudes)
            },
            'strength_stats': {
                'mean': np.mean(strengths),
                'std': np.std(strengths),
                'max': np.max(strengths)
            },
            'alignment_stats': {
                'mean': np.mean(alignments),
                'std': np.std(alignments)
            },
            'damping_stats': {
                'mean': np.mean(dampings),
                'std': np.std(dampings)
            },
            'coupling_stats': {
                'mean': np.mean(couplings),
                'std': np.std(couplings)
            },
            'harmonic_stats': {
                'mean': np.mean(harmonics),
                'std': np.std(harmonics)
            },
            'classification_distribution': class_counts,
            'mode_distribution': mode_counts,
            'resonance_modes': self.resonance_modes,
            'zeta_alignment': self.zeta_alignment
        }
        
    def compute_entropy_measures(self) -> Dict:
        """计算resonance系统的信息论测度"""
        all_traces = list(self.trace_universe.values())
        
        # Extract properties for entropy calculation
        properties = {
            'zeta_frequency': [t['zeta_frequency'] for t in all_traces],
            'resonance_amplitude': [t['resonance_amplitude'] for t in all_traces],
            'mode_coupling': [t['mode_coupling'] for t in all_traces],
            'phase_alignment': [t['phase_alignment'] for t in all_traces],
            'zeta_alignment': [t['zeta_alignment'] for t in all_traces],
            'resonance_strength': [t['resonance_strength'] for t in all_traces],
            'damping_factor': [t['damping_factor'] for t in all_traces],
            'harmonic_content': [t['harmonic_content'] for t in all_traces],
            'mode_index': [t['mode_index'] for t in all_traces],
            'resonance_classification': [t['resonance_classification'] for t in all_traces]
        }
        
        entropies = {}
        for prop_name, values in properties.items():
            if prop_name in ['mode_index', 'resonance_classification']:
                # Discrete entropy
                entropies[prop_name] = self._compute_discrete_entropy(values)
            else:
                # Continuous entropy (via binning)
                entropies[prop_name] = self._compute_continuous_entropy(values)
                
        return entropies
        
    def _compute_discrete_entropy(self, values: List) -> float:
        """计算离散值的熵"""
        if not values:
            return 0.0
            
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
            
        total = len(values)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)
                
        return entropy
        
    def _compute_continuous_entropy(self, values: List) -> float:
        """计算连续值的熵（通过分箱）"""
        if not values:
            return 0.0
            
        # Check for constant values
        if len(set(values)) == 1:
            return 0.0
            
        # Adaptive binning
        unique_values = len(set(values))
        bin_count = min(8, max(3, unique_values))
        
        try:
            hist, _ = np.histogram(values, bins=bin_count)
            total = sum(hist)
            if total == 0:
                return 0.0
                
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / total
                    entropy -= p * log2(p)
                    
            return entropy
        except:
            # Fallback to discrete entropy
            return self._compute_discrete_entropy(values)
            
    def build_resonance_network(self) -> nx.Graph:
        """构建resonance network"""
        G = nx.Graph()
        
        # Add nodes for each trace
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add edges between resonantly coupled traces
        traces = list(self.trace_universe.items())
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces[i+1:], i+1):
                # Connect if frequencies are similar or harmonically related
                freq_diff = abs(data1['zeta_frequency'] - data2['zeta_frequency'])
                harmonic_relation = abs(data1['zeta_frequency'] - 2 * data2['zeta_frequency']) % (2 * pi)
                
                coupling_strength = (data1['mode_coupling'] + data2['mode_coupling']) / 2
                resonance_product = data1['resonance_strength'] * data2['resonance_strength']
                
                if (freq_diff < 0.5 or harmonic_relation < 0.3) and coupling_strength > 0.2:
                    weight = coupling_strength * resonance_product
                    G.add_edge(val1, val2, weight=weight)
                    
        return G
        
    def analyze_categorical_structure(self) -> Dict:
        """分析resonance系统的范畴论结构"""
        # Group traces by resonance classification
        categories = {}
        for value, data in self.trace_universe.items():
            cls = data['resonance_classification']
            if cls not in categories:
                categories[cls] = []
            categories[cls].append(value)
            
        # Count morphisms (connections) between categories
        G = self.build_resonance_network()
        morphisms = {}
        total_morphisms = 0
        
        for cls1 in categories:
            for cls2 in categories:
                count = 0
                for v1 in categories[cls1]:
                    for v2 in categories[cls2]:
                        if G.has_edge(v1, v2):
                            count += 1
                morphisms[f"{cls1}->{cls2}"] = count
                total_morphisms += count
                
        # Morphism density
        total_possible = sum(len(cat1) * len(cat2) for cat1 in categories.values() 
                           for cat2 in categories.values())
        morphism_density = total_morphisms / total_possible if total_possible > 0 else 0
        
        return {
            'categories': {cls: len(traces) for cls, traces in categories.items()},
            'morphisms': morphisms,
            'total_morphisms': total_morphisms,
            'morphism_density': morphism_density,
            'category_count': len(categories)
        }

class TestCollapseResonance(unittest.TestCase):
    """测试collapse resonance系统的各项功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseResonanceSystem(max_trace_value=90, num_resonance_modes=8)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 10)
        
        # 检查基本trace性质
        for value, data in self.system.trace_universe.items():
            self.assertIn('zeta_frequency', data)
            self.assertIn('resonance_strength', data)
            self.assertIn('zeta_alignment', data)
            self.assertIn('resonance_classification', data)
            
    def test_zeta_frequency_computation(self):
        """测试ζ频率计算"""
        for value, data in self.system.trace_universe.items():
            freq = data['zeta_frequency']
            
            # Check frequency bounds
            self.assertGreaterEqual(freq, 0.0)
            self.assertLessEqual(freq, 2 * pi)
            
    def test_resonance_mode_detection(self):
        """测试resonance mode检测"""
        modes = self.system.resonance_modes
        
        self.assertIsInstance(modes, dict)
        
        # Check mode properties
        for mode_idx, analysis in modes.items():
            self.assertIn('trace_count', analysis)
            self.assertIn('mean_strength', analysis)
            self.assertIn('mean_alignment', analysis)
            self.assertGreaterEqual(analysis['mean_strength'], 0.0)
            self.assertLessEqual(analysis['mean_strength'], 1.0)
            
    def test_zeta_alignment_analysis(self):
        """测试ζ对齐分析"""
        alignment = self.system.zeta_alignment
        
        self.assertIn('mean_alignment', alignment)
        self.assertIn('mean_strength', alignment)
        self.assertIn('alignment_strength_correlation', alignment)
        
        # Check bounds
        self.assertGreaterEqual(alignment['mean_alignment'], 0.0)
        self.assertLessEqual(alignment['mean_alignment'], 1.0)
        
    def test_resonance_properties(self):
        """测试resonance性质"""
        props = self.system.analyze_resonance_properties()
        
        self.assertIn('frequency_stats', props)
        self.assertIn('strength_stats', props)
        self.assertIn('classification_distribution', props)
        self.assertGreater(props['total_traces'], 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropies = self.system.compute_entropy_measures()
        
        required_entropies = [
            'zeta_frequency', 'resonance_strength', 'zeta_alignment',
            'mode_coupling', 'resonance_classification'
        ]
        
        for entropy_name in required_entropies:
            self.assertIn(entropy_name, entropies)
            self.assertGreaterEqual(entropies[entropy_name], 0.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_resonance_network()
        
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
        # 检查节点属性
        for node in G.nodes():
            self.assertIn('zeta_frequency', G.nodes[node])
            self.assertIn('resonance_classification', G.nodes[node])
            
    def test_categorical_analysis(self):
        """测试范畴论分析"""
        cat_analysis = self.system.analyze_categorical_structure()
        
        self.assertIn('categories', cat_analysis)
        self.assertIn('morphisms', cat_analysis)
        self.assertIn('morphism_density', cat_analysis)
        self.assertGreater(cat_analysis['category_count'], 0)

def run_verification():
    """运行完整的验证过程"""
    print("="*80)
    print("Chapter 092: CollapseResonance Verification")
    print("从ψ=ψ(ψ)推导ζ-Aligned Trace Structures in Spectral Collapse Systems")
    print("="*80)
    
    # 创建系统
    system = CollapseResonanceSystem(max_trace_value=95, num_resonance_modes=10)
    
    # 1. 基础统计
    print("\n1. Resonance Foundation Analysis:")
    print("-" * 50)
    resonance_props = system.analyze_resonance_properties()
    
    print(f"Total traces analyzed: {resonance_props['total_traces']}")
    print(f"Frequency statistics: mean={resonance_props['frequency_stats']['mean']:.3f}, "
          f"std={resonance_props['frequency_stats']['std']:.3f}")
    print(f"Resonance strength: mean={resonance_props['strength_stats']['mean']:.3f}, "
          f"max={resonance_props['strength_stats']['max']:.3f}")
    print(f"Mean ζ-alignment: {resonance_props['alignment_stats']['mean']:.3f}")
    print(f"Mean damping factor: {resonance_props['damping_stats']['mean']:.3f}")
    print(f"Mean mode coupling: {resonance_props['coupling_stats']['mean']:.3f}")
    print(f"Mean harmonic content: {resonance_props['harmonic_stats']['mean']:.3f}")
    
    print("\nResonance Classification Distribution:")
    for cls, count in resonance_props['classification_distribution'].items():
        percentage = (count / resonance_props['total_traces']) * 100
        print(f"- {cls}: {count} traces ({percentage:.1f}%)")
        
    print("\nMode Distribution:")
    for mode, count in resonance_props['mode_distribution'].items():
        percentage = (count / resonance_props['total_traces']) * 100
        print(f"- Mode {mode}: {count} traces ({percentage:.1f}%)")
        
    # 2. ζ-alignment analysis
    print("\n2. ζ-Alignment Analysis:")
    print("-" * 50)
    zeta_alignment = resonance_props['zeta_alignment']
    
    print(f"Mean ζ-alignment: {zeta_alignment['mean_alignment']:.3f}")
    print(f"Alignment standard deviation: {zeta_alignment['alignment_std']:.3f}")
    print(f"Mean resonance strength: {zeta_alignment['mean_strength']:.3f}")
    print(f"Strength standard deviation: {zeta_alignment['strength_std']:.3f}")
    print(f"Alignment-strength correlation: {zeta_alignment['alignment_strength_correlation']:.3f}")
    print(f"Strong aligned traces: {zeta_alignment['strong_aligned_count']} "
          f"({zeta_alignment['strong_aligned_fraction']:.1%})")
          
    # 3. Resonance mode analysis
    print("\n3. Resonance Mode Analysis:")
    print("-" * 50)
    resonance_modes = resonance_props['resonance_modes']
    
    print(f"Active resonance modes: {len(resonance_modes)}")
    
    for mode_idx, analysis in resonance_modes.items():
        print(f"\nMode {mode_idx}:")
        print(f"  Trace count: {analysis['trace_count']}")
        print(f"  Mean strength: {analysis['mean_strength']:.3f}")
        print(f"  Mean alignment: {analysis['mean_alignment']:.3f}")
        print(f"  Mean damping: {analysis['mean_damping']:.3f}")
        print(f"  Frequency range: [{analysis['frequency_range'][0]:.3f}, {analysis['frequency_range'][1]:.3f}]")
        print(f"  Dominant type: {analysis['dominant_classification']}")
    
    # 4. 信息论分析
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    entropies = system.compute_entropy_measures()
    
    for prop, entropy in entropies.items():
        print(f"{prop.replace('_', ' ').title()} entropy: {entropy:.3f} bits")
        
    # 5. 网络分析
    print("\n5. Graph Theory Analysis:")
    print("-" * 50)
    G = system.build_resonance_network()
    
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        avg_weight = sum(edge_weights) / len(edge_weights)
        
        print(f"Average degree: {avg_degree:.3f}")
        print(f"Average edge weight: {avg_weight:.3f}")
        print(f"Connected components: {nx.number_connected_components(G)}")
        
        # Network density
        possible_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        density = G.number_of_edges() / possible_edges if possible_edges > 0 else 0
        print(f"Network density: {density:.3f}")
        
    # 6. 范畴论分析
    print("\n6. Category Theory Analysis:")
    print("-" * 50)
    cat_analysis = system.analyze_categorical_structure()
    
    print(f"Resonance categories: {cat_analysis['category_count']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for category, count in cat_analysis['categories'].items():
        print(f"- {category}: {count} objects")
        
    # 7. 可视化生成
    print("\n7. Visualization Generation:")
    print("-" * 50)
    
    try:
        # 创建resonance可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.1 ζ-Frequency vs Resonance Strength
        frequencies = [data['zeta_frequency'] for data in system.trace_universe.values()]
        strengths = [data['resonance_strength'] for data in system.trace_universe.values()]
        alignments = [data['zeta_alignment'] for data in system.trace_universe.values()]
        
        scatter = ax1.scatter(frequencies, strengths, c=alignments, cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('ζ-Frequency')
        ax1.set_ylabel('Resonance Strength')
        ax1.set_title('ζ-Frequency vs Resonance Strength')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='ζ-Alignment')
        
        # 7.2 Resonance Network
        if G.number_of_nodes() > 0:
            pos = {}
            for node in G.nodes():
                freq = G.nodes[node]['zeta_frequency']
                strength = G.nodes[node]['resonance_strength']
                pos[node] = (freq, strength + random.uniform(-0.02, 0.02))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=30, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=0.5)
            
        ax2.set_xlabel('ζ-Frequency')
        ax2.set_ylabel('Resonance Strength')
        ax2.set_title('Resonance Network Structure')
        ax2.grid(True, alpha=0.3)
        
        # 7.3 Mode Coupling vs Damping
        couplings = [data['mode_coupling'] for data in system.trace_universe.values()]
        dampings = [data['damping_factor'] for data in system.trace_universe.values()]
        classifications = [data['resonance_classification'] for data in system.trace_universe.values()]
        
        # Color by classification
        class_to_color = {cls: i for i, cls in enumerate(set(classifications))}
        colors = [class_to_color[cls] for cls in classifications]
        
        scatter = ax3.scatter(couplings, dampings, c=colors, cmap='tab10', alpha=0.7)
        ax3.set_xlabel('Mode Coupling')
        ax3.set_ylabel('Damping Factor')
        ax3.set_title('Mode Coupling vs Damping Factor')
        ax3.grid(True, alpha=0.3)
        
        # 7.4 Harmonic Content Distribution
        harmonics = [data['harmonic_content'] for data in system.trace_universe.values()]
        
        ax4.hist(harmonics, bins=15, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Harmonic Content')
        ax4.set_ylabel('Count')
        ax4.set_title('Harmonic Content Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-092-collapse-resonance-modes.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Resonance modes visualization saved")
        
        # 创建ζ-alignment分析可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.5 ζ-Alignment Distribution
        ax1.hist(alignments, bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('ζ-Alignment')
        ax1.set_ylabel('Count')
        ax1.set_title('ζ-Alignment Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 7.6 Phase Alignment vs ζ-Alignment
        phase_alignments = [data['phase_alignment'] for data in system.trace_universe.values()]
        
        scatter = ax2.scatter(phase_alignments, alignments, c=colors, cmap='tab10', alpha=0.7)
        ax2.set_xlabel('Phase Alignment')
        ax2.set_ylabel('ζ-Alignment')
        ax2.set_title('Phase Alignment vs ζ-Alignment')
        ax2.grid(True, alpha=0.3)
        
        # 7.7 Resonance Classification Distribution
        classification_counts = list(resonance_props['classification_distribution'].values())
        classification_labels = list(resonance_props['classification_distribution'].keys())
        
        ax3.pie(classification_counts, labels=classification_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Resonance Classification Distribution')
        
        # 7.8 Mode Index Distribution
        mode_indices = [data['mode_index'] for data in system.trace_universe.values()]
        mode_counts = {}
        for mode in mode_indices:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
        if mode_counts:
            modes = list(mode_counts.keys())
            counts = list(mode_counts.values())
            
            bars = ax4.bar(modes, counts, alpha=0.7)
            ax4.set_xlabel('Mode Index')
            ax4.set_ylabel('Count')
            ax4.set_title('Mode Index Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Color bars by mode strength
            if resonance_modes:
                for i, bar in enumerate(bars):
                    mode_idx = modes[i]
                    if mode_idx in resonance_modes:
                        strength = resonance_modes[mode_idx]['mean_strength']
                        bar.set_color(plt.cm.viridis(strength))
                        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-092-collapse-resonance-alignment.png',
                   dpi=300, bbox_inches='tight')
        print("✓ ζ-alignment analysis visualization saved")
        
        # 创建网络和范畴论可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.9 Network visualization by resonance strength
        if G.number_of_nodes() > 0:
            # Color nodes by resonance strength
            node_colors = [G.nodes[node]['resonance_strength'] for node in G.nodes()]
            node_sizes = [G.nodes[node]['zeta_alignment'] * 100 + 20 for node in G.nodes()]
            
            pos_spring = nx.spring_layout(G, k=1, iterations=50)
            
            scatter = ax1.scatter([pos_spring[node][0] for node in G.nodes()],
                                [pos_spring[node][1] for node in G.nodes()],
                                c=node_colors, s=node_sizes, cmap='plasma', alpha=0.8)
            
            # Draw edges
            for edge in G.edges():
                x_coords = [pos_spring[edge[0]][0], pos_spring[edge[1]][0]]
                y_coords = [pos_spring[edge[0]][1], pos_spring[edge[1]][1]]
                ax1.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
                
            plt.colorbar(scatter, ax=ax1, label='Resonance Strength', shrink=0.8)
        
        ax1.set_title('Resonance Network by Strength')
        ax1.set_xlabel('Network X-coordinate')
        ax1.set_ylabel('Network Y-coordinate')
        
        # 7.10 Morphism density by category
        if cat_analysis['categories']:
            categories = list(cat_analysis['categories'].keys())
            category_sizes = list(cat_analysis['categories'].values())
            
            ax2.bar(categories, category_sizes, alpha=0.7)
            ax2.set_ylabel('Object Count')
            ax2.set_title('Category Object Distribution')
            ax2.tick_params(axis='x', rotation=45)
            
        # 7.11 Entropy landscape
        entropy_names = list(entropies.keys())
        entropy_values = list(entropies.values())
        
        bars = ax3.bar(range(len(entropy_names)), entropy_values, alpha=0.7)
        ax3.set_xticks(range(len(entropy_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in entropy_names], 
                          rotation=45, ha='right')
        ax3.set_ylabel('Entropy (bits)')
        ax3.set_title('Information Entropy Distribution')
        
        # Color bars by entropy level
        max_entropy = max(entropy_values) if entropy_values else 1
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(entropy_values[i] / max_entropy))
            
        # 7.12 Resonance mode strength comparison
        if resonance_modes:
            mode_indices = list(resonance_modes.keys())
            mode_strengths = [resonance_modes[idx]['mean_strength'] for idx in mode_indices]
            mode_alignments = [resonance_modes[idx]['mean_alignment'] for idx in mode_indices]
            
            ax4.scatter(mode_indices, mode_strengths, s=100, alpha=0.7, label='Strength')
            ax4.scatter(mode_indices, mode_alignments, s=100, alpha=0.7, label='Alignment')
            ax4.set_xlabel('Mode Index')
            ax4.set_ylabel('Value')
            ax4.set_title('Mode Strength vs Alignment')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-092-collapse-resonance-network.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Network and categorical visualization saved")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with analysis...")
    
    # 8. 运行单元测试
    print("\n8. Running Unit Tests:")
    print("-" * 50)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*80)
    print("CollapseResonance Verification Complete")
    print("Key Findings:")
    print(f"- {resonance_props['total_traces']} φ-valid traces with ζ-aligned resonance analysis")
    print(f"- {len(resonance_modes)} active resonance modes identified")
    print(f"- Mean ζ-alignment: {zeta_alignment['mean_alignment']:.3f}")
    print(f"- Mean resonance strength: {zeta_alignment['mean_strength']:.3f}")
    print(f"- Strong aligned fraction: {zeta_alignment['strong_aligned_fraction']:.1%}")
    print(f"- {cat_analysis['category_count']} resonance categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print("="*80)

if __name__ == "__main__":
    run_verification()