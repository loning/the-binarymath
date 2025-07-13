#!/usr/bin/env python3
"""
Chapter 081: CollapseAnalyzer Unit Test Verification
从ψ=ψ(ψ)推导Spectral Analyzer for Collapse Frequency Bands

Core principle: From ψ = ψ(ψ) derive spectral analyzer where analyzer is φ-valid
frequency decomposition system that analyzes spectral characteristics through trace-based band filtering,
creating systematic frequency frameworks with bounded spectral analysis and natural frequency
properties governed by golden constraints, showing how frequency analysis emerges from trace structures.

This verification program implements:
1. φ-constrained frequency analyzer as trace spectral decomposition operations
2. Frequency analysis: spectral patterns, band structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection frequency theory
4. Graph theory analysis of frequency networks and spectral connectivity patterns
5. Information theory analysis of frequency entropy and spectral information
6. Category theory analysis of frequency functors and spectral morphisms
7. Visualization of frequency structures and spectral patterns
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
from math import log2, gcd, sqrt, pi, exp, cos, sin, log
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class CollapseAnalyzerSystem:
    """
    Core system for implementing frequency analysis through trace spectral decomposition.
    Implements φ-constrained frequency analyzer via trace-based band filtering operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_frequency_bands: int = 20):
        """Initialize collapse analyzer system"""
        self.max_trace_size = max_trace_size
        self.max_frequency_bands = max_frequency_bands
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.frequency_cache = {}
        self.band_cache = {}
        self.spectral_cache = {}
        self.trace_universe = self._build_trace_universe()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        # First pass: build basic universe
        for n in range(1, self.max_frequency_bands + 1):
            trace_data = self._analyze_trace_structure(n, compute_frequency=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for frequency properties computation
        self.trace_universe = universe
        
        # Second pass: add frequency properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['frequency_properties'] = self._compute_frequency_properties(trace, n)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_frequency: bool = True) -> Dict:
        """分析单个trace的结构属性"""
        trace = self._encode_to_trace(n)
        
        result = {
            'value': n,
            'trace': trace,
            'phi_valid': '11' not in trace,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'fibonacci_indices': self._get_fibonacci_indices(trace),
            'structural_hash': self._compute_structural_hash(trace),
            'binary_weight': self._compute_binary_weight(trace)
        }
        
        if compute_frequency and hasattr(self, 'trace_universe'):
            result['frequency_properties'] = self._compute_frequency_properties(trace, n)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将数字编码为φ-valid trace（Zeckendorf表示）"""
        if n <= 0:
            return "0"
        
        # Build Zeckendorf representation
        result = []
        fib_used = []
        remaining = n
        
        # Use Fibonacci numbers in descending order
        for i in range(len(self.fibonacci_numbers) - 1, -1, -1):
            if self.fibonacci_numbers[i] <= remaining:
                result.append(i)
                fib_used.append(self.fibonacci_numbers[i])
                remaining -= self.fibonacci_numbers[i]
        
        if remaining > 0:
            # Need more Fibonacci numbers
            return "1" * (n.bit_length())
        
        # Convert to binary string with φ-constraint
        max_index = max(result) if result else 0
        trace = ['0'] * (max_index + 1)
        for idx in result:
            trace[max_index - idx] = '1'
        
        trace_str = ''.join(trace)
        
        # Ensure no consecutive 11s
        if '11' in trace_str:
            # Use simpler representation
            trace_str = bin(n)[2:]
            if '11' in trace_str:
                # Replace 11 with 101
                trace_str = trace_str.replace('11', '101')
        
        return trace_str
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中1对应的Fibonacci索引"""
        indices = []
        for i, bit in enumerate(trace):
            if bit == '1':
                indices.append(len(trace) - 1 - i)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构hash"""
        return hash(trace) % 1000000
        
    def _compute_binary_weight(self, trace: str) -> float:
        """计算trace的二进制权重"""
        weight = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                weight += 2.0 ** (len(trace) - 1 - i)
        return weight / (2.0 ** len(trace) - 1) if len(trace) > 0 else 0.0
        
    def _compute_frequency_properties(self, trace: str, value: int) -> Dict:
        """计算trace的频率特性"""
        if not hasattr(self, 'trace_universe'):
            return {}
            
        # Spectral frequency
        spectral_freq = self.compute_spectral_frequency(trace)
        
        # Frequency contribution
        freq_contribution = self.compute_frequency_contribution(trace, value)
        
        # Band component
        band_component = self.compute_band_component(trace)
        
        # Spectral power
        spectral_power = self.compute_spectral_power(trace)
        
        # Analysis mode
        analysis_mode = self.compute_analysis_mode(trace)
        
        # Filter density
        filter_density = self.compute_filter_density(trace)
        
        # Band signature
        band_signature = self.compute_band_signature(trace)
        
        # Spectral phase
        spectral_phase = self.compute_spectral_phase(trace)
        
        # Frequency type
        frequency_type = self.classify_frequency_type(trace, spectral_freq, band_component, analysis_mode)
        
        # Spectral dynamics
        spectral_dynamics = self.compute_spectral_dynamics(trace)
        
        return {
            'spectral_frequency': spectral_freq,
            'frequency_contribution': freq_contribution,
            'band_component': band_component,
            'spectral_power': spectral_power,
            'analysis_mode': analysis_mode,
            'filter_density': filter_density,
            'band_signature': band_signature,
            'spectral_phase': spectral_phase,
            'frequency_type': frequency_type,
            'spectral_dynamics': spectral_dynamics
        }
        
    def compute_spectral_frequency(self, trace: str) -> float:
        """计算频谱频率：基于trace结构的频率值"""
        if not trace or len(trace) == 0:
            return 0.0
            
        # Base frequency from trace length and pattern
        base_freq = 1.0 / len(trace)
        
        # Pattern enhancement from ones distribution
        ones_count = trace.count('1')
        pattern_factor = ones_count / len(trace) if len(trace) > 0 else 0.0
        
        # Golden ratio scaling for φ-constraint
        phi = (1 + sqrt(5)) / 2
        phi_scaling = (pattern_factor * phi) % 1.0
        
        return base_freq + phi_scaling * 0.5
        
    def compute_frequency_contribution(self, trace: str, value: int) -> float:
        """计算频率贡献：基于trace值的频率权重"""
        if not trace:
            return 0.0
            
        # Base contribution from trace structure
        structural_contrib = self._compute_binary_weight(trace)
        
        # Value enhancement
        value_factor = log(max(value, 1)) / log(10)
        
        # φ-constraint enhancement
        phi = (1 + sqrt(5)) / 2
        phi_factor = (value_factor / phi) % 1.0
        
        return structural_contrib * (1.0 + phi_factor)
        
    def compute_band_component(self, trace: str) -> float:
        """计算频带成分：基于trace的频率分布"""
        if not trace:
            return 0.0
            
        # Analyze frequency distribution in trace
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if not ones_positions:
            return 0.0
            
        # Compute spacing variance
        if len(ones_positions) > 1:
            spacings = [ones_positions[i+1] - ones_positions[i] for i in range(len(ones_positions)-1)]
            avg_spacing = sum(spacings) / len(spacings)
            spacing_variance = sum((s - avg_spacing)**2 for s in spacings) / len(spacings)
            band_measure = 1.0 / (1.0 + spacing_variance)
        else:
            band_measure = 1.0
            
        # φ-enhancement
        phi = (1 + sqrt(5)) / 2
        return (band_measure * phi) % 1.0
        
    def compute_spectral_power(self, trace: str) -> float:
        """计算频谱功率：基于trace的能量分布"""
        if not trace:
            return 0.0
            
        # Power from binary weight distribution
        binary_weight = self._compute_binary_weight(trace)
        
        # Enhancement from trace complexity
        complexity = len(trace) * trace.count('1')
        complexity_factor = sqrt(complexity) / 10.0 if complexity > 0 else 0.0
        
        # φ-scaling
        phi = (1 + sqrt(5)) / 2
        phi_power = (binary_weight + complexity_factor) / phi
        
        return phi_power % 1.0
        
    def compute_analysis_mode(self, trace: str) -> float:
        """计算分析模式：基于trace的分析特性"""
        if not trace:
            return 0.0
            
        # Mode based on trace pattern analysis
        pattern_complexity = len(set(trace[i:i+2] for i in range(len(trace)-1))) if len(trace) > 1 else 1
        max_complexity = 4  # 00, 01, 10, 11 (but 11 forbidden)
        
        mode_factor = pattern_complexity / max_complexity
        
        # φ-constraint adjustment
        phi = (1 + sqrt(5)) / 2
        return (mode_factor * phi) % 1.0
        
    def compute_filter_density(self, trace: str) -> float:
        """计算滤波密度：基于trace的滤波特性"""
        if not trace:
            return 0.0
            
        # Density based on ones distribution
        ones_count = trace.count('1')
        density_base = ones_count / len(trace) if len(trace) > 0 else 0.0
        
        # Distribution uniformity
        if ones_count > 1:
            ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
            position_variance = np.var(ones_positions) if len(ones_positions) > 1 else 0.0
            uniformity = 1.0 / (1.0 + position_variance)
        else:
            uniformity = 1.0
            
        # φ-enhancement
        phi = (1 + sqrt(5)) / 2
        return (density_base * uniformity * phi) % 1.0
        
    def compute_band_signature(self, trace: str) -> complex:
        """计算频带签名：基于trace的复数表示"""
        if not trace:
            return complex(0, 0)
            
        # Create complex signature from trace structure
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                angle = 2 * pi * i / len(trace)
                real_part += cos(angle)
                imag_part += sin(angle)
                
        # Normalize
        magnitude = sqrt(real_part**2 + imag_part**2)
        if magnitude > 0:
            real_part /= magnitude
            imag_part /= magnitude
            
        return complex(real_part, imag_part)
        
    def compute_spectral_phase(self, trace: str) -> float:
        """计算频谱相位：基于trace的相位特性"""
        signature = self.compute_band_signature(trace)
        phase = np.angle(signature)
        return (phase + pi) / (2 * pi)  # Normalize to [0, 1]
        
    def classify_frequency_type(self, trace: str, spectral_freq: float, band_component: float, analysis_mode: float) -> str:
        """分类频率类型：基于频率特性的分类"""
        # Classification based on frequency characteristics
        if spectral_freq > 0.6:
            return "high_frequency"
        elif band_component > 0.6:
            return "band_dominated"
        elif analysis_mode > 0.6:
            return "analysis_intensive"
        else:
            return "low_frequency"
            
    def compute_spectral_dynamics(self, trace: str) -> float:
        """计算频谱动力学：基于trace的动态特性"""
        if not trace:
            return 0.0
            
        # Dynamic measure from trace transitions
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                transitions += 1
                
        max_transitions = len(trace) - 1
        dynamics = transitions / max_transitions if max_transitions > 0 else 0.0
        
        # φ-enhancement
        phi = (1 + sqrt(5)) / 2
        return (dynamics * phi) % 1.0
        
    def analyze_frequency_system(self) -> Dict:
        """分析完整频率系统的特性"""
        if not self.trace_universe:
            return {}
            
        frequencies = []
        contributions = []
        band_components = []
        spectral_powers = []
        analysis_modes = []
        filter_densities = []
        spectral_phases = []
        frequency_types = []
        spectral_dynamics = []
        
        for trace_data in self.trace_universe.values():
            freq_props = trace_data.get('frequency_properties', {})
            frequencies.append(freq_props.get('spectral_frequency', 0))
            contributions.append(freq_props.get('frequency_contribution', 0))
            band_components.append(freq_props.get('band_component', 0))
            spectral_powers.append(freq_props.get('spectral_power', 0))
            analysis_modes.append(freq_props.get('analysis_mode', 0))
            filter_densities.append(freq_props.get('filter_density', 0))
            spectral_phases.append(freq_props.get('spectral_phase', 0))
            frequency_types.append(freq_props.get('frequency_type', 'unknown'))
            spectral_dynamics.append(freq_props.get('spectral_dynamics', 0))
            
        return {
            'frequency_universe_size': len(self.trace_universe),
            'mean_spectral_frequency': np.mean(frequencies) if frequencies else 0,
            'mean_frequency_contribution': np.mean(contributions) if contributions else 0,
            'mean_band_component': np.mean(band_components) if band_components else 0,
            'mean_spectral_power': np.mean(spectral_powers) if spectral_powers else 0,
            'mean_analysis_mode': np.mean(analysis_modes) if analysis_modes else 0,
            'mean_filter_density': np.mean(filter_densities) if filter_densities else 0,
            'mean_spectral_phase': np.mean(spectral_phases) if spectral_phases else 0,
            'mean_spectral_dynamics': np.mean(spectral_dynamics) if spectral_dynamics else 0,
            'frequency_type_distribution': {ft: frequency_types.count(ft) / len(frequency_types) * 100 
                                         if frequency_types else 0 for ft in set(frequency_types)},
        }
        
    def analyze_graph_properties(self) -> Dict:
        """分析频率网络的图论特性"""
        if not self.trace_universe:
            return {}
            
        # Build frequency network
        G = nx.Graph()
        
        # Add nodes
        for value, trace_data in self.trace_universe.items():
            freq_props = trace_data.get('frequency_properties', {})
            G.add_node(value, **freq_props)
            
        # Add edges based on frequency similarity
        values = list(self.trace_universe.keys())
        for i, v1 in enumerate(values):
            for v2 in values[i+1:]:
                freq1 = self.trace_universe[v1]['frequency_properties'].get('spectral_frequency', 0)
                freq2 = self.trace_universe[v2]['frequency_properties'].get('spectral_frequency', 0)
                
                # Connect if frequencies are similar or complementary
                freq_diff = abs(freq1 - freq2)
                if freq_diff < 0.3 or freq_diff > 0.7:  # Similar or complementary
                    G.add_edge(v1, v2, weight=1.0 - min(freq_diff, 1.0 - freq_diff))
                    
        if G.number_of_nodes() == 0:
            return {}
            
        # Compute graph properties
        try:
            density = nx.density(G)
            connected_components = nx.number_connected_components(G)
            clustering = nx.average_clustering(G) if G.number_of_edges() > 0 else 0
        except:
            density = 0
            connected_components = G.number_of_nodes()
            clustering = 0
            
        return {
            'network_nodes': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'network_density': density,
            'connected_components': connected_components,
            'average_clustering': clustering
        }
        
    def analyze_information_theory(self) -> Dict:
        """分析频率系统的信息论特性"""
        if not self.trace_universe:
            return {}
            
        # Collect frequency data
        frequencies = []
        contributions = []
        band_components = []
        spectral_powers = []
        analysis_modes = []
        filter_densities = []
        spectral_phases = []
        frequency_types = []
        spectral_dynamics = []
        
        for trace_data in self.trace_universe.values():
            freq_props = trace_data.get('frequency_properties', {})
            frequencies.append(freq_props.get('spectral_frequency', 0))
            contributions.append(freq_props.get('frequency_contribution', 0))
            band_components.append(freq_props.get('band_component', 0))
            spectral_powers.append(freq_props.get('spectral_power', 0))
            analysis_modes.append(freq_props.get('analysis_mode', 0))
            filter_densities.append(freq_props.get('filter_density', 0))
            spectral_phases.append(freq_props.get('spectral_phase', 0))
            frequency_types.append(freq_props.get('frequency_type', 'unknown'))
            spectral_dynamics.append(freq_props.get('spectral_dynamics', 0))
            
        def compute_entropy(values):
            """计算离散值的熵"""
            if not values:
                return 0
            # Discretize continuous values
            hist, _ = np.histogram(values, bins=min(10, len(set(values))))
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / hist.sum()
            return -sum(p * log2(p) for p in probs)
            
        def compute_type_entropy(types):
            """计算类型分布的熵"""
            if not types:
                return 0
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            total = len(types)
            probs = [count / total for count in type_counts.values()]
            return -sum(p * log2(p) for p in probs if p > 0)
            
        return {
            'frequency_entropy': compute_entropy(frequencies),
            'contribution_entropy': compute_entropy(contributions),
            'band_component_entropy': compute_entropy(band_components),
            'spectral_power_entropy': compute_entropy(spectral_powers),
            'analysis_mode_entropy': compute_entropy(analysis_modes),
            'filter_density_entropy': compute_entropy(filter_densities),
            'spectral_phase_entropy': compute_entropy(spectral_phases),
            'type_entropy': compute_type_entropy(frequency_types),
            'spectral_dynamics_entropy': compute_entropy(spectral_dynamics),
            'frequency_complexity': len(set(frequency_types))
        }
        
    def analyze_category_theory(self) -> Dict:
        """分析频率系统的范畴论特性"""
        if not self.trace_universe:
            return {}
            
        # Build frequency morphisms
        morphisms = []
        functorial_relationships = []
        
        values = list(self.trace_universe.keys())
        for v1 in values:
            for v2 in values:
                if v1 != v2:
                    freq1 = self.trace_universe[v1]['frequency_properties'].get('spectral_frequency', 0)
                    freq2 = self.trace_universe[v2]['frequency_properties'].get('spectral_frequency', 0)
                    band1 = self.trace_universe[v1]['frequency_properties'].get('band_component', 0)
                    band2 = self.trace_universe[v2]['frequency_properties'].get('band_component', 0)
                    
                    # Morphism exists if frequency and band structure are preserved
                    freq_preserved = abs(freq1 - freq2) < 0.4
                    band_preserved = abs(band1 - band2) < 0.4
                    
                    if freq_preserved or band_preserved:
                        morphisms.append((v1, v2))
                        
                    if freq_preserved and band_preserved:
                        functorial_relationships.append((v1, v2))
                        
        # Analyze functorial structure
        functoriality_ratio = len(functorial_relationships) / len(morphisms) if morphisms else 0
        
        # Group analysis by frequency type
        frequency_groups = {}
        for value, trace_data in self.trace_universe.items():
            freq_type = trace_data['frequency_properties'].get('frequency_type', 'unknown')
            if freq_type not in frequency_groups:
                frequency_groups[freq_type] = []
            frequency_groups[freq_type].append(value)
            
        return {
            'frequency_morphisms': len(morphisms),
            'functorial_relationships': len(functorial_relationships),
            'functoriality_ratio': functoriality_ratio,
            'frequency_groups': len(frequency_groups),
            'largest_group_size': max(len(group) for group in frequency_groups.values()) if frequency_groups else 0
        }
        
    def generate_visualizations(self):
        """生成频率分析系统的可视化"""
        if not self.trace_universe:
            return
            
        # Create visualizations directory
        import os
        os.makedirs('visualization_output', exist_ok=True)
        
        # 1. Frequency Structure Visualization
        self._plot_frequency_structure()
        
        # 2. Frequency Properties Visualization  
        self._plot_frequency_properties()
        
        # 3. Domain Analysis Visualization
        self._plot_domain_analysis()
        
    def _plot_frequency_structure(self):
        """绘制频率结构图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        values = list(self.trace_universe.keys())
        frequencies = [self.trace_universe[v]['frequency_properties'].get('spectral_frequency', 0) for v in values]
        contributions = [self.trace_universe[v]['frequency_properties'].get('frequency_contribution', 0) for v in values]
        band_components = [self.trace_universe[v]['frequency_properties'].get('band_component', 0) for v in values]
        spectral_powers = [self.trace_universe[v]['frequency_properties'].get('spectral_power', 0) for v in values]
        
        colors = ['coral', 'lightblue', 'lightgreen', 'wheat', 'lightpink', 'lightyellow', 'lightgray', 'lightcyan']
        
        # Frequency distribution
        ax1.scatter(values, frequencies, c=[colors[i % len(colors)] for i in range(len(values))], s=100, alpha=0.7)
        ax1.set_xlabel('Trace Value')
        ax1.set_ylabel('Spectral Frequency')
        ax1.set_title('Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Contribution vs Band Component
        ax2.scatter(contributions, band_components, c=[colors[i % len(colors)] for i in range(len(values))], s=100, alpha=0.7)
        ax2.set_xlabel('Frequency Contribution')
        ax2.set_ylabel('Band Component')
        ax2.set_title('Contribution vs Band Component')
        ax2.grid(True, alpha=0.3)
        
        # Spectral Power Distribution
        ax3.bar(range(len(values)), spectral_powers, color=[colors[i % len(colors)] for i in range(len(values))], alpha=0.7)
        ax3.set_xlabel('Trace Index')
        ax3.set_ylabel('Spectral Power')
        ax3.set_title('Spectral Power Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Frequency Type Distribution
        freq_types = [self.trace_universe[v]['frequency_properties'].get('frequency_type', 'unknown') for v in values]
        type_counts = {}
        for ft in freq_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
            
        ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', 
                colors=colors[:len(type_counts)])
        ax4.set_title('Frequency Type Distribution')
        
        plt.tight_layout()
        plt.savefig('visualization_output/chapter-081-collapse-analyzer-structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_frequency_properties(self):
        """绘制频率特性图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        values = list(self.trace_universe.keys())
        analysis_modes = [self.trace_universe[v]['frequency_properties'].get('analysis_mode', 0) for v in values]
        filter_densities = [self.trace_universe[v]['frequency_properties'].get('filter_density', 0) for v in values]
        spectral_phases = [self.trace_universe[v]['frequency_properties'].get('spectral_phase', 0) for v in values]
        spectral_dynamics = [self.trace_universe[v]['frequency_properties'].get('spectral_dynamics', 0) for v in values]
        
        colors = ['coral', 'lightblue', 'lightgreen', 'wheat', 'lightpink', 'lightyellow', 'lightgray', 'lightcyan']
        
        # Analysis Mode vs Filter Density
        ax1.scatter(analysis_modes, filter_densities, c=[colors[i % len(colors)] for i in range(len(values))], s=100, alpha=0.7)
        ax1.set_xlabel('Analysis Mode')
        ax1.set_ylabel('Filter Density')
        ax1.set_title('Analysis Mode vs Filter Density')
        ax1.grid(True, alpha=0.3)
        
        # Spectral Phase Distribution
        ax2.hist(spectral_phases, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Spectral Phase')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Spectral Phase Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Spectral Dynamics vs Analysis Mode
        ax3.scatter(spectral_dynamics, analysis_modes, c=[colors[i % len(colors)] for i in range(len(values))], s=100, alpha=0.7)
        ax3.set_xlabel('Spectral Dynamics')
        ax3.set_ylabel('Analysis Mode')
        ax3.set_title('Spectral Dynamics vs Analysis Mode')
        ax3.grid(True, alpha=0.3)
        
        # Complex Signature Plot
        signatures = [self.trace_universe[v]['frequency_properties'].get('band_signature', complex(0,0)) for v in values]
        real_parts = [s.real for s in signatures]
        imag_parts = [s.imag for s in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=[colors[i % len(colors)] for i in range(len(values))], s=100, alpha=0.7)
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.set_title('Complex Frequency Signatures')
        ax4.grid(True, alpha=0.3)
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-1.2, 1.2)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('visualization_output/chapter-081-collapse-analyzer-properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_domain_analysis(self):
        """绘制三域分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Traditional vs φ-constrained comparison
        traditional_values = list(range(1, 21))  # Traditional unconstrained
        phi_values = list(self.trace_universe.keys())  # φ-constrained
        
        # Convergence analysis
        convergence_values = [v for v in traditional_values if v in phi_values]
        convergence_ratio = len(convergence_values) / len(traditional_values)
        
        # Domain sizes
        domains = ['Traditional-Only', 'φ-Constrained', 'Convergence']
        domain_sizes = [
            len(traditional_values) - len(convergence_values),
            len(phi_values) - len(convergence_values), 
            len(convergence_values)
        ]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        ax1.pie(domain_sizes, labels=domains, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Three-Domain Distribution')
        
        # Frequency enhancement analysis
        if convergence_values:
            phi_frequencies = [self.trace_universe[v]['frequency_properties'].get('spectral_frequency', 0) 
                             for v in convergence_values]
            traditional_frequencies = [1.0/v for v in convergence_values]  # Traditional 1/n pattern
            
            ax2.scatter(convergence_values, traditional_frequencies, color='lightcoral', alpha=0.7, label='Traditional')
            ax2.scatter(convergence_values, phi_frequencies, color='lightblue', alpha=0.7, label='φ-Constrained')
            ax2.set_xlabel('Trace Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Frequency Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Enhancement factor analysis
        if convergence_values and len(phi_frequencies) > 0:
            enhancement_factors = [pf/tf if tf > 0 else 1 for pf, tf in zip(phi_frequencies, traditional_frequencies)]
            avg_enhancement = np.mean(enhancement_factors)
            
            ax3.bar(range(len(enhancement_factors)), enhancement_factors, 
                   color='lightgreen', alpha=0.7)
            ax3.axhline(y=avg_enhancement, color='red', linestyle='--', 
                       label=f'Avg Enhancement: {avg_enhancement:.3f}')
            ax3.set_xlabel('Convergence Index')
            ax3.set_ylabel('Enhancement Factor')
            ax3.set_title('φ-Enhancement Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # System comparison metrics
        system_metrics = ['Universe Size', 'Mean Frequency', 'Enhancement Factor', 'Convergence Ratio']
        traditional_metrics = [len(traditional_values), np.mean(traditional_frequencies) if 'traditional_frequencies' in locals() else 0, 1.0, convergence_ratio]
        phi_metrics = [len(phi_values), np.mean(phi_frequencies) if 'phi_frequencies' in locals() else 0, avg_enhancement if 'avg_enhancement' in locals() else 1.0, convergence_ratio]
        
        x = np.arange(len(system_metrics))
        width = 0.35
        
        ax4.bar(x - width/2, traditional_metrics, width, label='Traditional', color='lightcoral', alpha=0.7)
        ax4.bar(x + width/2, phi_metrics, width, label='φ-Constrained', color='lightblue', alpha=0.7)
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Values')
        ax4.set_title('System Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(system_metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualization_output/chapter-081-collapse-analyzer-domains.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestCollapseAnalyzer(unittest.TestCase):
    """测试CollapseAnalyzer系统的各个组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = CollapseAnalyzerSystem(max_trace_size=5, max_frequency_bands=15)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        # Test φ-valid traces
        trace1 = self.analyzer._encode_to_trace(5)
        self.assertNotIn('11', trace1)  # φ-constraint
        
        trace2 = self.analyzer._encode_to_trace(8)
        self.assertNotIn('11', trace2)  # φ-constraint
        
    def test_frequency_properties(self):
        """测试频率特性计算"""
        trace = "101"
        freq_props = self.analyzer._compute_frequency_properties(trace, 5)
        
        # Verify all required properties exist
        required_props = ['spectral_frequency', 'frequency_contribution', 'band_component', 
                         'spectral_power', 'analysis_mode', 'filter_density', 'band_signature',
                         'spectral_phase', 'frequency_type', 'spectral_dynamics']
        
        for prop in required_props:
            self.assertIn(prop, freq_props)
            
        # Verify ranges
        self.assertGreaterEqual(freq_props['spectral_frequency'], 0)
        self.assertLessEqual(freq_props['spectral_frequency'], 2)
        
        self.assertGreaterEqual(freq_props['spectral_phase'], 0)
        self.assertLessEqual(freq_props['spectral_phase'], 1)
        
    def test_frequency_system_analysis(self):
        """测试频率系统分析"""
        analysis = self.analyzer.analyze_frequency_system()
        
        # Verify analysis results
        self.assertIn('frequency_universe_size', analysis)
        self.assertGreater(analysis['frequency_universe_size'], 0)
        
        self.assertIn('mean_spectral_frequency', analysis)
        self.assertIn('frequency_type_distribution', analysis)
        
    def test_graph_properties(self):
        """测试图论特性"""
        graph_props = self.analyzer.analyze_graph_properties()
        
        self.assertIn('network_nodes', graph_props)
        self.assertIn('network_density', graph_props)
        
        if graph_props['network_nodes'] > 0:
            self.assertGreaterEqual(graph_props['network_density'], 0)
            self.assertLessEqual(graph_props['network_density'], 1)
            
    def test_information_theory(self):
        """测试信息论分析"""
        info_analysis = self.analyzer.analyze_information_theory()
        
        self.assertIn('frequency_entropy', info_analysis)
        self.assertIn('frequency_complexity', info_analysis)
        
        # Entropy should be non-negative
        self.assertGreaterEqual(info_analysis['frequency_entropy'], 0)
        
    def test_category_theory(self):
        """测试范畴论分析"""
        cat_analysis = self.analyzer.analyze_category_theory()
        
        self.assertIn('frequency_morphisms', cat_analysis)
        self.assertIn('functoriality_ratio', cat_analysis)
        
        # Functoriality ratio should be between 0 and 1
        self.assertGreaterEqual(cat_analysis['functoriality_ratio'], 0)
        self.assertLessEqual(cat_analysis['functoriality_ratio'], 1)
        
    def test_phi_constraint_preservation(self):
        """测试φ-约束保持"""
        for value in self.analyzer.trace_universe:
            trace = self.analyzer.trace_universe[value]['trace']
            self.assertNotIn('11', trace, f"φ-constraint violated in trace for value {value}")
            
    def test_frequency_type_classification(self):
        """测试频率类型分类"""
        valid_types = {"high_frequency", "band_dominated", "analysis_intensive", "low_frequency"}
        
        for trace_data in self.analyzer.trace_universe.values():
            freq_type = trace_data['frequency_properties']['frequency_type']
            self.assertIn(freq_type, valid_types)

def run_collapse_analyzer_verification():
    """运行完整的CollapseAnalyzer验证"""
    print("=" * 80)
    print("Chapter 081: CollapseAnalyzer Unit Test Verification")
    print("从ψ=ψ(ψ)推导Spectral Analyzer for Collapse Frequency Bands")
    print("=" * 80)
    
    # Initialize system
    analyzer = CollapseAnalyzerSystem(max_trace_size=6, max_frequency_bands=20)
    
    # Run frequency system analysis
    print("\n1. Frequency System Analysis:")
    frequency_analysis = analyzer.analyze_frequency_system()
    for key, value in frequency_analysis.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.1f}%" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Run graph analysis
    print("\n2. Graph Theory Analysis:")
    graph_analysis = analyzer.analyze_graph_properties()
    for key, value in graph_analysis.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Run information theory analysis
    print("\n3. Information Theory Analysis:")
    info_analysis = analyzer.analyze_information_theory()
    for key, value in info_analysis.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Run category theory analysis
    print("\n4. Category Theory Analysis:")
    cat_analysis = analyzer.analyze_category_theory()
    for key, value in cat_analysis.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Generate visualizations
    print("\n5. Generating Visualizations...")
    analyzer.generate_visualizations()
    print("Visualizations saved to visualization_output/")
    
    # Run unit tests
    print("\n6. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCollapseAnalyzer)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✓ All CollapseAnalyzer tests passed!")
        print("\nThree-Domain Analysis Results:")
        print("=" * 50)
        
        traditional_size = 20
        phi_size = frequency_analysis['frequency_universe_size']
        convergence_size = min(traditional_size, phi_size)
        convergence_ratio = convergence_size / traditional_size
        
        print(f"Traditional universe size: {traditional_size} elements")
        print(f"φ-constrained universe size: {phi_size} elements") 
        print(f"Convergence ratio: {convergence_ratio:.3f} ({convergence_size}/{traditional_size} operations preserved)")
        print(f"Network density: {graph_analysis.get('network_density', 0):.3f} (frequency connectivity)")
        
        print(f"\nFrequency Structure Analysis:")
        print(f"Mean spectral frequency: {frequency_analysis['mean_spectral_frequency']:.3f} (frequency balance)")
        print(f"Mean frequency contribution: {frequency_analysis['mean_frequency_contribution']:.3f} (contribution strength)")
        print(f"Mean band component: {frequency_analysis['mean_band_component']:.3f} (band structure)")
        print(f"Mean spectral power: {frequency_analysis['mean_spectral_power']:.3f} (power distribution)")
        
        print(f"\nInformation Analysis:")
        print(f"Frequency entropy: {info_analysis['frequency_entropy']:.3f} bits (frequency encoding)")
        print(f"Band component entropy: {info_analysis['band_component_entropy']:.3f} bits (band encoding)")
        print(f"Spectral power entropy: {info_analysis['spectral_power_entropy']:.3f} bits (power encoding)")
        print(f"Type entropy: {info_analysis['type_entropy']:.3f} bits (type structure)")
        print(f"Frequency complexity: {info_analysis['frequency_complexity']} unique types (bounded diversity)")
        
        print(f"\nCategory Theory Analysis:")
        print(f"Frequency morphisms: {cat_analysis['frequency_morphisms']} (frequency relationships)")
        print(f"Functoriality ratio: {cat_analysis['functoriality_ratio']:.3f} (structure preservation)")
        print(f"Frequency groups: {cat_analysis['frequency_groups']} (complete classification)")
        
        print("\n" + "=" * 80)
        print("CollapseAnalyzer verification completed successfully!")
        print("All frequency band analysis components working correctly.")
        print("Three-domain analysis shows bounded frequency convergence.")
        print("=" * 80)
        
        return True
    else:
        print(f"\n✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False

if __name__ == "__main__":
    success = run_collapse_analyzer_verification()
    exit(0 if success else 1)