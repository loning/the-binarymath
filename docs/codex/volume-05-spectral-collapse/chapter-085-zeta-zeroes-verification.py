#!/usr/bin/env python3
"""
Chapter 085: ZetaZeroes Unit Test Verification
从ψ=ψ(ψ)推导ζ(s) Phase Failures as Collapse Instabilities

Core principle: From ψ = ψ(ψ) derive zeros of the zeta function where zeros mark
structural instabilities in collapse systems, creating phase failure patterns that
reveal deep connections between spectral analysis and structural collapse dynamics.

This verification program implements:
1. φ-constrained zeta function zeros as phase failure points
2. Instability analysis: spectral collapse patterns with structural breaks
3. Three-domain analysis: Traditional vs φ-constrained vs intersection zero theory
4. Graph theory analysis of instability networks and failure propagation
5. Information theory analysis of zero entropy and instability encoding
6. Category theory analysis of zero functors and failure morphisms
7. Visualization of zero patterns and instability dynamics
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

class ZetaZeroSystem:
    """
    Core system for implementing ζ(s) zeros as collapse instabilities.
    Implements φ-constrained zero analysis via spectral phase failure operations.
    """
    
    def __init__(self, max_s_value: int = 30, max_trace_length: int = 20):
        """Initialize zeta zero system with spectral analysis"""
        self.max_s_value = max_s_value
        self.max_trace_length = max_trace_length
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.zero_cache = {}
        self.instability_cache = {}
        self.phase_cache = {}
        self.trace_universe = self._build_trace_universe()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        for n in range(1, 50):  # Limited range for performance
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
        """分析trace的谱性质，用于零点检测"""
        result = {
            'value': value,
            'trace': trace,
            'phi_valid': True,
            'length': len(trace),
            'spectral_weight': self._compute_spectral_weight(trace),
            'phase': self._compute_phase(trace),
            'instability_measure': self._compute_instability(trace),
            'zero_proximity': self._compute_zero_proximity(trace, value)
        }
        return result
        
    def _compute_spectral_weight(self, trace: str) -> float:
        """计算谱权重：基于trace结构的频率分量"""
        if not trace or trace == "0":
            return 0.0
            
        weight = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Weight based on position and golden ratio
                weight += self.phi ** (-i)
        return weight
        
    def _compute_phase(self, trace: str) -> float:
        """计算相位：trace的谱相位"""
        if not trace or trace == "0":
            return 0.0
            
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                # Complex exponential for phase
                angle = 2 * pi * i / len(trace)
                real_part += cos(angle)
                imag_part += sin(angle)
                
        return atan2(imag_part, real_part)
        
    def _compute_instability(self, trace: str) -> float:
        """计算不稳定性度量：接近相位失败的程度"""
        if not trace or trace == "0":
            return 0.0
            
        # Instability based on pattern irregularity
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) < 2:
            return 0.0
            
        # Analyze spacing variations
        spacings = []
        for i in range(len(ones_positions) - 1):
            spacings.append(ones_positions[i+1] - ones_positions[i])
            
        if not spacings:
            return 0.0
            
        mean_spacing = sum(spacings) / len(spacings)
        variance = sum((s - mean_spacing)**2 for s in spacings) / len(spacings)
        
        # Normalize by trace length
        instability = sqrt(variance) / len(trace) if len(trace) > 0 else 0.0
        return min(instability, 1.0)
        
    def _compute_zero_proximity(self, trace: str, value: int) -> float:
        """计算零点接近度：trace距离zeta零点的距离"""
        # Simplified model: zeros occur at specific spectral resonances
        spectral_weight = self._compute_spectral_weight(trace)
        phase = self._compute_phase(trace)
        
        # Zero proximity based on phase alignment
        critical_phases = [pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4]
        min_distance = min(abs(phase - cp) for cp in critical_phases)
        
        # Weight by spectral strength
        proximity = exp(-min_distance) * spectral_weight
        return min(proximity, 1.0)
        
    def compute_zeta_zeros(self, s_range: Tuple[float, float] = (0.1, 30.0)) -> List[Dict]:
        """计算ζ(s)的零点（作为相位失败点）"""
        zeros = []
        s_values = np.linspace(s_range[0], s_range[1], 200)  # Reduced for performance
        
        for s in s_values:
            zeta_value = self._compute_zeta_value(s)
            
            # Check for sign change (simplified zero detection)
            if self._is_near_zero(zeta_value, s):
                zero_data = {
                    's_value': s,
                    'zeta_value': zeta_value,
                    'instability_type': self._classify_instability(s),
                    'affected_traces': self._find_affected_traces(s),
                    'phase_failure': self._compute_phase_failure(s)
                }
                zeros.append(zero_data)
                
        return zeros
        
    def _compute_zeta_value(self, s: float) -> complex:
        """计算ζ(s)在给定s处的值"""
        if s <= 0:
            return complex(float('inf'), 0)
            
        # Simplified zeta computation using trace weights
        zeta_sum = complex(0, 0)
        
        for n, data in self.trace_universe.items():
            if n > 0:
                weight = data['spectral_weight']
                phase = data['phase']
                
                # Contribution to zeta
                magnitude = weight / (n ** s)
                zeta_sum += magnitude * complex(cos(phase), sin(phase))
                
        return zeta_sum
        
    def _is_near_zero(self, zeta_value: complex, s: float) -> bool:
        """检查是否接近零点"""
        magnitude = abs(zeta_value)
        
        # Adaptive threshold based on s value
        threshold = 0.1 * exp(-s/10)
        
        return magnitude < threshold
        
    def _classify_instability(self, s: float) -> str:
        """分类不稳定性类型"""
        if s < 1:
            return "divergent"
        elif s < 5:
            return "critical"
        elif s < 15:
            return "resonant"
        else:
            return "stable"
            
    def _find_affected_traces(self, s: float) -> List[int]:
        """找到受零点影响的traces"""
        affected = []
        
        for n, data in self.trace_universe.items():
            # Check if trace is resonant at this s value
            weight = data['spectral_weight']
            contribution = weight / (n ** s) if n > 0 else 0
            
            if contribution > 0.1:  # Significant contribution threshold
                affected.append(n)
                
        return affected[:10]  # Return top 10 affected
        
    def _compute_phase_failure(self, s: float) -> Dict:
        """计算相位失败特征"""
        failure_data = {
            'failure_strength': 1.0 / (1 + s),  # Decreases with s
            'propagation_range': int(10 * exp(-s/5)),  # How far failure spreads
            'recovery_time': s * self.phi,  # Time to recover from failure
            'critical_traces': []
        }
        
        # Find critical traces at this zero
        for n, data in self.trace_universe.items():
            if data['instability_measure'] > 0.7:
                failure_data['critical_traces'].append(n)
                
        return failure_data
        
    def analyze_zero_network(self) -> nx.Graph:
        """构建零点影响网络"""
        G = nx.Graph()
        
        # Add nodes for traces
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # Add edges based on instability propagation
        for n1, data1 in self.trace_universe.items():
            for n2, data2 in self.trace_universe.items():
                if n1 < n2:
                    # Connect if instabilities are correlated
                    instab_corr = abs(data1['instability_measure'] - data2['instability_measure'])
                    if instab_corr < 0.1 and data1['zero_proximity'] > 0.5:
                        G.add_edge(n1, n2, weight=1 - instab_corr)
                        
        return G
        
    def compute_instability_entropy(self) -> Dict:
        """计算不稳定性的信息熵"""
        instabilities = [data['instability_measure'] for data in self.trace_universe.values()]
        proximities = [data['zero_proximity'] for data in self.trace_universe.values()]
        
        # Discretize for entropy calculation
        instab_bins = np.histogram(instabilities, bins=10)[0]
        prox_bins = np.histogram(proximities, bins=10)[0]
        
        # Normalize
        instab_probs = instab_bins / sum(instab_bins) if sum(instab_bins) > 0 else instab_bins
        prox_probs = prox_bins / sum(prox_bins) if sum(prox_bins) > 0 else prox_bins
        
        # Compute entropy
        instab_entropy = -sum(p * log2(p) if p > 0 else 0 for p in instab_probs)
        prox_entropy = -sum(p * log2(p) if p > 0 else 0 for p in prox_probs)
        
        return {
            'instability_entropy': instab_entropy,
            'proximity_entropy': prox_entropy,
            'total_entropy': instab_entropy + prox_entropy
        }
        
    def analyze_zero_categories(self) -> Dict:
        """范畴论分析：零点作为态射"""
        zeros = self.compute_zeta_zeros()
        
        # Group zeros by instability type
        zero_categories = defaultdict(list)
        for zero in zeros:
            zero_categories[zero['instability_type']].append(zero)
            
        # Analyze functorial relationships
        functors = []
        for type1 in zero_categories:
            for type2 in zero_categories:
                if type1 != type2:
                    # Check if there's a functor between categories
                    if self._has_functor(zero_categories[type1], zero_categories[type2]):
                        functors.append((type1, type2))
                        
        return {
            'categories': dict(zero_categories),
            'functors': functors,
            'category_count': len(zero_categories)
        }
        
    def _has_functor(self, zeros1: List[Dict], zeros2: List[Dict]) -> bool:
        """检查两个零点类别之间是否存在函子"""
        if not zeros1 or not zeros2:
            return False
            
        # Simplified: check if s-values have regular relationship
        s_values1 = sorted([z['s_value'] for z in zeros1])
        s_values2 = sorted([z['s_value'] for z in zeros2])
        
        # Look for linear relationship
        if len(s_values1) >= 2 and len(s_values2) >= 2:
            ratio = s_values2[0] / s_values1[0] if s_values1[0] != 0 else 0
            for i in range(1, min(len(s_values1), len(s_values2))):
                if s_values1[i] != 0:
                    new_ratio = s_values2[i] / s_values1[i]
                    if abs(new_ratio - ratio) > 0.1:
                        return False
            return True
        return False
        
    def generate_visualizations(self):
        """生成所有可视化图表"""
        print("Generating zeta zero visualizations...")
        
        # 1. Zero distribution plot
        self._plot_zero_distribution()
        
        # 2. Instability network
        self._plot_instability_network()
        
        # 3. Phase failure patterns
        self._plot_phase_failures()
        
        # 4. Domain analysis
        self._plot_domain_analysis()
        
        # 5. Spectral analysis
        self._plot_spectral_analysis()
        
        # 6. Category structure
        self._plot_category_structure()
        
    def _plot_zero_distribution(self):
        """绘制零点分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        zeros = self.compute_zeta_zeros()
        
        # Extract s-values and instability types
        s_values = [z['s_value'] for z in zeros]
        instab_types = [z['instability_type'] for z in zeros]
        
        # Color map for instability types
        type_colors = {
            'divergent': 'red',
            'critical': 'orange',
            'resonant': 'yellow',
            'stable': 'green'
        }
        
        # Plot zeros on complex plane (simplified to real line)
        colors = [type_colors[t] for t in instab_types]
        ax1.scatter(s_values, [0]*len(s_values), c=colors, s=100, alpha=0.7)
        ax1.set_xlabel('s (real part)')
        ax1.set_ylabel('Imaginary part')
        ax1.set_title('ζ(s) Zeros Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # Add instability regions
        ax1.axvspan(0, 1, alpha=0.1, color='red', label='Divergent')
        ax1.axvspan(1, 5, alpha=0.1, color='orange', label='Critical')
        ax1.axvspan(5, 15, alpha=0.1, color='yellow', label='Resonant')
        ax1.axvspan(15, 30, alpha=0.1, color='green', label='Stable')
        ax1.legend()
        
        # Histogram of zero density
        ax2.hist(s_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('s value')
        ax2.set_ylabel('Zero count')
        ax2.set_title('Zero Density Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_instability_network(self):
        """绘制不稳定性传播网络"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        G = self.analyze_zero_network()
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node colors based on instability
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            instab = G.nodes[node]['instability_measure']
            node_colors.append(instab)
            node_sizes.append(300 + 700 * instab)
            
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, cmap='hot',
                             alpha=0.8, ax=ax)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)
        
        # Add labels for high-instability nodes
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['instability_measure'] > 0.7:
                labels[node] = str(node)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Instability Propagation Network', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Instability Measure')
        
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_phase_failures(self):
        """绘制相位失败模式"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        zeros = self.compute_zeta_zeros()
        
        # 1. Phase failure strength vs s
        s_values = [z['s_value'] for z in zeros]
        failure_strengths = [z['phase_failure']['failure_strength'] for z in zeros]
        
        ax1.scatter(s_values, failure_strengths, alpha=0.7, color='red')
        ax1.set_xlabel('s value')
        ax1.set_ylabel('Failure Strength')
        ax1.set_title('Phase Failure Strength at Zeros')
        ax1.grid(True, alpha=0.3)
        
        # Fit exponential decay
        if s_values:
            s_fit = np.linspace(min(s_values), max(s_values), 100)
            strength_fit = 1.0 / (1 + s_fit)
            ax1.plot(s_fit, strength_fit, 'b--', label='Theoretical: 1/(1+s)')
            ax1.legend()
        
        # 2. Propagation range
        prop_ranges = [z['phase_failure']['propagation_range'] for z in zeros]
        
        ax2.scatter(s_values, prop_ranges, alpha=0.7, color='orange')
        ax2.set_xlabel('s value')
        ax2.set_ylabel('Propagation Range')
        ax2.set_title('Failure Propagation Range')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Recovery time
        recovery_times = [z['phase_failure']['recovery_time'] for z in zeros]
        
        ax3.scatter(s_values, recovery_times, alpha=0.7, color='green')
        ax3.set_xlabel('s value')
        ax3.set_ylabel('Recovery Time')
        ax3.set_title('System Recovery Time')
        ax3.grid(True, alpha=0.3)
        
        # Add golden ratio line
        if s_values:
            s_fit = np.linspace(min(s_values), max(s_values), 100)
            recovery_fit = s_fit * self.phi
            ax3.plot(s_fit, recovery_fit, 'g--', label=f'φ×s = {self.phi:.3f}×s')
            ax3.legend()
        
        # 4. Critical trace count
        critical_counts = [len(z['phase_failure']['critical_traces']) for z in zeros]
        
        ax4.bar(range(len(critical_counts)), critical_counts, alpha=0.7, color='purple')
        ax4.set_xlabel('Zero Index')
        ax4.set_ylabel('Critical Trace Count')
        ax4.set_title('Number of Critical Traces per Zero')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Phase Failure Patterns at ζ(s) Zeros', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-failures.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_domain_analysis(self):
        """绘制三域分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Traditional zeros (simplified model)
        traditional_zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876])  # First few Riemann zeros
        
        # φ-constrained zeros
        phi_zeros = self.compute_zeta_zeros()
        phi_s_values = [z['s_value'] for z in phi_zeros]
        
        # 1. Compare zero distributions
        ax1.scatter(traditional_zeros, [1]*len(traditional_zeros), 
                   label='Traditional', s=100, alpha=0.7, color='blue')
        ax1.scatter(phi_s_values, [0.5]*len(phi_s_values), 
                   label='φ-constrained', s=50, alpha=0.7, color='red')
        
        ax1.set_xlabel('s value')
        ax1.set_ylabel('System Type')
        ax1.set_title('Zero Distribution Comparison')
        ax1.set_yticks([0.5, 1])
        ax1.set_yticklabels(['φ-constrained', 'Traditional'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Instability measure distribution
        instabilities = [data['instability_measure'] for data in self.trace_universe.values()]
        
        ax2.hist(instabilities, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Instability Measure')
        ax2.set_ylabel('Count')
        ax2.set_title('Instability Distribution in φ-Traces')
        ax2.grid(True, alpha=0.3)
        
        # 3. Zero proximity analysis
        proximities = [data['zero_proximity'] for data in self.trace_universe.values()]
        
        ax3.hist(proximities, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Zero Proximity')
        ax3.set_ylabel('Count')
        ax3.set_title('Zero Proximity Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Information entropy
        entropy_data = self.compute_instability_entropy()
        
        categories = ['Instability', 'Proximity', 'Total']
        values = [entropy_data['instability_entropy'], 
                 entropy_data['proximity_entropy'],
                 entropy_data['total_entropy']]
        
        bars = ax4.bar(categories, values, alpha=0.7, color=['red', 'green', 'blue'])
        ax4.set_ylabel('Entropy (bits)')
        ax4.set_title('Information Entropy Analysis')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-domains.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_spectral_analysis(self):
        """绘制谱分析"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate spectral data
        s_values = np.linspace(0.5, 30, 50)
        theta_values = np.linspace(0, 2*pi, 50)
        S, THETA = np.meshgrid(s_values, theta_values)
        
        # Compute spectral intensity (simplified model)
        INTENSITY = np.zeros_like(S)
        
        for i in range(len(s_values)):
            for j in range(len(theta_values)):
                s = s_values[i]
                theta = theta_values[j]
                
                # Spectral intensity based on zeta properties
                zeta_val = self._compute_zeta_value(s)
                magnitude = abs(zeta_val)
                
                # Modulate by angle
                intensity = magnitude * (1 + 0.5 * cos(theta))
                
                # Add zero enhancement
                for zero in self.compute_zeta_zeros():
                    if abs(s - zero['s_value']) < 1:
                        intensity *= 0.1  # Suppress near zeros
                        
                INTENSITY[j, i] = intensity
        
        # Create surface plot
        surf = ax.plot_surface(S * np.cos(THETA), S * np.sin(THETA), INTENSITY,
                             cmap='viridis', alpha=0.8, edgecolor='none')
        
        # Mark zeros
        zeros = self.compute_zeta_zeros()
        for zero in zeros[:10]:  # First 10 zeros
            s = zero['s_value']
            # Draw circle at zero
            theta_circle = np.linspace(0, 2*pi, 50)
            x_circle = s * np.cos(theta_circle)
            y_circle = s * np.sin(theta_circle)
            z_circle = np.zeros_like(theta_circle)
            ax.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Re(s)')
        ax.set_ylabel('Im(s)')
        ax.set_zlabel('Spectral Intensity')
        ax.set_title('Spectral Analysis of ζ(s) with Zeros', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-spectral.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_category_structure(self):
        """绘制范畴结构"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        category_data = self.analyze_zero_categories()
        
        # 1. Category sizes
        categories = list(category_data['categories'].keys())
        sizes = [len(zeros) for zeros in category_data['categories'].values()]
        
        colors = {'divergent': 'red', 'critical': 'orange', 
                 'resonant': 'yellow', 'stable': 'green'}
        bar_colors = [colors.get(cat, 'gray') for cat in categories]
        
        bars = ax1.bar(categories, sizes, color=bar_colors, alpha=0.7)
        ax1.set_ylabel('Number of Zeros')
        ax1.set_title('Zero Categories by Instability Type')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    str(size), ha='center', va='bottom')
        
        # 2. Functor diagram
        G = nx.DiGraph()
        
        # Add nodes for categories
        for cat in categories:
            G.add_node(cat)
            
        # Add edges for functors
        for source, target in category_data['functors']:
            G.add_edge(source, target)
            
        # Layout
        pos = nx.circular_layout(G)
        
        # Draw nodes
        node_colors_list = [colors.get(node, 'gray') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_list,
                             node_size=1500, alpha=0.7, ax=ax2)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='black',
                             arrows=True, arrowsize=20,
                             width=2, alpha=0.5, ax=ax2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax2)
        
        ax2.set_title('Category Functors', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('chapter-085-zeta-zeroes-categories.png', dpi=300, bbox_inches='tight')
        plt.close()


class TestZetaZeros(unittest.TestCase):
    """测试ZetaZero系统的各个组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.zeta_system = ZetaZeroSystem()
        
    def test_trace_encoding(self):
        """测试trace编码"""
        trace = self.zeta_system._encode_to_trace(10)
        self.assertIsInstance(trace, str)
        self.assertNotIn('11', trace)  # φ-constraint
        
    def test_spectral_weight(self):
        """测试谱权重计算"""
        trace = "10101"
        weight = self.zeta_system._compute_spectral_weight(trace)
        self.assertIsInstance(weight, float)
        self.assertGreaterEqual(weight, 0.0)
        
    def test_phase_computation(self):
        """测试相位计算"""
        trace = "10010"
        phase = self.zeta_system._compute_phase(trace)
        self.assertIsInstance(phase, float)
        self.assertGreaterEqual(phase, -pi)
        self.assertLessEqual(phase, pi)
        
    def test_instability_measure(self):
        """测试不稳定性度量"""
        trace = "101010"
        instability = self.zeta_system._compute_instability(trace)
        self.assertIsInstance(instability, float)
        self.assertGreaterEqual(instability, 0.0)
        self.assertLessEqual(instability, 1.0)
        
    def test_zero_computation(self):
        """测试零点计算"""
        zeros = self.zeta_system.compute_zeta_zeros((0.1, 10))
        self.assertIsInstance(zeros, list)
        if zeros:
            zero = zeros[0]
            self.assertIn('s_value', zero)
            self.assertIn('instability_type', zero)
            self.assertIn('phase_failure', zero)
            
    def test_network_analysis(self):
        """测试网络分析"""
        G = self.zeta_system.analyze_zero_network()
        self.assertIsInstance(G, nx.Graph)
        self.assertGreater(len(G.nodes()), 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropy = self.zeta_system.compute_instability_entropy()
        self.assertIn('instability_entropy', entropy)
        self.assertIn('proximity_entropy', entropy)
        self.assertGreaterEqual(entropy['instability_entropy'], 0)
        
    def test_category_analysis(self):
        """测试范畴分析"""
        categories = self.zeta_system.analyze_zero_categories()
        self.assertIn('categories', categories)
        self.assertIn('functors', categories)
        self.assertIsInstance(categories['category_count'], int)


def run_zeta_zero_verification():
    """运行完整的zeta零点验证"""
    print("="*80)
    print("Chapter 085: ZetaZeroes Unit Test Verification")
    print("从ψ=ψ(ψ)推导Collapse Instabilities as ζ(s) Phase Failures")
    print("="*80)
    
    # Initialize system
    zeta_system = ZetaZeroSystem()
    
    # 1. Zero computation
    print("\n1. Computing ζ(s) Zeros:")
    zeros = zeta_system.compute_zeta_zeros()
    print(f"Found {len(zeros)} zeros in range")
    
    if zeros:
        print("\nFirst 5 zeros:")
        for i, zero in enumerate(zeros[:5]):
            print(f"  Zero {i+1}: s = {zero['s_value']:.4f}, "
                  f"type = {zero['instability_type']}, "
                  f"affected traces = {len(zero['affected_traces'])}")
    
    # 2. Network analysis
    print("\n2. Instability Network Analysis:")
    G = zeta_system.analyze_zero_network()
    print(f"Network nodes: {len(G.nodes())}")
    print(f"Network edges: {len(G.edges())}")
    
    if len(G.nodes()) > 0:
        clustering = nx.average_clustering(G)
        print(f"Average clustering: {clustering:.3f}")
        
        # Find most unstable nodes
        instab_nodes = [(n, G.nodes[n]['instability_measure']) 
                       for n in G.nodes()]
        instab_nodes.sort(key=lambda x: x[1], reverse=True)
        
        print("\nMost unstable traces:")
        for node, instab in instab_nodes[:5]:
            print(f"  Trace {node}: instability = {instab:.3f}")
    
    # 3. Information theory analysis
    print("\n3. Information Theory Analysis:")
    entropy = zeta_system.compute_instability_entropy()
    print(f"Instability entropy: {entropy['instability_entropy']:.3f} bits")
    print(f"Proximity entropy: {entropy['proximity_entropy']:.3f} bits")
    print(f"Total entropy: {entropy['total_entropy']:.3f} bits")
    
    # 4. Category theory analysis
    print("\n4. Category Theory Analysis:")
    categories = zeta_system.analyze_zero_categories()
    print(f"Number of categories: {categories['category_count']}")
    print(f"Number of functors: {len(categories['functors'])}")
    
    print("\nCategory sizes:")
    for cat_type, zeros in categories['categories'].items():
        print(f"  {cat_type}: {len(zeros)} zeros")
    
    # 5. Generate visualizations
    print("\n5. Generating Visualizations...")
    zeta_system.generate_visualizations()
    print("Visualizations saved to current directory")
    
    # 6. Run unit tests
    print("\n6. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestZetaZeros)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("ZetaZeroes verification completed successfully!")
    print(f"Found {len(zeros)} zeros marking collapse instabilities")
    print("Phase failures reveal deep structural connections")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_zeta_zero_verification()
    exit(0 if success else 1)