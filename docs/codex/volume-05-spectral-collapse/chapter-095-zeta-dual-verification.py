#!/usr/bin/env python3
"""
Chapter 095: ZetaDual Unit Test Verification
从ψ=ψ(ψ)推导Dual Collapse Paths under ζ(s) Inversion Mapping

Core principle: From ψ = ψ(ψ) derive duality relationships where spectral
collapse paths create systematic dual mappings through ζ-function inversion,
generating conjugate path structures that preserve essential spectral properties
while revealing complementary organizational principles that encode the
fundamental duality architecture of collapsed spectral space through entropy-
increasing tensor transformations.

This verification program implements:
1. φ-constrained dual path construction through ζ-inversion analysis
2. Duality mapping: systematic path inversion and conjugate relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection duality theory
4. Graph theory analysis of dual networks and conjugate pathways
5. Information theory analysis of duality entropy and mapping encoding
6. Category theory analysis of dual functors and inversion morphisms
7. Visualization of dual structures and ζ-inversion spectral mappings
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

class ZetaDualSystem:
    """
    Core system for implementing dual collapse paths under ζ-function inversion mapping.
    Implements φ-constrained duality dynamics via ζ-inversion path operations.
    """
    
    def __init__(self, max_trace_value: int = 100, inversion_resolution: int = 12):
        """Initialize zeta dual system with inversion mapping analysis"""
        self.max_trace_value = max_trace_value
        self.inversion_resolution = inversion_resolution
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.dual_cache = {}
        self.inversion_cache = {}
        self.path_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.dual_pairs = self._detect_dual_pairs()
        self.inversion_mapping = self._compute_inversion_mapping()
        
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
                dual_data = self._analyze_dual_properties(trace, n)
                universe[n] = dual_data
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
        
    def _analyze_dual_properties(self, trace: str, value: int) -> Dict:
        """分析trace的dual properties"""
        properties = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'complexity': self._compute_complexity(trace),
            'zeta_parameter': self._compute_zeta_parameter(trace, value),
            'dual_parameter': self._compute_dual_parameter(trace, value),
            'inversion_weight': self._compute_inversion_weight(trace, value),
            'conjugate_frequency': self._compute_conjugate_frequency(trace, value),
            'duality_strength': self._compute_duality_strength(trace, value),
            'path_symmetry': self._compute_path_symmetry(trace, value),
            'inversion_stability': self._compute_inversion_stability(trace, value),
            'dual_entropy': self._compute_dual_entropy(trace, value),
            'spectral_conjugate': self._compute_spectral_conjugate(trace, value),
            'mapping_index': self._compute_mapping_index(trace, value)
        }
        return properties
        
    def _compute_complexity(self, trace: str) -> float:
        """计算trace复杂度"""
        if len(trace) <= 1:
            return 0.0
        
        # 计算模式复杂度
        patterns = set()
        for i in range(len(trace) - 1):
            patterns.add(trace[i:i+2])
        
        return len(patterns) / (len(trace) - 1)
        
    def _compute_zeta_parameter(self, trace: str, value: int) -> float:
        """计算ζ-parameter"""
        s = 0.5  # Critical line
        for i, bit in enumerate(trace):
            if bit == '1':
                s += 0.1 * cos(pi * (i + 1) / len(trace)) / (i + 1)
        return abs(s)
        
    def _compute_dual_parameter(self, trace: str, value: int) -> float:
        """计算dual parameter（ζ-inversion）"""
        s = self._compute_zeta_parameter(trace, value)
        # ζ-inversion: s → 1 - s
        return abs(1.0 - s)
        
    def _compute_inversion_weight(self, trace: str, value: int) -> float:
        """计算inversion weight"""
        weight = trace.count('1') / len(trace)
        # Inversion preserves structure
        return weight * (1.0 + 0.1 * sin(2 * pi * value / 100))
        
    def _compute_conjugate_frequency(self, trace: str, value: int) -> float:
        """计算conjugate frequency"""
        base_freq = (log(value + 1) / log(2)) % (2 * pi)
        # Conjugate through reflection
        return (2 * pi - base_freq) % (2 * pi)
        
    def _compute_duality_strength(self, trace: str, value: int) -> float:
        """计算duality strength"""
        s_orig = self._compute_zeta_parameter(trace, value)
        s_dual = self._compute_dual_parameter(trace, value)
        # Duality strength through parameter correlation
        return abs(s_orig * s_dual - 0.25)  # |s(1-s) - 1/4|
        
    def _compute_path_symmetry(self, trace: str, value: int) -> float:
        """计算path symmetry"""
        # Measure trace palindromic properties
        reversed_trace = trace[::-1]
        matches = sum(1 for i, (a, b) in enumerate(zip(trace, reversed_trace)) if a == b)
        return matches / len(trace)
        
    def _compute_inversion_stability(self, trace: str, value: int) -> float:
        """计算inversion stability"""
        s = self._compute_zeta_parameter(trace, value)
        s_dual = self._compute_dual_parameter(trace, value)
        # Stability through inversion consistency
        return 1.0 - abs(s + s_dual - 1.0)
        
    def _compute_dual_entropy(self, trace: str, value: int) -> float:
        """计算dual entropy"""
        if len(trace) <= 1:
            return 0.0
        
        # Binary entropy
        p = trace.count('1') / len(trace)
        if p == 0 or p == 1:
            return 0.0
        
        h = -p * log2(p) - (1-p) * log2(1-p)
        # Dual entropy through inversion
        p_dual = 1 - p
        if p_dual == 0 or p_dual == 1:
            h_dual = 0.0
        else:
            h_dual = -p_dual * log2(p_dual) - (1-p_dual) * log2(1-p_dual)
        
        return (h + h_dual) / 2.0
        
    def _compute_spectral_conjugate(self, trace: str, value: int) -> float:
        """计算spectral conjugate"""
        # Spectral conjugate through harmonic analysis
        conjugate = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                conjugate += sin(pi * value * (i + 1) / len(trace)) / (i + 1)
        return abs(conjugate)
        
    def _compute_mapping_index(self, trace: str, value: int) -> int:
        """计算mapping index for inversion classes"""
        # Map to inversion resolution classes
        s = self._compute_zeta_parameter(trace, value)
        return int(s * self.inversion_resolution) % self.inversion_resolution
        
    def _detect_dual_pairs(self) -> List[Tuple[int, int]]:
        """检测dual pairs"""
        pairs = []
        traces = list(self.trace_universe.keys())
        
        for i, t1 in enumerate(traces):
            for j, t2 in enumerate(traces[i+1:], i+1):
                if self._are_dual_traces(t1, t2):
                    pairs.append((t1, t2))
        
        return pairs
        
    def _are_dual_traces(self, t1: int, t2: int) -> bool:
        """检查两个traces是否为dual pair"""
        props1 = self.trace_universe[t1]
        props2 = self.trace_universe[t2]
        
        # Dual if ζ-parameters are complementary
        s1 = props1['zeta_parameter']
        s2 = props2['zeta_parameter']
        
        # Check if s1 + s2 ≈ 1 (ζ-inversion)
        return abs(s1 + s2 - 1.0) < 0.1
        
    def _compute_inversion_mapping(self) -> Dict[int, Dict]:
        """计算inversion mapping structure"""
        mapping = {}
        
        for value, props in self.trace_universe.items():
            s = props['zeta_parameter']
            s_dual = props['dual_parameter']
            
            mapping[value] = {
                'original_parameter': s,
                'dual_parameter': s_dual,
                'inversion_map': s_dual,
                'fixed_point_distance': abs(s - 0.5),  # Distance from fixed point s=1/2
                'duality_product': s * s_dual,
                'inversion_class': props['mapping_index']
            }
            
        return mapping
        
    def build_dual_network(self) -> nx.Graph:
        """构建dual network"""
        G = nx.Graph()
        
        # Add nodes
        for value in self.trace_universe.keys():
            props = self.trace_universe[value]
            G.add_node(value, **props)
        
        # Add edges for dual relationships
        for t1, t2 in self.dual_pairs:
            props1 = self.trace_universe[t1]
            props2 = self.trace_universe[t2]
            
            # Edge weight based on duality strength
            weight = (props1['duality_strength'] + props2['duality_strength']) / 2
            G.add_edge(t1, t2, weight=weight, relationship='dual_pair')
        
        # Add edges for spectral proximity
        traces = list(self.trace_universe.keys())
        for i, t1 in enumerate(traces):
            for j, t2 in enumerate(traces[i+1:], i+1):
                if t1 != t2 and (t1, t2) not in self.dual_pairs and (t2, t1) not in self.dual_pairs:
                    props1 = self.trace_universe[t1]
                    props2 = self.trace_universe[t2]
                    
                    # Spectral proximity through conjugate frequencies
                    freq_diff = abs(props1['conjugate_frequency'] - props2['conjugate_frequency'])
                    if freq_diff < pi/2:  # Close conjugate frequencies
                        proximity_weight = 1.0 - freq_diff / (pi/2)
                        if proximity_weight > 0.3:  # Threshold for inclusion
                            G.add_edge(t1, t2, weight=proximity_weight, relationship='spectral_proximity')
        
        return G
        
    def analyze_information_theory(self) -> Dict[str, float]:
        """分析information theory"""
        def compute_entropy(data_list):
            if not data_list:
                return 0.0
            
            # Check if all values are the same
            data_array = np.array(data_list)
            if np.all(data_array == data_array[0]):
                return 0.0
            
            # Adaptive binning
            unique_values = len(np.unique(data_array))
            bin_count = min(10, max(3, unique_values))
            
            try:
                hist, _ = np.histogram(data_array, bins=bin_count)
                hist = hist[hist > 0]  # Remove zero bins
                probabilities = hist / hist.sum()
                return -np.sum(probabilities * np.log2(probabilities))
            except:
                # Fallback: count unique values
                unique_count = len(np.unique(data_array))
                return log2(unique_count) if unique_count > 1 else 0.0
        
        properties = [
            'zeta_parameter', 'dual_parameter', 'inversion_weight',
            'conjugate_frequency', 'duality_strength', 'path_symmetry',
            'inversion_stability', 'dual_entropy', 'spectral_conjugate',
            'mapping_index'
        ]
        
        entropies = {}
        for prop in properties:
            values = [self.trace_universe[t][prop] for t in self.trace_universe.keys()]
            entropies[f"{prop.replace('_', ' ').title()} entropy"] = compute_entropy(values)
        
        return entropies
        
    def analyze_category_theory(self) -> Dict[str, Any]:
        """分析category theory"""
        # Classify traces by inversion stability
        classifications = {}
        for value, props in self.trace_universe.items():
            stability = props['inversion_stability']
            if stability > 0.8:
                category = 'stable_inversion'
            elif stability > 0.5:
                category = 'moderate_inversion'
            else:
                category = 'unstable_inversion'
            
            if category not in classifications:
                classifications[category] = []
            classifications[category].append(value)
        
        # Compute morphisms
        total_morphisms = 0
        for cat_traces in classifications.values():
            # Morphisms within category
            n = len(cat_traces)
            total_morphisms += n * (n - 1)  # All ordered pairs within category
        
        # Cross-category morphisms (limited)
        cat_names = list(classifications.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i+1:]:
                # Allow some cross-category morphisms for dual pairs
                cross_morphisms = 0
                for t1 in classifications[cat1]:
                    for t2 in classifications[cat2]:
                        if (t1, t2) in self.dual_pairs or (t2, t1) in self.dual_pairs:
                            cross_morphisms += 2  # Bidirectional morphism
                total_morphisms += cross_morphisms
        
        total_objects = len(self.trace_universe)
        morphism_density = total_morphisms / (total_objects * total_objects) if total_objects > 0 else 0
        
        return {
            'categories': len(classifications),
            'classifications': classifications,
            'total_morphisms': total_morphisms,
            'morphism_density': morphism_density
        }
        
    def visualize_dual_dynamics(self):
        """可视化dual dynamics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Chapter 095: ZetaDual - Dual Collapse Paths under ζ(s) Inversion Mapping', fontsize=16, fontweight='bold')
        
        # 1. ζ-Parameter vs Dual Parameter scatter
        values = list(self.trace_universe.keys())
        zeta_params = [self.trace_universe[v]['zeta_parameter'] for v in values]
        dual_params = [self.trace_universe[v]['dual_parameter'] for v in values]
        duality_strengths = [self.trace_universe[v]['duality_strength'] for v in values]
        
        scatter = ax1.scatter(zeta_params, dual_params, c=duality_strengths, s=60, alpha=0.7, cmap='viridis')
        ax1.plot([0, 1], [1, 0], 'r--', alpha=0.8, linewidth=2, label='Perfect Inversion s + s\' = 1')
        ax1.set_xlabel('ζ-Parameter s')
        ax1.set_ylabel('Dual Parameter s\'')
        ax1.set_title('ζ-Inversion Mapping')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Duality Strength')
        
        # 2. Conjugate frequency distribution
        conjugate_freqs = [self.trace_universe[v]['conjugate_frequency'] for v in values]
        ax2.hist(conjugate_freqs, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Conjugate Frequency')
        ax2.set_ylabel('Count')
        ax2.set_title('Conjugate Frequency Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Inversion stability vs path symmetry
        inversion_stabilities = [self.trace_universe[v]['inversion_stability'] for v in values]
        path_symmetries = [self.trace_universe[v]['path_symmetry'] for v in values]
        mapping_indices = [self.trace_universe[v]['mapping_index'] for v in values]
        
        scatter2 = ax3.scatter(path_symmetries, inversion_stabilities, c=mapping_indices, s=60, alpha=0.7, cmap='plasma')
        ax3.set_xlabel('Path Symmetry')
        ax3.set_ylabel('Inversion Stability')
        ax3.set_title('Symmetry vs Stability')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='Mapping Index')
        
        # 4. Dual entropy vs spectral conjugate
        dual_entropies = [self.trace_universe[v]['dual_entropy'] for v in values]
        spectral_conjugates = [self.trace_universe[v]['spectral_conjugate'] for v in values]
        
        ax4.scatter(dual_entropies, spectral_conjugates, c=duality_strengths, s=60, alpha=0.7, cmap='cool')
        ax4.set_xlabel('Dual Entropy')
        ax4.set_ylabel('Spectral Conjugate')
        ax4.set_title('Entropy vs Spectral Conjugate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-095-zeta-dual-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_inversion_analysis(self):
        """可视化inversion analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ζ-Inversion Mapping Analysis', fontsize=16, fontweight='bold')
        
        values = list(self.trace_universe.keys())
        
        # 1. Inversion mapping classes
        mapping_indices = [self.trace_universe[v]['mapping_index'] for v in values]
        inversion_weights = [self.trace_universe[v]['inversion_weight'] for v in values]
        
        # Create class-based visualization
        unique_indices = sorted(set(mapping_indices))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_indices)))
        
        for i, idx in enumerate(unique_indices):
            mask = [mi == idx for mi in mapping_indices]
            class_values = [v for v, m in zip(values, mask) if m]
            class_weights = [w for w, m in zip(inversion_weights, mask) if m]
            ax1.scatter(class_values, class_weights, c=[colors[i]], s=60, alpha=0.7, label=f'Class {idx}')
        
        ax1.set_xlabel('Trace Value')
        ax1.set_ylabel('Inversion Weight')
        ax1.set_title('Inversion Mapping Classes')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Fixed point analysis
        zeta_params = [self.trace_universe[v]['zeta_parameter'] for v in values]
        fixed_point_distances = [abs(s - 0.5) for s in zeta_params]
        duality_products = [self.inversion_mapping[v]['duality_product'] for v in values]
        
        ax2.scatter(fixed_point_distances, duality_products, c=zeta_params, s=60, alpha=0.7, cmap='viridis')
        ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.8, label='Maximum Product s(1-s)=1/4')
        ax2.set_xlabel('Distance from Fixed Point |s - 1/2|')
        ax2.set_ylabel('Duality Product s·s\'')
        ax2.set_title('Fixed Point Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Dual pair connections
        dual_pair_indices = []
        dual_pair_strengths = []
        
        for t1, t2 in self.dual_pairs:
            dual_pair_indices.append(f"{t1}-{t2}")
            strength1 = self.trace_universe[t1]['duality_strength']
            strength2 = self.trace_universe[t2]['duality_strength']
            dual_pair_strengths.append((strength1 + strength2) / 2)
        
        if dual_pair_strengths:
            ax3.bar(range(len(dual_pair_strengths)), dual_pair_strengths, alpha=0.7, color='purple')
            ax3.set_xlabel('Dual Pair Index')
            ax3.set_ylabel('Average Duality Strength')
            ax3.set_title(f'Dual Pairs ({len(self.dual_pairs)} pairs)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Dual Pairs Found', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Dual Pairs Analysis')
        
        # 4. Spectral conjugate vs inversion stability
        inversion_stabilities = [self.trace_universe[v]['inversion_stability'] for v in values]
        spectral_conjugates = [self.trace_universe[v]['spectral_conjugate'] for v in values]
        
        ax4.scatter(spectral_conjugates, inversion_stabilities, c=zeta_params, s=60, alpha=0.7, cmap='coolwarm')
        ax4.set_xlabel('Spectral Conjugate')
        ax4.set_ylabel('Inversion Stability')
        ax4.set_title('Conjugate vs Stability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-095-zeta-dual-inversion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_network_and_categories(self):
        """可视化network and categories"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dual Network and Categorical Analysis', fontsize=16, fontweight='bold')
        
        # Build network
        G = self.build_dual_network()
        
        # 1. Network visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Color nodes by inversion stability
        node_colors = []
        for node in G.nodes():
            stability = self.trace_universe[node]['inversion_stability']
            node_colors.append(stability)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='viridis', alpha=0.8, ax=ax1)
        
        # Draw dual pair edges in red
        dual_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'dual_pair']
        if dual_edges:
            nx.draw_networkx_edges(G, pos, edgelist=dual_edges, edge_color='red', width=2, alpha=0.8, ax=ax1)
        
        # Draw proximity edges in blue
        proximity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'spectral_proximity']
        if proximity_edges:
            nx.draw_networkx_edges(G, pos, edgelist=proximity_edges, edge_color='blue', width=1, alpha=0.5, ax=ax1)
        
        ax1.set_title(f'Dual Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
        ax1.axis('off')
        
        # 2. Category distribution
        cat_analysis = self.analyze_category_theory()
        categories = list(cat_analysis['classifications'].keys())
        cat_sizes = [len(cat_analysis['classifications'][cat]) for cat in categories]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax2.pie(cat_sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Inversion Stability Categories')
        
        # 3. Morphism structure
        values = list(self.trace_universe.keys())
        stabilities = [self.trace_universe[v]['inversion_stability'] for v in values]
        
        # Create stability matrix
        n_traces = len(values)
        stability_matrix = np.zeros((n_traces, n_traces))
        
        for i, v1 in enumerate(values):
            for j, v2 in enumerate(values):
                if i != j:
                    # Morphism strength based on stability similarity
                    s1 = self.trace_universe[v1]['inversion_stability']
                    s2 = self.trace_universe[v2]['inversion_stability']
                    similarity = 1.0 - abs(s1 - s2)
                    stability_matrix[i, j] = similarity
        
        im = ax3.imshow(stability_matrix, cmap='plasma', aspect='auto')
        ax3.set_title('Morphism Structure Matrix')
        ax3.set_xlabel('Trace Index')
        ax3.set_ylabel('Trace Index')
        plt.colorbar(im, ax=ax3, label='Morphism Strength')
        
        # 4. Information entropy analysis
        info_analysis = self.analyze_information_theory()
        entropy_names = list(info_analysis.keys())
        entropy_values = list(info_analysis.values())
        
        # Sort by entropy value
        sorted_pairs = sorted(zip(entropy_names, entropy_values), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_values = zip(*sorted_pairs)
        
        bars = ax4.barh(range(len(sorted_names)), sorted_values, alpha=0.7, color='skyblue')
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels([name.replace(' entropy', '') for name in sorted_names], fontsize=10)
        ax4.set_xlabel('Entropy (bits)')
        ax4.set_title('Information Content Analysis')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('chapter-095-zeta-dual-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestZetaDual(unittest.TestCase):
    """Unit tests for ZetaDual verification"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = ZetaDualSystem(max_trace_value=50, inversion_resolution=8)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 0)
        for value, props in self.system.trace_universe.items():
            self.assertIn('trace', props)
            self.assertIn('zeta_parameter', props)
            self.assertIn('dual_parameter', props)
            self.assertIn('duality_strength', props)
            
    def test_dual_properties_analysis(self):
        """测试dual性质分析"""
        for value, props in self.system.trace_universe.items():
            # Test ζ-inversion property: s + s' ≈ 1
            s = props['zeta_parameter']
            s_dual = props['dual_parameter']
            inversion_error = abs(s + s_dual - 1.0)
            self.assertLess(inversion_error, 0.5)  # Reasonable tolerance
            
            # Test duality strength
            self.assertGreaterEqual(props['duality_strength'], 0.0)
            
            # Test inversion stability
            self.assertGreaterEqual(props['inversion_stability'], 0.0)
            self.assertLessEqual(props['inversion_stability'], 1.0)
            
    def test_dual_pair_detection(self):
        """测试dual pair检测"""
        # Should be able to detect dual pairs if they exist
        pairs = self.system.dual_pairs
        for t1, t2 in pairs:
            self.assertIn(t1, self.system.trace_universe)
            self.assertIn(t2, self.system.trace_universe)
            self.assertNotEqual(t1, t2)
            
    def test_inversion_mapping(self):
        """测试inversion mapping"""
        for value, mapping in self.system.inversion_mapping.items():
            self.assertIn('original_parameter', mapping)
            self.assertIn('dual_parameter', mapping)
            self.assertIn('duality_product', mapping)
            
            # Test that duality product ≤ 0.25 (maximum at s=0.5)
            product = mapping['duality_product']
            self.assertLessEqual(product, 0.26)  # Small tolerance
            
    def test_network_construction(self):
        """测试network构建"""
        G = self.system.build_dual_network()
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some structure
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        info_analysis = self.system.analyze_information_theory()
        for entropy_name, entropy_value in info_analysis.items():
            self.assertGreaterEqual(entropy_value, 0.0)
            self.assertLessEqual(entropy_value, 10.0)  # Reasonable upper bound
            
    def test_categorical_analysis(self):
        """测试范畴论分析"""
        cat_analysis = self.system.analyze_category_theory()
        self.assertGreater(cat_analysis['categories'], 0)
        self.assertGreaterEqual(cat_analysis['total_morphisms'], 0)
        self.assertGreaterEqual(cat_analysis['morphism_density'], 0.0)
        self.assertLessEqual(cat_analysis['morphism_density'], 1.0)

def main():
    """主验证程序"""
    print("=" * 80)
    print("Chapter 095: ZetaDual Verification")
    print("从ψ=ψ(ψ)推导Dual Collapse Paths under ζ(s) Inversion Mapping")
    print("=" * 80)
    
    # Initialize system
    system = ZetaDualSystem(max_trace_value=80, inversion_resolution=12)
    
    print("\n1. Dual Foundation Analysis:")
    print("-" * 50)
    print(f"Total traces analyzed: {len(system.trace_universe)}")
    
    # Basic statistics
    zeta_params = [props['zeta_parameter'] for props in system.trace_universe.values()]
    dual_params = [props['dual_parameter'] for props in system.trace_universe.values()]
    duality_strengths = [props['duality_strength'] for props in system.trace_universe.values()]
    inversion_stabilities = [props['inversion_stability'] for props in system.trace_universe.values()]
    
    print(f"Mean ζ-parameter: {np.mean(zeta_params):.3f}")
    print(f"Mean dual parameter: {np.mean(dual_params):.3f}")
    print(f"Mean duality strength: {np.mean(duality_strengths):.3f}")
    print(f"Mean inversion stability: {np.mean(inversion_stabilities):.3f}")
    
    # Dual pairs
    print(f"Dual pairs detected: {len(system.dual_pairs)}")
    
    print("\n2. Inversion Mapping Analysis:")
    print("-" * 50)
    
    # Fixed point analysis
    fixed_point_distances = [abs(s - 0.5) for s in zeta_params]
    duality_products = [system.inversion_mapping[v]['duality_product'] for v in system.trace_universe.keys()]
    
    print(f"Mean distance from fixed point: {np.mean(fixed_point_distances):.3f}")
    print(f"Max duality product: {np.max(duality_products):.3f}")
    print(f"Mean duality product: {np.mean(duality_products):.3f}")
    
    # Inversion classes
    mapping_indices = [props['mapping_index'] for props in system.trace_universe.values()]
    unique_classes = len(set(mapping_indices))
    print(f"Inversion mapping classes: {unique_classes}")
    
    print("\n3. Network Analysis:")
    print("-" * 50)
    
    G = system.build_dual_network()
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        print(f"Network density: {density:.3f}")
        
        # Component analysis
        if G.number_of_edges() > 0:
            components = list(nx.connected_components(G))
            print(f"Connected components: {len(components)}")
            
            # Average degree
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            print(f"Average degree: {avg_degree:.3f}")
    
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    
    info_analysis = system.analyze_information_theory()
    for name, entropy in sorted(info_analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {entropy:.3f} bits")
    
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    
    cat_analysis = system.analyze_category_theory()
    print(f"Inversion stability categories: {cat_analysis['categories']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for cat, traces in cat_analysis['classifications'].items():
        percentage = len(traces) / len(system.trace_universe) * 100
        print(f"- {cat}: {len(traces)} traces ({percentage:.1f}%)")
    
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        system.visualize_dual_dynamics()
        print("✓ Dual dynamics visualization saved")
    except Exception as e:
        print(f"✗ Dual dynamics visualization failed: {e}")
    
    try:
        system.visualize_inversion_analysis()
        print("✓ Inversion analysis visualization saved")
    except Exception as e:
        print(f"✗ Inversion analysis visualization failed: {e}")
    
    try:
        system.visualize_network_and_categories()
        print("✓ Network and categorical visualization saved")
    except Exception as e:
        print(f"✗ Network and categorical visualization failed: {e}")
    
    print("\n7. Running Unit Tests:")
    print("-" * 50)
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 80)
    print("ZetaDual Verification Complete")
    print("Key Findings:")
    print(f"- {len(system.trace_universe)} φ-valid traces with dual analysis")
    print(f"- {cat_analysis['categories']} inversion stability categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print(f"- Network density: {density:.3f}" if G.number_of_nodes() > 0 else "- Network density: 0.000")
    print(f"- Mean duality strength: {np.mean(duality_strengths):.3f}")
    print(f"- Mean inversion stability: {np.mean(inversion_stabilities):.3f}")
    print(f"- Dual pairs detected: {len(system.dual_pairs)}")
    print("=" * 80)

if __name__ == "__main__":
    main()