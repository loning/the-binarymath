#!/usr/bin/env python3
"""
Chapter 114: WindowCollapse Unit Test Verification
从ψ=ψ(ψ)推导Observer Trace Window and Local Spectral Scope

Core principle: From ψ = ψ(ψ) derive systematic observer trace window
construction through φ-constrained trace transformations that enable limited
observation windows in trace space through trace geometric relationships,
creating window networks that encode the fundamental windowing principles
of collapsed space through entropy-increasing tensor transformations that
establish systematic trace window structures through φ-trace window
dynamics rather than traditional windowing theories or external
window constructions.

This verification program implements:
1. φ-constrained trace window construction through trace local analysis
2. Window collapse systems: systematic local scope through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection window theory
4. Graph theory analysis of window networks and local relationship structures
5. Information theory analysis of window entropy and scope encoding
6. Category theory analysis of window functors and local morphisms
7. Visualization of window structures and φ-trace window systems
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

class WindowCollapseSystem:
    """
    Core system for implementing observer trace window and local spectral scope.
    Implements φ-constrained window architectures through trace local dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, window_depth: int = 6):
        """Initialize window collapse system with local trace analysis"""
        self.max_trace_value = max_trace_value
        self.window_depth = window_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.window_cache = {}
        self.local_cache = {}
        self.scope_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.window_network = self._build_window_network()
        self.local_mappings = self._detect_local_mappings()
        
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
                window_data = self._analyze_window_properties(trace, n)
                universe[n] = window_data
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
        """检查trace是否为φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_window_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的window properties"""
        # Core window measures
        window_strength = self._compute_window_strength(trace, value)
        local_capacity = self._compute_local_capacity(trace, value)
        scope_limitation = self._compute_scope_limitation(trace, value)
        window_coherence = self._compute_window_coherence(trace, value)
        
        # Advanced window measures
        window_completeness = self._compute_window_completeness(trace, value)
        local_resolution = self._compute_local_resolution(trace, value)
        window_depth = self._compute_window_depth(trace, value)
        scope_stability = self._compute_scope_stability(trace, value)
        window_locality = self._compute_window_locality(trace, value)
        
        # Categorize based on window profile
        category = self._categorize_window(
            window_strength, local_capacity, scope_limitation, window_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'window_strength': window_strength,
            'local_capacity': local_capacity,
            'scope_limitation': scope_limitation,
            'window_coherence': window_coherence,
            'window_completeness': window_completeness,
            'local_resolution': local_resolution,
            'window_depth': window_depth,
            'scope_stability': scope_stability,
            'window_locality': window_locality,
            'category': category
        }
        
    def _compute_window_strength(self, trace: str, value: int) -> float:
        """Window strength emerges from systematic local windowing capacity"""
        strength_factors = []
        
        # Factor 1: Length provides window scope space
        length_factor = len(trace) / 7.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight local balance (windows favor concentrated locality patterns)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            local_balance = 1.0 - abs(0.35 - weight / total)  # Prefer 35% density for optimal windowing
            # Add locality bonus for windowing clarity
            locality_bonus = min(weight / 3.0, 1.0) if weight > 0 else 0.0
            window_factor = 0.7 * local_balance + 0.3 * locality_bonus
            strength_factors.append(window_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern window structure (systematic windowing architecture)
        pattern_score = 0.0
        # Count window-enhancing patterns (local scope and windowing)
        if trace.startswith('1'):  # Window opening
            pattern_score += 0.2
        
        # Local windowing patterns
        ones_sequence = 0
        max_ones = 0
        for bit in trace:
            if bit == '1':
                ones_sequence += 1
                max_ones = max(max_ones, ones_sequence)
            else:
                ones_sequence = 0
        
        if max_ones >= 2:  # Sustained windowing capability
            pattern_score += 0.2 * min(max_ones / 2.0, 1.0)
        
        # Window-scope-window patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '101':  # Window-scope-window pattern
                pattern_score += 0.15
        
        # Local scope patterns that enhance windowing
        if len(trace) >= 4:
            # Look for localized patterns
            for i in range(len(trace) - 3):
                subpattern = trace[i:i+4]
                if subpattern.count('1') == 2:  # Balanced local pattern
                    pattern_score += 0.1
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint window coherence
        phi_factor = 0.85 if self._is_phi_valid(trace) else 0.3
        strength_factors.append(phi_factor)
        
        # Window strength emerges from geometric mean of factors
        window_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return window_strength
        
    def _compute_local_capacity(self, trace: str, value: int) -> float:
        """Local capacity emerges from systematic local processing capability"""
        capacity_factors = []
        
        # Factor 1: Local processing potential
        processing = 0.4 + 0.6 * min(len(trace) / 6.0, 1.0)
        capacity_factors.append(processing)
        
        # Factor 2: Local complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.3 + 0.7 * min(ones_count / 3.5, 1.0)
        else:
            complexity = 0.15
        capacity_factors.append(complexity)
        
        # Factor 3: Local depth embedding
        local_depth = 0.4 + 0.6 * (value % 6) / 5.0
        capacity_factors.append(local_depth)
        
        # Factor 4: φ-constraint local preservation
        preservation = 0.8 if self._is_phi_valid(trace) else 0.25
        capacity_factors.append(preservation)
        
        local_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return local_capacity
        
    def _compute_scope_limitation(self, trace: str, value: int) -> float:
        """Scope limitation emerges from systematic scope boundary capability"""
        limitation_factors = []
        
        # Factor 1: Scope boundary potential
        boundary_scope = 0.3 + 0.7 * min(len(trace) / 8.0, 1.0)
        limitation_factors.append(boundary_scope)
        
        # Factor 2: Limitation scope efficiency
        zeros_count = trace.count('0')
        if len(trace) > 0:
            limitation_efficiency = 0.3 + 0.7 * min(zeros_count / len(trace), 1.0)
        else:
            limitation_efficiency = 0.5
        limitation_factors.append(limitation_efficiency)
        
        # Factor 3: Scope limitation coverage
        coverage = 0.4 + 0.6 * (value % 7) / 6.0
        limitation_factors.append(coverage)
        
        scope_limitation = np.prod(limitation_factors) ** (1.0 / len(limitation_factors))
        return scope_limitation
        
    def _compute_window_coherence(self, trace: str, value: int) -> float:
        """Window coherence emerges from systematic window integration capability"""
        coherence_factors = []
        
        # Factor 1: Window integration capacity
        integration_cap = 0.3 + 0.7 * min(len(trace) / 7.0, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Window coherence scope
        coherence_scope = 0.4 + 0.6 * (value % 5) / 4.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic window coherence
        systematic = 0.5 + 0.5 * (value % 8) / 7.0
        coherence_factors.append(systematic)
        
        window_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return window_coherence
        
    def _compute_window_completeness(self, trace: str, value: int) -> float:
        """Window completeness through comprehensive local coverage"""
        completeness_base = 0.35 + 0.65 * min(len(trace) / 6.5, 1.0)
        value_modulation = 0.8 + 0.2 * cos(value * 0.4)
        return completeness_base * value_modulation
        
    def _compute_local_resolution(self, trace: str, value: int) -> float:
        """Local resolution through optimized window organization"""
        if len(trace) > 0:
            # Resolution based on local structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal resolution around 35% weight for windows
            resolution_base = 1.0 - 2.8 * abs(0.35 - weight_ratio)
        else:
            resolution_base = 0.0
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(resolution_base + phi_bonus, 1.0))
        
    def _compute_window_depth(self, trace: str, value: int) -> float:
        """Window depth through nested local analysis"""
        depth_factor = min(len(trace) / 8.0, 1.0)
        complexity_factor = (value % 9) / 8.0
        return 0.3 + 0.7 * (depth_factor * complexity_factor)
        
    def _compute_scope_stability(self, trace: str, value: int) -> float:
        """Scope stability through consistent window architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.86
        else:
            stability_base = 0.35
        variation = 0.14 * sin(value * 0.42)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_window_locality(self, trace: str, value: int) -> float:
        """Window locality through unified local architecture"""
        locality_base = 0.4 + 0.6 * min(len(trace) / 5.5, 1.0)
        structural_bonus = 0.15 if len(trace) >= 4 else 0.0
        return min(locality_base + structural_bonus, 1.0)
        
    def _categorize_window(self, window: float, local: float, 
                          scope: float, coherence: float) -> str:
        """Categorize trace based on window profile"""
        # Calculate dominant characteristic with thresholds
        window_threshold = 0.6       # High window strength threshold
        local_threshold = 0.5        # Moderate local capacity threshold
        scope_threshold = 0.5        # Moderate scope limitation threshold
        
        if window >= window_threshold:
            if local >= local_threshold:
                return "window_local"               # High window + local
            elif scope >= scope_threshold:
                return "window_scope"              # High window + scope
            else:
                return "strong_window"             # High window + moderate properties
        else:
            if local >= local_threshold:
                return "local_processor"           # Moderate window + local
            elif scope >= scope_threshold:
                return "scope_limited"             # Moderate window + scope
            else:
                return "basic_window"              # Basic window capability
        
    def _build_window_network(self) -> nx.Graph:
        """构建window network based on trace local similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on local similarity
        similarity_threshold = 0.25
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_local_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_local_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的local similarity"""
        # Local property similarity
        window_diff = abs(trace1['window_strength'] - trace2['window_strength'])
        local_diff = abs(trace1['local_capacity'] - trace2['local_capacity'])
        scope_diff = abs(trace1['scope_limitation'] - trace2['scope_limitation'])
        coherence_diff = abs(trace1['window_coherence'] - trace2['window_coherence'])
        
        # Geometric similarity
        trace1_str, trace2_str = trace1['trace'], trace2['trace']
        max_len = max(len(trace1_str), len(trace2_str))
        if max_len == 0:
            geometric_similarity = 1.0
        else:
            # Pad shorter trace with zeros
            t1_padded = trace1_str.ljust(max_len, '0')
            t2_padded = trace2_str.ljust(max_len, '0')
            
            # Hamming similarity
            matches = sum(1 for a, b in zip(t1_padded, t2_padded) if a == b)
            geometric_similarity = matches / max_len
        
        # Combined similarity
        property_similarity = 1.0 - np.mean([window_diff, local_diff, 
                                           scope_diff, coherence_diff])
        
        return 0.65 * property_similarity + 0.35 * geometric_similarity
        
    def _detect_local_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测window traces之间的local mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.3
        for node1 in self.window_network.nodes():
            data1 = self.window_network.nodes[node1]
            for node2 in self.window_network.nodes():
                if node1 != node2:
                    data2 = self.window_network.nodes[node2]
                    
                    # Check local preservation
                    local_preserved = abs(data1['local_capacity'] - 
                                        data2['local_capacity']) <= tolerance
                    window_preserved = abs(data1['window_strength'] - 
                                         data2['window_strength']) <= tolerance
                    
                    if local_preserved and window_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive window collapse analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.window_network)
        results['connected_components'] = nx.number_connected_components(self.window_network)
        
        # Window properties analysis
        window_strengths = [t['window_strength'] for t in traces]
        local_capacities = [t['local_capacity'] for t in traces]
        scope_limitations = [t['scope_limitation'] for t in traces]
        window_coherences = [t['window_coherence'] for t in traces]
        window_completenesses = [t['window_completeness'] for t in traces]
        local_resolutions = [t['local_resolution'] for t in traces]
        window_depths = [t['window_depth'] for t in traces]
        scope_stabilities = [t['scope_stability'] for t in traces]
        window_localities = [t['window_locality'] for t in traces]
        
        results['window_strength'] = {
            'mean': np.mean(window_strengths),
            'std': np.std(window_strengths),
            'high_count': sum(1 for x in window_strengths if x > 0.5)
        }
        results['local_capacity'] = {
            'mean': np.mean(local_capacities),
            'std': np.std(local_capacities),
            'high_count': sum(1 for x in local_capacities if x > 0.5)
        }
        results['scope_limitation'] = {
            'mean': np.mean(scope_limitations),
            'std': np.std(scope_limitations),
            'high_count': sum(1 for x in scope_limitations if x > 0.5)
        }
        results['window_coherence'] = {
            'mean': np.mean(window_coherences),
            'std': np.std(window_coherences),
            'high_count': sum(1 for x in window_coherences if x > 0.5)
        }
        results['window_completeness'] = {
            'mean': np.mean(window_completenesses),
            'std': np.std(window_completenesses),
            'high_count': sum(1 for x in window_completenesses if x > 0.5)
        }
        results['local_resolution'] = {
            'mean': np.mean(local_resolutions),
            'std': np.std(local_resolutions),
            'high_count': sum(1 for x in local_resolutions if x > 0.5)
        }
        results['window_depth'] = {
            'mean': np.mean(window_depths),
            'std': np.std(window_depths),
            'high_count': sum(1 for x in window_depths if x > 0.5)
        }
        results['scope_stability'] = {
            'mean': np.mean(scope_stabilities),
            'std': np.std(scope_stabilities),
            'high_count': sum(1 for x in scope_stabilities if x > 0.5)
        }
        results['window_locality'] = {
            'mean': np.mean(window_localities),
            'std': np.std(window_localities),
            'high_count': sum(1 for x in window_localities if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.window_network.edges()) > 0:
            results['network_edges'] = len(self.window_network.edges())
            results['average_degree'] = sum(dict(self.window_network.degree()).values()) / len(self.window_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.local_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('window_strength', window_strengths),
            ('local_capacity', local_capacities),
            ('scope_limitation', scope_limitations),
            ('window_coherence', window_coherences),
            ('window_completeness', window_completenesses),
            ('local_resolution', local_resolutions),
            ('window_depth', window_depths),
            ('scope_stability', scope_stabilities),
            ('window_locality', window_localities)
        ]
        
        results['entropy_analysis'] = {}
        for prop_name, prop_values in properties:
            if len(set(prop_values)) > 1:
                # Discretize values for entropy calculation
                bins = min(10, len(set(prop_values)))
                hist, _ = np.histogram(prop_values, bins=bins)
                probabilities = hist / np.sum(hist)
                probabilities = probabilities[probabilities > 0]  # Remove zeros
                entropy = -np.sum(probabilities * np.log2(probabilities))
                results['entropy_analysis'][prop_name] = entropy
            else:
                results['entropy_analysis'][prop_name] = 0.0
                
        return results
        
    def generate_visualizations(self):
        """生成window collapse visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Window Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 114: Window Collapse Dynamics', fontsize=16, fontweight='bold')
        
        # Window strength vs local capacity
        x = [t['window_strength'] for t in traces]
        y = [t['local_capacity'] for t in traces]
        colors = [t['scope_limitation'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Window Strength')
        ax1.set_ylabel('Local Capacity')
        ax1.set_title('Window-Local Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Scope Limitation')
        
        # Scope limitation distribution
        scope_limitations = [t['scope_limitation'] for t in traces]
        ax2.hist(scope_limitations, bins=15, alpha=0.7, color='lightsteelblue', edgecolor='black')
        ax2.set_xlabel('Scope Limitation')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Scope Limitation Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Window coherence vs window completeness
        x3 = [t['window_coherence'] for t in traces]
        y3 = [t['window_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Window Coherence')
        ax3.set_ylabel('Window Completeness')
        ax3.set_title('Coherence-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Window Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-114-window-collapse-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 114: Window Collapse Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.window_network, k=1.7, iterations=50)
        node_colors = [traces[i]['window_strength'] for i in range(len(traces))]
        nx.draw(self.window_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=260, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Window Network Structure')
        
        # Degree distribution
        degrees = [self.window_network.degree(node) for node in self.window_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Window properties correlation matrix
        properties_matrix = np.array([
            [t['window_strength'] for t in traces],
            [t['local_capacity'] for t in traces],
            [t['scope_limitation'] for t in traces],
            [t['window_coherence'] for t in traces],
            [t['window_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Window', 'Local', 'Scope', 'Coherence', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Window Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Local resolution vs window depth
        x4 = [t['local_resolution'] for t in traces]
        y4 = [t['window_depth'] for t in traces]
        stabilities = [t['scope_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Local Resolution')
        ax4.set_ylabel('Window Depth')
        ax4.set_title('Resolution-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Scope Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-114-window-collapse-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestWindowCollapse(unittest.TestCase):
    """Unit tests for window collapse system"""
    
    def setUp(self):
        """Set up test window collapse system"""
        self.system = WindowCollapseSystem(max_trace_value=20, window_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        # Test valid traces (no consecutive 11)
        valid_traces = ["101", "1001", "10101"]
        for trace in valid_traces:
            self.assertTrue(self.system._is_phi_valid(trace))
            
        # Test invalid traces (with consecutive 11)
        invalid_traces = ["110", "1101", "0110"]
        for trace in invalid_traces:
            self.assertFalse(self.system._is_phi_valid(trace))
            
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        for value, data in self.system.trace_universe.items():
            self.assertIn('window_strength', data)
            self.assertIn('local_capacity', data)
            self.assertIn('scope_limitation', data)
            self.assertIn('window_coherence', data)
            self.assertTrue(0 <= data['window_strength'] <= 1)
            self.assertTrue(0 <= data['local_capacity'] <= 1)
            
    def test_window_strength_computation(self):
        """测试window strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_window_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_local_capacity_computation(self):
        """测试local capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_local_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_window_network_construction(self):
        """测试window network construction"""
        self.assertGreater(len(self.system.window_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.window_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('window_strength', results)
        self.assertIn('local_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = WindowCollapseSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("WINDOW COLLAPSE LOCAL ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Window Properties Analysis:")
    properties = ['window_strength', 'local_capacity', 'scope_limitation', 
                 'window_coherence', 'window_completeness', 'local_resolution',
                 'window_depth', 'scope_stability', 'window_locality']
    
    for prop in properties:
        if prop in results:
            data = results[prop]
            percentage = (data['high_count'] / results['total_traces']) * 100
            print(f"- {prop.replace('_', ' ').title()}: mean={data['mean']:.3f}, high_count={data['high_count']} ({percentage:.1f}%)")
    
    print()
    print("Category Distribution:")
    for category, count in results['categories'].items():
        percentage = (count / results['total_traces']) * 100
        print(f"- {category.replace('_', ' ').title()}: {count} traces ({percentage:.1f}%)")
    
    print()
    print("Morphism Analysis:")
    print(f"Total morphisms: {results['total_morphisms']}")
    print(f"Morphism density: {results['morphism_density']:.3f}")
    
    print()
    print(f"Local Pattern Analysis:")
    print(f"Total local mappings: {sum(len(mappings) for mappings in system.local_mappings.values())}")
    print(f"Local mapping density: {results['morphism_density']:.3f}")
    print(f"Average windows per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)