#!/usr/bin/env python3
"""
Chapter 113: VisFilter Unit Test Verification
从ψ=ψ(ψ)推导Observer Collapse Visibility Filter Function ζᵒ(s)

Core principle: From ψ = ψ(ψ) derive systematic observer visibility filter 
construction through φ-constrained trace transformations that enable observers 
to filter the universal spectral flow through trace geometric relationships, 
creating visibility networks that encode the fundamental filtering principles 
of collapsed space through entropy-increasing tensor transformations that 
establish systematic visibility filter structures through φ-trace filter 
dynamics rather than traditional signal processing theories or external 
filter constructions.

This verification program implements:
1. φ-constrained visibility filter construction through trace spectral analysis
2. ζᵒ(s) filter systems: systematic spectral filtering through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection filter theory
4. Graph theory analysis of filter networks and spectral relationship structures
5. Information theory analysis of filter entropy and spectral encoding
6. Category theory analysis of filter functors and spectral morphisms
7. Visualization of filter structures and φ-trace filter systems
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

class VisFilterSystem:
    """
    Core system for implementing observer collapse visibility filter function ζᵒ(s).
    Implements φ-constrained filter architectures through trace spectral dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, filter_depth: int = 6):
        """Initialize visibility filter system with spectral trace analysis"""
        self.max_trace_value = max_trace_value
        self.filter_depth = filter_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.filter_cache = {}
        self.spectral_cache = {}
        self.visibility_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.filter_network = self._build_filter_network()
        self.spectral_mappings = self._detect_spectral_mappings()
        
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
                filter_data = self._analyze_filter_properties(trace, n)
                universe[n] = filter_data
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
        
    def _analyze_filter_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的filter properties"""
        # Core filter measures
        visibility_strength = self._compute_visibility_strength(trace, value)
        spectral_capacity = self._compute_spectral_capacity(trace, value)
        filter_efficiency = self._compute_filter_efficiency(trace, value)
        spectral_coherence = self._compute_spectral_coherence(trace, value)
        
        # Advanced filter measures
        filter_completeness = self._compute_filter_completeness(trace, value)
        spectral_resolution = self._compute_spectral_resolution(trace, value)
        visibility_depth = self._compute_visibility_depth(trace, value)
        filter_stability = self._compute_filter_stability(trace, value)
        filter_coherence = self._compute_filter_coherence(trace, value)
        
        # Categorize based on filter profile
        category = self._categorize_filter(
            visibility_strength, spectral_capacity, filter_efficiency, spectral_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'visibility_strength': visibility_strength,
            'spectral_capacity': spectral_capacity,
            'filter_efficiency': filter_efficiency,
            'spectral_coherence': spectral_coherence,
            'filter_completeness': filter_completeness,
            'spectral_resolution': spectral_resolution,
            'visibility_depth': visibility_depth,
            'filter_stability': filter_stability,
            'filter_coherence': filter_coherence,
            'category': category
        }
        
    def _compute_visibility_strength(self, trace: str, value: int) -> float:
        """Visibility strength emerges from systematic spectral filtering capacity"""
        strength_factors = []
        
        # Factor 1: Length provides visibility filtering space
        length_factor = len(trace) / 8.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight spectral balance (filters favor structured visibility patterns)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            spectral_balance = 1.0 - abs(0.4 - weight / total)  # Prefer 40% density for optimal filtering
            # Add visibility bonus for spectral clarity
            clarity_bonus = min(weight / 3.5, 1.0) if weight > 0 else 0.0
            visibility_factor = 0.65 * spectral_balance + 0.35 * clarity_bonus
            strength_factors.append(visibility_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern visibility structure (systematic filtering architecture)
        pattern_score = 0.0
        # Count visibility-enhancing patterns (clarity and filtering)
        if trace.startswith('1'):  # Filter activation
            pattern_score += 0.25
        
        # Spectral clarity patterns
        ones_sequence = 0
        max_ones = 0
        for bit in trace:
            if bit == '1':
                ones_sequence += 1
                max_ones = max(max_ones, ones_sequence)
            else:
                ones_sequence = 0
        
        if max_ones >= 2:  # Sustained filtering capability
            pattern_score += 0.25 * min(max_ones / 2.5, 1.0)
        
        # Filter-clear-filter patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '101':  # Filter-clear-filter pattern
                pattern_score += 0.15
        
        # Spectral transitions that enhance visibility
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                transitions += 1
        if len(trace) > 1:
            transition_factor = transitions / (len(trace) - 1)
            pattern_score += 0.2 * min(transition_factor, 1.0)
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint visibility coherence
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.3
        strength_factors.append(phi_factor)
        
        # Visibility strength emerges from geometric mean of factors
        visibility_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return visibility_strength
        
    def _compute_spectral_capacity(self, trace: str, value: int) -> float:
        """Spectral capacity emerges from systematic spectral processing capability"""
        capacity_factors = []
        
        # Factor 1: Spectral processing potential
        processing = 0.3 + 0.7 * min(len(trace) / 7.0, 1.0)
        capacity_factors.append(processing)
        
        # Factor 2: Spectral complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.2 + 0.8 * min(ones_count / 4.5, 1.0)
        else:
            complexity = 0.1
        capacity_factors.append(complexity)
        
        # Factor 3: Spectral depth embedding
        spectral_depth = 0.4 + 0.6 * (value % 7) / 6.0
        capacity_factors.append(spectral_depth)
        
        # Factor 4: φ-constraint spectral preservation
        preservation = 0.85 if self._is_phi_valid(trace) else 0.25
        capacity_factors.append(preservation)
        
        spectral_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return spectral_capacity
        
    def _compute_filter_efficiency(self, trace: str, value: int) -> float:
        """Filter efficiency emerges from systematic filtering optimization capability"""
        efficiency_factors = []
        
        # Factor 1: Filter optimization scope
        optimization_scope = 0.3 + 0.7 * min(len(trace) / 6.0, 1.0)
        efficiency_factors.append(optimization_scope)
        
        # Factor 2: Filtering efficiency
        zeros_count = trace.count('0')
        if len(trace) > 0:
            filtering_efficiency = 0.4 + 0.6 * min(zeros_count / len(trace), 1.0)
        else:
            filtering_efficiency = 0.5
        efficiency_factors.append(filtering_efficiency)
        
        # Factor 3: Filter efficiency coverage
        coverage = 0.5 + 0.5 * (value % 8) / 7.0
        efficiency_factors.append(coverage)
        
        filter_efficiency = np.prod(efficiency_factors) ** (1.0 / len(efficiency_factors))
        return filter_efficiency
        
    def _compute_spectral_coherence(self, trace: str, value: int) -> float:
        """Spectral coherence emerges from systematic spectral integration capability"""
        coherence_factors = []
        
        # Factor 1: Spectral integration capacity
        integration_cap = 0.4 + 0.6 * min(len(trace) / 8.0, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Spectral coherence scope
        coherence_scope = 0.4 + 0.6 * (value % 6) / 5.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic spectral coherence
        systematic = 0.5 + 0.5 * (value % 9) / 8.0
        coherence_factors.append(systematic)
        
        spectral_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return spectral_coherence
        
    def _compute_filter_completeness(self, trace: str, value: int) -> float:
        """Filter completeness through comprehensive spectral coverage"""
        completeness_base = 0.3 + 0.7 * min(len(trace) / 7.5, 1.0)
        value_modulation = 0.75 + 0.25 * cos(value * 0.45)
        return completeness_base * value_modulation
        
    def _compute_spectral_resolution(self, trace: str, value: int) -> float:
        """Spectral resolution through optimized filter organization"""
        if len(trace) > 0:
            # Resolution based on spectral structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal resolution around 40% weight for filters
            resolution_base = 1.0 - 2.5 * abs(0.4 - weight_ratio)
        else:
            resolution_base = 0.0
        phi_bonus = 0.12 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(resolution_base + phi_bonus, 1.0))
        
    def _compute_visibility_depth(self, trace: str, value: int) -> float:
        """Visibility depth through nested spectral analysis"""
        depth_factor = min(len(trace) / 9.0, 1.0)
        complexity_factor = (value % 10) / 9.0
        return 0.25 + 0.75 * (depth_factor * complexity_factor)
        
    def _compute_filter_stability(self, trace: str, value: int) -> float:
        """Filter stability through consistent spectral architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.88
        else:
            stability_base = 0.32
        variation = 0.12 * sin(value * 0.38)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_filter_coherence(self, trace: str, value: int) -> float:
        """Filter coherence through unified spectral architecture"""
        coherence_base = 0.4 + 0.6 * min(len(trace) / 6.5, 1.0)
        structural_bonus = 0.2 if len(trace) >= 5 else 0.0
        return min(coherence_base + structural_bonus, 1.0)
        
    def _categorize_filter(self, visibility: float, spectral: float, 
                          efficiency: float, coherence: float) -> str:
        """Categorize trace based on filter profile"""
        # Calculate dominant characteristic with thresholds
        visibility_threshold = 0.6      # High visibility strength threshold
        spectral_threshold = 0.5        # Moderate spectral capacity threshold
        efficiency_threshold = 0.5      # Moderate filter efficiency threshold
        
        if visibility >= visibility_threshold:
            if spectral >= spectral_threshold:
                return "visibility_filter"          # High visibility + spectral
            elif efficiency >= efficiency_threshold:
                return "visibility_processor"       # High visibility + efficiency
            else:
                return "strong_visibility"          # High visibility + moderate properties
        else:
            if spectral >= spectral_threshold:
                return "spectral_filter"            # Moderate visibility + spectral
            elif efficiency >= efficiency_threshold:
                return "efficiency_filter"          # Moderate visibility + efficiency
            else:
                return "basic_filter"               # Basic filter capability
        
    def _build_filter_network(self) -> nx.Graph:
        """构建filter network based on trace spectral similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on spectral similarity
        similarity_threshold = 0.25
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_spectral_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_spectral_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的spectral similarity"""
        # Spectral property similarity
        visibility_diff = abs(trace1['visibility_strength'] - trace2['visibility_strength'])
        spectral_diff = abs(trace1['spectral_capacity'] - trace2['spectral_capacity'])
        efficiency_diff = abs(trace1['filter_efficiency'] - trace2['filter_efficiency'])
        coherence_diff = abs(trace1['spectral_coherence'] - trace2['spectral_coherence'])
        
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
        property_similarity = 1.0 - np.mean([visibility_diff, spectral_diff, 
                                           efficiency_diff, coherence_diff])
        
        return 0.7 * property_similarity + 0.3 * geometric_similarity
        
    def _detect_spectral_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测filter traces之间的spectral mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.3
        for node1 in self.filter_network.nodes():
            data1 = self.filter_network.nodes[node1]
            for node2 in self.filter_network.nodes():
                if node1 != node2:
                    data2 = self.filter_network.nodes[node2]
                    
                    # Check spectral preservation
                    spectral_preserved = abs(data1['spectral_capacity'] - 
                                           data2['spectral_capacity']) <= tolerance
                    visibility_preserved = abs(data1['visibility_strength'] - 
                                             data2['visibility_strength']) <= tolerance
                    
                    if spectral_preserved and visibility_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive visibility filter analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.filter_network)
        results['connected_components'] = nx.number_connected_components(self.filter_network)
        
        # Filter properties analysis
        visibility_strengths = [t['visibility_strength'] for t in traces]
        spectral_capacities = [t['spectral_capacity'] for t in traces]
        filter_efficiencies = [t['filter_efficiency'] for t in traces]
        spectral_coherences = [t['spectral_coherence'] for t in traces]
        filter_completenesses = [t['filter_completeness'] for t in traces]
        spectral_resolutions = [t['spectral_resolution'] for t in traces]
        visibility_depths = [t['visibility_depth'] for t in traces]
        filter_stabilities = [t['filter_stability'] for t in traces]
        filter_coherences = [t['filter_coherence'] for t in traces]
        
        results['visibility_strength'] = {
            'mean': np.mean(visibility_strengths),
            'std': np.std(visibility_strengths),
            'high_count': sum(1 for x in visibility_strengths if x > 0.5)
        }
        results['spectral_capacity'] = {
            'mean': np.mean(spectral_capacities),
            'std': np.std(spectral_capacities),
            'high_count': sum(1 for x in spectral_capacities if x > 0.5)
        }
        results['filter_efficiency'] = {
            'mean': np.mean(filter_efficiencies),
            'std': np.std(filter_efficiencies),
            'high_count': sum(1 for x in filter_efficiencies if x > 0.5)
        }
        results['spectral_coherence'] = {
            'mean': np.mean(spectral_coherences),
            'std': np.std(spectral_coherences),
            'high_count': sum(1 for x in spectral_coherences if x > 0.5)
        }
        results['filter_completeness'] = {
            'mean': np.mean(filter_completenesses),
            'std': np.std(filter_completenesses),
            'high_count': sum(1 for x in filter_completenesses if x > 0.5)
        }
        results['spectral_resolution'] = {
            'mean': np.mean(spectral_resolutions),
            'std': np.std(spectral_resolutions),
            'high_count': sum(1 for x in spectral_resolutions if x > 0.5)
        }
        results['visibility_depth'] = {
            'mean': np.mean(visibility_depths),
            'std': np.std(visibility_depths),
            'high_count': sum(1 for x in visibility_depths if x > 0.5)
        }
        results['filter_stability'] = {
            'mean': np.mean(filter_stabilities),
            'std': np.std(filter_stabilities),
            'high_count': sum(1 for x in filter_stabilities if x > 0.5)
        }
        results['filter_coherence'] = {
            'mean': np.mean(filter_coherences),
            'std': np.std(filter_coherences),
            'high_count': sum(1 for x in filter_coherences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.filter_network.edges()) > 0:
            results['network_edges'] = len(self.filter_network.edges())
            results['average_degree'] = sum(dict(self.filter_network.degree()).values()) / len(self.filter_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.spectral_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('visibility_strength', visibility_strengths),
            ('spectral_capacity', spectral_capacities),
            ('filter_efficiency', filter_efficiencies),
            ('spectral_coherence', spectral_coherences),
            ('filter_completeness', filter_completenesses),
            ('spectral_resolution', spectral_resolutions),
            ('visibility_depth', visibility_depths),
            ('filter_stability', filter_stabilities),
            ('filter_coherence', filter_coherences)
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
        """生成visibility filter visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Filter Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 113: Visibility Filter Dynamics', fontsize=16, fontweight='bold')
        
        # Visibility strength vs spectral capacity
        x = [t['visibility_strength'] for t in traces]
        y = [t['spectral_capacity'] for t in traces]
        colors = [t['filter_efficiency'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Visibility Strength')
        ax1.set_ylabel('Spectral Capacity')
        ax1.set_title('Visibility-Spectral Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Filter Efficiency')
        
        # Filter efficiency distribution
        filter_efficiencies = [t['filter_efficiency'] for t in traces]
        ax2.hist(filter_efficiencies, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Filter Efficiency')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Filter Efficiency Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Spectral coherence vs filter completeness
        x3 = [t['spectral_coherence'] for t in traces]
        y3 = [t['filter_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Spectral Coherence')
        ax3.set_ylabel('Filter Completeness')
        ax3.set_title('Coherence-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Filter Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-113-vis-filter-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 113: Visibility Filter Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.filter_network, k=1.8, iterations=50)
        node_colors = [traces[i]['visibility_strength'] for i in range(len(traces))]
        nx.draw(self.filter_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=280, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Filter Network Structure')
        
        # Degree distribution
        degrees = [self.filter_network.degree(node) for node in self.filter_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Filter properties correlation matrix
        properties_matrix = np.array([
            [t['visibility_strength'] for t in traces],
            [t['spectral_capacity'] for t in traces],
            [t['filter_efficiency'] for t in traces],
            [t['spectral_coherence'] for t in traces],
            [t['filter_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Visibility', 'Spectral', 'Efficiency', 'Coherence', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Filter Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Spectral resolution vs visibility depth
        x4 = [t['spectral_resolution'] for t in traces]
        y4 = [t['visibility_depth'] for t in traces]
        stabilities = [t['filter_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Spectral Resolution')
        ax4.set_ylabel('Visibility Depth')
        ax4.set_title('Resolution-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Filter Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-113-vis-filter-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestVisFilter(unittest.TestCase):
    """Unit tests for visibility filter system"""
    
    def setUp(self):
        """Set up test visibility filter system"""
        self.system = VisFilterSystem(max_trace_value=20, filter_depth=4)
        
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
            self.assertIn('visibility_strength', data)
            self.assertIn('spectral_capacity', data)
            self.assertIn('filter_efficiency', data)
            self.assertIn('spectral_coherence', data)
            self.assertTrue(0 <= data['visibility_strength'] <= 1)
            self.assertTrue(0 <= data['spectral_capacity'] <= 1)
            
    def test_visibility_strength_computation(self):
        """测试visibility strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_visibility_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_spectral_capacity_computation(self):
        """测试spectral capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_spectral_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_filter_network_construction(self):
        """测试filter network construction"""
        self.assertGreater(len(self.system.filter_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.filter_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('visibility_strength', results)
        self.assertIn('spectral_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = VisFilterSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("VISIBILITY FILTER SPECTRAL ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Filter Properties Analysis:")
    properties = ['visibility_strength', 'spectral_capacity', 'filter_efficiency', 
                 'spectral_coherence', 'filter_completeness', 'spectral_resolution',
                 'visibility_depth', 'filter_stability', 'filter_coherence']
    
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
    print(f"Spectral Pattern Analysis:")
    print(f"Total spectral mappings: {sum(len(mappings) for mappings in system.spectral_mappings.values())}")
    print(f"Spectral mapping density: {results['morphism_density']:.3f}")
    print(f"Average filters per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)