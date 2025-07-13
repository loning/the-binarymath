#!/usr/bin/env python3
"""
Chapter 115: ObsResolution Unit Test Verification
从ψ=ψ(ψ)推导Trace Differentiation Capacity of Observer Rank

Core principle: From ψ = ψ(ψ) derive systematic observer resolution
construction through φ-constrained trace transformations that enable resolution
limits based on observer tensor rank through trace geometric relationships,
creating resolution networks that encode the fundamental differentiation
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic resolution structures through φ-trace resolution
dynamics rather than traditional resolution theories or external
resolution constructions.

This verification program implements:
1. φ-constrained resolution construction through trace rank analysis
2. Observer resolution systems: systematic differentiation through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection resolution theory
4. Graph theory analysis of resolution networks and rank relationship structures
5. Information theory analysis of resolution entropy and differentiation encoding
6. Category theory analysis of resolution functors and rank morphisms
7. Visualization of resolution structures and φ-trace resolution systems
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

class ObsResolutionSystem:
    """
    Core system for implementing trace differentiation capacity of observer rank.
    Implements φ-constrained resolution architectures through trace rank dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, resolution_depth: int = 6):
        """Initialize observer resolution system with rank trace analysis"""
        self.max_trace_value = max_trace_value
        self.resolution_depth = resolution_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.resolution_cache = {}
        self.rank_cache = {}
        self.differentiation_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.resolution_network = self._build_resolution_network()
        self.rank_mappings = self._detect_rank_mappings()
        
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
                resolution_data = self._analyze_resolution_properties(trace, n)
                universe[n] = resolution_data
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
        
    def _analyze_resolution_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的resolution properties"""
        # Core resolution measures
        resolution_strength = self._compute_resolution_strength(trace, value)
        rank_capacity = self._compute_rank_capacity(trace, value)
        differentiation_power = self._compute_differentiation_power(trace, value)
        resolution_coherence = self._compute_resolution_coherence(trace, value)
        
        # Advanced resolution measures
        resolution_completeness = self._compute_resolution_completeness(trace, value)
        rank_hierarchy = self._compute_rank_hierarchy(trace, value)
        resolution_depth = self._compute_resolution_depth(trace, value)
        differentiation_stability = self._compute_differentiation_stability(trace, value)
        resolution_precision = self._compute_resolution_precision(trace, value)
        
        # Categorize based on resolution profile
        category = self._categorize_resolution(
            resolution_strength, rank_capacity, differentiation_power, resolution_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'resolution_strength': resolution_strength,
            'rank_capacity': rank_capacity,
            'differentiation_power': differentiation_power,
            'resolution_coherence': resolution_coherence,
            'resolution_completeness': resolution_completeness,
            'rank_hierarchy': rank_hierarchy,
            'resolution_depth': resolution_depth,
            'differentiation_stability': differentiation_stability,
            'resolution_precision': resolution_precision,
            'category': category
        }
        
    def _compute_resolution_strength(self, trace: str, value: int) -> float:
        """Resolution strength emerges from systematic differentiation capacity"""
        strength_factors = []
        
        # Factor 1: Length provides resolution space (minimum 5 for meaningful resolution)
        length_factor = len(trace) / 5.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight differentiation balance (resolution favors specific density for clarity)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            # Resolution optimal at 30% density (0.3) for maximum differentiation
            differentiation_balance = 1.0 - abs(0.3 - weight / total)
            # Add resolution bonus for differentiation patterns
            pattern_bonus = min(weight / 2.5, 1.0) if weight > 0 else 0.0
            resolution_factor = 0.65 * differentiation_balance + 0.35 * pattern_bonus
            strength_factors.append(resolution_factor)
        else:
            strength_factors.append(0.15)
        
        # Factor 3: Pattern resolution structure (systematic differentiation architecture)
        pattern_score = 0.0
        # Count resolution-enhancing patterns (differentiation and clarity)
        if trace.startswith('1'):  # Resolution initialization
            pattern_score += 0.2
        
        # Differentiation patterns
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions enable differentiation
                transitions += 1
        if len(trace) > 1:
            transition_rate = transitions / (len(trace) - 1)
            pattern_score += 0.3 * min(transition_rate * 2, 1.0)  # Value transitions
        
        # Resolution clarity patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '100':  # Clear differentiation pattern
                pattern_score += 0.15
            elif trace[i:i+3] == '001':  # Resolution boundary pattern
                pattern_score += 0.15
        
        # Rank-based patterns
        if len(trace) >= 3:
            rank_patterns = 0
            for i in range(len(trace) - 2):
                if trace[i:i+3] in ['101', '010']:  # Rank differentiation patterns
                    rank_patterns += 1
            pattern_score += 0.2 * min(rank_patterns / 2.0, 1.0)
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint resolution coherence  
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.2
        strength_factors.append(phi_factor)
        
        # Resolution strength emerges from geometric mean of factors
        resolution_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return resolution_strength
        
    def _compute_rank_capacity(self, trace: str, value: int) -> float:
        """Rank capacity emerges from systematic rank hierarchy capability"""
        capacity_factors = []
        
        # Factor 1: Rank structural potential
        structural = 0.25 + 0.75 * min(len(trace) / 7.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Rank complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.2 + 0.8 * min(ones_count / 4.0, 1.0)
        else:
            complexity = 0.1
        capacity_factors.append(complexity)
        
        # Factor 3: Rank hierarchy depth (value modulo 7 for 7-level hierarchy)
        hierarchy_depth = 0.3 + 0.7 * (value % 7) / 6.0
        capacity_factors.append(hierarchy_depth)
        
        # Factor 4: φ-constraint rank preservation
        preservation = 0.85 if self._is_phi_valid(trace) else 0.2
        capacity_factors.append(preservation)
        
        rank_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return rank_capacity
        
    def _compute_differentiation_power(self, trace: str, value: int) -> float:
        """Differentiation power emerges from systematic differentiation capability"""
        power_factors = []
        
        # Factor 1: Differentiation scope
        diff_scope = 0.2 + 0.8 * min(len(trace) / 6.0, 1.0)
        power_factors.append(diff_scope)
        
        # Factor 2: Differentiation efficiency
        zeros_count = trace.count('0')
        if len(trace) > 0:
            diff_efficiency = 0.3 + 0.7 * (1.0 - abs(0.7 - zeros_count / len(trace)))
        else:
            diff_efficiency = 0.5
        power_factors.append(diff_efficiency)
        
        # Factor 3: Differentiation coverage (value modulo 8 for 8-level coverage)
        coverage = 0.3 + 0.7 * (value % 8) / 7.0
        power_factors.append(coverage)
        
        differentiation_power = np.prod(power_factors) ** (1.0 / len(power_factors))
        return differentiation_power
        
    def _compute_resolution_coherence(self, trace: str, value: int) -> float:
        """Resolution coherence emerges from systematic resolution integration capability"""
        coherence_factors = []
        
        # Factor 1: Resolution integration capacity
        integration_cap = 0.35 + 0.65 * min(len(trace) / 5.5, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Resolution coherence scope (value modulo 5 for 5-level scope)
        coherence_scope = 0.4 + 0.6 * (value % 5) / 4.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic resolution coherence (value modulo 6 for 6-fold symmetry)
        systematic = 0.5 + 0.5 * (value % 6) / 5.0
        coherence_factors.append(systematic)
        
        resolution_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return resolution_coherence
        
    def _compute_resolution_completeness(self, trace: str, value: int) -> float:
        """Resolution completeness through comprehensive differentiation coverage"""
        completeness_base = 0.3 + 0.7 * min(len(trace) / 7.5, 1.0)
        value_modulation = 0.75 + 0.25 * cos(value * 0.35)
        return completeness_base * value_modulation
        
    def _compute_rank_hierarchy(self, trace: str, value: int) -> float:
        """Rank hierarchy through optimized rank organization"""
        if len(trace) > 0:
            # Hierarchy based on rank structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal hierarchy around 30% weight for ranks
            hierarchy_base = 1.0 - 3.3 * abs(0.3 - weight_ratio)
        else:
            hierarchy_base = 0.0
        phi_bonus = 0.15 if self._is_phi_valid(trace) else 0.0
        return max(0.05, min(hierarchy_base + phi_bonus, 1.0))
        
    def _compute_resolution_depth(self, trace: str, value: int) -> float:
        """Resolution depth through nested differentiation analysis"""
        depth_factor = min(len(trace) / 10.0, 1.0)
        complexity_factor = (value % 11) / 10.0
        return 0.2 + 0.8 * (depth_factor * complexity_factor)
        
    def _compute_differentiation_stability(self, trace: str, value: int) -> float:
        """Differentiation stability through consistent resolution architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.87
        else:
            stability_base = 0.3
        variation = 0.13 * sin(value * 0.37)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_resolution_precision(self, trace: str, value: int) -> float:
        """Resolution precision through unified differentiation architecture"""
        precision_base = 0.35 + 0.65 * min(len(trace) / 6.0, 1.0)
        structural_bonus = 0.2 if len(trace) >= 5 else 0.0
        return min(precision_base + structural_bonus, 1.0)
        
    def _categorize_resolution(self, resolution: float, rank: float, 
                              differentiation: float, coherence: float) -> str:
        """Categorize trace based on resolution profile"""
        # Calculate dominant characteristic with thresholds
        resolution_threshold = 0.5    # Moderate resolution strength threshold
        rank_threshold = 0.5          # Moderate rank capacity threshold
        diff_threshold = 0.5          # Moderate differentiation power threshold
        
        if resolution >= resolution_threshold:
            if rank >= rank_threshold:
                return "resolution_rank"            # High resolution + rank
            elif differentiation >= diff_threshold:
                return "resolution_diff"            # High resolution + differentiation
            else:
                return "high_resolution"            # High resolution + moderate properties
        else:
            if rank >= rank_threshold:
                return "rank_resolver"              # Moderate resolution + rank
            elif differentiation >= diff_threshold:
                return "diff_resolver"              # Moderate resolution + differentiation
            else:
                return "basic_resolution"           # Basic resolution capability
        
    def _build_resolution_network(self) -> nx.Graph:
        """构建resolution network based on trace rank similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on rank similarity
        similarity_threshold = 0.2
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_rank_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_rank_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的rank similarity"""
        # Rank property similarity
        resolution_diff = abs(trace1['resolution_strength'] - trace2['resolution_strength'])
        rank_diff = abs(trace1['rank_capacity'] - trace2['rank_capacity'])
        diff_diff = abs(trace1['differentiation_power'] - trace2['differentiation_power'])
        coherence_diff = abs(trace1['resolution_coherence'] - trace2['resolution_coherence'])
        
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
        property_similarity = 1.0 - np.mean([resolution_diff, rank_diff, 
                                           diff_diff, coherence_diff])
        
        return 0.7 * property_similarity + 0.3 * geometric_similarity
        
    def _detect_rank_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测resolution traces之间的rank mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.25
        for node1 in self.resolution_network.nodes():
            data1 = self.resolution_network.nodes[node1]
            for node2 in self.resolution_network.nodes():
                if node1 != node2:
                    data2 = self.resolution_network.nodes[node2]
                    
                    # Check rank preservation
                    rank_preserved = abs(data1['rank_capacity'] - 
                                       data2['rank_capacity']) <= tolerance
                    resolution_preserved = abs(data1['resolution_strength'] - 
                                             data2['resolution_strength']) <= tolerance
                    
                    if rank_preserved and resolution_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive observer resolution analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.resolution_network)
        results['connected_components'] = nx.number_connected_components(self.resolution_network)
        
        # Resolution properties analysis
        resolution_strengths = [t['resolution_strength'] for t in traces]
        rank_capacities = [t['rank_capacity'] for t in traces]
        differentiation_powers = [t['differentiation_power'] for t in traces]
        resolution_coherences = [t['resolution_coherence'] for t in traces]
        resolution_completenesses = [t['resolution_completeness'] for t in traces]
        rank_hierarchies = [t['rank_hierarchy'] for t in traces]
        resolution_depths = [t['resolution_depth'] for t in traces]
        differentiation_stabilities = [t['differentiation_stability'] for t in traces]
        resolution_precisions = [t['resolution_precision'] for t in traces]
        
        results['resolution_strength'] = {
            'mean': np.mean(resolution_strengths),
            'std': np.std(resolution_strengths),
            'high_count': sum(1 for x in resolution_strengths if x > 0.5)
        }
        results['rank_capacity'] = {
            'mean': np.mean(rank_capacities),
            'std': np.std(rank_capacities),
            'high_count': sum(1 for x in rank_capacities if x > 0.5)
        }
        results['differentiation_power'] = {
            'mean': np.mean(differentiation_powers),
            'std': np.std(differentiation_powers),
            'high_count': sum(1 for x in differentiation_powers if x > 0.5)
        }
        results['resolution_coherence'] = {
            'mean': np.mean(resolution_coherences),
            'std': np.std(resolution_coherences),
            'high_count': sum(1 for x in resolution_coherences if x > 0.5)
        }
        results['resolution_completeness'] = {
            'mean': np.mean(resolution_completenesses),
            'std': np.std(resolution_completenesses),
            'high_count': sum(1 for x in resolution_completenesses if x > 0.5)
        }
        results['rank_hierarchy'] = {
            'mean': np.mean(rank_hierarchies),
            'std': np.std(rank_hierarchies),
            'high_count': sum(1 for x in rank_hierarchies if x > 0.5)
        }
        results['resolution_depth'] = {
            'mean': np.mean(resolution_depths),
            'std': np.std(resolution_depths),
            'high_count': sum(1 for x in resolution_depths if x > 0.5)
        }
        results['differentiation_stability'] = {
            'mean': np.mean(differentiation_stabilities),
            'std': np.std(differentiation_stabilities),
            'high_count': sum(1 for x in differentiation_stabilities if x > 0.5)
        }
        results['resolution_precision'] = {
            'mean': np.mean(resolution_precisions),
            'std': np.std(resolution_precisions),
            'high_count': sum(1 for x in resolution_precisions if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.resolution_network.edges()) > 0:
            results['network_edges'] = len(self.resolution_network.edges())
            results['average_degree'] = sum(dict(self.resolution_network.degree()).values()) / len(self.resolution_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.rank_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('resolution_strength', resolution_strengths),
            ('rank_capacity', rank_capacities),
            ('differentiation_power', differentiation_powers),
            ('resolution_coherence', resolution_coherences),
            ('resolution_completeness', resolution_completenesses),
            ('rank_hierarchy', rank_hierarchies),
            ('resolution_depth', resolution_depths),
            ('differentiation_stability', differentiation_stabilities),
            ('resolution_precision', resolution_precisions)
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
        """生成observer resolution visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Resolution Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 115: Observer Resolution Dynamics', fontsize=16, fontweight='bold')
        
        # Resolution strength vs rank capacity
        x = [t['resolution_strength'] for t in traces]
        y = [t['rank_capacity'] for t in traces]
        colors = [t['differentiation_power'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Resolution Strength')
        ax1.set_ylabel('Rank Capacity')
        ax1.set_title('Resolution-Rank Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Differentiation Power')
        
        # Differentiation power distribution
        differentiation_powers = [t['differentiation_power'] for t in traces]
        ax2.hist(differentiation_powers, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_xlabel('Differentiation Power')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Differentiation Power Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Resolution coherence vs resolution completeness
        x3 = [t['resolution_coherence'] for t in traces]
        y3 = [t['resolution_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Resolution Coherence')
        ax3.set_ylabel('Resolution Completeness')
        ax3.set_title('Coherence-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Resolution Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-115-obs-resolution-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 115: Observer Resolution Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.resolution_network, k=1.6, iterations=50)
        node_colors = [traces[i]['resolution_strength'] for i in range(len(traces))]
        nx.draw(self.resolution_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=270, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Resolution Network Structure')
        
        # Degree distribution
        degrees = [self.resolution_network.degree(node) for node in self.resolution_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Resolution properties correlation matrix
        properties_matrix = np.array([
            [t['resolution_strength'] for t in traces],
            [t['rank_capacity'] for t in traces],
            [t['differentiation_power'] for t in traces],
            [t['resolution_coherence'] for t in traces],
            [t['resolution_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Resolution', 'Rank', 'Differentiation', 'Coherence', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Resolution Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Rank hierarchy vs resolution depth
        x4 = [t['rank_hierarchy'] for t in traces]
        y4 = [t['resolution_depth'] for t in traces]
        stabilities = [t['differentiation_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Rank Hierarchy')
        ax4.set_ylabel('Resolution Depth')
        ax4.set_title('Hierarchy-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Differentiation Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-115-obs-resolution-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestObsResolution(unittest.TestCase):
    """Unit tests for observer resolution system"""
    
    def setUp(self):
        """Set up test observer resolution system"""
        self.system = ObsResolutionSystem(max_trace_value=20, resolution_depth=4)
        
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
            self.assertIn('resolution_strength', data)
            self.assertIn('rank_capacity', data)
            self.assertIn('differentiation_power', data)
            self.assertIn('resolution_coherence', data)
            self.assertTrue(0 <= data['resolution_strength'] <= 1)
            self.assertTrue(0 <= data['rank_capacity'] <= 1)
            
    def test_resolution_strength_computation(self):
        """测试resolution strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_resolution_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_rank_capacity_computation(self):
        """测试rank capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_rank_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_resolution_network_construction(self):
        """测试resolution network construction"""
        self.assertGreater(len(self.system.resolution_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.resolution_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('resolution_strength', results)
        self.assertIn('rank_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = ObsResolutionSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("OBSERVER RESOLUTION RANK ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Resolution Properties Analysis:")
    properties = ['resolution_strength', 'rank_capacity', 'differentiation_power', 
                 'resolution_coherence', 'resolution_completeness', 'rank_hierarchy',
                 'resolution_depth', 'differentiation_stability', 'resolution_precision']
    
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
    print(f"Rank Pattern Analysis:")
    print(f"Total rank mappings: {sum(len(mappings) for mappings in system.rank_mappings.values())}")
    print(f"Rank mapping density: {results['morphism_density']:.3f}")
    print(f"Average resolutions per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)