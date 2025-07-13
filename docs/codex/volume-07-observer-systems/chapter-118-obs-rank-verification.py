#!/usr/bin/env python3
"""
Chapter 118: ObsRank Unit Test Verification
从ψ=ψ(ψ)推导Observer Tensor Dimensionality Hierarchy

Core principle: From ψ = ψ(ψ) derive systematic observer rank
construction through φ-constrained trace transformations that enable hierarchical
structure of observer complexity through trace geometric relationships,
creating rank networks that encode the fundamental dimensionality
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic rank structures through φ-trace rank
dynamics rather than traditional dimensionality theories or external
rank constructions.

This verification program implements:
1. φ-constrained rank construction through trace hierarchy analysis
2. Observer rank systems: systematic dimensionality through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection rank theory
4. Graph theory analysis of rank networks and hierarchy relationship structures
5. Information theory analysis of rank entropy and dimensionality encoding
6. Category theory analysis of rank functors and hierarchy morphisms
7. Visualization of rank structures and φ-trace rank systems
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

class ObsRankSystem:
    """
    Core system for implementing observer tensor dimensionality hierarchy.
    Implements φ-constrained rank architectures through trace hierarchy dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, rank_depth: int = 7):
        """Initialize observer rank system with hierarchy trace analysis"""
        self.max_trace_value = max_trace_value
        self.rank_depth = rank_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.rank_cache = {}
        self.hierarchy_cache = {}
        self.dimension_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.rank_network = self._build_rank_network()
        self.hierarchy_mappings = self._detect_hierarchy_mappings()
        
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
                rank_data = self._analyze_rank_properties(trace, n)
                universe[n] = rank_data
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
        
    def _analyze_rank_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的rank properties"""
        # Core rank measures
        rank_strength = self._compute_rank_strength(trace, value)
        hierarchy_capacity = self._compute_hierarchy_capacity(trace, value)
        dimension_power = self._compute_dimension_power(trace, value)
        rank_coherence = self._compute_rank_coherence(trace, value)
        
        # Advanced rank measures
        rank_complexity = self._compute_rank_complexity(trace, value)
        hierarchy_depth = self._compute_hierarchy_depth(trace, value)
        dimension_stability = self._compute_dimension_stability(trace, value)
        rank_efficiency = self._compute_rank_efficiency(trace, value)
        hierarchy_balance = self._compute_hierarchy_balance(trace, value)
        
        # Categorize based on rank profile
        category = self._categorize_rank(
            rank_strength, hierarchy_capacity, dimension_power, rank_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'rank_strength': rank_strength,
            'hierarchy_capacity': hierarchy_capacity,
            'dimension_power': dimension_power,
            'rank_coherence': rank_coherence,
            'rank_complexity': rank_complexity,
            'hierarchy_depth': hierarchy_depth,
            'dimension_stability': dimension_stability,
            'rank_efficiency': rank_efficiency,
            'hierarchy_balance': hierarchy_balance,
            'category': category
        }
        
    def _compute_rank_strength(self, trace: str, value: int) -> float:
        """Rank strength emerges from systematic dimensionality capacity"""
        strength_factors = []
        
        # Factor 1: Length provides rank space (minimum 4 for meaningful hierarchy)
        length_factor = len(trace) / 4.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight hierarchy balance (rank favors specific density for structure)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            # Rank optimal at 35% density (0.35) for maximum hierarchy
            hierarchy_balance = 1.0 - abs(0.35 - weight / total)
            # Add rank bonus for dimensionality patterns
            pattern_bonus = min(weight / 3.0, 1.0) if weight > 0 else 0.0
            rank_factor = 0.6 * hierarchy_balance + 0.4 * pattern_bonus
            strength_factors.append(rank_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern rank structure (systematic hierarchy architecture)
        pattern_score = 0.0
        # Count rank-enhancing patterns (hierarchy and dimensionality)
        if trace.startswith('10'):  # Rank initialization pattern
            pattern_score += 0.25
        
        # Hierarchy patterns
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions enable hierarchy
                transitions += 1
        if len(trace) > 1:
            transition_rate = transitions / (len(trace) - 1)
            pattern_score += 0.25 * min(transition_rate * 1.5, 1.0)  # Value transitions
        
        # Dimensionality patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] in ['100', '001']:  # Dimension boundary patterns
                pattern_score += 0.15
        
        # Rank-specific patterns (value modulo 7 for 7-level hierarchy)
        if value % 7 in [0, 3, 6]:  # Key hierarchy levels
            pattern_score += 0.2
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint rank coherence  
        phi_factor = 0.88 if self._is_phi_valid(trace) else 0.25
        strength_factors.append(phi_factor)
        
        # Rank strength emerges from geometric mean of factors
        rank_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return rank_strength
        
    def _compute_hierarchy_capacity(self, trace: str, value: int) -> float:
        """Hierarchy capacity emerges from systematic structure capability"""
        capacity_factors = []
        
        # Factor 1: Structural hierarchy potential
        structural = 0.3 + 0.7 * min(len(trace) / 8.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Hierarchy complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.15 + 0.85 * min(ones_count / 4.5, 1.0)
        else:
            complexity = 0.05
        capacity_factors.append(complexity)
        
        # Factor 3: Hierarchy levels (value modulo 9 for 9-level structure)
        level_depth = 0.25 + 0.75 * (value % 9) / 8.0
        capacity_factors.append(level_depth)
        
        # Factor 4: φ-constraint hierarchy preservation
        preservation = 0.9 if self._is_phi_valid(trace) else 0.2
        capacity_factors.append(preservation)
        
        hierarchy_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return hierarchy_capacity
        
    def _compute_dimension_power(self, trace: str, value: int) -> float:
        """Dimension power emerges from systematic dimensionality capability"""
        power_factors = []
        
        # Factor 1: Dimension scope
        dim_scope = 0.2 + 0.8 * min(len(trace) / 6.5, 1.0)
        power_factors.append(dim_scope)
        
        # Factor 2: Dimension efficiency
        if len(trace) > 0:
            balance_ratio = trace.count('1') / len(trace)
            # Optimal around 35% for dimension structures
            dim_efficiency = 1.0 - 2.86 * abs(0.35 - balance_ratio)
            dim_efficiency = max(0.1, min(dim_efficiency, 1.0))
        else:
            dim_efficiency = 0.3
        power_factors.append(dim_efficiency)
        
        # Factor 3: Dimension coverage (value modulo 11 for 11-dimensional space)
        coverage = 0.25 + 0.75 * (value % 11) / 10.0
        power_factors.append(coverage)
        
        dimension_power = np.prod(power_factors) ** (1.0 / len(power_factors))
        return dimension_power
        
    def _compute_rank_coherence(self, trace: str, value: int) -> float:
        """Rank coherence emerges from systematic rank integration capability"""
        coherence_factors = []
        
        # Factor 1: Rank integration capacity
        integration_cap = 0.25 + 0.75 * min(len(trace) / 5.0, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Rank coherence scope (value modulo 4 for 4-fold symmetry)
        coherence_scope = 0.4 + 0.6 * (value % 4) / 3.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic rank coherence (value modulo 10 for decimal hierarchy)
        systematic = 0.5 + 0.5 * (value % 10) / 9.0
        coherence_factors.append(systematic)
        
        rank_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return rank_coherence
        
    def _compute_rank_complexity(self, trace: str, value: int) -> float:
        """Rank complexity through comprehensive hierarchy analysis"""
        if len(trace) == 0:
            return 0.1
            
        # Analyze rank pattern complexity
        patterns = set()
        for i in range(len(trace) - 1):
            patterns.add(trace[i:i+2])
        for i in range(len(trace) - 2):
            patterns.add(trace[i:i+3])
            
        pattern_diversity = len(patterns) / (len(trace) * 2) if len(trace) > 0 else 0
        value_complexity = (value % 13) / 12.0  # 13 for prime complexity levels
        
        return 0.3 + 0.7 * (0.6 * pattern_diversity + 0.4 * value_complexity)
        
    def _compute_hierarchy_depth(self, trace: str, value: int) -> float:
        """Hierarchy depth through nested rank analysis"""
        depth_factor = min(len(trace) / 9.0, 1.0)
        nesting_factor = (value % 12) / 11.0  # 12-level nesting
        return 0.2 + 0.8 * (depth_factor * nesting_factor)
        
    def _compute_dimension_stability(self, trace: str, value: int) -> float:
        """Dimension stability through consistent rank architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.85
        else:
            stability_base = 0.35
        variation = 0.15 * sin(value * 0.31)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_rank_efficiency(self, trace: str, value: int) -> float:
        """Rank efficiency through optimized hierarchy utilization"""
        if len(trace) > 0:
            # Efficiency based on rank structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal efficiency around 35% weight for ranks
            efficiency_base = 1.0 - 2.86 * abs(0.35 - weight_ratio)
        else:
            efficiency_base = 0.0
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(efficiency_base + phi_bonus, 1.0))
        
    def _compute_hierarchy_balance(self, trace: str, value: int) -> float:
        """Hierarchy balance through uniform rank distribution"""
        if len(trace) < 4:
            return 0.2
            
        # Analyze hierarchy distribution
        quarters = 4
        quarter_length = len(trace) // quarters
        quarter_weights = []
        
        for i in range(quarters):
            start = i * quarter_length
            end = min((i + 1) * quarter_length, len(trace))
            if end > start:
                segment = trace[start:end]
                weight = segment.count('1') / len(segment)
                quarter_weights.append(weight)
        
        if quarter_weights:
            mean_weight = np.mean(quarter_weights)
            std_weight = np.std(quarter_weights)
            balance = 1.0 - min(std_weight * 3, 1.0)  # Lower std means better balance
        else:
            balance = 0.2
            
        return balance
        
    def _categorize_rank(self, rank: float, hierarchy: float, 
                         dimension: float, coherence: float) -> str:
        """Categorize trace based on rank profile"""
        # Calculate dominant characteristic with thresholds
        rank_threshold = 0.6      # High rank strength threshold
        hierarchy_threshold = 0.5  # Moderate hierarchy capacity threshold
        dimension_threshold = 0.5  # Moderate dimension power threshold
        
        if rank >= rank_threshold:
            if hierarchy >= hierarchy_threshold:
                return "hierarchy_rank"              # High rank + hierarchy
            elif dimension >= dimension_threshold:
                return "dimension_rank"              # High rank + dimension
            else:
                return "strong_rank"                 # High rank + moderate properties
        else:
            if hierarchy >= hierarchy_threshold:
                return "hierarchy_structure"         # Moderate rank + hierarchy
            elif dimension >= dimension_threshold:
                return "dimension_structure"         # Moderate rank + dimension
            else:
                return "basic_rank"                  # Basic rank capability
        
    def _build_rank_network(self) -> nx.Graph:
        """构建rank network based on trace hierarchy similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on hierarchy similarity
        similarity_threshold = 0.2
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_hierarchy_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_hierarchy_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的hierarchy similarity"""
        # Hierarchy property similarity
        rank_diff = abs(trace1['rank_strength'] - trace2['rank_strength'])
        hierarchy_diff = abs(trace1['hierarchy_capacity'] - trace2['hierarchy_capacity'])
        dimension_diff = abs(trace1['dimension_power'] - trace2['dimension_power'])
        coherence_diff = abs(trace1['rank_coherence'] - trace2['rank_coherence'])
        
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
        property_similarity = 1.0 - np.mean([rank_diff, hierarchy_diff, 
                                           dimension_diff, coherence_diff])
        
        return 0.7 * property_similarity + 0.3 * geometric_similarity
        
    def _detect_hierarchy_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测rank traces之间的hierarchy mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.25
        for node1 in self.rank_network.nodes():
            data1 = self.rank_network.nodes[node1]
            for node2 in self.rank_network.nodes():
                if node1 != node2:
                    data2 = self.rank_network.nodes[node2]
                    
                    # Check hierarchy preservation
                    hierarchy_preserved = abs(data1['hierarchy_capacity'] - 
                                            data2['hierarchy_capacity']) <= tolerance
                    rank_preserved = abs(data1['rank_strength'] - 
                                       data2['rank_strength']) <= tolerance
                    
                    if hierarchy_preserved and rank_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive observer rank analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.rank_network)
        results['connected_components'] = nx.number_connected_components(self.rank_network)
        
        # Rank properties analysis
        rank_strengths = [t['rank_strength'] for t in traces]
        hierarchy_capacities = [t['hierarchy_capacity'] for t in traces]
        dimension_powers = [t['dimension_power'] for t in traces]
        rank_coherences = [t['rank_coherence'] for t in traces]
        rank_complexities = [t['rank_complexity'] for t in traces]
        hierarchy_depths = [t['hierarchy_depth'] for t in traces]
        dimension_stabilities = [t['dimension_stability'] for t in traces]
        rank_efficiencies = [t['rank_efficiency'] for t in traces]
        hierarchy_balances = [t['hierarchy_balance'] for t in traces]
        
        results['rank_strength'] = {
            'mean': np.mean(rank_strengths),
            'std': np.std(rank_strengths),
            'high_count': sum(1 for x in rank_strengths if x > 0.5)
        }
        results['hierarchy_capacity'] = {
            'mean': np.mean(hierarchy_capacities),
            'std': np.std(hierarchy_capacities),
            'high_count': sum(1 for x in hierarchy_capacities if x > 0.5)
        }
        results['dimension_power'] = {
            'mean': np.mean(dimension_powers),
            'std': np.std(dimension_powers),
            'high_count': sum(1 for x in dimension_powers if x > 0.5)
        }
        results['rank_coherence'] = {
            'mean': np.mean(rank_coherences),
            'std': np.std(rank_coherences),
            'high_count': sum(1 for x in rank_coherences if x > 0.5)
        }
        results['rank_complexity'] = {
            'mean': np.mean(rank_complexities),
            'std': np.std(rank_complexities),
            'high_count': sum(1 for x in rank_complexities if x > 0.5)
        }
        results['hierarchy_depth'] = {
            'mean': np.mean(hierarchy_depths),
            'std': np.std(hierarchy_depths),
            'high_count': sum(1 for x in hierarchy_depths if x > 0.5)
        }
        results['dimension_stability'] = {
            'mean': np.mean(dimension_stabilities),
            'std': np.std(dimension_stabilities),
            'high_count': sum(1 for x in dimension_stabilities if x > 0.5)
        }
        results['rank_efficiency'] = {
            'mean': np.mean(rank_efficiencies),
            'std': np.std(rank_efficiencies),
            'high_count': sum(1 for x in rank_efficiencies if x > 0.5)
        }
        results['hierarchy_balance'] = {
            'mean': np.mean(hierarchy_balances),
            'std': np.std(hierarchy_balances),
            'high_count': sum(1 for x in hierarchy_balances if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.rank_network.edges()) > 0:
            results['network_edges'] = len(self.rank_network.edges())
            results['average_degree'] = sum(dict(self.rank_network.degree()).values()) / len(self.rank_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.hierarchy_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('rank_strength', rank_strengths),
            ('hierarchy_capacity', hierarchy_capacities),
            ('dimension_power', dimension_powers),
            ('rank_coherence', rank_coherences),
            ('rank_complexity', rank_complexities),
            ('hierarchy_depth', hierarchy_depths),
            ('dimension_stability', dimension_stabilities),
            ('rank_efficiency', rank_efficiencies),
            ('hierarchy_balance', hierarchy_balances)
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
        """生成observer rank visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Rank Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 118: Observer Rank Dynamics', fontsize=16, fontweight='bold')
        
        # Rank strength vs hierarchy capacity
        x = [t['rank_strength'] for t in traces]
        y = [t['hierarchy_capacity'] for t in traces]
        colors = [t['dimension_power'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Rank Strength')
        ax1.set_ylabel('Hierarchy Capacity')
        ax1.set_title('Rank-Hierarchy Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Dimension Power')
        
        # Dimension power distribution
        dimension_powers = [t['dimension_power'] for t in traces]
        ax2.hist(dimension_powers, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Dimension Power')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Dimension Power Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Rank complexity vs hierarchy depth
        x3 = [t['rank_complexity'] for t in traces]
        y3 = [t['hierarchy_depth'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Rank Complexity')
        ax3.set_ylabel('Hierarchy Depth')
        ax3.set_title('Complexity-Depth Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Rank Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-118-obs-rank-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 118: Observer Rank Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.rank_network, k=1.5, iterations=50)
        node_colors = [traces[i]['rank_strength'] for i in range(len(traces))]
        nx.draw(self.rank_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=250, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Rank Network Structure')
        
        # Degree distribution
        degrees = [self.rank_network.degree(node) for node in self.rank_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='salmon', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Rank properties correlation matrix
        properties_matrix = np.array([
            [t['rank_strength'] for t in traces],
            [t['hierarchy_capacity'] for t in traces],
            [t['dimension_power'] for t in traces],
            [t['rank_coherence'] for t in traces],
            [t['dimension_stability'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Rank', 'Hierarchy', 'Dimension', 'Coherence', 'Stability']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Rank Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Rank efficiency vs hierarchy balance
        x4 = [t['rank_efficiency'] for t in traces]
        y4 = [t['hierarchy_balance'] for t in traces]
        stabilities = [t['dimension_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Rank Efficiency')
        ax4.set_ylabel('Hierarchy Balance')
        ax4.set_title('Efficiency-Balance Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Dimension Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-118-obs-rank-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestObsRank(unittest.TestCase):
    """Unit tests for observer rank system"""
    
    def setUp(self):
        """Set up test observer rank system"""
        self.system = ObsRankSystem(max_trace_value=20, rank_depth=4)
        
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
            self.assertIn('rank_strength', data)
            self.assertIn('hierarchy_capacity', data)
            self.assertIn('dimension_power', data)
            self.assertIn('rank_coherence', data)
            self.assertTrue(0 <= data['rank_strength'] <= 1)
            self.assertTrue(0 <= data['hierarchy_capacity'] <= 1)
            
    def test_rank_strength_computation(self):
        """测试rank strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_rank_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_hierarchy_capacity_computation(self):
        """测试hierarchy capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_hierarchy_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_rank_network_construction(self):
        """测试rank network construction"""
        self.assertGreater(len(self.system.rank_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.rank_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('rank_strength', results)
        self.assertIn('hierarchy_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = ObsRankSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("OBSERVER RANK HIERARCHY ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Rank Properties Analysis:")
    properties = ['rank_strength', 'hierarchy_capacity', 'dimension_power', 
                 'rank_coherence', 'rank_complexity', 'hierarchy_depth',
                 'dimension_stability', 'rank_efficiency', 'hierarchy_balance']
    
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
    print(f"Hierarchy Pattern Analysis:")
    print(f"Total hierarchy mappings: {sum(len(mappings) for mappings in system.hierarchy_mappings.values())}")
    print(f"Hierarchy mapping density: {results['morphism_density']:.3f}")
    print(f"Average ranks per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)