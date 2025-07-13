#!/usr/bin/env python3
"""
Chapter 117: ZetaModulate Unit Test Verification
从ψ=ψ(ψ)推导Observer ζ-Weight Modulation Function in Path Collapse

Core principle: From ψ = ψ(ψ) derive systematic zeta modulation
construction through φ-constrained trace transformations that enable observers
to modulate spectral weights through trace geometric relationships,
creating modulation networks that encode the fundamental weight adjustment
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic modulation structures through φ-trace modulation
dynamics rather than traditional signal modulation theories or external
modulation constructions.

This verification program implements:
1. φ-constrained modulation construction through trace spectral weight analysis
2. Zeta modulation systems: systematic weight adjustment through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection modulation theory
4. Graph theory analysis of modulation networks and weight relationship structures
5. Information theory analysis of modulation entropy and weight encoding
6. Category theory analysis of modulation functors and spectral morphisms
7. Visualization of modulation structures and φ-trace modulation systems
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

class ZetaModulateSystem:
    """
    Core system for implementing observer ζ-weight modulation function in path collapse.
    Implements φ-constrained modulation architectures through trace weight dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, modulation_depth: int = 6):
        """Initialize zeta modulation system with spectral weight analysis"""
        self.max_trace_value = max_trace_value
        self.modulation_depth = modulation_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.modulation_cache = {}
        self.spectral_cache = {}
        self.weight_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.modulation_network = self._build_modulation_network()
        self.weight_mappings = self._detect_weight_mappings()
        
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
                modulation_data = self._analyze_modulation_properties(trace, n)
                universe[n] = modulation_data
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
        
    def _analyze_modulation_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的modulation properties"""
        # Core modulation measures
        modulation_strength = self._compute_modulation_strength(trace, value)
        spectral_weight = self._compute_spectral_weight(trace, value)
        zeta_function = self._compute_zeta_function(trace, value)
        modulation_coherence = self._compute_modulation_coherence(trace, value)
        
        # Advanced modulation measures
        weight_distribution = self._compute_weight_distribution(trace, value)
        spectral_bandwidth = self._compute_spectral_bandwidth(trace, value)
        modulation_depth = self._compute_modulation_depth(trace, value)
        weight_stability = self._compute_weight_stability(trace, value)
        modulation_efficiency = self._compute_modulation_efficiency(trace, value)
        
        # Categorize based on modulation profile
        category = self._categorize_modulation(
            modulation_strength, spectral_weight, zeta_function, modulation_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'modulation_strength': modulation_strength,
            'spectral_weight': spectral_weight,
            'zeta_function': zeta_function,
            'modulation_coherence': modulation_coherence,
            'weight_distribution': weight_distribution,
            'spectral_bandwidth': spectral_bandwidth,
            'modulation_depth': modulation_depth,
            'weight_stability': weight_stability,
            'modulation_efficiency': modulation_efficiency,
            'category': category
        }
        
    def _compute_modulation_strength(self, trace: str, value: int) -> float:
        """Modulation strength emerges from systematic weight adjustment capacity"""
        strength_factors = []
        
        # Factor 1: Length provides modulation space (minimum 3 for meaningful modulation)
        length_factor = len(trace) / 3.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight modulation balance (modulation favors specific density for clarity)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            # Modulation optimal at 40% density (0.4) for maximum weight adjustment
            modulation_balance = 1.0 - abs(0.4 - weight / total)
            # Add modulation bonus for spectral patterns
            pattern_bonus = min(weight / 2.0, 1.0) if weight > 0 else 0.0
            modulation_factor = 0.6 * modulation_balance + 0.4 * pattern_bonus
            strength_factors.append(modulation_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern modulation structure (systematic weight architecture)
        pattern_score = 0.0
        # Count modulation-enhancing patterns (weight adjustment and spectral)
        if trace.startswith('10'):  # Modulation initialization pattern
            pattern_score += 0.25
        
        # Weight transition patterns
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions enable modulation
                transitions += 1
        if len(trace) > 1:
            transition_rate = transitions / (len(trace) - 1)
            pattern_score += 0.3 * min(transition_rate * 1.5, 1.0)  # Value transitions
        
        # Spectral modulation patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] in ['010', '101']:  # Spectral modulation patterns
                pattern_score += 0.15
        
        # Zeta-specific patterns (value modulo 5 for 5-level zeta function)
        if value % 5 == 0:
            pattern_score += 0.1
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint modulation coherence  
        phi_factor = 0.88 if self._is_phi_valid(trace) else 0.25
        strength_factors.append(phi_factor)
        
        # Modulation strength emerges from geometric mean of factors
        modulation_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return modulation_strength
        
    def _compute_spectral_weight(self, trace: str, value: int) -> float:
        """Spectral weight emerges from systematic weight distribution capability"""
        weight_factors = []
        
        # Factor 1: Spectral distribution potential
        distribution = 0.2 + 0.8 * min(len(trace) / 5.0, 1.0)
        weight_factors.append(distribution)
        
        # Factor 2: Weight complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.15 + 0.85 * min(ones_count / 3.5, 1.0)
        else:
            complexity = 0.05
        weight_factors.append(complexity)
        
        # Factor 3: Spectral depth (value modulo 7 for 7-level spectral hierarchy)
        spectral_depth = 0.3 + 0.7 * (value % 7) / 6.0
        weight_factors.append(spectral_depth)
        
        # Factor 4: φ-constraint weight preservation
        preservation = 0.9 if self._is_phi_valid(trace) else 0.2
        weight_factors.append(preservation)
        
        spectral_weight = np.prod(weight_factors) ** (1.0 / len(weight_factors))
        return spectral_weight
        
    def _compute_zeta_function(self, trace: str, value: int) -> float:
        """Zeta function ζ(s) emerges from systematic spectral modulation"""
        # Riemann zeta-inspired function adapted for φ-constraints
        if len(trace) == 0:
            return 0.1
            
        # Base zeta from trace structure
        s = len(trace) / 10.0 + 1.0  # s parameter from trace length
        
        # Compute zeta approximation (first 5 terms for efficiency)
        zeta_sum = 0.0
        for n in range(1, 6):  # 5 terms for 5-level approximation
            zeta_sum += 1.0 / (n ** s)
        
        # Modulate by trace properties
        weight_ratio = trace.count('1') / len(trace) if len(trace) > 0 else 0.5
        modulation = 0.5 + 0.5 * sin(2 * pi * weight_ratio)
        
        # φ-constraint bonus
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        
        zeta_function = min(zeta_sum * modulation + phi_bonus, 1.0)
        return zeta_function
        
    def _compute_modulation_coherence(self, trace: str, value: int) -> float:
        """Modulation coherence emerges from systematic spectral integration capability"""
        coherence_factors = []
        
        # Factor 1: Spectral integration capacity
        integration_cap = 0.25 + 0.75 * min(len(trace) / 4.5, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Modulation coherence scope (value modulo 4 for 4-phase coherence)
        coherence_scope = 0.4 + 0.6 * (value % 4) / 3.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic spectral coherence (value modulo 9 for 9-level coherence)
        systematic = 0.5 + 0.5 * (value % 9) / 8.0
        coherence_factors.append(systematic)
        
        modulation_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return modulation_coherence
        
    def _compute_weight_distribution(self, trace: str, value: int) -> float:
        """Weight distribution through comprehensive spectral coverage"""
        if len(trace) == 0:
            return 0.1
            
        # Analyze weight distribution uniformity
        segments = 4  # 4 segments for quaternary analysis
        segment_length = max(1, len(trace) // segments)
        segment_weights = []
        
        for i in range(segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(trace))
            segment = trace[start:end]
            weight = segment.count('1') / len(segment) if len(segment) > 0 else 0
            segment_weights.append(weight)
        
        # Compute distribution uniformity
        mean_weight = np.mean(segment_weights)
        std_weight = np.std(segment_weights)
        uniformity = 1.0 - min(std_weight * 2, 1.0)  # Lower std means better distribution
        
        return uniformity
        
    def _compute_spectral_bandwidth(self, trace: str, value: int) -> float:
        """Spectral bandwidth through frequency range analysis"""
        if len(trace) < 2:
            return 0.1
            
        # Analyze frequency content through transitions
        frequencies = []
        current_run = 1
        
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_run += 1
            else:
                frequencies.append(1.0 / current_run)
                current_run = 1
        frequencies.append(1.0 / current_run)
        
        # Compute bandwidth
        if frequencies:
            bandwidth = max(frequencies) - min(frequencies)
            normalized_bandwidth = min(bandwidth * 2, 1.0)
        else:
            normalized_bandwidth = 0.1
            
        return normalized_bandwidth
        
    def _compute_modulation_depth(self, trace: str, value: int) -> float:
        """Modulation depth through amplitude variation analysis"""
        depth_factor = min(len(trace) / 6.0, 1.0)
        complexity_factor = (value % 12) / 11.0  # 12-level depth for dodecaphonic modulation
        return 0.2 + 0.8 * (depth_factor * complexity_factor)
        
    def _compute_weight_stability(self, trace: str, value: int) -> float:
        """Weight stability through consistent modulation architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.85
        else:
            stability_base = 0.35
        variation = 0.15 * sin(value * 0.28)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_modulation_efficiency(self, trace: str, value: int) -> float:
        """Modulation efficiency through optimized weight adjustment"""
        if len(trace) > 0:
            # Efficiency based on modulation structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal efficiency around 40% weight for modulation
            efficiency_base = 1.0 - 2.5 * abs(0.4 - weight_ratio)
        else:
            efficiency_base = 0.0
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(efficiency_base + phi_bonus, 1.0))
        
    def _categorize_modulation(self, modulation: float, weight: float, 
                              zeta: float, coherence: float) -> str:
        """Categorize trace based on modulation profile"""
        # Calculate dominant characteristic with thresholds
        modulation_threshold = 0.6   # High modulation strength threshold
        weight_threshold = 0.5       # Moderate spectral weight threshold
        zeta_threshold = 0.5         # Moderate zeta function threshold
        
        if modulation >= modulation_threshold:
            if zeta >= zeta_threshold:
                return "zeta_modulator"              # High modulation + zeta
            elif weight >= weight_threshold:
                return "weight_modulator"            # High modulation + weight
            else:
                return "strong_modulator"            # High modulation + moderate properties
        else:
            if zeta >= zeta_threshold:
                return "zeta_processor"              # Moderate modulation + zeta
            elif weight >= weight_threshold:
                return "weight_processor"            # Moderate modulation + weight
            else:
                return "basic_modulator"             # Basic modulation capability
        
    def _build_modulation_network(self) -> nx.Graph:
        """构建modulation network based on trace spectral similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on spectral similarity
        similarity_threshold = 0.2
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
        modulation_diff = abs(trace1['modulation_strength'] - trace2['modulation_strength'])
        weight_diff = abs(trace1['spectral_weight'] - trace2['spectral_weight'])
        zeta_diff = abs(trace1['zeta_function'] - trace2['zeta_function'])
        coherence_diff = abs(trace1['modulation_coherence'] - trace2['modulation_coherence'])
        
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
        property_similarity = 1.0 - np.mean([modulation_diff, weight_diff, 
                                           zeta_diff, coherence_diff])
        
        return 0.7 * property_similarity + 0.3 * geometric_similarity
        
    def _detect_weight_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测modulation traces之间的weight mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.25
        for node1 in self.modulation_network.nodes():
            data1 = self.modulation_network.nodes[node1]
            for node2 in self.modulation_network.nodes():
                if node1 != node2:
                    data2 = self.modulation_network.nodes[node2]
                    
                    # Check weight preservation
                    weight_preserved = abs(data1['spectral_weight'] - 
                                         data2['spectral_weight']) <= tolerance
                    modulation_preserved = abs(data1['modulation_strength'] - 
                                             data2['modulation_strength']) <= tolerance
                    
                    if weight_preserved and modulation_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive zeta modulation analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.modulation_network)
        results['connected_components'] = nx.number_connected_components(self.modulation_network)
        
        # Modulation properties analysis
        modulation_strengths = [t['modulation_strength'] for t in traces]
        spectral_weights = [t['spectral_weight'] for t in traces]
        zeta_functions = [t['zeta_function'] for t in traces]
        modulation_coherences = [t['modulation_coherence'] for t in traces]
        weight_distributions = [t['weight_distribution'] for t in traces]
        spectral_bandwidths = [t['spectral_bandwidth'] for t in traces]
        modulation_depths = [t['modulation_depth'] for t in traces]
        weight_stabilities = [t['weight_stability'] for t in traces]
        modulation_efficiencies = [t['modulation_efficiency'] for t in traces]
        
        results['modulation_strength'] = {
            'mean': np.mean(modulation_strengths),
            'std': np.std(modulation_strengths),
            'high_count': sum(1 for x in modulation_strengths if x > 0.5)
        }
        results['spectral_weight'] = {
            'mean': np.mean(spectral_weights),
            'std': np.std(spectral_weights),
            'high_count': sum(1 for x in spectral_weights if x > 0.5)
        }
        results['zeta_function'] = {
            'mean': np.mean(zeta_functions),
            'std': np.std(zeta_functions),
            'high_count': sum(1 for x in zeta_functions if x > 0.5)
        }
        results['modulation_coherence'] = {
            'mean': np.mean(modulation_coherences),
            'std': np.std(modulation_coherences),
            'high_count': sum(1 for x in modulation_coherences if x > 0.5)
        }
        results['weight_distribution'] = {
            'mean': np.mean(weight_distributions),
            'std': np.std(weight_distributions),
            'high_count': sum(1 for x in weight_distributions if x > 0.5)
        }
        results['spectral_bandwidth'] = {
            'mean': np.mean(spectral_bandwidths),
            'std': np.std(spectral_bandwidths),
            'high_count': sum(1 for x in spectral_bandwidths if x > 0.5)
        }
        results['modulation_depth'] = {
            'mean': np.mean(modulation_depths),
            'std': np.std(modulation_depths),
            'high_count': sum(1 for x in modulation_depths if x > 0.5)
        }
        results['weight_stability'] = {
            'mean': np.mean(weight_stabilities),
            'std': np.std(weight_stabilities),
            'high_count': sum(1 for x in weight_stabilities if x > 0.5)
        }
        results['modulation_efficiency'] = {
            'mean': np.mean(modulation_efficiencies),
            'std': np.std(modulation_efficiencies),
            'high_count': sum(1 for x in modulation_efficiencies if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.modulation_network.edges()) > 0:
            results['network_edges'] = len(self.modulation_network.edges())
            results['average_degree'] = sum(dict(self.modulation_network.degree()).values()) / len(self.modulation_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.weight_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('modulation_strength', modulation_strengths),
            ('spectral_weight', spectral_weights),
            ('zeta_function', zeta_functions),
            ('modulation_coherence', modulation_coherences),
            ('weight_distribution', weight_distributions),
            ('spectral_bandwidth', spectral_bandwidths),
            ('modulation_depth', modulation_depths),
            ('weight_stability', weight_stabilities),
            ('modulation_efficiency', modulation_efficiencies)
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
        """生成zeta modulation visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Modulation Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 117: Zeta Modulation Dynamics', fontsize=16, fontweight='bold')
        
        # Modulation strength vs spectral weight
        x = [t['modulation_strength'] for t in traces]
        y = [t['spectral_weight'] for t in traces]
        colors = [t['zeta_function'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Modulation Strength')
        ax1.set_ylabel('Spectral Weight')
        ax1.set_title('Modulation-Weight Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Zeta Function')
        
        # Zeta function distribution
        zeta_functions = [t['zeta_function'] for t in traces]
        ax2.hist(zeta_functions, bins=15, alpha=0.7, color='indigo', edgecolor='black')
        ax2.set_xlabel('Zeta Function Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Zeta Function Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Weight distribution vs spectral bandwidth
        x3 = [t['weight_distribution'] for t in traces]
        y3 = [t['spectral_bandwidth'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Weight Distribution')
        ax3.set_ylabel('Spectral Bandwidth')
        ax3.set_title('Distribution-Bandwidth Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Modulation Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-117-zeta-modulate-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 117: Zeta Modulation Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.modulation_network, k=1.5, iterations=50)
        node_colors = [traces[i]['modulation_strength'] for i in range(len(traces))]
        nx.draw(self.modulation_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=250, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Modulation Network Structure')
        
        # Degree distribution
        degrees = [self.modulation_network.degree(node) for node in self.modulation_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='orchid', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Modulation properties correlation matrix
        properties_matrix = np.array([
            [t['modulation_strength'] for t in traces],
            [t['spectral_weight'] for t in traces],
            [t['zeta_function'] for t in traces],
            [t['modulation_coherence'] for t in traces],
            [t['weight_stability'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Modulation', 'Weight', 'Zeta', 'Coherence', 'Stability']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Modulation Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Modulation efficiency vs modulation depth
        x4 = [t['modulation_efficiency'] for t in traces]
        y4 = [t['modulation_depth'] for t in traces]
        stabilities = [t['weight_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Modulation Efficiency')
        ax4.set_ylabel('Modulation Depth')
        ax4.set_title('Efficiency-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Weight Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-117-zeta-modulate-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestZetaModulate(unittest.TestCase):
    """Unit tests for zeta modulation system"""
    
    def setUp(self):
        """Set up test zeta modulation system"""
        self.system = ZetaModulateSystem(max_trace_value=20, modulation_depth=4)
        
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
            self.assertIn('modulation_strength', data)
            self.assertIn('spectral_weight', data)
            self.assertIn('zeta_function', data)
            self.assertIn('modulation_coherence', data)
            self.assertTrue(0 <= data['modulation_strength'] <= 1)
            self.assertTrue(0 <= data['spectral_weight'] <= 1)
            
    def test_modulation_strength_computation(self):
        """测试modulation strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_modulation_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_zeta_function_computation(self):
        """测试zeta function computation"""
        trace = "1001"
        value = 8
        zeta = self.system._compute_zeta_function(trace, value)
        self.assertTrue(0 <= zeta <= 1)
        
    def test_modulation_network_construction(self):
        """测试modulation network construction"""
        self.assertGreater(len(self.system.modulation_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.modulation_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('modulation_strength', results)
        self.assertIn('spectral_weight', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = ZetaModulateSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("ZETA MODULATION SPECTRAL ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Modulation Properties Analysis:")
    properties = ['modulation_strength', 'spectral_weight', 'zeta_function', 
                 'modulation_coherence', 'weight_distribution', 'spectral_bandwidth',
                 'modulation_depth', 'weight_stability', 'modulation_efficiency']
    
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
    print(f"Weight Pattern Analysis:")
    print(f"Total weight mappings: {sum(len(mappings) for mappings in system.weight_mappings.values())}")
    print(f"Weight mapping density: {results['morphism_density']:.3f}")
    print(f"Average modulations per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)