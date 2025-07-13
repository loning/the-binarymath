#!/usr/bin/env python3
"""
Chapter 109: CollapseAxioms Unit Test Verification
从ψ=ψ(ψ)推导Structural Collapse Foundations of Formal Systems

Core principle: From ψ = ψ(ψ) derive systematic axiomatic foundations
through φ-constrained trace transformations that enable axiomatizing
collapse-based mathematics through trace geometric relationships, creating
axiom networks that encode the fundamental axiomatic principles of collapsed
space through entropy-increasing tensor transformations that establish
systematic axiomatic structures through φ-trace axiom dynamics rather than
traditional set-theoretic axiomatizations or external logical foundations.

This verification program implements:
1. φ-constrained axiom construction through trace foundation analysis
2. Axiomatic systems: systematic foundation building through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection axiom theory
4. Graph theory analysis of axiom networks and foundation relationship structures
5. Information theory analysis of axiom entropy and foundation encoding
6. Category theory analysis of axiom functors and foundation morphisms
7. Visualization of axiom structures and φ-trace foundation systems
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

class CollapseAxiomsSystem:
    """
    Core system for implementing structural collapse foundations of formal systems.
    Implements φ-constrained axiom architectures through trace foundation dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, axiom_depth: int = 6):
        """Initialize collapse axioms system with foundation trace analysis"""
        self.max_trace_value = max_trace_value
        self.axiom_depth = axiom_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.axiom_cache = {}
        self.foundation_cache = {}
        self.system_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.axiom_network = self._build_axiom_network()
        self.foundation_mappings = self._detect_foundation_mappings()
        
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
                axiom_data = self._analyze_axiom_properties(trace, n)
                universe[n] = axiom_data
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
        
    def _analyze_axiom_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的axiom properties"""
        # Core axiom measures
        axiom_strength = self._compute_axiom_strength(trace, value)
        foundation_power = self._compute_foundation_power(trace, value)
        independence_capacity = self._compute_independence_capacity(trace, value)
        completeness_degree = self._compute_completeness_degree(trace, value)
        
        # Advanced axiom measures
        axiom_consistency = self._compute_axiom_consistency(trace, value)
        foundation_minimality = self._compute_foundation_minimality(trace, value)
        axiom_depth = self._compute_axiom_depth(trace, value)
        system_stability = self._compute_system_stability(trace, value)
        axiom_coherence = self._compute_axiom_coherence(trace, value)
        
        # Categorize based on axiom profile
        category = self._categorize_axiom(
            axiom_strength, foundation_power, independence_capacity, completeness_degree
        )
        
        return {
            'trace': trace,
            'value': value,
            'axiom_strength': axiom_strength,
            'foundation_power': foundation_power,
            'independence_capacity': independence_capacity,
            'completeness_degree': completeness_degree,
            'axiom_consistency': axiom_consistency,
            'foundation_minimality': foundation_minimality,
            'axiom_depth': axiom_depth,
            'system_stability': system_stability,
            'axiom_coherence': axiom_coherence,
            'category': category
        }
        
    def _compute_axiom_strength(self, trace: str, value: int) -> float:
        """Axiom strength emerges from systematic foundation construction capacity"""
        strength_factors = []
        
        # Factor 1: Length provides axiom foundation space
        length_factor = len(trace) / 8.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight balance (optimal for axiom operations)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            balance = 1.0 - abs(0.4 - weight / total)  # Prefer moderate 1s density for axioms
            strength_factors.append(balance)
        else:
            strength_factors.append(0.5)
        
        # Factor 3: Pattern foundational structure (systematic axiom architecture)
        pattern_score = 0.0
        # Count foundational patterns (starting with 1, ending appropriately)
        if trace.startswith('1'):
            pattern_score += 0.3
        if len(trace) >= 3 and trace[1] == '0':  # Foundation gap
            pattern_score += 0.2
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '101':  # Axiom sandwich pattern
                pattern_score += 0.2
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint axiom coherence
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.1
        strength_factors.append(phi_factor)
        
        # Axiom strength emerges from geometric mean of factors
        axiom_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return axiom_strength
        
    def _compute_foundation_power(self, trace: str, value: int) -> float:
        """Foundation power emerges from systematic construction capability"""
        power_factors = []
        
        # Factor 1: Foundation generative power
        generative = 0.5 + 0.5 * min(len(trace) / 6.0, 1.0)
        power_factors.append(generative)
        
        # Factor 2: Construction capability (systematic building)
        ones_count = trace.count('1')
        if ones_count > 0:
            construction = 0.4 + 0.6 * min(ones_count / 3.0, 1.0)
        else:
            construction = 0.2
        power_factors.append(construction)
        
        # Factor 3: Foundation span depth
        span_depth = 0.3 + 0.7 * (value % 8) / 7.0
        power_factors.append(span_depth)
        
        # Factor 4: φ-constraint foundation preservation
        preservation = 0.85 if self._is_phi_valid(trace) else 0.25
        power_factors.append(preservation)
        
        foundation_power = np.prod(power_factors) ** (1.0 / len(power_factors))
        return foundation_power
        
    def _compute_independence_capacity(self, trace: str, value: int) -> float:
        """Independence capacity emerges from non-redundant axiom capability"""
        independence_factors = []
        
        # Factor 1: Independence uniqueness
        uniqueness = len(set(trace)) / 2.0  # Diversity in binary trace
        independence_factors.append(uniqueness)
        
        # Factor 2: Non-redundancy assessment
        redundancy_check = 1.0
        if len(trace) >= 4:
            # Check for immediate repetitions (redundancy indicators)
            for i in range(len(trace) - 1):
                if i < len(trace) - 3 and trace[i:i+2] == trace[i+2:i+4]:
                    redundancy_check *= 0.8
        independence_factors.append(redundancy_check)
        
        # Factor 3: Axiom independence space coverage
        coverage = 0.4 + 0.6 * (value % 9) / 8.0
        independence_factors.append(coverage)
        
        independence_capacity = np.prod(independence_factors) ** (1.0 / len(independence_factors))
        return independence_capacity
        
    def _compute_completeness_degree(self, trace: str, value: int) -> float:
        """Completeness degree emerges from systematic axiom coverage"""
        completeness_factors = []
        
        # Factor 1: Coverage capacity
        coverage_cap = 0.3 + 0.7 * min(trace.count('0') / 4.0, 1.0)
        completeness_factors.append(coverage_cap)
        
        # Factor 2: Completeness scope
        scope = 0.6 + 0.4 * (len(trace) % 4) / 3.0
        completeness_factors.append(scope)
        
        # Factor 3: Systematic completeness
        systematic = 0.5 + 0.5 * (value % 6) / 5.0
        completeness_factors.append(systematic)
        
        completeness_degree = np.prod(completeness_factors) ** (1.0 / len(completeness_factors))
        return completeness_degree
        
    def _compute_axiom_consistency(self, trace: str, value: int) -> float:
        """Axiom consistency through foundation coherence"""
        consistency_base = 0.4 + 0.6 * min(len(trace) / 5.0, 1.0)
        value_modulation = 0.7 + 0.3 * cos(value * 0.4)
        return consistency_base * value_modulation
        
    def _compute_foundation_minimality(self, trace: str, value: int) -> float:
        """Foundation minimality through optimized axiom economy"""
        minimality_base = 0.6 + 0.4 * (1.0 - min(len(trace) / 8.0, 1.0))  # Shorter is more minimal
        phi_bonus = 0.15 if self._is_phi_valid(trace) else 0.0
        return min(minimality_base + phi_bonus, 1.0)
        
    def _compute_axiom_depth(self, trace: str, value: int) -> float:
        """Axiom depth through nested foundation analysis"""
        depth_factor = min(len(trace) / 10.0, 1.0)
        complexity_factor = (value % 11) / 10.0
        return 0.3 + 0.7 * (depth_factor * complexity_factor)
        
    def _compute_system_stability(self, trace: str, value: int) -> float:
        """System stability through consistent axiom architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.9
        else:
            stability_base = 0.3
        variation = 0.15 * sin(value * 0.25)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_axiom_coherence(self, trace: str, value: int) -> float:
        """Axiom coherence through unified foundation architecture"""
        coherence_base = 0.5 + 0.5 * min(trace.count('1') / 3.0, 1.0)
        structural_bonus = 0.3 if len(trace) >= 5 else 0.0
        return min(coherence_base + structural_bonus, 1.0)
        
    def _categorize_axiom(self, strength: float, power: float, 
                         independence: float, completeness: float) -> str:
        """Categorize trace based on axiom profile"""
        # Calculate dominant characteristic with thresholds
        strength_threshold = 0.6   # High axiom strength threshold
        power_threshold = 0.7      # High foundation power threshold
        independence_threshold = 0.5  # Moderate independence threshold
        
        if strength >= strength_threshold:
            if power >= power_threshold:
                return "foundational_axiom"     # High strength + high power
            elif independence >= independence_threshold:
                return "independent_axiom"      # High strength + independence
            else:
                return "strong_axiom"          # High strength + moderate properties
        else:
            if power >= power_threshold:
                return "generative_foundation"  # Moderate strength + high power
            elif independence >= independence_threshold:
                return "minimal_axiom"          # Moderate strength + independence
            else:
                return "basic_axiom"           # Basic axiomatic capability
        
    def _build_axiom_network(self) -> nx.Graph:
        """构建axiom network based on trace foundation similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on axiom similarity
        similarity_threshold = 0.25
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_axiom_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_axiom_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的axiom similarity"""
        # Axiom property similarity
        strength_diff = abs(trace1['axiom_strength'] - trace2['axiom_strength'])
        power_diff = abs(trace1['foundation_power'] - trace2['foundation_power'])
        independence_diff = abs(trace1['independence_capacity'] - trace2['independence_capacity'])
        completeness_diff = abs(trace1['completeness_degree'] - trace2['completeness_degree'])
        
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
        property_similarity = 1.0 - np.mean([strength_diff, power_diff, 
                                           independence_diff, completeness_diff])
        
        return 0.7 * property_similarity + 0.3 * geometric_similarity
        
    def _detect_foundation_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测axiom traces之间的foundation mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.25
        for node1 in self.axiom_network.nodes():
            data1 = self.axiom_network.nodes[node1]
            for node2 in self.axiom_network.nodes():
                if node1 != node2:
                    data2 = self.axiom_network.nodes[node2]
                    
                    # Check foundation preservation
                    power_preserved = abs(data1['foundation_power'] - 
                                        data2['foundation_power']) <= tolerance
                    strength_preserved = abs(data1['axiom_strength'] - 
                                           data2['axiom_strength']) <= tolerance
                    
                    if power_preserved and strength_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive axiom system analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.axiom_network)
        results['connected_components'] = nx.number_connected_components(self.axiom_network)
        
        # Axiom properties analysis
        axiom_strengths = [t['axiom_strength'] for t in traces]
        foundation_powers = [t['foundation_power'] for t in traces]
        independence_capacities = [t['independence_capacity'] for t in traces]
        completeness_degrees = [t['completeness_degree'] for t in traces]
        axiom_consistencies = [t['axiom_consistency'] for t in traces]
        foundation_minimalities = [t['foundation_minimality'] for t in traces]
        axiom_depths = [t['axiom_depth'] for t in traces]
        system_stabilities = [t['system_stability'] for t in traces]
        axiom_coherences = [t['axiom_coherence'] for t in traces]
        
        results['axiom_strength'] = {
            'mean': np.mean(axiom_strengths),
            'std': np.std(axiom_strengths),
            'high_count': sum(1 for x in axiom_strengths if x > 0.5)
        }
        results['foundation_power'] = {
            'mean': np.mean(foundation_powers),
            'std': np.std(foundation_powers),
            'high_count': sum(1 for x in foundation_powers if x > 0.5)
        }
        results['independence_capacity'] = {
            'mean': np.mean(independence_capacities),
            'std': np.std(independence_capacities),
            'high_count': sum(1 for x in independence_capacities if x > 0.5)
        }
        results['completeness_degree'] = {
            'mean': np.mean(completeness_degrees),
            'std': np.std(completeness_degrees),
            'high_count': sum(1 for x in completeness_degrees if x > 0.5)
        }
        results['axiom_consistency'] = {
            'mean': np.mean(axiom_consistencies),
            'std': np.std(axiom_consistencies),
            'high_count': sum(1 for x in axiom_consistencies if x > 0.5)
        }
        results['foundation_minimality'] = {
            'mean': np.mean(foundation_minimalities),
            'std': np.std(foundation_minimalities),
            'high_count': sum(1 for x in foundation_minimalities if x > 0.5)
        }
        results['axiom_depth'] = {
            'mean': np.mean(axiom_depths),
            'std': np.std(axiom_depths),
            'high_count': sum(1 for x in axiom_depths if x > 0.5)
        }
        results['system_stability'] = {
            'mean': np.mean(system_stabilities),
            'std': np.std(system_stabilities),
            'high_count': sum(1 for x in system_stabilities if x > 0.5)
        }
        results['axiom_coherence'] = {
            'mean': np.mean(axiom_coherences),
            'std': np.std(axiom_coherences),
            'high_count': sum(1 for x in axiom_coherences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.axiom_network.edges()) > 0:
            results['network_edges'] = len(self.axiom_network.edges())
            results['average_degree'] = sum(dict(self.axiom_network.degree()).values()) / len(self.axiom_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.foundation_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('axiom_strength', axiom_strengths),
            ('foundation_power', foundation_powers),
            ('independence_capacity', independence_capacities),
            ('completeness_degree', completeness_degrees),
            ('axiom_consistency', axiom_consistencies),
            ('foundation_minimality', foundation_minimalities),
            ('axiom_depth', axiom_depths),
            ('system_stability', system_stabilities),
            ('axiom_coherence', axiom_coherences)
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
        """生成axiom system visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Axiom Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 109: Collapse Axioms Dynamics', fontsize=16, fontweight='bold')
        
        # Axiom strength vs foundation power
        x = [t['axiom_strength'] for t in traces]
        y = [t['foundation_power'] for t in traces]
        colors = [t['independence_capacity'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Axiom Strength')
        ax1.set_ylabel('Foundation Power')
        ax1.set_title('Strength-Foundation Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Independence Capacity')
        
        # Independence capacity distribution
        independence_capacities = [t['independence_capacity'] for t in traces]
        ax2.hist(independence_capacities, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Independence Capacity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Independence Capacity Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Completeness degree vs axiom consistency
        x3 = [t['completeness_degree'] for t in traces]
        y3 = [t['axiom_consistency'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Completeness Degree')
        ax3.set_ylabel('Axiom Consistency')
        ax3.set_title('Completeness-Consistency Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Axiom Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-109-collapse-axioms-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 109: Axiom Foundation Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.axiom_network, k=1.5, iterations=50)
        node_colors = [traces[i]['axiom_strength'] for i in range(len(traces))]
        nx.draw(self.axiom_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=250, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Axiom Network Structure')
        
        # Degree distribution
        degrees = [self.axiom_network.degree(node) for node in self.axiom_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Axiom properties correlation matrix
        properties_matrix = np.array([
            [t['axiom_strength'] for t in traces],
            [t['foundation_power'] for t in traces],
            [t['independence_capacity'] for t in traces],
            [t['completeness_degree'] for t in traces],
            [t['axiom_consistency'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Strength', 'Power', 'Independence', 'Completeness', 'Consistency']
        im = ax3.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Axiom Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Foundation minimality vs axiom depth
        x4 = [t['foundation_minimality'] for t in traces]
        y4 = [t['axiom_depth'] for t in traces]
        stabilities = [t['system_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='coolwarm', alpha=0.7, s=60)
        ax4.set_xlabel('Foundation Minimality')
        ax4.set_ylabel('Axiom Depth')
        ax4.set_title('Minimality-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='System Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-109-collapse-axioms-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestCollapseAxioms(unittest.TestCase):
    """Unit tests for collapse axioms system"""
    
    def setUp(self):
        """Set up test collapse axioms system"""
        self.system = CollapseAxiomsSystem(max_trace_value=20, axiom_depth=4)
        
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
            self.assertIn('axiom_strength', data)
            self.assertIn('foundation_power', data)
            self.assertIn('independence_capacity', data)
            self.assertIn('completeness_degree', data)
            self.assertTrue(0 <= data['axiom_strength'] <= 1)
            self.assertTrue(0 <= data['foundation_power'] <= 1)
            
    def test_axiom_strength_computation(self):
        """测试axiom strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_axiom_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_foundation_power_computation(self):
        """测试foundation power computation"""
        trace = "1001"
        value = 8
        power = self.system._compute_foundation_power(trace, value)
        self.assertTrue(0 <= power <= 1)
        
    def test_axiom_network_construction(self):
        """测试axiom network construction"""
        self.assertGreater(len(self.system.axiom_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.axiom_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('axiom_strength', results)
        self.assertIn('foundation_power', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = CollapseAxiomsSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("COLLAPSE AXIOMS STRUCTURAL FOUNDATION ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Axiom Properties Analysis:")
    properties = ['axiom_strength', 'foundation_power', 'independence_capacity', 
                 'completeness_degree', 'axiom_consistency', 'foundation_minimality',
                 'axiom_depth', 'system_stability', 'axiom_coherence']
    
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
    print(f"Foundation Pattern Analysis:")
    print(f"Total foundation mappings: {sum(len(mappings) for mappings in system.foundation_mappings.values())}")
    print(f"Foundation mapping density: {results['morphism_density']:.3f}")
    print(f"Average foundations per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)