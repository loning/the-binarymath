#!/usr/bin/env python3
"""
Chapter 111: CollapseEntropyTheorem Unit Test Verification
从ψ=ψ(ψ)推导Entropy-Bounded Meta-Structure Collapse Limit

Core principle: From ψ = ψ(ψ) derive systematic entropy-bounded meta-structure
analysis through φ-constrained trace transformations that enable understanding
entropy limits on self-description complexity through trace geometric
relationships, creating entropy networks that encode the fundamental
entropy principles of collapsed space through entropy-increasing tensor
transformations that establish systematic entropy-bounded structures
through φ-trace entropy dynamics rather than traditional
thermodynamic entropy or external information theoretical constructions.

This verification program implements:
1. φ-constrained entropy construction through trace meta-structure analysis
2. Entropy-bounded systems: systematic complexity limitation through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection entropy theory
4. Graph theory analysis of entropy networks and meta-structure relationship structures
5. Information theory analysis of entropy entropy and complexity encoding
6. Category theory analysis of entropy functors and meta-structure morphisms
7. Visualization of entropy structures and φ-trace entropy systems
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

class CollapseEntropyTheoremSystem:
    """
    Core system for implementing entropy-bounded meta-structure collapse limit.
    Implements φ-constrained entropy architectures through trace entropy dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, entropy_depth: int = 6):
        """Initialize collapse entropy theorem system with meta-structure trace analysis"""
        self.max_trace_value = max_trace_value
        self.entropy_depth = entropy_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.entropy_cache = {}
        self.meta_structure_cache = {}
        self.complexity_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.entropy_network = self._build_entropy_network()
        self.meta_structure_mappings = self._detect_meta_structure_mappings()
        
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
                entropy_data = self._analyze_entropy_properties(trace, n)
                universe[n] = entropy_data
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
        
    def _analyze_entropy_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的entropy properties"""
        # Core entropy measures
        entropy_strength = self._compute_entropy_strength(trace, value)
        meta_structure_capacity = self._compute_meta_structure_capacity(trace, value)
        complexity_limitation = self._compute_complexity_limitation(trace, value)
        entropy_boundary = self._compute_entropy_boundary(trace, value)
        
        # Advanced entropy measures
        entropy_completeness = self._compute_entropy_completeness(trace, value)
        meta_structure_efficiency = self._compute_meta_structure_efficiency(trace, value)
        entropy_depth = self._compute_entropy_depth(trace, value)
        complexity_stability = self._compute_complexity_stability(trace, value)
        entropy_coherence = self._compute_entropy_coherence(trace, value)
        
        # Categorize based on entropy profile
        category = self._categorize_entropy(
            entropy_strength, meta_structure_capacity, complexity_limitation, entropy_boundary
        )
        
        return {
            'trace': trace,
            'value': value,
            'entropy_strength': entropy_strength,
            'meta_structure_capacity': meta_structure_capacity,
            'complexity_limitation': complexity_limitation,
            'entropy_boundary': entropy_boundary,
            'entropy_completeness': entropy_completeness,
            'meta_structure_efficiency': meta_structure_efficiency,
            'entropy_depth': entropy_depth,
            'complexity_stability': complexity_stability,
            'entropy_coherence': entropy_coherence,
            'category': category
        }
        
    def _compute_entropy_strength(self, trace: str, value: int) -> float:
        """Entropy strength emerges from systematic complexity boundary capacity"""
        strength_factors = []
        
        # Factor 1: Length provides entropy complexity space
        length_factor = len(trace) / 10.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight entropy balance (entropy favors balanced structures with variation)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            entropy_balance = 1.0 - abs(0.5 - weight / total)  # Prefer 50% density for maximum entropy
            # Add entropy bonus for variation
            variation_bonus = min(weight * (total - weight) / (total * total / 4.0), 1.0) if total > 0 else 0.0
            entropy_factor = 0.7 * entropy_balance + 0.3 * variation_bonus
            strength_factors.append(entropy_factor)
        else:
            strength_factors.append(0.3)
        
        # Factor 3: Pattern entropy structure (systematic complexity architecture)
        pattern_score = 0.0
        # Count entropy-increasing patterns (transitions and complexity)
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions increase entropy
                transitions += 1
        if len(trace) > 1:
            transition_entropy = transitions / (len(trace) - 1)
            pattern_score += 0.5 * transition_entropy
        
        # Complexity patterns
        if len(set(trace)) > 1:  # Binary diversity
            pattern_score += 0.3
        if len(trace) >= 3:
            # Look for non-repeating subpatterns
            unique_subpatterns = set()
            for i in range(len(trace) - 2):
                unique_subpatterns.add(trace[i:i+3])
            if len(unique_subpatterns) > 1:
                pattern_score += 0.2
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint entropy coherence
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.2
        strength_factors.append(phi_factor)
        
        # Entropy strength emerges from geometric mean of factors
        entropy_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return entropy_strength
        
    def _compute_meta_structure_capacity(self, trace: str, value: int) -> float:
        """Meta-structure capacity emerges from systematic self-organization capability"""
        capacity_factors = []
        
        # Factor 1: Meta-structure organizational potential
        organizational = 0.5 + 0.5 * min(len(trace) / 8.0, 1.0)
        capacity_factors.append(organizational)
        
        # Factor 2: Structure complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.4 + 0.6 * min(ones_count / 5.0, 1.0)
        else:
            complexity = 0.1
        capacity_factors.append(complexity)
        
        # Factor 3: Meta-structure organization depth
        organization_depth = 0.3 + 0.7 * (value % 9) / 8.0
        capacity_factors.append(organization_depth)
        
        # Factor 4: φ-constraint meta-structure preservation
        preservation = 0.8 if self._is_phi_valid(trace) else 0.3
        capacity_factors.append(preservation)
        
        meta_structure_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return meta_structure_capacity
        
    def _compute_complexity_limitation(self, trace: str, value: int) -> float:
        """Complexity limitation emerges from systematic complexity boundary capability"""
        limitation_factors = []
        
        # Factor 1: Complexity boundary scope
        boundary_scope = 0.4 + 0.6 * (len(trace) / 7.0)  # Longer traces have more boundary
        limitation_factors.append(min(boundary_scope, 1.0))
        
        # Factor 2: Limitation efficiency
        zeros_count = trace.count('0')
        if len(trace) > 0:
            limitation_efficiency = 0.3 + 0.7 * min(zeros_count / len(trace), 1.0)
        else:
            limitation_efficiency = 0.5
        limitation_factors.append(limitation_efficiency)
        
        # Factor 3: Complexity boundary coverage
        coverage = 0.4 + 0.6 * (value % 11) / 10.0
        limitation_factors.append(coverage)
        
        complexity_limitation = np.prod(limitation_factors) ** (1.0 / len(limitation_factors))
        return complexity_limitation
        
    def _compute_entropy_boundary(self, trace: str, value: int) -> float:
        """Entropy boundary emerges from systematic entropy limitation capability"""
        boundary_factors = []
        
        # Factor 1: Entropy boundary capacity
        boundary_cap = 0.2 + 0.8 * min(len(trace) / 9.0, 1.0)
        boundary_factors.append(boundary_cap)
        
        # Factor 2: Boundary entropy scope
        boundary_scope = 0.5 + 0.5 * (value % 5) / 4.0
        boundary_factors.append(boundary_scope)
        
        # Factor 3: Systematic boundary entropy
        systematic = 0.6 + 0.4 * (value % 7) / 6.0
        boundary_factors.append(systematic)
        
        entropy_boundary = np.prod(boundary_factors) ** (1.0 / len(boundary_factors))
        return entropy_boundary
        
    def _compute_entropy_completeness(self, trace: str, value: int) -> float:
        """Entropy completeness through comprehensive complexity coverage"""
        completeness_base = 0.5 + 0.5 * min(len(trace) / 6.0, 1.0)
        value_modulation = 0.8 + 0.2 * cos(value * 0.6)
        return completeness_base * value_modulation
        
    def _compute_meta_structure_efficiency(self, trace: str, value: int) -> float:
        """Meta-structure efficiency through optimized entropy organization"""
        if len(trace) > 0:
            # Efficiency based on entropy maximization
            weight_ratio = trace.count('1') / len(trace)
            # Maximum efficiency at 50% weight (maximum entropy)
            efficiency_base = 1.0 - 2.0 * abs(0.5 - weight_ratio)
        else:
            efficiency_base = 0.0
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return min(efficiency_base + phi_bonus, 1.0)
        
    def _compute_entropy_depth(self, trace: str, value: int) -> float:
        """Entropy depth through nested complexity analysis"""
        depth_factor = min(len(trace) / 12.0, 1.0)
        complexity_factor = (value % 13) / 12.0
        return 0.3 + 0.7 * (depth_factor * complexity_factor)
        
    def _compute_complexity_stability(self, trace: str, value: int) -> float:
        """Complexity stability through consistent entropy architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.9
        else:
            stability_base = 0.4
        variation = 0.1 * sin(value * 0.4)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_entropy_coherence(self, trace: str, value: int) -> float:
        """Entropy coherence through unified complexity architecture"""
        coherence_base = 0.6 + 0.4 * min(len(trace) / 5.0, 1.0)
        structural_bonus = 0.2 if len(trace) >= 6 else 0.0
        return min(coherence_base + structural_bonus, 1.0)
        
    def _categorize_entropy(self, strength: float, capacity: float, 
                          limitation: float, boundary: float) -> str:
        """Categorize trace based on entropy profile"""
        # Calculate dominant characteristic with thresholds
        strength_threshold = 0.7      # High entropy strength threshold
        capacity_threshold = 0.6      # High meta-structure capacity threshold
        limitation_threshold = 0.5    # Moderate complexity limitation threshold
        
        if strength >= strength_threshold:
            if capacity >= capacity_threshold:
                return "entropy_maximizing"        # High strength + high capacity
            elif limitation >= limitation_threshold:
                return "entropy_bounded"           # High strength + limitation
            else:
                return "high_entropy"             # High strength + moderate properties
        else:
            if capacity >= capacity_threshold:
                return "meta_structure_organized"  # Moderate strength + high capacity
            elif limitation >= limitation_threshold:
                return "complexity_limited"       # Moderate strength + limitation
            else:
                return "basic_entropy"            # Basic entropy capability
        
    def _build_entropy_network(self) -> nx.Graph:
        """构建entropy network based on trace entropy similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on entropy similarity
        similarity_threshold = 0.2
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_entropy_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_entropy_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的entropy similarity"""
        # Entropy property similarity
        strength_diff = abs(trace1['entropy_strength'] - trace2['entropy_strength'])
        capacity_diff = abs(trace1['meta_structure_capacity'] - trace2['meta_structure_capacity'])
        limitation_diff = abs(trace1['complexity_limitation'] - trace2['complexity_limitation'])
        boundary_diff = abs(trace1['entropy_boundary'] - trace2['entropy_boundary'])
        
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
        property_similarity = 1.0 - np.mean([strength_diff, capacity_diff, 
                                           limitation_diff, boundary_diff])
        
        return 0.6 * property_similarity + 0.4 * geometric_similarity
        
    def _detect_meta_structure_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测entropy traces之间的meta-structure mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.25
        for node1 in self.entropy_network.nodes():
            data1 = self.entropy_network.nodes[node1]
            for node2 in self.entropy_network.nodes():
                if node1 != node2:
                    data2 = self.entropy_network.nodes[node2]
                    
                    # Check meta-structure preservation
                    capacity_preserved = abs(data1['meta_structure_capacity'] - 
                                           data2['meta_structure_capacity']) <= tolerance
                    strength_preserved = abs(data1['entropy_strength'] - 
                                           data2['entropy_strength']) <= tolerance
                    
                    if capacity_preserved and strength_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive entropy theorem analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.entropy_network)
        results['connected_components'] = nx.number_connected_components(self.entropy_network)
        
        # Entropy properties analysis
        entropy_strengths = [t['entropy_strength'] for t in traces]
        meta_structure_capacities = [t['meta_structure_capacity'] for t in traces]
        complexity_limitations = [t['complexity_limitation'] for t in traces]
        entropy_boundaries = [t['entropy_boundary'] for t in traces]
        entropy_completenesses = [t['entropy_completeness'] for t in traces]
        meta_structure_efficiencies = [t['meta_structure_efficiency'] for t in traces]
        entropy_depths = [t['entropy_depth'] for t in traces]
        complexity_stabilities = [t['complexity_stability'] for t in traces]
        entropy_coherences = [t['entropy_coherence'] for t in traces]
        
        results['entropy_strength'] = {
            'mean': np.mean(entropy_strengths),
            'std': np.std(entropy_strengths),
            'high_count': sum(1 for x in entropy_strengths if x > 0.5)
        }
        results['meta_structure_capacity'] = {
            'mean': np.mean(meta_structure_capacities),
            'std': np.std(meta_structure_capacities),
            'high_count': sum(1 for x in meta_structure_capacities if x > 0.5)
        }
        results['complexity_limitation'] = {
            'mean': np.mean(complexity_limitations),
            'std': np.std(complexity_limitations),
            'high_count': sum(1 for x in complexity_limitations if x > 0.5)
        }
        results['entropy_boundary'] = {
            'mean': np.mean(entropy_boundaries),
            'std': np.std(entropy_boundaries),
            'high_count': sum(1 for x in entropy_boundaries if x > 0.5)
        }
        results['entropy_completeness'] = {
            'mean': np.mean(entropy_completenesses),
            'std': np.std(entropy_completenesses),
            'high_count': sum(1 for x in entropy_completenesses if x > 0.5)
        }
        results['meta_structure_efficiency'] = {
            'mean': np.mean(meta_structure_efficiencies),
            'std': np.std(meta_structure_efficiencies),
            'high_count': sum(1 for x in meta_structure_efficiencies if x > 0.5)
        }
        results['entropy_depth'] = {
            'mean': np.mean(entropy_depths),
            'std': np.std(entropy_depths),
            'high_count': sum(1 for x in entropy_depths if x > 0.5)
        }
        results['complexity_stability'] = {
            'mean': np.mean(complexity_stabilities),
            'std': np.std(complexity_stabilities),
            'high_count': sum(1 for x in complexity_stabilities if x > 0.5)
        }
        results['entropy_coherence'] = {
            'mean': np.mean(entropy_coherences),
            'std': np.std(entropy_coherences),
            'high_count': sum(1 for x in entropy_coherences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.entropy_network.edges()) > 0:
            results['network_edges'] = len(self.entropy_network.edges())
            results['average_degree'] = sum(dict(self.entropy_network.degree()).values()) / len(self.entropy_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.meta_structure_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('entropy_strength', entropy_strengths),
            ('meta_structure_capacity', meta_structure_capacities),
            ('complexity_limitation', complexity_limitations),
            ('entropy_boundary', entropy_boundaries),
            ('entropy_completeness', entropy_completenesses),
            ('meta_structure_efficiency', meta_structure_efficiencies),
            ('entropy_depth', entropy_depths),
            ('complexity_stability', complexity_stabilities),
            ('entropy_coherence', entropy_coherences)
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
        """生成entropy theorem visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Entropy Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 111: Collapse Entropy Theorem Dynamics', fontsize=16, fontweight='bold')
        
        # Entropy strength vs meta-structure capacity
        x = [t['entropy_strength'] for t in traces]
        y = [t['meta_structure_capacity'] for t in traces]
        colors = [t['complexity_limitation'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Entropy Strength')
        ax1.set_ylabel('Meta-Structure Capacity')
        ax1.set_title('Entropy-MetaStructure Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Complexity Limitation')
        
        # Complexity limitation distribution
        complexity_limitations = [t['complexity_limitation'] for t in traces]
        ax2.hist(complexity_limitations, bins=15, alpha=0.7, color='gold', edgecolor='black')
        ax2.set_xlabel('Complexity Limitation')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Complexity Limitation Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Entropy boundary vs entropy completeness
        x3 = [t['entropy_boundary'] for t in traces]
        y3 = [t['entropy_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Entropy Boundary')
        ax3.set_ylabel('Entropy Completeness')
        ax3.set_title('Boundary-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Entropy Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-111-collapse-entropy-theorem-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 111: Entropy Meta-Structure Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.entropy_network, k=2.0, iterations=50)
        node_colors = [traces[i]['entropy_strength'] for i in range(len(traces))]
        nx.draw(self.entropy_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=300, alpha=0.8, with_labels=True, font_size=7)
        ax1.set_title('Entropy Network Structure')
        
        # Degree distribution
        degrees = [self.entropy_network.degree(node) for node in self.entropy_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='mediumorchid', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Entropy properties correlation matrix
        properties_matrix = np.array([
            [t['entropy_strength'] for t in traces],
            [t['meta_structure_capacity'] for t in traces],
            [t['complexity_limitation'] for t in traces],
            [t['entropy_boundary'] for t in traces],
            [t['entropy_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Strength', 'Capacity', 'Limitation', 'Boundary', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='viridis', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Entropy Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Meta-structure efficiency vs entropy depth
        x4 = [t['meta_structure_efficiency'] for t in traces]
        y4 = [t['entropy_depth'] for t in traces]
        stabilities = [t['complexity_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='cividis', alpha=0.7, s=60)
        ax4.set_xlabel('Meta-Structure Efficiency')
        ax4.set_ylabel('Entropy Depth')
        ax4.set_title('Efficiency-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Complexity Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-111-collapse-entropy-theorem-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestCollapseEntropyTheorem(unittest.TestCase):
    """Unit tests for collapse entropy theorem system"""
    
    def setUp(self):
        """Set up test collapse entropy theorem system"""
        self.system = CollapseEntropyTheoremSystem(max_trace_value=20, entropy_depth=4)
        
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
            self.assertIn('entropy_strength', data)
            self.assertIn('meta_structure_capacity', data)
            self.assertIn('complexity_limitation', data)
            self.assertIn('entropy_boundary', data)
            self.assertTrue(0 <= data['entropy_strength'] <= 1)
            self.assertTrue(0 <= data['meta_structure_capacity'] <= 1)
            
    def test_entropy_strength_computation(self):
        """测试entropy strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_entropy_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_meta_structure_capacity_computation(self):
        """测试meta-structure capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_meta_structure_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_entropy_network_construction(self):
        """测试entropy network construction"""
        self.assertGreater(len(self.system.entropy_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.entropy_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('entropy_strength', results)
        self.assertIn('meta_structure_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = CollapseEntropyTheoremSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("COLLAPSE ENTROPY THEOREM META-STRUCTURE ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Entropy Properties Analysis:")
    properties = ['entropy_strength', 'meta_structure_capacity', 'complexity_limitation', 
                 'entropy_boundary', 'entropy_completeness', 'meta_structure_efficiency',
                 'entropy_depth', 'complexity_stability', 'entropy_coherence']
    
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
    print(f"Meta-Structure Pattern Analysis:")
    print(f"Total meta-structure mappings: {sum(len(mappings) for mappings in system.meta_structure_mappings.values())}")
    print(f"Meta-structure mapping density: {results['morphism_density']:.3f}")
    print(f"Average structures per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)