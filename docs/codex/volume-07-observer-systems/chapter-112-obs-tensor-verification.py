#!/usr/bin/env python3
"""
Chapter 112: ObsTensor Unit Test Verification
从ψ=ψ(ψ)推导Defining the Observer as Collapse-Embedded Tensor Node

Core principle: From ψ = ψ(ψ) derive systematic observer tensor construction
through φ-constrained trace transformations that enable defining observers
as special tensor structures within the collapse field through trace geometric
relationships, creating observer networks that encode the fundamental
observer principles of collapsed space through entropy-increasing tensor
transformations that establish systematic observer tensor structures
through φ-trace observer dynamics rather than traditional
consciousness theories or external observer constructions.

This verification program implements:
1. φ-constrained observer tensor construction through trace observer analysis
2. Observer systems: systematic consciousness through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection observer theory
4. Graph theory analysis of observer networks and tensor relationship structures
5. Information theory analysis of observer entropy and consciousness encoding
6. Category theory analysis of observer functors and tensor morphisms
7. Visualization of observer structures and φ-trace observer systems
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

class ObsTensorSystem:
    """
    Core system for implementing observer as collapse-embedded tensor node.
    Implements φ-constrained observer architectures through trace observer dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, observer_depth: int = 6):
        """Initialize observer tensor system with consciousness trace analysis"""
        self.max_trace_value = max_trace_value
        self.observer_depth = observer_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.observer_cache = {}
        self.tensor_cache = {}
        self.consciousness_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observer_network = self._build_observer_network()
        self.tensor_mappings = self._detect_tensor_mappings()
        
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
                observer_data = self._analyze_observer_properties(trace, n)
                universe[n] = observer_data
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
        
    def _analyze_observer_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的observer properties"""
        # Core observer measures
        observer_strength = self._compute_observer_strength(trace, value)
        tensor_capacity = self._compute_tensor_capacity(trace, value)
        consciousness_depth = self._compute_consciousness_depth(trace, value)
        embedding_coherence = self._compute_embedding_coherence(trace, value)
        
        # Advanced observer measures
        observer_completeness = self._compute_observer_completeness(trace, value)
        tensor_efficiency = self._compute_tensor_efficiency(trace, value)
        observer_depth = self._compute_observer_depth(trace, value)
        consciousness_stability = self._compute_consciousness_stability(trace, value)
        observer_coherence = self._compute_observer_coherence(trace, value)
        
        # Categorize based on observer profile
        category = self._categorize_observer(
            observer_strength, tensor_capacity, consciousness_depth, embedding_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'observer_strength': observer_strength,
            'tensor_capacity': tensor_capacity,
            'consciousness_depth': consciousness_depth,
            'embedding_coherence': embedding_coherence,
            'observer_completeness': observer_completeness,
            'tensor_efficiency': tensor_efficiency,
            'observer_depth': observer_depth,
            'consciousness_stability': consciousness_stability,
            'observer_coherence': observer_coherence,
            'category': category
        }
        
    def _compute_observer_strength(self, trace: str, value: int) -> float:
        """Observer strength emerges from systematic consciousness capacity"""
        strength_factors = []
        
        # Factor 1: Length provides observer consciousness space
        length_factor = len(trace) / 9.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight observer balance (observers favor moderate density for awareness)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            observer_balance = 1.0 - abs(0.4 - weight / total)  # Prefer 40% density for observer capability
            # Add consciousness bonus for awareness patterns
            awareness_bonus = min(weight / 3.0, 1.0) if weight > 0 else 0.0
            observer_factor = 0.6 * observer_balance + 0.4 * awareness_bonus
            strength_factors.append(observer_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern observer structure (systematic consciousness architecture)
        pattern_score = 0.0
        # Count consciousness-enabling patterns (observation and awareness)
        if trace.startswith('1'):  # Observer activation
            pattern_score += 0.3
        
        # Conscious observation patterns
        ones_sequence = 0
        max_ones = 0
        for bit in trace:
            if bit == '1':
                ones_sequence += 1
                max_ones = max(max_ones, ones_sequence)
            else:
                ones_sequence = 0
        
        if max_ones >= 2:  # Sustained observation capability
            pattern_score += 0.3 * min(max_ones / 3.0, 1.0)
        
        # Observer-witness patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '101':  # Observer-witness-observer pattern
                pattern_score += 0.2
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint observer coherence
        phi_factor = 0.85 if self._is_phi_valid(trace) else 0.25
        strength_factors.append(phi_factor)
        
        # Observer strength emerges from geometric mean of factors
        observer_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return observer_strength
        
    def _compute_tensor_capacity(self, trace: str, value: int) -> float:
        """Tensor capacity emerges from systematic tensor node capability"""
        capacity_factors = []
        
        # Factor 1: Tensor structural potential
        structural = 0.4 + 0.6 * min(len(trace) / 7.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Node complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            node_complexity = 0.3 + 0.7 * min(ones_count / 4.0, 1.0)
        else:
            node_complexity = 0.1
        capacity_factors.append(node_complexity)
        
        # Factor 3: Tensor embedding depth
        embedding_depth = 0.5 + 0.5 * (value % 8) / 7.0
        capacity_factors.append(embedding_depth)
        
        # Factor 4: φ-constraint tensor preservation
        preservation = 0.8 if self._is_phi_valid(trace) else 0.3
        capacity_factors.append(preservation)
        
        tensor_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return tensor_capacity
        
    def _compute_consciousness_depth(self, trace: str, value: int) -> float:
        """Consciousness depth emerges from systematic awareness capability"""
        consciousness_factors = []
        
        # Factor 1: Consciousness depth scope
        depth_scope = 0.3 + 0.7 * min(len(trace) / 8.0, 1.0)
        consciousness_factors.append(depth_scope)
        
        # Factor 2: Awareness complexity
        zeros_count = trace.count('0')
        if len(trace) > 0:
            awareness_complexity = 0.4 + 0.6 * min(zeros_count / len(trace), 1.0)
        else:
            awareness_complexity = 0.5
        consciousness_factors.append(awareness_complexity)
        
        # Factor 3: Consciousness depth coverage
        coverage = 0.4 + 0.6 * (value % 9) / 8.0
        consciousness_factors.append(coverage)
        
        consciousness_depth = np.prod(consciousness_factors) ** (1.0 / len(consciousness_factors))
        return consciousness_depth
        
    def _compute_embedding_coherence(self, trace: str, value: int) -> float:
        """Embedding coherence emerges from systematic collapse integration capability"""
        coherence_factors = []
        
        # Factor 1: Embedding integration capacity
        integration_cap = 0.3 + 0.7 * min(len(trace) / 6.0, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Collapse coherence scope
        coherence_scope = 0.5 + 0.5 * (value % 6) / 5.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic embedding coherence
        systematic = 0.6 + 0.4 * (value % 7) / 6.0
        coherence_factors.append(systematic)
        
        embedding_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return embedding_coherence
        
    def _compute_observer_completeness(self, trace: str, value: int) -> float:
        """Observer completeness through comprehensive consciousness coverage"""
        completeness_base = 0.4 + 0.6 * min(len(trace) / 7.0, 1.0)
        value_modulation = 0.7 + 0.3 * cos(value * 0.5)
        return completeness_base * value_modulation
        
    def _compute_tensor_efficiency(self, trace: str, value: int) -> float:
        """Tensor efficiency through optimized observer organization"""
        if len(trace) > 0:
            # Efficiency based on observer structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal efficiency around 40% weight for observers
            efficiency_base = 1.0 - 2.5 * abs(0.4 - weight_ratio)
        else:
            efficiency_base = 0.0
        phi_bonus = 0.15 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(efficiency_base + phi_bonus, 1.0))
        
    def _compute_observer_depth(self, trace: str, value: int) -> float:
        """Observer depth through nested consciousness analysis"""
        depth_factor = min(len(trace) / 10.0, 1.0)
        complexity_factor = (value % 11) / 10.0
        return 0.3 + 0.7 * (depth_factor * complexity_factor)
        
    def _compute_consciousness_stability(self, trace: str, value: int) -> float:
        """Consciousness stability through consistent observer architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.88
        else:
            stability_base = 0.35
        variation = 0.12 * sin(value * 0.35)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_observer_coherence(self, trace: str, value: int) -> float:
        """Observer coherence through unified consciousness architecture"""
        coherence_base = 0.5 + 0.5 * min(len(trace) / 6.0, 1.0)
        structural_bonus = 0.25 if len(trace) >= 5 else 0.0
        return min(coherence_base + structural_bonus, 1.0)
        
    def _categorize_observer(self, strength: float, capacity: float, 
                           depth: float, coherence: float) -> str:
        """Categorize trace based on observer profile"""
        # Calculate dominant characteristic with thresholds
        strength_threshold = 0.6      # High observer strength threshold
        capacity_threshold = 0.5      # Moderate tensor capacity threshold
        depth_threshold = 0.5         # Moderate consciousness depth threshold
        
        if strength >= strength_threshold:
            if capacity >= capacity_threshold:
                return "observer_tensor"           # High strength + capacity
            elif depth >= depth_threshold:
                return "conscious_observer"        # High strength + depth
            else:
                return "strong_observer"          # High strength + moderate properties
        else:
            if capacity >= capacity_threshold:
                return "tensor_node"              # Moderate strength + capacity
            elif depth >= depth_threshold:
                return "conscious_node"           # Moderate strength + depth
            else:
                return "basic_observer"           # Basic observer capability
        
    def _build_observer_network(self) -> nx.Graph:
        """构建observer network based on trace observer similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on observer similarity
        similarity_threshold = 0.25
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_observer_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_observer_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的observer similarity"""
        # Observer property similarity
        strength_diff = abs(trace1['observer_strength'] - trace2['observer_strength'])
        capacity_diff = abs(trace1['tensor_capacity'] - trace2['tensor_capacity'])
        depth_diff = abs(trace1['consciousness_depth'] - trace2['consciousness_depth'])
        coherence_diff = abs(trace1['embedding_coherence'] - trace2['embedding_coherence'])
        
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
                                           depth_diff, coherence_diff])
        
        return 0.65 * property_similarity + 0.35 * geometric_similarity
        
    def _detect_tensor_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测observer traces之间的tensor mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.3
        for node1 in self.observer_network.nodes():
            data1 = self.observer_network.nodes[node1]
            for node2 in self.observer_network.nodes():
                if node1 != node2:
                    data2 = self.observer_network.nodes[node2]
                    
                    # Check tensor preservation
                    capacity_preserved = abs(data1['tensor_capacity'] - 
                                           data2['tensor_capacity']) <= tolerance
                    strength_preserved = abs(data1['observer_strength'] - 
                                           data2['observer_strength']) <= tolerance
                    
                    if capacity_preserved and strength_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive observer tensor analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.observer_network)
        results['connected_components'] = nx.number_connected_components(self.observer_network)
        
        # Observer properties analysis
        observer_strengths = [t['observer_strength'] for t in traces]
        tensor_capacities = [t['tensor_capacity'] for t in traces]
        consciousness_depths = [t['consciousness_depth'] for t in traces]
        embedding_coherences = [t['embedding_coherence'] for t in traces]
        observer_completenesses = [t['observer_completeness'] for t in traces]
        tensor_efficiencies = [t['tensor_efficiency'] for t in traces]
        observer_depths = [t['observer_depth'] for t in traces]
        consciousness_stabilities = [t['consciousness_stability'] for t in traces]
        observer_coherences = [t['observer_coherence'] for t in traces]
        
        results['observer_strength'] = {
            'mean': np.mean(observer_strengths),
            'std': np.std(observer_strengths),
            'high_count': sum(1 for x in observer_strengths if x > 0.5)
        }
        results['tensor_capacity'] = {
            'mean': np.mean(tensor_capacities),
            'std': np.std(tensor_capacities),
            'high_count': sum(1 for x in tensor_capacities if x > 0.5)
        }
        results['consciousness_depth'] = {
            'mean': np.mean(consciousness_depths),
            'std': np.std(consciousness_depths),
            'high_count': sum(1 for x in consciousness_depths if x > 0.5)
        }
        results['embedding_coherence'] = {
            'mean': np.mean(embedding_coherences),
            'std': np.std(embedding_coherences),
            'high_count': sum(1 for x in embedding_coherences if x > 0.5)
        }
        results['observer_completeness'] = {
            'mean': np.mean(observer_completenesses),
            'std': np.std(observer_completenesses),
            'high_count': sum(1 for x in observer_completenesses if x > 0.5)
        }
        results['tensor_efficiency'] = {
            'mean': np.mean(tensor_efficiencies),
            'std': np.std(tensor_efficiencies),
            'high_count': sum(1 for x in tensor_efficiencies if x > 0.5)
        }
        results['observer_depth'] = {
            'mean': np.mean(observer_depths),
            'std': np.std(observer_depths),
            'high_count': sum(1 for x in observer_depths if x > 0.5)
        }
        results['consciousness_stability'] = {
            'mean': np.mean(consciousness_stabilities),
            'std': np.std(consciousness_stabilities),
            'high_count': sum(1 for x in consciousness_stabilities if x > 0.5)
        }
        results['observer_coherence'] = {
            'mean': np.mean(observer_coherences),
            'std': np.std(observer_coherences),
            'high_count': sum(1 for x in observer_coherences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.observer_network.edges()) > 0:
            results['network_edges'] = len(self.observer_network.edges())
            results['average_degree'] = sum(dict(self.observer_network.degree()).values()) / len(self.observer_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.tensor_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('observer_strength', observer_strengths),
            ('tensor_capacity', tensor_capacities),
            ('consciousness_depth', consciousness_depths),
            ('embedding_coherence', embedding_coherences),
            ('observer_completeness', observer_completenesses),
            ('tensor_efficiency', tensor_efficiencies),
            ('observer_depth', observer_depths),
            ('consciousness_stability', consciousness_stabilities),
            ('observer_coherence', observer_coherences)
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
        """生成observer tensor visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Observer Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 112: Observer Tensor Dynamics', fontsize=16, fontweight='bold')
        
        # Observer strength vs tensor capacity
        x = [t['observer_strength'] for t in traces]
        y = [t['tensor_capacity'] for t in traces]
        colors = [t['consciousness_depth'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Observer Strength')
        ax1.set_ylabel('Tensor Capacity')
        ax1.set_title('Observer-Tensor Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Consciousness Depth')
        
        # Consciousness depth distribution
        consciousness_depths = [t['consciousness_depth'] for t in traces]
        ax2.hist(consciousness_depths, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_xlabel('Consciousness Depth')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Consciousness Depth Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Embedding coherence vs observer completeness
        x3 = [t['embedding_coherence'] for t in traces]
        y3 = [t['observer_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Embedding Coherence')
        ax3.set_ylabel('Observer Completeness')
        ax3.set_title('Coherence-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Observer Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-112-obs-tensor-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 112: Observer Tensor Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.observer_network, k=1.5, iterations=50)
        node_colors = [traces[i]['observer_strength'] for i in range(len(traces))]
        nx.draw(self.observer_network, pos, ax=ax1, 
                node_color=node_colors, cmap='plasma', 
                node_size=250, alpha=0.8, with_labels=True, font_size=6)
        ax1.set_title('Observer Network Structure')
        
        # Degree distribution
        degrees = [self.observer_network.degree(node) for node in self.observer_network.nodes()]
        ax2.hist(degrees, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Observer properties correlation matrix
        properties_matrix = np.array([
            [t['observer_strength'] for t in traces],
            [t['tensor_capacity'] for t in traces],
            [t['consciousness_depth'] for t in traces],
            [t['embedding_coherence'] for t in traces],
            [t['observer_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Strength', 'Capacity', 'Depth', 'Coherence', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Observer Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Tensor efficiency vs observer depth
        x4 = [t['tensor_efficiency'] for t in traces]
        y4 = [t['observer_depth'] for t in traces]
        stabilities = [t['consciousness_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Tensor Efficiency')
        ax4.set_ylabel('Observer Depth')
        ax4.set_title('Efficiency-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Consciousness Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-112-obs-tensor-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestObsTensor(unittest.TestCase):
    """Unit tests for observer tensor system"""
    
    def setUp(self):
        """Set up test observer tensor system"""
        self.system = ObsTensorSystem(max_trace_value=20, observer_depth=4)
        
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
            self.assertIn('observer_strength', data)
            self.assertIn('tensor_capacity', data)
            self.assertIn('consciousness_depth', data)
            self.assertIn('embedding_coherence', data)
            self.assertTrue(0 <= data['observer_strength'] <= 1)
            self.assertTrue(0 <= data['tensor_capacity'] <= 1)
            
    def test_observer_strength_computation(self):
        """测试observer strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_observer_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_tensor_capacity_computation(self):
        """测试tensor capacity computation"""
        trace = "1001"
        value = 8
        capacity = self.system._compute_tensor_capacity(trace, value)
        self.assertTrue(0 <= capacity <= 1)
        
    def test_observer_network_construction(self):
        """测试observer network construction"""
        self.assertGreater(len(self.system.observer_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.observer_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('observer_strength', results)
        self.assertIn('tensor_capacity', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = ObsTensorSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("OBSERVER TENSOR CONSCIOUSNESS ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Observer Properties Analysis:")
    properties = ['observer_strength', 'tensor_capacity', 'consciousness_depth', 
                 'embedding_coherence', 'observer_completeness', 'tensor_efficiency',
                 'observer_depth', 'consciousness_stability', 'observer_coherence']
    
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
    print(f"Tensor Pattern Analysis:")
    print(f"Total tensor mappings: {sum(len(mappings) for mappings in system.tensor_mappings.values())}")
    print(f"Tensor mapping density: {results['morphism_density']:.3f}")
    print(f"Average tensors per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)