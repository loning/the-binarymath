#!/usr/bin/env python3
"""
Chapter 108: ConsistencySpectrum Unit Test Verification
从ψ=ψ(ψ)推导Collapse Inconsistency Spectrum and Trace Conflict Detection

Core principle: From ψ = ψ(ψ) derive systematic inconsistency spectrum analysis
through φ-constrained trace transformations that enable systematic conflict detection
and consistency measurement through trace geometric relationships, creating
inconsistency networks that encode the fundamental consistency principles of
collapsed space through entropy-increasing tensor transformations that establish
systematic inconsistency analysis through φ-trace conflict dynamics rather than
traditional syntactic consistency checks or external logical verification.

This verification program implements:
1. φ-constrained inconsistency spectrum construction through trace conflict analysis
2. Consistency measurement: systematic conflict detection through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection consistency theory
4. Graph theory analysis of inconsistency networks and conflict relationship structures
5. Information theory analysis of consistency entropy and conflict detection encoding
6. Category theory analysis of consistency functors and inconsistency morphisms
7. Visualization of inconsistency structures and φ-trace conflict systems
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

class ConsistencySpectrumSystem:
    """
    Core system for implementing collapse inconsistency spectrum and trace conflict detection.
    Implements φ-constrained inconsistency architectures through trace conflict dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, consistency_depth: int = 6):
        """Initialize consistency spectrum system with conflict trace analysis"""
        self.max_trace_value = max_trace_value
        self.consistency_depth = consistency_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.consistency_cache = {}
        self.conflict_cache = {}
        self.spectrum_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.consistency_network = self._build_consistency_network()
        self.conflict_mappings = self._detect_conflict_mappings()
        
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
                consistency_data = self._analyze_consistency_properties(trace, n)
                universe[n] = consistency_data
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
        
    def _analyze_consistency_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的consistency properties"""
        # Core consistency measures
        consistency_strength = self._compute_consistency_strength(trace, value)
        conflict_detection = self._compute_conflict_detection(trace, value)
        spectrum_range = self._compute_spectrum_range(trace, value)
        inconsistency_tolerance = self._compute_inconsistency_tolerance(trace, value)
        
        # Advanced consistency measures
        consistency_completeness = self._compute_consistency_completeness(trace, value)
        conflict_efficiency = self._compute_conflict_efficiency(trace, value)
        consistency_depth = self._compute_consistency_depth(trace, value)
        spectrum_stability = self._compute_spectrum_stability(trace, value)
        consistency_coherence = self._compute_consistency_coherence(trace, value)
        
        # Categorize based on consistency profile
        category = self._categorize_consistency(
            consistency_strength, conflict_detection, spectrum_range, inconsistency_tolerance
        )
        
        return {
            'trace': trace,
            'value': value,
            'consistency_strength': consistency_strength,
            'conflict_detection': conflict_detection,
            'spectrum_range': spectrum_range,
            'inconsistency_tolerance': inconsistency_tolerance,
            'consistency_completeness': consistency_completeness,
            'conflict_efficiency': conflict_efficiency,
            'consistency_depth': consistency_depth,
            'spectrum_stability': spectrum_stability,
            'consistency_coherence': consistency_coherence,
            'category': category
        }
        
    def _compute_consistency_strength(self, trace: str, value: int) -> float:
        """Consistency strength emerges from systematic conflict resolution capacity"""
        strength_factors = []
        
        # Factor 1: Length provides conflict resolution space
        length_factor = len(trace) / 10.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight balance (optimal for consistency operations)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            balance = 1.0 - abs(0.5 - weight / total)
            strength_factors.append(balance)
        else:
            strength_factors.append(0.5)
        
        # Factor 3: Pattern regularity (systematic consistency structure)
        pattern_score = 0.0
        for i in range(len(trace) - 1):
            if trace[i] == '0' and trace[i+1] == '1':  # Rising edge
                pattern_score += 0.3
            elif trace[i] == '1' and trace[i+1] == '0':  # Falling edge
                pattern_score += 0.2
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint coherence
        phi_factor = 0.8 if self._is_phi_valid(trace) else 0.2
        strength_factors.append(phi_factor)
        
        # Consistency strength emerges from geometric mean of factors
        consistency_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return consistency_strength
        
    def _compute_conflict_detection(self, trace: str, value: int) -> float:
        """Conflict detection emerges from systematic inconsistency identification capacity"""
        detection_factors = []
        
        # Factor 1: Detection sensitivity (ability to identify conflicts)
        sensitivity = 0.6 + 0.4 * (len(trace) / 8.0)
        detection_factors.append(min(sensitivity, 1.0))
        
        # Factor 2: Resolution capability (systematic conflict handling)
        ones_count = trace.count('1')
        if ones_count > 0:
            resolution = 0.5 + 0.5 * min(ones_count / 4.0, 1.0)
        else:
            resolution = 0.3
        detection_factors.append(resolution)
        
        # Factor 3: Conflict analysis depth
        analysis_depth = 0.4 + 0.6 * (value % 7) / 6.0
        detection_factors.append(analysis_depth)
        
        # Factor 4: φ-constraint conflict preservation
        preservation = 0.9 if self._is_phi_valid(trace) else 0.3
        detection_factors.append(preservation)
        
        conflict_detection = np.prod(detection_factors) ** (1.0 / len(detection_factors))
        return conflict_detection
        
    def _compute_spectrum_range(self, trace: str, value: int) -> float:
        """Spectrum range emerges from inconsistency diversity exploration capability"""
        range_factors = []
        
        # Factor 1: Range variation potential
        variation = len(set(trace)) / 2.0  # Diversity in binary trace
        range_factors.append(variation)
        
        # Factor 2: Spectrum accessibility
        access = 0.5 + 0.5 * min(len(trace) / 6.0, 1.0)
        range_factors.append(access)
        
        # Factor 3: Inconsistency space coverage
        coverage = 0.3 + 0.7 * (value % 11) / 10.0
        range_factors.append(coverage)
        
        spectrum_range = np.prod(range_factors) ** (1.0 / len(range_factors))
        return spectrum_range
        
    def _compute_inconsistency_tolerance(self, trace: str, value: int) -> float:
        """Inconsistency tolerance emerges from systematic conflict accommodation"""
        tolerance_factors = []
        
        # Factor 1: Tolerance capacity
        capacity = 0.4 + 0.6 * min(trace.count('0') / 5.0, 1.0)
        tolerance_factors.append(capacity)
        
        # Factor 2: Accommodation flexibility
        flexibility = 0.5 + 0.5 * (len(trace) % 3) / 2.0
        tolerance_factors.append(flexibility)
        
        # Factor 3: Systematic tolerance
        systematic = 0.7 + 0.3 * (value % 5) / 4.0
        tolerance_factors.append(systematic)
        
        inconsistency_tolerance = np.prod(tolerance_factors) ** (1.0 / len(tolerance_factors))
        return inconsistency_tolerance
        
    def _compute_consistency_completeness(self, trace: str, value: int) -> float:
        """Consistency completeness through comprehensive conflict coverage"""
        completeness_base = 0.3 + 0.7 * min(len(trace) / 7.0, 1.0)
        value_modulation = 0.8 + 0.2 * cos(value * 0.3)
        return completeness_base * value_modulation
        
    def _compute_conflict_efficiency(self, trace: str, value: int) -> float:
        """Conflict efficiency through optimized inconsistency handling"""
        efficiency_base = 0.4 + 0.6 * (trace.count('1') / max(len(trace), 1))
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return min(efficiency_base + phi_bonus, 1.0)
        
    def _compute_consistency_depth(self, trace: str, value: int) -> float:
        """Consistency depth through nested consistency analysis"""
        depth_factor = min(len(trace) / 12.0, 1.0)
        complexity_factor = (value % 13) / 12.0
        return 0.2 + 0.8 * (depth_factor * complexity_factor)
        
    def _compute_spectrum_stability(self, trace: str, value: int) -> float:
        """Spectrum stability through consistent inconsistency measurement"""
        if self._is_phi_valid(trace):
            stability_base = 0.95
        else:
            stability_base = 0.4
        variation = 0.1 * sin(value * 0.2)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_consistency_coherence(self, trace: str, value: int) -> float:
        """Consistency coherence through unified conflict architecture"""
        coherence_base = 0.6 + 0.4 * min(trace.count('0') / 4.0, 1.0)
        structural_bonus = 0.2 if len(trace) >= 4 else 0.0
        return min(coherence_base + structural_bonus, 1.0)
        
    def _categorize_consistency(self, strength: float, detection: float, 
                              spectrum: float, tolerance: float) -> str:
        """Categorize trace based on consistency profile"""
        # Calculate dominant characteristic with thresholds
        detection_threshold = 0.7  # High conflict detection threshold
        tolerance_threshold = 0.6  # High inconsistency tolerance threshold
        
        if detection >= detection_threshold:
            if tolerance >= tolerance_threshold:
                return "conflict_tolerant"  # High detection + high tolerance
            else:
                return "conflict_sensitive"  # High detection + low tolerance
        else:
            if tolerance >= tolerance_threshold:
                return "inconsistency_accommodating"  # Low detection + high tolerance
            else:
                return "basic_consistency"  # Low detection + low tolerance
        
    def _build_consistency_network(self) -> nx.Graph:
        """构建consistency network based on trace consistency similarity"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.values())
        for trace_data in traces:
            G.add_node(trace_data['value'], **trace_data)
            
        # Add edges based on consistency similarity
        similarity_threshold = 0.3
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_consistency_similarity(trace1, trace2)
                if similarity >= similarity_threshold:
                    G.add_edge(trace1['value'], trace2['value'], 
                             weight=similarity, similarity=similarity)
                    
        return G
        
    def _compute_consistency_similarity(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的consistency similarity"""
        # Consistency property similarity
        consistency_diff = abs(trace1['consistency_strength'] - trace2['consistency_strength'])
        detection_diff = abs(trace1['conflict_detection'] - trace2['conflict_detection'])
        spectrum_diff = abs(trace1['spectrum_range'] - trace2['spectrum_range'])
        tolerance_diff = abs(trace1['inconsistency_tolerance'] - trace2['inconsistency_tolerance'])
        
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
        property_similarity = 1.0 - np.mean([consistency_diff, detection_diff, 
                                           spectrum_diff, tolerance_diff])
        
        return 0.6 * property_similarity + 0.4 * geometric_similarity
        
    def _detect_conflict_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """检测consistency traces之间的conflict mappings"""
        mappings = defaultdict(list)
        
        tolerance = 0.3
        for node1 in self.consistency_network.nodes():
            data1 = self.consistency_network.nodes[node1]
            for node2 in self.consistency_network.nodes():
                if node1 != node2:
                    data2 = self.consistency_network.nodes[node2]
                    
                    # Check conflict preservation
                    detection_preserved = abs(data1['conflict_detection'] - 
                                            data2['conflict_detection']) <= tolerance
                    consistency_preserved = abs(data1['consistency_strength'] - 
                                              data2['consistency_strength']) <= tolerance
                    
                    if detection_preserved and consistency_preserved:
                        mappings[data1['category']].append((node1, node2))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive consistency spectrum analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['network_density'] = nx.density(self.consistency_network)
        results['connected_components'] = nx.number_connected_components(self.consistency_network)
        
        # Consistency properties analysis
        consistency_strengths = [t['consistency_strength'] for t in traces]
        conflict_detections = [t['conflict_detection'] for t in traces]
        spectrum_ranges = [t['spectrum_range'] for t in traces]
        inconsistency_tolerances = [t['inconsistency_tolerance'] for t in traces]
        consistency_completenesses = [t['consistency_completeness'] for t in traces]
        conflict_efficiencies = [t['conflict_efficiency'] for t in traces]
        consistency_depths = [t['consistency_depth'] for t in traces]
        spectrum_stabilities = [t['spectrum_stability'] for t in traces]
        consistency_coherences = [t['consistency_coherence'] for t in traces]
        
        results['consistency_strength'] = {
            'mean': np.mean(consistency_strengths),
            'std': np.std(consistency_strengths),
            'high_count': sum(1 for x in consistency_strengths if x > 0.5)
        }
        results['conflict_detection'] = {
            'mean': np.mean(conflict_detections),
            'std': np.std(conflict_detections),
            'high_count': sum(1 for x in conflict_detections if x > 0.5)
        }
        results['spectrum_range'] = {
            'mean': np.mean(spectrum_ranges),
            'std': np.std(spectrum_ranges),
            'high_count': sum(1 for x in spectrum_ranges if x > 0.5)
        }
        results['inconsistency_tolerance'] = {
            'mean': np.mean(inconsistency_tolerances),
            'std': np.std(inconsistency_tolerances),
            'high_count': sum(1 for x in inconsistency_tolerances if x > 0.5)
        }
        results['consistency_completeness'] = {
            'mean': np.mean(consistency_completenesses),
            'std': np.std(consistency_completenesses),
            'high_count': sum(1 for x in consistency_completenesses if x > 0.5)
        }
        results['conflict_efficiency'] = {
            'mean': np.mean(conflict_efficiencies),
            'std': np.std(conflict_efficiencies),
            'high_count': sum(1 for x in conflict_efficiencies if x > 0.5)
        }
        results['consistency_depth'] = {
            'mean': np.mean(consistency_depths),
            'std': np.std(consistency_depths),
            'high_count': sum(1 for x in consistency_depths if x > 0.5)
        }
        results['spectrum_stability'] = {
            'mean': np.mean(spectrum_stabilities),
            'std': np.std(spectrum_stabilities),
            'high_count': sum(1 for x in spectrum_stabilities if x > 0.5)
        }
        results['consistency_coherence'] = {
            'mean': np.mean(consistency_coherences),
            'std': np.std(consistency_coherences),
            'high_count': sum(1 for x in consistency_coherences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.consistency_network.edges()) > 0:
            results['network_edges'] = len(self.consistency_network.edges())
            results['average_degree'] = sum(dict(self.consistency_network.degree()).values()) / len(self.consistency_network.nodes())
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            
        # Morphism analysis
        total_morphisms = sum(len(mappings) for mappings in self.conflict_mappings.values())
        results['total_morphisms'] = total_morphisms
        if len(traces) > 1:
            max_possible_morphisms = len(traces) * (len(traces) - 1)
            results['morphism_density'] = total_morphisms / max_possible_morphisms
        else:
            results['morphism_density'] = 0.0
            
        # Entropy analysis
        properties = [
            ('consistency_strength', consistency_strengths),
            ('conflict_detection', conflict_detections),
            ('spectrum_range', spectrum_ranges),
            ('inconsistency_tolerance', inconsistency_tolerances),
            ('consistency_completeness', consistency_completenesses),
            ('conflict_efficiency', conflict_efficiencies),
            ('consistency_depth', consistency_depths),
            ('spectrum_stability', spectrum_stabilities),
            ('consistency_coherence', consistency_coherences)
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
        """生成consistency spectrum visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Consistency Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 108: Consistency Spectrum Dynamics', fontsize=16, fontweight='bold')
        
        # Consistency strength vs conflict detection
        x = [t['consistency_strength'] for t in traces]
        y = [t['conflict_detection'] for t in traces]
        colors = [t['spectrum_range'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Consistency Strength')
        ax1.set_ylabel('Conflict Detection')
        ax1.set_title('Consistency-Conflict Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Spectrum Range')
        
        # Spectrum range distribution
        spectrum_ranges = [t['spectrum_range'] for t in traces]
        ax2.hist(spectrum_ranges, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Spectrum Range')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Spectrum Range Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Inconsistency tolerance vs consistency completeness
        x3 = [t['inconsistency_tolerance'] for t in traces]
        y3 = [t['consistency_completeness'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Inconsistency Tolerance')
        ax3.set_ylabel('Consistency Completeness')
        ax3.set_title('Tolerance-Completeness Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Consistency Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-108-consistency-spectrum-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 108: Consistency Network Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        pos = nx.spring_layout(self.consistency_network, k=2, iterations=50)
        node_colors = [traces[i]['consistency_strength'] for i in range(len(traces))]
        nx.draw(self.consistency_network, pos, ax=ax1, 
                node_color=node_colors, cmap='viridis', 
                node_size=300, alpha=0.8, with_labels=True, font_size=8)
        ax1.set_title('Consistency Network Structure')
        
        # Degree distribution
        degrees = [self.consistency_network.degree(node) for node in self.consistency_network.nodes()]
        ax2.hist(degrees, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Network Degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Consistency properties correlation matrix
        properties_matrix = np.array([
            [t['consistency_strength'] for t in traces],
            [t['conflict_detection'] for t in traces],
            [t['spectrum_range'] for t in traces],
            [t['inconsistency_tolerance'] for t in traces],
            [t['consistency_completeness'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Strength', 'Detection', 'Range', 'Tolerance', 'Completeness']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Consistency Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Conflict efficiency vs consistency depth
        x4 = [t['conflict_efficiency'] for t in traces]
        y4 = [t['consistency_depth'] for t in traces]
        stabilities = [t['spectrum_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='plasma', alpha=0.7, s=60)
        ax4.set_xlabel('Conflict Efficiency')
        ax4.set_ylabel('Consistency Depth')
        ax4.set_title('Efficiency-Depth Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Spectrum Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-108-consistency-spectrum-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestConsistencySpectrum(unittest.TestCase):
    """Unit tests for consistency spectrum system"""
    
    def setUp(self):
        """Set up test consistency spectrum system"""
        self.system = ConsistencySpectrumSystem(max_trace_value=20, consistency_depth=4)
        
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
            self.assertIn('consistency_strength', data)
            self.assertIn('conflict_detection', data)
            self.assertIn('spectrum_range', data)
            self.assertIn('inconsistency_tolerance', data)
            self.assertTrue(0 <= data['consistency_strength'] <= 1)
            self.assertTrue(0 <= data['conflict_detection'] <= 1)
            
    def test_consistency_strength_computation(self):
        """测试consistency strength computation"""
        trace = "101"
        value = 5
        strength = self.system._compute_consistency_strength(trace, value)
        self.assertTrue(0 <= strength <= 1)
        
    def test_conflict_detection_computation(self):
        """测试conflict detection computation"""
        trace = "1001"
        value = 8
        detection = self.system._compute_conflict_detection(trace, value)
        self.assertTrue(0 <= detection <= 1)
        
    def test_consistency_network_construction(self):
        """测试consistency network construction"""
        self.assertGreater(len(self.system.consistency_network.nodes()), 0)
        
        # Check network properties
        density = nx.density(self.system.consistency_network)
        self.assertTrue(0 <= density <= 1)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('consistency_strength', results)
        self.assertIn('conflict_detection', results)
        self.assertIn('categories', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = ConsistencySpectrumSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("CONSISTENCY SPECTRUM INCONSISTENCY ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Consistency Properties Analysis:")
    properties = ['consistency_strength', 'conflict_detection', 'spectrum_range', 
                 'inconsistency_tolerance', 'consistency_completeness', 'conflict_efficiency',
                 'consistency_depth', 'spectrum_stability', 'consistency_coherence']
    
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
    print(f"Conflict Pattern Analysis:")
    print(f"Total conflict mappings: {sum(len(mappings) for mappings in system.conflict_mappings.values())}")
    print(f"Conflict mapping density: {results['morphism_density']:.3f}")
    print(f"Average conflicts per trace: {results['total_morphisms'] / results['total_traces']:.1f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    for prop, entropy in results['entropy_analysis'].items():
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)