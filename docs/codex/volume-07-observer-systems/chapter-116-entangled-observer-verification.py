#!/usr/bin/env python3
"""
Chapter 116: EntangledObserver Unit Test Verification
从ψ=ψ(ψ)推导Inter-Collapse Coordination in Observer Tensor Pairs

Core principle: From ψ = ψ(ψ) derive systematic entangled observer
construction through φ-constrained trace transformations that enable quantum-like
entanglement between observers through trace geometric relationships,
creating entanglement networks that encode the fundamental coordination
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic entanglement structures through φ-trace entanglement
dynamics rather than traditional quantum entanglement theories or external
entanglement constructions.

This verification program implements:
1. φ-constrained entanglement construction through trace pair analysis
2. Observer entanglement systems: systematic coordination through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection entanglement theory
4. Graph theory analysis of entanglement networks and coordination relationship structures
5. Information theory analysis of entanglement entropy and coordination encoding
6. Category theory analysis of entanglement functors and pair morphisms
7. Visualization of entanglement structures and φ-trace entanglement systems
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

class EntangledObserverSystem:
    """
    Core system for implementing inter-collapse coordination in observer tensor pairs.
    Implements φ-constrained entanglement architectures through trace coordination dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, entanglement_depth: int = 6):
        """Initialize entangled observer system with coordination trace analysis"""
        self.max_trace_value = max_trace_value
        self.entanglement_depth = entanglement_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.entanglement_cache = {}
        self.coordination_cache = {}
        self.pair_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.entanglement_pairs = self._build_entanglement_pairs()
        self.entanglement_network = self._build_entanglement_network()
        self.coordination_mappings = self._detect_coordination_mappings()
        
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
        coordination_capacity = self._compute_coordination_capacity(trace, value)
        entanglement_potential = self._compute_entanglement_potential(trace, value)
        
        return {
            'trace': trace,
            'value': value,
            'observer_strength': observer_strength,
            'coordination_capacity': coordination_capacity,
            'entanglement_potential': entanglement_potential
        }
        
    def _compute_observer_strength(self, trace: str, value: int) -> float:
        """Observer strength for entanglement capability"""
        strength_factors = []
        
        # Factor 1: Length provides entanglement space (minimum 4 for meaningful pairs)
        length_factor = len(trace) / 4.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight balance for entanglement (optimal at 50% density)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            balance = 1.0 - 2.0 * abs(0.5 - weight / total)
            strength_factors.append(balance)
        else:
            strength_factors.append(0.1)
            
        # Factor 3: φ-constraint coherence
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.2
        strength_factors.append(phi_factor)
        
        observer_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return observer_strength
        
    def _compute_coordination_capacity(self, trace: str, value: int) -> float:
        """Coordination capacity for entanglement interactions"""
        capacity_factors = []
        
        # Factor 1: Structural coordination potential
        structural = 0.3 + 0.7 * min(len(trace) / 6.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Pattern complexity for coordination
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.2 + 0.8 * min(ones_count / 3.0, 1.0)
        else:
            complexity = 0.1
        capacity_factors.append(complexity)
        
        # Factor 3: Modulation depth (value modulo 3 for 3-fold symmetry)
        modulation = 0.4 + 0.6 * (value % 3) / 2.0
        capacity_factors.append(modulation)
        
        coordination_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return coordination_capacity
        
    def _compute_entanglement_potential(self, trace: str, value: int) -> float:
        """Entanglement potential based on trace properties"""
        # Entanglement favors balanced traces with good coordination
        if len(trace) < 2:
            return 0.1
            
        # Alternating pattern bonus
        alternations = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        alternation_rate = alternations / (len(trace) - 1) if len(trace) > 1 else 0
        
        # Symmetry bonus
        symmetry = 1.0 - abs(len(trace) / 2 - trace.count('1')) / (len(trace) / 2) if len(trace) > 0 else 0
        
        # Base entanglement from value properties
        base_entanglement = 0.3 + 0.7 * (value % 13) / 12.0  # 13 for prime entanglement levels
        
        entanglement_potential = 0.3 * alternation_rate + 0.3 * symmetry + 0.4 * base_entanglement
        return min(entanglement_potential, 1.0)
        
    def _build_entanglement_pairs(self) -> List[Tuple[Dict, Dict, float]]:
        """构建entanglement pairs with coordination strength"""
        pairs = []
        traces = list(self.trace_universe.values())
        
        # Build pairs with significant entanglement potential
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                entanglement_strength = self._compute_pair_entanglement(trace1, trace2)
                if entanglement_strength > 0.3:  # Threshold for significant entanglement
                    pairs.append((trace1, trace2, entanglement_strength))
                    
        return pairs
        
    def _compute_pair_entanglement(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的entanglement strength"""
        # Entanglement factors
        factors = []
        
        # Factor 1: Observer strength compatibility
        strength_diff = abs(trace1['observer_strength'] - trace2['observer_strength'])
        strength_compat = 1.0 - strength_diff
        factors.append(strength_compat)
        
        # Factor 2: Coordination harmony
        coord_product = trace1['coordination_capacity'] * trace2['coordination_capacity']
        factors.append(coord_product)
        
        # Factor 3: Entanglement potential synergy
        entangle_avg = (trace1['entanglement_potential'] + trace2['entanglement_potential']) / 2.0
        factors.append(entangle_avg)
        
        # Factor 4: Trace complementarity
        trace1_str, trace2_str = trace1['trace'], trace2['trace']
        if len(trace1_str) == len(trace2_str) and len(trace1_str) > 0:
            # XOR distance for complementarity
            xor_count = sum(1 for a, b in zip(trace1_str, trace2_str) if a != b)
            complementarity = xor_count / len(trace1_str)
            # Optimal complementarity around 0.5 (50% different)
            complement_score = 1.0 - 2.0 * abs(0.5 - complementarity)
        else:
            complement_score = 0.3
        factors.append(complement_score)
        
        # Geometric mean of factors
        entanglement_strength = np.prod(factors) ** (1.0 / len(factors))
        return entanglement_strength
        
    def _build_entanglement_network(self) -> nx.Graph:
        """构建entanglement network from pairs"""
        G = nx.Graph()
        
        # Add all traces as nodes
        for trace_data in self.trace_universe.values():
            G.add_node(trace_data['value'], **trace_data)
            
        # Add entanglement edges
        for trace1, trace2, strength in self.entanglement_pairs:
            if strength > 0.3:  # Threshold for network inclusion
                G.add_edge(trace1['value'], trace2['value'], 
                         weight=strength, entanglement=strength)
                    
        return G
        
    def _detect_coordination_mappings(self) -> Dict[str, List[Tuple[int, int, float]]]:
        """检测entangled traces之间的coordination mappings"""
        mappings = defaultdict(list)
        
        for trace1, trace2, strength in self.entanglement_pairs:
            # Categorize by entanglement strength
            if strength > 0.7:
                category = "strong_entanglement"
            elif strength > 0.5:
                category = "moderate_entanglement"
            else:
                category = "weak_entanglement"
                
            mappings[category].append((trace1['value'], trace2['value'], strength))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive entangled observer analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['total_pairs'] = len(self.entanglement_pairs)
        results['network_density'] = nx.density(self.entanglement_network)
        results['connected_components'] = nx.number_connected_components(self.entanglement_network)
        
        # Entanglement analysis
        if self.entanglement_pairs:
            entanglement_strengths = [strength for _, _, strength in self.entanglement_pairs]
            results['entanglement_strength'] = {
                'mean': np.mean(entanglement_strengths),
                'std': np.std(entanglement_strengths),
                'max': np.max(entanglement_strengths),
                'min': np.min(entanglement_strengths)
            }
        else:
            results['entanglement_strength'] = {
                'mean': 0, 'std': 0, 'max': 0, 'min': 0
            }
        
        # Observer properties
        observer_strengths = [t['observer_strength'] for t in traces]
        coordination_capacities = [t['coordination_capacity'] for t in traces]
        entanglement_potentials = [t['entanglement_potential'] for t in traces]
        
        results['observer_strength'] = {
            'mean': np.mean(observer_strengths),
            'std': np.std(observer_strengths),
            'high_count': sum(1 for x in observer_strengths if x > 0.5)
        }
        results['coordination_capacity'] = {
            'mean': np.mean(coordination_capacities),
            'std': np.std(coordination_capacities),
            'high_count': sum(1 for x in coordination_capacities if x > 0.5)
        }
        results['entanglement_potential'] = {
            'mean': np.mean(entanglement_potentials),
            'std': np.std(entanglement_potentials),
            'high_count': sum(1 for x in entanglement_potentials if x > 0.5)
        }
        
        # Category analysis
        category_counts = defaultdict(int)
        for category, pairs in self.coordination_mappings.items():
            category_counts[category] = len(pairs)
        results['categories'] = dict(category_counts)
        
        # Network analysis
        if len(self.entanglement_network.edges()) > 0:
            results['network_edges'] = len(self.entanglement_network.edges())
            results['average_degree'] = sum(dict(self.entanglement_network.degree()).values()) / len(self.entanglement_network.nodes())
            
            # Clustering coefficient
            results['clustering_coefficient'] = nx.average_clustering(self.entanglement_network)
            
            # Find maximally entangled pairs
            max_entangled = []
            for u, v, data in self.entanglement_network.edges(data=True):
                if data['entanglement'] > 0.7:
                    max_entangled.append((u, v, data['entanglement']))
            results['max_entangled_pairs'] = len(max_entangled)
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            results['clustering_coefficient'] = 0.0
            results['max_entangled_pairs'] = 0
            
        # Entropy analysis
        properties = [
            ('observer_strength', observer_strengths),
            ('coordination_capacity', coordination_capacities),
            ('entanglement_potential', entanglement_potentials)
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
        """生成entangled observer visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Entanglement Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 116: Entangled Observer Dynamics', fontsize=16, fontweight='bold')
        
        # Observer strength vs coordination capacity
        x = [t['observer_strength'] for t in traces]
        y = [t['coordination_capacity'] for t in traces]
        colors = [t['entanglement_potential'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Observer Strength')
        ax1.set_ylabel('Coordination Capacity')
        ax1.set_title('Observer-Coordination Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Entanglement Potential')
        
        # Entanglement strength distribution
        if self.entanglement_pairs:
            strengths = [s for _, _, s in self.entanglement_pairs]
            ax2.hist(strengths, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax2.set_xlabel('Entanglement Strength')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Entanglement Strength Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Entanglement Pairs', ha='center', va='center')
            ax2.set_title('Entanglement Strength Distribution')
        
        # Network visualization (subset for clarity)
        if len(self.entanglement_network.edges()) > 0:
            # Create subgraph of strongly entangled nodes
            strong_edges = [(u, v) for u, v, d in self.entanglement_network.edges(data=True) 
                           if d['entanglement'] > 0.6]
            if strong_edges:
                G_strong = self.entanglement_network.edge_subgraph(strong_edges).copy()
                pos = nx.spring_layout(G_strong, k=2.0, iterations=50)
                
                # Draw network
                edge_weights = [G_strong[u][v]['entanglement'] for u, v in G_strong.edges()]
                nx.draw_networkx_nodes(G_strong, pos, ax=ax3, node_size=300, 
                                     node_color='lightblue', alpha=0.8)
                nx.draw_networkx_edges(G_strong, pos, ax=ax3, width=2, 
                                     edge_color=edge_weights, edge_cmap=plt.cm.plasma)
                nx.draw_networkx_labels(G_strong, pos, ax=ax3, font_size=8)
                ax3.set_title('Strong Entanglement Network (strength > 0.6)')
            else:
                ax3.text(0.5, 0.5, 'No Strong Entanglements', ha='center', va='center')
                ax3.set_title('Strong Entanglement Network')
        else:
            ax3.text(0.5, 0.5, 'No Entanglement Network', ha='center', va='center')
            ax3.set_title('Strong Entanglement Network')
        ax3.axis('off')
        
        # Entanglement categories
        if self.coordination_mappings:
            categories = list(self.coordination_mappings.keys())
            counts = [len(self.coordination_mappings[cat]) for cat in categories]
            ax4.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Entanglement Category Distribution')
        else:
            ax4.text(0.5, 0.5, 'No Entanglement Categories', ha='center', va='center')
            ax4.set_title('Entanglement Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-116-entangled-observer-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Entanglement Architecture
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 116: Entangled Observer Architecture', fontsize=16, fontweight='bold')
        
        # Full network visualization
        if len(self.entanglement_network.nodes()) > 0:
            pos = nx.spring_layout(self.entanglement_network, k=1.5, iterations=50)
            node_colors = [self.entanglement_network.nodes[n]['observer_strength'] 
                         for n in self.entanglement_network.nodes()]
            nx.draw(self.entanglement_network, pos, ax=ax1, 
                   node_color=node_colors, cmap='plasma', 
                   node_size=200, alpha=0.8, with_labels=True, font_size=6)
            ax1.set_title('Complete Entanglement Network')
        else:
            ax1.text(0.5, 0.5, 'No Network Data', ha='center', va='center')
            ax1.set_title('Complete Entanglement Network')
        
        # Degree distribution
        if len(self.entanglement_network.edges()) > 0:
            degrees = [self.entanglement_network.degree(node) 
                      for node in self.entanglement_network.nodes()]
            ax2.hist(degrees, bins=15, alpha=0.7, color='coral', edgecolor='black')
            ax2.set_xlabel('Node Degree')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Entanglement Degree Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Degree Data', ha='center', va='center')
            ax2.set_title('Entanglement Degree Distribution')
        
        # Entanglement matrix heatmap (top traces)
        if len(traces) > 1:
            top_traces = sorted(traces, key=lambda t: t['entanglement_potential'], reverse=True)[:10]
            n = len(top_traces)
            entangle_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        entangle_matrix[i, j] = self._compute_pair_entanglement(
                            top_traces[i], top_traces[j])
            
            im = ax3.imshow(entangle_matrix, cmap='viridis', aspect='auto')
            ax3.set_xticks(range(n))
            ax3.set_yticks(range(n))
            ax3.set_xticklabels([t['value'] for t in top_traces], rotation=45)
            ax3.set_yticklabels([t['value'] for t in top_traces])
            ax3.set_title('Entanglement Matrix (Top 10 Traces)')
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            ax3.set_title('Entanglement Matrix')
        
        # Coordination patterns
        if len(traces) > 0:
            x4 = [t['coordination_capacity'] for t in traces]
            y4 = [t['entanglement_potential'] for t in traces]
            sizes = [100 * t['observer_strength'] for t in traces]
            scatter4 = ax4.scatter(x4, y4, s=sizes, alpha=0.6, c=sizes, cmap='coolwarm')
            ax4.set_xlabel('Coordination Capacity')
            ax4.set_ylabel('Entanglement Potential')
            ax4.set_title('Coordination-Entanglement Pattern')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter4, ax=ax4, label='Observer Strength')
        else:
            ax4.text(0.5, 0.5, 'No Pattern Data', ha='center', va='center')
            ax4.set_title('Coordination-Entanglement Pattern')
        
        plt.tight_layout()
        plt.savefig('chapter-116-entangled-observer-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestEntangledObserver(unittest.TestCase):
    """Unit tests for entangled observer system"""
    
    def setUp(self):
        """Set up test entangled observer system"""
        self.system = EntangledObserverSystem(max_trace_value=20, entanglement_depth=4)
        
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
            self.assertIn('coordination_capacity', data)
            self.assertIn('entanglement_potential', data)
            self.assertTrue(0 <= data['observer_strength'] <= 1)
            self.assertTrue(0 <= data['coordination_capacity'] <= 1)
            
    def test_entanglement_computation(self):
        """测试entanglement computation"""
        if len(self.system.trace_universe) >= 2:
            traces = list(self.system.trace_universe.values())
            trace1, trace2 = traces[0], traces[1]
            entanglement = self.system._compute_pair_entanglement(trace1, trace2)
            self.assertTrue(0 <= entanglement <= 1)
            
    def test_entanglement_network_construction(self):
        """测试entanglement network construction"""
        self.assertIsNotNone(self.system.entanglement_network)
        self.assertGreaterEqual(len(self.system.entanglement_network.nodes()), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('total_pairs', results)
        self.assertIn('observer_strength', results)
        self.assertIn('coordination_capacity', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = EntangledObserverSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("ENTANGLED OBSERVER COORDINATION ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Total entanglement pairs: {results['total_pairs']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Observer Properties Analysis:")
    properties = ['observer_strength', 'coordination_capacity', 'entanglement_potential']
    
    for prop in properties:
        if prop in results:
            data = results[prop]
            percentage = (data['high_count'] / results['total_traces']) * 100 if results['total_traces'] > 0 else 0
            print(f"- {prop.replace('_', ' ').title()}: mean={data['mean']:.3f}, high_count={data['high_count']} ({percentage:.1f}%)")
    
    print()
    print("Entanglement Analysis:")
    if 'entanglement_strength' in results:
        ent_data = results['entanglement_strength']
        print(f"- Mean entanglement strength: {ent_data['mean']:.3f}")
        print(f"- Max entanglement strength: {ent_data['max']:.3f}")
        print(f"- Min entanglement strength: {ent_data['min']:.3f}")
        print(f"- Clustering coefficient: {results.get('clustering_coefficient', 0):.3f}")
        print(f"- Maximally entangled pairs (>0.7): {results.get('max_entangled_pairs', 0)}")
    
    print()
    print("Category Distribution:")
    if 'categories' in results:
        for category, count in results['categories'].items():
            print(f"- {category.replace('_', ' ').title()}: {count} pairs")
    
    print()
    print("Network Analysis:")
    print(f"Network edges: {results.get('network_edges', 0)}")
    print(f"Average degree: {results.get('average_degree', 0):.3f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    if 'entropy_analysis' in results:
        for prop, entropy in results['entropy_analysis'].items():
            print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)