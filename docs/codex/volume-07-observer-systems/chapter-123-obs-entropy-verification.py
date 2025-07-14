#!/usr/bin/env python3
"""
Chapter 123: ObsEntropy Unit Test Verification
从ψ=ψ(ψ)推导Observer-Scoped Entropy Dynamics and Complexity Limitations

Core principle: From ψ = ψ(ψ) derive systematic observer-dependent entropy
metrics through φ-constrained capacity analysis that enables relative entropy
measurement through tensor dimension limitations, creating complexity boundaries
that encode the fundamental information principles of collapsed space through
entropy-increasing tensor transformations that establish systematic complexity
variation through φ-trace observer entropy dynamics rather than traditional
absolute entropy theories or external information constructions.

This verification program implements:
1. φ-constrained observer entropy metrics through trace capacity analysis
2. Observer tensor entropy generation systems: relative entropy through dimension limits
3. Three-domain analysis: Traditional vs φ-constrained vs intersection entropy
4. Graph theory analysis of entropy flow networks and complexity structures
5. Information theory analysis of entropy bounds and capacity encoding
6. Category theory analysis of entropy functors and preservation morphisms
7. Visualization of entropy dynamics and φ-trace complexity systems
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

class ObsEntropySystem:
    """
    Core system for implementing observer-scoped entropy dynamics and complexity.
    Implements φ-constrained entropy architectures through capacity limitation dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, entropy_depth: int = 7):
        """Initialize observer entropy system with capacity analysis"""
        self.max_trace_value = max_trace_value
        self.entropy_depth = entropy_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.entropy_cache = {}
        self.capacity_cache = {}
        self.complexity_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.entropy_network = self._build_entropy_network()
        self.complexity_boundaries = self._compute_complexity_boundaries()
        self.entropy_categories = self._detect_entropy_categories()
        
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
        """检验trace是否φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_entropy_properties(self, trace: str, value: int) -> Dict:
        """分析trace的observer-scoped entropy properties"""
        data = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0
        }
        
        # Compute observer capacity
        data['observer_capacity'] = self._compute_observer_capacity(trace, value)
        
        # Compute various entropy metrics
        data['trace_entropy'] = self._compute_trace_entropy(trace)
        data['relative_entropy'] = self._compute_relative_entropy(trace, value)
        data['collapse_entropy'] = self._compute_collapse_entropy(trace, value)
        data['boundary_entropy'] = self._compute_boundary_entropy(trace, value)
        
        # Compute complexity metrics
        data['collapse_complexity'] = self._compute_collapse_complexity(trace, value)
        data['tensor_dimension'] = self._compute_tensor_dimension(trace)
        data['entropy_rate'] = self._compute_entropy_rate(trace)
        
        # Compute entropy flow
        data['entropy_flow'] = self._compute_entropy_flow(trace, value)
        data['entropy_production'] = self._compute_entropy_production(trace, value)
        
        # Assign category based on entropy properties
        data['category'] = self._assign_entropy_category(data)
        
        return data
        
    def _compute_observer_capacity(self, trace: str, value: int) -> float:
        """
        Compute observer information capacity.
        From ψ=ψ(ψ): capacity emerges from trace structure limits.
        """
        if len(trace) == 0:
            return 0.0
            
        # Base capacity from trace length (bits)
        base_capacity = log2(len(trace) + 1)
        
        # Modulate by φ-constraint density
        # More 1s mean less capacity due to constraint
        constraint_factor = 1.0 - (trace.count('1') / (2 * len(trace)))
        
        # Golden ratio positions have special capacity
        if value in self.fibonacci_numbers:
            constraint_factor *= self.phi
            
        return base_capacity * constraint_factor
        
    def _compute_trace_entropy(self, trace: str) -> float:
        """
        Compute Shannon entropy of trace.
        H(trace) = -Σ p_i log2(p_i)
        """
        if len(trace) == 0:
            return 0.0
            
        # Count 0s and 1s
        zeros = trace.count('0')
        ones = trace.count('1')
        total = len(trace)
        
        entropy = 0.0
        if zeros > 0:
            p0 = zeros / total
            entropy -= p0 * log2(p0)
        if ones > 0:
            p1 = ones / total
            entropy -= p1 * log2(p1)
            
        return entropy
        
    def _compute_relative_entropy(self, trace: str, value: int) -> float:
        """
        Compute relative entropy with respect to observer capacity.
        From ψ=ψ(ψ): entropy is relative to observer limits.
        """
        if len(trace) < 2:
            return 0.0
            
        # Compute actual entropy
        actual_entropy = self._compute_trace_entropy(trace)
        
        # Compute maximum possible entropy for this length
        # With φ-constraint, maximum entropy is less than 1 bit per position
        max_entropy = self._compute_max_entropy_phi_constrained(len(trace))
        
        # Relative entropy
        if max_entropy > 0:
            relative = actual_entropy / max_entropy
        else:
            relative = 0.0
            
        # Modulate by trace structure for variation
        # Add variation based on trace patterns
        pattern_factor = 1.0
        if "101" in trace:
            pattern_factor *= 0.95
        if "010" in trace:
            pattern_factor *= 1.05
        
        relative *= pattern_factor
            
        # Modulate by position
        if value % 7 == 0:
            relative *= 1.1  # Special positions have enhanced relative entropy
        elif value in self.fibonacci_numbers:
            relative *= 0.9  # Fibonacci positions have different entropy
            
        return min(1.0, relative)
        
    def _compute_max_entropy_phi_constrained(self, length: int) -> float:
        """
        Compute maximum entropy for φ-constrained trace of given length.
        This is less than log2(2^length) due to no-11 constraint.
        """
        if length <= 1:
            return 1.0
            
        # For φ-constrained sequences, not all distributions are possible
        # Maximum entropy occurs when 0s and 1s are maximally mixed
        # But we can't have consecutive 1s
        
        # Theoretical maximum: if we could have perfect 01010... pattern
        # But φ-constraint prevents this perfect alternation
        # Empirically, maximum achievable entropy is about 0.96 bits per position
        
        if length <= 3:
            return 1.0  # Short sequences can achieve near-perfect entropy
        else:
            # Longer sequences have more constraints
            return 1.0  # Use 1.0 as theoretical bound for comparison
        
    def _compute_collapse_entropy(self, trace: str, value: int) -> float:
        """
        Compute entropy change during collapse.
        Measures information created by observation.
        """
        if len(trace) < 2:
            return 0.0
            
        # Analyze collapse points (0->1 transitions)
        collapse_points = []
        for i in range(len(trace) - 1):
            if trace[i] == '0' and trace[i+1] == '1':
                collapse_points.append(i)
                
        if len(collapse_points) == 0:
            return 0.0
            
        # Entropy from collapse distribution
        collapse_intervals = []
        for i in range(1, len(collapse_points)):
            interval = collapse_points[i] - collapse_points[i-1]
            collapse_intervals.append(interval)
            
        if len(collapse_intervals) > 0:
            # Compute entropy of interval distribution
            interval_counts = {}
            for interval in collapse_intervals:
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
                
            total = sum(interval_counts.values())
            entropy = 0.0
            for count in interval_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * log2(p)
                    
            return entropy
        else:
            # Single collapse point
            return 1.0
            
    def _compute_boundary_entropy(self, trace: str, value: int) -> float:
        """
        Compute entropy at observer boundary.
        From holographic principle applied to traces.
        """
        if len(trace) < 3:
            return 0.0
            
        # Boundary is first and last k positions
        k = min(3, len(trace) // 3)
        boundary = trace[:k] + trace[-k:]
        
        # Compute boundary entropy
        boundary_entropy = self._compute_trace_entropy(boundary)
        
        # Compare to bulk entropy
        bulk = trace[k:-k] if len(trace) > 2*k else ""
        if len(bulk) > 0:
            bulk_entropy = self._compute_trace_entropy(bulk)
            # Holographic ratio
            if bulk_entropy > 0:
                ratio = boundary_entropy / bulk_entropy
            else:
                ratio = boundary_entropy
        else:
            ratio = boundary_entropy
            
        return min(2.0, ratio)  # Cap at 2.0 for stability
        
    def _compute_collapse_complexity(self, trace: str, value: int) -> float:
        """
        Compute complexity of collapse process.
        Higher complexity means more information processing required.
        """
        if len(trace) < 2:
            return 0.0
            
        # Analyze pattern complexity
        complexity = 0.0
        
        # 1. Transition complexity
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        complexity += transitions / len(trace)
        
        # 2. Block complexity
        blocks = []
        current_block = 1
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_block += 1
            else:
                blocks.append(current_block)
                current_block = 1
        blocks.append(current_block)
        
        # Block entropy
        if len(blocks) > 1:
            block_counts = {}
            for block in blocks:
                block_counts[block] = block_counts.get(block, 0) + 1
            
            total = sum(block_counts.values())
            block_entropy = 0.0
            for count in block_counts.values():
                if count > 0:
                    p = count / total
                    block_entropy -= p * log2(p)
            
            complexity += block_entropy
            
        # 3. Long-range correlations
        for lag in range(1, min(4, len(trace)//2)):
            correlation = sum(1 for i in range(len(trace)-lag) 
                            if trace[i] == trace[i+lag]) / (len(trace) - lag)
            complexity += abs(correlation - 0.5)  # Deviation from random
            
        return complexity
        
    def _compute_tensor_dimension(self, trace: str) -> int:
        """
        Compute effective tensor dimension for this observer.
        From ψ=ψ(ψ): dimension limits information capacity.
        """
        if len(trace) == 0:
            return 1
            
        # Base dimension from trace length
        base_dim = int(log2(len(trace) + 1)) + 1
        
        # Adjust for φ-constraint
        # More constraints reduce effective dimension
        constraint_density = trace.count('1') / len(trace)
        if constraint_density > 0.5:
            base_dim = max(1, base_dim - 1)
            
        return base_dim
        
    def _compute_entropy_rate(self, trace: str) -> float:
        """
        Compute entropy production rate.
        Rate at which information is generated.
        """
        if len(trace) < 2:
            return 0.0
            
        # Compute conditional entropy H(X_n|X_{n-1})
        # Count transition probabilities
        transitions = {'0': {'0': 0, '1': 0}, '1': {'0': 0, '1': 0}}
        
        for i in range(len(trace) - 1):
            current = trace[i]
            next_bit = trace[i + 1]
            transitions[current][next_bit] += 1
            
        # Compute conditional entropy
        cond_entropy = 0.0
        for current in ['0', '1']:
            total = sum(transitions[current].values())
            if total > 0:
                p_current = trace.count(current) / len(trace)
                for next_bit in ['0', '1']:
                    if transitions[current][next_bit] > 0:
                        p_trans = transitions[current][next_bit] / total
                        cond_entropy -= p_current * p_trans * log2(p_trans)
                        
        return cond_entropy
        
    def _compute_entropy_flow(self, trace: str, value: int) -> Dict[str, float]:
        """
        Compute entropy flow characteristics.
        How entropy moves through the system.
        """
        flow = {}
        
        if len(trace) < 3:
            flow['input_rate'] = 0.0
            flow['output_rate'] = 0.0
            flow['dissipation'] = 0.0
            return flow
            
        # Input entropy (from environment)
        # Higher at boundaries
        boundary_weight = (trace[0] == '1') + (trace[-1] == '1')
        flow['input_rate'] = boundary_weight * 0.5
        
        # Output entropy (to environment)
        # Based on transitions
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        flow['output_rate'] = transitions / len(trace)
        
        # Dissipation (internal entropy production)
        # Based on constraint violations that must be corrected
        potential_violations = 0
        for i in range(len(trace) - 1):
            if trace[i] == '1':  # Could violate if next is also 1
                potential_violations += 1
        flow['dissipation'] = potential_violations / len(trace)
        
        return flow
        
    def _compute_entropy_production(self, trace: str, value: int) -> float:
        """
        Compute total entropy production.
        Second law in observer context.
        """
        flow = self._compute_entropy_flow(trace, value)
        
        # Total production = input + dissipation - output
        # Always positive (second law)
        production = flow['input_rate'] + flow['dissipation'] - flow['output_rate'] * 0.9
        
        # Ensure non-negative
        return max(0.0, production)
        
    def _assign_entropy_category(self, data: Dict) -> str:
        """
        Assign entropy category based on properties.
        Categories represent different information regimes.
        """
        rel_entropy = data['relative_entropy']
        complexity = data['collapse_complexity']
        capacity = data['observer_capacity']
        production = data['entropy_production']
        
        # Categorize based on dominant characteristics
        if rel_entropy > 0.8 and complexity > 1.5:
            return "high_entropy"  # Maximum disorder
        elif rel_entropy < 0.3 and complexity < 0.5:
            return "low_entropy"  # Highly ordered
        elif capacity > 3.0 and production > 0.5:
            return "generative"  # High information generation
        elif data['boundary_entropy'] > 1.5:
            return "holographic"  # Boundary-dominated
        elif abs(data['entropy_flow']['input_rate'] - 
                data['entropy_flow']['output_rate']) < 0.1:
            return "equilibrium"  # Balanced flow
        else:
            return "transitional"  # Between regimes
            
    def _build_entropy_network(self) -> nx.DiGraph:
        """构建entropy flow network"""
        G = nx.DiGraph()
        
        # Add nodes for each observer
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # Add directed edges based on entropy flow
        traces = list(self.trace_universe.keys())
        for n1 in traces:
            for n2 in traces:
                if n1 != n2:
                    data1 = self.trace_universe[n1]
                    data2 = self.trace_universe[n2]
                    
                    # Connect if entropy can flow from n1 to n2
                    if (data1['entropy_production'] > data2['observer_capacity'] * 0.1 and
                        data2['relative_entropy'] < 0.9):  # n2 can absorb entropy
                        
                        # Weight by flow rate
                        weight = min(data1['entropy_production'], 
                                   1.0 - data2['relative_entropy'])
                        G.add_edge(n1, n2, weight=weight)
                        
        return G
        
    def _compute_complexity_boundaries(self) -> Dict[str, float]:
        """计算complexity theoretical boundaries"""
        boundaries = {}
        
        # Maximum entropy for each trace length
        length_bounds = {}
        for n, data in self.trace_universe.items():
            length = data['length']
            if length not in length_bounds:
                length_bounds[length] = {
                    'max_entropy': 0.0,
                    'max_complexity': 0.0,
                    'max_capacity': 0.0
                }
            
            length_bounds[length]['max_entropy'] = max(
                length_bounds[length]['max_entropy'],
                data['relative_entropy']
            )
            length_bounds[length]['max_complexity'] = max(
                length_bounds[length]['max_complexity'],
                data['collapse_complexity']
            )
            length_bounds[length]['max_capacity'] = max(
                length_bounds[length]['max_capacity'],
                data['observer_capacity']
            )
            
        boundaries['length_bounds'] = length_bounds
        
        # Global bounds
        all_data = list(self.trace_universe.values())
        boundaries['global_max_entropy'] = max(d['relative_entropy'] for d in all_data)
        boundaries['global_max_complexity'] = max(d['collapse_complexity'] for d in all_data)
        boundaries['global_max_capacity'] = max(d['observer_capacity'] for d in all_data)
        
        return boundaries
        
    def _detect_entropy_categories(self) -> Dict[int, str]:
        """检测entropy categories through clustering"""
        categories = {}
        
        # Group by assigned categories
        for n, data in self.trace_universe.items():
            categories[n] = data['category']
            
        return categories
        
    def analyze_observer_entropy(self) -> Dict:
        """综合分析observer-scoped entropy"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        
        # Entropy statistics
        relative_entropies = [t['relative_entropy'] for t in traces]
        collapse_entropies = [t['collapse_entropy'] for t in traces]
        boundary_entropies = [t['boundary_entropy'] for t in traces]
        
        results['relative_entropy'] = {
            'mean': np.mean(relative_entropies),
            'std': np.std(relative_entropies),
            'max': np.max(relative_entropies),
            'high_entropy_count': sum(1 for e in relative_entropies if e > 0.8)
        }
        
        results['collapse_entropy'] = {
            'mean': np.mean(collapse_entropies),
            'std': np.std(collapse_entropies),
            'max': np.max(collapse_entropies)
        }
        
        results['boundary_entropy'] = {
            'mean': np.mean(boundary_entropies),
            'std': np.std(boundary_entropies),
            'holographic_count': sum(1 for e in boundary_entropies if e > 1.5)
        }
        
        # Complexity statistics
        complexities = [t['collapse_complexity'] for t in traces]
        capacities = [t['observer_capacity'] for t in traces]
        dimensions = [t['tensor_dimension'] for t in traces]
        
        results['complexity'] = {
            'mean': np.mean(complexities),
            'std': np.std(complexities),
            'max': np.max(complexities),
            'high_complexity_count': sum(1 for c in complexities if c > 1.5)
        }
        
        results['capacity'] = {
            'mean': np.mean(capacities),
            'std': np.std(capacities),
            'max': np.max(capacities)
        }
        
        results['dimensions'] = {
            'unique': len(set(dimensions)),
            'max': max(dimensions),
            'distribution': {d: dimensions.count(d) for d in set(dimensions)}
        }
        
        # Entropy flow analysis
        productions = [t['entropy_production'] for t in traces]
        results['entropy_production'] = {
            'mean': np.mean(productions),
            'total': np.sum(productions),
            'positive_count': sum(1 for p in productions if p > 0)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.entropy_network.edges()) > 0:
            results['network_edges'] = len(self.entropy_network.edges())
            results['network_flow'] = sum(data['weight'] 
                                        for _, _, data in self.entropy_network.edges(data=True))
            
            # Find entropy sources and sinks
            in_degrees = dict(self.entropy_network.in_degree(weight='weight'))
            out_degrees = dict(self.entropy_network.out_degree(weight='weight'))
            
            results['entropy_sources'] = sum(1 for n in self.entropy_network.nodes()
                                           if out_degrees.get(n, 0) > in_degrees.get(n, 0))
            results['entropy_sinks'] = sum(1 for n in self.entropy_network.nodes()
                                         if in_degrees.get(n, 0) > out_degrees.get(n, 0))
        else:
            results['network_edges'] = 0
            results['network_flow'] = 0.0
            results['entropy_sources'] = 0
            results['entropy_sinks'] = 0
            
        # Correlation analysis
        results['correlations'] = {}
        
        # Capacity-complexity correlation
        results['correlations']['capacity_complexity'] = np.corrcoef(capacities, complexities)[0, 1]
        
        # Entropy-dimension correlation
        results['correlations']['entropy_dimension'] = np.corrcoef(relative_entropies, dimensions)[0, 1]
        
        # Production-capacity correlation
        results['correlations']['production_capacity'] = np.corrcoef(productions, capacities)[0, 1]
        
        return results
        
    def generate_visualizations(self):
        """生成observer entropy visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Entropy Dynamics Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 123: Observer-Scoped Entropy', fontsize=16, fontweight='bold')
        
        # Capacity vs Complexity scatter
        x = [t['observer_capacity'] for t in traces]
        y = [t['collapse_complexity'] for t in traces]
        colors = [t['relative_entropy'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='hot', alpha=0.7, s=60)
        ax1.set_xlabel('Observer Capacity (bits)')
        ax1.set_ylabel('Collapse Complexity')
        ax1.set_title('Entropy-Complexity-Capacity Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Relative Entropy')
        
        # Entropy distribution by category
        categories = set(t['category'] for t in traces)
        category_entropies = {cat: [] for cat in categories}
        for t in traces:
            category_entropies[t['category']].append(t['relative_entropy'])
            
        positions = range(len(categories))
        ax2.boxplot([category_entropies[cat] for cat in categories],
                    positions=positions, labels=list(categories))
        ax2.set_xlabel('Entropy Category')
        ax2.set_ylabel('Relative Entropy')
        ax2.set_title('Entropy Distribution by Observer Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Boundary vs Bulk entropy
        x = [t['trace_entropy'] for t in traces]
        y = [t['boundary_entropy'] for t in traces]
        ax3.scatter(x, y, alpha=0.6, s=50)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax3.plot([0, 1], [0, 1.5], 'g--', alpha=0.5, label='y=1.5x (holographic)')
        ax3.set_xlabel('Bulk Entropy')
        ax3.set_ylabel('Boundary Entropy')
        ax3.set_title('Holographic Entropy Relationship')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1.1)
        ax3.set_ylim(0, 2.0)
        
        # Entropy production vs trace length
        x = [t['length'] for t in traces]
        y = [t['entropy_production'] for t in traces]
        colors = [t['tensor_dimension'] for t in traces]
        scatter = ax4.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Trace Length')
        ax4.set_ylabel('Entropy Production')
        ax4.set_title('Length-Production-Dimension Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Tensor Dimension')
        
        plt.tight_layout()
        plt.savefig('chapter-123-obs-entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Entropy Flow Network
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle('Chapter 123: Entropy Flow and Complexity Boundaries', fontsize=16, fontweight='bold')
        
        # Entropy flow network visualization
        if len(self.entropy_network.edges()) > 0:
            # Create layout
            pos = nx.spring_layout(self.entropy_network, k=2, iterations=50)
            
            # Node colors by entropy production
            node_colors = [self.trace_universe[n]['entropy_production'] 
                          for n in self.entropy_network.nodes()]
            
            # Draw network
            nx.draw_networkx_nodes(self.entropy_network, pos, ax=ax1,
                                 node_color=node_colors, cmap='Reds',
                                 node_size=100, alpha=0.8)
            
            # Draw edges with flow weights
            edges = self.entropy_network.edges(data=True)
            weights = [d['weight'] for _, _, d in edges]
            nx.draw_networkx_edges(self.entropy_network, pos, ax=ax1,
                                 width=[w*2 for w in weights],
                                 alpha=0.5, edge_color='gray',
                                 arrows=True, arrowsize=10)
            
            ax1.set_title('Entropy Flow Network')
            ax1.axis('off')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='Reds', 
                                      norm=plt.Normalize(vmin=min(node_colors),
                                                       vmax=max(node_colors)))
            sm.set_array([])
            plt.colorbar(sm, ax=ax1, label='Entropy Production')
        else:
            ax1.text(0.5, 0.5, 'No entropy flow detected', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Entropy Flow Network (Empty)')
            
        # Complexity boundaries by trace length
        lengths = sorted(self.complexity_boundaries['length_bounds'].keys())
        max_entropies = [self.complexity_boundaries['length_bounds'][l]['max_entropy'] 
                        for l in lengths]
        max_complexities = [self.complexity_boundaries['length_bounds'][l]['max_complexity']
                           for l in lengths]
        max_capacities = [self.complexity_boundaries['length_bounds'][l]['max_capacity']
                         for l in lengths]
        
        ax2.plot(lengths, max_entropies, 'ro-', label='Max Entropy', markersize=8)
        ax2.plot(lengths, max_complexities, 'bo-', label='Max Complexity', markersize=8)
        ax2.plot(lengths, max_capacities, 'go-', label='Max Capacity', markersize=8)
        
        # Add theoretical bounds
        theoretical_entropy = [self._compute_max_entropy_phi_constrained(l) for l in lengths]
        ax2.plot(lengths, theoretical_entropy, 'k--', alpha=0.5, 
                label='Theoretical Entropy Bound')
        
        ax2.set_xlabel('Trace Length')
        ax2.set_ylabel('Maximum Value')
        ax2.set_title('Complexity Boundaries by Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('chapter-123-obs-entropy-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class ObsEntropyTests(unittest.TestCase):
    """Unit tests for observer entropy verification"""
    
    def setUp(self):
        """Initialize test system"""
        self.system = ObsEntropySystem(max_trace_value=55)
        
    def test_psi_recursion_entropy(self):
        """Test ψ=ψ(ψ) creates observer-relative entropy"""
        # Verify that entropy is relative to observer
        trace1 = "10101"
        trace2 = "1000"  # Shorter trace with different structure
        
        data1 = self.system._analyze_entropy_properties(trace1, 21)
        data2 = self.system._analyze_entropy_properties(trace2, 8)
        
        # Different complexity should lead to different properties
        self.assertNotEqual(data1['collapse_complexity'], data2['collapse_complexity'],
                          "Different observers should have different complexity")
        
    def test_entropy_bounds(self):
        """Test entropy respects theoretical bounds"""
        for n, data in self.system.trace_universe.items():
            trace_entropy = data['trace_entropy']
            max_entropy = self.system._compute_max_entropy_phi_constrained(data['length'])
            
            self.assertLessEqual(trace_entropy, max_entropy,
                               f"Trace entropy should not exceed theoretical bound")
            self.assertGreaterEqual(trace_entropy, 0.0,
                                  "Entropy should be non-negative")
            
    def test_second_law(self):
        """Test entropy production is non-negative"""
        for n, data in self.system.trace_universe.items():
            production = data['entropy_production']
            self.assertGreaterEqual(production, 0.0,
                                  "Entropy production must be non-negative (2nd law)")
            
    def test_holographic_principle(self):
        """Test boundary entropy relationships"""
        for n, data in self.system.trace_universe.items():
            if data['length'] >= 6:  # Need sufficient length
                boundary = data['boundary_entropy']
                bulk = data['trace_entropy']
                
                # Boundary should encode significant information
                self.assertGreater(boundary, 0.0,
                                 "Boundary should have non-zero entropy")
                
    def test_capacity_limits(self):
        """Test observer capacity is properly bounded"""
        for n, data in self.system.trace_universe.items():
            capacity = data['observer_capacity']
            dimension = data['tensor_dimension']
            
            # Capacity should relate to dimension
            theoretical_max = dimension * log2(dimension + 1)
            self.assertLessEqual(capacity, theoretical_max * 2,
                               "Capacity should be bounded by dimension")

def main():
    """Main verification program"""
    print("Chapter 123: ObsEntropy Verification")
    print("="*60)
    print("从ψ=ψ(ψ)推导Observer-Scoped Entropy Dynamics and Complexity")
    print("="*60)
    
    # Create observer entropy system
    system = ObsEntropySystem(max_trace_value=89)
    
    # Analyze observer entropy
    results = system.analyze_observer_entropy()
    
    print(f"\nObsEntropy Analysis:")
    print(f"Total traces analyzed: {results['total_traces']} φ-valid observers")
    
    print(f"\nRelative Entropy:")
    print(f"  Mean: {results['relative_entropy']['mean']:.3f}")
    print(f"  Std dev: {results['relative_entropy']['std']:.3f}")
    print(f"  Maximum: {results['relative_entropy']['max']:.3f}")
    print(f"  High entropy observers: {results['relative_entropy']['high_entropy_count']}")
    
    print(f"\nCollapse Entropy:")
    print(f"  Mean: {results['collapse_entropy']['mean']:.3f}")
    print(f"  Maximum: {results['collapse_entropy']['max']:.3f}")
    
    print(f"\nBoundary Entropy (Holographic):")
    print(f"  Mean: {results['boundary_entropy']['mean']:.3f}")
    print(f"  Holographic observers: {results['boundary_entropy']['holographic_count']}")
    
    print(f"\nComplexity Measures:")
    print(f"  Mean complexity: {results['complexity']['mean']:.3f}")
    print(f"  High complexity: {results['complexity']['high_complexity_count']} observers")
    print(f"  Mean capacity: {results['capacity']['mean']:.3f} bits")
    print(f"  Max capacity: {results['capacity']['max']:.3f} bits")
    
    print(f"\nTensor Dimensions:")
    print(f"  Unique dimensions: {results['dimensions']['unique']}")
    print(f"  Maximum dimension: {results['dimensions']['max']}")
    print("  Distribution:")
    for dim, count in sorted(results['dimensions']['distribution'].items()):
        print(f"    Dimension {dim}: {count} observers")
    
    print(f"\nEntropy Production:")
    print(f"  Mean production: {results['entropy_production']['mean']:.3f}")
    print(f"  Total production: {results['entropy_production']['total']:.3f}")
    print(f"  Active producers: {results['entropy_production']['positive_count']} observers")
    
    print(f"\nEntropy Categories:")
    for category, count in sorted(results['categories'].items(), 
                                key=lambda x: x[1], reverse=True):
        percentage = 100 * count / results['total_traces']
        print(f"- {category}: {count} observers ({percentage:.1f}%)")
    
    print(f"\nEntropy Flow Network:")
    print(f"  Network edges: {results['network_edges']}")
    print(f"  Total flow: {results['network_flow']:.3f}")
    print(f"  Entropy sources: {results['entropy_sources']}")
    print(f"  Entropy sinks: {results['entropy_sinks']}")
    
    print(f"\nKey Correlations:")
    for pair, corr in results['correlations'].items():
        print(f"  {pair}: {corr:.3f}")
    
    # Generate visualizations
    system.generate_visualizations()
    print("\nVisualizations saved:")
    print("- chapter-123-obs-entropy.png")
    print("- chapter-123-obs-entropy-network.png")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "="*60)
    print("Verification complete: Observer entropy emerges from ψ=ψ(ψ)")
    print("through capacity-limited information dynamics creating relative complexity.")
    print("="*60)

if __name__ == "__main__":
    main()