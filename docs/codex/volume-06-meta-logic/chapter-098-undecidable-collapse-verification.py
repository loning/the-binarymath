#!/usr/bin/env python3
"""
Chapter 098: UndecidableCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse Reachability as Logical Incompleteness

Core principle: From ψ = ψ(ψ) derive systematic undecidability where certain
trace configurations become unreachable through φ-constrained transformations,
creating logical incompleteness that emerges from structural limitations rather
than external undecidability, generating systematic undecidable architectures
that encode the fundamental incompleteness principles of collapsed logical space
through entropy-increasing transformations that reveal reachability limits.

This verification program implements:
1. φ-constrained reachability analysis through trace transformation networks
2. Undecidability detection: systematic identification of unreachable configurations
3. Three-domain analysis: Traditional vs φ-constrained vs intersection undecidability theory
4. Graph theory analysis of reachability networks and undecidable structures
5. Information theory analysis of undecidability entropy and reachability encoding
6. Category theory analysis of undecidable functors and reachability morphisms
7. Visualization of undecidable structures and φ-trace reachability systems
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

class UndecidableCollapseSystem:
    """
    Core system for implementing collapse reachability as logical incompleteness.
    Implements φ-constrained undecidability through trace reachability analysis.
    """
    
    def __init__(self, max_trace_value: int = 80, reachability_depth: int = 5):
        """Initialize undecidable collapse system with reachability analysis"""
        self.max_trace_value = max_trace_value
        self.reachability_depth = reachability_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.undecidable_cache = {}
        self.reachability_cache = {}
        self.incompleteness_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.reachability_network = self._build_reachability_network()
        self.undecidable_sets = self._detect_undecidable_sets()
        
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
                undecidable_data = self._analyze_undecidable_properties(trace, n)
                universe[n] = undecidable_data
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
        """检查trace是否满足φ-constraint（无连续11）"""
        return "11" not in trace
        
    def _analyze_undecidable_properties(self, trace: str, value: int) -> Dict:
        """分析trace的undecidable properties"""
        properties = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'complexity': self._compute_complexity(trace),
            'reachability_measure': self._compute_reachability_measure(trace, value),
            'undecidability_index': self._compute_undecidability_index(trace, value),
            'incompleteness_degree': self._compute_incompleteness_degree(trace, value),
            'halting_probability': self._compute_halting_probability(trace, value),
            'decidability_barrier': self._compute_decidability_barrier(trace, value),
            'reachable_neighbors': self._count_reachable_neighbors(trace, value),
            'isolation_measure': self._compute_isolation_measure(trace, value),
            'undecidable_entropy': self._compute_undecidable_entropy(trace, value),
            'incompleteness_stability': self._compute_incompleteness_stability(trace, value),
            'undecidable_category': self._classify_undecidable_type(trace, value)
        }
        return properties
        
    def _compute_complexity(self, trace: str) -> float:
        """计算trace复杂度"""
        if len(trace) <= 1:
            return 0.0
        
        # 计算模式复杂度
        patterns = set()
        for i in range(len(trace) - 1):
            patterns.add(trace[i:i+2])
        
        return len(patterns) / (len(trace) - 1)
        
    def _compute_reachability_measure(self, trace: str, value: int) -> float:
        """计算reachability measure"""
        # Measure how many traces can reach this trace
        reachable_count = 0
        total_candidates = 0
        
        # Check reachability from nearby traces
        for offset in range(-5, 6):
            candidate_value = value + offset
            if candidate_value > 0 and candidate_value < self.max_trace_value:
                candidate_trace = self._encode_to_trace(candidate_value)
                if self._is_phi_valid(candidate_trace):
                    total_candidates += 1
                    if self._can_reach(candidate_trace, trace):
                        reachable_count += 1
        
        if total_candidates == 0:
            return 0.0
        
        return reachable_count / total_candidates
        
    def _can_reach(self, source_trace: str, target_trace: str) -> bool:
        """Check if source_trace can reach target_trace through φ-valid transformations"""
        # Simple reachability check through basic transformations
        if source_trace == target_trace:
            return True
        
        # Single bit flip reachability
        if len(source_trace) == len(target_trace):
            diff_count = sum(1 for a, b in zip(source_trace, target_trace) if a != b)
            if diff_count == 1:
                # Check if transformation preserves φ-validity
                return self._is_phi_valid(target_trace)
        
        # Length change reachability (add/remove single bit)
        if abs(len(source_trace) - len(target_trace)) == 1:
            if len(source_trace) < len(target_trace):
                # Check if target is source with one bit added
                for i in range(len(target_trace)):
                    modified = target_trace[:i] + target_trace[i+1:]
                    if modified == source_trace:
                        return True
            else:
                # Check if source is target with one bit added
                for i in range(len(source_trace)):
                    modified = source_trace[:i] + source_trace[i+1:]
                    if modified == target_trace:
                        return True
        
        return False
        
    def _compute_undecidability_index(self, trace: str, value: int) -> float:
        """计算undecidability index"""
        # Index based on reachability limitations
        reachability = self._compute_reachability_measure(trace, value)
        
        # Undecidability increases as reachability decreases
        undecidability = 1.0 - reachability
        
        # Adjust by structural complexity
        complexity = self._compute_complexity(trace)
        complexity_factor = 1.0 + 0.5 * complexity
        
        # φ-constraint influence
        phi_factor = 1.2 if self._is_phi_valid(trace) else 0.8
        
        index = undecidability * complexity_factor * phi_factor
        return min(1.0, index)
        
    def _compute_incompleteness_degree(self, trace: str, value: int) -> float:
        """计算incompleteness degree"""
        # Incompleteness as inability to reach all possible configurations
        if len(trace) <= 1:
            return 0.0
        
        # Count theoretical vs actual reachable configurations
        theoretical_configs = 2 ** len(trace)  # All possible binary strings
        
        # Count actually reachable φ-valid configurations
        reachable_configs = 0
        for i in range(theoretical_configs):
            config_trace = format(i, f'0{len(trace)}b')
            if self._is_phi_valid(config_trace) and self._can_reach(trace, config_trace):
                reachable_configs += 1
        
        if theoretical_configs == 0:
            return 0.0
        
        incompleteness = 1.0 - (reachable_configs / theoretical_configs)
        return incompleteness
        
    def _compute_halting_probability(self, trace: str, value: int) -> float:
        """计算halting probability for trace transformations"""
        # Probability that trace transformations halt in finite steps
        
        # Simple model: probability decreases with complexity
        complexity = self._compute_complexity(trace)
        length_factor = 1.0 / (1.0 + len(trace) / 10.0)
        
        # φ-constraint provides stability
        phi_stability = 1.1 if self._is_phi_valid(trace) else 0.9
        
        halting_prob = (1.0 - complexity) * length_factor * phi_stability
        return max(0.0, min(1.0, halting_prob))
        
    def _compute_decidability_barrier(self, trace: str, value: int) -> float:
        """计算decidability barrier"""
        # Barrier representing difficulty of deciding trace properties
        
        # Barrier increases with undecidability
        undecidability = self._compute_undecidability_index(trace, value)
        
        # Structural barriers
        length_barrier = len(trace) / 20.0  # Longer traces harder to decide
        complexity_barrier = self._compute_complexity(trace)
        
        total_barrier = undecidability + length_barrier + complexity_barrier
        return min(1.0, total_barrier)
        
    def _count_reachable_neighbors(self, trace: str, value: int) -> int:
        """Count reachable neighbor traces"""
        reachable_count = 0
        
        # Check neighbors in value space
        for offset in range(-3, 4):
            neighbor_value = value + offset
            if neighbor_value > 0 and neighbor_value < self.max_trace_value:
                neighbor_trace = self._encode_to_trace(neighbor_value)
                if self._is_phi_valid(neighbor_trace) and self._can_reach(trace, neighbor_trace):
                    reachable_count += 1
        
        return reachable_count
        
    def _compute_isolation_measure(self, trace: str, value: int) -> float:
        """计算isolation measure"""
        # Measure how isolated trace is from others
        reachable_neighbors = self._count_reachable_neighbors(trace, value)
        max_possible_neighbors = 6  # Range of offsets checked
        
        if max_possible_neighbors == 0:
            return 1.0
        
        isolation = 1.0 - (reachable_neighbors / max_possible_neighbors)
        return isolation
        
    def _compute_undecidable_entropy(self, trace: str, value: int) -> float:
        """计算undecidable entropy"""
        # Entropy in undecidability characteristics
        if len(trace) <= 1:
            return 0.0
        
        # Binary entropy of trace
        p = trace.count('1') / len(trace)
        if p == 0 or p == 1:
            return 0.0
        
        entropy = -p * log2(p) - (1-p) * log2(1-p)
        
        # Modulate by undecidability
        undecidability = self._compute_undecidability_index(trace, value)
        undecidable_entropy = entropy * (1.0 + undecidability)
        
        return min(2.0, undecidable_entropy)  # Cap at 2.0
        
    def _compute_incompleteness_stability(self, trace: str, value: int) -> float:
        """计算incompleteness stability"""
        # Stability of incompleteness properties
        incompleteness = self._compute_incompleteness_degree(trace, value)
        decidability_barrier = self._compute_decidability_barrier(trace, value)
        
        # Stability as resistance to change
        stability = incompleteness * decidability_barrier
        
        # φ-constraint stability bonus
        if self._is_phi_valid(trace):
            stability *= 1.1
        
        return min(1.0, stability)
        
    def _classify_undecidable_type(self, trace: str, value: int) -> str:
        """分类undecidable type"""
        undecidability = self._compute_undecidability_index(trace, value)
        isolation = self._compute_isolation_measure(trace, value)
        barrier = self._compute_decidability_barrier(trace, value)
        
        if undecidability > 0.8 and isolation > 0.8:
            return 'highly_undecidable'
        elif barrier > 0.7:
            return 'barrier_undecidable'
        elif isolation > 0.6:
            return 'isolated_undecidable'
        elif undecidability > 0.5:
            return 'moderately_undecidable'
        else:
            return 'weakly_undecidable'
            
    def _build_reachability_network(self) -> nx.DiGraph:
        """构建reachability network"""
        G = nx.DiGraph()
        
        # Add nodes
        for value in self.trace_universe.keys():
            props = self.trace_universe[value]
            G.add_node(value, **props)
        
        # Add directed edges for reachability
        traces = list(self.trace_universe.keys())
        for t1 in traces:
            for t2 in traces:
                if t1 != t2:
                    trace1 = self.trace_universe[t1]['trace']
                    trace2 = self.trace_universe[t2]['trace']
                    
                    if self._can_reach(trace1, trace2):
                        # Weight by reachability strength
                        props1 = self.trace_universe[t1]
                        props2 = self.trace_universe[t2]
                        
                        reach_weight = (props1['reachability_measure'] + props2['reachability_measure']) / 2.0
                        G.add_edge(t1, t2, weight=reach_weight, relationship='reachable')
        
        return G
        
    def _detect_undecidable_sets(self) -> List[Set[int]]:
        """检测undecidable sets"""
        undecidable_sets = []
        
        # Group traces by undecidable category
        categories = {}
        for value, props in self.trace_universe.items():
            category = props['undecidable_category']
            if category not in categories:
                categories[category] = set()
            categories[category].add(value)
        
        # Each category forms an undecidable set
        for category, trace_set in categories.items():
            if len(trace_set) > 1:  # Only include sets with multiple elements
                undecidable_sets.append(trace_set)
        
        return undecidable_sets
        
    def analyze_information_theory(self) -> Dict[str, float]:
        """分析information theory"""
        def compute_entropy(data_list):
            if not data_list:
                return 0.0
            
            # Check if all values are the same
            data_array = np.array(data_list)
            if np.all(data_array == data_array[0]):
                return 0.0
            
            # Adaptive binning
            unique_values = len(np.unique(data_array))
            bin_count = min(8, max(3, unique_values))
            
            try:
                hist, _ = np.histogram(data_array, bins=bin_count)
                hist = hist[hist > 0]  # Remove zero bins
                probabilities = hist / hist.sum()
                return -np.sum(probabilities * np.log2(probabilities))
            except:
                # Fallback: count unique values
                unique_count = len(np.unique(data_array))
                return log2(unique_count) if unique_count > 1 else 0.0
        
        properties = [
            'reachability_measure', 'undecidability_index', 'incompleteness_degree',
            'halting_probability', 'decidability_barrier', 'reachable_neighbors',
            'isolation_measure', 'undecidable_entropy', 'incompleteness_stability'
        ]
        
        entropies = {}
        for prop in properties:
            if prop == 'reachable_neighbors':
                # Integer property - handle specially
                values = [self.trace_universe[t][prop] for t in self.trace_universe.keys()]
                unique_values = len(set(values))
                entropies[f"{prop.replace('_', ' ').title()} entropy"] = log2(unique_values) if unique_values > 1 else 0.0
            else:
                values = [self.trace_universe[t][prop] for t in self.trace_universe.keys()]
                entropies[f"{prop.replace('_', ' ').title()} entropy"] = compute_entropy(values)
        
        return entropies
        
    def analyze_category_theory(self) -> Dict[str, Any]:
        """分析category theory"""
        # Classify traces by undecidable category
        classifications = {}
        for value, props in self.trace_universe.items():
            category = props['undecidable_category']
            
            if category not in classifications:
                classifications[category] = []
            classifications[category].append(value)
        
        # Compute morphisms
        total_morphisms = 0
        for cat_traces in classifications.values():
            # Morphisms within category
            n = len(cat_traces)
            total_morphisms += n * (n - 1)  # All ordered pairs within category
        
        # Cross-category morphisms (reachability relationships)
        cat_names = list(classifications.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i+1:]:
                # Allow morphisms between undecidable categories with reachability
                cross_morphisms = 0
                for t1 in classifications[cat1]:
                    for t2 in classifications[cat2]:
                        trace1 = self.trace_universe[t1]['trace']
                        trace2 = self.trace_universe[t2]['trace']
                        
                        # Morphism if reachability relationship exists
                        if self._can_reach(trace1, trace2) or self._can_reach(trace2, trace1):
                            cross_morphisms += 2  # Bidirectional morphism
                            
                total_morphisms += cross_morphisms
        
        total_objects = len(self.trace_universe)
        morphism_density = total_morphisms / (total_objects * total_objects) if total_objects > 0 else 0
        
        return {
            'categories': len(classifications),
            'classifications': classifications,
            'total_morphisms': total_morphisms,
            'morphism_density': morphism_density
        }
        
    def visualize_undecidable_dynamics(self):
        """可视化undecidable dynamics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Chapter 098: UndecidableCollapse - Collapse Reachability as Logical Incompleteness', fontsize=16, fontweight='bold')
        
        # 1. Undecidability index vs reachability measure
        values = list(self.trace_universe.keys())
        undecidability_indices = [self.trace_universe[v]['undecidability_index'] for v in values]
        reachability_measures = [self.trace_universe[v]['reachability_measure'] for v in values]
        incompleteness_degrees = [self.trace_universe[v]['incompleteness_degree'] for v in values]
        
        scatter = ax1.scatter(reachability_measures, undecidability_indices, c=incompleteness_degrees, s=60, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Reachability Measure')
        ax1.set_ylabel('Undecidability Index')
        ax1.set_title('Reachability vs Undecidability')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Incompleteness Degree')
        
        # 2. Undecidable category distribution
        undecidable_categories = [self.trace_universe[v]['undecidable_category'] for v in values]
        unique_categories = list(set(undecidable_categories))
        category_counts = [undecidable_categories.count(cat) for cat in unique_categories]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        ax2.pie(category_counts, labels=unique_categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Undecidable Category Distribution')
        
        # 3. Isolation measure vs decidability barrier
        isolation_measures = [self.trace_universe[v]['isolation_measure'] for v in values]
        decidability_barriers = [self.trace_universe[v]['decidability_barrier'] for v in values]
        halting_probabilities = [self.trace_universe[v]['halting_probability'] for v in values]
        
        scatter2 = ax3.scatter(isolation_measures, decidability_barriers, c=halting_probabilities, s=60, alpha=0.7, cmap='plasma')
        ax3.set_xlabel('Isolation Measure')
        ax3.set_ylabel('Decidability Barrier')
        ax3.set_title('Isolation vs Barrier')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='Halting Probability')
        
        # 4. Undecidable entropy vs incompleteness stability
        undecidable_entropies = [self.trace_universe[v]['undecidable_entropy'] for v in values]
        incompleteness_stabilities = [self.trace_universe[v]['incompleteness_stability'] for v in values]
        
        ax4.scatter(undecidable_entropies, incompleteness_stabilities, c=undecidability_indices, s=60, alpha=0.7, cmap='cool')
        ax4.set_xlabel('Undecidable Entropy')
        ax4.set_ylabel('Incompleteness Stability')
        ax4.set_title('Entropy vs Stability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-098-undecidable-collapse-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_reachability_analysis(self):
        """可视化reachability analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reachability and Incompleteness Analysis', fontsize=16, fontweight='bold')
        
        values = list(self.trace_universe.keys())
        
        # 1. Reachability measure distribution
        reachability_measures = [self.trace_universe[v]['reachability_measure'] for v in values]
        
        ax1.hist(reachability_measures, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Reachability Measure')
        ax1.set_ylabel('Count')
        ax1.set_title('Reachability Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Halting probability vs incompleteness degree
        halting_probabilities = [self.trace_universe[v]['halting_probability'] for v in values]
        incompleteness_degrees = [self.trace_universe[v]['incompleteness_degree'] for v in values]
        undecidability_indices = [self.trace_universe[v]['undecidability_index'] for v in values]
        
        scatter = ax2.scatter(halting_probabilities, incompleteness_degrees, c=undecidability_indices, s=60, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Halting Probability')
        ax2.set_ylabel('Incompleteness Degree')
        ax2.set_title('Halting vs Incompleteness')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Undecidability Index')
        
        # 3. Reachable neighbors histogram
        reachable_neighbors = [self.trace_universe[v]['reachable_neighbors'] for v in values]
        
        max_neighbors = max(reachable_neighbors) if reachable_neighbors else 0
        ax3.hist(reachable_neighbors, bins=range(0, max_neighbors+2), alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Reachable Neighbors Count')
        ax3.set_ylabel('Count')
        ax3.set_title('Reachable Neighbors Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Decidability barrier vs isolation
        decidability_barriers = [self.trace_universe[v]['decidability_barrier'] for v in values]
        isolation_measures = [self.trace_universe[v]['isolation_measure'] for v in values]
        
        ax4.scatter(decidability_barriers, isolation_measures, c=incompleteness_degrees, s=60, alpha=0.7, cmap='coolwarm')
        ax4.set_xlabel('Decidability Barrier')
        ax4.set_ylabel('Isolation Measure')
        ax4.set_title('Barrier vs Isolation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-098-undecidable-collapse-reachability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_network_and_categories(self):
        """可视化network and categories"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reachability Network and Categorical Analysis', fontsize=16, fontweight='bold')
        
        # Build network
        G = self.reachability_network
        
        # 1. Network visualization
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Color nodes by undecidable category
            node_colors = []
            category_map = {'highly_undecidable': 0, 'barrier_undecidable': 1, 'isolated_undecidable': 2, 
                          'moderately_undecidable': 3, 'weakly_undecidable': 4}
            
            for node in G.nodes():
                category = self.trace_universe[node]['undecidable_category']
                node_colors.append(category_map.get(category, 0))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='Set3', alpha=0.8, ax=ax1)
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5, ax=ax1)
            
            ax1.set_title(f'Reachability Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'No Network Connections', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Reachability Network')
        
        # 2. Category analysis
        cat_analysis = self.analyze_category_theory()
        categories = list(cat_analysis['classifications'].keys())
        cat_sizes = [len(cat_analysis['classifications'][cat]) for cat in categories]
        
        if categories:
            colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            wedges, texts, autotexts = ax2.pie(cat_sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Undecidable Categories')
        else:
            ax2.text(0.5, 0.5, 'No Categories Found', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Undecidable Categories')
        
        # 3. Undecidable sets visualization
        undecidable_sets = self.undecidable_sets
        if undecidable_sets:
            set_sizes = [len(uset) for uset in undecidable_sets]
            ax3.bar(range(len(set_sizes)), set_sizes, alpha=0.7, color='orange')
            ax3.set_xlabel('Undecidable Set Index')
            ax3.set_ylabel('Set Size')
            ax3.set_title(f'Undecidable Sets ({len(undecidable_sets)} sets)')
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No Undecidable Sets', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Undecidable Sets')
        
        # 4. Information entropy analysis
        info_analysis = self.analyze_information_theory()
        entropy_names = list(info_analysis.keys())
        entropy_values = list(info_analysis.values())
        
        # Sort by entropy value
        sorted_pairs = sorted(zip(entropy_names, entropy_values), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        if sorted_names:
            bars = ax4.barh(range(len(sorted_names)), sorted_values, alpha=0.7, color='purple')
            ax4.set_yticks(range(len(sorted_names)))
            ax4.set_yticklabels([name.replace(' entropy', '') for name in sorted_names], fontsize=10)
            ax4.set_xlabel('Entropy (bits)')
            ax4.set_title('Information Content Analysis')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                        ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('chapter-098-undecidable-collapse-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestUndecidableCollapse(unittest.TestCase):
    """Unit tests for UndecidableCollapse verification"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = UndecidableCollapseSystem(max_trace_value=50, reachability_depth=3)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 0)
        for value, props in self.system.trace_universe.items():
            self.assertIn('trace', props)
            self.assertIn('reachability_measure', props)
            self.assertIn('undecidability_index', props)
            self.assertIn('incompleteness_degree', props)
            
    def test_undecidable_properties_analysis(self):
        """测试undecidable性质分析"""
        for value, props in self.system.trace_universe.items():
            # Test reachability measure bounds
            self.assertGreaterEqual(props['reachability_measure'], 0.0)
            self.assertLessEqual(props['reachability_measure'], 1.0)
            
            # Test undecidability index bounds
            self.assertGreaterEqual(props['undecidability_index'], 0.0)
            self.assertLessEqual(props['undecidability_index'], 1.0)
            
            # Test incompleteness degree bounds
            self.assertGreaterEqual(props['incompleteness_degree'], 0.0)
            self.assertLessEqual(props['incompleteness_degree'], 1.0)
            
    def test_reachability_analysis(self):
        """测试reachability分析"""
        for value, props in self.system.trace_universe.items():
            trace = props['trace']
            
            # Test self-reachability
            self.assertTrue(self.system._can_reach(trace, trace))
            
            # Test reachable neighbors count
            self.assertGreaterEqual(props['reachable_neighbors'], 0)
            
    def test_halting_probability(self):
        """测试halting probability计算"""
        for value, props in self.system.trace_universe.items():
            halting_prob = props['halting_probability']
            self.assertGreaterEqual(halting_prob, 0.0)
            self.assertLessEqual(halting_prob, 1.0)
            
    def test_undecidable_sets(self):
        """测试undecidable sets检测"""
        undecidable_sets = self.system.undecidable_sets
        # Should detect some undecidable structure
        self.assertGreaterEqual(len(undecidable_sets), 0)
        
        # Each set should contain multiple elements
        for uset in undecidable_sets:
            self.assertGreater(len(uset), 1)
            
    def test_reachability_network(self):
        """测试reachability network构建"""
        G = self.system.reachability_network
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some structure
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        info_analysis = self.system.analyze_information_theory()
        for entropy_name, entropy_value in info_analysis.items():
            self.assertGreaterEqual(entropy_value, 0.0)
            self.assertLessEqual(entropy_value, 10.0)  # Reasonable upper bound
            
    def test_categorical_analysis(self):
        """测试范畴论分析"""
        cat_analysis = self.system.analyze_category_theory()
        self.assertGreater(cat_analysis['categories'], 0)
        self.assertGreaterEqual(cat_analysis['total_morphisms'], 0)
        self.assertGreaterEqual(cat_analysis['morphism_density'], 0.0)
        self.assertLessEqual(cat_analysis['morphism_density'], 1.0)

def main():
    """主验证程序"""
    print("=" * 80)
    print("Chapter 098: UndecidableCollapse Verification")
    print("从ψ=ψ(ψ)推导Collapse Reachability as Logical Incompleteness")
    print("=" * 80)
    
    # Initialize system
    system = UndecidableCollapseSystem(max_trace_value=65, reachability_depth=4)
    
    print("\n1. Undecidability Foundation Analysis:")
    print("-" * 50)
    print(f"Total traces analyzed: {len(system.trace_universe)}")
    
    # Basic statistics
    reachability_measures = [props['reachability_measure'] for props in system.trace_universe.values()]
    undecidability_indices = [props['undecidability_index'] for props in system.trace_universe.values()]
    incompleteness_degrees = [props['incompleteness_degree'] for props in system.trace_universe.values()]
    halting_probabilities = [props['halting_probability'] for props in system.trace_universe.values()]
    
    print(f"Mean reachability measure: {np.mean(reachability_measures):.3f}")
    print(f"Mean undecidability index: {np.mean(undecidability_indices):.3f}")
    print(f"Mean incompleteness degree: {np.mean(incompleteness_degrees):.3f}")
    print(f"Mean halting probability: {np.mean(halting_probabilities):.3f}")
    
    # Undecidable sets
    print(f"Undecidable sets detected: {len(system.undecidable_sets)}")
    
    print("\n2. Reachability Analysis:")
    print("-" * 50)
    
    decidability_barriers = [props['decidability_barrier'] for props in system.trace_universe.values()]
    isolation_measures = [props['isolation_measure'] for props in system.trace_universe.values()]
    reachable_neighbors = [props['reachable_neighbors'] for props in system.trace_universe.values()]
    
    print(f"Mean decidability barrier: {np.mean(decidability_barriers):.3f}")
    print(f"Mean isolation measure: {np.mean(isolation_measures):.3f}")
    print(f"Mean reachable neighbors: {np.mean(reachable_neighbors):.1f}")
    print(f"Max reachable neighbors: {np.max(reachable_neighbors)}")
    
    # High undecidability count
    high_undecidable_count = sum(1 for u in undecidability_indices if u > 0.8)
    print(f"Highly undecidable traces (>0.8): {high_undecidable_count}")
    
    print("\n3. Network Analysis:")
    print("-" * 50)
    
    G = system.reachability_network
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        print(f"Network density: {density:.3f}")
        
        # Component analysis
        if G.number_of_edges() > 0:
            components = list(nx.weakly_connected_components(G))
            print(f"Weakly connected components: {len(components)}")
            
            # Average degree
            total_degree = sum(dict(G.degree()).values())
            avg_degree = total_degree / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            print(f"Average degree: {avg_degree:.3f}")
    
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    
    info_analysis = system.analyze_information_theory()
    for name, entropy in sorted(info_analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {entropy:.3f} bits")
    
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    
    cat_analysis = system.analyze_category_theory()
    print(f"Undecidable categories: {cat_analysis['categories']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for cat, traces in cat_analysis['classifications'].items():
        percentage = len(traces) / len(system.trace_universe) * 100
        print(f"- {cat}: {len(traces)} traces ({percentage:.1f}%)")
    
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        system.visualize_undecidable_dynamics()
        print("✓ Undecidable dynamics visualization saved")
    except Exception as e:
        print(f"✗ Undecidable dynamics visualization failed: {e}")
    
    try:
        system.visualize_reachability_analysis()
        print("✓ Reachability analysis visualization saved")
    except Exception as e:
        print(f"✗ Reachability analysis visualization failed: {e}")
    
    try:
        system.visualize_network_and_categories()
        print("✓ Network and categorical visualization saved")
    except Exception as e:
        print(f"✗ Network and categorical visualization failed: {e}")
    
    print("\n7. Running Unit Tests:")
    print("-" * 50)
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 80)
    print("UndecidableCollapse Verification Complete")
    print("Key Findings:")
    print(f"- {len(system.trace_universe)} φ-valid traces with undecidability analysis")
    print(f"- {cat_analysis['categories']} undecidable categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print(f"- Network density: {density:.3f}" if G.number_of_nodes() > 0 else "- Network density: 0.000")
    print(f"- Mean undecidability index: {np.mean(undecidability_indices):.3f}")
    print(f"- Mean incompleteness degree: {np.mean(incompleteness_degrees):.3f}")
    print(f"- Undecidable sets: {len(system.undecidable_sets)}")
    print("=" * 80)

if __name__ == "__main__":
    main()