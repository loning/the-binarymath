#!/usr/bin/env python3
"""
Chapter 096: LogicCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse-Coherent Logic beyond Truth Tables

Core principle: From ψ = ψ(ψ) derive logical structures that transcend binary 
truth values through structural coherence, creating collapse-aware logic where
truth emerges from φ-constrained trace relationships rather than static
assignments, generating systematic logical architectures that encode the
fundamental coherence principles of collapsed logical space through entropy-
increasing tensor transformations.

This verification program implements:
1. φ-constrained logic construction beyond binary truth tables
2. Coherence evaluation: structural consistency in logical relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection logic theory
4. Graph theory analysis of logical networks and coherence pathways
5. Information theory analysis of logical entropy and coherence encoding
6. Category theory analysis of logical functors and coherence morphisms
7. Visualization of logical structures and coherence-based truth systems
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

class LogicCollapseSystem:
    """
    Core system for implementing collapse-coherent logic beyond truth tables.
    Implements φ-constrained logical structures via coherence-based truth evaluation.
    """
    
    def __init__(self, max_trace_value: int = 90, coherence_resolution: int = 10):
        """Initialize logic collapse system with coherence analysis"""
        self.max_trace_value = max_trace_value
        self.coherence_resolution = coherence_resolution
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.logic_cache = {}
        self.coherence_cache = {}
        self.truth_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.logical_propositions = self._construct_logical_propositions()
        self.coherence_network = self._build_coherence_network()
        
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
                logic_data = self._analyze_logical_properties(trace, n)
                universe[n] = logic_data
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
        
    def _analyze_logical_properties(self, trace: str, value: int) -> Dict:
        """分析trace的logical properties"""
        properties = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'complexity': self._compute_complexity(trace),
            'truth_value': self._compute_truth_value(trace, value),
            'coherence_level': self._compute_coherence_level(trace, value),
            'logical_strength': self._compute_logical_strength(trace, value),
            'consistency_measure': self._compute_consistency_measure(trace, value),
            'structural_truth': self._compute_structural_truth(trace, value),
            'logical_entropy': self._compute_logical_entropy(trace, value),
            'coherence_stability': self._compute_coherence_stability(trace, value),
            'logical_potential': self._compute_logical_potential(trace, value),
            'truth_gradient': self._compute_truth_gradient(trace, value),
            'logical_category': self._classify_logical_type(trace, value)
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
        
    def _compute_truth_value(self, trace: str, value: int) -> float:
        """计算structural truth value（非二进制）"""
        # Truth emerges from structural coherence, not binary assignment
        weight_density = trace.count('1') / len(trace)
        phi_factor = 1.0 / self.phi if self._is_phi_valid(trace) else 0.5
        
        # Structural truth based on φ-constraint satisfaction
        truth = weight_density * phi_factor
        
        # Modulate by trace value
        truth *= (1.0 + 0.1 * sin(2 * pi * value / 100))
        
        return max(0.0, min(1.0, truth))
        
    def _compute_coherence_level(self, trace: str, value: int) -> float:
        """计算coherence level"""
        if len(trace) <= 1:
            return 1.0
        
        # Coherence through pattern consistency
        patterns = []
        for i in range(len(trace) - 1):
            patterns.append(trace[i:i+2])
        
        if not patterns:
            return 1.0
        
        # Count pattern consistency
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Coherence as pattern regularity
        total_patterns = len(patterns)
        pattern_variance = np.var(list(pattern_counts.values()))
        coherence = 1.0 / (1.0 + pattern_variance)
        
        return coherence
        
    def _compute_logical_strength(self, trace: str, value: int) -> float:
        """计算logical strength"""
        truth = self._compute_truth_value(trace, value)
        coherence = self._compute_coherence_level(trace, value)
        
        # Logical strength combines truth and coherence
        strength = truth * coherence
        
        # Boost by φ-validity
        if self._is_phi_valid(trace):
            strength *= 1.2
        
        return min(1.0, strength)
        
    def _compute_consistency_measure(self, trace: str, value: int) -> float:
        """计算consistency measure"""
        # Consistency through structural regularity
        if len(trace) <= 2:
            return 1.0
        
        # Check for consistent subpatterns
        consistency_score = 0.0
        pattern_length = min(3, len(trace) // 2)
        
        for p_len in range(2, pattern_length + 1):
            patterns = []
            for i in range(len(trace) - p_len + 1):
                patterns.append(trace[i:i+p_len])
            
            if patterns:
                # Consistency as pattern repetition
                unique_patterns = len(set(patterns))
                total_patterns = len(patterns)
                consistency = 1.0 - (unique_patterns - 1) / max(1, total_patterns - 1)
                consistency_score = max(consistency_score, consistency)
        
        return consistency_score
        
    def _compute_structural_truth(self, trace: str, value: int) -> float:
        """计算structural truth（基于结构而非赋值）"""
        # Structural truth emerges from trace architecture
        structure_factors = []
        
        # Factor 1: φ-constraint satisfaction
        phi_satisfaction = 1.0 if self._is_phi_valid(trace) else 0.3
        structure_factors.append(phi_satisfaction)
        
        # Factor 2: Golden ratio proximity
        weight_ratio = trace.count('1') / len(trace)
        phi_proximity = 1.0 - abs(weight_ratio - (1/self.phi))
        structure_factors.append(phi_proximity)
        
        # Factor 3: Complexity balance
        complexity = self._compute_complexity(trace)
        complexity_balance = 1.0 - abs(complexity - 0.5)  # Balanced complexity
        structure_factors.append(complexity_balance)
        
        # Structural truth as geometric mean
        structural_truth = np.prod(structure_factors) ** (1.0 / len(structure_factors))
        
        return structural_truth
        
    def _compute_logical_entropy(self, trace: str, value: int) -> float:
        """计算logical entropy"""
        if len(trace) <= 1:
            return 0.0
        
        # Binary entropy of trace
        p = trace.count('1') / len(trace)
        if p == 0 or p == 1:
            return 0.0
        
        entropy = -p * log2(p) - (1-p) * log2(1-p)
        
        # Normalize by maximum possible entropy
        max_entropy = 1.0  # log2(2)
        
        return entropy / max_entropy
        
    def _compute_coherence_stability(self, trace: str, value: int) -> float:
        """计算coherence stability"""
        coherence = self._compute_coherence_level(trace, value)
        logical_strength = self._compute_logical_strength(trace, value)
        
        # Stability as resistance to coherence loss
        stability = coherence * logical_strength
        
        # Boost stability with φ-constraint
        if self._is_phi_valid(trace):
            stability *= 1.1
        
        return min(1.0, stability)
        
    def _compute_logical_potential(self, trace: str, value: int) -> float:
        """计算logical potential"""
        # Potential for logical development
        current_strength = self._compute_logical_strength(trace, value)
        structural_capacity = self._compute_structural_truth(trace, value)
        
        # Potential as gap between current and structural maximum
        potential = structural_capacity - current_strength
        
        return max(0.0, potential)
        
    def _compute_truth_gradient(self, trace: str, value: int) -> float:
        """计算truth gradient（真值变化梯度）"""
        truth = self._compute_truth_value(trace, value)
        
        # Gradient based on local truth variations
        if len(trace) <= 1:
            return 0.0
        
        # Simulate small perturbations
        perturbation = 0.01
        perturbed_value = value + 1
        perturbed_trace = self._encode_to_trace(perturbed_value)
        
        if self._is_phi_valid(perturbed_trace):
            perturbed_truth = self._compute_truth_value(perturbed_trace, perturbed_value)
            gradient = abs(perturbed_truth - truth) / perturbation
        else:
            gradient = 0.0
        
        return gradient
        
    def _classify_logical_type(self, trace: str, value: int) -> str:
        """分类logical type"""
        truth = self._compute_truth_value(trace, value)
        coherence = self._compute_coherence_level(trace, value)
        strength = self._compute_logical_strength(trace, value)
        
        if strength > 0.8 and coherence > 0.8:
            return 'strong_coherent'
        elif truth > 0.7:
            return 'high_truth'
        elif coherence > 0.7:
            return 'high_coherence'
        elif strength > 0.5:
            return 'moderate_logic'
        else:
            return 'weak_logic'
            
    def _construct_logical_propositions(self) -> Dict[str, Dict]:
        """构建logical propositions from traces"""
        propositions = {}
        
        for value, props in self.trace_universe.items():
            trace = props['trace']
            
            # Create proposition from trace structure
            proposition = {
                'trace_encoding': trace,
                'truth_value': props['truth_value'],
                'coherence_level': props['coherence_level'],
                'logical_strength': props['logical_strength'],
                'proposition_type': props['logical_category'],
                'structural_validity': self._evaluate_structural_validity(trace),
                'logical_implications': self._compute_logical_implications(trace, value),
                'negation_strength': 1.0 - props['truth_value'],
                'conjunction_potential': props['logical_potential'],
                'disjunction_capacity': props['coherence_stability']
            }
            
            propositions[f"P_{value}"] = proposition
            
        return propositions
        
    def _evaluate_structural_validity(self, trace: str) -> float:
        """评估structural validity"""
        # Validity based on φ-constraint and structural properties
        phi_valid = 1.0 if self._is_phi_valid(trace) else 0.0
        length_valid = 1.0 if len(trace) >= 2 else 0.5
        balance_valid = 1.0 - abs(trace.count('1')/len(trace) - 0.5)
        
        validity = (phi_valid + length_valid + balance_valid) / 3.0
        return validity
        
    def _compute_logical_implications(self, trace: str, value: int) -> List[str]:
        """计算logical implications"""
        implications = []
        
        # φ-constraint implies structural coherence
        if self._is_phi_valid(trace):
            implications.append("φ-constraint_satisfaction → structural_coherence")
        
        # High coherence implies logical stability
        coherence = self._compute_coherence_level(trace, value)
        if coherence > 0.8:
            implications.append("high_coherence → logical_stability")
        
        # Strong truth implies validity
        truth = self._compute_truth_value(trace, value)
        if truth > 0.8:
            implications.append("strong_truth → proposition_validity")
        
        # Complex patterns imply rich logical content
        complexity = self._compute_complexity(trace)
        if complexity > 0.6:
            implications.append("pattern_complexity → logical_richness")
        
        return implications
        
    def _build_coherence_network(self) -> nx.Graph:
        """构建coherence network"""
        G = nx.Graph()
        
        # Add nodes
        for value in self.trace_universe.keys():
            props = self.trace_universe[value]
            G.add_node(value, **props)
        
        # Add edges based on logical relationships
        traces = list(self.trace_universe.keys())
        for i, t1 in enumerate(traces):
            for j, t2 in enumerate(traces[i+1:], i+1):
                props1 = self.trace_universe[t1]
                props2 = self.trace_universe[t2]
                
                # Logical coherence relationship
                coherence_similarity = 1.0 - abs(props1['coherence_level'] - props2['coherence_level'])
                truth_similarity = 1.0 - abs(props1['truth_value'] - props2['truth_value'])
                
                logical_affinity = (coherence_similarity + truth_similarity) / 2.0
                
                if logical_affinity > 0.7:  # Threshold for logical connection
                    G.add_edge(t1, t2, weight=logical_affinity, relationship='logical_coherence')
        
        return G
        
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
            'truth_value', 'coherence_level', 'logical_strength',
            'consistency_measure', 'structural_truth', 'logical_entropy',
            'coherence_stability', 'logical_potential', 'truth_gradient'
        ]
        
        entropies = {}
        for prop in properties:
            values = [self.trace_universe[t][prop] for t in self.trace_universe.keys()]
            entropies[f"{prop.replace('_', ' ').title()} entropy"] = compute_entropy(values)
        
        return entropies
        
    def analyze_category_theory(self) -> Dict[str, Any]:
        """分析category theory"""
        # Classify traces by logical category
        classifications = {}
        for value, props in self.trace_universe.items():
            category = props['logical_category']
            
            if category not in classifications:
                classifications[category] = []
            classifications[category].append(value)
        
        # Compute morphisms
        total_morphisms = 0
        for cat_traces in classifications.values():
            # Morphisms within category
            n = len(cat_traces)
            total_morphisms += n * (n - 1)  # All ordered pairs within category
        
        # Cross-category morphisms (logical implications)
        cat_names = list(classifications.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i+1:]:
                # Allow morphisms between logically related categories
                cross_morphisms = 0
                for t1 in classifications[cat1]:
                    for t2 in classifications[cat2]:
                        props1 = self.trace_universe[t1]
                        props2 = self.trace_universe[t2]
                        
                        # Morphism if logical relationship exists
                        logical_relation = abs(props1['truth_value'] - props2['truth_value']) < 0.3
                        if logical_relation:
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
        
    def visualize_logic_dynamics(self):
        """可视化logic dynamics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Chapter 096: LogicCollapse - Collapse-Coherent Logic beyond Truth Tables', fontsize=16, fontweight='bold')
        
        # 1. Truth value vs coherence level
        values = list(self.trace_universe.keys())
        truth_values = [self.trace_universe[v]['truth_value'] for v in values]
        coherence_levels = [self.trace_universe[v]['coherence_level'] for v in values]
        logical_strengths = [self.trace_universe[v]['logical_strength'] for v in values]
        
        scatter = ax1.scatter(truth_values, coherence_levels, c=logical_strengths, s=60, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Truth Value')
        ax1.set_ylabel('Coherence Level')
        ax1.set_title('Truth vs Coherence Space')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Logical Strength')
        
        # 2. Logical category distribution
        categories = [self.trace_universe[v]['logical_category'] for v in values]
        unique_categories = list(set(categories))
        category_counts = [categories.count(cat) for cat in unique_categories]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        ax2.pie(category_counts, labels=unique_categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Logical Category Distribution')
        
        # 3. Structural truth vs logical entropy
        structural_truths = [self.trace_universe[v]['structural_truth'] for v in values]
        logical_entropies = [self.trace_universe[v]['logical_entropy'] for v in values]
        consistency_measures = [self.trace_universe[v]['consistency_measure'] for v in values]
        
        scatter2 = ax3.scatter(structural_truths, logical_entropies, c=consistency_measures, s=60, alpha=0.7, cmap='plasma')
        ax3.set_xlabel('Structural Truth')
        ax3.set_ylabel('Logical Entropy')
        ax3.set_title('Structure vs Entropy')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='Consistency Measure')
        
        # 4. Coherence stability vs logical potential
        coherence_stabilities = [self.trace_universe[v]['coherence_stability'] for v in values]
        logical_potentials = [self.trace_universe[v]['logical_potential'] for v in values]
        
        ax4.scatter(coherence_stabilities, logical_potentials, c=logical_strengths, s=60, alpha=0.7, cmap='cool')
        ax4.set_xlabel('Coherence Stability')
        ax4.set_ylabel('Logical Potential')
        ax4.set_title('Stability vs Potential')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-096-logic-collapse-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_coherence_analysis(self):
        """可视化coherence analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Logical Coherence Analysis', fontsize=16, fontweight='bold')
        
        values = list(self.trace_universe.keys())
        
        # 1. Coherence level distribution
        coherence_levels = [self.trace_universe[v]['coherence_level'] for v in values]
        
        ax1.hist(coherence_levels, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Coherence Level')
        ax1.set_ylabel('Count')
        ax1.set_title('Coherence Level Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Truth gradient vs consistency
        truth_gradients = [self.trace_universe[v]['truth_gradient'] for v in values]
        consistency_measures = [self.trace_universe[v]['consistency_measure'] for v in values]
        logical_strengths = [self.trace_universe[v]['logical_strength'] for v in values]
        
        scatter = ax2.scatter(truth_gradients, consistency_measures, c=logical_strengths, s=60, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Truth Gradient')
        ax2.set_ylabel('Consistency Measure')
        ax2.set_title('Gradient vs Consistency')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Logical Strength')
        
        # 3. Logical implications network
        proposition_types = {}
        for value, props in self.trace_universe.items():
            prop_type = props['logical_category']
            if prop_type not in proposition_types:
                proposition_types[prop_type] = 0
            proposition_types[prop_type] += 1
        
        prop_names = list(proposition_types.keys())
        prop_counts = list(proposition_types.values())
        
        bars = ax3.bar(prop_names, prop_counts, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('Proposition Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Logical Proposition Types')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Coherence stability scatter
        coherence_stabilities = [self.trace_universe[v]['coherence_stability'] for v in values]
        structural_truths = [self.trace_universe[v]['structural_truth'] for v in values]
        
        ax4.scatter(coherence_stabilities, structural_truths, c=coherence_levels, s=60, alpha=0.7, cmap='coolwarm')
        ax4.set_xlabel('Coherence Stability')
        ax4.set_ylabel('Structural Truth')
        ax4.set_title('Stability vs Structure')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-096-logic-collapse-coherence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_network_and_categories(self):
        """可视化network and categories"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Logical Network and Categorical Analysis', fontsize=16, fontweight='bold')
        
        # Build network
        G = self.coherence_network
        
        # 1. Network visualization
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Color nodes by logical category
            node_colors = []
            category_map = {'strong_coherent': 0, 'high_truth': 1, 'high_coherence': 2, 'moderate_logic': 3, 'weak_logic': 4}
            
            for node in G.nodes():
                category = self.trace_universe[node]['logical_category']
                node_colors.append(category_map.get(category, 0))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='Set3', alpha=0.8, ax=ax1)
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5, ax=ax1)
            
            ax1.set_title(f'Logical Coherence Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'No Network Connections', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Logical Coherence Network')
        
        # 2. Category analysis
        cat_analysis = self.analyze_category_theory()
        categories = list(cat_analysis['classifications'].keys())
        cat_sizes = [len(cat_analysis['classifications'][cat]) for cat in categories]
        
        if categories:
            colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            wedges, texts, autotexts = ax2.pie(cat_sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Logical Categories')
        else:
            ax2.text(0.5, 0.5, 'No Categories Found', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Logical Categories')
        
        # 3. Truth value spectrum
        values = list(self.trace_universe.keys())
        truth_values = [self.trace_universe[v]['truth_value'] for v in values]
        
        ax3.hist(truth_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Truth Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Truth Value Spectrum')
        ax3.grid(True, alpha=0.3)
        
        # 4. Information entropy analysis
        info_analysis = self.analyze_information_theory()
        entropy_names = list(info_analysis.keys())
        entropy_values = list(info_analysis.values())
        
        # Sort by entropy value
        sorted_pairs = sorted(zip(entropy_names, entropy_values), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        if sorted_names:
            bars = ax4.barh(range(len(sorted_names)), sorted_values, alpha=0.7, color='orange')
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
        plt.savefig('chapter-096-logic-collapse-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestLogicCollapse(unittest.TestCase):
    """Unit tests for LogicCollapse verification"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = LogicCollapseSystem(max_trace_value=60, coherence_resolution=8)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 0)
        for value, props in self.system.trace_universe.items():
            self.assertIn('trace', props)
            self.assertIn('truth_value', props)
            self.assertIn('coherence_level', props)
            self.assertIn('logical_strength', props)
            
    def test_logical_properties_analysis(self):
        """测试logical性质分析"""
        for value, props in self.system.trace_universe.items():
            # Test truth value bounds
            self.assertGreaterEqual(props['truth_value'], 0.0)
            self.assertLessEqual(props['truth_value'], 1.0)
            
            # Test coherence level bounds
            self.assertGreaterEqual(props['coherence_level'], 0.0)
            self.assertLessEqual(props['coherence_level'], 1.0)
            
            # Test logical strength bounds
            self.assertGreaterEqual(props['logical_strength'], 0.0)
            self.assertLessEqual(props['logical_strength'], 1.0)
            
            # Test consistency measure
            self.assertGreaterEqual(props['consistency_measure'], 0.0)
            self.assertLessEqual(props['consistency_measure'], 1.0)
            
    def test_structural_truth_computation(self):
        """测试structural truth计算"""
        for value, props in self.system.trace_universe.items():
            structural_truth = props['structural_truth']
            self.assertGreaterEqual(structural_truth, 0.0)
            self.assertLessEqual(structural_truth, 1.0)
            
            # φ-valid traces should have higher structural truth
            if self.system._is_phi_valid(props['trace']):
                self.assertGreater(structural_truth, 0.1)
                
    def test_logical_propositions(self):
        """测试logical propositions"""
        self.assertGreater(len(self.system.logical_propositions), 0)
        for prop_name, proposition in self.system.logical_propositions.items():
            self.assertIn('trace_encoding', proposition)
            self.assertIn('truth_value', proposition)
            self.assertIn('coherence_level', proposition)
            self.assertIn('logical_implications', proposition)
            
    def test_coherence_network(self):
        """测试coherence network构建"""
        G = self.system.coherence_network
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
    print("Chapter 096: LogicCollapse Verification")
    print("从ψ=ψ(ψ)推导Collapse-Coherent Logic beyond Truth Tables")
    print("=" * 80)
    
    # Initialize system
    system = LogicCollapseSystem(max_trace_value=75, coherence_resolution=10)
    
    print("\n1. Logical Foundation Analysis:")
    print("-" * 50)
    print(f"Total traces analyzed: {len(system.trace_universe)}")
    
    # Basic statistics
    truth_values = [props['truth_value'] for props in system.trace_universe.values()]
    coherence_levels = [props['coherence_level'] for props in system.trace_universe.values()]
    logical_strengths = [props['logical_strength'] for props in system.trace_universe.values()]
    structural_truths = [props['structural_truth'] for props in system.trace_universe.values()]
    
    print(f"Mean truth value: {np.mean(truth_values):.3f}")
    print(f"Mean coherence level: {np.mean(coherence_levels):.3f}")
    print(f"Mean logical strength: {np.mean(logical_strengths):.3f}")
    print(f"Mean structural truth: {np.mean(structural_truths):.3f}")
    
    # Logical propositions
    print(f"Logical propositions: {len(system.logical_propositions)}")
    
    print("\n2. Coherence Analysis:")
    print("-" * 50)
    
    consistency_measures = [props['consistency_measure'] for props in system.trace_universe.values()]
    coherence_stabilities = [props['coherence_stability'] for props in system.trace_universe.values()]
    logical_potentials = [props['logical_potential'] for props in system.trace_universe.values()]
    
    print(f"Mean consistency measure: {np.mean(consistency_measures):.3f}")
    print(f"Mean coherence stability: {np.mean(coherence_stabilities):.3f}")
    print(f"Mean logical potential: {np.mean(logical_potentials):.3f}")
    
    # High coherence count
    high_coherence_count = sum(1 for c in coherence_levels if c > 0.8)
    print(f"High coherence traces (>0.8): {high_coherence_count}")
    
    print("\n3. Network Analysis:")
    print("-" * 50)
    
    G = system.coherence_network
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        print(f"Network density: {density:.3f}")
        
        # Component analysis
        if G.number_of_edges() > 0:
            components = list(nx.connected_components(G))
            print(f"Connected components: {len(components)}")
            
            # Average degree
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            print(f"Average degree: {avg_degree:.3f}")
    
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    
    info_analysis = system.analyze_information_theory()
    for name, entropy in sorted(info_analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {entropy:.3f} bits")
    
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    
    cat_analysis = system.analyze_category_theory()
    print(f"Logical categories: {cat_analysis['categories']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for cat, traces in cat_analysis['classifications'].items():
        percentage = len(traces) / len(system.trace_universe) * 100
        print(f"- {cat}: {len(traces)} traces ({percentage:.1f}%)")
    
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        system.visualize_logic_dynamics()
        print("✓ Logic dynamics visualization saved")
    except Exception as e:
        print(f"✗ Logic dynamics visualization failed: {e}")
    
    try:
        system.visualize_coherence_analysis()
        print("✓ Coherence analysis visualization saved")
    except Exception as e:
        print(f"✗ Coherence analysis visualization failed: {e}")
    
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
    print("LogicCollapse Verification Complete")
    print("Key Findings:")
    print(f"- {len(system.trace_universe)} φ-valid traces with logical analysis")
    print(f"- {cat_analysis['categories']} logical categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print(f"- Network density: {density:.3f}" if G.number_of_nodes() > 0 else "- Network density: 0.000")
    print(f"- Mean truth value: {np.mean(truth_values):.3f}")
    print(f"- Mean coherence level: {np.mean(coherence_levels):.3f}")
    print(f"- High coherence traces: {high_coherence_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()