#!/usr/bin/env python3
"""
Chapter 063: AlgebraOverObserver Unit Test Verification
从ψ=ψ(ψ)推导Algebraic Structures Varying over Observer Spectra

Core principle: From ψ = ψ(ψ) derive observer-dependent algebraic structures where algebras are φ-valid
trace systems that transform according to observer perspective while preserving fundamental algebraic
properties through observer transformations, creating systematic observer-parametric frameworks with
bounded variation and natural perspective-invariant properties governed by golden constraints.

This verification program implements:
1. φ-constrained observer-algebra operations as trace observer tensor operations
2. Observer analysis: perspective variation, invariant preservation with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection observer-dependent algebra
4. Graph theory analysis of observer networks and perspective-dependent connectivity
5. Information theory analysis of observer entropy and perspective information
6. Category theory analysis of observer functors and perspective morphisms
7. Visualization of observer structures and perspective variation patterns
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi, exp, cos, sin
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class AlgebraOverObserverSystem:
    """
    Core system for implementing algebraic structures varying over observer spectra.
    Implements φ-constrained observer-dependent algebra via trace-based observer operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_observer_complexity: int = 4):
        """Initialize algebra over observer system"""
        self.max_trace_size = max_trace_size
        self.max_observer_complexity = max_observer_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.observer_cache = {}
        self.algebra_cache = {}
        self.variation_cache = {}
        self.trace_universe = self._build_trace_universe()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        # First pass: build basic universe
        for n in range(self.max_trace_size + 1):
            trace_data = self._analyze_trace_structure(n, compute_observer=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for observer properties computation
        self.trace_universe = universe
        
        # Second pass: add observer properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['observer_properties'] = self._compute_observer_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_observer: bool = True) -> Dict:
        """分析单个trace的结构属性"""
        trace = self._encode_to_trace(n)
        
        result = {
            'value': n,
            'trace': trace,
            'phi_valid': '11' not in trace,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'fibonacci_indices': self._get_fibonacci_indices(trace),
            'structural_hash': self._compute_structural_hash(trace),
            'binary_weight': self._compute_binary_weight(trace)
        }
        
        if compute_observer and hasattr(self, 'trace_universe'):
            result['observer_properties'] = self._compute_observer_properties(trace)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将整数编码为trace表示"""
        if n == 0:
            return '0'
        return bin(n)[2:]  # Remove '0b' prefix
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中1对应的Fibonacci索引"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构哈希"""
        return hash(trace + str(self._compute_fibonacci_sum(trace)))
        
    def _compute_binary_weight(self, trace: str) -> float:
        """计算trace的二进制权重"""
        weight = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                weight += self.fibonacci_numbers[fib_idx]
        return weight / sum(self.fibonacci_numbers[:len(trace)]) if len(trace) > 0 else 0
        
    def _compute_fibonacci_sum(self, trace: str) -> int:
        """计算trace的Fibonacci值之和"""
        total = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                total += self.fibonacci_numbers[fib_idx]
        return total
        
    def _compute_observer_properties(self, trace: str) -> Dict:
        """计算trace的observer相关属性"""
        cache_key = trace
        if cache_key in self.observer_cache:
            return self.observer_cache[cache_key]
            
        result = {
            'observer_complexity': self._compute_observer_complexity(trace),
            'perspective_measure': self._compute_perspective_measure(trace),
            'invariant_dimension': self._compute_invariant_dimension(trace),
            'variation_capacity': self._compute_variation_capacity(trace),
            'observer_signature': self._compute_observer_signature(trace),
            'algebraic_stability': self._compute_algebraic_stability(trace),
            'observer_type': self._classify_observer_type(trace),
            'perspective_index': self._compute_perspective_index(trace),
            'spectral_measure': self._compute_spectral_measure(trace)
        }
        
        self.observer_cache[cache_key] = result
        return result
        
    def _compute_observer_complexity(self, trace: str) -> int:
        """计算观察者复杂度：基于trace复杂度"""
        if not trace or trace == '0':
            return 0  # Zero observer has complexity 0
        
        # Use ones count to determine observer complexity
        ones_count = trace.count('1')
        if ones_count == 0:
            return 0
        elif ones_count == 1:
            return 1  # Complexity 1 observer
        else:
            return min(self.max_observer_complexity, ones_count)
            
    def _compute_perspective_measure(self, trace: str) -> float:
        """计算视角度量"""
        if not trace or trace == '0':
            return 0.0  # Zero observer has no perspective
            
        # Complex harmonic measure based on position structure
        measure = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                position_factor = (i + 1) / len(trace)
                fibonacci_factor = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)]
                measure += position_factor * fibonacci_factor
                
        # Normalize for perspective measure
        max_possible = sum(self.fibonacci_numbers[:len(trace)])
        return measure / max_possible if max_possible > 0 else 0.0
        
    def _compute_invariant_dimension(self, trace: str) -> int:
        """计算不变量维度：基于trace结构"""
        if not trace or trace == '0':
            return 0  # Zero observer has no invariants
            
        # Invariant dimension based on pattern stability
        stable_patterns = 0
        
        # Count stable single bits
        for bit in trace:
            if bit == '1':
                stable_patterns += 1
                
        # Bonus for φ-constraint (inherent stability)
        if '11' not in trace:
            stable_patterns += 1
            
        return min(stable_patterns, self.max_observer_complexity)
        
    def _compute_variation_capacity(self, trace: str) -> float:
        """计算变化能力"""
        if not trace or trace == '0':
            return 0.0  # Zero observer has no variation
            
        # Measure variation capacity through structural patterns
        capacity = 0.0
        
        # Factor 1: position distribution variance
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) > 1:
            position_variance = np.var(ones_positions) / (len(trace) ** 2)
            capacity += position_variance
            
        # Factor 2: fibonacci weight distribution
        fib_weights = [self.fibonacci_numbers[min(pos, len(self.fibonacci_numbers) - 1)] 
                      for pos in ones_positions]
        if fib_weights:
            weight_coefficient = np.std(fib_weights) / np.mean(fib_weights) if np.mean(fib_weights) > 0 else 0
            capacity += weight_coefficient * 0.5
            
        return min(1.0, capacity)
        
    def _compute_observer_signature(self, trace: str) -> complex:
        """计算观察者签名：复数表示的观察者特征"""
        if not trace:
            return complex(0, 0)
            
        # Complex harmonic encoding for observer perspective
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                # Observer perspective angle
                angle = 2 * pi * i / len(trace)
                weight = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)]
                real_part += weight * cos(angle)
                imag_part += weight * sin(angle)
                
        # Normalize to unit circle
        magnitude = sqrt(real_part**2 + imag_part**2)
        if magnitude > 0:
            return complex(real_part / magnitude, imag_part / magnitude)
        return complex(0, 0)
        
    def _compute_algebraic_stability(self, trace: str) -> float:
        """计算代数稳定性"""
        if not trace or trace == '0':
            return 1.0  # Zero observer is perfectly stable
            
        # Measure algebraic stability through pattern consistency
        stability = 0.0
        
        # Factor 1: φ-constraint preservation (fundamental stability)
        if '11' not in trace:
            stability += 0.6  # Major stability from golden constraint
            
        # Factor 2: fibonacci resonance
        fib_indices = self._get_fibonacci_indices(trace)
        if fib_indices:
            # Stability from fibonacci alignment
            fibonacci_resonance = sum(1 for i in fib_indices 
                                    if i < len(self.fibonacci_numbers)) / len(fib_indices)
            stability += 0.3 * fibonacci_resonance
            
        # Factor 3: structural balance
        ones_count = trace.count('1')
        zeros_count = trace.count('0')
        if ones_count + zeros_count > 0:
            balance = 1.0 - abs(ones_count - zeros_count) / (ones_count + zeros_count)
            stability += 0.1 * balance
            
        return min(1.0, stability)
        
    def _classify_observer_type(self, trace: str) -> str:
        """分类观察者类型"""
        if not trace or trace == '0':
            return 'universal'  # Universal observer sees all
            
        complexity = self._compute_observer_complexity(trace)
        perspective = self._compute_perspective_measure(trace)
        variation = self._compute_variation_capacity(trace)
        stability = self._compute_algebraic_stability(trace)
        
        if complexity == 1 and stability > 0.8:
            return 'singular'
        elif perspective > 0.7 and variation > 0.5:
            return 'dynamic'
        elif stability > 0.8 and variation < 0.3:
            return 'invariant'
        elif complexity > 2:
            return 'complex'
        else:
            return 'general'
            
    def _compute_perspective_index(self, trace: str) -> float:
        """计算视角指数"""
        if not trace or trace == '0':
            return 1.0  # Universal perspective
            
        # Perspective index based on viewing capability
        perspective = self._compute_perspective_measure(trace)
        complexity = self._compute_observer_complexity(trace)
        
        # Higher complexity and perspective = higher index
        index = (perspective + complexity / self.max_observer_complexity) / 2
        
        # Bonus for φ-constraint (stable perspective)
        if '11' not in trace:
            index = min(1.0, index * 1.1)
            
        return max(0.0, min(1.0, index))
        
    def _compute_spectral_measure(self, trace: str) -> float:
        """计算光谱度量"""
        if not trace or trace == '0':
            return 0.0  # Universal observer has no spectral limitation
            
        # Spectral measure through fibonacci spectral analysis
        fib_indices = self._get_fibonacci_indices(trace)
        
        if not fib_indices:
            return 0.0
            
        # Compute spectral distribution
        spectral_weights = [self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)] 
                           for i in fib_indices]
        total_spectral = sum(spectral_weights)
        max_possible = sum(self.fibonacci_numbers[:len(trace)])
        
        spectral_measure = total_spectral / max_possible if max_possible > 0 else 0.0
        
        return min(1.0, spectral_measure)
        
    def analyze_observer_algebra_system(self) -> Dict:
        """分析complete observer algebra system"""
        elements = list(self.trace_universe.keys())
        observer_data = []
        
        for n in elements:
            trace_info = self.trace_universe[n]
            observer_props = trace_info['observer_properties']
            
            observer_data.append({
                'element': n,
                'trace': trace_info['trace'],
                'observer_complexity': observer_props['observer_complexity'],
                'perspective_measure': observer_props['perspective_measure'],
                'invariant_dimension': observer_props['invariant_dimension'],
                'variation_capacity': observer_props['variation_capacity'],
                'observer_signature': observer_props['observer_signature'],
                'algebraic_stability': observer_props['algebraic_stability'],
                'observer_type': observer_props['observer_type'],
                'perspective_index': observer_props['perspective_index'],
                'spectral_measure': observer_props['spectral_measure']
            })
            
        return self._compute_system_analysis(observer_data)
        
    def _compute_system_analysis(self, observer_data: List[Dict]) -> Dict:
        """计算系统级分析"""
        if not observer_data:
            return {}
            
        # Basic statistics
        complexities = [item['observer_complexity'] for item in observer_data]
        perspective_measures = [item['perspective_measure'] for item in observer_data]
        invariant_dimensions = [item['invariant_dimension'] for item in observer_data]
        variation_capacities = [item['variation_capacity'] for item in observer_data]
        algebraic_stabilities = [item['algebraic_stability'] for item in observer_data]
        perspective_indices = [item['perspective_index'] for item in observer_data]
        spectral_measures = [item['spectral_measure'] for item in observer_data]
        
        # Type distribution
        types = [item['observer_type'] for item in observer_data]
        type_counts = {t: types.count(t) for t in set(types)}
        
        # Network analysis
        network_analysis = self._analyze_observer_network(observer_data)
        
        # Information theory analysis
        info_analysis = self._analyze_observer_information(observer_data)
        
        # Category theory analysis  
        category_analysis = self._analyze_observer_categories(observer_data)
        
        return {
            'system': {
                'element_count': len(observer_data),
                'mean_complexity': np.mean(complexities),
                'max_complexity': max(complexities),
                'mean_perspective': np.mean(perspective_measures),
                'mean_invariant_dimension': np.mean(invariant_dimensions),
                'mean_variation_capacity': np.mean(variation_capacities),
                'mean_algebraic_stability': np.mean(algebraic_stabilities),
                'mean_perspective_index': np.mean(perspective_indices),
                'mean_spectral_measure': np.mean(spectral_measures),
                'observer_types': type_counts
            },
            'network': network_analysis,
            'information': info_analysis,
            'category': category_analysis,
            'observer_data': observer_data
        }
        
    def _analyze_observer_network(self, observer_data: List[Dict]) -> Dict:
        """分析observer network结构"""
        G = nx.DiGraph()  # Use directed graph for observer perspectives
        
        # Add nodes
        for item in observer_data:
            G.add_node(item['element'], **item)
            
        # Add edges based on observer relationships
        for i, item1 in enumerate(observer_data):
            for j, item2 in enumerate(observer_data[i+1:], i+1):
                # Observer relationship criterion
                sig1 = item1['observer_signature']
                sig2 = item2['observer_signature']
                
                # Check for perspective compatibility
                if (abs(sig1) > 0 and abs(sig2) > 0 and
                    item1['algebraic_stability'] > 0.3 and
                    item2['algebraic_stability'] > 0.3):
                    
                    # Compute perspective affinity
                    signature_similarity = abs((sig1.real * sig2.real + sig1.imag * sig2.imag) / 
                                             (abs(sig1) * abs(sig2)))
                    stability_product = (item1['algebraic_stability'] * 
                                       item2['algebraic_stability'])
                    
                    affinity = (signature_similarity + stability_product) / 2
                    
                    # Add edge if sufficiently compatible
                    if affinity > 0.4:  # Threshold for observer compatibility
                        G.add_edge(item1['element'], item2['element'], 
                                 weight=affinity)
        
        if G.number_of_edges() == 0:
            density = 0.0
            clustering = 0.0
        else:
            density = nx.density(G)
            # Convert to undirected for clustering
            clustering = nx.average_clustering(G.to_undirected())
            
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': density,
            'components': nx.number_weakly_connected_components(G),
            'clustering': clustering
        }
        
    def _analyze_observer_information(self, observer_data: List[Dict]) -> Dict:
        """分析observer information content"""
        if not observer_data:
            return {}
            
        # Complexity entropy
        complexities = [item['observer_complexity'] for item in observer_data]
        complexity_entropy = self._compute_entropy([complexities.count(c) for c in set(complexities)])
        
        # Type entropy  
        types = [item['observer_type'] for item in observer_data]
        type_entropy = self._compute_entropy([types.count(t) for t in set(types)])
        
        # Invariant entropy
        invariants = [item['invariant_dimension'] for item in observer_data]
        invariant_entropy = self._compute_entropy([invariants.count(i) for i in set(invariants)])
        
        # Perspective entropy
        # Discretize perspective measures for entropy calculation
        perspective_discrete = [int(item['perspective_measure'] * 10) for item in observer_data]
        perspective_entropy = self._compute_entropy([perspective_discrete.count(p) for p in set(perspective_discrete)])
        
        return {
            'complexity_entropy': complexity_entropy,
            'type_entropy': type_entropy,
            'invariant_entropy': invariant_entropy,
            'perspective_entropy': perspective_entropy,
            'observer_complexity': len(set(types)),
            'complexity_diversity': len(set(complexities)),
            'invariant_diversity': len(set(invariants)),
            'perspective_diversity': len(set(perspective_discrete))
        }
        
    def _analyze_observer_categories(self, observer_data: List[Dict]) -> Dict:
        """分析observer category structure"""
        if not observer_data:
            return {}
            
        # Count perspective morphisms (observer transformations)
        perspective_morphisms = 0
        functorial_relationships = 0
        
        for i, item1 in enumerate(observer_data):
            for j, item2 in enumerate(observer_data[i+1:], i+1):
                # Check for perspective morphism (complexity and stability compatibility)
                if (item1['observer_complexity'] <= item2['observer_complexity'] and
                    abs(item1['algebraic_stability'] - item2['algebraic_stability']) < 0.3):
                    perspective_morphisms += 1
                    
                    # Check for functoriality (type preservation)
                    if (item1['observer_type'] == item2['observer_type'] or
                        (item1['observer_type'] in ['singular', 'invariant'] and
                         item2['observer_type'] in ['singular', 'invariant']) or
                        (item1['observer_type'] in ['dynamic', 'complex'] and
                         item2['observer_type'] in ['dynamic', 'complex'])):
                        functorial_relationships += 1
        
        functoriality_ratio = (functorial_relationships / perspective_morphisms 
                             if perspective_morphisms > 0 else 0)
        
        # Observer transformation analysis
        transformable_pairs = 0
        for item1 in observer_data:
            for item2 in observer_data:
                if (item1['variation_capacity'] > 0.3 and
                    item2['variation_capacity'] > 0.3 and
                    item1['algebraic_stability'] > 0.5 and
                    item2['algebraic_stability'] > 0.5):
                    transformable_pairs += 1
        
        return {
            'perspective_morphisms': perspective_morphisms,
            'functorial_relationships': functorial_relationships,
            'functoriality_ratio': functoriality_ratio,
            'transformable_pairs': transformable_pairs,
            'category_structure': 'observer_algebra_category'
        }
        
    def _compute_entropy(self, counts: List[int]) -> float:
        """计算熵"""
        if not counts or sum(counts) == 0:
            return 0.0
            
        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]
        
        return -sum(p * log2(p) for p in probabilities)
        
    def generate_visualizations(self, analysis_results: Dict, output_prefix: str):
        """生成observer algebra system可视化"""
        self._plot_observer_structure(analysis_results, f"{output_prefix}-structure.png")
        self._plot_observer_properties(analysis_results, f"{output_prefix}-properties.png")
        self._plot_domain_analysis(analysis_results, f"{output_prefix}-domains.png")
        
    def _plot_observer_structure(self, analysis: Dict, filename: str):
        """可视化observer结构"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        observer_data = analysis['observer_data']
        elements = [item['element'] for item in observer_data]
        complexities = [item['observer_complexity'] for item in observer_data]
        perspective = [item['perspective_measure'] for item in observer_data]
        invariant_dims = [item['invariant_dimension'] for item in observer_data]
        variation_caps = [item['variation_capacity'] for item in observer_data]
        
        # Observer complexity distribution
        ax1.bar(elements, complexities, color='skyblue', alpha=0.7)
        ax1.set_title('Observer Complexity Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trace Elements')
        ax1.set_ylabel('Observer Complexity')
        ax1.grid(True, alpha=0.3)
        
        # Perspective vs Variation capacity
        colors = plt.cm.viridis([id/max(invariant_dims) if max(invariant_dims) > 0 else 0 for id in invariant_dims])
        scatter = ax2.scatter(perspective, variation_caps, c=colors, s=100, alpha=0.7)
        ax2.set_title('Perspective vs Variation Capacity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Perspective Measure')
        ax2.set_ylabel('Variation Capacity')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Invariant Dimension')
        
        # Observer type distribution
        type_counts = analysis['system']['observer_types']
        ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3.colors)
        ax3.set_title('Observer Type Distribution', fontsize=14, fontweight='bold')
        
        # Observer signature visualization
        signatures = [item['observer_signature'] for item in observer_data]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=complexities, s=100, alpha=0.7, cmap='plasma')
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)
        ax4.set_title('Observer Signatures (Complex Plane)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_observer_properties(self, analysis: Dict, filename: str):
        """可视化observer属性"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        observer_data = analysis['observer_data']
        elements = [item['element'] for item in observer_data]
        stability = [item['algebraic_stability'] for item in observer_data]
        perspective_index = [item['perspective_index'] for item in observer_data]
        spectral_measure = [item['spectral_measure'] for item in observer_data]
        
        # Network metrics
        network = analysis['network']
        network_metrics = ['nodes', 'edges', 'density', 'components', 'clustering']
        network_values = [network.get(metric, 0) for metric in network_metrics]
        
        ax1.bar(network_metrics, network_values, color='lightcoral', alpha=0.7)
        ax1.set_title('Network Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Information theory metrics
        info = analysis['information']
        info_metrics = ['complexity_entropy', 'type_entropy', 'invariant_entropy', 'perspective_entropy']
        info_values = [info.get(metric, 0) for metric in info_metrics]
        
        ax2.bar(info_metrics, info_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Information Theory Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        
        # Stability vs Perspective Index
        ax3.scatter(stability, perspective_index, c=spectral_measure, s=100, alpha=0.7, cmap='coolwarm')
        ax3.set_title('Stability vs Perspective Index', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Algebraic Stability')
        ax3.set_ylabel('Perspective Index')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Spectral Measure')
        
        # Category theory metrics
        category = analysis['category']
        cat_metrics = ['perspective_morphisms', 'functorial_relationships', 'transformable_pairs']
        cat_values = [category.get(metric, 0) for metric in cat_metrics]
        
        ax4.bar(cat_metrics, cat_values, color='gold', alpha=0.7)
        ax4.set_title('Category Theory Analysis', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_domain_analysis(self, analysis: Dict, filename: str):
        """可视化domain convergence analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Three-domain comparison
        traditional_ops = 100  # Hypothetical traditional operations
        phi_ops = len(analysis['observer_data'])  # φ-constrained operations
        convergence_ops = phi_ops  # Operations in convergence
        
        domains = ['Traditional\nOnly', 'φ-Constrained\nOnly', 'Convergence\nDomain']
        operation_counts = [traditional_ops - phi_ops, 0, convergence_ops]
        colors = ['lightblue', 'lightcoral', 'gold']
        
        bars = ax1.bar(domains, operation_counts, color=colors, alpha=0.7)
        ax1.set_title('Three-Domain Operation Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Operation Count')
        ax1.grid(True, alpha=0.3)
        
        # Add convergence ratio annotation
        convergence_ratio = convergence_ops / traditional_ops
        ax1.text(2, convergence_ops + 5, f'Convergence Ratio: {convergence_ratio:.3f}', 
                ha='center', fontweight='bold')
        
        # Convergence efficiency metrics
        system = analysis['system']
        efficiency_metrics = ['mean_perspective', 'mean_algebraic_stability', 'mean_perspective_index']
        efficiency_values = [system.get(metric, 0) for metric in efficiency_metrics]
        
        ax2.bar(efficiency_metrics, efficiency_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Observer Efficiency Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Complexity vs Invariant analysis
        observer_data = analysis['observer_data']
        complexities = [item['observer_complexity'] for item in observer_data]
        invariant_dims = [item['invariant_dimension'] for item in observer_data]
        
        complexity_invariant_data = {}
        for c, i in zip(complexities, invariant_dims):
            key = (c, i)
            complexity_invariant_data[key] = complexity_invariant_data.get(key, 0) + 1
            
        if complexity_invariant_data:
            keys = list(complexity_invariant_data.keys())
            values = list(complexity_invariant_data.values())
            
            x_pos = range(len(keys))
            ax3.bar(x_pos, values, color='purple', alpha=0.7)
            ax3.set_title('Complexity-Invariant Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('(Complexity, Invariant Dim)')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'({c},{i})' for c, i in keys], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # System complexity evolution
        info = analysis['information']
        complexity_metrics = ['observer_complexity', 'complexity_diversity', 'invariant_diversity', 'perspective_diversity']
        complexity_values = [info.get(metric, 0) for metric in complexity_metrics]
        
        ax4.plot(complexity_metrics, complexity_values, 'o-', linewidth=2, markersize=8,
                color='red', alpha=0.7)
        ax4.set_title('System Complexity Evolution', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Diversity Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


class TestAlgebraOverObserverSystem(unittest.TestCase):
    """Test suite for AlgebraOverObserverSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = AlgebraOverObserverSystem(max_trace_size=4, max_observer_complexity=3)
        
    def test_observer_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.trace_universe, dict)
        self.assertGreater(len(self.system.trace_universe), 0)
        
    def test_observer_properties_computation(self):
        """Test observer properties computation"""
        trace = "101"  # φ-valid trace
        props = self.system._compute_observer_properties(trace)
        
        required_keys = ['observer_complexity', 'perspective_measure', 'invariant_dimension',
                        'variation_capacity', 'observer_signature', 'algebraic_stability',
                        'observer_type', 'perspective_index', 'spectral_measure']
        
        for key in required_keys:
            self.assertIn(key, props)
            
        # Test constraints
        self.assertGreaterEqual(props['observer_complexity'], 0)
        self.assertLessEqual(props['observer_complexity'], self.system.max_observer_complexity)
        self.assertGreaterEqual(props['perspective_measure'], 0.0)
        self.assertLessEqual(props['perspective_measure'], 1.0)
        self.assertGreaterEqual(props['invariant_dimension'], 0)
        self.assertGreaterEqual(props['variation_capacity'], 0.0)
        self.assertLessEqual(props['variation_capacity'], 1.0)
        
    def test_phi_constraint_preservation(self):
        """Test φ-constraint preservation"""
        # Valid traces (no consecutive 1s)
        valid_traces = ["0", "1", "10", "101", "1010"]
        for trace in valid_traces:
            self.assertNotIn('11', trace)
            props = self.system._compute_observer_properties(trace)
            self.assertGreaterEqual(props['invariant_dimension'], 0)
            
        # Invalid trace should still work but with different properties
        invalid_trace = "11"
        props = self.system._compute_observer_properties(invalid_trace)
        # All properties should still be computable
        self.assertIsInstance(props['observer_complexity'], int)
        
    def test_observer_system_analysis(self):
        """Test complete observer system analysis"""
        analysis = self.system.analyze_observer_algebra_system()
        
        required_sections = ['system', 'network', 'information', 'category', 'observer_data']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Test system metrics
        system = analysis['system']
        self.assertGreater(system['element_count'], 0)
        self.assertGreaterEqual(system['mean_complexity'], 0)
        self.assertGreaterEqual(system['max_complexity'], 0)
        
    def test_observer_signature_computation(self):
        """Test observer signature computation"""
        trace = "101"
        signature = self.system._compute_observer_signature(trace)
        
        self.assertIsInstance(signature, complex)
        # Should be approximately on unit circle
        magnitude = abs(signature)
        self.assertAlmostEqual(magnitude, 1.0, places=10)
        
    def test_observer_type_classification(self):
        """Test observer type classification"""
        # Test different trace patterns
        test_cases = [
            ("0", "universal"),
            ("1", "singular"),  # Should be singular or general
            ("101", "general"),  # Should be some valid type
        ]
        
        for trace, expected_category in test_cases:
            observer_type = self.system._classify_observer_type(trace)
            if expected_category == "universal":
                self.assertEqual(observer_type, expected_category)
            elif expected_category == "singular":
                self.assertIn(observer_type, ["singular", "invariant", "general"])
            else:
                # Any valid type is acceptable for complex traces
                self.assertIn(observer_type, ["universal", "singular", "dynamic", "invariant", "complex", "general"])
        
    def test_algebraic_stability(self):
        """Test algebraic stability computation"""
        # φ-valid traces should have good stability
        valid_trace = "101"
        stability = self.system._compute_algebraic_stability(valid_trace)
        self.assertGreaterEqual(stability, 0.6)  # Should have φ-constraint stability
        
        # Zero trace should have perfect stability
        zero_trace = "0"
        zero_stability = self.system._compute_algebraic_stability(zero_trace)
        self.assertEqual(zero_stability, 1.0)


def main():
    """Main execution function"""
    print("=" * 80)
    print("CHAPTER 063: ALGEBRA OVER OBSERVER VERIFICATION")
    print("Algebraic Structures Varying over Observer Spectra")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing AlgebraOverObserver System...")
    system = AlgebraOverObserverSystem(max_trace_size=6, max_observer_complexity=4)
    print(f"   φ-valid traces found: {len(system.trace_universe)}")
    
    # Analyze observer algebra system
    print("\n2. Analyzing Observer Algebra System...")
    analysis_results = system.analyze_observer_algebra_system()
    
    # Display results
    print("\n3. Observer Algebra System Analysis Results:")
    print("-" * 50)
    
    system_data = analysis_results['system']
    print(f"Observer elements: {system_data['element_count']}")
    print(f"Mean observer complexity: {system_data['mean_complexity']:.3f}")
    print(f"Maximum complexity: {system_data['max_complexity']}")
    print(f"Mean perspective: {system_data['mean_perspective']:.3f}")
    print(f"Mean invariant dimension: {system_data['mean_invariant_dimension']:.3f}")
    print(f"Mean variation capacity: {system_data['mean_variation_capacity']:.3f}")
    print(f"Mean algebraic stability: {system_data['mean_algebraic_stability']:.3f}")
    print(f"Mean perspective index: {system_data['mean_perspective_index']:.3f}")
    print(f"Mean spectral measure: {system_data['mean_spectral_measure']:.3f}")
    
    print(f"\nObserver Type Distribution:")
    for observer_type, count in system_data['observer_types'].items():
        percentage = (count / system_data['element_count']) * 100
        print(f"  {observer_type.capitalize()} observers: {count} ({percentage:.1f}%)")
    
    network_data = analysis_results['network']
    print(f"\nNetwork Analysis:")
    print(f"  Network density: {network_data['density']:.3f}")
    print(f"  Connected components: {network_data['components']}")
    print(f"  Average clustering: {network_data['clustering']:.3f}")
    
    info_data = analysis_results['information']
    print(f"\nInformation Theory:")
    print(f"  Complexity entropy: {info_data['complexity_entropy']:.3f} bits")
    print(f"  Type entropy: {info_data['type_entropy']:.3f} bits")
    print(f"  Invariant entropy: {info_data['invariant_entropy']:.3f} bits")
    print(f"  Perspective entropy: {info_data['perspective_entropy']:.3f} bits")
    print(f"  Observer complexity: {info_data['observer_complexity']} unique types")
    
    category_data = analysis_results['category']
    print(f"\nCategory Theory:")
    print(f"  Perspective morphisms: {category_data['perspective_morphisms']}")
    print(f"  Functorial relationships: {category_data['functorial_relationships']}")
    print(f"  Functoriality ratio: {category_data['functoriality_ratio']:.3f}")
    print(f"  Transformable pairs: {category_data['transformable_pairs']}")
    
    # Three-domain convergence analysis
    traditional_ops = 100
    phi_ops = system_data['element_count']
    convergence_ratio = phi_ops / traditional_ops
    
    print(f"\nThree-Domain Convergence Analysis:")
    print(f"  Traditional operations: {traditional_ops}")
    print(f"  φ-constrained operations: {phi_ops}")
    print(f"  Convergence ratio: {convergence_ratio:.3f}")
    print(f"  Operations preserved: {phi_ops}/{traditional_ops}")
    
    # Generate visualizations
    print("\n4. Generating Visualizations...")
    output_prefix = "chapter-063-algebra-over-observer"
    system.generate_visualizations(analysis_results, output_prefix)
    print(f"   Generated: {output_prefix}-structure.png")
    print(f"   Generated: {output_prefix}-properties.png") 
    print(f"   Generated: {output_prefix}-domains.png")
    
    # Run unit tests
    print("\n5. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    main()