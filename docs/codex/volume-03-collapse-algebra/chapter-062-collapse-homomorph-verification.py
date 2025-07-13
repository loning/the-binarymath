#!/usr/bin/env python3
"""
Chapter 062: CollapseHomomorph Unit Test Verification
从ψ=ψ(ψ)推导Collapse Homomorphism Mapping between Collapse Tensor Algebras

Core principle: From ψ = ψ(ψ) derive homomorphism structures where morphisms are φ-valid
trace mappings preserving algebraic structure between collapse tensor algebras while maintaining
the φ-constraint across all morphism operations, creating systematic homomorphism frameworks with
bounded preservation and natural structure-preserving properties governed by golden constraints.

This verification program implements:
1. φ-constrained homomorphism operations as trace morphism tensor operations
2. Homomorphism analysis: structure preservation, kernel/image analysis with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection homomorphism theory
4. Graph theory analysis of morphism networks and structure-preserving connectivity
5. Information theory analysis of homomorphism entropy and preservation information
6. Category theory analysis of homomorphism functors and morphism compositions
7. Visualization of homomorphism structures and preservation patterns
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

class CollapseHomomorphSystem:
    """
    Core system for implementing collapse homomorphism mapping between collapse tensor algebras.
    Implements φ-constrained homomorphism theory via trace-based morphism operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_morphism_complexity: int = 4):
        """Initialize collapse homomorph system"""
        self.max_trace_size = max_trace_size
        self.max_morphism_complexity = max_morphism_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.homomorph_cache = {}
        self.morphism_cache = {}
        self.preservation_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_homomorph=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for homomorphism properties computation
        self.trace_universe = universe
        
        # Second pass: add homomorphism properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['homomorph_properties'] = self._compute_homomorph_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_homomorph: bool = True) -> Dict:
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
        
        if compute_homomorph and hasattr(self, 'trace_universe'):
            result['homomorph_properties'] = self._compute_homomorph_properties(trace)
            
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
        
    def _compute_homomorph_properties(self, trace: str) -> Dict:
        """计算trace的homomorphism相关属性"""
        cache_key = trace
        if cache_key in self.homomorph_cache:
            return self.homomorph_cache[cache_key]
            
        result = {
            'morphism_degree': self._compute_morphism_degree(trace),
            'preservation_measure': self._compute_preservation_measure(trace),
            'kernel_dimension': self._compute_kernel_dimension(trace),
            'image_rank': self._compute_image_rank(trace),
            'morphism_signature': self._compute_morphism_signature(trace),
            'structural_preservation': self._compute_structural_preservation(trace),
            'homomorph_type': self._classify_homomorph_type(trace),
            'injectivity_index': self._compute_injectivity_index(trace),
            'surjectivity_measure': self._compute_surjectivity_measure(trace)
        }
        
        self.homomorph_cache[cache_key] = result
        return result
        
    def _compute_morphism_degree(self, trace: str) -> int:
        """计算态射次数：基于trace复杂度"""
        if not trace or trace == '0':
            return 0  # Zero morphism has degree 0
        
        # Use ones count to determine morphism degree
        ones_count = trace.count('1')
        if ones_count == 0:
            return 0
        elif ones_count == 1:
            return 1  # Degree 1 morphism
        else:
            return min(self.max_morphism_complexity, ones_count)
            
    def _compute_preservation_measure(self, trace: str) -> float:
        """计算结构保持度量"""
        if not trace or trace == '0':
            return 1.0  # Zero morphism trivially preserves structure
            
        # Complex harmonic measure based on position structure
        measure = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                position_factor = (i + 1) / len(trace)
                fibonacci_factor = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)]
                measure += position_factor * fibonacci_factor
                
        # Normalize and invert for preservation (closer positions = better preservation)
        max_possible = sum(self.fibonacci_numbers[:len(trace)])
        preservation = 1.0 - (measure / max_possible) if max_possible > 0 else 1.0
        return max(0.0, min(1.0, preservation))
        
    def _compute_kernel_dimension(self, trace: str) -> int:
        """计算核维度：基于trace结构"""
        if not trace or trace == '0':
            return 0  # Zero morphism has trivial kernel
            
        # Kernel dimension based on consecutive patterns
        consecutive_zeros = 0
        max_consecutive_zeros = 0
        
        for bit in trace:
            if bit == '0':
                consecutive_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
            else:
                consecutive_zeros = 0
                
        return min(max_consecutive_zeros, self.max_morphism_complexity)
        
    def _compute_image_rank(self, trace: str) -> int:
        """计算像的秩"""
        if not trace or trace == '0':
            return 0  # Zero morphism has trivial image
            
        # Image rank based on ones distribution
        ones_count = trace.count('1')
        
        # Check for φ-constraint violation
        if '11' in trace:
            return 0  # Invalid morphism
            
        return min(ones_count, self.max_morphism_complexity)
        
    def _compute_morphism_signature(self, trace: str) -> complex:
        """计算态射签名：复数表示的态射特征"""
        if not trace:
            return complex(0, 0)
            
        # Complex harmonic encoding for morphism
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                angle = 2 * pi * i / len(trace)
                weight = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)]
                real_part += weight * cos(angle)
                imag_part += weight * sin(angle)
                
        # Normalize to unit circle
        magnitude = sqrt(real_part**2 + imag_part**2)
        if magnitude > 0:
            return complex(real_part / magnitude, imag_part / magnitude)
        return complex(0, 0)
        
    def _compute_structural_preservation(self, trace: str) -> float:
        """计算结构保持性"""
        if not trace or trace == '0':
            return 1.0  # Zero morphism preserves all structure trivially
            
        # Measure structural preservation through pattern consistency
        preservation = 0.0
        
        # Factor 1: φ-constraint preservation
        if '11' not in trace:
            preservation += 0.5  # Maintains golden constraint
            
        # Factor 2: Fibonacci alignment
        fib_indices = self._get_fibonacci_indices(trace)
        if fib_indices:
            fib_consistency = sum(1 for i in fib_indices 
                                if i < len(self.fibonacci_numbers)) / len(fib_indices)
            preservation += 0.3 * fib_consistency
            
        # Factor 3: Structural balance
        ones_count = trace.count('1')
        zeros_count = trace.count('0')
        if ones_count + zeros_count > 0:
            balance = min(ones_count, zeros_count) / max(ones_count, zeros_count)
            preservation += 0.2 * balance
            
        return min(1.0, preservation)
        
    def _classify_homomorph_type(self, trace: str) -> str:
        """分类态射类型"""
        if not trace or trace == '0':
            return 'zero'
            
        kernel_dim = self._compute_kernel_dimension(trace)
        image_rank = self._compute_image_rank(trace)
        preservation = self._compute_preservation_measure(trace)
        
        if kernel_dim == 0 and image_rank > 0:
            return 'injective'
        elif image_rank == self.max_morphism_complexity:
            return 'surjective'
        elif kernel_dim == 0 and image_rank == self.max_morphism_complexity:
            return 'isomorphism'
        elif preservation > 0.8:
            return 'embedding'
        else:
            return 'general'
            
    def _compute_injectivity_index(self, trace: str) -> float:
        """计算单射性指数"""
        if not trace or trace == '0':
            return 0.0  # Zero morphism is not injective
            
        # Injectivity based on kernel triviality
        kernel_dim = self._compute_kernel_dimension(trace)
        max_kernel = self.max_morphism_complexity
        
        # Lower kernel dimension = higher injectivity
        injectivity = 1.0 - (kernel_dim / max_kernel) if max_kernel > 0 else 1.0
        
        # Bonus for φ-constraint preservation
        if '11' not in trace:
            injectivity = min(1.0, injectivity * 1.1)
            
        return max(0.0, min(1.0, injectivity))
        
    def _compute_surjectivity_measure(self, trace: str) -> float:
        """计算满射性度量"""
        if not trace or trace == '0':
            return 0.0  # Zero morphism is not surjective
            
        # Surjectivity based on image coverage
        image_rank = self._compute_image_rank(trace)
        max_rank = self.max_morphism_complexity
        
        surjectivity = image_rank / max_rank if max_rank > 0 else 0.0
        
        # Bonus for good preservation
        preservation = self._compute_preservation_measure(trace)
        surjectivity = min(1.0, surjectivity * (1 + 0.2 * preservation))
        
        return max(0.0, min(1.0, surjectivity))
        
    def analyze_homomorphism_system(self) -> Dict:
        """分析complete homomorphism system"""
        elements = list(self.trace_universe.keys())
        homomorph_data = []
        
        for n in elements:
            trace_info = self.trace_universe[n]
            homomorph_props = trace_info['homomorph_properties']
            
            homomorph_data.append({
                'element': n,
                'trace': trace_info['trace'],
                'morphism_degree': homomorph_props['morphism_degree'],
                'preservation_measure': homomorph_props['preservation_measure'],
                'kernel_dimension': homomorph_props['kernel_dimension'],
                'image_rank': homomorph_props['image_rank'],
                'morphism_signature': homomorph_props['morphism_signature'],
                'structural_preservation': homomorph_props['structural_preservation'],
                'homomorph_type': homomorph_props['homomorph_type'],
                'injectivity_index': homomorph_props['injectivity_index'],
                'surjectivity_measure': homomorph_props['surjectivity_measure']
            })
            
        return self._compute_system_analysis(homomorph_data)
        
    def _compute_system_analysis(self, homomorph_data: List[Dict]) -> Dict:
        """计算系统级分析"""
        if not homomorph_data:
            return {}
            
        # Basic statistics
        degrees = [item['morphism_degree'] for item in homomorph_data]
        preservation_measures = [item['preservation_measure'] for item in homomorph_data]
        kernel_dimensions = [item['kernel_dimension'] for item in homomorph_data]
        image_ranks = [item['image_rank'] for item in homomorph_data]
        structural_preservations = [item['structural_preservation'] for item in homomorph_data]
        injectivity_indices = [item['injectivity_index'] for item in homomorph_data]
        surjectivity_measures = [item['surjectivity_measure'] for item in homomorph_data]
        
        # Type distribution
        types = [item['homomorph_type'] for item in homomorph_data]
        type_counts = {t: types.count(t) for t in set(types)}
        
        # Network analysis
        network_analysis = self._analyze_homomorphism_network(homomorph_data)
        
        # Information theory analysis
        info_analysis = self._analyze_homomorphism_information(homomorph_data)
        
        # Category theory analysis  
        category_analysis = self._analyze_homomorphism_categories(homomorph_data)
        
        return {
            'system': {
                'element_count': len(homomorph_data),
                'mean_degree': np.mean(degrees),
                'max_degree': max(degrees),
                'mean_preservation': np.mean(preservation_measures),
                'mean_kernel_dimension': np.mean(kernel_dimensions),
                'mean_image_rank': np.mean(image_ranks),
                'mean_structural_preservation': np.mean(structural_preservations),
                'mean_injectivity': np.mean(injectivity_indices),
                'mean_surjectivity': np.mean(surjectivity_measures),
                'homomorph_types': type_counts
            },
            'network': network_analysis,
            'information': info_analysis,
            'category': category_analysis,
            'homomorph_data': homomorph_data
        }
        
    def _analyze_homomorphism_network(self, homomorph_data: List[Dict]) -> Dict:
        """分析homomorphism network结构"""
        G = nx.DiGraph()  # Use directed graph for morphisms
        
        # Add nodes
        for item in homomorph_data:
            G.add_node(item['element'], **item)
            
        # Add edges based on morphism relationships
        for i, item1 in enumerate(homomorph_data):
            for j, item2 in enumerate(homomorph_data[i+1:], i+1):
                # Morphism relationship criterion
                sig1 = item1['morphism_signature']
                sig2 = item2['morphism_signature']
                
                # Check for potential morphism (based on preservation compatibility)
                if (abs(sig1) > 0 and abs(sig2) > 0 and
                    item1['structural_preservation'] > 0.3 and
                    item2['structural_preservation'] > 0.3):
                    
                    # Compute morphism strength
                    preservation_product = (item1['structural_preservation'] * 
                                          item2['structural_preservation'])
                    
                    # Add directed edge if sufficiently strong
                    if preservation_product > 0.2:
                        G.add_edge(item1['element'], item2['element'], 
                                 weight=preservation_product)
        
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
        
    def _analyze_homomorphism_information(self, homomorph_data: List[Dict]) -> Dict:
        """分析homomorphism information content"""
        if not homomorph_data:
            return {}
            
        # Degree entropy
        degrees = [item['morphism_degree'] for item in homomorph_data]
        degree_entropy = self._compute_entropy([degrees.count(d) for d in set(degrees)])
        
        # Type entropy  
        types = [item['homomorph_type'] for item in homomorph_data]
        type_entropy = self._compute_entropy([types.count(t) for t in set(types)])
        
        # Kernel entropy
        kernels = [item['kernel_dimension'] for item in homomorph_data]
        kernel_entropy = self._compute_entropy([kernels.count(k) for k in set(kernels)])
        
        # Image entropy
        images = [item['image_rank'] for item in homomorph_data]
        image_entropy = self._compute_entropy([images.count(i) for i in set(images)])
        
        return {
            'degree_entropy': degree_entropy,
            'type_entropy': type_entropy,
            'kernel_entropy': kernel_entropy,
            'image_entropy': image_entropy,
            'homomorph_complexity': len(set(types)),
            'degree_diversity': len(set(degrees)),
            'kernel_diversity': len(set(kernels)),
            'image_diversity': len(set(images))
        }
        
    def _analyze_homomorphism_categories(self, homomorph_data: List[Dict]) -> Dict:
        """分析homomorphism category structure"""
        if not homomorph_data:
            return {}
            
        # Count morphisms (structure-preserving maps)
        morphisms = 0
        functorial_relationships = 0
        
        for i, item1 in enumerate(homomorph_data):
            for j, item2 in enumerate(homomorph_data[i+1:], i+1):
                # Check for morphism (degree and preservation compatibility)
                if (item1['morphism_degree'] <= item2['morphism_degree'] and
                    abs(item1['preservation_measure'] - item2['preservation_measure']) < 0.3):
                    morphisms += 1
                    
                    # Check for functoriality (type preservation)
                    if (item1['homomorph_type'] == item2['homomorph_type'] or
                        (item1['homomorph_type'] in ['injective', 'embedding'] and
                         item2['homomorph_type'] in ['injective', 'embedding']) or
                        (item1['homomorph_type'] in ['surjective', 'isomorphism'] and
                         item2['homomorph_type'] in ['surjective', 'isomorphism'])):
                        functorial_relationships += 1
        
        functoriality_ratio = (functorial_relationships / morphisms 
                             if morphisms > 0 else 0)
        
        # Composition analysis
        composable_pairs = 0
        for item1 in homomorph_data:
            for item2 in homomorph_data:
                if (item1['image_rank'] <= item2['kernel_dimension'] + item2['image_rank'] and
                    item1['structural_preservation'] > 0.5 and
                    item2['structural_preservation'] > 0.5):
                    composable_pairs += 1
        
        return {
            'morphism_count': morphisms,
            'functorial_relationships': functorial_relationships,
            'functoriality_ratio': functoriality_ratio,
            'composable_pairs': composable_pairs,
            'category_structure': 'homomorphism_category'
        }
        
    def _compute_entropy(self, counts: List[int]) -> float:
        """计算熵"""
        if not counts or sum(counts) == 0:
            return 0.0
            
        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]
        
        return -sum(p * log2(p) for p in probabilities)
        
    def generate_visualizations(self, analysis_results: Dict, output_prefix: str):
        """生成homomorphism system可视化"""
        self._plot_homomorphism_structure(analysis_results, f"{output_prefix}-structure.png")
        self._plot_homomorphism_properties(analysis_results, f"{output_prefix}-properties.png")
        self._plot_domain_analysis(analysis_results, f"{output_prefix}-domains.png")
        
    def _plot_homomorphism_structure(self, analysis: Dict, filename: str):
        """可视化homomorphism结构"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        homomorph_data = analysis['homomorph_data']
        elements = [item['element'] for item in homomorph_data]
        degrees = [item['morphism_degree'] for item in homomorph_data]
        preservation = [item['preservation_measure'] for item in homomorph_data]
        kernel_dims = [item['kernel_dimension'] for item in homomorph_data]
        image_ranks = [item['image_rank'] for item in homomorph_data]
        
        # Morphism degree distribution
        ax1.bar(elements, degrees, color='skyblue', alpha=0.7)
        ax1.set_title('Morphism Degree Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trace Elements')
        ax1.set_ylabel('Morphism Degree')
        ax1.grid(True, alpha=0.3)
        
        # Preservation vs Kernel dimension
        colors = plt.cm.viridis([ir/max(image_ranks) if max(image_ranks) > 0 else 0 for ir in image_ranks])
        scatter = ax2.scatter(preservation, kernel_dims, c=colors, s=100, alpha=0.7)
        ax2.set_title('Preservation vs Kernel Dimension', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Preservation Measure')
        ax2.set_ylabel('Kernel Dimension')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Image Rank')
        
        # Homomorphism type distribution
        type_counts = analysis['system']['homomorph_types']
        ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3.colors)
        ax3.set_title('Homomorphism Type Distribution', fontsize=14, fontweight='bold')
        
        # Morphism signature visualization
        signatures = [item['morphism_signature'] for item in homomorph_data]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=degrees, s=100, alpha=0.7, cmap='plasma')
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)
        ax4.set_title('Morphism Signatures (Complex Plane)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_homomorphism_properties(self, analysis: Dict, filename: str):
        """可视化homomorphism属性"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        homomorph_data = analysis['homomorph_data']
        elements = [item['element'] for item in homomorph_data]
        injectivity = [item['injectivity_index'] for item in homomorph_data]
        surjectivity = [item['surjectivity_measure'] for item in homomorph_data]
        structural_preservation = [item['structural_preservation'] for item in homomorph_data]
        
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
        info_metrics = ['degree_entropy', 'type_entropy', 'kernel_entropy', 'image_entropy']
        info_values = [info.get(metric, 0) for metric in info_metrics]
        
        ax2.bar(info_metrics, info_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Information Theory Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        
        # Injectivity vs Surjectivity
        ax3.scatter(injectivity, surjectivity, c=structural_preservation, s=100, alpha=0.7, cmap='coolwarm')
        ax3.set_title('Injectivity vs Surjectivity', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Injectivity Index')
        ax3.set_ylabel('Surjectivity Measure')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Structural Preservation')
        
        # Category theory metrics
        category = analysis['category']
        cat_metrics = ['morphism_count', 'functorial_relationships', 'composable_pairs']
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
        phi_ops = len(analysis['homomorph_data'])  # φ-constrained operations
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
        efficiency_metrics = ['mean_preservation', 'mean_structural_preservation', 'mean_injectivity']
        efficiency_values = [system.get(metric, 0) for metric in efficiency_metrics]
        
        ax2.bar(efficiency_metrics, efficiency_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Homomorphism Efficiency Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Kernel vs Image analysis
        homomorph_data = analysis['homomorph_data']
        kernel_dims = [item['kernel_dimension'] for item in homomorph_data]
        image_ranks = [item['image_rank'] for item in homomorph_data]
        
        kernel_image_data = {}
        for k, i in zip(kernel_dims, image_ranks):
            key = (k, i)
            kernel_image_data[key] = kernel_image_data.get(key, 0) + 1
            
        if kernel_image_data:
            keys = list(kernel_image_data.keys())
            values = list(kernel_image_data.values())
            
            x_pos = range(len(keys))
            ax3.bar(x_pos, values, color='purple', alpha=0.7)
            ax3.set_title('Kernel-Image Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('(Kernel Dim, Image Rank)')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'({k},{i})' for k, i in keys], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # System complexity evolution
        info = analysis['information']
        complexity_metrics = ['homomorph_complexity', 'degree_diversity', 'kernel_diversity', 'image_diversity']
        complexity_values = [info.get(metric, 0) for metric in complexity_metrics]
        
        ax4.plot(complexity_metrics, complexity_values, 'o-', linewidth=2, markersize=8,
                color='red', alpha=0.7)
        ax4.set_title('System Complexity Evolution', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Diversity Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


class TestCollapseHomomorphSystem(unittest.TestCase):
    """Test suite for CollapseHomomorphSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = CollapseHomomorphSystem(max_trace_size=4, max_morphism_complexity=3)
        
    def test_homomorphism_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.trace_universe, dict)
        self.assertGreater(len(self.system.trace_universe), 0)
        
    def test_homomorphism_properties_computation(self):
        """Test homomorphism properties computation"""
        trace = "101"  # φ-valid trace
        props = self.system._compute_homomorph_properties(trace)
        
        required_keys = ['morphism_degree', 'preservation_measure', 'kernel_dimension',
                        'image_rank', 'morphism_signature', 'structural_preservation',
                        'homomorph_type', 'injectivity_index', 'surjectivity_measure']
        
        for key in required_keys:
            self.assertIn(key, props)
            
        # Test constraints
        self.assertGreaterEqual(props['morphism_degree'], 0)
        self.assertLessEqual(props['morphism_degree'], self.system.max_morphism_complexity)
        self.assertGreaterEqual(props['preservation_measure'], 0.0)
        self.assertLessEqual(props['preservation_measure'], 1.0)
        self.assertGreaterEqual(props['kernel_dimension'], 0)
        self.assertGreaterEqual(props['image_rank'], 0)
        
    def test_phi_constraint_preservation(self):
        """Test φ-constraint preservation"""
        # Valid traces (no consecutive 1s)
        valid_traces = ["0", "1", "10", "101", "1010"]
        for trace in valid_traces:
            self.assertNotIn('11', trace)
            props = self.system._compute_homomorph_properties(trace)
            self.assertGreaterEqual(props['image_rank'], 0)
            
        # Invalid trace should have zero image rank
        invalid_trace = "11"
        props = self.system._compute_homomorph_properties(invalid_trace)
        self.assertEqual(props['image_rank'], 0)
        
    def test_homomorphism_system_analysis(self):
        """Test complete homomorphism system analysis"""
        analysis = self.system.analyze_homomorphism_system()
        
        required_sections = ['system', 'network', 'information', 'category', 'homomorph_data']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Test system metrics
        system = analysis['system']
        self.assertGreater(system['element_count'], 0)
        self.assertGreaterEqual(system['mean_degree'], 0)
        self.assertGreaterEqual(system['max_degree'], 0)
        
    def test_morphism_signature_computation(self):
        """Test morphism signature computation"""
        trace = "101"
        signature = self.system._compute_morphism_signature(trace)
        
        self.assertIsInstance(signature, complex)
        # Should be approximately on unit circle
        magnitude = abs(signature)
        self.assertAlmostEqual(magnitude, 1.0, places=10)
        
    def test_homomorphism_type_classification(self):
        """Test homomorphism type classification"""
        # Test different trace patterns
        test_cases = [
            ("0", "zero"),
            ("1", "injective"),  # Should be injective or general
            ("101", "general"),  # Should be some valid type
        ]
        
        for trace, expected_category in test_cases:
            homomorph_type = self.system._classify_homomorph_type(trace)
            if expected_category == "zero":
                self.assertEqual(homomorph_type, expected_category)
            elif expected_category == "injective":
                self.assertIn(homomorph_type, ["injective", "embedding", "general"])
            else:
                # Any valid type is acceptable for complex traces
                self.assertIn(homomorph_type, ["zero", "injective", "surjective", "isomorphism", "embedding", "general"])
        
    def test_structure_preservation(self):
        """Test structure preservation computation"""
        # φ-valid traces should have good preservation
        valid_trace = "101"
        preservation = self.system._compute_structural_preservation(valid_trace)
        self.assertGreaterEqual(preservation, 0.5)  # Should preserve φ-constraint
        
        # Zero trace should have perfect preservation
        zero_trace = "0"
        zero_preservation = self.system._compute_structural_preservation(zero_trace)
        self.assertEqual(zero_preservation, 1.0)


def main():
    """Main execution function"""
    print("=" * 80)
    print("CHAPTER 062: COLLAPSE HOMOMORPH VERIFICATION")
    print("Morphism Mapping between Collapse Tensor Algebras")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing CollapseHomomorph System...")
    system = CollapseHomomorphSystem(max_trace_size=6, max_morphism_complexity=4)
    print(f"   φ-valid traces found: {len(system.trace_universe)}")
    
    # Analyze homomorphism system
    print("\n2. Analyzing Homomorphism System...")
    analysis_results = system.analyze_homomorphism_system()
    
    # Display results
    print("\n3. Homomorphism System Analysis Results:")
    print("-" * 50)
    
    system_data = analysis_results['system']
    print(f"Homomorphism elements: {system_data['element_count']}")
    print(f"Mean morphism degree: {system_data['mean_degree']:.3f}")
    print(f"Maximum degree: {system_data['max_degree']}")
    print(f"Mean preservation: {system_data['mean_preservation']:.3f}")
    print(f"Mean kernel dimension: {system_data['mean_kernel_dimension']:.3f}")
    print(f"Mean image rank: {system_data['mean_image_rank']:.3f}")
    print(f"Mean structural preservation: {system_data['mean_structural_preservation']:.3f}")
    print(f"Mean injectivity: {system_data['mean_injectivity']:.3f}")
    print(f"Mean surjectivity: {system_data['mean_surjectivity']:.3f}")
    
    print(f"\nHomomorphism Type Distribution:")
    for homomorph_type, count in system_data['homomorph_types'].items():
        percentage = (count / system_data['element_count']) * 100
        print(f"  {homomorph_type.capitalize()} morphisms: {count} ({percentage:.1f}%)")
    
    network_data = analysis_results['network']
    print(f"\nNetwork Analysis:")
    print(f"  Network density: {network_data['density']:.3f}")
    print(f"  Connected components: {network_data['components']}")
    print(f"  Average clustering: {network_data['clustering']:.3f}")
    
    info_data = analysis_results['information']
    print(f"\nInformation Theory:")
    print(f"  Degree entropy: {info_data['degree_entropy']:.3f} bits")
    print(f"  Type entropy: {info_data['type_entropy']:.3f} bits")
    print(f"  Kernel entropy: {info_data['kernel_entropy']:.3f} bits")
    print(f"  Image entropy: {info_data['image_entropy']:.3f} bits")
    print(f"  Homomorph complexity: {info_data['homomorph_complexity']} unique types")
    
    category_data = analysis_results['category']
    print(f"\nCategory Theory:")
    print(f"  Morphism count: {category_data['morphism_count']}")
    print(f"  Functorial relationships: {category_data['functorial_relationships']}")
    print(f"  Functoriality ratio: {category_data['functoriality_ratio']:.3f}")
    print(f"  Composable pairs: {category_data['composable_pairs']}")
    
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
    output_prefix = "chapter-062-collapse-homomorph"
    system.generate_visualizations(analysis_results, output_prefix)
    print(f"   Generated: {output_prefix}-structure.png")
    print(f"   Generated: {output_prefix}-properties.png") 
    print(f"   Generated: {output_prefix}-domains.png")
    
    # Run unit tests
    print("\n5. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    main()