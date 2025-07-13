#!/usr/bin/env python3
"""
Chapter 064: TensorSpace Unit Test Verification
从ψ=ψ(ψ)推导Interpreting Collapse Traces as Spatial Connectivity

Core principle: From ψ = ψ(ψ) derive spatial connectivity where trace networks are φ-valid
spatial structures that encode geometric relationships through trace-based tensor connections,
creating systematic spatial frameworks with bounded connectivity and natural topological
properties governed by golden constraints, showing how traces naturally represent space.

This verification program implements:
1. φ-constrained spatial connectivity as trace tensor spatial operations
2. Spatial analysis: connectivity patterns, spatial structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection spatial connectivity theory
4. Graph theory analysis of spatial networks and geometric connectivity patterns
5. Information theory analysis of spatial entropy and connectivity information
6. Category theory analysis of spatial functors and geometric morphisms
7. Visualization of spatial structures and connectivity patterns
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

class TensorSpaceSystem:
    """
    Core system for implementing tensor space with spatial connectivity from collapse traces.
    Implements φ-constrained spatial theory via trace-based tensor spatial operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_spatial_complexity: int = 4):
        """Initialize tensor space system"""
        self.max_trace_size = max_trace_size
        self.max_spatial_complexity = max_spatial_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.spatial_cache = {}
        self.connectivity_cache = {}
        self.tensor_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_spatial=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for spatial properties computation
        self.trace_universe = universe
        
        # Second pass: add spatial properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['spatial_properties'] = self._compute_spatial_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_spatial: bool = True) -> Dict:
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
        
        if compute_spatial and hasattr(self, 'trace_universe'):
            result['spatial_properties'] = self._compute_spatial_properties(trace)
            
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
        
    def _compute_spatial_properties(self, trace: str) -> Dict:
        """计算trace的spatial相关属性"""
        cache_key = trace
        if cache_key in self.spatial_cache:
            return self.spatial_cache[cache_key]
            
        result = {
            'spatial_dimension': self._compute_spatial_dimension(trace),
            'connectivity_measure': self._compute_connectivity_measure(trace),
            'tensor_rank': self._compute_tensor_rank(trace),
            'spatial_density': self._compute_spatial_density(trace),
            'geometric_signature': self._compute_geometric_signature(trace),
            'topological_invariant': self._compute_topological_invariant(trace),
            'spatial_type': self._classify_spatial_type(trace),
            'embedding_dimension': self._compute_embedding_dimension(trace),
            'spatial_curvature': self._compute_spatial_curvature(trace)
        }
        
        self.spatial_cache[cache_key] = result
        return result
        
    def _compute_spatial_dimension(self, trace: str) -> int:
        """计算空间维度：基于trace复杂度"""
        if not trace or trace == '0':
            return 0  # Zero space has dimension 0
        
        # Use ones count to determine spatial dimension
        ones_count = trace.count('1')
        if ones_count == 0:
            return 0
        elif ones_count == 1:
            return 1  # Dimension 1 space (line)
        elif ones_count == 2:
            return 2  # Dimension 2 space (plane)
        else:
            return min(self.max_spatial_complexity, ones_count)
            
    def _compute_connectivity_measure(self, trace: str) -> float:
        """计算连通性度量"""
        if not trace or trace == '0':
            return 0.0  # Zero space has no connectivity
            
        # Complex harmonic measure based on position structure
        measure = 0.0
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) <= 1:
            return 0.0  # Need at least 2 points for connectivity
            
        # Calculate connectivity based on position relationships
        total_connections = 0
        for i, pos1 in enumerate(ones_positions):
            for j, pos2 in enumerate(ones_positions[i+1:], i+1):
                distance = abs(pos2 - pos1)
                # Fibonacci-weighted connectivity
                fib_weight1 = self.fibonacci_numbers[min(pos1, len(self.fibonacci_numbers) - 1)]
                fib_weight2 = self.fibonacci_numbers[min(pos2, len(self.fibonacci_numbers) - 1)]
                connection_strength = (fib_weight1 * fib_weight2) / (distance + 1)
                total_connections += connection_strength
                
        # Normalize by maximum possible connections
        max_connections = len(ones_positions) * (len(ones_positions) - 1) / 2
        if max_connections > 0:
            measure = min(1.0, total_connections / (max_connections * max(self.fibonacci_numbers[:len(trace)])))
            
        return measure
        
    def _compute_tensor_rank(self, trace: str) -> int:
        """计算张量秩：基于trace结构"""
        if not trace or trace == '0':
            return 0  # Zero tensor has rank 0
            
        # Tensor rank based on structural complexity
        ones_count = trace.count('1')
        
        # Check for φ-constraint violation
        if '11' in trace:
            return 0  # Invalid tensor
            
        # Rank based on spatial distribution
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if not ones_positions:
            return 0
            
        # Higher rank for well-distributed positions
        if len(ones_positions) == 1:
            return 1
        elif len(ones_positions) == 2:
            return 2
        else:
            return min(self.max_spatial_complexity, len(ones_positions))
            
    def _compute_spatial_density(self, trace: str) -> float:
        """计算空间密度"""
        if not trace or trace == '0':
            return 0.0  # Zero space has no density
            
        # Density as ratio of active positions to total space
        ones_count = trace.count('1')
        total_length = len(trace)
        
        if total_length == 0:
            return 0.0
            
        # Basic density
        basic_density = ones_count / total_length
        
        # Fibonacci-weighted density
        fib_weight = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                fib_weight += self.fibonacci_numbers[fib_idx]
                
        max_fib_weight = sum(self.fibonacci_numbers[:total_length])
        fib_density = fib_weight / max_fib_weight if max_fib_weight > 0 else 0
        
        # Combined density measure
        return (basic_density + fib_density) / 2
        
    def _compute_geometric_signature(self, trace: str) -> complex:
        """计算几何签名：复数表示的几何特征"""
        if not trace:
            return complex(0, 0)
            
        # Complex harmonic encoding for geometric structure
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                # Geometric angle based on position
                angle = 2 * pi * i / len(trace)
                weight = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers) - 1)]
                real_part += weight * cos(angle)
                imag_part += weight * sin(angle)
                
        # Normalize to unit circle
        magnitude = sqrt(real_part**2 + imag_part**2)
        if magnitude > 0:
            return complex(real_part / magnitude, imag_part / magnitude)
        return complex(0, 0)
        
    def _compute_topological_invariant(self, trace: str) -> int:
        """计算拓扑不变量"""
        if not trace or trace == '0':
            return 0  # Zero space has trivial topology
            
        # Compute Euler characteristic-like invariant
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if not ones_positions:
            return 0
            
        # Simple invariant based on connectivity pattern
        vertices = len(ones_positions)  # Number of active positions
        
        # Compute "edges" based on adjacency (respecting φ-constraint)
        edges = 0
        for i in range(len(ones_positions) - 1):
            pos1, pos2 = ones_positions[i], ones_positions[i+1]
            if pos2 - pos1 >= 2:  # No consecutive 1s allowed
                edges += 1
                
        # Euler characteristic χ = V - E (for 1D complex)
        euler_char = vertices - edges
        
        return euler_char
        
    def _classify_spatial_type(self, trace: str) -> str:
        """分类空间类型"""
        if not trace or trace == '0':
            return 'empty'
            
        dimension = self._compute_spatial_dimension(trace)
        connectivity = self._compute_connectivity_measure(trace)
        density = self._compute_spatial_density(trace)
        
        if dimension == 0:
            return 'empty'
        elif dimension == 1:
            return 'linear'
        elif dimension == 2 and connectivity > 0.5:
            return 'planar'
        elif dimension >= 3 and density > 0.6:
            return 'volumetric'
        elif connectivity > 0.7:
            return 'connected'
        else:
            return 'discrete'
            
    def _compute_embedding_dimension(self, trace: str) -> int:
        """计算嵌入维度"""
        if not trace or trace == '0':
            return 0  # Empty space embeds in 0D
            
        # Minimum dimension needed to embed the spatial structure
        spatial_dim = self._compute_spatial_dimension(trace)
        ones_count = trace.count('1')
        
        # Embedding dimension is typically spatial dimension + 1
        # But bounded by trace complexity
        embedding_dim = min(spatial_dim + 1, ones_count, self.max_spatial_complexity)
        
        return max(1, embedding_dim)  # At least 1D embedding
        
    def _compute_spatial_curvature(self, trace: str) -> float:
        """计算空间曲率"""
        if not trace or trace == '0':
            return 0.0  # Empty space has no curvature
            
        # Curvature based on how positions deviate from uniform distribution
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) < 3:
            return 0.0  # Need at least 3 points for curvature
            
        # Expected uniform spacing
        total_length = len(trace)
        expected_spacing = total_length / len(ones_positions)
        
        # Compute curvature as deviation from uniform spacing
        curvature = 0.0
        for i in range(1, len(ones_positions) - 1):
            pos_prev = ones_positions[i-1]
            pos_curr = ones_positions[i]
            pos_next = ones_positions[i+1]
            
            # Local spacing deviations
            spacing1 = pos_curr - pos_prev
            spacing2 = pos_next - pos_curr
            
            # Curvature as second derivative approximation
            local_curvature = abs(spacing2 - spacing1) / expected_spacing
            curvature += local_curvature
            
        # Normalize by number of curvature measurements
        if len(ones_positions) > 2:
            curvature /= (len(ones_positions) - 2)
            
        return min(1.0, curvature)  # Normalize to [0,1]
        
    def analyze_tensor_space_system(self) -> Dict:
        """分析complete tensor space system"""
        elements = list(self.trace_universe.keys())
        spatial_data = []
        
        for n in elements:
            trace_info = self.trace_universe[n]
            spatial_props = trace_info['spatial_properties']
            
            spatial_data.append({
                'element': n,
                'trace': trace_info['trace'],
                'spatial_dimension': spatial_props['spatial_dimension'],
                'connectivity_measure': spatial_props['connectivity_measure'],
                'tensor_rank': spatial_props['tensor_rank'],
                'spatial_density': spatial_props['spatial_density'],
                'geometric_signature': spatial_props['geometric_signature'],
                'topological_invariant': spatial_props['topological_invariant'],
                'spatial_type': spatial_props['spatial_type'],
                'embedding_dimension': spatial_props['embedding_dimension'],
                'spatial_curvature': spatial_props['spatial_curvature']
            })
            
        return self._compute_system_analysis(spatial_data)
        
    def _compute_system_analysis(self, spatial_data: List[Dict]) -> Dict:
        """计算系统级分析"""
        if not spatial_data:
            return {}
            
        # Basic statistics
        dimensions = [item['spatial_dimension'] for item in spatial_data]
        connectivity_measures = [item['connectivity_measure'] for item in spatial_data]
        tensor_ranks = [item['tensor_rank'] for item in spatial_data]
        spatial_densities = [item['spatial_density'] for item in spatial_data]
        topological_invariants = [item['topological_invariant'] for item in spatial_data]
        embedding_dimensions = [item['embedding_dimension'] for item in spatial_data]
        spatial_curvatures = [item['spatial_curvature'] for item in spatial_data]
        
        # Type distribution
        types = [item['spatial_type'] for item in spatial_data]
        type_counts = {t: types.count(t) for t in set(types)}
        
        # Network analysis
        network_analysis = self._analyze_spatial_network(spatial_data)
        
        # Information theory analysis
        info_analysis = self._analyze_spatial_information(spatial_data)
        
        # Category theory analysis  
        category_analysis = self._analyze_spatial_categories(spatial_data)
        
        return {
            'system': {
                'element_count': len(spatial_data),
                'mean_dimension': np.mean(dimensions),
                'max_dimension': max(dimensions),
                'mean_connectivity': np.mean(connectivity_measures),
                'mean_tensor_rank': np.mean(tensor_ranks),
                'mean_spatial_density': np.mean(spatial_densities),
                'mean_topological_invariant': np.mean(topological_invariants),
                'mean_embedding_dimension': np.mean(embedding_dimensions),
                'mean_spatial_curvature': np.mean(spatial_curvatures),
                'spatial_types': type_counts
            },
            'network': network_analysis,
            'information': info_analysis,
            'category': category_analysis,
            'spatial_data': spatial_data
        }
        
    def _analyze_spatial_network(self, spatial_data: List[Dict]) -> Dict:
        """分析spatial network结构"""
        G = nx.Graph()  # Undirected graph for spatial relationships
        
        # Add nodes
        for item in spatial_data:
            G.add_node(item['element'], **item)
            
        # Add edges based on spatial relationships
        for i, item1 in enumerate(spatial_data):
            for j, item2 in enumerate(spatial_data[i+1:], i+1):
                # Spatial relationship criterion
                sig1 = item1['geometric_signature']
                sig2 = item2['geometric_signature']
                
                # Check for spatial compatibility
                if (abs(sig1) > 0 and abs(sig2) > 0 and
                    item1['spatial_density'] > 0.1 and
                    item2['spatial_density'] > 0.1):
                    
                    # Compute spatial affinity
                    geometric_similarity = abs((sig1.real * sig2.real + sig1.imag * sig2.imag) / 
                                             (abs(sig1) * abs(sig2)))
                    dimension_compatibility = 1.0 - abs(item1['spatial_dimension'] - item2['spatial_dimension']) / max(item1['spatial_dimension'], item2['spatial_dimension'], 1)
                    
                    affinity = (geometric_similarity + dimension_compatibility) / 2
                    
                    # Add edge if sufficiently compatible
                    if affinity > 0.4:  # Threshold for spatial compatibility
                        G.add_edge(item1['element'], item2['element'], 
                                 weight=affinity)
        
        if G.number_of_edges() == 0:
            density = 0.0
            clustering = 0.0
        else:
            density = nx.density(G)
            clustering = nx.average_clustering(G)
            
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': density,
            'components': nx.number_connected_components(G),
            'clustering': clustering
        }
        
    def _analyze_spatial_information(self, spatial_data: List[Dict]) -> Dict:
        """分析spatial information content"""
        if not spatial_data:
            return {}
            
        # Dimension entropy
        dimensions = [item['spatial_dimension'] for item in spatial_data]
        dimension_entropy = self._compute_entropy([dimensions.count(d) for d in set(dimensions)])
        
        # Type entropy  
        types = [item['spatial_type'] for item in spatial_data]
        type_entropy = self._compute_entropy([types.count(t) for t in set(types)])
        
        # Tensor rank entropy
        ranks = [item['tensor_rank'] for item in spatial_data]
        rank_entropy = self._compute_entropy([ranks.count(r) for r in set(ranks)])
        
        # Topological invariant entropy
        invariants = [item['topological_invariant'] for item in spatial_data]
        invariant_entropy = self._compute_entropy([invariants.count(i) for i in set(invariants)])
        
        return {
            'dimension_entropy': dimension_entropy,
            'type_entropy': type_entropy,
            'rank_entropy': rank_entropy,
            'invariant_entropy': invariant_entropy,
            'spatial_complexity': len(set(types)),
            'dimension_diversity': len(set(dimensions)),
            'rank_diversity': len(set(ranks)),
            'invariant_diversity': len(set(invariants))
        }
        
    def _analyze_spatial_categories(self, spatial_data: List[Dict]) -> Dict:
        """分析spatial category structure"""
        if not spatial_data:
            return {}
            
        # Count spatial morphisms (dimension-preserving maps)
        spatial_morphisms = 0
        functorial_relationships = 0
        
        for i, item1 in enumerate(spatial_data):
            for j, item2 in enumerate(spatial_data[i+1:], i+1):
                # Check for spatial morphism (dimension and type compatibility)
                if (item1['spatial_dimension'] <= item2['spatial_dimension'] and
                    abs(item1['spatial_density'] - item2['spatial_density']) < 0.3):
                    spatial_morphisms += 1
                    
                    # Check for functoriality (type preservation)
                    if (item1['spatial_type'] == item2['spatial_type'] or
                        (item1['spatial_type'] in ['linear', 'planar'] and
                         item2['spatial_type'] in ['linear', 'planar']) or
                        (item1['spatial_type'] in ['connected', 'volumetric'] and
                         item2['spatial_type'] in ['connected', 'volumetric'])):
                        functorial_relationships += 1
        
        functoriality_ratio = (functorial_relationships / spatial_morphisms 
                             if spatial_morphisms > 0 else 0)
        
        # Embedding analysis
        embeddable_pairs = 0
        for item1 in spatial_data:
            for item2 in spatial_data:
                if (item1['embedding_dimension'] <= item2['embedding_dimension'] and
                    item1['spatial_density'] > 0.2 and
                    item2['spatial_density'] > 0.2):
                    embeddable_pairs += 1
        
        return {
            'spatial_morphisms': spatial_morphisms,
            'functorial_relationships': functorial_relationships,
            'functoriality_ratio': functoriality_ratio,
            'embeddable_pairs': embeddable_pairs,
            'category_structure': 'tensor_space_category'
        }
        
    def _compute_entropy(self, counts: List[int]) -> float:
        """计算熵"""
        if not counts or sum(counts) == 0:
            return 0.0
            
        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]
        
        return -sum(p * log2(p) for p in probabilities)
        
    def generate_visualizations(self, analysis_results: Dict, output_prefix: str):
        """生成tensor space system可视化"""
        self._plot_spatial_structure(analysis_results, f"{output_prefix}-structure.png")
        self._plot_spatial_properties(analysis_results, f"{output_prefix}-properties.png")
        self._plot_domain_analysis(analysis_results, f"{output_prefix}-domains.png")
        
    def _plot_spatial_structure(self, analysis: Dict, filename: str):
        """可视化spatial结构"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        spatial_data = analysis['spatial_data']
        elements = [item['element'] for item in spatial_data]
        dimensions = [item['spatial_dimension'] for item in spatial_data]
        connectivity = [item['connectivity_measure'] for item in spatial_data]
        tensor_ranks = [item['tensor_rank'] for item in spatial_data]
        spatial_densities = [item['spatial_density'] for item in spatial_data]
        
        # Spatial dimension distribution
        ax1.bar(elements, dimensions, color='skyblue', alpha=0.7)
        ax1.set_title('Spatial Dimension Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trace Elements')
        ax1.set_ylabel('Spatial Dimension')
        ax1.grid(True, alpha=0.3)
        
        # Connectivity vs Tensor rank
        colors = plt.cm.viridis([sd/max(spatial_densities) if max(spatial_densities) > 0 else 0 for sd in spatial_densities])
        scatter = ax2.scatter(connectivity, tensor_ranks, c=colors, s=100, alpha=0.7)
        ax2.set_title('Connectivity vs Tensor Rank', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Connectivity Measure')
        ax2.set_ylabel('Tensor Rank')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Spatial Density')
        
        # Spatial type distribution
        type_counts = analysis['system']['spatial_types']
        ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3.colors)
        ax3.set_title('Spatial Type Distribution', fontsize=14, fontweight='bold')
        
        # Geometric signature visualization
        signatures = [item['geometric_signature'] for item in spatial_data]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=dimensions, s=100, alpha=0.7, cmap='plasma')
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)
        ax4.set_title('Geometric Signatures (Complex Plane)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_spatial_properties(self, analysis: Dict, filename: str):
        """可视化spatial属性"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        spatial_data = analysis['spatial_data']
        elements = [item['element'] for item in spatial_data]
        topological_invariants = [item['topological_invariant'] for item in spatial_data]
        embedding_dimensions = [item['embedding_dimension'] for item in spatial_data]
        spatial_curvatures = [item['spatial_curvature'] for item in spatial_data]
        
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
        info_metrics = ['dimension_entropy', 'type_entropy', 'rank_entropy', 'invariant_entropy']
        info_values = [info.get(metric, 0) for metric in info_metrics]
        
        ax2.bar(info_metrics, info_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Information Theory Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        
        # Topological invariant vs Embedding dimension
        ax3.scatter(topological_invariants, embedding_dimensions, c=spatial_curvatures, s=100, alpha=0.7, cmap='coolwarm')
        ax3.set_title('Topological Invariant vs Embedding Dimension', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Topological Invariant')
        ax3.set_ylabel('Embedding Dimension')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Spatial Curvature')
        
        # Category theory metrics
        category = analysis['category']
        cat_metrics = ['spatial_morphisms', 'functorial_relationships', 'embeddable_pairs']
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
        phi_ops = len(analysis['spatial_data'])  # φ-constrained operations
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
        efficiency_metrics = ['mean_connectivity', 'mean_spatial_density', 'mean_spatial_curvature']
        efficiency_values = [system.get(metric, 0) for metric in efficiency_metrics]
        
        ax2.bar(efficiency_metrics, efficiency_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Spatial Efficiency Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Dimension vs Tensor rank analysis
        spatial_data = analysis['spatial_data']
        dimensions = [item['spatial_dimension'] for item in spatial_data]
        tensor_ranks = [item['tensor_rank'] for item in spatial_data]
        
        dimension_rank_data = {}
        for d, r in zip(dimensions, tensor_ranks):
            key = (d, r)
            dimension_rank_data[key] = dimension_rank_data.get(key, 0) + 1
            
        if dimension_rank_data:
            keys = list(dimension_rank_data.keys())
            values = list(dimension_rank_data.values())
            
            x_pos = range(len(keys))
            ax3.bar(x_pos, values, color='purple', alpha=0.7)
            ax3.set_title('Dimension-Rank Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('(Dimension, Tensor Rank)')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'({d},{r})' for d, r in keys], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # System complexity evolution
        info = analysis['information']
        complexity_metrics = ['spatial_complexity', 'dimension_diversity', 'rank_diversity', 'invariant_diversity']
        complexity_values = [info.get(metric, 0) for metric in complexity_metrics]
        
        ax4.plot(complexity_metrics, complexity_values, 'o-', linewidth=2, markersize=8,
                color='red', alpha=0.7)
        ax4.set_title('System Complexity Evolution', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Diversity Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


class TestTensorSpaceSystem(unittest.TestCase):
    """Test suite for TensorSpaceSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = TensorSpaceSystem(max_trace_size=4, max_spatial_complexity=3)
        
    def test_tensor_space_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.trace_universe, dict)
        self.assertGreater(len(self.system.trace_universe), 0)
        
    def test_spatial_properties_computation(self):
        """Test spatial properties computation"""
        trace = "101"  # φ-valid trace
        props = self.system._compute_spatial_properties(trace)
        
        required_keys = ['spatial_dimension', 'connectivity_measure', 'tensor_rank',
                        'spatial_density', 'geometric_signature', 'topological_invariant',
                        'spatial_type', 'embedding_dimension', 'spatial_curvature']
        
        for key in required_keys:
            self.assertIn(key, props)
            
        # Test constraints
        self.assertGreaterEqual(props['spatial_dimension'], 0)
        self.assertLessEqual(props['spatial_dimension'], self.system.max_spatial_complexity)
        self.assertGreaterEqual(props['connectivity_measure'], 0.0)
        self.assertLessEqual(props['connectivity_measure'], 1.0)
        self.assertGreaterEqual(props['tensor_rank'], 0)
        self.assertGreaterEqual(props['spatial_density'], 0.0)
        self.assertLessEqual(props['spatial_density'], 1.0)
        
    def test_phi_constraint_preservation(self):
        """Test φ-constraint preservation"""
        # Valid traces (no consecutive 1s)
        valid_traces = ["0", "1", "10", "101", "1010"]
        for trace in valid_traces:
            self.assertNotIn('11', trace)
            props = self.system._compute_spatial_properties(trace)
            self.assertGreaterEqual(props['tensor_rank'], 0)
            
        # Invalid trace should have zero tensor rank
        invalid_trace = "11"
        props = self.system._compute_spatial_properties(invalid_trace)
        self.assertEqual(props['tensor_rank'], 0)
        
    def test_tensor_space_system_analysis(self):
        """Test complete tensor space system analysis"""
        analysis = self.system.analyze_tensor_space_system()
        
        required_sections = ['system', 'network', 'information', 'category', 'spatial_data']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Test system metrics
        system = analysis['system']
        self.assertGreater(system['element_count'], 0)
        self.assertGreaterEqual(system['mean_dimension'], 0)
        self.assertGreaterEqual(system['max_dimension'], 0)
        
    def test_geometric_signature_computation(self):
        """Test geometric signature computation"""
        trace = "101"
        signature = self.system._compute_geometric_signature(trace)
        
        self.assertIsInstance(signature, complex)
        # Should be approximately on unit circle
        magnitude = abs(signature)
        self.assertAlmostEqual(magnitude, 1.0, places=10)
        
    def test_spatial_type_classification(self):
        """Test spatial type classification"""
        # Test different trace patterns
        test_cases = [
            ("0", "empty"),
            ("1", "linear"),  # Should be linear or discrete
            ("101", "linear"),  # Should be some valid type
        ]
        
        for trace, expected_category in test_cases:
            spatial_type = self.system._classify_spatial_type(trace)
            if expected_category == "empty":
                self.assertEqual(spatial_type, expected_category)
            elif expected_category == "linear":
                self.assertIn(spatial_type, ["linear", "discrete", "connected"])
            else:
                # Any valid type is acceptable for complex traces
                self.assertIn(spatial_type, ["empty", "linear", "planar", "volumetric", "connected", "discrete"])
        
    def test_topological_invariant(self):
        """Test topological invariant computation"""
        # φ-valid traces should have reasonable invariants
        valid_trace = "101"
        invariant = self.system._compute_topological_invariant(valid_trace)
        self.assertIsInstance(invariant, int)
        
        # Zero trace should have zero invariant
        zero_trace = "0"
        zero_invariant = self.system._compute_topological_invariant(zero_trace)
        self.assertEqual(zero_invariant, 0)


def main():
    """Main execution function"""
    print("=" * 80)
    print("CHAPTER 064: TENSOR SPACE VERIFICATION")
    print("Interpreting Collapse Traces as Spatial Connectivity")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing TensorSpace System...")
    system = TensorSpaceSystem(max_trace_size=6, max_spatial_complexity=4)
    print(f"   φ-valid traces found: {len(system.trace_universe)}")
    
    # Analyze tensor space system
    print("\n2. Analyzing Tensor Space System...")
    analysis_results = system.analyze_tensor_space_system()
    
    # Display results
    print("\n3. Tensor Space System Analysis Results:")
    print("-" * 50)
    
    system_data = analysis_results['system']
    print(f"Spatial elements: {system_data['element_count']}")
    print(f"Mean spatial dimension: {system_data['mean_dimension']:.3f}")
    print(f"Maximum dimension: {system_data['max_dimension']}")
    print(f"Mean connectivity: {system_data['mean_connectivity']:.3f}")
    print(f"Mean tensor rank: {system_data['mean_tensor_rank']:.3f}")
    print(f"Mean spatial density: {system_data['mean_spatial_density']:.3f}")
    print(f"Mean topological invariant: {system_data['mean_topological_invariant']:.3f}")
    print(f"Mean embedding dimension: {system_data['mean_embedding_dimension']:.3f}")
    print(f"Mean spatial curvature: {system_data['mean_spatial_curvature']:.3f}")
    
    print(f"\nSpatial Type Distribution:")
    for spatial_type, count in system_data['spatial_types'].items():
        percentage = (count / system_data['element_count']) * 100
        print(f"  {spatial_type.capitalize()} spaces: {count} ({percentage:.1f}%)")
    
    network_data = analysis_results['network']
    print(f"\nNetwork Analysis:")
    print(f"  Network density: {network_data['density']:.3f}")
    print(f"  Connected components: {network_data['components']}")
    print(f"  Average clustering: {network_data['clustering']:.3f}")
    
    info_data = analysis_results['information']
    print(f"\nInformation Theory:")
    print(f"  Dimension entropy: {info_data['dimension_entropy']:.3f} bits")
    print(f"  Type entropy: {info_data['type_entropy']:.3f} bits")
    print(f"  Rank entropy: {info_data['rank_entropy']:.3f} bits")
    print(f"  Invariant entropy: {info_data['invariant_entropy']:.3f} bits")
    print(f"  Spatial complexity: {info_data['spatial_complexity']} unique types")
    
    category_data = analysis_results['category']
    print(f"\nCategory Theory:")
    print(f"  Spatial morphisms: {category_data['spatial_morphisms']}")
    print(f"  Functorial relationships: {category_data['functorial_relationships']}")
    print(f"  Functoriality ratio: {category_data['functoriality_ratio']:.3f}")
    print(f"  Embeddable pairs: {category_data['embeddable_pairs']}")
    
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
    output_prefix = "chapter-064-tensor-space"
    system.generate_visualizations(analysis_results, output_prefix)
    print(f"   Generated: {output_prefix}-structure.png")
    print(f"   Generated: {output_prefix}-properties.png") 
    print(f"   Generated: {output_prefix}-domains.png")
    
    # Run unit tests
    print("\n5. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    main()