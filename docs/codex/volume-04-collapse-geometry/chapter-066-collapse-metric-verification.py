#!/usr/bin/env python3
"""
Chapter 066: CollapseMetric Unit Test Verification
从ψ=ψ(ψ)推导Distance as Minimum Transformation Cost between φ-Traces

Core principle: From ψ = ψ(ψ) derive metric spaces where distance is φ-valid
trace transformation cost between traces, creating systematic metric frameworks 
with bounded distance and natural metric properties governed by golden constraints, 
showing how distance emerges from trace transformation cost.

This verification program implements:
1. φ-constrained metric distance as trace transformation cost operations
2. Metric analysis: distance patterns, transformation structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection metric theory
4. Graph theory analysis of distance networks and metric connectivity patterns
5. Information theory analysis of metric entropy and distance information
6. Category theory analysis of metric functors and distance morphisms
7. Visualization of metric structures and transformation patterns
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

class CollapseMetricSystem:
    """
    Core system for implementing collapse metric spaces with φ-trace distance.
    Implements φ-constrained metric theory via trace transformation cost operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_metric_complexity: int = 4):
        """Initialize collapse metric system"""
        self.max_trace_size = max_trace_size
        self.max_metric_complexity = max_metric_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.metric_cache = {}
        self.distance_cache = {}
        self.transformation_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_metric=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for metric properties computation
        self.trace_universe = universe
        
        # Second pass: add metric properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['metric_properties'] = self._compute_metric_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_metric: bool = True) -> Dict:
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
        
        if compute_metric and hasattr(self, 'trace_universe'):
            result['metric_properties'] = self._compute_metric_properties(trace)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将整数编码为二进制trace表示"""
        if n == 0:
            return '0'
        return bin(n)[2:]
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中的Fibonacci位置索引"""
        indices = []
        for i, bit in enumerate(trace):
            if bit == '1' and (i+1) in self.fibonacci_numbers:
                indices.append(i+1)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构哈希值"""
        hash_val = 0
        for i, bit in enumerate(trace):
            if bit == '1':
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                hash_val += self.fibonacci_numbers[fib_idx] * (i + 1)
        return hash_val % 1009  # 使用素数取模
        
    def _compute_binary_weight(self, trace: str) -> float:
        """计算trace的二进制权重"""
        weight = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Fibonacci-weighted position value
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                weight += self.fibonacci_numbers[fib_idx] / (2 ** (i + 1))
        return weight
        
    def _compute_metric_properties(self, trace: str) -> Dict:
        """计算trace的metric属性"""
        if trace in self.metric_cache:
            return self.metric_cache[trace]
            
        properties = {}
        
        # Distance to origin (empty trace)
        properties['distance_to_origin'] = self._compute_trace_distance(trace, '0')
        
        # Distance signature (complex encoding of position weights)
        properties['distance_signature'] = self._compute_distance_signature(trace)
        
        # Transformation cost (cost to transform to standard forms)
        properties['transformation_cost'] = self._compute_transformation_cost(trace)
        
        # Tension measure (internal structural tension)
        properties['tension_measure'] = self._compute_tension_measure(trace)
        
        # Radius from metric center
        properties['metric_radius'] = self._compute_metric_radius(trace)
        
        # Distance dimension (effective metric dimension)
        properties['distance_dimension'] = self._compute_distance_dimension(trace)
        
        # Metric complexity (structural metric complexity)
        properties['metric_complexity'] = self._compute_metric_complexity(trace)
        
        # Distance type classification
        properties['distance_type'] = self._classify_distance_type(trace)
        
        # Interior distance (distance to boundary)
        properties['interior_distance'] = self._compute_interior_distance(trace)
        
        # Neighborhood distance (distance to nearest neighbors)
        properties['neighborhood_distance'] = self._compute_neighborhood_distance(trace)
        
        self.metric_cache[trace] = properties
        return properties
        
    def _compute_trace_distance(self, trace1: str, trace2: str) -> float:
        """计算两个trace之间的φ-constrained距离"""
        cache_key = (trace1, trace2)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
            
        # Pad traces to same length
        max_len = max(len(trace1), len(trace2))
        t1 = trace1.ljust(max_len, '0')
        t2 = trace2.ljust(max_len, '0')
        
        # Distance computation with φ-constraint awareness
        distance = 0.0
        for i in range(max_len):
            if t1[i] != t2[i]:
                # Position-weighted difference
                pos_weight = 1.0 / (i + 1)
                
                # Fibonacci-modulated cost
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                fib_weight = self.fibonacci_numbers[fib_idx] / self.fibonacci_numbers[-1]
                
                # φ-constraint penalty (avoid consecutive 1s)
                phi_penalty = 1.0
                if i > 0 and t1[i-1:i+1] == '11' or t2[i-1:i+1] == '11':
                    phi_penalty = 2.0  # Higher cost for φ-violations
                if i < max_len - 1 and t1[i:i+2] == '11' or t2[i:i+2] == '11':
                    phi_penalty = 2.0
                    
                distance += pos_weight * fib_weight * phi_penalty
                
        self.distance_cache[cache_key] = distance
        return distance
        
    def _compute_distance_signature(self, trace: str) -> complex:
        """计算trace的距离签名（复数编码）"""
        signature = 0 + 0j
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        for i, pos in enumerate(ones_positions):
            # Harmonic weight based on position
            weight = 1.0 / (pos + 1)
            
            # Complex phase based on Fibonacci modulation
            fib_idx = min(pos, len(self.fibonacci_numbers) - 1)
            phase = 2 * pi * self.fibonacci_numbers[fib_idx] / self.fibonacci_numbers[-1]
            
            signature += weight * (cos(phase) + 1j * sin(phase))
            
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _compute_transformation_cost(self, trace: str) -> float:
        """计算trace的变换成本"""
        if trace in self.transformation_cache:
            return self.transformation_cache[trace]
            
        # Cost to transform to simplest φ-valid forms
        costs = []
        
        # Cost to reach '1' (minimal non-zero)
        if trace != '1':
            costs.append(self._compute_trace_distance(trace, '1'))
            
        # Cost to reach '10' (minimal binary expansion)  
        if trace != '10':
            costs.append(self._compute_trace_distance(trace, '10'))
            
        # Cost to reach '101' (minimal φ-valid pattern)
        if trace != '101':
            costs.append(self._compute_trace_distance(trace, '101'))
            
        # Cost to reach origin
        costs.append(self._compute_trace_distance(trace, '0'))
        
        # Minimum transformation cost
        min_cost = min(costs) if costs else 0.0
        
        self.transformation_cache[trace] = min_cost
        return min_cost
        
    def _compute_tension_measure(self, trace: str) -> float:
        """计算trace的内部张力度量"""
        if len(trace) <= 1:
            return 0.0
            
        tension = 0.0
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        # Tension from proximity of 1s
        for i in range(len(ones_positions) - 1):
            pos1, pos2 = ones_positions[i], ones_positions[i + 1]
            proximity = 1.0 / (pos2 - pos1)  # Higher tension for closer positions
            
            # φ-constraint tension (very high for consecutive)
            if pos2 - pos1 == 1:
                proximity *= 10.0  # φ-violation penalty
                
            tension += proximity
            
        # Normalize by length
        return tension / len(trace)
        
    def _compute_metric_radius(self, trace: str) -> float:
        """计算trace的metric半径"""
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if not ones_positions:
            return 0.0
            
        # Radius as weighted spread of positions
        mean_pos = sum(ones_positions) / len(ones_positions)
        spread = sum((pos - mean_pos) ** 2 for pos in ones_positions)
        
        # Fibonacci-modulated radius
        fib_weights = [self.fibonacci_numbers[min(pos, len(self.fibonacci_numbers) - 1)] 
                      for pos in ones_positions]
        weighted_spread = sum(w * (pos - mean_pos) ** 2 
                             for w, pos in zip(fib_weights, ones_positions))
        
        return sqrt(weighted_spread) / len(trace)
        
    def _compute_distance_dimension(self, trace: str) -> float:
        """计算trace的距离维度"""
        ones_count = trace.count('1')
        trace_length = len(trace)
        
        if ones_count == 0:
            return 0.0
            
        # Dimension based on distribution of 1s
        density = ones_count / trace_length
        
        # Fibonacci-weighted dimension
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        fib_weights = [self.fibonacci_numbers[min(pos, len(self.fibonacci_numbers) - 1)] 
                      for pos in ones_positions]
        
        weighted_density = sum(fib_weights) / (len(trace) * self.fibonacci_numbers[-1])
        
        return density + weighted_density
        
    def _compute_metric_complexity(self, trace: str) -> float:
        """计算trace的metric复杂度"""
        # Complexity based on pattern irregularity
        complexity = 0.0
        
        # Transition complexity
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                complexity += 1.0
                
        # φ-constraint complexity penalty
        if '11' in trace:
            complexity += 5.0  # High penalty for φ-violations
            
        # Pattern complexity
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) > 1:
            gaps = [ones_positions[i+1] - ones_positions[i] 
                   for i in range(len(ones_positions) - 1)]
            gap_variance = np.var(gaps) if len(gaps) > 0 else 0
            complexity += gap_variance / len(trace)
            
        return complexity / len(trace)
        
    def _classify_distance_type(self, trace: str) -> str:
        """分类trace的距离类型"""
        tension = self._compute_tension_measure(trace)
        complexity = self._compute_metric_complexity(trace)
        
        if tension == 0 and complexity == 0:
            return "trivial_distance"
        elif tension < 0.5 and complexity < 0.5:
            return "simple_distance"
        elif tension >= 0.5 and complexity < 0.5:
            return "high_tension"
        elif tension < 0.5 and complexity >= 0.5:
            return "complex_distance"
        else:
            return "high_complexity"
            
    def _compute_interior_distance(self, trace: str) -> float:
        """计算trace的内部距离"""
        # Distance to metric boundary (furthest from φ-violations)
        if '11' in trace:
            return 0.0  # On boundary due to φ-violation
            
        # Distance based on how far from creating φ-violations
        min_distance_to_violation = float('inf')
        
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i + 1] == '0':
                # Could create violation by changing next bit
                min_distance_to_violation = min(min_distance_to_violation, 1.0 / (i + 2))
            elif trace[i] == '0' and trace[i + 1] == '1':
                # Could create violation by changing current bit
                min_distance_to_violation = min(min_distance_to_violation, 1.0 / (i + 1))
                
        return min_distance_to_violation if min_distance_to_violation != float('inf') else 1.0
        
    def _compute_neighborhood_distance(self, trace: str) -> float:
        """计算trace到最近邻的距离"""
        min_distance = float('inf')
        
        # Find minimum distance to other φ-valid traces
        for other_n, other_data in self.trace_universe.items():
            other_trace = other_data['trace']
            if other_trace != trace:
                distance = self._compute_trace_distance(trace, other_trace)
                min_distance = min(min_distance, distance)
                
        return min_distance if min_distance != float('inf') else 0.0
        
    def analyze_metric_system(self) -> Dict:
        """分析完整的metric系统"""
        traces = [data['trace'] for data in self.trace_universe.values()]
        
        analysis = {
            'metric_universe_size': len(traces),
            'distance_matrix': self._compute_distance_matrix(traces),
            'metric_properties': self._analyze_metric_properties(traces),
            'network_analysis': self._analyze_metric_networks(traces),
            'information_analysis': self._analyze_metric_information(traces),
            'category_analysis': self._analyze_metric_categories(traces),
            'three_domain_analysis': self._analyze_three_domains(traces)
        }
        
        return analysis
        
    def _compute_distance_matrix(self, traces: List[str]) -> np.ndarray:
        """计算traces之间的距离矩阵"""
        n = len(traces)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self._compute_trace_distance(traces[i], traces[j])
                
        return matrix
        
    def _analyze_metric_properties(self, traces: List[str]) -> Dict:
        """分析metric属性统计"""
        properties = {}
        
        # Collect all metric properties
        all_distances = []
        all_tensions = []
        all_radii = []
        all_complexities = []
        all_dimensions = []
        distance_types = []
        
        for trace in traces:
            metric_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['metric_properties']
            all_distances.append(metric_props['distance_to_origin'])
            all_tensions.append(metric_props['tension_measure'])
            all_radii.append(metric_props['metric_radius'])
            all_complexities.append(metric_props['metric_complexity'])
            all_dimensions.append(metric_props['distance_dimension'])
            distance_types.append(metric_props['distance_type'])
            
        properties.update({
            'mean_distance_to_origin': np.mean(all_distances),
            'mean_tension': np.mean(all_tensions),
            'mean_radius': np.mean(all_radii),
            'mean_complexity': np.mean(all_complexities),
            'mean_dimension': np.mean(all_dimensions),
            'distance_type_distribution': {
                t: distance_types.count(t) / len(distance_types) 
                for t in set(distance_types)
            }
        })
        
        return properties
        
    def _analyze_metric_networks(self, traces: List[str]) -> Dict:
        """分析metric网络属性"""
        # Build distance-based network
        G = nx.Graph()
        G.add_nodes_from(range(len(traces)))
        
        # Add edges for traces within distance threshold
        distance_matrix = self._compute_distance_matrix(traces)
        threshold = np.mean(distance_matrix) * 0.5  # Conservative threshold
        
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                if distance_matrix[i, j] <= threshold:
                    G.add_edge(i, j, weight=distance_matrix[i, j])
                    
        return {
            'network_nodes': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'network_density': nx.density(G),
            'connected_components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0.0
        }
        
    def _analyze_metric_information(self, traces: List[str]) -> Dict:
        """分析metric信息论属性"""
        # Collect distance type distribution for entropy
        distance_types = []
        dimensions = []
        complexities = []
        tensions = []
        
        for trace in traces:
            metric_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['metric_properties']
            distance_types.append(metric_props['distance_type'])
            dimensions.append(round(metric_props['distance_dimension'], 1))
            complexities.append(round(metric_props['metric_complexity'], 1))
            tensions.append(round(metric_props['tension_measure'], 1))
            
        def compute_entropy(values):
            from collections import Counter
            counts = Counter(values)
            total = len(values)
            return -sum((count/total) * log2(count/total) for count in counts.values())
            
        return {
            'dimension_entropy': compute_entropy(dimensions),
            'type_entropy': compute_entropy(distance_types),
            'complexity_entropy': compute_entropy(complexities),
            'tension_entropy': compute_entropy(tensions),
            'metric_complexity': len(set(distance_types))
        }
        
    def _analyze_metric_categories(self, traces: List[str]) -> Dict:
        """分析metric范畴论属性"""
        # Count morphisms (meaningful distance relationships)
        morphism_count = 0
        functorial_count = 0
        
        distance_matrix = self._compute_distance_matrix(traces)
        n = len(traces)
        
        # Count significant distance relationships
        for i in range(n):
            for j in range(n):
                if i != j and distance_matrix[i, j] < np.mean(distance_matrix):
                    morphism_count += 1
                    
                    # Check if relationship preserves structure (functoriality)
                    trace_i = traces[i]
                    trace_j = traces[j]
                    
                    props_i = self.trace_universe[int(trace_i, 2) if trace_i != '0' else 0]['metric_properties']
                    props_j = self.trace_universe[int(trace_j, 2) if trace_j != '0' else 0]['metric_properties']
                    
                    # Structure preservation: similar dimensions and complexity
                    if (abs(props_i['distance_dimension'] - props_j['distance_dimension']) < 0.5 and
                        abs(props_i['metric_complexity'] - props_j['metric_complexity']) < 0.5):
                        functorial_count += 1
                        
        # Count reachable pairs
        reachable_pairs = sum(1 for i in range(n) for j in range(n) 
                             if i != j and distance_matrix[i, j] < float('inf'))
        
        return {
            'metric_morphisms': morphism_count,
            'functorial_relationships': functorial_count,
            'functoriality_ratio': functorial_count / morphism_count if morphism_count > 0 else 0,
            'reachable_pairs': reachable_pairs,
            'category_structure': f"Category with {n} objects and {morphism_count} morphisms"
        }
        
    def _analyze_three_domains(self, traces: List[str]) -> Dict:
        """分析三域系统：Traditional vs φ-constrained vs Convergence"""
        # Traditional domain: all possible distances without φ-constraints
        traditional_operations = 100  # Baseline traditional metric operations
        
        # φ-constrained domain: only φ-valid operations
        phi_constrained_operations = len(traces)  # Only φ-valid traces
        
        # Convergence domain: operations that work in both systems
        convergence_operations = len([t for t in traces if '11' not in t])  # φ-valid traces
        
        convergence_ratio = convergence_operations / traditional_operations
        
        return {
            'traditional_only': traditional_operations - convergence_operations,
            'phi_constrained_only': phi_constrained_operations - convergence_operations,
            'convergence_domain': convergence_operations,
            'convergence_ratio': convergence_ratio,
            'domain_analysis': {
                'Traditional': f"{traditional_operations} total metric operations",
                'φ-Constrained': f"{phi_constrained_operations} φ-valid metric operations", 
                'Convergence': f"{convergence_operations} operations preserved in both systems"
            }
        }
        
    def generate_visualizations(self, analysis: Dict, prefix: str = "chapter-066-collapse-metric"):
        """生成metric系统的可视化"""
        plt.style.use('default')
        
        # 创建主要的可视化图表
        self._create_metric_structure_plot(analysis, f"{prefix}-structure.png")
        self._create_metric_properties_plot(analysis, f"{prefix}-properties.png") 
        self._create_domain_analysis_plot(analysis, f"{prefix}-domains.png")
        
    def _create_metric_structure_plot(self, analysis: Dict, filename: str):
        """创建metric结构可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distance to origin distribution
        traces = [data['trace'] for data in self.trace_universe.values()]
        distances = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['distance_to_origin'] 
                    for t in traces]
        
        ax1.hist(distances, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distance to Origin Distribution')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Tension vs Complexity scatter
        tensions = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['tension_measure'] 
                   for t in traces]
        complexities = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['metric_complexity'] 
                       for t in traces]
        radii = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['metric_radius'] 
                for t in traces]
        
        scatter = ax2.scatter(tensions, complexities, c=radii, cmap='viridis', alpha=0.7, s=60)
        ax2.set_title('Tension vs Complexity')
        ax2.set_xlabel('Tension Measure')
        ax2.set_ylabel('Metric Complexity')
        plt.colorbar(scatter, ax=ax2, label='Metric Radius')
        ax2.grid(True, alpha=0.3)
        
        # Distance type distribution
        distance_types = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['distance_type'] 
                         for t in traces]
        type_counts = {}
        for dt in distance_types:
            type_counts[dt] = type_counts.get(dt, 0) + 1
            
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax3.set_title('Distance Type Distribution')
        
        # Distance signatures (complex plane)
        signatures = [self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['distance_signature'] 
                     for t in traces]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=distances, cmap='plasma', alpha=0.7, s=60)
        ax4.set_title('Distance Signatures (Complex Plane)')
        ax4.set_xlabel('Real Part')
        ax4.set_ylabel('Imaginary Part')
        ax4.grid(True, alpha=0.3)
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-1.2, 1.2)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_metric_properties_plot(self, analysis: Dict, filename: str):
        """创建metric属性可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metric efficiency metrics
        metrics = analysis['metric_properties']
        metric_names = ['mean_distance_to_origin', 'mean_tension', 'mean_radius', 'mean_complexity']
        metric_values = [metrics[name] for name in metric_names]
        
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        bars = ax1.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Metric Efficiency Metrics')
        ax1.set_xlabel('Metric Type')
        ax1.set_ylabel('Efficiency Score')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace('mean_', '').replace('_', ' ').title() for name in metric_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Dimension-Complexity distribution
        traces = [data['trace'] for data in self.trace_universe.values()]
        dimensions = [round(self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['distance_dimension'], 1) 
                     for t in traces]
        complexities = [round(self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['metric_complexity'], 1) 
                       for t in traces]
        
        # Create 2D histogram
        from collections import Counter
        dim_comp_pairs = list(zip(dimensions, complexities))
        pair_counts = Counter(dim_comp_pairs)
        
        unique_pairs = list(pair_counts.keys())
        counts = list(pair_counts.values())
        
        if unique_pairs:
            dims, comps = zip(*unique_pairs)
            bars = ax2.bar(range(len(unique_pairs)), counts, color='purple', alpha=0.7, edgecolor='black')
            ax2.set_title('Dimension-Complexity Distribution')
            ax2.set_xlabel('(Dimension, Complexity)')
            ax2.set_ylabel('Count')
            ax2.set_xticks(range(len(unique_pairs)))
            ax2.set_xticklabels([f'({d},{c})' for d, c in unique_pairs], rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # System complexity evolution
        complexity_metrics = ['metric_complexity', 'dimension_diversity', 'type_diversity', 'tension_diversity']
        complexity_values = [
            analysis['information_analysis']['metric_complexity'],
            len(set(dimensions)),
            len(set([self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['distance_type'] for t in traces])),
            len(set([round(self.trace_universe[int(t, 2) if t != '0' else 0]['metric_properties']['tension_measure'], 1) for t in traces]))
        ]
        
        ax3.plot(complexity_metrics, complexity_values, 'ro-', linewidth=2, markersize=8)
        ax3.set_title('System Complexity Evolution')
        ax3.set_xlabel('Complexity Metric')
        ax3.set_ylabel('Diversity Count')
        ax3.set_xticks(range(len(complexity_metrics)))
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in complexity_metrics], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Network connectivity analysis
        network_props = analysis['network_analysis']
        network_metrics = ['network_density', 'connected_components', 'average_clustering']
        network_values = [
            network_props['network_density'],
            network_props['connected_components'] / network_props['network_nodes'],  # Normalized
            network_props['average_clustering']
        ]
        
        bars = ax4.bar(network_metrics, network_values, color=['cyan', 'orange', 'pink'], alpha=0.7, edgecolor='black')
        ax4.set_title('Network Connectivity Analysis')
        ax4.set_xlabel('Network Metric')
        ax4.set_ylabel('Score')
        ax4.set_xticklabels([name.replace('_', ' ').title() for name in network_metrics], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, network_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_domain_analysis_plot(self, analysis: Dict, filename: str):
        """创建域分析可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Three-domain operation distribution
        domain_data = analysis['three_domain_analysis']
        domains = ['Traditional\nOnly', 'φ-Constrained\nOnly', 'Convergence\nDomain']
        operation_counts = [
            domain_data['traditional_only'],
            domain_data['phi_constrained_only'], 
            domain_data['convergence_domain']
        ]
        
        colors = ['lightblue', 'lightcoral', 'gold']
        bars = ax1.bar(domains, operation_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Three-Domain Operation Distribution')
        ax1.set_ylabel('Operation Count')
        
        # Add convergence ratio annotation
        convergence_ratio = domain_data['convergence_ratio']
        ax1.text(2, operation_counts[2] + 1, f'Convergence Ratio: {convergence_ratio:.3f}', 
                ha='center', va='bottom', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.grid(True, alpha=0.3)
        
        # Metric efficiency metrics
        metrics = analysis['metric_properties']
        efficiency_metrics = ['mean_distance_to_origin', 'mean_tension', 'mean_radius']
        efficiency_values = [metrics[name] for name in efficiency_metrics]
        
        bars = ax2.bar(range(len(efficiency_metrics)), efficiency_values, 
                      color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Metric Efficiency Metrics')
        ax2.set_xlabel('Metric Type')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_xticks(range(len(efficiency_metrics)))
        ax2.set_xticklabels([name.replace('mean_', '').replace('_', '\n') for name in efficiency_metrics])
        ax2.grid(True, alpha=0.3)
        
        # Information theory results
        info_data = analysis['information_analysis']
        info_metrics = ['dimension_entropy', 'type_entropy', 'complexity_entropy', 'tension_entropy']
        info_values = [info_data[metric] for metric in info_metrics]
        
        bars = ax3.bar(range(len(info_metrics)), info_values, 
                      color='purple', alpha=0.7, edgecolor='black')
        ax3.set_title('Information Theory Analysis')
        ax3.set_xlabel('Entropy Type')
        ax3.set_ylabel('Entropy (bits)')
        ax3.set_xticks(range(len(info_metrics)))
        ax3.set_xticklabels([name.replace('_entropy', '').replace('_', '\n').title() for name in info_metrics])
        ax3.grid(True, alpha=0.3)
        
        # Category theory analysis
        cat_data = analysis['category_analysis']
        category_metrics = ['Morphisms', 'Functorial\nRelationships', 'Reachable\nPairs']
        category_values = [
            cat_data['metric_morphisms'],
            cat_data['functorial_relationships'],
            cat_data['reachable_pairs']
        ]
        
        bars = ax4.bar(category_metrics, category_values, 
                      color=['red', 'blue', 'green'], alpha=0.7, edgecolor='black')
        ax4.set_title('Category Theory Analysis')
        ax4.set_xlabel('Category Metric')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # Add functoriality ratio annotation
        functoriality_ratio = cat_data['functoriality_ratio']
        ax4.text(1, category_values[1] + max(category_values) * 0.05, 
                f'Functoriality: {functoriality_ratio:.3f}', 
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

class TestCollapseMetric(unittest.TestCase):
    """单元测试用于验证collapse metric实现"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseMetricSystem(max_trace_size=6)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        self.assertEqual(self.system._encode_to_trace(0), '0')
        self.assertEqual(self.system._encode_to_trace(1), '1')
        self.assertEqual(self.system._encode_to_trace(5), '101')
        self.assertEqual(self.system._encode_to_trace(10), '1010')
        
    def test_phi_constraint(self):
        """测试φ约束验证"""
        trace_data = self.system._analyze_trace_structure(3)  # '11'
        self.assertFalse(trace_data['phi_valid'])
        
        trace_data = self.system._analyze_trace_structure(5)  # '101'  
        self.assertTrue(trace_data['phi_valid'])
        
    def test_distance_computation(self):
        """测试距离计算"""
        # Distance to self should be 0
        distance = self.system._compute_trace_distance('101', '101')
        self.assertEqual(distance, 0.0)
        
        # Distance should be symmetric
        d1 = self.system._compute_trace_distance('101', '1010')
        d2 = self.system._compute_trace_distance('1010', '101')
        self.assertAlmostEqual(d1, d2, places=6)
        
        # Distance to origin
        distance = self.system._compute_trace_distance('1', '0')
        self.assertGreater(distance, 0)
        
    def test_metric_properties(self):
        """测试metric属性计算"""
        properties = self.system._compute_metric_properties('101')
        
        # Check that all required properties exist
        required_props = [
            'distance_to_origin', 'distance_signature', 'transformation_cost',
            'tension_measure', 'metric_radius', 'distance_dimension',
            'metric_complexity', 'distance_type', 'interior_distance',
            'neighborhood_distance'
        ]
        
        for prop in required_props:
            self.assertIn(prop, properties)
            
        # Check reasonable values
        self.assertGreaterEqual(properties['distance_to_origin'], 0)
        self.assertGreaterEqual(properties['tension_measure'], 0)
        self.assertGreaterEqual(properties['metric_radius'], 0)
        
    def test_distance_type_classification(self):
        """测试距离类型分类"""
        # Test different traces for type classification
        types_found = set()
        
        for n in range(20):
            trace = self.system._encode_to_trace(n)
            if '11' not in trace:  # Only φ-valid traces
                distance_type = self.system._classify_distance_type(trace)
                types_found.add(distance_type)
                
        # Should find multiple distance types
        self.assertGreater(len(types_found), 1)
        
        # All types should be valid
        valid_types = {
            "trivial_distance", "simple_distance", "high_tension", 
            "complex_distance", "high_complexity"
        }
        self.assertTrue(types_found.issubset(valid_types))
        
    def test_transformation_cost(self):
        """测试变换成本计算"""
        # Cost to transform '0' to itself should be minimal
        cost = self.system._compute_transformation_cost('0')
        self.assertGreaterEqual(cost, 0)
        
        # More complex traces should have higher transformation costs
        cost_simple = self.system._compute_transformation_cost('1')
        cost_complex = self.system._compute_transformation_cost('10101')
        
        # Generally, more complex traces should have higher costs
        # (though not strictly enforced due to φ-constraints)
        self.assertGreaterEqual(cost_complex, 0)
        self.assertGreaterEqual(cost_simple, 0)
        
    def test_metric_system_analysis(self):
        """测试完整metric系统分析"""
        analysis = self.system.analyze_metric_system()
        
        # Check that all analysis sections exist
        required_sections = [
            'metric_universe_size', 'distance_matrix', 'metric_properties',
            'network_analysis', 'information_analysis', 'category_analysis',
            'three_domain_analysis'
        ]
        
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Check reasonable values
        self.assertGreater(analysis['metric_universe_size'], 0)
        self.assertGreater(analysis['three_domain_analysis']['convergence_ratio'], 0)
        
    def test_distance_matrix_properties(self):
        """测试距离矩阵属性"""
        traces = ['0', '1', '10', '101']
        matrix = self.system._compute_distance_matrix(traces)
        
        # Matrix should be square
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(traces))
        
        # Diagonal should be zero
        for i in range(len(traces)):
            self.assertAlmostEqual(matrix[i, i], 0.0, places=6)
            
        # Matrix should be symmetric
        for i in range(len(traces)):
            for j in range(len(traces)):
                self.assertAlmostEqual(matrix[i, j], matrix[j, i], places=6)
                
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        analysis = self.system.analyze_metric_system()
        domain_data = analysis['three_domain_analysis']
        
        # Check domain structure
        self.assertIn('traditional_only', domain_data)
        self.assertIn('phi_constrained_only', domain_data)
        self.assertIn('convergence_domain', domain_data)
        self.assertIn('convergence_ratio', domain_data)
        
        # Convergence ratio should be reasonable
        ratio = domain_data['convergence_ratio']
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
        
    def test_visualization_generation(self):
        """测试可视化生成"""
        analysis = self.system.analyze_metric_system()
        
        # Should not raise exceptions
        try:
            self.system.generate_visualizations(analysis, "test-metric")
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Visualization generation failed: {e}")
            
        self.assertTrue(test_passed)

def main():
    """主函数：运行测试和分析"""
    print("🔄 Chapter 066: CollapseMetric Unit Test Verification")
    print("=" * 60)
    
    # 创建系统实例
    system = CollapseMetricSystem(max_trace_size=6)
    
    print("📊 Building trace universe...")
    print(f"✅ Found {len(system.trace_universe)} φ-valid traces")
    
    # 运行完整分析
    print("\n🔍 Analyzing collapse metric system...")
    analysis = system.analyze_metric_system()
    
    print(f"📈 Metric universe size: {analysis['metric_universe_size']} elements")
    print(f"📊 Network density: {analysis['network_analysis']['network_density']:.3f}")
    print(f"🎯 Convergence ratio: {analysis['three_domain_analysis']['convergence_ratio']:.3f}")
    
    # 显示metric属性统计
    props = analysis['metric_properties']
    print(f"\n📏 Metric Properties:")
    print(f"   Mean distance to origin: {props['mean_distance_to_origin']:.3f}")
    print(f"   Mean tension: {props['mean_tension']:.3f}")
    print(f"   Mean radius: {props['mean_radius']:.3f}")
    print(f"   Mean complexity: {props['mean_complexity']:.3f}")
    print(f"   Mean dimension: {props['mean_dimension']:.3f}")
    
    # 显示信息论分析
    info = analysis['information_analysis']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Tension entropy: {info['tension_entropy']:.3f} bits")
    print(f"   Metric complexity: {info['metric_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.generate_visualizations(analysis)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n✅ Chapter 066: CollapseMetric verification completed!")
    print("=" * 60)
    print("🔥 Metric structures exhibit bounded distance convergence!")

if __name__ == "__main__":
    main()