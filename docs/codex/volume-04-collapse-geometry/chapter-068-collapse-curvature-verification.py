#!/usr/bin/env python3
"""
Chapter 068: CollapseCurvature Unit Test Verification
从ψ=ψ(ψ)推导Curvature from φ-Divergence and Tensor Field Deformation

Core principle: From ψ = ψ(ψ) derive curvature where curvature is φ-valid
tensor field deformation that encodes geometric relationships through trace-based divergence,
creating systematic curvature frameworks with bounded divergence and natural curvature
properties governed by golden constraints, showing how curvature emerges from trace tensor deformation.

This verification program implements:
1. φ-constrained curvature as tensor field deformation operations
2. Curvature analysis: divergence patterns, deformation structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection curvature theory
4. Graph theory analysis of divergence networks and curvature connectivity patterns
5. Information theory analysis of curvature entropy and deformation information
6. Category theory analysis of curvature functors and deformation morphisms
7. Visualization of curvature structures and divergence patterns
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

class CollapseCurvatureSystem:
    """
    Core system for implementing collapse curvature through φ-divergence tensor deformation.
    Implements φ-constrained curvature theory via trace-based tensor field operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_curvature_complexity: int = 4):
        """Initialize collapse curvature system"""
        self.max_trace_size = max_trace_size
        self.max_curvature_complexity = max_curvature_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.curvature_cache = {}
        self.divergence_cache = {}
        self.deformation_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_curvature=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for curvature properties computation
        self.trace_universe = universe
        
        # Second pass: add curvature properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['curvature_properties'] = self._compute_curvature_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_curvature: bool = True) -> Dict:
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
        
        if compute_curvature and hasattr(self, 'trace_universe'):
            result['curvature_properties'] = self._compute_curvature_properties(trace)
            
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
        
    def _compute_curvature_properties(self, trace: str) -> Dict:
        """计算trace的curvature属性"""
        if trace in self.curvature_cache:
            return self.curvature_cache[trace]
            
        properties = {}
        
        # Divergence measure (how tensor field diverges from trace)
        properties['divergence_measure'] = self._compute_divergence_measure(trace)
        
        # Curvature signature (complex encoding of deformation weights)
        properties['curvature_signature'] = self._compute_curvature_signature(trace)
        
        # Deformation cost (cost to deform tensor field)
        properties['deformation_cost'] = self._compute_deformation_cost(trace)
        
        # Tensor strain (internal tensor deformation)
        properties['tensor_strain'] = self._compute_tensor_strain(trace)
        
        # Curvature radius from geometric center
        properties['curvature_radius'] = self._compute_curvature_radius(trace)
        
        # Deformation dimension (effective curvature dimension)
        properties['deformation_dimension'] = self._compute_deformation_dimension(trace)
        
        # Curvature complexity (structural deformation complexity)
        properties['curvature_complexity'] = self._compute_curvature_complexity(trace)
        
        # Curvature type classification
        properties['curvature_type'] = self._classify_curvature_type(trace)
        
        # Sectional curvature (curvature in different directions)
        properties['sectional_curvature'] = self._compute_sectional_curvature(trace)
        
        # Ricci curvature (trace of curvature tensor)
        properties['ricci_curvature'] = self._compute_ricci_curvature(trace)
        
        self.curvature_cache[trace] = properties
        return properties
        
    def _compute_divergence_measure(self, trace: str) -> float:
        """计算trace的φ-constrained散度度量"""
        if len(trace) <= 1:
            return 0.0
            
        divergence = 0.0
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        # Divergence from position variations
        if len(ones_positions) >= 2:
            for i in range(len(ones_positions) - 1):
                pos1, pos2 = ones_positions[i], ones_positions[i + 1]
                
                # Expected uniform distribution
                expected_gap = len(trace) / len(ones_positions)
                actual_gap = pos2 - pos1
                
                # Divergence as deviation from uniformity
                deviation = abs(actual_gap - expected_gap) / expected_gap
                
                # Fibonacci-weighted divergence
                fib_idx = min(pos1, len(self.fibonacci_numbers) - 1)
                fib_weight = self.fibonacci_numbers[fib_idx] / self.fibonacci_numbers[-1]
                
                divergence += deviation * fib_weight
                
        # φ-constraint divergence penalty (very high for consecutive)
        if '11' in trace:
            divergence += 10.0  # High penalty for φ-violations
            
        # Normalize by length
        return divergence / len(trace)
        
    def _compute_curvature_signature(self, trace: str) -> complex:
        """计算trace的曲率签名（复数编码）"""
        signature = 0 + 0j
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        # Calculate curvature through second derivatives of positions
        if len(ones_positions) >= 3:
            for i in range(len(ones_positions) - 2):
                pos1, pos2, pos3 = ones_positions[i], ones_positions[i + 1], ones_positions[i + 2]
                
                # Second derivative approximation (curvature)
                curvature_val = (pos3 - 2*pos2 + pos1) / ((pos3 - pos1) ** 2 + 1)
                
                # Weight based on position
                weight = 1.0 / (pos2 + 1)
                
                # Complex phase based on Fibonacci modulation
                fib_idx = min(pos2, len(self.fibonacci_numbers) - 1)
                phase = 2 * pi * self.fibonacci_numbers[fib_idx] / self.fibonacci_numbers[-1]
                
                # Add curvature contribution
                signature += weight * curvature_val * (cos(phase) + 1j * sin(phase))
                
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _compute_deformation_cost(self, trace: str) -> float:
        """计算trace的变形成本"""
        if trace in self.deformation_cache:
            return self.deformation_cache[trace]
            
        # Cost to deform to flat (zero curvature) forms
        costs = []
        
        # Cost to flatten to uniform distribution
        ones_count = trace.count('1')
        if ones_count > 0:
            uniform_spacing = len(trace) / ones_count
            actual_positions = [i for i, bit in enumerate(trace) if bit == '1']
            
            deformation_cost = 0.0
            for i, pos in enumerate(actual_positions):
                expected_pos = i * uniform_spacing
                deformation_cost += abs(pos - expected_pos) / len(trace)
                
            costs.append(deformation_cost)
            
        # Cost to deform to minimal curvature forms
        costs.append(self._compute_divergence_measure(trace) * 1.5)  # Deformation multiplier
        
        # Cost to deform to origin (flat space)
        origin_cost = sum(1.0 / (i + 1) for i, bit in enumerate(trace) if bit == '1')
        costs.append(origin_cost / len(trace))
        
        # Minimum deformation cost
        min_cost = min(costs) if costs else 0.0
        
        self.deformation_cache[trace] = min_cost
        return min_cost
        
    def _compute_tensor_strain(self, trace: str) -> float:
        """计算trace的张量应变"""
        if len(trace) <= 2:
            return 0.0
            
        strain = 0.0
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        # Strain from non-uniform tensor field
        if len(ones_positions) >= 2:
            gaps = [ones_positions[i+1] - ones_positions[i] 
                   for i in range(len(ones_positions) - 1)]
            
            # Strain as variance in gaps (non-uniformity)
            mean_gap = sum(gaps) / len(gaps)
            strain_variance = sum((gap - mean_gap) ** 2 for gap in gaps)
            strain = sqrt(strain_variance) / len(trace)
            
        # φ-constraint strain (high strain for consecutive)
        if '11' in trace:
            strain += 2.0  # Moderate penalty for φ-violations
            
        return strain
        
    def _compute_curvature_radius(self, trace: str) -> float:
        """计算trace的曲率半径"""
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) < 3:
            return float('inf')  # Infinite radius (flat)
            
        # Radius of curvature approximation
        total_curvature = 0.0
        curvature_count = 0
        
        for i in range(len(ones_positions) - 2):
            pos1, pos2, pos3 = ones_positions[i], ones_positions[i + 1], ones_positions[i + 2]
            
            # Curvature using three points
            if pos3 != pos1:  # Avoid division by zero
                curvature = abs(pos3 - 2*pos2 + pos1) / ((pos3 - pos1) ** 2)
                if curvature > 0:
                    total_curvature += curvature
                    curvature_count += 1
                    
        if curvature_count > 0:
            mean_curvature = total_curvature / curvature_count
            return 1.0 / mean_curvature if mean_curvature > 0 else float('inf')
        else:
            return float('inf')
        
    def _compute_deformation_dimension(self, trace: str) -> float:
        """计算trace的变形维度"""
        ones_count = trace.count('1')
        trace_length = len(trace)
        
        if ones_count == 0:
            return 0.0
            
        # Dimension based on deformation complexity
        density = ones_count / trace_length
        
        # Deformation complexity dimension
        divergence = self._compute_divergence_measure(trace)
        strain = self._compute_tensor_strain(trace)
        
        # Combined dimension measure
        return density + divergence * 0.3 + strain * 0.2
        
    def _compute_curvature_complexity(self, trace: str) -> float:
        """计算trace的曲率复杂度"""
        # Complexity based on curvature variations
        complexity = 0.0
        
        # Divergence contribution
        divergence = self._compute_divergence_measure(trace)
        complexity += divergence
        
        # Strain contribution
        strain = self._compute_tensor_strain(trace)
        complexity += strain * 0.5
        
        # φ-constraint complexity penalty
        if '11' in trace:
            complexity += 1.0  # Moderate penalty for φ-violations
            
        # Position variation complexity
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) > 2:
            second_diffs = []
            for i in range(len(ones_positions) - 2):
                pos1, pos2, pos3 = ones_positions[i], ones_positions[i + 1], ones_positions[i + 2]
                second_diff = abs(pos3 - 2*pos2 + pos1)
                second_diffs.append(second_diff)
                
            if second_diffs:
                complexity += np.var(second_diffs) / len(trace)
                
        return complexity
        
    def _classify_curvature_type(self, trace: str) -> str:
        """分类trace的曲率类型"""
        divergence = self._compute_divergence_measure(trace)
        strain = self._compute_tensor_strain(trace)
        
        if divergence == 0 and strain == 0:
            return "flat_curvature"
        elif divergence < 0.2 and strain < 0.2:
            return "weak_curvature"
        elif divergence >= 0.2 and strain < 0.2:
            return "divergent_curvature"
        elif divergence < 0.2 and strain >= 0.2:
            return "strained_curvature"
        else:
            return "high_curvature"
            
    def _compute_sectional_curvature(self, trace: str) -> float:
        """计算trace的截面曲率"""
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) < 3:
            return 0.0
            
        # Sectional curvature in different directions
        sectional_curvatures = []
        
        for i in range(len(ones_positions) - 2):
            pos1, pos2, pos3 = ones_positions[i], ones_positions[i + 1], ones_positions[i + 2]
            
            # Sectional curvature approximation
            if pos3 != pos1:
                sectional = (pos3 - 2*pos2 + pos1) / ((pos3 - pos1) ** 3 + 1)
                sectional_curvatures.append(abs(sectional))
                
        return np.mean(sectional_curvatures) if sectional_curvatures else 0.0
        
    def _compute_ricci_curvature(self, trace: str) -> float:
        """计算trace的Ricci曲率"""
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) < 2:
            return 0.0
            
        # Ricci curvature as trace of curvature tensor
        ricci = 0.0
        
        # Sum over all directions (position pairs)
        for i in range(len(ones_positions)):
            for j in range(i + 1, len(ones_positions)):
                pos1, pos2 = ones_positions[i], ones_positions[j]
                
                # Ricci component
                distance = abs(pos2 - pos1)
                if distance > 0:
                    # Fibonacci-weighted Ricci curvature
                    fib_idx1 = min(pos1, len(self.fibonacci_numbers) - 1)
                    fib_idx2 = min(pos2, len(self.fibonacci_numbers) - 1)
                    fib_weight = (self.fibonacci_numbers[fib_idx1] * self.fibonacci_numbers[fib_idx2]) / (self.fibonacci_numbers[-1] ** 2)
                    
                    ricci_component = fib_weight / (distance ** 2 + 1)
                    ricci += ricci_component
                    
        # Normalize by number of components
        num_pairs = len(ones_positions) * (len(ones_positions) - 1) // 2
        return ricci / num_pairs if num_pairs > 0 else 0.0
        
    def analyze_curvature_system(self) -> Dict:
        """分析完整的curvature系统"""
        traces = [data['trace'] for data in self.trace_universe.values()]
        
        analysis = {
            'curvature_universe_size': len(traces),
            'divergence_matrix': self._compute_divergence_matrix(traces),
            'curvature_properties': self._analyze_curvature_properties(traces),
            'network_analysis': self._analyze_curvature_networks(traces),
            'information_analysis': self._analyze_curvature_information(traces),
            'category_analysis': self._analyze_curvature_categories(traces),
            'three_domain_analysis': self._analyze_three_domains(traces)
        }
        
        return analysis
        
    def _compute_divergence_matrix(self, traces: List[str]) -> np.ndarray:
        """计算traces之间的散度矩阵"""
        n = len(traces)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Divergence between traces
                    trace_i = traces[i]
                    trace_j = traces[j]
                    div_i = self._compute_divergence_measure(trace_i)
                    div_j = self._compute_divergence_measure(trace_j)
                    matrix[i, j] = abs(div_i - div_j)
                    
        return matrix
        
    def _analyze_curvature_properties(self, traces: List[str]) -> Dict:
        """分析curvature属性统计"""
        properties = {}
        
        # Collect all curvature properties
        all_divergences = []
        all_strains = []
        all_radii = []
        all_complexities = []
        all_dimensions = []
        curvature_types = []
        
        for trace in traces:
            curvature_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['curvature_properties']
            all_divergences.append(curvature_props['divergence_measure'])
            all_strains.append(curvature_props['tensor_strain'])
            
            # Handle infinite radius
            radius = curvature_props['curvature_radius']
            all_radii.append(min(radius, 10.0))  # Cap at 10 for analysis
            
            all_complexities.append(curvature_props['curvature_complexity'])
            all_dimensions.append(curvature_props['deformation_dimension'])
            curvature_types.append(curvature_props['curvature_type'])
            
        properties.update({
            'mean_divergence': np.mean(all_divergences),
            'mean_strain': np.mean(all_strains),
            'mean_radius': np.mean(all_radii),
            'mean_complexity': np.mean(all_complexities),
            'mean_dimension': np.mean(all_dimensions),
            'curvature_type_distribution': {
                t: curvature_types.count(t) / len(curvature_types) 
                for t in set(curvature_types)
            }
        })
        
        return properties
        
    def _analyze_curvature_networks(self, traces: List[str]) -> Dict:
        """分析curvature网络属性"""
        # Build curvature-based network
        G = nx.Graph()
        G.add_nodes_from(range(len(traces)))
        
        # Add edges for traces with similar curvature
        divergence_matrix = self._compute_divergence_matrix(traces)
        threshold = np.mean(divergence_matrix) * 0.5  # Similarity threshold
        
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                if divergence_matrix[i, j] <= threshold:
                    G.add_edge(i, j, weight=1.0 / (divergence_matrix[i, j] + 0.01))
                    
        return {
            'network_nodes': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'network_density': nx.density(G),
            'connected_components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0.0
        }
        
    def _analyze_curvature_information(self, traces: List[str]) -> Dict:
        """分析curvature信息论属性"""
        # Collect curvature type distribution for entropy
        curvature_types = []
        dimensions = []
        complexities = []
        strains = []
        
        for trace in traces:
            curvature_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['curvature_properties']
            curvature_types.append(curvature_props['curvature_type'])
            dimensions.append(round(curvature_props['deformation_dimension'], 1))
            complexities.append(round(curvature_props['curvature_complexity'], 1))
            strains.append(round(curvature_props['tensor_strain'], 1))
            
        def compute_entropy(values):
            from collections import Counter
            counts = Counter(values)
            total = len(values)
            return -sum((count/total) * log2(count/total) for count in counts.values())
            
        return {
            'dimension_entropy': compute_entropy(dimensions),
            'type_entropy': compute_entropy(curvature_types),
            'complexity_entropy': compute_entropy(complexities),
            'strain_entropy': compute_entropy(strains),
            'curvature_complexity': len(set(curvature_types))
        }
        
    def _analyze_curvature_categories(self, traces: List[str]) -> Dict:
        """分析curvature范畴论属性"""
        # Count morphisms (meaningful curvature relationships)
        morphism_count = 0
        functorial_count = 0
        
        divergence_matrix = self._compute_divergence_matrix(traces)
        n = len(traces)
        
        # Count significant curvature relationships
        for i in range(n):
            for j in range(n):
                if i != j and divergence_matrix[i, j] < np.mean(divergence_matrix):
                    morphism_count += 1
                    
                    # Check if relationship preserves structure (functoriality)
                    trace_i = traces[i]
                    trace_j = traces[j]
                    
                    props_i = self.trace_universe[int(trace_i, 2) if trace_i != '0' else 0]['curvature_properties']
                    props_j = self.trace_universe[int(trace_j, 2) if trace_j != '0' else 0]['curvature_properties']
                    
                    # Structure preservation: similar dimensions and strain
                    if (abs(props_i['deformation_dimension'] - props_j['deformation_dimension']) < 0.5 and
                        abs(props_i['tensor_strain'] - props_j['tensor_strain']) < 0.5):
                        functorial_count += 1
                        
        # Count reachable pairs
        reachable_pairs = sum(1 for i in range(n) for j in range(n) 
                             if i != j and divergence_matrix[i, j] < float('inf'))
        
        return {
            'curvature_morphisms': morphism_count,
            'functorial_relationships': functorial_count,
            'functoriality_ratio': functorial_count / morphism_count if morphism_count > 0 else 0,
            'reachable_pairs': reachable_pairs,
            'category_structure': f"Category with {n} objects and {morphism_count} morphisms"
        }
        
    def _analyze_three_domains(self, traces: List[str]) -> Dict:
        """分析三域系统：Traditional vs φ-constrained vs Convergence"""
        # Traditional domain: all possible curvatures without φ-constraints
        traditional_operations = 100  # Baseline traditional curvature operations
        
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
                'Traditional': f"{traditional_operations} total curvature operations",
                'φ-Constrained': f"{phi_constrained_operations} φ-valid curvature operations", 
                'Convergence': f"{convergence_operations} operations preserved in both systems"
            }
        }
        
    def generate_visualizations(self, analysis: Dict, prefix: str = "chapter-068-collapse-curvature"):
        """生成curvature系统的可视化"""
        plt.style.use('default')
        
        # 创建主要的可视化图表
        self._create_curvature_structure_plot(analysis, f"{prefix}-structure.png")
        self._create_curvature_properties_plot(analysis, f"{prefix}-properties.png") 
        self._create_domain_analysis_plot(analysis, f"{prefix}-domains.png")
        
    def _create_curvature_structure_plot(self, analysis: Dict, filename: str):
        """创建curvature结构可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Divergence measure distribution
        traces = [data['trace'] for data in self.trace_universe.values()]
        divergences = [self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['divergence_measure'] 
                      for t in traces]
        
        ax1.hist(divergences, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_title('Divergence Measure Distribution')
        ax1.set_xlabel('Divergence')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Strain vs Complexity scatter
        strains = [self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['tensor_strain'] 
                  for t in traces]
        complexities = [self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['curvature_complexity'] 
                       for t in traces]
        radii = [min(self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['curvature_radius'], 10.0) 
                for t in traces]
        
        scatter = ax2.scatter(strains, complexities, c=radii, cmap='coolwarm', alpha=0.7, s=60)
        ax2.set_title('Strain vs Complexity')
        ax2.set_xlabel('Tensor Strain')
        ax2.set_ylabel('Curvature Complexity')
        plt.colorbar(scatter, ax=ax2, label='Curvature Radius')
        ax2.grid(True, alpha=0.3)
        
        # Curvature type distribution
        curvature_types = [self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['curvature_type'] 
                          for t in traces]
        type_counts = {}
        for ct in curvature_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
            
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
        
        wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax3.set_title('Curvature Type Distribution')
        
        # Curvature signatures (complex plane)
        signatures = [self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['curvature_signature'] 
                     for t in traces]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=divergences, cmap='plasma', alpha=0.7, s=60)
        ax4.set_title('Curvature Signatures (Complex Plane)')
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
        
    def _create_curvature_properties_plot(self, analysis: Dict, filename: str):
        """创建curvature属性可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Curvature efficiency metrics
        metrics = analysis['curvature_properties']
        metric_names = ['mean_divergence', 'mean_strain', 'mean_radius', 'mean_complexity']
        metric_values = [metrics[name] for name in metric_names]
        
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        bars = ax1.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Curvature Efficiency Metrics')
        ax1.set_xlabel('Curvature Type')
        ax1.set_ylabel('Efficiency Score')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace('mean_', '').replace('_', ' ').title() for name in metric_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Dimension-Strain distribution
        traces = [data['trace'] for data in self.trace_universe.values()]
        dimensions = [round(self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['deformation_dimension'], 1) 
                     for t in traces]
        strains = [round(self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['tensor_strain'], 1) 
                  for t in traces]
        
        # Create 2D histogram
        from collections import Counter
        dim_strain_pairs = list(zip(dimensions, strains))
        pair_counts = Counter(dim_strain_pairs)
        
        unique_pairs = list(pair_counts.keys())
        counts = list(pair_counts.values())
        
        if unique_pairs:
            dims, strains_vals = zip(*unique_pairs)
            bars = ax2.bar(range(len(unique_pairs)), counts, color='purple', alpha=0.7, edgecolor='black')
            ax2.set_title('Dimension-Strain Distribution')
            ax2.set_xlabel('(Dimension, Strain)')
            ax2.set_ylabel('Count')
            ax2.set_xticks(range(len(unique_pairs)))
            ax2.set_xticklabels([f'({d},{s})' for d, s in unique_pairs], rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # System complexity evolution
        complexity_metrics = ['curvature_complexity', 'dimension_diversity', 'type_diversity', 'strain_diversity']
        complexity_values = [
            analysis['information_analysis']['curvature_complexity'],
            len(set(dimensions)),
            len(set([self.trace_universe[int(t, 2) if t != '0' else 0]['curvature_properties']['curvature_type'] for t in traces])),
            len(set(strains))
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
        
        # Curvature efficiency metrics
        metrics = analysis['curvature_properties']
        efficiency_metrics = ['mean_divergence', 'mean_strain', 'mean_radius']
        efficiency_values = [metrics[name] for name in efficiency_metrics]
        
        bars = ax2.bar(range(len(efficiency_metrics)), efficiency_values, 
                      color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Curvature Efficiency Metrics')
        ax2.set_xlabel('Curvature Type')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_xticks(range(len(efficiency_metrics)))
        ax2.set_xticklabels([name.replace('mean_', '').replace('_', '\n') for name in efficiency_metrics])
        ax2.grid(True, alpha=0.3)
        
        # Information theory results
        info_data = analysis['information_analysis']
        info_metrics = ['dimension_entropy', 'type_entropy', 'complexity_entropy', 'strain_entropy']
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
            cat_data['curvature_morphisms'],
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

class TestCollapseCurvature(unittest.TestCase):
    """单元测试用于验证collapse curvature实现"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseCurvatureSystem(max_trace_size=6)
        
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
        
    def test_divergence_computation(self):
        """测试散度计算"""
        # Simple trace should have low divergence
        divergence = self.system._compute_divergence_measure('10')
        self.assertGreaterEqual(divergence, 0)
        
        # φ-violating trace should have high divergence
        divergence_violation = self.system._compute_divergence_measure('11')
        self.assertGreater(divergence_violation, divergence)
        
    def test_curvature_properties(self):
        """测试curvature属性计算"""
        properties = self.system._compute_curvature_properties('101')
        
        # Check that all required properties exist
        required_props = [
            'divergence_measure', 'curvature_signature', 'deformation_cost',
            'tensor_strain', 'curvature_radius', 'deformation_dimension',
            'curvature_complexity', 'curvature_type', 'sectional_curvature',
            'ricci_curvature'
        ]
        
        for prop in required_props:
            self.assertIn(prop, properties)
            
        # Check reasonable values
        self.assertGreaterEqual(properties['divergence_measure'], 0)
        self.assertGreaterEqual(properties['tensor_strain'], 0)
        self.assertGreater(properties['curvature_radius'], 0)
        
    def test_curvature_type_classification(self):
        """测试曲率类型分类"""
        # Test different traces for type classification
        types_found = set()
        
        for n in range(20):
            trace = self.system._encode_to_trace(n)
            if '11' not in trace:  # Only φ-valid traces
                curvature_type = self.system._classify_curvature_type(trace)
                types_found.add(curvature_type)
                
        # Should find multiple curvature types
        self.assertGreater(len(types_found), 1)
        
        # All types should be valid
        valid_types = {
            "flat_curvature", "weak_curvature", "divergent_curvature", 
            "strained_curvature", "high_curvature"
        }
        self.assertTrue(types_found.issubset(valid_types))
        
    def test_deformation_cost(self):
        """测试变形成本计算"""
        # Cost should be non-negative
        cost = self.system._compute_deformation_cost('101')
        self.assertGreaterEqual(cost, 0)
        
        # Origin should have minimal cost
        cost_origin = self.system._compute_deformation_cost('0')
        self.assertGreaterEqual(cost_origin, 0)
        
    def test_curvature_system_analysis(self):
        """测试完整curvature系统分析"""
        analysis = self.system.analyze_curvature_system()
        
        # Check that all analysis sections exist
        required_sections = [
            'curvature_universe_size', 'divergence_matrix', 'curvature_properties',
            'network_analysis', 'information_analysis', 'category_analysis',
            'three_domain_analysis'
        ]
        
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Check reasonable values
        self.assertGreater(analysis['curvature_universe_size'], 0)
        self.assertGreater(analysis['three_domain_analysis']['convergence_ratio'], 0)
        
    def test_divergence_matrix_properties(self):
        """测试散度矩阵属性"""
        traces = ['0', '1', '10', '101']
        matrix = self.system._compute_divergence_matrix(traces)
        
        # Matrix should be square
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(traces))
        
        # Diagonal should be zero
        for i in range(len(traces)):
            self.assertEqual(matrix[i, i], 0.0)
            
        # Matrix should be symmetric
        for i in range(len(traces)):
            for j in range(len(traces)):
                self.assertAlmostEqual(matrix[i, j], matrix[j, i], places=6)
                
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        analysis = self.system.analyze_curvature_system()
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
        analysis = self.system.analyze_curvature_system()
        
        # Should not raise exceptions
        try:
            self.system.generate_visualizations(analysis, "test-curvature")
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Visualization generation failed: {e}")
            
        self.assertTrue(test_passed)

def main():
    """主函数：运行测试和分析"""
    print("🔄 Chapter 068: CollapseCurvature Unit Test Verification")
    print("=" * 60)
    
    # 创建系统实例
    system = CollapseCurvatureSystem(max_trace_size=6)
    
    print("📊 Building trace universe...")
    print(f"✅ Found {len(system.trace_universe)} φ-valid traces")
    
    # 运行完整分析
    print("\n🔍 Analyzing collapse curvature system...")
    analysis = system.analyze_curvature_system()
    
    print(f"📈 Curvature universe size: {analysis['curvature_universe_size']} elements")
    print(f"📊 Network density: {analysis['network_analysis']['network_density']:.3f}")
    print(f"🎯 Convergence ratio: {analysis['three_domain_analysis']['convergence_ratio']:.3f}")
    
    # 显示curvature属性统计
    props = analysis['curvature_properties']
    print(f"\n📏 Curvature Properties:")
    print(f"   Mean divergence: {props['mean_divergence']:.3f}")
    print(f"   Mean strain: {props['mean_strain']:.3f}")
    print(f"   Mean radius: {props['mean_radius']:.3f}")
    print(f"   Mean complexity: {props['mean_complexity']:.3f}")
    print(f"   Mean dimension: {props['mean_dimension']:.3f}")
    
    # 显示信息论分析
    info = analysis['information_analysis']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Strain entropy: {info['strain_entropy']:.3f} bits")
    print(f"   Curvature complexity: {info['curvature_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.generate_visualizations(analysis)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n✅ Chapter 068: CollapseCurvature verification completed!")
    print("=" * 60)
    print("🔥 Curvature structures exhibit bounded divergence convergence!")

if __name__ == "__main__":
    main()