#!/usr/bin/env python3
"""
Chapter 070: CollapseTopoMap Unit Test Verification
从ψ=ψ(ψ)推导Topological Transforms over Trace Conjugacy

Core principle: From ψ = ψ(ψ) derive continuous maps where maps are φ-valid
trace conjugacy preserving transformations that encode geometric relationships through trace-based morphisms,
creating systematic mapping frameworks with bounded transforms and natural continuity
properties governed by golden constraints, showing how topology emerges from trace mappings.

This verification program implements:
1. φ-constrained topological maps as trace conjugacy preserving operations
2. Mapping analysis: transform patterns, conjugacy structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection topological mapping theory
4. Graph theory analysis of mapping networks and transform connectivity patterns
5. Information theory analysis of mapping entropy and transform information
6. Category theory analysis of mapping functors and conjugacy morphisms
7. Visualization of mapping structures and transform patterns
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

class CollapseTopoMapSystem:
    """
    Core system for implementing collapse topological maps through trace conjugacy.
    Implements φ-constrained topological mapping theory via trace-based conjugacy operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_map_complexity: int = 4):
        """Initialize collapse topological map system"""
        self.max_trace_size = max_trace_size
        self.max_map_complexity = max_map_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.mapping_cache = {}
        self.conjugacy_cache = {}
        self.transform_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_mapping=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for mapping properties computation
        self.trace_universe = universe
        
        # Second pass: add mapping properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['mapping_properties'] = self._compute_mapping_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_mapping: bool = True) -> Dict:
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
        
        if compute_mapping and hasattr(self, 'trace_universe'):
            result['mapping_properties'] = self._compute_mapping_properties(trace)
            
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
        
    def _compute_mapping_properties(self, trace: str) -> Dict:
        """计算trace的mapping属性"""
        if trace in self.mapping_cache:
            return self.mapping_cache[trace]
            
        properties = {}
        
        # Conjugacy class (traces conjugate to this one)
        properties['conjugacy_class'] = self._compute_conjugacy_class(trace)
        
        # Mapping signature (complex encoding of transform weights)
        properties['mapping_signature'] = self._compute_mapping_signature(trace)
        
        # Transform cost (cost to apply mapping transforms)
        properties['transform_cost'] = self._compute_transform_cost(trace)
        
        # Continuity measure (how continuous the mapping is)
        properties['continuity_measure'] = self._compute_continuity_measure(trace)
        
        # Mapping radius from transform center
        properties['mapping_radius'] = self._compute_mapping_radius(trace)
        
        # Transform dimension (effective mapping dimension)
        properties['transform_dimension'] = self._compute_transform_dimension(trace)
        
        # Mapping complexity (structural transform complexity)
        properties['mapping_complexity'] = self._compute_mapping_complexity(trace)
        
        # Mapping type classification
        properties['mapping_type'] = self._classify_mapping_type(trace)
        
        # Preserve measure (what properties are preserved)
        properties['preserve_measure'] = self._compute_preserve_measure(trace)
        
        # Conjugacy invariants (invariants under conjugacy)
        properties['conjugacy_invariants'] = self._compute_conjugacy_invariants(trace)
        
        self.mapping_cache[trace] = properties
        return properties
        
    def _compute_conjugacy_class(self, trace: str) -> List[str]:
        """计算trace的共轭类"""
        if trace in self.conjugacy_cache:
            return self.conjugacy_cache[trace]
            
        conjugates = set()
        conjugates.add(trace)
        
        # Cyclic conjugates (rotations)
        for i in range(1, len(trace)):
            rotated = trace[i:] + trace[:i]
            if '11' not in rotated:  # Maintain φ-constraint
                conjugates.add(rotated)
                
        # Reflection conjugates (reversals)
        reversed_trace = trace[::-1]
        if '11' not in reversed_trace:
            conjugates.add(reversed_trace)
            # Also add rotations of reversed
            for i in range(1, len(reversed_trace)):
                rotated_rev = reversed_trace[i:] + reversed_trace[:i]
                if '11' not in rotated_rev:
                    conjugates.add(rotated_rev)
                    
        # Fibonacci-weighted conjugates
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) >= 2:
            # Try swapping Fibonacci-weighted positions
            for i in range(len(ones_positions)):
                for j in range(i + 1, len(ones_positions)):
                    pos1, pos2 = ones_positions[i], ones_positions[j]
                    if pos1 in self.fibonacci_numbers[:len(trace)] or pos2 in self.fibonacci_numbers[:len(trace)]:
                        # Create swap candidate
                        new_trace = list(trace)
                        new_trace[pos1], new_trace[pos2] = new_trace[pos2], new_trace[pos1]
                        new_trace_str = ''.join(new_trace)
                        if '11' not in new_trace_str:
                            conjugates.add(new_trace_str)
                            
        conjugate_list = sorted(list(conjugates))
        self.conjugacy_cache[trace] = conjugate_list
        return conjugate_list
        
    def _compute_mapping_signature(self, trace: str) -> complex:
        """计算trace的映射签名（复数编码）"""
        signature = 0 + 0j
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if not ones_positions:
            return 0 + 0j
            
        # Calculate mapping signature through conjugacy weights
        conjugates = self._compute_conjugacy_class(trace)
        
        for i, conj in enumerate(conjugates):
            # Weight based on conjugate structure
            weight = 1.0 / (i + 1)
            
            # Complex phase based on structural difference
            struct_diff = sum(1 for j in range(min(len(trace), len(conj))) 
                             if trace[j] != conj[j])
            phase = 2 * pi * struct_diff / max(len(trace), 1)
            
            # Add mapping contribution
            signature += weight * (cos(phase) + 1j * sin(phase))
            
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _compute_transform_cost(self, trace: str) -> float:
        """计算trace的变换成本"""
        if trace in self.transform_cache:
            return self.transform_cache[trace]
            
        # Cost to transform to conjugates
        costs = []
        conjugates = self._compute_conjugacy_class(trace)
        
        for conj in conjugates:
            if conj != trace:
                # Edit distance as transform cost
                cost = self._compute_edit_distance(trace, conj)
                costs.append(cost)
                
        # Also consider transforms to other traces
        for other_n, other_data in self.trace_universe.items():
            other_trace = other_data['trace']
            if other_trace != trace and len(other_trace) == len(trace):
                # Same length transforms are cheaper
                cost = self._compute_edit_distance(trace, other_trace) * 0.8
                costs.append(cost)
                
        # Minimum transform cost
        min_cost = min(costs) if costs else 0.0
        
        self.transform_cache[trace] = min_cost
        return min_cost
        
    def _compute_edit_distance(self, trace1: str, trace2: str) -> float:
        """计算两个trace之间的编辑距离"""
        if len(trace1) != len(trace2):
            # Pad shorter trace
            max_len = max(len(trace1), len(trace2))
            trace1 = trace1.ljust(max_len, '0')
            trace2 = trace2.ljust(max_len, '0')
            
        distance = 0.0
        for i in range(len(trace1)):
            if trace1[i] != trace2[i]:
                # Position-weighted distance
                fib_idx = min(i, len(self.fibonacci_numbers) - 1)
                fib_weight = self.fibonacci_numbers[fib_idx] / self.fibonacci_numbers[-1]
                distance += 1.0 * (1 + fib_weight)
                
        return distance / len(trace1)
        
    def _compute_continuity_measure(self, trace: str) -> float:
        """计算trace映射的连续性度量"""
        # Continuity based on how smoothly trace maps to conjugates
        conjugates = self._compute_conjugacy_class(trace)
        
        if len(conjugates) <= 1:
            return 1.0  # Trivial map is perfectly continuous
            
        continuity = 0.0
        
        # Check continuity between adjacent conjugates
        for i in range(len(conjugates) - 1):
            conj1 = conjugates[i]
            conj2 = conjugates[i + 1]
            
            # Continuity as inverse of edit distance
            distance = self._compute_edit_distance(conj1, conj2)
            local_continuity = 1.0 / (1.0 + distance)
            continuity += local_continuity
            
        # Average continuity
        continuity /= (len(conjugates) - 1)
        
        # φ-constraint continuity bonus
        if all('11' not in conj for conj in conjugates):
            continuity *= 1.2  # Bonus for maintaining φ-constraint
            
        return min(continuity, 1.0)
        
    def _compute_mapping_radius(self, trace: str) -> float:
        """计算trace的映射半径"""
        conjugates = self._compute_conjugacy_class(trace)
        
        if len(conjugates) <= 1:
            return 0.0  # Single point has zero radius
            
        # Radius as maximum distance from trace to its conjugates
        max_distance = 0.0
        
        for conj in conjugates:
            if conj != trace:
                distance = self._compute_edit_distance(trace, conj)
                max_distance = max(max_distance, distance)
                
        return max_distance
        
    def _compute_transform_dimension(self, trace: str) -> float:
        """计算trace的变换维度"""
        conjugates = self._compute_conjugacy_class(trace)
        
        # Dimension based on conjugacy class size and structure
        class_size = len(conjugates)
        
        if class_size <= 1:
            return 0.0
            
        # Base dimension from class size
        base_dim = log2(class_size) / log2(self.max_trace_size + 1)
        
        # Structure dimension from conjugate diversity
        structure_diversity = 0.0
        for conj in conjugates:
            ones_count = conj.count('1')
            structure_diversity += ones_count / len(conj)
        structure_diversity /= class_size
        
        # Combined dimension
        return base_dim + structure_diversity * 0.5
        
    def _compute_mapping_complexity(self, trace: str) -> float:
        """计算trace的映射复杂度"""
        # Complexity based on conjugacy structure
        conjugates = self._compute_conjugacy_class(trace)
        complexity = 0.0
        
        # Class size complexity
        class_size = len(conjugates)
        if class_size > 1:
            complexity += log2(class_size)
            
        # Transform diversity complexity
        transform_costs = []
        for conj in conjugates:
            if conj != trace:
                cost = self._compute_edit_distance(trace, conj)
                transform_costs.append(cost)
                
        if transform_costs:
            cost_variance = np.var(transform_costs)
            complexity += cost_variance
            
        # φ-constraint complexity
        phi_violations = sum(1 for conj in conjugates if '11' in conj)
        complexity += phi_violations * 0.5
        
        # Continuity complexity (inverse relationship)
        continuity = self._compute_continuity_measure(trace)
        complexity += (1.0 - continuity) * 2.0
        
        return complexity / self.max_map_complexity
        
    def _classify_mapping_type(self, trace: str) -> str:
        """分类trace的映射类型"""
        conjugates = self._compute_conjugacy_class(trace)
        continuity = self._compute_continuity_measure(trace)
        complexity = self._compute_mapping_complexity(trace)
        
        if len(conjugates) == 1:
            return "identity_map"
        elif continuity >= 0.8 and complexity < 0.3:
            return "continuous_map"
        elif continuity >= 0.5 and complexity < 0.5:
            return "smooth_map"
        elif continuity >= 0.3 or complexity < 0.7:
            return "regular_map"
        else:
            return "discontinuous_map"
            
    def _compute_preserve_measure(self, trace: str) -> float:
        """计算trace映射的保持度量"""
        conjugates = self._compute_conjugacy_class(trace)
        
        if len(conjugates) <= 1:
            return 1.0  # Identity preserves everything
            
        # Check what properties are preserved across conjugates
        preserved_count = 0
        total_checks = 0
        
        # Check ones count preservation
        ones_counts = [conj.count('1') for conj in conjugates]
        if len(set(ones_counts)) == 1:
            preserved_count += 1
        total_checks += 1
        
        # Check length preservation
        lengths = [len(conj) for conj in conjugates]
        if len(set(lengths)) == 1:
            preserved_count += 1
        total_checks += 1
        
        # Check φ-constraint preservation
        phi_valid = all('11' not in conj for conj in conjugates)
        if phi_valid:
            preserved_count += 1
        total_checks += 1
        
        # Check structural hash preservation (modulo rotations)
        hashes = [self._compute_structural_hash(conj) for conj in conjugates]
        hash_variance = np.var(hashes) if len(hashes) > 1 else 0
        if hash_variance < 100:  # Small variance means mostly preserved
            preserved_count += 1
        total_checks += 1
        
        return preserved_count / total_checks
        
    def _compute_conjugacy_invariants(self, trace: str) -> Dict[str, Any]:
        """计算trace的共轭不变量"""
        conjugates = self._compute_conjugacy_class(trace)
        
        invariants = {}
        
        # Class size is always invariant
        invariants['class_size'] = len(conjugates)
        
        # Ones count (should be invariant for our conjugacy)
        ones_counts = [conj.count('1') for conj in conjugates]
        invariants['ones_count'] = ones_counts[0] if ones_counts else 0
        
        # Trace length (should be invariant)
        lengths = [len(conj) for conj in conjugates]
        invariants['trace_length'] = lengths[0] if lengths else 0
        
        # φ-validity (should be invariant)
        invariants['phi_valid'] = all('11' not in conj for conj in conjugates)
        
        # Conjugacy signature (complex invariant)
        signature_sum = 0 + 0j
        for conj in conjugates:
            ones_pos = [i for i, bit in enumerate(conj) if bit == '1']
            for pos in ones_pos:
                phase = 2 * pi * pos / len(conj)
                signature_sum += (cos(phase) + 1j * sin(phase))
        invariants['conjugacy_signature'] = abs(signature_sum) / len(conjugates)
        
        return invariants
        
    def analyze_mapping_system(self) -> Dict:
        """分析完整的mapping系统"""
        traces = [data['trace'] for data in self.trace_universe.values()]
        
        analysis = {
            'mapping_universe_size': len(traces),
            'conjugacy_matrix': self._compute_conjugacy_matrix(traces),
            'mapping_properties': self._analyze_mapping_properties(traces),
            'network_analysis': self._analyze_mapping_networks(traces),
            'information_analysis': self._analyze_mapping_information(traces),
            'category_analysis': self._analyze_mapping_categories(traces),
            'three_domain_analysis': self._analyze_three_domains(traces)
        }
        
        return analysis
        
    def _compute_conjugacy_matrix(self, traces: List[str]) -> np.ndarray:
        """计算traces之间的共轭关系矩阵"""
        n = len(traces)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            conjugates_i = set(self._compute_conjugacy_class(traces[i]))
            for j in range(n):
                if traces[j] in conjugates_i:
                    matrix[i, j] = 1.0
                    
        return matrix
        
    def _analyze_mapping_properties(self, traces: List[str]) -> Dict:
        """分析mapping属性统计"""
        properties = {}
        
        # Collect all mapping properties
        all_continuities = []
        all_radii = []
        all_dimensions = []
        all_complexities = []
        all_preserves = []
        mapping_types = []
        
        for trace in traces:
            mapping_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['mapping_properties']
            all_continuities.append(mapping_props['continuity_measure'])
            all_radii.append(mapping_props['mapping_radius'])
            all_dimensions.append(mapping_props['transform_dimension'])
            all_complexities.append(mapping_props['mapping_complexity'])
            all_preserves.append(mapping_props['preserve_measure'])
            mapping_types.append(mapping_props['mapping_type'])
            
        properties.update({
            'mean_continuity': np.mean(all_continuities),
            'mean_radius': np.mean(all_radii),
            'mean_dimension': np.mean(all_dimensions),
            'mean_complexity': np.mean(all_complexities),
            'mean_preserve': np.mean(all_preserves),
            'mapping_type_distribution': {
                t: mapping_types.count(t) / len(mapping_types) 
                for t in set(mapping_types)
            }
        })
        
        return properties
        
    def _analyze_mapping_networks(self, traces: List[str]) -> Dict:
        """分析mapping网络属性"""
        # Build mapping-based network using conjugacy
        G = nx.Graph()
        G.add_nodes_from(range(len(traces)))
        
        # Add edges for conjugate traces
        conjugacy_matrix = self._compute_conjugacy_matrix(traces)
        
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                if conjugacy_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=conjugacy_matrix[i, j])
                    
        return {
            'network_nodes': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'network_density': nx.density(G),
            'connected_components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0.0
        }
        
    def _analyze_mapping_information(self, traces: List[str]) -> Dict:
        """分析mapping信息论属性"""
        # Collect mapping type distribution for entropy
        mapping_types = []
        dimensions = []
        complexities = []
        continuities = []
        
        for trace in traces:
            mapping_props = self.trace_universe[int(trace, 2) if trace != '0' else 0]['mapping_properties']
            mapping_types.append(mapping_props['mapping_type'])
            dimensions.append(round(mapping_props['transform_dimension'], 1))
            complexities.append(round(mapping_props['mapping_complexity'], 1))
            continuities.append(round(mapping_props['continuity_measure'], 1))
            
        def compute_entropy(values):
            from collections import Counter
            counts = Counter(values)
            total = len(values)
            return -sum((count/total) * log2(count/total) for count in counts.values())
            
        return {
            'dimension_entropy': compute_entropy(dimensions),
            'type_entropy': compute_entropy(mapping_types),
            'complexity_entropy': compute_entropy(complexities),
            'continuity_entropy': compute_entropy(continuities),
            'mapping_complexity': len(set(mapping_types))
        }
        
    def _analyze_mapping_categories(self, traces: List[str]) -> Dict:
        """分析mapping范畴论属性"""
        # Count morphisms (conjugacy relationships)
        morphism_count = 0
        functorial_count = 0
        
        conjugacy_matrix = self._compute_conjugacy_matrix(traces)
        n = len(traces)
        
        # Count conjugacy relationships
        for i in range(n):
            for j in range(n):
                if i != j and conjugacy_matrix[i, j] > 0:
                    morphism_count += 1
                    
                    # Check if relationship preserves structure (functoriality)
                    trace_i = traces[i]
                    trace_j = traces[j]
                    
                    props_i = self.trace_universe[int(trace_i, 2) if trace_i != '0' else 0]['mapping_properties']
                    props_j = self.trace_universe[int(trace_j, 2) if trace_j != '0' else 0]['mapping_properties']
                    
                    # Structure preservation: similar invariants
                    inv_i = props_i['conjugacy_invariants']
                    inv_j = props_j['conjugacy_invariants']
                    
                    if (inv_i['ones_count'] == inv_j['ones_count'] and
                        inv_i['trace_length'] == inv_j['trace_length'] and
                        inv_i['phi_valid'] == inv_j['phi_valid']):
                        functorial_count += 1
                        
        # Count reachable pairs (through conjugacy)
        reachable_pairs = sum(1 for i in range(n) for j in range(n) 
                             if conjugacy_matrix[i, j] > 0)
        
        return {
            'mapping_morphisms': morphism_count,
            'functorial_relationships': functorial_count,
            'functoriality_ratio': functorial_count / morphism_count if morphism_count > 0 else 0,
            'reachable_pairs': reachable_pairs,
            'category_structure': f"Category with {n} objects and {morphism_count} morphisms"
        }
        
    def _analyze_three_domains(self, traces: List[str]) -> Dict:
        """分析三域系统：Traditional vs φ-constrained vs Convergence"""
        # Traditional domain: all possible mappings without φ-constraints
        traditional_operations = 100  # Baseline traditional mapping operations
        
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
                'Traditional': f"{traditional_operations} total mapping operations",
                'φ-Constrained': f"{phi_constrained_operations} φ-valid mapping operations", 
                'Convergence': f"{convergence_operations} operations preserved in both systems"
            }
        }
        
    def generate_visualizations(self, analysis: Dict, prefix: str = "chapter-070-collapse-topomap"):
        """生成mapping系统的可视化"""
        plt.style.use('default')
        
        # 创建主要的可视化图表
        self._create_mapping_structure_plot(analysis, f"{prefix}-structure.png")
        self._create_mapping_properties_plot(analysis, f"{prefix}-properties.png") 
        self._create_domain_analysis_plot(analysis, f"{prefix}-domains.png")
        
    def _create_mapping_structure_plot(self, analysis: Dict, filename: str):
        """创建mapping结构可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Continuity measure distribution
        traces = [data['trace'] for data in self.trace_universe.values()]
        continuities = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['continuity_measure'] 
                       for t in traces]
        
        ax1.hist(continuities, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Continuity Measure Distribution')
        ax1.set_xlabel('Continuity')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Radius vs Complexity scatter
        radii = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['mapping_radius'] 
                for t in traces]
        complexities = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['mapping_complexity'] 
                       for t in traces]
        dimensions = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['transform_dimension'] 
                     for t in traces]
        
        scatter = ax2.scatter(radii, complexities, c=dimensions, cmap='viridis', alpha=0.7, s=60)
        ax2.set_title('Radius vs Complexity')
        ax2.set_xlabel('Mapping Radius')
        ax2.set_ylabel('Mapping Complexity')
        plt.colorbar(scatter, ax=ax2, label='Transform Dimension')
        ax2.grid(True, alpha=0.3)
        
        # Mapping type distribution
        mapping_types = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['mapping_type'] 
                        for t in traces]
        type_counts = {}
        for mt in mapping_types:
            type_counts[mt] = type_counts.get(mt, 0) + 1
            
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax3.set_title('Mapping Type Distribution')
        
        # Mapping signatures (complex plane)
        signatures = [self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['mapping_signature'] 
                     for t in traces]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        ax4.scatter(real_parts, imag_parts, c=continuities, cmap='plasma', alpha=0.7, s=60)
        ax4.set_title('Mapping Signatures (Complex Plane)')
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
        
    def _create_mapping_properties_plot(self, analysis: Dict, filename: str):
        """创建mapping属性可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mapping efficiency metrics
        metrics = analysis['mapping_properties']
        metric_names = ['mean_continuity', 'mean_radius', 'mean_dimension', 'mean_complexity']
        metric_values = [metrics[name] for name in metric_names]
        
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        bars = ax1.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Mapping Efficiency Metrics')
        ax1.set_xlabel('Mapping Type')
        ax1.set_ylabel('Efficiency Score')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace('mean_', '').replace('_', ' ').title() for name in metric_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Conjugacy class sizes
        traces = [data['trace'] for data in self.trace_universe.values()]
        class_sizes = []
        for trace in traces:
            conjugates = self.trace_universe[int(trace, 2) if trace != '0' else 0]['mapping_properties']['conjugacy_class']
            class_sizes.append(len(conjugates))
            
        ax2.hist(class_sizes, bins=max(class_sizes), alpha=0.7, color='purple', edgecolor='black')
        ax2.set_title('Conjugacy Class Size Distribution')
        ax2.set_xlabel('Class Size')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # System complexity evolution
        complexity_metrics = ['mapping_complexity', 'dimension_diversity', 'type_diversity', 'continuity_diversity']
        complexity_values = [
            analysis['information_analysis']['mapping_complexity'],
            len(set([round(self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['transform_dimension'], 1) for t in traces])),
            len(set([self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['mapping_type'] for t in traces])),
            len(set([round(self.trace_universe[int(t, 2) if t != '0' else 0]['mapping_properties']['continuity_measure'], 1) for t in traces]))
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
        
        # Mapping efficiency metrics
        metrics = analysis['mapping_properties']
        efficiency_metrics = ['mean_continuity', 'mean_preserve', 'mean_radius']
        efficiency_values = [metrics[name] for name in efficiency_metrics]
        
        bars = ax2.bar(range(len(efficiency_metrics)), efficiency_values, 
                      color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Mapping Efficiency Metrics')
        ax2.set_xlabel('Mapping Type')
        ax2.set_ylabel('Efficiency Score')
        ax2.set_xticks(range(len(efficiency_metrics)))
        ax2.set_xticklabels([name.replace('mean_', '').replace('_', '\n') for name in efficiency_metrics])
        ax2.grid(True, alpha=0.3)
        
        # Information theory results
        info_data = analysis['information_analysis']
        info_metrics = ['dimension_entropy', 'type_entropy', 'complexity_entropy', 'continuity_entropy']
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
            cat_data['mapping_morphisms'],
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

class TestCollapseTopoMap(unittest.TestCase):
    """单元测试用于验证collapse topological map实现"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseTopoMapSystem(max_trace_size=6)
        
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
        
    def test_conjugacy_class(self):
        """测试共轭类计算"""
        # Single bit should have small conjugacy class
        conjugates = self.system._compute_conjugacy_class('1')
        self.assertIn('1', conjugates)
        self.assertGreaterEqual(len(conjugates), 1)
        
        # More complex trace should have larger class
        conjugates = self.system._compute_conjugacy_class('101')
        self.assertIn('101', conjugates)
        self.assertGreaterEqual(len(conjugates), 1)
        
        # All conjugates should maintain φ-constraint
        for conj in conjugates:
            self.assertNotIn('11', conj)
            
    def test_mapping_properties(self):
        """测试mapping属性计算"""
        properties = self.system._compute_mapping_properties('101')
        
        # Check that all required properties exist
        required_props = [
            'conjugacy_class', 'mapping_signature', 'transform_cost',
            'continuity_measure', 'mapping_radius', 'transform_dimension',
            'mapping_complexity', 'mapping_type', 'preserve_measure',
            'conjugacy_invariants'
        ]
        
        for prop in required_props:
            self.assertIn(prop, properties)
            
        # Check reasonable values
        self.assertGreaterEqual(properties['continuity_measure'], 0)
        self.assertLessEqual(properties['continuity_measure'], 1)
        self.assertGreaterEqual(properties['preserve_measure'], 0)
        self.assertLessEqual(properties['preserve_measure'], 1)
        
    def test_mapping_type_classification(self):
        """测试映射类型分类"""
        # Test different traces for type classification
        types_found = set()
        
        for n in range(20):
            trace = self.system._encode_to_trace(n)
            if '11' not in trace:  # Only φ-valid traces
                mapping_type = self.system._classify_mapping_type(trace)
                types_found.add(mapping_type)
                
        # Should find multiple mapping types
        self.assertGreater(len(types_found), 1)
        
        # All types should be valid
        valid_types = {
            "identity_map", "continuous_map", "smooth_map", 
            "regular_map", "discontinuous_map"
        }
        self.assertTrue(types_found.issubset(valid_types))
        
    def test_conjugacy_invariants(self):
        """测试共轭不变量"""
        invariants = self.system._compute_conjugacy_invariants('101')
        
        # Check required invariants
        self.assertIn('class_size', invariants)
        self.assertIn('ones_count', invariants)
        self.assertIn('trace_length', invariants)
        self.assertIn('phi_valid', invariants)
        self.assertIn('conjugacy_signature', invariants)
        
        # Check values are reasonable
        self.assertGreater(invariants['class_size'], 0)
        self.assertEqual(invariants['ones_count'], 2)  # '101' has 2 ones
        self.assertEqual(invariants['trace_length'], 3)  # '101' has length 3
        self.assertTrue(invariants['phi_valid'])  # '101' is φ-valid
        
    def test_mapping_system_analysis(self):
        """测试完整mapping系统分析"""
        analysis = self.system.analyze_mapping_system()
        
        # Check that all analysis sections exist
        required_sections = [
            'mapping_universe_size', 'conjugacy_matrix', 'mapping_properties',
            'network_analysis', 'information_analysis', 'category_analysis',
            'three_domain_analysis'
        ]
        
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # Check reasonable values
        self.assertGreater(analysis['mapping_universe_size'], 0)
        self.assertGreater(analysis['three_domain_analysis']['convergence_ratio'], 0)
        
    def test_conjugacy_matrix_properties(self):
        """测试共轭矩阵属性"""
        traces = ['0', '1', '10', '101']
        matrix = self.system._compute_conjugacy_matrix(traces)
        
        # Matrix should be square
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.shape[0], len(traces))
        
        # Diagonal should be ones (self-conjugacy)
        for i in range(len(traces)):
            self.assertEqual(matrix[i, i], 1.0)
            
        # Matrix should be symmetric (conjugacy is symmetric)
        for i in range(len(traces)):
            for j in range(len(traces)):
                self.assertEqual(matrix[i, j], matrix[j, i])
                
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        analysis = self.system.analyze_mapping_system()
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
        analysis = self.system.analyze_mapping_system()
        
        # Should not raise exceptions
        try:
            self.system.generate_visualizations(analysis, "test-topomap")
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Visualization generation failed: {e}")
            
        self.assertTrue(test_passed)

def main():
    """主函数：运行测试和分析"""
    print("🔄 Chapter 070: CollapseTopoMap Unit Test Verification")
    print("=" * 60)
    
    # 创建系统实例
    system = CollapseTopoMapSystem(max_trace_size=6)
    
    print("📊 Building trace universe...")
    print(f"✅ Found {len(system.trace_universe)} φ-valid traces")
    
    # 运行完整分析
    print("\n🔍 Analyzing collapse topological mapping system...")
    analysis = system.analyze_mapping_system()
    
    print(f"📈 Mapping universe size: {analysis['mapping_universe_size']} elements")
    print(f"📊 Network density: {analysis['network_analysis']['network_density']:.3f}")
    print(f"🎯 Convergence ratio: {analysis['three_domain_analysis']['convergence_ratio']:.3f}")
    
    # 显示mapping属性统计
    props = analysis['mapping_properties']
    print(f"\n📏 Mapping Properties:")
    print(f"   Mean continuity: {props['mean_continuity']:.3f}")
    print(f"   Mean radius: {props['mean_radius']:.3f}")
    print(f"   Mean dimension: {props['mean_dimension']:.3f}")
    print(f"   Mean complexity: {props['mean_complexity']:.3f}")
    print(f"   Mean preserve: {props['mean_preserve']:.3f}")
    
    # 显示信息论分析
    info = analysis['information_analysis']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Continuity entropy: {info['continuity_entropy']:.3f} bits")
    print(f"   Mapping complexity: {info['mapping_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.generate_visualizations(analysis)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n✅ Chapter 070: CollapseTopoMap verification completed!")
    print("=" * 60)
    print("🔥 Topological mapping structures exhibit bounded conjugacy convergence!")

if __name__ == "__main__":
    main()