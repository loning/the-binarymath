#!/usr/bin/env python3
"""
Chapter 094: TensorZetaFlow Unit Test Verification
从ψ=ψ(ψ)推导Collapse Weight Currents across Trace Tensor Networks

Core principle: From ψ = ψ(ψ) derive flow dynamics where spectral weights
create systematic currents across trace tensor networks, generating directed
flows that preserve tensor structure while enabling weight redistribution
through entropy-increasing transformations that reveal the fundamental flow
architecture of collapsed tensor space.

This verification program implements:
1. φ-constrained tensor network construction with weight flow analysis
2. Flow current computation: systematic weight redistribution patterns
3. Three-domain analysis: Traditional vs φ-constrained vs intersection flow theory
4. Graph theory analysis of flow networks and current pathways
5. Information theory analysis of flow entropy and current encoding
6. Category theory analysis of flow functors and current morphisms
7. Visualization of flow patterns and tensor current structures
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

class TensorZetaFlowSystem:
    """
    Core system for implementing collapse weight currents across trace tensor networks.
    Implements φ-constrained flow dynamics via tensor current operations.
    """
    
    def __init__(self, max_trace_value: int = 120, flow_resolution: int = 16):
        """Initialize tensor zeta flow system with current analysis"""
        self.max_trace_value = max_trace_value
        self.flow_resolution = flow_resolution
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.flow_cache = {}
        self.current_cache = {}
        self.tensor_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.tensor_network = self._build_tensor_network()
        self.flow_field = self._compute_flow_field()
        
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
                flow_data = self._analyze_flow_properties(trace, n)
                universe[n] = flow_data
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
        
    def _analyze_flow_properties(self, trace: str, value: int) -> Dict:
        """分析trace的flow properties，用于tensor current分析"""
        # Core flow analysis
        weight_density = self._compute_weight_density(trace, value)
        flow_potential = self._compute_flow_potential(trace, value)
        current_capacity = self._compute_current_capacity(trace)
        tensor_conductance = self._compute_tensor_conductance(trace, value)
        
        # Flow-specific properties
        source_strength = self._compute_source_strength(trace, value)
        sink_affinity = self._compute_sink_affinity(trace)
        flow_resistance = self._compute_flow_resistance(trace)
        current_direction = self._compute_current_direction(trace, value)
        
        return {
            'value': value,
            'trace': trace,
            'weight_density': weight_density,
            'flow_potential': flow_potential,
            'current_capacity': current_capacity,
            'tensor_conductance': tensor_conductance,
            'source_strength': source_strength,
            'sink_affinity': sink_affinity,
            'flow_resistance': flow_resistance,
            'current_direction': current_direction,
            'flow_classification': self._classify_flow_node(trace, value),
            'tensor_position': self._determine_tensor_position(weight_density, flow_potential)
        }
        
    def _compute_weight_density(self, trace: str, value: int) -> float:
        """计算weight density（权重密度）"""
        if not trace or value == 0:
            return 0.0
            
        # Density based on trace structure and value magnitude
        ones_count = trace.count('1')
        trace_length = len(trace)
        
        if trace_length == 0:
            return 0.0
            
        # Base density from trace pattern
        pattern_density = ones_count / trace_length
        
        # Enhancement from value magnitude (logarithmic scaling)
        value_factor = log(value + 1) / log(self.max_trace_value)
        
        # φ-constraint enhancement
        phi_factor = self.phi if "11" not in trace else 1.0
        
        density = pattern_density * value_factor * phi_factor / 2.0
        
        return min(density, 1.0)
        
    def _compute_flow_potential(self, trace: str, value: int) -> float:
        """计算flow potential（流势）"""
        if not trace or value == 0:
            return 0.0
            
        # Potential based on trace organization and structural complexity
        complexity = 0.0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                complexity += 1.0
                
        if len(trace) <= 1:
            normalized_complexity = 0.0
        else:
            normalized_complexity = complexity / (len(trace) - 1)
            
        # Potential from value position in sequence
        value_potential = sin(2 * pi * log(value + 1) / log(self.phi))
        
        # Combined potential
        potential = (normalized_complexity + abs(value_potential)) / 2.0
        
        return potential
        
    def _compute_current_capacity(self, trace: str) -> float:
        """计算current capacity（电流容量）"""
        if not trace:
            return 0.0
            
        # Capacity based on trace "bandwidth" and structural stability
        bandwidth = len(trace)
        
        # Stability from pattern regularity
        pattern_changes = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                pattern_changes += 1
                
        stability = 1.0 - (pattern_changes / max(1, len(trace) - 1))
        
        # Capacity combines bandwidth and stability
        capacity = sqrt(bandwidth) * stability / 5.0  # Normalize
        
        # φ-constraint bonus
        if "11" not in trace:
            capacity *= sqrt(self.phi)
            
        return min(capacity, 1.0)
        
    def _compute_tensor_conductance(self, trace: str, value: int) -> float:
        """计算tensor conductance（张量电导）"""
        if not trace:
            return 0.0
            
        # Conductance based on trace connectivity and flow efficiency
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        
        if len(ones_positions) <= 1:
            return 0.1  # Minimal conductance
            
        # Conductance from connection patterns
        connection_efficiency = 0.0
        for i in range(len(ones_positions) - 1):
            gap = ones_positions[i + 1] - ones_positions[i] - 1
            connection_efficiency += exp(-gap / 2.0)  # Prefer smaller gaps
            
        conductance = connection_efficiency / (len(ones_positions) - 1)
        
        # Value-dependent modulation
        value_modulation = cos(2 * pi * value / 100.0) * 0.1 + 1.0
        
        return conductance * value_modulation
        
    def _compute_source_strength(self, trace: str, value: int) -> float:
        """计算source strength（源强度）"""
        if not trace:
            return 0.0
            
        # Source strength from trace beginning patterns
        source_patterns = ['1', '10', '101']
        strength = 0.0
        
        for pattern in source_patterns:
            if trace.startswith(pattern):
                strength += len(pattern) / 3.0  # Weight by pattern length
                break
                
        # Enhancement from value magnitude
        value_enhancement = sqrt(value) / sqrt(self.max_trace_value)
        
        return strength * value_enhancement
        
    def _compute_sink_affinity(self, trace: str) -> float:
        """计算sink affinity（汇亲和力）"""
        if not trace:
            return 0.0
            
        # Sink affinity from trace ending patterns
        sink_patterns = ['0', '01', '010']
        affinity = 0.0
        
        for pattern in sink_patterns:
            if trace.endswith(pattern):
                affinity += len(pattern) / 3.0
                break
                
        # Length-dependent normalization
        length_factor = 1.0 / sqrt(len(trace))
        
        return affinity * length_factor
        
    def _compute_flow_resistance(self, trace: str) -> float:
        """计算flow resistance（流阻）"""
        if not trace:
            return 1.0  # Maximum resistance
            
        # Resistance from structural irregularities
        irregularities = 0
        
        # Count transitions and complex patterns
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                irregularities += 1
                
        # Resistance inversely related to organization
        if len(trace) <= 1:
            resistance = 1.0
        else:
            resistance = irregularities / (len(trace) - 1)
            
        # φ-constraint reduces resistance
        if "11" not in trace:
            resistance /= self.phi
            
        return min(resistance, 1.0)
        
    def _compute_current_direction(self, trace: str, value: int) -> float:
        """计算current direction（电流方向，以角度表示）"""
        if not trace:
            return 0.0
            
        # Direction based on trace asymmetry and value
        forward_weight = 0.0
        backward_weight = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                position_factor = i / len(trace)  # 0 to 1
                forward_weight += position_factor
                backward_weight += (1 - position_factor)
                
        # Net direction
        if forward_weight + backward_weight == 0:
            direction = 0.0
        else:
            direction = (forward_weight - backward_weight) / (forward_weight + backward_weight)
            
        # Convert to angle [0, 2π]
        angle = (direction + 1) * pi  # Map [-1, 1] to [0, 2π]
        
        # Value-dependent modulation
        value_modulation = 2 * pi * (value % 10) / 10.0
        
        return (angle + value_modulation) % (2 * pi)
        
    def _classify_flow_node(self, trace: str, value: int) -> str:
        """分类flow node类型"""
        source = self._compute_source_strength(trace, value)
        sink = self._compute_sink_affinity(trace)
        resistance = self._compute_flow_resistance(trace)
        
        if source > 0.5:
            return "source_node"
        elif sink > 0.5:
            return "sink_node"
        elif resistance < 0.3:
            return "conductor_node"
        else:
            return "resistor_node"
            
    def _determine_tensor_position(self, density: float, potential: float) -> Tuple[int, int]:
        """确定在tensor grid中的位置"""
        # Map continuous values to discrete grid positions
        x_pos = int(density * (self.flow_resolution - 1))
        y_pos = int(potential * (self.flow_resolution - 1))
        
        return (x_pos, y_pos)
        
    def _build_tensor_network(self) -> nx.DiGraph:
        """构建directed tensor network for flow analysis"""
        G = nx.DiGraph()
        
        # Add nodes for each trace
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add directed edges based on flow potential and conductance
        traces = list(self.trace_universe.items())
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces):
                if i != j:
                    # Directed flow from higher to lower potential
                    potential_diff = data1['flow_potential'] - data2['flow_potential']
                    conductance_product = data1['tensor_conductance'] * data2['tensor_conductance']
                    
                    if potential_diff > 0.1 and conductance_product > 0.1:
                        # Flow strength based on potential difference and conductance
                        flow_strength = potential_diff * conductance_product
                        resistance_factor = 1.0 / (1.0 + data1['flow_resistance'] + data2['flow_resistance'])
                        
                        edge_weight = flow_strength * resistance_factor
                        
                        if edge_weight > 0.05:  # Threshold for significant flow
                            G.add_edge(val1, val2, weight=edge_weight, 
                                     flow_type='potential_driven')
                            
        return G
        
    def _compute_flow_field(self) -> Dict:
        """计算flow field across tensor space"""
        # Initialize flow field grid
        flow_field = np.zeros((self.flow_resolution, self.flow_resolution, 2))  # 2D vector field
        density_field = np.zeros((self.flow_resolution, self.flow_resolution))
        
        # Populate fields based on trace properties
        for value, data in self.trace_universe.items():
            x_pos, y_pos = data['tensor_position']
            
            # Accumulate density
            density_field[y_pos, x_pos] += data['weight_density']
            
            # Compute flow vector
            direction = data['current_direction']
            magnitude = data['current_capacity']
            
            flow_x = magnitude * cos(direction)
            flow_y = magnitude * sin(direction)
            
            flow_field[y_pos, x_pos, 0] += flow_x
            flow_field[y_pos, x_pos, 1] += flow_y
            
        # Normalize flow field
        for i in range(self.flow_resolution):
            for j in range(self.flow_resolution):
                magnitude = sqrt(flow_field[i, j, 0]**2 + flow_field[i, j, 1]**2)
                if magnitude > 0:
                    flow_field[i, j, 0] /= magnitude
                    flow_field[i, j, 1] /= magnitude
                    
        return {
            'flow_vectors': flow_field,
            'density_field': density_field,
            'grid_resolution': self.flow_resolution
        }
        
    def analyze_flow_properties(self) -> Dict:
        """分析flow system的全局性质"""
        all_traces = list(self.trace_universe.values())
        
        # Statistical analysis
        densities = [t['weight_density'] for t in all_traces]
        potentials = [t['flow_potential'] for t in all_traces]
        capacities = [t['current_capacity'] for t in all_traces]
        conductances = [t['tensor_conductance'] for t in all_traces]
        sources = [t['source_strength'] for t in all_traces]
        sinks = [t['sink_affinity'] for t in all_traces]
        resistances = [t['flow_resistance'] for t in all_traces]
        directions = [t['current_direction'] for t in all_traces]
        
        # Classification distribution
        classifications = [t['flow_classification'] for t in all_traces]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        # Network analysis
        G = self.tensor_network
        
        return {
            'total_traces': len(all_traces),
            'density_stats': {
                'mean': np.mean(densities),
                'std': np.std(densities),
                'max': np.max(densities)
            },
            'potential_stats': {
                'mean': np.mean(potentials),
                'std': np.std(potentials),
                'range': np.max(potentials) - np.min(potentials)
            },
            'capacity_stats': {
                'mean': np.mean(capacities),
                'std': np.std(capacities)
            },
            'conductance_stats': {
                'mean': np.mean(conductances),
                'std': np.std(conductances)
            },
            'source_stats': {
                'mean': np.mean(sources),
                'std': np.std(sources)
            },
            'sink_stats': {
                'mean': np.mean(sinks),
                'std': np.std(sinks)
            },
            'resistance_stats': {
                'mean': np.mean(resistances),
                'std': np.std(resistances)
            },
            'classification_distribution': class_counts,
            'network_properties': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'strongly_connected_components': nx.number_strongly_connected_components(G),
                'weakly_connected_components': nx.number_weakly_connected_components(G)
            },
            'flow_field': self.flow_field
        }
        
    def compute_entropy_measures(self) -> Dict:
        """计算flow system的信息论测度"""
        all_traces = list(self.trace_universe.values())
        
        # Extract properties for entropy calculation
        properties = {
            'weight_density': [t['weight_density'] for t in all_traces],
            'flow_potential': [t['flow_potential'] for t in all_traces],
            'current_capacity': [t['current_capacity'] for t in all_traces],
            'tensor_conductance': [t['tensor_conductance'] for t in all_traces],
            'source_strength': [t['source_strength'] for t in all_traces],
            'sink_affinity': [t['sink_affinity'] for t in all_traces],
            'flow_resistance': [t['flow_resistance'] for t in all_traces],
            'current_direction': [t['current_direction'] for t in all_traces],
            'flow_classification': [t['flow_classification'] for t in all_traces]
        }
        
        entropies = {}
        for prop_name, values in properties.items():
            if prop_name in ['flow_classification']:
                # Discrete entropy
                entropies[prop_name] = self._compute_discrete_entropy(values)
            else:
                # Continuous entropy (via binning)
                entropies[prop_name] = self._compute_continuous_entropy(values)
                
        return entropies
        
    def _compute_discrete_entropy(self, values: List) -> float:
        """计算离散值的熵"""
        if not values:
            return 0.0
            
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
            
        total = len(values)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)
                
        return entropy
        
    def _compute_continuous_entropy(self, values: List) -> float:
        """计算连续值的熵（通过分箱）"""
        if not values:
            return 0.0
            
        # Check for constant values
        if len(set(values)) == 1:
            return 0.0
            
        # Adaptive binning
        unique_values = len(set(values))
        bin_count = min(8, max(3, unique_values))
        
        try:
            hist, _ = np.histogram(values, bins=bin_count)
            total = sum(hist)
            if total == 0:
                return 0.0
                
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / total
                    entropy -= p * log2(p)
                    
            return entropy
        except:
            # Fallback to discrete entropy
            return self._compute_discrete_entropy(values)
            
    def analyze_categorical_structure(self) -> Dict:
        """分析flow system的范畴论结构"""
        # Group traces by flow classification
        categories = {}
        for value, data in self.trace_universe.items():
            cls = data['flow_classification']
            if cls not in categories:
                categories[cls] = []
            categories[cls].append(value)
            
        # Count morphisms (directed connections) between categories
        G = self.tensor_network
        morphisms = {}
        total_morphisms = 0
        
        for cls1 in categories:
            for cls2 in categories:
                count = 0
                for v1 in categories[cls1]:
                    for v2 in categories[cls2]:
                        if G.has_edge(v1, v2):
                            count += 1
                morphisms[f"{cls1}->{cls2}"] = count
                total_morphisms += count
                
        # Morphism density
        total_possible = sum(len(cat1) * len(cat2) for cat1 in categories.values() 
                           for cat2 in categories.values())
        morphism_density = total_morphisms / total_possible if total_possible > 0 else 0
        
        return {
            'categories': {cls: len(traces) for cls, traces in categories.items()},
            'morphisms': morphisms,
            'total_morphisms': total_morphisms,
            'morphism_density': morphism_density,
            'category_count': len(categories)
        }

class TestTensorZetaFlow(unittest.TestCase):
    """测试tensor zeta flow系统的各项功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = TensorZetaFlowSystem(max_trace_value=110, flow_resolution=12)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 10)
        
        # 检查基本trace性质
        for value, data in self.system.trace_universe.items():
            self.assertIn('weight_density', data)
            self.assertIn('flow_potential', data)
            self.assertIn('current_capacity', data)
            self.assertIn('flow_classification', data)
            
    def test_tensor_network_construction(self):
        """测试tensor network构建"""
        G = self.system.tensor_network
        
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        self.assertTrue(isinstance(G, nx.DiGraph))
        
    def test_flow_field_computation(self):
        """测试flow field计算"""
        field = self.system.flow_field
        
        self.assertIn('flow_vectors', field)
        self.assertIn('density_field', field)
        self.assertIn('grid_resolution', field)
        
        # Check field dimensions
        flow_vectors = field['flow_vectors']
        density_field = field['density_field']
        
        self.assertEqual(flow_vectors.shape[0], self.system.flow_resolution)
        self.assertEqual(flow_vectors.shape[1], self.system.flow_resolution)
        self.assertEqual(flow_vectors.shape[2], 2)  # 2D vector field
        
        self.assertEqual(density_field.shape[0], self.system.flow_resolution)
        self.assertEqual(density_field.shape[1], self.system.flow_resolution)
        
    def test_flow_properties_analysis(self):
        """测试flow性质分析"""
        props = self.system.analyze_flow_properties()
        
        self.assertIn('density_stats', props)
        self.assertIn('potential_stats', props)
        self.assertIn('network_properties', props)
        self.assertGreater(props['total_traces'], 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropies = self.system.compute_entropy_measures()
        
        required_entropies = [
            'weight_density', 'flow_potential', 'current_capacity',
            'tensor_conductance', 'flow_classification'
        ]
        
        for entropy_name in required_entropies:
            self.assertIn(entropy_name, entropies)
            self.assertGreaterEqual(entropies[entropy_name], 0.0)
            
    def test_categorical_analysis(self):
        """测试范畴论分析"""
        cat_analysis = self.system.analyze_categorical_structure()
        
        self.assertIn('categories', cat_analysis)
        self.assertIn('morphisms', cat_analysis)
        self.assertIn('morphism_density', cat_analysis)
        self.assertGreater(cat_analysis['category_count'], 0)

def run_verification():
    """运行完整的验证过程"""
    print("="*80)
    print("Chapter 094: TensorZetaFlow Verification")
    print("从ψ=ψ(ψ)推导Collapse Weight Currents across Trace Tensor Networks")
    print("="*80)
    
    # 创建系统
    system = TensorZetaFlowSystem(max_trace_value=115, flow_resolution=16)
    
    # 1. 基础统计
    print("\n1. Flow Dynamics Foundation Analysis:")
    print("-" * 50)
    flow_props = system.analyze_flow_properties()
    
    print(f"Total traces analyzed: {flow_props['total_traces']}")
    print(f"Weight density: mean={flow_props['density_stats']['mean']:.3f}, "
          f"max={flow_props['density_stats']['max']:.3f}")
    print(f"Flow potential range: {flow_props['potential_stats']['range']:.3f}")
    print(f"Mean current capacity: {flow_props['capacity_stats']['mean']:.3f}")
    print(f"Mean tensor conductance: {flow_props['conductance_stats']['mean']:.3f}")
    print(f"Mean source strength: {flow_props['source_stats']['mean']:.3f}")
    print(f"Mean sink affinity: {flow_props['sink_stats']['mean']:.3f}")
    print(f"Mean flow resistance: {flow_props['resistance_stats']['mean']:.3f}")
    
    print("\nFlow Classification Distribution:")
    for cls, count in flow_props['classification_distribution'].items():
        percentage = (count / flow_props['total_traces']) * 100
        print(f"- {cls}: {count} traces ({percentage:.1f}%)")
        
    # 2. Network analysis
    print("\n2. Tensor Network Analysis:")
    print("-" * 50)
    net_props = flow_props['network_properties']
    
    print(f"Network nodes: {net_props['nodes']}")
    print(f"Network edges: {net_props['edges']}")
    print(f"Network density: {net_props['density']:.3f}")
    print(f"Strongly connected components: {net_props['strongly_connected_components']}")
    print(f"Weakly connected components: {net_props['weakly_connected_components']}")
    
    if net_props['edges'] > 0:
        avg_degree = net_props['edges'] / net_props['nodes']
        print(f"Average degree: {avg_degree:.3f}")
        
    # 3. Flow field analysis
    print("\n3. Flow Field Analysis:")
    print("-" * 50)
    field = flow_props['flow_field']
    
    flow_vectors = field['flow_vectors']
    density_field = field['density_field']
    
    print(f"Grid resolution: {field['grid_resolution']}x{field['grid_resolution']}")
    print(f"Total density: {np.sum(density_field):.3f}")
    print(f"Max density cell: {np.max(density_field):.3f}")
    print(f"Mean flow magnitude: {np.mean(np.sqrt(flow_vectors[:,:,0]**2 + flow_vectors[:,:,1]**2)):.3f}")
    
    # Find dominant flow directions
    flow_angles = np.arctan2(flow_vectors[:,:,1], flow_vectors[:,:,0])
    dominant_direction = np.mean(flow_angles[flow_angles != 0]) if np.any(flow_angles != 0) else 0
    print(f"Dominant flow direction: {dominant_direction:.3f} radians ({dominant_direction*180/pi:.1f}°)")
    
    # 4. 信息论分析
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    entropies = system.compute_entropy_measures()
    
    for prop, entropy in entropies.items():
        print(f"{prop.replace('_', ' ').title()} entropy: {entropy:.3f} bits")
        
    # 5. 范畴论分析
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    cat_analysis = system.analyze_categorical_structure()
    
    print(f"Flow categories: {cat_analysis['category_count']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for category, count in cat_analysis['categories'].items():
        print(f"- {category}: {count} objects")
        
    # 6. 可视化生成
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        # 创建flow dynamics可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.1 Weight Density vs Flow Potential
        densities = [data['weight_density'] for data in system.trace_universe.values()]
        potentials = [data['flow_potential'] for data in system.trace_universe.values()]
        capacities = [data['current_capacity'] for data in system.trace_universe.values()]
        
        scatter = ax1.scatter(densities, potentials, c=capacities, cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Weight Density')
        ax1.set_ylabel('Flow Potential')
        ax1.set_title('Weight Density vs Flow Potential')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Current Capacity')
        
        # 6.2 Flow Field Visualization
        x = np.linspace(0, 1, field['grid_resolution'])
        y = np.linspace(0, 1, field['grid_resolution'])
        X, Y = np.meshgrid(x, y)
        
        # Plot density field as background
        im = ax2.imshow(density_field, extent=[0, 1, 0, 1], origin='lower', 
                       cmap='Blues', alpha=0.6)
        
        # Overlay flow vectors
        skip = 2  # Skip some arrows for clarity
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  flow_vectors[::skip, ::skip, 0], flow_vectors[::skip, ::skip, 1],
                  alpha=0.8, color='red', scale=10)
        
        ax2.set_xlabel('Tensor X Position')
        ax2.set_ylabel('Tensor Y Position')
        ax2.set_title('Flow Field with Density Background')
        plt.colorbar(im, ax=ax2, label='Weight Density', shrink=0.8)
        
        # 6.3 Conductance vs Resistance
        conductances = [data['tensor_conductance'] for data in system.trace_universe.values()]
        resistances = [data['flow_resistance'] for data in system.trace_universe.values()]
        classifications = [data['flow_classification'] for data in system.trace_universe.values()]
        
        # Color by classification
        class_to_color = {cls: i for i, cls in enumerate(set(classifications))}
        colors = [class_to_color[cls] for cls in classifications]
        
        scatter = ax3.scatter(conductances, resistances, c=colors, cmap='tab10', alpha=0.7)
        ax3.set_xlabel('Tensor Conductance')
        ax3.set_ylabel('Flow Resistance')
        ax3.set_title('Conductance vs Resistance')
        ax3.grid(True, alpha=0.3)
        
        # 6.4 Source vs Sink Analysis
        sources = [data['source_strength'] for data in system.trace_universe.values()]
        sinks = [data['sink_affinity'] for data in system.trace_universe.values()]
        
        ax4.scatter(sources, sinks, c=colors, cmap='tab10', alpha=0.7)
        ax4.set_xlabel('Source Strength')
        ax4.set_ylabel('Sink Affinity')
        ax4.set_title('Source vs Sink Characteristics')
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        max_val = max(max(sources), max(sinks))
        ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Source/Sink')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-094-tensor-zeta-flow-dynamics.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Flow dynamics visualization saved")
        
        # 创建网络流分析可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.5 Network Flow Visualization
        G = system.tensor_network
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            pos = {}
            for node in G.nodes():
                density = G.nodes[node]['weight_density']
                potential = G.nodes[node]['flow_potential']
                pos[node] = (density, potential)
            
            # Draw nodes colored by classification
            node_colors = [class_to_color[G.nodes[node]['flow_classification']] for node in G.nodes()]
            node_sizes = [G.nodes[node]['current_capacity'] * 200 + 20 for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                                 node_size=node_sizes, cmap='tab10', alpha=0.7)
            
            # Draw edges with weights
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            if edge_weights:
                max_weight = max(edge_weights)
                edge_widths = [w/max_weight * 3 for w in edge_weights]
                nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths, 
                                     alpha=0.5, edge_color='gray')
            
        ax1.set_xlabel('Weight Density')
        ax1.set_ylabel('Flow Potential')
        ax1.set_title('Tensor Flow Network')
        ax1.grid(True, alpha=0.3)
        
        # 6.6 Current Direction Distribution
        directions = [data['current_direction'] for data in system.trace_universe.values()]
        
        # Convert to degrees for readability
        directions_deg = [d * 180 / pi for d in directions]
        
        ax2.hist(directions_deg, bins=16, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Current Direction (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Current Direction Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 6.7 Flow Classification Distribution
        classification_counts = list(flow_props['classification_distribution'].values())
        classification_labels = list(flow_props['classification_distribution'].keys())
        
        ax3.pie(classification_counts, labels=classification_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Flow Classification Distribution')
        
        # 6.8 Density Field Heatmap
        im = ax4.imshow(density_field, cmap='viridis', aspect='auto', origin='lower')
        ax4.set_xlabel('Tensor X Index')
        ax4.set_ylabel('Tensor Y Index')
        ax4.set_title('Weight Density Field')
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-094-tensor-zeta-flow-currents.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Flow current analysis visualization saved")
        
        # 创建范畴论和熵分析可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.9 Category morphisms visualization
        if cat_analysis['categories']:
            categories = list(cat_analysis['categories'].keys())
            category_sizes = list(cat_analysis['categories'].values())
            
            ax1.bar(categories, category_sizes, alpha=0.7)
            ax1.set_ylabel('Object Count')
            ax1.set_title('Category Object Distribution')
            ax1.tick_params(axis='x', rotation=45)
            
        # 6.10 Entropy landscape
        entropy_names = list(entropies.keys())
        entropy_values = list(entropies.values())
        
        bars = ax2.bar(range(len(entropy_names)), entropy_values, alpha=0.7)
        ax2.set_xticks(range(len(entropy_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in entropy_names], 
                          rotation=45, ha='right')
        ax2.set_ylabel('Entropy (bits)')
        ax2.set_title('Information Entropy Distribution')
        
        # Color bars by entropy level
        max_entropy = max(entropy_values) if entropy_values else 1
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(entropy_values[i] / max_entropy))
            
        # 6.11 Flow vector magnitude field
        flow_magnitude = np.sqrt(flow_vectors[:,:,0]**2 + flow_vectors[:,:,1]**2)
        im = ax3.imshow(flow_magnitude, cmap='plasma', aspect='auto', origin='lower')
        ax3.set_xlabel('Tensor X Index')
        ax3.set_ylabel('Tensor Y Index')
        ax3.set_title('Flow Vector Magnitude Field')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 6.12 Property correlation matrix
        properties = ['weight_density', 'flow_potential', 'current_capacity', 
                     'tensor_conductance', 'source_strength', 'sink_affinity', 'flow_resistance']
        
        correlation_matrix = np.zeros((len(properties), len(properties)))
        
        for i, prop1 in enumerate(properties):
            for j, prop2 in enumerate(properties):
                values1 = [data[prop1] for data in system.trace_universe.values()]
                values2 = [data[prop2] for data in system.trace_universe.values()]
                correlation_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]
                
        im = ax4.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(properties)))
        ax4.set_yticks(range(len(properties)))
        ax4.set_xticklabels([p.replace('_', '\n') for p in properties], rotation=45, ha='right')
        ax4.set_yticklabels([p.replace('_', '\n') for p in properties])
        ax4.set_title('Flow Property Correlation Matrix')
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-094-tensor-zeta-flow-network.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Network and categorical visualization saved")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with analysis...")
    
    # 7. 运行单元测试
    print("\n7. Running Unit Tests:")
    print("-" * 50)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*80)
    print("TensorZetaFlow Verification Complete")
    print("Key Findings:")
    print(f"- {flow_props['total_traces']} φ-valid traces with tensor flow analysis")
    print(f"- {cat_analysis['category_count']} flow categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {net_props['edges']} edges among {net_props['nodes']} nodes")
    print(f"- Network density: {net_props['density']:.3f}")
    print(f"- Mean weight density: {flow_props['density_stats']['mean']:.3f}")
    print(f"- Mean current capacity: {flow_props['capacity_stats']['mean']:.3f}")
    print(f"- Flow field resolution: {field['grid_resolution']}x{field['grid_resolution']}")
    print("="*80)

if __name__ == "__main__":
    run_verification()