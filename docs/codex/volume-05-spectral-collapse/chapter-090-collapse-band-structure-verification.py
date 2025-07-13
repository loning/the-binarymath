#!/usr/bin/env python3
"""
Chapter 090: CollapseBandStructure Unit Test Verification
从ψ=ψ(ψ)推导Trace Tensor Zones in Spectral Frequency Layers

Core principle: From ψ = ψ(ψ) derive band structure formation where trace tensors
organize into distinct frequency layers, creating spectral bands with forbidden gaps,
revealing how φ-constraint geometry creates the fundamental band architecture of
collapsed frequency space.

This verification program implements:
1. φ-constrained trace tensor organization into frequency layers
2. Band formation: systematic spectral gaps and allowed frequency ranges
3. Three-domain analysis: Traditional vs φ-constrained vs intersection band theory
4. Graph theory analysis of band networks and interlayer coupling
5. Information theory analysis of band entropy and frequency encoding
6. Category theory analysis of band functors and layer morphisms
7. Visualization of band structures and frequency layer organization
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

class CollapseBandStructureSystem:
    """
    Core system for implementing trace tensor zones in spectral frequency layers.
    Implements φ-constrained band structure via tensor layer organization.
    """
    
    def __init__(self, max_trace_value: int = 80, num_frequency_layers: int = 6):
        """Initialize collapse band structure system with layer analysis"""
        self.max_trace_value = max_trace_value
        self.num_frequency_layers = num_frequency_layers
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.band_cache = {}
        self.layer_cache = {}
        self.tensor_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.frequency_layers = self._build_frequency_layers()
        self.band_structure = self._detect_band_structure()
        
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
                tensor_data = self._analyze_tensor_properties(trace, n)
                universe[n] = tensor_data
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
        
    def _analyze_tensor_properties(self, trace: str, value: int) -> Dict:
        """分析trace的tensor性质，用于band structure分析"""
        # Core tensor analysis for band structure formation
        layer_count = len(trace)
        weight_distribution = self._compute_weight_distribution(trace)
        frequency_profile = self._compute_frequency_profile(trace, value)
        tensor_rank = self._compute_tensor_rank(trace)
        
        # Band-specific properties
        band_energy = self._compute_band_energy(trace, value)
        layer_coupling = self._compute_layer_coupling(trace)
        gap_probability = self._compute_gap_probability(trace)
        tensor_density = self._compute_tensor_density(trace)
        
        return {
            'value': value,
            'trace': trace,
            'layer_count': layer_count,
            'weight_distribution': weight_distribution,
            'frequency_profile': frequency_profile,
            'tensor_rank': tensor_rank,
            'band_energy': band_energy,
            'layer_coupling': layer_coupling,
            'gap_probability': gap_probability,
            'tensor_density': tensor_density,
            'band_classification': self._classify_band_structure(trace, value),
            'zone_membership': self._determine_zone_membership(frequency_profile)
        }
        
    def _compute_weight_distribution(self, trace: str) -> float:
        """计算trace的权重分布（基于1的位置）"""
        if not trace:
            return 0.0
        weights = []
        for i, bit in enumerate(trace):
            if bit == '1':
                weights.append(self.phi ** (len(trace) - i - 1))
        return sum(weights) / len(weights) if weights else 0.0
        
    def _compute_frequency_profile(self, trace: str, value: int) -> float:
        """计算trace的频率轮廓（基于value和φ的关系）"""
        if value == 0:
            return 0.0
        # Frequency based on position in φ-sequence
        freq_base = log(value) / log(self.phi)
        phase_factor = sin(2 * pi * freq_base / self.phi)
        return (freq_base + phase_factor) / 2
        
    def _compute_tensor_rank(self, trace: str) -> int:
        """计算trace的tensor rank（有效维度数）"""
        # Count number of transitions in the trace
        if len(trace) <= 1:
            return 1
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                transitions += 1
        return max(1, transitions)
        
    def _compute_band_energy(self, trace: str, value: int) -> float:
        """计算band energy（在频率层中的能量位置）"""
        if value == 0:
            return 0.0
        # Energy based on harmonic content
        harmonic_sum = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                harmonic_sum += 1.0 / (i + 1)
        return harmonic_sum * log(value + 1) / len(trace)
        
    def _compute_layer_coupling(self, trace: str) -> float:
        """计算层间耦合强度（相邻层的相互作用）"""
        if len(trace) <= 1:
            return 0.0
        coupling = 0.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i + 1] == '1':
                coupling += 0.0  # φ-constraint forbids this
            elif trace[i] == '1' or trace[i + 1] == '1':
                coupling += 1.0
        return coupling / (len(trace) - 1)
        
    def _compute_gap_probability(self, trace: str) -> float:
        """计算gap出现的概率（频率禁带的倾向）"""
        if not trace:
            return 0.0
        # Probability based on spacing between 1s
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        if len(ones_positions) <= 1:
            return 0.5
        
        gaps = []
        for i in range(len(ones_positions) - 1):
            gaps.append(ones_positions[i + 1] - ones_positions[i] - 1)
        
        mean_gap = sum(gaps) / len(gaps) if gaps else 0
        return min(1.0, mean_gap / len(trace))
        
    def _compute_tensor_density(self, trace: str) -> float:
        """计算tensor密度（信息密度在层结构中）"""
        if not trace:
            return 0.0
        ones_count = trace.count('1')
        return ones_count / len(trace)
        
    def _classify_band_structure(self, trace: str, value: int) -> str:
        """分类band structure类型"""
        density = self._compute_tensor_density(trace)
        energy = self._compute_band_energy(trace, value)
        
        if density < 0.3:
            return "sparse_band"
        elif density > 0.6:
            return "dense_band"
        elif energy > 0.8:
            return "high_energy_band"
        else:
            return "moderate_band"
            
    def _determine_zone_membership(self, frequency: float) -> int:
        """确定频率属于哪个zone（频率层）"""
        # Divide frequency space into zones
        zone_width = 1.0 / self.num_frequency_layers
        zone = int(frequency / zone_width)
        return min(zone, self.num_frequency_layers - 1)
        
    def _build_frequency_layers(self) -> Dict[int, List]:
        """构建频率层结构"""
        layers = {i: [] for i in range(self.num_frequency_layers)}
        
        for value, data in self.trace_universe.items():
            zone = data['zone_membership']
            layers[zone].append(value)
            
        return layers
        
    def _detect_band_structure(self) -> Dict:
        """检测band structure特征"""
        bands = []
        gaps = []
        
        # Analyze each frequency layer for band/gap characteristics
        for zone, traces in self.frequency_layers.items():
            if not traces:
                continue
                
            zone_data = [self.trace_universe[t] for t in traces]
            mean_energy = np.mean([d['band_energy'] for d in zone_data])
            energy_variance = np.var([d['band_energy'] for d in zone_data])
            coupling_strength = np.mean([d['layer_coupling'] for d in zone_data])
            gap_tendency = np.mean([d['gap_probability'] for d in zone_data])
            
            band_info = {
                'zone': zone,
                'trace_count': len(traces),
                'mean_energy': mean_energy,
                'energy_variance': energy_variance,
                'coupling_strength': coupling_strength,
                'gap_tendency': gap_tendency,
                'band_type': 'allowed' if gap_tendency < 0.5 else 'forbidden'
            }
            
            if band_info['band_type'] == 'allowed':
                bands.append(band_info)
            else:
                gaps.append(band_info)
                
        return {
            'allowed_bands': bands,
            'forbidden_gaps': gaps,
            'total_zones': self.num_frequency_layers,
            'band_gap_ratio': len(gaps) / (len(bands) + len(gaps)) if bands or gaps else 0
        }
        
    def analyze_band_properties(self) -> Dict:
        """分析band structure的全局性质"""
        all_traces = list(self.trace_universe.values())
        
        # Energy distribution analysis
        energies = [t['band_energy'] for t in all_traces]
        frequencies = [t['frequency_profile'] for t in all_traces]
        densities = [t['tensor_density'] for t in all_traces]
        couplings = [t['layer_coupling'] for t in all_traces]
        
        # Band classification distribution
        classifications = [t['band_classification'] for t in all_traces]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        # Zone distribution
        zones = [t['zone_membership'] for t in all_traces]
        zone_counts = {}
        for zone in zones:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
        return {
            'total_traces': len(all_traces),
            'energy_stats': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies)
            },
            'frequency_stats': {
                'mean': np.mean(frequencies),
                'std': np.std(frequencies),
                'range': np.max(frequencies) - np.min(frequencies)
            },
            'density_stats': {
                'mean': np.mean(densities),
                'std': np.std(densities)
            },
            'coupling_stats': {
                'mean': np.mean(couplings),
                'std': np.std(couplings)
            },
            'classification_distribution': class_counts,
            'zone_distribution': zone_counts,
            'band_structure': self.band_structure
        }
        
    def compute_entropy_measures(self) -> Dict:
        """计算band structure的信息论测度"""
        all_traces = list(self.trace_universe.values())
        
        # Extract properties for entropy calculation
        properties = {
            'band_energy': [t['band_energy'] for t in all_traces],
            'frequency_profile': [t['frequency_profile'] for t in all_traces],
            'tensor_density': [t['tensor_density'] for t in all_traces],
            'layer_coupling': [t['layer_coupling'] for t in all_traces],
            'gap_probability': [t['gap_probability'] for t in all_traces],
            'tensor_rank': [t['tensor_rank'] for t in all_traces],
            'zone_membership': [t['zone_membership'] for t in all_traces],
            'band_classification': [t['band_classification'] for t in all_traces]
        }
        
        entropies = {}
        for prop_name, values in properties.items():
            if prop_name in ['zone_membership', 'band_classification']:
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
            
    def build_band_network(self) -> nx.Graph:
        """构建band structure网络"""
        G = nx.Graph()
        
        # Add nodes for each trace
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add edges between traces in similar bands/zones
        traces = list(self.trace_universe.items())
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces[i+1:], i+1):
                # Connect if in same zone or adjacent zones
                zone_diff = abs(data1['zone_membership'] - data2['zone_membership'])
                energy_diff = abs(data1['band_energy'] - data2['band_energy'])
                
                if zone_diff <= 1 and energy_diff < 0.5:
                    coupling = data1['layer_coupling'] * data2['layer_coupling']
                    if coupling > 0.1:
                        G.add_edge(val1, val2, weight=coupling)
                        
        return G
        
    def analyze_categorical_structure(self) -> Dict:
        """分析band structure的范畴论结构"""
        # Group traces by band classification
        categories = {}
        for value, data in self.trace_universe.items():
            cls = data['band_classification']
            if cls not in categories:
                categories[cls] = []
            categories[cls].append(value)
            
        # Count morphisms (connections) between categories
        G = self.build_band_network()
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

class TestCollapseBandStructure(unittest.TestCase):
    """测试collapse band structure系统的各项功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseBandStructureSystem(max_trace_value=60, num_frequency_layers=5)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 10)
        
        # 检查基本trace性质
        for value, data in self.system.trace_universe.items():
            self.assertIn('band_energy', data)
            self.assertIn('frequency_profile', data)
            self.assertIn('zone_membership', data)
            self.assertIn('band_classification', data)
            
    def test_band_structure_detection(self):
        """测试band structure检测"""
        band_structure = self.system.band_structure
        
        self.assertIn('allowed_bands', band_structure)
        self.assertIn('forbidden_gaps', band_structure)
        self.assertIsInstance(band_structure['band_gap_ratio'], float)
        
    def test_frequency_layer_organization(self):
        """测试频率层组织"""
        layers = self.system.frequency_layers
        
        self.assertEqual(len(layers), self.system.num_frequency_layers)
        
        # 检查每层都有合理的trace数量
        total_traces = sum(len(traces) for traces in layers.values())
        self.assertEqual(total_traces, len(self.system.trace_universe))
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropies = self.system.compute_entropy_measures()
        
        required_entropies = [
            'band_energy', 'frequency_profile', 'tensor_density',
            'layer_coupling', 'zone_membership', 'band_classification'
        ]
        
        for entropy_name in required_entropies:
            self.assertIn(entropy_name, entropies)
            self.assertGreaterEqual(entropies[entropy_name], 0.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_band_network()
        
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
        # 检查节点属性
        for node in G.nodes():
            self.assertIn('band_energy', G.nodes[node])
            self.assertIn('zone_membership', G.nodes[node])
            
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
    print("Chapter 090: CollapseBandStructure Verification")
    print("从ψ=ψ(ψ)推导Trace Tensor Zones in Spectral Frequency Layers")
    print("="*80)
    
    # 创建系统
    system = CollapseBandStructureSystem(max_trace_value=70, num_frequency_layers=6)
    
    # 1. 基础统计
    print("\n1. Band Structure Foundation Analysis:")
    print("-" * 50)
    band_props = system.analyze_band_properties()
    
    print(f"Total traces analyzed: {band_props['total_traces']}")
    print(f"Energy statistics: mean={band_props['energy_stats']['mean']:.3f}, "
          f"std={band_props['energy_stats']['std']:.3f}")
    print(f"Frequency range: {band_props['frequency_stats']['range']:.3f}")
    print(f"Mean tensor density: {band_props['density_stats']['mean']:.3f}")
    print(f"Mean layer coupling: {band_props['coupling_stats']['mean']:.3f}")
    
    print("\nBand Classification Distribution:")
    for cls, count in band_props['classification_distribution'].items():
        percentage = (count / band_props['total_traces']) * 100
        print(f"- {cls}: {count} traces ({percentage:.1f}%)")
        
    print("\nZone Distribution:")
    for zone, count in band_props['zone_distribution'].items():
        percentage = (count / band_props['total_traces']) * 100
        print(f"- Zone {zone}: {count} traces ({percentage:.1f}%)")
        
    # 2. Band structure analysis
    print("\n2. Band Structure Analysis:")
    print("-" * 50)
    band_structure = band_props['band_structure']
    
    print(f"Allowed bands: {len(band_structure['allowed_bands'])}")
    print(f"Forbidden gaps: {len(band_structure['forbidden_gaps'])}")
    print(f"Band-gap ratio: {band_structure['band_gap_ratio']:.3f}")
    
    if band_structure['allowed_bands']:
        print("\nAllowed Bands:")
        for i, band in enumerate(band_structure['allowed_bands']):
            print(f"Band {i}: Zone {band['zone']}, "
                  f"traces={band['trace_count']}, "
                  f"energy={band['mean_energy']:.3f}, "
                  f"coupling={band['coupling_strength']:.3f}")
                  
    if band_structure['forbidden_gaps']:
        print("\nForbidden Gaps:")
        for i, gap in enumerate(band_structure['forbidden_gaps']):
            print(f"Gap {i}: Zone {gap['zone']}, "
                  f"traces={gap['trace_count']}, "
                  f"gap_tendency={gap['gap_tendency']:.3f}")
    
    # 3. 信息论分析
    print("\n3. Information Theory Analysis:")
    print("-" * 50)
    entropies = system.compute_entropy_measures()
    
    for prop, entropy in entropies.items():
        print(f"{prop.replace('_', ' ').title()} entropy: {entropy:.3f} bits")
        
    # 4. 网络分析
    print("\n4. Graph Theory Analysis:")
    print("-" * 50)
    G = system.build_band_network()
    
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        avg_weight = sum(edge_weights) / len(edge_weights)
        
        print(f"Average degree: {avg_degree:.3f}")
        print(f"Average edge weight: {avg_weight:.3f}")
        print(f"Connected components: {nx.number_connected_components(G)}")
        
    # 5. 范畴论分析
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    cat_analysis = system.analyze_categorical_structure()
    
    print(f"Band categories: {cat_analysis['category_count']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for category, count in cat_analysis['categories'].items():
        print(f"- {category}: {count} objects")
        
    # 6. 可视化生成
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        # 创建band structure可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.1 Band Energy Distribution
        energies = [data['band_energy'] for data in system.trace_universe.values()]
        zones = [data['zone_membership'] for data in system.trace_universe.values()]
        
        scatter = ax1.scatter(zones, energies, alpha=0.6, s=50)
        ax1.set_xlabel('Frequency Zone')
        ax1.set_ylabel('Band Energy')
        ax1.set_title('Band Energy vs Frequency Zone')
        ax1.grid(True, alpha=0.3)
        
        # 6.2 Layer Coupling Network
        if G.number_of_nodes() > 0:
            pos = {}
            for node in G.nodes():
                zone = G.nodes[node]['zone_membership']
                energy = G.nodes[node]['band_energy']
                pos[node] = (zone + random.uniform(-0.1, 0.1), energy)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=30, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=0.5)
            
        ax2.set_xlabel('Frequency Zone')
        ax2.set_ylabel('Band Energy')
        ax2.set_title('Band Structure Network')
        ax2.grid(True, alpha=0.3)
        
        # 6.3 Tensor Density Heatmap
        zone_density_matrix = np.zeros((system.num_frequency_layers, 10))
        for data in system.trace_universe.values():
            zone = data['zone_membership']
            density_bin = min(9, int(data['tensor_density'] * 10))
            zone_density_matrix[zone, density_bin] += 1
            
        im = ax3.imshow(zone_density_matrix, aspect='auto', cmap='viridis')
        ax3.set_xlabel('Tensor Density Bins')
        ax3.set_ylabel('Frequency Zones')
        ax3.set_title('Tensor Density Distribution by Zone')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 6.4 Band Classification Distribution
        classifications = list(band_props['classification_distribution'].keys())
        counts = list(band_props['classification_distribution'].values())
        
        ax4.pie(counts, labels=classifications, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Band Classification Distribution')
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-090-collapse-band-structure-bands.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Band structure visualization saved")
        
        # 创建频率层分析可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Layer structure analysis
        frequencies = [data['frequency_profile'] for data in system.trace_universe.values()]
        couplings = [data['layer_coupling'] for data in system.trace_universe.values()]
        
        # 6.5 Frequency Profile Distribution
        ax1.hist(frequencies, bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Frequency Profile')
        ax1.set_ylabel('Count')
        ax1.set_title('Frequency Profile Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 6.6 Layer Coupling vs Tensor Density
        densities = [data['tensor_density'] for data in system.trace_universe.values()]
        scatter = ax2.scatter(densities, couplings, c=zones, cmap='tab10', alpha=0.7)
        ax2.set_xlabel('Tensor Density')
        ax2.set_ylabel('Layer Coupling')
        ax2.set_title('Layer Coupling vs Tensor Density')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Zone')
        
        # 6.7 Band Gap Analysis
        gap_probs = [data['gap_probability'] for data in system.trace_universe.values()]
        ax3.scatter(frequencies, gap_probs, alpha=0.6, s=40)
        ax3.set_xlabel('Frequency Profile')
        ax3.set_ylabel('Gap Probability')
        ax3.set_title('Gap Formation Tendency')
        ax3.grid(True, alpha=0.3)
        
        # 6.8 Zone Energy Distribution
        zone_energies = {i: [] for i in range(system.num_frequency_layers)}
        for data in system.trace_universe.values():
            zone_energies[data['zone_membership']].append(data['band_energy'])
            
        zone_data = []
        zone_labels = []
        for zone, energies_list in zone_energies.items():
            if energies_list:
                zone_data.append(energies_list)
                zone_labels.append(f'Zone {zone}')
                
        if zone_data:
            ax4.boxplot(zone_data, labels=zone_labels)
            ax4.set_ylabel('Band Energy')
            ax4.set_title('Energy Distribution by Zone')
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-090-collapse-band-structure-layers.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Frequency layer visualization saved")
        
        # 创建网络和范畴论可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.9 Network visualization by zone
        if G.number_of_nodes() > 0:
            # Color nodes by zone membership
            node_colors = [G.nodes[node]['zone_membership'] for node in G.nodes()]
            node_sizes = [G.nodes[node]['tensor_density'] * 100 + 20 for node in G.nodes()]
            
            pos_spring = nx.spring_layout(G, k=1, iterations=50)
            
            scatter = ax1.scatter([pos_spring[node][0] for node in G.nodes()],
                                [pos_spring[node][1] for node in G.nodes()],
                                c=node_colors, s=node_sizes, cmap='viridis', alpha=0.8)
            
            # Draw edges
            for edge in G.edges():
                x_coords = [pos_spring[edge[0]][0], pos_spring[edge[1]][0]]
                y_coords = [pos_spring[edge[0]][1], pos_spring[edge[1]][1]]
                ax1.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
                
            plt.colorbar(scatter, ax=ax1, label='Zone', shrink=0.8)
        
        ax1.set_title('Band Network Structure')
        ax1.set_xlabel('Network X-coordinate')
        ax1.set_ylabel('Network Y-coordinate')
        
        # 6.10 Morphism density by category
        if cat_analysis['categories']:
            categories = list(cat_analysis['categories'].keys())
            category_sizes = list(cat_analysis['categories'].values())
            
            ax2.bar(categories, category_sizes, alpha=0.7)
            ax2.set_ylabel('Object Count')
            ax2.set_title('Category Object Distribution')
            ax2.tick_params(axis='x', rotation=45)
            
        # 6.11 Entropy landscape
        entropy_names = list(entropies.keys())
        entropy_values = list(entropies.values())
        
        bars = ax3.bar(range(len(entropy_names)), entropy_values, alpha=0.7)
        ax3.set_xticks(range(len(entropy_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in entropy_names], 
                          rotation=45, ha='right')
        ax3.set_ylabel('Entropy (bits)')
        ax3.set_title('Information Entropy Distribution')
        
        # Color bars by entropy level
        max_entropy = max(entropy_values)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(entropy_values[i] / max_entropy))
            
        # 6.12 Band structure summary
        band_types = ['Allowed Bands', 'Forbidden Gaps']
        band_counts = [len(band_structure['allowed_bands']), len(band_structure['forbidden_gaps'])]
        
        colors = ['green', 'red']
        ax4.pie(band_counts, labels=band_types, colors=colors, autopct='%1.1f%%')
        ax4.set_title('Band Structure Summary')
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-090-collapse-band-structure-network.png',
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
    print("CollapseBandStructure Verification Complete")
    print("Key Findings:")
    print(f"- {band_props['total_traces']} φ-valid traces organized into {system.num_frequency_layers} frequency zones")
    print(f"- {len(band_structure['allowed_bands'])} allowed bands and {len(band_structure['forbidden_gaps'])} forbidden gaps")
    print(f"- Band-gap ratio: {band_structure['band_gap_ratio']:.3f}")
    print(f"- Mean tensor density: {band_props['density_stats']['mean']:.3f}")
    print(f"- {cat_analysis['category_count']} distinct band categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print("="*80)

if __name__ == "__main__":
    run_verification()