#!/usr/bin/env python3
"""
Chapter 091: ZetaNormal Unit Test Verification
从ψ=ψ(ψ)推导Spectral Normal Form Collapse across Conjugate Frequencies

Core principle: From ψ = ψ(ψ) derive spectral normalization procedures where
traces transform into canonical normal forms across conjugate frequency pairs,
revealing how φ-constraints create systematic normalization mappings that preserve
essential spectral structure while organizing conjugate relationships.

This verification program implements:
1. φ-constrained spectral normalization through canonical form reduction
2. Conjugate frequency analysis: systematic pairing and transformation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection normalization
4. Graph theory analysis of conjugate networks and normalization paths
5. Information theory analysis of normal form entropy and canonical encoding
6. Category theory analysis of normalization functors and form morphisms
7. Visualization of normal forms and conjugate frequency structures
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

class ZetaNormalSystem:
    """
    Core system for implementing spectral normal form collapse across conjugate frequencies.
    Implements φ-constrained normalization via canonical form operations.
    """
    
    def __init__(self, max_trace_value: int = 90, num_conjugate_pairs: int = 8):
        """Initialize zeta normal system with conjugate frequency analysis"""
        self.max_trace_value = max_trace_value
        self.num_conjugate_pairs = num_conjugate_pairs
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.normal_cache = {}
        self.conjugate_cache = {}
        self.canonical_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.conjugate_pairs = self._build_conjugate_pairs()
        self.normal_forms = self._compute_normal_forms()
        
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
                normal_data = self._analyze_normal_properties(trace, n)
                universe[n] = normal_data
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
        
    def _analyze_normal_properties(self, trace: str, value: int) -> Dict:
        """分析trace的normal form性质，用于conjugate frequency分析"""
        # Core normalization analysis
        canonical_form = self._compute_canonical_form(trace)
        conjugate_frequency = self._compute_conjugate_frequency(trace, value)
        normalization_weight = self._compute_normalization_weight(trace)
        spectral_signature = self._compute_spectral_signature(trace, value)
        
        # Normal form specific properties
        form_complexity = self._compute_form_complexity(canonical_form)
        conjugate_coupling = self._compute_conjugate_coupling(trace, value)
        normal_stability = self._compute_normal_stability(canonical_form)
        canonical_distance = self._compute_canonical_distance(trace, canonical_form)
        
        return {
            'value': value,
            'trace': trace,
            'canonical_form': canonical_form,
            'conjugate_frequency': conjugate_frequency,
            'normalization_weight': normalization_weight,
            'spectral_signature': spectral_signature,
            'form_complexity': form_complexity,
            'conjugate_coupling': conjugate_coupling,
            'normal_stability': normal_stability,
            'canonical_distance': canonical_distance,
            'normal_classification': self._classify_normal_form(canonical_form, value),
            'conjugate_partner': self._find_conjugate_partner(conjugate_frequency)
        }
        
    def _compute_canonical_form(self, trace: str) -> str:
        """计算trace的canonical form（标准化形式）"""
        if not trace:
            return "0"
        
        # Normalize by removing leading zeros and applying φ-constraint reduction
        canonical = trace.lstrip("0") or "0"
        
        # Apply φ-constraint normalization rules
        while "11" in canonical:
            # Replace 11 with 101 (Fibonacci property)
            canonical = canonical.replace("11", "101", 1)
            
        # Reduce to minimal form
        while canonical.startswith("0") and len(canonical) > 1:
            canonical = canonical[1:]
            
        return canonical
        
    def _compute_conjugate_frequency(self, trace: str, value: int) -> float:
        """计算conjugate frequency（共轭频率）"""
        if value == 0:
            return 0.0
            
        # Conjugate frequency based on φ-relationship and trace structure
        base_freq = log(value + 1) / log(self.phi)
        
        # Modulation based on trace structure
        phase_factor = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                phase_factor += cos(2 * pi * i / len(trace))
                
        conjugate_freq = base_freq + phase_factor / len(trace)
        
        # Ensure conjugate pairing (map to [0, π] for conjugate symmetry)
        return (conjugate_freq % (2 * pi)) if conjugate_freq >= 0 else (2 * pi + conjugate_freq % (2 * pi))
        
    def _compute_normalization_weight(self, trace: str) -> float:
        """计算normalization weight（规范化权重）"""
        if not trace:
            return 0.0
            
        # Weight based on trace complexity and structure
        ones_count = trace.count('1')
        total_length = len(trace)
        
        if total_length == 0:
            return 0.0
            
        density_factor = ones_count / total_length
        length_factor = 1.0 / sqrt(total_length)
        
        return density_factor * length_factor
        
    def _compute_spectral_signature(self, trace: str, value: int) -> float:
        """计算spectral signature（谱特征）"""
        if value == 0:
            return 0.0
            
        # Signature based on harmonic content
        signature = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                harmonic = (i + 1) * self.phi
                signature += sin(2 * pi * value / harmonic) / (i + 1)
                
        return abs(signature) / len(trace) if trace else 0.0
        
    def _compute_form_complexity(self, canonical_form: str) -> int:
        """计算form complexity（形式复杂度）"""
        if not canonical_form:
            return 0
            
        # Complexity based on pattern variations
        complexity = 0
        
        # Count transitions
        for i in range(len(canonical_form) - 1):
            if canonical_form[i] != canonical_form[i + 1]:
                complexity += 1
                
        # Add pattern complexity
        ones_count = canonical_form.count('1')
        complexity += ones_count
        
        return complexity
        
    def _compute_conjugate_coupling(self, trace: str, value: int) -> float:
        """计算conjugate coupling strength（共轭耦合强度）"""
        if not trace or value == 0:
            return 0.0
            
        # Coupling based on symmetry properties
        coupling = 0.0
        
        # Check for palindromic patterns (self-conjugate tendency)
        for i in range(len(trace) // 2):
            if trace[i] == trace[-(i+1)]:
                coupling += 1.0
                
        coupling /= len(trace) // 2 if len(trace) > 1 else 1
        
        # Modulate by frequency
        conjugate_freq = self._compute_conjugate_frequency(trace, value)
        coupling *= abs(sin(conjugate_freq))
        
        return coupling
        
    def _compute_normal_stability(self, canonical_form: str) -> float:
        """计算normal form stability（标准形稳定性）"""
        if not canonical_form:
            return 0.0
            
        # Stability based on form entropy and structure
        ones_count = canonical_form.count('1')
        total_length = len(canonical_form)
        
        if total_length == 0:
            return 0.0
            
        # Entropy-based stability
        if ones_count == 0 or ones_count == total_length:
            entropy = 0.0
        else:
            p1 = ones_count / total_length
            p0 = 1 - p1
            entropy = -(p1 * log2(p1) + p0 * log2(p0))
            
        # Stability inversely related to entropy (more organized = more stable)
        max_entropy = 1.0  # Maximum entropy for binary
        stability = 1.0 - (entropy / max_entropy)
        
        return stability
        
    def _compute_canonical_distance(self, trace: str, canonical_form: str) -> int:
        """计算到canonical form的距离"""
        if trace == canonical_form:
            return 0
            
        # Simple edit distance approximation
        return abs(len(trace) - len(canonical_form)) + sum(1 for i in range(min(len(trace), len(canonical_form))) 
                                                         if trace[i] != canonical_form[i])
        
    def _classify_normal_form(self, canonical_form: str, value: int) -> str:
        """分类normal form类型"""
        complexity = self._compute_form_complexity(canonical_form)
        stability = self._compute_normal_stability(canonical_form)
        
        if complexity <= 2:
            return "simple_normal"
        elif stability > 0.7:
            return "stable_normal"
        elif complexity > 6:
            return "complex_normal"
        else:
            return "balanced_normal"
            
    def _find_conjugate_partner(self, frequency: float) -> float:
        """寻找conjugate partner frequency"""
        # Conjugate partner is π - frequency (for conjugate symmetry)
        return pi - frequency if frequency <= pi else 2*pi - frequency
        
    def _build_conjugate_pairs(self) -> List[Tuple]:
        """构建conjugate frequency pairs"""
        pairs = []
        
        # Group traces by conjugate frequency regions
        freq_groups = {}
        for value, data in self.trace_universe.items():
            freq = data['conjugate_frequency']
            region = int(freq * self.num_conjugate_pairs / (2 * pi))
            if region not in freq_groups:
                freq_groups[region] = []
            freq_groups[region].append(value)
            
        # Form pairs within regions
        for region, traces in freq_groups.items():
            for i in range(0, len(traces) - 1, 2):
                if i + 1 < len(traces):
                    pairs.append((traces[i], traces[i + 1]))
                    
        return pairs
        
    def _compute_normal_forms(self) -> Dict:
        """计算系统中所有normal forms的统计"""
        forms = {}
        canonical_counts = {}
        
        for value, data in self.trace_universe.items():
            canonical = data['canonical_form']
            classification = data['normal_classification']
            
            if canonical not in canonical_counts:
                canonical_counts[canonical] = 0
            canonical_counts[canonical] += 1
            
            if classification not in forms:
                forms[classification] = []
            forms[classification].append(value)
            
        return {
            'canonical_distribution': canonical_counts,
            'classification_groups': forms,
            'total_unique_forms': len(canonical_counts),
            'most_common_form': max(canonical_counts.items(), key=lambda x: x[1]) if canonical_counts else None
        }
        
    def analyze_normalization_properties(self) -> Dict:
        """分析normalization的全局性质"""
        all_traces = list(self.trace_universe.values())
        
        # Statistical analysis
        weights = [t['normalization_weight'] for t in all_traces]
        frequencies = [t['conjugate_frequency'] for t in all_traces]
        stabilities = [t['normal_stability'] for t in all_traces]
        complexities = [t['form_complexity'] for t in all_traces]
        signatures = [t['spectral_signature'] for t in all_traces]
        distances = [t['canonical_distance'] for t in all_traces]
        
        # Conjugate analysis
        couplings = [t['conjugate_coupling'] for t in all_traces]
        
        # Classification distribution
        classifications = [t['normal_classification'] for t in all_traces]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        return {
            'total_traces': len(all_traces),
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            },
            'frequency_stats': {
                'mean': np.mean(frequencies),
                'std': np.std(frequencies),
                'range': np.max(frequencies) - np.min(frequencies)
            },
            'stability_stats': {
                'mean': np.mean(stabilities),
                'std': np.std(stabilities)
            },
            'complexity_stats': {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'max': np.max(complexities)
            },
            'signature_stats': {
                'mean': np.mean(signatures),
                'std': np.std(signatures)
            },
            'coupling_stats': {
                'mean': np.mean(couplings),
                'std': np.std(couplings)
            },
            'distance_stats': {
                'mean': np.mean(distances),
                'std': np.std(distances)
            },
            'classification_distribution': class_counts,
            'conjugate_pairs': len(self.conjugate_pairs),
            'normal_forms': self.normal_forms
        }
        
    def compute_entropy_measures(self) -> Dict:
        """计算normal form系统的信息论测度"""
        all_traces = list(self.trace_universe.values())
        
        # Extract properties for entropy calculation
        properties = {
            'normalization_weight': [t['normalization_weight'] for t in all_traces],
            'conjugate_frequency': [t['conjugate_frequency'] for t in all_traces],
            'spectral_signature': [t['spectral_signature'] for t in all_traces],
            'form_complexity': [t['form_complexity'] for t in all_traces],
            'conjugate_coupling': [t['conjugate_coupling'] for t in all_traces],
            'normal_stability': [t['normal_stability'] for t in all_traces],
            'canonical_distance': [t['canonical_distance'] for t in all_traces],
            'normal_classification': [t['normal_classification'] for t in all_traces],
            'canonical_form': [t['canonical_form'] for t in all_traces]
        }
        
        entropies = {}
        for prop_name, values in properties.items():
            if prop_name in ['normal_classification', 'canonical_form']:
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
            
    def build_conjugate_network(self) -> nx.Graph:
        """构建conjugate frequency网络"""
        G = nx.Graph()
        
        # Add nodes for each trace
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add edges between conjugate pairs and similar frequencies
        traces = list(self.trace_universe.items())
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces[i+1:], i+1):
                # Connect if frequencies are conjugate or similar
                freq_diff = abs(data1['conjugate_frequency'] - data2['conjugate_frequency'])
                conjugate_diff = abs(data1['conjugate_frequency'] - data2['conjugate_partner'])
                
                coupling_strength = max(data1['conjugate_coupling'], data2['conjugate_coupling'])
                
                if (freq_diff < 0.5 or conjugate_diff < 0.5) and coupling_strength > 0.1:
                    weight = coupling_strength * exp(-min(freq_diff, conjugate_diff))
                    G.add_edge(val1, val2, weight=weight)
                    
        return G
        
    def analyze_categorical_structure(self) -> Dict:
        """分析normal form系统的范畴论结构"""
        # Group traces by normal form classification
        categories = {}
        for value, data in self.trace_universe.items():
            cls = data['normal_classification']
            if cls not in categories:
                categories[cls] = []
            categories[cls].append(value)
            
        # Count morphisms (connections) between categories
        G = self.build_conjugate_network()
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

class TestZetaNormal(unittest.TestCase):
    """测试zeta normal系统的各项功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = ZetaNormalSystem(max_trace_value=80, num_conjugate_pairs=6)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 10)
        
        # 检查基本trace性质
        for value, data in self.system.trace_universe.items():
            self.assertIn('canonical_form', data)
            self.assertIn('conjugate_frequency', data)
            self.assertIn('normal_classification', data)
            self.assertIn('normalization_weight', data)
            
    def test_canonical_form_computation(self):
        """测试canonical form计算"""
        # Test basic cases
        self.assertEqual(self.system._compute_canonical_form("101"), "101")
        self.assertEqual(self.system._compute_canonical_form("0101"), "101")
        
    def test_conjugate_frequency_analysis(self):
        """测试conjugate frequency分析"""
        for value, data in self.system.trace_universe.items():
            freq = data['conjugate_frequency']
            partner = data['conjugate_partner']
            
            # Check frequency bounds
            self.assertGreaterEqual(freq, 0.0)
            self.assertLessEqual(freq, 2 * pi)
            
            # Check conjugate relationship
            self.assertGreaterEqual(partner, 0.0)
            self.assertLessEqual(partner, 2 * pi)
            
    def test_normalization_properties(self):
        """测试normalization性质"""
        props = self.system.analyze_normalization_properties()
        
        self.assertIn('weight_stats', props)
        self.assertIn('stability_stats', props)
        self.assertIn('complexity_stats', props)
        self.assertGreater(props['total_traces'], 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropies = self.system.compute_entropy_measures()
        
        required_entropies = [
            'normalization_weight', 'conjugate_frequency', 'spectral_signature',
            'normal_stability', 'normal_classification'
        ]
        
        for entropy_name in required_entropies:
            self.assertIn(entropy_name, entropies)
            self.assertGreaterEqual(entropies[entropy_name], 0.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_conjugate_network()
        
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
        # 检查节点属性
        for node in G.nodes():
            self.assertIn('conjugate_frequency', G.nodes[node])
            self.assertIn('normal_classification', G.nodes[node])
            
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
    print("Chapter 091: ZetaNormal Verification")
    print("从ψ=ψ(ψ)推导Spectral Normal Form Collapse across Conjugate Frequencies")
    print("="*80)
    
    # 创建系统
    system = ZetaNormalSystem(max_trace_value=85, num_conjugate_pairs=8)
    
    # 1. 基础统计
    print("\n1. Normal Form Foundation Analysis:")
    print("-" * 50)
    normal_props = system.analyze_normalization_properties()
    
    print(f"Total traces analyzed: {normal_props['total_traces']}")
    print(f"Weight statistics: mean={normal_props['weight_stats']['mean']:.3f}, "
          f"std={normal_props['weight_stats']['std']:.3f}")
    print(f"Frequency range: {normal_props['frequency_stats']['range']:.3f}")
    print(f"Mean stability: {normal_props['stability_stats']['mean']:.3f}")
    print(f"Max complexity: {normal_props['complexity_stats']['max']}")
    print(f"Mean conjugate coupling: {normal_props['coupling_stats']['mean']:.3f}")
    print(f"Mean canonical distance: {normal_props['distance_stats']['mean']:.3f}")
    
    print("\nNormal Form Classification Distribution:")
    for cls, count in normal_props['classification_distribution'].items():
        percentage = (count / normal_props['total_traces']) * 100
        print(f"- {cls}: {count} traces ({percentage:.1f}%)")
        
    print(f"\nConjugate pairs identified: {normal_props['conjugate_pairs']}")
    print(f"Unique canonical forms: {normal_props['normal_forms']['total_unique_forms']}")
    
    if normal_props['normal_forms']['most_common_form']:
        most_common, count = normal_props['normal_forms']['most_common_form']
        print(f"Most common canonical form: '{most_common}' ({count} instances)")
        
    # 2. Normal form analysis
    print("\n2. Canonical Form Analysis:")
    print("-" * 50)
    normal_forms = normal_props['normal_forms']
    
    print(f"Total unique canonical forms: {normal_forms['total_unique_forms']}")
    
    print("\nCanonical Form Distribution (top 5):")
    sorted_forms = sorted(normal_forms['canonical_distribution'].items(), 
                         key=lambda x: x[1], reverse=True)
    for i, (form, count) in enumerate(sorted_forms[:5]):
        percentage = (count / normal_props['total_traces']) * 100
        print(f"{i+1}. '{form}': {count} instances ({percentage:.1f}%)")
        
    print("\nClassification Groups:")
    for cls, traces in normal_forms['classification_groups'].items():
        print(f"- {cls}: {len(traces)} traces")
        if traces:
            # Show example traces
            examples = traces[:3]
            example_forms = [system.trace_universe[t]['canonical_form'] for t in examples]
            print(f"  Examples: {examples} -> {example_forms}")
    
    # 3. 信息论分析
    print("\n3. Information Theory Analysis:")
    print("-" * 50)
    entropies = system.compute_entropy_measures()
    
    for prop, entropy in entropies.items():
        print(f"{prop.replace('_', ' ').title()} entropy: {entropy:.3f} bits")
        
    # 4. 网络分析
    print("\n4. Graph Theory Analysis:")
    print("-" * 50)
    G = system.build_conjugate_network()
    
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        avg_weight = sum(edge_weights) / len(edge_weights)
        
        print(f"Average degree: {avg_degree:.3f}")
        print(f"Average edge weight: {avg_weight:.3f}")
        print(f"Connected components: {nx.number_connected_components(G)}")
        
        # Network density
        possible_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        density = G.number_of_edges() / possible_edges if possible_edges > 0 else 0
        print(f"Network density: {density:.3f}")
        
    # 5. 范畴论分析
    print("\n5. Category Theory Analysis:")
    print("-" * 50)
    cat_analysis = system.analyze_categorical_structure()
    
    print(f"Normal form categories: {cat_analysis['category_count']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for category, count in cat_analysis['categories'].items():
        print(f"- {category}: {count} objects")
        
    # 6. 可视化生成
    print("\n6. Visualization Generation:")
    print("-" * 50)
    
    try:
        # 创建normal form可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.1 Conjugate Frequency Distribution
        frequencies = [data['conjugate_frequency'] for data in system.trace_universe.values()]
        weights = [data['normalization_weight'] for data in system.trace_universe.values()]
        
        scatter = ax1.scatter(frequencies, weights, alpha=0.6, s=50)
        ax1.set_xlabel('Conjugate Frequency')
        ax1.set_ylabel('Normalization Weight')
        ax1.set_title('Conjugate Frequency vs Normalization Weight')
        ax1.grid(True, alpha=0.3)
        
        # 6.2 Conjugate Network
        if G.number_of_nodes() > 0:
            pos = {}
            for node in G.nodes():
                freq = G.nodes[node]['conjugate_frequency']
                weight = G.nodes[node]['normalization_weight']
                pos[node] = (freq, weight + random.uniform(-0.02, 0.02))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=30, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=0.5)
            
        ax2.set_xlabel('Conjugate Frequency')
        ax2.set_ylabel('Normalization Weight')
        ax2.set_title('Conjugate Network Structure')
        ax2.grid(True, alpha=0.3)
        
        # 6.3 Stability vs Complexity
        stabilities = [data['normal_stability'] for data in system.trace_universe.values()]
        complexities = [data['form_complexity'] for data in system.trace_universe.values()]
        classifications = [data['normal_classification'] for data in system.trace_universe.values()]
        
        # Color by classification
        class_to_color = {cls: i for i, cls in enumerate(set(classifications))}
        colors = [class_to_color[cls] for cls in classifications]
        
        scatter = ax3.scatter(complexities, stabilities, c=colors, cmap='tab10', alpha=0.7)
        ax3.set_xlabel('Form Complexity')
        ax3.set_ylabel('Normal Stability')
        ax3.set_title('Normal Form Stability vs Complexity')
        ax3.grid(True, alpha=0.3)
        
        # 6.4 Canonical Distance Distribution
        distances = [data['canonical_distance'] for data in system.trace_universe.values()]
        
        ax4.hist(distances, bins=max(1, len(set(distances))), alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Canonical Distance')
        ax4.set_ylabel('Count')
        ax4.set_title('Canonical Distance Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-091-zeta-normal-forms.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Normal form visualization saved")
        
        # 创建频率分析可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.5 Spectral Signature Analysis
        signatures = [data['spectral_signature'] for data in system.trace_universe.values()]
        
        ax1.hist(signatures, bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Spectral Signature')
        ax1.set_ylabel('Count')
        ax1.set_title('Spectral Signature Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 6.6 Conjugate Coupling vs Frequency
        couplings = [data['conjugate_coupling'] for data in system.trace_universe.values()]
        
        scatter = ax2.scatter(frequencies, couplings, c=colors, cmap='tab10', alpha=0.7)
        ax2.set_xlabel('Conjugate Frequency')
        ax2.set_ylabel('Conjugate Coupling')
        ax2.set_title('Conjugate Coupling vs Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 6.7 Normal Form Classification Distribution
        classification_counts = list(normal_props['classification_distribution'].values())
        classification_labels = list(normal_props['classification_distribution'].keys())
        
        ax3.pie(classification_counts, labels=classification_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Normal Form Classification Distribution')
        
        # 6.8 Canonical Form Frequency Distribution
        if normal_forms['canonical_distribution']:
            # Show top canonical forms
            top_forms = sorted(normal_forms['canonical_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)[:8]
            
            forms = [f[0] if len(f[0]) <= 6 else f[0][:6]+'...' for f in top_forms]
            counts = [f[1] for f in top_forms]
            
            bars = ax4.bar(range(len(forms)), counts, alpha=0.7)
            ax4.set_xticks(range(len(forms)))
            ax4.set_xticklabels(forms, rotation=45, ha='right')
            ax4.set_ylabel('Count')
            ax4.set_title('Top Canonical Forms')
            
            # Color bars by frequency
            max_count = max(counts)
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(counts[i] / max_count))
                
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-091-zeta-normal-frequency.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Frequency analysis visualization saved")
        
        # 创建网络和范畴论可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 6.9 Network visualization with conjugate pairs
        if G.number_of_nodes() > 0:
            # Color nodes by classification
            node_colors = [class_to_color[G.nodes[node]['normal_classification']] for node in G.nodes()]
            node_sizes = [G.nodes[node]['conjugate_coupling'] * 100 + 20 for node in G.nodes()]
            
            pos_circular = nx.circular_layout(G)
            
            scatter = ax1.scatter([pos_circular[node][0] for node in G.nodes()],
                                [pos_circular[node][1] for node in G.nodes()],
                                c=node_colors, s=node_sizes, cmap='tab10', alpha=0.8)
            
            # Draw edges
            for edge in G.edges():
                x_coords = [pos_circular[edge[0]][0], pos_circular[edge[1]][0]]
                y_coords = [pos_circular[edge[0]][1], pos_circular[edge[1]][1]]
                ax1.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
                
        ax1.set_title('Conjugate Network Structure')
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
        max_entropy = max(entropy_values) if entropy_values else 1
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(entropy_values[i] / max_entropy))
            
        # 6.12 Conjugate pairing analysis
        pair_strengths = []
        for pair in system.conjugate_pairs:
            if len(pair) == 2:
                data1 = system.trace_universe.get(pair[0], {})
                data2 = system.trace_universe.get(pair[1], {})
                if data1 and data2:
                    strength = (data1.get('conjugate_coupling', 0) + data2.get('conjugate_coupling', 0)) / 2
                    pair_strengths.append(strength)
                    
        if pair_strengths:
            ax4.hist(pair_strengths, bins=10, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Conjugate Pair Strength')
            ax4.set_ylabel('Count')
            ax4.set_title('Conjugate Pair Strength Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-091-zeta-normal-network.png',
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
    print("ZetaNormal Verification Complete")
    print("Key Findings:")
    print(f"- {normal_props['total_traces']} φ-valid traces with {normal_forms['total_unique_forms']} unique canonical forms")
    print(f"- {normal_props['conjugate_pairs']} conjugate pairs identified")
    print(f"- Mean normalization weight: {normal_props['weight_stats']['mean']:.3f}")
    print(f"- Mean normal stability: {normal_props['stability_stats']['mean']:.3f}")
    print(f"- {cat_analysis['category_count']} normal form categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    if normal_forms['most_common_form']:
        most_common, count = normal_forms['most_common_form']
        print(f"- Most common canonical form: '{most_common}' ({count} instances)")
    print("="*80)

if __name__ == "__main__":
    run_verification()