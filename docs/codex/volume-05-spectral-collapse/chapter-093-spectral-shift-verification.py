#!/usr/bin/env python3
"""
Chapter 093: SpectralShift Unit Test Verification
从ψ=ψ(ψ)推导Structural Phase Dynamics under ζ-Tuned Collapse

Core principle: From ψ = ψ(ψ) derive phase transition dynamics where spectral
parameters undergo systematic shifts through ζ-function tuning, creating
discrete phase states with critical transition points that reveal the
fundamental phase architecture of collapsed spectral space through entropy-
increasing tensor transformations.

This verification program implements:
1. φ-constrained phase state detection through ζ-tuning analysis
2. Spectral shift dynamics: systematic parameter evolution and transitions
3. Three-domain analysis: Traditional vs φ-constrained vs intersection phase theory
4. Graph theory analysis of phase networks and transition pathways
5. Information theory analysis of phase entropy and transition encoding
6. Category theory analysis of phase functors and transition morphisms
7. Visualization of phase transitions and ζ-tuned spectral shifts
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

class SpectralShiftSystem:
    """
    Core system for implementing structural phase dynamics under ζ-tuned collapse.
    Implements φ-constrained phase transitions via ζ-function parameter tuning.
    """
    
    def __init__(self, max_trace_value: int = 110, num_phase_states: int = 12):
        """Initialize spectral shift system with phase transition analysis"""
        self.max_trace_value = max_trace_value
        self.num_phase_states = num_phase_states
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.phase_cache = {}
        self.shift_cache = {}
        self.transition_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.phase_states = self._detect_phase_states()
        self.transition_matrix = self._compute_transition_matrix()
        
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
                phase_data = self._analyze_phase_properties(trace, n)
                universe[n] = phase_data
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
        
    def _analyze_phase_properties(self, trace: str, value: int) -> Dict:
        """分析trace的phase properties，用于spectral shift分析"""
        # Core phase analysis
        zeta_parameter = self._compute_zeta_parameter(trace, value)
        phase_order = self._compute_phase_order(trace, value)
        transition_potential = self._compute_transition_potential(trace)
        spectral_shift = self._compute_spectral_shift(trace, value)
        
        # Phase-specific properties
        phase_stability = self._compute_phase_stability(trace, value)
        critical_field = self._compute_critical_field(trace)
        order_parameter = self._compute_order_parameter(trace, value)
        entropy_production = self._compute_entropy_production(trace, value)
        
        return {
            'value': value,
            'trace': trace,
            'zeta_parameter': zeta_parameter,
            'phase_order': phase_order,
            'transition_potential': transition_potential,
            'spectral_shift': spectral_shift,
            'phase_stability': phase_stability,
            'critical_field': critical_field,
            'order_parameter': order_parameter,
            'entropy_production': entropy_production,
            'phase_classification': self._classify_phase_state(trace, value),
            'transition_pathway': self._determine_transition_pathway(phase_order, zeta_parameter)
        }
        
    def _compute_zeta_parameter(self, trace: str, value: int) -> float:
        """计算ζ-tuning parameter（ζ调控参数）"""
        if value == 0:
            return 0.5  # Critical line default
            
        # ζ-parameter based on trace structure and critical line proximity
        # Real part modulation around critical line s = 1/2 + it
        real_modulation = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Modulation based on Fibonacci positioning
                real_modulation += ((-1) ** i) / (2 ** (i + 1))
                
        # Keep near critical line with trace-dependent variation
        zeta_param = 0.5 + real_modulation * 0.1  # Small perturbation around 1/2
        
        return max(0.1, min(0.9, zeta_param))  # Bounded in [0.1, 0.9]
        
    def _compute_phase_order(self, trace: str, value: int) -> float:
        """计算phase order parameter（相序参数）"""
        if not trace:
            return 0.0
            
        # Order parameter based on trace organization and φ-constraint satisfaction
        ones_count = trace.count('1')
        trace_length = len(trace)
        
        if trace_length == 0:
            return 0.0
            
        # Base order from density
        density_order = ones_count / trace_length
        
        # Enhancement from structure
        structure_factor = 0.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i + 1] == '0':
                structure_factor += 1.0  # Preferred 10 pattern
            elif trace[i] == '0' and trace[i + 1] == '1':
                structure_factor += 0.5  # Acceptable 01 pattern
                
        structure_order = structure_factor / max(1, len(trace) - 1)
        
        # φ-constraint enhancement
        phi_enhancement = self.phi if "11" not in trace else 1.0
        
        order = (density_order + structure_order) * phi_enhancement / 3.0
        
        return min(order, 1.0)
        
    def _compute_transition_potential(self, trace: str) -> float:
        """计算transition potential（相变势能）"""
        if not trace:
            return 0.0
            
        # Potential based on structural instabilities and constraint violations
        potential = 0.0
        
        # Edge effects (boundary instabilities)
        if trace.startswith('1'):
            potential += 0.1
        if trace.endswith('1'):
            potential += 0.1
            
        # Pattern instabilities
        for i in range(len(trace) - 2):
            pattern = trace[i:i+3]
            if pattern == "101":  # Stable pattern
                potential -= 0.05
            elif pattern == "010":  # Metastable pattern
                potential += 0.03
            elif pattern == "100" or pattern == "001":  # Edge patterns
                potential += 0.02
                
        # Length-dependent normalization
        potential /= sqrt(len(trace))
        
        return max(0.0, potential)
        
    def _compute_spectral_shift(self, trace: str, value: int) -> float:
        """计算spectral shift magnitude（谱偏移幅度）"""
        if value == 0:
            return 0.0
            
        # Shift based on value evolution and trace transformation
        base_shift = log(value + 1) / log(self.phi)
        
        # Modulation by trace structure
        trace_modulation = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                # Harmonic shift contribution
                trace_modulation += sin(2 * pi * (i + 1) / len(trace)) / (i + 1)
                
        spectral_shift = base_shift + abs(trace_modulation)
        
        # Normalize to [0, 2π] range
        return spectral_shift % (2 * pi)
        
    def _compute_phase_stability(self, trace: str, value: int) -> float:
        """计算phase stability（相稳定性）"""
        if not trace:
            return 0.0
            
        # Stability based on order parameter and transition potential
        order = self._compute_phase_order(trace, value)
        potential = self._compute_transition_potential(trace)
        
        # Higher order and lower potential = higher stability
        stability = order / (1.0 + potential)
        
        # φ-constraint bonus
        if "11" not in trace:
            stability *= self.phi
            
        return min(stability, 1.0)
        
    def _compute_critical_field(self, trace: str) -> float:
        """计算critical field strength（临界场强）"""
        if not trace:
            return 1.0
            
        # Critical field needed to induce phase transition
        # Based on trace length and structure complexity
        
        complexity = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                complexity += 1
                
        # Longer, more complex traces need stronger fields
        field_strength = (complexity + 1) / len(trace)
        
        # φ-constraint modification
        if "11" not in trace:
            field_strength /= self.phi  # Easier transitions for φ-valid traces
            
        return field_strength
        
    def _compute_order_parameter(self, trace: str, value: int) -> float:
        """计算order parameter（序参量）"""
        # This is the same as phase_order but with different normalization
        phase_order = self._compute_phase_order(trace, value)
        
        # Map to typical order parameter range [-1, 1]
        order_param = 2 * phase_order - 1
        
        return order_param
        
    def _compute_entropy_production(self, trace: str, value: int) -> float:
        """计算entropy production rate（熵产生率）"""
        if not trace or value == 0:
            return 0.0
            
        # Entropy production from trace evolution and transitions
        # Based on information content and structural changes
        
        # Information entropy of trace
        ones_count = trace.count('1')
        zeros_count = len(trace) - ones_count
        
        if ones_count == 0 or zeros_count == 0:
            entropy = 0.0
        else:
            p1 = ones_count / len(trace)
            p0 = 1 - p1
            entropy = -(p1 * log2(p1) + p0 * log2(p0))
            
        # Production rate modulated by transition potential
        potential = self._compute_transition_potential(trace)
        production_rate = entropy * potential * value / 100.0
        
        return production_rate
        
    def _classify_phase_state(self, trace: str, value: int) -> str:
        """分类phase state类型"""
        order = self._compute_phase_order(trace, value)
        stability = self._compute_phase_stability(trace, value)
        potential = self._compute_transition_potential(trace)
        
        if order > 0.7 and stability > 0.6:
            return "ordered_phase"
        elif order < 0.3 and potential > 0.3:
            return "disordered_phase"
        elif potential > 0.5:
            return "critical_phase"
        else:
            return "metastable_phase"
            
    def _determine_transition_pathway(self, phase_order: float, zeta_param: float) -> int:
        """确定transition pathway index"""
        # Map phase space to discrete transition pathways
        order_bin = int(phase_order * 4)  # 0-3
        param_bin = int((zeta_param - 0.1) / 0.8 * 3)  # 0-2 (for range 0.1-0.9)
        
        pathway = order_bin * 3 + param_bin
        return min(pathway, self.num_phase_states - 1)
        
    def _detect_phase_states(self) -> Dict:
        """检测系统中的phase states"""
        states = {}
        
        # Group traces by phase classification and transition pathway
        for value, data in self.trace_universe.items():
            classification = data['phase_classification']
            pathway = data['transition_pathway']
            
            key = (classification, pathway)
            if key not in states:
                states[key] = []
            states[key].append(value)
            
        # Analyze each phase state
        state_analysis = {}
        for (classification, pathway), traces in states.items():
            if not traces:
                continue
                
            state_data = [self.trace_universe[t] for t in traces]
            
            analysis = {
                'classification': classification,
                'pathway': pathway,
                'trace_count': len(traces),
                'mean_order': np.mean([d['phase_order'] for d in state_data]),
                'mean_stability': np.mean([d['phase_stability'] for d in state_data]),
                'mean_shift': np.mean([d['spectral_shift'] for d in state_data]),
                'mean_entropy_production': np.mean([d['entropy_production'] for d in state_data]),
                'zeta_param_range': (
                    np.min([d['zeta_parameter'] for d in state_data]),
                    np.max([d['zeta_parameter'] for d in state_data])
                )
            }
            
            state_analysis[f"{classification}_{pathway}"] = analysis
            
        return state_analysis
        
    def _compute_transition_matrix(self) -> np.ndarray:
        """计算phase transition matrix"""
        # Initialize transition matrix
        matrix = np.zeros((self.num_phase_states, self.num_phase_states))
        
        # Compute transition probabilities based on trace relationships
        traces = list(self.trace_universe.items())
        
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces):
                if i != j:
                    # Transition probability based on parameter similarity
                    param_diff = abs(data1['zeta_parameter'] - data2['zeta_parameter'])
                    order_diff = abs(data1['phase_order'] - data2['phase_order'])
                    
                    # Similar states have higher transition probability
                    transition_prob = exp(-(param_diff + order_diff))
                    
                    pathway1 = data1['transition_pathway']
                    pathway2 = data2['transition_pathway']
                    
                    matrix[pathway1, pathway2] += transition_prob
                    
        # Normalize rows to make proper transition probabilities
        for i in range(self.num_phase_states):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:
                matrix[i, :] /= row_sum
                
        return matrix
        
    def analyze_phase_properties(self) -> Dict:
        """分析phase system的全局性质"""
        all_traces = list(self.trace_universe.values())
        
        # Statistical analysis
        zeta_params = [t['zeta_parameter'] for t in all_traces]
        phase_orders = [t['phase_order'] for t in all_traces]
        stabilities = [t['phase_stability'] for t in all_traces]
        shifts = [t['spectral_shift'] for t in all_traces]
        potentials = [t['transition_potential'] for t in all_traces]
        entropies = [t['entropy_production'] for t in all_traces]
        
        # Classification distribution
        classifications = [t['phase_classification'] for t in all_traces]
        class_counts = {}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        # Pathway distribution
        pathways = [t['transition_pathway'] for t in all_traces]
        pathway_counts = {}
        for pathway in pathways:
            pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
            
        return {
            'total_traces': len(all_traces),
            'zeta_param_stats': {
                'mean': np.mean(zeta_params),
                'std': np.std(zeta_params),
                'range': np.max(zeta_params) - np.min(zeta_params)
            },
            'phase_order_stats': {
                'mean': np.mean(phase_orders),
                'std': np.std(phase_orders),
                'max': np.max(phase_orders)
            },
            'stability_stats': {
                'mean': np.mean(stabilities),
                'std': np.std(stabilities)
            },
            'shift_stats': {
                'mean': np.mean(shifts),
                'std': np.std(shifts),
                'range': np.max(shifts) - np.min(shifts)
            },
            'potential_stats': {
                'mean': np.mean(potentials),
                'std': np.std(potentials)
            },
            'entropy_stats': {
                'mean': np.mean(entropies),
                'std': np.std(entropies)
            },
            'classification_distribution': class_counts,
            'pathway_distribution': pathway_counts,
            'phase_states': self.phase_states,
            'transition_matrix': self.transition_matrix
        }
        
    def compute_entropy_measures(self) -> Dict:
        """计算phase system的信息论测度"""
        all_traces = list(self.trace_universe.values())
        
        # Extract properties for entropy calculation
        properties = {
            'zeta_parameter': [t['zeta_parameter'] for t in all_traces],
            'phase_order': [t['phase_order'] for t in all_traces],
            'transition_potential': [t['transition_potential'] for t in all_traces],
            'spectral_shift': [t['spectral_shift'] for t in all_traces],
            'phase_stability': [t['phase_stability'] for t in all_traces],
            'critical_field': [t['critical_field'] for t in all_traces],
            'order_parameter': [t['order_parameter'] for t in all_traces],
            'entropy_production': [t['entropy_production'] for t in all_traces],
            'transition_pathway': [t['transition_pathway'] for t in all_traces],
            'phase_classification': [t['phase_classification'] for t in all_traces]
        }
        
        entropies = {}
        for prop_name, values in properties.items():
            if prop_name in ['transition_pathway', 'phase_classification']:
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
            
    def build_phase_network(self) -> nx.Graph:
        """构建phase transition network"""
        G = nx.Graph()
        
        # Add nodes for each trace
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add edges between traces with similar phase properties
        traces = list(self.trace_universe.items())
        for i, (val1, data1) in enumerate(traces):
            for j, (val2, data2) in enumerate(traces[i+1:], i+1):
                # Connect if phase parameters are similar
                param_diff = abs(data1['zeta_parameter'] - data2['zeta_parameter'])
                order_diff = abs(data1['phase_order'] - data2['phase_order'])
                stability_product = data1['phase_stability'] * data2['phase_stability']
                
                if param_diff < 0.2 and order_diff < 0.3 and stability_product > 0.1:
                    weight = stability_product * exp(-param_diff - order_diff)
                    G.add_edge(val1, val2, weight=weight)
                    
        return G
        
    def analyze_categorical_structure(self) -> Dict:
        """分析phase system的范畴论结构"""
        # Group traces by phase classification
        categories = {}
        for value, data in self.trace_universe.items():
            cls = data['phase_classification']
            if cls not in categories:
                categories[cls] = []
            categories[cls].append(value)
            
        # Count morphisms (connections) between categories
        G = self.build_phase_network()
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

class TestSpectralShift(unittest.TestCase):
    """测试spectral shift系统的各项功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = SpectralShiftSystem(max_trace_value=100, num_phase_states=10)
        
    def test_trace_universe_construction(self):
        """测试trace universe构建"""
        self.assertGreater(len(self.system.trace_universe), 10)
        
        # 检查基本trace性质
        for value, data in self.system.trace_universe.items():
            self.assertIn('zeta_parameter', data)
            self.assertIn('phase_order', data)
            self.assertIn('phase_classification', data)
            self.assertIn('spectral_shift', data)
            
    def test_zeta_parameter_computation(self):
        """测试ζ参数计算"""
        for value, data in self.system.trace_universe.items():
            param = data['zeta_parameter']
            
            # Check parameter bounds
            self.assertGreaterEqual(param, 0.1)
            self.assertLessEqual(param, 0.9)
            
    def test_phase_state_detection(self):
        """测试phase state检测"""
        states = self.system.phase_states
        
        self.assertIsInstance(states, dict)
        
        # Check state properties
        for state_key, analysis in states.items():
            self.assertIn('trace_count', analysis)
            self.assertIn('mean_order', analysis)
            self.assertIn('mean_stability', analysis)
            self.assertGreaterEqual(analysis['mean_order'], 0.0)
            self.assertLessEqual(analysis['mean_order'], 1.0)
            
    def test_transition_matrix_computation(self):
        """测试transition matrix计算"""
        matrix = self.system.transition_matrix
        
        self.assertEqual(matrix.shape, (self.system.num_phase_states, self.system.num_phase_states))
        
        # Check that rows sum to approximately 1 (transition probabilities)
        for i in range(self.system.num_phase_states):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:  # Some rows might be zero if no traces in that state
                self.assertAlmostEqual(row_sum, 1.0, places=5)
                
    def test_phase_properties(self):
        """测试phase性质"""
        props = self.system.analyze_phase_properties()
        
        self.assertIn('zeta_param_stats', props)
        self.assertIn('phase_order_stats', props)
        self.assertIn('classification_distribution', props)
        self.assertGreater(props['total_traces'], 0)
        
    def test_entropy_computation(self):
        """测试熵计算"""
        entropies = self.system.compute_entropy_measures()
        
        required_entropies = [
            'zeta_parameter', 'phase_order', 'spectral_shift',
            'phase_stability', 'phase_classification'
        ]
        
        for entropy_name in required_entropies:
            self.assertIn(entropy_name, entropies)
            self.assertGreaterEqual(entropies[entropy_name], 0.0)
            
    def test_network_construction(self):
        """测试网络构建"""
        G = self.system.build_phase_network()
        
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
        # 检查节点属性
        for node in G.nodes():
            self.assertIn('zeta_parameter', G.nodes[node])
            self.assertIn('phase_classification', G.nodes[node])
            
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
    print("Chapter 093: SpectralShift Verification")
    print("从ψ=ψ(ψ)推导Structural Phase Dynamics under ζ-Tuned Collapse")
    print("="*80)
    
    # 创建系统
    system = SpectralShiftSystem(max_trace_value=105, num_phase_states=12)
    
    # 1. 基础统计
    print("\n1. Phase Dynamics Foundation Analysis:")
    print("-" * 50)
    phase_props = system.analyze_phase_properties()
    
    print(f"Total traces analyzed: {phase_props['total_traces']}")
    print(f"ζ-parameter statistics: mean={phase_props['zeta_param_stats']['mean']:.3f}, "
          f"std={phase_props['zeta_param_stats']['std']:.3f}")
    print(f"Phase order: mean={phase_props['phase_order_stats']['mean']:.3f}, "
          f"max={phase_props['phase_order_stats']['max']:.3f}")
    print(f"Mean stability: {phase_props['stability_stats']['mean']:.3f}")
    print(f"Spectral shift range: {phase_props['shift_stats']['range']:.3f}")
    print(f"Mean transition potential: {phase_props['potential_stats']['mean']:.3f}")
    print(f"Mean entropy production: {phase_props['entropy_stats']['mean']:.3f}")
    
    print("\nPhase Classification Distribution:")
    for cls, count in phase_props['classification_distribution'].items():
        percentage = (count / phase_props['total_traces']) * 100
        print(f"- {cls}: {count} traces ({percentage:.1f}%)")
        
    print("\nTransition Pathway Distribution:")
    for pathway, count in phase_props['pathway_distribution'].items():
        percentage = (count / phase_props['total_traces']) * 100
        print(f"- Pathway {pathway}: {count} traces ({percentage:.1f}%)")
        
    # 2. Phase state analysis
    print("\n2. Phase State Analysis:")
    print("-" * 50)
    phase_states = phase_props['phase_states']
    
    print(f"Detected phase states: {len(phase_states)}")
    
    for state_key, analysis in phase_states.items():
        print(f"\nState {state_key}:")
        print(f"  Classification: {analysis['classification']}")
        print(f"  Pathway: {analysis['pathway']}")
        print(f"  Trace count: {analysis['trace_count']}")
        print(f"  Mean order: {analysis['mean_order']:.3f}")
        print(f"  Mean stability: {analysis['mean_stability']:.3f}")
        print(f"  Mean shift: {analysis['mean_shift']:.3f}")
        print(f"  Mean entropy production: {analysis['mean_entropy_production']:.3f}")
        print(f"  ζ-parameter range: [{analysis['zeta_param_range'][0]:.3f}, {analysis['zeta_param_range'][1]:.3f}]")
        
    # 3. Transition matrix analysis
    print("\n3. Transition Matrix Analysis:")
    print("-" * 50)
    transition_matrix = phase_props['transition_matrix']
    
    print(f"Transition matrix shape: {transition_matrix.shape}")
    print(f"Matrix trace (self-transitions): {np.trace(transition_matrix):.3f}")
    print(f"Maximum transition probability: {np.max(transition_matrix):.3f}")
    print(f"Mean off-diagonal probability: {np.mean(transition_matrix[transition_matrix != np.diag(np.diag(transition_matrix))]):.3f}")
    
    # Find dominant transitions
    max_indices = np.unravel_index(np.argmax(transition_matrix), transition_matrix.shape)
    print(f"Strongest transition: {max_indices[0]} → {max_indices[1]} (prob: {transition_matrix[max_indices]:.3f})")
    
    # 4. 信息论分析
    print("\n4. Information Theory Analysis:")
    print("-" * 50)
    entropies = system.compute_entropy_measures()
    
    for prop, entropy in entropies.items():
        print(f"{prop.replace('_', ' ').title()} entropy: {entropy:.3f} bits")
        
    # 5. 网络分析
    print("\n5. Graph Theory Analysis:")
    print("-" * 50)
    G = system.build_phase_network()
    
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
        
    # 6. 范畴论分析
    print("\n6. Category Theory Analysis:")
    print("-" * 50)
    cat_analysis = system.analyze_categorical_structure()
    
    print(f"Phase categories: {cat_analysis['category_count']}")
    print(f"Total morphisms: {cat_analysis['total_morphisms']}")
    print(f"Morphism density: {cat_analysis['morphism_density']:.3f}")
    
    print("\nCategory Distribution:")
    for category, count in cat_analysis['categories'].items():
        print(f"- {category}: {count} objects")
        
    # 7. 可视化生成
    print("\n7. Visualization Generation:")
    print("-" * 50)
    
    try:
        # 创建phase dynamics可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.1 ζ-Parameter vs Phase Order
        zeta_params = [data['zeta_parameter'] for data in system.trace_universe.values()]
        phase_orders = [data['phase_order'] for data in system.trace_universe.values()]
        stabilities = [data['phase_stability'] for data in system.trace_universe.values()]
        
        scatter = ax1.scatter(zeta_params, phase_orders, c=stabilities, cmap='plasma', alpha=0.7, s=50)
        ax1.set_xlabel('ζ-Parameter')
        ax1.set_ylabel('Phase Order')
        ax1.set_title('ζ-Parameter vs Phase Order')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Phase Stability')
        
        # 7.2 Phase Network
        if G.number_of_nodes() > 0:
            pos = {}
            for node in G.nodes():
                zeta_param = G.nodes[node]['zeta_parameter']
                phase_order = G.nodes[node]['phase_order']
                pos[node] = (zeta_param, phase_order + random.uniform(-0.02, 0.02))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=30, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=0.5)
            
        ax2.set_xlabel('ζ-Parameter')
        ax2.set_ylabel('Phase Order')
        ax2.set_title('Phase Transition Network')
        ax2.grid(True, alpha=0.3)
        
        # 7.3 Spectral Shift vs Transition Potential
        shifts = [data['spectral_shift'] for data in system.trace_universe.values()]
        potentials = [data['transition_potential'] for data in system.trace_universe.values()]
        classifications = [data['phase_classification'] for data in system.trace_universe.values()]
        
        # Color by classification
        class_to_color = {cls: i for i, cls in enumerate(set(classifications))}
        colors = [class_to_color[cls] for cls in classifications]
        
        scatter = ax3.scatter(shifts, potentials, c=colors, cmap='tab10', alpha=0.7)
        ax3.set_xlabel('Spectral Shift')
        ax3.set_ylabel('Transition Potential')
        ax3.set_title('Spectral Shift vs Transition Potential')
        ax3.grid(True, alpha=0.3)
        
        # 7.4 Entropy Production Distribution
        entropies_prod = [data['entropy_production'] for data in system.trace_universe.values()]
        
        ax4.hist(entropies_prod, bins=15, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Entropy Production')
        ax4.set_ylabel('Count')
        ax4.set_title('Entropy Production Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-093-spectral-shift-dynamics.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Phase dynamics visualization saved")
        
        # 创建transition matrix可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.5 Transition Matrix Heatmap
        im = ax1.imshow(transition_matrix, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Target State')
        ax1.set_ylabel('Source State')
        ax1.set_title('Phase Transition Matrix')
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 7.6 Critical Field vs Order Parameter
        critical_fields = [data['critical_field'] for data in system.trace_universe.values()]
        order_params = [data['order_parameter'] for data in system.trace_universe.values()]
        
        scatter = ax2.scatter(critical_fields, order_params, c=colors, cmap='tab10', alpha=0.7)
        ax2.set_xlabel('Critical Field')
        ax2.set_ylabel('Order Parameter')
        ax2.set_title('Critical Field vs Order Parameter')
        ax2.grid(True, alpha=0.3)
        
        # 7.7 Phase Classification Distribution
        classification_counts = list(phase_props['classification_distribution'].values())
        classification_labels = list(phase_props['classification_distribution'].keys())
        
        ax3.pie(classification_counts, labels=classification_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Phase Classification Distribution')
        
        # 7.8 Pathway Distribution
        pathway_counts = list(phase_props['pathway_distribution'].values())
        pathway_labels = [f"Path {p}" for p in phase_props['pathway_distribution'].keys()]
        
        if len(pathway_labels) <= 10:  # Only show if manageable number
            bars = ax4.bar(pathway_labels, pathway_counts, alpha=0.7)
            ax4.set_ylabel('Count')
            ax4.set_title('Transition Pathway Distribution')
            ax4.tick_params(axis='x', rotation=45)
            
            # Color bars by pathway activity
            max_count = max(pathway_counts) if pathway_counts else 1
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(pathway_counts[i] / max_count))
        else:
            ax4.hist(list(phase_props['pathway_distribution'].keys()), 
                    weights=pathway_counts, bins=10, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Pathway Index')
            ax4.set_ylabel('Count')
            ax4.set_title('Pathway Distribution Histogram')
            
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-093-spectral-shift-transitions.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Transition analysis visualization saved")
        
        # 创建网络和范畴论可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 7.9 Network visualization by phase state
        if G.number_of_nodes() > 0:
            # Color nodes by phase classification
            node_colors = [class_to_color[G.nodes[node]['phase_classification']] for node in G.nodes()]
            node_sizes = [G.nodes[node]['phase_stability'] * 100 + 20 for node in G.nodes()]
            
            pos_spring = nx.spring_layout(G, k=1, iterations=50)
            
            scatter = ax1.scatter([pos_spring[node][0] for node in G.nodes()],
                                [pos_spring[node][1] for node in G.nodes()],
                                c=node_colors, s=node_sizes, cmap='tab10', alpha=0.8)
            
            # Draw edges
            for edge in G.edges():
                x_coords = [pos_spring[edge[0]][0], pos_spring[edge[1]][0]]
                y_coords = [pos_spring[edge[0]][1], pos_spring[edge[1]][1]]
                ax1.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
                
        ax1.set_title('Phase Network by Classification')
        ax1.set_xlabel('Network X-coordinate')
        ax1.set_ylabel('Network Y-coordinate')
        
        # 7.10 Morphism density by category
        if cat_analysis['categories']:
            categories = list(cat_analysis['categories'].keys())
            category_sizes = list(cat_analysis['categories'].values())
            
            ax2.bar(categories, category_sizes, alpha=0.7)
            ax2.set_ylabel('Object Count')
            ax2.set_title('Category Object Distribution')
            ax2.tick_params(axis='x', rotation=45)
            
        # 7.11 Entropy landscape
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
            
        # 7.12 Phase space trajectory
        # Create a 2D phase space plot
        ax4.scatter(zeta_params, phase_orders, c=shifts, cmap='coolwarm', alpha=0.7, s=40)
        
        # Add trajectory arrows for some representative transitions
        if transition_matrix.shape[0] > 0:
            # Find strongest transitions and plot arrows
            strong_transitions = np.where(transition_matrix > 0.1)
            for i, (src, tgt) in enumerate(zip(strong_transitions[0], strong_transitions[1])):
                if i < 5:  # Limit number of arrows
                    # Get representative points for source and target states
                    src_traces = [v for v, d in system.trace_universe.items() 
                                if d['transition_pathway'] == src]
                    tgt_traces = [v for v, d in system.trace_universe.items() 
                                if d['transition_pathway'] == tgt]
                    
                    if src_traces and tgt_traces:
                        src_data = system.trace_universe[src_traces[0]]
                        tgt_data = system.trace_universe[tgt_traces[0]]
                        
                        ax4.annotate('', xy=(tgt_data['zeta_parameter'], tgt_data['phase_order']),
                                   xytext=(src_data['zeta_parameter'], src_data['phase_order']),
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
        
        ax4.set_xlabel('ζ-Parameter')
        ax4.set_ylabel('Phase Order')
        ax4.set_title('Phase Space with Transition Trajectories')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-05-spectral-collapse/chapter-093-spectral-shift-network.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Network and categorical visualization saved")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with analysis...")
    
    # 8. 运行单元测试
    print("\n8. Running Unit Tests:")
    print("-" * 50)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*80)
    print("SpectralShift Verification Complete")
    print("Key Findings:")
    print(f"- {phase_props['total_traces']} φ-valid traces with ζ-tuned phase analysis")
    print(f"- {len(phase_states)} distinct phase states identified")
    print(f"- Mean ζ-parameter: {phase_props['zeta_param_stats']['mean']:.3f}")
    print(f"- Mean phase order: {phase_props['phase_order_stats']['mean']:.3f}")
    print(f"- Mean stability: {phase_props['stability_stats']['mean']:.3f}")
    print(f"- {cat_analysis['category_count']} phase categories with {cat_analysis['total_morphisms']} morphisms")
    print(f"- Network connectivity: {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    print("="*80)

if __name__ == "__main__":
    run_verification()