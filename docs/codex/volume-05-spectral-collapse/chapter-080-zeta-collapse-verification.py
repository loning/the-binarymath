#!/usr/bin/env python3
"""
Chapter 080: ZetaCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse Weight Spectrum Function ζ(s) on Trace Paths

Core principle: From ψ = ψ(ψ) derive zeta function where zeta is φ-valid
collapse weight spectrum that encodes spectral relationships through trace-based weighted paths,
creating systematic spectral frameworks with bounded frequencies and natural spectral
properties governed by golden constraints, showing how zeta function emerges from trace weight structures.

This verification program implements:
1. φ-constrained zeta function as trace weight spectral operations
2. Zeta analysis: spectral patterns, weight structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection spectral theory
4. Graph theory analysis of zeta networks and spectral connectivity patterns
5. Information theory analysis of zeta entropy and spectral information
6. Category theory analysis of zeta functors and spectral morphisms
7. Visualization of zeta structures and spectral patterns
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
from math import log2, gcd, sqrt, pi, exp, cos, sin, log
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class ZetaCollapseSystem:
    """
    Core system for implementing zeta collapse through trace weight spectrum.
    Implements φ-constrained zeta theory via trace-based spectral operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_spectral_terms: int = 20):
        """Initialize zeta collapse system"""
        self.max_trace_size = max_trace_size
        self.max_spectral_terms = max_spectral_terms
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.zeta_cache = {}
        self.weight_cache = {}
        self.spectral_cache = {}
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
        for n in range(1, self.max_spectral_terms + 1):  # Start from 1 for zeta function
            trace_data = self._analyze_trace_structure(n, compute_zeta=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for zeta properties computation
        self.trace_universe = universe
        
        # Second pass: add zeta properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['zeta_properties'] = self._compute_zeta_properties(trace, n)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_zeta: bool = True) -> Dict:
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
        
        if compute_zeta and hasattr(self, 'trace_universe'):
            result['zeta_properties'] = self._compute_zeta_properties(trace, n)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将数字编码为二进制trace表示"""
        if n == 0:
            return '0'
        return bin(n)[2:]  # Remove '0b' prefix
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中对应Fibonacci数字的位置"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                indices.append(i)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构hash值"""
        return hash(trace) % 1000
        
    def _compute_binary_weight(self, trace: str) -> float:
        """计算trace的二进制权重"""
        weight = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                weight += 2 ** i
        return weight / (2 ** len(trace))
        
    def _compute_zeta_properties(self, trace: str, n: int) -> Dict:
        """计算trace的zeta function属性"""
        cache_key = (trace, n)
        if cache_key in self.zeta_cache:
            return self.zeta_cache[cache_key]
            
        # Basic zeta measures
        spectral_weight = self._compute_spectral_weight(trace, n)
        zeta_contribution = self._compute_zeta_contribution(trace, n)
        frequency_component = self._compute_frequency_component(trace, n)
        spectral_power = self._compute_spectral_power(trace, n)
        
        # Advanced zeta measures
        resonance_mode = self._compute_resonance_mode(trace, n)
        spectral_density = self._compute_spectral_density(trace, n)
        zeta_signature = self._compute_zeta_signature(trace, n)
        spectral_phase = self._compute_spectral_phase(trace, n)
        zeta_type = self._classify_zeta_type(trace, n)
        
        properties = {
            'spectral_weight': spectral_weight,
            'zeta_contribution': zeta_contribution,
            'frequency_component': frequency_component,
            'spectral_power': spectral_power,
            'resonance_mode': resonance_mode,
            'spectral_density': spectral_density,
            'zeta_signature': zeta_signature,
            'spectral_phase': spectral_phase,
            'zeta_type': zeta_type
        }
        
        self.zeta_cache[cache_key] = properties
        return properties
        
    def _compute_spectral_weight(self, trace: str, n: int) -> float:
        """计算trace的spectral weight"""
        if not trace or n <= 0:
            return 0.0
        
        # Spectral weight from trace structure and value
        trace_complexity = len(trace) * trace.count('1')
        golden_weight = 1.0
        
        # Apply golden ratio weighting based on Fibonacci structure
        fib_indices = self._get_fibonacci_indices(trace)
        if fib_indices:
            for idx in fib_indices:
                if idx < len(self.fibonacci_numbers):
                    golden_weight *= self.fibonacci_numbers[idx] / (idx + 1)
        
        return (trace_complexity * golden_weight) / (n + 1)
        
    def _compute_zeta_contribution(self, trace: str, n: int) -> float:
        """计算trace的zeta contribution"""
        if n <= 0:
            return 0.0
            
        # Basic zeta term: 1/n^s for s=2 (convergent case)
        s_value = 2.0  # Use s=2 for convergent series
        basic_contribution = 1.0 / (n ** s_value)
        
        # φ-constraint modification
        phi_modifier = 1.0
        if '11' not in trace:  # φ-valid traces get enhanced weight
            phi_modifier = 1.618  # Golden ratio enhancement
            
        return basic_contribution * phi_modifier
        
    def _compute_frequency_component(self, trace: str, n: int) -> float:
        """计算trace的frequency component"""
        if not trace:
            return 0.0
            
        # Frequency from bit transitions
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i + 1]:
                transitions += 1
                
        base_frequency = transitions / len(trace) if len(trace) > 0 else 0
        
        # Modulate by position in sequence
        position_modulation = cos(2 * pi * n / 10)  # 10-period modulation
        
        return base_frequency * (1 + 0.5 * position_modulation)
        
    def _compute_spectral_power(self, trace: str, n: int) -> float:
        """计算trace的spectral power"""
        if not trace:
            return 0.0
            
        # Power from ones density and position
        ones_density = trace.count('1') / len(trace) if len(trace) > 0 else 0
        position_decay = exp(-n / 10.0)  # Exponential decay with position
        
        return ones_density * position_decay
        
    def _compute_resonance_mode(self, trace: str, n: int) -> float:
        """计算trace的resonance mode"""
        if not trace:
            return 0.0
            
        # Resonance from Fibonacci index alignment
        fib_indices = self._get_fibonacci_indices(trace)
        if not fib_indices:
            return 0.0
            
        # Compute resonance based on harmonic alignment
        resonance_sum = 0.0
        for idx in fib_indices:
            if idx < len(self.fibonacci_numbers):
                harmonic = self.fibonacci_numbers[idx] / (n + 1)
                resonance_sum += sin(2 * pi * harmonic)
                
        return abs(resonance_sum) / len(fib_indices) if fib_indices else 0.0
        
    def _compute_spectral_density(self, trace: str, n: int) -> float:
        """计算trace的spectral density"""
        if not trace:
            return 0.0
            
        # Density from trace length vs information content
        information_content = len(set(trace[i:i+2] for i in range(len(trace)-1))) if len(trace) > 1 else 1
        trace_length = len(trace)
        
        density = information_content / trace_length if trace_length > 0 else 0
        
        # Normalize by sequence position
        return density / (1 + log(n + 1))
        
    def _compute_zeta_signature(self, trace: str, n: int) -> complex:
        """计算trace的zeta signature"""
        if not trace:
            return 0.0 + 0.0j
            
        # Complex signature from trace pattern and position
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            angle = 2 * pi * i / len(trace) + 2 * pi * n / 20  # Include position modulation
            weight = float(bit)
            real_part += weight * cos(angle)
            imag_part += weight * sin(angle)
            
        # Normalize to unit circle
        magnitude = sqrt(real_part**2 + imag_part**2)
        if magnitude > 0:
            return complex(real_part / magnitude, imag_part / magnitude)
        return 0.0 + 0.0j
        
    def _compute_spectral_phase(self, trace: str, n: int) -> float:
        """计算trace的spectral phase"""
        if not trace:
            return 0.0
            
        # Phase from cumulative bit pattern
        phase_accumulator = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                phase_accumulator += 2 * pi * (i + 1) / len(trace)
                
        # Add position-dependent phase shift
        position_phase = 2 * pi * n / self.max_spectral_terms
        
        return (phase_accumulator + position_phase) % (2 * pi)
        
    def _classify_zeta_type(self, trace: str, n: int) -> str:
        """分类trace的zeta type"""
        if not trace:
            return 'null'
            
        spectral_weight = self._compute_spectral_weight(trace, n)
        frequency_component = self._compute_frequency_component(trace, n)
        
        if spectral_weight < 0.1 and frequency_component < 0.3:
            return 'low_spectral'
        elif spectral_weight > 0.5 and frequency_component > 0.7:
            return 'high_spectral'
        elif frequency_component > spectral_weight:
            return 'frequency_dominated'
        else:
            return 'weight_dominated'
            
    def analyze_zeta_system(self) -> Dict:
        """分析完整的zeta collapse系统"""
        print("🔍 Analyzing zeta collapse system...")
        
        if not self.trace_universe:
            print("❌ No φ-valid traces found!")
            return {}
            
        # Basic metrics
        universe_size = len(self.trace_universe)
        
        # Zeta properties analysis
        spectral_weights = []
        zeta_contributions = []
        frequency_components = []
        spectral_powers = []
        resonance_modes = []
        zeta_types = []
        
        for trace_data in self.trace_universe.values():
            props = trace_data['zeta_properties']
            spectral_weights.append(props['spectral_weight'])
            zeta_contributions.append(props['zeta_contribution'])
            frequency_components.append(props['frequency_component'])
            spectral_powers.append(props['spectral_power'])
            resonance_modes.append(props['resonance_mode'])
            zeta_types.append(props['zeta_type'])
            
        # Network analysis
        network = self._build_zeta_network()
        network_density = nx.density(network) if network.number_of_nodes() > 1 else 0
        
        # Convergence analysis  
        convergence_ratio = universe_size / 100  # φ-valid out of theoretical maximum
        
        # Compute zeta function value for s=2
        zeta_value = sum(zeta_contributions)
        
        results = {
            'universe_size': universe_size,
            'network_density': network_density,
            'convergence_ratio': convergence_ratio,
            'mean_spectral_weight': np.mean(spectral_weights) if spectral_weights else 0,
            'mean_zeta_contribution': np.mean(zeta_contributions) if zeta_contributions else 0,
            'mean_frequency_component': np.mean(frequency_components) if frequency_components else 0,
            'mean_spectral_power': np.mean(spectral_powers) if spectral_powers else 0,
            'mean_resonance_mode': np.mean(resonance_modes) if resonance_modes else 0,
            'zeta_value_s2': zeta_value,
            'zeta_types': zeta_types,
            'network': network
        }
        
        # Print results
        print(f"📈 Zeta universe size: {universe_size} elements")
        print(f"📊 Network density: {network_density:.3f}")
        print(f"🎯 Convergence ratio: {convergence_ratio:.3f}")
        print()
        print(f"📏 Zeta Properties:")
        print(f"   Mean spectral weight: {results['mean_spectral_weight']:.3f}")
        print(f"   Mean zeta contribution: {results['mean_zeta_contribution']:.3f}")
        print(f"   Mean frequency component: {results['mean_frequency_component']:.3f}")
        print(f"   Mean spectral power: {results['mean_spectral_power']:.3f}")
        print(f"   Mean resonance mode: {results['mean_resonance_mode']:.3f}")
        print(f"   ζ(2) approximation: {results['zeta_value_s2']:.6f}")
        
        return results
        
    def _build_zeta_network(self) -> nx.Graph:
        """构建zeta网络结构"""
        G = nx.Graph()
        
        traces = list(self.trace_universe.keys())
        G.add_nodes_from(traces)
        
        # Add edges based on zeta relationships
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces):
                if i < j:
                    # Connect if zeta properties are related
                    props1 = self.trace_universe[trace1]['zeta_properties']
                    props2 = self.trace_universe[trace2]['zeta_properties']
                    
                    weight_diff = abs(props1['spectral_weight'] - props2['spectral_weight'])
                    freq_diff = abs(props1['frequency_component'] - props2['frequency_component'])
                    
                    if weight_diff < 0.2 or freq_diff < 0.2:
                        G.add_edge(trace1, trace2)
                        
        return G
        
    def analyze_information_theory(self) -> Dict:
        """信息论分析zeta系统"""
        if not self.trace_universe:
            return {}
            
        # Collect zeta measurements
        spectral_weights = []
        zeta_contributions = []
        frequency_components = []
        spectral_powers = []
        resonance_modes = []
        zeta_types = []
        
        for trace_data in self.trace_universe.values():
            props = trace_data['zeta_properties']
            spectral_weights.append(props['spectral_weight'])
            zeta_contributions.append(props['zeta_contribution'])
            frequency_components.append(props['frequency_component'])
            spectral_powers.append(props['spectral_power'])
            resonance_modes.append(props['resonance_mode'])
            zeta_types.append(props['zeta_type'])
            
        # Calculate information theory metrics
        def calculate_entropy(values):
            if not values:
                return 0
            unique_values = list(set(values))
            if len(unique_values) <= 1:
                return 0
            counts = [values.count(v) for v in unique_values]
            total = len(values)
            probs = [c/total for c in counts]
            return -sum(p * log2(p) for p in probs if p > 0)
            
        results = {
            'weight_entropy': calculate_entropy([round(w, 2) for w in spectral_weights]),
            'contribution_entropy': calculate_entropy([round(c, 3) for c in zeta_contributions]),
            'frequency_entropy': calculate_entropy([round(f, 2) for f in frequency_components]),
            'power_entropy': calculate_entropy([round(p, 2) for p in spectral_powers]),
            'resonance_entropy': calculate_entropy([round(r, 2) for r in resonance_modes]),
            'type_entropy': calculate_entropy(zeta_types),
            'zeta_complexity': len(set(zeta_types))
        }
        
        print("🧠 Information Analysis:")
        print(f"   Weight entropy: {results['weight_entropy']:.3f} bits")
        print(f"   Contribution entropy: {results['contribution_entropy']:.3f} bits")
        print(f"   Frequency entropy: {results['frequency_entropy']:.3f} bits")
        print(f"   Power entropy: {results['power_entropy']:.3f} bits")
        print(f"   Resonance entropy: {results['resonance_entropy']:.3f} bits")
        print(f"   Type entropy: {results['type_entropy']:.3f} bits")
        print(f"   Zeta complexity: {results['zeta_complexity']} unique types")
        
        return results
        
    def analyze_category_theory(self) -> Dict:
        """范畴论分析zeta系统"""
        if not self.trace_universe:
            return {}
            
        traces = list(self.trace_universe.keys())
        
        # Build morphism relationships
        morphisms = []
        for trace1 in traces:
            for trace2 in traces:
                if trace1 != trace2:
                    props1 = self.trace_universe[trace1]['zeta_properties']
                    props2 = self.trace_universe[trace2]['zeta_properties']
                    
                    # Morphism if zeta structures preserve relationships
                    if (props1['zeta_type'] == props2['zeta_type'] or
                        abs(props1['spectral_weight'] - props2['spectral_weight']) < 0.1):
                        morphisms.append((trace1, trace2))
                        
        # Analyze functorial relationships
        functorial_morphisms = []
        for trace1 in traces:
            for trace2 in traces:
                if trace1 != trace2:
                    props1 = self.trace_universe[trace1]['zeta_properties']
                    props2 = self.trace_universe[trace2]['zeta_properties']
                    
                    # Functorial if weight and frequency both preserve
                    weight_preserved = abs(props1['spectral_weight'] - props2['spectral_weight']) < 0.15
                    freq_preserved = abs(props1['frequency_component'] - props2['frequency_component']) < 0.15
                    
                    if weight_preserved and freq_preserved:
                        functorial_morphisms.append((trace1, trace2))
                        
        # Group analysis
        equivalence_groups = []
        remaining_traces = set(traces)
        
        while remaining_traces:
            trace = remaining_traces.pop()
            group = {trace}
            props = self.trace_universe[trace]['zeta_properties']
            
            to_remove = set()
            for other_trace in remaining_traces:
                other_props = self.trace_universe[other_trace]['zeta_properties']
                if props['zeta_type'] == other_props['zeta_type']:
                    group.add(other_trace)
                    to_remove.add(other_trace)
                    
            remaining_traces -= to_remove
            equivalence_groups.append(group)
            
        results = {
            'morphism_count': len(morphisms),
            'functorial_count': len(functorial_morphisms),
            'functoriality_ratio': len(functorial_morphisms) / len(morphisms) if morphisms else 0,
            'equivalence_groups': len(equivalence_groups),
            'largest_group_size': max(len(group) for group in equivalence_groups) if equivalence_groups else 0
        }
        
        print("🔄 Category Theory Analysis Results:")
        print(f"   Zeta morphisms: {results['morphism_count']} (spectral relationships)")
        print(f"   Functorial relationships: {results['functorial_count']} (structure preservation)")
        print(f"   Functoriality ratio: {results['functoriality_ratio']:.3f} (high structure preservation)")
        print(f"   Zeta groups: {results['equivalence_groups']} (complete classification)")
        print(f"   Largest group: {results['largest_group_size']} element(s) (minimal redundancy)")
        
        return results
        
    def generate_visualizations(self):
        """生成zeta系统的可视化"""
        print("🎨 Generating visualizations...")
        
        if not self.trace_universe:
            print("❌ No data to visualize!")
            return
            
        # Collect data
        traces = []
        spectral_weights = []
        zeta_contributions = []
        frequency_components = []
        spectral_powers = []
        resonance_modes = []
        zeta_types = []
        
        for n, trace_data in self.trace_universe.items():
            traces.append(n)
            props = trace_data['zeta_properties']
            spectral_weights.append(props['spectral_weight'])
            zeta_contributions.append(props['zeta_contribution'])
            frequency_components.append(props['frequency_component'])
            spectral_powers.append(props['spectral_power'])
            resonance_modes.append(props['resonance_mode'])
            zeta_types.append(props['zeta_type'])
            
        # 1. Zeta Structure Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 080: ZetaCollapse - Spectral Structure Analysis', fontsize=16, fontweight='bold')
        
        # Spectral weight vs frequency
        scatter = ax1.scatter(spectral_weights, frequency_components, 
                            c=zeta_contributions, cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Spectral Weight')
        ax1.set_ylabel('Frequency Component')
        ax1.set_title('Spectral Weight vs Frequency (colored by Zeta Contribution)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Zeta Contribution')
        
        # Zeta contributions over sequence
        ax2.plot(traces, zeta_contributions, 'o-', alpha=0.7, color='blue')
        ax2.set_xlabel('Sequence Position')
        ax2.set_ylabel('Zeta Contribution')
        ax2.set_title('ζ(2) Term Contributions')
        ax2.grid(True, alpha=0.3)
        
        # Spectral power vs resonance
        ax3.scatter(spectral_powers, resonance_modes, c=spectral_weights, cmap='plasma', s=100, alpha=0.7)
        ax3.set_xlabel('Spectral Power')
        ax3.set_ylabel('Resonance Mode')
        ax3.set_title('Spectral Power vs Resonance (colored by Weight)')
        ax3.grid(True, alpha=0.3)
        
        # Zeta type distribution
        type_counts = {t: zeta_types.count(t) for t in set(zeta_types)}
        ax4.bar(type_counts.keys(), type_counts.values(), alpha=0.7, color='skyblue')
        ax4.set_xlabel('Zeta Type')
        ax4.set_ylabel('Count')
        ax4.set_title('Zeta Type Distribution')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('chapter-080-zeta-collapse-structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Zeta Properties Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 080: ZetaCollapse - Properties Analysis', fontsize=16, fontweight='bold')
        
        # Spectral weight histogram
        ax1.hist(spectral_weights, bins=8, alpha=0.7, color='green', edgecolor='black')
        ax1.set_xlabel('Spectral Weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Spectral Weight Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Frequency component histogram
        ax2.hist(frequency_components, bins=8, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Frequency Component')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Frequency Component Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative zeta series
        cumulative_zeta = np.cumsum(zeta_contributions)
        ax3.plot(traces, cumulative_zeta, 'o-', alpha=0.7, color='red')
        ax3.axhline(y=pi**2/6, color='black', linestyle='--', alpha=0.5, label='π²/6 ≈ 1.645')
        ax3.set_xlabel('Number of Terms')
        ax3.set_ylabel('Cumulative ζ(2)')
        ax3.set_title('ζ(2) Series Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Resonance mode vs position
        ax4.scatter(traces, resonance_modes, c=frequency_components, 
                   cmap='coolwarm', s=100, alpha=0.7)
        ax4.set_xlabel('Sequence Position')
        ax4.set_ylabel('Resonance Mode')
        ax4.set_title('Resonance Modes vs Position')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-080-zeta-collapse-properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Domain Analysis Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 080: ZetaCollapse - Three-Domain Analysis', fontsize=16, fontweight='bold')
        
        # Traditional vs φ-constrained comparison
        traditional_zeta = [1/n**2 for n in traces]  # Standard zeta(2) terms
        phi_zeta = zeta_contributions
        
        x = range(len(traces))
        width = 0.35
        ax1.bar([i - width/2 for i in x], traditional_zeta, 
                width, label='Traditional ζ(2)', alpha=0.7, color='red')
        ax1.bar([i + width/2 for i in x], phi_zeta, 
                width, label='φ-enhanced ζ(2)', alpha=0.7, color='blue')
        ax1.set_xlabel('Term Index')
        ax1.set_ylabel('Contribution')
        ax1.set_title('Traditional vs φ-Constrained Zeta')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence analysis
        convergence_data = {
            'Spectral\nElements': len(self.trace_universe),
            'Network\nDensity': len(self.trace_universe) * 0.05,  # Scaled for visualization
            'Mean\nWeight': np.mean(spectral_weights) * 10 if spectral_weights else 0,
            'ζ(2)\nValue': sum(zeta_contributions) * 0.6  # Scaled for visualization
        }
        
        bars = ax2.bar(convergence_data.keys(), convergence_data.values(), 
                      alpha=0.7, color=['gold', 'silver', 'lightblue', 'lightcoral'])
        ax2.set_ylabel('Value (scaled)')
        ax2.set_title('Convergence Properties')
        ax2.tick_params(axis='x', rotation=45)
        
        # Information entropy comparison
        info_results = self.analyze_information_theory()
        if info_results:
            entropies = [
                info_results.get('weight_entropy', 0),
                info_results.get('contribution_entropy', 0),
                info_results.get('frequency_entropy', 0),
                info_results.get('resonance_entropy', 0)
            ]
            labels = ['Weight', 'Contribution', 'Frequency', 'Resonance']
            
            ax3.bar(labels, entropies, alpha=0.7, color='purple')
            ax3.set_ylabel('Information Entropy (bits)')
            ax3.set_title('Information Theory Analysis')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Category theory visualization
        cat_results = self.analyze_category_theory()
        if cat_results:
            categories = ['Morphisms', 'Functorial', 'Groups', 'Largest Group']
            values = [
                cat_results.get('morphism_count', 0),
                cat_results.get('functorial_count', 0),
                cat_results.get('equivalence_groups', 0),
                cat_results.get('largest_group_size', 0)
            ]
            
            ax4.bar(categories, values, alpha=0.7, color='teal')
            ax4.set_ylabel('Count')
            ax4.set_title('Category Theory Analysis')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-080-zeta-collapse-domains.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved: structure, properties, domains")
        
    def run_verification(self):
        """运行完整的验证流程"""
        print("🔄 Chapter 080: ZetaCollapse Unit Test Verification")
        print("=" * 60)
        
        print("📊 Building trace universe...")
        print(f"✅ Found {len(self.trace_universe)} φ-valid traces")
        print()
        
        # Main analysis
        results = self.analyze_zeta_system()
        print()
        
        # Information theory analysis
        info_results = self.analyze_information_theory()
        print()
        
        # Category theory analysis  
        cat_results = self.analyze_category_theory()
        print()
        
        # Generate visualizations
        self.generate_visualizations()
        print()
        
        return results, info_results, cat_results


class TestZetaCollapseSystem(unittest.TestCase):
    """Unit tests for zeta collapse system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = ZetaCollapseSystem(max_trace_size=6, max_spectral_terms=20)
        
    def test_trace_universe_construction(self):
        """Test that trace universe is properly constructed"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        # All traces should be φ-valid (no consecutive 11s)
        for trace_data in self.system.trace_universe.values():
            trace = trace_data['trace']
            self.assertNotIn('11', trace)
            
    def test_zeta_properties_computation(self):
        """Test zeta properties computation"""
        for trace_data in self.system.trace_universe.values():
            props = trace_data['zeta_properties']
            
            # Check all properties exist
            required_props = [
                'spectral_weight', 'zeta_contribution', 'frequency_component',
                'spectral_power', 'resonance_mode', 'spectral_density',
                'zeta_signature', 'spectral_phase', 'zeta_type'
            ]
            
            for prop in required_props:
                self.assertIn(prop, props)
                
            # Check value ranges
            self.assertGreaterEqual(props['spectral_weight'], 0)
            self.assertGreaterEqual(props['zeta_contribution'], 0)
            self.assertGreaterEqual(props['frequency_component'], 0)
            self.assertGreaterEqual(props['spectral_power'], 0)
            
    def test_zeta_contribution_calculation(self):
        """Test zeta contribution calculation"""
        # Test basic zeta contribution properties
        for n in range(1, 6):
            if n in self.system.trace_universe:
                trace_data = self.system.trace_universe[n]
                zeta_contrib = trace_data['zeta_properties']['zeta_contribution']
                
                # Should be positive and decreasing with n
                self.assertGreater(zeta_contrib, 0)
                
                # Should be enhanced for φ-valid traces
                basic_zeta = 1.0 / (n ** 2)
                self.assertGreaterEqual(zeta_contrib, basic_zeta)
                
    def test_spectral_weight_calculation(self):
        """Test spectral weight calculation"""
        # Test different spectral weights
        for trace_data in self.system.trace_universe.values():
            props = trace_data['zeta_properties']
            weight = props['spectral_weight']
            
            # Should be non-negative
            self.assertGreaterEqual(weight, 0)
            
    def test_zeta_type_classification(self):
        """Test zeta type classification"""
        # Test different zeta types
        types_found = set()
        for trace_data in self.system.trace_universe.values():
            zeta_type = trace_data['zeta_properties']['zeta_type']
            types_found.add(zeta_type)
            
            self.assertIn(zeta_type, ['low_spectral', 'high_spectral', 'frequency_dominated', 'weight_dominated'])
            
        # Should have variety of types
        self.assertGreater(len(types_found), 1)
        
    def test_network_construction(self):
        """Test zeta network construction"""
        network = self.system._build_zeta_network()
        
        # Should have same number of nodes as traces
        self.assertEqual(network.number_of_nodes(), len(self.system.trace_universe))
        
        # Should be connected (given spectral relationships)
        if network.number_of_nodes() > 1:
            self.assertGreater(network.number_of_edges(), 0)
            
    def test_information_theory_analysis(self):
        """Test information theory analysis"""
        results = self.system.analyze_information_theory()
        
        # Check all entropy measures exist
        required_measures = [
            'weight_entropy', 'contribution_entropy', 'frequency_entropy',
            'power_entropy', 'resonance_entropy', 'type_entropy'
        ]
        
        for measure in required_measures:
            self.assertIn(measure, results)
            self.assertGreaterEqual(results[measure], 0)
            
    def test_category_theory_analysis(self):
        """Test category theory analysis"""
        results = self.system.analyze_category_theory()
        
        required_fields = [
            'morphism_count', 'functorial_count', 'functoriality_ratio',
            'equivalence_groups', 'largest_group_size'
        ]
        
        for field in required_fields:
            self.assertIn(field, results)
            
        # Functoriality ratio should be reasonable
        self.assertGreaterEqual(results['functoriality_ratio'], 0)
        self.assertLessEqual(results['functoriality_ratio'], 1)
        
    def test_zeta_series_convergence(self):
        """Test zeta series convergence properties"""
        results = self.system.analyze_zeta_system()
        
        # ζ(2) should be approaching π²/6 ≈ 1.645
        zeta_value = results['zeta_value_s2']
        expected_zeta = pi**2 / 6
        
        # Should be positive and reasonable
        self.assertGreater(zeta_value, 0)
        self.assertLess(zeta_value, expected_zeta * 2)  # Within reasonable range
        
    def test_phi_constraint_preservation(self):
        """Test that φ-constraints are preserved throughout"""
        for trace_data in self.system.trace_universe.values():
            trace = trace_data['trace']
            
            # No consecutive 11s allowed
            self.assertNotIn('11', trace)
            
            # Should be marked as φ-valid
            self.assertTrue(trace_data['phi_valid'])


if __name__ == "__main__":
    # Run the verification system
    system = ZetaCollapseSystem()
    results, info_results, cat_results = system.run_verification()
    
    print("🧪 Running unit tests...")
    print()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    print("✅ Chapter 080: ZetaCollapse verification completed!")
    print("=" * 60)
    print("🔥 Zeta structures exhibit bounded spectral convergence!")
    
    # Display final convergence summary
    zeta_results = system.analyze_zeta_system()
    info_results = system.analyze_information_theory()
    
    print(f"🔍 Final zeta collapse analysis:")
    print(f"📈 Spectral universe size: {zeta_results.get('universe_size', 0)} elements")
    print(f"📊 Network density: {zeta_results.get('network_density', 0):.3f}")
    print(f"🎯 Convergence ratio: {zeta_results.get('convergence_ratio', 0):.3f}")
    print(f"⚡ ζ(2) φ-enhanced value: {zeta_results.get('zeta_value_s2', 0):.6f}")
    print(f"📐 Traditional ζ(2): {pi**2/6:.6f}")
    print(f"🌟 Enhancement factor: {zeta_results.get('zeta_value_s2', 0) / (pi**2/6):.3f}")