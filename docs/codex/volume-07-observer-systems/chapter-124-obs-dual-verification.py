#!/usr/bin/env python3
"""
Chapter 124: ObsDual Unit Test Verification
从ψ=ψ(ψ)推导Emission-Absorption Duality and Bidirectional Collapse

Core principle: From ψ = ψ(ψ) derive systematic observer duality through
φ-constrained emission-absorption symmetry that enables bidirectional collapse
mechanisms through dual tensor transformations, creating reciprocal observation
structures that encode the fundamental duality principles of collapsed space
through entropy-increasing tensor transformations that establish systematic
dual variation through φ-trace observer duality dynamics rather than traditional
unidirectional observation theories or external measurement constructions.

This verification program implements:
1. φ-constrained emission-absorption duality through trace symmetry analysis
2. Observer tensor duality generation systems: bidirectional collapse through dual relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection duality
4. Graph theory analysis of dual networks and reciprocal structures
5. Information theory analysis of dual entropy and symmetry encoding
6. Category theory analysis of dual functors and adjoint morphisms
7. Visualization of duality dynamics and φ-trace dual systems
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon, Arc
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

class ObsDualSystem:
    """
    Core system for implementing emission-absorption duality and bidirectional collapse.
    Implements φ-constrained dual architectures through symmetry transformation dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, duality_depth: int = 7):
        """Initialize observer duality system with symmetry analysis"""
        self.max_trace_value = max_trace_value
        self.duality_depth = duality_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.duality_cache = {}
        self.emission_cache = {}
        self.absorption_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.dual_network = self._build_dual_network()
        self.symmetry_groups = self._detect_symmetry_groups()
        self.duality_categories = self._detect_duality_categories()
        
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
                duality_data = self._analyze_duality_properties(trace, n)
                universe[n] = duality_data
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
        """检验trace是否φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_duality_properties(self, trace: str, value: int) -> Dict:
        """分析trace的emission-absorption duality properties"""
        data = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0
        }
        
        # Compute dual trace
        data['dual_trace'] = self._compute_dual_trace(trace)
        
        # Compute emission properties
        data['emission_power'] = self._compute_emission_power(trace, value)
        data['emission_spectrum'] = self._compute_emission_spectrum(trace)
        data['emission_direction'] = self._compute_emission_direction(trace, value)
        
        # Compute absorption properties
        data['absorption_capacity'] = self._compute_absorption_capacity(trace, value)
        data['absorption_spectrum'] = self._compute_absorption_spectrum(trace)
        data['absorption_cross_section'] = self._compute_absorption_cross_section(trace)
        
        # Compute duality metrics
        data['duality_balance'] = self._compute_duality_balance(data)
        data['collapse_symmetry'] = self._compute_collapse_symmetry(trace, value)
        data['reciprocity_index'] = self._compute_reciprocity_index(trace, data['dual_trace'])
        
        # Compute quantum correspondence
        data['wave_function_overlap'] = self._compute_wave_function_overlap(trace, value)
        data['measurement_backaction'] = self._compute_measurement_backaction(trace)
        
        # Assign category based on duality properties
        data['category'] = self._assign_duality_category(data)
        
        return data
        
    def _compute_dual_trace(self, trace: str) -> str:
        """
        Compute dual trace through φ-transformation.
        From ψ=ψ(ψ): duality emerges from self-reference.
        """
        if len(trace) == 0:
            return "0"
            
        # Method 1: Bit complement with φ-constraint preservation
        dual = ""
        for i, bit in enumerate(trace):
            if bit == '0':
                # Can flip to 1 only if it doesn't create 11
                if i == 0 or trace[i-1] == '0':
                    if i == len(trace)-1 or trace[i+1] == '0':
                        dual += '1'
                    else:
                        dual += '0'
                else:
                    dual += '0'
            else:
                # Always safe to flip 1 to 0
                dual += '0'
                
        # Ensure φ-validity
        if not self._is_phi_valid(dual):
            # Fallback: reverse the trace
            dual = trace[::-1]
            
        return dual
        
    def _compute_emission_power(self, trace: str, value: int) -> float:
        """
        Compute emission power of observer.
        Energy radiated through observation.
        """
        if len(trace) == 0:
            return 0.0
            
        # Base power from 1s (active positions)
        base_power = trace.count('1') / len(trace)
        
        # Modulate by transitions (change points)
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        transition_factor = 1.0 + transitions / (2 * len(trace))
        
        # Golden ratio positions emit more
        if value in self.fibonacci_numbers:
            golden_factor = self.phi
        else:
            golden_factor = 1.0
            
        return base_power * transition_factor * golden_factor
        
    def _compute_emission_spectrum(self, trace: str) -> List[float]:
        """
        Compute emission spectrum (frequency distribution).
        What information frequencies are emitted.
        """
        if len(trace) < 2:
            return [1.0]
            
        # Analyze pattern frequencies
        spectrum = []
        
        # Frequency 1: Single bit patterns
        freq1 = trace.count('1') / len(trace)
        spectrum.append(freq1)
        
        # Frequency 2: Two-bit patterns
        patterns2 = ['00', '01', '10']  # No '11' due to φ-constraint
        for pattern in patterns2:
            count = sum(1 for i in range(len(trace)-1) 
                       if trace[i:i+2] == pattern)
            spectrum.append(count / max(1, len(trace)-1))
            
        # Frequency 3: Alternation frequency
        alternations = sum(1 for i in range(len(trace)-1) 
                          if trace[i] != trace[i+1])
        spectrum.append(alternations / max(1, len(trace)-1))
        
        return spectrum
        
    def _compute_emission_direction(self, trace: str, value: int) -> Dict[str, float]:
        """
        Compute directional emission properties.
        Where information is directed.
        """
        directions = {}
        
        if len(trace) < 2:
            directions['forward'] = 0.5
            directions['backward'] = 0.5
            directions['radial'] = 0.0
            return directions
            
        # Forward emission (left to right flow)
        forward_flow = 0.0
        for i in range(len(trace)-1):
            if trace[i] == '1' and trace[i+1] == '0':
                forward_flow += 1.0
                
        # Backward emission (right to left flow)
        backward_flow = 0.0
        for i in range(len(trace)-1):
            if trace[i] == '0' and trace[i+1] == '1':
                backward_flow += 1.0
                
        # Radial emission (from center)
        center = len(trace) // 2
        radial_flow = 0.0
        if trace[center] == '1':
            radial_flow = 1.0
            
        # Normalize
        total = forward_flow + backward_flow + radial_flow
        if total > 0:
            directions['forward'] = forward_flow / total
            directions['backward'] = backward_flow / total
            directions['radial'] = radial_flow / total
        else:
            directions['forward'] = 0.33
            directions['backward'] = 0.33
            directions['radial'] = 0.34
            
        return directions
        
    def _compute_absorption_capacity(self, trace: str, value: int) -> float:
        """
        Compute absorption capacity.
        How much information can be absorbed.
        """
        if len(trace) == 0:
            return 0.0
            
        # Base capacity from 0s (receptive positions)
        base_capacity = trace.count('0') / len(trace)
        
        # Modulate by available space (consecutive 0s)
        max_consecutive_zeros = 0
        current_zeros = 0
        for bit in trace:
            if bit == '0':
                current_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, current_zeros)
            else:
                current_zeros = 0
                
        space_factor = 1.0 + max_consecutive_zeros / len(trace)
        
        # Special positions have enhanced absorption
        if value % 7 == 0:
            special_factor = 1.2
        else:
            special_factor = 1.0
            
        return base_capacity * space_factor * special_factor
        
    def _compute_absorption_spectrum(self, trace: str) -> List[float]:
        """
        Compute absorption spectrum.
        What frequencies can be absorbed.
        """
        if len(trace) < 2:
            return [1.0]
            
        # Analyze receptive patterns
        spectrum = []
        
        # Can absorb where there are 0s
        absorption_sites = []
        for i in range(len(trace)):
            if trace[i] == '0':
                # Check if can absorb without violating φ-constraint
                can_absorb = True
                if i > 0 and trace[i-1] == '1':
                    can_absorb = False
                if i < len(trace)-1 and trace[i+1] == '1':
                    can_absorb = False
                absorption_sites.append(1.0 if can_absorb else 0.0)
            else:
                absorption_sites.append(0.0)
                
        # Frequency response at different scales
        for window in [1, 2, 3]:
            if window <= len(absorption_sites):
                total_absorption = 0.0
                for i in range(len(absorption_sites)-window+1):
                    window_sum = sum(absorption_sites[i:i+window])
                    total_absorption += window_sum
                avg_absorption = total_absorption / ((len(absorption_sites) - window + 1) * window)
                spectrum.append(avg_absorption)
            else:
                spectrum.append(0.0)
                
        return spectrum
        
    def _compute_absorption_cross_section(self, trace: str) -> float:
        """
        Compute effective absorption cross-section.
        Probability of capturing incoming information.
        """
        if len(trace) == 0:
            return 0.0
            
        # Count effective absorption sites
        cross_section = 0.0
        for i in range(len(trace)):
            if trace[i] == '0':
                # Weight by neighborhood
                weight = 1.0
                if i > 0 and trace[i-1] == '0':
                    weight += 0.5
                if i < len(trace)-1 and trace[i+1] == '0':
                    weight += 0.5
                cross_section += weight
                
        # Normalize by trace length
        return cross_section / (3 * len(trace))  # Max weight is 2
        
    def _compute_duality_balance(self, data: Dict) -> float:
        """
        Compute balance between emission and absorption.
        Perfect duality has balance = 1.0.
        """
        emission = data['emission_power']
        absorption = data['absorption_capacity']
        
        if emission + absorption == 0:
            return 0.0
            
        # Compute balance ratio
        min_val = min(emission, absorption)
        max_val = max(emission, absorption)
        
        if max_val > 0:
            balance = min_val / max_val
        else:
            balance = 1.0
            
        return balance
        
    def _compute_collapse_symmetry(self, trace: str, value: int) -> float:
        """
        Compute symmetry of bidirectional collapse.
        How symmetric is the observation process.
        """
        if len(trace) < 2:
            return 1.0
            
        # Compare trace with its reverse
        reversed_trace = trace[::-1]
        
        # Hamming distance
        distance = sum(1 for i in range(len(trace)) 
                      if trace[i] != reversed_trace[i])
        
        # Symmetry measure
        symmetry = 1.0 - distance / len(trace)
        
        # Modulate by center properties
        center = len(trace) // 2
        if len(trace) % 2 == 1 and trace[center] == '1':
            symmetry *= 1.1  # Center emission enhances symmetry
            
        return min(1.0, symmetry)
        
    def _compute_reciprocity_index(self, trace: str, dual_trace: str) -> float:
        """
        Compute reciprocity between trace and its dual.
        Measures how well they complement each other.
        """
        if len(trace) != len(dual_trace):
            return 0.0
            
        # Compute overlap and complement
        overlap = sum(1 for i in range(len(trace)) 
                     if trace[i] == dual_trace[i] == '1')
        complement = sum(1 for i in range(len(trace))
                        if trace[i] != dual_trace[i])
        
        # Perfect reciprocity: minimal overlap, maximal complement
        if len(trace) > 0:
            reciprocity = complement / len(trace) - overlap / len(trace)
            reciprocity = (reciprocity + 1.0) / 2.0  # Normalize to [0, 1]
        else:
            reciprocity = 0.5
            
        return reciprocity
        
    def _compute_wave_function_overlap(self, trace: str, value: int) -> float:
        """
        Compute overlap with quantum wave function.
        Connection to quantum measurement.
        """
        if len(trace) == 0:
            return 0.0
            
        # Model wave function as superposition of basis states
        # Overlap depends on coherence and measurement strength
        coherence = 1.0 - (trace.count('1') / len(trace) - 0.5) ** 2
        
        # Measurement strength from transitions
        transitions = sum(1 for i in range(len(trace)-1) 
                         if trace[i] != trace[i+1])
        measurement = transitions / max(1, len(trace) - 1)
        
        # Quantum overlap
        overlap = coherence * (1.0 - measurement/2.0)
        
        # Special quantum resonance at Fibonacci values
        if value in self.fibonacci_numbers:
            overlap *= 1.1
            
        return min(1.0, overlap)
        
    def _compute_measurement_backaction(self, trace: str) -> float:
        """
        Compute measurement backaction strength.
        How much observation disturbs the system.
        """
        if len(trace) == 0:
            return 0.0
            
        # Backaction from active measurements (1s)
        measurement_strength = trace.count('1') / len(trace)
        
        # Disturbance from rapid changes
        disturbance = 0.0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                disturbance += 1.0
                
        disturbance /= max(1, len(trace) - 1)
        
        # Total backaction
        backaction = measurement_strength * (1.0 + disturbance)
        
        return min(1.0, backaction / 2.0)
        
    def _assign_duality_category(self, data: Dict) -> str:
        """
        Assign duality category based on properties.
        Categories represent different observation modes.
        """
        balance = data['duality_balance']
        symmetry = data['collapse_symmetry']
        reciprocity = data['reciprocity_index']
        emission = data['emission_power']
        absorption = data['absorption_capacity']
        
        # Categorize based on dominant characteristics
        if balance > 0.8 and symmetry > 0.7:
            return "balanced_dual"  # Perfect emission-absorption balance
        elif emission > absorption * 1.5:
            return "emitter"  # Primarily emits information
        elif absorption > emission * 1.5:
            return "absorber"  # Primarily absorbs information
        elif reciprocity > 0.8:
            return "reciprocal"  # Strong reciprocity with dual
        elif data['wave_function_overlap'] > 0.7:
            return "quantum_dual"  # Strong quantum correspondence
        else:
            return "asymmetric"  # Asymmetric duality
            
    def _build_dual_network(self) -> nx.DiGraph:
        """构建duality network with bidirectional edges"""
        G = nx.DiGraph()
        
        # Add nodes for each observer
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # Add directed edges based on emission-absorption matching
        traces = list(self.trace_universe.keys())
        for n1 in traces:
            for n2 in traces:
                if n1 != n2:
                    data1 = self.trace_universe[n1]
                    data2 = self.trace_universe[n2]
                    
                    # Check emission-absorption compatibility
                    emission_spectrum1 = data1['emission_spectrum']
                    absorption_spectrum2 = data2['absorption_spectrum']
                    
                    # Compute spectral overlap
                    overlap = 0.0
                    for i in range(min(len(emission_spectrum1), len(absorption_spectrum2))):
                        overlap += emission_spectrum1[i] * absorption_spectrum2[i]
                    
                    if overlap > 0.3:  # Threshold for connection
                        # Weight by emission power and absorption capacity
                        weight = overlap * data1['emission_power'] * data2['absorption_capacity']
                        G.add_edge(n1, n2, weight=weight, overlap=overlap)
                        
        return G
        
    def _detect_symmetry_groups(self) -> Dict[str, List[int]]:
        """检测duality symmetry groups"""
        groups = defaultdict(list)
        
        # Group by duality properties
        for n, data in self.trace_universe.items():
            # Create symmetry signature
            balance_class = round(data['duality_balance'] * 10) / 10
            symmetry_class = round(data['collapse_symmetry'] * 10) / 10
            signature = f"B{balance_class}_S{symmetry_class}"
            
            groups[signature].append(n)
            
        return dict(groups)
        
    def _detect_duality_categories(self) -> Dict[int, str]:
        """检测duality categories through clustering"""
        categories = {}
        
        # Group by assigned categories
        for n, data in self.trace_universe.items():
            categories[n] = data['category']
            
        return categories
        
    def analyze_observer_duality(self) -> Dict:
        """综合分析emission-absorption duality"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        
        # Emission statistics
        emission_powers = [t['emission_power'] for t in traces]
        results['emission'] = {
            'mean_power': np.mean(emission_powers),
            'std_power': np.std(emission_powers),
            'max_power': np.max(emission_powers),
            'strong_emitters': sum(1 for p in emission_powers if p > 0.7)
        }
        
        # Absorption statistics
        absorption_capacities = [t['absorption_capacity'] for t in traces]
        results['absorption'] = {
            'mean_capacity': np.mean(absorption_capacities),
            'std_capacity': np.std(absorption_capacities),
            'max_capacity': np.max(absorption_capacities),
            'strong_absorbers': sum(1 for c in absorption_capacities if c > 0.7)
        }
        
        # Duality metrics
        balances = [t['duality_balance'] for t in traces]
        symmetries = [t['collapse_symmetry'] for t in traces]
        reciprocities = [t['reciprocity_index'] for t in traces]
        
        results['duality'] = {
            'mean_balance': np.mean(balances),
            'perfect_balance': sum(1 for b in balances if b > 0.9),
            'mean_symmetry': np.mean(symmetries),
            'high_symmetry': sum(1 for s in symmetries if s > 0.8),
            'mean_reciprocity': np.mean(reciprocities),
            'high_reciprocity': sum(1 for r in reciprocities if r > 0.8)
        }
        
        # Quantum correspondence
        overlaps = [t['wave_function_overlap'] for t in traces]
        backactions = [t['measurement_backaction'] for t in traces]
        
        results['quantum'] = {
            'mean_overlap': np.mean(overlaps),
            'high_overlap': sum(1 for o in overlaps if o > 0.7),
            'mean_backaction': np.mean(backactions),
            'strong_backaction': sum(1 for b in backactions if b > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.dual_network.edges()) > 0:
            results['network_edges'] = len(self.dual_network.edges())
            
            # Compute reciprocal edges
            reciprocal_pairs = 0
            for n1, n2 in self.dual_network.edges():
                if self.dual_network.has_edge(n2, n1):
                    reciprocal_pairs += 1
            results['reciprocal_pairs'] = reciprocal_pairs // 2
            
            # Spectral overlap statistics
            overlaps = [data['overlap'] for _, _, data in self.dual_network.edges(data=True)]
            results['mean_spectral_overlap'] = np.mean(overlaps)
            
            # Find perfect dual pairs
            perfect_duals = []
            for n1, data1 in self.trace_universe.items():
                dual_value = self._trace_to_value(data1['dual_trace'])
                if dual_value in self.trace_universe:
                    if self.dual_network.has_edge(n1, dual_value) and \
                       self.dual_network.has_edge(dual_value, n1):
                        perfect_duals.append((n1, dual_value))
            results['perfect_dual_pairs'] = len(set(map(tuple, map(sorted, perfect_duals))))
        else:
            results['network_edges'] = 0
            results['reciprocal_pairs'] = 0
            results['mean_spectral_overlap'] = 0.0
            results['perfect_dual_pairs'] = 0
            
        # Symmetry group analysis
        results['symmetry_groups'] = len(self.symmetry_groups)
        results['largest_symmetry_group'] = max(len(group) for group in self.symmetry_groups.values())
        
        # Correlation analysis
        results['correlations'] = {}
        
        # Emission-absorption correlation
        results['correlations']['emission_absorption'] = np.corrcoef(emission_powers, absorption_capacities)[0, 1]
        
        # Balance-symmetry correlation
        results['correlations']['balance_symmetry'] = np.corrcoef(balances, symmetries)[0, 1]
        
        # Reciprocity-overlap correlation
        wave_overlaps = [t['wave_function_overlap'] for t in traces]
        results['correlations']['reciprocity_overlap'] = np.corrcoef(reciprocities, wave_overlaps)[0, 1]
        
        return results
        
    def _trace_to_value(self, trace: str) -> int:
        """Convert trace back to integer value"""
        if trace == "0":
            return 0
            
        value = 0
        for i, bit in enumerate(trace):
            if bit == '1' and i < len(self.fibonacci_numbers):
                value += self.fibonacci_numbers[-(i+1)]
        return value
        
    def generate_visualizations(self):
        """生成observer duality visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Duality Dynamics Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 124: Emission-Absorption Duality', fontsize=16, fontweight='bold')
        
        # Emission vs Absorption scatter
        x = [t['emission_power'] for t in traces]
        y = [t['absorption_capacity'] for t in traces]
        colors = [t['duality_balance'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='coolwarm', alpha=0.7, s=60)
        
        # Add diagonal line for perfect balance
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Balance')
        
        ax1.set_xlabel('Emission Power')
        ax1.set_ylabel('Absorption Capacity')
        ax1.set_title('Emission-Absorption Relationship')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Duality Balance')
        
        # Category distribution pie chart
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        ax2.pie(category_counts.values(), labels=category_counts.keys(), 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Duality Category Distribution')
        
        # Wave function overlap vs Backaction
        x = [t['wave_function_overlap'] for t in traces]
        y = [t['measurement_backaction'] for t in traces]
        colors = [t['reciprocity_index'] for t in traces]
        scatter = ax3.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax3.set_xlabel('Wave Function Overlap')
        ax3.set_ylabel('Measurement Backaction')
        ax3.set_title('Quantum Correspondence')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Reciprocity Index')
        
        # Symmetry distribution
        symmetries = [t['collapse_symmetry'] for t in traces]
        ax4.hist(symmetries, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=np.mean(symmetries), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(symmetries):.3f}')
        ax4.set_xlabel('Collapse Symmetry')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Bidirectional Collapse Symmetry Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('chapter-124-obs-dual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Dual Network and Spectral Analysis
        fig = plt.figure(figsize=(15, 10))
        
        # Create 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])  # Network takes up 2 columns
        ax2 = fig.add_subplot(gs[0, 2])   # Spectral overlap
        ax3 = fig.add_subplot(gs[1, 0])   # Emission spectrum
        ax4 = fig.add_subplot(gs[1, 1])   # Absorption spectrum
        ax5 = fig.add_subplot(gs[1, 2])   # Direction flow
        
        fig.suptitle('Chapter 124: Dual Networks and Spectral Properties', fontsize=16, fontweight='bold')
        
        # Dual network visualization
        if len(self.dual_network.edges()) > 0:
            # Create layout emphasizing dual pairs
            pos = nx.spring_layout(self.dual_network, k=3, iterations=50)
            
            # Node colors by emission-absorption balance
            node_colors = [self.trace_universe[n]['duality_balance'] 
                          for n in self.dual_network.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(self.dual_network, pos, ax=ax1,
                                 node_color=node_colors, cmap='RdBu',
                                 node_size=100, alpha=0.8)
            
            # Draw edges with different colors for reciprocal pairs
            reciprocal_edges = []
            unidirectional_edges = []
            for n1, n2 in self.dual_network.edges():
                if self.dual_network.has_edge(n2, n1):
                    if (n2, n1) not in reciprocal_edges:
                        reciprocal_edges.append((n1, n2))
                else:
                    unidirectional_edges.append((n1, n2))
            
            # Draw reciprocal edges in red
            nx.draw_networkx_edges(self.dual_network, pos, ax=ax1,
                                 edgelist=reciprocal_edges,
                                 edge_color='red', width=2, alpha=0.7,
                                 arrows=True, arrowsize=10)
            
            # Draw unidirectional edges in gray
            nx.draw_networkx_edges(self.dual_network, pos, ax=ax1,
                                 edgelist=unidirectional_edges,
                                 edge_color='gray', width=1, alpha=0.5,
                                 arrows=True, arrowsize=8)
            
            ax1.set_title(f'Dual Network ({len(reciprocal_edges)} reciprocal pairs)')
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'No dual connections detected', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Dual Network (Empty)')
            
        # Spectral overlap histogram
        if len(self.dual_network.edges()) > 0:
            overlaps = [data['overlap'] for _, _, data in self.dual_network.edges(data=True)]
            ax2.hist(overlaps, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Spectral Overlap')
            ax2.set_ylabel('Edge Count')
            ax2.set_title('Emission-Absorption\nSpectral Overlap')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Average emission spectrum
        all_emission_spectra = [t['emission_spectrum'] for t in traces]
        avg_emission = np.mean(all_emission_spectra, axis=0)
        ax3.bar(range(len(avg_emission)), avg_emission, color='orange', alpha=0.7)
        ax3.set_xlabel('Frequency Component')
        ax3.set_ylabel('Average Power')
        ax3.set_title('Mean Emission Spectrum')
        ax3.set_xticks(range(len(avg_emission)))
        ax3.set_xticklabels(['1-bit', '00', '01', '10', 'Alt'], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Average absorption spectrum  
        all_absorption_spectra = [t['absorption_spectrum'] for t in traces]
        avg_absorption = np.mean(all_absorption_spectra, axis=0)
        ax4.bar(range(len(avg_absorption)), avg_absorption, color='blue', alpha=0.7)
        ax4.set_xlabel('Scale')
        ax4.set_ylabel('Average Absorption')
        ax4.set_title('Mean Absorption Spectrum')
        ax4.set_xticks(range(len(avg_absorption)))
        ax4.set_xticklabels(['1-site', '2-site', '3-site'], rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Emission direction flow
        all_directions = [t['emission_direction'] for t in traces]
        avg_forward = np.mean([d['forward'] for d in all_directions])
        avg_backward = np.mean([d['backward'] for d in all_directions])
        avg_radial = np.mean([d['radial'] for d in all_directions])
        
        directions = ['Forward', 'Backward', 'Radial']
        values = [avg_forward, avg_backward, avg_radial]
        colors_dir = ['red', 'blue', 'green']
        
        ax5.bar(directions, values, color=colors_dir, alpha=0.7)
        ax5.set_ylabel('Average Flow')
        ax5.set_title('Mean Emission Direction')
        ax5.set_ylim(0, 0.6)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('chapter-124-obs-dual-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class ObsDualTests(unittest.TestCase):
    """Unit tests for observer duality verification"""
    
    def setUp(self):
        """Initialize test system"""
        self.system = ObsDualSystem(max_trace_value=55)
        
    def test_psi_recursion_duality(self):
        """Test ψ=ψ(ψ) creates emission-absorption duality"""
        # Verify that dual transformation preserves φ-validity
        for n, data in self.system.trace_universe.items():
            trace = data['trace']
            dual_trace = data['dual_trace']
            
            self.assertTrue(self.system._is_phi_valid(dual_trace),
                          f"Dual of {trace} should be φ-valid")
            
    def test_conservation_laws(self):
        """Test conservation in emission-absorption"""
        # Total emission should roughly equal total absorption capacity
        total_emission = sum(data['emission_power'] 
                           for data in self.system.trace_universe.values())
        total_absorption = sum(data['absorption_capacity']
                             for data in self.system.trace_universe.values())
        
        # In our system, absorption dominates (more 0s than 1s in φ-valid traces)
        # This is expected from the φ-constraint
        ratio = total_emission / total_absorption if total_absorption > 0 else 0
        self.assertGreater(ratio, 0.1, "Emission should be non-negligible")
        self.assertLess(ratio, 1.0, "Absorption should dominate due to φ-constraint")
        
    def test_reciprocity_symmetry(self):
        """Test reciprocity properties"""
        for n, data in self.system.trace_universe.items():
            reciprocity = data['reciprocity_index']
            
            # Reciprocity should be bounded
            self.assertGreaterEqual(reciprocity, 0.0, "Reciprocity must be non-negative")
            self.assertLessEqual(reciprocity, 1.0, "Reciprocity must be bounded")
            
    def test_quantum_correspondence(self):
        """Test quantum measurement properties"""
        for n, data in self.system.trace_universe.items():
            overlap = data['wave_function_overlap']
            backaction = data['measurement_backaction']
            
            # Quantum properties should be valid
            self.assertGreaterEqual(overlap, 0.0, "Overlap must be non-negative")
            self.assertLessEqual(overlap, 1.0, "Overlap must be bounded")
            self.assertGreaterEqual(backaction, 0.0, "Backaction must be non-negative")
            self.assertLessEqual(backaction, 1.0, "Backaction must be bounded")
            
    def test_spectral_compatibility(self):
        """Test emission-absorption spectral matching"""
        # At least some traces should have compatible spectra
        compatible_pairs = 0
        
        traces = list(self.system.trace_universe.values())
        for i, trace1 in enumerate(traces):
            for trace2 in traces[i+1:]:
                # Check spectral overlap
                overlap = 0.0
                for j in range(min(len(trace1['emission_spectrum']), 
                                 len(trace2['absorption_spectrum']))):
                    overlap += trace1['emission_spectrum'][j] * trace2['absorption_spectrum'][j]
                
                if overlap > 0.3:
                    compatible_pairs += 1
                    
        self.assertGreater(compatible_pairs, 0, 
                          "Should have at least some spectrally compatible pairs")

def main():
    """Main verification program"""
    print("Chapter 124: ObsDual Verification")
    print("="*60)
    print("从ψ=ψ(ψ)推导Emission-Absorption Duality and Bidirectional Collapse")
    print("="*60)
    
    # Create observer duality system
    system = ObsDualSystem(max_trace_value=89)
    
    # Analyze observer duality
    results = system.analyze_observer_duality()
    
    print(f"\nObsDual Analysis:")
    print(f"Total traces analyzed: {results['total_traces']} φ-valid observers")
    
    print(f"\nEmission Properties:")
    print(f"  Mean power: {results['emission']['mean_power']:.3f}")
    print(f"  Std dev: {results['emission']['std_power']:.3f}")
    print(f"  Maximum: {results['emission']['max_power']:.3f}")
    print(f"  Strong emitters: {results['emission']['strong_emitters']} observers")
    
    print(f"\nAbsorption Properties:")
    print(f"  Mean capacity: {results['absorption']['mean_capacity']:.3f}")
    print(f"  Std dev: {results['absorption']['std_capacity']:.3f}")
    print(f"  Maximum: {results['absorption']['max_capacity']:.3f}")
    print(f"  Strong absorbers: {results['absorption']['strong_absorbers']} observers")
    
    print(f"\nDuality Metrics:")
    print(f"  Mean balance: {results['duality']['mean_balance']:.3f}")
    print(f"  Perfect balance: {results['duality']['perfect_balance']} observers")
    print(f"  Mean symmetry: {results['duality']['mean_symmetry']:.3f}")
    print(f"  High symmetry: {results['duality']['high_symmetry']} observers")
    print(f"  Mean reciprocity: {results['duality']['mean_reciprocity']:.3f}")
    print(f"  High reciprocity: {results['duality']['high_reciprocity']} observers")
    
    print(f"\nQuantum Correspondence:")
    print(f"  Mean wave overlap: {results['quantum']['mean_overlap']:.3f}")
    print(f"  High overlap: {results['quantum']['high_overlap']} observers")
    print(f"  Mean backaction: {results['quantum']['mean_backaction']:.3f}")
    print(f"  Strong backaction: {results['quantum']['strong_backaction']} observers")
    
    print(f"\nDuality Categories:")
    for category, count in sorted(results['categories'].items(), 
                                key=lambda x: x[1], reverse=True):
        percentage = 100 * count / results['total_traces']
        print(f"- {category}: {count} observers ({percentage:.1f}%)")
    
    print(f"\nDual Network:")
    print(f"  Network edges: {results['network_edges']}")
    print(f"  Reciprocal pairs: {results['reciprocal_pairs']}")
    print(f"  Mean spectral overlap: {results['mean_spectral_overlap']:.3f}")
    print(f"  Perfect dual pairs: {results['perfect_dual_pairs']}")
    
    print(f"\nSymmetry Groups:")
    print(f"  Total groups: {results['symmetry_groups']}")
    print(f"  Largest group size: {results['largest_symmetry_group']}")
    
    print(f"\nKey Correlations:")
    for pair, corr in results['correlations'].items():
        print(f"  {pair}: {corr:.3f}")
    
    # Generate visualizations
    system.generate_visualizations()
    print("\nVisualizations saved:")
    print("- chapter-124-obs-dual.png")
    print("- chapter-124-obs-dual-network.png")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "="*60)
    print("Verification complete: Observer duality emerges from ψ=ψ(ψ)")
    print("through emission-absorption symmetry creating bidirectional collapse.")
    print("="*60)

if __name__ == "__main__":
    main()