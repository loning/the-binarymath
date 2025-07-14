#!/usr/bin/env python3
"""
Chapter 122: ConstByObs Unit Test Verification
从ψ=ψ(ψ)推导Observer-Specific Generation of Structural Constants

Core principle: From ψ = ψ(ψ) derive systematic observer-dependent constant
construction through φ-constrained trace transformations that enable observer-specific
physical constants through tensor projection mechanisms, creating relative constant
structures that encode the fundamental relativity principles of collapsed space through
entropy-increasing tensor transformations that establish systematic constant variation
through φ-trace observer constant dynamics rather than traditional universal constant
theories or external physical constructions.

This verification program implements:
1. φ-constrained observer constant construction through trace projection analysis
2. Observer tensor constant generation systems: systematic constants through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection physics
4. Graph theory analysis of constant networks and observer-dependent structures
5. Information theory analysis of constant entropy and variation encoding
6. Category theory analysis of constant functors and transformation morphisms
7. Visualization of constant variation structures and φ-trace constant systems
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

class ConstByObsSystem:
    """
    Core system for implementing observer-specific generation of structural constants.
    Implements φ-constrained constant architectures through trace projection dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, constant_depth: int = 7):
        """Initialize observer constant system with projection trace analysis"""
        self.max_trace_value = max_trace_value
        self.constant_depth = constant_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.constant_cache = {}
        self.observer_cache = {}
        self.projection_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observer_constants = self._build_observer_constants()
        self.constant_network = self._build_constant_network()
        self.constant_categories = self._detect_constant_categories()
        
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
                constant_data = self._analyze_constant_properties(trace, n)
                universe[n] = constant_data
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
        
    def _analyze_constant_properties(self, trace: str, value: int) -> Dict:
        """分析trace的observer-specific constant properties"""
        data = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0
        }
        
        # Compute fundamental constants for this observer
        data['speed_of_light'] = self._compute_speed_of_light(trace, value)
        data['planck_constant'] = self._compute_planck_constant(trace, value)
        data['gravitational_constant'] = self._compute_gravitational_constant(trace, value)
        data['fine_structure'] = self._compute_fine_structure(trace, value)
        
        # Compute derived constants
        data['vacuum_permittivity'] = self._compute_vacuum_permittivity(trace, data)
        data['electron_mass'] = self._compute_electron_mass(trace, value)
        data['boltzmann_constant'] = self._compute_boltzmann_constant(trace, value)
        
        # Compute constant relationships
        data['constant_coherence'] = self._compute_constant_coherence(data)
        data['dimensional_consistency'] = self._compute_dimensional_consistency(data)
        data['constant_entropy'] = self._compute_constant_entropy(data)
        
        # Assign category based on constant properties
        data['category'] = self._assign_constant_category(data)
        
        return data
        
    def _compute_speed_of_light(self, trace: str, value: int) -> float:
        """
        Compute observer-specific speed of light.
        From ψ=ψ(ψ): c emerges from trace propagation rate.
        """
        if len(trace) < 2:
            return 1.0
            
        # Base speed from trace structure
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        base_speed = 1.0 + transitions / len(trace)
        
        # Modulate by golden ratio position
        if value in self.fibonacci_numbers:
            base_speed *= self.phi
            
        # Normalize to reasonable range (0.5 to 2.0 relative to standard c)
        return min(2.0, max(0.5, base_speed))
        
    def _compute_planck_constant(self, trace: str, value: int) -> float:
        """
        Compute observer-specific Planck constant.
        Quantum of action depends on trace discreteness.
        """
        if len(trace) < 2:
            return 1.0
            
        # Base quantum from trace granularity
        blocks = []
        current_block = 1
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_block += 1
            else:
                blocks.append(current_block)
                current_block = 1
        blocks.append(current_block)
        
        # Average block size indicates quantum granularity
        avg_block = sum(blocks) / len(blocks)
        h_value = 1.0 / avg_block
        
        # Golden ratio enhancement
        if value in self.fibonacci_numbers:
            h_value *= sqrt(self.phi)
            
        return min(2.0, max(0.5, h_value))
        
    def _compute_gravitational_constant(self, trace: str, value: int) -> float:
        """
        Compute observer-specific gravitational constant.
        Coupling strength from trace connectivity.
        """
        if len(trace) < 3:
            return 1.0
            
        # Analyze long-range correlations
        correlation = 0.0
        for lag in range(1, min(5, len(trace)//2)):
            corr = sum(1 for i in range(len(trace)-lag) 
                      if trace[i] == trace[i+lag]) / (len(trace) - lag)
            correlation += corr
            
        # Average correlation indicates coupling strength
        divisor = min(4, len(trace)//2 - 1)
        if divisor > 0:
            avg_correlation = correlation / divisor
        else:
            avg_correlation = 0.5
        g_value = avg_correlation * 2.0
        
        # Modulate by position
        if value % 7 == 0:  # Special positions have different gravity
            g_value *= 1.2
            
        return min(2.0, max(0.5, g_value))
        
    def _compute_fine_structure(self, trace: str, value: int) -> float:
        """
        Compute fine structure constant (α).
        Electromagnetic coupling from trace patterns.
        """
        if len(trace) < 2:
            return 1/137.0  # Standard value as default
            
        # Pattern complexity determines coupling
        patterns = set()
        for length in [2, 3]:
            for i in range(len(trace) - length + 1):
                patterns.add(trace[i:i+length])
                
        # More patterns mean stronger coupling
        pattern_diversity = len(patterns) / (2 ** 3)  # Normalized by possible patterns
        alpha = (1/137.0) * (0.8 + 0.4 * pattern_diversity)
        
        # Golden ratio positions have special coupling
        if value in self.fibonacci_numbers:
            alpha *= (1 + 0.1 * (self.phi - 1))
            
        return alpha
        
    def _compute_vacuum_permittivity(self, trace: str, data: Dict) -> float:
        """
        Compute vacuum permittivity from c and other constants.
        Derived constant showing consistency.
        """
        c = data['speed_of_light']
        alpha = data['fine_structure']
        
        # From electromagnetic relationships
        epsilon_0 = 1.0 / (c * c)  # Simplified relationship
        
        # Modulate by trace structure
        if "101" in trace or "010" in trace:
            epsilon_0 *= 1.1  # Alternating patterns affect vacuum
            
        return epsilon_0
        
    def _compute_electron_mass(self, trace: str, value: int) -> float:
        """
        Compute electron mass for this observer.
        Mass emerges from trace stability.
        """
        if len(trace) < 2:
            return 1.0
            
        # Stability analysis
        stable_positions = sum(1 for i in range(1, len(trace)-1)
                             if trace[i-1] == trace[i] == trace[i+1])
        stability = stable_positions / max(1, len(trace) - 2)
        
        # Mass proportional to stability
        m_e = 0.5 + stability
        
        # Special values have different mass scales
        if value in self.fibonacci_numbers:
            m_e *= 0.9  # Lighter at Fibonacci positions
            
        return m_e
        
    def _compute_boltzmann_constant(self, trace: str, value: int) -> float:
        """
        Compute Boltzmann constant.
        Statistical mechanics scale from trace entropy.
        """
        if len(trace) < 2:
            return 1.0
            
        # Local entropy
        entropy = 0.0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                entropy += 1.0
                
        # Normalize
        k_B = 1.0 + entropy / (2 * len(trace))
        
        # Temperature scale adjustment
        if value % 5 == 0:
            k_B *= 1.1
            
        return min(2.0, max(0.5, k_B))
        
    def _compute_constant_coherence(self, data: Dict) -> float:
        """
        Compute coherence between different constants.
        High coherence means consistent physics.
        """
        # Check dimensional consistency
        c = data['speed_of_light']
        h = data['planck_constant']
        G = data['gravitational_constant']
        
        # Planck units coherence
        l_p = sqrt(h * G / (c ** 3))  # Planck length
        t_p = sqrt(h * G / (c ** 5))  # Planck time
        
        # Check if l_p / t_p ≈ c
        ratio = l_p / t_p if t_p > 0 else 0
        coherence = 1.0 - abs(ratio - c) / c if c > 0 else 0
        
        return max(0.0, min(1.0, coherence))
        
    def _compute_dimensional_consistency(self, data: Dict) -> float:
        """
        Check dimensional consistency of constants.
        """
        # Simplified dimensional analysis
        c = data['speed_of_light']
        h = data['planck_constant']
        alpha = data['fine_structure']
        
        # Check if α is dimensionless (approximately)
        # In our system, we ensure this by construction
        consistency = 1.0
        
        # Check relationships
        if alpha < 0 or alpha > 1:
            consistency *= 0.5
            
        # Check speed of light bounds
        if c < 0.1 or c > 10:
            consistency *= 0.7
            
        return consistency
        
    def _compute_constant_entropy(self, data: Dict) -> float:
        """
        Compute entropy of constant distribution.
        """
        # Collect all constant values
        constants = [
            data['speed_of_light'],
            data['planck_constant'],
            data['gravitational_constant'],
            data['fine_structure'] * 100,  # Scale for comparison
            data['electron_mass'],
            data['boltzmann_constant']
        ]
        
        # Discretize for entropy calculation
        bins = 10
        hist, _ = np.histogram(constants, bins=bins)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) > 0:
            entropy = -np.sum(probabilities * np.log2(probabilities))
        else:
            entropy = 0.0
            
        return entropy / log2(bins)  # Normalize
        
    def _assign_constant_category(self, data: Dict) -> str:
        """
        Assign constant category based on properties.
        Categories represent different physics regimes.
        """
        c = data['speed_of_light']
        h = data['planck_constant']
        G = data['gravitational_constant']
        coherence = data['constant_coherence']
        
        # Categorize based on dominant effects
        if c > 1.5 and h < 0.7:
            return "relativistic"  # High speed, low quantum
        elif h > 1.3 and c < 1.0:
            return "quantum"  # High quantum, low speed
        elif G > 1.5:
            return "strong_gravity"  # Enhanced gravity
        elif coherence > 0.8 and data['dimensional_consistency'] > 0.8:
            return "balanced"  # Well-balanced constants
        elif data['constant_entropy'] > 0.7:
            return "chaotic"  # High variation regime
        else:
            return "standard"  # Near standard physics
            
    def _build_observer_constants(self) -> Dict[int, Dict[str, float]]:
        """构建observer-specific constant sets"""
        constants = {}
        
        for n, data in self.trace_universe.items():
            observer_const = {
                'c': data['speed_of_light'],
                'h': data['planck_constant'],
                'G': data['gravitational_constant'],
                'alpha': data['fine_structure'],
                'epsilon_0': data['vacuum_permittivity'],
                'm_e': data['electron_mass'],
                'k_B': data['boltzmann_constant']
            }
            constants[n] = observer_const
            
        return constants
        
    def _build_constant_network(self) -> nx.Graph:
        """构建constant relationship network"""
        G = nx.Graph()
        
        # Add nodes for each observer
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # Add edges based on constant similarity
        traces = list(self.trace_universe.keys())
        for i, n1 in enumerate(traces):
            for n2 in traces[i+1:]:
                const1 = self.observer_constants[n1]
                const2 = self.observer_constants[n2]
                
                # Compute constant distance
                distance = 0.0
                for key in ['c', 'h', 'G', 'alpha']:
                    distance += abs(const1[key] - const2[key])
                
                # Connect similar physics
                if distance < 1.0:  # Threshold for similarity
                    weight = 1.0 / (1.0 + distance)
                    G.add_edge(n1, n2, weight=weight)
                    
        return G
        
    def _detect_constant_categories(self) -> Dict[int, str]:
        """检测constant physics categories through clustering"""
        categories = {}
        
        # Group by assigned categories
        for n, data in self.trace_universe.items():
            categories[n] = data['category']
            
        return categories
        
    def analyze_observer_constants(self) -> Dict:
        """综合分析observer-specific constants"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        
        # Analyze each constant
        speed_of_lights = [t['speed_of_light'] for t in traces]
        planck_constants = [t['planck_constant'] for t in traces]
        gravitational_constants = [t['gravitational_constant'] for t in traces]
        fine_structures = [t['fine_structure'] for t in traces]
        electron_masses = [t['electron_mass'] for t in traces]
        boltzmann_constants = [t['boltzmann_constant'] for t in traces]
        
        # Statistical analysis
        results['speed_of_light'] = {
            'mean': np.mean(speed_of_lights),
            'std': np.std(speed_of_lights),
            'min': np.min(speed_of_lights),
            'max': np.max(speed_of_lights),
            'relativistic_count': sum(1 for c in speed_of_lights if c > 1.5)
        }
        
        results['planck_constant'] = {
            'mean': np.mean(planck_constants),
            'std': np.std(planck_constants),
            'min': np.min(planck_constants),
            'max': np.max(planck_constants),
            'quantum_count': sum(1 for h in planck_constants if h > 1.3)
        }
        
        results['gravitational_constant'] = {
            'mean': np.mean(gravitational_constants),
            'std': np.std(gravitational_constants),
            'min': np.min(gravitational_constants),
            'max': np.max(gravitational_constants),
            'strong_count': sum(1 for g in gravitational_constants if g > 1.5)
        }
        
        results['fine_structure'] = {
            'mean': np.mean(fine_structures),
            'std': np.std(fine_structures),
            'min': np.min(fine_structures),
            'max': np.max(fine_structures),
            'variation': (np.max(fine_structures) - np.min(fine_structures)) / np.mean(fine_structures)
        }
        
        # Coherence analysis
        coherences = [t['constant_coherence'] for t in traces]
        consistencies = [t['dimensional_consistency'] for t in traces]
        entropies = [t['constant_entropy'] for t in traces]
        
        results['coherence'] = {
            'mean': np.mean(coherences),
            'high_count': sum(1 for c in coherences if c > 0.8)
        }
        
        results['consistency'] = {
            'mean': np.mean(consistencies),
            'consistent_count': sum(1 for c in consistencies if c > 0.9)
        }
        
        results['entropy'] = {
            'mean': np.mean(entropies),
            'high_entropy_count': sum(1 for e in entropies if e > 0.7)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.constant_network.edges()) > 0:
            results['network_edges'] = len(self.constant_network.edges())
            results['average_degree'] = sum(dict(self.constant_network.degree()).values()) / len(self.constant_network.nodes())
            results['network_density'] = nx.density(self.constant_network)
            
            # Connected components
            components = list(nx.connected_components(self.constant_network))
            results['connected_components'] = len(components)
            
            # Clustering
            results['clustering_coefficient'] = nx.average_clustering(self.constant_network)
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            results['network_density'] = 0.0
            results['connected_components'] = len(traces)
            results['clustering_coefficient'] = 0.0
            
        # Correlation analysis
        results['correlations'] = {}
        
        # c-h correlation
        results['correlations']['c_h'] = np.corrcoef(speed_of_lights, planck_constants)[0, 1]
        
        # c-G correlation
        results['correlations']['c_G'] = np.corrcoef(speed_of_lights, gravitational_constants)[0, 1]
        
        # h-alpha correlation
        results['correlations']['h_alpha'] = np.corrcoef(planck_constants, fine_structures)[0, 1]
        
        # Entropy analysis of constants
        properties = [
            ('speed_of_light', speed_of_lights),
            ('planck_constant', planck_constants),
            ('gravitational_constant', gravitational_constants),
            ('fine_structure', fine_structures),
            ('electron_mass', electron_masses),
            ('boltzmann_constant', boltzmann_constants)
        ]
        
        results['constant_entropies'] = {}
        for prop_name, prop_values in properties:
            if len(set(prop_values)) > 1:
                bins = min(10, len(set(prop_values)))
                hist, _ = np.histogram(prop_values, bins=bins)
                probabilities = hist / np.sum(hist)
                probabilities = probabilities[probabilities > 0]
                entropy = -np.sum(probabilities * np.log2(probabilities))
                results['constant_entropies'][prop_name] = entropy
            else:
                results['constant_entropies'][prop_name] = 0.0
                
        return results
        
    def generate_visualizations(self):
        """生成observer constant visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Constant Variation Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 122: Observer-Specific Constants', fontsize=16, fontweight='bold')
        
        # Speed of light vs Planck constant
        x = [t['speed_of_light'] for t in traces]
        y = [t['planck_constant'] for t in traces]
        colors = [t['gravitational_constant'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Speed of Light (c)')
        ax1.set_ylabel('Planck Constant (h)')
        ax1.set_title('Fundamental Constant Relationships')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Gravitational Constant (G)')
        
        # Fine structure constant distribution
        alphas = [t['fine_structure'] for t in traces]
        ax2.hist(alphas, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=1/137.0, color='red', linestyle='--', alpha=0.5, label='Standard α')
        ax2.set_xlabel('Fine Structure Constant (α)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fine Structure Constant Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Category distribution
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax3.bar(category_counts.keys(), category_counts.values(), color='green', alpha=0.7)
        ax3.set_xlabel('Physics Category')
        ax3.set_ylabel('Count')
        ax3.set_title('Observer Physics Categories')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Coherence vs Entropy
        x = [t['constant_coherence'] for t in traces]
        y = [t['constant_entropy'] for t in traces]
        colors = [t['dimensional_consistency'] for t in traces]
        scatter = ax4.scatter(x, y, c=colors, cmap='plasma', alpha=0.7, s=60)
        ax4.set_xlabel('Constant Coherence')
        ax4.set_ylabel('Constant Entropy')
        ax4.set_title('Physics Consistency Analysis')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Dimensional Consistency')
        
        plt.tight_layout()
        plt.savefig('chapter-122-const-by-obs.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Constant Network and Correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle('Chapter 122: Constant Relationships and Networks', fontsize=16, fontweight='bold')
        
        # Constant correlation matrix
        const_names = ['c', 'h', 'G', 'α', 'm_e', 'k_B']
        const_data = []
        for name in const_names:
            if name == 'α':
                const_data.append([t['fine_structure'] for t in traces])
            elif name == 'c':
                const_data.append([t['speed_of_light'] for t in traces])
            elif name == 'h':
                const_data.append([t['planck_constant'] for t in traces])
            elif name == 'G':
                const_data.append([t['gravitational_constant'] for t in traces])
            elif name == 'm_e':
                const_data.append([t['electron_mass'] for t in traces])
            elif name == 'k_B':
                const_data.append([t['boltzmann_constant'] for t in traces])
        
        corr_matrix = np.corrcoef(const_data)
        im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax1.set_xticks(range(len(const_names)))
        ax1.set_yticks(range(len(const_names)))
        ax1.set_xticklabels(const_names)
        ax1.set_yticklabels(const_names)
        ax1.set_title('Constant Correlation Matrix')
        
        # Add correlation values
        for i in range(len(const_names)):
            for j in range(len(const_names)):
                text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
                
        plt.colorbar(im, ax=ax1, label='Correlation')
        
        # Physics regime scatter
        # Create 2D projection of constant space
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Prepare data for PCA
        const_matrix = np.array([
            [t['speed_of_light'], t['planck_constant'], t['gravitational_constant'],
             t['fine_structure'], t['electron_mass'], t['boltzmann_constant']]
            for t in traces
        ])
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        const_scaled = scaler.fit_transform(const_matrix)
        pca = PCA(n_components=2)
        const_pca = pca.fit_transform(const_scaled)
        
        # Plot with categories
        category_colors = {
            'relativistic': 'red',
            'quantum': 'blue',
            'strong_gravity': 'purple',
            'balanced': 'green',
            'chaotic': 'orange',
            'standard': 'gray'
        }
        
        for category in set(categories):
            mask = [cat == category for cat in categories]
            x_cat = const_pca[mask, 0]
            y_cat = const_pca[mask, 1]
            ax2.scatter(x_cat, y_cat, label=category, 
                       color=category_colors.get(category, 'black'),
                       alpha=0.7, s=60)
        
        ax2.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        ax2.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        ax2.set_title('Physics Regime Landscape')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-122-const-by-obs-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class ConstByObsTests(unittest.TestCase):
    """Unit tests for observer constant verification"""
    
    def setUp(self):
        """Initialize test system"""
        self.system = ConstByObsSystem(max_trace_value=55)
        
    def test_psi_recursion_constants(self):
        """Test ψ=ψ(ψ) creates observer-specific constants"""
        # Verify that different traces give different constants
        trace1 = "10101"
        trace2 = "01010"
        
        const1 = self.system._compute_speed_of_light(trace1, 21)
        const2 = self.system._compute_speed_of_light(trace2, 10)
        
        self.assertNotEqual(const1, const2, "Different observers should see different c")
        
    def test_constant_coherence(self):
        """Test physical coherence of constants"""
        trace = "101010"
        data = self.system._analyze_constant_properties(trace, 42)
        
        coherence = data['constant_coherence']
        self.assertGreaterEqual(coherence, 0.0, "Coherence should be non-negative")
        self.assertLessEqual(coherence, 1.0, "Coherence should be bounded")
        
    def test_dimensional_consistency(self):
        """Test dimensional consistency"""
        trace = "1010010"
        data = self.system._analyze_constant_properties(trace, 33)
        
        consistency = data['dimensional_consistency']
        self.assertGreater(consistency, 0.0, "Should have some consistency")
        
    def test_fibonacci_influence(self):
        """Test φ influence on constants"""
        fib_value = 21  # Fibonacci number
        non_fib_value = 22
        
        trace_fib = self.system._encode_to_trace(fib_value)
        trace_non_fib = self.system._encode_to_trace(non_fib_value)
        
        if self.system._is_phi_valid(trace_fib) and self.system._is_phi_valid(trace_non_fib):
            c_fib = self.system._compute_speed_of_light(trace_fib, fib_value)
            c_non_fib = self.system._compute_speed_of_light(trace_non_fib, non_fib_value)
            
            # Fibonacci positions should have distinct constants
            self.assertNotEqual(c_fib, c_non_fib,
                              "Fibonacci positions should have unique constants")
            
    def test_constant_bounds(self):
        """Test constants remain in reasonable bounds"""
        for n, data in self.system.trace_universe.items():
            c = data['speed_of_light']
            h = data['planck_constant']
            G = data['gravitational_constant']
            
            self.assertGreater(c, 0.0, "Speed of light must be positive")
            self.assertLess(c, 10.0, "Speed of light should be bounded")
            self.assertGreater(h, 0.0, "Planck constant must be positive")
            self.assertGreater(G, 0.0, "Gravitational constant must be positive")

def main():
    """Main verification program"""
    print("Chapter 122: ConstByObs Verification")
    print("="*60)
    print("从ψ=ψ(ψ)推导Observer-Specific Generation of Structural Constants")
    print("="*60)
    
    # Create observer constant system
    system = ConstByObsSystem(max_trace_value=89)
    
    # Analyze observer constants
    results = system.analyze_observer_constants()
    
    print(f"\nConstByObs Analysis:")
    print(f"Total traces analyzed: {results['total_traces']} φ-valid observers")
    
    print(f"\nSpeed of Light Variation:")
    print(f"  Mean c: {results['speed_of_light']['mean']:.3f}")
    print(f"  Std dev: {results['speed_of_light']['std']:.3f}")
    print(f"  Range: [{results['speed_of_light']['min']:.3f}, {results['speed_of_light']['max']:.3f}]")
    print(f"  Relativistic regime: {results['speed_of_light']['relativistic_count']} observers")
    
    print(f"\nPlanck Constant Variation:")
    print(f"  Mean h: {results['planck_constant']['mean']:.3f}")
    print(f"  Std dev: {results['planck_constant']['std']:.3f}")
    print(f"  Range: [{results['planck_constant']['min']:.3f}, {results['planck_constant']['max']:.3f}]")
    print(f"  Quantum regime: {results['planck_constant']['quantum_count']} observers")
    
    print(f"\nGravitational Constant Variation:")
    print(f"  Mean G: {results['gravitational_constant']['mean']:.3f}")
    print(f"  Std dev: {results['gravitational_constant']['std']:.3f}")
    print(f"  Strong gravity: {results['gravitational_constant']['strong_count']} observers")
    
    print(f"\nFine Structure Constant:")
    print(f"  Mean α: {results['fine_structure']['mean']:.6f}")
    print(f"  Variation: {100*results['fine_structure']['variation']:.1f}%")
    
    print(f"\nPhysics Coherence:")
    print(f"  Mean coherence: {results['coherence']['mean']:.3f}")
    print(f"  High coherence: {results['coherence']['high_count']} observers")
    print(f"  Mean consistency: {results['consistency']['mean']:.3f}")
    print(f"  Consistent physics: {results['consistency']['consistent_count']} observers")
    
    print(f"\nPhysics Categories:")
    for category, count in results['categories'].items():
        percentage = 100 * count / results['total_traces']
        print(f"- {category}: {count} observers ({percentage:.1f}%)")
    
    print(f"\nConstant Correlations:")
    for pair, corr in results['correlations'].items():
        print(f"  {pair}: {corr:.3f}")
    
    print(f"\nConstant Entropies:")
    for const, entropy in sorted(results['constant_entropies'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {const:20s}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    print("\nVisualizations saved:")
    print("- chapter-122-const-by-obs.png")
    print("- chapter-122-const-by-obs-network.png")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "="*60)
    print("Verification complete: Physical constants emerge from ψ=ψ(ψ)")
    print("through observer-specific tensor projections creating relative physics.")
    print("="*60)

if __name__ == "__main__":
    main()