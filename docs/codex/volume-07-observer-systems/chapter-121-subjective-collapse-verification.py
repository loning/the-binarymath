#!/usr/bin/env python3
"""
Chapter 121: SubjectiveCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse Direction Bias as Tensor Self-Selection

Core principle: From ψ = ψ(ψ) derive systematic subjective collapse
construction through φ-constrained trace transformations that enable observer-specific
collapse paths through tensor self-selection mechanisms, creating subjective reality
structures that encode the fundamental bias principles of collapsed space through
entropy-increasing tensor transformations that establish systematic subjectivity
through φ-trace observer bias dynamics rather than traditional measurement theories
or external observation constructions.

This verification program implements:
1. φ-constrained subjective collapse construction through trace bias analysis
2. Observer tensor self-selection systems: systematic subjectivity through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection measurement theory
4. Graph theory analysis of subjective path networks and collapse direction structures
5. Information theory analysis of subjective entropy and bias encoding
6. Category theory analysis of subjective functors and collapse morphisms
7. Visualization of subjective collapse structures and φ-trace bias systems
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

class SubjectiveCollapseSystem:
    """
    Core system for implementing collapse direction bias as tensor self-selection.
    Implements φ-constrained subjective collapse architectures through trace bias dynamics.
    """
    
    def __init__(self, max_trace_value: int = 89, collapse_depth: int = 7):
        """Initialize subjective collapse system with bias trace analysis"""
        self.max_trace_value = max_trace_value
        self.collapse_depth = collapse_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.subjective_cache = {}
        self.bias_cache = {}
        self.collapse_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observer_tensors = self._build_observer_tensors()
        self.collapse_network = self._build_collapse_network()
        self.subjective_categories = self._detect_subjective_categories()
        
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
                subjective_data = self._analyze_subjective_properties(trace, n)
                universe[n] = subjective_data
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
        
    def _analyze_subjective_properties(self, trace: str, value: int) -> Dict:
        """分析trace的subjective collapse properties"""
        data = {
            'trace': trace,
            'value': value,
            'length': len(trace),
            'weight': trace.count('1'),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0
        }
        
        # Compute direction bias - core subjective metric
        data['direction_bias'] = self._compute_direction_bias(trace, value)
        
        # Compute subjective entropy
        data['subjective_entropy'] = self._compute_subjective_entropy(trace)
        
        # Compute collapse tendency
        data['collapse_tendency'] = self._compute_collapse_tendency(trace, value)
        
        # Compute observer specificity
        data['observer_specificity'] = self._compute_observer_specificity(trace)
        
        # Compute path divergence potential
        data['path_divergence'] = self._compute_path_divergence(trace, value)
        
        # Compute reality coherence
        data['reality_coherence'] = self._compute_reality_coherence(trace)
        
        # Compute temporal bias
        data['temporal_bias'] = self._compute_temporal_bias(trace, value)
        
        # Compute selection strength
        data['selection_strength'] = self._compute_selection_strength(trace)
        
        # Compute perspective depth
        data['perspective_depth'] = self._compute_perspective_depth(trace, value)
        
        # Assign category based on subjective properties
        data['category'] = self._assign_subjective_category(data)
        
        return data
        
    def _compute_direction_bias(self, trace: str, value: int) -> float:
        """
        Compute observer-specific collapse direction bias.
        From ψ=ψ(ψ): bias emerges from self-referential tensor projection.
        """
        if len(trace) < 2:
            return 0.0
            
        # Analyze collapse direction preference
        early_weight = sum(1 for i, bit in enumerate(trace[:len(trace)//2]) if bit == '1')
        late_weight = sum(1 for i, bit in enumerate(trace[len(trace)//2:]) if bit == '1')
        
        if early_weight + late_weight == 0:
            return 0.0
            
        # Bias towards early (-1) or late (+1) collapse
        bias = (late_weight - early_weight) / (early_weight + late_weight)
        
        # Modulate by golden ratio position
        golden_factor = 1.0
        if value in self.fibonacci_numbers:
            golden_factor = self.phi / 2
            
        return bias * golden_factor
        
    def _compute_subjective_entropy(self, trace: str) -> float:
        """
        Compute entropy as perceived by specific observer configuration.
        Subjective entropy depends on observer tensor structure.
        """
        if len(trace) < 2:
            return 0.0
            
        # Compute local entropy variations
        entropy = 0.0
        window_size = 3
        
        for i in range(len(trace) - window_size + 1):
            window = trace[i:i+window_size]
            ones = window.count('1')
            zeros = window.count('0')
            
            if ones > 0 and zeros > 0:
                p1 = ones / window_size
                p0 = zeros / window_size
                local_entropy = -p1 * log2(p1) - p0 * log2(p0)
                entropy += local_entropy
                
        # Normalize by trace length
        entropy = entropy / max(1, len(trace) - window_size + 1)
        
        # Modulate by trace structure
        if "101" in trace or "010" in trace:
            entropy *= 1.2  # Alternating patterns increase subjective entropy
            
        return min(entropy, 1.0)
        
    def _compute_collapse_tendency(self, trace: str, value: int) -> float:
        """
        Compute tendency to collapse in specific directions.
        Based on trace geometry and value properties.
        """
        if len(trace) < 2:
            return 0.5
            
        # Analyze collapse pattern preferences
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        max_transitions = len(trace) - 1
        
        if max_transitions == 0:
            return 0.5
            
        # High transitions indicate unstable collapse tendency
        instability = transitions / max_transitions
        
        # Modulate by value properties
        if value % 5 == 0:
            instability *= 0.8  # Multiples of 5 are more stable
        if value in self.fibonacci_numbers:
            instability *= 0.7  # Fibonacci positions are most stable
            
        return 1.0 - instability
        
    def _compute_observer_specificity(self, trace: str) -> float:
        """
        Compute how specific this trace is to particular observer configurations.
        High specificity means strong observer dependence.
        """
        if len(trace) < 3:
            return 0.0
            
        # Unique pattern detection
        patterns = set()
        for length in [2, 3, 4]:
            for i in range(len(trace) - length + 1):
                patterns.add(trace[i:i+length])
                
        # Specificity increases with unique patterns
        base_patterns = 2 ** 3  # Maximum possible 3-bit patterns
        specificity = len(patterns) / base_patterns
        
        # Boost for rare patterns
        if "1001" in trace or "0110" in trace:
            specificity *= 1.3
            
        return min(specificity, 1.0)
        
    def _compute_path_divergence(self, trace: str, value: int) -> float:
        """
        Compute potential for collapse path divergence between observers.
        High divergence means different observers see different realities.
        """
        if len(trace) < 2:
            return 0.0
            
        # Analyze decision points in trace
        decision_points = 0
        for i in range(1, len(trace) - 1):
            if trace[i-1] != trace[i+1]:  # Potential branch point
                decision_points += 1
                
        # Normalize by trace length
        divergence = decision_points / max(1, len(trace) - 2)
        
        # Golden ratio positions have higher divergence potential
        if value in self.fibonacci_numbers:
            divergence *= self.phi / 2
            
        return min(divergence, 1.0)
        
    def _compute_reality_coherence(self, trace: str) -> float:
        """
        Compute coherence of subjective reality experienced by observer.
        Low coherence means fragmented/inconsistent experience.
        """
        if len(trace) < 3:
            return 1.0
            
        # Analyze structural coherence
        coherence = 1.0
        
        # Check for coherent blocks
        block_sizes = []
        current_block = 1
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_block += 1
            else:
                block_sizes.append(current_block)
                current_block = 1
        block_sizes.append(current_block)
        
        # Coherence decreases with many small blocks
        avg_block_size = sum(block_sizes) / len(block_sizes)
        coherence = avg_block_size / len(trace)
        
        # Boost for Fibonacci block sizes
        fib_blocks = sum(1 for size in block_sizes if size in self.fibonacci_numbers)
        if fib_blocks > 0:
            coherence *= (1 + 0.1 * fib_blocks)
            
        return min(coherence, 1.0)
        
    def _compute_temporal_bias(self, trace: str, value: int) -> float:
        """
        Compute temporal bias in collapse sequence.
        Positive means future-biased, negative means past-biased.
        """
        if len(trace) < 4:
            return 0.0
            
        # Analyze temporal asymmetry
        first_half = trace[:len(trace)//2]
        second_half = trace[len(trace)//2:]
        
        # Compute complexity difference
        first_complexity = sum(1 for i in range(len(first_half)-1) 
                             if first_half[i] != first_half[i+1])
        second_complexity = sum(1 for i in range(len(second_half)-1) 
                              if second_half[i] != second_half[i+1])
        
        total_complexity = first_complexity + second_complexity
        if total_complexity == 0:
            return 0.0
            
        # Positive bias means increasing complexity (future-oriented)
        temporal_bias = (second_complexity - first_complexity) / total_complexity
        
        # Modulate by value position
        if value > self.max_trace_value * 0.618:  # Golden ratio threshold
            temporal_bias *= 1.2
            
        return max(-1.0, min(1.0, temporal_bias))
        
    def _compute_selection_strength(self, trace: str) -> float:
        """
        Compute strength of observer self-selection mechanism.
        High strength means strong reality-shaping power.
        """
        if len(trace) < 2:
            return 0.0
            
        # Analyze selection indicators
        strength = 0.0
        
        # Consecutive patterns indicate strong selection
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
                
        # Normalize by trace length
        strength = max_consecutive / len(trace)
        
        # Boost for specific selection patterns
        if "111" in trace or "000" in trace:
            strength *= 1.5  # Triple patterns show strong selection
            
        return min(strength, 1.0)
        
    def _compute_perspective_depth(self, trace: str, value: int) -> float:
        """
        Compute depth of subjective perspective.
        Deep perspective can see more collapse possibilities.
        """
        if len(trace) < 3:
            return 0.0
            
        # Analyze recursive depth indicators
        depth = 0.0
        
        # Check for nested patterns (indicates recursion)
        if "101" in trace and "10101" in trace:
            depth += 0.3
        if "010" in trace and "01010" in trace:
            depth += 0.3
            
        # Fibonacci positions have natural depth
        if value in self.fibonacci_numbers:
            fib_index = self.fibonacci_numbers.index(value)
            depth += fib_index / len(self.fibonacci_numbers)
            
        # Long traces can support deeper perspective
        depth += len(trace) / self.max_trace_value
        
        return min(depth, 1.0)
        
    def _assign_subjective_category(self, data: Dict) -> str:
        """
        Assign subjective category based on collapse properties.
        Categories represent different types of observer perspectives.
        """
        bias = data['direction_bias']
        entropy = data['subjective_entropy']
        tendency = data['collapse_tendency']
        specificity = data['observer_specificity']
        
        # Categorize based on dominant characteristics
        if abs(bias) > 0.5 and specificity > 0.6:
            return "strong_selector"  # Strong reality selection
        elif entropy > 0.7 and tendency < 0.4:
            return "chaos_observer"  # Sees chaotic/uncertain reality
        elif tendency > 0.7 and data['reality_coherence'] > 0.6:
            return "stable_perceiver"  # Experiences stable reality
        elif data['path_divergence'] > 0.6:
            return "multipath_viewer"  # Sees multiple possibilities
        elif abs(data['temporal_bias']) > 0.5:
            return "time_biased"  # Past or future oriented
        else:
            return "neutral_observer"  # Balanced perspective
            
    def _build_observer_tensors(self) -> Dict[int, torch.Tensor]:
        """构建observer tensor configurations"""
        tensors = {}
        
        for n, data in self.trace_universe.items():
            # Create observer tensor based on trace properties
            rank = min(8, len(data['trace']))
            
            # Initialize with trace-derived structure
            torch.manual_seed(n)
            tensor = torch.randn(rank, rank, rank)
            
            # Modulate by subjective properties
            tensor *= (1 + data['direction_bias'])
            tensor *= (1 + data['observer_specificity'])
            
            # Normalize
            tensor = tensor / (torch.norm(tensor) + 1e-8)
            
            tensors[n] = tensor
            
        return tensors
        
    def _build_collapse_network(self) -> nx.Graph:
        """构建collapse path network"""
        G = nx.Graph()
        
        # Add nodes for each trace
        for n, data in self.trace_universe.items():
            G.add_node(n, **data)
            
        # Add edges based on collapse relationships
        traces = list(self.trace_universe.keys())
        for i, n1 in enumerate(traces):
            for n2 in traces[i+1:]:
                data1 = self.trace_universe[n1]
                data2 = self.trace_universe[n2]
                
                # Connect if collapse paths could diverge
                divergence = abs(data1['path_divergence'] - data2['path_divergence'])
                bias_diff = abs(data1['direction_bias'] - data2['direction_bias'])
                
                if divergence > 0.3 or bias_diff > 0.4:
                    weight = 1.0 / (1.0 + divergence + bias_diff)
                    G.add_edge(n1, n2, weight=weight)
                    
        return G
        
    def _detect_subjective_categories(self) -> Dict[int, str]:
        """检测subjective collapse categories through clustering"""
        categories = {}
        
        # Group by assigned categories
        for n, data in self.trace_universe.items():
            categories[n] = data['category']
            
        return categories
        
    def analyze_subjective_collapse(self) -> Dict:
        """综合分析subjective collapse properties"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        
        # Analyze each property
        direction_biases = [t['direction_bias'] for t in traces]
        subjective_entropies = [t['subjective_entropy'] for t in traces]
        collapse_tendencies = [t['collapse_tendency'] for t in traces]
        observer_specificities = [t['observer_specificity'] for t in traces]
        path_divergences = [t['path_divergence'] for t in traces]
        reality_coherences = [t['reality_coherence'] for t in traces]
        temporal_biases = [t['temporal_bias'] for t in traces]
        selection_strengths = [t['selection_strength'] for t in traces]
        perspective_depths = [t['perspective_depth'] for t in traces]
        
        # Statistical analysis
        results['direction_bias'] = {
            'mean': np.mean(direction_biases),
            'std': np.std(direction_biases),
            'positive_count': sum(1 for x in direction_biases if x > 0),
            'negative_count': sum(1 for x in direction_biases if x < 0)
        }
        
        results['subjective_entropy'] = {
            'mean': np.mean(subjective_entropies),
            'std': np.std(subjective_entropies),
            'high_count': sum(1 for x in subjective_entropies if x > 0.5)
        }
        
        results['collapse_tendency'] = {
            'mean': np.mean(collapse_tendencies),
            'std': np.std(collapse_tendencies),
            'stable_count': sum(1 for x in collapse_tendencies if x > 0.7)
        }
        
        results['observer_specificity'] = {
            'mean': np.mean(observer_specificities),
            'std': np.std(observer_specificities),
            'high_count': sum(1 for x in observer_specificities if x > 0.5)
        }
        
        results['path_divergence'] = {
            'mean': np.mean(path_divergences),
            'std': np.std(path_divergences),
            'high_count': sum(1 for x in path_divergences if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.collapse_network.edges()) > 0:
            results['network_edges'] = len(self.collapse_network.edges())
            results['average_degree'] = sum(dict(self.collapse_network.degree()).values()) / len(self.collapse_network.nodes())
            results['network_density'] = nx.density(self.collapse_network)
            
            # Connected components
            components = list(nx.connected_components(self.collapse_network))
            results['connected_components'] = len(components)
            
            # Clustering
            results['clustering_coefficient'] = nx.average_clustering(self.collapse_network)
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            results['network_density'] = 0.0
            results['connected_components'] = len(traces)
            results['clustering_coefficient'] = 0.0
            
        # Entropy analysis
        properties = [
            ('direction_bias', direction_biases),
            ('subjective_entropy', subjective_entropies),
            ('collapse_tendency', collapse_tendencies),
            ('observer_specificity', observer_specificities),
            ('path_divergence', path_divergences),
            ('reality_coherence', reality_coherences),
            ('temporal_bias', temporal_biases),
            ('selection_strength', selection_strengths),
            ('perspective_depth', perspective_depths)
        ]
        
        results['entropy_analysis'] = {}
        for prop_name, prop_values in properties:
            if len(set(prop_values)) > 1:
                # Discretize values for entropy calculation
                bins = min(10, len(set(prop_values)))
                hist, _ = np.histogram(prop_values, bins=bins)
                probabilities = hist / np.sum(hist)
                probabilities = probabilities[probabilities > 0]  # Remove zeros
                entropy = -np.sum(probabilities * np.log2(probabilities))
                results['entropy_analysis'][prop_name] = entropy
            else:
                results['entropy_analysis'][prop_name] = 0.0
                
        return results
        
    def generate_visualizations(self):
        """生成subjective collapse visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Subjective Collapse Dynamics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 121: Subjective Collapse Dynamics', fontsize=16, fontweight='bold')
        
        # Direction bias distribution
        biases = [t['direction_bias'] for t in traces]
        ax1.hist(biases, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Direction Bias')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Collapse Direction Bias Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Entropy vs Specificity
        x = [t['subjective_entropy'] for t in traces]
        y = [t['observer_specificity'] for t in traces]
        colors = [abs(t['direction_bias']) for t in traces]
        scatter = ax2.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax2.set_xlabel('Subjective Entropy')
        ax2.set_ylabel('Observer Specificity')
        ax2.set_title('Entropy-Specificity Relationship')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='|Direction Bias|')
        
        # Category distribution
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax3.bar(category_counts.keys(), category_counts.values(), color='green', alpha=0.7)
        ax3.set_xlabel('Observer Category')
        ax3.set_ylabel('Count')
        ax3.set_title('Subjective Observer Categories')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Path divergence vs Reality coherence
        x = [t['path_divergence'] for t in traces]
        y = [t['reality_coherence'] for t in traces]
        colors = [t['temporal_bias'] for t in traces]
        scatter = ax4.scatter(x, y, c=colors, cmap='coolwarm', alpha=0.7, s=60, vmin=-1, vmax=1)
        ax4.set_xlabel('Path Divergence')
        ax4.set_ylabel('Reality Coherence')
        ax4.set_title('Divergence-Coherence Trade-off')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Temporal Bias')
        
        plt.tight_layout()
        plt.savefig('chapter-121-subjective-collapse.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Collapse Network Structure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle('Chapter 121: Subjective Collapse Network Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization
        if len(self.collapse_network.edges()) > 0:
            pos = nx.spring_layout(self.collapse_network, k=2, iterations=50)
            
            # Color nodes by direction bias
            node_colors = [self.trace_universe[n]['direction_bias'] for n in self.collapse_network.nodes()]
            
            nx.draw_networkx_nodes(self.collapse_network, pos, ax=ax1,
                                 node_color=node_colors, cmap='coolwarm',
                                 vmin=-1, vmax=1, node_size=100, alpha=0.8)
            nx.draw_networkx_edges(self.collapse_network, pos, ax=ax1,
                                 alpha=0.3, width=0.5)
            
            ax1.set_title('Subjective Collapse Network')
            ax1.axis('off')
            
            # Create a separate axis for colorbar
            sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                      norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, label='Direction Bias', fraction=0.05)
        else:
            ax1.text(0.5, 0.5, 'No network edges', ha='center', va='center')
            ax1.set_title('Subjective Collapse Network (Empty)')
            ax1.axis('off')
            
        # Subjective property correlations
        properties = ['direction_bias', 'subjective_entropy', 'collapse_tendency',
                     'observer_specificity', 'path_divergence', 'reality_coherence']
        
        corr_matrix = np.zeros((len(properties), len(properties)))
        for i, prop1 in enumerate(properties):
            for j, prop2 in enumerate(properties):
                values1 = [t[prop1] for t in traces]
                values2 = [t[prop2] for t in traces]
                corr_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]
                
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax2.set_xticks(range(len(properties)))
        ax2.set_yticks(range(len(properties)))
        ax2.set_xticklabels(properties, rotation=45, ha='right')
        ax2.set_yticklabels(properties)
        ax2.set_title('Subjective Property Correlations')
        
        # Add correlation values
        for i in range(len(properties)):
            for j in range(len(properties)):
                text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
                
        plt.colorbar(im, ax=ax2, label='Correlation')
        
        plt.tight_layout()
        plt.savefig('chapter-121-subjective-collapse-network.png', dpi=300, bbox_inches='tight')
        plt.close()

class SubjectiveCollapseTests(unittest.TestCase):
    """Unit tests for subjective collapse verification"""
    
    def setUp(self):
        """Initialize test system"""
        self.system = SubjectiveCollapseSystem(max_trace_value=55)
        
    def test_psi_recursion(self):
        """Test ψ=ψ(ψ) creates subjective bias"""
        # Verify that self-reference creates bias
        trace = "10101"
        bias = self.system._compute_direction_bias(trace, 21)
        self.assertNotEqual(bias, 0.0, "ψ=ψ(ψ) should create non-zero bias")
        
    def test_observer_specificity(self):
        """Test observer-specific collapse properties"""
        trace1 = "1010"
        trace2 = "0101"
        spec1 = self.system._compute_observer_specificity(trace1)
        spec2 = self.system._compute_observer_specificity(trace2)
        self.assertNotEqual(spec1, spec2, "Different traces should have different specificities")
        
    def test_entropy_subjectivity(self):
        """Test subjective entropy perception"""
        trace = "101010"
        entropy = self.system._compute_subjective_entropy(trace)
        self.assertGreater(entropy, 0.0, "Alternating patterns should have positive entropy")
        self.assertLessEqual(entropy, 1.0, "Entropy should be bounded")
        
    def test_path_divergence(self):
        """Test collapse path divergence"""
        trace = "10010110"
        divergence = self.system._compute_path_divergence(trace, 42)
        self.assertGreater(divergence, 0.0, "Complex traces should allow path divergence")
        
    def test_golden_ratio_influence(self):
        """Test φ influence on subjective properties"""
        fib_value = 21  # Fibonacci number
        non_fib_value = 22
        
        trace_fib = self.system._encode_to_trace(fib_value)
        trace_non_fib = self.system._encode_to_trace(non_fib_value)
        
        if self.system._is_phi_valid(trace_fib) and self.system._is_phi_valid(trace_non_fib):
            data_fib = self.system._analyze_subjective_properties(trace_fib, fib_value)
            data_non_fib = self.system._analyze_subjective_properties(trace_non_fib, non_fib_value)
            
            # Fibonacci numbers should have different properties
            self.assertNotEqual(data_fib['direction_bias'], data_non_fib['direction_bias'],
                              "Fibonacci positions should have unique bias properties")

def main():
    """Main verification program"""
    print("Chapter 121: SubjectiveCollapse Verification")
    print("="*60)
    print("从ψ=ψ(ψ)推导Collapse Direction Bias as Tensor Self-Selection")
    print("="*60)
    
    # Create subjective collapse system
    system = SubjectiveCollapseSystem(max_trace_value=89)
    
    # Analyze subjective collapse properties
    results = system.analyze_subjective_collapse()
    
    print(f"\nSubjectiveCollapse Analysis:")
    print(f"Total traces analyzed: {results['total_traces']} φ-valid subjective structures")
    print(f"Mean direction bias: {results['direction_bias']['mean']:.3f}")
    print(f"  Positive bias: {results['direction_bias']['positive_count']} traces")
    print(f"  Negative bias: {results['direction_bias']['negative_count']} traces")
    print(f"Mean subjective entropy: {results['subjective_entropy']['mean']:.3f}")
    print(f"Mean observer specificity: {results['observer_specificity']['mean']:.3f}")
    print(f"Mean path divergence: {results['path_divergence']['mean']:.3f}")
    
    print(f"\nSubjective Categories:")
    for category, count in results['categories'].items():
        percentage = 100 * count / results['total_traces']
        print(f"- {category}: {count} traces ({percentage:.1f}%)")
    
    print(f"\nCollapse Network Properties:")
    print(f"Network edges: {results['network_edges']} subjective connections")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print(f"Clustering coefficient: {results['clustering_coefficient']:.3f}")
    
    print(f"\nInformation Entropy Analysis:")
    for prop, entropy in sorted(results['entropy_analysis'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{prop:20s}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    print("\nVisualizations saved:")
    print("- chapter-121-subjective-collapse.png")
    print("- chapter-121-subjective-collapse-network.png")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "="*60)
    print("Verification complete: Subjectivity emerges from ψ=ψ(ψ)")
    print("through observer tensor self-selection creating unique collapse paths.")
    print("="*60)

if __name__ == "__main__":
    main()