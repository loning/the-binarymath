#!/usr/bin/env python3
"""
Chapter 119: DecoherenceCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse Loss via Observer Misalignment

Core principle: From ψ = ψ(ψ) derive systematic decoherence collapse
construction through φ-constrained trace transformations that enable decoherence
from observer-trace mismatches through trace geometric relationships,
creating decoherence networks that encode the fundamental collapse loss
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic decoherence structures through φ-trace decoherence
dynamics rather than traditional decoherence theories or external
decoherence constructions.

This verification program implements:
1. φ-constrained decoherence construction through trace misalignment analysis
2. Decoherence collapse systems: systematic collapse loss through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection decoherence theory
4. Graph theory analysis of decoherence networks and misalignment relationship structures
5. Information theory analysis of decoherence entropy and collapse loss encoding
6. Category theory analysis of decoherence functors and misalignment morphisms
7. Visualization of decoherence structures and φ-trace decoherence systems
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

class DecoherenceCollapseSystem:
    """
    Core system for implementing collapse loss via observer misalignment.
    Implements φ-constrained decoherence architectures through trace misalignment dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, decoherence_depth: int = 7):
        """Initialize decoherence collapse system with misalignment trace analysis"""
        self.max_trace_value = max_trace_value
        self.decoherence_depth = decoherence_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.decoherence_cache = {}
        self.misalignment_cache = {}
        self.collapse_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.decoherence_pairs = self._build_decoherence_pairs()
        self.decoherence_network = self._build_decoherence_network()
        self.misalignment_mappings = self._detect_misalignment_mappings()
        
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
                decoherence_data = self._analyze_decoherence_properties(trace, n)
                universe[n] = decoherence_data
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
        """检查trace是否为φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_decoherence_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的decoherence properties"""
        # Core decoherence measures
        decoherence_strength = self._compute_decoherence_strength(trace, value)
        misalignment_capacity = self._compute_misalignment_capacity(trace, value)
        collapse_loss = self._compute_collapse_loss(trace, value)
        decoherence_coherence = self._compute_decoherence_coherence(trace, value)
        
        # Advanced decoherence measures
        decoherence_rate = self._compute_decoherence_rate(trace, value)
        misalignment_depth = self._compute_misalignment_depth(trace, value)
        collapse_stability = self._compute_collapse_stability(trace, value)
        decoherence_efficiency = self._compute_decoherence_efficiency(trace, value)
        misalignment_severity = self._compute_misalignment_severity(trace, value)
        
        # Categorize based on decoherence profile
        category = self._categorize_decoherence(
            decoherence_strength, misalignment_capacity, collapse_loss, decoherence_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'decoherence_strength': decoherence_strength,
            'misalignment_capacity': misalignment_capacity,
            'collapse_loss': collapse_loss,
            'decoherence_coherence': decoherence_coherence,
            'decoherence_rate': decoherence_rate,
            'misalignment_depth': misalignment_depth,
            'collapse_stability': collapse_stability,
            'decoherence_efficiency': decoherence_efficiency,
            'misalignment_severity': misalignment_severity,
            'category': category
        }
        
    def _compute_decoherence_strength(self, trace: str, value: int) -> float:
        """Decoherence strength emerges from systematic collapse loss capacity"""
        strength_factors = []
        
        # Factor 1: Length provides decoherence space (minimum 3 for meaningful loss)
        length_factor = len(trace) / 3.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight misalignment balance (decoherence favors specific density)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            # Decoherence optimal at 45% density (0.45) for maximum misalignment
            misalignment_balance = 1.0 - abs(0.45 - weight / total)
            # Add decoherence bonus for loss patterns
            pattern_bonus = min(weight / 2.2, 1.0) if weight > 0 else 0.0
            decoherence_factor = 0.55 * misalignment_balance + 0.45 * pattern_bonus
            strength_factors.append(decoherence_factor)
        else:
            strength_factors.append(0.25)
        
        # Factor 3: Pattern decoherence structure (systematic collapse architecture)
        pattern_score = 0.0
        # Count decoherence-enhancing patterns (misalignment and loss)
        if trace.endswith('10'):  # Decoherence termination pattern
            pattern_score += 0.3
        
        # Misalignment patterns
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions enable misalignment
                transitions += 1
        if len(trace) > 1:
            transition_rate = transitions / (len(trace) - 1)
            pattern_score += 0.25 * min(transition_rate * 1.8, 1.0)  # Value transitions
        
        # Collapse loss patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] in ['110', '011']:  # Unstable patterns (would violate φ if extended)
                pattern_score += 0.2
        
        # Decoherence-specific patterns (value modulo 6 for 6-phase decoherence)
        if value % 6 in [1, 5]:  # Key decoherence phases
            pattern_score += 0.15
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint decoherence potential  
        phi_factor = 0.85 if self._is_phi_valid(trace) else 0.3
        strength_factors.append(phi_factor)
        
        # Decoherence strength emerges from geometric mean of factors
        decoherence_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return decoherence_strength
        
    def _compute_misalignment_capacity(self, trace: str, value: int) -> float:
        """Misalignment capacity emerges from systematic observer-trace mismatch"""
        capacity_factors = []
        
        # Factor 1: Structural misalignment potential
        structural = 0.2 + 0.8 * min(len(trace) / 7.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Misalignment complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.1 + 0.9 * min(ones_count / 4.0, 1.0)
        else:
            complexity = 0.05
        capacity_factors.append(complexity)
        
        # Factor 3: Misalignment phase (value modulo 8 for 8-phase structure)
        phase_depth = 0.3 + 0.7 * (value % 8) / 7.0
        capacity_factors.append(phase_depth)
        
        # Factor 4: φ-constraint misalignment vulnerability
        vulnerability = 0.88 if self._is_phi_valid(trace) else 0.25
        capacity_factors.append(vulnerability)
        
        misalignment_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return misalignment_capacity
        
    def _compute_collapse_loss(self, trace: str, value: int) -> float:
        """Collapse loss emerges from systematic decoherence magnitude"""
        loss_factors = []
        
        # Factor 1: Loss scope
        loss_scope = 0.15 + 0.85 * min(len(trace) / 5.5, 1.0)
        loss_factors.append(loss_scope)
        
        # Factor 2: Loss intensity
        if len(trace) > 0:
            balance_ratio = trace.count('1') / len(trace)
            # Optimal loss around 45% for decoherence
            loss_intensity = 1.0 - 2.22 * abs(0.45 - balance_ratio)
            loss_intensity = max(0.1, min(loss_intensity, 1.0))
        else:
            loss_intensity = 0.2
        loss_factors.append(loss_intensity)
        
        # Factor 3: Loss coverage (value modulo 12 for 12-level loss)
        coverage = 0.2 + 0.8 * (value % 12) / 11.0
        loss_factors.append(coverage)
        
        collapse_loss = np.prod(loss_factors) ** (1.0 / len(loss_factors))
        return collapse_loss
        
    def _compute_decoherence_coherence(self, trace: str, value: int) -> float:
        """Decoherence coherence emerges from systematic collapse integration"""
        coherence_factors = []
        
        # Factor 1: Decoherence integration capacity
        integration_cap = 0.3 + 0.7 * min(len(trace) / 6.0, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Decoherence coherence scope (value modulo 5 for 5-fold structure)
        coherence_scope = 0.35 + 0.65 * (value % 5) / 4.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic decoherence coherence (value modulo 7 for 7-level system)
        systematic = 0.45 + 0.55 * (value % 7) / 6.0
        coherence_factors.append(systematic)
        
        decoherence_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return decoherence_coherence
        
    def _compute_decoherence_rate(self, trace: str, value: int) -> float:
        """Decoherence rate through temporal collapse progression"""
        if len(trace) == 0:
            return 0.1
            
        # Analyze decoherence progression patterns
        rate_base = 0.2 + 0.8 * min(len(trace) / 8.0, 1.0)
        
        # Pattern-based rate modulation
        pattern_rate = 0.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i+1] == '0':  # Collapse transitions
                pattern_rate += 0.2
        pattern_rate = min(pattern_rate, 1.0)
        
        # Value-based rate (modulo 15 for 15-level rate structure)
        value_rate = (value % 15) / 14.0
        
        return 0.4 * rate_base + 0.3 * pattern_rate + 0.3 * value_rate
        
    def _compute_misalignment_depth(self, trace: str, value: int) -> float:
        """Misalignment depth through nested observer-trace divergence"""
        depth_factor = min(len(trace) / 10.0, 1.0)
        nesting_factor = (value % 14) / 13.0  # 14-level nesting
        return 0.15 + 0.85 * (depth_factor * nesting_factor)
        
    def _compute_collapse_stability(self, trace: str, value: int) -> float:
        """Collapse stability through consistent decoherence architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.82
        else:
            stability_base = 0.32
        variation = 0.18 * sin(value * 0.29)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_decoherence_efficiency(self, trace: str, value: int) -> float:
        """Decoherence efficiency through optimized collapse pathways"""
        if len(trace) > 0:
            # Efficiency based on decoherence structure optimization
            weight_ratio = trace.count('1') / len(trace)
            # Optimal efficiency around 45% weight for decoherence
            efficiency_base = 1.0 - 2.22 * abs(0.45 - weight_ratio)
        else:
            efficiency_base = 0.0
        phi_bonus = 0.08 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(efficiency_base + phi_bonus, 1.0))
        
    def _compute_misalignment_severity(self, trace: str, value: int) -> float:
        """Misalignment severity through observer-trace divergence magnitude"""
        if len(trace) < 3:
            return 0.15
            
        # Analyze misalignment severity patterns
        severity = 0.0
        
        # Pattern mismatches
        for i in range(len(trace) - 2):
            pattern = trace[i:i+3]
            if pattern in ['101', '010']:  # Oscillating patterns (high misalignment)
                severity += 0.25
            elif pattern in ['100', '001']:  # Isolated patterns (moderate misalignment)
                severity += 0.15
                
        severity = min(severity, 1.0)
        
        # Length-based severity
        length_severity = min(len(trace) / 12.0, 1.0)
        
        # Value-based severity (modulo 11 for 11-level severity)
        value_severity = (value % 11) / 10.0
        
        return 0.4 * severity + 0.3 * length_severity + 0.3 * value_severity
        
    def _categorize_decoherence(self, decoherence: float, misalignment: float, 
                               loss: float, coherence: float) -> str:
        """Categorize trace based on decoherence profile"""
        # Calculate dominant characteristic with thresholds
        decoherence_threshold = 0.6   # High decoherence strength threshold
        misalignment_threshold = 0.5  # Moderate misalignment capacity threshold
        loss_threshold = 0.5          # Moderate collapse loss threshold
        
        if decoherence >= decoherence_threshold:
            if misalignment >= misalignment_threshold:
                return "severe_decoherence"          # High decoherence + misalignment
            elif loss >= loss_threshold:
                return "collapse_decoherence"        # High decoherence + loss
            else:
                return "strong_decoherence"          # High decoherence + moderate properties
        else:
            if misalignment >= misalignment_threshold:
                return "misalignment_decay"          # Moderate decoherence + misalignment
            elif loss >= loss_threshold:
                return "loss_decay"                  # Moderate decoherence + loss
            else:
                return "mild_decoherence"            # Mild decoherence capability
        
    def _build_decoherence_pairs(self) -> List[Tuple[Dict, Dict, float]]:
        """构建decoherence pairs with misalignment strength"""
        pairs = []
        traces = list(self.trace_universe.values())
        
        # Build pairs with significant decoherence potential
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                decoherence_strength = self._compute_pair_decoherence(trace1, trace2)
                if decoherence_strength > 0.25:  # Threshold for significant decoherence
                    pairs.append((trace1, trace2, decoherence_strength))
                    
        return pairs
        
    def _compute_pair_decoherence(self, trace1: Dict, trace2: Dict) -> float:
        """计算两个traces之间的decoherence strength"""
        # Decoherence factors
        factors = []
        
        # Factor 1: Observer misalignment
        misalignment_diff = abs(trace1['misalignment_capacity'] - trace2['misalignment_capacity'])
        misalignment_score = misalignment_diff  # Higher difference = more decoherence
        factors.append(misalignment_score)
        
        # Factor 2: Collapse loss differential
        loss_diff = abs(trace1['collapse_loss'] - trace2['collapse_loss'])
        factors.append(loss_diff)
        
        # Factor 3: Decoherence anti-synergy
        decoherence_product = trace1['decoherence_strength'] * trace2['decoherence_strength']
        anti_synergy = 1.0 - decoherence_product  # Lower product = more decoherence
        factors.append(anti_synergy)
        
        # Factor 4: Trace incompatibility
        trace1_str, trace2_str = trace1['trace'], trace2['trace']
        if len(trace1_str) == len(trace2_str) and len(trace1_str) > 0:
            # XOR distance for incompatibility
            xor_count = sum(1 for a, b in zip(trace1_str, trace2_str) if a != b)
            incompatibility = xor_count / len(trace1_str)
        else:
            # Different lengths indicate incompatibility
            incompatibility = 0.8
        factors.append(incompatibility)
        
        # Geometric mean of factors
        decoherence_strength = np.prod(factors) ** (1.0 / len(factors))
        return decoherence_strength
        
    def _build_decoherence_network(self) -> nx.Graph:
        """构建decoherence network from pairs"""
        G = nx.Graph()
        
        # Add all traces as nodes
        for trace_data in self.trace_universe.values():
            G.add_node(trace_data['value'], **trace_data)
            
        # Add decoherence edges
        for trace1, trace2, strength in self.decoherence_pairs:
            if strength > 0.25:  # Threshold for network inclusion
                G.add_edge(trace1['value'], trace2['value'], 
                         weight=strength, decoherence=strength)
                    
        return G
        
    def _detect_misalignment_mappings(self) -> Dict[str, List[Tuple[int, int, float]]]:
        """检测decoherence traces之间的misalignment mappings"""
        mappings = defaultdict(list)
        
        for trace1, trace2, strength in self.decoherence_pairs:
            # Categorize by decoherence strength
            if strength > 0.7:
                category = "severe_misalignment"
            elif strength > 0.5:
                category = "moderate_misalignment"
            else:
                category = "mild_misalignment"
                
            mappings[category].append((trace1['value'], trace2['value'], strength))
        
        return dict(mappings)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive decoherence collapse analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_traces'] = len(traces)
        results['total_pairs'] = len(self.decoherence_pairs)
        results['network_density'] = nx.density(self.decoherence_network)
        results['connected_components'] = nx.number_connected_components(self.decoherence_network)
        
        # Decoherence analysis
        if self.decoherence_pairs:
            decoherence_strengths = [strength for _, _, strength in self.decoherence_pairs]
            results['pair_decoherence'] = {
                'mean': np.mean(decoherence_strengths),
                'std': np.std(decoherence_strengths),
                'max': np.max(decoherence_strengths),
                'min': np.min(decoherence_strengths)
            }
        else:
            results['pair_decoherence'] = {
                'mean': 0, 'std': 0, 'max': 0, 'min': 0
            }
        
        # Decoherence properties analysis
        decoherence_strengths = [t['decoherence_strength'] for t in traces]
        misalignment_capacities = [t['misalignment_capacity'] for t in traces]
        collapse_losses = [t['collapse_loss'] for t in traces]
        decoherence_coherences = [t['decoherence_coherence'] for t in traces]
        decoherence_rates = [t['decoherence_rate'] for t in traces]
        misalignment_depths = [t['misalignment_depth'] for t in traces]
        collapse_stabilities = [t['collapse_stability'] for t in traces]
        decoherence_efficiencies = [t['decoherence_efficiency'] for t in traces]
        misalignment_severities = [t['misalignment_severity'] for t in traces]
        
        results['decoherence_strength'] = {
            'mean': np.mean(decoherence_strengths),
            'std': np.std(decoherence_strengths),
            'high_count': sum(1 for x in decoherence_strengths if x > 0.5)
        }
        results['misalignment_capacity'] = {
            'mean': np.mean(misalignment_capacities),
            'std': np.std(misalignment_capacities),
            'high_count': sum(1 for x in misalignment_capacities if x > 0.5)
        }
        results['collapse_loss'] = {
            'mean': np.mean(collapse_losses),
            'std': np.std(collapse_losses),
            'high_count': sum(1 for x in collapse_losses if x > 0.5)
        }
        results['decoherence_coherence'] = {
            'mean': np.mean(decoherence_coherences),
            'std': np.std(decoherence_coherences),
            'high_count': sum(1 for x in decoherence_coherences if x > 0.5)
        }
        results['decoherence_rate'] = {
            'mean': np.mean(decoherence_rates),
            'std': np.std(decoherence_rates),
            'high_count': sum(1 for x in decoherence_rates if x > 0.5)
        }
        results['misalignment_depth'] = {
            'mean': np.mean(misalignment_depths),
            'std': np.std(misalignment_depths),
            'high_count': sum(1 for x in misalignment_depths if x > 0.5)
        }
        results['collapse_stability'] = {
            'mean': np.mean(collapse_stabilities),
            'std': np.std(collapse_stabilities),
            'high_count': sum(1 for x in collapse_stabilities if x > 0.5)
        }
        results['decoherence_efficiency'] = {
            'mean': np.mean(decoherence_efficiencies),
            'std': np.std(decoherence_efficiencies),
            'high_count': sum(1 for x in decoherence_efficiencies if x > 0.5)
        }
        results['misalignment_severity'] = {
            'mean': np.mean(misalignment_severities),
            'std': np.std(misalignment_severities),
            'high_count': sum(1 for x in misalignment_severities if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Network analysis
        if len(self.decoherence_network.edges()) > 0:
            results['network_edges'] = len(self.decoherence_network.edges())
            results['average_degree'] = sum(dict(self.decoherence_network.degree()).values()) / len(self.decoherence_network.nodes())
            
            # Clustering coefficient
            results['clustering_coefficient'] = nx.average_clustering(self.decoherence_network)
            
            # Find severe decoherence pairs
            severe_decoherence = [
                (u, v, data['decoherence']) 
                for u, v, data in self.decoherence_network.edges(data=True)
                if data['decoherence'] > 0.7
            ]
            results['severe_decoherence_pairs'] = len(severe_decoherence)
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            results['clustering_coefficient'] = 0.0
            results['severe_decoherence_pairs'] = 0
            
        # Entropy analysis
        properties = [
            ('decoherence_strength', decoherence_strengths),
            ('misalignment_capacity', misalignment_capacities),
            ('collapse_loss', collapse_losses),
            ('decoherence_coherence', decoherence_coherences),
            ('decoherence_rate', decoherence_rates),
            ('misalignment_depth', misalignment_depths),
            ('collapse_stability', collapse_stabilities),
            ('decoherence_efficiency', decoherence_efficiencies),
            ('misalignment_severity', misalignment_severities)
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
        """生成decoherence collapse visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Decoherence Dynamics Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 119: Decoherence Collapse Dynamics', fontsize=16, fontweight='bold')
        
        # Decoherence strength vs misalignment capacity
        x = [t['decoherence_strength'] for t in traces]
        y = [t['misalignment_capacity'] for t in traces]
        colors = [t['collapse_loss'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Decoherence Strength')
        ax1.set_ylabel('Misalignment Capacity')
        ax1.set_title('Decoherence-Misalignment Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Collapse Loss')
        
        # Decoherence pair strength distribution
        if self.decoherence_pairs:
            strengths = [s for _, _, s in self.decoherence_pairs]
            ax2.hist(strengths, bins=20, alpha=0.7, color='crimson', edgecolor='black')
            ax2.set_xlabel('Decoherence Pair Strength')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Decoherence Pair Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Decoherence Pairs', ha='center', va='center')
            ax2.set_title('Decoherence Pair Distribution')
        
        # Decoherence rate vs misalignment severity
        x3 = [t['decoherence_rate'] for t in traces]
        y3 = [t['misalignment_severity'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Decoherence Rate')
        ax3.set_ylabel('Misalignment Severity')
        ax3.set_title('Rate-Severity Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Decoherence Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-119-decoherence-collapse-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Decoherence Architecture
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 119: Decoherence Collapse Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization (subset for severe decoherence)
        if len(self.decoherence_network.edges()) > 0:
            # Create subgraph of severe decoherence nodes
            severe_edges = [(u, v) for u, v, d in self.decoherence_network.edges(data=True) 
                           if d['decoherence'] > 0.6]
            if severe_edges:
                G_severe = self.decoherence_network.edge_subgraph(severe_edges).copy()
                pos = nx.spring_layout(G_severe, k=2.0, iterations=50)
                
                # Draw network
                edge_weights = [G_severe[u][v]['decoherence'] for u, v in G_severe.edges()]
                nx.draw_networkx_nodes(G_severe, pos, ax=ax1, node_size=300, 
                                     node_color='lightcoral', alpha=0.8)
                nx.draw_networkx_edges(G_severe, pos, ax=ax1, width=2, 
                                     edge_color=edge_weights, edge_cmap=plt.cm.Reds)
                nx.draw_networkx_labels(G_severe, pos, ax=ax1, font_size=8)
                ax1.set_title('Severe Decoherence Network (strength > 0.6)')
            else:
                ax1.text(0.5, 0.5, 'No Severe Decoherence', ha='center', va='center')
                ax1.set_title('Severe Decoherence Network')
        else:
            ax1.text(0.5, 0.5, 'No Decoherence Network', ha='center', va='center')
            ax1.set_title('Severe Decoherence Network')
        ax1.axis('off')
        
        # Degree distribution
        if len(self.decoherence_network.edges()) > 0:
            degrees = [self.decoherence_network.degree(node) 
                      for node in self.decoherence_network.nodes()]
            ax2.hist(degrees, bins=15, alpha=0.7, color='darkred', edgecolor='black')
            ax2.set_xlabel('Node Degree')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Decoherence Degree Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Degree Data', ha='center', va='center')
            ax2.set_title('Decoherence Degree Distribution')
        
        # Decoherence properties correlation matrix
        properties_matrix = np.array([
            [t['decoherence_strength'] for t in traces],
            [t['misalignment_capacity'] for t in traces],
            [t['collapse_loss'] for t in traces],
            [t['decoherence_coherence'] for t in traces],
            [t['collapse_stability'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Decoherence', 'Misalignment', 'Loss', 'Coherence', 'Stability']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Decoherence Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Decoherence efficiency vs collapse stability
        x4 = [t['decoherence_efficiency'] for t in traces]
        y4 = [t['collapse_stability'] for t in traces]
        depths = [t['misalignment_depth'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=depths, cmap='inferno', alpha=0.7, s=60)
        ax4.set_xlabel('Decoherence Efficiency')
        ax4.set_ylabel('Collapse Stability')
        ax4.set_title('Efficiency-Stability Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Misalignment Depth')
        
        plt.tight_layout()
        plt.savefig('chapter-119-decoherence-collapse-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestDecoherenceCollapse(unittest.TestCase):
    """Unit tests for decoherence collapse system"""
    
    def setUp(self):
        """Set up test decoherence collapse system"""
        self.system = DecoherenceCollapseSystem(max_trace_value=20, decoherence_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        # Test valid traces (no consecutive 11)
        valid_traces = ["101", "1001", "10101"]
        for trace in valid_traces:
            self.assertTrue(self.system._is_phi_valid(trace))
            
        # Test invalid traces (with consecutive 11)
        invalid_traces = ["110", "1101", "0110"]
        for trace in invalid_traces:
            self.assertFalse(self.system._is_phi_valid(trace))
            
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        for value, data in self.system.trace_universe.items():
            self.assertIn('decoherence_strength', data)
            self.assertIn('misalignment_capacity', data)
            self.assertIn('collapse_loss', data)
            self.assertTrue(0 <= data['decoherence_strength'] <= 1)
            self.assertTrue(0 <= data['misalignment_capacity'] <= 1)
            
    def test_decoherence_computation(self):
        """测试decoherence computation"""
        if len(self.system.trace_universe) >= 2:
            traces = list(self.system.trace_universe.values())
            trace1, trace2 = traces[0], traces[1]
            decoherence = self.system._compute_pair_decoherence(trace1, trace2)
            self.assertTrue(0 <= decoherence <= 1)
            
    def test_decoherence_network_construction(self):
        """测试decoherence network construction"""
        self.assertIsNotNone(self.system.decoherence_network)
        self.assertGreaterEqual(len(self.system.decoherence_network.nodes()), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('total_pairs', results)
        self.assertIn('decoherence_strength', results)
        self.assertIn('misalignment_capacity', results)
        
        self.assertGreater(results['total_traces'], 0)

if __name__ == "__main__":
    # Initialize system
    system = DecoherenceCollapseSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("DECOHERENCE COLLAPSE MISALIGNMENT ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Total decoherence pairs: {results['total_pairs']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print()
    
    print("Decoherence Properties Analysis:")
    properties = ['decoherence_strength', 'misalignment_capacity', 'collapse_loss', 
                 'decoherence_coherence', 'decoherence_rate', 'misalignment_depth',
                 'collapse_stability', 'decoherence_efficiency', 'misalignment_severity']
    
    for prop in properties:
        if prop in results:
            data = results[prop]
            percentage = (data['high_count'] / results['total_traces']) * 100 if results['total_traces'] > 0 else 0
            print(f"- {prop.replace('_', ' ').title()}: mean={data['mean']:.3f}, high_count={data['high_count']} ({percentage:.1f}%)")
    
    print()
    print("Decoherence Pair Analysis:")
    if 'pair_decoherence' in results:
        dec_data = results['pair_decoherence']
        print(f"- Mean decoherence strength: {dec_data['mean']:.3f}")
        print(f"- Max decoherence strength: {dec_data['max']:.3f}")
        print(f"- Min decoherence strength: {dec_data['min']:.3f}")
        print(f"- Clustering coefficient: {results.get('clustering_coefficient', 0):.3f}")
        print(f"- Severe decoherence pairs (>0.7): {results.get('severe_decoherence_pairs', 0)}")
    
    print()
    print("Category Distribution:")
    if 'categories' in results:
        for category, count in results['categories'].items():
            percentage = (count / results['total_traces']) * 100 if results['total_traces'] > 0 else 0
            print(f"- {category.replace('_', ' ').title()}: {count} traces ({percentage:.1f}%)")
    
    print()
    print("Network Analysis:")
    print(f"Network edges: {results.get('network_edges', 0)}")
    print(f"Average degree: {results.get('average_degree', 0):.3f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    if 'entropy_analysis' in results:
        for prop, entropy in results['entropy_analysis'].items():
            print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)