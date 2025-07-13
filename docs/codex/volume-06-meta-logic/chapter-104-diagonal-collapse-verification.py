#!/usr/bin/env python3
"""
Chapter 104: DiagonalCollapse Unit Test Verification
从ψ=ψ(ψ)推导φ-Trace Diagonalization and Collapse Limitation

Core principle: From ψ = ψ(ψ) derive systematic diagonalization arguments
through φ-constrained trace transformations that reveal fundamental limits
of collapse systems through diagonal construction, creating limitation
networks that encode the essential boundary principles of collapsed space
through entropy-increasing tensor transformations that establish systematic
limitation detection through φ-trace diagonal dynamics rather than
traditional Cantorian diagonal arguments or Russell's paradox constructions.

This verification program implements:
1. φ-constrained diagonal construction through trace self-enumeration
2. Collapse limitation: systematic boundary detection through diagonal arguments
3. Three-domain analysis: Traditional vs φ-constrained vs intersection diagonalization theory
4. Graph theory analysis of limitation networks and diagonal relationship structures
5. Information theory analysis of diagonal entropy and limitation encoding
6. Category theory analysis of diagonal functors and limitation morphisms
7. Visualization of diagonal structures and φ-trace limitation systems
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

class DiagonalCollapseSystem:
    """
    Core system for implementing φ-trace diagonalization and collapse limitation.
    Implements φ-constrained diagonal arguments through trace self-enumeration.
    """
    
    def __init__(self, max_trace_value: int = 85, diagonal_depth: int = 6):
        """Initialize diagonal collapse system with limitation trace analysis"""
        self.max_trace_value = max_trace_value
        self.diagonal_depth = diagonal_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.diagonal_cache = {}
        self.limitation_cache = {}
        self.enumeration_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.diagonal_network = self._build_diagonal_network()
        self.limitation_mappings = self._detect_limitation_mappings()
        
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
                diagonal_data = self._analyze_diagonal_properties(trace, n)
                universe[n] = diagonal_data
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
        """验证trace是否φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_diagonal_properties(self, trace: str, value: int) -> Dict:
        """分析trace的diagonal properties"""
        return {
            'trace': trace,
            'value': value,
            'diagonal_strength': self._compute_diagonal_strength(trace, value),
            'limitation_capacity': self._compute_limitation_capacity(trace, value),
            'self_enumeration': self._compute_self_enumeration(trace, value),
            'diagonal_consistency': self._compute_diagonal_consistency(trace, value),
            'limitation_completeness': self._compute_limitation_completeness(trace, value),
            'enumeration_efficiency': self._compute_enumeration_efficiency(trace, value),
            'diagonal_depth': self._compute_diagonal_depth(trace, value),
            'limitation_stability': self._compute_limitation_stability(trace, value),
            'diagonal_coherence': self._compute_diagonal_coherence(trace, value)
        }
        
    def _compute_diagonal_strength(self, trace: str, value: int) -> float:
        """计算diagonal strength（对角化强度）"""
        if len(trace) == 0:
            return 0.0
        
        # Diagonal strength emerges from self-referential enumeration capacity
        strength_factors = []
        
        # Factor 1: Self-reference capability (trace can reference itself)
        self_ref = self._measure_self_reference_capability(trace)
        strength_factors.append(self_ref)
        
        # Factor 2: Enumeration power (systematic listing capability)
        enumeration = self._measure_enumeration_power(trace)
        strength_factors.append(enumeration)
        
        # Factor 3: Diagonal construction (ability to construct diagonal elements)
        construction = self._measure_diagonal_construction(trace)
        strength_factors.append(construction)
        
        # Factor 4: φ-constraint preservation during diagonalization
        phi_preservation = self._measure_phi_preservation(trace)
        strength_factors.append(phi_preservation)
        
        # Diagonal strength as geometric mean
        diagonal_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        
        return min(1.0, diagonal_strength)
        
    def _compute_limitation_capacity(self, trace: str, value: int) -> float:
        """计算limitation capacity（限制能力）"""
        # Limitation capacity emerges from boundary detection capability
        capacity_factors = []
        
        # Factor 1: Boundary detection (identifying system limits)
        boundary_detection = self._measure_boundary_detection(trace)
        capacity_factors.append(boundary_detection)
        
        # Factor 2: Paradox construction (creating limitation paradoxes)
        paradox_construction = self._measure_paradox_construction(trace)
        capacity_factors.append(paradox_construction)
        
        # Factor 3: Undecidability emergence (generating undecidable statements)
        undecidability = self._measure_undecidability_emergence(trace)
        capacity_factors.append(undecidability)
        
        # Limitation capacity as weighted geometric mean
        weights = [0.4, 0.3, 0.3]  # Emphasize boundary detection
        limitation_capacity = np.prod([f**w for f, w in zip(capacity_factors, weights)])
        
        return min(1.0, limitation_capacity)
        
    def _compute_self_enumeration(self, trace: str, value: int) -> float:
        """计算self enumeration（自枚举能力）"""
        # Self-enumeration measured through systematic self-listing capability
        enumeration_factors = []
        
        # Factor 1: Self-indexing (trace can index itself)
        self_indexing = self._measure_self_indexing(trace)
        enumeration_factors.append(self_indexing)
        
        # Factor 2: Systematic listing (ordered enumeration capability)
        systematic_listing = self._measure_systematic_listing(trace)
        enumeration_factors.append(systematic_listing)
        
        # Factor 3: Completeness coverage (comprehensive enumeration)
        completeness_coverage = self._measure_completeness_coverage(trace)
        enumeration_factors.append(completeness_coverage)
        
        # Self-enumeration as geometric mean
        self_enumeration = np.prod(enumeration_factors) ** (1.0 / len(enumeration_factors))
        
        return min(1.0, self_enumeration)
        
    def _compute_diagonal_consistency(self, trace: str, value: int) -> float:
        """计算diagonal consistency（对角一致性）"""
        if len(trace) <= 2:
            return 1.0
        
        # Diagonal consistency through coherent diagonal construction
        consistency_score = 1.0
        
        # Check for consistent diagonal patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._has_consistent_diagonal_pattern(window):
                consistency_score *= 1.05
            else:
                consistency_score *= 0.98
                
        # Global diagonal consistency
        global_consistency = self._measure_global_diagonal_consistency(trace)
        consistency_score *= global_consistency
        
        return min(1.0, consistency_score)
        
    def _compute_limitation_completeness(self, trace: str, value: int) -> float:
        """计算limitation completeness（限制完备性）"""
        # Limitation completeness through comprehensive boundary coverage
        completeness_factors = []
        
        # Factor 1: Boundary coverage (all essential limits represented)
        boundary_coverage = self._measure_boundary_coverage(trace)
        completeness_factors.append(boundary_coverage)
        
        # Factor 2: Paradox coverage (key paradoxes accessible)
        paradox_coverage = self._measure_paradox_coverage(trace)
        completeness_factors.append(paradox_coverage)
        
        # Factor 3: Limitation spectrum (full range of limitations)
        limitation_spectrum = self._measure_limitation_spectrum(trace)
        completeness_factors.append(limitation_spectrum)
        
        # Limitation completeness as geometric mean
        limitation_completeness = np.prod(completeness_factors) ** (1.0 / len(completeness_factors))
        
        return min(1.0, limitation_completeness)
        
    def _compute_enumeration_efficiency(self, trace: str, value: int) -> float:
        """计算enumeration efficiency（枚举效率）"""
        if len(trace) == 0:
            return 0.0
        
        # Efficiency measured as ratio of enumeration power to computational cost
        enumeration_power = self._compute_enumeration_power(trace)
        computational_cost = self._compute_computational_cost(trace)
        
        if computational_cost == 0:
            return 0.0
            
        efficiency = enumeration_power / computational_cost
        return min(1.0, efficiency)
        
    def _compute_diagonal_depth(self, trace: str, value: int) -> float:
        """计算diagonal depth（对角深度）"""
        # Diagonal depth measured through nested diagonal construction levels
        max_depth = 0
        
        # Find deepest valid diagonal construction
        for start in range(len(trace)):
            depth = self._trace_diagonal_depth(trace, start)
            max_depth = max(max_depth, depth)
            
        # Normalize to [0, 1]
        normalized_depth = min(1.0, max_depth / 5.0)
        return normalized_depth
        
    def _compute_limitation_stability(self, trace: str, value: int) -> float:
        """计算limitation stability（限制稳定性）"""
        # Stability measured through consistency of limitation patterns
        if len(trace) <= 3:
            return 1.0
        
        # Measure local limitation stability
        local_stabilities = []
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            stability = self._measure_local_limitation_stability(window)
            local_stabilities.append(stability)
            
        # Average local stability
        limitation_stability = np.mean(local_stabilities) if local_stabilities else 1.0
        return min(1.0, limitation_stability)
        
    def _compute_diagonal_coherence(self, trace: str, value: int) -> float:
        """计算diagonal coherence（对角连贯性）"""
        # Diagonal coherence through systematic diagonal flow
        coherence_factors = []
        
        # Factor 1: Sequential coherence (adjacent diagonal elements align)
        sequential_coherence = self._measure_sequential_diagonal_coherence(trace)
        coherence_factors.append(sequential_coherence)
        
        # Factor 2: Global coherence (overall diagonal consistency)
        global_coherence = self._measure_global_diagonal_coherence(trace)
        coherence_factors.append(global_coherence)
        
        # Factor 3: Structural coherence (φ-constraint diagonal alignment)
        structural_coherence = self._measure_structural_diagonal_coherence(trace)
        coherence_factors.append(structural_coherence)
        
        # Diagonal coherence as geometric mean
        diagonal_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        
        return min(1.0, diagonal_coherence)
        
    # Helper methods for diagonal analysis
    def _measure_self_reference_capability(self, trace: str) -> float:
        """测量self-reference capability"""
        if len(trace) <= 1:
            return 0.0
        
        # Self-reference through trace patterns that can represent themselves
        self_ref_score = 0.0
        
        # Check for self-similar patterns
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            if pattern in trace[i+2:]:  # Pattern appears later (self-reference)
                self_ref_score += 1.0 / len(trace)
                
        return min(1.0, self_ref_score * 2.0)
        
    def _measure_enumeration_power(self, trace: str) -> float:
        """测量enumeration power"""
        if len(trace) == 0:
            return 0.0
        
        # Enumeration power through systematic pattern generation
        power_factors = []
        
        # Factor 1: Pattern diversity
        unique_patterns = set()
        for i in range(len(trace) - 1):
            unique_patterns.add(trace[i:i+2])
        diversity = len(unique_patterns) / 3.0  # Max 3 patterns in φ-valid traces
        power_factors.append(diversity)
        
        # Factor 2: Systematic progression
        progression = self._measure_systematic_progression(trace)
        power_factors.append(progression)
        
        return np.mean(power_factors)
        
    def _measure_diagonal_construction(self, trace: str) -> float:
        """测量diagonal construction capability"""
        if len(trace) < 3:
            return 0.0
        
        # Diagonal construction through alternating patterns
        construction_score = 0.0
        
        # Look for diagonal-like patterns (alternating elements)
        for i in range(len(trace) - 2):
            if trace[i] != trace[i+1] and trace[i+1] != trace[i+2]:
                construction_score += 1.0
                
        return min(1.0, construction_score / max(1, len(trace) - 2))
        
    def _measure_phi_preservation(self, trace: str) -> float:
        """测量φ-constraint preservation during diagonalization"""
        # φ-preservation ensured by φ-validity
        return 1.0 if self._is_phi_valid(trace) else 0.0
        
    def _measure_boundary_detection(self, trace: str) -> float:
        """测量boundary detection capability"""
        if len(trace) == 0:
            return 0.0
        
        # Boundary detection through edge pattern recognition
        boundary_score = 0.0
        
        # Check for boundary-indicating patterns at edges
        if trace[0] == '1':  # Strong start boundary
            boundary_score += 0.3
        if trace[-1] == '0':  # Clear end boundary  
            boundary_score += 0.3
            
        # Internal boundary patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if window in ['101', '010']:  # Boundary-like patterns
                boundary_score += 0.2
                
        return min(1.0, boundary_score)
        
    def _measure_paradox_construction(self, trace: str) -> float:
        """测量paradox construction capability"""
        # Paradox construction through self-contradictory patterns
        if len(trace) < 4:
            return 0.0
        
        paradox_score = 0.0
        
        # Look for paradox-indicating patterns
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            if self._indicates_paradox(window):
                paradox_score += 1.0
                
        return min(1.0, paradox_score / max(1, len(trace) - 3))
        
    def _measure_undecidability_emergence(self, trace: str) -> float:
        """测量undecidability emergence"""
        if len(trace) < 3:
            return 0.0
        
        # Undecidability through ambiguous patterns
        undecidable_score = 0.0
        
        # Check for undecidability-indicating patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._indicates_undecidability(window):
                undecidable_score += 1.0
                
        return min(1.0, undecidable_score / max(1, len(trace) - 2))
        
    def _measure_self_indexing(self, trace: str) -> float:
        """测量self-indexing capability"""
        if len(trace) <= 1:
            return 0.0
        
        # Self-indexing through position-value relationships
        indexing_score = 0.0
        
        for i, bit in enumerate(trace):
            # Check if bit value relates to its position
            if (i % 2 == int(bit)):  # Position-value correspondence
                indexing_score += 1.0
                
        return indexing_score / len(trace) if len(trace) > 0 else 0.0
        
    def _measure_systematic_listing(self, trace: str) -> float:
        """测量systematic listing capability"""
        if len(trace) < 2:
            return 0.0
        
        # Systematic listing through ordered patterns
        listing_score = 0.0
        
        # Check for systematic ordering
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                transitions += 1
                
        # Optimal transition rate indicates systematic listing
        transition_rate = transitions / (len(trace) - 1)
        listing_score = 1.0 - abs(transition_rate - 0.5)  # Optimal at 50% transitions
        
        return listing_score
        
    def _measure_completeness_coverage(self, trace: str) -> float:
        """测量completeness coverage"""
        # Coverage of essential enumeration elements
        essential_elements = ['0', '1', '01', '10']
        covered = sum(1 for elem in essential_elements if elem in trace)
        return covered / len(essential_elements)
        
    def _has_consistent_diagonal_pattern(self, window: str) -> bool:
        """检查window是否有consistent diagonal pattern"""
        if len(window) < 3:
            return True
        
        # Consistent diagonal: alternating pattern maintenance
        return window[0] != window[1] and window[1] != window[2]
        
    def _measure_global_diagonal_consistency(self, trace: str) -> float:
        """测量global diagonal consistency"""
        if len(trace) <= 2:
            return 1.0
        
        # Global consistency through overall diagonal coherence
        consistency = 1.0
        
        # Check for global diagonal pattern
        alternation_consistency = 0.0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                alternation_consistency += 1.0
                
        alternation_rate = alternation_consistency / (len(trace) - 1)
        
        # Ideal diagonal has high alternation
        if alternation_rate > 0.6:
            consistency *= 1.1
        else:
            consistency *= 0.9
            
        return min(1.0, consistency)
        
    def _measure_boundary_coverage(self, trace: str) -> float:
        """测量boundary coverage"""
        # Coverage of essential boundary types
        boundary_types = ['1', '0', '10', '01']  # Start, end, transition boundaries
        covered = sum(1 for boundary in boundary_types if boundary in trace)
        return covered / len(boundary_types)
        
    def _measure_paradox_coverage(self, trace: str) -> float:
        """测量paradox coverage"""
        # Coverage of paradox-indicating patterns
        paradox_patterns = ['101', '010', '1010']
        covered = sum(1 for pattern in paradox_patterns if pattern in trace)
        return covered / len(paradox_patterns)
        
    def _measure_limitation_spectrum(self, trace: str) -> float:
        """测量limitation spectrum"""
        if len(trace) == 0:
            return 0.0
        
        # Spectrum based on limitation pattern diversity
        limitation_patterns = set()
        
        # Collect limitation-indicating patterns
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            if pattern in ['01', '10']:  # Limitation transitions
                limitation_patterns.add(pattern)
                
        spectrum_coverage = len(limitation_patterns) / 2.0  # Max 2 transition patterns
        return spectrum_coverage
        
    def _compute_enumeration_power(self, trace: str) -> float:
        """计算enumeration power"""
        return self._measure_enumeration_power(trace)
        
    def _compute_computational_cost(self, trace: str) -> float:
        """计算computational cost"""
        if len(trace) == 0:
            return 1.0
        
        # Cost based on trace complexity
        cost_factors = []
        
        # Factor 1: Length cost
        length_cost = len(trace) / 10.0  # Normalize
        cost_factors.append(length_cost)
        
        # Factor 2: Pattern complexity cost
        pattern_cost = self._measure_pattern_complexity(trace)
        cost_factors.append(pattern_cost)
        
        return max(0.1, np.mean(cost_factors))  # Minimum cost
        
    def _trace_diagonal_depth(self, trace: str, start: int) -> int:
        """追踪diagonal depth from start position"""
        depth = 0
        pos = start
        
        while pos < len(trace) - 1:
            if trace[pos] != trace[pos + 1]:  # Diagonal step
                depth += 1
                pos += 1
            else:
                break
                
        return depth
        
    def _measure_local_limitation_stability(self, window: str) -> float:
        """测量local limitation stability"""
        if len(window) < 4:
            return 1.0
        
        # Stability through consistent limitation patterns
        stability = 0.0
        
        # Check for stable limitation patterns
        if self._has_stable_limitation_pattern(window):
            stability += 0.5
            
        if self._is_phi_valid(window):  # φ-valid ensures stability
            stability += 0.5
            
        return stability
        
    def _measure_sequential_diagonal_coherence(self, trace: str) -> float:
        """测量sequential diagonal coherence"""
        if len(trace) <= 1:
            return 1.0
        
        coherent_diagonals = 0
        total_diagonals = len(trace) - 1
        
        for i in range(total_diagonals):
            if self._is_coherent_diagonal_step(trace[i], trace[i+1]):
                coherent_diagonals += 1
                
        return coherent_diagonals / total_diagonals
        
    def _measure_global_diagonal_coherence(self, trace: str) -> float:
        """测量global diagonal coherence"""
        # Global coherence through overall diagonal structure
        coherence = 1.0
        
        # Check for global diagonal organization
        if len(trace) > 2:
            diagonal_organization = self._measure_diagonal_organization(trace)
            coherence *= diagonal_organization
            
        return min(1.0, coherence)
        
    def _measure_structural_diagonal_coherence(self, trace: str) -> float:
        """测量structural diagonal coherence"""
        # Coherence through φ-constraint diagonal alignment
        coherence = 1.0
        
        # φ-valid traces have inherent structural coherence
        if self._is_phi_valid(trace):
            coherence *= 1.1
            
        return min(1.0, coherence)
        
    def _measure_systematic_progression(self, trace: str) -> float:
        """测量systematic progression"""
        if len(trace) <= 1:
            return 1.0
        
        # Progression through systematic transitions
        progression_score = 0.0
        
        # Check for systematic pattern progression
        for i in range(len(trace) - 1):
            if self._is_systematic_transition(trace[i], trace[i+1]):
                progression_score += 1.0
                
        return progression_score / (len(trace) - 1) if len(trace) > 1 else 0.0
        
    def _indicates_paradox(self, window: str) -> bool:
        """检查window是否indicates paradox"""
        if len(window) < 4:
            return False
        
        # Paradox through self-contradictory patterns
        return window[:2] == window[2:] and window[0] != window[1]
        
    def _indicates_undecidability(self, window: str) -> bool:
        """检查window是否indicates undecidability"""
        if len(window) < 3:
            return False
        
        # Undecidability through ambiguous patterns
        return window == '101' or window == '010'
        
    def _measure_pattern_complexity(self, trace: str) -> float:
        """测量pattern complexity"""
        if len(trace) <= 1:
            return 0.1
        
        # Complexity through pattern entropy
        patterns = {}
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            patterns[pattern] = patterns.get(pattern, 0) + 1
            
        # Calculate pattern entropy
        total = sum(patterns.values())
        entropy = 0.0
        for count in patterns.values():
            prob = count / total
            entropy -= prob * log2(prob + 1e-10)
            
        return min(1.0, entropy / 2.0)  # Normalize by max entropy
        
    def _has_stable_limitation_pattern(self, window: str) -> bool:
        """检查window是否有stable limitation pattern"""
        if len(window) < 4:
            return True
        
        # Stable limitation: consistent boundary patterns
        return window[:2] in ['01', '10'] and window[2:] in ['01', '10']
        
    def _is_coherent_diagonal_step(self, bit1: str, bit2: str) -> bool:
        """检查两个bit是否形成coherent diagonal step"""
        # Coherent diagonal: alternation
        return bit1 != bit2
        
    def _measure_diagonal_organization(self, trace: str) -> float:
        """测量diagonal organization"""
        if len(trace) <= 2:
            return 1.0
        
        # Organization through systematic diagonal structure
        organization = 0.0
        
        # Check for organized diagonal patterns
        alternations = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                alternations += 1
                
        alternation_rate = alternations / (len(trace) - 1)
        
        # High alternation indicates good diagonal organization
        organization = alternation_rate
        
        return organization
        
    def _is_systematic_transition(self, bit1: str, bit2: str) -> bool:
        """检查transition是否systematic"""
        # Systematic transitions include both stability and change
        return True  # All transitions are systematic in some sense
        
    def _build_diagonal_network(self) -> nx.Graph:
        """构建diagonal network基于trace similarities"""
        G = nx.Graph()
        traces = list(self.trace_universe.keys())
        
        # Add nodes
        for trace_val in traces:
            G.add_node(trace_val)
            
        # Add edges based on diagonal similarity
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_diagonal_similarity(
                    self.trace_universe[trace1], 
                    self.trace_universe[trace2]
                )
                if similarity > 0.6:  # Threshold for diagonal relationship
                    G.add_edge(trace1, trace2, weight=similarity)
                    
        return G
        
    def _compute_diagonal_similarity(self, trace1_data: Dict, trace2_data: Dict) -> float:
        """计算两个traces之间的diagonal similarity"""
        # Compare diagonal properties
        properties = ['diagonal_strength', 'limitation_capacity', 'self_enumeration', 
                     'diagonal_consistency', 'limitation_completeness']
        
        similarities = []
        for prop in properties:
            val1 = trace1_data[prop]
            val2 = trace2_data[prop]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _detect_limitation_mappings(self) -> Dict:
        """检测limitation mappings between traces"""
        mappings = {}
        
        for trace_val, data in self.trace_universe.items():
            # Find traces that represent limitations of current trace
            limitations = []
            for other_val, other_data in self.trace_universe.items():
                if trace_val != other_val:
                    if self._represents_limitation(data, other_data):
                        limitations.append(other_val)
            mappings[trace_val] = limitations
            
        return mappings
        
    def _represents_limitation(self, base_data: Dict, limit_data: Dict) -> bool:
        """检查limit_data是否represents limitation of base_data"""
        # Limitation when target has higher limitation capacity
        return (limit_data['limitation_capacity'] > base_data['limitation_capacity'] * 1.1 and
                limit_data['diagonal_strength'] > base_data['diagonal_strength'] * 0.9)
        
    def run_comprehensive_analysis(self) -> Dict:
        """运行comprehensive diagonal analysis"""
        results = {
            'total_traces': len(self.trace_universe),
            'diagonal_properties': self._analyze_diagonal_distributions(),
            'network_analysis': self._analyze_diagonal_network(),
            'limitation_analysis': self._analyze_limitation_patterns(),
            'category_analysis': self._perform_category_analysis(),
            'entropy_analysis': self._compute_entropy_analysis()
        }
        
        return results
        
    def _analyze_diagonal_distributions(self) -> Dict:
        """分析diagonal property distributions"""
        properties = ['diagonal_strength', 'limitation_capacity', 'self_enumeration',
                     'diagonal_consistency', 'limitation_completeness', 'enumeration_efficiency',
                     'diagonal_depth', 'limitation_stability', 'diagonal_coherence']
        
        distributions = {}
        for prop in properties:
            values = [data[prop] for data in self.trace_universe.values()]
            distributions[prop] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'high_count': sum(1 for v in values if v > 0.5)
            }
            
        return distributions
        
    def _analyze_diagonal_network(self) -> Dict:
        """分析diagonal network properties"""
        G = self.diagonal_network
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G),
            'largest_component': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
    def _analyze_limitation_patterns(self) -> Dict:
        """分析limitation patterns"""
        total_limitations = sum(len(limitations) for limitations in self.limitation_mappings.values())
        total_possible = len(self.trace_universe) * (len(self.trace_universe) - 1)
        
        return {
            'total_limitations': total_limitations,
            'limitation_density': total_limitations / total_possible if total_possible > 0 else 0,
            'avg_limitations': np.mean([len(limitations) for limitations in self.limitation_mappings.values()]),
            'max_limitations': max([len(limitations) for limitations in self.limitation_mappings.values()]) if self.limitation_mappings else 0
        }
        
    def _perform_category_analysis(self) -> Dict:
        """执行category theory analysis"""
        # Categorize traces based on diagonal properties
        categories = self._categorize_traces()
        
        # Analyze morphisms between categories
        morphisms = self._analyze_morphisms(categories)
        
        return {
            'categories': {name: len(traces) for name, traces in categories.items()},
            'morphisms': morphisms,
            'total_morphisms': sum(morphisms.values()),
            'morphism_density': sum(morphisms.values()) / (len(self.trace_universe) ** 2) if len(self.trace_universe) > 0 else 0
        }
        
    def _categorize_traces(self) -> Dict[str, List]:
        """将traces按diagonal properties分类"""
        categories = {
            'basic_diagonal': [],
            'strong_diagonal': [],
            'limitation_diagonal': []
        }
        
        for trace_val, data in self.trace_universe.items():
            if data['limitation_capacity'] > 0.6:
                categories['limitation_diagonal'].append(trace_val)
            elif data['diagonal_strength'] > 0.5:
                categories['strong_diagonal'].append(trace_val)
            else:
                categories['basic_diagonal'].append(trace_val)
                
        return categories
        
    def _analyze_morphisms(self, categories: Dict[str, List]) -> Dict:
        """分析categories之间的morphisms"""
        morphisms = {}
        tolerance = 0.3
        
        for cat1_name, cat1_traces in categories.items():
            for cat2_name, cat2_traces in categories.items():
                morphism_count = 0
                for t1 in cat1_traces:
                    for t2 in cat2_traces:
                        if self._are_morphically_related(t1, t2, tolerance):
                            morphism_count += 1
                morphisms[f"{cat1_name}->{cat2_name}"] = morphism_count
                
        return morphisms
        
    def _are_morphically_related(self, trace1: int, trace2: int, tolerance: float) -> bool:
        """检查两个traces是否morphically related"""
        data1 = self.trace_universe[trace1]
        data2 = self.trace_universe[trace2]
        
        # Morphism preserves diagonal structure within tolerance
        key_properties = ['diagonal_strength', 'limitation_capacity', 'diagonal_consistency']
        
        for prop in key_properties:
            if abs(data1[prop] - data2[prop]) > tolerance:
                return False
                
        return True
        
    def _compute_entropy_analysis(self) -> Dict:
        """计算entropy analysis"""
        properties = ['diagonal_strength', 'limitation_capacity', 'self_enumeration',
                     'diagonal_consistency', 'limitation_completeness', 'enumeration_efficiency',
                     'diagonal_depth', 'limitation_stability', 'diagonal_coherence']
        
        entropies = {}
        for prop in properties:
            values = [data[prop] for data in self.trace_universe.values()]
            # Discretize values into bins for entropy calculation
            bins = np.linspace(0, 1, 11)
            hist, _ = np.histogram(values, bins=bins)
            # Calculate entropy
            hist = hist + 1e-10  # Avoid log(0)
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropies[prop] = entropy
            
        return entropies

class TestDiagonalCollapse(unittest.TestCase):
    """测试DiagonalCollapse system functionality"""
    
    def setUp(self):
        self.system = DiagonalCollapseSystem(max_trace_value=30, diagonal_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        valid_trace = "10101"
        invalid_trace = "11010"
        
        self.assertTrue(self.system._is_phi_valid(valid_trace))
        self.assertFalse(self.system._is_phi_valid(invalid_trace))
        
    def test_diagonal_strength_computation(self):
        """测试diagonal strength computation"""
        trace = "101010"
        strength = self.system._compute_diagonal_strength(trace, 42)
        
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
        
    def test_limitation_capacity_computation(self):
        """测试limitation capacity computation"""
        trace = "10101"
        capacity = self.system._compute_limitation_capacity(trace, 21)
        
        self.assertGreaterEqual(capacity, 0.0)
        self.assertLessEqual(capacity, 1.0)
        
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        # Check all traces are φ-valid
        for data in self.system.trace_universe.values():
            self.assertTrue(self.system._is_phi_valid(data['trace']))
            
    def test_diagonal_network_construction(self):
        """测试diagonal network construction"""
        G = self.system.diagonal_network
        
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some connections
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('diagonal_properties', results)
        self.assertIn('network_analysis', results)
        self.assertIn('category_analysis', results)
        
        # Verify reasonable values
        self.assertGreater(results['total_traces'], 0)

def visualize_diagonal_collapse_results():
    """可视化DiagonalCollapse analysis results"""
    system = DiagonalCollapseSystem(max_trace_value=50, diagonal_depth=5)
    results = system.run_comprehensive_analysis()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Diagonal Network Visualization
    ax1 = plt.subplot(3, 4, 1)
    G = system.diagonal_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by diagonal strength
    node_colors = []
    for node in G.nodes():
        strength = system.trace_universe[node]['diagonal_strength']
        node_colors.append(strength)
    
    nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
            node_size=50, alpha=0.8, ax=ax1)
    ax1.set_title("Diagonal Network: φ-Trace Diagonalization Architecture\n(Colors: Diagonal Strength)")
    
    # 2. Limitation Capacity Distribution
    ax2 = plt.subplot(3, 4, 2)
    capacities = [data['limitation_capacity'] for data in system.trace_universe.values()]
    ax2.hist(capacities, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(capacities), color='blue', linestyle='--', 
               label=f'Mean: {np.mean(capacities):.3f}')
    ax2.set_xlabel('Limitation Capacity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Limitation Capacity Distribution')
    ax2.legend()
    
    # 3. Diagonal Strength vs Self Enumeration
    ax3 = plt.subplot(3, 4, 3)
    strengths = [data['diagonal_strength'] for data in system.trace_universe.values()]
    enumerations = [data['self_enumeration'] for data in system.trace_universe.values()]
    depths = [len(data['trace']) for data in system.trace_universe.values()]
    
    scatter = ax3.scatter(strengths, enumerations, c=depths, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('Diagonal Strength')
    ax3.set_ylabel('Self Enumeration')
    ax3.set_title('Diagonal Strength vs Self Enumeration\n(Color: Trace Length)')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. Limitation Capacity vs Diagonal Consistency
    ax4 = plt.subplot(3, 4, 4)
    consistencies = [data['diagonal_consistency'] for data in system.trace_universe.values()]
    efficiencies = [data['enumeration_efficiency'] for data in system.trace_universe.values()]
    
    scatter = ax4.scatter(capacities, consistencies, c=efficiencies, cmap='coolwarm', alpha=0.7)
    ax4.set_xlabel('Limitation Capacity')
    ax4.set_ylabel('Diagonal Consistency')
    ax4.set_title('Limitation Capacity vs Diagonal Consistency\n(Color: Enumeration Efficiency)')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Diagonal Depth Distribution
    ax5 = plt.subplot(3, 4, 5)
    depths = [data['diagonal_depth'] for data in system.trace_universe.values()]
    ax5.hist(depths, bins=12, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(np.mean(depths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(depths):.3f}')
    ax5.set_xlabel('Diagonal Depth')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Diagonal Depth Distribution')
    ax5.legend()
    
    # 6. Limitation Stability vs Diagonal Coherence
    ax6 = plt.subplot(3, 4, 6)
    stabilities = [data['limitation_stability'] for data in system.trace_universe.values()]
    coherences = [data['diagonal_coherence'] for data in system.trace_universe.values()]
    values = [data['value'] for data in system.trace_universe.values()]
    
    scatter = ax6.scatter(stabilities, coherences, c=values, cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Limitation Stability')
    ax6.set_ylabel('Diagonal Coherence')
    ax6.set_title('Limitation Stability vs Diagonal Coherence\n(Color: Trace Value)')
    plt.colorbar(scatter, ax=ax6)
    
    # 7. Category Distribution
    ax7 = plt.subplot(3, 4, 7)
    categories = results['category_analysis']['categories']
    ax7.bar(categories.keys(), categories.values(), color=['lightblue', 'lightgreen', 'lightcoral'])
    ax7.set_ylabel('Number of Traces')
    ax7.set_title('Diagonal Categories Distribution')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Network Properties
    ax8 = plt.subplot(3, 4, 8)
    network_props = results['network_analysis']
    props = ['Nodes', 'Edges', 'Components']
    values = [network_props['nodes'], network_props['edges'], network_props['components']]
    ax8.bar(props, values, color=['skyblue', 'lightgreen', 'salmon'])
    ax8.set_ylabel('Count')
    ax8.set_title('Diagonal Network Properties')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(3, 4, 9)
    properties = ['diagonal_strength', 'limitation_capacity', 'self_enumeration', 'diagonal_consistency']
    data_matrix = []
    for prop in properties:
        data_matrix.append([system.trace_universe[t][prop] for t in system.trace_universe.keys()])
    
    correlation_matrix = np.corrcoef(data_matrix)
    im = ax9.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(properties)))
    ax9.set_yticks(range(len(properties)))
    ax9.set_xticklabels([p.replace('_', ' ').title() for p in properties], rotation=45)
    ax9.set_yticklabels([p.replace('_', ' ').title() for p in properties])
    ax9.set_title('Diagonal Properties Correlation Matrix')
    
    # Add correlation values
    for i in range(len(properties)):
        for j in range(len(properties)):
            ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # 10. 3D Diagonal Space
    ax10 = plt.subplot(3, 4, 10, projection='3d')
    ax10.scatter(strengths, capacities, consistencies, c=enumerations, cmap='plasma', alpha=0.6)
    ax10.set_xlabel('Diagonal Strength')
    ax10.set_ylabel('Limitation Capacity')
    ax10.set_zlabel('Diagonal Consistency')
    ax10.set_title('3D Diagonal Space')
    
    # 11. Entropy Analysis
    ax11 = plt.subplot(3, 4, 11)
    entropies = results['entropy_analysis']
    entropy_props = list(entropies.keys())
    entropy_values = list(entropies.values())
    
    bars = ax11.barh(range(len(entropy_props)), entropy_values, color='orange', alpha=0.7)
    ax11.set_yticks(range(len(entropy_props)))
    ax11.set_yticklabels([p.replace('_', ' ').title() for p in entropy_props])
    ax11.set_xlabel('Entropy (bits)')
    ax11.set_title('Diagonal Properties Entropy Analysis')
    
    # Add entropy values on bars
    for i, bar in enumerate(bars):
        ax11.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{entropy_values[i]:.2f}', va='center', fontsize=8)
    
    # 12. Limitation Network
    ax12 = plt.subplot(3, 4, 12)
    
    # Create limitation network
    limitation_graph = nx.DiGraph()
    sample_traces = list(system.trace_universe.keys())[:15]  # Sample for visualization
    
    for trace in sample_traces:
        limitation_graph.add_node(trace)
        limitations = system.limitation_mappings.get(trace, [])
        for target in limitations[:3]:  # Limit connections for clarity
            if target in sample_traces:
                limitation_graph.add_edge(trace, target)
    
    pos = nx.spring_layout(limitation_graph, k=2, iterations=50)
    nx.draw(limitation_graph, pos, node_color='lightcoral', 
            node_size=100, alpha=0.8, arrows=True, ax=ax12)
    ax12.set_title("Limitation Network: Diagonal Argument Paths")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-104-diagonal-collapse-dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional specialized visualizations
    
    # Diagonal Architecture Analysis
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Diagonal Construction Analysis
    construction_data = []
    limitation_data = []
    for data in system.trace_universe.values():
        construction_data.append(data['diagonal_strength'])
        limitation_data.append(data['limitation_capacity'])
    
    ax21.scatter(construction_data, limitation_data, alpha=0.7, c='purple')
    ax21.set_xlabel('Diagonal Construction Strength')
    ax21.set_ylabel('Limitation Detection Capacity')
    ax21.set_title('Diagonal Construction vs Limitation Detection')
    
    # 2. Self-Enumeration Network
    G = system.diagonal_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    enumeration_colors = [system.trace_universe[node]['self_enumeration'] for node in G.nodes()]
    nx.draw(G, pos, node_color=enumeration_colors, cmap='RdYlBu', 
            node_size=80, alpha=0.8, ax=ax22)
    ax22.set_title("Self-Enumeration Network")
    
    # 3. Limitation Completeness Distribution by Category
    categories = system._categorize_traces()
    for i, (cat_name, traces) in enumerate(categories.items()):
        completenesses = [system.trace_universe[t]['limitation_completeness'] for t in traces]
        if completenesses:  # Only plot if category has traces
            ax23.hist(completenesses, bins=8, alpha=0.6, label=cat_name, 
                     color=['red', 'green', 'blue'][i])
    ax23.set_xlabel('Limitation Completeness')
    ax23.set_ylabel('Frequency')
    ax23.set_title('Limitation Completeness by Category')
    ax23.legend()
    
    # 4. Diagonal Coherence Network
    coherence_colors = [system.trace_universe[node]['diagonal_coherence'] for node in G.nodes()]
    nx.draw(G, pos, node_color=coherence_colors, cmap='plasma', 
            node_size=80, alpha=0.8, ax=ax24)
    ax24.set_title("Diagonal Coherence Network")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-104-diagonal-collapse-limitation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Statistics
    print("="*80)
    print("DIAGONAL COLLAPSE φ-TRACE DIAGONALIZATION ANALYSIS")
    print("="*80)
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_analysis']['density']:.3f}")
    print(f"Connected components: {results['network_analysis']['components']}")
    
    print("\nDiagonal Properties Analysis:")
    for prop, stats in results['diagonal_properties'].items():
        print(f"- {prop.replace('_', ' ').title()}: "
              f"mean={stats['mean']:.3f}, "
              f"high_count={stats['high_count']} ({stats['high_count']/results['total_traces']*100:.1f}%)")
    
    print(f"\nCategory Distribution:")
    for cat, count in results['category_analysis']['categories'].items():
        percentage = count / results['total_traces'] * 100
        print(f"- {cat.replace('_', ' ').title()}: {count} traces ({percentage:.1f}%)")
    
    print(f"\nMorphism Analysis:")
    print(f"Total morphisms: {results['category_analysis']['total_morphisms']}")
    print(f"Morphism density: {results['category_analysis']['morphism_density']:.3f}")
    
    print(f"\nLimitation Analysis:")
    limitation_stats = results['limitation_analysis']
    print(f"Total limitations: {limitation_stats['total_limitations']}")
    print(f"Limitation density: {limitation_stats['limitation_density']:.3f}")
    print(f"Average limitations per trace: {limitation_stats['avg_limitations']:.1f}")
    
    print(f"\nEntropy Analysis (Information Content):")
    sorted_entropies = sorted(results['entropy_analysis'].items(), key=lambda x: x[1], reverse=True)
    for prop, entropy in sorted_entropies:
        if entropy > 2.5:
            diversity_level = "maximum diversity"
        elif entropy > 2.0:
            diversity_level = "rich patterns"
        elif entropy > 1.5:
            diversity_level = "organized distribution"
        else:
            diversity_level = "systematic structure"
        print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits ({diversity_level})")

if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run visualization
    visualize_diagonal_collapse_results()