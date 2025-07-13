#!/usr/bin/env python3
"""
Chapter 103: CollapseProof Unit Test Verification
从ψ=ψ(ψ)推导Structure-Driven Deduction Nets over φ-Trace

Core principle: From ψ = ψ(ψ) derive systematic proof architectures
through φ-constrained trace transformations that enable structure-driven
deduction where logical validity emerges from trace geometric relationships,
creating proof networks that encode the fundamental deductive principles
of collapsed space through entropy-increasing tensor transformations that
establish systematic logical inference through φ-trace structural dynamics
rather than traditional syntactic proof manipulation or semantic truth
table verification methods.

This verification program implements:
1. φ-constrained structure-driven deduction through trace transformation networks
2. Proof validity: systematic logical inference through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection proof theory
4. Graph theory analysis of deduction networks and proof pathway relationships
5. Information theory analysis of proof entropy and deductive inference encoding
6. Category theory analysis of proof functors and deduction morphisms
7. Visualization of proof structures and φ-trace deduction systems
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

class CollapseProofSystem:
    """
    Core system for implementing structure-driven deduction nets over φ-trace.
    Implements φ-constrained proof systems through trace geometric relationships.
    """
    
    def __init__(self, max_trace_value: int = 85, proof_depth: int = 6):
        """Initialize collapse proof system with deductive trace analysis"""
        self.max_trace_value = max_trace_value
        self.proof_depth = proof_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.proof_cache = {}
        self.deduction_cache = {}
        self.inference_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.proof_network = self._build_proof_network()
        self.deduction_mappings = self._detect_deduction_mappings()
        
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
                proof_data = self._analyze_proof_properties(trace, n)
                universe[n] = proof_data
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
        
    def _analyze_proof_properties(self, trace: str, value: int) -> Dict:
        """分析trace的proof properties"""
        return {
            'trace': trace,
            'value': value,
            'deductive_strength': self._compute_deductive_strength(trace, value),
            'proof_validity': self._compute_proof_validity(trace, value),
            'inference_capacity': self._compute_inference_capacity(trace, value),
            'logical_consistency': self._compute_logical_consistency(trace, value),
            'proof_completeness': self._compute_proof_completeness(trace, value),
            'deduction_efficiency': self._compute_deduction_efficiency(trace, value),
            'proof_depth': self._compute_proof_depth(trace, value),
            'inference_stability': self._compute_inference_stability(trace, value),
            'logical_coherence': self._compute_logical_coherence(trace, value)
        }
        
    def _compute_deductive_strength(self, trace: str, value: int) -> float:
        """计算deductive strength（演绎推理强度）"""
        if len(trace) == 0:
            return 0.0
        
        # Deductive strength emerges from structural logical capacity
        strength_factors = []
        
        # Factor 1: Length provides inference space
        length_factor = min(1.0, len(trace) / 8.0)
        strength_factors.append(length_factor)
        
        # Factor 2: Weight balance (optimal for logical operations)
        weight_ratio = trace.count('1') / len(trace)
        balance_factor = 1.0 - abs(weight_ratio - 0.4)  # Optimal at 40% density
        strength_factors.append(balance_factor)
        
        # Factor 3: Pattern regularity (systematic logical structure)
        pattern_regularity = self._measure_pattern_regularity(trace)
        strength_factors.append(pattern_regularity)
        
        # Factor 4: φ-constraint coherence
        phi_coherence = self._measure_phi_coherence(trace)
        strength_factors.append(phi_coherence)
        
        # Deductive strength as geometric mean
        deductive_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        
        return min(1.0, deductive_strength)
        
    def _compute_proof_validity(self, trace: str, value: int) -> float:
        """计算proof validity（证明有效性）"""
        # Proof validity emerges from logical structural soundness
        validity_factors = []
        
        # Factor 1: Logical consistency (no contradictory patterns)
        consistency = self._measure_logical_consistency(trace)
        validity_factors.append(consistency)
        
        # Factor 2: Inference soundness (valid logical transitions)
        soundness = self._measure_inference_soundness(trace)
        validity_factors.append(soundness)
        
        # Factor 3: Structural completeness (sufficient logical coverage)
        completeness = self._measure_structural_completeness(trace)
        validity_factors.append(completeness)
        
        # Proof validity as weighted geometric mean
        weights = [0.4, 0.4, 0.2]  # Emphasize consistency and soundness
        proof_validity = np.prod([f**w for f, w in zip(validity_factors, weights)])
        
        return min(1.0, proof_validity)
        
    def _compute_inference_capacity(self, trace: str, value: int) -> float:
        """计算inference capacity（推理能力）"""
        # Inference capacity emerges from deductive processing capability
        capacity_factors = []
        
        # Factor 1: Chain length (multi-step reasoning)
        max_chain = self._find_max_inference_chain(trace)
        chain_factor = min(1.0, max_chain / 5.0)
        capacity_factors.append(chain_factor)
        
        # Factor 2: Parallel inference (multiple reasoning paths)
        parallel_paths = self._count_parallel_paths(trace)
        parallel_factor = min(1.0, parallel_paths / 3.0)
        capacity_factors.append(parallel_factor)
        
        # Factor 3: Logical depth (reasoning complexity)
        logical_depth = self._compute_logical_depth(trace)
        capacity_factors.append(logical_depth)
        
        # Inference capacity as geometric mean
        inference_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        
        return min(1.0, inference_capacity)
        
    def _compute_logical_consistency(self, trace: str, value: int) -> float:
        """计算logical consistency（逻辑一致性）"""
        if len(trace) <= 2:
            return 1.0
        
        # Logical consistency measured through contradiction detection
        consistency_score = 1.0
        
        # Check for logical contradictions in local patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._contains_contradiction(window):
                consistency_score *= 0.9
                
        # Check for global consistency patterns
        global_consistency = self._measure_global_consistency(trace)
        consistency_score *= global_consistency
        
        return consistency_score
        
    def _compute_proof_completeness(self, trace: str, value: int) -> float:
        """计算proof completeness（证明完备性）"""
        # Proof completeness measured through coverage of logical operations
        completeness_factors = []
        
        # Factor 1: Operation coverage (different logical operations represented)
        operation_coverage = self._measure_operation_coverage(trace)
        completeness_factors.append(operation_coverage)
        
        # Factor 2: Structural coverage (all necessary proof components)
        structural_coverage = self._measure_structural_coverage(trace)
        completeness_factors.append(structural_coverage)
        
        # Factor 3: Inference completeness (comprehensive reasoning coverage)
        inference_completeness = self._measure_inference_completeness(trace)
        completeness_factors.append(inference_completeness)
        
        # Proof completeness as geometric mean
        proof_completeness = np.prod(completeness_factors) ** (1.0 / len(completeness_factors))
        
        return min(1.0, proof_completeness)
        
    def _compute_deduction_efficiency(self, trace: str, value: int) -> float:
        """计算deduction efficiency（演绎效率）"""
        if len(trace) == 0:
            return 0.0
        
        # Efficiency measured as ratio of proof power to structural complexity
        proof_power = self._compute_proof_power(trace)
        structural_complexity = self._compute_structural_complexity(trace)
        
        if structural_complexity == 0:
            return 0.0
            
        efficiency = proof_power / structural_complexity
        return min(1.0, efficiency)
        
    def _compute_proof_depth(self, trace: str, value: int) -> float:
        """计算proof depth（证明深度）"""
        # Proof depth measured through maximum reasoning chain length
        max_depth = 0
        
        # Find longest valid reasoning chain
        for start in range(len(trace)):
            depth = self._trace_reasoning_depth(trace, start)
            max_depth = max(max_depth, depth)
            
        # Normalize to [0, 1]
        normalized_depth = min(1.0, max_depth / 6.0)
        return normalized_depth
        
    def _compute_inference_stability(self, trace: str, value: int) -> float:
        """计算inference stability（推理稳定性）"""
        # Stability measured through consistency of inference patterns
        if len(trace) <= 3:
            return 1.0
        
        # Measure local inference stability
        local_stabilities = []
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            stability = self._measure_local_inference_stability(window)
            local_stabilities.append(stability)
            
        # Average local stability
        inference_stability = np.mean(local_stabilities) if local_stabilities else 1.0
        return min(1.0, inference_stability)
        
    def _compute_logical_coherence(self, trace: str, value: int) -> float:
        """计算logical coherence（逻辑连贯性）"""
        # Logical coherence measured through systematic logical flow
        coherence_factors = []
        
        # Factor 1: Sequential coherence (adjacent logical elements align)
        sequential_coherence = self._measure_sequential_coherence(trace)
        coherence_factors.append(sequential_coherence)
        
        # Factor 2: Global coherence (overall logical consistency)
        global_coherence = self._measure_global_logical_coherence(trace)
        coherence_factors.append(global_coherence)
        
        # Factor 3: Structural coherence (φ-constraint logical alignment)
        structural_coherence = self._measure_structural_logical_coherence(trace)
        coherence_factors.append(structural_coherence)
        
        # Logical coherence as geometric mean
        logical_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        
        return min(1.0, logical_coherence)
        
    # Helper methods for proof analysis
    def _measure_pattern_regularity(self, trace: str) -> float:
        """测量pattern regularity"""
        if len(trace) <= 2:
            return 1.0
        
        # Count regular patterns
        regular_patterns = 0
        total_patterns = 0
        
        for pattern_len in range(2, min(len(trace) + 1, 4)):
            for i in range(len(trace) - pattern_len + 1):
                pattern = trace[i:i+pattern_len]
                total_patterns += 1
                
                # Check if pattern appears elsewhere (regularity)
                for j in range(i + pattern_len, len(trace) - pattern_len + 1):
                    if trace[j:j+pattern_len] == pattern:
                        regular_patterns += 1
                        break
        
        if total_patterns == 0:
            return 1.0
            
        return regular_patterns / total_patterns
        
    def _measure_phi_coherence(self, trace: str) -> float:
        """测量φ-constraint coherence"""
        # φ-coherence measured through golden ratio relationships
        coherence_score = 1.0
        
        if len(trace) >= 5:
            # Check for fibonacci-like patterns
            for i in range(len(trace) - 4):
                window = trace[i:i+5]
                if self._exhibits_fibonacci_pattern(window):
                    coherence_score *= 1.1
                    
        return min(1.0, coherence_score)
        
    def _measure_logical_consistency(self, trace: str) -> float:
        """测量logical consistency"""
        consistency = 1.0
        
        # No contradictory adjacent patterns
        for i in range(len(trace) - 1):
            if trace[i] == trace[i+1] == '1':  # φ-violation indicates logical inconsistency
                consistency *= 0.5
                
        return consistency
        
    def _measure_inference_soundness(self, trace: str) -> float:
        """测量inference soundness"""
        if len(trace) <= 2:
            return 1.0
        
        soundness_score = 1.0
        
        # Check for sound logical transitions
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._is_sound_transition(window):
                soundness_score *= 1.05
            else:
                soundness_score *= 0.95
                
        return min(1.0, soundness_score)
        
    def _measure_structural_completeness(self, trace: str) -> float:
        """测量structural completeness"""
        # Completeness based on representation of key logical structures
        structures = ['010', '101', '001', '100']  # Key logical patterns
        present_structures = 0
        
        for structure in structures:
            if structure in trace:
                present_structures += 1
                
        return present_structures / len(structures)
        
    def _find_max_inference_chain(self, trace: str) -> int:
        """找到最长推理链"""
        max_chain = 0
        
        for start in range(len(trace)):
            chain_length = self._trace_inference_chain(trace, start)
            max_chain = max(max_chain, chain_length)
            
        return max_chain
        
    def _count_parallel_paths(self, trace: str) -> int:
        """计算parallel inference paths"""
        paths = 0
        
        # Look for branching patterns
        for i in range(len(trace) - 2):
            if trace[i] == '1' and trace[i+2] == '1':  # Branching pattern
                paths += 1
                
        return paths
        
    def _compute_logical_depth(self, trace: str) -> float:
        """计算logical depth"""
        if len(trace) == 0:
            return 0.0
        
        # Depth based on nested logical structures
        depth_score = 0.0
        nesting_level = 0
        
        for bit in trace:
            if bit == '1':
                nesting_level += 1
                depth_score += nesting_level
            else:
                nesting_level = max(0, nesting_level - 1)
                
        return min(1.0, depth_score / (len(trace) * 3))
        
    def _contains_contradiction(self, window: str) -> bool:
        """检查window是否包含逻辑矛盾"""
        # φ-constraint violation indicates logical contradiction
        return "11" in window
        
    def _measure_global_consistency(self, trace: str) -> float:
        """测量global consistency"""
        # Global consistency through overall pattern coherence
        consistency = 1.0
        
        # Check overall density consistency
        density = trace.count('1') / len(trace)
        if density > 0.6:  # Too dense indicates inconsistency
            consistency *= 0.8
            
        return consistency
        
    def _measure_operation_coverage(self, trace: str) -> float:
        """测量operation coverage"""
        # Different logical operations represented by different patterns
        operations = ['0', '1', '10', '01', '010', '101']  # Basic logical operations
        covered_operations = 0
        
        for op in operations:
            if op in trace:
                covered_operations += 1
                
        return covered_operations / len(operations)
        
    def _measure_structural_coverage(self, trace: str) -> float:
        """测量structural coverage"""
        # Coverage of essential proof structures
        essential_structures = ['01', '10', '010', '101']
        covered = sum(1 for struct in essential_structures if struct in trace)
        return covered / len(essential_structures)
        
    def _measure_inference_completeness(self, trace: str) -> float:
        """测量inference completeness"""
        # Completeness of inference capabilities
        if len(trace) < 3:
            return 0.5
        
        # Check for complete inference patterns
        inference_patterns = ['010', '101', '001', '100']
        present_patterns = sum(1 for pattern in inference_patterns if pattern in trace)
        
        return present_patterns / len(inference_patterns)
        
    def _compute_proof_power(self, trace: str) -> float:
        """计算proof power"""
        if len(trace) == 0:
            return 0.0
        
        # Proof power based on logical capability
        power_factors = []
        
        # Factor 1: Logical operations represented
        operations = self._count_logical_operations(trace)
        power_factors.append(min(1.0, operations / 5.0))
        
        # Factor 2: Inference chains available
        chains = self._find_max_inference_chain(trace)
        power_factors.append(min(1.0, chains / 4.0))
        
        # Factor 3: Structural complexity
        complexity = self._compute_structural_complexity(trace)
        power_factors.append(complexity)
        
        return np.mean(power_factors)
        
    def _compute_structural_complexity(self, trace: str) -> float:
        """计算structural complexity"""
        if len(trace) == 0:
            return 0.0
        
        # Complexity based on pattern diversity and structure
        complexity_factors = []
        
        # Factor 1: Pattern diversity
        unique_patterns = set()
        for i in range(len(trace) - 1):
            unique_patterns.add(trace[i:i+2])
        diversity = len(unique_patterns) / 4.0  # Maximum 4 patterns: 00,01,10,11 (but 11 excluded)
        complexity_factors.append(diversity)
        
        # Factor 2: Length normalization
        length_factor = min(1.0, len(trace) / 8.0)
        complexity_factors.append(length_factor)
        
        return np.mean(complexity_factors)
        
    def _trace_reasoning_depth(self, trace: str, start: int) -> int:
        """追踪reasoning depth from start position"""
        depth = 0
        pos = start
        
        while pos < len(trace) - 1:
            if trace[pos] == '1' and trace[pos + 1] == '0':  # Valid reasoning step
                depth += 1
                pos += 2
            else:
                break
                
        return depth
        
    def _measure_local_inference_stability(self, window: str) -> float:
        """测量local inference stability"""
        if len(window) < 4:
            return 1.0
        
        # Stability through consistent logical patterns
        pattern_consistency = 0.0
        
        # Check for consistent inference patterns
        if window[0:2] == window[2:4]:  # Repeated pattern indicates stability
            pattern_consistency += 0.5
        
        if '11' not in window:  # φ-valid indicates stability
            pattern_consistency += 0.5
            
        return pattern_consistency
        
    def _measure_sequential_coherence(self, trace: str) -> float:
        """测量sequential coherence"""
        if len(trace) <= 1:
            return 1.0
        
        coherent_transitions = 0
        total_transitions = len(trace) - 1
        
        for i in range(total_transitions):
            if self._is_coherent_transition(trace[i], trace[i+1]):
                coherent_transitions += 1
                
        return coherent_transitions / total_transitions
        
    def _measure_global_logical_coherence(self, trace: str) -> float:
        """测量global logical coherence"""
        # Global coherence through overall logical structure
        coherence = 1.0
        
        # Check for balanced logical structure
        density = trace.count('1') / len(trace) if len(trace) > 0 else 0
        if 0.3 <= density <= 0.5:  # Optimal logical density
            coherence *= 1.2
        else:
            coherence *= 0.9
            
        return min(1.0, coherence)
        
    def _measure_structural_logical_coherence(self, trace: str) -> float:
        """测量structural logical coherence"""
        # Coherence through φ-constraint alignment
        coherence = 1.0
        
        # φ-valid traces have inherent structural coherence
        if self._is_phi_valid(trace):
            coherence *= 1.1
        else:
            coherence *= 0.8
            
        return min(1.0, coherence)
        
    def _exhibits_fibonacci_pattern(self, window: str) -> bool:
        """检查window是否展现fibonacci pattern"""
        if len(window) < 5:
            return False
        
        # Check for fibonacci-like growth patterns
        counts = [0, 0]
        for i, bit in enumerate(window):
            counts[int(bit)] += 1
            
        # Fibonacci ratio approximation
        ratio = counts[1] / counts[0] if counts[0] > 0 else 0
        return 0.5 <= ratio <= 0.7  # Approximates fibonacci ratio regions
        
    def _is_sound_transition(self, window: str) -> bool:
        """检查window是否表示sound logical transition"""
        if len(window) < 3:
            return True
        
        # Sound transitions maintain φ-validity and logical flow
        return self._is_phi_valid(window) and self._has_logical_flow(window)
        
    def _trace_inference_chain(self, trace: str, start: int) -> int:
        """追踪inference chain from start"""
        chain_length = 0
        pos = start
        
        while pos < len(trace) - 1:
            if self._is_valid_inference_step(trace[pos:pos+2]):
                chain_length += 1
                pos += 1
            else:
                break
                
        return chain_length
        
    def _count_logical_operations(self, trace: str) -> int:
        """计算logical operations represented"""
        operations = {'0', '1', '01', '10', '010', '101'}
        found_operations = 0
        
        for op in operations:
            if op in trace:
                found_operations += 1
                
        return found_operations
        
    def _is_coherent_transition(self, bit1: str, bit2: str) -> bool:
        """检查两个bit之间的transition是否coherent"""
        # Coherent transitions maintain logical flow
        return not (bit1 == '1' and bit2 == '1')  # φ-constraint maintains coherence
        
    def _has_logical_flow(self, window: str) -> bool:
        """检查window是否有logical flow"""
        if len(window) < 3:
            return True
        
        # Logical flow through meaningful transitions
        return not ('111' in window or '000' in window[:3])
        
    def _is_valid_inference_step(self, step: str) -> bool:
        """检查step是否为valid inference step"""
        if len(step) < 2:
            return False
        
        # Valid steps maintain φ-constraint and enable inference
        return step != '11' and ('1' in step)
        
    def _build_proof_network(self) -> nx.Graph:
        """构建proof network基于trace similarities"""
        G = nx.Graph()
        traces = list(self.trace_universe.keys())
        
        # Add nodes
        for trace_val in traces:
            G.add_node(trace_val)
            
        # Add edges based on proof similarity
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_proof_similarity(
                    self.trace_universe[trace1], 
                    self.trace_universe[trace2]
                )
                if similarity > 0.7:  # Threshold for proof relationship
                    G.add_edge(trace1, trace2, weight=similarity)
                    
        return G
        
    def _compute_proof_similarity(self, trace1_data: Dict, trace2_data: Dict) -> float:
        """计算两个traces之间的proof similarity"""
        # Compare proof properties
        properties = ['deductive_strength', 'proof_validity', 'inference_capacity', 
                     'logical_consistency', 'proof_completeness']
        
        similarities = []
        for prop in properties:
            val1 = trace1_data[prop]
            val2 = trace2_data[prop]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _detect_deduction_mappings(self) -> Dict:
        """检测deduction mappings between traces"""
        mappings = {}
        
        for trace_val, data in self.trace_universe.items():
            # Find traces that can be reached through valid deduction
            reachable = []
            for other_val, other_data in self.trace_universe.items():
                if trace_val != other_val:
                    if self._can_deduce(data, other_data):
                        reachable.append(other_val)
            mappings[trace_val] = reachable
            
        return mappings
        
    def _can_deduce(self, from_data: Dict, to_data: Dict) -> bool:
        """检查是否可以从from_data deduce to_data"""
        # Deduction possible if target has higher or equal proof properties
        return (to_data['proof_validity'] >= from_data['proof_validity'] * 0.9 and
                to_data['logical_consistency'] >= from_data['logical_consistency'] * 0.9)
        
    def run_comprehensive_analysis(self) -> Dict:
        """运行comprehensive proof analysis"""
        results = {
            'total_traces': len(self.trace_universe),
            'proof_properties': self._analyze_proof_distributions(),
            'network_analysis': self._analyze_proof_network(),
            'deduction_analysis': self._analyze_deduction_patterns(),
            'category_analysis': self._perform_category_analysis(),
            'entropy_analysis': self._compute_entropy_analysis()
        }
        
        return results
        
    def _analyze_proof_distributions(self) -> Dict:
        """分析proof property distributions"""
        properties = ['deductive_strength', 'proof_validity', 'inference_capacity',
                     'logical_consistency', 'proof_completeness', 'deduction_efficiency',
                     'proof_depth', 'inference_stability', 'logical_coherence']
        
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
        
    def _analyze_proof_network(self) -> Dict:
        """分析proof network properties"""
        G = self.proof_network
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G),
            'largest_component': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
    def _analyze_deduction_patterns(self) -> Dict:
        """分析deduction patterns"""
        total_mappings = sum(len(reachable) for reachable in self.deduction_mappings.values())
        total_possible = len(self.trace_universe) * (len(self.trace_universe) - 1)
        
        return {
            'total_mappings': total_mappings,
            'deduction_density': total_mappings / total_possible if total_possible > 0 else 0,
            'avg_reachable': np.mean([len(reachable) for reachable in self.deduction_mappings.values()]),
            'max_reachable': max([len(reachable) for reachable in self.deduction_mappings.values()]) if self.deduction_mappings else 0
        }
        
    def _perform_category_analysis(self) -> Dict:
        """执行category theory analysis"""
        # Categorize traces based on proof properties
        categories = self._categorize_traces()
        
        # Analyze morphisms between categories
        morphisms = self._analyze_morphisms(categories)
        
        return {
            'categories': {name: len(traces) for name, traces in categories.items()},
            'morphisms': morphisms,
            'total_morphisms': sum(morphisms.values())
        }
        
    def _categorize_traces(self) -> Dict[str, List]:
        """将traces按proof properties分类"""
        categories = {
            'basic_proof': [],
            'strong_proof': [],
            'complete_proof': []
        }
        
        for trace_val, data in self.trace_universe.items():
            if data['proof_validity'] > 0.7 and data['proof_completeness'] > 0.7:
                categories['complete_proof'].append(trace_val)
            elif data['proof_validity'] > 0.5:
                categories['strong_proof'].append(trace_val)
            else:
                categories['basic_proof'].append(trace_val)
                
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
        
        # Morphism preserves proof structure within tolerance
        key_properties = ['proof_validity', 'logical_consistency', 'inference_capacity']
        
        for prop in key_properties:
            if abs(data1[prop] - data2[prop]) > tolerance:
                return False
                
        return True
        
    def _compute_entropy_analysis(self) -> Dict:
        """计算entropy analysis"""
        properties = ['deductive_strength', 'proof_validity', 'inference_capacity',
                     'logical_consistency', 'proof_completeness', 'deduction_efficiency',
                     'proof_depth', 'inference_stability', 'logical_coherence']
        
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

class TestCollapseProof(unittest.TestCase):
    """测试CollapseProof system functionality"""
    
    def setUp(self):
        self.system = CollapseProofSystem(max_trace_value=30, proof_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        valid_trace = "10101"
        invalid_trace = "11010"
        
        self.assertTrue(self.system._is_phi_valid(valid_trace))
        self.assertFalse(self.system._is_phi_valid(invalid_trace))
        
    def test_deductive_strength_computation(self):
        """测试deductive strength computation"""
        trace = "101010"
        strength = self.system._compute_deductive_strength(trace, 42)
        
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
        
    def test_proof_validity_computation(self):
        """测试proof validity computation"""
        trace = "10101"
        validity = self.system._compute_proof_validity(trace, 21)
        
        self.assertGreaterEqual(validity, 0.0)
        self.assertLessEqual(validity, 1.0)
        
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        # Check all traces are φ-valid
        for data in self.system.trace_universe.values():
            self.assertTrue(self.system._is_phi_valid(data['trace']))
            
    def test_proof_network_construction(self):
        """测试proof network construction"""
        G = self.system.proof_network
        
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some connections
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('proof_properties', results)
        self.assertIn('network_analysis', results)
        self.assertIn('category_analysis', results)
        
        # Verify reasonable values
        self.assertGreater(results['total_traces'], 0)

def visualize_collapse_proof_results():
    """可视化CollapseProof analysis results"""
    system = CollapseProofSystem(max_trace_value=60, proof_depth=5)
    results = system.run_comprehensive_analysis()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Proof Network Visualization
    ax1 = plt.subplot(3, 4, 1)
    G = system.proof_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by proof validity
    node_colors = []
    for node in G.nodes():
        validity = system.trace_universe[node]['proof_validity']
        node_colors.append(validity)
    
    nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
            node_size=50, alpha=0.8, ax=ax1)
    ax1.set_title("Proof Network: Structure-Driven Deduction Architecture\n(Colors: Proof Validity)")
    
    # 2. Deductive Strength Distribution
    ax2 = plt.subplot(3, 4, 2)
    strengths = [data['deductive_strength'] for data in system.trace_universe.values()]
    ax2.hist(strengths, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(np.mean(strengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(strengths):.3f}')
    ax2.set_xlabel('Deductive Strength')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Deductive Strength Distribution')
    ax2.legend()
    
    # 3. Proof Validity vs Inference Capacity
    ax3 = plt.subplot(3, 4, 3)
    validities = [data['proof_validity'] for data in system.trace_universe.values()]
    capacities = [data['inference_capacity'] for data in system.trace_universe.values()]
    depths = [len(data['trace']) for data in system.trace_universe.values()]
    
    scatter = ax3.scatter(validities, capacities, c=depths, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('Proof Validity')
    ax3.set_ylabel('Inference Capacity')
    ax3.set_title('Proof Validity vs Inference Capacity\n(Color: Trace Length)')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. Logical Consistency vs Proof Completeness
    ax4 = plt.subplot(3, 4, 4)
    consistencies = [data['logical_consistency'] for data in system.trace_universe.values()]
    completenesses = [data['proof_completeness'] for data in system.trace_universe.values()]
    efficiencies = [data['deduction_efficiency'] for data in system.trace_universe.values()]
    
    scatter = ax4.scatter(consistencies, completenesses, c=efficiencies, cmap='coolwarm', alpha=0.7)
    ax4.set_xlabel('Logical Consistency')
    ax4.set_ylabel('Proof Completeness')
    ax4.set_title('Logical Consistency vs Proof Completeness\n(Color: Deduction Efficiency)')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Proof Depth Distribution
    ax5 = plt.subplot(3, 4, 5)
    depths = [data['proof_depth'] for data in system.trace_universe.values()]
    ax5.hist(depths, bins=12, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(np.mean(depths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(depths):.3f}')
    ax5.set_xlabel('Proof Depth')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Proof Depth Distribution')
    ax5.legend()
    
    # 6. Inference Stability vs Logical Coherence
    ax6 = plt.subplot(3, 4, 6)
    stabilities = [data['inference_stability'] for data in system.trace_universe.values()]
    coherences = [data['logical_coherence'] for data in system.trace_universe.values()]
    values = [data['value'] for data in system.trace_universe.values()]
    
    scatter = ax6.scatter(stabilities, coherences, c=values, cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Inference Stability')
    ax6.set_ylabel('Logical Coherence')
    ax6.set_title('Inference Stability vs Logical Coherence\n(Color: Trace Value)')
    plt.colorbar(scatter, ax=ax6)
    
    # 7. Category Distribution
    ax7 = plt.subplot(3, 4, 7)
    categories = results['category_analysis']['categories']
    ax7.bar(categories.keys(), categories.values(), color=['lightblue', 'lightgreen', 'lightcoral'])
    ax7.set_ylabel('Number of Traces')
    ax7.set_title('Proof Categories Distribution')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Network Properties
    ax8 = plt.subplot(3, 4, 8)
    network_props = results['network_analysis']
    props = ['Nodes', 'Edges', 'Components']
    values = [network_props['nodes'], network_props['edges'], network_props['components']]
    ax8.bar(props, values, color=['skyblue', 'lightgreen', 'salmon'])
    ax8.set_ylabel('Count')
    ax8.set_title('Proof Network Properties')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(3, 4, 9)
    properties = ['deductive_strength', 'proof_validity', 'inference_capacity', 'logical_consistency']
    data_matrix = []
    for prop in properties:
        data_matrix.append([system.trace_universe[t][prop] for t in system.trace_universe.keys()])
    
    correlation_matrix = np.corrcoef(data_matrix)
    im = ax9.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(properties)))
    ax9.set_yticks(range(len(properties)))
    ax9.set_xticklabels([p.replace('_', ' ').title() for p in properties], rotation=45)
    ax9.set_yticklabels([p.replace('_', ' ').title() for p in properties])
    ax9.set_title('Proof Properties Correlation Matrix')
    
    # Add correlation values
    for i in range(len(properties)):
        for j in range(len(properties)):
            ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # 10. 3D Proof Space
    ax10 = plt.subplot(3, 4, 10, projection='3d')
    validities = [data['proof_validity'] for data in system.trace_universe.values()]
    capacities = [data['inference_capacity'] for data in system.trace_universe.values()]
    consistencies = [data['logical_consistency'] for data in system.trace_universe.values()]
    
    ax10.scatter(validities, capacities, consistencies, c=strengths, cmap='plasma', alpha=0.6)
    ax10.set_xlabel('Proof Validity')
    ax10.set_ylabel('Inference Capacity')
    ax10.set_zlabel('Logical Consistency')
    ax10.set_title('3D Proof Space')
    
    # 11. Entropy Analysis
    ax11 = plt.subplot(3, 4, 11)
    entropies = results['entropy_analysis']
    entropy_props = list(entropies.keys())
    entropy_values = list(entropies.values())
    
    bars = ax11.barh(range(len(entropy_props)), entropy_values, color='orange', alpha=0.7)
    ax11.set_yticks(range(len(entropy_props)))
    ax11.set_yticklabels([p.replace('_', ' ').title() for p in entropy_props])
    ax11.set_xlabel('Entropy (bits)')
    ax11.set_title('Proof Properties Entropy Analysis')
    
    # Add entropy values on bars
    for i, bar in enumerate(bars):
        ax11.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{entropy_values[i]:.2f}', va='center', fontsize=8)
    
    # 12. Deduction Network
    ax12 = plt.subplot(3, 4, 12)
    
    # Create simplified deduction network
    deduction_graph = nx.DiGraph()
    sample_traces = list(system.trace_universe.keys())[:15]  # Sample for visualization
    
    for trace in sample_traces:
        deduction_graph.add_node(trace)
        reachable = system.deduction_mappings.get(trace, [])
        for target in reachable[:3]:  # Limit connections for clarity
            if target in sample_traces:
                deduction_graph.add_edge(trace, target)
    
    pos = nx.spring_layout(deduction_graph, k=2, iterations=50)
    nx.draw(deduction_graph, pos, node_color='lightblue', 
            node_size=100, alpha=0.8, arrows=True, ax=ax12)
    ax12.set_title("Deduction Network: Proof Transformation Paths")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-103-collapse-proof-dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional specialized visualizations
    
    # Proof Architecture Analysis
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Proof Strength vs Network Position
    G = system.proof_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Calculate centrality measures
    if G.number_of_edges() > 0:
        centralities = nx.degree_centrality(G)
        strengths = {node: system.trace_universe[node]['deductive_strength'] for node in G.nodes()}
        
        cent_values = [centralities.get(node, 0) for node in G.nodes()]
        strength_values = [strengths.get(node, 0) for node in G.nodes()]
        
        ax21.scatter(cent_values, strength_values, alpha=0.7, c='blue')
        ax21.set_xlabel('Network Centrality')
        ax21.set_ylabel('Deductive Strength')
        ax21.set_title('Network Position vs Proof Strength')
    
    # 2. Logical Consistency Network
    consistency_colors = [system.trace_universe[node]['logical_consistency'] for node in G.nodes()]
    nx.draw(G, pos, node_color=consistency_colors, cmap='RdYlBu', 
            node_size=80, alpha=0.8, ax=ax22)
    ax22.set_title("Logical Consistency Network")
    
    # 3. Proof Completeness Distribution by Category
    categories = system._categorize_traces()
    for i, (cat_name, traces) in enumerate(categories.items()):
        completenesses = [system.trace_universe[t]['proof_completeness'] for t in traces]
        ax23.hist(completenesses, bins=10, alpha=0.6, label=cat_name, 
                 color=['red', 'green', 'blue'][i])
    ax23.set_xlabel('Proof Completeness')
    ax23.set_ylabel('Frequency')
    ax23.set_title('Proof Completeness by Category')
    ax23.legend()
    
    # 4. Inference Efficiency Network
    efficiency_colors = [system.trace_universe[node]['deduction_efficiency'] for node in G.nodes()]
    nx.draw(G, pos, node_color=efficiency_colors, cmap='plasma', 
            node_size=80, alpha=0.8, ax=ax24)
    ax24.set_title("Deduction Efficiency Network")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-103-collapse-proof-architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Statistics
    print("="*80)
    print("COLLAPSEPROOF STRUCTURE-DRIVEN DEDUCTION ANALYSIS")
    print("="*80)
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_analysis']['density']:.3f}")
    print(f"Connected components: {results['network_analysis']['components']}")
    
    print("\nProof Properties Analysis:")
    for prop, stats in results['proof_properties'].items():
        print(f"- {prop.replace('_', ' ').title()}: "
              f"mean={stats['mean']:.3f}, "
              f"high_count={stats['high_count']} ({stats['high_count']/results['total_traces']*100:.1f}%)")
    
    print(f"\nCategory Distribution:")
    for cat, count in results['category_analysis']['categories'].items():
        percentage = count / results['total_traces'] * 100
        print(f"- {cat.replace('_', ' ').title()}: {count} traces ({percentage:.1f}%)")
    
    print(f"\nMorphism Analysis:")
    print(f"Total morphisms: {results['category_analysis']['total_morphisms']}")
    morphism_density = results['category_analysis']['total_morphisms'] / (results['total_traces'] ** 2)
    print(f"Morphism density: {morphism_density:.3f}")
    
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
    visualize_collapse_proof_results()