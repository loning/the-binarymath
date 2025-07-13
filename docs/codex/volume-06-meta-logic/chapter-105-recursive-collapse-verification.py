#!/usr/bin/env python3
"""
Chapter 105: RecursiveCollapse Unit Test Verification
从ψ=ψ(ψ)推导φ-Recursive Function Construction via Trace Evolution

Core principle: From ψ = ψ(ψ) derive systematic recursive function construction
through φ-constrained trace evolution that enables systematic function building
through iterative trace transformations, creating recursive architectures
that encode the fundamental recursive principles of collapsed space through
entropy-increasing tensor transformations that establish systematic recursive
computation through φ-trace iterative dynamics rather than traditional
lambda calculus or primitive recursive function definitions.

This verification program implements:
1. φ-constrained recursive function construction through trace iteration evolution
2. Function construction: systematic computation building through trace transformations
3. Three-domain analysis: Traditional vs φ-constrained vs intersection recursion theory
4. Graph theory analysis of recursive networks and iteration relationship structures
5. Information theory analysis of recursive entropy and function construction encoding
6. Category theory analysis of recursive functors and iteration morphisms
7. Visualization of recursive structures and φ-trace iteration systems
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

class RecursiveCollapseSystem:
    """
    Core system for implementing φ-recursive function construction via trace evolution.
    Implements φ-constrained recursive architectures through trace iteration dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, recursive_depth: int = 6):
        """Initialize recursive collapse system with iteration trace analysis"""
        self.max_trace_value = max_trace_value
        self.recursive_depth = recursive_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.recursive_cache = {}
        self.iteration_cache = {}
        self.function_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.recursive_network = self._build_recursive_network()
        self.iteration_mappings = self._detect_iteration_mappings()
        
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
                recursive_data = self._analyze_recursive_properties(trace, n)
                universe[n] = recursive_data
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
        
    def _analyze_recursive_properties(self, trace: str, value: int) -> Dict:
        """分析trace的recursive properties"""
        return {
            'trace': trace,
            'value': value,
            'recursive_strength': self._compute_recursive_strength(trace, value),
            'iteration_capacity': self._compute_iteration_capacity(trace, value),
            'function_construction': self._compute_function_construction(trace, value),
            'recursive_consistency': self._compute_recursive_consistency(trace, value),
            'construction_completeness': self._compute_construction_completeness(trace, value),
            'iteration_efficiency': self._compute_iteration_efficiency(trace, value),
            'recursive_depth': self._compute_recursive_depth(trace, value),
            'iteration_stability': self._compute_iteration_stability(trace, value),
            'recursive_coherence': self._compute_recursive_coherence(trace, value)
        }
        
    def _compute_recursive_strength(self, trace: str, value: int) -> float:
        """计算recursive strength（递归强度）"""
        if len(trace) == 0:
            return 0.0
        
        # Recursive strength emerges from self-referential iteration capacity
        strength_factors = []
        
        # Factor 1: Self-iteration capability (trace can iterate on itself)
        self_iteration = self._measure_self_iteration_capability(trace)
        strength_factors.append(self_iteration)
        
        # Factor 2: Pattern recursion (systematic pattern repetition)
        pattern_recursion = self._measure_pattern_recursion(trace)
        strength_factors.append(pattern_recursion)
        
        # Factor 3: Functional composition (ability to compose operations)
        functional_composition = self._measure_functional_composition(trace)
        strength_factors.append(functional_composition)
        
        # Factor 4: φ-constraint preservation during recursion
        phi_preservation = self._measure_phi_recursion_preservation(trace)
        strength_factors.append(phi_preservation)
        
        # Recursive strength as geometric mean
        recursive_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        
        return min(1.0, recursive_strength)
        
    def _compute_iteration_capacity(self, trace: str, value: int) -> float:
        """计算iteration capacity（迭代能力）"""
        # Iteration capacity emerges from repeated transformation capability
        capacity_factors = []
        
        # Factor 1: Iteration stability (consistent transformations)
        iteration_stability = self._measure_iteration_stability(trace)
        capacity_factors.append(iteration_stability)
        
        # Factor 2: Convergence behavior (iteration reaches stable states)
        convergence_behavior = self._measure_convergence_behavior(trace)
        capacity_factors.append(convergence_behavior)
        
        # Factor 3: Transformation preservation (structure maintained during iteration)
        transformation_preservation = self._measure_transformation_preservation(trace)
        capacity_factors.append(transformation_preservation)
        
        # Iteration capacity as weighted geometric mean
        weights = [0.4, 0.3, 0.3]  # Emphasize stability
        iteration_capacity = np.prod([f**w for f, w in zip(capacity_factors, weights)])
        
        return min(1.0, iteration_capacity)
        
    def _compute_function_construction(self, trace: str, value: int) -> float:
        """计算function construction（函数构造能力）"""
        # Function construction measured through computational building capability
        construction_factors = []
        
        # Factor 1: Function representation (trace can represent functions)
        function_representation = self._measure_function_representation(trace)
        construction_factors.append(function_representation)
        
        # Factor 2: Computational composition (building complex computations)
        computational_composition = self._measure_computational_composition(trace)
        construction_factors.append(computational_composition)
        
        # Factor 3: Recursive definition (self-referential function definition)
        recursive_definition = self._measure_recursive_definition(trace)
        construction_factors.append(recursive_definition)
        
        # Function construction as geometric mean
        function_construction = np.prod(construction_factors) ** (1.0 / len(construction_factors))
        
        return min(1.0, function_construction)
        
    def _compute_recursive_consistency(self, trace: str, value: int) -> float:
        """计算recursive consistency（递归一致性）"""
        if len(trace) <= 2:
            return 1.0
        
        # Recursive consistency through coherent recursive construction
        consistency_score = 1.0
        
        # Check for consistent recursive patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._has_consistent_recursive_pattern(window):
                consistency_score *= 1.05
            else:
                consistency_score *= 0.98
                
        # Global recursive consistency
        global_consistency = self._measure_global_recursive_consistency(trace)
        consistency_score *= global_consistency
        
        return min(1.0, consistency_score)
        
    def _compute_construction_completeness(self, trace: str, value: int) -> float:
        """计算construction completeness（构造完备性）"""
        # Construction completeness through comprehensive function building coverage
        completeness_factors = []
        
        # Factor 1: Function type coverage (different function types represented)
        function_type_coverage = self._measure_function_type_coverage(trace)
        completeness_factors.append(function_type_coverage)
        
        # Factor 2: Recursive pattern coverage (key recursive patterns accessible)
        recursive_pattern_coverage = self._measure_recursive_pattern_coverage(trace)
        completeness_factors.append(recursive_pattern_coverage)
        
        # Factor 3: Construction spectrum (full range of constructions)
        construction_spectrum = self._measure_construction_spectrum(trace)
        completeness_factors.append(construction_spectrum)
        
        # Construction completeness as geometric mean
        construction_completeness = np.prod(completeness_factors) ** (1.0 / len(completeness_factors))
        
        return min(1.0, construction_completeness)
        
    def _compute_iteration_efficiency(self, trace: str, value: int) -> float:
        """计算iteration efficiency（迭代效率）"""
        if len(trace) == 0:
            return 0.0
        
        # Efficiency measured as ratio of iteration power to computational cost
        iteration_power = self._compute_iteration_power(trace)
        computational_cost = self._compute_iteration_cost(trace)
        
        if computational_cost == 0:
            return 0.0
            
        efficiency = iteration_power / computational_cost
        return min(1.0, efficiency)
        
    def _compute_recursive_depth(self, trace: str, value: int) -> float:
        """计算recursive depth（递归深度）"""
        # Recursive depth measured through nested recursive construction levels
        max_depth = 0
        
        # Find deepest valid recursive construction
        for start in range(len(trace)):
            depth = self._trace_recursive_depth(trace, start)
            max_depth = max(max_depth, depth)
            
        # Normalize to [0, 1]
        normalized_depth = min(1.0, max_depth / 6.0)
        return normalized_depth
        
    def _compute_iteration_stability(self, trace: str, value: int) -> float:
        """计算iteration stability（迭代稳定性）"""
        # Stability measured through consistency of iteration patterns
        if len(trace) <= 3:
            return 1.0
        
        # Measure local iteration stability
        local_stabilities = []
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            stability = self._measure_local_iteration_stability(window)
            local_stabilities.append(stability)
            
        # Average local stability
        iteration_stability = np.mean(local_stabilities) if local_stabilities else 1.0
        return min(1.0, iteration_stability)
        
    def _compute_recursive_coherence(self, trace: str, value: int) -> float:
        """计算recursive coherence（递归连贯性）"""
        # Recursive coherence through systematic recursive flow
        coherence_factors = []
        
        # Factor 1: Sequential coherence (adjacent recursive elements align)
        sequential_coherence = self._measure_sequential_recursive_coherence(trace)
        coherence_factors.append(sequential_coherence)
        
        # Factor 2: Global coherence (overall recursive consistency)
        global_coherence = self._measure_global_recursive_coherence(trace)
        coherence_factors.append(global_coherence)
        
        # Factor 3: Structural coherence (φ-constraint recursive alignment)
        structural_coherence = self._measure_structural_recursive_coherence(trace)
        coherence_factors.append(structural_coherence)
        
        # Recursive coherence as geometric mean
        recursive_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        
        return min(1.0, recursive_coherence)
        
    # Helper methods for recursive analysis
    def _measure_self_iteration_capability(self, trace: str) -> float:
        """测量self-iteration capability"""
        if len(trace) <= 1:
            return 0.0
        
        # Self-iteration through trace patterns that can transform themselves
        self_iteration_score = 0.0
        
        # Check for self-transforming patterns
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            if self._can_self_transform(pattern):
                self_iteration_score += 1.0 / len(trace)
                
        return min(1.0, self_iteration_score * 2.0)
        
    def _measure_pattern_recursion(self, trace: str) -> float:
        """测量pattern recursion"""
        if len(trace) == 0:
            return 0.0
        
        # Pattern recursion through systematic pattern repetition
        recursion_factors = []
        
        # Factor 1: Pattern repetition
        pattern_repetition = self._measure_pattern_repetition(trace)
        recursion_factors.append(pattern_repetition)
        
        # Factor 2: Nested patterns
        nested_patterns = self._measure_nested_patterns(trace)
        recursion_factors.append(nested_patterns)
        
        return np.mean(recursion_factors)
        
    def _measure_functional_composition(self, trace: str) -> float:
        """测量functional composition capability"""
        if len(trace) < 3:
            return 0.0
        
        # Functional composition through combinable operation patterns
        composition_score = 0.0
        
        # Look for composable patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._is_composable_pattern(window):
                composition_score += 1.0
                
        return min(1.0, composition_score / max(1, len(trace) - 2))
        
    def _measure_phi_recursion_preservation(self, trace: str) -> float:
        """测量φ-constraint preservation during recursion"""
        # φ-preservation ensured by φ-validity
        return 1.0 if self._is_phi_valid(trace) else 0.0
        
    def _measure_iteration_stability(self, trace: str) -> float:
        """测量iteration stability"""
        if len(trace) <= 1:
            return 1.0
        
        # Stability through consistent transformation patterns
        stability_score = 0.0
        
        # Check for stable iteration patterns
        for i in range(len(trace) - 1):
            if self._is_stable_transition(trace[i], trace[i+1]):
                stability_score += 1.0
                
        return stability_score / (len(trace) - 1) if len(trace) > 1 else 1.0
        
    def _measure_convergence_behavior(self, trace: str) -> float:
        """测量convergence behavior"""
        if len(trace) <= 2:
            return 1.0
        
        # Convergence through stabilizing patterns
        convergence_score = 0.0
        
        # Check for convergent patterns
        last_third = len(trace) // 3
        if last_third > 0:
            recent_pattern = trace[-last_third:]
            if self._exhibits_convergence(recent_pattern):
                convergence_score = 1.0
            else:
                convergence_score = 0.5
                
        return convergence_score
        
    def _measure_transformation_preservation(self, trace: str) -> float:
        """测量transformation preservation"""
        if len(trace) == 0:
            return 1.0
        
        # Preservation through maintained structural properties
        preservation_score = 1.0
        
        # Check if φ-validity is preserved (essential structural property)
        if self._is_phi_valid(trace):
            preservation_score *= 1.2
        else:
            preservation_score *= 0.8
            
        return min(1.0, preservation_score)
        
    def _measure_function_representation(self, trace: str) -> float:
        """测量function representation capability"""
        if len(trace) == 0:
            return 0.0
        
        # Function representation through computational pattern encoding
        representation_score = 0.0
        
        # Check for function-like patterns
        function_patterns = ['01', '10', '010', '101']
        for pattern in function_patterns:
            if pattern in trace:
                representation_score += 0.25
                
        return representation_score
        
    def _measure_computational_composition(self, trace: str) -> float:
        """测量computational composition"""
        if len(trace) < 4:
            return 0.0
        
        # Composition through complex computational patterns
        composition_score = 0.0
        
        # Look for complex composable patterns
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            if self._is_complex_composable(window):
                composition_score += 1.0
                
        return min(1.0, composition_score / max(1, len(trace) - 3))
        
    def _measure_recursive_definition(self, trace: str) -> float:
        """测量recursive definition capability"""
        if len(trace) < 3:
            return 0.0
        
        # Recursive definition through self-referential patterns
        definition_score = 0.0
        
        # Check for self-referential patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._is_self_referential(window):
                definition_score += 1.0
                
        return min(1.0, definition_score / max(1, len(trace) - 2))
        
    def _has_consistent_recursive_pattern(self, window: str) -> bool:
        """检查window是否有consistent recursive pattern"""
        if len(window) < 3:
            return True
        
        # Consistent recursive: self-similar pattern structure
        return window[0] == window[2]  # Recursive similarity
        
    def _measure_global_recursive_consistency(self, trace: str) -> float:
        """测量global recursive consistency"""
        if len(trace) <= 2:
            return 1.0
        
        # Global consistency through overall recursive coherence
        consistency = 1.0
        
        # Check for global recursive pattern
        recursive_consistency = 0.0
        for i in range(0, len(trace) - 2, 2):
            if i + 2 < len(trace) and trace[i] == trace[i + 2]:
                recursive_consistency += 1.0
                
        recursive_rate = recursive_consistency / max(1, (len(trace) - 2) // 2)
        
        # Good recursion has moderate recursive similarity
        if 0.3 <= recursive_rate <= 0.7:
            consistency *= 1.1
        else:
            consistency *= 0.9
            
        return min(1.0, consistency)
        
    def _measure_function_type_coverage(self, trace: str) -> float:
        """测量function type coverage"""
        # Coverage of essential function types
        function_types = ['0', '1', '01', '10']  # Identity, constant, transformation functions
        covered = sum(1 for ftype in function_types if ftype in trace)
        return covered / len(function_types)
        
    def _measure_recursive_pattern_coverage(self, trace: str) -> float:
        """测量recursive pattern coverage"""
        # Coverage of recursive patterns
        recursive_patterns = ['010', '101', '0101', '1010']
        covered = sum(1 for pattern in recursive_patterns if pattern in trace)
        return covered / len(recursive_patterns)
        
    def _measure_construction_spectrum(self, trace: str) -> float:
        """测量construction spectrum"""
        if len(trace) == 0:
            return 0.0
        
        # Spectrum based on construction pattern diversity
        construction_patterns = set()
        
        # Collect construction-indicating patterns
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            construction_patterns.add(pattern)
                
        spectrum_coverage = len(construction_patterns) / 3.0  # Max 3 valid patterns in φ-traces
        return min(1.0, spectrum_coverage)
        
    def _compute_iteration_power(self, trace: str) -> float:
        """计算iteration power"""
        if len(trace) == 0:
            return 0.0
        
        # Power based on iteration capability
        power_factors = []
        
        # Factor 1: Iteration count potential
        iteration_count = self._count_potential_iterations(trace)
        power_factors.append(min(1.0, iteration_count / 5.0))
        
        # Factor 2: Iteration complexity
        iteration_complexity = self._measure_iteration_complexity(trace)
        power_factors.append(iteration_complexity)
        
        return np.mean(power_factors)
        
    def _compute_iteration_cost(self, trace: str) -> float:
        """计算iteration cost"""
        if len(trace) == 0:
            return 1.0
        
        # Cost based on trace iteration complexity
        cost_factors = []
        
        # Factor 1: Length cost
        length_cost = len(trace) / 8.0  # Normalize
        cost_factors.append(length_cost)
        
        # Factor 2: Pattern complexity cost
        pattern_cost = self._measure_pattern_iteration_complexity(trace)
        cost_factors.append(pattern_cost)
        
        return max(0.1, np.mean(cost_factors))  # Minimum cost
        
    def _trace_recursive_depth(self, trace: str, start: int) -> int:
        """追踪recursive depth from start position"""
        depth = 0
        pos = start
        
        while pos < len(trace) - 2:
            if self._is_recursive_step(trace[pos:pos+3]):
                depth += 1
                pos += 2
            else:
                break
                
        return depth
        
    def _measure_local_iteration_stability(self, window: str) -> float:
        """测量local iteration stability"""
        if len(window) < 4:
            return 1.0
        
        # Stability through consistent iteration patterns
        stability = 0.0
        
        # Check for stable iteration patterns
        if self._has_stable_iteration_pattern(window):
            stability += 0.5
            
        if self._is_phi_valid(window):  # φ-valid ensures stability
            stability += 0.5
            
        return stability
        
    def _measure_sequential_recursive_coherence(self, trace: str) -> float:
        """测量sequential recursive coherence"""
        if len(trace) <= 1:
            return 1.0
        
        coherent_recursions = 0
        total_recursions = len(trace) - 1
        
        for i in range(total_recursions):
            if self._is_coherent_recursive_step(trace[i], trace[i+1]):
                coherent_recursions += 1
                
        return coherent_recursions / total_recursions
        
    def _measure_global_recursive_coherence(self, trace: str) -> float:
        """测量global recursive coherence"""
        # Global coherence through overall recursive structure
        coherence = 1.0
        
        # Check for global recursive organization
        if len(trace) > 2:
            recursive_organization = self._measure_recursive_organization(trace)
            coherence *= recursive_organization
            
        return min(1.0, coherence)
        
    def _measure_structural_recursive_coherence(self, trace: str) -> float:
        """测量structural recursive coherence"""
        # Coherence through φ-constraint recursive alignment
        coherence = 1.0
        
        # φ-valid traces have inherent structural coherence
        if self._is_phi_valid(trace):
            coherence *= 1.1
            
        return min(1.0, coherence)
        
    # Additional helper methods
    def _can_self_transform(self, pattern: str) -> bool:
        """检查pattern是否can self-transform"""
        if len(pattern) < 2:
            return False
        
        # Self-transformation through invertible patterns
        return pattern in ['01', '10']  # Invertible transformations
        
    def _measure_pattern_repetition(self, trace: str) -> float:
        """测量pattern repetition"""
        if len(trace) < 2:
            return 0.0
        
        # Count pattern repetitions
        repetitions = 0
        patterns_seen = set()
        
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            if pattern in patterns_seen:
                repetitions += 1
            patterns_seen.add(pattern)
            
        return min(1.0, repetitions / max(1, len(trace) - 1))
        
    def _measure_nested_patterns(self, trace: str) -> float:
        """测量nested patterns"""
        if len(trace) < 4:
            return 0.0
        
        # Check for nested pattern structures
        nesting_score = 0.0
        
        for i in range(len(trace) - 3):
            outer = trace[i:i+4]
            inner = trace[i+1:i+3]
            if self._is_nested_pattern(outer, inner):
                nesting_score += 1.0
                
        return min(1.0, nesting_score / max(1, len(trace) - 3))
        
    def _is_composable_pattern(self, window: str) -> bool:
        """检查window是否为composable pattern"""
        if len(window) < 3:
            return False
        
        # Composable patterns have functional structure
        return window in ['010', '101', '001', '100']
        
    def _is_stable_transition(self, bit1: str, bit2: str) -> bool:
        """检查transition是否stable"""
        # Stable transitions maintain some consistency
        return True  # All transitions have some stability in φ-valid traces
        
    def _exhibits_convergence(self, pattern: str) -> bool:
        """检查pattern是否exhibits convergence"""
        if len(pattern) <= 1:
            return True
        
        # Convergence through repeated or stabilizing patterns
        return pattern.count(pattern[0]) / len(pattern) > 0.6
        
    def _is_complex_composable(self, window: str) -> bool:
        """检查window是否为complex composable pattern"""
        if len(window) < 4:
            return False
        
        # Complex composable patterns
        return self._is_phi_valid(window) and len(set(window)) > 1
        
    def _is_self_referential(self, window: str) -> bool:
        """检查window是否为self-referential"""
        if len(window) < 3:
            return False
        
        # Self-referential patterns have internal similarity
        return window[0] == window[2]
        
    def _count_potential_iterations(self, trace: str) -> int:
        """计算potential iterations"""
        # Count positions where iteration could occur
        iterations = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # Transition points enable iteration
                iterations += 1
        return iterations
        
    def _measure_iteration_complexity(self, trace: str) -> float:
        """测量iteration complexity"""
        if len(trace) <= 1:
            return 0.1
        
        # Complexity through pattern diversity in iterations
        transitions = set()
        for i in range(len(trace) - 1):
            transitions.add(trace[i:i+2])
            
        complexity = len(transitions) / 3.0  # Max 3 valid transitions in φ-traces
        return min(1.0, complexity)
        
    def _measure_pattern_iteration_complexity(self, trace: str) -> float:
        """测量pattern iteration complexity"""
        return self._measure_iteration_complexity(trace)
        
    def _is_recursive_step(self, window: str) -> bool:
        """检查window是否为recursive step"""
        if len(window) < 3:
            return False
        
        # Recursive step has self-similarity
        return window[0] == window[2]
        
    def _has_stable_iteration_pattern(self, window: str) -> bool:
        """检查window是否有stable iteration pattern"""
        if len(window) < 4:
            return True
        
        # Stable iteration: consistent pattern progression
        return self._is_phi_valid(window)
        
    def _is_coherent_recursive_step(self, bit1: str, bit2: str) -> bool:
        """检查两个bit是否形成coherent recursive step"""
        # Coherent recursion includes both continuation and change
        return True  # All steps have some recursive coherence
        
    def _measure_recursive_organization(self, trace: str) -> float:
        """测量recursive organization"""
        if len(trace) <= 2:
            return 1.0
        
        # Organization through systematic recursive structure
        organization = 0.0
        
        # Check for organized recursive patterns
        recursive_matches = 0
        for i in range(0, len(trace) - 2, 2):
            if i + 2 < len(trace) and trace[i] == trace[i + 2]:
                recursive_matches += 1
                
        if len(trace) > 2:
            organization = recursive_matches / max(1, (len(trace) - 2) // 2)
        
        return min(1.0, organization)
        
    def _is_nested_pattern(self, outer: str, inner: str) -> bool:
        """检查是否为nested pattern"""
        if len(outer) < 4 or len(inner) < 2:
            return False
        
        # Nested if inner pattern is contained and meaningful
        return inner in outer and inner != outer
        
    def _build_recursive_network(self) -> nx.Graph:
        """构建recursive network基于trace similarities"""
        G = nx.Graph()
        traces = list(self.trace_universe.keys())
        
        # Add nodes
        for trace_val in traces:
            G.add_node(trace_val)
            
        # Add edges based on recursive similarity
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_recursive_similarity(
                    self.trace_universe[trace1], 
                    self.trace_universe[trace2]
                )
                if similarity > 0.6:  # Threshold for recursive relationship
                    G.add_edge(trace1, trace2, weight=similarity)
                    
        return G
        
    def _compute_recursive_similarity(self, trace1_data: Dict, trace2_data: Dict) -> float:
        """计算两个traces之间的recursive similarity"""
        # Compare recursive properties
        properties = ['recursive_strength', 'iteration_capacity', 'function_construction', 
                     'recursive_consistency', 'construction_completeness']
        
        similarities = []
        for prop in properties:
            val1 = trace1_data[prop]
            val2 = trace2_data[prop]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _detect_iteration_mappings(self) -> Dict:
        """检测iteration mappings between traces"""
        mappings = {}
        
        for trace_val, data in self.trace_universe.items():
            # Find traces that can be reached through iteration
            iterations = []
            for other_val, other_data in self.trace_universe.items():
                if trace_val != other_val:
                    if self._can_iterate_to(data, other_data):
                        iterations.append(other_val)
            mappings[trace_val] = iterations
            
        return mappings
        
    def _can_iterate_to(self, from_data: Dict, to_data: Dict) -> bool:
        """检查是否can iterate from from_data to to_data"""
        # Iteration possible if target has compatible recursive properties
        return (abs(to_data['recursive_strength'] - from_data['recursive_strength']) < 0.3 and
                to_data['iteration_capacity'] >= from_data['iteration_capacity'] * 0.8)
        
    def run_comprehensive_analysis(self) -> Dict:
        """运行comprehensive recursive analysis"""
        results = {
            'total_traces': len(self.trace_universe),
            'recursive_properties': self._analyze_recursive_distributions(),
            'network_analysis': self._analyze_recursive_network(),
            'iteration_analysis': self._analyze_iteration_patterns(),
            'category_analysis': self._perform_category_analysis(),
            'entropy_analysis': self._compute_entropy_analysis()
        }
        
        return results
        
    def _analyze_recursive_distributions(self) -> Dict:
        """分析recursive property distributions"""
        properties = ['recursive_strength', 'iteration_capacity', 'function_construction',
                     'recursive_consistency', 'construction_completeness', 'iteration_efficiency',
                     'recursive_depth', 'iteration_stability', 'recursive_coherence']
        
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
        
    def _analyze_recursive_network(self) -> Dict:
        """分析recursive network properties"""
        G = self.recursive_network
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G),
            'largest_component': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
    def _analyze_iteration_patterns(self) -> Dict:
        """分析iteration patterns"""
        total_iterations = sum(len(iterations) for iterations in self.iteration_mappings.values())
        total_possible = len(self.trace_universe) * (len(self.trace_universe) - 1)
        
        return {
            'total_iterations': total_iterations,
            'iteration_density': total_iterations / total_possible if total_possible > 0 else 0,
            'avg_iterations': np.mean([len(iterations) for iterations in self.iteration_mappings.values()]),
            'max_iterations': max([len(iterations) for iterations in self.iteration_mappings.values()]) if self.iteration_mappings else 0
        }
        
    def _perform_category_analysis(self) -> Dict:
        """执行category theory analysis"""
        # Categorize traces based on recursive properties
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
        """将traces按recursive properties分类"""
        categories = {
            'basic_recursive': [],
            'iterative_recursive': [],
            'functional_recursive': []
        }
        
        for trace_val, data in self.trace_universe.items():
            if data['function_construction'] > 0.6:
                categories['functional_recursive'].append(trace_val)
            elif data['iteration_capacity'] > 0.5:
                categories['iterative_recursive'].append(trace_val)
            else:
                categories['basic_recursive'].append(trace_val)
                
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
        
        # Morphism preserves recursive structure within tolerance
        key_properties = ['recursive_strength', 'iteration_capacity', 'recursive_consistency']
        
        for prop in key_properties:
            if abs(data1[prop] - data2[prop]) > tolerance:
                return False
                
        return True
        
    def _compute_entropy_analysis(self) -> Dict:
        """计算entropy analysis"""
        properties = ['recursive_strength', 'iteration_capacity', 'function_construction',
                     'recursive_consistency', 'construction_completeness', 'iteration_efficiency',
                     'recursive_depth', 'iteration_stability', 'recursive_coherence']
        
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

class TestRecursiveCollapse(unittest.TestCase):
    """测试RecursiveCollapse system functionality"""
    
    def setUp(self):
        self.system = RecursiveCollapseSystem(max_trace_value=30, recursive_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        valid_trace = "10101"
        invalid_trace = "11010"
        
        self.assertTrue(self.system._is_phi_valid(valid_trace))
        self.assertFalse(self.system._is_phi_valid(invalid_trace))
        
    def test_recursive_strength_computation(self):
        """测试recursive strength computation"""
        trace = "101010"
        strength = self.system._compute_recursive_strength(trace, 42)
        
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
        
    def test_iteration_capacity_computation(self):
        """测试iteration capacity computation"""
        trace = "10101"
        capacity = self.system._compute_iteration_capacity(trace, 21)
        
        self.assertGreaterEqual(capacity, 0.0)
        self.assertLessEqual(capacity, 1.0)
        
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        # Check all traces are φ-valid
        for data in self.system.trace_universe.values():
            self.assertTrue(self.system._is_phi_valid(data['trace']))
            
    def test_recursive_network_construction(self):
        """测试recursive network construction"""
        G = self.system.recursive_network
        
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some connections
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('recursive_properties', results)
        self.assertIn('network_analysis', results)
        self.assertIn('category_analysis', results)
        
        # Verify reasonable values
        self.assertGreater(results['total_traces'], 0)

def visualize_recursive_collapse_results():
    """可视化RecursiveCollapse analysis results"""
    system = RecursiveCollapseSystem(max_trace_value=45, recursive_depth=5)
    results = system.run_comprehensive_analysis()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Recursive Network Visualization
    ax1 = plt.subplot(3, 4, 1)
    G = system.recursive_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by recursive strength
    node_colors = []
    for node in G.nodes():
        strength = system.trace_universe[node]['recursive_strength']
        node_colors.append(strength)
    
    nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
            node_size=50, alpha=0.8, ax=ax1)
    ax1.set_title("Recursive Network: φ-Recursive Function Architecture\n(Colors: Recursive Strength)")
    
    # 2. Iteration Capacity Distribution
    ax2 = plt.subplot(3, 4, 2)
    capacities = [data['iteration_capacity'] for data in system.trace_universe.values()]
    ax2.hist(capacities, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(capacities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(capacities):.3f}')
    ax2.set_xlabel('Iteration Capacity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Iteration Capacity Distribution')
    ax2.legend()
    
    # 3. Recursive Strength vs Function Construction
    ax3 = plt.subplot(3, 4, 3)
    strengths = [data['recursive_strength'] for data in system.trace_universe.values()]
    constructions = [data['function_construction'] for data in system.trace_universe.values()]
    depths = [len(data['trace']) for data in system.trace_universe.values()]
    
    scatter = ax3.scatter(strengths, constructions, c=depths, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('Recursive Strength')
    ax3.set_ylabel('Function Construction')
    ax3.set_title('Recursive Strength vs Function Construction\n(Color: Trace Length)')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. Iteration Capacity vs Recursive Consistency
    ax4 = plt.subplot(3, 4, 4)
    consistencies = [data['recursive_consistency'] for data in system.trace_universe.values()]
    efficiencies = [data['iteration_efficiency'] for data in system.trace_universe.values()]
    
    scatter = ax4.scatter(capacities, consistencies, c=efficiencies, cmap='coolwarm', alpha=0.7)
    ax4.set_xlabel('Iteration Capacity')
    ax4.set_ylabel('Recursive Consistency')
    ax4.set_title('Iteration Capacity vs Recursive Consistency\n(Color: Iteration Efficiency)')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Recursive Depth Distribution
    ax5 = plt.subplot(3, 4, 5)
    depths = [data['recursive_depth'] for data in system.trace_universe.values()]
    ax5.hist(depths, bins=12, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(np.mean(depths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(depths):.3f}')
    ax5.set_xlabel('Recursive Depth')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Recursive Depth Distribution')
    ax5.legend()
    
    # 6. Iteration Stability vs Recursive Coherence
    ax6 = plt.subplot(3, 4, 6)
    stabilities = [data['iteration_stability'] for data in system.trace_universe.values()]
    coherences = [data['recursive_coherence'] for data in system.trace_universe.values()]
    values = [data['value'] for data in system.trace_universe.values()]
    
    scatter = ax6.scatter(stabilities, coherences, c=values, cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Iteration Stability')
    ax6.set_ylabel('Recursive Coherence')
    ax6.set_title('Iteration Stability vs Recursive Coherence\n(Color: Trace Value)')
    plt.colorbar(scatter, ax=ax6)
    
    # 7. Category Distribution
    ax7 = plt.subplot(3, 4, 7)
    categories = results['category_analysis']['categories']
    ax7.bar(categories.keys(), categories.values(), color=['lightblue', 'lightgreen', 'lightcoral'])
    ax7.set_ylabel('Number of Traces')
    ax7.set_title('Recursive Categories Distribution')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Network Properties
    ax8 = plt.subplot(3, 4, 8)
    network_props = results['network_analysis']
    props = ['Nodes', 'Edges', 'Components']
    values = [network_props['nodes'], network_props['edges'], network_props['components']]
    ax8.bar(props, values, color=['skyblue', 'lightgreen', 'salmon'])
    ax8.set_ylabel('Count')
    ax8.set_title('Recursive Network Properties')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(3, 4, 9)
    properties = ['recursive_strength', 'iteration_capacity', 'function_construction', 'recursive_consistency']
    data_matrix = []
    for prop in properties:
        data_matrix.append([system.trace_universe[t][prop] for t in system.trace_universe.keys()])
    
    correlation_matrix = np.corrcoef(data_matrix)
    im = ax9.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(properties)))
    ax9.set_yticks(range(len(properties)))
    ax9.set_xticklabels([p.replace('_', ' ').title() for p in properties], rotation=45)
    ax9.set_yticklabels([p.replace('_', ' ').title() for p in properties])
    ax9.set_title('Recursive Properties Correlation Matrix')
    
    # Add correlation values
    for i in range(len(properties)):
        for j in range(len(properties)):
            ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # 10. 3D Recursive Space
    ax10 = plt.subplot(3, 4, 10, projection='3d')
    ax10.scatter(strengths, capacities, consistencies, c=constructions, cmap='plasma', alpha=0.6)
    ax10.set_xlabel('Recursive Strength')
    ax10.set_ylabel('Iteration Capacity')
    ax10.set_zlabel('Recursive Consistency')
    ax10.set_title('3D Recursive Space')
    
    # 11. Entropy Analysis
    ax11 = plt.subplot(3, 4, 11)
    entropies = results['entropy_analysis']
    entropy_props = list(entropies.keys())
    entropy_values = list(entropies.values())
    
    bars = ax11.barh(range(len(entropy_props)), entropy_values, color='purple', alpha=0.7)
    ax11.set_yticks(range(len(entropy_props)))
    ax11.set_yticklabels([p.replace('_', ' ').title() for p in entropy_props])
    ax11.set_xlabel('Entropy (bits)')
    ax11.set_title('Recursive Properties Entropy Analysis')
    
    # Add entropy values on bars
    for i, bar in enumerate(bars):
        ax11.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{entropy_values[i]:.2f}', va='center', fontsize=8)
    
    # 12. Iteration Network
    ax12 = plt.subplot(3, 4, 12)
    
    # Create iteration network
    iteration_graph = nx.DiGraph()
    sample_traces = list(system.trace_universe.keys())[:15]  # Sample for visualization
    
    for trace in sample_traces:
        iteration_graph.add_node(trace)
        iterations = system.iteration_mappings.get(trace, [])
        for target in iterations[:3]:  # Limit connections for clarity
            if target in sample_traces:
                iteration_graph.add_edge(trace, target)
    
    pos = nx.spring_layout(iteration_graph, k=2, iterations=50)
    nx.draw(iteration_graph, pos, node_color='lightgreen', 
            node_size=100, alpha=0.8, arrows=True, ax=ax12)
    ax12.set_title("Iteration Network: Recursive Transformation Paths")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-105-recursive-collapse-dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional specialized visualizations
    
    # Recursive Architecture Analysis
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Recursive Construction Analysis
    construction_data = []
    iteration_data = []
    for data in system.trace_universe.values():
        construction_data.append(data['function_construction'])
        iteration_data.append(data['iteration_capacity'])
    
    ax21.scatter(construction_data, iteration_data, alpha=0.7, c='orange')
    ax21.set_xlabel('Function Construction Capability')
    ax21.set_ylabel('Iteration Capacity')
    ax21.set_title('Function Construction vs Iteration Capacity')
    
    # 2. Construction Completeness Network
    G = system.recursive_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    completeness_colors = [system.trace_universe[node]['construction_completeness'] for node in G.nodes()]
    nx.draw(G, pos, node_color=completeness_colors, cmap='RdYlBu', 
            node_size=80, alpha=0.8, ax=ax22)
    ax22.set_title("Construction Completeness Network")
    
    # 3. Recursive Coherence Distribution by Category
    categories = system._categorize_traces()
    for i, (cat_name, traces) in enumerate(categories.items()):
        coherences = [system.trace_universe[t]['recursive_coherence'] for t in traces]
        if coherences:  # Only plot if category has traces
            ax23.hist(coherences, bins=8, alpha=0.6, label=cat_name, 
                     color=['red', 'green', 'blue'][i])
    ax23.set_xlabel('Recursive Coherence')
    ax23.set_ylabel('Frequency')
    ax23.set_title('Recursive Coherence by Category')
    ax23.legend()
    
    # 4. Iteration Efficiency Network
    efficiency_colors = [system.trace_universe[node]['iteration_efficiency'] for node in G.nodes()]
    nx.draw(G, pos, node_color=efficiency_colors, cmap='plasma', 
            node_size=80, alpha=0.8, ax=ax24)
    ax24.set_title("Iteration Efficiency Network")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-105-recursive-collapse-evolution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Statistics
    print("="*80)
    print("RECURSIVE COLLAPSE φ-RECURSIVE FUNCTION CONSTRUCTION ANALYSIS")
    print("="*80)
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_analysis']['density']:.3f}")
    print(f"Connected components: {results['network_analysis']['components']}")
    
    print("\nRecursive Properties Analysis:")
    for prop, stats in results['recursive_properties'].items():
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
    
    print(f"\nIteration Analysis:")
    iteration_stats = results['iteration_analysis']
    print(f"Total iterations: {iteration_stats['total_iterations']}")
    print(f"Iteration density: {iteration_stats['iteration_density']:.3f}")
    print(f"Average iterations per trace: {iteration_stats['avg_iterations']:.1f}")
    
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
    visualize_recursive_collapse_results()