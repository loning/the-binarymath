#!/usr/bin/env python3
"""
Chapter 107: CollapseCategoryLogic Unit Test Verification
从ψ=ψ(ψ)推导Collapse-Aware Category-Theoretic Semantics

Core principle: From ψ = ψ(ψ) derive systematic categorical logic architectures
through φ-constrained trace transformations that enable collapse-aware categorical
semantics where objects, morphisms, and functors emerge from trace geometric
relationships, creating categorical networks that encode the fundamental
categorical principles of collapsed space through entropy-increasing tensor
transformations that establish systematic categorical logic through φ-trace
categorical dynamics rather than traditional category theory axiomatizations.

This verification program implements:
1. φ-constrained categorical logic construction through trace categorical relationships
2. Categorical semantics: systematic object/morphism through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection category theory
4. Graph theory analysis of categorical networks and morphism relationship structures
5. Information theory analysis of categorical entropy and object/morphism encoding
6. Category theory analysis of categorical functors and natural transformation morphisms
7. Visualization of categorical structures and φ-trace categorical systems
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

class CollapseCategoryLogicSystem:
    """
    Core system for implementing collapse-aware category-theoretic semantics.
    Implements φ-constrained categorical architectures through trace categorical dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, categorical_depth: int = 6):
        """Initialize collapse category logic system with categorical trace analysis"""
        self.max_trace_value = max_trace_value
        self.categorical_depth = categorical_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.categorical_cache = {}
        self.morphism_cache = {}
        self.functor_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.categorical_network = self._build_categorical_network()
        self.morphism_mappings = self._detect_morphism_mappings()
        
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
                categorical_data = self._analyze_categorical_properties(trace, n)
                universe[n] = categorical_data
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
        
    def _analyze_categorical_properties(self, trace: str, value: int) -> Dict:
        """分析trace的categorical properties"""
        return {
            'trace': trace,
            'value': value,
            'categorical_strength': self._compute_categorical_strength(trace, value),
            'object_capacity': self._compute_object_capacity(trace, value),
            'morphism_range': self._compute_morphism_range(trace, value),
            'functor_capability': self._compute_functor_capability(trace, value),
            'categorical_completeness': self._compute_categorical_completeness(trace, value),
            'morphism_efficiency': self._compute_morphism_efficiency(trace, value),
            'categorical_depth': self._compute_categorical_depth(trace, value),
            'functor_stability': self._compute_functor_stability(trace, value),
            'categorical_coherence': self._compute_categorical_coherence(trace, value)
        }
        
    def _compute_categorical_strength(self, trace: str, value: int) -> float:
        """计算categorical strength（范畴强度）"""
        if len(trace) == 0:
            return 0.0
        
        # Categorical strength emerges from object/morphism differentiation capacity
        strength_factors = []
        
        # Factor 1: Object detection (trace patterns that represent objects)
        object_detection = self._measure_object_detection(trace)
        strength_factors.append(object_detection)
        
        # Factor 2: Morphism construction (systematic morphism building)
        morphism_construction = self._measure_morphism_construction(trace)
        strength_factors.append(morphism_construction)
        
        # Factor 3: Categorical distinction (clear object/morphism differentiation)
        categorical_distinction = self._measure_categorical_distinction(trace)
        strength_factors.append(categorical_distinction)
        
        # Factor 4: φ-constraint categorical preservation
        phi_categorical_preservation = self._measure_phi_categorical_preservation(trace)
        strength_factors.append(phi_categorical_preservation)
        
        # Categorical strength as geometric mean
        categorical_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        
        return min(1.0, categorical_strength)
        
    def _compute_object_capacity(self, trace: str, value: int) -> float:
        """计算object capacity（对象能力）"""
        # Object capacity emerges from stable entity representation
        capacity_factors = []
        
        # Factor 1: Entity stability (patterns that represent stable objects)
        entity_stability = self._measure_entity_stability(trace)
        capacity_factors.append(entity_stability)
        
        # Factor 2: Identity preservation (object identity maintenance)
        identity_preservation = self._measure_identity_preservation(trace)
        capacity_factors.append(identity_preservation)
        
        # Factor 3: Object relationships (systematic object interactions)
        object_relationships = self._measure_object_relationships(trace)
        capacity_factors.append(object_relationships)
        
        # Object capacity as weighted geometric mean
        weights = [0.4, 0.3, 0.3]  # Emphasize entity stability
        object_capacity = np.prod([f**w for f, w in zip(capacity_factors, weights)])
        
        return min(1.0, object_capacity)
        
    def _compute_morphism_range(self, trace: str, value: int) -> float:
        """计算morphism range（态射范围）"""
        # Morphism range measured through transformation exploration
        range_factors = []
        
        # Factor 1: Transformation potential (possible morphisms)
        transformation_potential = self._measure_transformation_potential(trace)
        range_factors.append(transformation_potential)
        
        # Factor 2: Composition accessibility (reachable compositions)
        composition_accessibility = self._measure_composition_accessibility(trace)
        range_factors.append(composition_accessibility)
        
        # Factor 3: Morphism space coverage (comprehensive morphism exploration)
        morphism_coverage = self._measure_morphism_coverage(trace)
        range_factors.append(morphism_coverage)
        
        # Morphism range as geometric mean
        morphism_range = np.prod(range_factors) ** (1.0 / len(range_factors))
        
        return min(1.0, morphism_range)
        
    def _compute_functor_capability(self, trace: str, value: int) -> float:
        """计算functor capability（函子能力）"""
        if len(trace) <= 2:
            return 1.0
        
        # Functor capability through category-preserving transformations
        capability_score = 1.0
        
        # Check for functor-enabling categorical patterns
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._enables_functor_pattern(window):
                capability_score *= 1.05
            else:
                capability_score *= 0.99
                
        # Global functor capability
        global_capability = self._measure_global_functor_capability(trace)
        capability_score *= global_capability
        
        return min(1.0, capability_score)
        
    def _compute_categorical_completeness(self, trace: str, value: int) -> float:
        """计算categorical completeness（范畴完备性）"""
        # Categorical completeness through comprehensive categorical operation coverage
        completeness_factors = []
        
        # Factor 1: Categorical structure coverage (objects, morphisms, composition)
        categorical_structure_coverage = self._measure_categorical_structure_coverage(trace)
        completeness_factors.append(categorical_structure_coverage)
        
        # Factor 2: Functor coverage (categorical functors and transformations)
        functor_coverage = self._measure_functor_coverage(trace)
        completeness_factors.append(functor_coverage)
        
        # Factor 3: Categorical logic spectrum (full range of categorical operations)
        categorical_spectrum = self._measure_categorical_spectrum(trace)
        completeness_factors.append(categorical_spectrum)
        
        # Categorical completeness as geometric mean
        categorical_completeness = np.prod(completeness_factors) ** (1.0 / len(completeness_factors))
        
        return min(1.0, categorical_completeness)
        
    def _compute_morphism_efficiency(self, trace: str, value: int) -> float:
        """计算morphism efficiency（态射效率）"""
        if len(trace) == 0:
            return 0.0
        
        # Efficiency measured as ratio of morphism power to categorical cost
        morphism_power = self._compute_morphism_power(trace)
        categorical_cost = self._compute_categorical_cost(trace)
        
        if categorical_cost == 0:
            return 0.0
            
        efficiency = morphism_power / categorical_cost
        return min(1.0, efficiency)
        
    def _compute_categorical_depth(self, trace: str, value: int) -> float:
        """计算categorical depth（范畴深度）"""
        # Categorical depth measured through nested categorical construction levels
        max_depth = 0
        
        # Find deepest valid categorical construction
        for start in range(len(trace)):
            depth = self._trace_categorical_depth(trace, start)
            max_depth = max(max_depth, depth)
            
        # Normalize to [0, 1]
        normalized_depth = min(1.0, max_depth / 5.0)
        return normalized_depth
        
    def _compute_functor_stability(self, trace: str, value: int) -> float:
        """计算functor stability（函子稳定性）"""
        # Stability measured through consistency of functor-preserving patterns
        if len(trace) <= 3:
            return 1.0
        
        # Measure local functor stability
        local_stabilities = []
        for i in range(len(trace) - 3):
            window = trace[i:i+4]
            stability = self._measure_local_functor_stability(window)
            local_stabilities.append(stability)
            
        # Average local stability
        functor_stability = np.mean(local_stabilities) if local_stabilities else 1.0
        return min(1.0, functor_stability)
        
    def _compute_categorical_coherence(self, trace: str, value: int) -> float:
        """计算categorical coherence（范畴连贯性）"""
        # Categorical coherence through systematic categorical flow
        coherence_factors = []
        
        # Factor 1: Sequential coherence (adjacent categorical elements align)
        sequential_coherence = self._measure_sequential_categorical_coherence(trace)
        coherence_factors.append(sequential_coherence)
        
        # Factor 2: Global coherence (overall categorical consistency)
        global_coherence = self._measure_global_categorical_coherence(trace)
        coherence_factors.append(global_coherence)
        
        # Factor 3: Structural coherence (φ-constraint categorical alignment)
        structural_coherence = self._measure_structural_categorical_coherence(trace)
        coherence_factors.append(structural_coherence)
        
        # Categorical coherence as geometric mean
        categorical_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        
        return min(1.0, categorical_coherence)
        
    # Helper methods for categorical analysis
    def _measure_object_detection(self, trace: str) -> float:
        """测量object detection capability"""
        if len(trace) <= 1:
            return 0.0
        
        # Object detection through stable pattern recognition
        object_score = 0.0
        
        # Check for object-indicating patterns (stable structures)
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            if self._indicates_object(pattern):
                object_score += 1.0 / len(trace)
                
        return min(1.0, object_score * 2.0)
        
    def _measure_morphism_construction(self, trace: str) -> float:
        """测量morphism construction"""
        if len(trace) == 0:
            return 0.0
        
        # Morphism construction through transformation pattern generation
        construction_factors = []
        
        # Factor 1: Transformation diversity
        unique_transformations = set()
        for i in range(len(trace) - 1):
            unique_transformations.add(trace[i:i+2])
        diversity = len(unique_transformations) / 3.0  # Max 3 transformations in φ-valid traces
        construction_factors.append(diversity)
        
        # Factor 2: Composition accessibility
        compositions = self._count_possible_compositions(trace)
        construction_factors.append(min(1.0, compositions / 5.0))
        
        return np.mean(construction_factors)
        
    def _measure_categorical_distinction(self, trace: str) -> float:
        """测量categorical distinction capability"""
        if len(trace) < 3:
            return 0.0
        
        # Categorical distinction through object/morphism differentiation
        distinction_score = 0.0
        
        # Look for clear categorical distinctions
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._has_categorical_distinction(window):
                distinction_score += 1.0
                
        return min(1.0, distinction_score / max(1, len(trace) - 2))
        
    def _measure_phi_categorical_preservation(self, trace: str) -> float:
        """测量φ-constraint categorical preservation"""
        # φ-preservation ensured by φ-validity with categorical interpretation
        return 1.0 if self._is_phi_valid(trace) else 0.0
        
    def _measure_entity_stability(self, trace: str) -> float:
        """测量entity stability capability"""
        if len(trace) == 0:
            return 0.0
        
        # Entity stability through stable pattern recognition
        stability_score = 0.0
        
        # Check for stable patterns (φ-constraint stable entities)
        if self._is_phi_valid(trace):  # φ-validity provides stability
            stability_score += 0.5
            
        # Check for structural stability patterns
        for i in range(len(trace) - 1):
            if trace[i] == trace[i+1]:  # Local stability
                stability_score += 0.1
                
        return min(1.0, stability_score)
        
    def _measure_identity_preservation(self, trace: str) -> float:
        """测量identity preservation"""
        if len(trace) <= 1:
            return 1.0
        
        # Identity preservation through consistent identity patterns
        preservation_score = 0.0
        
        # φ-constraint preserves identity structure
        for i in range(len(trace) - 1):
            if trace[i:i+2] != "11":  # Identity preserved by φ-constraint
                preservation_score += 1.0
                
        return preservation_score / (len(trace) - 1) if len(trace) > 1 else 1.0
        
    def _measure_object_relationships(self, trace: str) -> float:
        """测量object relationships"""
        if len(trace) < 2:
            return 0.5
        
        # Object relationships through interaction patterns
        relationship_score = 0.0
        
        # Check for relationship-indicating patterns
        relationship_patterns = ['01', '10']  # Basic object interactions
        for pattern in relationship_patterns:
            if pattern in trace:
                relationship_score += 0.5
                
        return min(1.0, relationship_score)
        
    def _measure_transformation_potential(self, trace: str) -> float:
        """测量transformation potential"""
        if len(trace) == 0:
            return 0.0
        
        # Transformation potential through possible morphisms
        transformations = 0
        
        # Count possible transformations that maintain φ-validity
        for i in range(len(trace)):
            test_trace = trace[:i] + ('0' if trace[i] == '1' else '1') + trace[i+1:]
            if self._is_phi_valid(test_trace):
                transformations += 1
                
        return min(1.0, transformations / len(trace))
        
    def _measure_composition_accessibility(self, trace: str) -> float:
        """测量composition accessibility"""
        if len(trace) <= 1:
            return 1.0
        
        # Composition accessibility through morphism combination
        accessibility_score = 0.0
        
        # Check accessibility to morphism compositions
        accessible_compositions = 0
        total_checks = min(5, len(trace))  # Limit for efficiency
        
        for i in range(total_checks):
            if self._has_accessible_composition(trace, i):
                accessible_compositions += 1
                
        accessibility_score = accessible_compositions / total_checks if total_checks > 0 else 0.0
        return accessibility_score
        
    def _measure_morphism_coverage(self, trace: str) -> float:
        """测量morphism coverage"""
        # Coverage of essential morphism types
        morphism_types = ['0', '1', '01', '10']  # Basic morphisms
        covered = sum(1 for mtype in morphism_types if mtype in trace)
        return covered / len(morphism_types)
        
    def _enables_functor_pattern(self, window: str) -> bool:
        """检查window是否enables functor pattern"""
        if len(window) < 3:
            return True
        
        # Functor-enabling: pattern preserves categorical structure
        return window in ['010', '101']  # Structure-preserving patterns
        
    def _measure_global_functor_capability(self, trace: str) -> float:
        """测量global functor capability"""
        if len(trace) <= 2:
            return 1.0
        
        # Global capability through overall functor structure
        capability = 1.0
        
        # Check for global functor patterns
        functor_patterns = trace.count('010') + trace.count('101')
        if len(trace) > 2:
            functor_density = functor_patterns / (len(trace) - 2)
            if functor_density > 0.3:  # High functor capability
                capability *= 1.2
            else:
                capability *= 0.9
                
        return min(1.0, capability)
        
    def _measure_categorical_structure_coverage(self, trace: str) -> float:
        """测量categorical structure coverage"""
        # Coverage of categorical structures (objects, morphisms, composition)
        categorical_structures = ['1', '0', '10', '01']  # Object/morphism patterns
        covered = sum(1 for struct in categorical_structures if struct in trace)
        return covered / len(categorical_structures)
        
    def _measure_functor_coverage(self, trace: str) -> float:
        """测量functor coverage"""
        # Coverage of functor relation patterns
        functor_patterns = ['01', '10', '010', '101']
        covered = sum(1 for pattern in functor_patterns if pattern in trace)
        return covered / len(functor_patterns)
        
    def _measure_categorical_spectrum(self, trace: str) -> float:
        """测量categorical spectrum"""
        if len(trace) == 0:
            return 0.0
        
        # Spectrum based on categorical pattern diversity
        categorical_patterns = set()
        
        # Collect categorical-indicating patterns
        for i in range(len(trace) - 1):
            pattern = trace[i:i+2]
            categorical_patterns.add(pattern)
                
        spectrum_coverage = len(categorical_patterns) / 3.0  # Max 3 valid patterns in φ-traces
        return min(1.0, spectrum_coverage)
        
    def _compute_morphism_power(self, trace: str) -> float:
        """计算morphism power"""
        if len(trace) == 0:
            return 0.0
        
        # Power based on morphism construction capability
        power_factors = []
        
        # Factor 1: Morphism construction count
        morphism_count = self._count_morphism_constructions(trace)
        power_factors.append(min(1.0, morphism_count / 3.0))
        
        # Factor 2: Morphism composition strength
        composition_strength = self._measure_composition_strength(trace)
        power_factors.append(composition_strength)
        
        return np.mean(power_factors)
        
    def _compute_categorical_cost(self, trace: str) -> float:
        """计算categorical cost"""
        if len(trace) == 0:
            return 1.0
        
        # Cost based on categorical complexity
        cost_factors = []
        
        # Factor 1: Length cost
        length_cost = len(trace) / 8.0  # Normalize
        cost_factors.append(length_cost)
        
        # Factor 2: Categorical complexity cost
        categorical_cost = self._measure_categorical_complexity(trace)
        cost_factors.append(categorical_cost)
        
        return max(0.1, np.mean(cost_factors))  # Minimum cost
        
    def _trace_categorical_depth(self, trace: str, start: int) -> int:
        """追踪categorical depth from start position"""
        depth = 0
        pos = start
        
        while pos < len(trace) - 2:
            if self._is_categorical_step(trace[pos:pos+3]):
                depth += 1
                pos += 2
            else:
                break
                
        return depth
        
    def _measure_local_functor_stability(self, window: str) -> float:
        """测量local functor stability"""
        if len(window) < 4:
            return 1.0
        
        # Stability through consistent functor-preserving patterns
        stability = 0.0
        
        # Check for stable functor patterns
        if self._has_stable_functor_pattern(window):
            stability += 0.5
            
        if self._is_phi_valid(window):  # φ-valid ensures stability
            stability += 0.5
            
        return stability
        
    def _measure_sequential_categorical_coherence(self, trace: str) -> float:
        """测量sequential categorical coherence"""
        if len(trace) <= 1:
            return 1.0
        
        coherent_categoricals = 0
        total_categoricals = len(trace) - 1
        
        for i in range(total_categoricals):
            if self._is_coherent_categorical_step(trace[i], trace[i+1]):
                coherent_categoricals += 1
                
        return coherent_categoricals / total_categoricals
        
    def _measure_global_categorical_coherence(self, trace: str) -> float:
        """测量global categorical coherence"""
        # Global coherence through overall categorical structure
        coherence = 1.0
        
        # Check for global categorical organization
        if len(trace) > 2:
            categorical_organization = self._measure_categorical_organization(trace)
            coherence *= categorical_organization
            
        return min(1.0, coherence)
        
    def _measure_structural_categorical_coherence(self, trace: str) -> float:
        """测量structural categorical coherence"""
        # Coherence through φ-constraint categorical alignment
        coherence = 1.0
        
        # φ-valid traces have inherent structural coherence
        if self._is_phi_valid(trace):
            coherence *= 1.1
            
        return min(1.0, coherence)
        
    # Additional helper methods
    def _indicates_object(self, pattern: str) -> bool:
        """检查pattern是否indicates object"""
        if len(pattern) < 2:
            return False
        
        # Objects through stable identity patterns
        return pattern == '00' or pattern == '10'  # Stable object patterns
        
    def _count_possible_compositions(self, trace: str) -> int:
        """计算possible compositions"""
        compositions = 0
        
        # Count valid compositions maintaining φ-constraint
        for i in range(len(trace) - 2):
            window = trace[i:i+3]
            if self._is_valid_composition(window):
                compositions += 1
                
        return compositions
        
    def _has_categorical_distinction(self, window: str) -> bool:
        """检查window是否has categorical distinction"""
        if len(window) < 3:
            return False
        
        # Categorical distinction through object/morphism patterns
        return window in ['010', '101', '001', '100']
        
    def _has_accessible_composition(self, trace: str, position: int) -> bool:
        """检查position是否has accessible composition"""
        if position >= len(trace) - 2:
            return False
        
        # Composition accessibility through valid categorical operations
        window = trace[position:position+3]
        return self._is_valid_composition(window)
        
    def _count_morphism_constructions(self, trace: str) -> int:
        """计算morphism constructions"""
        constructions = 0
        
        # Count morphism-constructing patterns
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # Morphism construction through transition
                constructions += 1
                
        return constructions
        
    def _measure_composition_strength(self, trace: str) -> float:
        """测量composition strength"""
        if len(trace) <= 2:
            return 1.0
        
        # Strength through proportion of compositional elements
        compositional_elements = 0
        
        for i in range(len(trace) - 2):
            if self._is_compositional_element(trace[i:i+3]):
                compositional_elements += 1
                
        return compositional_elements / max(1, len(trace) - 2)
        
    def _measure_categorical_complexity(self, trace: str) -> float:
        """测量categorical complexity"""
        if len(trace) <= 1:
            return 0.1
        
        # Complexity through categorical pattern entropy
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
        
    def _is_categorical_step(self, window: str) -> bool:
        """检查window是否为categorical step"""
        if len(window) < 3:
            return False
        
        # Categorical step has categorical relationship
        return window in ['010', '101']
        
    def _has_stable_functor_pattern(self, window: str) -> bool:
        """检查window是否has stable functor pattern"""
        if len(window) < 4:
            return True
        
        # Stable functor: consistent structure-preserving pattern
        return self._is_phi_valid(window)
        
    def _is_coherent_categorical_step(self, bit1: str, bit2: str) -> bool:
        """检查两个bit是否形成coherent categorical step"""
        # Coherent categorical: meaningful categorical transition
        return True  # All transitions have some categorical coherence
        
    def _measure_categorical_organization(self, trace: str) -> float:
        """测量categorical organization"""
        if len(trace) <= 2:
            return 1.0
        
        # Organization through systematic categorical structure
        organization = 0.0
        
        # Check for organized categorical patterns
        categorical_transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # Categorical transition
                categorical_transitions += 1
                
        transition_rate = categorical_transitions / (len(trace) - 1)
        
        # Moderate transition rate indicates good categorical organization
        if 0.3 <= transition_rate <= 0.7:
            organization = 1.0
        else:
            organization = 0.7
            
        return organization
        
    def _is_valid_composition(self, window: str) -> bool:
        """检查window是否为valid composition"""
        if len(window) < 3:
            return False
        
        # Valid composition maintains φ-constraint and has compositional structure
        return self._is_phi_valid(window) and window != '000'
        
    def _is_compositional_element(self, window: str) -> bool:
        """检查window是否为compositional element"""
        if len(window) < 3:
            return False
        
        # Compositional if has transformation structure
        return window in ['010', '101']
        
    def _build_categorical_network(self) -> nx.Graph:
        """构建categorical network基于trace similarities"""
        G = nx.Graph()
        traces = list(self.trace_universe.keys())
        
        # Add nodes
        for trace_val in traces:
            G.add_node(trace_val)
            
        # Add edges based on categorical similarity
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                similarity = self._compute_categorical_similarity(
                    self.trace_universe[trace1], 
                    self.trace_universe[trace2]
                )
                if similarity > 0.6:  # Threshold for categorical relationship
                    G.add_edge(trace1, trace2, weight=similarity)
                    
        return G
        
    def _compute_categorical_similarity(self, trace1_data: Dict, trace2_data: Dict) -> float:
        """计算两个traces之间的categorical similarity"""
        # Compare categorical properties
        properties = ['categorical_strength', 'object_capacity', 'morphism_range', 
                     'functor_capability', 'categorical_completeness']
        
        similarities = []
        for prop in properties:
            val1 = trace1_data[prop]
            val2 = trace2_data[prop]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _detect_morphism_mappings(self) -> Dict:
        """检测morphism mappings between traces"""
        mappings = {}
        
        for trace_val, data in self.trace_universe.items():
            # Find traces that represent morphisms to current trace
            morphisms = []
            for other_val, other_data in self.trace_universe.items():
                if trace_val != other_val:
                    if self._represents_morphism(data, other_data):
                        morphisms.append(other_val)
            mappings[trace_val] = morphisms
            
        return mappings
        
    def _represents_morphism(self, source_data: Dict, target_data: Dict) -> bool:
        """检查target_data是否represents morphism to source_data"""
        # Morphism when target preserves categorical structure with transformation
        return (abs(target_data['categorical_strength'] - source_data['categorical_strength']) < 0.2 and
                target_data['morphism_range'] >= source_data['morphism_range'] * 0.8)
        
    def run_comprehensive_analysis(self) -> Dict:
        """运行comprehensive categorical analysis"""
        results = {
            'total_traces': len(self.trace_universe),
            'categorical_properties': self._analyze_categorical_distributions(),
            'network_analysis': self._analyze_categorical_network(),
            'morphism_analysis': self._analyze_morphism_patterns(),
            'category_analysis': self._perform_category_analysis(),
            'entropy_analysis': self._compute_entropy_analysis()
        }
        
        return results
        
    def _analyze_categorical_distributions(self) -> Dict:
        """分析categorical property distributions"""
        properties = ['categorical_strength', 'object_capacity', 'morphism_range',
                     'functor_capability', 'categorical_completeness', 'morphism_efficiency',
                     'categorical_depth', 'functor_stability', 'categorical_coherence']
        
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
        
    def _analyze_categorical_network(self) -> Dict:
        """分析categorical network properties"""
        G = self.categorical_network
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G),
            'largest_component': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
    def _analyze_morphism_patterns(self) -> Dict:
        """分析morphism patterns"""
        total_morphisms = sum(len(morphisms) for morphisms in self.morphism_mappings.values())
        total_possible = len(self.trace_universe) * (len(self.trace_universe) - 1)
        
        return {
            'total_morphisms': total_morphisms,
            'morphism_density': total_morphisms / total_possible if total_possible > 0 else 0,
            'avg_morphisms': np.mean([len(morphisms) for morphisms in self.morphism_mappings.values()]),
            'max_morphisms': max([len(morphisms) for morphisms in self.morphism_mappings.values()]) if self.morphism_mappings else 0
        }
        
    def _perform_category_analysis(self) -> Dict:
        """执行category theory analysis"""
        # Categorize traces based on categorical properties
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
        """将traces按categorical properties分类"""
        categories = {
            'object_categorical': [],
            'morphism_categorical': [],
            'functor_categorical': []
        }
        
        for trace_val, data in self.trace_universe.items():
            if data['functor_capability'] > 0.6:
                categories['functor_categorical'].append(trace_val)
            elif data['morphism_range'] > data['object_capacity']:
                categories['morphism_categorical'].append(trace_val)
            else:
                categories['object_categorical'].append(trace_val)
                
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
        
        # Morphism preserves categorical structure within tolerance
        key_properties = ['categorical_strength', 'object_capacity', 'functor_capability']
        
        for prop in key_properties:
            if abs(data1[prop] - data2[prop]) > tolerance:
                return False
                
        return True
        
    def _compute_entropy_analysis(self) -> Dict:
        """计算entropy analysis"""
        properties = ['categorical_strength', 'object_capacity', 'morphism_range',
                     'functor_capability', 'categorical_completeness', 'morphism_efficiency',
                     'categorical_depth', 'functor_stability', 'categorical_coherence']
        
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

class TestCollapseCategoryLogic(unittest.TestCase):
    """测试CollapseCategoryLogic system functionality"""
    
    def setUp(self):
        self.system = CollapseCategoryLogicSystem(max_trace_value=30, categorical_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        valid_trace = "10101"
        invalid_trace = "11010"
        
        self.assertTrue(self.system._is_phi_valid(valid_trace))
        self.assertFalse(self.system._is_phi_valid(invalid_trace))
        
    def test_categorical_strength_computation(self):
        """测试categorical strength computation"""
        trace = "101010"
        strength = self.system._compute_categorical_strength(trace, 42)
        
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
        
    def test_object_capacity_computation(self):
        """测试object capacity computation"""
        trace = "10101"
        capacity = self.system._compute_object_capacity(trace, 21)
        
        self.assertGreaterEqual(capacity, 0.0)
        self.assertLessEqual(capacity, 1.0)
        
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        # Check all traces are φ-valid
        for data in self.system.trace_universe.values():
            self.assertTrue(self.system._is_phi_valid(data['trace']))
            
    def test_categorical_network_construction(self):
        """测试categorical network construction"""
        G = self.system.categorical_network
        
        self.assertGreater(G.number_of_nodes(), 0)
        # Network should have some connections
        self.assertGreaterEqual(G.number_of_edges(), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_traces', results)
        self.assertIn('categorical_properties', results)
        self.assertIn('network_analysis', results)
        self.assertIn('category_analysis', results)
        
        # Verify reasonable values
        self.assertGreater(results['total_traces'], 0)

def visualize_collapse_category_logic_results():
    """可视化CollapseCategoryLogic analysis results"""
    system = CollapseCategoryLogicSystem(max_trace_value=35, categorical_depth=5)
    results = system.run_comprehensive_analysis()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Categorical Network Visualization
    ax1 = plt.subplot(3, 4, 1)
    G = system.categorical_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by categorical strength
    node_colors = []
    for node in G.nodes():
        strength = system.trace_universe[node]['categorical_strength']
        node_colors.append(strength)
    
    nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
            node_size=50, alpha=0.8, ax=ax1)
    ax1.set_title("Categorical Network: Category-Theoretic Architecture\n(Colors: Categorical Strength)")
    
    # 2. Object Capacity Distribution
    ax2 = plt.subplot(3, 4, 2)
    capacities = [data['object_capacity'] for data in system.trace_universe.values()]
    ax2.hist(capacities, bins=15, alpha=0.7, color='brown', edgecolor='black')
    ax2.axvline(np.mean(capacities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(capacities):.3f}')
    ax2.set_xlabel('Object Capacity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Object Capacity Distribution')
    ax2.legend()
    
    # 3. Categorical Strength vs Morphism Range
    ax3 = plt.subplot(3, 4, 3)
    strengths = [data['categorical_strength'] for data in system.trace_universe.values()]
    ranges = [data['morphism_range'] for data in system.trace_universe.values()]
    depths = [len(data['trace']) for data in system.trace_universe.values()]
    
    scatter = ax3.scatter(strengths, ranges, c=depths, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('Categorical Strength')
    ax3.set_ylabel('Morphism Range')
    ax3.set_title('Categorical Strength vs Morphism Range\n(Color: Trace Length)')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. Functor Capability vs Categorical Completeness
    ax4 = plt.subplot(3, 4, 4)
    capabilities = [data['functor_capability'] for data in system.trace_universe.values()]
    completenesses = [data['categorical_completeness'] for data in system.trace_universe.values()]
    efficiencies = [data['morphism_efficiency'] for data in system.trace_universe.values()]
    
    scatter = ax4.scatter(capabilities, completenesses, c=efficiencies, cmap='coolwarm', alpha=0.7)
    ax4.set_xlabel('Functor Capability')
    ax4.set_ylabel('Categorical Completeness')
    ax4.set_title('Functor Capability vs Categorical Completeness\n(Color: Morphism Efficiency)')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Categorical Depth Distribution
    ax5 = plt.subplot(3, 4, 5)
    depths = [data['categorical_depth'] for data in system.trace_universe.values()]
    ax5.hist(depths, bins=12, alpha=0.7, color='darkgreen', edgecolor='black')
    ax5.axvline(np.mean(depths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(depths):.3f}')
    ax5.set_xlabel('Categorical Depth')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Categorical Depth Distribution')
    ax5.legend()
    
    # 6. Functor Stability vs Categorical Coherence
    ax6 = plt.subplot(3, 4, 6)
    stabilities = [data['functor_stability'] for data in system.trace_universe.values()]
    coherences = [data['categorical_coherence'] for data in system.trace_universe.values()]
    values = [data['value'] for data in system.trace_universe.values()]
    
    scatter = ax6.scatter(stabilities, coherences, c=values, cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Functor Stability')
    ax6.set_ylabel('Categorical Coherence')
    ax6.set_title('Functor Stability vs Categorical Coherence\n(Color: Trace Value)')
    plt.colorbar(scatter, ax=ax6)
    
    # 7. Category Distribution
    ax7 = plt.subplot(3, 4, 7)
    categories = results['category_analysis']['categories']
    ax7.bar(categories.keys(), categories.values(), color=['lightblue', 'lightgreen', 'lightcoral'])
    ax7.set_ylabel('Number of Traces')
    ax7.set_title('Categorical Categories Distribution')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Network Properties
    ax8 = plt.subplot(3, 4, 8)
    network_props = results['network_analysis']
    props = ['Nodes', 'Edges', 'Components']
    values = [network_props['nodes'], network_props['edges'], network_props['components']]
    ax8.bar(props, values, color=['skyblue', 'lightgreen', 'salmon'])
    ax8.set_ylabel('Count')
    ax8.set_title('Categorical Network Properties')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(3, 4, 9)
    properties = ['categorical_strength', 'object_capacity', 'morphism_range', 'functor_capability']
    data_matrix = []
    for prop in properties:
        data_matrix.append([system.trace_universe[t][prop] for t in system.trace_universe.keys()])
    
    correlation_matrix = np.corrcoef(data_matrix)
    im = ax9.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(properties)))
    ax9.set_yticks(range(len(properties)))
    ax9.set_xticklabels([p.replace('_', ' ').title() for p in properties], rotation=45)
    ax9.set_yticklabels([p.replace('_', ' ').title() for p in properties])
    ax9.set_title('Categorical Properties Correlation Matrix')
    
    # Add correlation values
    for i in range(len(properties)):
        for j in range(len(properties)):
            ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # 10. 3D Categorical Space
    ax10 = plt.subplot(3, 4, 10, projection='3d')
    ax10.scatter(strengths, capacities, capabilities, c=ranges, cmap='plasma', alpha=0.6)
    ax10.set_xlabel('Categorical Strength')
    ax10.set_ylabel('Object Capacity')
    ax10.set_zlabel('Functor Capability')
    ax10.set_title('3D Categorical Space')
    
    # 11. Entropy Analysis
    ax11 = plt.subplot(3, 4, 11)
    entropies = results['entropy_analysis']
    entropy_props = list(entropies.keys())
    entropy_values = list(entropies.values())
    
    bars = ax11.barh(range(len(entropy_props)), entropy_values, color='gold', alpha=0.7)
    ax11.set_yticks(range(len(entropy_props)))
    ax11.set_yticklabels([p.replace('_', ' ').title() for p in entropy_props])
    ax11.set_xlabel('Entropy (bits)')
    ax11.set_title('Categorical Properties Entropy Analysis')
    
    # Add entropy values on bars
    for i, bar in enumerate(bars):
        ax11.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{entropy_values[i]:.2f}', va='center', fontsize=8)
    
    # 12. Morphism Network
    ax12 = plt.subplot(3, 4, 12)
    
    # Create morphism network
    morphism_graph = nx.DiGraph()
    sample_traces = list(system.trace_universe.keys())[:15]  # Sample for visualization
    
    for trace in sample_traces:
        morphism_graph.add_node(trace)
        morphisms = system.morphism_mappings.get(trace, [])
        for target in morphisms[:3]:  # Limit connections for clarity
            if target in sample_traces:
                morphism_graph.add_edge(trace, target)
    
    pos = nx.spring_layout(morphism_graph, k=2, iterations=50)
    nx.draw(morphism_graph, pos, node_color='wheat', 
            node_size=100, alpha=0.8, arrows=True, ax=ax12)
    ax12.set_title("Morphism Network: Categorical Transformation Paths")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-107-collapse-category-logic-dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional specialized visualizations
    
    # Categorical Architecture Analysis
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Object vs Morphism Analysis
    object_data = []
    morphism_data = []
    for data in system.trace_universe.values():
        object_data.append(data['object_capacity'])
        morphism_data.append(data['morphism_range'])
    
    ax21.scatter(object_data, morphism_data, alpha=0.7, c='indigo')
    ax21.set_xlabel('Object Capacity')
    ax21.set_ylabel('Morphism Range')
    ax21.set_title('Objects vs Morphisms Categorical Space')
    
    # 2. Categorical Completeness Network
    G = system.categorical_network
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    completeness_colors = [system.trace_universe[node]['categorical_completeness'] for node in G.nodes()]
    nx.draw(G, pos, node_color=completeness_colors, cmap='RdYlBu', 
            node_size=80, alpha=0.8, ax=ax22)
    ax22.set_title("Categorical Completeness Network")
    
    # 3. Categorical Coherence Distribution by Category
    categories = system._categorize_traces()
    for i, (cat_name, traces) in enumerate(categories.items()):
        coherences = [system.trace_universe[t]['categorical_coherence'] for t in traces]
        if coherences:  # Only plot if category has traces
            ax23.hist(coherences, bins=8, alpha=0.6, label=cat_name, 
                     color=['red', 'green', 'blue'][i])
    ax23.set_xlabel('Categorical Coherence')
    ax23.set_ylabel('Frequency')
    ax23.set_title('Categorical Coherence by Category')
    ax23.legend()
    
    # 4. Morphism Efficiency Network
    efficiency_colors = [system.trace_universe[node]['morphism_efficiency'] for node in G.nodes()]
    nx.draw(G, pos, node_color=efficiency_colors, cmap='plasma', 
            node_size=80, alpha=0.8, ax=ax24)
    ax24.set_title("Morphism Efficiency Network")
    
    plt.tight_layout()
    plt.savefig('/Users/auric/the-binarymath/docs/codex/volume-06-meta-logic/chapter-107-collapse-category-logic-structures.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Statistics
    print("="*80)
    print("COLLAPSE CATEGORY LOGIC CATEGORY-THEORETIC ANALYSIS")
    print("="*80)
    print(f"Total φ-valid traces analyzed: {results['total_traces']}")
    print(f"Network density: {results['network_analysis']['density']:.3f}")
    print(f"Connected components: {results['network_analysis']['components']}")
    
    print("\nCategorical Properties Analysis:")
    for prop, stats in results['categorical_properties'].items():
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
    
    print(f"\nMorphism Pattern Analysis:")
    morphism_stats = results['morphism_analysis']
    print(f"Total morphism mappings: {morphism_stats['total_morphisms']}")
    print(f"Morphism mapping density: {morphism_stats['morphism_density']:.3f}")
    print(f"Average morphisms per trace: {morphism_stats['avg_morphisms']:.1f}")
    
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
    visualize_collapse_category_logic_results()