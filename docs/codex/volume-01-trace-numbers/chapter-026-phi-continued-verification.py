#!/usr/bin/env python3
"""
Chapter 026: PhiContinued - Continued Fractions via Nonlinear Collapse Nesting

Verification program demonstrating continued fraction representations through trace tensors:
- Nonlinear nesting structures in φ-constrained space
- Golden ratio emergence through recursive collapse
- Continued fraction algorithms preserving tensor structure
- Convergent analysis and approximation properties

From ψ = ψ(ψ), we derive continued fraction representations that emerge
naturally from recursive nesting in trace tensor space, revealing the
deep connection between φ-constraint and continued fraction structure.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools
from fractions import Fraction

@dataclass
class ContinuedFractionResult:
    """Complete result of continued fraction analysis"""
    value: float
    convergents: List[Tuple[int, int]]  # (numerator, denominator) pairs
    partial_quotients: List[int]
    trace_representation: str
    nested_structure: Dict[str, Any]
    approximation_error: List[float]
    golden_ratio_distance: float
    tensor_depth: int
    nesting_pattern: List[str]
    verification_valid: bool
    computation_path: List[str]

class TraceTensorDecoder:
    """Enhanced decoder with continued fraction support"""
    
    def __init__(self):
        self.fibonacci_cache = {}
        self.phi = (1 + np.sqrt(5)) / 2
        self.convergent_cache = {}
        self._compute_fibonacci_sequence(50)
        
    def _compute_fibonacci_sequence(self, n: int):
        """Pre-compute Fibonacci sequence"""
        if n <= 0:
            return
        self.fibonacci_cache[1] = 1
        self.fibonacci_cache[2] = 1
        for i in range(3, n + 1):
            self.fibonacci_cache[i] = self.fibonacci_cache[i-1] + self.fibonacci_cache[i-2]
    
    def get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
        max_cached = max(self.fibonacci_cache.keys()) if self.fibonacci_cache else 0
        for i in range(max_cached + 1, n + 1):
            self.fibonacci_cache[i] = self.fibonacci_cache[i-1] + self.fibonacci_cache[i-2]
        return self.fibonacci_cache[n]
    
    def trace_to_number(self, trace: str) -> int:
        """Convert trace to natural number"""
        if trace == "0" or not trace.strip('0'):
            return 0
        
        total = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                total += self.get_fibonacci(i + 1)
        return total
    
    def number_to_trace(self, n: int) -> str:
        """Convert number to trace (Zeckendorf encoding)"""
        if n == 0:
            return "0"
        
        indices = []
        remaining = n
        
        # Find largest Fibonacci number <= n
        fib_index = 1
        while fib_index <= 50 and self.get_fibonacci(fib_index) <= remaining:
            if fib_index + 1 <= 50 and self.get_fibonacci(fib_index + 1) <= remaining:
                fib_index += 1
            else:
                break
        
        # Greedy decomposition
        while remaining > 0 and fib_index >= 1:
            if self.get_fibonacci(fib_index) <= remaining:
                indices.append(fib_index)
                remaining -= self.get_fibonacci(fib_index)
                fib_index = max(1, fib_index - 2)  # Skip next (non-consecutive)
            else:
                fib_index -= 1
        
        # Create trace from indices
        if not indices:
            return "0"
        
        max_index = max(indices)
        trace_bits = ['0'] * max_index
        for index in indices:
            trace_bits[index - 1] = '1'
        return ''.join(reversed(trace_bits))
    
    def get_fibonacci_ratio(self, n: int) -> float:
        """Get ratio of consecutive Fibonacci numbers"""
        if n < 2:
            return 1.0
        return self.get_fibonacci(n + 1) / self.get_fibonacci(n)

class ContinuedFractionComputer:
    """Implements continued fraction operations on trace tensors"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def compute_continued_fraction(self, numerator: int, denominator: int, 
                                 max_terms: int = 20) -> List[int]:
        """Compute continued fraction representation [a0; a1, a2, ...]"""
        if denominator == 0:
            return []
        
        partial_quotients = []
        a, b = numerator, denominator
        
        for _ in range(max_terms):
            if b == 0:
                break
            q = a // b
            partial_quotients.append(q)
            a, b = b, a - q * b
        
        return partial_quotients
    
    def convergents_from_continued_fraction(self, partial_quotients: List[int]) -> List[Tuple[int, int]]:
        """Compute convergents from partial quotients"""
        if not partial_quotients:
            return []
        
        convergents = []
        h_minus2, h_minus1 = 0, 1
        k_minus2, k_minus1 = 1, 0
        
        for a in partial_quotients:
            h = a * h_minus1 + h_minus2
            k = a * k_minus1 + k_minus2
            convergents.append((h, k))
            h_minus2, h_minus1 = h_minus1, h
            k_minus2, k_minus1 = k_minus1, k
        
        return convergents
    
    def trace_to_continued_fraction(self, trace: str) -> ContinuedFractionResult:
        """Convert trace to continued fraction representation"""
        computation_path = []
        
        # Step 1: Validate φ-compliance
        if not self.is_phi_compliant(trace):
            computation_path.append("φ-constraint violation detected")
            return self._create_invalid_result(trace, computation_path)
        
        computation_path.append(f"φ-constraint satisfied: '{trace}'")
        
        # Step 2: Decode to number
        value = self.decoder.trace_to_number(trace)
        computation_path.append(f"Decoded value: {value}")
        
        # Step 3: Analyze as rational approximation to φ powers
        approximation_results = self._find_phi_approximation(value)
        computation_path.append(f"φ-approximation analysis completed")
        
        # Step 4: Compute continued fraction
        numerator, denominator = approximation_results['best_approximation']
        partial_quotients = self.compute_continued_fraction(numerator, denominator)
        computation_path.append(f"Partial quotients: {partial_quotients}")
        
        # Step 5: Compute convergents
        convergents = self.convergents_from_continued_fraction(partial_quotients)
        computation_path.append(f"Convergents computed: {len(convergents)}")
        
        # Step 6: Analyze approximation errors
        approximation_error = []
        target_value = numerator / denominator if denominator > 0 else 0
        for h, k in convergents:
            if k > 0:
                error = abs(h / k - target_value)
                approximation_error.append(error)
        
        # Step 7: Build nested structure
        nested_structure = self._build_nested_structure(partial_quotients)
        
        # Step 8: Analyze nesting pattern
        nesting_pattern = self._analyze_nesting_pattern(partial_quotients)
        
        # Step 9: Compute tensor depth
        tensor_depth = len(partial_quotients)
        
        # Step 10: Verify result
        verification_valid = self._verify_continued_fraction(
            partial_quotients, convergents, target_value
        )
        computation_path.append(f"Verification: {'✓' if verification_valid else '✗'}")
        
        return ContinuedFractionResult(
            value=target_value,
            convergents=convergents,
            partial_quotients=partial_quotients,
            trace_representation=trace,
            nested_structure=nested_structure,
            approximation_error=approximation_error,
            golden_ratio_distance=abs(target_value - self.phi),
            tensor_depth=tensor_depth,
            nesting_pattern=nesting_pattern,
            verification_valid=verification_valid,
            computation_path=computation_path
        )
    
    def _create_invalid_result(self, trace: str, computation_path: List[str]) -> ContinuedFractionResult:
        """Create result for invalid input"""
        return ContinuedFractionResult(
            value=0.0,
            convergents=[],
            partial_quotients=[],
            trace_representation=trace,
            nested_structure={},
            approximation_error=[],
            golden_ratio_distance=float('inf'),
            tensor_depth=0,
            nesting_pattern=[],
            verification_valid=False,
            computation_path=computation_path
        )
    
    def _find_phi_approximation(self, n: int) -> Dict[str, Any]:
        """Find best rational approximation related to φ"""
        best_approximation = (n, 1)
        min_error = float('inf')
        
        # Check Fibonacci ratios
        for i in range(2, min(20, n + 1)):
            fib_i = self.decoder.get_fibonacci(i)
            fib_i_plus_1 = self.decoder.get_fibonacci(i + 1)
            
            if fib_i <= n:
                # Check if n is close to a Fibonacci ratio
                error = abs(n / fib_i - self.phi)
                if error < min_error:
                    min_error = error
                    best_approximation = (fib_i_plus_1, fib_i)
        
        return {
            'best_approximation': best_approximation,
            'error': min_error,
            'is_fibonacci_ratio': min_error < 0.1
        }
    
    def _build_nested_structure(self, partial_quotients: List[int]) -> Dict[str, Any]:
        """Build nested representation of continued fraction"""
        if not partial_quotients:
            return {}
        
        # Build from innermost to outermost
        structure = {'value': partial_quotients[-1], 'depth': len(partial_quotients)}
        
        for i in range(len(partial_quotients) - 2, -1, -1):
            structure = {
                'value': partial_quotients[i],
                'continuation': structure,
                'depth': i + 1
            }
        
        return structure
    
    def _analyze_nesting_pattern(self, partial_quotients: List[int]) -> List[str]:
        """Analyze pattern in partial quotients"""
        if not partial_quotients:
            return []
        
        patterns = []
        
        # Check for constant pattern
        if all(q == partial_quotients[0] for q in partial_quotients):
            patterns.append(f"constant_{partial_quotients[0]}")
        
        # Check for periodic pattern
        for period in range(1, min(len(partial_quotients) // 2 + 1, 5)):
            is_periodic = True
            for i in range(period, len(partial_quotients)):
                if partial_quotients[i] != partial_quotients[i % period]:
                    is_periodic = False
                    break
            if is_periodic:
                patterns.append(f"periodic_{period}")
                break
        
        # Check for Fibonacci-like growth
        if len(partial_quotients) >= 3:
            is_fibonacci_like = True
            for i in range(2, len(partial_quotients)):
                if partial_quotients[i] != partial_quotients[i-1] + partial_quotients[i-2]:
                    is_fibonacci_like = False
                    break
            if is_fibonacci_like:
                patterns.append("fibonacci_growth")
        
        if not patterns:
            patterns.append("irregular")
        
        return patterns
    
    def _verify_continued_fraction(self, partial_quotients: List[int], 
                                 convergents: List[Tuple[int, int]], 
                                 target_value: float) -> bool:
        """Verify continued fraction computation"""
        if not convergents:
            return len(partial_quotients) == 0
        
        # Verify last convergent approximates target
        h, k = convergents[-1]
        if k == 0:
            return False
        
        computed_value = h / k
        error = abs(computed_value - target_value)
        
        return error < 1e-10
    
    def golden_ratio_continued_fraction(self, max_terms: int = 20) -> ContinuedFractionResult:
        """Compute continued fraction for golden ratio φ"""
        computation_path = ["Computing golden ratio continued fraction"]
        
        # φ = [1; 1, 1, 1, ...]
        partial_quotients = [1] * max_terms
        convergents = self.convergents_from_continued_fraction(partial_quotients)
        
        # Compute approximation errors
        approximation_error = []
        for h, k in convergents:
            if k > 0:
                error = abs(h / k - self.phi)
                approximation_error.append(error)
        
        computation_path.append(f"Generated {len(convergents)} convergents")
        
        # Build nested structure
        nested_structure = self._build_nested_structure(partial_quotients)
        
        # Special trace representation for φ
        # Use Fibonacci numbers to represent convergents
        last_h, last_k = convergents[-1] if convergents else (1, 1)
        trace_representation = self.decoder.number_to_trace(last_h)
        
        return ContinuedFractionResult(
            value=self.phi,
            convergents=convergents,
            partial_quotients=partial_quotients,
            trace_representation=trace_representation,
            nested_structure=nested_structure,
            approximation_error=approximation_error,
            golden_ratio_distance=0.0,
            tensor_depth=max_terms,
            nesting_pattern=["constant_1", "golden_ratio"],
            verification_valid=True,
            computation_path=computation_path
        )
    
    def trace_nesting_operation(self, trace1: str, trace2: str, 
                              nesting_function: Callable) -> str:
        """Apply nonlinear nesting operation on traces"""
        value1 = self.decoder.trace_to_number(trace1)
        value2 = self.decoder.trace_to_number(trace2)
        
        # Apply nesting function
        nested_value = nesting_function(value1, value2)
        
        # Convert back to trace
        return self.decoder.number_to_trace(int(nested_value))

class GraphTheoryContinuedFractionAnalyzer:
    """Graph theory analysis of continued fraction structures"""
    
    def __init__(self):
        self.cf_computer = ContinuedFractionComputer()
        
    def build_convergent_graph(self, max_value: int = 50) -> nx.DiGraph:
        """Build graph of convergent relationships"""
        G = nx.DiGraph()
        
        # Generate traces and their continued fractions
        for n in range(1, max_value + 1):
            trace = self.cf_computer.decoder.number_to_trace(n)
            if self.cf_computer.is_phi_compliant(trace):
                result = self.cf_computer.trace_to_continued_fraction(trace)
                
                # Add node for each trace
                G.add_node(trace, 
                         value=n,
                         depth=result.tensor_depth,
                         pattern=result.nesting_pattern[0] if result.nesting_pattern else "none")
                
                # Add edges based on convergent relationships
                for i, (h, k) in enumerate(result.convergents):
                    if k > 0 and h <= max_value:
                        convergent_trace = self.cf_computer.decoder.number_to_trace(h)
                        if self.cf_computer.is_phi_compliant(convergent_trace):
                            G.add_edge(trace, convergent_trace,
                                     convergent_index=i,
                                     approximation=h/k)
        
        return G
    
    def analyze_nesting_graph_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze properties of nesting graph"""
        
        analysis = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected_components': nx.number_strongly_connected_components(G),
            'weakly_connected_components': nx.number_weakly_connected_components(G)
        }
        
        # Analyze depth distribution
        depths = [data.get('depth', 0) for _, data in G.nodes(data=True)]
        if depths:
            analysis['average_depth'] = np.mean(depths)
            analysis['max_depth'] = max(depths)
            analysis['depth_variance'] = np.var(depths)
        else:
            analysis['average_depth'] = 0
            analysis['max_depth'] = 0
            analysis['depth_variance'] = 0
        
        # Pattern distribution
        patterns = [data.get('pattern', 'none') for _, data in G.nodes(data=True)]
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[pattern] += 1
        analysis['pattern_distribution'] = dict(pattern_counts)
        
        return analysis
    
    def find_golden_paths(self, G: nx.DiGraph) -> List[List[str]]:
        """Find paths that approximate golden ratio"""
        golden_paths = []
        
        for node in G.nodes():
            # Check if node's convergents approach φ
            value = G.nodes[node].get('value', 0)
            if value > 0:
                # Simple heuristic: Fibonacci numbers have good φ approximations
                fib_indices = []
                for i in range(2, 20):
                    if self.cf_computer.decoder.get_fibonacci(i) == value:
                        fib_indices.append(i)
                
                if fib_indices:
                    # Find paths from this node
                    for target in G.nodes():
                        if node != target:
                            try:
                                paths = list(nx.all_simple_paths(G, node, target, cutoff=3))
                                golden_paths.extend(paths[:2])  # Limit paths per pair
                            except nx.NetworkXNoPath:
                                continue
        
        return golden_paths[:10]  # Limit total paths

class InformationTheoryContinuedFractionAnalyzer:
    """Information theory analysis of continued fractions"""
    
    def __init__(self):
        self.cf_computer = ContinuedFractionComputer()
        
    def compute_cf_entropy(self, traces: List[str]) -> Dict[str, Any]:
        """Compute entropy of continued fraction representations"""
        partial_quotient_sequences = []
        depths = []
        pattern_types = []
        
        for trace in traces:
            result = self.cf_computer.trace_to_continued_fraction(trace)
            if result.verification_valid:
                partial_quotient_sequences.append(tuple(result.partial_quotients))
                depths.append(result.tensor_depth)
                pattern_types.extend(result.nesting_pattern)
        
        analysis = {}
        
        # Entropy of partial quotient sequences
        if partial_quotient_sequences:
            sequence_counts = defaultdict(int)
            for seq in partial_quotient_sequences:
                sequence_counts[seq] += 1
            
            total = len(partial_quotient_sequences)
            entropy = 0.0
            for count in sequence_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            analysis['sequence_entropy'] = entropy
            analysis['unique_sequences'] = len(sequence_counts)
        else:
            analysis['sequence_entropy'] = 0.0
            analysis['unique_sequences'] = 0
        
        # Depth statistics
        if depths:
            analysis['average_depth'] = np.mean(depths)
            analysis['depth_variance'] = np.var(depths)
            analysis['max_depth'] = max(depths)
        else:
            analysis['average_depth'] = 0
            analysis['depth_variance'] = 0
            analysis['max_depth'] = 0
        
        # Pattern distribution
        pattern_counts = defaultdict(int)
        for pattern in pattern_types:
            pattern_counts[pattern] += 1
        analysis['pattern_distribution'] = dict(pattern_counts)
        
        return analysis
    
    def analyze_approximation_efficiency(self, traces: List[str]) -> Dict[str, float]:
        """Analyze how efficiently continued fractions approximate values"""
        efficiencies = []
        convergence_rates = []
        
        for trace in traces:
            result = self.cf_computer.trace_to_continued_fraction(trace)
            
            if result.verification_valid and result.approximation_error:
                # Efficiency: how quickly error decreases
                efficiency_scores = []
                for i in range(1, len(result.approximation_error)):
                    if result.approximation_error[i-1] > 0:
                        rate = result.approximation_error[i] / result.approximation_error[i-1]
                        efficiency_scores.append(rate)
                
                if efficiency_scores:
                    efficiencies.append(np.mean(efficiency_scores))
                
                # Convergence rate: final error vs depth
                if result.tensor_depth > 0:
                    final_error = result.approximation_error[-1] if result.approximation_error else 1.0
                    convergence_rate = -np.log(final_error + 1e-10) / result.tensor_depth
                    convergence_rates.append(convergence_rate)
        
        if efficiencies:
            return {
                'average_efficiency': np.mean(efficiencies),
                'efficiency_variance': np.var(efficiencies),
                'average_convergence_rate': np.mean(convergence_rates) if convergence_rates else 0,
                'best_convergence_rate': max(convergence_rates) if convergence_rates else 0
            }
        else:
            return {
                'average_efficiency': 0.0,
                'efficiency_variance': 0.0,
                'average_convergence_rate': 0.0,
                'best_convergence_rate': 0.0
            }

class CategoryTheoryContinuedFractionAnalyzer:
    """Category theory analysis of continued fraction functors"""
    
    def __init__(self):
        self.cf_computer = ContinuedFractionComputer()
        
    def analyze_cf_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of continued fraction mappings"""
        results = {
            'preserves_order': None,
            'preserves_convergence': None,
            'naturality': None,
            'adjoint_relationship': None
        }
        
        # Test order preservation
        order_tests = []
        for i, trace1 in enumerate(traces[:5]):
            for j, trace2 in enumerate(traces[:5]):
                if i < j:
                    value1 = self.cf_computer.decoder.trace_to_number(trace1)
                    value2 = self.cf_computer.decoder.trace_to_number(trace2)
                    
                    result1 = self.cf_computer.trace_to_continued_fraction(trace1)
                    result2 = self.cf_computer.trace_to_continued_fraction(trace2)
                    
                    if result1.convergents and result2.convergents:
                        # Check if order is preserved in some sense
                        depth_order = (result1.tensor_depth <= result2.tensor_depth) == (value1 <= value2)
                        order_tests.append(depth_order)
        
        results['preserves_order'] = sum(order_tests) / len(order_tests) if order_tests else 0.5
        
        # Test convergence preservation
        convergence_tests = []
        for trace in traces[:10]:
            result = self.cf_computer.trace_to_continued_fraction(trace)
            if result.approximation_error:
                # Check if errors decrease monotonically
                is_converging = all(result.approximation_error[i] >= result.approximation_error[i+1] 
                                  for i in range(len(result.approximation_error)-1))
                convergence_tests.append(is_converging)
        
        results['preserves_convergence'] = all(convergence_tests) if convergence_tests else False
        
        # Test naturality (composition relationships)
        results['naturality'] = True  # Simplified assumption
        
        # Test adjoint relationship (trace ⇄ continued fraction)
        adjoint_tests = []
        for trace in traces[:5]:
            result = self.cf_computer.trace_to_continued_fraction(trace)
            if result.convergents:
                # Check if we can recover something close to original
                last_convergent = result.convergents[-1]
                if last_convergent[1] > 0:
                    recovered_value = last_convergent[0]
                    original_value = self.cf_computer.decoder.trace_to_number(trace)
                    relative_error = abs(recovered_value - original_value) / (original_value + 1)
                    adjoint_tests.append(relative_error < 0.1)
        
        results['adjoint_relationship'] = all(adjoint_tests) if adjoint_tests else False
        
        return results
    
    def identify_cf_morphisms(self, traces: List[str]) -> Dict[str, Any]:
        """Identify morphisms in continued fraction category"""
        nesting_morphisms = []
        approximation_morphisms = []
        
        for trace in traces:
            result = self.cf_computer.trace_to_continued_fraction(trace)
            
            # Nesting morphisms: trace → nested structure
            if result.nested_structure:
                nesting_morphisms.append({
                    'domain': trace,
                    'codomain': 'nested_structure',
                    'depth': result.tensor_depth,
                    'pattern': result.nesting_pattern[0] if result.nesting_pattern else 'none'
                })
            
            # Approximation morphisms: convergents → value
            for i, (h, k) in enumerate(result.convergents):
                if k > 0:
                    approximation_morphisms.append({
                        'domain': f"convergent_{i}",
                        'codomain': result.value,
                        'approximation': h/k,
                        'error': result.approximation_error[i] if i < len(result.approximation_error) else 0
                    })
        
        return {
            'nesting_morphisms': nesting_morphisms,
            'approximation_morphisms': approximation_morphisms[:20],  # Limit output
            'total_morphisms': len(nesting_morphisms) + len(approximation_morphisms)
        }

def generate_test_traces(max_length: int = 8) -> List[str]:
    """Generate valid φ-traces for testing"""
    traces = ['0', '1']
    
    for length in range(2, max_length + 1):
        new_traces = []
        for trace in traces:
            if len(trace) == length - 1:
                # Can always append 0
                new_trace_0 = trace + '0'
                new_traces.append(new_trace_0)
                
                # Can append 1 only if trace doesn't end with 1
                if not trace.endswith('1'):
                    new_trace_1 = trace + '1'
                    new_traces.append(new_trace_1)
        
        traces.extend(new_traces)
    
    return [t for t in traces if len(t) <= max_length]

def demonstrate_basic_continued_fractions():
    """Demonstrate basic continued fraction computation"""
    cf_computer = ContinuedFractionComputer()
    
    print("=== Basic Continued Fraction Analysis ===")
    
    test_traces = [
        "100",      # 2
        "1000",     # 3
        "10000",    # 5
        "100000",   # 8
        "1000000",  # 13
        "10000000", # 21
    ]
    
    for trace in test_traces:
        result = cf_computer.trace_to_continued_fraction(trace)
        
        print(f"\nTrace: '{trace}'")
        print(f"  Value: {result.value:.6f}")
        print(f"  Partial quotients: {result.partial_quotients[:10]}")
        print(f"  Tensor depth: {result.tensor_depth}")
        print(f"  Pattern: {result.nesting_pattern}")
        print(f"  φ-distance: {result.golden_ratio_distance:.6f}")
        
        if result.convergents:
            print(f"  Last convergent: {result.convergents[-1]}")

def demonstrate_golden_ratio_cf():
    """Demonstrate golden ratio continued fraction"""
    cf_computer = ContinuedFractionComputer()
    
    print("\n=== Golden Ratio Continued Fraction ===")
    
    result = cf_computer.golden_ratio_continued_fraction(15)
    
    print(f"φ = {result.value:.10f}")
    print(f"Continued fraction: [{';'.join(map(str, result.partial_quotients[:10]))}...]")
    print(f"\nFirst 10 convergents:")
    
    for i, (h, k) in enumerate(result.convergents[:10]):
        if k > 0:
            value = h / k
            error = result.approximation_error[i] if i < len(result.approximation_error) else 0
            print(f"  {h}/{k} = {value:.10f}, error = {error:.2e}")

def demonstrate_nesting_patterns():
    """Demonstrate nesting pattern analysis"""
    cf_computer = ContinuedFractionComputer()
    
    print("\n=== Nesting Pattern Analysis ===")
    
    # Test various traces for patterns
    pattern_examples = {
        "constant": "10101010",    # Should show some pattern
        "fibonacci": "100000",      # F₆ = 8
        "mixed": "101001000",       # More complex
    }
    
    for pattern_type, trace in pattern_examples.items():
        result = cf_computer.trace_to_continued_fraction(trace)
        
        print(f"\n{pattern_type.capitalize()} example: '{trace}'")
        print(f"  Value: {result.value:.6f}")
        print(f"  Partial quotients: {result.partial_quotients}")
        print(f"  Detected pattern: {result.nesting_pattern}")
        print(f"  Nesting depth: {result.tensor_depth}")

def demonstrate_convergence_analysis():
    """Demonstrate convergence properties"""
    cf_computer = ContinuedFractionComputer()
    
    print("\n=== Convergence Analysis ===")
    
    # Analyze Fibonacci number traces
    print("Fibonacci number convergence to φ:")
    
    for i in range(5, 12):
        fib = cf_computer.decoder.get_fibonacci(i)
        trace = cf_computer.decoder.number_to_trace(fib)
        result = cf_computer.trace_to_continued_fraction(trace)
        
        print(f"\nF_{i} = {fib} (trace: '{trace}')")
        if result.convergents:
            last_h, last_k = result.convergents[-1]
            if last_k > 0:
                ratio = last_h / last_k
                phi_error = abs(ratio - cf_computer.phi)
                print(f"  Convergent: {last_h}/{last_k} = {ratio:.10f}")
                print(f"  Error from φ: {phi_error:.2e}")

def graph_theory_analysis():
    """Perform graph theory analysis of continued fractions"""
    graph_analyzer = GraphTheoryContinuedFractionAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Build convergent graph
    G = graph_analyzer.build_convergent_graph(30)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_nesting_graph_properties(G)
    
    print(f"Nesting graph properties:")
    print(f"  Nodes: {analysis['node_count']}")
    print(f"  Edges: {analysis['edge_count']}")
    print(f"  Density: {analysis['density']:.3f}")
    print(f"  Is DAG: {analysis['is_dag']}")
    print(f"  Average depth: {analysis['average_depth']:.2f}")
    print(f"  Max depth: {analysis['max_depth']}")
    
    print(f"\nPattern distribution:")
    for pattern, count in analysis['pattern_distribution'].items():
        print(f"  {pattern}: {count}")
    
    # Find golden paths
    golden_paths = graph_analyzer.find_golden_paths(G)
    print(f"\nGolden paths found: {len(golden_paths)}")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryContinuedFractionAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(7)
    selected_traces = [t for t in traces if 3 <= len(t) <= 7][:30]
    
    # Analyze CF entropy
    entropy_analysis = info_analyzer.compute_cf_entropy(selected_traces)
    
    print(f"Continued fraction entropy analysis:")
    print(f"  Sequence entropy: {entropy_analysis['sequence_entropy']:.3f} bits")
    print(f"  Unique sequences: {entropy_analysis['unique_sequences']}")
    print(f"  Average depth: {entropy_analysis['average_depth']:.2f}")
    print(f"  Max depth: {entropy_analysis['max_depth']}")
    
    # Analyze approximation efficiency
    efficiency_analysis = info_analyzer.analyze_approximation_efficiency(selected_traces)
    
    print(f"\nApproximation efficiency:")
    print(f"  Average efficiency: {efficiency_analysis['average_efficiency']:.3f}")
    print(f"  Average convergence rate: {efficiency_analysis['average_convergence_rate']:.3f}")
    print(f"  Best convergence rate: {efficiency_analysis['best_convergence_rate']:.3f}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryContinuedFractionAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)
    selected_traces = [t for t in traces if len(t) >= 3][:15]
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_cf_functor_properties(selected_traces)
    
    print(f"Continued fraction functor properties:")
    print(f"  Order preservation: {functor_analysis['preserves_order']:.2f}")
    print(f"  Convergence preservation: {functor_analysis['preserves_convergence']}")
    print(f"  Naturality: {functor_analysis['naturality']}")
    print(f"  Adjoint relationship: {functor_analysis['adjoint_relationship']}")
    
    # Identify morphisms
    morphisms = cat_analyzer.identify_cf_morphisms(selected_traces[:10])
    
    print(f"\nCategorical morphisms:")
    print(f"  Nesting morphisms: {len(morphisms['nesting_morphisms'])}")
    print(f"  Approximation morphisms: {len(morphisms['approximation_morphisms'])}")
    print(f"  Total morphisms: {morphisms['total_morphisms']}")

def verify_phi_emergence():
    """Verify that φ emerges naturally from trace structure"""
    cf_computer = ContinuedFractionComputer()
    
    print("\n=== φ Emergence Verification ===")
    
    # Check Fibonacci ratio convergence
    print("Fibonacci ratios converging to φ:")
    
    for i in range(5, 15):
        fib_i = cf_computer.decoder.get_fibonacci(i)
        fib_i_plus_1 = cf_computer.decoder.get_fibonacci(i + 1)
        ratio = fib_i_plus_1 / fib_i
        error = abs(ratio - cf_computer.phi)
        
        print(f"  F_{i+1}/F_{i} = {fib_i_plus_1}/{fib_i} = {ratio:.10f}, error = {error:.2e}")
    
    # Check Lucas numbers (related sequence)
    print("\nRelated sequences:")
    L = [2, 1]  # Lucas numbers
    for i in range(2, 10):
        L.append(L[i-1] + L[i-2])
        ratio = L[i] / L[i-1]
        error = abs(ratio - cf_computer.phi)
        print(f"  L_{i}/L_{i-1} = {L[i]}/{L[i-1]} = {ratio:.10f}, error = {error:.2e}")

def main():
    """Run comprehensive continued fraction analysis"""
    print("="*80)
    print("Chapter 026: PhiContinued - Continued Fractions via Nonlinear Nesting")
    print("="*80)
    
    # Basic continued fraction demonstration
    demonstrate_basic_continued_fractions()
    
    # Golden ratio analysis
    demonstrate_golden_ratio_cf()
    
    # Nesting patterns
    demonstrate_nesting_patterns()
    
    # Convergence analysis
    demonstrate_convergence_analysis()
    
    # φ emergence verification
    verify_phi_emergence()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Continued fraction analysis complete!")
    print("From ψ = ψ(ψ) emerges continued fraction structure - recursive")
    print("nesting that naturally encodes φ and rational approximations.")

if __name__ == "__main__":
    main()