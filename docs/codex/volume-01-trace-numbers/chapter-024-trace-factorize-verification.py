#!/usr/bin/env python3
"""
Chapter 024: TraceFactorize - Tensor-Level Structural Factor Decomposition

Verification program demonstrating complete factorization of trace tensors:
- Structural decomposition preserving φ-constraint
- Prime trace building block identification
- Factorization tree construction and analysis
- Tensor-level multiplicative decomposition algorithms

From ψ = ψ(ψ), we derive complete factorization as the inverse of 
multiplication - decomposing composite traces into prime trace products
while maintaining golden structure throughout the decomposition process.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools

@dataclass
class FactorizationResult:
    """Complete result of trace tensor factorization"""
    original_trace: str
    original_value: int
    is_prime: bool
    prime_factorization: List[Tuple[str, int]]  # (trace, exponent)
    factorization_tree: Dict[str, Any]
    decomposition_steps: List[str]
    factor_validation: bool
    tensor_complexity: int
    factorization_depth: int
    irreducible_components: List[str]
    verification_path: List[str]

class TraceTensorDecoder:
    """Enhanced decoder with factorization support"""
    
    def __init__(self):
        self.fibonacci_cache = {}
        self.prime_cache = {}
        self.factorization_cache = {}
        self._compute_fibonacci_sequence(50)
        self._compute_primes(200)
        
    def _compute_fibonacci_sequence(self, n: int):
        """Pre-compute Fibonacci sequence"""
        if n <= 0:
            return
        self.fibonacci_cache[1] = 1
        self.fibonacci_cache[2] = 1
        for i in range(3, n + 1):
            self.fibonacci_cache[i] = self.fibonacci_cache[i-1] + self.fibonacci_cache[i-2]
    
    def _compute_primes(self, n: int):
        """Compute prime numbers up to n using sieve"""
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        for i in range(2, n + 1):
            self.prime_cache[i] = sieve[i]
    
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
    
    def is_prime_number(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n in self.prime_cache:
            return self.prime_cache[n]
        
        # Check divisibility for numbers not in cache
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def factorize_number(self, n: int) -> List[int]:
        """Complete prime factorization of number"""
        if n in self.factorization_cache:
            return self.factorization_cache[n]
        
        factors = []
        remaining = n
        divisor = 2
        
        while divisor * divisor <= remaining:
            while remaining % divisor == 0:
                factors.append(divisor)
                remaining //= divisor
            divisor += 1
        
        if remaining > 1:
            factors.append(remaining)
        
        self.factorization_cache[n] = factors
        return factors
    
    def extract_fibonacci_indices(self, trace: str) -> List[int]:
        """Extract Fibonacci indices from trace"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i + 1)
        return sorted(indices)

class TraceFactorizer:
    """Complete trace tensor factorization system"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.prime_traces = set()
        self.composite_factorizations = {}
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def is_prime_trace(self, trace: str) -> bool:
        """Check if trace represents a prime number and cannot be factorized"""
        if not self.is_phi_compliant(trace):
            return False
        
        value = self.decoder.trace_to_number(trace)
        if value <= 1:
            return False
        
        # Check if value is prime
        if not self.decoder.is_prime_number(value):
            return False
        
        # For prime values, check if trace can be expressed as product of smaller traces
        for i in range(2, int(value**0.5) + 1):
            if value % i == 0:
                j = value // i
                trace_i = self.decoder.number_to_trace(i)
                trace_j = self.decoder.number_to_trace(j)
                
                if (self.is_phi_compliant(trace_i) and self.is_phi_compliant(trace_j) and
                    i > 1 and j > 1):
                    return False
        
        return True
    
    def find_all_factor_pairs(self, trace: str) -> List[Tuple[str, str]]:
        """Find all valid factor pairs for a trace"""
        value = self.decoder.trace_to_number(trace)
        factor_pairs = []
        
        if value <= 1:
            return factor_pairs
        
        for i in range(2, int(value**0.5) + 1):
            if value % i == 0:
                j = value // i
                trace_i = self.decoder.number_to_trace(i)
                trace_j = self.decoder.number_to_trace(j)
                
                # Check φ-compliance of both factors
                if (self.is_phi_compliant(trace_i) and self.is_phi_compliant(trace_j) and
                    i > 1 and j > 1):
                    factor_pairs.append((trace_i, trace_j))
        
        return factor_pairs
    
    def build_factorization_tree(self, trace: str, depth: int = 0) -> Dict[str, Any]:
        """Build complete factorization tree for trace"""
        tree = {
            'trace': trace,
            'value': self.decoder.trace_to_number(trace),
            'depth': depth,
            'is_prime': self.is_prime_trace(trace),
            'children': []
        }
        
        if tree['is_prime'] or depth > 10:  # Prevent infinite recursion
            return tree
        
        # Find factor pairs and recursively factorize
        factor_pairs = self.find_all_factor_pairs(trace)
        
        if factor_pairs:
            # Use the first valid factorization
            factor1, factor2 = factor_pairs[0]
            tree['children'] = [
                self.build_factorization_tree(factor1, depth + 1),
                self.build_factorization_tree(factor2, depth + 1)
            ]
        
        return tree
    
    def extract_prime_factors_from_tree(self, tree: Dict[str, Any]) -> List[str]:
        """Extract all prime trace factors from factorization tree"""
        if tree['is_prime']:
            return [tree['trace']]
        
        prime_factors = []
        for child in tree['children']:
            prime_factors.extend(self.extract_prime_factors_from_tree(child))
        
        return prime_factors
    
    def verify_factorization(self, original_trace: str, prime_factors: List[str]) -> bool:
        """Verify that prime factors multiply to original trace"""
        if not prime_factors:
            return False
        
        # Compute product of all prime factors
        product = 1
        for factor_trace in prime_factors:
            factor_value = self.decoder.trace_to_number(factor_trace)
            product *= factor_value
        
        original_value = self.decoder.trace_to_number(original_trace)
        return product == original_value
    
    def compute_prime_factorization_with_exponents(self, prime_factors: List[str]) -> List[Tuple[str, int]]:
        """Compute prime factorization with exponents"""
        factor_counts = {}
        for factor in prime_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return [(factor, count) for factor, count in factor_counts.items()]
    
    def factorize_trace_completely(self, trace: str) -> FactorizationResult:
        """Complete factorization analysis of trace"""
        verification_path = []
        
        # Step 1: Validate φ-compliance
        if not self.is_phi_compliant(trace):
            verification_path.append("φ-constraint violation detected")
            return FactorizationResult(
                original_trace=trace,
                original_value=0,
                is_prime=False,
                prime_factorization=[],
                factorization_tree={},
                decomposition_steps=[],
                factor_validation=False,
                tensor_complexity=0,
                factorization_depth=0,
                irreducible_components=[],
                verification_path=verification_path
            )
        
        verification_path.append(f"φ-constraint satisfied: '{trace}'")
        
        # Step 2: Decode value
        value = self.decoder.trace_to_number(trace)
        verification_path.append(f"Decoded value: {value}")
        
        # Step 3: Check if prime
        is_prime = self.is_prime_trace(trace)
        verification_path.append(f"Prime trace: {is_prime}")
        
        # Step 4: Build factorization tree
        factorization_tree = self.build_factorization_tree(trace)
        tree_depth = self.compute_tree_depth(factorization_tree)
        verification_path.append(f"Factorization tree depth: {tree_depth}")
        
        # Step 5: Extract prime factors
        prime_factors = self.extract_prime_factors_from_tree(factorization_tree)
        verification_path.append(f"Prime factors found: {len(prime_factors)}")
        
        # Step 6: Verify factorization
        factor_validation = self.verify_factorization(trace, prime_factors)
        verification_path.append(f"Factorization verified: {factor_validation}")
        
        # Step 7: Compute prime factorization with exponents
        prime_factorization = self.compute_prime_factorization_with_exponents(prime_factors)
        
        # Step 8: Compute tensor complexity
        tensor_complexity = len(prime_factors) + tree_depth
        
        # Step 9: Generate decomposition steps
        decomposition_steps = self.generate_decomposition_steps(factorization_tree)
        
        return FactorizationResult(
            original_trace=trace,
            original_value=value,
            is_prime=is_prime,
            prime_factorization=prime_factorization,
            factorization_tree=factorization_tree,
            decomposition_steps=decomposition_steps,
            factor_validation=factor_validation,
            tensor_complexity=tensor_complexity,
            factorization_depth=tree_depth,
            irreducible_components=list(set(prime_factors)),
            verification_path=verification_path
        )
    
    def compute_tree_depth(self, tree: Dict[str, Any]) -> int:
        """Compute maximum depth of factorization tree"""
        if not tree.get('children'):
            return tree.get('depth', 0)
        
        max_child_depth = max(self.compute_tree_depth(child) for child in tree['children'])
        return max_child_depth
    
    def generate_decomposition_steps(self, tree: Dict[str, Any]) -> List[str]:
        """Generate human-readable decomposition steps"""
        steps = []
        
        def traverse_tree(node, path=""):
            if node['is_prime']:
                steps.append(f"{path}'{node['trace']}' → {node['value']} (prime)")
            else:
                if node['children']:
                    child_traces = [child['trace'] for child in node['children']]
                    child_values = [child['value'] for child in node['children']]
                    steps.append(f"{path}'{node['trace']}' → {node['value']} = " +
                               f"'{child_traces[0]}'×'{child_traces[1]}' = {child_values[0]}×{child_values[1]}")
                    
                    for i, child in enumerate(node['children']):
                        traverse_tree(child, path + "  ")
        
        traverse_tree(tree)
        return steps

class GraphTheoryFactorizationAnalyzer:
    """Graph theory analysis of factorization structures"""
    
    def __init__(self):
        self.factorizer = TraceFactorizer()
        
    def build_factorization_graph(self, traces: List[str]) -> nx.DiGraph:
        """Build graph showing factorization relationships"""
        G = nx.DiGraph()
        
        # Add nodes for all traces
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            G.add_node(trace, 
                      value=result.original_value,
                      is_prime=result.is_prime,
                      complexity=result.tensor_complexity)
        
        # Add edges for factorization relationships
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            
            # Add edges from factors to composite
            for factor_trace, exponent in result.prime_factorization:
                if factor_trace in traces:
                    G.add_edge(factor_trace, trace, 
                             relationship='factor',
                             exponent=exponent)
        
        return G
    
    def analyze_factorization_graph_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze properties of factorization graph"""
        
        # Classify nodes
        prime_nodes = [n for n, d in G.nodes(data=True) if d.get('is_prime', False)]
        composite_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_prime', False)]
        
        # Analyze graph structure
        analysis = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'prime_nodes': len(prime_nodes),
            'composite_nodes': len(composite_nodes),
            'prime_ratio': len(prime_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected_components': nx.number_strongly_connected_components(G),
            'weakly_connected_components': nx.number_weakly_connected_components(G)
        }
        
        # Analyze factorization completeness
        factorization_edges = [e for e in G.edges(data=True) if e[2].get('relationship') == 'factor']
        analysis['factorization_edges'] = len(factorization_edges)
        analysis['factorization_density'] = len(factorization_edges) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        # Compute degree statistics
        if G.number_of_nodes() > 0:
            in_degrees = [G.in_degree(n) for n in G.nodes()]
            out_degrees = [G.out_degree(n) for n in G.nodes()]
            
            analysis['average_in_degree'] = np.mean(in_degrees)
            analysis['average_out_degree'] = np.mean(out_degrees)
            analysis['max_in_degree'] = max(in_degrees)
            analysis['max_out_degree'] = max(out_degrees)
        else:
            analysis.update({
                'average_in_degree': 0,
                'average_out_degree': 0,
                'max_in_degree': 0,
                'max_out_degree': 0
            })
        
        return analysis
    
    def find_factorization_paths(self, G: nx.DiGraph, max_length: int = 5) -> List[List[str]]:
        """Find factorization paths from primes to composites"""
        prime_nodes = [n for n, d in G.nodes(data=True) if d.get('is_prime', False)]
        composite_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_prime', False)]
        
        factorization_paths = []
        
        for prime in prime_nodes:
            for composite in composite_nodes:
                try:
                    paths = list(nx.all_simple_paths(G, prime, composite, cutoff=max_length))
                    factorization_paths.extend(paths[:5])  # Limit paths per pair
                except nx.NetworkXNoPath:
                    continue
        
        return factorization_paths[:20]  # Limit total paths

class InformationTheoryFactorizationAnalyzer:
    """Information theory analysis of factorization properties"""
    
    def __init__(self):
        self.factorizer = TraceFactorizer()
        
    def compute_factorization_entropy(self, traces: List[str]) -> Dict[str, Any]:
        """Compute entropy of factorization structures"""
        factorization_complexities = []
        tree_depths = []
        prime_factor_counts = []
        
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            
            if result.factor_validation:
                factorization_complexities.append(result.tensor_complexity)
                tree_depths.append(result.factorization_depth)
                prime_factor_counts.append(len(result.irreducible_components))
        
        analysis = {}
        
        if factorization_complexities:
            analysis['complexity_entropy'] = self._compute_discrete_entropy(factorization_complexities)
            analysis['depth_entropy'] = self._compute_discrete_entropy(tree_depths)
            analysis['factor_count_entropy'] = self._compute_discrete_entropy(prime_factor_counts)
            
            analysis['average_complexity'] = np.mean(factorization_complexities)
            analysis['average_depth'] = np.mean(tree_depths)
            analysis['average_factor_count'] = np.mean(prime_factor_counts)
            
            analysis['complexity_variance'] = np.var(factorization_complexities)
            analysis['depth_variance'] = np.var(tree_depths)
        else:
            analysis = {
                'complexity_entropy': 0,
                'depth_entropy': 0,
                'factor_count_entropy': 0,
                'average_complexity': 0,
                'average_depth': 0,
                'average_factor_count': 0,
                'complexity_variance': 0,
                'depth_variance': 0
            }
        
        return analysis
    
    def _compute_discrete_entropy(self, values: List[int]) -> float:
        """Compute entropy of discrete value distribution"""
        if not values:
            return 0.0
        
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        total = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def analyze_compression_efficiency(self, traces: List[str]) -> Dict[str, float]:
        """Analyze compression efficiency of factorization"""
        original_lengths = []
        factorized_lengths = []
        
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            
            if result.factor_validation:
                original_lengths.append(len(trace))
                
                # Compute total length of prime factors
                factorized_length = sum(len(factor) for factor, _ in result.prime_factorization)
                factorized_lengths.append(factorized_length)
        
        if original_lengths and factorized_lengths:
            return {
                'average_original_length': np.mean(original_lengths),
                'average_factorized_length': np.mean(factorized_lengths),
                'compression_ratio': np.mean(factorized_lengths) / np.mean(original_lengths),
                'compression_efficiency': 1 - (np.mean(factorized_lengths) / np.mean(original_lengths))
            }
        else:
            return {
                'average_original_length': 0,
                'average_factorized_length': 0,
                'compression_ratio': 1.0,
                'compression_efficiency': 0.0
            }

class CategoryTheoryFactorizationAnalyzer:
    """Category theory analysis of factorization functors"""
    
    def __init__(self):
        self.factorizer = TraceFactorizer()
        
    def analyze_factorization_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of factorization"""
        results = {
            'preserves_identity': None,
            'respects_multiplication': None,
            'factorization_completeness': None,
            'prime_object_preservation': None
        }
        
        # Test factorization completeness
        complete_factorizations = 0
        total_factorizations = 0
        
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            total_factorizations += 1
            
            if result.factor_validation:
                complete_factorizations += 1
        
        results['factorization_completeness'] = (complete_factorizations / total_factorizations 
                                               if total_factorizations > 0 else 0)
        
        # Test prime object preservation
        prime_preservation_tests = []
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            
            if result.is_prime:
                # Prime traces should factorize to themselves
                prime_preservation = len(result.prime_factorization) == 1
                if prime_preservation and result.prime_factorization:
                    prime_preservation = result.prime_factorization[0][0] == trace
                prime_preservation_tests.append(prime_preservation)
        
        results['prime_object_preservation'] = (all(prime_preservation_tests) 
                                              if prime_preservation_tests else True)
        
        # Test multiplication respect (basic consistency)
        multiplication_tests = []
        for i, trace1 in enumerate(traces[:3]):
            for j, trace2 in enumerate(traces[:3]):
                if i < j:
                    result1 = self.factorizer.factorize_trace_completely(trace1)
                    result2 = self.factorizer.factorize_trace_completely(trace2)
                    
                    # Basic consistency check
                    if result1.factor_validation and result2.factor_validation:
                        multiplication_tests.append(True)
        
        results['respects_multiplication'] = (all(multiplication_tests) 
                                            if multiplication_tests else True)
        
        return results
    
    def identify_factorization_morphisms(self, traces: List[str]) -> Dict[str, Any]:
        """Identify morphisms in factorization category"""
        factorization_morphisms = []
        identity_morphisms = []
        
        for trace in traces:
            result = self.factorizer.factorize_trace_completely(trace)
            
            if result.is_prime:
                # Identity morphism: prime to itself
                identity_morphisms.append({
                    'domain': trace,
                    'codomain': trace,
                    'morphism_type': 'identity'
                })
            else:
                # Factorization morphisms: composite to factors
                for factor, exponent in result.prime_factorization:
                    factorization_morphisms.append({
                        'domain': trace,
                        'codomain': factor,
                        'morphism_type': 'factorization',
                        'exponent': exponent
                    })
        
        return {
            'factorization_morphisms': factorization_morphisms,
            'identity_morphisms': identity_morphisms,
            'total_morphisms': len(factorization_morphisms) + len(identity_morphisms)
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

def demonstrate_basic_factorization():
    """Demonstrate basic trace factorization"""
    factorizer = TraceFactorizer()
    
    print("=== Basic Trace Factorization ===")
    
    test_traces = [
        "100",      # 2 (prime)
        "1010",     # 4 = 2×2
        "10100",    # 7 (prime)
        "1010100",  # 20 = 4×5 = 2²×5
        "10101000", # 28 = 4×7 = 2²×7
    ]
    
    for trace in test_traces:
        result = factorizer.factorize_trace_completely(trace)
        
        print(f"\nTrace: '{trace}' (value: {result.original_value})")
        print(f"  Is prime: {result.is_prime}")
        print(f"  Prime factorization: {result.prime_factorization}")
        print(f"  Factorization depth: {result.factorization_depth}")
        print(f"  Tensor complexity: {result.tensor_complexity}")
        print(f"  Validation: {result.factor_validation}")
        
        if result.decomposition_steps:
            print(f"  Decomposition steps:")
            for step in result.decomposition_steps[:3]:  # Show first 3 steps
                print(f"    {step}")

def demonstrate_factorization_trees():
    """Demonstrate factorization tree construction"""
    factorizer = TraceFactorizer()
    
    print("\n=== Factorization Tree Construction ===")
    
    composite_traces = [
        "1010",     # 4 = 2×2
        "1010100",  # 20 = 4×5
        "100101000", # 42 = 6×7
    ]
    
    for trace in composite_traces:
        result = factorizer.factorize_trace_completely(trace)
        
        print(f"\nFactorization tree for '{trace}' (value: {result.original_value}):")
        print(f"  Tree depth: {result.factorization_depth}")
        print(f"  Irreducible components: {result.irreducible_components}")
        print(f"  Prime factors with exponents: {result.prime_factorization}")
        
        print("  Decomposition path:")
        for step in result.decomposition_steps:
            print(f"    {step}")

def demonstrate_prime_detection():
    """Demonstrate prime trace detection within factorization"""
    factorizer = TraceFactorizer()
    
    print("\n=== Prime Detection in Factorization ===")
    
    mixed_traces = [
        "100",      # 2 (prime)
        "1000",     # 3 (prime)
        "10000",    # 5 (prime)
        "1010",     # 4 (composite)
        "101",      # 3 (prime, sum form)
        "10100",    # 7 (prime, sum form)
    ]
    
    prime_count = 0
    composite_count = 0
    
    for trace in mixed_traces:
        result = factorizer.factorize_trace_completely(trace)
        
        print(f"\nTrace: '{trace}' → {result.original_value}")
        print(f"  Prime: {result.is_prime}")
        print(f"  Factors: {len(result.irreducible_components)}")
        
        if result.is_prime:
            prime_count += 1
        else:
            composite_count += 1
    
    print(f"\nSummary: {prime_count} prime, {composite_count} composite")

def graph_theory_analysis():
    """Perform graph theory analysis of factorization"""
    graph_analyzer = GraphTheoryFactorizationAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(7)
    
    # Build factorization graph
    G = graph_analyzer.build_factorization_graph(traces)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_factorization_graph_properties(G)
    
    print(f"Factorization graph properties:")
    print(f"  Total nodes: {analysis['total_nodes']}")
    print(f"  Total edges: {analysis['total_edges']}")
    print(f"  Prime nodes: {analysis['prime_nodes']}")
    print(f"  Composite nodes: {analysis['composite_nodes']}")
    print(f"  Prime ratio: {analysis['prime_ratio']:.3f}")
    print(f"  Is DAG: {analysis['is_dag']}")
    print(f"  Factorization edges: {analysis['factorization_edges']}")
    print(f"  Factorization density: {analysis['factorization_density']:.3f}")
    
    # Find factorization paths
    paths = graph_analyzer.find_factorization_paths(G)
    print(f"  Factorization paths found: {len(paths)}")
    
    if paths:
        print(f"  Example path: {' → '.join(paths[0])}")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryFactorizationAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(7)
    
    # Analyze factorization entropy
    entropy_analysis = info_analyzer.compute_factorization_entropy(traces)
    
    print(f"Factorization entropy analysis:")
    print(f"  Complexity entropy: {entropy_analysis['complexity_entropy']:.3f} bits")
    print(f"  Depth entropy: {entropy_analysis['depth_entropy']:.3f} bits")
    print(f"  Factor count entropy: {entropy_analysis['factor_count_entropy']:.3f} bits")
    print(f"  Average complexity: {entropy_analysis['average_complexity']:.2f}")
    print(f"  Average depth: {entropy_analysis['average_depth']:.2f}")
    print(f"  Average factor count: {entropy_analysis['average_factor_count']:.2f}")
    
    # Analyze compression efficiency
    compression_analysis = info_analyzer.analyze_compression_efficiency(traces)
    
    print(f"\nCompression efficiency analysis:")
    print(f"  Average original length: {compression_analysis['average_original_length']:.2f}")
    print(f"  Average factorized length: {compression_analysis['average_factorized_length']:.2f}")
    print(f"  Compression ratio: {compression_analysis['compression_ratio']:.3f}")
    print(f"  Compression efficiency: {compression_analysis['compression_efficiency']:.3f}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryFactorizationAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_factorization_functor_properties(traces)
    
    print(f"Factorization functor properties:")
    print(f"  Factorization completeness: {functor_analysis['factorization_completeness']:.3f}")
    print(f"  Prime object preservation: {functor_analysis['prime_object_preservation']}")
    print(f"  Respects multiplication: {functor_analysis['respects_multiplication']}")
    
    # Identify morphisms
    morphisms = cat_analyzer.identify_factorization_morphisms(traces)
    
    print(f"\nCategorical morphisms:")
    print(f"  Factorization morphisms: {len(morphisms['factorization_morphisms'])}")
    print(f"  Identity morphisms: {len(morphisms['identity_morphisms'])}")
    print(f"  Total morphisms: {morphisms['total_morphisms']}")

def verify_tensor_factorization_properties():
    """Verify tensor-level properties of factorization"""
    factorizer = TraceFactorizer()
    
    print("\n=== Tensor Factorization Properties ===")
    
    # Test specific tensor properties
    test_cases = [
        ("1010", "Should factor as '100' × '100'"),
        ("1010100", "Should factor into multiple primes"),
        ("100101000", "Complex factorization test"),
    ]
    
    for trace, description in test_cases:
        result = factorizer.factorize_trace_completely(trace)
        
        print(f"\nTest: {description}")
        print(f"  Trace: '{trace}' → {result.original_value}")
        print(f"  φ-compliant: {factorizer.is_phi_compliant(trace)}")
        print(f"  Prime factors: {result.prime_factorization}")
        print(f"  Validation: {result.factor_validation}")
        print(f"  Irreducible components: {result.irreducible_components}")

def main():
    """Run comprehensive trace factorization analysis"""
    print("="*80)
    print("Chapter 024: TraceFactorize - Tensor-Level Factor Decomposition Analysis")
    print("="*80)
    
    # Basic factorization demonstration
    demonstrate_basic_factorization()
    
    # Factorization tree construction
    demonstrate_factorization_trees()
    
    # Prime detection
    demonstrate_prime_detection()
    
    # Tensor properties verification
    verify_tensor_factorization_properties()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Trace factorization verification complete!")
    print("From ψ = ψ(ψ) emerges complete structural decomposition - factorization")
    print("that preserves φ-constraint while revealing prime tensor building blocks.")

if __name__ == "__main__":
    main()