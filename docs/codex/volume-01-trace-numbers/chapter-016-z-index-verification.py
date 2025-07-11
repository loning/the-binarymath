#!/usr/bin/env python3
"""
Chapter 016: ZIndex - Zeckendorf Decomposition of Natural Numbers into Non-Overlapping Trace Seeds

Verification program demonstrating how every natural number uniquely decomposes into
non-consecutive Fibonacci numbers, establishing the mapping from ℕ to φ-traces.

From ψ = ψ(ψ), we derive the fundamental arithmetic structure of the golden base universe.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class FibonacciSystem:
    """Core Fibonacci number system for Zeckendorf decomposition"""
    
    def __init__(self, max_fib: int = 50):
        """Initialize Fibonacci sequence up to max_fib terms"""
        self.fib = [1, 2]  # Start with F1=1, F2=2 (ignoring F0)
        while len(self.fib) < max_fib:
            self.fib.append(self.fib[-1] + self.fib[-2])
            
    def get_fibonacci(self, k: int) -> int:
        """Get kth Fibonacci number (1-indexed)"""
        if k <= 0:
            return 0
        if k <= len(self.fib):
            return self.fib[k-1]
        # Extend if needed
        while len(self.fib) < k:
            self.fib.append(self.fib[-1] + self.fib[-2])
        return self.fib[k-1]

class ZeckendorfDecomposer:
    """Decompose natural numbers into Zeckendorf representation"""
    
    def __init__(self):
        self.fib_system = FibonacciSystem()
        
    def decompose(self, n: int) -> List[int]:
        """
        Decompose n into sum of non-consecutive Fibonacci numbers.
        Returns list of Fibonacci indices (1-based).
        """
        if n == 0:
            return []
            
        indices = []
        remaining = n
        
        # Greedy algorithm: always take largest possible Fibonacci
        k = 1
        while self.fib_system.get_fibonacci(k) <= n:
            k += 1
        k -= 1  # Back to largest Fib ≤ n
        
        while remaining > 0 and k >= 1:
            fib_k = self.fib_system.get_fibonacci(k)
            if fib_k <= remaining:
                indices.append(k)
                remaining -= fib_k
                k -= 2  # Skip next to ensure non-consecutive
            else:
                k -= 1
                
        return sorted(indices)
    
    def verify_decomposition(self, n: int, indices: List[int]) -> bool:
        """Verify that indices give valid Zeckendorf decomposition of n"""
        if not indices:
            return n == 0
            
        # Check non-consecutive
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] == 1:
                return False
                
        # Check sum
        total = sum(self.fib_system.get_fibonacci(k) for k in indices)
        return total == n
    
    def to_binary_trace(self, indices: List[int], min_length: Optional[int] = None) -> str:
        """
        Convert Zeckendorf indices to binary trace.
        Bit i is 1 if Fibonacci F_{i+1} is in the decomposition.
        """
        if not indices:
            return "0" if min_length is None else "0" * min_length
            
        max_index = max(indices)
        length = max(max_index, min_length or 0)
        
        trace = ['0'] * length
        for idx in indices:
            trace[idx-1] = '1'  # Convert to 0-based for trace
            
        # Reverse to get standard binary order (LSB first)
        return ''.join(reversed(trace))
    
    def from_binary_trace(self, trace: str) -> Tuple[List[int], int]:
        """
        Convert binary trace back to Zeckendorf indices and value.
        Returns (indices, value).
        """
        # Reverse to match index order
        reversed_trace = trace[::-1]
        
        indices = []
        value = 0
        
        for i, bit in enumerate(reversed_trace):
            if bit == '1':
                idx = i + 1  # Convert to 1-based Fibonacci index
                indices.append(idx)
                value += self.fib_system.get_fibonacci(idx)
                
        return sorted(indices), value

class ZIndexMapper:
    """Map between natural numbers and φ-traces via Zeckendorf"""
    
    def __init__(self):
        self.decomposer = ZeckendorfDecomposer()
        self.cache = {}  # Cache for efficiency
        
    def n_to_trace(self, n: int) -> str:
        """Map natural number to unique φ-trace"""
        if n in self.cache:
            return self.cache[n]
            
        indices = self.decomposer.decompose(n)
        trace = self.decomposer.to_binary_trace(indices)
        
        # Verify φ-constraint
        assert '11' not in trace, f"Invalid trace for n={n}: {trace}"
        
        self.cache[n] = trace
        return trace
    
    def trace_to_n(self, trace: str) -> int:
        """Map φ-trace back to natural number"""
        # First verify φ-constraint
        if '11' in trace:
            raise ValueError(f"Invalid φ-trace: {trace}")
            
        indices, value = self.decomposer.from_binary_trace(trace)
        return value
    
    def generate_trace_sequence(self, start: int, count: int) -> List[Tuple[int, str]]:
        """Generate sequence of (n, trace) pairs"""
        return [(n, self.n_to_trace(n)) for n in range(start, start + count)]

class GraphTheoryAnalyzer:
    """Analyze Zeckendorf decomposition from graph theory perspective"""
    
    def __init__(self):
        self.decomposer = ZeckendorfDecomposer()
        
    def build_decomposition_tree(self, max_n: int) -> nx.DiGraph:
        """Build tree showing decomposition paths"""
        G = nx.DiGraph()
        
        for n in range(1, max_n + 1):
            indices = self.decomposer.decompose(n)
            
            # Add node for n
            G.add_node(n, indices=indices)
            
            # Add edges showing decomposition steps
            if indices:
                # Remove largest Fibonacci to get parent
                parent_indices = indices[:-1]
                if parent_indices:
                    parent_n = sum(self.decomposer.fib_system.get_fibonacci(k) 
                                 for k in parent_indices)
                    G.add_edge(parent_n, n, 
                             fib_added=self.decomposer.fib_system.get_fibonacci(indices[-1]))
                else:
                    # Direct child of 0
                    G.add_edge(0, n, fib_added=n)
                    
        return G
    
    def analyze_tree_properties(self, G: nx.DiGraph) -> Dict:
        """Analyze properties of decomposition tree"""
        properties = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'max_degree': max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
            'height': nx.dag_longest_path_length(G) if G.number_of_nodes() > 0 else 0
        }
        
        # Find branching patterns
        branching = defaultdict(int)
        for node in G.nodes():
            out_degree = G.out_degree(node)
            branching[out_degree] += 1
            
        properties['branching_distribution'] = dict(branching)
        
        return properties

class InformationTheoryAnalyzer:
    """Analyze Zeckendorf encoding from information theory perspective"""
    
    def __init__(self):
        self.mapper = ZIndexMapper()
        
    def compute_encoding_efficiency(self, max_n: int) -> Dict:
        """Compute efficiency of Zeckendorf encoding"""
        results = {
            'n_values': [],
            'trace_lengths': [],
            'standard_binary_lengths': [],
            'compression_ratios': []
        }
        
        for n in range(1, max_n + 1):
            trace = self.mapper.n_to_trace(n)
            trace_len = len(trace.rstrip('0'))  # Effective length
            binary_len = n.bit_length()
            
            results['n_values'].append(n)
            results['trace_lengths'].append(trace_len)
            results['standard_binary_lengths'].append(binary_len)
            
            if binary_len > 0:
                ratio = trace_len / binary_len
                results['compression_ratios'].append(ratio)
            else:
                results['compression_ratios'].append(1.0)
                
        # Compute average metrics
        results['avg_trace_length'] = np.mean(results['trace_lengths'])
        results['avg_binary_length'] = np.mean(results['standard_binary_lengths'])
        results['avg_compression_ratio'] = np.mean(results['compression_ratios'])
        
        return results
    
    def compute_trace_entropy(self, traces: List[str]) -> float:
        """Compute entropy of trace distribution"""
        # Count bit frequencies
        bit_counts = {'0': 0, '1': 0}
        total_bits = 0
        
        for trace in traces:
            for bit in trace:
                bit_counts[bit] += 1
                total_bits += 1
                
        # Compute entropy
        entropy = 0.0
        for count in bit_counts.values():
            if count > 0:
                p = count / total_bits
                entropy -= p * np.log2(p)
                
        return entropy

class CategoryTheoryAnalyzer:
    """Analyze Zeckendorf from category theory perspective"""
    
    def __init__(self):
        self.mapper = ZIndexMapper()
        self.decomposer = ZeckendorfDecomposer()
        
    def verify_functor_properties(self, max_n: int) -> Dict:
        """Verify that Z-index mapping forms a functor"""
        results = {
            'preserves_identity': True,
            'preserves_composition': True,
            'is_injective': True,
            'counterexamples': []
        }
        
        # Check identity preservation: 0 maps to empty trace
        if self.mapper.n_to_trace(0) != "0":
            results['preserves_identity'] = False
            results['counterexamples'].append(('identity', 0, self.mapper.n_to_trace(0)))
            
        # Check injectivity
        trace_to_n_map = {}
        for n in range(max_n + 1):
            trace = self.mapper.n_to_trace(n)
            if trace in trace_to_n_map:
                results['is_injective'] = False
                results['counterexamples'].append(('injectivity', n, trace_to_n_map[trace]))
            trace_to_n_map[trace] = n
            
        # Check composition (simplified as additive structure)
        # For Zeckendorf, we need special addition that maintains non-consecutive property
        
        return results
    
    def analyze_morphisms(self, max_n: int) -> Dict:
        """Analyze morphism structure in Zeckendorf category"""
        morphisms = {
            'successor_preserving': [],
            'fibonacci_preserving': [],
            'structure_preserving': []
        }
        
        for n in range(max_n):
            # Successor morphism
            trace_n = self.mapper.n_to_trace(n)
            trace_n1 = self.mapper.n_to_trace(n + 1)
            
            # Check if successor has simple trace relationship
            if len(trace_n1) == len(trace_n) + 1:
                morphisms['successor_preserving'].append(n)
                
            # Check Fibonacci relationships
            indices_n = self.decomposer.decompose(n)
            if len(indices_n) == 1:  # Pure Fibonacci number
                morphisms['fibonacci_preserving'].append(n)
                
        return morphisms

def visualize_decomposition_examples():
    """Visualize Zeckendorf decomposition for example numbers"""
    decomposer = ZeckendorfDecomposer()
    mapper = ZIndexMapper()
    
    print("=== Zeckendorf Decomposition Examples ===")
    print(f"{'n':>4} | {'Fibonacci Sum':<30} | {'Indices':<15} | {'φ-Trace':<20}")
    print("-" * 75)
    
    examples = [0, 1, 2, 3, 4, 5, 8, 10, 13, 20, 30, 50, 100]
    
    for n in examples:
        indices = decomposer.decompose(n)
        fib_sum = []
        for idx in indices:
            fib_val = decomposer.fib_system.get_fibonacci(idx)
            fib_sum.append(f"F{idx}={fib_val}")
        
        fib_str = " + ".join(fib_sum) if fib_sum else "0"
        indices_str = str(indices) if indices else "[]"
        trace = mapper.n_to_trace(n)
        
        print(f"{n:4d} | {fib_str:<30} | {indices_str:<15} | {trace:<20}")

def demonstrate_uniqueness():
    """Demonstrate uniqueness of Zeckendorf decomposition"""
    print("\n=== Uniqueness of Zeckendorf Decomposition ===")
    
    decomposer = ZeckendorfDecomposer()
    
    # Show that greedy algorithm always gives unique result
    n = 50
    indices = decomposer.decompose(n)
    
    print(f"\nDecomposing n = {n}:")
    print(f"Greedy algorithm gives: {indices}")
    
    # Verify no other valid decomposition exists
    # Try some alternative decompositions
    alt_attempts = [
        [1, 3, 5, 7],  # Would sum to 1+3+8+21 = 33 ≠ 50
        [2, 4, 6],      # Would sum to 2+5+13 = 20 ≠ 50
        [1, 4, 6, 7]    # Contains consecutive (6,7), invalid
    ]
    
    print("\nVerifying alternatives fail:")
    for alt in alt_attempts:
        total = sum(decomposer.fib_system.get_fibonacci(k) for k in alt)
        consecutive = any(alt[i+1] - alt[i] == 1 for i in range(len(alt)-1))
        valid = total == n and not consecutive
        print(f"  {alt}: sum={total}, consecutive={consecutive}, valid={valid}")

def analyze_trace_patterns():
    """Analyze patterns in φ-traces"""
    print("\n=== φ-Trace Pattern Analysis ===")
    
    mapper = ZIndexMapper()
    
    # Generate traces for first 50 numbers
    traces = [mapper.n_to_trace(n) for n in range(50)]
    
    # Pattern statistics
    pattern_counts = defaultdict(int)
    
    for trace in traces:
        # Count patterns of length 2, 3
        for length in [2, 3]:
            for i in range(len(trace) - length + 1):
                pattern = trace[i:i+length]
                pattern_counts[pattern] += 1
                
    print("\nMost common patterns:")
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_patterns[:10]:
        print(f"  '{pattern}': {count} occurrences")
        
    # Verify no '11' pattern
    assert '11' not in pattern_counts, "Found forbidden '11' pattern!"
    print("\n✓ Verified: No '11' patterns found")

def graph_analysis():
    """Perform graph-theoretic analysis"""
    print("\n=== Graph Theory Analysis ===")
    
    analyzer = GraphTheoryAnalyzer()
    
    # Build decomposition tree
    G = analyzer.build_decomposition_tree(30)
    properties = analyzer.analyze_tree_properties(G)
    
    print(f"\nDecomposition tree properties (n ≤ 30):")
    print(f"  Nodes: {properties['nodes']}")
    print(f"  Edges: {properties['edges']}")
    print(f"  Max degree: {properties['max_degree']}")
    print(f"  Height: {properties['height']}")
    print(f"  Branching distribution: {properties['branching_distribution']}")
    
    # Visualize path structure
    print("\nExample paths in decomposition tree:")
    for target in [13, 20, 30]:
        if target in G:
            # Find path from 0 to target
            path = nx.shortest_path(G, 0, target)
            path_desc = []
            for i in range(len(path)-1):
                edge_data = G.get_edge_data(path[i], path[i+1])
                path_desc.append(f"+F{edge_data['fib_added']}")
            print(f"  Path to {target}: {' '.join(path_desc)}")

def information_analysis():
    """Perform information-theoretic analysis"""
    print("\n=== Information Theory Analysis ===")
    
    analyzer = InformationTheoryAnalyzer()
    
    # Analyze encoding efficiency
    results = analyzer.compute_encoding_efficiency(100)
    
    print(f"\nEncoding efficiency (n ≤ 100):")
    print(f"  Average trace length: {results['avg_trace_length']:.2f}")
    print(f"  Average binary length: {results['avg_binary_length']:.2f}")
    print(f"  Average expansion ratio: {results['avg_compression_ratio']:.2f}")
    
    # Compute entropy
    traces = [analyzer.mapper.n_to_trace(n) for n in range(1, 101)]
    entropy = analyzer.compute_trace_entropy(traces)
    print(f"  Trace bit entropy: {entropy:.3f} bits")
    print(f"  (Maximum possible: 1.000 bits)")

def category_analysis():
    """Perform category-theoretic analysis"""
    print("\n=== Category Theory Analysis ===")
    
    analyzer = CategoryTheoryAnalyzer()
    
    # Verify functor properties
    functor_results = analyzer.verify_functor_properties(50)
    
    print(f"\nFunctor properties:")
    print(f"  Preserves identity: {functor_results['preserves_identity']}")
    print(f"  Injective: {functor_results['is_injective']}")
    
    if functor_results['counterexamples']:
        print(f"  Counterexamples found: {functor_results['counterexamples']}")
        
    # Analyze morphisms
    morphisms = analyzer.analyze_morphisms(30)
    
    print(f"\nMorphism analysis:")
    print(f"  Fibonacci-preserving numbers: {morphisms['fibonacci_preserving'][:10]}...")
    print(f"  (These are pure Fibonacci numbers)")

def verify_correspondence():
    """Verify bijection between ℕ and φ-traces"""
    print("\n=== Verifying ℕ ↔ φ-Trace Correspondence ===")
    
    mapper = ZIndexMapper()
    
    # Test round-trip conversion
    test_range = 1000
    print(f"\nTesting round-trip conversion for n ∈ [0, {test_range})...")
    
    for n in range(test_range):
        trace = mapper.n_to_trace(n)
        n_recovered = mapper.trace_to_n(trace)
        
        if n != n_recovered:
            print(f"  ERROR: n={n} → trace='{trace}' → n'={n_recovered}")
            break
    else:
        print(f"  ✓ All {test_range} round-trip conversions successful")
        
    # Verify no duplicate traces
    print(f"\nVerifying uniqueness of traces...")
    trace_set = set()
    
    for n in range(test_range):
        trace = mapper.n_to_trace(n)
        if trace in trace_set:
            print(f"  ERROR: Duplicate trace '{trace}' for n={n}")
            break
        trace_set.add(trace)
    else:
        print(f"  ✓ All {test_range} traces are unique")

def main():
    """Run comprehensive Zeckendorf analysis"""
    print("="*80)
    print("Chapter 016: ZIndex - Zeckendorf Decomposition into φ-Traces")
    print("="*80)
    
    # Basic examples
    visualize_decomposition_examples()
    
    # Uniqueness
    demonstrate_uniqueness()
    
    # Pattern analysis
    analyze_trace_patterns()
    
    # Graph theory
    graph_analysis()
    
    # Information theory
    information_analysis()
    
    # Category theory
    category_analysis()
    
    # Verify correspondence
    verify_correspondence()
    
    print("\n" + "="*80)
    print("Zeckendorf decomposition verified!")
    print("From ψ = ψ(ψ) emerges the golden arithmetic of trace space.")

if __name__ == "__main__":
    main()