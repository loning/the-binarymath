#!/usr/bin/env python3
"""
Chapter 017: FibEncode - φ-Safe Trace Construction from Individual Fibonacci Components

Verification program demonstrating how to encode Fibonacci components into traces
while maintaining the φ-constraint (no consecutive 11s).

From ψ = ψ(ψ), we derive the safe construction principles for trace arithmetic.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class FibonacciEncoder:
    """Encode Fibonacci numbers as φ-safe traces"""
    
    def __init__(self):
        self.fib_cache = {1: 1, 2: 2}
        self._precompute_fibs(50)
        
    def _precompute_fibs(self, max_n: int):
        """Precompute Fibonacci numbers"""
        for i in range(3, max_n + 1):
            self.fib_cache[i] = self.fib_cache[i-1] + self.fib_cache[i-2]
            
    def get_fib(self, n: int) -> int:
        """Get nth Fibonacci number (1-indexed)"""
        if n in self.fib_cache:
            return self.fib_cache[n]
        # Compute if needed
        self.fib_cache[n] = self.get_fib(n-1) + self.get_fib(n-2)
        return self.fib_cache[n]
    
    def encode_single_fib(self, n: int) -> str:
        """Encode single Fibonacci F_n as trace"""
        if n <= 0:
            return "0"
        
        # F_n is represented by a 1 at position n-1
        trace = ['0'] * n
        trace[n-1] = '1'
        return ''.join(reversed(trace))  # LSB first
    
    def encode_fib_list(self, indices: List[int]) -> str:
        """Encode list of Fibonacci indices as combined trace"""
        if not indices:
            return "0"
            
        max_idx = max(indices)
        trace = ['0'] * max_idx
        
        for idx in indices:
            trace[idx-1] = '1'
            
        return ''.join(reversed(trace))
    
    def verify_no_consecutive(self, indices: List[int]) -> bool:
        """Verify that Fibonacci indices are non-consecutive"""
        sorted_indices = sorted(indices)
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i+1] - sorted_indices[i] == 1:
                return False
        return True
    
    def safe_combine_traces(self, trace1: str, trace2: str) -> Optional[str]:
        """Safely combine two traces, returning None if would create 11"""
        # Align traces to same length
        max_len = max(len(trace1), len(trace2))
        t1 = trace1.ljust(max_len, '0')
        t2 = trace2.ljust(max_len, '0')
        
        # Simple OR combination
        result = []
        for i in range(max_len):
            bit = '1' if t1[i] == '1' or t2[i] == '1' else '0'
            result.append(bit)
            
        combined = ''.join(result)
        
        # Final verification
        if '11' in combined:
            return None
            
        return combined

class TraceComponentAnalyzer:
    """Analyze structure of Fibonacci trace components"""
    
    def __init__(self):
        self.encoder = FibonacciEncoder()
        
    def analyze_component_patterns(self, max_fib: int = 20):
        """Analyze patterns in individual Fibonacci traces"""
        patterns = {
            'trace_lengths': [],
            'one_positions': [],
            'spacing_patterns': []
        }
        
        for n in range(1, max_fib + 1):
            trace = self.encoder.encode_single_fib(n)
            patterns['trace_lengths'].append(len(trace.rstrip('0')))
            
            # Find position of the single 1
            one_pos = trace.find('1')
            patterns['one_positions'].append(one_pos)
            
            # Spacing from end
            patterns['spacing_patterns'].append(len(trace) - one_pos - 1)
            
        return patterns
    
    def analyze_combination_safety(self, max_fib: int = 10):
        """Analyze which Fibonacci pairs can be safely combined"""
        safety_matrix = np.zeros((max_fib, max_fib), dtype=int)
        
        for i in range(1, max_fib + 1):
            for j in range(1, max_fib + 1):
                if abs(i - j) > 1:  # Non-consecutive indices
                    trace1 = self.encoder.encode_single_fib(i)
                    trace2 = self.encoder.encode_single_fib(j)
                    combined = self.encoder.safe_combine_traces(trace1, trace2)
                    if combined is not None:
                        safety_matrix[i-1, j-1] = 1
                        
        return safety_matrix

class GraphTheoryFibAnalyzer:
    """Graph theory analysis of Fibonacci encoding"""
    
    def __init__(self):
        self.encoder = FibonacciEncoder()
        
    def build_combination_graph(self, max_fib: int = 12) -> nx.Graph:
        """Build graph of safe Fibonacci combinations"""
        G = nx.Graph()
        
        # Add nodes for each Fibonacci
        for i in range(1, max_fib + 1):
            G.add_node(i, value=self.encoder.get_fib(i))
            
        # Add edges for safe combinations
        for i in range(1, max_fib + 1):
            for j in range(i + 2, max_fib + 1):  # Skip consecutive
                # Verify combination is safe
                trace1 = self.encoder.encode_single_fib(i)
                trace2 = self.encoder.encode_single_fib(j)
                if self.encoder.safe_combine_traces(trace1, trace2):
                    G.add_edge(i, j)
                    
        return G
    
    def analyze_encoding_paths(self, G: nx.Graph, target_value: int) -> List[List[int]]:
        """Find all ways to encode a target value using graph paths"""
        paths = []
        nodes = list(G.nodes())
        
        # Try all subsets (simplified - real implementation would be more efficient)
        from itertools import combinations
        
        for r in range(1, min(len(nodes), 5) + 1):
            for subset in combinations(nodes, r):
                # Check if subset is independent (no edges between them)
                is_independent = True
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        if G.has_edge(subset[i], subset[j]):
                            is_independent = False
                            break
                    if not is_independent:
                        break
                        
                if is_independent:
                    # Check if sum equals target
                    total = sum(self.encoder.get_fib(n) for n in subset)
                    if total == target_value:
                        paths.append(list(subset))
                        
        return paths

class InformationTheoryFibAnalyzer:
    """Information theory analysis of Fibonacci encoding"""
    
    def __init__(self):
        self.encoder = FibonacciEncoder()
        
    def compute_encoding_density(self, max_n: int = 100) -> Dict:
        """Compute information density of Fibonacci encodings"""
        results = {
            'values': [],
            'trace_lengths': [],
            'effective_lengths': [],
            'densities': []
        }
        
        for n in range(1, max_n + 1):
            # Get Zeckendorf decomposition (simplified)
            remaining = n
            indices = []
            k = 20  # Start from a high Fibonacci
            
            while remaining > 0 and k >= 1:
                fib_k = self.encoder.get_fib(k)
                if fib_k <= remaining:
                    indices.append(k)
                    remaining -= fib_k
                    k -= 2  # Skip next for non-consecutive
                else:
                    k -= 1
                    
            if indices:
                trace = self.encoder.encode_fib_list(indices)
                results['values'].append(n)
                results['trace_lengths'].append(len(trace))
                results['effective_lengths'].append(len(trace.rstrip('0')))
                
                # Information density = log2(value) / trace_length
                if len(trace) > 0:
                    density = np.log2(n) / len(trace.rstrip('0'))
                    results['densities'].append(density)
                    
        return results
    
    def analyze_component_entropy(self, max_fib: int = 15) -> float:
        """Analyze entropy of Fibonacci component distribution"""
        # Count how often each Fibonacci appears in decompositions
        fib_counts = defaultdict(int)
        total_fibs = 0
        
        for n in range(1, 100):  # Analyze first 100 numbers
            # Get decomposition
            remaining = n
            k = 20
            
            while remaining > 0 and k >= 1:
                fib_k = self.encoder.get_fib(k)
                if fib_k <= remaining:
                    fib_counts[k] += 1
                    total_fibs += 1
                    remaining -= fib_k
                    k -= 2
                else:
                    k -= 1
                    
        # Compute entropy
        entropy = 0.0
        for count in fib_counts.values():
            if count > 0:
                p = count / total_fibs
                entropy -= p * np.log2(p)
                
        return entropy

class CategoryTheoryFibAnalyzer:
    """Category theory analysis of Fibonacci encoding"""
    
    def __init__(self):
        self.encoder = FibonacciEncoder()
        
    def verify_functor_properties(self) -> Dict:
        """Verify that Fibonacci encoding forms a functor"""
        results = {
            'preserves_identity': True,
            'preserves_composition': True,
            'examples': []
        }
        
        # Identity: F_0 (if defined) or empty set maps to empty trace
        empty_trace = self.encoder.encode_fib_list([])
        if empty_trace != "0":
            results['preserves_identity'] = False
            
        # Composition: encoding preserves non-overlapping union
        test_cases = [
            ([1, 3], [5, 7]),
            ([2, 4], [6, 8]),
            ([1, 4], [7, 10])
        ]
        
        for set1, set2 in test_cases:
            # Encode separately
            trace1 = self.encoder.encode_fib_list(set1)
            trace2 = self.encoder.encode_fib_list(set2)
            
            # Encode union
            union_trace = self.encoder.encode_fib_list(set1 + set2)
            
            # Try to combine separate traces
            combined = self.encoder.safe_combine_traces(trace1, trace2)
            
            if combined != union_trace:
                results['preserves_composition'] = False
                results['examples'].append({
                    'set1': set1,
                    'set2': set2,
                    'expected': union_trace,
                    'got': combined
                })
                
        return results
    
    def analyze_morphisms(self) -> Dict:
        """Analyze morphism structure in Fibonacci encoding"""
        morphisms = {
            'inclusion_morphisms': [],
            'disjoint_morphisms': [],
            'composition_examples': []
        }
        
        # Inclusion morphisms: single Fib → multiple Fibs
        for i in range(1, 10):
            for j in range(i+2, 12):  # Non-consecutive
                morphisms['inclusion_morphisms'].append({
                    'from': [i],
                    'to': [i, j],
                    'type': 'inclusion'
                })
                
        # Disjoint morphisms: separate components
        morphisms['disjoint_morphisms'] = [
            {'components': [[1, 3], [5, 7]], 'type': 'disjoint_union'},
            {'components': [[2, 4], [6, 8]], 'type': 'disjoint_union'}
        ]
        
        return morphisms

def demonstrate_basic_encoding():
    """Demonstrate basic Fibonacci encoding"""
    encoder = FibonacciEncoder()
    
    print("=== Basic Fibonacci Encoding ===")
    print(f"{'Fib Index':<10} {'Value':<10} {'Trace':<20} {'Verification':<15}")
    print("-" * 60)
    
    for n in range(1, 11):
        fib_val = encoder.get_fib(n)
        trace = encoder.encode_single_fib(n)
        has_11 = '11' in trace
        
        print(f"F_{n:<8} {fib_val:<10} {trace:<20} {'✗ Has 11!' if has_11 else '✓ Valid'}")

def demonstrate_safe_combination():
    """Demonstrate safe trace combination"""
    encoder = FibonacciEncoder()
    
    print("\n=== Safe Trace Combination ===")
    
    test_cases = [
        ([1, 3], "Non-consecutive Fibonacci"),
        ([2, 5], "Non-consecutive Fibonacci"),
        ([1, 2], "Consecutive - should handle safely"),
        ([3, 5, 7], "Multiple non-consecutive"),
        ([1, 3, 6, 8], "Larger set")
    ]
    
    for indices, description in test_cases:
        print(f"\nCombining F_{indices} ({description}):")
        
        # Check if indices are valid
        if encoder.verify_no_consecutive(indices):
            trace = encoder.encode_fib_list(indices)
            value = sum(encoder.get_fib(i) for i in indices)
            print(f"  Value: {value}")
            print(f"  Trace: {trace}")
            print(f"  Valid: {'✓' if '11' not in trace else '✗'}")
        else:
            print("  ✗ Contains consecutive Fibonacci indices!")

def analyze_component_structure():
    """Analyze structure of Fibonacci components"""
    analyzer = TraceComponentAnalyzer()
    
    print("\n=== Component Structure Analysis ===")
    
    patterns = analyzer.analyze_component_patterns(15)
    
    print("\nFibonacci component patterns:")
    print(f"{'Index':<8} {'Trace Length':<15} {'1-Position':<12} {'Spacing':<10}")
    print("-" * 50)
    
    for i in range(15):
        print(f"F_{i+1:<6} {patterns['trace_lengths'][i]:<15} "
              f"{patterns['one_positions'][i]:<12} {patterns['spacing_patterns'][i]:<10}")

def graph_analysis():
    """Perform graph theory analysis"""
    graph_analyzer = GraphTheoryFibAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Build combination graph
    G = graph_analyzer.build_combination_graph(10)
    
    print(f"\nCombination graph properties:")
    print(f"  Nodes (Fibonacci indices): {list(G.nodes())}")
    print(f"  Edges (safe combinations): {G.number_of_edges()}")
    print(f"  Graph density: {nx.density(G):.3f}")
    
    # Find cliques (sets of mutually compatible Fibonacci numbers)
    cliques = list(nx.find_cliques(G))
    print(f"\nMaximal cliques (fully compatible sets):")
    for clique in sorted(cliques, key=len, reverse=True)[:5]:
        values = [graph_analyzer.encoder.get_fib(i) for i in clique]
        print(f"  Indices {clique} → values {values} → sum {sum(values)}")
    
    # Find encoding paths for specific values
    print("\nEncoding paths for specific values:")
    for target in [20, 30, 50]:
        paths = graph_analyzer.analyze_encoding_paths(G, target)
        if paths:
            print(f"  {target}: {paths[0]} (and {len(paths)-1} more)" if len(paths) > 1 else f"  {target}: {paths[0]}")
        else:
            print(f"  {target}: No exact encoding found with small Fibonacci numbers")

def information_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryFibAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Encoding density
    density_results = info_analyzer.compute_encoding_density(50)
    
    avg_density = np.mean(density_results['densities']) if density_results['densities'] else 0
    print(f"\nEncoding density (first 50 numbers):")
    print(f"  Average density: {avg_density:.3f} bits/position")
    print(f"  Average trace length: {np.mean(density_results['effective_lengths']):.1f}")
    
    # Component entropy
    entropy = info_analyzer.analyze_component_entropy()
    print(f"\nComponent distribution entropy: {entropy:.3f} bits")
    print(f"  (Higher entropy = more uniform use of Fibonacci numbers)")

def category_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryFibAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Verify functor properties
    functor_results = cat_analyzer.verify_functor_properties()
    
    print(f"\nFunctor properties:")
    print(f"  Preserves identity: {functor_results['preserves_identity']}")
    print(f"  Preserves composition: {functor_results['preserves_composition']}")
    
    if functor_results['examples']:
        print(f"  Counterexamples: {functor_results['examples']}")
    
    # Analyze morphisms
    morphisms = cat_analyzer.analyze_morphisms()
    
    print(f"\nMorphism structure:")
    print(f"  Inclusion morphisms: {len(morphisms['inclusion_morphisms'])}")
    print(f"  Example: {morphisms['inclusion_morphisms'][0] if morphisms['inclusion_morphisms'] else 'None'}")

def visualize_safety_matrix():
    """Visualize which Fibonacci pairs can be safely combined"""
    analyzer = TraceComponentAnalyzer()
    
    print("\n=== Combination Safety Matrix ===")
    
    safety_matrix = analyzer.analyze_combination_safety(10)
    
    print("\nSafety matrix (1 = safe to combine, 0 = unsafe):")
    print("    ", end="")
    for j in range(1, 11):
        print(f"F{j:<2}", end=" ")
    print()
    
    for i in range(1, 11):
        print(f"F{i:<2}: ", end="")
        for j in range(1, 11):
            print(f"{safety_matrix[i-1, j-1]:<3}", end="")
        print()
    
    print("\nNote: Diagonal and adjacent indices are always 0 (unsafe)")

def demonstrate_encoding_algorithm():
    """Demonstrate complete encoding algorithm"""
    encoder = FibonacciEncoder()
    
    print("\n=== Complete Encoding Algorithm ===")
    
    def encode_number(n: int) -> Tuple[List[int], str]:
        """Encode a number using Zeckendorf decomposition"""
        remaining = n
        indices = []
        k = 20  # Start from high index
        
        while remaining > 0 and k >= 1:
            fib_k = encoder.get_fib(k)
            if fib_k <= remaining:
                indices.append(k)
                remaining -= fib_k
                k -= 2  # Skip next for non-consecutive
            else:
                k -= 1
                
        return indices, encoder.encode_fib_list(indices)
    
    print(f"{'Number':<10} {'Fibonacci Sum':<30} {'Trace':<20} {'Valid':<10}")
    print("-" * 75)
    
    for n in [10, 20, 33, 50, 89, 100]:
        indices, trace = encode_number(n)
        fib_sum = ' + '.join(f"F{i}" for i in indices)
        values = ' + '.join(str(encoder.get_fib(i)) for i in indices)
        
        print(f"{n:<10} {fib_sum:<30} {trace:<20} {'✓' if '11' not in trace else '✗'}")
        print(f"{'':<10} = {values}")
        print()

def main():
    """Run comprehensive Fibonacci encoding analysis"""
    print("="*80)
    print("Chapter 017: FibEncode - φ-Safe Trace Construction")
    print("="*80)
    
    # Basic encoding
    demonstrate_basic_encoding()
    
    # Safe combination
    demonstrate_safe_combination()
    
    # Component structure
    analyze_component_structure()
    
    # Graph analysis
    graph_analysis()
    
    # Information analysis
    information_analysis()
    
    # Category analysis
    category_analysis()
    
    # Safety matrix
    visualize_safety_matrix()
    
    # Complete algorithm
    demonstrate_encoding_algorithm()
    
    print("\n" + "="*80)
    print("Fibonacci encoding principles verified!")
    print("From ψ = ψ(ψ) emerges safe arithmetic in trace space.")

if __name__ == "__main__":
    main()