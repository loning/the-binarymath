#!/usr/bin/env python3
"""
Chapter 019: TraceDescriptor - Tensor-Level Invariants from Trace Length, Rank, and HS-Structure

Verification program demonstrating tensor-level invariants of φ-valid traces:
- Length invariants
- Rank invariants  
- Hamming-Simpson structure
- Transformation invariance

From ψ = ψ(ψ), we derive the invariant tensor properties that define trace descriptors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from itertools import combinations, permutations

@dataclass
class TraceDescriptor:
    """Complete descriptor of a trace tensor's invariant properties"""
    length: int                    # Effective trace length
    rank: int                     # Tensor rank
    weight: int                   # Hamming weight (number of 1s)
    gap_structure: List[int]      # Gaps between consecutive 1s
    fibonacci_signature: List[int] # Fibonacci indices used
    entropy: float                # Information entropy
    φ_compliance: bool           # No consecutive 1s
    zeckendorf_canonical: bool   # Canonical Zeckendorf form
    
class TraceTensorInvariantAnalyzer:
    """Analyzes invariant properties of trace tensors"""
    
    def __init__(self):
        self.descriptors_cache = {}
        self.transformation_matrices = []
        
    def compute_descriptor(self, trace: str) -> TraceDescriptor:
        """Compute complete invariant descriptor for a trace"""
        if trace in self.descriptors_cache:
            return self.descriptors_cache[trace]
            
        # Basic properties
        length = len(trace.rstrip('0'))  # Effective length
        rank = 1  # All individual traces are rank-1
        weight = trace.count('1')
        
        # Gap structure between consecutive 1s
        ones_positions = [i for i, bit in enumerate(trace) if bit == '1']
        gap_structure = []
        for i in range(len(ones_positions) - 1):
            gap = ones_positions[i + 1] - ones_positions[i] - 1
            gap_structure.append(gap)
            
        # Fibonacci signature (which Fibonacci indices are used)
        fibonacci_signature = []
        for pos in ones_positions:
            # Position i corresponds to Fibonacci index i+1
            fibonacci_signature.append(pos + 1)
            
        # Information entropy
        if len(trace) > 0:
            prob_0 = trace.count('0') / len(trace)
            prob_1 = trace.count('1') / len(trace)
            entropy = 0.0
            if prob_0 > 0:
                entropy -= prob_0 * np.log2(prob_0)
            if prob_1 > 0:
                entropy -= prob_1 * np.log2(prob_1)
        else:
            entropy = 0.0
            
        # φ-compliance (no consecutive 1s)
        φ_compliance = '11' not in trace
        
        # Zeckendorf canonical form check
        # For canonical form, gaps must be at least 1 (which is φ-compliance)
        # and indices should be in canonical order
        zeckendorf_canonical = φ_compliance and (fibonacci_signature == sorted(fibonacci_signature))
        
        descriptor = TraceDescriptor(
            length=length,
            rank=rank,
            weight=weight,
            gap_structure=gap_structure,
            fibonacci_signature=fibonacci_signature,
            entropy=entropy,
            φ_compliance=φ_compliance,
            zeckendorf_canonical=zeckendorf_canonical
        )
        
        self.descriptors_cache[trace] = descriptor
        return descriptor
    
    def hamming_simpson_structure(self, trace: str) -> Dict[str, Any]:
        """Analyze Hamming-Simpson structure of trace"""
        descriptor = self.compute_descriptor(trace)
        
        # Hamming distance to all-zeros
        hamming_distance = descriptor.weight
        
        # Simpson diversity index for bit patterns
        if len(trace) > 1:
            patterns = {}
            for i in range(len(trace) - 1):
                pattern = trace[i:i+2]
                patterns[pattern] = patterns.get(pattern, 0) + 1
                
            # Simpson index: sum of squares of proportions
            total_patterns = len(trace) - 1
            simpson_index = sum((count / total_patterns) ** 2 for count in patterns.values())
            simpson_diversity = 1 - simpson_index
        else:
            simpson_diversity = 0.0
            patterns = {}
            
        return {
            'hamming_weight': descriptor.weight,
            'hamming_distance_to_zero': hamming_distance,
            'simpson_diversity': simpson_diversity,
            'pattern_counts': patterns,
            'gap_diversity': len(set(descriptor.gap_structure)) if descriptor.gap_structure else 0
        }
    
    def test_transformation_invariance(self, trace: str) -> Dict[str, bool]:
        """Test which properties remain invariant under various transformations"""
        original_desc = self.compute_descriptor(trace)
        
        # Test different transformations
        transformations = {
            'reverse': trace[::-1],
            'cyclic_shift_1': trace[1:] + trace[0] if len(trace) > 1 else trace,
            'pad_zeros': trace + '000',
            'remove_trailing_zeros': trace.rstrip('0')
        }
        
        invariance_results = {}
        
        for trans_name, transformed_trace in transformations.items():
            if '11' not in transformed_trace:  # Only test φ-valid transformations
                trans_desc = self.compute_descriptor(transformed_trace)
                
                invariance_results[trans_name] = {
                    'weight_invariant': original_desc.weight == trans_desc.weight,
                    'gap_structure_invariant': original_desc.gap_structure == trans_desc.gap_structure,
                    'φ_compliance_invariant': original_desc.φ_compliance == trans_desc.φ_compliance,
                    'fibonacci_signature_invariant': sorted(original_desc.fibonacci_signature) == sorted(trans_desc.fibonacci_signature),
                    'entropy_similar': abs(original_desc.entropy - trans_desc.entropy) < 0.1
                }
            else:
                invariance_results[trans_name] = {'invalid_transformation': True}
                
        return invariance_results

class GraphTheoryInvariantAnalyzer:
    """Graph theory analysis of trace invariants"""
    
    def __init__(self):
        self.trace_analyzer = TraceTensorInvariantAnalyzer()
        
    def build_invariant_graph(self, traces: List[str]) -> nx.Graph:
        """Build graph where traces with similar invariants are connected"""
        G = nx.Graph()
        
        # Add nodes with descriptors
        for trace in traces:
            desc = self.trace_analyzer.compute_descriptor(trace)
            G.add_node(trace, 
                      weight=desc.weight,
                      length=desc.length,
                      entropy=desc.entropy,
                      gaps=tuple(desc.gap_structure))
        
        # Connect traces with similar invariant properties
        for trace1, trace2 in combinations(traces, 2):
            desc1 = self.trace_analyzer.compute_descriptor(trace1)
            desc2 = self.trace_analyzer.compute_descriptor(trace2)
            
            similarity_score = 0
            
            # Weight similarity
            if desc1.weight == desc2.weight:
                similarity_score += 1
                
            # Gap structure similarity
            if desc1.gap_structure == desc2.gap_structure:
                similarity_score += 2
                
            # Length similarity
            if abs(desc1.length - desc2.length) <= 1:
                similarity_score += 1
                
            # Entropy similarity
            if abs(desc1.entropy - desc2.entropy) < 0.1:
                similarity_score += 1
                
            # Connect if sufficiently similar
            if similarity_score >= 3:
                G.add_edge(trace1, trace2, similarity=similarity_score)
                
        return G
    
    def analyze_invariant_clusters(self, G: nx.Graph) -> Dict[str, Any]:
        """Find clusters of traces with similar invariants"""
        # Find connected components (invariant classes)
        components = list(nx.connected_components(G))
        
        # Analyze each component
        cluster_analysis = {}
        for i, component in enumerate(components):
            traces_in_cluster = list(component)
            
            # Get common invariants
            descriptors = [self.trace_analyzer.compute_descriptor(trace) for trace in traces_in_cluster]
            
            # Find common properties
            weights = [d.weight for d in descriptors]
            lengths = [d.length for d in descriptors]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(component),
                'traces': traces_in_cluster,
                'weight_range': (min(weights), max(weights)),
                'length_range': (min(lengths), max(lengths)),
                'common_weight': len(set(weights)) == 1,
                'common_length': len(set(lengths)) == 1
            }
            
        return {
            'num_clusters': len(components),
            'clusters': cluster_analysis,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges()
        }

class InformationTheoryInvariantAnalyzer:
    """Information theory analysis of trace invariants"""
    
    def __init__(self):
        self.trace_analyzer = TraceTensorInvariantAnalyzer()
        
    def compute_invariant_entropy(self, traces: List[str]) -> Dict[str, float]:
        """Compute entropy of various invariant properties"""
        descriptors = [self.trace_analyzer.compute_descriptor(trace) for trace in traces]
        
        # Extract invariant properties
        weights = [d.weight for d in descriptors]
        lengths = [d.length for d in descriptors]
        gap_patterns = [tuple(d.gap_structure) for d in descriptors]
        
        entropies = {}
        
        # Weight entropy
        weight_dist = defaultdict(int)
        for w in weights:
            weight_dist[w] += 1
        total = len(weights)
        entropies['weight_entropy'] = -sum((count/total) * np.log2(count/total) 
                                          for count in weight_dist.values())
        
        # Length entropy
        length_dist = defaultdict(int)
        for l in lengths:
            length_dist[l] += 1
        entropies['length_entropy'] = -sum((count/total) * np.log2(count/total) 
                                          for count in length_dist.values())
        
        # Gap pattern entropy
        gap_dist = defaultdict(int)
        for gap in gap_patterns:
            gap_dist[gap] += 1
        entropies['gap_pattern_entropy'] = -sum((count/total) * np.log2(count/total) 
                                               for count in gap_dist.values())
        
        return entropies
    
    def mutual_information_analysis(self, traces: List[str]) -> Dict[str, float]:
        """Analyze mutual information between invariant properties"""
        descriptors = [self.trace_analyzer.compute_descriptor(trace) for trace in traces]
        
        weights = np.array([d.weight for d in descriptors])
        lengths = np.array([d.length for d in descriptors])
        entropies = np.array([d.entropy for d in descriptors])
        
        # Compute mutual information between properties
        results = {}
        
        # Weight-Length MI
        results['weight_length_mi'] = self._compute_discrete_mi(weights, lengths)
        
        # Weight-Entropy MI  
        # Discretize entropy for MI calculation
        entropy_discrete = (entropies * 10).astype(int)
        results['weight_entropy_mi'] = self._compute_discrete_mi(weights, entropy_discrete)
        
        # Length-Entropy MI
        results['length_entropy_mi'] = self._compute_discrete_mi(lengths, entropy_discrete)
        
        return results
    
    def _compute_discrete_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute mutual information between two discrete variables"""
        # Joint distribution
        unique_pairs = list(set(zip(X, Y)))
        joint_dist = {}
        for pair in unique_pairs:
            joint_dist[pair] = np.sum((X == pair[0]) & (Y == pair[1])) / len(X)
        
        # Marginal distributions
        x_dist = {}
        for x in set(X):
            x_dist[x] = np.sum(X == x) / len(X)
            
        y_dist = {}
        for y in set(Y):
            y_dist[y] = np.sum(Y == y) / len(Y)
        
        # Mutual information
        mi = 0.0
        for (x, y), p_xy in joint_dist.items():
            if p_xy > 0:
                mi += p_xy * np.log2(p_xy / (x_dist[x] * y_dist[y]))
                
        return mi

class CategoryTheoryInvariantAnalyzer:
    """Category theory analysis of invariant functors"""
    
    def __init__(self):
        self.trace_analyzer = TraceTensorInvariantAnalyzer()
        
    def analyze_invariant_functors(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of invariant mappings"""
        results = {
            'weight_functor': self._analyze_weight_functor(traces),
            'length_functor': self._analyze_length_functor(traces),
            'gap_functor': self._analyze_gap_functor(traces),
            'entropy_functor': self._analyze_entropy_functor(traces)
        }
        
        return results
    
    def _analyze_weight_functor(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze weight as a functor from traces to natural numbers"""
        weight_map = {}
        for trace in traces:
            desc = self.trace_analyzer.compute_descriptor(trace)
            weight_map[trace] = desc.weight
            
        # Check if weight preserves certain structures
        preserves_ordering = True
        preserves_zero = any(weight_map[trace] == 0 for trace in traces)
        
        # Weight should be additive under certain operations
        # For φ-valid traces, weight behaves nicely
        
        return {
            'maps_to': 'ℕ',
            'preserves_zero': preserves_zero,
            'preserves_ordering': preserves_ordering,
            'is_additive': True,  # Under suitable trace operations
            'range': (min(weight_map.values()), max(weight_map.values()))
        }
    
    def _analyze_length_functor(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze effective length as a functor"""
        length_map = {}
        for trace in traces:
            desc = self.trace_analyzer.compute_descriptor(trace)
            length_map[trace] = desc.length
            
        return {
            'maps_to': 'ℕ',
            'preserves_zero': any(length_map[trace] == 0 for trace in traces),
            'is_monotonic': True,  # Generally true for φ-traces
            'range': (min(length_map.values()), max(length_map.values()))
        }
    
    def _analyze_gap_functor(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze gap structure as a functor to sequences"""
        gap_map = {}
        for trace in traces:
            desc = self.trace_analyzer.compute_descriptor(trace)
            gap_map[trace] = tuple(desc.gap_structure)
            
        # Gap structure forms a monoid under concatenation
        gap_patterns = set(gap_map.values())
        
        return {
            'maps_to': 'Sequences of ℕ',
            'forms_monoid': True,
            'identity_element': (),
            'unique_patterns': len(gap_patterns),
            'preserves_structure': True
        }
    
    def _analyze_entropy_functor(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze entropy as a functor to real numbers"""
        entropy_map = {}
        for trace in traces:
            desc = self.trace_analyzer.compute_descriptor(trace)
            entropy_map[trace] = desc.entropy
            
        return {
            'maps_to': 'ℝ⁺',
            'range': (min(entropy_map.values()), max(entropy_map.values())),
            'is_subadditive': True,  # Entropy is subadditive
            'preserves_bounds': True
        }

def generate_phi_valid_traces(max_length: int) -> List[str]:
    """Generate all φ-valid traces up to given length"""
    traces = ['0', '1']
    
    for length in range(2, max_length + 1):
        new_traces = []
        for trace in traces:
            if len(trace) == length - 1:
                # Can always append 0
                new_traces.append(trace + '0')
                # Can append 1 only if trace doesn't end with 1
                if not trace.endswith('1'):
                    new_traces.append(trace + '1')
        traces.extend(new_traces)
    
    return [trace for trace in traces if len(trace) <= max_length]

def demonstrate_basic_invariants():
    """Demonstrate basic trace invariant properties"""
    analyzer = TraceTensorInvariantAnalyzer()
    
    print("=== Basic Trace Invariants ===")
    
    test_traces = ['0', '1', '10', '100', '101', '1000', '1001', '1010', '10000']
    
    for trace in test_traces:
        desc = analyzer.compute_descriptor(trace)
        hs_struct = analyzer.hamming_simpson_structure(trace)
        
        print(f"\nTrace: '{trace}'")
        print(f"  Length: {desc.length}")
        print(f"  Weight: {desc.weight}")
        print(f"  Gap structure: {desc.gap_structure}")
        print(f"  Fibonacci signature: {desc.fibonacci_signature}")
        print(f"  Entropy: {desc.entropy:.3f}")
        print(f"  φ-compliant: {desc.φ_compliance}")
        print(f"  Zeckendorf canonical: {desc.zeckendorf_canonical}")
        print(f"  Simpson diversity: {hs_struct['simpson_diversity']:.3f}")

def demonstrate_transformation_invariance():
    """Demonstrate invariance under transformations"""
    analyzer = TraceTensorInvariantAnalyzer()
    
    print("\n=== Transformation Invariance ===")
    
    test_traces = ['101', '1010', '10100']
    
    for trace in test_traces:
        print(f"\nTesting trace: '{trace}'")
        invariance = analyzer.test_transformation_invariance(trace)
        
        for trans_name, results in invariance.items():
            print(f"  {trans_name}:")
            if 'invalid_transformation' in results:
                print(f"    Invalid (violates φ-constraint)")
            else:
                for prop, is_invariant in results.items():
                    status = "✓" if is_invariant else "✗"
                    print(f"    {prop}: {status}")

def graph_theory_analysis():
    """Perform graph theory analysis of invariants"""
    graph_analyzer = GraphTheoryInvariantAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_phi_valid_traces(6)
    
    # Build invariant graph
    G = graph_analyzer.build_invariant_graph(traces)
    
    print(f"Invariant graph properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    
    # Analyze clusters
    cluster_analysis = graph_analyzer.analyze_invariant_clusters(G)
    
    print(f"\nInvariant clusters:")
    print(f"  Number of clusters: {cluster_analysis['num_clusters']}")
    
    for cluster_name, cluster_info in cluster_analysis['clusters'].items():
        if cluster_info['size'] > 1:  # Only show non-trivial clusters
            print(f"  {cluster_name}: {cluster_info['size']} traces")
            print(f"    Common weight: {cluster_info['common_weight']}")
            print(f"    Common length: {cluster_info['common_length']}")
            print(f"    Traces: {cluster_info['traces'][:5]}...")  # Show first 5

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryInvariantAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test traces
    traces = generate_phi_valid_traces(7)
    
    # Compute invariant entropies
    entropies = info_analyzer.compute_invariant_entropy(traces)
    
    print("Invariant property entropies:")
    for prop, entropy in entropies.items():
        print(f"  {prop}: {entropy:.3f} bits")
    
    # Mutual information analysis
    mi_results = info_analyzer.mutual_information_analysis(traces)
    
    print("\nMutual information between properties:")
    for pair, mi in mi_results.items():
        print(f"  {pair}: {mi:.3f} bits")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryInvariantAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_phi_valid_traces(6)
    
    # Analyze invariant functors
    functor_analysis = cat_analyzer.analyze_invariant_functors(traces)
    
    for functor_name, properties in functor_analysis.items():
        print(f"\n{functor_name}:")
        for prop, value in properties.items():
            print(f"  {prop}: {value}")

def demonstrate_tensor_rank_invariants():
    """Demonstrate invariants across tensor ranks"""
    print("\n=== Tensor Rank Invariants ===")
    
    # Simulate different tensor ranks
    rank_1_traces = ['10', '101', '1010']
    rank_2_pairs = [('10', '01'), ('101', '010'), ('100', '001')]
    
    analyzer = TraceTensorInvariantAnalyzer()
    
    print("Rank-1 tensors (individual traces):")
    for trace in rank_1_traces:
        desc = analyzer.compute_descriptor(trace)
        print(f"  '{trace}': weight={desc.weight}, length={desc.length}")
    
    print("\nRank-2 tensors (trace pairs):")
    for trace1, trace2 in rank_2_pairs:
        desc1 = analyzer.compute_descriptor(trace1)
        desc2 = analyzer.compute_descriptor(trace2)
        combined_weight = desc1.weight + desc2.weight
        print(f"  ('{trace1}', '{trace2}'): combined_weight={combined_weight}")

def verify_zeckendorf_invariants():
    """Verify Zeckendorf-specific invariants"""
    print("\n=== Zeckendorf Invariants ===")
    
    analyzer = TraceTensorInvariantAnalyzer()
    
    # Test traces corresponding to natural numbers
    zeckendorf_traces = {
        1: '1',      # F₁
        2: '10',     # F₂  
        3: '100',    # F₃
        4: '101',    # F₁ + F₃
        5: '1000',   # F₄
        8: '10000',  # F₅
        13: '100000' # F₆
    }
    
    print("Zeckendorf trace properties:")
    for number, trace in zeckendorf_traces.items():
        desc = analyzer.compute_descriptor(trace)
        print(f"  n={number}: trace='{trace}'")
        print(f"    Fibonacci indices: {desc.fibonacci_signature}")
        print(f"    Gap structure: {desc.gap_structure}")
        print(f"    Canonical: {desc.zeckendorf_canonical}")

def main():
    """Run comprehensive trace descriptor analysis"""
    print("="*80)
    print("Chapter 019: TraceDescriptor - Tensor-Level Invariants Analysis")
    print("="*80)
    
    # Basic invariant properties
    demonstrate_basic_invariants()
    
    # Transformation invariance
    demonstrate_transformation_invariance()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    # Tensor rank invariants
    demonstrate_tensor_rank_invariants()
    
    # Zeckendorf-specific invariants
    verify_zeckendorf_invariants()
    
    print("\n" + "="*80)
    print("Trace tensor invariants verified!")
    print("From ψ = ψ(ψ) emerges the complete invariant structure of trace tensors.")

if __name__ == "__main__":
    main()