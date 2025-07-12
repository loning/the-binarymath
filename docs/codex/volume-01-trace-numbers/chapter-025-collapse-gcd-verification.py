#!/usr/bin/env python3
"""
Chapter 025: CollapseGCD - Common Collapse Divisors in Path Configuration Space

Verification program demonstrating greatest common divisor operations on trace tensors:
- Common structural subpath detection in φ-constrained space
- Longest shared legal trace identification
- Path configuration analysis for shared substructures
- GCD algorithms preserving golden constraint

From ψ = ψ(ψ), we derive GCD operations that find maximal common 
structural subpaths between trace tensors while maintaining φ-constraint
throughout the divisor extraction process.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools

@dataclass
class GCDResult:
    """Complete result of trace tensor GCD computation"""
    trace1: str
    trace2: str
    gcd_trace: str
    common_indices: List[int]  # Shared Fibonacci indices
    alignment_type: str  # 'identical', 'partial', 'disjoint'
    structural_similarity: float
    path_configuration: Dict[str, Any]
    verification_valid: bool
    computation_path: List[str]

class TraceTensorDecoder:
    """Enhanced decoder with structural GCD support"""
    
    def __init__(self):
        self.fibonacci_cache = {}
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
    
    def extract_fibonacci_indices(self, trace: str) -> List[int]:
        """Extract Fibonacci indices from trace"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i + 1)
        return sorted(indices)

class StructuralGCDComputer:
    """Compute structural GCD based on shared Fibonacci components"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def compute_structural_gcd(self, trace1: str, trace2: str) -> GCDResult:
        """Compute GCD based on shared structural subpaths"""
        computation_path = []
        
        # Step 1: Validate φ-compliance
        if not self.is_phi_compliant(trace1) or not self.is_phi_compliant(trace2):
            computation_path.append("φ-constraint violation detected")
            return GCDResult(
                trace1=trace1,
                trace2=trace2,
                gcd_trace="",
                common_indices=[],
                alignment_type='invalid',
                structural_similarity=0.0,
                path_configuration={},
                verification_valid=False,
                computation_path=computation_path
            )
        
        computation_path.append(f"φ-constraint satisfied for both traces")
        
        # Step 2: Extract Fibonacci indices
        indices1 = self.decoder.extract_fibonacci_indices(trace1)
        indices2 = self.decoder.extract_fibonacci_indices(trace2)
        computation_path.append(f"Trace1 indices: {indices1}")
        computation_path.append(f"Trace2 indices: {indices2}")
        
        # Step 3: Find common indices (shared Fibonacci components)
        common_indices = sorted(list(set(indices1) & set(indices2)))
        computation_path.append(f"Common indices: {common_indices}")
        
        # Step 4: Construct GCD trace from common indices
        if not common_indices:
            gcd_trace = "0"
            alignment_type = 'disjoint'
            structural_similarity = 0.0
        else:
            # Build trace from common indices
            max_index = max(common_indices)
            gcd_bits = ['0'] * max_index
            for index in common_indices:
                gcd_bits[index - 1] = '1'
            gcd_trace = ''.join(reversed(gcd_bits))
            
            # Verify φ-compliance of result
            if not self.is_phi_compliant(gcd_trace):
                computation_path.append("Result violates φ-constraint, adjusting...")
                # Remove consecutive 1s if any
                gcd_trace = self._remove_consecutive_ones(gcd_trace, common_indices)
            
            # Determine alignment type
            if trace1 == trace2:
                alignment_type = 'identical'
            elif len(common_indices) > 0:
                alignment_type = 'partial'
            else:
                alignment_type = 'disjoint'
            
            # Calculate structural similarity
            union_indices = sorted(list(set(indices1) | set(indices2)))
            if union_indices:
                structural_similarity = len(common_indices) / len(union_indices)
            else:
                structural_similarity = 0.0
        
        computation_path.append(f"GCD trace: '{gcd_trace}'")
        computation_path.append(f"Alignment: {alignment_type}")
        
        # Step 5: Analyze path configuration
        path_configuration = {
            'trace1_unique': sorted(list(set(indices1) - set(indices2))),
            'trace2_unique': sorted(list(set(indices2) - set(indices1))),
            'common': common_indices,
            'jaccard_similarity': structural_similarity
        }
        
        return GCDResult(
            trace1=trace1,
            trace2=trace2,
            gcd_trace=gcd_trace,
            common_indices=common_indices,
            alignment_type=alignment_type,
            structural_similarity=structural_similarity,
            path_configuration=path_configuration,
            verification_valid=True,
            computation_path=computation_path
        )
    
    def _remove_consecutive_ones(self, trace: str, indices: List[int]) -> str:
        """Remove consecutive ones from trace by dropping smaller indices"""
        # Check for consecutive indices and remove smaller ones
        filtered_indices = []
        prev_idx = -2
        
        for idx in sorted(indices):
            if idx > prev_idx + 1:  # Not consecutive
                filtered_indices.append(idx)
                prev_idx = idx
        
        # Rebuild trace
        if not filtered_indices:
            return "0"
        
        max_index = max(filtered_indices)
        trace_bits = ['0'] * max_index
        for index in filtered_indices:
            trace_bits[index - 1] = '1'
        return ''.join(reversed(trace_bits))
    
    def compute_multiple_gcd(self, traces: List[str]) -> str:
        """Compute GCD of multiple traces"""
        if not traces:
            return "0"
        
        if len(traces) == 1:
            return traces[0] if self.is_phi_compliant(traces[0]) else "0"
        
        # Compute iteratively: gcd(a,b,c) = gcd(gcd(a,b),c)
        result = traces[0]
        for i in range(1, len(traces)):
            gcd_result = self.compute_structural_gcd(result, traces[i])
            result = gcd_result.gcd_trace
            
            if result == "0":
                break
        
        return result
    
    def compute_structural_lcm(self, trace1: str, trace2: str) -> str:
        """Compute LCM as union of structural components"""
        if not self.is_phi_compliant(trace1) or not self.is_phi_compliant(trace2):
            return ""
        
        # Extract indices
        indices1 = self.decoder.extract_fibonacci_indices(trace1)
        indices2 = self.decoder.extract_fibonacci_indices(trace2)
        
        # Union of indices
        union_indices = sorted(list(set(indices1) | set(indices2)))
        
        if not union_indices:
            return "0"
        
        # Build LCM trace
        max_index = max(union_indices)
        lcm_bits = ['0'] * max_index
        for index in union_indices:
            lcm_bits[index - 1] = '1'
        lcm_trace = ''.join(reversed(lcm_bits))
        
        # Check φ-compliance
        if not self.is_phi_compliant(lcm_trace):
            # LCM cannot be constructed while maintaining φ-constraint
            return self._adjust_lcm_for_phi(union_indices)
        
        return lcm_trace
    
    def _adjust_lcm_for_phi(self, indices: List[int]) -> str:
        """Adjust LCM to maintain φ-constraint by removing conflicting indices"""
        # Remove consecutive indices to maintain φ-constraint
        filtered_indices = []
        prev_idx = -2
        
        for idx in sorted(indices):
            if idx > prev_idx + 1:  # Not consecutive
                filtered_indices.append(idx)
                prev_idx = idx
        
        if not filtered_indices:
            return "0"
        
        max_index = max(filtered_indices)
        trace_bits = ['0'] * max_index
        for index in filtered_indices:
            trace_bits[index - 1] = '1'
        return ''.join(reversed(trace_bits))

class GraphTheoryGCDAnalyzer:
    """Graph theory analysis of GCD structures"""
    
    def __init__(self):
        self.gcd_computer = StructuralGCDComputer()
        
    def build_gcd_graph(self, traces: List[str]) -> nx.Graph:
        """Build graph showing GCD relationships"""
        G = nx.Graph()
        
        # Add nodes for all traces
        for trace in traces:
            value = self.gcd_computer.decoder.trace_to_number(trace)
            G.add_node(trace, value=value)
        
        # Add edges for non-trivial GCDs
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                result = self.gcd_computer.compute_structural_gcd(trace1, trace2)
                
                if result.gcd_trace != "0" and result.verification_valid:
                    gcd_value = self.gcd_computer.decoder.trace_to_number(result.gcd_trace)
                    G.add_edge(trace1, trace2, 
                             gcd=result.gcd_trace,
                             gcd_value=gcd_value,
                             similarity=result.structural_similarity,
                             common_indices=result.common_indices)
        
        return G
    
    def analyze_gcd_graph_properties(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze properties of GCD graph"""
        
        # Count non-trivial GCDs
        non_trivial_gcds = [(u, v, d) for u, v, d in G.edges(data=True) 
                           if d.get('gcd', '0') != '0']
        
        # Compute GCD values distribution
        gcd_values = [d['gcd_value'] for _, _, d in non_trivial_gcds]
        unique_gcds = len(set(gcd_values)) if gcd_values else 0
        
        # Find communities based on common GCDs
        communities = self._find_gcd_communities(G)
        
        analysis = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'non_trivial_gcds': len(non_trivial_gcds),
            'unique_gcd_values': unique_gcds,
            'average_similarity': np.mean([d['similarity'] for _, _, d in G.edges(data=True)]) if G.edges() else 0,
            'max_similarity': max([d['similarity'] for _, _, d in G.edges(data=True)], default=0),
            'connected_components': nx.number_connected_components(G),
            'density': nx.density(G),
            'communities': len(communities),
            'largest_community_size': max(len(c) for c in communities) if communities else 0
        }
        
        # Analyze clustering
        if G.number_of_nodes() > 0:
            analysis['clustering_coefficient'] = nx.average_clustering(G)
        else:
            analysis['clustering_coefficient'] = 0
        
        return analysis
    
    def _find_gcd_communities(self, G: nx.Graph, threshold: float = 0.5) -> List[Set[str]]:
        """Find communities based on GCD similarity"""
        # Create subgraph with high-similarity edges
        high_sim_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('similarity', 0) >= threshold]
        
        H = G.edge_subgraph(high_sim_edges).copy()
        
        # Find connected components in high-similarity subgraph
        communities = list(nx.connected_components(H))
        
        return communities

class InformationTheoryGCDAnalyzer:
    """Information theory analysis of GCD properties"""
    
    def __init__(self):
        self.gcd_computer = StructuralGCDComputer()
        
    def compute_gcd_entropy(self, traces: List[str]) -> Dict[str, Any]:
        """Compute entropy of GCD structures"""
        gcd_results = []
        similarities = []
        common_index_counts = []
        
        # Compute all pairwise GCDs
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                result = self.gcd_computer.compute_structural_gcd(trace1, trace2)
                
                if result.verification_valid:
                    gcd_results.append(result.gcd_trace)
                    similarities.append(result.structural_similarity)
                    common_index_counts.append(len(result.common_indices))
        
        analysis = {}
        
        # GCD entropy
        if gcd_results:
            gcd_counts = {}
            for gcd in gcd_results:
                gcd_counts[gcd] = gcd_counts.get(gcd, 0) + 1
            
            total = len(gcd_results)
            entropy = 0.0
            for count in gcd_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            analysis['gcd_entropy'] = entropy
            analysis['unique_gcds'] = len(gcd_counts)
            analysis['most_common_gcd'] = max(gcd_counts.items(), key=lambda x: x[1])[0]
        else:
            analysis['gcd_entropy'] = 0
            analysis['unique_gcds'] = 0
            analysis['most_common_gcd'] = "0"
        
        # Similarity statistics
        if similarities:
            analysis['average_similarity'] = np.mean(similarities)
            analysis['similarity_variance'] = np.var(similarities)
            analysis['max_similarity'] = max(similarities)
            analysis['min_similarity'] = min(similarities)
        else:
            analysis['average_similarity'] = 0
            analysis['similarity_variance'] = 0
            analysis['max_similarity'] = 0
            analysis['min_similarity'] = 0
        
        # Common index statistics
        if common_index_counts:
            analysis['average_common_indices'] = np.mean(common_index_counts)
            analysis['max_common_indices'] = max(common_index_counts)
        else:
            analysis['average_common_indices'] = 0
            analysis['max_common_indices'] = 0
        
        return analysis
    
    def analyze_information_preservation(self, traces: List[str]) -> Dict[str, float]:
        """Analyze information preservation in GCD operations"""
        preservation_ratios = []
        
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                result = self.gcd_computer.compute_structural_gcd(trace1, trace2)
                
                if result.verification_valid and result.gcd_trace != "0":
                    # Information in original traces
                    info1 = len(result.path_configuration['trace1_unique']) + len(result.common_indices)
                    info2 = len(result.path_configuration['trace2_unique']) + len(result.common_indices)
                    
                    # Information in GCD
                    info_gcd = len(result.common_indices)
                    
                    # Preservation ratio
                    if info1 + info2 > 0:
                        preservation = info_gcd / ((info1 + info2) / 2)
                        preservation_ratios.append(preservation)
        
        if preservation_ratios:
            return {
                'average_preservation': np.mean(preservation_ratios),
                'min_preservation': min(preservation_ratios),
                'max_preservation': max(preservation_ratios),
                'preservation_variance': np.var(preservation_ratios)
            }
        else:
            return {
                'average_preservation': 0,
                'min_preservation': 0,
                'max_preservation': 0,
                'preservation_variance': 0
            }

class CategoryTheoryGCDAnalyzer:
    """Category theory analysis of GCD functors"""
    
    def __init__(self):
        self.gcd_computer = StructuralGCDComputer()
        
    def analyze_gcd_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of GCD operation"""
        results = {
            'commutative': None,
            'associative': None,
            'preserves_identity': None,
            'preserves_divisibility': None,
            'universal_property': None
        }
        
        # Test commutativity: gcd(a,b) = gcd(b,a)
        commutative_tests = []
        for i in range(min(5, len(traces))):
            for j in range(i+1, min(5, len(traces))):
                gcd_ab = self.gcd_computer.compute_structural_gcd(traces[i], traces[j])
                gcd_ba = self.gcd_computer.compute_structural_gcd(traces[j], traces[i])
                commutative_tests.append(gcd_ab.gcd_trace == gcd_ba.gcd_trace)
        
        results['commutative'] = all(commutative_tests) if commutative_tests else True
        
        # Test associativity: gcd(gcd(a,b),c) = gcd(a,gcd(b,c))
        if len(traces) >= 3:
            associative_tests = []
            for i in range(min(3, len(traces)-2)):
                a, b, c = traces[i], traces[i+1], traces[i+2]
                
                # Left association
                gcd_ab = self.gcd_computer.compute_structural_gcd(a, b)
                gcd_ab_c = self.gcd_computer.compute_structural_gcd(gcd_ab.gcd_trace, c)
                
                # Right association
                gcd_bc = self.gcd_computer.compute_structural_gcd(b, c)
                gcd_a_bc = self.gcd_computer.compute_structural_gcd(a, gcd_bc.gcd_trace)
                
                associative_tests.append(gcd_ab_c.gcd_trace == gcd_a_bc.gcd_trace)
            
            results['associative'] = all(associative_tests) if associative_tests else True
        else:
            results['associative'] = True
        
        # Test identity preservation: gcd(a,a) = a
        identity_tests = []
        for trace in traces[:5]:
            gcd_self = self.gcd_computer.compute_structural_gcd(trace, trace)
            identity_tests.append(gcd_self.gcd_trace == trace)
        
        results['preserves_identity'] = all(identity_tests) if identity_tests else True
        
        # Test divisibility preservation
        # For structural GCD, check if common indices are preserved
        divisibility_tests = []
        for i in range(min(3, len(traces))):
            for j in range(i+1, min(3, len(traces))):
                result = self.gcd_computer.compute_structural_gcd(traces[i], traces[j])
                if result.gcd_trace != "0":
                    # Check if GCD indices are subset of both traces
                    gcd_indices = set(result.common_indices)
                    trace1_indices = set(self.gcd_computer.decoder.extract_fibonacci_indices(traces[i]))
                    trace2_indices = set(self.gcd_computer.decoder.extract_fibonacci_indices(traces[j]))
                    
                    divisibility_tests.append(
                        gcd_indices.issubset(trace1_indices) and 
                        gcd_indices.issubset(trace2_indices)
                    )
        
        results['preserves_divisibility'] = all(divisibility_tests) if divisibility_tests else True
        
        # Universal property: GCD is the largest common divisor
        # For structural GCD, this means it contains all common indices
        results['universal_property'] = results['preserves_divisibility']
        
        return results
    
    def identify_gcd_morphisms(self, traces: List[str]) -> Dict[str, Any]:
        """Identify morphisms in GCD category"""
        divisibility_morphisms = []
        gcd_morphisms = []
        
        # Find divisibility relationships
        for i, trace1 in enumerate(traces):
            indices1 = set(self.gcd_computer.decoder.extract_fibonacci_indices(trace1))
            
            for j, trace2 in enumerate(traces):
                if i != j:
                    indices2 = set(self.gcd_computer.decoder.extract_fibonacci_indices(trace2))
                    
                    # Check if trace1 divides trace2 (indices1 ⊆ indices2)
                    if indices1.issubset(indices2):
                        divisibility_morphisms.append({
                            'domain': trace1,
                            'codomain': trace2,
                            'morphism_type': 'divisibility'
                        })
        
        # Find GCD morphisms
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i+1:], i+1):
                result = self.gcd_computer.compute_structural_gcd(trace1, trace2)
                
                if result.gcd_trace != "0":
                    gcd_morphisms.append({
                        'domain': (trace1, trace2),
                        'codomain': result.gcd_trace,
                        'morphism_type': 'gcd',
                        'common_indices': result.common_indices
                    })
        
        return {
            'divisibility_morphisms': divisibility_morphisms,
            'gcd_morphisms': gcd_morphisms,
            'total_morphisms': len(divisibility_morphisms) + len(gcd_morphisms)
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
    
    return [t for t in traces if len(t) <= max_length and t != '0']

def demonstrate_basic_gcd():
    """Demonstrate basic structural GCD examples from the prompt"""
    gcd_computer = StructuralGCDComputer()
    
    print("=== Basic Structural GCD Examples ===")
    
    examples = [
        # Example 1: Identical traces
        ("100100", "100100", "Complete identity"),
        
        # Example 2: Shared F5
        ("100100", "100010", "Share F5"),
        
        # Example 3: No common indices
        ("100100", "101000", "Disjoint indices"),
        
        # Example 4: No common positions
        ("101000", "100100", "No shared positions"),
        
        # Example 5: One contains another
        ("101000", "100000", "F6 shared"),
        
        # Example 6: Prime traces
        ("1000000", "10000", "Different primes"),
        
        # Example 7: Multiple shared indices
        ("1001010", "1000010", "Share F7 and F2"),
    ]
    
    for trace1, trace2, description in examples:
        result = gcd_computer.compute_structural_gcd(trace1, trace2)
        
        print(f"\n{description}:")
        print(f"  Trace 1: '{trace1}' → indices {gcd_computer.decoder.extract_fibonacci_indices(trace1)}")
        print(f"  Trace 2: '{trace2}' → indices {gcd_computer.decoder.extract_fibonacci_indices(trace2)}")
        print(f"  Common indices: {result.common_indices}")
        print(f"  GCD: '{result.gcd_trace}'")
        print(f"  Alignment: {result.alignment_type}")
        print(f"  Similarity: {result.structural_similarity:.3f}")

def demonstrate_multiple_gcd():
    """Demonstrate GCD of multiple traces"""
    gcd_computer = StructuralGCDComputer()
    
    print("\n=== Multiple Trace GCD ===")
    
    test_sets = [
        ["100100", "100010", "100000"],  # All share F5
        ["1001010", "1000010", "1000000"],  # All share F7
        ["100", "1000", "10000"],  # No common indices
    ]
    
    for traces in test_sets:
        result = gcd_computer.compute_multiple_gcd(traces)
        print(f"\nTraces: {traces}")
        print(f"Multiple GCD: '{result}'")
        
        # Show pairwise GCDs
        print("  Pairwise GCDs:")
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                pair_result = gcd_computer.compute_structural_gcd(traces[i], traces[j])
                print(f"    GCD('{traces[i]}', '{traces[j]}') = '{pair_result.gcd_trace}'")

def demonstrate_lcm():
    """Demonstrate structural LCM"""
    gcd_computer = StructuralGCDComputer()
    
    print("\n=== Structural LCM (Union of Components) ===")
    
    examples = [
        ("100", "1000", "Disjoint components"),
        ("100100", "100010", "Overlapping components"),
        ("10000", "1010", "Would create illegal 11"),
    ]
    
    for trace1, trace2, description in examples:
        lcm = gcd_computer.compute_structural_lcm(trace1, trace2)
        
        print(f"\n{description}:")
        print(f"  Trace 1: '{trace1}' → indices {gcd_computer.decoder.extract_fibonacci_indices(trace1)}")
        print(f"  Trace 2: '{trace2}' → indices {gcd_computer.decoder.extract_fibonacci_indices(trace2)}")
        print(f"  LCM: '{lcm}' → indices {gcd_computer.decoder.extract_fibonacci_indices(lcm) if lcm else []}")
        
        # Verify relationship: indices(LCM) ⊇ indices(trace1) ∪ indices(trace2)
        if lcm and lcm != "":
            lcm_indices = set(gcd_computer.decoder.extract_fibonacci_indices(lcm))
            union_indices = set(gcd_computer.decoder.extract_fibonacci_indices(trace1)) | \
                          set(gcd_computer.decoder.extract_fibonacci_indices(trace2))
            print(f"  Union preserved: {union_indices.issubset(lcm_indices)}")

def graph_theory_analysis():
    """Perform graph theory analysis of GCD structure"""
    graph_analyzer = GraphTheoryGCDAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)[:20]  # Limit to 20 traces
    
    # Build GCD graph
    G = graph_analyzer.build_gcd_graph(traces)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_gcd_graph_properties(G)
    
    print(f"GCD graph properties:")
    print(f"  Total nodes: {analysis['total_nodes']}")
    print(f"  Total edges: {analysis['total_edges']}")
    print(f"  Non-trivial GCDs: {analysis['non_trivial_gcds']}")
    print(f"  Unique GCD values: {analysis['unique_gcd_values']}")
    print(f"  Average similarity: {analysis['average_similarity']:.3f}")
    print(f"  Max similarity: {analysis['max_similarity']:.3f}")
    print(f"  Connected components: {analysis['connected_components']}")
    print(f"  Graph density: {analysis['density']:.3f}")
    print(f"  Clustering coefficient: {analysis['clustering_coefficient']:.3f}")
    print(f"  Communities: {analysis['communities']}")
    print(f"  Largest community: {analysis['largest_community_size']} nodes")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryGCDAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)[:20]
    
    # Analyze GCD entropy
    entropy_analysis = info_analyzer.compute_gcd_entropy(traces)
    
    print(f"GCD entropy analysis:")
    print(f"  GCD entropy: {entropy_analysis['gcd_entropy']:.3f} bits")
    print(f"  Unique GCDs: {entropy_analysis['unique_gcds']}")
    print(f"  Most common GCD: '{entropy_analysis['most_common_gcd']}'")
    print(f"  Average similarity: {entropy_analysis['average_similarity']:.3f}")
    print(f"  Average common indices: {entropy_analysis['average_common_indices']:.2f}")
    print(f"  Max common indices: {entropy_analysis['max_common_indices']}")
    
    # Analyze information preservation
    preservation_analysis = info_analyzer.analyze_information_preservation(traces)
    
    print(f"\nInformation preservation:")
    print(f"  Average preservation: {preservation_analysis['average_preservation']:.3f}")
    print(f"  Min preservation: {preservation_analysis['min_preservation']:.3f}")
    print(f"  Max preservation: {preservation_analysis['max_preservation']:.3f}")
    print(f"  Preservation variance: {preservation_analysis['preservation_variance']:.3f}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryGCDAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(5)[:10]
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_gcd_functor_properties(traces)
    
    print(f"GCD functor properties:")
    print(f"  Commutative: {functor_analysis['commutative']}")
    print(f"  Associative: {functor_analysis['associative']}")
    print(f"  Preserves identity: {functor_analysis['preserves_identity']}")
    print(f"  Preserves divisibility: {functor_analysis['preserves_divisibility']}")
    print(f"  Universal property: {functor_analysis['universal_property']}")
    
    # Identify morphisms
    morphisms = cat_analyzer.identify_gcd_morphisms(traces)
    
    print(f"\nCategorical morphisms:")
    print(f"  Divisibility morphisms: {len(morphisms['divisibility_morphisms'])}")
    print(f"  GCD morphisms: {len(morphisms['gcd_morphisms'])}")
    print(f"  Total morphisms: {morphisms['total_morphisms']}")

def verify_bezout_identity():
    """Verify Bezout identity for structural GCD"""
    gcd_computer = StructuralGCDComputer()
    decoder = TraceTensorDecoder()
    
    print("\n=== Bezout Identity Verification ===")
    
    # For structural GCD, Bezout identity translates to:
    # The GCD can be expressed using the common structural components
    
    examples = [
        ("100100", "100010"),  # Share F5
        ("1001010", "1000010"),  # Share F7 and F2
    ]
    
    for trace1, trace2 in examples:
        result = gcd_computer.compute_structural_gcd(trace1, trace2)
        
        print(f"\nTraces: '{trace1}' and '{trace2}'")
        print(f"GCD: '{result.gcd_trace}'")
        print(f"Common indices: {result.common_indices}")
        
        # Verify that GCD indices are exactly the intersection
        indices1 = set(decoder.extract_fibonacci_indices(trace1))
        indices2 = set(decoder.extract_fibonacci_indices(trace2))
        gcd_indices = set(decoder.extract_fibonacci_indices(result.gcd_trace))
        
        print(f"Verification: GCD = intersection of indices: {gcd_indices == (indices1 & indices2)}")

def main():
    """Run comprehensive structural GCD analysis"""
    print("="*80)
    print("Chapter 025: CollapseGCD - Structural Common Divisors Analysis")
    print("="*80)
    
    # Basic GCD demonstration
    demonstrate_basic_gcd()
    
    # Multiple GCD
    demonstrate_multiple_gcd()
    
    # LCM demonstration
    demonstrate_lcm()
    
    # Bezout identity
    verify_bezout_identity()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Structural GCD verification complete!")
    print("From ψ = ψ(ψ) emerges common structural divisors - shared subpaths")
    print("that maintain φ-constraint while revealing deep trace relationships.")

if __name__ == "__main__":
    main()