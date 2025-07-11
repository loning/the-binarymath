#!/usr/bin/env python3
"""
Chapter 021: CollapseAdd - φ-Conformal Trace Merging under Entropy Bound

Verification program demonstrating φ-conformal addition operations on trace tensors:
- Trace-level addition preserving φ-constraint
- Entropy bounds during merging operations
- Conformal addition algorithms
- Information-theoretic analysis of addition complexity

From ψ = ψ(ψ), we derive addition operations that maintain golden structure
while enabling complete arithmetic on constrained tensor representations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass

@dataclass
class AdditionResult:
    """Complete result of φ-conformal trace addition"""
    operand1: str
    operand2: str
    trace_result: str
    decoded_result: int
    fibonacci_components: List[int]
    is_phi_compliant: bool
    entropy_before: float
    entropy_after: float
    entropy_change: float
    operation_valid: bool
    addition_path: List[str]

class TraceTensorDecoder:
    """Reuse decoder from Chapter 020 for validation"""
    
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

class ConformalTraceAdder:
    """Implements φ-conformal addition operations on trace tensors"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.addition_cache = {}
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check if trace satisfies φ-constraint (no consecutive 11s)"""
        return '11' not in trace
    
    def compute_trace_entropy(self, trace: str) -> float:
        """Compute information entropy of trace"""
        if not trace or len(trace) == 0:
            return 0.0
        
        ones = trace.count('1')
        zeros = trace.count('0')
        total = len(trace)
        
        if ones == 0 or zeros == 0:
            return 0.0
        
        p1 = ones / total
        p0 = zeros / total
        return -p1 * np.log2(p1) - p0 * np.log2(p0)
    
    def naive_trace_addition(self, trace1: str, trace2: str) -> AdditionResult:
        """Naive addition: decode → add → encode"""
        addition_path = []
        
        # Step 1: Decode both traces
        num1 = self.decoder.trace_to_number(trace1)
        num2 = self.decoder.trace_to_number(trace2)
        addition_path.append(f"Decode: '{trace1}' → {num1}, '{trace2}' → {num2}")
        
        # Step 2: Add numbers
        sum_result = num1 + num2
        addition_path.append(f"Add: {num1} + {num2} = {sum_result}")
        
        # Step 3: Encode result
        result_trace = self.decoder.number_to_trace(sum_result)
        addition_path.append(f"Encode: {sum_result} → '{result_trace}'")
        
        # Compute entropies
        entropy_before = (self.compute_trace_entropy(trace1) + self.compute_trace_entropy(trace2)) / 2
        entropy_after = self.compute_trace_entropy(result_trace)
        entropy_change = entropy_after - entropy_before
        
        # Verify φ-compliance
        is_compliant = self.is_phi_compliant(result_trace)
        
        # Get Fibonacci components
        fibonacci_components = []
        for i, bit in enumerate(reversed(result_trace)):
            if bit == '1':
                fibonacci_components.append(i + 1)
        
        return AdditionResult(
            operand1=trace1,
            operand2=trace2,
            trace_result=result_trace,
            decoded_result=sum_result,
            fibonacci_components=fibonacci_components,
            is_phi_compliant=is_compliant,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_change=entropy_change,
            operation_valid=is_compliant,
            addition_path=addition_path
        )
    
    def direct_fibonacci_addition(self, trace1: str, trace2: str) -> AdditionResult:
        """Direct addition by combining Fibonacci components"""
        addition_path = []
        
        # Extract Fibonacci indices from both traces
        indices1 = []
        for i, bit in enumerate(reversed(trace1)):
            if bit == '1':
                indices1.append(i + 1)
        
        indices2 = []
        for i, bit in enumerate(reversed(trace2)):
            if bit == '1':
                indices2.append(i + 1)
        
        addition_path.append(f"Extract indices: {indices1} + {indices2}")
        
        # Combine Fibonacci values
        total = 0
        for idx in indices1:
            total += self.decoder.get_fibonacci(idx)
        for idx in indices2:
            total += self.decoder.get_fibonacci(idx)
        
        addition_path.append(f"Sum Fibonacci values: {total}")
        
        # Re-encode using Zeckendorf
        result_trace = self.decoder.number_to_trace(total)
        addition_path.append(f"Re-encode: {total} → '{result_trace}'")
        
        # Compute entropies
        entropy_before = (self.compute_trace_entropy(trace1) + self.compute_trace_entropy(trace2)) / 2
        entropy_after = self.compute_trace_entropy(result_trace)
        
        # Get final Fibonacci components
        final_components = []
        for i, bit in enumerate(reversed(result_trace)):
            if bit == '1':
                final_components.append(i + 1)
        
        return AdditionResult(
            operand1=trace1,
            operand2=trace2,
            trace_result=result_trace,
            decoded_result=total,
            fibonacci_components=final_components,
            is_phi_compliant=self.is_phi_compliant(result_trace),
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_change=entropy_after - entropy_before,
            operation_valid=self.is_phi_compliant(result_trace),
            addition_path=addition_path
        )
    
    def entropy_bounded_addition(self, trace1: str, trace2: str, max_entropy_increase: float = 0.5) -> Optional[AdditionResult]:
        """Addition with entropy bound constraints"""
        # Try naive addition first
        result = self.naive_trace_addition(trace1, trace2)
        
        # Check entropy constraint
        if result.entropy_change <= max_entropy_increase:
            result.addition_path.append(f"Entropy bound satisfied: Δ={result.entropy_change:.3f} ≤ {max_entropy_increase}")
            return result
        else:
            result.addition_path.append(f"Entropy bound violated: Δ={result.entropy_change:.3f} > {max_entropy_increase}")
            result.operation_valid = False
            return result

class GraphTheoryAdditionAnalyzer:
    """Graph theory analysis of addition operations"""
    
    def __init__(self):
        self.adder = ConformalTraceAdder()
        
    def build_addition_graph(self, traces: List[str]) -> nx.DiGraph:
        """Build graph showing addition relationships"""
        G = nx.DiGraph()
        
        # Add nodes for all traces
        for trace in traces:
            num_value = self.adder.decoder.trace_to_number(trace)
            G.add_node(trace, value=num_value, entropy=self.adder.compute_trace_entropy(trace))
        
        # Add edges for addition operations
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces):
                if i <= j:  # Avoid duplicates (addition is commutative)
                    result = self.adder.naive_trace_addition(trace1, trace2)
                    if result.operation_valid and result.trace_result in traces:
                        G.add_edge(trace1, result.trace_result, 
                                  operand2=trace2, 
                                  operation='add',
                                  entropy_change=result.entropy_change)
        
        return G
    
    def analyze_addition_graph_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze properties of addition graph"""
        analysis = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'strongly_connected': nx.is_strongly_connected(G),
            'weakly_connected': nx.is_weakly_connected(G)
        }
        
        # Analyze closure properties
        additions_count = 0
        closed_additions = 0
        
        for edge in G.edges(data=True):
            if edge[2].get('operation') == 'add':
                additions_count += 1
                # Check if result is in the original trace set
                if edge[1] in G.nodes():
                    closed_additions += 1
        
        analysis['addition_closure_rate'] = closed_additions / additions_count if additions_count > 0 else 0
        analysis['total_additions'] = additions_count
        analysis['closed_additions'] = closed_additions
        
        return analysis

class InformationTheoryAdditionAnalyzer:
    """Information theory analysis of addition operations"""
    
    def __init__(self):
        self.adder = ConformalTraceAdder()
        
    def analyze_entropy_behavior(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze entropy changes during addition"""
        entropy_changes = []
        valid_operations = 0
        invalid_operations = 0
        
        entropy_stats = {
            'min_change': float('inf'),
            'max_change': float('-inf'),
            'zero_change_count': 0,
            'negative_change_count': 0,
            'positive_change_count': 0
        }
        
        for trace1, trace2 in test_pairs:
            result = self.adder.naive_trace_addition(trace1, trace2)
            
            if result.operation_valid:
                valid_operations += 1
                entropy_changes.append(result.entropy_change)
                
                # Update statistics
                entropy_stats['min_change'] = min(entropy_stats['min_change'], result.entropy_change)
                entropy_stats['max_change'] = max(entropy_stats['max_change'], result.entropy_change)
                
                if abs(result.entropy_change) < 1e-10:
                    entropy_stats['zero_change_count'] += 1
                elif result.entropy_change < 0:
                    entropy_stats['negative_change_count'] += 1
                else:
                    entropy_stats['positive_change_count'] += 1
            else:
                invalid_operations += 1
        
        return {
            'total_operations': len(test_pairs),
            'valid_operations': valid_operations,
            'invalid_operations': invalid_operations,
            'validity_rate': valid_operations / len(test_pairs) if test_pairs else 0,
            'average_entropy_change': np.mean(entropy_changes) if entropy_changes else 0,
            'entropy_std': np.std(entropy_changes) if entropy_changes else 0,
            'entropy_statistics': entropy_stats
        }
    
    def compute_addition_information_bounds(self, traces: List[str]) -> Dict[str, float]:
        """Compute information-theoretic bounds on addition"""
        entropies = [self.adder.compute_trace_entropy(trace) for trace in traces]
        
        return {
            'min_entropy': min(entropies) if entropies else 0,
            'max_entropy': max(entropies) if entropies else 0,
            'average_entropy': np.mean(entropies) if entropies else 0,
            'entropy_range': max(entropies) - min(entropies) if entropies else 0,
            'theoretical_max_increase': np.log2(2),  # Maximum for binary sequences
            'phi_constraint_bound': np.log2(1.618)  # Golden ratio bound
        }

class CategoryTheoryAdditionAnalyzer:
    """Category theory analysis of addition functor"""
    
    def __init__(self):
        self.adder = ConformalTraceAdder()
        
    def analyze_addition_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of trace addition"""
        results = {
            'preserves_identity': None,
            'is_commutative': None,
            'is_associative': None,
            'distributes_over_composition': None
        }
        
        # Test identity preservation: a + 0 = a
        zero_trace = "0"
        identity_tests = []
        for trace in traces[:5]:  # Test first 5 traces
            result = self.adder.naive_trace_addition(trace, zero_trace)
            original_num = self.adder.decoder.trace_to_number(trace)
            identity_preserved = result.decoded_result == original_num
            identity_tests.append(identity_preserved)
        
        results['preserves_identity'] = all(identity_tests) if identity_tests else False
        
        # Test commutativity: a + b = b + a
        commutativity_tests = []
        for i, trace1 in enumerate(traces[:3]):
            for j, trace2 in enumerate(traces[:3]):
                if i < j:
                    result1 = self.adder.naive_trace_addition(trace1, trace2)
                    result2 = self.adder.naive_trace_addition(trace2, trace1)
                    commutative = result1.decoded_result == result2.decoded_result
                    commutativity_tests.append(commutative)
        
        results['is_commutative'] = all(commutativity_tests) if commutativity_tests else False
        
        # Test associativity: (a + b) + c = a + (b + c)
        associativity_tests = []
        test_traces = traces[:3]  # Use first 3 traces
        for i, a in enumerate(test_traces):
            for j, b in enumerate(test_traces):
                for k, c in enumerate(test_traces):
                    if i <= j <= k:  # Avoid redundant tests
                        # Compute (a + b) + c
                        ab = self.adder.naive_trace_addition(a, b)
                        if ab.operation_valid:
                            abc_left = self.adder.naive_trace_addition(ab.trace_result, c)
                            
                            # Compute a + (b + c)
                            bc = self.adder.naive_trace_addition(b, c)
                            if bc.operation_valid:
                                abc_right = self.adder.naive_trace_addition(a, bc.trace_result)
                                
                                if abc_left.operation_valid and abc_right.operation_valid:
                                    associative = abc_left.decoded_result == abc_right.decoded_result
                                    associativity_tests.append(associative)
        
        results['is_associative'] = all(associativity_tests) if associativity_tests else False
        
        return results

def generate_test_traces(max_length: int = 6) -> List[str]:
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

def demonstrate_basic_addition():
    """Demonstrate basic conformal trace addition"""
    adder = ConformalTraceAdder()
    
    print("=== Basic Conformal Trace Addition ===")
    
    test_cases = [
        ("1", "1"),      # 1 + 1 = 2
        ("10", "1"),     # 1 + 1 = 2 (different encoding)
        ("100", "1"),    # 2 + 1 = 3
        ("101", "10"),   # 3 + 1 = 4
        ("1000", "100"), # 3 + 2 = 5
    ]
    
    for trace1, trace2 in test_cases:
        result = adder.naive_trace_addition(trace1, trace2)
        
        print(f"\nAddition: '{trace1}' + '{trace2}'")
        print(f"  Result trace: '{result.trace_result}'")
        print(f"  Decoded result: {result.decoded_result}")
        print(f"  φ-compliant: {result.is_phi_compliant}")
        print(f"  Entropy change: {result.entropy_change:.3f}")
        print(f"  Operation valid: {result.operation_valid}")
        if len(result.addition_path) <= 3:
            print(f"  Path: {' → '.join(result.addition_path)}")

def demonstrate_entropy_bounded_addition():
    """Demonstrate entropy-bounded addition operations"""
    adder = ConformalTraceAdder()
    
    print("\n=== Entropy-Bounded Addition ===")
    
    test_cases = [
        ("1", "1", 0.1),     # Low entropy bound
        ("101", "1010", 0.5), # Medium entropy bound
        ("10100", "10010", 1.0), # High entropy bound
    ]
    
    for trace1, trace2, max_entropy in test_cases:
        result = adder.entropy_bounded_addition(trace1, trace2, max_entropy)
        
        print(f"\nBounded addition: '{trace1}' + '{trace2}' (max Δ={max_entropy})")
        if result:
            print(f"  Result: '{result.trace_result}'")
            print(f"  Entropy change: {result.entropy_change:.3f}")
            print(f"  Bound satisfied: {result.operation_valid}")
        else:
            print(f"  Operation rejected due to entropy bound")

def graph_theory_analysis():
    """Perform graph theory analysis of addition operations"""
    graph_analyzer = GraphTheoryAdditionAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(5)
    
    # Build addition graph
    G = graph_analyzer.build_addition_graph(traces)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_addition_graph_properties(G)
    
    print(f"Addition graph properties:")
    print(f"  Nodes (traces): {analysis['node_count']}")
    print(f"  Edges (operations): {analysis['edge_count']}")
    print(f"  Graph density: {analysis['density']:.3f}")
    print(f"  Strongly connected: {analysis['strongly_connected']}")
    print(f"  Weakly connected: {analysis['weakly_connected']}")
    print(f"  Addition closure rate: {analysis['addition_closure_rate']:.3f}")
    print(f"  Closed additions: {analysis['closed_additions']}/{analysis['total_additions']}")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryAdditionAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test pairs
    traces = generate_test_traces(5)
    test_pairs = [(traces[i], traces[j]) for i in range(len(traces)) 
                  for j in range(i, min(i+5, len(traces)))]
    
    # Analyze entropy behavior
    entropy_analysis = info_analyzer.analyze_entropy_behavior(test_pairs)
    
    print(f"Entropy behavior analysis:")
    print(f"  Total operations: {entropy_analysis['total_operations']}")
    print(f"  Valid operations: {entropy_analysis['valid_operations']}")
    print(f"  Validity rate: {entropy_analysis['validity_rate']:.3f}")
    print(f"  Average entropy change: {entropy_analysis['average_entropy_change']:.3f}")
    print(f"  Entropy std deviation: {entropy_analysis['entropy_std']:.3f}")
    
    stats = entropy_analysis['entropy_statistics']
    print(f"  Entropy changes: {stats['negative_change_count']} negative, "
          f"{stats['zero_change_count']} zero, {stats['positive_change_count']} positive")
    
    # Compute information bounds
    bounds = info_analyzer.compute_addition_information_bounds(traces)
    print(f"\nInformation bounds:")
    print(f"  Entropy range: {bounds['entropy_range']:.3f}")
    print(f"  Average entropy: {bounds['average_entropy']:.3f}")
    print(f"  φ-constraint bound: {bounds['phi_constraint_bound']:.3f}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryAdditionAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(4)
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_addition_functor_properties(traces)
    
    print(f"Addition functor properties:")
    print(f"  Preserves identity: {functor_analysis['preserves_identity']}")
    print(f"  Is commutative: {functor_analysis['is_commutative']}")
    print(f"  Is associative: {functor_analysis['is_associative']}")

def demonstrate_fibonacci_component_addition():
    """Demonstrate direct Fibonacci component addition"""
    adder = ConformalTraceAdder()
    
    print("\n=== Direct Fibonacci Component Addition ===")
    
    test_cases = [
        ("1", "100"),    # F₁ + F₃ = 1 + 2 = 3
        ("10", "1000"),  # F₂ + F₄ = 1 + 3 = 4  
        ("101", "1010"), # (F₁+F₃) + (F₂+F₄) = 3 + 4 = 7
    ]
    
    for trace1, trace2 in test_cases:
        naive_result = adder.naive_trace_addition(trace1, trace2)
        direct_result = adder.direct_fibonacci_addition(trace1, trace2)
        
        print(f"\nComparing methods for '{trace1}' + '{trace2}':")
        print(f"  Naive result: {naive_result.decoded_result}")
        print(f"  Direct result: {direct_result.decoded_result}")
        print(f"  Results match: {naive_result.decoded_result == direct_result.decoded_result}")
        print(f"  Both φ-compliant: {naive_result.is_phi_compliant and direct_result.is_phi_compliant}")

def main():
    """Run comprehensive conformal addition analysis"""
    print("="*80)
    print("Chapter 021: CollapseAdd - φ-Conformal Trace Addition Analysis")
    print("="*80)
    
    # Basic addition demonstration
    demonstrate_basic_addition()
    
    # Entropy-bounded addition
    demonstrate_entropy_bounded_addition()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    # Fibonacci component addition
    demonstrate_fibonacci_component_addition()
    
    print("\n" + "="*80)
    print("Conformal addition verification complete!")
    print("From ψ = ψ(ψ) emerges φ-preserving arithmetic - addition that")
    print("maintains golden structure while enabling complete tensor arithmetic.")

if __name__ == "__main__":
    main()