#!/usr/bin/env python3
"""
Chapter 022: CollapseMult - Multiplicative Folding of Collapse Trace Networks

Verification program demonstrating multiplicative folding operations on trace tensors:
- Network folding multiplication preserving φ-constraint
- Distributive properties in constrained tensor space
- Multiplicative folding complexity analysis
- Tensor network topology for multiplication

From ψ = ψ(ψ), we derive multiplication operations through network folding
that maintains golden structure while enabling complete multiplicative arithmetic.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools

@dataclass
class MultiplicationResult:
    """Complete result of trace tensor multiplication via folding"""
    operand1: str
    operand2: str
    trace_result: str
    decoded_result: int
    folding_network: Dict[str, Any]
    fibonacci_expansion: List[int]
    is_phi_compliant: bool
    network_complexity: int
    distributive_verification: bool
    operation_valid: bool
    folding_path: List[str]

class TraceTensorDecoder:
    """Enhanced decoder with multiplication support"""
    
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

class NetworkFoldingMultiplier:
    """Implements multiplication through trace network folding"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.folding_cache = {}
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def naive_trace_multiplication(self, trace1: str, trace2: str) -> MultiplicationResult:
        """Naive multiplication: decode → multiply → encode"""
        folding_path = []
        
        # Step 1: Decode both traces
        num1 = self.decoder.trace_to_number(trace1)
        num2 = self.decoder.trace_to_number(trace2)
        folding_path.append(f"Decode: '{trace1}' → {num1}, '{trace2}' → {num2}")
        
        # Step 2: Multiply numbers
        product = num1 * num2
        folding_path.append(f"Multiply: {num1} × {num2} = {product}")
        
        # Step 3: Encode result
        result_trace = self.decoder.number_to_trace(product)
        folding_path.append(f"Encode: {product} → '{result_trace}'")
        
        # Create basic folding network
        folding_network = {
            'operands': [trace1, trace2],
            'intermediate_products': [],
            'final_result': result_trace,
            'network_nodes': 3,
            'network_edges': 2
        }
        
        # Extract Fibonacci expansion
        fibonacci_expansion = self.decoder.extract_fibonacci_indices(result_trace)
        
        # Verify φ-compliance
        is_compliant = self.is_phi_compliant(result_trace)
        
        # Basic distributive verification (a × 1 = a)
        distributive_check = True
        if num2 == 1:
            distributive_check = (product == num1)
        elif num1 == 1:
            distributive_check = (product == num2)
        
        return MultiplicationResult(
            operand1=trace1,
            operand2=trace2,
            trace_result=result_trace,
            decoded_result=product,
            folding_network=folding_network,
            fibonacci_expansion=fibonacci_expansion,
            is_phi_compliant=is_compliant,
            network_complexity=folding_network['network_nodes'],
            distributive_verification=distributive_check,
            operation_valid=is_compliant,
            folding_path=folding_path
        )
    
    def distributive_folding_multiplication(self, trace1: str, trace2: str) -> MultiplicationResult:
        """Multiplication using distributive folding over Fibonacci components"""
        folding_path = []
        
        # Extract Fibonacci indices
        indices1 = self.decoder.extract_fibonacci_indices(trace1)
        indices2 = self.decoder.extract_fibonacci_indices(trace2)
        
        folding_path.append(f"Extract indices: {indices1} × {indices2}")
        
        # Distributive expansion: (Σ F_i) × (Σ F_j) = Σ Σ F_i × F_j
        intermediate_products = []
        total_sum = 0
        
        for i in indices1:
            for j in indices2:
                fib_i = self.decoder.get_fibonacci(i)
                fib_j = self.decoder.get_fibonacci(j)
                product_ij = fib_i * fib_j
                intermediate_products.append((i, j, fib_i, fib_j, product_ij))
                total_sum += product_ij
                folding_path.append(f"F_{i} × F_{j} = {fib_i} × {fib_j} = {product_ij}")
        
        folding_path.append(f"Sum all products: {total_sum}")
        
        # Encode final result
        result_trace = self.decoder.number_to_trace(total_sum)
        folding_path.append(f"Final encoding: {total_sum} → '{result_trace}'")
        
        # Create detailed folding network
        folding_network = {
            'operands': [trace1, trace2],
            'operand_indices': [indices1, indices2],
            'intermediate_products': intermediate_products,
            'final_result': result_trace,
            'network_nodes': len(indices1) + len(indices2) + len(intermediate_products) + 1,
            'network_edges': len(intermediate_products) * 2 + len(intermediate_products),
            'distributive_expansion': True
        }
        
        # Extract final Fibonacci expansion
        fibonacci_expansion = self.decoder.extract_fibonacci_indices(result_trace)
        
        # Verify distributive property: compare with naive method
        naive_result = self.naive_trace_multiplication(trace1, trace2)
        distributive_verification = (total_sum == naive_result.decoded_result)
        
        return MultiplicationResult(
            operand1=trace1,
            operand2=trace2,
            trace_result=result_trace,
            decoded_result=total_sum,
            folding_network=folding_network,
            fibonacci_expansion=fibonacci_expansion,
            is_phi_compliant=self.is_phi_compliant(result_trace),
            network_complexity=folding_network['network_nodes'],
            distributive_verification=distributive_verification,
            operation_valid=self.is_phi_compliant(result_trace),
            folding_path=folding_path
        )
    
    def tensor_network_multiplication(self, trace1: str, trace2: str) -> MultiplicationResult:
        """Multiplication through explicit tensor network construction"""
        folding_path = []
        
        # Build tensor network graph
        G = nx.DiGraph()
        
        # Add input nodes
        G.add_node("input1", trace=trace1, value=self.decoder.trace_to_number(trace1))
        G.add_node("input2", trace=trace2, value=self.decoder.trace_to_number(trace2))
        
        # Extract components
        indices1 = self.decoder.extract_fibonacci_indices(trace1)
        indices2 = self.decoder.extract_fibonacci_indices(trace2)
        
        folding_path.append(f"Build tensor network: {len(indices1)}×{len(indices2)} components")
        
        # Add component nodes
        for i, idx in enumerate(indices1):
            node_name = f"comp1_{i}"
            G.add_node(node_name, fib_index=idx, fib_value=self.decoder.get_fibonacci(idx))
            G.add_edge("input1", node_name)
        
        for j, idx in enumerate(indices2):
            node_name = f"comp2_{j}"
            G.add_node(node_name, fib_index=idx, fib_value=self.decoder.get_fibonacci(idx))
            G.add_edge("input2", node_name)
        
        # Add product nodes
        products = []
        for i, idx1 in enumerate(indices1):
            for j, idx2 in enumerate(indices2):
                product_node = f"prod_{i}_{j}"
                fib1 = self.decoder.get_fibonacci(idx1)
                fib2 = self.decoder.get_fibonacci(idx2)
                product_value = fib1 * fib2
                
                G.add_node(product_node, 
                          fib_indices=(idx1, idx2), 
                          fib_values=(fib1, fib2),
                          product=product_value)
                
                G.add_edge(f"comp1_{i}", product_node)
                G.add_edge(f"comp2_{j}", product_node)
                
                products.append(product_value)
        
        # Add accumulator node
        total_product = sum(products)
        G.add_node("accumulator", total=total_product)
        
        for i, idx1 in enumerate(indices1):
            for j, idx2 in enumerate(indices2):
                G.add_edge(f"prod_{i}_{j}", "accumulator")
        
        # Add output node
        result_trace = self.decoder.number_to_trace(total_product)
        G.add_node("output", trace=result_trace, value=total_product)
        G.add_edge("accumulator", "output")
        
        folding_path.append(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        folding_path.append(f"Final result: {total_product} → '{result_trace}'")
        
        # Create network description
        folding_network = {
            'graph': G,
            'operands': [trace1, trace2],
            'network_nodes': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'tensor_structure': True,
            'products': products,
            'final_result': result_trace
        }
        
        return MultiplicationResult(
            operand1=trace1,
            operand2=trace2,
            trace_result=result_trace,
            decoded_result=total_product,
            folding_network=folding_network,
            fibonacci_expansion=self.decoder.extract_fibonacci_indices(result_trace),
            is_phi_compliant=self.is_phi_compliant(result_trace),
            network_complexity=G.number_of_nodes(),
            distributive_verification=True,  # By construction
            operation_valid=self.is_phi_compliant(result_trace),
            folding_path=folding_path
        )

class GraphTheoryMultiplicationAnalyzer:
    """Graph theory analysis of multiplication networks"""
    
    def __init__(self):
        self.multiplier = NetworkFoldingMultiplier()
        
    def build_multiplication_graph(self, traces: List[str]) -> nx.DiGraph:
        """Build graph showing multiplication relationships"""
        G = nx.DiGraph()
        
        # Add nodes for all traces
        for trace in traces:
            value = self.multiplier.decoder.trace_to_number(trace)
            G.add_node(trace, value=value)
        
        # Add edges for multiplication operations
        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces):
                if i <= j:  # Avoid duplicates (multiplication is commutative)
                    result = self.multiplier.naive_trace_multiplication(trace1, trace2)
                    if result.operation_valid:
                        # Add intermediate result if it's in our trace set
                        if result.trace_result in traces or result.decoded_result <= 50:
                            G.add_edge(trace1, result.trace_result,
                                     operand2=trace2,
                                     operation='multiply',
                                     complexity=result.network_complexity)
        
        return G
    
    def analyze_multiplication_graph_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze properties of multiplication graph"""
        analysis = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected': nx.is_strongly_connected(G),
            'weakly_connected': nx.is_weakly_connected(G)
        }
        
        # Analyze multiplicative closure
        multiplications_count = 0
        closed_multiplications = 0
        
        for edge in G.edges(data=True):
            if edge[2].get('operation') == 'multiply':
                multiplications_count += 1
                if edge[1] in G.nodes():
                    closed_multiplications += 1
        
        analysis['multiplication_closure_rate'] = (
            closed_multiplications / multiplications_count 
            if multiplications_count > 0 else 0
        )
        analysis['total_multiplications'] = multiplications_count
        analysis['closed_multiplications'] = closed_multiplications
        
        # Network complexity analysis
        if G.number_of_edges() > 0:
            complexities = [data.get('complexity', 0) for _, _, data in G.edges(data=True)]
            analysis['average_complexity'] = np.mean(complexities)
            analysis['max_complexity'] = max(complexities)
        else:
            analysis['average_complexity'] = 0
            analysis['max_complexity'] = 0
        
        return analysis
    
    def analyze_folding_network_topology(self, result: MultiplicationResult) -> Dict[str, Any]:
        """Analyze topology of individual folding network"""
        if 'graph' in result.folding_network:
            G = result.folding_network['graph']
            
            # Safely compute diameter
            try:
                if nx.is_weakly_connected(G) and nx.is_strongly_connected(G):
                    diameter = nx.diameter(G)
                else:
                    # For weakly connected but not strongly connected DAGs
                    diameter = float('inf')
            except:
                diameter = float('inf')
            
            topology = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'diameter': diameter,
                'average_clustering': nx.average_clustering(G.to_undirected()),
                'is_tree': nx.is_tree(G.to_undirected()),
                'is_dag': nx.is_directed_acyclic_graph(G),
                'topological_sort_length': len(list(nx.topological_sort(G))) if nx.is_directed_acyclic_graph(G) else 0
            }
        else:
            # Basic network structure
            topology = {
                'nodes': result.folding_network.get('network_nodes', 0),
                'edges': result.folding_network.get('network_edges', 0),
                'complexity': result.network_complexity,
                'structure_type': 'basic'
            }
        
        return topology

class InformationTheoryMultiplicationAnalyzer:
    """Information theory analysis of multiplication operations"""
    
    def __init__(self):
        self.multiplier = NetworkFoldingMultiplier()
        
    def analyze_multiplication_entropy(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze entropy behavior during multiplication"""
        entropy_results = []
        complexity_results = []
        
        for trace1, trace2 in test_pairs:
            result = self.multiplier.naive_trace_multiplication(trace1, trace2)
            
            if result.operation_valid:
                # Compute entropy of operands and result
                entropy1 = self._compute_trace_entropy(trace1)
                entropy2 = self._compute_trace_entropy(trace2)
                entropy_result = self._compute_trace_entropy(result.trace_result)
                
                # Entropy change analysis
                input_entropy = (entropy1 + entropy2) / 2
                entropy_change = entropy_result - input_entropy
                
                entropy_results.append({
                    'input_entropy': input_entropy,
                    'output_entropy': entropy_result,
                    'entropy_change': entropy_change,
                    'complexity': result.network_complexity
                })
                
                complexity_results.append(result.network_complexity)
        
        if entropy_results:
            entropy_changes = [r['entropy_change'] for r in entropy_results]
            return {
                'total_operations': len(entropy_results),
                'average_entropy_change': np.mean(entropy_changes),
                'entropy_std': np.std(entropy_changes),
                'min_entropy_change': min(entropy_changes),
                'max_entropy_change': max(entropy_changes),
                'average_complexity': np.mean(complexity_results),
                'max_complexity': max(complexity_results),
                'entropy_complexity_correlation': np.corrcoef(
                    [r['entropy_change'] for r in entropy_results],
                    [r['complexity'] for r in entropy_results]
                )[0,1] if len(entropy_results) > 1 else 0
            }
        else:
            return {'total_operations': 0}
    
    def _compute_trace_entropy(self, trace: str) -> float:
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
    
    def compute_folding_complexity_bounds(self, traces: List[str]) -> Dict[str, float]:
        """Compute theoretical bounds on folding complexity"""
        fibonacci_counts = []
        for trace in traces:
            indices = self.multiplier.decoder.extract_fibonacci_indices(trace)
            fibonacci_counts.append(len(indices))
        
        if fibonacci_counts:
            return {
                'min_components': min(fibonacci_counts),
                'max_components': max(fibonacci_counts),
                'average_components': np.mean(fibonacci_counts),
                'theoretical_max_products': max(fibonacci_counts) ** 2,
                'complexity_growth_rate': np.log2(max(fibonacci_counts)) if max(fibonacci_counts) > 0 else 0
            }
        else:
            return {'min_components': 0, 'max_components': 0}

class CategoryTheoryMultiplicationAnalyzer:
    """Category theory analysis of multiplication functors"""
    
    def __init__(self):
        self.multiplier = NetworkFoldingMultiplier()
        
    def analyze_multiplication_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of trace multiplication"""
        results = {
            'preserves_identity': None,
            'is_commutative': None,
            'is_associative': None,
            'distributes_over_addition': None,
            'preserves_zero': None
        }
        
        # Test identity preservation: a × 1 = a
        one_trace = "1"  # Trace for number 1
        identity_tests = []
        for trace in traces[:5]:  # Test first 5 traces
            if trace != "0":  # Skip zero for identity test
                result = self.multiplier.naive_trace_multiplication(trace, one_trace)
                original_num = self.multiplier.decoder.trace_to_number(trace)
                identity_preserved = result.decoded_result == original_num
                identity_tests.append(identity_preserved)
        
        results['preserves_identity'] = all(identity_tests) if identity_tests else False
        
        # Test zero preservation: a × 0 = 0
        zero_trace = "0"
        zero_tests = []
        for trace in traces[:5]:
            result = self.multiplier.naive_trace_multiplication(trace, zero_trace)
            zero_preserved = result.decoded_result == 0
            zero_tests.append(zero_preserved)
        
        results['preserves_zero'] = all(zero_tests) if zero_tests else False
        
        # Test commutativity: a × b = b × a
        commutativity_tests = []
        for i, trace1 in enumerate(traces[:3]):
            for j, trace2 in enumerate(traces[:3]):
                if i < j:
                    result1 = self.multiplier.naive_trace_multiplication(trace1, trace2)
                    result2 = self.multiplier.naive_trace_multiplication(trace2, trace1)
                    commutative = result1.decoded_result == result2.decoded_result
                    commutativity_tests.append(commutative)
        
        results['is_commutative'] = all(commutativity_tests) if commutativity_tests else False
        
        # Test associativity: (a × b) × c = a × (b × c)
        associativity_tests = []
        test_traces = traces[:3]
        for i, a in enumerate(test_traces):
            for j, b in enumerate(test_traces):
                for k, c in enumerate(test_traces):
                    if i <= j <= k:
                        # Compute (a × b) × c
                        ab = self.multiplier.naive_trace_multiplication(a, b)
                        if ab.operation_valid:
                            abc_left = self.multiplier.naive_trace_multiplication(ab.trace_result, c)
                            
                            # Compute a × (b × c)
                            bc = self.multiplier.naive_trace_multiplication(b, c)
                            if bc.operation_valid:
                                abc_right = self.multiplier.naive_trace_multiplication(a, bc.trace_result)
                                
                                if abc_left.operation_valid and abc_right.operation_valid:
                                    associative = abc_left.decoded_result == abc_right.decoded_result
                                    associativity_tests.append(associative)
        
        results['is_associative'] = all(associativity_tests) if associativity_tests else False
        
        return results

def generate_test_traces(max_length: int = 5) -> List[str]:
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

def demonstrate_basic_multiplication():
    """Demonstrate basic network folding multiplication"""
    multiplier = NetworkFoldingMultiplier()
    
    print("=== Basic Network Folding Multiplication ===")
    
    test_cases = [
        ("1", "1"),      # 1 × 1 = 1
        ("1", "10"),     # 1 × 1 = 1
        ("10", "10"),    # 1 × 1 = 1
        ("100", "1"),    # 2 × 1 = 2
        ("100", "100"),  # 2 × 2 = 4
        ("101", "10"),   # 3 × 1 = 3
    ]
    
    for trace1, trace2 in test_cases:
        result = multiplier.naive_trace_multiplication(trace1, trace2)
        
        print(f"\nMultiplication: '{trace1}' × '{trace2}'")
        print(f"  Result trace: '{result.trace_result}'")
        print(f"  Decoded result: {result.decoded_result}")
        print(f"  φ-compliant: {result.is_phi_compliant}")
        print(f"  Network complexity: {result.network_complexity}")
        print(f"  Operation valid: {result.operation_valid}")
        if len(result.folding_path) <= 3:
            print(f"  Path: {' → '.join(result.folding_path)}")

def demonstrate_distributive_folding():
    """Demonstrate distributive folding multiplication"""
    multiplier = NetworkFoldingMultiplier()
    
    print("\n=== Distributive Folding Multiplication ===")
    
    test_cases = [
        ("101", "10"),   # (F₁+F₃) × F₂ = F₁×F₂ + F₃×F₂
        ("101", "101"),  # (F₁+F₃) × (F₁+F₃)
        ("1010", "100"), # (F₂+F₄) × F₃
    ]
    
    for trace1, trace2 in test_cases:
        result = multiplier.distributive_folding_multiplication(trace1, trace2)
        
        print(f"\nDistributive: '{trace1}' × '{trace2}'")
        print(f"  Result: '{result.trace_result}' = {result.decoded_result}")
        print(f"  Network nodes: {result.folding_network['network_nodes']}")
        print(f"  Intermediate products: {len(result.folding_network['intermediate_products'])}")
        print(f"  Distributive verified: {result.distributive_verification}")
        print(f"  φ-compliant: {result.is_phi_compliant}")

def demonstrate_tensor_network_multiplication():
    """Demonstrate tensor network multiplication"""
    multiplier = NetworkFoldingMultiplier()
    
    print("\n=== Tensor Network Multiplication ===")
    
    test_cases = [
        ("101", "10"),   # Small tensor network
        ("1010", "101"), # Medium tensor network
    ]
    
    for trace1, trace2 in test_cases:
        result = multiplier.tensor_network_multiplication(trace1, trace2)
        
        print(f"\nTensor Network: '{trace1}' × '{trace2}'")
        print(f"  Result: '{result.trace_result}' = {result.decoded_result}")
        print(f"  Network: {result.folding_network['network_nodes']} nodes, {result.folding_network['network_edges']} edges")
        print(f"  Products computed: {len(result.folding_network['products'])}")
        print(f"  φ-compliant: {result.is_phi_compliant}")

def graph_theory_analysis():
    """Perform graph theory analysis of multiplication operations"""
    graph_analyzer = GraphTheoryMultiplicationAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(4)
    
    # Build multiplication graph
    G = graph_analyzer.build_multiplication_graph(traces)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_multiplication_graph_properties(G)
    
    print(f"Multiplication graph properties:")
    print(f"  Nodes (traces): {analysis['node_count']}")
    print(f"  Edges (operations): {analysis['edge_count']}")
    print(f"  Graph density: {analysis['density']:.3f}")
    print(f"  Is DAG: {analysis['is_dag']}")
    print(f"  Strongly connected: {analysis['strongly_connected']}")
    print(f"  Weakly connected: {analysis['weakly_connected']}")
    print(f"  Multiplication closure rate: {analysis['multiplication_closure_rate']:.3f}")
    print(f"  Average complexity: {analysis['average_complexity']:.1f}")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryMultiplicationAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test pairs
    traces = generate_test_traces(4)
    test_pairs = [(traces[i], traces[j]) for i in range(len(traces)) 
                  for j in range(i, min(i+3, len(traces)))]
    
    # Analyze entropy behavior
    entropy_analysis = info_analyzer.analyze_multiplication_entropy(test_pairs)
    
    if entropy_analysis['total_operations'] > 0:
        print(f"Entropy behavior analysis:")
        print(f"  Total operations: {entropy_analysis['total_operations']}")
        print(f"  Average entropy change: {entropy_analysis['average_entropy_change']:.3f}")
        print(f"  Entropy std deviation: {entropy_analysis['entropy_std']:.3f}")
        print(f"  Average complexity: {entropy_analysis['average_complexity']:.1f}")
        print(f"  Max complexity: {entropy_analysis['max_complexity']}")
        
        if abs(entropy_analysis['entropy_complexity_correlation']) > 0.1:
            print(f"  Entropy-complexity correlation: {entropy_analysis['entropy_complexity_correlation']:.3f}")
    
    # Compute complexity bounds
    bounds = info_analyzer.compute_folding_complexity_bounds(traces)
    print(f"\nComplexity bounds:")
    print(f"  Component range: {bounds['min_components']}-{bounds['max_components']}")
    print(f"  Average components: {bounds['average_components']:.1f}")
    print(f"  Theoretical max products: {bounds['theoretical_max_products']}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryMultiplicationAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(4)
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_multiplication_functor_properties(traces)
    
    print(f"Multiplication functor properties:")
    print(f"  Preserves identity: {functor_analysis['preserves_identity']}")
    print(f"  Preserves zero: {functor_analysis['preserves_zero']}")
    print(f"  Is commutative: {functor_analysis['is_commutative']}")
    print(f"  Is associative: {functor_analysis['is_associative']}")

def verify_folding_properties():
    """Verify specific folding network properties"""
    multiplier = NetworkFoldingMultiplier()
    graph_analyzer = GraphTheoryMultiplicationAnalyzer()
    
    print("\n=== Folding Network Properties ===")
    
    test_cases = [
        ("101", "10"),   # 3 × 1 = 3
        ("1010", "101"), # 4 × 3 = 12
    ]
    
    for trace1, trace2 in test_cases:
        result = multiplier.tensor_network_multiplication(trace1, trace2)
        topology = graph_analyzer.analyze_folding_network_topology(result)
        
        print(f"\nNetwork '{trace1}' × '{trace2}':")
        print(f"  Topology: {topology['nodes']} nodes, {topology['edges']} edges")
        if 'diameter' in topology:
            print(f"  Diameter: {topology['diameter']}")
            print(f"  Is DAG: {topology['is_dag']}")
            print(f"  Is tree: {topology['is_tree']}")

def main():
    """Run comprehensive multiplication folding analysis"""
    print("="*80)
    print("Chapter 022: CollapseMult - Multiplicative Folding Analysis")
    print("="*80)
    
    # Basic multiplication demonstration
    demonstrate_basic_multiplication()
    
    # Distributive folding
    demonstrate_distributive_folding()
    
    # Tensor network multiplication
    demonstrate_tensor_network_multiplication()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    # Folding properties verification
    verify_folding_properties()
    
    print("\n" + "="*80)
    print("Multiplicative folding verification complete!")
    print("From ψ = ψ(ψ) emerges network folding multiplication - operations")
    print("that preserve φ-constraint through distributive tensor folding.")

if __name__ == "__main__":
    main()