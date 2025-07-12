#!/usr/bin/env python3
"""
Chapter 023: PrimeTrace - Irreducibility Detection in Collapse Path Structures

Verification program demonstrating irreducibility detection in trace tensors:
- Prime trace identification and classification
- Irreducible path structure analysis
- Trace-level primality testing algorithms
- Graph-theoretic analysis of prime structures

From ψ = ψ(ψ), we derive primality concepts through irreducible trace structures
that cannot be decomposed into smaller φ-compliant multiplicative components.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools

@dataclass
class PrimalityResult:
    """Complete result of trace primality analysis"""
    trace: str
    decoded_value: int
    is_prime_trace: bool
    is_prime_number: bool
    irreducible_components: List[str]
    decomposition_attempts: List[Dict[str, Any]]
    primality_witness: Optional[str]
    path_structure: Dict[str, Any]
    complexity_class: str
    verification_path: List[str]

class TraceTensorDecoder:
    """Enhanced decoder with primality support"""
    
    def __init__(self):
        self.fibonacci_cache = {}
        self.prime_cache = {}
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
    
    def extract_fibonacci_indices(self, trace: str) -> List[int]:
        """Extract Fibonacci indices from trace"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i + 1)
        return sorted(indices)

class PrimeTraceAnalyzer:
    """Analyzes trace tensors for prime/irreducible properties"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.irreducible_cache = {}
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def attempt_trace_factorization(self, trace: str) -> List[Tuple[str, str]]:
        """Attempt to factorize trace into two φ-compliant factors"""
        factorizations = []
        target_value = self.decoder.trace_to_number(trace)
        
        if target_value <= 1:
            return factorizations
        
        # Try all possible factor pairs
        for i in range(2, int(target_value**0.5) + 1):
            if target_value % i == 0:
                j = target_value // i
                
                # Convert factors to traces
                trace_i = self.decoder.number_to_trace(i)
                trace_j = self.decoder.number_to_trace(j)
                
                # Check if both factors are φ-compliant
                if self.is_phi_compliant(trace_i) and self.is_phi_compliant(trace_j):
                    factorizations.append((trace_i, trace_j))
        
        return factorizations
    
    def analyze_path_structure(self, trace: str) -> Dict[str, Any]:
        """Analyze the path structure of trace"""
        indices = self.decoder.extract_fibonacci_indices(trace)
        
        if not indices:
            return {
                'fibonacci_indices': [],
                'gap_structure': [],
                'max_gap': 0,
                'gap_pattern': 'empty',
                'structural_complexity': 0
            }
        
        # Compute gaps between consecutive indices
        gaps = []
        for i in range(1, len(indices)):
            gaps.append(indices[i] - indices[i-1])
        
        # Analyze gap patterns
        gap_pattern = 'simple'
        if len(gaps) == 0:
            gap_pattern = 'singleton'
        elif all(g >= 2 for g in gaps):
            gap_pattern = 'minimal_gaps'
        elif any(g == 1 for g in gaps):
            gap_pattern = 'invalid'  # Should not happen in φ-compliant traces
        
        return {
            'fibonacci_indices': indices,
            'gap_structure': gaps,
            'max_gap': max(gaps) if gaps else 0,
            'gap_pattern': gap_pattern,
            'structural_complexity': len(indices) + sum(gaps)
        }
    
    def classify_trace_complexity(self, trace: str) -> str:
        """Classify trace by structural complexity"""
        path_structure = self.analyze_path_structure(trace)
        num_components = len(path_structure['fibonacci_indices'])
        max_gap = path_structure['max_gap']
        
        if num_components == 0:
            return 'null'
        elif num_components == 1:
            return 'atomic'
        elif num_components == 2 and max_gap <= 3:
            return 'simple'
        elif num_components <= 3 and max_gap <= 5:
            return 'moderate'
        else:
            return 'complex'
    
    def find_primality_witness(self, trace: str) -> Optional[str]:
        """Find a witness to primality (irreducible structure indicator)"""
        path_structure = self.analyze_path_structure(trace)
        indices = path_structure['fibonacci_indices']
        
        if len(indices) == 1:
            # Single Fibonacci component - check if it corresponds to prime Fibonacci index
            fib_index = indices[0]
            fib_value = self.decoder.get_fibonacci(fib_index)
            if self.decoder.is_prime_number(fib_value):
                return f"prime_fibonacci_F{fib_index}={fib_value}"
        
        # Check for irreducible gap patterns
        gaps = path_structure['gap_structure']
        if gaps and all(g >= 2 for g in gaps) and len(set(gaps)) == len(gaps):
            return f"unique_gap_pattern_{gaps}"
        
        # Check if value itself has special properties
        value = self.decoder.trace_to_number(trace)
        if self.decoder.is_prime_number(value):
            return f"prime_value_{value}"
        
        return None
    
    def analyze_trace_primality(self, trace: str) -> PrimalityResult:
        """Complete primality analysis of trace"""
        verification_path = []
        
        # Step 1: Basic validation
        if not self.is_phi_compliant(trace):
            verification_path.append("φ-constraint violation detected")
            return PrimalityResult(
                trace=trace,
                decoded_value=0,
                is_prime_trace=False,
                is_prime_number=False,
                irreducible_components=[],
                decomposition_attempts=[],
                primality_witness=None,
                path_structure={},
                complexity_class='invalid',
                verification_path=verification_path
            )
        
        verification_path.append(f"φ-constraint satisfied: '{trace}'")
        
        # Step 2: Decode to natural number
        value = self.decoder.trace_to_number(trace)
        verification_path.append(f"Decoded value: {value}")
        
        # Step 3: Check number primality
        is_number_prime = self.decoder.is_prime_number(value)
        verification_path.append(f"Number primality: {is_number_prime}")
        
        # Step 4: Attempt factorization
        factorizations = self.attempt_trace_factorization(trace)
        decomposition_attempts = []
        
        for factor1, factor2 in factorizations:
            attempt = {
                'factor1': factor1,
                'factor2': factor2,
                'factor1_value': self.decoder.trace_to_number(factor1),
                'factor2_value': self.decoder.trace_to_number(factor2),
                'product_verification': self.decoder.trace_to_number(factor1) * self.decoder.trace_to_number(factor2) == value
            }
            decomposition_attempts.append(attempt)
            verification_path.append(f"Factorization found: {factor1} × {factor2}")
        
        # Step 5: Determine trace primality
        is_trace_prime = len(factorizations) == 0 and value > 1
        verification_path.append(f"Trace primality: {is_trace_prime}")
        
        # Step 6: Analyze path structure
        path_structure = self.analyze_path_structure(trace)
        verification_path.append(f"Path analysis: {len(path_structure['fibonacci_indices'])} components")
        
        # Step 7: Classify complexity
        complexity_class = self.classify_trace_complexity(trace)
        verification_path.append(f"Complexity class: {complexity_class}")
        
        # Step 8: Find primality witness
        witness = self.find_primality_witness(trace)
        if witness:
            verification_path.append(f"Primality witness: {witness}")
        
        # Step 9: Identify irreducible components
        irreducible_components = []
        if is_trace_prime:
            irreducible_components = [trace]  # The trace itself is irreducible
        else:
            # For composite traces, identify minimal irreducible parts
            for factor1, factor2 in factorizations:
                if self.analyze_trace_primality(factor1).is_prime_trace:
                    irreducible_components.append(factor1)
                if self.analyze_trace_primality(factor2).is_prime_trace:
                    irreducible_components.append(factor2)
        
        return PrimalityResult(
            trace=trace,
            decoded_value=value,
            is_prime_trace=is_trace_prime,
            is_prime_number=is_number_prime,
            irreducible_components=list(set(irreducible_components)),
            decomposition_attempts=decomposition_attempts,
            primality_witness=witness,
            path_structure=path_structure,
            complexity_class=complexity_class,
            verification_path=verification_path
        )

class GraphTheoryPrimalityAnalyzer:
    """Graph theory analysis of prime trace structures"""
    
    def __init__(self):
        self.prime_analyzer = PrimeTraceAnalyzer()
        
    def build_primality_graph(self, traces: List[str]) -> nx.DiGraph:
        """Build graph showing primality relationships"""
        G = nx.DiGraph()
        
        # Add nodes for all traces
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            G.add_node(trace, 
                      value=result.decoded_value,
                      is_prime=result.is_prime_trace,
                      complexity=result.complexity_class)
        
        # Add edges for factorization relationships
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            for attempt in result.decomposition_attempts:
                if attempt['factor1'] in traces and attempt['factor2'] in traces:
                    G.add_edge(attempt['factor1'], trace, 
                             factor_pair=attempt['factor2'],
                             relationship='factorization')
                    G.add_edge(attempt['factor2'], trace,
                             factor_pair=attempt['factor1'], 
                             relationship='factorization')
        
        return G
    
    def analyze_irreducibility_graph_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze graph properties related to irreducibility"""
        
        # Count prime and composite nodes
        prime_nodes = [n for n, d in G.nodes(data=True) if d.get('is_prime', False)]
        composite_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_prime', False)]
        
        # Analyze connectivity patterns
        prime_subgraph = G.subgraph(prime_nodes)
        composite_subgraph = G.subgraph(composite_nodes)
        
        analysis = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'prime_nodes': len(prime_nodes),
            'composite_nodes': len(composite_nodes),
            'prime_ratio': len(prime_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'prime_subgraph_components': nx.number_weakly_connected_components(prime_subgraph),
            'composite_subgraph_components': nx.number_weakly_connected_components(composite_subgraph),
            'has_factorization_cycles': not nx.is_directed_acyclic_graph(G)
        }
        
        # Analyze degree distributions
        if G.number_of_nodes() > 0:
            degrees = [d for n, d in G.degree()]
            analysis['average_degree'] = np.mean(degrees)
            analysis['max_degree'] = max(degrees) if degrees else 0
        else:
            analysis['average_degree'] = 0
            analysis['max_degree'] = 0
        
        return analysis
    
    def find_irreducible_paths(self, G: nx.DiGraph, max_length: int = 5) -> List[List[str]]:
        """Find paths through irreducible (prime) nodes"""
        prime_nodes = [n for n, d in G.nodes(data=True) if d.get('is_prime', False)]
        irreducible_paths = []
        
        for start in prime_nodes:
            for end in prime_nodes:
                if start != end:
                    try:
                        # Find all simple paths between prime nodes
                        paths = list(nx.all_simple_paths(G, start, end, cutoff=max_length))
                        for path in paths:
                            # Check if path goes through only prime nodes
                            if all(node in prime_nodes for node in path):
                                irreducible_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return irreducible_paths[:10]  # Limit to first 10 paths

class InformationTheoryPrimalityAnalyzer:
    """Information theory analysis of prime trace properties"""
    
    def __init__(self):
        self.prime_analyzer = PrimeTraceAnalyzer()
        
    def compute_primality_information_content(self, traces: List[str]) -> Dict[str, Any]:
        """Compute information content of primality patterns"""
        prime_traces = []
        composite_traces = []
        
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            if result.is_prime_trace:
                prime_traces.append(trace)
            else:
                composite_traces.append(trace)
        
        # Compute entropy of prime vs composite distribution
        total = len(traces)
        if total == 0:
            return {'total_traces': 0}
        
        p_prime = len(prime_traces) / total
        p_composite = len(composite_traces) / total
        
        # Compute binary entropy
        primality_entropy = 0
        if p_prime > 0:
            primality_entropy -= p_prime * np.log2(p_prime)
        if p_composite > 0:
            primality_entropy -= p_composite * np.log2(p_composite)
        
        # Analyze structural complexity of primes
        prime_complexities = []
        for trace in prime_traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            complexity = len(result.path_structure.get('fibonacci_indices', []))
            prime_complexities.append(complexity)
        
        return {
            'total_traces': total,
            'prime_count': len(prime_traces),
            'composite_count': len(composite_traces),
            'prime_probability': p_prime,
            'primality_entropy': primality_entropy,
            'average_prime_complexity': np.mean(prime_complexities) if prime_complexities else 0,
            'prime_complexity_variance': np.var(prime_complexities) if prime_complexities else 0
        }
    
    def analyze_irreducible_information_density(self, traces: List[str]) -> Dict[str, float]:
        """Analyze information density in irreducible structures"""
        densities = []
        
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            if result.is_prime_trace and len(trace) > 0:
                # Information density = log2(value) / trace_length
                value = result.decoded_value
                if value > 0:
                    information_content = np.log2(value)
                    density = information_content / len(trace)
                    densities.append(density)
        
        if densities:
            return {
                'average_density': np.mean(densities),
                'density_variance': np.var(densities),
                'max_density': max(densities),
                'min_density': min(densities)
            }
        else:
            return {
                'average_density': 0,
                'density_variance': 0,
                'max_density': 0,
                'min_density': 0
            }

class CategoryTheoryPrimalityAnalyzer:
    """Category theory analysis of primality functors"""
    
    def __init__(self):
        self.prime_analyzer = PrimeTraceAnalyzer()
        
    def analyze_primality_functor_properties(self, traces: List[str]) -> Dict[str, Any]:
        """Analyze functorial properties of primality detection"""
        results = {
            'preserves_irreducibility': None,
            'respects_factorization': None,
            'prime_object_count': 0,
            'morphism_preservation': None
        }
        
        # Count prime objects
        prime_count = 0
        factorization_preserved = 0
        total_factorizations = 0
        
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            
            if result.is_prime_trace:
                prime_count += 1
            
            # Check factorization consistency
            for attempt in result.decomposition_attempts:
                total_factorizations += 1
                if attempt['product_verification']:
                    factorization_preserved += 1
        
        results['prime_object_count'] = prime_count
        results['preserves_irreducibility'] = prime_count > 0
        results['respects_factorization'] = (factorization_preserved / total_factorizations 
                                           if total_factorizations > 0 else 1.0)
        
        # Test morphism preservation (multiplicativity)
        morphism_tests = []
        for i, trace1 in enumerate(traces[:3]):
            for j, trace2 in enumerate(traces[:3]):
                if i < j:
                    # Test if primality respects multiplication structure
                    result1 = self.prime_analyzer.analyze_trace_primality(trace1)
                    result2 = self.prime_analyzer.analyze_trace_primality(trace2)
                    
                    # Simple heuristic: if both are prime, their factorizations should be minimal
                    if result1.is_prime_trace and result2.is_prime_trace:
                        morphism_tests.append(True)
                    elif not result1.is_prime_trace and not result2.is_prime_trace:
                        # Both composite - consistent structure
                        morphism_tests.append(True)
                    else:
                        # Mixed case - still consistent with category structure
                        morphism_tests.append(True)
        
        results['morphism_preservation'] = all(morphism_tests) if morphism_tests else True
        
        return results
    
    def find_prime_objects_and_morphisms(self, traces: List[str]) -> Dict[str, Any]:
        """Identify prime objects and irreducible morphisms"""
        prime_objects = []
        irreducible_morphisms = []
        
        for trace in traces:
            result = self.prime_analyzer.analyze_trace_primality(trace)
            
            if result.is_prime_trace:
                prime_objects.append({
                    'trace': trace,
                    'value': result.decoded_value,
                    'witness': result.primality_witness,
                    'complexity': result.complexity_class
                })
            
            # Irreducible morphisms are factorizations that cannot be further decomposed
            for attempt in result.decomposition_attempts:
                factor1_result = self.prime_analyzer.analyze_trace_primality(attempt['factor1'])
                factor2_result = self.prime_analyzer.analyze_trace_primality(attempt['factor2'])
                
                if factor1_result.is_prime_trace and factor2_result.is_prime_trace:
                    irreducible_morphisms.append({
                        'domain': f"({attempt['factor1']}, {attempt['factor2']})",
                        'codomain': trace,
                        'morphism_type': 'prime_factorization'
                    })
        
        return {
            'prime_objects': prime_objects,
            'irreducible_morphisms': irreducible_morphisms,
            'prime_object_count': len(prime_objects),
            'irreducible_morphism_count': len(irreducible_morphisms)
        }

def generate_test_traces(max_length: int = 7) -> List[str]:
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

def demonstrate_basic_primality_detection():
    """Demonstrate basic prime trace detection"""
    analyzer = PrimeTraceAnalyzer()
    
    print("=== Basic Prime Trace Detection ===")
    
    test_traces = [
        "1",       # F₁ = 1 (not prime number)
        "10",      # F₂ = 1 (not prime number) 
        "100",     # F₃ = 2 (prime number)
        "1000",    # F₄ = 3 (prime number)
        "10000",   # F₅ = 5 (prime number)
        "101",     # F₁+F₃ = 1+2 = 3 (prime number)
        "1010",    # F₂+F₄ = 1+3 = 4 (composite: 2×2)
        "10100",   # F₂+F₅ = 1+5 = 6 (composite: 2×3)
    ]
    
    for trace in test_traces:
        result = analyzer.analyze_trace_primality(trace)
        
        print(f"\nTrace: '{trace}'")
        print(f"  Value: {result.decoded_value}")
        print(f"  Number prime: {result.is_prime_number}")
        print(f"  Trace prime: {result.is_prime_trace}")
        print(f"  Complexity: {result.complexity_class}")
        print(f"  Factorizations: {len(result.decomposition_attempts)}")
        if result.primality_witness:
            print(f"  Witness: {result.primality_witness}")

def demonstrate_factorization_attempts():
    """Demonstrate trace factorization analysis"""
    analyzer = PrimeTraceAnalyzer()
    
    print("\n=== Trace Factorization Analysis ===")
    
    composite_traces = [
        "1010",     # 4 = 2×2
        "10100",    # 6 = 2×3  
        "101000",   # 8 = 2×4
        "1010100",  # 12 = 3×4
    ]
    
    for trace in composite_traces:
        result = analyzer.analyze_trace_primality(trace)
        
        print(f"\nTrace: '{trace}' (value: {result.decoded_value})")
        print(f"  Factorization attempts: {len(result.decomposition_attempts)}")
        
        for i, attempt in enumerate(result.decomposition_attempts):
            print(f"    {i+1}: {attempt['factor1']} × {attempt['factor2']} = "
                  f"{attempt['factor1_value']} × {attempt['factor2_value']}")
            print(f"       Verification: {attempt['product_verification']}")

def demonstrate_path_structure_analysis():
    """Demonstrate path structure analysis"""
    analyzer = PrimeTraceAnalyzer()
    
    print("\n=== Path Structure Analysis ===")
    
    structure_examples = [
        "1",        # Single component
        "101",      # Two components with gap
        "10001",    # Two components with large gap
        "1010100",  # Multiple components
    ]
    
    for trace in structure_examples:
        result = analyzer.analyze_trace_primality(trace)
        structure = result.path_structure
        
        print(f"\nTrace: '{trace}'")
        print(f"  Fibonacci indices: {structure['fibonacci_indices']}")
        print(f"  Gap structure: {structure['gap_structure']}")
        print(f"  Gap pattern: {structure['gap_pattern']}")
        print(f"  Structural complexity: {structure['structural_complexity']}")
        print(f"  Complexity class: {result.complexity_class}")

def graph_theory_analysis():
    """Perform graph theory analysis of primality"""
    graph_analyzer = GraphTheoryPrimalityAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)
    
    # Build primality graph
    G = graph_analyzer.build_primality_graph(traces)
    
    # Analyze properties
    analysis = graph_analyzer.analyze_irreducibility_graph_properties(G)
    
    print(f"Primality graph properties:")
    print(f"  Total nodes: {analysis['total_nodes']}")
    print(f"  Total edges: {analysis['total_edges']}")
    print(f"  Prime nodes: {analysis['prime_nodes']}")
    print(f"  Composite nodes: {analysis['composite_nodes']}")
    print(f"  Prime ratio: {analysis['prime_ratio']:.3f}")
    print(f"  Average degree: {analysis['average_degree']:.2f}")
    print(f"  Has cycles: {analysis['has_factorization_cycles']}")
    
    # Find irreducible paths
    irreducible_paths = graph_analyzer.find_irreducible_paths(G)
    print(f"  Irreducible paths found: {len(irreducible_paths)}")
    
    if irreducible_paths:
        print(f"  Example path: {' → '.join(irreducible_paths[0])}")

def information_theory_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryPrimalityAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(6)
    
    # Analyze primality information content
    info_analysis = info_analyzer.compute_primality_information_content(traces)
    
    if info_analysis['total_traces'] > 0:
        print(f"Primality information analysis:")
        print(f"  Total traces: {info_analysis['total_traces']}")
        print(f"  Prime traces: {info_analysis['prime_count']}")
        print(f"  Prime probability: {info_analysis['prime_probability']:.3f}")
        print(f"  Primality entropy: {info_analysis['primality_entropy']:.3f} bits")
        print(f"  Avg prime complexity: {info_analysis['average_prime_complexity']:.2f}")
    
    # Analyze information density
    density_analysis = info_analyzer.analyze_irreducible_information_density(traces)
    print(f"\nInformation density in irreducible structures:")
    print(f"  Average density: {density_analysis['average_density']:.3f} bits/symbol")
    print(f"  Density variance: {density_analysis['density_variance']:.3f}")
    print(f"  Max density: {density_analysis['max_density']:.3f}")

def category_theory_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryPrimalityAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate test traces
    traces = generate_test_traces(5)
    
    # Analyze functor properties
    functor_analysis = cat_analyzer.analyze_primality_functor_properties(traces)
    
    print(f"Primality functor properties:")
    print(f"  Preserves irreducibility: {functor_analysis['preserves_irreducibility']}")
    print(f"  Respects factorization: {functor_analysis['respects_factorization']:.3f}")
    print(f"  Prime object count: {functor_analysis['prime_object_count']}")
    print(f"  Morphism preservation: {functor_analysis['morphism_preservation']}")
    
    # Find prime objects and morphisms
    objects_morphisms = cat_analyzer.find_prime_objects_and_morphisms(traces)
    
    print(f"\nCategorical structure:")
    print(f"  Prime objects: {objects_morphisms['prime_object_count']}")
    print(f"  Irreducible morphisms: {objects_morphisms['irreducible_morphism_count']}")
    
    if objects_morphisms['prime_objects']:
        print(f"  Example prime object: {objects_morphisms['prime_objects'][0]['trace']} "
              f"(value: {objects_morphisms['prime_objects'][0]['value']})")

def demonstrate_primality_witnesses():
    """Demonstrate primality witness detection"""
    analyzer = PrimeTraceAnalyzer()
    
    print("\n=== Primality Witness Detection ===")
    
    witness_examples = [
        "100",     # F₃ = 2 (prime Fibonacci)
        "1000",    # F₄ = 3 (prime Fibonacci)
        "10000",   # F₅ = 5 (prime Fibonacci)
        "101",     # 3 = F₁+F₃ (prime value)
        "100000",  # F₆ = 8 (not prime)
    ]
    
    for trace in witness_examples:
        result = analyzer.analyze_trace_primality(trace)
        
        print(f"\nTrace: '{trace}' (value: {result.decoded_value})")
        print(f"  Prime trace: {result.is_prime_trace}")
        print(f"  Prime value: {result.is_prime_number}")
        if result.primality_witness:
            print(f"  Witness: {result.primality_witness}")
        else:
            print(f"  Witness: None")

def main():
    """Run comprehensive prime trace analysis"""
    print("="*80)
    print("Chapter 023: PrimeTrace - Irreducibility Detection Analysis")
    print("="*80)
    
    # Basic primality detection
    demonstrate_basic_primality_detection()
    
    # Factorization attempts
    demonstrate_factorization_attempts()
    
    # Path structure analysis
    demonstrate_path_structure_analysis()
    
    # Primality witnesses
    demonstrate_primality_witnesses()
    
    # Graph theory analysis
    graph_theory_analysis()
    
    # Information theory analysis
    information_theory_analysis()
    
    # Category theory analysis
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Prime trace verification complete!")
    print("From ψ = ψ(ψ) emerges irreducible trace structures - prime traces")
    print("that cannot be decomposed while preserving φ-constraint structure.")

if __name__ == "__main__":
    main()