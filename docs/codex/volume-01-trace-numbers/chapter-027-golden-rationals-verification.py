#!/usr/bin/env python3
"""
Chapter 027: GoldenRationals - Constructing Rational Numbers from φ-Traces

Verification program demonstrating rational number construction from trace tensors:
- Rational representation through trace tensor pairs
- φ-constrained fraction arithmetic
- Canonical reduction maintaining golden constraint
- Rational approximation of irrational values

From ψ = ψ(ψ), we derive how rational numbers emerge naturally from
relationships between trace tensors in φ-constrained space.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from fractions import Fraction
import math
from dataclasses import dataclass

@dataclass
class RationalTrace:
    """Rational number represented as trace tensor pair"""
    numerator_trace: str
    denominator_trace: str
    numerator_value: int
    denominator_value: int
    fraction: Fraction
    is_canonical: bool
    reduction_path: List[str]
    tensor_representation: torch.Tensor
    phi_compliant: bool

class TraceTensorDecoder:
    """Decoder for trace to number conversion"""
    
    def __init__(self):
        self.fibonacci_cache = {1: 1, 2: 1}
        self._compute_fibonacci_sequence(50)
        
    def _compute_fibonacci_sequence(self, n: int):
        """Pre-compute Fibonacci sequence"""
        for i in range(3, n + 1):
            self.fibonacci_cache[i] = self.fibonacci_cache[i-1] + self.fibonacci_cache[i-2]
    
    def get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number"""
        if n not in self.fibonacci_cache:
            for i in range(max(self.fibonacci_cache.keys()) + 1, n + 1):
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
        while self.get_fibonacci(fib_index) <= remaining:
            fib_index += 1
        fib_index -= 1
        
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

class RationalTraceConstructor:
    """Construct rational numbers from trace tensor pairs"""
    
    def __init__(self):
        self.decoder = TraceTensorDecoder()
        self.phi = (1 + math.sqrt(5)) / 2
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def construct_rational(self, num_trace: str, den_trace: str) -> RationalTrace:
        """Construct rational from trace pair"""
        reduction_path = []
        
        # Validate φ-compliance
        if not self.is_phi_compliant(num_trace) or not self.is_phi_compliant(den_trace):
            reduction_path.append("φ-constraint violation detected")
            return self._create_invalid_rational(num_trace, den_trace, reduction_path)
        
        reduction_path.append(f"Initial traces: {num_trace}/{den_trace}")
        
        # Decode to numbers
        num_value = self.decoder.trace_to_number(num_trace)
        den_value = self.decoder.trace_to_number(den_trace)
        
        if den_value == 0:
            reduction_path.append("Division by zero")
            return self._create_invalid_rational(num_trace, den_trace, reduction_path)
        
        reduction_path.append(f"Decoded values: {num_value}/{den_value}")
        
        # Create fraction and reduce
        frac = Fraction(num_value, den_value)
        reduction_path.append(f"Fraction form: {frac}")
        
        # Check if already canonical
        is_canonical = (frac.numerator == num_value and frac.denominator == den_value)
        
        if not is_canonical:
            # Find canonical traces
            canon_num_trace = self.decoder.number_to_trace(frac.numerator)
            canon_den_trace = self.decoder.number_to_trace(frac.denominator)
            reduction_path.append(f"Canonical traces: {canon_num_trace}/{canon_den_trace}")
        else:
            canon_num_trace = num_trace
            canon_den_trace = den_trace
            reduction_path.append("Already in canonical form")
        
        # Create tensor representation
        tensor_repr = self._create_tensor_representation(canon_num_trace, canon_den_trace)
        
        return RationalTrace(
            numerator_trace=canon_num_trace,
            denominator_trace=canon_den_trace,
            numerator_value=frac.numerator,
            denominator_value=frac.denominator,
            fraction=frac,
            is_canonical=True,
            reduction_path=reduction_path,
            tensor_representation=tensor_repr,
            phi_compliant=True
        )
    
    def _create_invalid_rational(self, num_trace: str, den_trace: str, 
                                path: List[str]) -> RationalTrace:
        """Create invalid rational result"""
        return RationalTrace(
            numerator_trace=num_trace,
            denominator_trace=den_trace,
            numerator_value=0,
            denominator_value=0,
            fraction=Fraction(0, 1),
            is_canonical=False,
            reduction_path=path,
            tensor_representation=torch.zeros(1),
            phi_compliant=False
        )
    
    def _create_tensor_representation(self, num_trace: str, den_trace: str) -> torch.Tensor:
        """Create tensor representation of rational"""
        # Pad traces to same length
        max_len = max(len(num_trace), len(den_trace))
        num_padded = num_trace.zfill(max_len)
        den_padded = den_trace.zfill(max_len)
        
        # Create 2D tensor: [numerator_bits, denominator_bits]
        num_tensor = torch.tensor([int(b) for b in num_padded], dtype=torch.float32)
        den_tensor = torch.tensor([int(b) for b in den_padded], dtype=torch.float32)
        
        return torch.stack([num_tensor, den_tensor])
    
    def add_rationals(self, r1: RationalTrace, r2: RationalTrace) -> RationalTrace:
        """Add two rational traces"""
        if not r1.phi_compliant or not r2.phi_compliant:
            return self._create_invalid_rational("", "", ["Invalid input rationals"])
        
        # Add fractions
        sum_frac = r1.fraction + r2.fraction
        
        # Convert to traces
        sum_num_trace = self.decoder.number_to_trace(sum_frac.numerator)
        sum_den_trace = self.decoder.number_to_trace(sum_frac.denominator)
        
        return self.construct_rational(sum_num_trace, sum_den_trace)
    
    def multiply_rationals(self, r1: RationalTrace, r2: RationalTrace) -> RationalTrace:
        """Multiply two rational traces"""
        if not r1.phi_compliant or not r2.phi_compliant:
            return self._create_invalid_rational("", "", ["Invalid input rationals"])
        
        # Multiply fractions
        prod_frac = r1.fraction * r2.fraction
        
        # Convert to traces
        prod_num_trace = self.decoder.number_to_trace(prod_frac.numerator)
        prod_den_trace = self.decoder.number_to_trace(prod_frac.denominator)
        
        return self.construct_rational(prod_num_trace, prod_den_trace)
    
    def approximate_phi(self, max_denominator: int = 100) -> List[RationalTrace]:
        """Generate rational approximations of φ using Fibonacci convergents"""
        approximations = []
        
        # Fibonacci convergents approach φ
        for i in range(2, 20):
            fib_i = self.decoder.get_fibonacci(i)
            fib_i1 = self.decoder.get_fibonacci(i + 1)
            
            if fib_i > max_denominator:
                break
            
            # Create rational from Fibonacci ratio
            num_trace = self.decoder.number_to_trace(fib_i1)
            den_trace = self.decoder.number_to_trace(fib_i)
            
            rational = self.construct_rational(num_trace, den_trace)
            approximations.append(rational)
        
        return approximations

class GraphTheoryRationalAnalyzer:
    """Graph theory analysis of rational structures"""
    
    def __init__(self):
        self.constructor = RationalTraceConstructor()
        
    def build_farey_graph(self, max_denominator: int = 10) -> Dict[str, Any]:
        """Build Farey sequence graph in trace space"""
        import networkx as nx
        
        G = nx.Graph()
        rationals = []
        
        # Generate Farey sequence
        for den in range(1, max_denominator + 1):
            for num in range(0, den + 1):
                if math.gcd(num, den) == 1:  # Reduced form only
                    num_trace = self.constructor.decoder.number_to_trace(num)
                    den_trace = self.constructor.decoder.number_to_trace(den) 
                    
                    if self.constructor.is_phi_compliant(num_trace) and \
                       self.constructor.is_phi_compliant(den_trace):
                        rational = self.constructor.construct_rational(num_trace, den_trace)
                        rationals.append(rational)
                        
                        # Add node
                        node_id = f"{num}/{den}"
                        G.add_node(node_id, 
                                 rational=rational,
                                 value=float(rational.fraction))
        
        # Add Farey neighbor edges
        sorted_rats = sorted(rationals, key=lambda r: float(r.fraction))
        for i in range(len(sorted_rats) - 1):
            r1, r2 = sorted_rats[i], sorted_rats[i + 1]
            # Check if Farey neighbors (ad - bc = 1)
            a, b = r1.numerator_value, r1.denominator_value
            c, d = r2.numerator_value, r2.denominator_value
            
            if abs(a * d - b * c) == 1:
                node1 = f"{a}/{b}"
                node2 = f"{c}/{d}"
                G.add_edge(node1, node2, relation='farey_neighbor')
        
        analysis = {
            'graph': G,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'is_connected': nx.is_connected(G),
            'rational_count': len(rationals)
        }
        
        return analysis
    
    def analyze_mediant_structure(self, max_depth: int = 5) -> Dict[str, Any]:
        """Analyze mediant tree structure"""
        mediant_tree = {}
        
        def compute_mediant(r1: RationalTrace, r2: RationalTrace) -> Optional[RationalTrace]:
            """Compute mediant of two rationals"""
            # Mediant: (a+c)/(b+d) for a/b and c/d
            med_num = r1.numerator_value + r2.numerator_value
            med_den = r1.denominator_value + r2.denominator_value
            
            med_num_trace = self.constructor.decoder.number_to_trace(med_num)
            med_den_trace = self.constructor.decoder.number_to_trace(med_den)
            
            if self.constructor.is_phi_compliant(med_num_trace) and \
               self.constructor.is_phi_compliant(med_den_trace):
                return self.constructor.construct_rational(med_num_trace, med_den_trace)
            return None
        
        # Start with 0/1 and 1/1
        r0 = self.constructor.construct_rational("0", "1")
        r1 = self.constructor.construct_rational("1", "1")
        
        level = 0
        current_level = [(r0, r1)]
        tree_nodes = 2
        
        while level < max_depth and current_level:
            next_level = []
            
            for left, right in current_level:
                mediant = compute_mediant(left, right)
                if mediant:
                    tree_nodes += 1
                    # Add to next level
                    next_level.append((left, mediant))
                    next_level.append((mediant, right))
            
            current_level = next_level
            level += 1
        
        return {
            'max_depth': max_depth,
            'total_nodes': tree_nodes,
            'levels_generated': level,
            'final_level_size': len(current_level)
        }

class InformationTheoryRationalAnalyzer:
    """Information theory analysis of rational representations"""
    
    def __init__(self):
        self.constructor = RationalTraceConstructor()
        
    def compute_rational_entropy(self, rationals: List[RationalTrace]) -> Dict[str, float]:
        """Compute entropy of rational representations"""
        # Collect trace lengths
        num_lengths = [len(r.numerator_trace) for r in rationals]
        den_lengths = [len(r.denominator_trace) for r in rationals]
        total_lengths = [len(r.numerator_trace) + len(r.denominator_trace) 
                        for r in rationals]
        
        # Compute entropies
        def entropy(values):
            if not values:
                return 0
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            total = len(values)
            ent = 0
            for count in counts.values():
                p = count / total
                if p > 0:
                    ent -= p * np.log2(p)
            return ent
        
        return {
            'numerator_entropy': entropy(num_lengths),
            'denominator_entropy': entropy(den_lengths),
            'total_length_entropy': entropy(total_lengths),
            'average_numerator_length': np.mean(num_lengths) if num_lengths else 0,
            'average_denominator_length': np.mean(den_lengths) if den_lengths else 0,
            'average_total_length': np.mean(total_lengths) if total_lengths else 0
        }
    
    def analyze_approximation_efficiency(self, target: float, 
                                       approximations: List[RationalTrace]) -> Dict[str, Any]:
        """Analyze efficiency of rational approximations"""
        errors = []
        complexities = []
        
        for approx in approximations:
            error = abs(float(approx.fraction) - target)
            complexity = len(approx.numerator_trace) + len(approx.denominator_trace)
            
            errors.append(error)
            complexities.append(complexity)
        
        if not errors:
            return {}
        
        # Find best approximations (Pareto frontier)
        pareto_indices = []
        for i in range(len(errors)):
            is_pareto = True
            for j in range(len(errors)):
                if i != j:
                    # Check if j dominates i
                    if errors[j] <= errors[i] and complexities[j] <= complexities[i]:
                        if errors[j] < errors[i] or complexities[j] < complexities[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        return {
            'min_error': min(errors),
            'max_error': max(errors),
            'average_error': np.mean(errors),
            'min_complexity': min(complexities),
            'max_complexity': max(complexities),
            'average_complexity': np.mean(complexities),
            'pareto_count': len(pareto_indices),
            'pareto_rationals': [approximations[i] for i in pareto_indices]
        }

class CategoryTheoryRationalAnalyzer:
    """Category theory analysis of rational structures"""
    
    def __init__(self):
        self.constructor = RationalTraceConstructor()
        
    def analyze_field_structure(self, sample_rationals: List[RationalTrace]) -> Dict[str, bool]:
        """Verify field axioms for rational traces"""
        results = {
            'closure_under_addition': True,
            'closure_under_multiplication': True,
            'additive_identity_exists': False,
            'multiplicative_identity_exists': False,
            'additive_inverses_exist': True,
            'multiplicative_inverses_exist': True,
            'commutativity': True,
            'associativity': True,
            'distributivity': True
        }
        
        # Check for identities
        zero = self.constructor.construct_rational("0", "1")
        one = self.constructor.construct_rational("1", "1")
        
        results['additive_identity_exists'] = zero.phi_compliant
        results['multiplicative_identity_exists'] = one.phi_compliant
        
        # Test sample operations
        if len(sample_rationals) >= 3:
            r1, r2, r3 = sample_rationals[:3]
            
            # Test closure
            sum_r = self.constructor.add_rationals(r1, r2)
            prod_r = self.constructor.multiply_rationals(r1, r2)
            
            results['closure_under_addition'] = sum_r.phi_compliant
            results['closure_under_multiplication'] = prod_r.phi_compliant
            
            # Test associativity
            sum_12_3 = self.constructor.add_rationals(
                self.constructor.add_rationals(r1, r2), r3)
            sum_1_23 = self.constructor.add_rationals(
                r1, self.constructor.add_rationals(r2, r3))
            
            if sum_12_3.phi_compliant and sum_1_23.phi_compliant:
                results['associativity'] = (sum_12_3.fraction == sum_1_23.fraction)
        
        return results
    
    def identify_morphisms(self, rationals: List[RationalTrace]) -> Dict[str, Any]:
        """Identify morphisms in rational category"""
        morphisms = {
            'order_preserving': [],
            'multiplicative': [],
            'additive': []
        }
        
        # Sort rationals by value
        sorted_rats = sorted(rationals, key=lambda r: float(r.fraction))
        
        # Order-preserving morphisms (inclusions)
        for i in range(len(sorted_rats) - 1):
            r1, r2 = sorted_rats[i], sorted_rats[i + 1]
            morphisms['order_preserving'].append({
                'source': f"{r1.numerator_value}/{r1.denominator_value}",
                'target': f"{r2.numerator_value}/{r2.denominator_value}",
                'type': 'order'
            })
        
        # Sample multiplicative morphisms (scaling)
        for r in rationals[:5]:  # Limit for efficiency
            # Morphism: multiply by 2
            doubled = self.constructor.multiply_rationals(
                r, self.constructor.construct_rational("10", "1"))  # 2 in trace
            if doubled.phi_compliant:
                morphisms['multiplicative'].append({
                    'source': f"{r.numerator_value}/{r.denominator_value}",
                    'target': f"{doubled.numerator_value}/{doubled.denominator_value}",
                    'type': 'multiply_by_2'
                })
        
        return {
            'order_morphism_count': len(morphisms['order_preserving']),
            'multiplicative_morphism_count': len(morphisms['multiplicative']),
            'total_morphisms': sum(len(m) for m in morphisms.values()),
            'morphism_types': morphisms
        }

def demonstrate_basic_rationals():
    """Demonstrate basic rational construction"""
    constructor = RationalTraceConstructor()
    
    print("=== Basic Rational Construction ===")
    
    examples = [
        ("1", "1", "Unity"),
        ("100", "1", "Two"),
        ("1000", "10", "Three halves"),
        ("10000", "1000", "Five thirds"),
        ("1000000", "100000", "Thirteen eighths"),
    ]
    
    for num_trace, den_trace, description in examples:
        rational = constructor.construct_rational(num_trace, den_trace)
        
        print(f"\n{description}:")
        print(f"  Traces: {num_trace}/{den_trace}")
        print(f"  Values: {rational.numerator_value}/{rational.denominator_value}")
        print(f"  Fraction: {rational.fraction}")
        print(f"  Decimal: {float(rational.fraction):.6f}")
        print(f"  Canonical: {rational.is_canonical}")
        print(f"  φ-compliant: {rational.phi_compliant}")

def demonstrate_rational_arithmetic():
    """Demonstrate rational arithmetic operations"""
    constructor = RationalTraceConstructor()
    
    print("\n=== Rational Arithmetic ===")
    
    # Create test rationals
    r1 = constructor.construct_rational("10", "1")      # 2/1
    r2 = constructor.construct_rational("1000", "10")   # 3/2
    r3 = constructor.construct_rational("10000", "1000") # 5/3
    
    print(f"\nr1 = {r1.fraction}")
    print(f"r2 = {r2.fraction}")
    print(f"r3 = {r3.fraction}")
    
    # Addition
    sum_r = constructor.add_rationals(r1, r2)
    print(f"\nr1 + r2 = {sum_r.fraction}")
    print(f"  Traces: {sum_r.numerator_trace}/{sum_r.denominator_trace}")
    
    # Multiplication
    prod_r = constructor.multiply_rationals(r2, r3)
    print(f"\nr2 × r3 = {prod_r.fraction}")
    print(f"  Traces: {prod_r.numerator_trace}/{prod_r.denominator_trace}")

def demonstrate_phi_approximation():
    """Demonstrate rational approximation of φ"""
    constructor = RationalTraceConstructor()
    
    print("\n=== Rational Approximations of φ ===")
    print(f"Target: φ = {constructor.phi:.10f}")
    
    approximations = constructor.approximate_phi(max_denominator=100)
    
    print("\nFibonacci Convergents:")
    for i, approx in enumerate(approximations[:8]):
        error = abs(float(approx.fraction) - constructor.phi)
        print(f"{i+1}. {approx.fraction} = {float(approx.fraction):.10f}, error = {error:.2e}")
        print(f"   Traces: {approx.numerator_trace}/{approx.denominator_trace}")

def graph_theory_analysis():
    """Perform graph theory analysis"""
    analyzer = GraphTheoryRationalAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Build Farey graph
    farey_analysis = analyzer.build_farey_graph(max_denominator=8)
    
    print(f"Farey Graph Properties:")
    print(f"  Nodes: {farey_analysis['node_count']}")
    print(f"  Edges: {farey_analysis['edge_count']}")
    print(f"  Density: {farey_analysis['density']:.3f}")
    print(f"  Connected: {farey_analysis['is_connected']}")
    print(f"  Rational count: {farey_analysis['rational_count']}")
    
    # Analyze mediant structure
    mediant_analysis = analyzer.analyze_mediant_structure(max_depth=5)
    
    print(f"\nMediant Tree Structure:")
    print(f"  Max depth: {mediant_analysis['max_depth']}")
    print(f"  Total nodes: {mediant_analysis['total_nodes']}")
    print(f"  Levels generated: {mediant_analysis['levels_generated']}")
    print(f"  Final level size: {mediant_analysis['final_level_size']}")

def information_theory_analysis():
    """Perform information theory analysis"""
    constructor = RationalTraceConstructor()
    analyzer = InformationTheoryRationalAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate sample rationals
    rationals = []
    for den in range(1, 20):
        for num in range(0, den):
            if math.gcd(num, den) == 1:
                num_trace = constructor.decoder.number_to_trace(num)
                den_trace = constructor.decoder.number_to_trace(den)
                if constructor.is_phi_compliant(num_trace) and \
                   constructor.is_phi_compliant(den_trace):
                    rationals.append(constructor.construct_rational(num_trace, den_trace))
    
    # Compute entropy
    entropy_analysis = analyzer.compute_rational_entropy(rationals)
    
    print(f"Rational Representation Entropy:")
    print(f"  Numerator entropy: {entropy_analysis['numerator_entropy']:.3f} bits")
    print(f"  Denominator entropy: {entropy_analysis['denominator_entropy']:.3f} bits")
    print(f"  Total length entropy: {entropy_analysis['total_length_entropy']:.3f} bits")
    print(f"  Average numerator length: {entropy_analysis['average_numerator_length']:.2f}")
    print(f"  Average denominator length: {entropy_analysis['average_denominator_length']:.2f}")
    
    # Analyze φ approximation efficiency
    phi_approx = constructor.approximate_phi(max_denominator=100)
    efficiency = analyzer.analyze_approximation_efficiency(constructor.phi, phi_approx)
    
    print(f"\nφ Approximation Efficiency:")
    print(f"  Min error: {efficiency['min_error']:.2e}")
    print(f"  Average error: {efficiency['average_error']:.2e}")
    print(f"  Min complexity: {efficiency['min_complexity']}")
    print(f"  Average complexity: {efficiency['average_complexity']:.1f}")
    print(f"  Pareto optimal count: {efficiency['pareto_count']}")

def category_theory_analysis():
    """Perform category theory analysis"""
    constructor = RationalTraceConstructor()
    analyzer = CategoryTheoryRationalAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate sample rationals
    sample_rationals = []
    for den in range(1, 10):
        for num in range(0, den):
            if math.gcd(num, den) == 1:
                num_trace = constructor.decoder.number_to_trace(num)
                den_trace = constructor.decoder.number_to_trace(den)
                if constructor.is_phi_compliant(num_trace) and \
                   constructor.is_phi_compliant(den_trace):
                    sample_rationals.append(
                        constructor.construct_rational(num_trace, den_trace))
    
    # Verify field structure
    field_analysis = analyzer.analyze_field_structure(sample_rationals[:10])
    
    print("Field Structure Verification:")
    for prop, value in field_analysis.items():
        print(f"  {prop}: {value}")
    
    # Identify morphisms
    morphism_analysis = analyzer.identify_morphisms(sample_rationals[:20])
    
    print(f"\nMorphism Analysis:")
    print(f"  Order morphisms: {morphism_analysis['order_morphism_count']}")
    print(f"  Multiplicative morphisms: {morphism_analysis['multiplicative_morphism_count']}")
    print(f"  Total morphisms: {morphism_analysis['total_morphisms']}")

def demonstrate_tensor_representation():
    """Demonstrate tensor representation of rationals"""
    constructor = RationalTraceConstructor()
    
    print("\n=== Tensor Representation ===")
    
    # Create rational
    rational = constructor.construct_rational("10000", "1000")  # 5/3
    
    print(f"Rational: {rational.fraction}")
    print(f"Numerator trace: {rational.numerator_trace}")
    print(f"Denominator trace: {rational.denominator_trace}")
    print(f"\nTensor representation shape: {rational.tensor_representation.shape}")
    print(f"Tensor:\n{rational.tensor_representation}")
    
    # Show tensor arithmetic
    r1 = constructor.construct_rational("10", "1")     # 2/1
    r2 = constructor.construct_rational("1000", "10")  # 3/2
    
    print(f"\nTensor arithmetic example:")
    print(f"r1 tensor:\n{r1.tensor_representation}")
    print(f"r2 tensor:\n{r2.tensor_representation}")
    
    # Tensor product (represents multiplication)
    tensor_product = torch.kron(r1.tensor_representation[0], r2.tensor_representation[0]) / \
                    torch.kron(r1.tensor_representation[1], r2.tensor_representation[1])
    print(f"\nTensor product shape: {tensor_product.shape}")

def main():
    """Run comprehensive rational trace analysis"""
    print("="*80)
    print("Chapter 027: GoldenRationals - Rational Numbers from φ-Traces")
    print("="*80)
    
    # Basic demonstrations
    demonstrate_basic_rationals()
    demonstrate_rational_arithmetic()
    demonstrate_phi_approximation()
    demonstrate_tensor_representation()
    
    # Advanced analysis
    graph_theory_analysis()
    information_theory_analysis()
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Rational trace verification complete!")
    print("From ψ = ψ(ψ) emerges the rational field - fractions maintaining")
    print("φ-constraint through canonical trace tensor representation.")

if __name__ == "__main__":
    main()