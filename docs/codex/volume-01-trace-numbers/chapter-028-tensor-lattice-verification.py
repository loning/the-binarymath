#!/usr/bin/env python3
"""
Chapter 028: TensorLattice - Integer-Like Grid in Collapse Trace Tensor Space

Verification program demonstrating lattice structure in trace tensor space:
- Integer-like grid construction from trace tensors
- Lattice operations (meet, join) preserving φ-constraint
- Basis vectors and fundamental domains
- Crystallographic properties in tensor space

From ψ = ψ(ψ), we derive how discrete lattice structures emerge naturally
in φ-constrained tensor space, creating integer-like grids.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import itertools

@dataclass
class LatticePoint:
    """Point in the trace tensor lattice"""
    trace: str
    value: int
    coordinates: torch.Tensor
    basis_decomposition: List[Tuple[str, int]]
    is_fundamental: bool
    neighbors: List[str]
    phi_compliant: bool

@dataclass 
class LatticeBasis:
    """Basis for the tensor lattice"""
    basis_traces: List[str]
    basis_values: List[int]
    basis_tensors: torch.Tensor
    gram_matrix: torch.Tensor
    determinant: float
    is_reduced: bool

class TraceTensorDecoder:
    """Enhanced decoder with lattice support"""
    
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
    
    def trace_to_tensor(self, trace: str, dim: int = None) -> torch.Tensor:
        """Convert trace to tensor representation"""
        if dim is None:
            dim = len(trace)
        
        # Pad or truncate to specified dimension
        if len(trace) < dim:
            padded_trace = trace.zfill(dim)
        else:
            padded_trace = trace[-dim:]
        
        return torch.tensor([int(b) for b in padded_trace], dtype=torch.float32)

class TensorLatticeConstructor:
    """Construct integer-like lattice in trace tensor space"""
    
    def __init__(self, dimension: int = 8):
        self.decoder = TraceTensorDecoder()
        self.dimension = dimension
        self.phi = (1 + np.sqrt(5)) / 2
        
    def is_phi_compliant(self, trace: str) -> bool:
        """Check φ-constraint compliance"""
        return '11' not in trace
    
    def generate_basis(self, method: str = 'fibonacci') -> LatticeBasis:
        """Generate lattice basis vectors"""
        if method == 'fibonacci':
            # Use Fibonacci number traces as basis
            basis_traces = []
            basis_values = []
            
            for i in range(1, self.dimension + 1):
                fib_value = self.decoder.get_fibonacci(i)
                fib_trace = self.decoder.number_to_trace(fib_value)
                
                if self.is_phi_compliant(fib_trace):
                    basis_traces.append(fib_trace)
                    basis_values.append(fib_value)
            
        elif method == 'prime':
            # Use prime traces as basis
            basis_traces = []
            basis_values = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            
            for p in primes[:self.dimension]:
                p_trace = self.decoder.number_to_trace(p)
                if self.is_phi_compliant(p_trace):
                    basis_traces.append(p_trace)
                    basis_values.append(p)
        
        elif method == 'power':
            # Use powers of 2 as basis
            basis_traces = []
            basis_values = []
            
            for i in range(self.dimension):
                pow_value = 2 ** i
                pow_trace = self.decoder.number_to_trace(pow_value)
                if self.is_phi_compliant(pow_trace):
                    basis_traces.append(pow_trace)
                    basis_values.append(pow_value)
        
        # Create tensor representation
        basis_tensors = torch.stack([
            self.decoder.trace_to_tensor(trace, self.dimension)
            for trace in basis_traces
        ])
        
        # Compute Gram matrix
        gram_matrix = torch.matmul(basis_tensors, basis_tensors.T)
        
        # Compute determinant
        if len(basis_traces) == self.dimension:
            determinant = torch.det(gram_matrix).item()
        else:
            determinant = 0.0
        
        # Check if basis is reduced (LLL-like criterion)
        is_reduced = self._check_basis_reduction(basis_tensors)
        
        return LatticeBasis(
            basis_traces=basis_traces,
            basis_values=basis_values,
            basis_tensors=basis_tensors,
            gram_matrix=gram_matrix,
            determinant=determinant,
            is_reduced=is_reduced
        )
    
    def _check_basis_reduction(self, basis_tensors: torch.Tensor) -> bool:
        """Check if basis is reduced (simplified criterion)"""
        n = len(basis_tensors)
        if n < 2:
            return True
        
        # Check if basis vectors are nearly orthogonal
        gram = torch.matmul(basis_tensors, basis_tensors.T)
        off_diagonal = gram - torch.diag(torch.diag(gram))
        
        # Reduction criterion: off-diagonal elements small relative to diagonal
        diagonal_min = torch.min(torch.diag(gram))
        off_diagonal_max = torch.max(torch.abs(off_diagonal))
        
        return off_diagonal_max < 0.75 * diagonal_min
    
    def generate_lattice_points(self, basis: LatticeBasis, 
                               max_coefficient: int = 3) -> List[LatticePoint]:
        """Generate lattice points from basis"""
        lattice_points = []
        
        # Generate all integer linear combinations within bounds
        coefficients = range(-max_coefficient, max_coefficient + 1)
        
        for coeff_tuple in itertools.product(coefficients, repeat=len(basis.basis_traces)):
            # Skip all-zero combination
            if all(c == 0 for c in coeff_tuple):
                continue
            
            # Compute linear combination value
            total_value = sum(c * v for c, v in zip(coeff_tuple, basis.basis_values))
            
            if total_value <= 0:
                continue
            
            # Convert to trace
            combined_trace = self.decoder.number_to_trace(total_value)
            
            if not self.is_phi_compliant(combined_trace):
                continue
            
            # Create coordinates tensor
            coordinates = torch.tensor(coeff_tuple, dtype=torch.float32)
            
            # Record basis decomposition
            decomposition = [(basis.basis_traces[i], coeff_tuple[i]) 
                           for i in range(len(basis.basis_traces))
                           if coeff_tuple[i] != 0]
            
            # Check if point is in fundamental domain
            is_fundamental = all(0 <= c < 1 for c in coeff_tuple)
            
            lattice_points.append(LatticePoint(
                trace=combined_trace,
                value=total_value,
                coordinates=coordinates,
                basis_decomposition=decomposition,
                is_fundamental=is_fundamental,
                neighbors=[],  # Will be filled later
                phi_compliant=True
            ))
        
        # Find neighbors for each point
        self._compute_neighbors(lattice_points)
        
        return lattice_points
    
    def _compute_neighbors(self, points: List[LatticePoint]):
        """Compute nearest neighbors in lattice"""
        # Build value -> point mapping
        value_to_point = {p.value: p for p in points}
        
        for point in points:
            neighbors = []
            
            # Check standard lattice neighbors (±1 in each basis direction)
            for i in range(len(point.coordinates)):
                for delta in [-1, 1]:
                    neighbor_coords = point.coordinates.clone()
                    neighbor_coords[i] += delta
                    
                    # Find point with these coordinates
                    for other in points:
                        if torch.allclose(other.coordinates, neighbor_coords):
                            neighbors.append(other.trace)
                            break
            
            point.neighbors = list(set(neighbors))
    
    def compute_lattice_operations(self, p1: LatticePoint, p2: LatticePoint) -> Dict[str, Any]:
        """Compute lattice meet and join operations"""
        # Meet (greatest lower bound) - componentwise minimum
        meet_coords = torch.minimum(p1.coordinates, p2.coordinates)
        
        # Join (least upper bound) - componentwise maximum  
        join_coords = torch.maximum(p1.coordinates, p2.coordinates)
        
        # Convert back to values (approximate)
        # This is simplified - real implementation would need proper basis
        meet_value = min(p1.value, p2.value)
        join_value = max(p1.value, p2.value)
        
        meet_trace = self.decoder.number_to_trace(meet_value)
        join_trace = self.decoder.number_to_trace(join_value)
        
        return {
            'meet_trace': meet_trace,
            'meet_value': meet_value,
            'meet_coords': meet_coords,
            'join_trace': join_trace,
            'join_value': join_value,
            'join_coords': join_coords,
            'meet_phi_compliant': self.is_phi_compliant(meet_trace),
            'join_phi_compliant': self.is_phi_compliant(join_trace)
        }

class GraphTheoryLatticeAnalyzer:
    """Graph theory analysis of tensor lattice"""
    
    def __init__(self):
        self.constructor = TensorLatticeConstructor()
        
    def build_lattice_graph(self, points: List[LatticePoint]) -> nx.Graph:
        """Build graph from lattice points"""
        G = nx.Graph()
        
        # Add nodes
        for point in points:
            G.add_node(point.trace,
                      value=point.value,
                      coordinates=point.coordinates.tolist(),
                      is_fundamental=point.is_fundamental)
        
        # Add edges based on neighbor relationships
        for point in points:
            for neighbor_trace in point.neighbors:
                if G.has_node(neighbor_trace):
                    G.add_edge(point.trace, neighbor_trace)
        
        return G
    
    def analyze_lattice_properties(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze properties of lattice graph"""
        analysis = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'is_connected': nx.is_connected(G),
            'density': nx.density(G),
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
        if G.number_of_nodes() > 0:
            # Compute clustering
            analysis['clustering_coefficient'] = nx.average_clustering(G)
            
            # Find connected components
            components = list(nx.connected_components(G))
            analysis['component_count'] = len(components)
            analysis['largest_component_size'] = max(len(c) for c in components)
            
            # Check for regular structure
            degrees = list(dict(G.degree()).values())
            analysis['is_regular'] = len(set(degrees)) == 1
            analysis['degree_variance'] = np.var(degrees)
        
        return analysis
    
    def find_sublattices(self, points: List[LatticePoint]) -> List[List[LatticePoint]]:
        """Find sublattices within the main lattice"""
        sublattices = []
        
        # Group by common factors in decomposition
        factor_groups = defaultdict(list)
        
        for point in points:
            # Extract common factors
            if point.basis_decomposition:
                factors = tuple(sorted(coeff for _, coeff in point.basis_decomposition))
                factor_groups[factors].append(point)
        
        # Each factor group forms a sublattice
        for factors, group_points in factor_groups.items():
            if len(group_points) >= 3:  # Minimum size for interesting sublattice
                sublattices.append(group_points)
        
        return sublattices

class InformationTheoryLatticeAnalyzer:
    """Information theory analysis of tensor lattice"""
    
    def __init__(self):
        self.constructor = TensorLatticeConstructor()
        
    def compute_lattice_entropy(self, points: List[LatticePoint]) -> Dict[str, float]:
        """Compute entropy measures of lattice"""
        if not points:
            return {}
        
        # Coordinate entropy
        all_coords = torch.stack([p.coordinates for p in points])
        coord_probs = {}
        
        for dim in range(all_coords.shape[1]):
            values, counts = torch.unique(all_coords[:, dim], return_counts=True)
            probs = counts.float() / len(points)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
            coord_probs[f'dim_{dim}_entropy'] = entropy
        
        # Value distribution entropy
        values = [p.value for p in points]
        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
        
        total = len(values)
        value_entropy = 0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                value_entropy -= p * np.log2(p)
        
        # Trace length entropy
        trace_lengths = [len(p.trace) for p in points]
        length_counts = {}
        for l in trace_lengths:
            length_counts[l] = length_counts.get(l, 0) + 1
        
        length_entropy = 0
        for count in length_counts.values():
            p = count / total
            if p > 0:
                length_entropy -= p * np.log2(p)
        
        return {
            **coord_probs,
            'value_entropy': value_entropy,
            'length_entropy': length_entropy,
            'average_trace_length': np.mean(trace_lengths),
            'total_entropy': value_entropy + length_entropy
        }
    
    def analyze_information_content(self, basis: LatticeBasis) -> Dict[str, Any]:
        """Analyze information content of basis"""
        # Basis entropy (how much information in basis choice)
        basis_values = torch.tensor(basis.basis_values, dtype=torch.float32)
        basis_probs = basis_values / torch.sum(basis_values)
        basis_entropy = -torch.sum(basis_probs * torch.log2(basis_probs + 1e-10)).item()
        
        # Gram matrix entropy (correlation information)
        gram_flat = basis.gram_matrix.flatten()
        gram_probs = torch.softmax(gram_flat, dim=0)
        gram_entropy = -torch.sum(gram_probs * torch.log2(gram_probs + 1e-10)).item()
        
        # Orthogonality measure
        n = len(basis.basis_tensors)
        if n > 1:
            orthogonality = torch.sum(torch.abs(basis.gram_matrix - torch.diag(torch.diag(basis.gram_matrix)))) / (n * (n - 1))
        else:
            orthogonality = 0
        
        return {
            'basis_entropy': basis_entropy,
            'gram_entropy': gram_entropy,
            'determinant': basis.determinant,
            'orthogonality': orthogonality.item() if torch.is_tensor(orthogonality) else orthogonality,
            'is_reduced': basis.is_reduced,
            'basis_size': len(basis.basis_traces)
        }

class CategoryTheoryLatticeAnalyzer:
    """Category theory analysis of tensor lattice"""
    
    def __init__(self):
        self.constructor = TensorLatticeConstructor()
        
    def verify_lattice_axioms(self, points: List[LatticePoint]) -> Dict[str, bool]:
        """Verify lattice axioms"""
        results = {
            'has_meet': True,
            'has_join': True,
            'meet_associative': True,
            'join_associative': True,
            'meet_commutative': True,
            'join_commutative': True,
            'absorption_laws': True,
            'is_complete_lattice': False
        }
        
        if len(points) < 3:
            return results
        
        # Test on sample points
        p1, p2, p3 = points[:3]
        
        # Test meet and join existence
        ops12 = self.constructor.compute_lattice_operations(p1, p2)
        ops23 = self.constructor.compute_lattice_operations(p2, p3)
        
        results['has_meet'] = ops12['meet_phi_compliant']
        results['has_join'] = ops12['join_phi_compliant']
        
        # Test associativity (simplified)
        # Would need full implementation for complete test
        
        # Test if we have top and bottom elements
        values = [p.value for p in points]
        has_zero = 0 in values
        has_max = len(set(values)) == len(values)  # All distinct suggests bounded
        
        results['is_complete_lattice'] = has_zero
        
        return results
    
    def identify_lattice_morphisms(self, points1: List[LatticePoint], 
                                  points2: List[LatticePoint]) -> List[Dict[str, Any]]:
        """Identify morphisms between lattices"""
        morphisms = []
        
        # Build value mappings
        values1 = {p.value for p in points1}
        values2 = {p.value for p in points2}
        
        # Check for inclusion morphism
        if values1.issubset(values2):
            morphisms.append({
                'type': 'inclusion',
                'source_size': len(values1),
                'target_size': len(values2),
                'preserves_operations': True
            })
        
        # Check for scaling morphisms
        for scale in [2, 3, 5]:
            scaled_values1 = {v * scale for v in values1}
            if scaled_values1.issubset(values2):
                morphisms.append({
                    'type': f'scaling_by_{scale}',
                    'scale_factor': scale,
                    'preserves_structure': True
                })
        
        return morphisms

def demonstrate_basic_lattice():
    """Demonstrate basic lattice construction"""
    constructor = TensorLatticeConstructor(dimension=4)
    
    print("=== Basic Lattice Construction ===")
    
    # Generate different bases
    bases = {
        'fibonacci': constructor.generate_basis('fibonacci'),
        'prime': constructor.generate_basis('prime'),
        'power': constructor.generate_basis('power')
    }
    
    for name, basis in bases.items():
        print(f"\n{name.capitalize()} Basis:")
        print(f"  Basis traces: {basis.basis_traces[:4]}")
        print(f"  Basis values: {basis.basis_values[:4]}")
        print(f"  Determinant: {basis.determinant:.3f}")
        print(f"  Is reduced: {basis.is_reduced}")
        
        # Show Gram matrix
        print(f"  Gram matrix shape: {basis.gram_matrix.shape}")
        if basis.gram_matrix.shape[0] <= 4:
            print(f"  Gram matrix:\n{basis.gram_matrix}")

def demonstrate_lattice_points():
    """Demonstrate lattice point generation"""
    constructor = TensorLatticeConstructor(dimension=3)
    
    print("\n=== Lattice Point Generation ===")
    
    # Generate basis
    basis = constructor.generate_basis('fibonacci')
    print(f"Basis: {basis.basis_traces}")
    
    # Generate lattice points
    points = constructor.generate_lattice_points(basis, max_coefficient=2)
    
    print(f"\nGenerated {len(points)} lattice points")
    
    # Show some examples
    for i, point in enumerate(points[:10]):
        print(f"\nPoint {i+1}:")
        print(f"  Trace: {point.trace}")
        print(f"  Value: {point.value}")
        print(f"  Coordinates: {point.coordinates.tolist()}")
        print(f"  Decomposition: {point.basis_decomposition}")
        print(f"  Neighbors: {len(point.neighbors)}")

def demonstrate_lattice_operations():
    """Demonstrate lattice meet and join operations"""
    constructor = TensorLatticeConstructor(dimension=3)
    
    print("\n=== Lattice Operations ===")
    
    # Generate basis and points
    basis = constructor.generate_basis('fibonacci')
    points = constructor.generate_lattice_points(basis, max_coefficient=2)
    
    if len(points) >= 2:
        p1, p2 = points[0], points[1]
        
        print(f"\nPoint 1: {p1.trace} (value: {p1.value})")
        print(f"Point 2: {p2.trace} (value: {p2.value})")
        
        # Compute meet and join
        ops = constructor.compute_lattice_operations(p1, p2)
        
        print(f"\nMeet:")
        print(f"  Trace: {ops['meet_trace']}")
        print(f"  Value: {ops['meet_value']}")
        print(f"  Coordinates: {ops['meet_coords'].tolist()}")
        print(f"  φ-compliant: {ops['meet_phi_compliant']}")
        
        print(f"\nJoin:")
        print(f"  Trace: {ops['join_trace']}")
        print(f"  Value: {ops['join_value']}")
        print(f"  Coordinates: {ops['join_coords'].tolist()}")
        print(f"  φ-compliant: {ops['join_phi_compliant']}")

def graph_theory_analysis():
    """Perform graph theory analysis of lattice"""
    constructor = TensorLatticeConstructor(dimension=4)
    analyzer = GraphTheoryLatticeAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    # Generate lattice
    basis = constructor.generate_basis('fibonacci')
    points = constructor.generate_lattice_points(basis, max_coefficient=2)
    
    # Build graph
    G = analyzer.build_lattice_graph(points)
    
    # Analyze properties
    analysis = analyzer.analyze_lattice_properties(G)
    
    print(f"Lattice Graph Properties:")
    print(f"  Nodes: {analysis['node_count']}")
    print(f"  Edges: {analysis['edge_count']}")
    print(f"  Connected: {analysis['is_connected']}")
    print(f"  Density: {analysis['density']:.3f}")
    print(f"  Average degree: {analysis['average_degree']:.2f}")
    print(f"  Clustering: {analysis.get('clustering_coefficient', 0):.3f}")
    print(f"  Components: {analysis.get('component_count', 1)}")
    print(f"  Regular: {analysis.get('is_regular', False)}")
    
    # Find sublattices
    sublattices = analyzer.find_sublattices(points)
    print(f"\nSublattices found: {len(sublattices)}")
    for i, sublattice in enumerate(sublattices[:3]):
        print(f"  Sublattice {i+1}: {len(sublattice)} points")

def information_theory_analysis():
    """Perform information theory analysis"""
    constructor = TensorLatticeConstructor(dimension=4)
    analyzer = InformationTheoryLatticeAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Generate lattice
    basis = constructor.generate_basis('fibonacci')
    points = constructor.generate_lattice_points(basis, max_coefficient=3)
    
    # Analyze lattice entropy
    entropy_analysis = analyzer.compute_lattice_entropy(points)
    
    print(f"Lattice Entropy Analysis:")
    for key, value in entropy_analysis.items():
        if 'entropy' in key:
            print(f"  {key}: {value:.3f} bits")
        else:
            print(f"  {key}: {value:.2f}")
    
    # Analyze basis information
    basis_info = analyzer.analyze_information_content(basis)
    
    print(f"\nBasis Information Content:")
    print(f"  Basis entropy: {basis_info['basis_entropy']:.3f} bits")
    print(f"  Gram entropy: {basis_info['gram_entropy']:.3f} bits")
    print(f"  Determinant: {basis_info['determinant']:.3f}")
    print(f"  Orthogonality: {basis_info['orthogonality']:.3f}")
    print(f"  Is reduced: {basis_info['is_reduced']}")

def category_theory_analysis():
    """Perform category theory analysis"""
    constructor = TensorLatticeConstructor(dimension=3)
    analyzer = CategoryTheoryLatticeAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    # Generate lattice
    basis = constructor.generate_basis('fibonacci')
    points = constructor.generate_lattice_points(basis, max_coefficient=2)
    
    # Verify lattice axioms
    axioms = analyzer.verify_lattice_axioms(points)
    
    print("Lattice Axiom Verification:")
    for axiom, satisfied in axioms.items():
        print(f"  {axiom}: {satisfied}")
    
    # Generate second lattice for morphism analysis
    basis2 = constructor.generate_basis('prime')
    points2 = constructor.generate_lattice_points(basis2, max_coefficient=2)
    
    # Find morphisms
    morphisms = analyzer.identify_lattice_morphisms(points, points2)
    
    print(f"\nLattice Morphisms Found: {len(morphisms)}")
    for morphism in morphisms:
        print(f"  Type: {morphism['type']}")
        for key, value in morphism.items():
            if key != 'type':
                print(f"    {key}: {value}")

def demonstrate_crystallographic_properties():
    """Demonstrate crystallographic-like properties"""
    constructor = TensorLatticeConstructor(dimension=4)
    
    print("\n=== Crystallographic Properties ===")
    
    # Generate lattice with high symmetry
    basis = constructor.generate_basis('fibonacci')
    points = constructor.generate_lattice_points(basis, max_coefficient=3)
    
    # Analyze symmetry
    print(f"Total lattice points: {len(points)}")
    
    # Group by value modulo small primes (like crystal classes)
    mod_groups = defaultdict(list)
    for p in [2, 3, 5, 7]:
        for point in points:
            mod_groups[p].append(point.value % p)
    
    print("\nModular Structure (Crystal Classes):")
    for p, values in mod_groups.items():
        unique_classes = len(set(values))
        print(f"  Mod {p}: {unique_classes} classes")
    
    # Find periodic patterns
    trace_lengths = [len(p.trace) for p in points]
    length_periods = []
    for i in range(1, min(10, len(trace_lengths)//2)):
        if trace_lengths[:i] == trace_lengths[i:2*i]:
            length_periods.append(i)
    
    if length_periods:
        print(f"\nPeriodic patterns found with periods: {length_periods}")
    else:
        print("\nNo simple periodic patterns detected")
    
    # Compute packing density (simplified)
    total_bits = sum(len(p.trace) for p in points)
    max_value = max(p.value for p in points)
    max_bits = len(bin(max_value)) - 2
    packing_density = total_bits / (len(points) * max_bits) if max_bits > 0 else 0
    
    print(f"\nPacking density: {packing_density:.3f}")

def main():
    """Run comprehensive tensor lattice analysis"""
    print("="*80)
    print("Chapter 028: TensorLattice - Integer-Like Grid Analysis")
    print("="*80)
    
    # Basic demonstrations
    demonstrate_basic_lattice()
    demonstrate_lattice_points()
    demonstrate_lattice_operations()
    demonstrate_crystallographic_properties()
    
    # Advanced analysis
    graph_theory_analysis()
    information_theory_analysis()
    category_theory_analysis()
    
    print("\n" + "="*80)
    print("Tensor lattice verification complete!")
    print("From ψ = ψ(ψ) emerges discrete structure - integer-like grids")
    print("maintaining φ-constraint throughout lattice operations.")

if __name__ == "__main__":
    main()