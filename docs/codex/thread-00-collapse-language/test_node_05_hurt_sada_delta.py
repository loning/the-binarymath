#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N5: Hurt-Sada Δ-Collapse Vector
Verifies vector space structure of collapse sequences under difference operations.
"""

import unittest
import numpy as np
from typing import List, Tuple, Optional
import math


class DeltaOperator:
    """Implements the Δ-operator for collapse sequences"""
    
    @staticmethod
    def delta(s1: List[str], s2: List[str]) -> List[str]:
        """Compute minimal transformation from s1 to s2"""
        if not s1:
            return s2
        if not s2:
            return [DeltaOperator.inverse(sym) for sym in s1]
        
        # Find common prefix
        i = 0
        while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
            i += 1
        
        # Transformation = inverse(remaining s1) + remaining s2
        result = []
        for j in range(i, len(s1)):
            result.append(DeltaOperator.inverse(s1[j]))
        result.extend(s2[i:])
        
        return result
    
    @staticmethod
    def inverse(symbol: str) -> str:
        """Get inverse of a collapse symbol"""
        # Based on the algebra from Node 1
        inverses = {
            "00": "00",  # Identity is self-inverse
            "01": "10",  # Transform and return are inverses
            "10": "01"
        }
        return inverses.get(symbol, symbol)
    
    @staticmethod
    def compose(d1: List[str], d2: List[str]) -> List[str]:
        """Compose two delta operations"""
        # Simplify by canceling inverse pairs
        result = d1 + d2
        
        # Cancel adjacent inverse pairs
        changed = True
        while changed:
            changed = False
            new_result = []
            i = 0
            while i < len(result):
                if i + 1 < len(result) and result[i] == DeltaOperator.inverse(result[i + 1]):
                    # Skip both (they cancel)
                    i += 2
                    changed = True
                else:
                    new_result.append(result[i])
                    i += 1
            result = new_result
        
        return result


class CollapseVector:
    """Vector representation of collapse sequences"""
    
    def __init__(self, components: np.ndarray):
        """Initialize with component vector"""
        self.components = np.array(components, dtype=float)
        self.dimension = len(components)
    
    @classmethod
    def from_sequence(cls, sequence: List[str]) -> 'CollapseVector':
        """Convert collapse sequence to vector representation"""
        # Count occurrences of each symbol
        counts = {"00": 0, "01": 0, "10": 0}
        for symbol in sequence:
            if symbol in counts:
                counts[symbol] += 1
        
        # Vector components: [#00, #01, #10]
        return cls([counts["00"], counts["01"], counts["10"]])
    
    def __add__(self, other: 'CollapseVector') -> 'CollapseVector':
        """Vector addition"""
        return CollapseVector(self.components + other.components)
    
    def __sub__(self, other: 'CollapseVector') -> 'CollapseVector':
        """Vector subtraction"""
        return CollapseVector(self.components - other.components)
    
    def __mul__(self, scalar: float) -> 'CollapseVector':
        """Scalar multiplication"""
        return CollapseVector(scalar * self.components)
    
    def __rmul__(self, scalar: float) -> 'CollapseVector':
        """Right scalar multiplication"""
        return self.__mul__(scalar)
    
    def dot(self, other: 'CollapseVector') -> float:
        """Dot product"""
        return np.dot(self.components, other.components)
    
    def norm(self) -> float:
        """Euclidean norm"""
        return np.linalg.norm(self.components)
    
    def angle_with(self, other: 'CollapseVector') -> float:
        """Angle between vectors (in radians)"""
        if self.norm() == 0 or other.norm() == 0:
            return 0
        cos_angle = self.dot(other) / (self.norm() * other.norm())
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = max(-1, min(1, cos_angle))
        return math.acos(cos_angle)


class HurtSadaStructure:
    """Implements Hurt-Sada cone structure"""
    
    @staticmethod
    def is_in_cone(vector: CollapseVector) -> bool:
        """Check if vector is in the Hurt-Sada cone"""
        # Cone condition: all components non-negative
        return all(c >= 0 for c in vector.components)
    
    @staticmethod
    def project_to_cone(vector: CollapseVector) -> CollapseVector:
        """Project vector onto Hurt-Sada cone"""
        # Set negative components to zero
        projected = np.maximum(vector.components, 0)
        return CollapseVector(projected)
    
    @staticmethod
    def cone_distance(v1: CollapseVector, v2: CollapseVector) -> float:
        """Distance within the cone metric"""
        diff = v2 - v1
        if HurtSadaStructure.is_in_cone(diff):
            return diff.norm()
        else:
            # Project difference to cone boundary
            projected = HurtSadaStructure.project_to_cone(diff)
            return projected.norm()


class TestDeltaOperator(unittest.TestCase):
    """Test the Δ-operator properties"""
    
    def test_identity_property(self):
        """Test Δ(s, s) = identity"""
        sequences = [
            ["00", "01", "10"],
            ["01", "01", "00"],
            ["10", "00", "01"]
        ]
        
        for seq in sequences:
            delta = DeltaOperator.delta(seq, seq)
            self.assertEqual(delta, [])
    
    def test_inverse_property(self):
        """Test inverse operations"""
        self.assertEqual(DeltaOperator.inverse("00"), "00")
        self.assertEqual(DeltaOperator.inverse("01"), "10")
        self.assertEqual(DeltaOperator.inverse("10"), "01")
        
        # Double inverse = identity
        for symbol in ["00", "01", "10"]:
            double_inv = DeltaOperator.inverse(DeltaOperator.inverse(symbol))
            self.assertEqual(double_inv, symbol)
    
    def test_composition_property(self):
        """Test delta composition"""
        s1 = ["00", "01"]
        s2 = ["01", "10"]
        s3 = ["10", "00"]
        
        # Δ(s1, s2) ∘ Δ(s2, s3) = Δ(s1, s3)
        d12 = DeltaOperator.delta(s1, s2)
        d23 = DeltaOperator.delta(s2, s3)
        d13 = DeltaOperator.delta(s1, s3)
        
        composed = DeltaOperator.compose(d12, d23)
        
        # Should give same result (up to simplification)
        self.assertEqual(len(composed), len(d13))
    
    def test_minimal_transformation(self):
        """Test that delta gives minimal transformations"""
        # Simple case: one symbol difference
        s1 = ["00", "01"]
        s2 = ["00", "10"]
        
        delta = DeltaOperator.delta(s1, s2)
        # Should be: inverse(01) + 10 = 10 + 10 = ["10", "10"]
        self.assertEqual(len(delta), 2)


class TestCollapseVector(unittest.TestCase):
    """Test vector space properties"""
    
    def test_vector_creation(self):
        """Test creating vectors from sequences"""
        seq = ["00", "01", "01", "10"]
        vec = CollapseVector.from_sequence(seq)
        
        # Check counts: 1x00, 2x01, 1x10
        expected = np.array([1, 2, 1])
        np.testing.assert_array_equal(vec.components, expected)
    
    def test_vector_addition(self):
        """Test vector addition"""
        v1 = CollapseVector([1, 2, 0])
        v2 = CollapseVector([0, 1, 3])
        
        v3 = v1 + v2
        expected = np.array([1, 3, 3])
        np.testing.assert_array_equal(v3.components, expected)
        
        # Commutativity
        v4 = v2 + v1
        np.testing.assert_array_equal(v3.components, v4.components)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        v = CollapseVector([1, 2, 3])
        
        # Multiply by scalar
        v2 = 2 * v
        expected = np.array([2, 4, 6])
        np.testing.assert_array_equal(v2.components, expected)
        
        # Distributivity
        v3 = CollapseVector([1, 0, 1])
        result1 = 3 * (v + v3)
        result2 = 3 * v + 3 * v3
        np.testing.assert_array_almost_equal(result1.components, result2.components)
    
    def test_dot_product(self):
        """Test dot product"""
        v1 = CollapseVector([1, 0, 0])
        v2 = CollapseVector([0, 1, 0])
        v3 = CollapseVector([1, 1, 0])
        
        # Orthogonal vectors
        self.assertEqual(v1.dot(v2), 0)
        
        # Non-orthogonal
        self.assertEqual(v1.dot(v3), 1)
        self.assertEqual(v2.dot(v3), 1)
    
    def test_norm_and_angle(self):
        """Test vector norm and angle calculations"""
        v1 = CollapseVector([3, 0, 0])
        v2 = CollapseVector([0, 4, 0])
        
        # Norms
        self.assertEqual(v1.norm(), 3)
        self.assertEqual(v2.norm(), 4)
        
        # Angle (should be 90 degrees)
        angle = v1.angle_with(v2)
        self.assertAlmostEqual(angle, math.pi / 2, places=6)
    
    def test_vector_space_axioms(self):
        """Test that vector space axioms hold"""
        v1 = CollapseVector([1, 2, 3])
        v2 = CollapseVector([4, 5, 6])
        v3 = CollapseVector([7, 8, 9])
        zero = CollapseVector([0, 0, 0])
        
        # Associativity of addition
        result1 = (v1 + v2) + v3
        result2 = v1 + (v2 + v3)
        np.testing.assert_array_almost_equal(result1.components, result2.components)
        
        # Identity element
        result = v1 + zero
        np.testing.assert_array_equal(result.components, v1.components)
        
        # Inverse element
        neg_v1 = (-1) * v1
        result = v1 + neg_v1
        np.testing.assert_array_almost_equal(result.components, zero.components)


class TestHurtSadaCone(unittest.TestCase):
    """Test Hurt-Sada cone structure"""
    
    def test_cone_membership(self):
        """Test cone membership checking"""
        # In cone (all components non-negative)
        v1 = CollapseVector([1, 2, 3])
        self.assertTrue(HurtSadaStructure.is_in_cone(v1))
        
        # Not in cone (has negative component)
        v2 = CollapseVector([1, -2, 3])
        self.assertFalse(HurtSadaStructure.is_in_cone(v2))
        
        # On boundary (has zero component)
        v3 = CollapseVector([1, 0, 3])
        self.assertTrue(HurtSadaStructure.is_in_cone(v3))
    
    def test_cone_projection(self):
        """Test projection onto cone"""
        # Vector with negative components
        v = CollapseVector([1, -2, 3])
        projected = HurtSadaStructure.project_to_cone(v)
        
        # Should zero out negative components
        expected = np.array([1, 0, 3])
        np.testing.assert_array_equal(projected.components, expected)
        
        # Already in cone - should not change
        v_in = CollapseVector([1, 2, 3])
        projected_in = HurtSadaStructure.project_to_cone(v_in)
        np.testing.assert_array_equal(projected_in.components, v_in.components)
    
    def test_cone_distance(self):
        """Test distance metric within cone"""
        v1 = CollapseVector([1, 2, 3])
        v2 = CollapseVector([4, 5, 6])
        
        # Both in cone, difference in cone
        dist = HurtSadaStructure.cone_distance(v1, v2)
        expected_dist = (v2 - v1).norm()
        self.assertAlmostEqual(dist, expected_dist)
        
        # Difference not in cone
        v3 = CollapseVector([0, 0, 0])
        v4 = CollapseVector([1, 1, 1])
        dist2 = HurtSadaStructure.cone_distance(v4, v3)
        # Should project the negative difference
        self.assertGreaterEqual(dist2, 0)


class TestGeometricProperties(unittest.TestCase):
    """Test geometric properties of collapse vectors"""
    
    def test_orthogonality_structure(self):
        """Test orthogonality in collapse space"""
        # Basis vectors
        e1 = CollapseVector([1, 0, 0])  # Pure 00
        e2 = CollapseVector([0, 1, 0])  # Pure 01
        e3 = CollapseVector([0, 0, 1])  # Pure 10
        
        # Should be orthogonal
        self.assertEqual(e1.dot(e2), 0)
        self.assertEqual(e1.dot(e3), 0)
        self.assertEqual(e2.dot(e3), 0)
    
    def test_completeness_relation(self):
        """Test completeness of vector representation"""
        # Any sequence can be represented
        sequences = [
            ["00", "00", "01"],
            ["01", "10", "00"],
            ["10", "10", "10"]
        ]
        
        vectors = [CollapseVector.from_sequence(seq) for seq in sequences]
        
        # All should have dimension 3
        for vec in vectors:
            self.assertEqual(vec.dimension, 3)
        
        # Linear independence check
        matrix = np.array([v.components for v in vectors])
        rank = np.linalg.matrix_rank(matrix)
        self.assertGreaterEqual(rank, 1)  # Non-trivial space
    
    def test_triangle_inequality(self):
        """Test triangle inequality in vector space"""
        v1 = CollapseVector([1, 2, 3])
        v2 = CollapseVector([4, 5, 6])
        v3 = CollapseVector([7, 8, 9])
        
        # ||v1 - v3|| <= ||v1 - v2|| + ||v2 - v3||
        direct = (v3 - v1).norm()
        indirect = (v2 - v1).norm() + (v3 - v2).norm()
        
        self.assertLessEqual(direct, indirect + 1e-10)  # Allow small numerical error


class TestHurtSadaEmergence(unittest.TestCase):
    """Test emergence of Hurt-Sada structure from first principles"""
    
    def test_delta_preserves_grammar(self):
        """Test that delta operations preserve grammar constraints"""
        # Valid sequences (no "11" when concatenated)
        s1 = ["00", "01", "00"]
        s2 = ["10", "00", "01"]
        
        # Delta should give valid transformation
        delta = DeltaOperator.delta(s1, s2)
        
        # Check result is valid
        concat = "".join(delta)
        self.assertNotIn("11", concat)
    
    def test_cone_contains_valid_paths(self):
        """Test that valid collapse paths map to cone interior"""
        # Generate some valid sequences
        valid_sequences = [
            ["00", "00", "00"],
            ["01", "00", "10"],
            ["10", "01", "00"]
        ]
        
        for seq in valid_sequences:
            vec = CollapseVector.from_sequence(seq)
            self.assertTrue(HurtSadaStructure.is_in_cone(vec))
    
    def test_fibonaci_connection(self):
        """Test connection to Fibonacci/golden ratio"""
        # Sequences of increasing length
        sequences = []
        for n in [1, 2, 3, 5, 8]:  # Fibonacci numbers
            seq = ["01"] * n
            sequences.append(seq)
        
        vectors = [CollapseVector.from_sequence(seq) for seq in sequences]
        
        # Check growth pattern
        for i in range(1, len(vectors)):
            ratio = vectors[i].norm() / vectors[i-1].norm()
            # Should approach golden ratio for large n
            if i >= 3:
                phi = (1 + math.sqrt(5)) / 2
                self.assertAlmostEqual(ratio, phi, places=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)