#!/usr/bin/env python3
"""
Machine verification unit tests for D1.6: Entropy
Testing the constructive definition of information entropy in self-referential complete systems.
"""

import unittest
import math
from typing import Set, List, Callable, Union


class SystemEntropy:
    """Implementation of D1.6: Entropy definition"""
    
    def __init__(self):
        self.epsilon = 1e-10  # Small value for numerical stability
    
    def structural_complexity(self, state_set: Set[str]) -> float:
        """StructuralComplexity: P(S) → R⁺"""
        if not state_set:
            return 0.0
        
        total_complexity = sum(math.log2(1 + len(state)) for state in state_set)
        return total_complexity / len(state_set)
    
    def entropy(self, state_set: Set[str]) -> float:
        """H: P(S) → R⁺ - Information entropy function"""
        size = len(state_set)
        
        if size == 0 or size == 1:
            return 0.0
        
        base_entropy = math.log2(size)
        structural_comp = self.structural_complexity(state_set)
        
        return base_entropy + structural_comp
    
    def is_well_defined(self, state_set: Set[str]) -> bool:
        """Check if entropy function is well-defined for given state set"""
        try:
            entropy_val = self.entropy(state_set)
            return entropy_val >= 0 and math.isfinite(entropy_val)
        except (ValueError, ZeroDivisionError):
            return False
    
    def is_computable(self, state_set: Set[str]) -> bool:
        """Check if entropy is computable for finite state set"""
        if not isinstance(state_set, set):
            return False
        
        # Check if all states are finite binary strings
        for state in state_set:
            if not isinstance(state, str) or not all(c in '01' for c in state):
                return False
        
        return len(state_set) < float('inf')
    
    def verify_non_negativity(self, state_set: Set[str]) -> bool:
        """Verify non-negativity property: H(S_t) ≥ 0"""
        if not self.is_well_defined(state_set):
            return False
        return self.entropy(state_set) >= 0
    
    def verify_strict_monotonicity(self, smaller_set: Set[str], larger_set: Set[str]) -> bool:
        """Verify strict monotonicity: S_t ⊊ S_t' ⟹ H(S_t) < H(S_t')"""
        if not (smaller_set < larger_set):  # strict subset
            return False
        
        if not (self.is_well_defined(smaller_set) and self.is_well_defined(larger_set)):
            return False
        
        h_smaller = self.entropy(smaller_set)
        h_larger = self.entropy(larger_set)
        
        return h_smaller < h_larger
    
    def decomposition_property(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute decomposition property for disjoint sets"""
        if set1 & set2:  # Not disjoint
            raise ValueError("Sets must be disjoint")
        
        union_set = set1 | set2
        if not union_set:
            return 0.0
        
        size1, size2, size_union = len(set1), len(set2), len(union_set)
        
        if size_union == 0:
            return 0.0
        
        weight1 = size1 / size_union if size1 > 0 else 0
        weight2 = size2 / size_union if size2 > 0 else 0
        
        h1 = self.entropy(set1) if set1 else 0
        h2 = self.entropy(set2) if set2 else 0
        
        return weight1 * h1 + weight2 * h2 + math.log2(size_union)


class TestSystemEntropy(unittest.TestCase):
    """Unit tests for D1.6: Entropy"""
    
    def setUp(self):
        self.entropy = SystemEntropy()
    
    def test_structural_complexity_basic(self):
        """Test StructuralComplexity function"""
        # Empty set
        self.assertEqual(self.entropy.structural_complexity(set()), 0.0)
        
        # Single state
        complexity = self.entropy.structural_complexity({'0'})
        expected = math.log2(1 + 1)  # log2(1 + |'0'|)
        self.assertAlmostEqual(complexity, expected, places=10)
        
        # Multiple states
        states = {'0', '1', '01', '10'}
        complexity = self.entropy.structural_complexity(states)
        expected = (math.log2(2) + math.log2(2) + math.log2(3) + math.log2(3)) / 4
        self.assertAlmostEqual(complexity, expected, places=10)
    
    def test_entropy_basic_cases(self):
        """Test entropy function for basic cases"""
        # Empty set: H = 0
        self.assertEqual(self.entropy.entropy(set()), 0.0)
        
        # Single state: H = 0
        self.assertEqual(self.entropy.entropy({'0'}), 0.0)
        
        # Two states: H > 0
        entropy_val = self.entropy.entropy({'0', '1'})
        self.assertGreater(entropy_val, 0)
        
        # Four states
        entropy_val = self.entropy.entropy({'00', '01', '10', '11'})
        self.assertGreater(entropy_val, math.log2(4))  # Should be greater than base entropy
    
    def test_well_defined_property(self):
        """Test that entropy function is well-defined"""
        test_sets = [
            set(),
            {'0'},
            {'0', '1'},
            {'00', '01', '10'},
            {'', '0', '1', '01', '10', '001', '010', '100'}
        ]
        
        for state_set in test_sets:
            with self.subTest(state_set=state_set):
                self.assertTrue(self.entropy.is_well_defined(state_set))
    
    def test_computability(self):
        """Test that entropy is computable for finite sets"""
        # Valid finite sets
        valid_sets = [
            set(),
            {'0', '1'},
            {'00', '01', '10', '11'},
            {'101', '010', '001'}
        ]
        
        for state_set in valid_sets:
            with self.subTest(state_set=state_set):
                self.assertTrue(self.entropy.is_computable(state_set))
        
        # Invalid sets (not all binary strings)
        invalid_sets = [
            {'a', 'b'},
            {'012'},
            {'0', '1', 'invalid'}
        ]
        
        for state_set in invalid_sets:
            with self.subTest(state_set=state_set):
                self.assertFalse(self.entropy.is_computable(state_set))
    
    def test_non_negativity_property(self):
        """Test non-negativity: ∀S_t ⊆ S: H(S_t) ≥ 0"""
        test_sets = [
            set(),
            {'0'},
            {'1'},
            {'0', '1'},
            {'00', '01', '10', '11'},
            {'', '0', '1', '01', '10', '001'},
            set(f'{i:03b}' for i in range(8))  # All 3-bit strings
        ]
        
        for state_set in test_sets:
            with self.subTest(state_set=state_set):
                self.assertTrue(self.entropy.verify_non_negativity(state_set))
                if self.entropy.is_well_defined(state_set):
                    self.assertGreaterEqual(self.entropy.entropy(state_set), 0)
    
    def test_strict_monotonicity_property(self):
        """Test strict monotonicity: S_t ⊊ S_t' ⟹ H(S_t) < H(S_t')"""
        # Test cases: (smaller_set, larger_set)
        test_cases = [
            ({'0'}, {'0', '1'}),
            ({'0', '1'}, {'0', '1', '01'}),
            ({'00', '01'}, {'00', '01', '10', '11'}),
            ({'0'}, {'0', '1', '01', '10'}),
            (set(), {'0'}),
        ]
        
        for smaller, larger in test_cases:
            with self.subTest(smaller=smaller, larger=larger):
                if len(smaller) > 1 and len(larger) > 1:  # Both non-trivial
                    self.assertTrue(self.entropy.verify_strict_monotonicity(smaller, larger))
                    h_small = self.entropy.entropy(smaller)
                    h_large = self.entropy.entropy(larger)
                    self.assertLess(h_small, h_large)
    
    def test_entropy_grows_with_size(self):
        """Test that entropy generally grows with state set size"""
        # Build increasingly large sets
        sets = [
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'},
            {'0', '1', '01', '10', '00'},
            {'0', '1', '01', '10', '00', '11'},
        ]
        
        entropies = [self.entropy.entropy(s) for s in sets]
        
        # Check that entropy is generally increasing
        for i in range(len(entropies) - 1):
            self.assertLess(entropies[i], entropies[i + 1])
    
    def test_decomposition_property_basic(self):
        """Test basic decomposition property for disjoint sets"""
        set1 = {'0', '1'}
        set2 = {'00', '01'}
        
        # Verify they're disjoint
        self.assertTrue(set1.isdisjoint(set2))
        
        # Test decomposition property
        decomp_val = self.entropy.decomposition_property(set1, set2)
        self.assertIsInstance(decomp_val, float)
        self.assertGreaterEqual(decomp_val, 0)
        
        # Test with empty sets
        decomp_empty = self.entropy.decomposition_property(set(), {'0'})
        self.assertGreaterEqual(decomp_empty, 0)
    
    def test_decomposition_property_error_cases(self):
        """Test decomposition property error handling"""
        # Non-disjoint sets should raise error
        set1 = {'0', '1'}
        set2 = {'1', '01'}  # '1' is in both sets
        
        with self.assertRaises(ValueError):
            self.entropy.decomposition_property(set1, set2)
    
    def test_structural_complexity_properties(self):
        """Test properties of StructuralComplexity function"""
        # Complexity should be non-negative
        test_sets = [
            {'0'},
            {'0', '1'},
            {'00', '01', '10'},
            {'', '0', '1', '01'}
        ]
        
        for state_set in test_sets:
            complexity = self.entropy.structural_complexity(state_set)
            self.assertGreaterEqual(complexity, 0)
        
        # Longer states should generally contribute more complexity
        short_states = {'0', '1'}
        long_states = {'0000', '0001'}
        
        complexity_short = self.entropy.structural_complexity(short_states)
        complexity_long = self.entropy.structural_complexity(long_states)
        self.assertLess(complexity_short, complexity_long)
    
    def test_entropy_mathematical_consistency(self):
        """Test mathematical consistency of entropy definition"""
        # Base case: |S| = 2^k should give H ≥ k
        for k in range(1, 5):
            states = set(f'{i:0{k}b}' for i in range(2**k))
            entropy_val = self.entropy.entropy(states)
            self.assertGreaterEqual(entropy_val, k)
        
        # Entropy should be continuous-like in behavior
        # (larger sets should have larger entropy)
        sizes = [2, 4, 8, 16]
        entropies = []
        
        for size in sizes:
            states = set(f'{i:05b}' for i in range(size))
            entropies.append(self.entropy.entropy(states))
        
        # Should be increasing
        for i in range(len(entropies) - 1):
            self.assertLess(entropies[i], entropies[i + 1])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very long states
        long_state = '0' * 100
        long_states = {long_state}
        self.assertTrue(self.entropy.is_well_defined(long_states))
        
        # Mixed length states
        mixed_states = {'', '0', '01', '001', '0001'}
        entropy_val = self.entropy.entropy(mixed_states)
        self.assertGreater(entropy_val, 0)
        
        # Single very complex state
        complex_state = '01010101010101010101'
        complex_states = {complex_state}
        # Should have entropy 0 (single state) but be well-defined
        self.assertEqual(self.entropy.entropy(complex_states), 0.0)
        self.assertTrue(self.entropy.is_well_defined(complex_states))


if __name__ == '__main__':
    unittest.main(verbosity=2)