#!/usr/bin/env python3
"""
Machine verification unit tests for D1.6: Entropy Definition
Testing the formal definition of entropy in self-referential complete systems.
Based on the verified formula: H = ln(|S_t|)
"""

import unittest
import math
from typing import Set, List, Dict, Any


class EntropySystem:
    """System for testing entropy properties"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.ln_phi = math.log(self.phi)   # ln(φ) ≈ 0.4812
    
    def compute_entropy(self, S_t: Set[str]) -> float:
        """
        Compute information entropy of a state set
        
        H(S_t) = ln(|S_t|)
        """
        if len(S_t) == 0:
            return 0.0
        
        return math.log(len(S_t))
    
    def compute_entropy_increase(self, S_t: Set[str], S_t_plus_1: Set[str]) -> float:
        """
        Compute entropy increase
        
        ΔH = ln(|S_{t+1}|/|S_t|) = ln(growth rate)
        """
        if len(S_t) == 0:
            return self.compute_entropy(S_t_plus_1)
        
        growth_rate = len(S_t_plus_1) / len(S_t)
        return math.log(growth_rate)
    
    def verify_non_negativity(self, state_sets: List[Set[str]]) -> bool:
        """Verify H(S_t) ≥ 0 for all state sets"""
        for S_t in state_sets:
            if self.compute_entropy(S_t) < 0:
                return False
        return True
    
    def verify_monotonicity(self, S1: Set[str], S2: Set[str]) -> bool:
        """Verify if S1 ⊂ S2 then H(S1) < H(S2)"""
        if S1 == S2:
            return True
        
        if S1.issubset(S2) and S1 != S2:
            H1 = self.compute_entropy(S1)
            H2 = self.compute_entropy(S2)
            return H1 < H2
        
        return True
    
    def verify_entropy_increase_bound(self, S_t: Set[str], S_t_plus_1: Set[str]) -> bool:
        """Verify entropy increase equals ln(growth rate)"""
        if len(S_t) == 0:
            return True
        
        # Compute actual entropy increase
        H_t = self.compute_entropy(S_t)
        H_t_plus_1 = self.compute_entropy(S_t_plus_1)
        actual_increase = H_t_plus_1 - H_t
        
        # Compute theoretical value
        growth_rate = len(S_t_plus_1) / len(S_t)
        theoretical_increase = math.log(growth_rate)
        
        # Check equality (within numerical tolerance)
        return abs(actual_increase - theoretical_increase) < 1e-10
    
    def evolve_with_no11_constraint(self, S_t: Set[str]) -> Set[str]:
        """Evolve system respecting no-11 constraint"""
        S_next = set()
        
        for s in S_t:
            # Always can append 0
            S_next.add(s + '0')
            
            # Can append 1 only if last bit is not 1
            if not s or s[-1] != '1':
                S_next.add(s + '1')
        
        return S_next
    
    def generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate Fibonacci sequence"""
        if n <= 0:
            return []
        if n == 1:
            return [1]
        
        fib = [1, 1]  # F_1 = 1, F_2 = 1 (standard)
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        
        return fib
    
    def verify_fibonacci_growth(self, evolution_sequence: List[Set[str]]) -> bool:
        """Verify state count follows Fibonacci sequence"""
        sizes = [len(S) for S in evolution_sequence]
        
        # Starting from {0, 1}, sizes should be 2, 3, 5, 8, 13, ...
        # which is F_3, F_4, F_5, F_6, F_7, ...
        expected = self.generate_fibonacci_sequence(len(sizes) + 2)[2:]
        
        for i, (actual, expected_val) in enumerate(zip(sizes, expected)):
            if actual != expected_val:
                return False
        
        return True
    
    def shannon_to_natural_conversion(self, H_shannon_bits: float) -> float:
        """Convert Shannon entropy (bits) to natural units (nats)"""
        return H_shannon_bits * math.log(2)
    
    def natural_to_shannon_conversion(self, H_nats: float) -> float:
        """Convert natural entropy (nats) to Shannon units (bits)"""
        return H_nats / math.log(2)


class TestEntropyDefinition(unittest.TestCase):
    """Unit tests for D1.6: Entropy Definition"""
    
    def setUp(self):
        self.system = EntropySystem()
        # Test state sets
        self.test_sets = [
            set(),  # Empty set
            {'0'},  # Single state
            {'0', '1'},  # Two states
            {'00', '01', '10'},  # Three states (no '11')
            {'000', '001', '010', '100', '101'},  # Five states
        ]
    
    def test_entropy_computation(self):
        """Test basic entropy computation H = ln(|S_t|)"""
        # Test specific values
        self.assertEqual(self.system.compute_entropy(set()), 0.0)
        self.assertEqual(self.system.compute_entropy({'0'}), 0.0)  # ln(1) = 0
        self.assertAlmostEqual(self.system.compute_entropy({'0', '1'}), math.log(2))
        self.assertAlmostEqual(self.system.compute_entropy({'0', '1', '2'}), math.log(3))
        
        # Test general formula
        for n in range(1, 10):
            S = {str(i) for i in range(n)}
            H = self.system.compute_entropy(S)
            self.assertAlmostEqual(H, math.log(n))
    
    def test_non_negativity(self):
        """Test Property D1.6.1: Non-negativity"""
        self.assertTrue(self.system.verify_non_negativity(self.test_sets))
        
        # Test with larger sets
        large_sets = [
            {bin(i)[2:] for i in range(100)},
            {str(i) for i in range(1000)}
        ]
        self.assertTrue(self.system.verify_non_negativity(large_sets))
    
    def test_monotonicity(self):
        """Test Property D1.6.2: Monotonicity"""
        # Test with nested sets
        S1 = {'0', '1'}
        S2 = {'0', '1', '00'}
        S3 = {'0', '1', '00', '01', '10'}
        
        self.assertTrue(self.system.verify_monotonicity(S1, S2))
        self.assertTrue(self.system.verify_monotonicity(S2, S3))
        self.assertTrue(self.system.verify_monotonicity(S1, S3))
    
    def test_entropy_increase_formula(self):
        """Test Property D1.6.4: Entropy increase equals ln(growth rate)"""
        # Test various transitions
        test_cases = [
            ({'0', '1'}, {'00', '01', '10'}),  # 2 → 3
            ({'00', '01', '10'}, {'000', '001', '010', '100', '101'}),  # 3 → 5
            ({'a'}, {'a0', 'a1'}),  # 1 → 2
        ]
        
        for S_t, S_t_plus_1 in test_cases:
            self.assertTrue(
                self.system.verify_entropy_increase_bound(S_t, S_t_plus_1)
            )
    
    def test_fibonacci_evolution(self):
        """Test that no-11 constraint leads to Fibonacci growth"""
        # Start with {0, 1}
        S_t = {'0', '1'}
        evolution = [S_t]
        
        # Evolve for several steps
        for _ in range(10):
            S_t = self.system.evolve_with_no11_constraint(S_t)
            evolution.append(S_t)
        
        # Verify Fibonacci growth
        self.assertTrue(self.system.verify_fibonacci_growth(evolution))
        
        # Verify entropy increases
        for i in range(len(evolution) - 1):
            H_i = self.system.compute_entropy(evolution[i])
            H_next = self.system.compute_entropy(evolution[i + 1])
            self.assertGreater(H_next, H_i)
    
    def test_entropy_increase_bound(self):
        """Test that entropy increase approaches ln(φ)"""
        # Evolve system and track entropy increases
        S_t = {'0', '1'}
        increases = []
        
        for _ in range(20):
            S_next = self.system.evolve_with_no11_constraint(S_t)
            delta_H = self.system.compute_entropy_increase(S_t, S_next)
            increases.append(delta_H)
            S_t = S_next
        
        # Later increases should approach ln(φ)
        last_increases = increases[-5:]
        for delta_H in last_increases:
            # Should be close to ln(φ) ≈ 0.4812
            self.assertAlmostEqual(delta_H, self.system.ln_phi, places=3)
    
    def test_shannon_entropy_relation(self):
        """Test Property D1.6.5: Relation to Shannon entropy"""
        # For uniform distribution over n states:
        # Shannon: H_S = log₂(n) bits
        # Natural: H_N = ln(n) nats
        # Relation: H_N = H_S × ln(2)
        
        for n in [2, 4, 8, 16, 32]:
            S = {str(i) for i in range(n)}
            
            # Our entropy (natural units)
            H_natural = self.system.compute_entropy(S)
            
            # Shannon entropy
            H_shannon_bits = math.log2(n)
            
            # Convert and compare
            H_shannon_converted = self.system.shannon_to_natural_conversion(H_shannon_bits)
            self.assertAlmostEqual(H_natural, H_shannon_converted)
            
            # Reverse conversion
            H_natural_to_bits = self.system.natural_to_shannon_conversion(H_natural)
            self.assertAlmostEqual(H_natural_to_bits, H_shannon_bits)
    
    def test_thermodynamic_correspondence(self):
        """Test correspondence with Boltzmann entropy"""
        # S = k_B ln(Ω)
        # With k_B = 1 (natural units), S = ln(Ω)
        # Where Ω is number of microstates
        
        # Our entropy with state count as "microstates"
        for n in [1, 10, 100, 1000]:
            S = {str(i) for i in range(n)}
            H = self.system.compute_entropy(S)
            
            # Boltzmann entropy (k_B = 1)
            S_boltzmann = math.log(n) if n > 0 else 0
            
            self.assertAlmostEqual(H, S_boltzmann)
    
    def test_specific_examples(self):
        """Test specific calculation examples"""
        # Example 1: Empty set
        self.assertEqual(self.system.compute_entropy(set()), 0.0)
        
        # Example 2: Two states
        S2 = {'0', '1'}
        H2 = self.system.compute_entropy(S2)
        self.assertAlmostEqual(H2, math.log(2), places=10)
        self.assertAlmostEqual(H2, 0.693147, places=5)
        
        # Example 3: Three states
        S3 = {'00', '01', '10'}
        H3 = self.system.compute_entropy(S3)
        self.assertAlmostEqual(H3, math.log(3), places=10)
        self.assertAlmostEqual(H3, 1.098612, places=5)
        
        # Example 4: Entropy increase
        delta_H = H3 - H2
        growth_rate = 3 / 2
        self.assertAlmostEqual(delta_H, math.log(growth_rate), places=10)
        self.assertAlmostEqual(delta_H, 0.405465, places=5)
    
    def test_long_term_behavior(self):
        """Test long-term entropy growth rate"""
        # Evolve for many steps
        S_t = {'0', '1'}
        growth_rates = []
        
        for _ in range(50):
            S_next = self.system.evolve_with_no11_constraint(S_t)
            if len(S_t) > 0:
                rate = len(S_next) / len(S_t)
                growth_rates.append(rate)
            S_t = S_next
        
        # Check convergence to golden ratio
        last_rates = growth_rates[-10:]
        for rate in last_rates:
            self.assertAlmostEqual(rate, self.system.phi, places=4)
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency of the entropy formula"""
        # Test that H(S1 ∪ S2) = ln(|S1 ∪ S2|) when S1 ∩ S2 = ∅
        S1 = {'00', '01'}
        S2 = {'10', '11'}  # Note: '11' included for this test
        
        self.assertEqual(S1.intersection(S2), set())  # Disjoint
        
        H_union = self.system.compute_entropy(S1.union(S2))
        expected = math.log(len(S1) + len(S2))
        
        self.assertAlmostEqual(H_union, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)