#!/usr/bin/env python3
"""
Machine verification unit tests for D1.4: Time Metric Definition
Testing the formal definition of time metric in self-referential complete systems.
"""

import unittest
from math import log2, sqrt
from typing import Any, List, Tuple, Set, Dict


class TimeMetricSystem:
    """System for testing time metric properties"""
    
    def __init__(self):
        self.phi = (1 + sqrt(5)) / 2  # Golden ratio
        self.epsilon = log2(self.phi)  # Minimum time quantum ≈ 0.694
        self.states = {}  # Cache for states and their entropies
    
    def compute_entropy(self, state: str) -> float:
        """Compute entropy H(s) = log2(|s|) for a state"""
        if not state:  # Empty state
            return 0.0
        return log2(len(state))
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """Compute minimum edit distance between two states"""
        # Implementation of Levenshtein distance
        if s1 == s2:
            return 0
        
        m, n = len(s1), len(s2)
        
        # Create distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def compute_time_metric(self, s: str, t: str) -> float:
        """
        Compute time metric τ(s,t) from state s to state t
        
        Definition:
        τ(s,t) = {
            0                    if s = t
            H(t) - H(s)         if H(t) > H(s)
            ε + d_E(s,t) * log2(φ)  otherwise
        }
        """
        if s == t:
            return 0.0
        
        H_s = self.compute_entropy(s)
        H_t = self.compute_entropy(t)
        
        if H_t > H_s:
            return H_t - H_s
        else:
            d_E = self.edit_distance(s, t)
            return self.epsilon + d_E * log2(self.phi)
    
    def verify_non_negativity(self, states: List[str]) -> bool:
        """Verify τ(s,t) ≥ 0 for all states"""
        for s in states:
            for t in states:
                if self.compute_time_metric(s, t) < 0:
                    return False
        return True
    
    def verify_identity(self, states: List[str]) -> bool:
        """Verify τ(s,t) = 0 iff s = t"""
        for s in states:
            for t in states:
                tau = self.compute_time_metric(s, t)
                if s == t:
                    if tau != 0:
                        return False
                else:
                    if tau == 0:
                        return False
        return True
    
    def verify_asymmetry(self, s: str, t: str) -> bool:
        """Check if τ(s,t) ≠ τ(t,s) for given states"""
        tau_st = self.compute_time_metric(s, t)
        tau_ts = self.compute_time_metric(t, s)
        return tau_st != tau_ts
    
    def verify_causality(self, s: str, t: str, u: str) -> bool:
        """Verify causality condition: τ(s,u) ≥ τ(s,t) + τ(t,u)"""
        tau_st = self.compute_time_metric(s, t)
        tau_tu = self.compute_time_metric(t, u)
        tau_su = self.compute_time_metric(s, u)
        
        # Allow for small numerical error
        return tau_su >= tau_st + tau_tu - 1e-10
    
    def verify_discreteness(self, states: List[str]) -> bool:
        """Verify τ(s,t) > 0 for all s ≠ t (and minimum non-zero value exists)"""
        min_nonzero_tau = float('inf')
        
        for s in states:
            for t in states:
                if s != t:
                    tau = self.compute_time_metric(s, t)
                    if tau <= 0:
                        return False
                    min_nonzero_tau = min(min_nonzero_tau, tau)
        
        # The minimum non-zero time should exist
        return min_nonzero_tau > 0 and min_nonzero_tau < float('inf')
    
    def verify_entropy_monotonicity(self, s: str, t: str) -> bool:
        """Verify that if H(t) > H(s), then τ(s,t) = H(t) - H(s)"""
        H_s = self.compute_entropy(s)
        H_t = self.compute_entropy(t)
        
        if H_t > H_s:
            tau = self.compute_time_metric(s, t)
            return abs(tau - (H_t - H_s)) < 1e-10
        return True
    
    def generate_evolution_sequence(self, initial: str, steps: int) -> List[str]:
        """Generate a sequence of states with increasing entropy"""
        sequence = [initial]
        current = initial
        
        for i in range(steps):
            # Add a bit to increase entropy
            current = current + str(i % 2)
            sequence.append(current)
        
        return sequence
    
    def verify_time_arrow(self, sequence: List[str]) -> bool:
        """Verify time arrow property in an evolution sequence"""
        for i in range(len(sequence) - 1):
            s_i = sequence[i]
            s_next = sequence[i + 1]
            
            # Forward time should be positive
            tau_forward = self.compute_time_metric(s_i, s_next)
            if tau_forward <= 0:
                return False
            
            # If entropy increases, verify specific value
            H_i = self.compute_entropy(s_i)
            H_next = self.compute_entropy(s_next)
            
            if H_next > H_i:
                expected = H_next - H_i
                if abs(tau_forward - expected) > 1e-10:
                    return False
        
        return True
    
    def verify_time_quantization(self, states: List[str]) -> bool:
        """Verify time values show discrete structure"""
        # Collect all non-zero time values
        time_values = set()
        
        for s in states:
            for t in states:
                tau = self.compute_time_metric(s, t)
                if tau > 0:
                    time_values.add(round(tau, 10))  # Round to avoid floating point issues
        
        # Time values should form a discrete set
        # The minimum positive value should exist
        if not time_values:
            return True
        
        min_tau = min(time_values)
        return min_tau > 0
    
    def create_causal_chain(self, length: int) -> List[str]:
        """Create a causal chain of states"""
        chain = ['0']  # Initial state
        
        for i in range(1, length):
            # Each state has more information than the previous
            prev = chain[-1]
            new_state = prev + str(i % 2)
            chain.append(new_state)
        
        return chain


class TestTimeMetricDefinition(unittest.TestCase):
    """Unit tests for D1.4: Time Metric Definition"""
    
    def setUp(self):
        self.system = TimeMetricSystem()
        # Test states of various lengths
        self.test_states = [
            '',      # Empty state
            '0',     # Single bit
            '1',
            '00',    # Two bits
            '01',
            '10',
            '11',
            '010',   # Three bits
            '101',
            '0101',  # Four bits
            '1010'
        ]
    
    def test_time_metric_computation(self):
        """Test basic time metric computation"""
        # Test identity
        for state in self.test_states:
            tau = self.system.compute_time_metric(state, state)
            self.assertEqual(tau, 0.0)
        
        # Test entropy increase case
        tau = self.system.compute_time_metric('0', '00')
        H_0 = self.system.compute_entropy('0')
        H_00 = self.system.compute_entropy('00')
        self.assertAlmostEqual(tau, H_00 - H_0)
        
        # Test general case
        tau = self.system.compute_time_metric('11', '00')
        self.assertGreaterEqual(tau, self.system.epsilon)
    
    def test_quasi_metric_axioms(self):
        """Test Property D1.4.1: Quasi-metric axioms"""
        # Non-negativity
        self.assertTrue(self.system.verify_non_negativity(self.test_states))
        
        # Identity
        self.assertTrue(self.system.verify_identity(self.test_states))
        
        # Test asymmetry exists
        asymmetry_found = False
        for s in self.test_states:
            for t in self.test_states:
                if s != t and self.system.verify_asymmetry(s, t):
                    asymmetry_found = True
                    break
            if asymmetry_found:
                break
        self.assertTrue(asymmetry_found)
    
    def test_causality_condition(self):
        """Test Property D1.4.2: Causality condition"""
        # Test with a causal chain
        chain = self.system.create_causal_chain(5)
        
        for i in range(len(chain) - 2):
            s = chain[i]
            t = chain[i + 1]
            u = chain[i + 2]
            
            # Since this is a causal chain with increasing entropy,
            # the causality condition should hold
            self.assertTrue(self.system.verify_causality(s, t, u))
    
    def test_discreteness(self):
        """Test Property D1.4.3: Discreteness"""
        self.assertTrue(self.system.verify_discreteness(self.test_states))
        
        # Find minimum non-zero time value
        min_tau = float('inf')
        for s in self.test_states:
            for t in self.test_states:
                if s != t:
                    tau = self.system.compute_time_metric(s, t)
                    self.assertGreater(tau, 0)  # All non-identical states have positive time
                    min_tau = min(min_tau, tau)
        
        # The minimum time quantum exists
        self.assertGreater(min_tau, 0)
        self.assertLess(min_tau, float('inf'))
    
    def test_entropy_monotonicity_relation(self):
        """Test Property D1.4.4: Entropy monotonicity relation"""
        for s in self.test_states:
            for t in self.test_states:
                self.assertTrue(self.system.verify_entropy_monotonicity(s, t))
    
    def test_time_quantization(self):
        """Test Property D1.4.5: Time quantization"""
        self.assertTrue(self.system.verify_time_quantization(self.test_states))
    
    def test_time_arrow(self):
        """Test time arrow in evolution sequences"""
        # Generate evolution sequence
        sequence = self.system.generate_evolution_sequence('0', 10)
        
        # Verify time arrow property
        self.assertTrue(self.system.verify_time_arrow(sequence))
        
        # Verify entropy always increases
        for i in range(len(sequence) - 1):
            H_i = self.system.compute_entropy(sequence[i])
            H_next = self.system.compute_entropy(sequence[i + 1])
            self.assertGreater(H_next, H_i)
    
    def test_edit_distance_properties(self):
        """Test edit distance computation"""
        # Identity
        self.assertEqual(self.system.edit_distance('010', '010'), 0)
        
        # Single substitution
        self.assertEqual(self.system.edit_distance('010', '011'), 1)
        
        # Single insertion/deletion
        self.assertEqual(self.system.edit_distance('01', '010'), 1)
        self.assertEqual(self.system.edit_distance('010', '01'), 1)
        
        # Multiple operations
        self.assertEqual(self.system.edit_distance('', '0101'), 4)
    
    def test_golden_ratio_constants(self):
        """Test golden ratio related constants"""
        # Verify φ value
        phi = self.system.phi
        self.assertAlmostEqual(phi * phi, phi + 1)  # φ² = φ + 1
        self.assertAlmostEqual(phi, (1 + sqrt(5)) / 2)
        
        # Verify ε value
        epsilon = self.system.epsilon
        self.assertAlmostEqual(epsilon, log2(phi))
        self.assertAlmostEqual(epsilon, 0.6942419136306174)
    
    def test_specific_examples(self):
        """Test specific example calculations"""
        # Example 1: Same length, different content
        tau = self.system.compute_time_metric('00', '11')
        d_E = self.system.edit_distance('00', '11')
        expected = self.system.epsilon + d_E * log2(self.system.phi)
        self.assertAlmostEqual(tau, expected)
        
        # Example 2: Increasing entropy
        tau = self.system.compute_time_metric('0', '000')
        H_0 = log2(1)    # H('0') = log2(1) = 0
        H_000 = log2(3)  # H('000') = log2(3)
        self.assertAlmostEqual(tau, H_000 - H_0)
        
        # Example 3: Decreasing entropy
        tau = self.system.compute_time_metric('0000', '00')
        self.assertGreaterEqual(tau, self.system.epsilon)
    
    def test_mathematical_representation(self):
        """Test mathematical representation formulas"""
        # Test quasi-metric space property
        states = self.test_states[:5]  # Use subset for efficiency
        
        # For any three states, modified triangle inequality may not hold
        # but causality should hold for proper evolution sequences
        chain = self.system.create_causal_chain(3)
        s, t, u = chain[0], chain[1], chain[2]
        
        tau_st = self.system.compute_time_metric(s, t)
        tau_tu = self.system.compute_time_metric(t, u)
        tau_su = self.system.compute_time_metric(s, u)
        
        # In a causal chain, the direct path should be at least
        # as long as the sum of intermediate steps
        self.assertGreaterEqual(tau_su, tau_st + tau_tu - 1e-10)
    
    def test_entropy_driven_evolution(self):
        """Test entropy-driven evolution formula"""
        # Create states with strictly increasing entropy
        S_t = '0'
        S_t1 = '01'
        
        tau = self.system.compute_time_metric(S_t, S_t1)
        H_diff = self.system.compute_entropy(S_t1) - self.system.compute_entropy(S_t)
        
        self.assertAlmostEqual(tau, H_diff)
        self.assertGreater(tau, 0)
    
    def test_discrete_time_structure(self):
        """Test discrete time structure"""
        # Generate time values
        time_values = set()
        
        # Collect various time values
        test_pairs = [
            ('0', '00'),    # Entropy increase: H(00) - H(0) = 1 - 0 = 1
            ('0', '000'),   # Entropy increase: H(000) - H(0) = log2(3) - 0
            ('00', '0'),    # Edit distance case
            ('01', '10'),   # Edit distance case
            ('', '0'),      # From empty to single bit
        ]
        
        for s, t in test_pairs:
            tau = self.system.compute_time_metric(s, t)
            if tau > 0:
                time_values.add(round(tau, 10))
        
        # Verify we have discrete values
        self.assertGreater(len(time_values), 0)
        
        # The minimum positive time exists
        min_tau = min(time_values)
        self.assertGreater(min_tau, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)