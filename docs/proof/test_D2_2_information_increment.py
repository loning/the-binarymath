#!/usr/bin/env python3
"""
Machine verification unit tests for D2.2: Information Increment
Testing the formal definition of information increment in self-referential complete systems.
"""

import unittest
import math
from typing import Set, List, Dict, Any


class InformationIncrementSystem:
    """System for testing information increment properties"""
    
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        self.ln_phi = math.log(self.phi)  # ln(φ) ≈ 0.4812
        
    def phi_valid(self, s: str) -> bool:
        """Check if string is φ-valid (no consecutive 11)"""
        return '11' not in s
    
    def phi_concat(self, a: str, b: str) -> str:
        """φ-valid concatenation operation"""
        if not a or not b:
            return a + b
        
        # Check if concatenation would create 11
        if a[-1] == '1' and b[0] == '1':
            return a + '0' + b  # Insert 0 to avoid 11
        else:
            return a + b  # Direct concatenation
    
    def self_ref(self, s: str) -> str:
        """Simple self-referential function for testing"""
        if not s:
            return '0'
        
        # Encode length as binary
        length = len(s)
        length_bits = bin(length)[2:]  # Remove '0b' prefix
        
        # Simple checksum
        checksum = sum(int(bit) for bit in s) % 2
        checksum_bit = str(checksum)
        
        return self.phi_concat(length_bits, checksum_bit)
    
    def collapse_operator(self, s: str) -> str:
        """Collapse operator implementation"""
        if not self.phi_valid(s):
            raise ValueError(f"Input must be φ-valid: {s}")
        
        # Compute self-reference
        self_ref_part = self.self_ref(s)
        
        # φ-valid concatenation
        result = self.phi_concat(s, self_ref_part)
        
        # Ensure result is φ-valid
        if not self.phi_valid(result):
            # Simple fix: replace 11 with 101
            result = result.replace('11', '101')
        
        return result
    
    def compute_entropy(self, s: str) -> float:
        """Compute entropy H(s) = ln(|s|)"""
        if not s:
            return 0.0
        return math.log(len(s))
    
    def compute_information_increment(self, s: str) -> float:
        """Compute information increment ΔI(s)"""
        if not s:
            return 0.0
        
        # Apply collapse operator
        collapsed_s = self.collapse_operator(s)
        
        # Compute entropy increment
        entropy_before = self.compute_entropy(s)
        entropy_after = self.compute_entropy(collapsed_s)
        
        delta_I = entropy_after - entropy_before
        
        return delta_I
    
    def compute_cumulative_information(self, s0: str, n_steps: int) -> float:
        """Compute cumulative information increment"""
        total_info = 0.0
        current_state = s0
        
        for step in range(n_steps):
            # Compute information increment for current step
            delta_I = self.compute_information_increment(current_state)
            total_info += delta_I
            
            # Update state
            current_state = self.collapse_operator(current_state)
        
        return total_info
    
    def verify_positivity(self, s: str) -> bool:
        """Verify that ΔI(s) > 0"""
        try:
            delta_I = self.compute_information_increment(s)
            return delta_I > 0
        except:
            return False
    
    def verify_upper_bound(self, s: str) -> bool:
        """Verify that ΔI(s) ≤ ln(φ)"""
        try:
            delta_I = self.compute_information_increment(s)
            return delta_I <= self.ln_phi + 1e-10
        except:
            return False
    
    def verify_lower_bound(self, states: List[str]) -> bool:
        """Verify that there exists c > 0 such that ΔI(s) ≥ c"""
        increments = []
        for s in states:
            try:
                delta_I = self.compute_information_increment(s)
                increments.append(delta_I)
            except:
                continue
        
        if not increments:
            return False
        
        # Check if minimum increment is positive
        min_increment = min(increments)
        return min_increment > 0
    
    def verify_cumulativity(self, s0: str, n_steps: int) -> bool:
        """Verify cumulative property"""
        # Method 1: Direct cumulative calculation
        cumulative_1 = self.compute_cumulative_information(s0, n_steps)
        
        # Method 2: Step-by-step calculation
        cumulative_2 = 0.0
        current_state = s0
        
        for _ in range(n_steps):
            delta_I = self.compute_information_increment(current_state)
            cumulative_2 += delta_I
            current_state = self.collapse_operator(current_state)
        
        # Should be approximately equal
        return abs(cumulative_1 - cumulative_2) < 1e-10
    
    def verify_growth_rate_formula(self, s: str) -> bool:
        """Verify that ΔI(s) = ln(|Ξ(s)|/|s|)"""
        try:
            collapsed_s = self.collapse_operator(s)
            
            # Compute information increment
            delta_I = self.compute_information_increment(s)
            
            # Compute growth rate
            growth_rate = len(collapsed_s) / len(s)
            expected_delta_I = math.log(growth_rate)
            
            # Should be approximately equal
            return abs(delta_I - expected_delta_I) < 1e-10
        except:
            return False
    
    def analyze_limit_behavior(self, s0: str, n_steps: int) -> List[float]:
        """Analyze limit behavior of information increments"""
        increments = []
        current_state = s0
        
        for _ in range(n_steps):
            delta_I = self.compute_information_increment(current_state)
            increments.append(delta_I)
            current_state = self.collapse_operator(current_state)
        
        return increments


class TestInformationIncrement(unittest.TestCase):
    """Unit tests for D2.2: Information Increment"""
    
    def setUp(self):
        self.system = InformationIncrementSystem()
        self.test_states = ['0', '1', '01', '10', '001', '010', '100', '101']
    
    def test_information_increment_computation(self):
        """Test basic information increment computation"""
        for state in self.test_states:
            with self.subTest(state=state):
                delta_I = self.system.compute_information_increment(state)
                self.assertIsInstance(delta_I, float)
                self.assertGreater(delta_I, 0, f"ΔI({state}) should be positive")
    
    def test_positivity_property(self):
        """Test Property D2.2.1: Positivity"""
        for state in self.test_states:
            with self.subTest(state=state):
                self.assertTrue(self.system.verify_positivity(state),
                              f"Positivity should hold for {state}")
    
    def test_upper_bound_property(self):
        """Test Property D2.2.2: Upper bound"""
        for state in self.test_states:
            with self.subTest(state=state):
                self.assertTrue(self.system.verify_upper_bound(state),
                              f"Upper bound should hold for {state}")
    
    def test_lower_bound_property(self):
        """Test Property D2.2.3: Lower bound"""
        self.assertTrue(self.system.verify_lower_bound(self.test_states),
                      "Lower bound property should hold")
    
    def test_cumulativity_property(self):
        """Test Property D2.2.4: Cumulativity"""
        for state in self.test_states[:3]:  # Test first few states
            with self.subTest(state=state):
                self.assertTrue(self.system.verify_cumulativity(state, 3),
                              f"Cumulativity should hold for {state}")
    
    def test_growth_rate_formula(self):
        """Test that ΔI(s) = ln(|Ξ(s)|/|s|)"""
        for state in self.test_states:
            with self.subTest(state=state):
                self.assertTrue(self.system.verify_growth_rate_formula(state),
                              f"Growth rate formula should hold for {state}")
    
    def test_cumulative_information_computation(self):
        """Test cumulative information computation"""
        s0 = '0'
        n_steps = 5
        
        cumulative_info = self.system.compute_cumulative_information(s0, n_steps)
        
        # Should be positive
        self.assertGreater(cumulative_info, 0)
        
        # Should be sum of individual increments
        individual_sum = 0.0
        current_state = s0
        
        for _ in range(n_steps):
            delta_I = self.system.compute_information_increment(current_state)
            individual_sum += delta_I
            current_state = self.system.collapse_operator(current_state)
        
        self.assertAlmostEqual(cumulative_info, individual_sum, places=10)
    
    def test_entropy_relationship(self):
        """Test relationship with entropy"""
        for state in self.test_states:
            with self.subTest(state=state):
                # ΔI(s) = H(Ξ(s)) - H(s)
                collapsed_state = self.system.collapse_operator(state)
                
                delta_I = self.system.compute_information_increment(state)
                entropy_before = self.system.compute_entropy(state)
                entropy_after = self.system.compute_entropy(collapsed_state)
                
                expected_delta_I = entropy_after - entropy_before
                self.assertAlmostEqual(delta_I, expected_delta_I, places=10)
    
    def test_phi_constraint_bound(self):
        """Test that information increments approach ln(φ) bound"""
        s0 = '0'
        increments = self.system.analyze_limit_behavior(s0, 10)
        
        # All increments should be positive
        for i, delta_I in enumerate(increments):
            self.assertGreater(delta_I, 0, f"Increment {i} should be positive")
        
        # All increments should be bounded by ln(φ)
        for i, delta_I in enumerate(increments):
            self.assertLessEqual(delta_I, self.system.ln_phi + 1e-10,
                               f"Increment {i} should be bounded by ln(φ)")
        
        # Later increments should approach ln(φ)
        if len(increments) >= 5:
            later_increments = increments[-3:]
            for delta_I in later_increments:
                # Should be reasonably close to ln(φ)
                self.assertLess(abs(delta_I - self.system.ln_phi), 0.2,
                              f"Later increments should approach ln(φ)")
    
    def test_specific_calculations(self):
        """Test specific calculation examples"""
        # Example 1: Single bit
        s1 = '0'
        delta_I1 = self.system.compute_information_increment(s1)
        self.assertGreater(delta_I1, 0)
        
        # Example 2: Two bits
        s2 = '01'
        delta_I2 = self.system.compute_information_increment(s2)
        self.assertGreater(delta_I2, 0)
        
        # Example 3: Longer string
        s3 = '010'
        delta_I3 = self.system.compute_information_increment(s3)
        self.assertGreater(delta_I3, 0)
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency"""
        # Test that cumulative information equals final entropy minus initial entropy
        s0 = '0'
        n_steps = 3
        
        # Cumulative information
        cumulative_info = self.system.compute_cumulative_information(s0, n_steps)
        
        # Final state entropy minus initial entropy
        final_state = s0
        for _ in range(n_steps):
            final_state = self.system.collapse_operator(final_state)
        
        entropy_initial = self.system.compute_entropy(s0)
        entropy_final = self.system.compute_entropy(final_state)
        entropy_difference = entropy_final - entropy_initial
        
        self.assertAlmostEqual(cumulative_info, entropy_difference, places=10)
    
    def test_non_negative_increments(self):
        """Test that all increments are non-negative"""
        for state in self.test_states:
            with self.subTest(state=state):
                delta_I = self.system.compute_information_increment(state)
                self.assertGreaterEqual(delta_I, 0,
                                      f"Information increment should be non-negative for {state}")
    
    def test_collapse_operator_consistency(self):
        """Test consistency with collapse operator"""
        for state in self.test_states:
            with self.subTest(state=state):
                # Applying collapse operator should increase length
                collapsed_state = self.system.collapse_operator(state)
                self.assertGreater(len(collapsed_state), len(state),
                                 f"Collapse should increase length: {state} -> {collapsed_state}")
                
                # Should maintain φ-validity
                self.assertTrue(self.system.phi_valid(collapsed_state),
                              f"Collapsed state should be φ-valid: {collapsed_state}")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty string (if supported)
        try:
            delta_I = self.system.compute_information_increment('')
            self.assertGreaterEqual(delta_I, 0)
        except:
            pass  # Empty string might not be supported
        
        # Test single character
        delta_I = self.system.compute_information_increment('0')
        self.assertGreater(delta_I, 0)
        
        # Test longer valid strings
        for length in [5, 10, 15]:
            # Create a valid string of given length
            test_string = '0' + '10' * ((length - 1) // 2)
            if len(test_string) == length and self.system.phi_valid(test_string):
                delta_I = self.system.compute_information_increment(test_string)
                self.assertGreater(delta_I, 0)
                self.assertLessEqual(delta_I, self.system.ln_phi + 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)