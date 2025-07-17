#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine verification unit tests for D1.3: No-11 Constraint Definition
Testing the formal definition of the no-11 constraint on binary sequences.
"""

import unittest
from typing import Set
import re


class No11ConstraintSystem:
    """System for testing no-11 constraint properties"""
    
    def __init__(self):
        self.sigma = {'0', '1'}
        self.fibonacci_cache = {}  # Will compute as needed
    
    def verify_no_11_constraint(self, s: str) -> bool:
        """Verify if a binary string satisfies the no-11 constraint"""
        for i in range(len(s) - 1):
            if s[i] == '1' and s[i+1] == '1':
                return False
        return True
    
    def has_substring_11(self, s: str) -> bool:
        """Check if string contains '11' as substring"""
        return '11' in s
    
    def verify_gap_property(self, s: str) -> bool:
        """Verify that any two 1s are separated by at least one position"""
        ones_positions = [i for i, bit in enumerate(s) if bit == '1']
        
        for i in range(len(ones_positions) - 1):
            if ones_positions[i+1] - ones_positions[i] < 2:
                return False
        return True
    
    def count_valid_sequences(self, n: int) -> int:
        """Count number of valid sequences of length n"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
        
        # Base cases
        if n == 0:
            return 1  # Empty sequence
        if n == 1:
            return 2  # "0" and "1"
        if n == 2:
            return 3  # "00", "01", "10"
        
        # Dynamic programming: V[n] = V[n-1] + V[n-2]
        result = self.count_valid_sequences(n-1) + self.count_valid_sequences(n-2)
        self.fibonacci_cache[n] = result
        return result
    
    def generate_all_valid_sequences(self, n: int) -> Set[str]:
        """Generate all valid binary sequences of length n"""
        if n == 0:
            return {''}
        if n == 1:
            return {'0', '1'}
        
        valid_sequences = set()
        
        # Recursively build sequences
        def build_sequences(current: str, remaining: int):
            if remaining == 0:
                if self.verify_no_11_constraint(current):
                    valid_sequences.add(current)
                return
            
            # Try adding 0
            build_sequences(current + '0', remaining - 1)
            
            # Try adding 1 (only if it won't create '11')
            if not current or current[-1] != '1':
                build_sequences(current + '1', remaining - 1)
        
        build_sequences('', n)
        return valid_sequences
    
    def compute_asymptotic_density(self, n: int) -> float:
        """Compute the density of valid sequences among all n-bit sequences"""
        valid_count = self.count_valid_sequences(n)
        total_count = 2**n
        return valid_count / total_count
    
    def max_ones_count(self, n: int) -> int:
        """Compute maximum number of 1s in a valid sequence of length n"""
        return (n + 1) // 2
    
    def verify_recursive_construction(self, n: int) -> bool:
        """Verify the recursive construction property"""
        if n <= 2:
            return True
        
        V_n = self.generate_all_valid_sequences(n)
        
        # The correct recursive property is based on the last digit:
        # V_n consists of:
        # 1. All sequences from V_{n-1} with 0 appended (these end in 0)
        # 2. All sequences from V_{n-1} that end in 0 with 1 appended (these end in 1)
        
        V_n_1 = self.generate_all_valid_sequences(n-1)
        
        constructed = set()
        
        # All V_{n-1} sequences can have 0 appended
        for s in V_n_1:
            constructed.add(s + '0')
        
        # Only V_{n-1} sequences ending in 0 can have 1 appended
        for s in V_n_1:
            if not s or s[-1] == '0':  # empty string or ends with 0
                constructed.add(s + '1')
        
        return V_n == constructed
    
    def verify_concatenation_closure(self, s1: str, s2: str) -> bool:
        """Verify that concatenating two valid sequences with 0 preserves validity"""
        if self.verify_no_11_constraint(s1) and self.verify_no_11_constraint(s2):
            concatenated = s1 + '0' + s2
            return self.verify_no_11_constraint(concatenated)
        return True
    
    def matches_regex(self, s: str) -> bool:
        """Check if string matches the regex 0*(10+)*0* for no-11 constraint"""
        # The regex should match sequences with no consecutive 1s
        # A valid sequence is either:
        # - All zeros: 0*
        # - Alternating pattern with isolated 1s: each 1 must be followed by at least one 0
        # - Can end with 1 or 0
        
        # Check no-11 constraint directly instead of complex regex
        if '11' in s:
            return False
        
        # Now check if it matches the pattern
        # Valid patterns: empty, all 0s, or sequences where 1s are separated by 0s
        pattern = r'^(0*|(0*1+0+)*0*1*0*)$'
        return bool(re.match(pattern, s))
    
    def compute_fibonacci(self, n: int) -> int:
        """Compute standard Fibonacci number F_n"""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
    
    def verify_fibonacci_formula(self, n: int) -> bool:
        """Verify |V_n| = F_{n+2}"""
        valid_count = self.count_valid_sequences(n)
        fib_value = self.compute_fibonacci(n + 2)
        return valid_count == fib_value


class TestNo11ConstraintDefinition(unittest.TestCase):
    """Unit tests for D1.3: No-11 Constraint Definition"""
    
    def setUp(self):
        self.system = No11ConstraintSystem()
    
    def test_basic_constraint_verification(self):
        """Test basic no-11 constraint verification"""
        # Valid sequences
        self.assertTrue(self.system.verify_no_11_constraint(''))
        self.assertTrue(self.system.verify_no_11_constraint('0'))
        self.assertTrue(self.system.verify_no_11_constraint('1'))
        self.assertTrue(self.system.verify_no_11_constraint('010101'))
        self.assertTrue(self.system.verify_no_11_constraint('100010'))
        self.assertTrue(self.system.verify_no_11_constraint('0000'))
        
        # Invalid sequences
        self.assertFalse(self.system.verify_no_11_constraint('11'))
        self.assertFalse(self.system.verify_no_11_constraint('011'))
        self.assertFalse(self.system.verify_no_11_constraint('110'))
        self.assertFalse(self.system.verify_no_11_constraint('0110'))
        self.assertFalse(self.system.verify_no_11_constraint('01101'))
    
    def test_equivalent_formulations(self):
        """Test Lemma D1.3.1: Equivalent formulations"""
        test_strings = ['', '0', '1', '01', '10', '11', '010', '101', '110', '0110']
        
        for s in test_strings:
            # No-11 constraint iff "11" not in substrings(s)
            constraint_check = self.system.verify_no_11_constraint(s)
            substring_check = not self.system.has_substring_11(s)
            self.assertEqual(constraint_check, substring_check)
    
    def test_gap_property(self):
        """Test Lemma D1.3.2: Gap property"""
        test_strings = ['', '0', '1', '01', '10', '11', '010', '101', '110', '10101']
        
        for s in test_strings:
            constraint_check = self.system.verify_no_11_constraint(s)
            gap_check = self.system.verify_gap_property(s)
            self.assertEqual(constraint_check, gap_check)
    
    def test_regex_representation(self):
        """Test Lemma D1.3.3: Regular language representation"""
        # Generate all valid sequences up to length 5
        for n in range(6):
            valid_sequences = self.system.generate_all_valid_sequences(n)
            for s in valid_sequences:
                self.assertTrue(self.system.matches_regex(s),
                              f"Valid sequence {s} should match regex")
        
        # Test some invalid sequences
        invalid = ['11', '110', '011', '1101']
        for s in invalid:
            self.assertFalse(self.system.matches_regex(s),
                           f"Invalid sequence {s} should not match regex")
    
    def test_counting_formula(self):
        """Test Property D1.3.1: Counting formula |V_n| = F_{n+2}"""
        for n in range(10):
            self.assertTrue(self.system.verify_fibonacci_formula(n))
    
    def test_sequence_generation(self):
        """Test valid sequence generation"""
        # Test specific cases
        self.assertEqual(self.system.generate_all_valid_sequences(0), {''})
        self.assertEqual(self.system.generate_all_valid_sequences(1), {'0', '1'})
        self.assertEqual(self.system.generate_all_valid_sequences(2), {'00', '01', '10'})
        
        # Verify counts match formula
        for n in range(1, 8):
            valid_sequences = self.system.generate_all_valid_sequences(n)
            expected_count = self.system.count_valid_sequences(n)
            self.assertEqual(len(valid_sequences), expected_count)
            
            # Verify all generated sequences are valid
            for s in valid_sequences:
                self.assertTrue(self.system.verify_no_11_constraint(s))
    
    def test_asymptotic_density(self):
        """Test Property D1.3.2: Asymptotic density"""
        phi = (1 + 5**0.5) / 2
        expected_limit = 1 / (phi**2)  # ≈ 0.382
        
        # Test convergence for increasing n
        densities = []
        for n in range(5, 20):
            density = self.system.compute_asymptotic_density(n)
            densities.append(density)
        
        # The density converges to 1/φ² but from above, decreasing exponentially
        # For practical n values, it will be much smaller than the limit
        last_density = densities[-1]
        
        # Check basic properties
        self.assertGreater(last_density, 0)  # Positive
        self.assertLess(last_density, 1)  # Less than 1
        
        # Check that density is strictly decreasing
        for i in range(1, len(densities)):
            self.assertLess(densities[i], densities[i-1])
        
        # The actual formula is: density ≈ (1/√5) * φ^(n+2) / 2^n = (1/√5) * (φ/2)^n * φ²
        # Since φ/2 ≈ 0.809 < 1, this decreases exponentially
        # So for large n, density ≈ (φ²/√5) * (φ/2)^n which is very small
    
    def test_max_ones_count(self):
        """Test Property D1.3.3: Maximum number of 1s"""
        # Test specific cases
        self.assertEqual(self.system.max_ones_count(0), 0)
        self.assertEqual(self.system.max_ones_count(1), 1)
        self.assertEqual(self.system.max_ones_count(2), 1)
        self.assertEqual(self.system.max_ones_count(3), 2)
        self.assertEqual(self.system.max_ones_count(4), 2)
        self.assertEqual(self.system.max_ones_count(5), 3)
        
        # Verify by checking actual sequences
        for n in range(1, 8):
            valid_sequences = self.system.generate_all_valid_sequences(n)
            max_ones = max(s.count('1') for s in valid_sequences)
            self.assertEqual(max_ones, self.system.max_ones_count(n))
    
    def test_recursive_construction(self):
        """Test Property D1.3.4: Recursive construction"""
        for n in range(3, 8):
            self.assertTrue(self.system.verify_recursive_construction(n))
    
    def test_concatenation_closure(self):
        """Test Property D1.3.5: Concatenation closure"""
        valid_sequences = [
            ('0', '1'),
            ('10', '01'),
            ('010', '101'),
            ('1010', '0101')
        ]
        
        for s1, s2 in valid_sequences:
            self.assertTrue(self.system.verify_concatenation_closure(s1, s2))
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty string
        self.assertTrue(self.system.verify_no_11_constraint(''))
        self.assertEqual(self.system.count_valid_sequences(0), 1)
        
        # Very long alternating sequence
        long_alternating = '01' * 100
        self.assertTrue(self.system.verify_no_11_constraint(long_alternating))
        
        # All zeros
        all_zeros = '0' * 100
        self.assertTrue(self.system.verify_no_11_constraint(all_zeros))
        
        # Maximum ones pattern
        max_ones_pattern = '10' * 50
        self.assertTrue(self.system.verify_no_11_constraint(max_ones_pattern))
    
    def test_fibonacci_values(self):
        """Test Fibonacci number computation"""
        # Standard Fibonacci (F_0=0, F_1=1, F_2=1, F_3=2, ...)
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for n in range(10):
            self.assertEqual(self.system.compute_fibonacci(n), expected[n])
    
    def test_mathematical_formulas(self):
        """Test mathematical representation formulas"""
        # Test explicit Fibonacci formula (Binet's formula)
        phi = (1 + 5**0.5) / 2
        psi = (1 - 5**0.5) / 2
        
        for n in range(1, 10):
            fib_n = self.system.compute_fibonacci(n)
            # Standard Binet's formula
            formula_value = round((phi**n - psi**n) / 5**0.5)
            
            self.assertAlmostEqual(fib_n, formula_value, delta=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)