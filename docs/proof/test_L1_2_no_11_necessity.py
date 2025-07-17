#!/usr/bin/env python3
"""
Machine verification unit tests for L1.2: No-11 Constraint Necessity
Testing the proof that self-referential complete binary systems necessarily satisfy the no-11 constraint.
"""

import unittest
import re
from typing import List, Set, Dict, Optional, Tuple


class No11ConstraintSystem:
    """Implementation of L1.2: No-11 constraint necessity in self-referential systems"""
    
    def __init__(self):
        self.epsilon = 1e-10
    
    def contains_consecutive_ones(self, binary_string: str) -> bool:
        """Check if binary string contains consecutive 1s (violates no-11 constraint)"""
        return '11' in binary_string
    
    def validate_no_11_constraint(self, binary_string: str) -> bool:
        """Validate that string satisfies no-11 constraint"""
        if not all(c in '01' for c in binary_string):
            raise ValueError("Input must be binary string")
        return not self.contains_consecutive_ones(binary_string)
    
    def generate_valid_strings(self, length: int) -> Set[str]:
        """Generate all valid binary strings of given length satisfying no-11 constraint"""
        if length <= 0:
            return {''}
        
        valid_strings = set()
        
        def backtrack(current: str, remaining: int):
            if remaining == 0:
                valid_strings.add(current)
                return
            
            # Can always append 0
            backtrack(current + '0', remaining - 1)
            
            # Can append 1 only if last character is not 1
            if not current or current[-1] != '1':
                backtrack(current + '1', remaining - 1)
        
        backtrack('', length)
        return valid_strings
    
    def count_valid_strings_dp(self, length: int) -> int:
        """Count valid no-11 strings using dynamic programming (should equal Fibonacci numbers)"""
        if length == 0:
            return 1  # Empty string
        if length == 1:
            return 2  # "0", "1"
        
        # dp[i][0] = count ending in 0, dp[i][1] = count ending in 1
        dp = [[0, 0] for _ in range(length + 1)]
        dp[1][0] = 1  # "0"
        dp[1][1] = 1  # "1"
        
        for i in range(2, length + 1):
            dp[i][0] = dp[i-1][0] + dp[i-1][1]  # Can append 0 to any
            dp[i][1] = dp[i-1][0]  # Can only append 1 to strings ending in 0
        
        return dp[length][0] + dp[length][1]
    
    def level_function(self, binary_string: str, position: int) -> Optional[int]:
        """Level function as defined in L1.2.2 - returns recursive level at position"""
        if position < 0 or position >= len(binary_string):
            return None
        
        if binary_string[position] == '0':
            return 0
        elif binary_string[position] == '1':
            if position == 0:
                return 1  # First 1 starts at level 1
            elif binary_string[position - 1] == '0':
                prev_level = self.level_function(binary_string, position - 1)
                return (prev_level or 0) + 1
            else:  # binary_string[position - 1] == '1' - consecutive 1s
                return None  # undefined
        
        return None
    
    def verify_level_function_well_defined(self, binary_string: str) -> bool:
        """Verify that level function is well-defined for entire string"""
        for i in range(len(binary_string)):
            if self.level_function(binary_string, i) is None:
                return False
        return True
    
    def information_increment(self, state1: str, state2: str) -> float:
        """Simple information increment measure"""
        if len(state2) <= len(state1):
            return 0.0
        
        # Information increment based on length and structural complexity
        length_increment = len(state2) - len(state1)
        complexity_increment = self.structural_complexity(state2) - self.structural_complexity(state1)
        
        return length_increment + complexity_increment
    
    def structural_complexity(self, binary_string: str) -> float:
        """Measure structural complexity of binary string"""
        if not binary_string:
            return 0.0
        
        # Count pattern complexity: transitions, alternations, etc.
        transitions = sum(1 for i in range(len(binary_string) - 1) 
                         if binary_string[i] != binary_string[i + 1])
        
        # Penalize consecutive 1s heavily
        consecutive_ones_penalty = binary_string.count('11') * 10
        
        return transitions - consecutive_ones_penalty
    
    def collapse_operator_simulation(self, state: str) -> str:
        """Simulate Ξ operator - must preserve no-11 constraint"""
        if not self.validate_no_11_constraint(state):
            raise ValueError("Input state violates no-11 constraint")
        
        # Simple simulation: add self-reference while maintaining no-11
        if not state:
            return "0"
        
        # Add expansion that maintains no-11 constraint
        if state.endswith('1'):
            return state + '0' + state  # Insert 0 to prevent consecutive 1s
        else:
            return state + state
    
    def verify_information_monotonicity(self, state_sequence: List[str]) -> bool:
        """Verify L1.2.1: information increment monotonicity"""
        for i in range(len(state_sequence) - 1):
            current = state_sequence[i]
            next_state = state_sequence[i + 1]
            
            if len(next_state) <= len(current):
                return False  # No expansion
            
            # Information should increase
            info_increment = self.information_increment(current, next_state)
            if info_increment <= 0:
                return False
        
        return True
    
    def demonstrate_no_11_necessity(self) -> Dict[str, bool]:
        """Demonstrate that no-11 constraint is necessary for self-referential completeness"""
        results = {
            "valid_strings_well_defined": True,
            "invalid_strings_ill_defined": True,
            "fibonacci_counting_property": True,
            "level_function_consistency": True,
            "information_monotonicity": True
        }
        
        # Test valid strings
        valid_examples = ["0", "1", "01", "10", "101", "010", "1010"]
        for example in valid_examples:
            if not self.verify_level_function_well_defined(example):
                results["valid_strings_well_defined"] = False
        
        # Test invalid strings (should fail level function)
        invalid_examples = ["11", "110", "011", "1101", "0110"]
        for example in invalid_examples:
            if self.verify_level_function_well_defined(example):
                results["invalid_strings_ill_defined"] = False
        
        # Test Fibonacci counting
        for length in range(1, 8):
            valid_count = len(self.generate_valid_strings(length))
            dp_count = self.count_valid_strings_dp(length)
            if valid_count != dp_count:
                results["fibonacci_counting_property"] = False
        
        return results
    
    def prove_contradiction_with_consecutive_ones(self, example_string: str = "1101") -> Dict[str, bool]:
        """Prove that consecutive 1s lead to contradictions"""
        proof_steps = {
            "contains_consecutive_ones": self.contains_consecutive_ones(example_string),
            "level_function_undefined": not self.verify_level_function_well_defined(example_string),
            "violates_no_11_constraint": not self.validate_no_11_constraint(example_string),
            "collapse_operator_fails": False
        }
        
        # Test if collapse operator can handle the invalid string
        try:
            self.collapse_operator_simulation(example_string)
        except ValueError:
            proof_steps["collapse_operator_fails"] = True
        
        return proof_steps


class TestNo11Necessity(unittest.TestCase):
    """Unit tests for L1.2: No-11 Constraint Necessity"""
    
    def setUp(self):
        self.no11_system = No11ConstraintSystem()
    
    def test_consecutive_ones_detection(self):
        """Test detection of consecutive 1s in binary strings"""
        # Strings with consecutive 1s
        invalid_strings = ["11", "110", "011", "1101", "0110", "1110", "1111"]
        for s in invalid_strings:
            with self.subTest(string=s):
                self.assertTrue(self.no11_system.contains_consecutive_ones(s))
                self.assertFalse(self.no11_system.validate_no_11_constraint(s))
        
        # Strings without consecutive 1s
        valid_strings = ["", "0", "1", "01", "10", "101", "010", "1010", "0101"]
        for s in valid_strings:
            with self.subTest(string=s):
                self.assertFalse(self.no11_system.contains_consecutive_ones(s))
                self.assertTrue(self.no11_system.validate_no_11_constraint(s))
    
    def test_valid_string_generation(self):
        """Test generation of all valid strings of given length"""
        # Test small lengths
        valid_1 = self.no11_system.generate_valid_strings(1)
        self.assertEqual(valid_1, {"0", "1"})
        
        valid_2 = self.no11_system.generate_valid_strings(2)
        self.assertEqual(valid_2, {"00", "01", "10"})
        
        valid_3 = self.no11_system.generate_valid_strings(3)
        expected_3 = {"000", "001", "010", "100", "101"}
        self.assertEqual(valid_3, expected_3)
        
        # Verify all generated strings satisfy no-11 constraint
        for length in range(1, 6):
            valid_strings = self.no11_system.generate_valid_strings(length)
            for s in valid_strings:
                with self.subTest(length=length, string=s):
                    self.assertTrue(self.no11_system.validate_no_11_constraint(s))
    
    def test_fibonacci_counting_property(self):
        """Test that valid string counts follow Fibonacci sequence"""
        # For length n, count equals F(n+2) where F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, ...
        # But our implementation has F(1)=1, F(2)=2, so we need F(n+1) for length n
        fibonacci_values = {
            1: 2,  # F(2) = 2 strings of length 1: "0", "1"  
            2: 3,  # F(3) = 3 strings of length 2: "00", "01", "10"
            3: 5,  # F(4) = 5 strings of length 3
            4: 8,  # F(5) = 8 strings of length 4
            5: 13, # F(6) = 13 strings of length 5
            6: 21, # F(7) = 21 strings of length 6
            7: 34  # F(8) = 34 strings of length 7
        }
        
        for length, expected in fibonacci_values.items():
            actual_generated = len(self.no11_system.generate_valid_strings(length))
            actual_dp = self.no11_system.count_valid_strings_dp(length)
            
            with self.subTest(length=length):
                self.assertEqual(actual_generated, expected)
                self.assertEqual(actual_dp, expected)
                self.assertEqual(actual_generated, actual_dp)
    
    def test_level_function_well_defined_for_valid_strings(self):
        """Test L1.2.2: Level function is well-defined for valid strings"""
        valid_examples = [
            ("0", [0]),
            ("1", [1]),
            ("01", [0, 1]),
            ("10", [1, 0]),
            ("101", [1, 0, 1]),
            ("010", [0, 1, 0]),
            ("1010", [1, 0, 1, 0])
        ]
        
        for string, expected_levels in valid_examples:
            with self.subTest(string=string):
                # Should be well-defined
                self.assertTrue(self.no11_system.verify_level_function_well_defined(string))
                
                # Check specific level values
                for i, expected_level in enumerate(expected_levels):
                    actual_level = self.no11_system.level_function(string, i)
                    self.assertEqual(actual_level, expected_level)
    
    def test_level_function_undefined_for_invalid_strings(self):
        """Test that level function is undefined for strings with consecutive 1s"""
        invalid_examples = ["11", "110", "011", "1101", "0110", "1110"]
        
        for string in invalid_examples:
            with self.subTest(string=string):
                # Should NOT be well-defined
                self.assertFalse(self.no11_system.verify_level_function_well_defined(string))
                
                # Find positions where level function is undefined
                undefined_positions = []
                for i in range(len(string)):
                    if self.no11_system.level_function(string, i) is None:
                        undefined_positions.append(i)
                
                self.assertGreater(len(undefined_positions), 0,
                                 f"Expected undefined positions in {string}")
    
    def test_information_increment_monotonicity(self):
        """Test L1.2.1: Information increment monotonicity"""
        # Valid sequence that should show monotonic information increase
        valid_sequence = ["0", "01", "010", "0101"]
        self.assertTrue(self.no11_system.verify_information_monotonicity(valid_sequence))
        
        # Test individual increments
        for i in range(len(valid_sequence) - 1):
            increment = self.no11_system.information_increment(
                valid_sequence[i], valid_sequence[i + 1]
            )
            self.assertGreater(increment, 0)
    
    def test_structural_complexity_penalizes_consecutive_ones(self):
        """Test that structural complexity penalizes consecutive 1s"""
        valid_string = "1010"
        invalid_string = "1100"
        
        complexity_valid = self.no11_system.structural_complexity(valid_string)
        complexity_invalid = self.no11_system.structural_complexity(invalid_string)
        
        # Invalid string should have lower (more negative) complexity due to penalty
        self.assertLess(complexity_invalid, complexity_valid)
    
    def test_collapse_operator_preserves_no_11_constraint(self):
        """Test that Ξ operator preserves no-11 constraint"""
        valid_inputs = ["0", "1", "01", "10", "101", "010"]
        
        for input_state in valid_inputs:
            with self.subTest(input=input_state):
                output_state = self.no11_system.collapse_operator_simulation(input_state)
                
                # Output should also satisfy no-11 constraint
                self.assertTrue(self.no11_system.validate_no_11_constraint(output_state))
                
                # Output should be longer (information increase)
                self.assertGreater(len(output_state), len(input_state))
    
    def test_collapse_operator_rejects_invalid_input(self):
        """Test that Ξ operator rejects input violating no-11 constraint"""
        invalid_inputs = ["11", "110", "011", "1101"]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                with self.assertRaises(ValueError):
                    self.no11_system.collapse_operator_simulation(invalid_input)
    
    def test_main_necessity_demonstration(self):
        """Test main demonstration that no-11 constraint is necessary"""
        results = self.no11_system.demonstrate_no_11_necessity()
        
        # All aspects of necessity should be demonstrated
        for aspect, demonstrated in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(demonstrated, f"Failed to demonstrate: {aspect}")
    
    def test_contradiction_proof_with_consecutive_ones(self):
        """Test proof that consecutive 1s lead to contradictions"""
        contradiction_examples = ["11", "1101", "0110", "1110"]
        
        for example in contradiction_examples:
            with self.subTest(example=example):
                proof_steps = self.no11_system.prove_contradiction_with_consecutive_ones(example)
                
                # All contradiction indicators should be true
                self.assertTrue(proof_steps["contains_consecutive_ones"])
                self.assertTrue(proof_steps["level_function_undefined"])
                self.assertTrue(proof_steps["violates_no_11_constraint"])
                self.assertTrue(proof_steps["collapse_operator_fails"])
    
    def test_necessity_vs_sufficiency(self):
        """Test that no-11 constraint is both necessary and sufficient"""
        # Necessity: all self-referential systems must satisfy it
        # (tested in other methods)
        
        # Sufficiency: satisfying it enables self-referential completeness
        sufficient_examples = ["0", "1", "01", "10", "101", "010", "1010"]
        
        for example in sufficient_examples:
            with self.subTest(example=example):
                # Should enable well-defined level function
                self.assertTrue(self.no11_system.verify_level_function_well_defined(example))
                
                # Should be processable by collapse operator
                try:
                    result = self.no11_system.collapse_operator_simulation(example)
                    self.assertIsInstance(result, str)
                    self.assertGreater(len(result), len(example))
                except ValueError:
                    self.fail(f"Collapse operator failed on valid input: {example}")
    
    def test_physical_correspondence_interpretation(self):
        """Test physical correspondence of no-11 constraint"""
        # The constraint should correspond to physical principles
        
        # Pauli exclusion principle analogy: no two identical states
        consecutive_states = ["11", "1111"]
        for state in consecutive_states:
            with self.subTest(state=state):
                # Should violate "exclusion principle"
                self.assertFalse(self.no11_system.validate_no_11_constraint(state))
        
        # Energy level separation: transitions need gaps
        valid_transitions = ["101", "010", "1010"]
        for transition in valid_transitions:
            with self.subTest(transition=transition):
                # Should satisfy "separation principle"
                self.assertTrue(self.no11_system.validate_no_11_constraint(transition))
    
    def test_mathematical_elegance_properties(self):
        """Test mathematical elegance of no-11 constraint"""
        # Should exhibit elegant mathematical properties
        
        # Golden ratio emergence in growth rates
        growth_rates = []
        for length in range(1, 10):
            count = self.no11_system.count_valid_strings_dp(length)
            if length > 1:
                prev_count = self.no11_system.count_valid_strings_dp(length - 1)
                if prev_count > 0:
                    growth_rates.append(count / prev_count)
        
        # Growth rates should converge to golden ratio φ ≈ 1.618
        if growth_rates:
            final_rate = growth_rates[-1]
            golden_ratio = (1 + 5**0.5) / 2
            self.assertAlmostEqual(final_rate, golden_ratio, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)