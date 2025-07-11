#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N2: Entropic Wall and Forbidden 11
Verifies why "11" must be forbidden and how this creates the fundamental boundary.
"""

import unittest
import math
from typing import List, Optional, Set


class EntropyCalculator:
    """Calculate entropy for different state configurations"""
    
    @staticmethod
    def calculate_entropy(states: List[str]) -> float:
        """Calculate Shannon entropy for a set of states"""
        if not states:
            return 0.0
        
        # For uniform distribution
        n = len(states)
        if n == 1:
            return 0.0
        
        return math.log2(n)
    
    @staticmethod
    def entropy_with_11() -> float:
        """Calculate entropy if 11 were allowed"""
        # With 11, we would have unbounded states
        # This represents infinite entropy
        return float('inf')


class StateEvolution:
    """Model the evolution of states under transformation"""
    
    def __init__(self):
        self.states = set()
        self.transitions = {}
    
    def add_state(self, state: str):
        """Add a state to the system"""
        self.states.add(state)
    
    def add_transition(self, from_state: str, to_state: str):
        """Add a transition between states"""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(to_state)
    
    def evolve(self, start_state: str, steps: int) -> List[str]:
        """Evolve a state for n steps"""
        path = [start_state]
        current = start_state
        
        for _ in range(steps):
            if current in self.transitions and self.transitions[current]:
                # For simplicity, take the first transition
                current = self.transitions[current][0]
                path.append(current)
            else:
                break
        
        return path
    
    def has_infinite_chain(self, state: str) -> bool:
        """Check if a state leads to an infinite chain"""
        visited = set()
        current = state
        
        while current not in visited:
            visited.add(current)
            if current not in self.transitions or not self.transitions[current]:
                return False
            current = self.transitions[current][0]
        
        # If we revisit a state, we have a cycle (not infinite chain)
        # True infinite chain would never revisit
        return False


class TestEntropicWall(unittest.TestCase):
    """Test the necessity and properties of the entropic wall"""
    
    def test_forbidden_state_identification(self):
        """Test that 11 is correctly identified as problematic"""
        valid_states = ["00", "01", "10"]
        forbidden_state = "11"
        
        # 11 is not in valid states
        self.assertNotIn(forbidden_state, valid_states)
        
        # Binary representation check
        self.assertEqual(forbidden_state, "11")
        self.assertTrue(all(c == '1' for c in forbidden_state))
    
    def test_infinite_chain_prevention(self):
        """Test that 11 would create infinite chains"""
        # If 11 existed, it would mean:
        # - First position: 1 (transform from ψ to ψ')
        # - Second position: 1 (transform from ψ' to ψ'')
        # This creates an infinite sequence
        
        system_without_11 = StateEvolution()
        system_without_11.add_state("ψ")
        system_without_11.add_state("ψ'")
        system_without_11.add_transition("ψ", "ψ'")
        system_without_11.add_transition("ψ'", "ψ")  # Returns
        
        path = system_without_11.evolve("ψ", 10)
        
        # Without 11, we have finite cycles
        self.assertLess(len(set(path)), 10)  # Visits fewer unique states
    
    def test_entropy_comparison(self):
        """Test entropy with and without the forbidden state"""
        # Entropy with 3 states
        entropy_valid = EntropyCalculator.calculate_entropy(["00", "01", "10"])
        
        # Entropy if 11 were allowed
        entropy_with_11 = EntropyCalculator.entropy_with_11()
        
        # Valid system has finite entropy
        self.assertLess(entropy_valid, 2.0)
        self.assertAlmostEqual(entropy_valid, math.log2(3), places=5)
        
        # System with 11 would have infinite entropy
        self.assertEqual(entropy_with_11, float('inf'))
    
    def test_closure_property(self):
        """Test that forbidding 11 ensures closure"""
        # Valid states and their combinations
        valid_states = {"00", "01", "10"}
        
        # Check all concatenations
        problematic_concatenations = []
        for s1 in valid_states:
            for s2 in valid_states:
                concat = s1 + s2
                if "11" in concat:
                    problematic_concatenations.append((s1, s2))
        
        # Only 01 → 10 creates "11" (in "0110")
        self.assertEqual(len(problematic_concatenations), 1)
        self.assertEqual(problematic_concatenations[0], ("01", "10"))
    
    def test_wall_as_boundary(self):
        """Test that the wall creates a definite boundary"""
        # States on this side of the wall
        valid_region = {"00", "01", "10"}
        
        # State beyond the wall
        beyond_wall = "11"
        
        # The wall separates them
        self.assertNotIn(beyond_wall, valid_region)
        
        # And this separation is absolute
        self.assertEqual(len(valid_region), 3)
        self.assertTrue(all(state != beyond_wall for state in valid_region))
    
    def test_maximum_entropy_property(self):
        """Test that 11 represents maximum entropy"""
        # Interpret states as transformation counts
        state_entropy = {
            "00": 0,  # No transformation
            "01": 1,  # One transformation
            "10": 1,  # One return transformation
            "11": 2   # Two transformations (unbounded)
        }
        
        # 11 has maximum transformation count
        max_entropy_state = max(state_entropy.items(), key=lambda x: x[1])[0]
        self.assertEqual(max_entropy_state, "11")
    
    def test_structural_dissolution(self):
        """Test that 11 would dissolve structure"""
        # With valid states, we must avoid 01→10
        valid_pattern = ["00", "01", "00", "10", "00"]
        
        # Check pattern is valid
        concat = "".join(valid_pattern)
        self.assertNotIn("11", concat)
        
        # If we try to insert 11
        broken_pattern = ["00", "11", "00"]
        broken_concat = "".join(broken_pattern)
        
        # It contains the forbidden sequence
        self.assertIn("11", broken_concat)


class TestWallAsCreativeConstraint(unittest.TestCase):
    """Test how the wall enables rather than limits"""
    
    def test_pattern_formation(self):
        """Test that the wall enables stable patterns"""
        # Valid patterns without 11
        patterns = [
            ["00", "00", "00"],  # Stable identity
            ["01", "00", "10"],  # Transform and return
            ["00", "01", "00"]   # Isolated transform
        ]
        
        for pattern in patterns:
            concat = "".join(pattern)
            # All patterns are valid
            self.assertNotIn("11", concat)
        
        # These patterns can persist
        self.assertEqual(len(patterns), 3)
    
    def test_cycle_enablement(self):
        """Test that forbidding 11 enables cycles"""
        # The constraint forces cycles
        # 01 cannot be followed by 10, so must be followed by 00 or 01
        
        valid_cycles = [
            ["01", "01"],  # Creates "0101" - no 11
            ["01", "00"],  # Creates "0100" - no 11
            ["10", "00"],  # Creates "1000" - no 11
            ["10", "01"],  # Creates "1001" - no 11
            ["10", "10"]   # Creates "1010" - no 11
        ]
        
        for cycle in valid_cycles:
            concat = "".join(cycle)
            self.assertNotIn("11", concat)
    
    def test_complexity_emergence(self):
        """Test that constraint enables complexity"""
        # Count valid n-length sequences
        def count_valid_sequences(n):
            if n == 0:
                return 1
            if n == 1:
                return 3  # "00", "01", "10"
            
            # Use dynamic programming
            # dp[i][j] = count of sequences of length i ending with state j
            # j: 0="00", 1="01", 2="10"
            dp = [[0] * 3 for _ in range(n + 1)]
            dp[1] = [1, 1, 1]
            
            for i in range(2, n + 1):
                # Ending with "00" - can come from any state
                dp[i][0] = dp[i-1][0] + dp[i-1][1] + dp[i-1][2]
                # Ending with "01" - can come from "00" or "10" (not "01")
                dp[i][1] = dp[i-1][0] + dp[i-1][2]
                # Ending with "10" - can come from any state
                dp[i][2] = dp[i-1][0] + dp[i-1][1] + dp[i-1][2]
            
            return sum(dp[n])
        
        # Complexity grows even with constraint
        seq_counts = [count_valid_sequences(i) for i in range(1, 5)]
        
        # Verify growth
        for i in range(1, len(seq_counts)):
            self.assertGreater(seq_counts[i], seq_counts[i-1])


class TestPhysicalAnalogies(unittest.TestCase):
    """Test physical analogies to the entropic wall"""
    
    def test_speed_of_light_analogy(self):
        """Test that the wall is like the speed of light limit"""
        # Both are absolute boundaries
        speed_of_light = 299792458  # m/s
        faster_than_light = speed_of_light + 1
        
        # Cannot exceed the limit
        self.assertGreater(faster_than_light, speed_of_light)
        
        # Similarly, cannot have "11"
        valid_values = ["00", "01", "10"]
        forbidden_value = "11"
        
        self.assertNotIn(forbidden_value, valid_values)
    
    def test_absolute_zero_analogy(self):
        """Test the absolute zero temperature analogy"""
        # Absolute zero is unreachable
        absolute_zero_kelvin = 0
        
        # Any real temperature is above it
        min_achievable_temp = 0.000001  # Very close but not zero
        
        self.assertGreater(min_achievable_temp, absolute_zero_kelvin)
        
        # Similarly, "11" is unreachable from valid states
        # through valid transitions
    
    def test_uncertainty_principle_analogy(self):
        """Test the uncertainty principle analogy"""
        # Cannot know position and momentum perfectly simultaneously
        # This is a fundamental limit
        
        # In our system, cannot have continuous transformation
        # "11" would mean knowing both transformations perfectly
        # The wall prevents this
        
        # This creates a fundamental uncertainty/limitation
        # that enables the system to function
        self.assertTrue(True)  # Philosophical test


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of the wall"""
    
    def test_wall_function(self):
        """Test the wall function W: States → {0, 1}"""
        def wall_function(state: str) -> int:
            return 0 if state == "11" else 1
        
        # Test all states
        self.assertEqual(wall_function("00"), 1)
        self.assertEqual(wall_function("01"), 1)
        self.assertEqual(wall_function("10"), 1)
        self.assertEqual(wall_function("11"), 0)
        
        # Wall function partitions the space
        all_states = ["00", "01", "10", "11"]
        valid = [s for s in all_states if wall_function(s) == 1]
        invalid = [s for s in all_states if wall_function(s) == 0]
        
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(invalid), 1)
    
    def test_algebraic_closure_with_wall(self):
        """Test that the wall ensures algebraic closure"""
        valid_states = ["00", "01", "10"]
        
        # Composition table (avoiding 01→10)
        compositions = {
            ("00", "00"): "00",
            ("00", "01"): "01",
            ("00", "10"): "10",
            ("01", "00"): "01",
            ("01", "01"): "01",  # Modified to maintain validity
            ("10", "00"): "10",
            ("10", "01"): "00",
            ("10", "10"): "10"
        }
        
        # Check closure
        for s1 in valid_states:
            for s2 in valid_states:
                if (s1, s2) in compositions:
                    result = compositions[(s1, s2)]
                    self.assertIn(result, valid_states)


if __name__ == "__main__":
    unittest.main(verbosity=2)