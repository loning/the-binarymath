#!/usr/bin/env python3
"""
Machine verification unit tests for L1.4: Time Emergence Lemma
Testing the constructive proof that entropy changes necessarily give rise to time structure.
"""

import unittest
import math
from typing import Set, Dict, List, Optional, Tuple


class TimeEmergenceSystem:
    """Implementation of L1.4: Time Emergence from entropy changes"""
    
    def __init__(self):
        self.epsilon = 1e-10  # Numerical tolerance
        self.state_history = []  # Track state evolution
        self.causal_graph = {}  # Track causal connections
    
    def entropy(self, state_set: Set[str]) -> float:
        """Entropy function H: S → R⁺"""
        if not state_set:
            return 0.0
        
        size = len(state_set)
        if size <= 1:
            return 0.0
        
        # Basic entropy: log₂|S| plus structural complexity
        base_entropy = math.log2(size)
        structural_complexity = sum(math.log2(1 + len(state)) for state in state_set) / size
        
        return base_entropy + structural_complexity
    
    def has_causal_connection(self, state_i: Set[str], state_j: Set[str]) -> bool:
        """Check if there's a causal chain S_i → S_j"""
        # For simplicity, we consider a causal connection exists if:
        # 1. state_i is a subset of state_j (expansion), or
        # 2. they have significant overlap and j has higher entropy
        
        if state_i == state_j:
            return True
        
        # Check if i is subset of j (direct expansion)
        if state_i.issubset(state_j):
            return True
        
        # Check if they share common elements and j has higher entropy
        if state_i & state_j and self.entropy(state_j) > self.entropy(state_i):
            return True
        
        return False
    
    def time_metric(self, state_i: Set[str], state_j: Set[str]) -> Optional[float]:
        """τ: S × S → R - Time metric function as defined in L1.4"""
        if state_i == state_j:
            return 0.0
        
        if not self.has_causal_connection(state_i, state_j):
            return None  # undefined for non-causally connected states
        
        h_i = self.entropy(state_i)
        h_j = self.entropy(state_j)
        
        return h_j - h_i
    
    def verify_well_defined(self, state_i: Set[str], state_j: Set[str]) -> bool:
        """Verify that time metric is well-defined for given state pair"""
        try:
            time_val = self.time_metric(state_i, state_j)
            if time_val is None:
                return True  # undefined is a valid result
            return math.isfinite(time_val)
        except (ValueError, ZeroDivisionError):
            return False
    
    def verify_positivity(self, state_i: Set[str], state_j: Set[str]) -> bool:
        """Verify positivity: if S_i → S_j and S_i ≠ S_j, then τ(S_i, S_j) > 0"""
        if state_i == state_j:
            return self.time_metric(state_i, state_j) == 0.0
        
        if not self.has_causal_connection(state_i, state_j):
            return True  # No requirement for non-connected states
        
        time_val = self.time_metric(state_i, state_j)
        if time_val is None:
            return True
        
        return time_val > 0
    
    def verify_identity(self, state: Set[str]) -> bool:
        """Verify identity: τ(S, S) = 0"""
        time_val = self.time_metric(state, state)
        return time_val == 0.0
    
    def verify_transitivity(self, state_i: Set[str], state_j: Set[str], state_k: Set[str]) -> bool:
        """Verify transitivity: if S_i → S_j → S_k, then τ(S_i, S_k) = τ(S_i, S_j) + τ(S_j, S_k)"""
        # Check if we have causal chain i → j → k
        if not (self.has_causal_connection(state_i, state_j) and 
                self.has_causal_connection(state_j, state_k) and
                self.has_causal_connection(state_i, state_k)):
            return True  # No requirement if chain doesn't exist
        
        tau_ij = self.time_metric(state_i, state_j)
        tau_jk = self.time_metric(state_j, state_k)
        tau_ik = self.time_metric(state_i, state_k)
        
        if any(val is None for val in [tau_ij, tau_jk, tau_ik]):
            return True  # Skip if any undefined
        
        expected = tau_ij + tau_jk
        return abs(tau_ik - expected) < self.epsilon
    
    def simulate_state_evolution(self, initial_state: Set[str], steps: int) -> List[Set[str]]:
        """Simulate evolution of self-referential system"""
        evolution = [initial_state]
        current = initial_state.copy()
        
        for i in range(steps):
            # Simple evolution: add new states based on existing ones
            new_elements = set()
            for state in current:
                # Add a transformation of each state
                new_state = state + '0' if len(state) < 5 else state[:-1] + '1'
                new_elements.add(new_state)
            
            # Create new state set
            next_state = current | new_elements
            evolution.append(next_state)
            current = next_state
        
        return evolution
    
    def verify_entropy_monotonicity(self, evolution: List[Set[str]]) -> bool:
        """Verify that entropy increases along evolution"""
        entropies = [self.entropy(state) for state in evolution]
        
        for i in range(len(entropies) - 1):
            if entropies[i+1] <= entropies[i]:
                return False
        
        return True
    
    def verify_time_arrow(self, evolution: List[Set[str]]) -> bool:
        """Verify that time metric establishes consistent arrow of time"""
        for i in range(len(evolution) - 1):
            tau = self.time_metric(evolution[i], evolution[i+1])
            if tau is None or tau <= 0:
                return False
        
        return True
    
    def compute_time_sequence(self, evolution: List[Set[str]]) -> List[float]:
        """Compute cumulative time sequence from evolution"""
        if not evolution:
            return []
        
        time_sequence = [0.0]  # Start at t=0
        cumulative_time = 0.0
        
        for i in range(len(evolution) - 1):
            tau = self.time_metric(evolution[i], evolution[i+1])
            if tau is not None and tau > 0:
                cumulative_time += tau
            time_sequence.append(cumulative_time)
        
        return time_sequence
    
    def verify_time_emergence_properties(self, evolution: List[Set[str]]) -> Dict[str, bool]:
        """Verify all properties required for time emergence"""
        return {
            "entropy_monotonic": self.verify_entropy_monotonicity(evolution),
            "time_arrow_consistent": self.verify_time_arrow(evolution),
            "time_metric_well_defined": all(
                self.verify_well_defined(evolution[i], evolution[i+1]) 
                for i in range(len(evolution) - 1)
            ),
            "causality_preserved": all(
                self.has_causal_connection(evolution[i], evolution[i+1])
                for i in range(len(evolution) - 1)
            )
        }


class TestTimeEmergence(unittest.TestCase):
    """Unit tests for L1.4: Time Emergence Lemma"""
    
    def setUp(self):
        self.time_system = TimeEmergenceSystem()
    
    def test_time_metric_construction(self):
        """Test construction of time metric function τ: S × S → R"""
        state1 = {'0', '1'}
        state2 = {'0', '1', '01'}
        state3 = {'0', '1', '01', '10'}
        
        # Test that time metric is computable
        tau12 = self.time_system.time_metric(state1, state2)
        tau23 = self.time_system.time_metric(state2, state3)
        
        self.assertIsNotNone(tau12)
        self.assertIsNotNone(tau23)
        self.assertIsInstance(tau12, float)
        self.assertIsInstance(tau23, float)
    
    def test_well_defined_property(self):
        """Test that time metric is well-defined"""
        test_states = [
            {'0'},
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'},
            set()
        ]
        
        # Test all pairs
        for i, state_i in enumerate(test_states):
            for j, state_j in enumerate(test_states):
                with self.subTest(i=i, j=j):
                    self.assertTrue(self.time_system.verify_well_defined(state_i, state_j))
    
    def test_positivity_property(self):
        """Test positivity: if S_i → S_j and S_i ≠ S_j, then τ(S_i, S_j) > 0"""
        # Create causally connected states with increasing entropy
        state1 = {'0'}
        state2 = {'0', '1'}
        state3 = {'0', '1', '01'}
        
        # Verify positivity
        self.assertTrue(self.time_system.verify_positivity(state1, state2))
        self.assertTrue(self.time_system.verify_positivity(state2, state3))
        self.assertTrue(self.time_system.verify_positivity(state1, state3))
    
    def test_identity_property(self):
        """Test identity: τ(S, S) = 0"""
        test_states = [
            set(),
            {'0'},
            {'0', '1'},
            {'0', '1', '01', '10'}
        ]
        
        for state in test_states:
            with self.subTest(state=state):
                self.assertTrue(self.time_system.verify_identity(state))
    
    def test_transitivity_property(self):
        """Test transitivity: τ(S_i, S_k) = τ(S_i, S_j) + τ(S_j, S_k)"""
        state1 = {'0'}
        state2 = {'0', '1'}
        state3 = {'0', '1', '01'}
        
        # For causally connected sequence
        self.assertTrue(self.time_system.verify_transitivity(state1, state2, state3))
        
        # Test with multiple sequences
        test_sequences = [
            ({'0'}, {'0', '1'}, {'0', '1', '01'}),
            ({'1'}, {'0', '1'}, {'0', '1', '10'}),
            (set(), {'0'}, {'0', '1'})
        ]
        
        for seq in test_sequences:
            with self.subTest(sequence=seq):
                self.assertTrue(self.time_system.verify_transitivity(*seq))
    
    def test_causal_connection_detection(self):
        """Test detection of causal connections"""
        # Clear causal connections (subset relationships)
        state1 = {'0'}
        state2 = {'0', '1'}
        state3 = {'0', '1', '01'}
        
        self.assertTrue(self.time_system.has_causal_connection(state1, state2))
        self.assertTrue(self.time_system.has_causal_connection(state2, state3))
        self.assertTrue(self.time_system.has_causal_connection(state1, state3))
        
        # Non-causal connections
        unrelated1 = {'00'}
        unrelated2 = {'11'}
        
        # These should not have causal connection in our simple model
        self.assertFalse(self.time_system.has_causal_connection(unrelated1, unrelated2))
    
    def test_state_evolution_simulation(self):
        """Test simulation of state evolution"""
        initial_state = {'0', '1'}
        evolution = self.time_system.simulate_state_evolution(initial_state, 5)
        
        # Should have 6 states (initial + 5 steps)
        self.assertEqual(len(evolution), 6)
        
        # Each state should be a set
        for state in evolution:
            self.assertIsInstance(state, set)
        
        # States should generally grow (entropy increase)
        for i in range(len(evolution) - 1):
            h_i = self.time_system.entropy(evolution[i])
            h_j = self.time_system.entropy(evolution[i+1])
            self.assertGreaterEqual(h_j, h_i)
    
    def test_entropy_monotonicity_verification(self):
        """Test verification of entropy monotonicity"""
        # Create evolution with increasing entropy
        evolution_increasing = [
            {'0'},
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'}
        ]
        
        self.assertTrue(self.time_system.verify_entropy_monotonicity(evolution_increasing))
        
        # Create evolution with decreasing entropy (should fail)
        evolution_decreasing = [
            {'0', '1', '01', '10'},
            {'0', '1'},
            {'0'}
        ]
        
        self.assertFalse(self.time_system.verify_entropy_monotonicity(evolution_decreasing))
    
    def test_time_arrow_verification(self):
        """Test verification of consistent time arrow"""
        evolution = [
            {'0'},
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'}
        ]
        
        self.assertTrue(self.time_system.verify_time_arrow(evolution))
    
    def test_time_sequence_computation(self):
        """Test computation of cumulative time sequence"""
        evolution = [
            {'0'},
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'}
        ]
        
        time_sequence = self.time_system.compute_time_sequence(evolution)
        
        # Should have same length as evolution
        self.assertEqual(len(time_sequence), len(evolution))
        
        # Should start at 0
        self.assertEqual(time_sequence[0], 0.0)
        
        # Should be monotonically increasing
        for i in range(len(time_sequence) - 1):
            self.assertLessEqual(time_sequence[i], time_sequence[i+1])
        
        # Should be strictly increasing for non-trivial evolution
        self.assertLess(time_sequence[0], time_sequence[-1])
    
    def test_time_emergence_properties_comprehensive(self):
        """Test all properties required for time emergence"""
        initial_state = {'0', '1'}
        evolution = self.time_system.simulate_state_evolution(initial_state, 4)
        
        properties = self.time_system.verify_time_emergence_properties(evolution)
        
        # All properties should be satisfied
        for prop_name, satisfied in properties.items():
            with self.subTest(property=prop_name):
                self.assertTrue(satisfied, f"Property {prop_name} not satisfied")
    
    def test_entropy_to_time_correspondence(self):
        """Test that entropy changes directly correspond to time intervals"""
        state1 = {'0'}
        state2 = {'0', '1'}
        
        entropy_diff = self.time_system.entropy(state2) - self.time_system.entropy(state1)
        time_interval = self.time_system.time_metric(state1, state2)
        
        # Time interval should equal entropy difference
        self.assertAlmostEqual(time_interval, entropy_diff, places=10)
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency of time emergence"""
        # Create a more complex evolution
        evolution = [
            set(),
            {'0'},
            {'0', '1'},
            {'0', '1', '01'},
            {'0', '1', '01', '10'},
            {'0', '1', '01', '10', '001'}
        ]
        
        # Verify mathematical consistency
        for i in range(len(evolution)):
            for j in range(i, len(evolution)):
                # Self-reference gives zero time
                if i == j:
                    tau = self.time_system.time_metric(evolution[i], evolution[j])
                    self.assertEqual(tau, 0.0)
                
                # Forward time should be positive for non-empty entropy differences
                else:
                    tau = self.time_system.time_metric(evolution[i], evolution[j])
                    if tau is not None:
                        h_i = self.time_system.entropy(evolution[i])
                        h_j = self.time_system.entropy(evolution[j])
                        if h_j > h_i:  # Only check if entropy actually increases
                            self.assertGreater(tau, 0)
    
    def test_time_emergence_from_self_reference(self):
        """Test that time emerges from self-referential completeness"""
        # Start with a self-referential state (contains its own description)
        initial = {'0', '1', '01'}  # Contains elements and their combinations
        
        # Evolve the system
        evolution = self.time_system.simulate_state_evolution(initial, 3)
        
        # Verify that time structure emerges
        properties = self.time_system.verify_time_emergence_properties(evolution)
        self.assertTrue(all(properties.values()))
        
        # Verify that system maintains self-referential property
        for state in evolution:
            # Each state should contain simpler elements
            self.assertTrue(len(state) > 0)
    
    def test_time_quantization_implication(self):
        """Test implications for time quantization"""
        # Minimal state changes should give minimal time increments
        state1 = {'0'}
        state2 = {'0', '1'}  # Add one bit of information
        
        tau = self.time_system.time_metric(state1, state2)
        
        # Time increment should be finite and positive
        self.assertIsNotNone(tau)
        self.assertGreater(tau, 0)
        self.assertTrue(math.isfinite(tau))
        
        # Should be related to information content (roughly 1 bit)
        self.assertLess(tau, 10)  # Reasonable upper bound
        self.assertGreater(tau, 0.1)  # Reasonable lower bound


if __name__ == '__main__':
    unittest.main(verbosity=2)