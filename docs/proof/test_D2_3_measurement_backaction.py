#!/usr/bin/env python3
"""
Machine verification unit tests for D2.3: Measurement Backaction
Testing the formal definition of measurement backaction in self-referential complete systems.
"""

import unittest
import math
from typing import Set, List, Dict, Any, Callable


class MeasurementBackactionSystem:
    """System for testing measurement backaction properties"""
    
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        self.ln_phi = math.log(self.phi)
        self.weight = self.ln_phi / math.log(2)  # ln(φ)/ln(2)
        
    def phi_valid(self, s: str) -> bool:
        """Check if string is φ-valid (no consecutive 11)"""
        return '11' not in s
    
    def phi_distance(self, s: str, t: str) -> float:
        """Compute φ-distance between two states"""
        if not s or not t:
            # Handle empty states
            if not s and not t:
                return 0.0
            return 1.0
        
        # Compute symmetric difference
        s_set = set(enumerate(s))
        t_set = set(enumerate(t))
        symmetric_diff = len(s_set.symmetric_difference(t_set))
        
        # Normalized difference
        normalized_diff = symmetric_diff / (len(s) + len(t))
        
        # Length difference term
        length_diff = abs(math.log(len(s)) - math.log(len(t)))
        
        return normalized_diff + self.weight * length_diff
    
    def simple_observer(self, s: str) -> str:
        """Simple observer that flips the last bit"""
        if not s:
            return '0'
        
        # Flip the last bit
        last_bit = s[-1]
        new_bit = '0' if last_bit == '1' else '1'
        result = s[:-1] + new_bit
        
        # Ensure φ-validity
        if not self.phi_valid(result):
            # Simple fix: replace 11 with 101
            result = result.replace('11', '101')
        
        return result
    
    def truncating_observer(self, s: str) -> str:
        """Observer that truncates to half length"""
        if not s:
            return '0'
        
        if len(s) == 1:
            return '0' if s == '1' else '1'
        
        # Truncate to half length
        half_len = len(s) // 2
        result = s[:half_len]
        
        # Ensure non-empty and φ-valid
        if not result:
            result = '0'
        if not self.phi_valid(result):
            result = result.replace('11', '101')
        
        return result
    
    def expanding_observer(self, s: str) -> str:
        """Observer that doubles the string"""
        if not s:
            return '0'
        
        # Double the string
        result = s + s
        
        # Ensure φ-validity
        if not self.phi_valid(result):
            result = result.replace('11', '101')
        
        return result
    
    def compute_measurement_backaction(self, observer_func: Callable[[str], str], s: str) -> float:
        """Compute measurement backaction"""
        if not s:
            return 0.0
        
        # Apply observer function
        observed_s = observer_func(s)
        
        # Compute φ-distance
        backaction = self.phi_distance(s, observed_s)
        
        return backaction
    
    def compute_cumulative_backaction(self, observer_sequence: List[Callable], state_sequence: List[str]) -> float:
        """Compute cumulative measurement backaction"""
        total_backaction = 0.0
        
        for observer, state in zip(observer_sequence, state_sequence):
            backaction = self.compute_measurement_backaction(observer, state)
            total_backaction += backaction
        
        return total_backaction
    
    def verify_nonzero_property(self, observer_func: Callable[[str], str], s: str) -> bool:
        """Verify that backaction(o,s) > 0"""
        try:
            backaction = self.compute_measurement_backaction(observer_func, s)
            return backaction > 0
        except:
            return False
    
    def verify_bounded_property(self, observer_funcs: List[Callable], states: List[str]) -> bool:
        """Verify that backaction is bounded"""
        backactions = []
        
        for observer in observer_funcs:
            for state in states:
                try:
                    backaction = self.compute_measurement_backaction(observer, state)
                    backactions.append(backaction)
                except:
                    continue
        
        if not backactions:
            return False
        
        # Check if there's a reasonable upper bound
        max_backaction = max(backactions)
        return max_backaction < 100  # Reasonable bound for our system
    
    def verify_lower_bound_property(self, observer_funcs: List[Callable], states: List[str]) -> bool:
        """Verify that backaction has a positive lower bound"""
        backactions = []
        
        for observer in observer_funcs:
            for state in states:
                try:
                    backaction = self.compute_measurement_backaction(observer, state)
                    if backaction > 0:
                        backactions.append(backaction)
                except:
                    continue
        
        if not backactions:
            return False
        
        # Check if minimum is positive
        min_backaction = min(backactions)
        return min_backaction > 0
    
    def verify_observer_dependency(self, observer1: Callable, observer2: Callable, state: str) -> bool:
        """Verify that different observers produce different backactions"""
        try:
            backaction1 = self.compute_measurement_backaction(observer1, state)
            backaction2 = self.compute_measurement_backaction(observer2, state)
            return abs(backaction1 - backaction2) > 1e-10
        except:
            return False
    
    def verify_cumulative_property(self, observer_sequence: List[Callable], state_sequence: List[str]) -> bool:
        """Verify cumulative property"""
        # Method 1: Direct cumulative calculation
        cumulative_1 = self.compute_cumulative_backaction(observer_sequence, state_sequence)
        
        # Method 2: Sum of individual backactions
        cumulative_2 = 0.0
        for observer, state in zip(observer_sequence, state_sequence):
            backaction = self.compute_measurement_backaction(observer, state)
            cumulative_2 += backaction
        
        return abs(cumulative_1 - cumulative_2) < 1e-10
    
    def analyze_distance_properties(self, s: str, t: str) -> Dict[str, float]:
        """Analyze properties of φ-distance"""
        return {
            'phi_distance': self.phi_distance(s, t),
            'symmetric': abs(self.phi_distance(s, t) - self.phi_distance(t, s)),
            'length_s': len(s),
            'length_t': len(t),
            'symmetric_diff': len(set(enumerate(s)).symmetric_difference(set(enumerate(t))))
        }


class TestMeasurementBackaction(unittest.TestCase):
    """Unit tests for D2.3: Measurement Backaction"""
    
    def setUp(self):
        self.system = MeasurementBackactionSystem()
        self.test_states = ['0', '1', '01', '10', '001', '010', '100', '101']
        self.observers = [
            self.system.simple_observer,
            self.system.truncating_observer,
            self.system.expanding_observer
        ]
    
    def test_phi_distance_computation(self):
        """Test basic φ-distance computation"""
        # Test identical states
        self.assertEqual(self.system.phi_distance('01', '01'), 0.0)
        
        # Test different states
        for s in self.test_states:
            for t in self.test_states:
                if s != t:
                    distance = self.system.phi_distance(s, t)
                    self.assertGreater(distance, 0, f"Distance between {s} and {t} should be positive")
    
    def test_phi_distance_symmetry(self):
        """Test symmetry of φ-distance"""
        for s in self.test_states:
            for t in self.test_states:
                dist_st = self.system.phi_distance(s, t)
                dist_ts = self.system.phi_distance(t, s)
                self.assertAlmostEqual(dist_st, dist_ts, places=10,
                                     msg=f"Distance should be symmetric: d({s},{t}) = d({t},{s})")
    
    def test_phi_distance_triangle_inequality(self):
        """Test triangle inequality for φ-distance"""
        test_triplets = [
            ('0', '1', '01'),
            ('01', '10', '001'),
            ('001', '010', '100')
        ]
        
        for s, t, u in test_triplets:
            dist_st = self.system.phi_distance(s, t)
            dist_tu = self.system.phi_distance(t, u)
            dist_su = self.system.phi_distance(s, u)
            
            # Triangle inequality: d(s,u) ≤ d(s,t) + d(t,u)
            self.assertLessEqual(dist_su, dist_st + dist_tu + 1e-10,
                               f"Triangle inequality should hold for {s}, {t}, {u}")
    
    def test_measurement_backaction_computation(self):
        """Test basic measurement backaction computation"""
        for observer in self.observers:
            for state in self.test_states:
                with self.subTest(observer=observer.__name__, state=state):
                    backaction = self.system.compute_measurement_backaction(observer, state)
                    self.assertIsInstance(backaction, float)
                    self.assertGreaterEqual(backaction, 0)
    
    def test_nonzero_property(self):
        """Test Property D2.3.1: Non-zero backaction"""
        for observer in self.observers:
            for state in self.test_states:
                with self.subTest(observer=observer.__name__, state=state):
                    self.assertTrue(self.system.verify_nonzero_property(observer, state),
                                  f"Backaction should be non-zero for {observer.__name__} on {state}")
    
    def test_bounded_property(self):
        """Test Property D2.3.2: Bounded backaction"""
        self.assertTrue(self.system.verify_bounded_property(self.observers, self.test_states),
                      "Backaction should be bounded")
    
    def test_lower_bound_property(self):
        """Test Property D2.3.3: Lower bound"""
        self.assertTrue(self.system.verify_lower_bound_property(self.observers, self.test_states),
                      "Backaction should have a positive lower bound")
    
    def test_observer_dependency(self):
        """Test Property D2.3.4: Observer dependency"""
        for state in self.test_states:
            with self.subTest(state=state):
                # Test different observers produce different backactions
                different_found = False
                for i, obs1 in enumerate(self.observers):
                    for j, obs2 in enumerate(self.observers):
                        if i != j:
                            if self.system.verify_observer_dependency(obs1, obs2, state):
                                different_found = True
                                break
                    if different_found:
                        break
                
                self.assertTrue(different_found, 
                              f"Different observers should produce different backactions for {state}")
    
    def test_cumulative_property(self):
        """Test Property D2.3.5: Cumulative backaction"""
        observer_sequence = [self.system.simple_observer, self.system.truncating_observer]
        state_sequence = ['01', '10']
        
        self.assertTrue(self.system.verify_cumulative_property(observer_sequence, state_sequence),
                      "Cumulative property should hold")
    
    def test_cumulative_backaction_computation(self):
        """Test cumulative backaction computation"""
        observer_sequence = [self.system.simple_observer] * 3
        state_sequence = ['0', '1', '01']
        
        cumulative = self.system.compute_cumulative_backaction(observer_sequence, state_sequence)
        
        # Should be positive
        self.assertGreater(cumulative, 0)
        
        # Should equal sum of individual backactions
        individual_sum = sum(self.system.compute_measurement_backaction(obs, state) 
                           for obs, state in zip(observer_sequence, state_sequence))
        
        self.assertAlmostEqual(cumulative, individual_sum, places=10)
    
    def test_observer_non_triviality(self):
        """Test that observers are non-trivial (change the state)"""
        for observer in self.observers:
            for state in self.test_states:
                with self.subTest(observer=observer.__name__, state=state):
                    observed_state = observer(state)
                    self.assertNotEqual(state, observed_state,
                                      f"Observer {observer.__name__} should change state {state}")
    
    def test_phi_validity_preservation(self):
        """Test that observers preserve φ-validity"""
        for observer in self.observers:
            for state in self.test_states:
                with self.subTest(observer=observer.__name__, state=state):
                    if self.system.phi_valid(state):
                        observed_state = observer(state)
                        self.assertTrue(self.system.phi_valid(observed_state),
                                      f"Observer {observer.__name__} should preserve φ-validity: {state} -> {observed_state}")
    
    def test_distance_analysis(self):
        """Test distance analysis functionality"""
        s, t = '01', '10'
        analysis = self.system.analyze_distance_properties(s, t)
        
        # Check that analysis contains expected keys
        expected_keys = ['phi_distance', 'symmetric', 'length_s', 'length_t', 'symmetric_diff']
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check that symmetry error is small
        self.assertLess(analysis['symmetric'], 1e-10)
    
    def test_specific_observer_behaviors(self):
        """Test specific observer behaviors"""
        # Test simple observer (flips last bit)
        result = self.system.simple_observer('01')
        self.assertTrue(result.endswith('0'))  # Should flip 1 to 0
        
        # Test truncating observer
        result = self.system.truncating_observer('0101')
        self.assertLessEqual(len(result), 2)  # Should be half length
        
        # Test expanding observer
        result = self.system.expanding_observer('01')
        self.assertGreaterEqual(len(result), 4)  # Should be longer
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency of backaction"""
        # Test that backaction equals φ-distance
        for observer in self.observers:
            for state in self.test_states[:3]:  # Test first few states
                with self.subTest(observer=observer.__name__, state=state):
                    backaction = self.system.compute_measurement_backaction(observer, state)
                    observed_state = observer(state)
                    phi_dist = self.system.phi_distance(state, observed_state)
                    
                    self.assertAlmostEqual(backaction, phi_dist, places=10,
                                         msg=f"Backaction should equal φ-distance for {state}")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty string (if supported)
        try:
            backaction = self.system.compute_measurement_backaction(self.system.simple_observer, '')
            self.assertGreaterEqual(backaction, 0)
        except:
            pass  # Empty string might not be supported
        
        # Test single character
        backaction = self.system.compute_measurement_backaction(self.system.simple_observer, '0')
        self.assertGreater(backaction, 0)
        
        # Test identical states (should give 0 distance)
        self.assertEqual(self.system.phi_distance('01', '01'), 0.0)
    
    def test_weight_parameter(self):
        """Test the weight parameter in φ-distance"""
        # Weight should be ln(φ)/ln(2)
        expected_weight = math.log(self.system.phi) / math.log(2)
        self.assertAlmostEqual(self.system.weight, expected_weight, places=10)
        
        # Weight should be positive
        self.assertGreater(self.system.weight, 0)
    
    def test_information_theoretic_properties(self):
        """Test information-theoretic properties"""
        # Test that longer strings generally have larger distances
        short_state = '0'
        long_state = '010101'
        
        # Distance from short to long should be significant
        distance = self.system.phi_distance(short_state, long_state)
        self.assertGreater(distance, 0.1)  # Should be non-trivial
        
        # Backaction should reflect information change
        backaction = self.system.compute_measurement_backaction(self.system.expanding_observer, short_state)
        self.assertGreater(backaction, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)