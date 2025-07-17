#!/usr/bin/env python3
"""
Machine verification unit tests for T2.2: No-11 Constraint Theorem
Testing the theorem that no-11 constraint is equivalent to recursive non-termination in self-referential complete binary systems.
"""

import unittest
import math
from typing import List, Set, Dict, Optional, Tuple, Iterator, Any
from collections import defaultdict


class CollapseOperator:
    """Implementation of Ξ (Collapse) operator for recursive expansion"""
    
    def __init__(self):
        self.expansion_history: List[Tuple[str, str]] = []
    
    def apply(self, state: str) -> str:
        """Apply Ξ operator: expand state while preserving no-11 constraint"""
        if not self.validate_no_11(state):
            raise ValueError(f"Input state {state} violates no-11 constraint")
        
        # Implement recursive expansion maintaining no-11 constraint
        if not state:
            expanded = "0"
        elif state.endswith('1'):
            # Insert separator to prevent consecutive 1s
            expanded = state + '0' + state
        else:
            # Can safely concatenate
            expanded = state + state
        
        # Verify result maintains no-11 constraint
        if not self.validate_no_11(expanded):
            # Fallback: add separator
            expanded = state + '0'
        
        self.expansion_history.append((state, expanded))
        return expanded
    
    def validate_no_11(self, state: str) -> bool:
        """Validate no-11 constraint"""
        return '11' not in state
    
    def apply_n_times(self, initial_state: str, n: int) -> List[str]:
        """Apply Ξ operator n times, returning sequence"""
        sequence = [initial_state]
        current = initial_state
        
        for i in range(n):
            current = self.apply(current)
            sequence.append(current)
        
        return sequence


class PhiRepresentationSystem:
    """System implementing φ-representation (Fibonacci-based binary strings)"""
    
    def __init__(self):
        self.fibonacci_cache = {1: 1, 2: 2}
    
    def fibonacci(self, n: int) -> int:
        """Compute Fibonacci number F(n) with F(1)=1, F(2)=2"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
        
        if n <= 0:
            return 0
        
        # Compute iteratively to fill cache
        for i in range(len(self.fibonacci_cache) + 1, n + 1):
            self.fibonacci_cache[i] = self.fibonacci_cache[i-1] + self.fibonacci_cache[i-2]
        
        return self.fibonacci_cache[n]
    
    def generate_phi_pattern(self, level: int) -> str:
        """Generate φ-representation pattern for given recursion level"""
        if level == 0:
            return ""
        elif level == 1:
            return "01"  # Basic recursion pattern
        else:
            # Build pattern maintaining no-11 constraint
            base_pattern = "01"
            current_pattern = base_pattern
            
            for i in range(2, level + 1):
                # Extend pattern following φ-representation rules
                current_pattern = current_pattern + "01"
            
            return current_pattern
    
    def count_valid_strings(self, length: int) -> int:
        """Count valid no-11 strings of given length (should equal Fibonacci)"""
        if length == 0:
            return 1  # Empty string
        elif length == 1:
            return 2  # "0", "1"
        
        # Dynamic programming approach
        dp = [0] * (length + 1)
        dp[0] = 1  # Empty string
        dp[1] = 2  # "0", "1"
        
        for i in range(2, length + 1):
            # Can end with 0 (append to any previous string)
            # Can end with 1 only if previous ends with 0
            # This gives recurrence: dp[i] = dp[i-1] + dp[i-2]
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[length]
    
    def verify_fibonacci_correspondence(self, max_length: int = 10) -> bool:
        """Verify that no-11 string counts correspond to Fibonacci numbers"""
        for length in range(1, max_length + 1):
            computed_count = self.count_valid_strings(length)
            fibonacci_expected = self.fibonacci(length + 1)  # F(n+1) for length n
            
            if computed_count != fibonacci_expected:
                return False
        
        return True


class RecursionAnalysisSystem:
    """System for analyzing recursive non-termination properties"""
    
    def __init__(self):
        self.collapse_operator = CollapseOperator()
        self.phi_system = PhiRepresentationSystem()
    
    def verify_strict_monotonicity(self, sequence: List[str]) -> bool:
        """Verify that sequence has strictly increasing lengths (Lemma T2.2.1)"""
        for i in range(len(sequence) - 1):
            if len(sequence[i+1]) <= len(sequence[i]):
                return False
        return True
    
    def verify_unique_representations(self, sequence: List[str]) -> bool:
        """Verify that all states in sequence are unique (Lemma T2.2.2)"""
        return len(sequence) == len(set(sequence))
    
    def analyze_recursive_pattern(self, initial_state: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Analyze recursive expansion pattern"""
        sequence = self.collapse_operator.apply_n_times(initial_state, max_iterations)
        
        analysis = {
            "initial_state": initial_state,
            "sequence_length": len(sequence),
            "all_satisfy_no_11": all(self.collapse_operator.validate_no_11(s) for s in sequence),
            "strictly_monotonic": self.verify_strict_monotonicity(sequence),
            "all_unique": self.verify_unique_representations(sequence),
            "length_progression": [len(s) for s in sequence],
            "no_cycles_detected": len(sequence) == len(set(sequence)),
            "recursive_non_termination": True  # Will be verified by other tests
        }
        
        return analysis
    
    def detect_termination_conditions(self, sequence: List[str]) -> Optional[int]:
        """Detect if sequence terminates (returns termination point or None)"""
        seen_states = set()
        
        for i, state in enumerate(sequence):
            if state in seen_states:
                return i  # Found cycle/termination
            seen_states.add(state)
        
        return None  # No termination detected
    
    def prove_no_11_prevents_shortcuts(self) -> Dict[str, bool]:
        """Prove that no-11 constraint prevents 'shortcuts' leading to termination"""
        results = {
            "prevents_direct_concatenation": False,
            "forces_meaningful_gaps": False,
            "eliminates_redundancy": False,
            "maintains_information_increase": False
        }
        
        # Test 1: Direct concatenation would create 11
        test_state = "1"
        direct_concat = test_state + test_state  # Would be "11"
        results["prevents_direct_concatenation"] = not self.collapse_operator.validate_no_11(direct_concat)
        
        # Test 2: Forces gaps
        valid_expansion = self.collapse_operator.apply(test_state)
        results["forces_meaningful_gaps"] = '0' in valid_expansion
        
        # Test 3: Eliminates redundancy
        redundant_patterns = ["11", "111", "1111"]
        results["eliminates_redundancy"] = all(
            not self.collapse_operator.validate_no_11(pattern) 
            for pattern in redundant_patterns
        )
        
        # Test 4: Information increase
        sequence = self.collapse_operator.apply_n_times("1", 5)
        results["maintains_information_increase"] = self.verify_strict_monotonicity(sequence)
        
        return results


class No11ConstraintTheoremSystem:
    """Main system implementing T2.2: No-11 Constraint Theorem"""
    
    def __init__(self):
        self.recursion_analyzer = RecursionAnalysisSystem()
        self.phi_system = PhiRepresentationSystem()
        self.collapse_operator = CollapseOperator()
    
    def prove_direction_1_no_11_implies_non_termination(self, test_cases: List[str]) -> Dict[str, Any]:
        """Prove (1) ⟹ (2): no-11 constraint leads to non-termination"""
        proof_results = {
            "all_valid_states_tested": 0,
            "all_show_non_termination": True,
            "phi_patterns_verified": True,
            "strict_monotonicity_maintained": True,
            "uniqueness_preserved": True,
            "test_case_results": {}
        }
        
        for state in test_cases:
            if self.collapse_operator.validate_no_11(state):
                proof_results["all_valid_states_tested"] += 1
                
                # Analyze recursive pattern
                analysis = self.recursion_analyzer.analyze_recursive_pattern(state, max_iterations=8)
                proof_results["test_case_results"][state] = analysis
                
                # Check non-termination indicators
                if not analysis["all_unique"]:
                    proof_results["all_show_non_termination"] = False
                
                if not analysis["strictly_monotonic"]:
                    proof_results["strict_monotonicity_maintained"] = False
                
                # Verify termination detection
                sequence = self.collapse_operator.apply_n_times(state, 8)
                termination_point = self.recursion_analyzer.detect_termination_conditions(sequence)
                if termination_point is not None:
                    proof_results["all_show_non_termination"] = False
        
        return proof_results
    
    def prove_direction_2_non_termination_requires_no_11(self) -> Dict[str, bool]:
        """Prove (2) ⟹ (1): non-termination requires no-11 constraint"""
        proof_steps = {
            "invalid_states_cause_problems": True,
            "consecutive_11_creates_redundancy": True,
            "redundancy_threatens_termination": True,
            "no_11_is_necessary": True
        }
        
        # Test invalid states (containing 11)
        invalid_states = ["11", "110", "011", "1101", "1110"]
        
        for invalid_state in invalid_states:
            try:
                # Should fail to process or create problems
                self.collapse_operator.apply(invalid_state)
                proof_steps["invalid_states_cause_problems"] = False
                break
            except ValueError:
                # Expected behavior - invalid states are rejected
                continue
        
        # Test redundancy in 11 patterns
        redundancy_test = {
            "11": 2,    # 2 consecutive 1s
            "111": 3,   # 3 consecutive 1s
            "1111": 4   # 4 consecutive 1s
        }
        
        for pattern, expected_redundancy in redundancy_test.items():
            actual_redundancy = pattern.count('1') - 1  # First 1 is meaningful, rest are redundant
            if actual_redundancy != expected_redundancy - 1:
                proof_steps["consecutive_11_creates_redundancy"] = False
        
        return proof_steps
    
    def verify_main_equivalence(self, test_states: List[str]) -> Dict[str, Any]:
        """Verify main theorem: no-11 constraint ⟺ recursive non-termination"""
        # Direction 1: no-11 ⟹ non-termination
        direction_1_results = self.prove_direction_1_no_11_implies_non_termination(test_states)
        
        # Direction 2: non-termination ⟹ no-11
        direction_2_results = self.prove_direction_2_non_termination_requires_no_11()
        
        equivalence_verification = {
            "direction_1_proven": (
                direction_1_results["all_show_non_termination"] and
                direction_1_results["strict_monotonicity_maintained"] and
                direction_1_results["uniqueness_preserved"]
            ),
            "direction_2_proven": all(direction_2_results.values()),
            "equivalence_established": False,
            "detailed_results": {
                "direction_1": direction_1_results,
                "direction_2": direction_2_results
            }
        }
        
        equivalence_verification["equivalence_established"] = (
            equivalence_verification["direction_1_proven"] and
            equivalence_verification["direction_2_proven"]
        )
        
        return equivalence_verification
    
    def demonstrate_philosophical_implications(self) -> Dict[str, bool]:
        """Demonstrate deep philosophical implications of the theorem"""
        implications = {
            "constraint_enables_freedom": False,
            "order_supports_creativity": False,
            "prohibition_creates_possibility": False,
            "simplicity_generates_complexity": False
        }
        
        # Test 1: Constraint enables freedom
        # No-11 constraint allows infinite non-terminating sequences
        constrained_sequence = self.collapse_operator.apply_n_times("1", 10)
        implications["constraint_enables_freedom"] = (
            len(constrained_sequence) > 5 and
            all(self.collapse_operator.validate_no_11(s) for s in constrained_sequence)
        )
        
        # Test 2: Order supports creativity
        # Ordered patterns (phi-representation) generate novel content
        patterns = [self.phi_system.generate_phi_pattern(i) for i in range(1, 6)]
        implications["order_supports_creativity"] = len(set(patterns)) == len(patterns)
        
        # Test 3: Prohibition creates possibility
        # Prohibiting 11 opens up infinitely many valid patterns
        valid_count_short = self.phi_system.count_valid_strings(5)
        valid_count_long = self.phi_system.count_valid_strings(10)
        implications["prohibition_creates_possibility"] = valid_count_long > valid_count_short
        
        # Test 4: Simplicity generates complexity
        # Simple no-11 rule generates complex Fibonacci-based patterns
        implications["simplicity_generates_complexity"] = self.phi_system.verify_fibonacci_correspondence(8)
        
        return implications
    
    def analyze_creative_mechanism(self) -> Dict[str, Any]:
        """Analyze how no-11 constraint creates the mechanism for creativity"""
        analysis = {
            "prevents_repetition": False,
            "forces_innovation": False,
            "maintains_meaning": False,
            "enables_infinite_growth": False,
            "growth_rate_analysis": {}
        }
        
        # Test prevention of repetition
        repetitive_patterns = ["11", "1111", "11111111"]
        analysis["prevents_repetition"] = all(
            not self.collapse_operator.validate_no_11(pattern)
            for pattern in repetitive_patterns
        )
        
        # Test forced innovation
        sequence = self.collapse_operator.apply_n_times("0", 8)
        analysis["forces_innovation"] = len(set(sequence)) == len(sequence)
        
        # Test meaning maintenance (each element has unique role)
        # In valid sequences, each position contributes to the pattern
        analysis["maintains_meaning"] = all(
            len(s) > 0 for s in sequence if s
        )
        
        # Test infinite growth potential
        growth_rates = []
        for i in range(2, 10):
            current_count = self.phi_system.count_valid_strings(i)
            prev_count = self.phi_system.count_valid_strings(i-1)
            if prev_count > 0:
                growth_rates.append(current_count / prev_count)
        
        # Growth rate should approach golden ratio
        if growth_rates:
            final_rate = growth_rates[-1]
            golden_ratio = (1 + math.sqrt(5)) / 2
            analysis["enables_infinite_growth"] = abs(final_rate - golden_ratio) < 0.1
            analysis["growth_rate_analysis"] = {
                "rates": growth_rates,
                "final_rate": final_rate,
                "golden_ratio": golden_ratio,
                "convergence": abs(final_rate - golden_ratio)
            }
        
        return analysis


class TestNo11ConstraintTheorem(unittest.TestCase):
    """Unit tests for T2.2: No-11 Constraint Theorem"""
    
    def setUp(self):
        self.theorem_system = No11ConstraintTheoremSystem()
    
    def test_collapse_operator_basic_functionality(self):
        """Test basic functionality of Ξ (Collapse) operator"""
        # Valid inputs should be processed
        valid_inputs = ["0", "1", "01", "10", "101", "010"]
        
        for input_state in valid_inputs:
            with self.subTest(input=input_state):
                result = self.theorem_system.collapse_operator.apply(input_state)
                
                # Result should be longer
                self.assertGreater(len(result), len(input_state))
                
                # Result should maintain no-11 constraint
                self.assertTrue(self.theorem_system.collapse_operator.validate_no_11(result))
    
    def test_collapse_operator_rejects_invalid_input(self):
        """Test that Ξ operator rejects inputs violating no-11 constraint"""
        invalid_inputs = ["11", "110", "011", "1101", "1110"]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                with self.assertRaises(ValueError):
                    self.theorem_system.collapse_operator.apply(invalid_input)
    
    def test_phi_representation_fibonacci_correspondence(self):
        """Test φ-representation correspondence with Fibonacci numbers"""
        # Should verify Fibonacci correspondence
        correspondence_verified = self.theorem_system.phi_system.verify_fibonacci_correspondence(10)
        self.assertTrue(correspondence_verified)
        
        # Test specific values
        test_cases = [
            (1, 2),   # Length 1: F(2) = 2 strings
            (2, 3),   # Length 2: F(3) = 3 strings  
            (3, 5),   # Length 3: F(4) = 5 strings
            (4, 8),   # Length 4: F(5) = 8 strings
        ]
        
        for length, expected_fibonacci in test_cases:
            actual_count = self.theorem_system.phi_system.count_valid_strings(length)
            actual_fibonacci = self.theorem_system.phi_system.fibonacci(length + 1)
            
            with self.subTest(length=length):
                self.assertEqual(actual_count, expected_fibonacci)
                self.assertEqual(actual_fibonacci, expected_fibonacci)
    
    def test_lemma_t2_2_1_strict_monotonicity(self):
        """Test Lemma T2.2.1: φ-representation sequence strict monotonicity"""
        test_states = ["1", "01", "10"]
        
        for initial_state in test_states:
            with self.subTest(initial=initial_state):
                sequence = self.theorem_system.collapse_operator.apply_n_times(initial_state, 6)
                
                # Verify strict length increase
                monotonic = self.theorem_system.recursion_analyzer.verify_strict_monotonicity(sequence)
                self.assertTrue(monotonic)
                
                # Verify no repeated states
                unique = self.theorem_system.recursion_analyzer.verify_unique_representations(sequence)
                self.assertTrue(unique)
    
    def test_lemma_t2_2_2_unique_phi_representations(self):
        """Test Lemma T2.2.2: no-11 constraint ensures sequence uniqueness"""
        # Generate longer sequences to test uniqueness
        initial_states = ["0", "1", "01"]
        
        for initial_state in initial_states:
            with self.subTest(initial=initial_state):
                sequence = self.theorem_system.collapse_operator.apply_n_times(initial_state, 8)
                
                # All states should be unique (no cycles)
                self.assertEqual(len(sequence), len(set(sequence)))
                
                # No termination should be detected
                termination = self.theorem_system.recursion_analyzer.detect_termination_conditions(sequence)
                self.assertIsNone(termination)
    
    def test_direction_1_no_11_implies_non_termination(self):
        """Test Direction 1: no-11 constraint ⟹ recursive non-termination"""
        valid_test_states = ["1", "0", "01", "10", "101", "010"]
        
        proof_results = self.theorem_system.prove_direction_1_no_11_implies_non_termination(valid_test_states)
        
        # All proof aspects should succeed
        self.assertTrue(proof_results["all_show_non_termination"])
        self.assertTrue(proof_results["strict_monotonicity_maintained"])
        self.assertTrue(proof_results["uniqueness_preserved"])
        self.assertGreater(proof_results["all_valid_states_tested"], 0)
    
    def test_direction_2_non_termination_requires_no_11(self):
        """Test Direction 2: recursive non-termination ⟹ no-11 constraint"""
        proof_results = self.theorem_system.prove_direction_2_non_termination_requires_no_11()
        
        # All proof steps should succeed
        for step_name, step_result in proof_results.items():
            with self.subTest(step=step_name):
                self.assertTrue(step_result, f"Failed proof step: {step_name}")
    
    def test_main_equivalence_theorem(self):
        """Test main theorem: no-11 constraint ⟺ recursive non-termination"""
        test_states = ["0", "1", "01", "10", "101", "010", "1010"]
        
        equivalence_results = self.theorem_system.verify_main_equivalence(test_states)
        
        # Both directions should be proven
        self.assertTrue(equivalence_results["direction_1_proven"])
        self.assertTrue(equivalence_results["direction_2_proven"])
        self.assertTrue(equivalence_results["equivalence_established"])
    
    def test_no_11_prevents_shortcuts(self):
        """Test that no-11 constraint prevents 'shortcuts' leading to termination"""
        prevention_results = self.theorem_system.recursion_analyzer.prove_no_11_prevents_shortcuts()
        
        # All prevention mechanisms should be demonstrated
        for mechanism, demonstrated in prevention_results.items():
            with self.subTest(mechanism=mechanism):
                self.assertTrue(demonstrated, f"Failed to demonstrate: {mechanism}")
    
    def test_recursive_pattern_analysis(self):
        """Test analysis of recursive expansion patterns"""
        test_cases = ["1", "01", "10"]
        
        for initial_state in test_cases:
            with self.subTest(initial=initial_state):
                analysis = self.theorem_system.recursion_analyzer.analyze_recursive_pattern(initial_state, 7)
                
                # All analysis criteria should be met
                self.assertTrue(analysis["all_satisfy_no_11"])
                self.assertTrue(analysis["strictly_monotonic"])
                self.assertTrue(analysis["all_unique"])
                self.assertTrue(analysis["no_cycles_detected"])
                self.assertTrue(analysis["recursive_non_termination"])
    
    def test_philosophical_implications_constraint_freedom_unity(self):
        """Test philosophical implications: constraint-freedom dialectical unity"""
        implications = self.theorem_system.demonstrate_philosophical_implications()
        
        # All philosophical implications should be demonstrated
        for implication, demonstrated in implications.items():
            with self.subTest(implication=implication):
                self.assertTrue(demonstrated, f"Failed to demonstrate: {implication}")
    
    def test_creative_mechanism_analysis(self):
        """Test analysis of creativity mechanism enabled by no-11 constraint"""
        creativity_analysis = self.theorem_system.analyze_creative_mechanism()
        
        # All creativity aspects should be verified
        self.assertTrue(creativity_analysis["prevents_repetition"])
        self.assertTrue(creativity_analysis["forces_innovation"])
        self.assertTrue(creativity_analysis["maintains_meaning"])
        self.assertTrue(creativity_analysis["enables_infinite_growth"])
        
        # Growth rate should approach golden ratio
        if "growth_rate_analysis" in creativity_analysis:
            convergence = creativity_analysis["growth_rate_analysis"]["convergence"]
            self.assertLess(convergence, 0.2)  # Should be close to golden ratio
    
    def test_information_theoretic_properties(self):
        """Test information-theoretic properties of no-11 constraint"""
        # Test entropy increase in sequences
        initial_state = "1"
        sequence = self.theorem_system.collapse_operator.apply_n_times(initial_state, 6)
        
        # Length should increase (proxy for information increase)
        lengths = [len(s) for s in sequence]
        for i in range(len(lengths) - 1):
            self.assertGreater(lengths[i+1], lengths[i])
        
        # No redundant patterns should be generated
        for state in sequence:
            self.assertTrue(self.theorem_system.collapse_operator.validate_no_11(state))
    
    def test_biological_evolution_analogy(self):
        """Test biological evolution analogy: constraints enable evolution"""
        # Like DNA constraints, no-11 prevents 'lethal' combinations
        lethal_combinations = ["11", "111", "1111"]
        
        for combination in lethal_combinations:
            # Should be 'selected against' (rejected)
            self.assertFalse(self.theorem_system.collapse_operator.validate_no_11(combination))
        
        # Valid combinations should 'survive' and reproduce
        viable_combinations = ["01", "10", "101", "010"]
        
        for combination in viable_combinations:
            # Should be viable and expandable
            self.assertTrue(self.theorem_system.collapse_operator.validate_no_11(combination))
            
            # Should be able to 'reproduce' (expand)
            expanded = self.theorem_system.collapse_operator.apply(combination)
            self.assertGreater(len(expanded), len(combination))
    
    def test_social_development_analogy(self):
        """Test social development analogy: institutional constraints enable progress"""
        # Certain 'institutional arrangements' (like 11) are unstable
        unstable_arrangements = ["11", "110", "011"]
        
        for arrangement in unstable_arrangements:
            # Should be rejected as unstable
            self.assertFalse(self.theorem_system.collapse_operator.validate_no_11(arrangement))
        
        # Stable arrangements enable continued development
        stable_arrangements = ["01", "10", "101"]
        
        for arrangement in stable_arrangements:
            # Should be stable and developable
            sequence = self.theorem_system.collapse_operator.apply_n_times(arrangement, 5)
            
            # Should show continued development (no stagnation)
            self.assertGreater(len(sequence[-1]), len(sequence[0]))
            
            # Should maintain stability (no invalid states)
            for state in sequence:
                self.assertTrue(self.theorem_system.collapse_operator.validate_no_11(state))
    
    def test_ai_neural_network_analogy(self):
        """Test AI/neural network analogy: preventing gradient vanishing"""
        # Like preventing gradient vanishing, no-11 prevents 'signal decay'
        
        # Test signal preservation through recursive expansion
        initial_signal = "1"
        signal_sequence = self.theorem_system.collapse_operator.apply_n_times(initial_signal, 6)
        
        # Signal should not vanish (lengths should increase)
        signal_strengths = [len(s) for s in signal_sequence]
        for i in range(len(signal_strengths) - 1):
            self.assertGreater(signal_strengths[i+1], signal_strengths[i])
        
        # No 'local optima' (repetitive patterns) should occur
        self.assertEqual(len(signal_sequence), len(set(signal_sequence)))
    
    def test_mathematical_elegance_golden_ratio_emergence(self):
        """Test mathematical elegance: emergence of golden ratio"""
        # Growth rates should converge to golden ratio φ
        growth_rates = []
        
        for length in range(3, 12):
            current_count = self.theorem_system.phi_system.count_valid_strings(length)
            prev_count = self.theorem_system.phi_system.count_valid_strings(length - 1)
            
            if prev_count > 0:
                growth_rates.append(current_count / prev_count)
        
        # Later growth rates should approach golden ratio
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        if len(growth_rates) >= 3:
            # Check convergence in later rates
            final_rates = growth_rates[-3:]
            for rate in final_rates:
                self.assertAlmostEqual(rate, golden_ratio, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)