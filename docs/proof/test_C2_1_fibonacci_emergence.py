#!/usr/bin/env python3
"""
Machine verification unit tests for C2.1: Fibonacci Emergence Corollary
Testing the corollary that self-referential complete systems follow Fibonacci counting laws.
"""

import unittest
import math
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from itertools import product


@dataclass
class BinarySequence:
    """Represents a binary sequence with no-11 constraint validation"""
    sequence: str
    
    def __post_init__(self):
        if not self.is_valid_binary():
            raise ValueError(f"Invalid binary sequence: {self.sequence}")
    
    def is_valid_binary(self) -> bool:
        """Check if sequence contains only 0s and 1s"""
        return all(bit in '01' for bit in self.sequence)
    
    def satisfies_no_11_constraint(self) -> bool:
        """Check if sequence satisfies no-11 constraint"""
        return "11" not in self.sequence
    
    def length(self) -> int:
        """Get sequence length"""
        return len(self.sequence)
    
    def append_bit(self, bit: str) -> 'BinarySequence':
        """Append bit and return new sequence"""
        return BinarySequence(self.sequence + bit)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, BinarySequence) and self.sequence == other.sequence
    
    def __hash__(self) -> int:
        return hash(self.sequence)
    
    def __str__(self) -> str:
        return self.sequence


class FibonacciSequenceGenerator:
    """Generates Fibonacci numbers"""
    
    def __init__(self):
        self._cache = {0: 0, 1: 1, 2: 1}
    
    def fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (F_0 = 0, F_1 = 1, F_2 = 1, ...)"""
        if n in self._cache:
            return self._cache[n]
        
        if n < 0:
            raise ValueError("Fibonacci number index must be non-negative")
        
        # Calculate iteratively to avoid recursion depth issues
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
            self._cache[i] = b
        
        return b
    
    def fibonacci_sequence(self, count: int) -> List[int]:
        """Get first 'count' Fibonacci numbers"""
        return [self.fibonacci(i) for i in range(count)]


class No11ConstraintSequenceCounter:
    """Counts valid sequences under no-11 constraint"""
    
    def __init__(self):
        self.fib_gen = FibonacciSequenceGenerator()
        self._count_cache = {}
    
    def count_valid_sequences_brute_force(self, length: int) -> int:
        """Count valid sequences by brute force enumeration"""
        if length <= 0:
            return 0
        
        count = 0
        # Generate all possible binary sequences of given length
        for bits in product('01', repeat=length):
            sequence_str = ''.join(bits)
            if "11" not in sequence_str:
                count += 1
        
        return count
    
    def count_valid_sequences_algorithm(self, length: int) -> int:
        """Count using the algorithm from Lemma C2.1.3"""
        if length <= 0:
            return 0
        if length == 1:
            return 2
        if length == 2:
            return 3
        
        if length in self._count_cache:
            return self._count_cache[length]
        
        prev2 = 2  # V_1
        prev1 = 3  # V_2
        
        for i in range(3, length + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        self._count_cache[length] = current
        return current
    
    def generate_valid_sequences(self, length: int) -> Set[BinarySequence]:
        """Generate all valid sequences of given length"""
        if length <= 0:
            return set()
        
        valid_sequences = set()
        
        # Generate all possible sequences and filter valid ones
        for bits in product('01', repeat=length):
            sequence_str = ''.join(bits)
            sequence = BinarySequence(sequence_str)
            if sequence.satisfies_no_11_constraint():
                valid_sequences.add(sequence)
        
        return valid_sequences
    
    def verify_fibonacci_relationship(self, max_length: int = 10) -> Dict[int, bool]:
        """Verify V_n = F_{n+2} relationship"""
        results = {}
        
        for n in range(1, max_length + 1):
            v_n = self.count_valid_sequences_algorithm(n)
            f_n_plus_2 = self.fib_gen.fibonacci(n + 2)
            results[n] = (v_n == f_n_plus_2)
        
        return results


class SelfReferentialSystemWithNo11:
    """Self-referential system that enforces no-11 constraint"""
    
    def __init__(self):
        self.sequence_counter = No11ConstraintSequenceCounter()
        self.states = set()
        self.state_count_by_length = {}
    
    def add_state_from_sequence(self, sequence: BinarySequence):
        """Add state if it satisfies no-11 constraint"""
        if sequence.satisfies_no_11_constraint():
            self.states.add(sequence)
            length = sequence.length()
            if length not in self.state_count_by_length:
                self.state_count_by_length[length] = 0
            self.state_count_by_length[length] += 1
    
    def generate_all_valid_states(self, max_length: int):
        """Generate all valid states up to max_length"""
        for length in range(1, max_length + 1):
            valid_sequences = self.sequence_counter.generate_valid_sequences(length)
            for sequence in valid_sequences:
                self.add_state_from_sequence(sequence)
    
    def verify_fibonacci_state_counting(self, max_length: int = 8) -> Dict[str, bool]:
        """Verify that state counting follows Fibonacci pattern"""
        results = {
            "fibonacci_pattern_verified": True,
            "recursive_relationship_satisfied": True,
            "boundary_conditions_correct": True
        }
        
        self.generate_all_valid_states(max_length)
        
        # Check Fibonacci relationship
        fib_verification = self.sequence_counter.verify_fibonacci_relationship(max_length)
        if not all(fib_verification.values()):
            results["fibonacci_pattern_verified"] = False
        
        # Check recursive relationship V_n = V_{n-1} + V_{n-2}
        for n in range(3, max_length + 1):
            v_n = self.sequence_counter.count_valid_sequences_algorithm(n)
            v_n_1 = self.sequence_counter.count_valid_sequences_algorithm(n - 1)
            v_n_2 = self.sequence_counter.count_valid_sequences_algorithm(n - 2)
            
            if v_n != v_n_1 + v_n_2:
                results["recursive_relationship_satisfied"] = False
        
        # Check boundary conditions
        v_1 = self.sequence_counter.count_valid_sequences_algorithm(1)
        v_2 = self.sequence_counter.count_valid_sequences_algorithm(2)
        
        if v_1 != 2 or v_2 != 3:
            results["boundary_conditions_correct"] = False
        
        return results


class FibonacciEmergenceSystem:
    """Main system implementing C2.1: Fibonacci Emergence Corollary"""
    
    def __init__(self):
        self.fib_gen = FibonacciSequenceGenerator()
        self.sequence_counter = No11ConstraintSequenceCounter()
        self.self_ref_system = SelfReferentialSystemWithNo11()
    
    def prove_state_counting_regularity_lemma(self, max_length: int = 10) -> Dict[str, bool]:
        """Prove Lemma C2.1.1: State counting follows V_n = V_{n-1} + V_{n-2}"""
        results = {
            "recursive_formula_verified": True,
            "construction_method_correct": True,
            "no_11_constraint_enforced": True
        }
        
        # Verify recursive formula
        for n in range(3, max_length + 1):
            v_n = self.sequence_counter.count_valid_sequences_algorithm(n)
            v_n_1 = self.sequence_counter.count_valid_sequences_algorithm(n - 1)
            v_n_2 = self.sequence_counter.count_valid_sequences_algorithm(n - 2)
            
            if v_n != v_n_1 + v_n_2:
                results["recursive_formula_verified"] = False
        
        # Verify construction method (compare brute force with algorithm)
        for n in range(1, min(8, max_length + 1)):  # Limit for computational efficiency
            brute_force_count = self.sequence_counter.count_valid_sequences_brute_force(n)
            algorithm_count = self.sequence_counter.count_valid_sequences_algorithm(n)
            
            if brute_force_count != algorithm_count:
                results["construction_method_correct"] = False
        
        # Verify no-11 constraint enforcement
        test_sequences = ["11", "011", "110", "1011", "1101", "0110"]
        for seq_str in test_sequences:
            if len(seq_str) <= max_length:
                sequence = BinarySequence(seq_str)
                if sequence.satisfies_no_11_constraint():
                    results["no_11_constraint_enforced"] = False
        
        return results
    
    def prove_boundary_conditions_lemma(self) -> Dict[str, bool]:
        """Prove Lemma C2.1.2: Boundary conditions V_1 = 2, V_2 = 3"""
        results = {
            "v_1_equals_2": False,
            "v_2_equals_3": False,
            "fibonacci_correspondence": False
        }
        
        # Check V_1 = 2
        v_1 = self.sequence_counter.count_valid_sequences_algorithm(1)
        valid_1_sequences = self.sequence_counter.generate_valid_sequences(1)
        
        if v_1 == 2 and len(valid_1_sequences) == 2:
            # Should be {"0", "1"}
            expected_sequences = {BinarySequence("0"), BinarySequence("1")}
            if valid_1_sequences == expected_sequences:
                results["v_1_equals_2"] = True
        
        # Check V_2 = 3
        v_2 = self.sequence_counter.count_valid_sequences_algorithm(2)
        valid_2_sequences = self.sequence_counter.generate_valid_sequences(2)
        
        if v_2 == 3 and len(valid_2_sequences) == 3:
            # Should be {"00", "01", "10"} (excluding "11")
            expected_sequences = {BinarySequence("00"), BinarySequence("01"), BinarySequence("10")}
            if valid_2_sequences == expected_sequences:
                results["v_2_equals_3"] = True
        
        # Check Fibonacci correspondence: V_n = F_{n+2}
        fibonacci_correspondence = True
        for n in range(1, 6):  # Test first few values
            v_n = self.sequence_counter.count_valid_sequences_algorithm(n)
            f_n_plus_2 = self.fib_gen.fibonacci(n + 2)
            if v_n != f_n_plus_2:
                fibonacci_correspondence = False
                break
        
        results["fibonacci_correspondence"] = fibonacci_correspondence
        
        return results
    
    def prove_algorithm_correctness_lemma(self, max_length: int = 8) -> Dict[str, bool]:
        """Prove Lemma C2.1.3: Algorithm correctness"""
        results = {
            "algorithm_implementation_correct": True,
            "boundary_conditions_handled": True,
            "recursive_relation_implemented": True,
            "output_consistency_verified": True
        }
        
        # Compare algorithm output with brute force for small lengths
        for n in range(1, max_length + 1):
            algorithm_result = self.sequence_counter.count_valid_sequences_algorithm(n)
            
            # For small n, compare with brute force
            if n <= 6:  # Limit computational complexity
                brute_force_result = self.sequence_counter.count_valid_sequences_brute_force(n)
                if algorithm_result != brute_force_result:
                    results["algorithm_implementation_correct"] = False
        
        # Check boundary conditions
        if (self.sequence_counter.count_valid_sequences_algorithm(1) != 2 or
            self.sequence_counter.count_valid_sequences_algorithm(2) != 3):
            results["boundary_conditions_handled"] = False
        
        # Check recursive relation implementation
        for n in range(3, max_length + 1):
            v_n = self.sequence_counter.count_valid_sequences_algorithm(n)
            v_n_1 = self.sequence_counter.count_valid_sequences_algorithm(n - 1)
            v_n_2 = self.sequence_counter.count_valid_sequences_algorithm(n - 2)
            
            if v_n != v_n_1 + v_n_2:
                results["recursive_relation_implemented"] = False
        
        # Check output consistency (same input should give same output)
        test_lengths = [1, 2, 3, 4, 5]
        for length in test_lengths:
            result1 = self.sequence_counter.count_valid_sequences_algorithm(length)
            result2 = self.sequence_counter.count_valid_sequences_algorithm(length)
            if result1 != result2:
                results["output_consistency_verified"] = False
        
        return results
    
    def verify_fibonacci_emergence_in_self_referential_systems(self, max_length: int = 8) -> Dict[str, bool]:
        """Verify Fibonacci emergence in self-referential systems"""
        results = {
            "fibonacci_counting_in_self_ref_systems": True,
            "no_11_constraint_necessity": True,
            "state_space_structure_correct": True
        }
        
        # Test self-referential system state counting
        system_verification = self.self_ref_system.verify_fibonacci_state_counting(max_length)
        if not all(system_verification.values()):
            results["fibonacci_counting_in_self_ref_systems"] = False
        
        # Test no-11 constraint necessity
        # Compare counts with and without constraint
        for n in range(1, min(6, max_length + 1)):
            total_sequences = 2**n  # All possible binary sequences of length n
            valid_sequences = self.sequence_counter.count_valid_sequences_algorithm(n)
            
            if n >= 2:  # For n >= 2, some sequences should be excluded
                if valid_sequences >= total_sequences:
                    results["no_11_constraint_necessity"] = False
        
        # Verify state space structure
        self.self_ref_system.generate_all_valid_states(max_length)
        for length in range(1, max_length + 1):
            expected_count = self.sequence_counter.count_valid_sequences_algorithm(length)
            actual_count = len([s for s in self.self_ref_system.states if s.length() == length])
            
            if expected_count != actual_count:
                results["state_space_structure_correct"] = False
        
        return results
    
    def prove_main_fibonacci_emergence_theorem(self, max_length: int = 8) -> Dict[str, bool]:
        """Prove main theorem: Self-referential systems follow Fibonacci counting"""
        
        # Combine all lemma proofs
        counting_regularity = self.prove_state_counting_regularity_lemma(max_length)
        boundary_conditions = self.prove_boundary_conditions_lemma()
        algorithm_correctness = self.prove_algorithm_correctness_lemma(max_length)
        fibonacci_emergence = self.verify_fibonacci_emergence_in_self_referential_systems(max_length)
        
        return {
            "state_counting_regularity_proven": all(counting_regularity.values()),
            "boundary_conditions_proven": all(boundary_conditions.values()),
            "algorithm_correctness_proven": all(algorithm_correctness.values()),
            "fibonacci_emergence_verified": all(fibonacci_emergence.values()),
            "main_theorem_proven": (
                all(counting_regularity.values()) and
                all(boundary_conditions.values()) and
                all(algorithm_correctness.values()) and
                all(fibonacci_emergence.values())
            )
        }


class TestFibonacciEmergence(unittest.TestCase):
    """Unit tests for C2.1: Fibonacci Emergence Corollary"""
    
    def setUp(self):
        self.fibonacci_system = FibonacciEmergenceSystem()
        self.max_test_length = 8  # Balance between thoroughness and speed
    
    def test_binary_sequence_basic_properties(self):
        """Test basic properties of binary sequences"""
        # Valid sequences
        valid_seq = BinarySequence("010")
        self.assertTrue(valid_seq.is_valid_binary())
        self.assertEqual(valid_seq.length(), 3)
        
        # No-11 constraint
        self.assertTrue(BinarySequence("010").satisfies_no_11_constraint())
        self.assertFalse(BinarySequence("011").satisfies_no_11_constraint())
        self.assertFalse(BinarySequence("110").satisfies_no_11_constraint())
        
        # Invalid binary should raise error
        with self.assertRaises(ValueError):
            BinarySequence("012")
    
    def test_fibonacci_sequence_generator(self):
        """Test Fibonacci sequence generation"""
        fib_gen = FibonacciSequenceGenerator()
        
        # Test known Fibonacci values
        expected_fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, expected in enumerate(expected_fibs):
            self.assertEqual(fib_gen.fibonacci(i), expected)
        
        # Test sequence generation
        fib_sequence = fib_gen.fibonacci_sequence(10)
        self.assertEqual(fib_sequence, expected_fibs)
    
    def test_no_11_constraint_sequence_counting(self):
        """Test sequence counting under no-11 constraint"""
        counter = No11ConstraintSequenceCounter()
        
        # Test known values
        self.assertEqual(counter.count_valid_sequences_algorithm(1), 2)  # "0", "1"
        self.assertEqual(counter.count_valid_sequences_algorithm(2), 3)  # "00", "01", "10"
        self.assertEqual(counter.count_valid_sequences_algorithm(3), 5)  # Should follow Fibonacci
        
        # Compare brute force with algorithm for small values
        for n in range(1, 6):
            brute_force = counter.count_valid_sequences_brute_force(n)
            algorithm = counter.count_valid_sequences_algorithm(n)
            self.assertEqual(brute_force, algorithm, f"Mismatch at length {n}")
    
    def test_fibonacci_relationship_verification(self):
        """Test V_n = F_{n+2} relationship"""
        counter = No11ConstraintSequenceCounter()
        fib_verification = counter.verify_fibonacci_relationship(self.max_test_length)
        
        for n, verified in fib_verification.items():
            with self.subTest(length=n):
                self.assertTrue(verified, f"Fibonacci relationship failed at length {n}")
    
    def test_valid_sequence_generation(self):
        """Test generation of valid sequences"""
        counter = No11ConstraintSequenceCounter()
        
        # Test length 1
        valid_1 = counter.generate_valid_sequences(1)
        expected_1 = {BinarySequence("0"), BinarySequence("1")}
        self.assertEqual(valid_1, expected_1)
        
        # Test length 2
        valid_2 = counter.generate_valid_sequences(2)
        expected_2 = {BinarySequence("00"), BinarySequence("01"), BinarySequence("10")}
        self.assertEqual(valid_2, expected_2)
        
        # Verify no "11" sequences are included
        for seq in valid_2:
            self.assertFalse("11" in seq.sequence)
    
    def test_state_counting_regularity_lemma(self):
        """Test Lemma C2.1.1: State counting regularity"""
        results = self.fibonacci_system.prove_state_counting_regularity_lemma(self.max_test_length)
        
        for aspect, proven in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(proven, f"Failed to prove state counting regularity aspect: {aspect}")
    
    def test_boundary_conditions_lemma(self):
        """Test Lemma C2.1.2: Boundary conditions"""
        results = self.fibonacci_system.prove_boundary_conditions_lemma()
        
        for aspect, proven in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(proven, f"Failed to prove boundary conditions aspect: {aspect}")
    
    def test_algorithm_correctness_lemma(self):
        """Test Lemma C2.1.3: Algorithm correctness"""
        results = self.fibonacci_system.prove_algorithm_correctness_lemma(self.max_test_length)
        
        for aspect, proven in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(proven, f"Failed to prove algorithm correctness aspect: {aspect}")
    
    def test_self_referential_system_fibonacci_counting(self):
        """Test Fibonacci counting in self-referential systems"""
        system = SelfReferentialSystemWithNo11()
        results = system.verify_fibonacci_state_counting(self.max_test_length)
        
        for aspect, verified in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(verified, f"Failed to verify self-referential system aspect: {aspect}")
    
    def test_fibonacci_emergence_verification(self):
        """Test verification of Fibonacci emergence in self-referential systems"""
        results = self.fibonacci_system.verify_fibonacci_emergence_in_self_referential_systems(self.max_test_length)
        
        for aspect, verified in results.items():
            with self.subTest(aspect=aspect):
                self.assertTrue(verified, f"Failed to verify Fibonacci emergence aspect: {aspect}")
    
    def test_main_fibonacci_emergence_theorem(self):
        """Test main theorem C2.1: Fibonacci emergence"""
        results = self.fibonacci_system.prove_main_fibonacci_emergence_theorem(self.max_test_length)
        
        # Test each component
        self.assertTrue(results["state_counting_regularity_proven"])
        self.assertTrue(results["boundary_conditions_proven"])
        self.assertTrue(results["algorithm_correctness_proven"])
        self.assertTrue(results["fibonacci_emergence_verified"])
        
        # Test main theorem
        self.assertTrue(results["main_theorem_proven"])
    
    def test_recursive_relationship_verification(self):
        """Test that V_n = V_{n-1} + V_{n-2} holds for all tested lengths"""
        counter = No11ConstraintSequenceCounter()
        
        for n in range(3, self.max_test_length + 1):
            v_n = counter.count_valid_sequences_algorithm(n)
            v_n_1 = counter.count_valid_sequences_algorithm(n - 1)
            v_n_2 = counter.count_valid_sequences_algorithm(n - 2)
            
            with self.subTest(length=n):
                self.assertEqual(v_n, v_n_1 + v_n_2, 
                               f"Recursive relationship failed at length {n}: {v_n} ≠ {v_n_1} + {v_n_2}")
    
    def test_no_11_constraint_effectiveness(self):
        """Test that no-11 constraint actually reduces sequence count"""
        counter = No11ConstraintSequenceCounter()
        
        for n in range(2, min(6, self.max_test_length + 1)):  # Starting from 2 where constraint matters
            total_possible = 2**n  # All binary sequences of length n
            valid_count = counter.count_valid_sequences_algorithm(n)
            
            with self.subTest(length=n):
                self.assertLess(valid_count, total_possible, 
                               f"No-11 constraint should reduce count at length {n}")
    
    def test_fibonacci_property_in_nature_analogy(self):
        """Test that our counting matches known Fibonacci patterns in nature"""
        # This is more of a conceptual test - verify our sequence matches
        # the mathematical properties that appear in natural Fibonacci phenomena
        
        counter = No11ConstraintSequenceCounter()
        fib_gen = FibonacciSequenceGenerator()
        
        # Golden ratio convergence test
        ratios = []
        for n in range(3, self.max_test_length + 1):
            v_n = counter.count_valid_sequences_algorithm(n)
            v_n_1 = counter.count_valid_sequences_algorithm(n - 1)
            if v_n_1 > 0:
                ratios.append(v_n / v_n_1)
        
        # Later ratios should approach golden ratio (φ ≈ 1.618)
        if len(ratios) >= 3:
            golden_ratio = (1 + math.sqrt(5)) / 2
            last_ratio = ratios[-1]
            self.assertAlmostEqual(last_ratio, golden_ratio, delta=0.1, 
                                 msg="Ratio should approach golden ratio")
    
    def test_edge_cases_and_robustness(self):
        """Test edge cases and robustness"""
        counter = No11ConstraintSequenceCounter()
        
        # Test length 0
        self.assertEqual(counter.count_valid_sequences_algorithm(0), 0)
        
        # Test negative length
        self.assertEqual(counter.count_valid_sequences_algorithm(-1), 0)
        
        # Test very small sequences
        self.assertEqual(len(counter.generate_valid_sequences(0)), 0)
        self.assertEqual(len(counter.generate_valid_sequences(1)), 2)
        
        # Test sequence operations
        seq = BinarySequence("01")
        extended_0 = seq.append_bit("0")
        extended_1 = seq.append_bit("1")
        
        self.assertEqual(extended_0.sequence, "010")
        self.assertEqual(extended_1.sequence, "011")
        self.assertTrue(extended_0.satisfies_no_11_constraint())
        self.assertFalse(extended_1.satisfies_no_11_constraint())


if __name__ == '__main__':
    unittest.main(verbosity=2)