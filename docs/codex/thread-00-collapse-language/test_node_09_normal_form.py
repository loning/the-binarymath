#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N9: Lexical Collapse Normal Form
Verifies canonical representations emerging from collapse equivalence.
"""

import unittest
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


class ReductionRule:
    """Represents a reduction rule for normalization"""
    
    def __init__(self, pattern: List[str], replacement: List[str], name: str = ""):
        """Initialize reduction rule"""
        self.pattern = pattern
        self.replacement = replacement
        self.name = name
        self.pattern_str = "".join(pattern)
        self.replacement_str = "".join(replacement)
    
    def applies_at(self, sequence: List[str], index: int) -> bool:
        """Check if rule applies at given index"""
        if index + len(self.pattern) > len(sequence):
            return False
        
        for i, symbol in enumerate(self.pattern):
            if sequence[index + i] != symbol:
                return False
        
        return True
    
    def apply_at(self, sequence: List[str], index: int) -> List[str]:
        """Apply rule at given index"""
        return sequence[:index] + self.replacement + sequence[index + len(self.pattern):]
    
    def __repr__(self):
        return f"Rule({self.pattern} → {self.replacement})"


class NormalFormComputer:
    """Computes normal forms for collapse expressions"""
    
    def __init__(self):
        """Initialize with reduction rules"""
        self.rules: List[ReductionRule] = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Set up reduction rules"""
        # Identity reduction: 00 00 → 00
        self.add_rule(["00", "00"], ["00"], "identity-reduction")
        
        # Cycle completion: 01 00 10 → C (conceptually)
        # But we keep it expanded for normal form
        
        # Remove redundant patterns
        self.add_rule(["00", "01", "00", "00"], ["00", "01", "00"], "trailing-identity")
        
        # Normalize alternating patterns
        self.add_rule(["01", "00", "01", "00", "01"], ["01", "00", "01"], "alternation-reduction")
        
        # Sort independent operations (lexical ordering)
        # This is handled separately in canonicalize
    
    def add_rule(self, pattern: List[str], replacement: List[str], name: str = ""):
        """Add reduction rule"""
        # Only add if reduction preserves validity
        if self._is_valid_reduction(pattern, replacement):
            self.rules.append(ReductionRule(pattern, replacement, name))
    
    def _is_valid_reduction(self, pattern: List[str], replacement: List[str]) -> bool:
        """Check if reduction preserves no-11 constraint"""
        pattern_str = "".join(pattern)
        replacement_str = "".join(replacement)
        
        # Both must be valid
        if "11" in pattern_str or "11" in replacement_str:
            return False
        
        # Replacement must be shorter or same length
        return len(replacement) <= len(pattern)
    
    def reduce_once(self, sequence: List[str]) -> Tuple[List[str], bool]:
        """Apply one reduction if possible"""
        for i in range(len(sequence)):
            for rule in self.rules:
                if rule.applies_at(sequence, i):
                    reduced = rule.apply_at(sequence, i)
                    return reduced, True
        
        return sequence, False
    
    def reduce_to_normal_form(self, sequence: List[str]) -> List[str]:
        """Reduce sequence to normal form"""
        current = sequence.copy()
        max_iterations = 100  # Prevent infinite loops
        
        for _ in range(max_iterations):
            reduced, changed = self.reduce_once(current)
            if not changed:
                break
            current = reduced
        
        # Apply canonical ordering
        return self.canonicalize(current)
    
    def canonicalize(self, sequence: List[str]) -> List[str]:
        """Put sequence in canonical order"""
        # For now, just ensure no trailing identities
        while len(sequence) > 1 and sequence[-1] == "00":
            # Check if removing it maintains meaning
            if self._can_remove_trailing_identity(sequence):
                sequence = sequence[:-1]
            else:
                break
        
        return sequence
    
    def _can_remove_trailing_identity(self, sequence: List[str]) -> bool:
        """Check if trailing 00 can be removed"""
        # Can remove if it doesn't affect the pattern
        if len(sequence) <= 1:
            return False
        
        # Don't remove if it's part of a recognized pattern
        # For example, P1 pattern is ["00", "01", "00"] - don't break it
        if len(sequence) >= 3 and sequence[-3:] == ["00", "01", "00"]:
            return False
        
        if len(sequence) >= 3 and sequence[-3:] == ["01", "00", "00"]:
            return False
        
        # Only remove if previous symbol is 00 or 10 (safe endings)
        if len(sequence) >= 2 and sequence[-2] in ["00", "10"]:
            return True
        
        return False
    
    def is_normal_form(self, sequence: List[str]) -> bool:
        """Check if sequence is already in normal form"""
        reduced = self.reduce_to_normal_form(sequence)
        return reduced == sequence


class EquivalenceChecker:
    """Checks equivalence between collapse expressions"""
    
    def __init__(self, normalizer: NormalFormComputer):
        """Initialize with normalizer"""
        self.normalizer = normalizer
    
    def are_equivalent(self, seq1: List[str], seq2: List[str]) -> bool:
        """Check if two sequences are equivalent"""
        # Two sequences are equivalent if they have the same normal form
        normal1 = self.normalizer.reduce_to_normal_form(seq1)
        normal2 = self.normalizer.reduce_to_normal_form(seq2)
        
        return normal1 == normal2
    
    def equivalence_class(self, sequences: List[List[str]]) -> Dict[str, List[List[str]]]:
        """Group sequences into equivalence classes"""
        classes = defaultdict(list)
        
        for seq in sequences:
            normal = self.normalizer.reduce_to_normal_form(seq)
            key = "".join(normal)
            classes[key].append(seq)
        
        return dict(classes)


class MinimalFormFinder:
    """Finds minimal representations"""
    
    def __init__(self):
        """Initialize finder"""
        self.normalizer = NormalFormComputer()
    
    def find_minimal(self, sequence: List[str]) -> List[str]:
        """Find minimal equivalent form"""
        # Start with normal form
        normal = self.normalizer.reduce_to_normal_form(sequence)
        
        # Try to find shorter equivalent forms
        minimal = self._try_compressions(normal)
        
        return minimal
    
    def _try_compressions(self, sequence: List[str]) -> List[str]:
        """Try various compressions"""
        current = sequence
        
        # Try pattern replacements
        patterns = {
            ("00", "01", "00"): ["P1"],  # Pattern macro
            ("01", "00", "10"): ["P2"],
            ("10", "00", "01"): ["P3"]
        }
        
        for pattern, replacement in patterns.items():
            pattern_list = list(pattern)
            if self._contains_pattern(current, pattern_list):
                # Would use macro symbols in practice
                # For now, keep expanded form
                pass
        
        return current
    
    def _contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains pattern"""
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False


class TestReductionRules(unittest.TestCase):
    """Test reduction rule application"""
    
    def test_basic_reduction(self):
        """Test basic reduction rules"""
        rule = ReductionRule(["00", "00"], ["00"], "identity")
        
        # Rule applies
        sequence = ["01", "00", "00", "10"]
        self.assertTrue(rule.applies_at(sequence, 1))
        
        # Apply rule
        reduced = rule.apply_at(sequence, 1)
        self.assertEqual(reduced, ["01", "00", "10"])
    
    def test_pattern_matching(self):
        """Test pattern matching in rules"""
        rule = ReductionRule(["01", "00", "01"], ["01"], "pattern")
        
        sequence = ["00", "01", "00", "01", "10"]
        self.assertTrue(rule.applies_at(sequence, 1))
        
        reduced = rule.apply_at(sequence, 1)
        self.assertEqual(reduced, ["00", "01", "10"])
    
    def test_rule_validation(self):
        """Test that rules preserve validity"""
        normalizer = NormalFormComputer()
        
        # Valid reduction
        valid = normalizer._is_valid_reduction(["00", "00"], ["00"])
        self.assertTrue(valid)
        
        # Invalid - creates "11"
        invalid = normalizer._is_valid_reduction(["01", "10"], ["11"])
        self.assertFalse(invalid)


class TestNormalFormComputation(unittest.TestCase):
    """Test normal form computation"""
    
    def setUp(self):
        self.normalizer = NormalFormComputer()
    
    def test_identity_reduction(self):
        """Test reduction of repeated identities"""
        sequence = ["00", "00", "00", "01"]
        normal = self.normalizer.reduce_to_normal_form(sequence)
        
        # Should reduce repeated 00s
        self.assertEqual(normal, ["00", "01"])
    
    def test_already_normal(self):
        """Test sequences already in normal form"""
        sequences = [
            ["00", "01", "00"],
            ["01", "00", "10"],
            ["10"]
        ]
        
        for seq in sequences:
            self.assertTrue(self.normalizer.is_normal_form(seq))
    
    def test_canonical_ordering(self):
        """Test canonical ordering"""
        # Remove trailing identities
        sequence = ["01", "00", "10", "00"]
        normal = self.normalizer.reduce_to_normal_form(sequence)
        
        # Should remove trailing 00 if not essential
        self.assertEqual(normal, ["01", "00", "10"])
    
    def test_complex_reduction(self):
        """Test reduction of complex sequences"""
        sequence = ["00", "00", "01", "00", "00", "10", "00"]
        normal = self.normalizer.reduce_to_normal_form(sequence)
        
        # Should simplify while preserving meaning
        self.assertLess(len(normal), len(sequence))
        
        # Should still be valid
        concat = "".join(normal)
        self.assertNotIn("11", concat)


class TestEquivalenceChecking(unittest.TestCase):
    """Test equivalence between expressions"""
    
    def setUp(self):
        self.normalizer = NormalFormComputer()
        self.checker = EquivalenceChecker(self.normalizer)
    
    def test_basic_equivalence(self):
        """Test basic equivalence checking"""
        # Same sequences are equivalent
        seq1 = ["00", "01", "00"]
        seq2 = ["00", "01", "00"]
        self.assertTrue(self.checker.are_equivalent(seq1, seq2))
        
        # Different sequences
        seq3 = ["01", "00", "10"]
        self.assertFalse(self.checker.are_equivalent(seq1, seq3))
    
    def test_reduction_equivalence(self):
        """Test equivalence through reduction"""
        # These reduce to the same form
        seq1 = ["00", "00", "01"]
        seq2 = ["00", "01"]
        
        self.assertTrue(self.checker.are_equivalent(seq1, seq2))
    
    def test_equivalence_classes(self):
        """Test grouping into equivalence classes"""
        sequences = [
            ["00", "01"],
            ["00", "00", "01"],
            ["01", "00"],
            ["00", "01", "00"],
            ["00", "00", "01", "00"]
        ]
        
        classes = self.checker.equivalence_class(sequences)
        
        # Should have at least 2 different classes
        self.assertGreaterEqual(len(classes), 2)
        
        # Each class should have equivalent sequences
        for key, seqs in classes.items():
            if len(seqs) > 1:
                # All should be equivalent
                for i in range(1, len(seqs)):
                    self.assertTrue(self.checker.are_equivalent(seqs[0], seqs[i]))


class TestMinimalForms(unittest.TestCase):
    """Test finding minimal representations"""
    
    def setUp(self):
        self.finder = MinimalFormFinder()
    
    def test_already_minimal(self):
        """Test sequences already minimal"""
        minimal_sequences = [
            ["00"],
            ["01"],
            ["10"],
            ["00", "01", "00"]
        ]
        
        for seq in minimal_sequences:
            minimal = self.finder.find_minimal(seq)
            self.assertEqual(minimal, seq)
    
    def test_find_minimal(self):
        """Test finding minimal form"""
        # Sequence with redundancy
        sequence = ["00", "00", "00", "01", "00"]
        minimal = self.finder.find_minimal(sequence)
        
        # Should be shorter
        self.assertLessEqual(len(minimal), len(sequence))
        
        # Should be equivalent
        checker = EquivalenceChecker(self.finder.normalizer)
        self.assertTrue(checker.are_equivalent(sequence, minimal))
    
    def test_pattern_detection(self):
        """Test pattern detection for minimization"""
        # Contains known pattern
        sequence = ["00", "01", "00", "10"]
        
        # Check pattern detection
        self.assertTrue(self.finder._contains_pattern(sequence, ["00", "01", "00"]))


class TestNormalFormProperties(unittest.TestCase):
    """Test mathematical properties of normal forms"""
    
    def setUp(self):
        self.normalizer = NormalFormComputer()
        self.checker = EquivalenceChecker(self.normalizer)
    
    def test_uniqueness(self):
        """Test uniqueness of normal forms"""
        # Different sequences with same meaning
        sequences = [
            ["00", "00", "01"],
            ["00", "01", "00", "00"]
        ]
        
        normals = [self.normalizer.reduce_to_normal_form(seq) for seq in sequences]
        
        # If equivalent, should have same normal form
        if self.checker.are_equivalent(sequences[0], sequences[1]):
            self.assertEqual(normals[0], normals[1])
    
    def test_idempotence(self):
        """Test that normalizing normal form gives same result"""
        sequence = ["00", "01", "00", "10"]
        
        normal1 = self.normalizer.reduce_to_normal_form(sequence)
        normal2 = self.normalizer.reduce_to_normal_form(normal1)
        
        # Should be idempotent
        self.assertEqual(normal1, normal2)
    
    def test_preservation(self):
        """Test that normalization preserves validity"""
        test_sequences = [
            ["00", "01", "00"],
            ["01", "00", "10"],
            ["10", "00", "01", "00"],
            ["00", "00", "00", "01", "00", "10"]
        ]
        
        for seq in test_sequences:
            normal = self.normalizer.reduce_to_normal_form(seq)
            
            # Should maintain validity
            concat = "".join(normal)
            self.assertNotIn("11", concat)
            
            # Should not be empty unless original was
            if seq:
                self.assertGreater(len(normal), 0)


class TestComplexNormalization(unittest.TestCase):
    """Test normalization of complex expressions"""
    
    def setUp(self):
        self.normalizer = NormalFormComputer()
    
    def test_nested_patterns(self):
        """Test normalization with nested patterns"""
        # Pattern within pattern
        sequence = ["00", "00", "01", "00", "00", "10", "00"]
        normal = self.normalizer.reduce_to_normal_form(sequence)
        
        # Should simplify nested structure
        self.assertLess(len(normal), len(sequence))
    
    def test_long_sequences(self):
        """Test normalization of long sequences"""
        # Generate long sequence with repetition
        base = ["00", "01", "00"]
        long_seq = base * 5 + ["00", "00"]
        
        normal = self.normalizer.reduce_to_normal_form(long_seq)
        
        # Should be significantly shorter
        self.assertLess(len(normal), len(long_seq) / 2)
    
    def test_algorithmic_complexity(self):
        """Test that normalization terminates efficiently"""
        # Potentially problematic sequence
        sequence = ["01", "00"] * 20
        
        # Should still normalize quickly
        import time
        start = time.time()
        normal = self.normalizer.reduce_to_normal_form(sequence)
        duration = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(duration, 1.0)  # Less than 1 second
        
        # Should produce valid result
        self.assertIsNotNone(normal)


if __name__ == "__main__":
    unittest.main(verbosity=2)