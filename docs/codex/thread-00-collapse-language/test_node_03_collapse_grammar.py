#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N3: Collapse Grammar Rules
Verifies the grammatical rules that govern valid collapse sequences.
"""

import unittest
from typing import List, Set, Tuple, Optional


class GrammarValidator:
    """Validates sequences according to collapse grammar rules"""
    
    @staticmethod
    def is_valid_sequence(symbols: List[str]) -> bool:
        """Check if a sequence of symbols is grammatically valid"""
        if not symbols:
            return True
        
        # Check each symbol is valid
        valid_symbols = {"00", "01", "10"}
        if not all(s in valid_symbols for s in symbols):
            return False
        
        # Check for forbidden adjacencies
        concat = "".join(symbols)
        return "11" not in concat
    
    @staticmethod
    def get_forbidden_adjacencies() -> Set[Tuple[str, str]]:
        """Return the set of forbidden symbol adjacencies"""
        # Only 01 → 10 is forbidden (creates "0110" with "11")
        return {("01", "10")}
    
    @staticmethod
    def can_follow(s1: str, s2: str) -> bool:
        """Check if s2 can follow s1"""
        concat = s1 + s2
        return "11" not in concat


class ContextFreeGrammar:
    """Implements the context-free grammar for collapse language"""
    
    def __init__(self):
        # Grammar rules from the corrected version
        self.rules = {
            "S": ["ε", "A", "B", "C"],
            "A": ["00A", "00B", "00C", "00", "ε"],
            "B": ["01A", "01B", "01", "ε"],
            "C": ["10A", "10B", "10C", "10", "ε"]
        }
    
    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is terminal"""
        return symbol in ["00", "01", "10", "ε"]
    
    def derive(self, start: str, max_depth: int = 5) -> List[str]:
        """Derive strings from a non-terminal up to max depth"""
        if max_depth == 0:
            return [""]  # Terminate derivation
            
        if self.is_terminal(start):
            return [start] if start != "ε" else [""]
        
        if start not in self.rules:
            return []
        
        results = set()  # Use set to avoid duplicates
        for production in self.rules[start]:
            if production == "ε":
                results.add("")
            elif self.is_terminal(production):
                results.add(production)
            else:
                # Handle productions like "00A", "01B", etc.
                if len(production) >= 3 and production[:2] in ["00", "01", "10"]:
                    prefix = production[:2]
                    rest = production[2:]
                    for suffix in self.derive(rest, max_depth - 1):
                        results.add(prefix + suffix)
                elif len(production) == 2 and production in ["00", "01", "10"]:
                    # Terminal symbol
                    results.add(production)
                else:
                    # Single non-terminal
                    results.update(self.derive(production, max_depth - 1))
        
        return list(results)


class TestGrammarRules(unittest.TestCase):
    """Test the grammatical rules of collapse language"""
    
    def test_valid_adjacencies(self):
        """Test all valid symbol adjacencies"""
        validator = GrammarValidator()
        
        valid_adjacencies = [
            ("00", "00"), ("00", "01"), ("00", "10"),
            ("01", "00"), ("01", "01"),
            ("10", "00"), ("10", "01"), ("10", "10")
        ]
        
        for s1, s2 in valid_adjacencies:
            self.assertTrue(validator.can_follow(s1, s2),
                          f"{s1} → {s2} should be valid")
    
    def test_forbidden_adjacency(self):
        """Test the forbidden adjacency"""
        validator = GrammarValidator()
        
        # Only 01 → 10 is forbidden
        self.assertFalse(validator.can_follow("01", "10"))
        
        # Verify it creates "11"
        concat = "01" + "10"
        self.assertEqual(concat, "0110")
        self.assertIn("11", concat)
    
    def test_sequence_validation(self):
        """Test validation of complete sequences"""
        validator = GrammarValidator()
        
        # Valid sequences
        valid_sequences = [
            ["00", "01", "00"],
            ["10", "10", "10"],
            ["01", "01", "00"],
            ["00", "00", "00"]
        ]
        
        for seq in valid_sequences:
            self.assertTrue(validator.is_valid_sequence(seq),
                          f"Sequence {seq} should be valid")
        
        # Invalid sequences
        invalid_sequences = [
            ["01", "10"],  # Creates "0110" with "11"
            ["00", "01", "10", "00"]  # Contains "01", "10" adjacency
        ]
        
        for seq in invalid_sequences:
            self.assertFalse(validator.is_valid_sequence(seq),
                           f"Sequence {seq} should be invalid")
    
    def test_constraint_consistency(self):
        """Test that constraints are consistent with the no-11 rule"""
        validator = GrammarValidator()
        symbols = ["00", "01", "10"]
        
        for s1 in symbols:
            for s2 in symbols:
                can_follow = validator.can_follow(s1, s2)
                has_11 = "11" in (s1 + s2)
                
                # can_follow should be True iff no "11" in concatenation
                self.assertEqual(can_follow, not has_11,
                               f"Inconsistency for {s1} → {s2}")


class TestContextFreeGrammar(unittest.TestCase):
    """Test the context-free grammar formalization"""
    
    def test_grammar_productions(self):
        """Test that grammar produces valid strings"""
        cfg = ContextFreeGrammar()
        
        # Generate some strings from S
        strings = cfg.derive("S", max_depth=3)
        
        # Remove empty string for validation
        non_empty = [s for s in strings if s]
        
        # All should be valid
        validator = GrammarValidator()
        for s in non_empty:
            # Split into symbols (pairs of characters)
            symbols = [s[i:i+2] for i in range(0, len(s), 2)]
            self.assertTrue(validator.is_valid_sequence(symbols),
                          f"Grammar produced invalid string: {s}")
    
    def test_no_forbidden_patterns(self):
        """Test that grammar never produces forbidden patterns"""
        cfg = ContextFreeGrammar()
        
        # Generate strings from each non-terminal
        for start in ["A", "B", "C"]:
            strings = cfg.derive(start, max_depth=4)
            
            for s in strings:
                # Check no "11" appears
                self.assertNotIn("11", s,
                               f"Grammar produced string with '11': {s}")
    
    def test_language_characterization(self):
        """Test that the language is correctly characterized"""
        cfg = ContextFreeGrammar()
        
        # Generate a sample of the language
        strings = set()
        for start in ["S", "A", "B", "C"]:
            strings.update(cfg.derive(start, max_depth=3))
        
        # Check properties
        for s in strings:
            if s:  # Non-empty
                # No "11" substring
                self.assertNotIn("11", s)
                
                # All characters are 0 or 1
                self.assertTrue(all(c in "01" for c in s))
                
                # Length is even (symbols are pairs)
                self.assertEqual(len(s) % 2, 0)


class TestProductionRules(unittest.TestCase):
    """Test specific production rules"""
    
    def test_core_productions(self):
        """Test the core production rules"""
        # According to corrected grammar:
        # A can produce sequences starting with 00
        # B can produce sequences starting with 01
        # C can produce sequences starting with 10
        
        cfg = ContextFreeGrammar()
        
        # Test A productions
        a_strings = cfg.derive("A", max_depth=2)
        for s in a_strings:
            if s and len(s) >= 2:
                self.assertEqual(s[:2], "00",
                               f"A production should start with 00: {s}")
        
        # Test B productions
        b_strings = cfg.derive("B", max_depth=2)
        for s in b_strings:
            if s and len(s) >= 2:
                self.assertEqual(s[:2], "01",
                               f"B production should start with 01: {s}")
        
        # Test C productions
        c_strings = cfg.derive("C", max_depth=2)
        for s in c_strings:
            if s and len(s) >= 2:
                self.assertEqual(s[:2], "10",
                               f"C production should start with 10: {s}")
    
    def test_no_invalid_cycles(self):
        """Test that production rules don't create invalid cycles"""
        cfg = ContextFreeGrammar()
        
        # B → 01... cannot lead directly to 10...
        # This is ensured by the grammar structure
        
        b_strings = cfg.derive("B", max_depth=3)
        
        for s in b_strings:
            # Check that 01 is never directly followed by 10
            for i in range(0, len(s) - 3, 2):
                if s[i:i+2] == "01":
                    self.assertNotEqual(s[i+2:i+4], "10",
                                      f"Found 01→10 in: {s}")


class TestAlgebraicProperties(unittest.TestCase):
    """Test algebraic properties of the grammar"""
    
    def test_monoid_structure(self):
        """Test that valid words form a monoid under concatenation"""
        validator = GrammarValidator()
        
        # Test cases
        words = [
            ["00"],
            ["01", "00"],
            ["10", "00"],
            []  # Empty word
        ]
        
        # Closure: concatenating valid words should produce valid words
        # (if no forbidden adjacency is created)
        for w1 in words:
            for w2 in words:
                combined = w1 + w2
                
                # Check if combination is valid
                if validator.is_valid_sequence(w1) and validator.is_valid_sequence(w2):
                    # Need to check boundary
                    if w1 and w2:
                        boundary_valid = validator.can_follow(w1[-1], w2[0])
                        if boundary_valid:
                            self.assertTrue(validator.is_valid_sequence(combined))
        
        # Identity: empty word
        empty = []
        word = ["00", "01"]
        self.assertEqual(empty + word, word)
        self.assertEqual(word + empty, word)
        
        # Associativity of concatenation
        w1, w2, w3 = ["00"], ["01"], ["00"]
        self.assertEqual((w1 + w2) + w3, w1 + (w2 + w3))
    
    def test_pumping_property(self):
        """Test modified pumping property"""
        # Certain patterns can be pumped while maintaining validity
        
        # Pattern 1: (00)* can be pumped
        base = ["00"]
        pumped = base * 5  # Pump 5 times
        validator = GrammarValidator()
        self.assertTrue(validator.is_valid_sequence(pumped))
        
        # Pattern 2: Complete cycles can be pumped
        # But we must be careful about boundaries
        cycle = ["10", "00", "01", "00"]  # Valid cycle
        self.assertTrue(validator.is_valid_sequence(cycle))
        
        # Pumping twice should still be valid
        double_cycle = cycle + cycle
        self.assertTrue(validator.is_valid_sequence(double_cycle))


if __name__ == "__main__":
    unittest.main(verbosity=2)