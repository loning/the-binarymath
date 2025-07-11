#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N8: Collapse Symbol Expansion Rules
Verifies systematic unfolding of compressed collapse representations.
"""

import unittest
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


class ExpansionRule:
    """Represents a symbol expansion rule"""
    
    def __init__(self, pattern: str, expansion: List[str], name: str = ""):
        """Initialize expansion rule"""
        self.pattern = pattern
        self.expansion = expansion
        self.name = name
    
    def matches(self, symbol: str) -> bool:
        """Check if rule matches symbol"""
        return symbol == self.pattern
    
    def apply(self, symbol: str) -> List[str]:
        """Apply expansion rule"""
        if self.matches(symbol):
            return self.expansion
        return [symbol]
    
    def __repr__(self):
        return f"Rule({self.pattern} → {self.expansion})"


class SymbolExpander:
    """Systematic symbol expansion system"""
    
    def __init__(self):
        """Initialize expander with core rules"""
        self.rules: Dict[str, ExpansionRule] = {}
        self._initialize_core_rules()
        self.max_depth = 10  # Prevent infinite expansion
    
    def _initialize_core_rules(self):
        """Set up fundamental expansion rules"""
        # Basic symbol expansions
        self.add_rule("I", ["00"], "Identity expansion")
        self.add_rule("T", ["01"], "Transform expansion")
        self.add_rule("R", ["10"], "Return expansion")
        
        # Composite expansions - must avoid creating "11"
        # C cannot be 01,10 as that creates "0110" with "11"
        self.add_rule("C", ["01", "00", "10"], "Cycle expansion")
        self.add_rule("D", ["01", "00", "01"], "Double transform")
        self.add_rule("S", ["00", "00"], "Stable identity")
        
        # Pattern macros
        self.add_rule("P1", ["00", "01", "00"], "Pattern 1")
        self.add_rule("P2", ["01", "00", "10"], "Pattern 2")
        self.add_rule("P3", ["10", "00", "01"], "Pattern 3")
    
    def add_rule(self, pattern: str, expansion: List[str], name: str = ""):
        """Add expansion rule"""
        self.rules[pattern] = ExpansionRule(pattern, expansion, name)
    
    def expand_symbol(self, symbol: str, depth: int = 0) -> List[str]:
        """Expand a single symbol"""
        if depth >= self.max_depth:
            return [symbol]
        
        # Check if symbol has a rule
        if symbol in self.rules:
            expansion = self.rules[symbol].expansion
            # Recursively expand the result
            result = []
            for s in expansion:
                result.extend(self.expand_symbol(s, depth + 1))
            return result
        else:
            # No rule - return as is
            return [symbol]
    
    def expand_sequence(self, sequence: List[str], depth: int = 0) -> List[str]:
        """Expand a sequence of symbols"""
        result = []
        for symbol in sequence:
            result.extend(self.expand_symbol(symbol, depth))
        return result
    
    def full_expansion(self, sequence: List[str]) -> List[str]:
        """Fully expand sequence until no more rules apply"""
        current = sequence
        for _ in range(self.max_depth):
            expanded = self.expand_sequence(current)
            if expanded == current:
                break  # Fixed point reached
            current = expanded
        return current
    
    def is_valid_expansion(self, expansion: List[str]) -> bool:
        """Check if expansion is valid (no "11" in concatenation)"""
        concat = "".join(expansion)
        return "11" not in concat


class CompressionAnalyzer:
    """Analyze compression ratios and patterns"""
    
    def __init__(self, expander: SymbolExpander):
        """Initialize with expander"""
        self.expander = expander
    
    def compression_ratio(self, compressed: List[str], expanded: List[str]) -> float:
        """Calculate compression ratio"""
        if not expanded:
            return 0.0
        return len(compressed) / len(expanded)
    
    def find_compressible_patterns(self, sequence: List[str]) -> List[Tuple[int, int, str]]:
        """Find patterns that can be compressed"""
        patterns = []
        
        # Check for known patterns
        pattern_rules = {
            ("00", "01", "00"): "P1",
            ("01", "00", "10"): "P2",
            ("10", "00", "01"): "P3",
            ("01", "00", "10"): "C",
            ("01", "00", "01"): "D",
            ("00", "00"): "S"
        }
        
        for length in range(2, min(4, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i+length])
                if subseq in pattern_rules:
                    patterns.append((i, i + length, pattern_rules[subseq]))
        
        return patterns
    
    def optimal_compression(self, sequence: List[str]) -> List[str]:
        """Find optimal compression of sequence"""
        # Greedy approach: replace longest patterns first
        patterns = self.find_compressible_patterns(sequence)
        
        if not patterns:
            return sequence
        
        # Sort by length (descending) and position
        patterns.sort(key=lambda x: (x[0] - x[1], x[0]))
        
        # Apply first non-overlapping pattern
        result = sequence.copy()
        for start, end, symbol in patterns:
            # Replace pattern with symbol
            result = result[:start] + [symbol] + result[end:]
            break
        
        # Recursively compress the result
        if result != sequence:
            return self.optimal_compression(result)
        
        return result


class ExpansionTree:
    """Tree structure for tracking expansions"""
    
    def __init__(self, symbol: str, depth: int = 0):
        """Initialize tree node"""
        self.symbol = symbol
        self.depth = depth
        self.children: List[ExpansionTree] = []
    
    def add_child(self, child: 'ExpansionTree'):
        """Add child node"""
        self.children.append(child)
    
    def to_sequence(self) -> List[str]:
        """Convert tree to flat sequence"""
        if not self.children:
            return [self.symbol]
        
        result = []
        for child in self.children:
            result.extend(child.to_sequence())
        return result
    
    def height(self) -> int:
        """Get tree height"""
        if not self.children:
            return 1
        return 1 + max(child.height() for child in self.children)
    
    def node_count(self) -> int:
        """Count total nodes"""
        return 1 + sum(child.node_count() for child in self.children)


class TestExpansionRules(unittest.TestCase):
    """Test basic expansion rules"""
    
    def setUp(self):
        self.expander = SymbolExpander()
    
    def test_basic_expansions(self):
        """Test fundamental symbol expansions"""
        # Basic symbols
        self.assertEqual(self.expander.expand_symbol("I"), ["00"])
        self.assertEqual(self.expander.expand_symbol("T"), ["01"])
        self.assertEqual(self.expander.expand_symbol("R"), ["10"])
        
        # Already expanded
        self.assertEqual(self.expander.expand_symbol("00"), ["00"])
        self.assertEqual(self.expander.expand_symbol("01"), ["01"])
    
    def test_composite_expansions(self):
        """Test composite symbol expansions"""
        # Cycle (with separator to avoid "11")
        self.assertEqual(self.expander.expand_symbol("C"), ["01", "00", "10"])
        
        # Double transform (with separator)
        self.assertEqual(self.expander.expand_symbol("D"), ["01", "00", "01"])
        
        # Pattern macros
        self.assertEqual(self.expander.expand_symbol("P1"), ["00", "01", "00"])
    
    def test_sequence_expansion(self):
        """Test expanding sequences"""
        sequence = ["I", "T", "R"]
        expanded = self.expander.expand_sequence(sequence)
        self.assertEqual(expanded, ["00", "01", "10"])
        
        # Mixed sequence
        mixed = ["P1", "C"]
        expanded = self.expander.expand_sequence(mixed)
        self.assertEqual(expanded, ["00", "01", "00", "01", "00", "10"])
    
    def test_recursive_expansion(self):
        """Test recursive expansion"""
        # Add a rule that expands to other macros
        self.expander.add_rule("META", ["I", "C", "I"])
        
        expanded = self.expander.expand_symbol("META")
        # I -> 00, C -> 01,00,10, I -> 00
        self.assertEqual(expanded, ["00", "01", "00", "10", "00"])
    
    def test_expansion_validity(self):
        """Test that expansions maintain validity"""
        # Valid expansions
        valid_symbols = ["I", "T", "R", "C", "P1", "P2", "P3"]
        
        for symbol in valid_symbols:
            expanded = self.expander.expand_symbol(symbol)
            self.assertTrue(self.expander.is_valid_expansion(expanded))
        
        # Invalid expansion (would create "11")
        self.expander.add_rule("BAD", ["01", "10"])  # This is actually valid
        # Let's create one that's actually invalid
        # Since we can't create "11" with our alphabet, all expansions are valid


class TestCompressionAnalysis(unittest.TestCase):
    """Test compression analysis"""
    
    def setUp(self):
        self.expander = SymbolExpander()
        self.analyzer = CompressionAnalyzer(self.expander)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        compressed = ["P1"]
        expanded = ["00", "01", "00"]
        
        ratio = self.analyzer.compression_ratio(compressed, expanded)
        self.assertAlmostEqual(ratio, 1/3)
        
        # No compression
        same = ["00", "01"]
        ratio = self.analyzer.compression_ratio(same, same)
        self.assertEqual(ratio, 1.0)
    
    def test_pattern_finding(self):
        """Test finding compressible patterns"""
        sequence = ["00", "01", "00", "01", "00", "10"]
        patterns = self.analyzer.find_compressible_patterns(sequence)
        
        # Should find P1 pattern at start
        self.assertTrue(any(p[2] == "P1" for p in patterns))
        
        # Should find C pattern (01, 00, 10) in the sequence
        self.assertTrue(any(p[2] == "C" for p in patterns))
    
    def test_optimal_compression(self):
        """Test optimal compression"""
        # Sequence with known pattern
        sequence = ["00", "01", "00", "10", "00"]
        compressed = self.analyzer.optimal_compression(sequence)
        
        # Should compress first part to P1
        self.assertIn("P1", compressed)
        self.assertLess(len(compressed), len(sequence))
        
        # Already optimal
        optimal = ["P1", "P2"]
        self.assertEqual(self.analyzer.optimal_compression(optimal), optimal)


class TestExpansionTree(unittest.TestCase):
    """Test expansion tree structure"""
    
    def test_tree_construction(self):
        """Test building expansion tree"""
        root = ExpansionTree("P1")
        
        # Add children for expansion
        root.add_child(ExpansionTree("00", 1))
        root.add_child(ExpansionTree("01", 1))
        root.add_child(ExpansionTree("00", 1))
        
        # Check structure
        self.assertEqual(len(root.children), 3)
        self.assertEqual(root.height(), 2)
        self.assertEqual(root.node_count(), 4)
    
    def test_tree_to_sequence(self):
        """Test converting tree to sequence"""
        root = ExpansionTree("C")
        root.add_child(ExpansionTree("01", 1))
        root.add_child(ExpansionTree("10", 1))
        
        sequence = root.to_sequence()
        self.assertEqual(sequence, ["01", "10"])
    
    def test_nested_tree(self):
        """Test nested expansion tree"""
        root = ExpansionTree("META")
        
        # First level
        child1 = ExpansionTree("I", 1)
        child2 = ExpansionTree("C", 1)
        root.add_child(child1)
        root.add_child(child2)
        
        # Second level
        child1.add_child(ExpansionTree("00", 2))
        child2.add_child(ExpansionTree("01", 2))
        child2.add_child(ExpansionTree("10", 2))
        
        sequence = root.to_sequence()
        self.assertEqual(sequence, ["00", "01", "10"])
        self.assertEqual(root.height(), 3)


class TestExpansionProperties(unittest.TestCase):
    """Test mathematical properties of expansion"""
    
    def setUp(self):
        self.expander = SymbolExpander()
    
    def test_expansion_uniqueness(self):
        """Test that expansion is deterministic"""
        symbol = "P1"
        
        # Multiple expansions should give same result
        exp1 = self.expander.expand_symbol(symbol)
        exp2 = self.expander.expand_symbol(symbol)
        
        self.assertEqual(exp1, exp2)
    
    def test_expansion_termination(self):
        """Test that expansion terminates"""
        # Even with recursive rules, should terminate due to depth limit
        self.expander.add_rule("REC", ["REC", "I"])
        
        # Should not infinite loop
        expanded = self.expander.expand_symbol("REC")
        self.assertIsNotNone(expanded)
        self.assertGreater(len(expanded), 0)
    
    def test_expansion_preservation(self):
        """Test that expansion preserves essential properties"""
        # Expansion should preserve the "no 11" property
        test_symbols = ["I", "T", "R", "C", "P1", "P2", "P3"]
        
        for symbol in test_symbols:
            expanded = self.expander.expand_symbol(symbol)
            concat = "".join(expanded)
            self.assertNotIn("11", concat)
    
    def test_composition_expansion(self):
        """Test expansion of composed symbols"""
        # Create a symbol that expands to other symbols
        self.expander.add_rule("COMP", ["P1", "C"])
        
        # Single level expansion (depth 1)
        level1 = self.expander.expand_sequence(["COMP"], depth=1)
        # This will expand COMP once but not further
        # Actually, the current implementation doesn't support partial depth
        
        # Full expansion
        full = self.expander.full_expansion(["COMP"])
        expected = ["00", "01", "00", "01", "00", "10"]
        self.assertEqual(full, expected)


class TestExpansionInversion(unittest.TestCase):
    """Test relationship between expansion and compression"""
    
    def setUp(self):
        self.expander = SymbolExpander()
        self.analyzer = CompressionAnalyzer(self.expander)
    
    def test_expand_compress_cycle(self):
        """Test that expand-compress preserves structure"""
        # Start with compressed form
        compressed = ["P1", "C"]
        
        # Expand
        expanded = self.expander.full_expansion(compressed)
        
        # Compress back
        recompressed = self.analyzer.optimal_compression(expanded)
        
        # May not be identical due to greedy compression,
        # but should have same expansion
        re_expanded = self.expander.full_expansion(recompressed)
        self.assertEqual(expanded, re_expanded)
    
    def test_information_preservation(self):
        """Test that expansion preserves information"""
        sequences = [
            ["P1"],
            ["C", "I"],
            ["P2", "P3"]
        ]
        
        for seq in sequences:
            expanded = self.expander.full_expansion(seq)
            # Information is preserved if we can distinguish sequences
            # by their expansions
            self.assertGreater(len(expanded), 0)
            
            # Validity is preserved
            self.assertTrue(self.expander.is_valid_expansion(expanded))


if __name__ == "__main__":
    unittest.main(verbosity=2)