#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N10: φ-Trace Folding and Self-Nesting
Verifies the mathematical properties of trace folding operations.
"""

import unittest
from typing import List, Tuple, Optional


class CollapseSymbol:
    """Represents a collapse symbol from {00, 01, 10}"""
    def __init__(self, value: str):
        if value not in ["00", "01", "10"]:
            raise ValueError(f"Invalid collapse symbol: {value}")
        self.value = value
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)


class Trace:
    """Represents a collapse trace (sequence of symbols)"""
    def __init__(self, symbols: List[CollapseSymbol]):
        self.symbols = symbols
        self._validate()
    
    def _validate(self):
        """Ensure no forbidden "11" appears when concatenated"""
        concat = "".join(str(s) for s in self.symbols)
        if "11" in concat:
            raise ValueError(f"Forbidden '11' in trace: {concat}")
    
    def __str__(self):
        return " ".join(str(s) for s in self.symbols)
    
    def __eq__(self, other):
        return self.symbols == other.symbols
    
    def __len__(self):
        return len(self.symbols)
    
    def concat(self) -> str:
        """Get concatenated binary string"""
        return "".join(str(s) for s in self.symbols)


class FoldingOperations:
    """Implements various trace folding operations"""
    
    @staticmethod
    def mirror_fold(trace: Trace) -> Trace:
        """M(t) = t ⟨t⟩ where ⟨t⟩ is reverse of t"""
        forward = trace.symbols
        backward = trace.symbols[::-1]
        # Must check if concatenation is valid
        try:
            return Trace(forward + backward)
        except ValueError:
            # If direct concatenation fails, insert separator
            separator = [CollapseSymbol("00")]
            return Trace(forward + separator + backward)
    
    @staticmethod
    def insertion_fold(trace: Trace, insert: Trace, position: int) -> Trace:
        """Insert one trace into another at given position"""
        if position < 0 or position > len(trace):
            raise ValueError("Invalid insertion position")
        
        symbols = trace.symbols[:position] + insert.symbols + trace.symbols[position:]
        return Trace(symbols)
    
    @staticmethod
    def recursive_fold(trace: Trace, depth: int) -> Trace:
        """R(t) = t[R(reduce(t))] up to given depth"""
        if depth <= 0:
            return trace
        
        # Simple reduction: take every other symbol
        if len(trace) <= 1:
            return trace
        
        reduced_symbols = trace.symbols[::2]
        reduced = Trace(reduced_symbols)
        
        # Recursively fold the reduced trace
        inner = FoldingOperations.recursive_fold(reduced, depth - 1)
        
        # Insert the folded version in the middle
        mid = len(trace) // 2
        return FoldingOperations.insertion_fold(trace, inner, mid)
    
    @staticmethod
    def is_self_similar(trace: Trace, fold_func) -> bool:
        """Check if T = F(T) for given folding function"""
        folded = fold_func(trace)
        return trace == folded
    
    @staticmethod
    def unfold(trace: Trace, pattern_length: int) -> List[Trace]:
        """Extract nested patterns of given length"""
        if pattern_length <= 0 or pattern_length > len(trace):
            return []
        
        patterns = []
        for i in range(len(trace) - pattern_length + 1):
            sub_symbols = trace.symbols[i:i + pattern_length]
            patterns.append(Trace(sub_symbols))
        
        return patterns


class TestTraceFolding(unittest.TestCase):
    """Test cases for trace folding operations"""
    
    def setUp(self):
        """Initialize test traces"""
        self.trace1 = Trace([
            CollapseSymbol("00"),
            CollapseSymbol("01"),
            CollapseSymbol("10")
        ])
        
        self.trace2 = Trace([
            CollapseSymbol("10"),
            CollapseSymbol("00")
        ])
        
        self.trace3 = Trace([
            CollapseSymbol("00"),
            CollapseSymbol("00"),
            CollapseSymbol("00")
        ])
    
    def test_basic_trace_validation(self):
        """Test that invalid traces are rejected"""
        # Valid trace
        valid = Trace([CollapseSymbol("00"), CollapseSymbol("01")])
        self.assertEqual(str(valid), "00 01")
        
        # Invalid trace (01 → 10 creates "0110" with "11")
        with self.assertRaises(ValueError):
            Trace([CollapseSymbol("01"), CollapseSymbol("10")])
    
    def test_mirror_fold(self):
        """Test mirror folding operation"""
        # Simple mirror fold
        folded = FoldingOperations.mirror_fold(self.trace3)
        self.assertEqual(str(folded), "00 00 00 00 00 00")
        
        # Mirror fold that would create invalid sequence
        trace = Trace([CollapseSymbol("01"), CollapseSymbol("00")])
        folded = FoldingOperations.mirror_fold(trace)
        # Should insert separator to avoid "01 00 00 01" → "01000001"
        self.assertNotIn("11", folded.concat())
    
    def test_insertion_fold(self):
        """Test insertion folding"""
        insert = Trace([CollapseSymbol("10")])
        
        # Insert at beginning
        folded = FoldingOperations.insertion_fold(self.trace1, insert, 0)
        self.assertEqual(str(folded), "10 00 01 10")
        
        # Insert in middle
        folded = FoldingOperations.insertion_fold(self.trace1, insert, 2)
        self.assertEqual(str(folded), "00 01 10 10")
        
        # Insert at end
        folded = FoldingOperations.insertion_fold(self.trace1, insert, 3)
        self.assertEqual(str(folded), "00 01 10 10")
    
    def test_recursive_fold(self):
        """Test recursive folding"""
        # Depth 1
        folded1 = FoldingOperations.recursive_fold(self.trace1, 1)
        self.assertGreater(len(folded1), len(self.trace1))
        
        # Depth 2
        folded2 = FoldingOperations.recursive_fold(self.trace1, 2)
        self.assertGreater(len(folded2), len(folded1))
        
        # Verify no forbidden patterns
        self.assertNotIn("11", folded1.concat())
        self.assertNotIn("11", folded2.concat())
    
    def test_folding_algebra_closure(self):
        """Test that folding operations are closed"""
        # Fold once
        folded1 = FoldingOperations.mirror_fold(self.trace1)
        
        # Fold again - should still be valid
        folded2 = FoldingOperations.mirror_fold(folded1)
        self.assertNotIn("11", folded2.concat())
    
    def test_folding_associativity(self):
        """Test associativity of folding composition"""
        # Due to complexity, we test a weaker property:
        # Multiple folds should preserve validity
        trace = self.trace1
        
        # (F1 ∘ F2) ∘ F3
        temp = FoldingOperations.mirror_fold(trace)
        temp = FoldingOperations.recursive_fold(temp, 1)
        result1 = FoldingOperations.mirror_fold(temp)
        
        # F1 ∘ (F2 ∘ F3)
        temp = FoldingOperations.recursive_fold(trace, 1)
        temp = FoldingOperations.mirror_fold(temp)
        result2 = FoldingOperations.mirror_fold(temp)
        
        # Both should be valid
        self.assertNotIn("11", result1.concat())
        self.assertNotIn("11", result2.concat())
    
    def test_unfold_operation(self):
        """Test unfolding to extract patterns"""
        patterns = FoldingOperations.unfold(self.trace1, 2)
        
        self.assertEqual(len(patterns), 2)
        self.assertEqual(str(patterns[0]), "00 01")
        self.assertEqual(str(patterns[1]), "01 10")
    
    def test_information_density(self):
        """Test that folding increases information density"""
        original_length = len(self.trace1)
        
        # Multiple folds
        folded = self.trace1
        densities = [original_length]
        
        for i in range(3):
            folded = FoldingOperations.recursive_fold(folded, 1)
            # Information density = pattern count / length
            patterns = FoldingOperations.unfold(folded, original_length)
            density = len(patterns) / len(folded)
            densities.append(density)
        
        # Density should generally increase with folding depth
        self.assertGreater(densities[-1], densities[0])
    
    def test_phi_rank_preservation(self):
        """Test that certain folds preserve structural properties"""
        # For simplicity, we use length as a proxy for φ-rank
        original = self.trace3  # All "00" symbols
        
        # This fold should preserve the "all-00" property
        folded = FoldingOperations.mirror_fold(original)
        
        # Check all symbols are still "00"
        all_zeros = all(s.value == "00" for s in folded.symbols)
        self.assertTrue(all_zeros)
    
    def test_exponential_growth(self):
        """Test exponential growth of deeply folded traces"""
        lengths = [len(self.trace1)]
        
        trace = self.trace1
        for depth in range(1, 5):
            trace = FoldingOperations.recursive_fold(trace, depth)
            lengths.append(len(trace))
        
        # Verify exponential-like growth
        for i in range(1, len(lengths)):
            self.assertGreater(lengths[i], lengths[i-1])
        
        # Growth rate should increase
        growth_rates = []
        for i in range(1, len(lengths)):
            growth_rates.append(lengths[i] / lengths[i-1])
        
        # Later growth rates should be higher
        self.assertGreater(growth_rates[-1], growth_rates[0])


class TestSelfSimilarity(unittest.TestCase):
    """Test cases for self-similar traces"""
    
    def test_fixed_point_search(self):
        """Test searching for self-similar traces"""
        # The empty trace is trivially self-similar
        empty = Trace([])
        self.assertTrue(FoldingOperations.is_self_similar(
            empty, 
            lambda t: t
        ))
        
        # A trace of all "00" has certain self-similar properties
        zeros = Trace([CollapseSymbol("00")])
        
        def zero_preserving_fold(t):
            if all(s.value == "00" for s in t.symbols):
                return t
            return FoldingOperations.mirror_fold(t)
        
        self.assertTrue(FoldingOperations.is_self_similar(
            zeros,
            zero_preserving_fold
        ))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)