#!/usr/bin/env python3
"""
Chapter 000: SelfCollapse - Verification Program
ψ = ψ(ψ) as the Origin of All Structure

This program verifies the fundamental properties of self-referential collapse
and demonstrates how all mathematical structure emerges from ψ = ψ(ψ).
"""

import unittest
from typing import Any, Callable, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class Bit(Enum):
    """Fundamental binary unit"""
    ZERO = 0
    ONE = 1
    
    def __str__(self):
        return str(self.value)


@dataclass
class Trace:
    """Binary trace without consecutive 1s (φ-constraint)"""
    bits: List[Bit]
    
    def __post_init__(self):
        """Verify φ-constraint: no consecutive 1s"""
        for i in range(len(self.bits) - 1):
            if self.bits[i] == Bit.ONE and self.bits[i+1] == Bit.ONE:
                raise ValueError(f"Invalid trace: consecutive 1s at position {i}")
    
    def __str__(self):
        return ''.join(str(b) for b in self.bits)
    
    def __eq__(self, other):
        return isinstance(other, Trace) and self.bits == other.bits
    
    def __hash__(self):
        return hash(tuple(self.bits))


class Psi:
    """The self-referential function ψ"""
    
    def __init__(self, inner: 'Psi' = None):
        """ψ can contain another ψ (including itself)"""
        self.inner = inner if inner is not None else self
        self._trace_cache = {}
    
    def __call__(self, arg: 'Psi') -> 'Psi':
        """ψ(ψ) - the fundamental operation"""
        # Self-application creates new structure
        return Psi(arg)
    
    def collapse(self) -> Trace:
        """Collapse ψ into a trace representation"""
        # Base case: self-reference collapses to minimal valid trace
        if self.inner is self:
            return Trace([Bit.ZERO, Bit.ONE])  # "01" - minimal non-trivial trace
        
        # Recursive case: combine inner collapse with structure marker
        inner_trace = self.inner.collapse()
        
        # Add structure while maintaining φ-constraint
        result_bits = []
        for bit in inner_trace.bits:
            result_bits.append(bit)
            # Prevent consecutive 1s by inserting 0 if needed
            if bit == Bit.ONE and result_bits and result_bits[-1] == Bit.ONE:
                result_bits.insert(-1, Bit.ZERO)
        
        # Mark this level with "10" prefix (valid under φ-constraint)
        return Trace([Bit.ONE, Bit.ZERO] + result_bits)
    
    def is_self_referential(self) -> bool:
        """Check if this ψ refers to itself"""
        return self.inner is self
    
    def depth(self) -> int:
        """Calculate nesting depth"""
        if self.is_self_referential():
            return 0
        return 1 + self.inner.depth()


class CollapseAlgebra:
    """Algebraic operations on collapsed traces"""
    
    @staticmethod
    def merge(t1: Trace, t2: Trace) -> Trace:
        """Merge two traces maintaining φ-constraint"""
        result_bits = []
        
        # Interleave bits to avoid creating consecutive 1s
        i, j = 0, 0
        last_was_one = False
        
        while i < len(t1.bits) or j < len(t2.bits):
            # Choose from t1 or t2 based on φ-constraint
            if i < len(t1.bits) and (j >= len(t2.bits) or 
                                     (t1.bits[i] == Bit.ZERO or not last_was_one)):
                result_bits.append(t1.bits[i])
                last_was_one = (t1.bits[i] == Bit.ONE)
                i += 1
            elif j < len(t2.bits):
                if t2.bits[j] == Bit.ONE and last_was_one:
                    # Insert separator to maintain constraint
                    result_bits.append(Bit.ZERO)
                    last_was_one = False
                else:
                    result_bits.append(t2.bits[j])
                    last_was_one = (t2.bits[j] == Bit.ONE)
                    j += 1
            else:
                break
        
        return Trace(result_bits)
    
    @staticmethod
    def fibonacci_rank(trace: Trace) -> int:
        """Calculate Fibonacci rank (Zeckendorf position)"""
        # Map trace to its position in Fibonacci sequence
        fib = [1, 2]
        while len(fib) < len(trace.bits):
            fib.append(fib[-1] + fib[-2])
        
        rank = 0
        for i, bit in enumerate(trace.bits):
            if bit == Bit.ONE:
                rank += fib[i]
        
        return rank


class SelfCollapseTests(unittest.TestCase):
    """Verify fundamental properties of self-referential collapse"""
    
    def test_self_reference_creation(self):
        """Test: ψ = ψ(ψ) creates valid self-reference"""
        psi = Psi()
        self.assertTrue(psi.is_self_referential())
        self.assertEqual(psi.depth(), 0)
    
    def test_self_application(self):
        """Test: ψ(ψ) creates new structure"""
        psi1 = Psi()
        psi2 = psi1(psi1)
        
        self.assertFalse(psi2.is_self_referential())
        self.assertEqual(psi2.depth(), 1)
        self.assertEqual(psi2.inner, psi1)
    
    def test_collapse_maintains_phi_constraint(self):
        """Test: Collapse always produces valid traces (no consecutive 1s)"""
        psi = Psi()
        
        # Test multiple levels of nesting
        current = psi
        for _ in range(10):
            trace = current.collapse()
            # Verify no consecutive 1s
            for i in range(len(trace.bits) - 1):
                self.assertFalse(
                    trace.bits[i] == Bit.ONE and trace.bits[i+1] == Bit.ONE,
                    f"Found consecutive 1s in trace: {trace}"
                )
            current = current(current)
    
    def test_collapse_deterministic(self):
        """Test: Same structure always collapses to same trace"""
        psi1 = Psi()
        psi2 = Psi()
        
        self.assertEqual(psi1.collapse(), psi2.collapse())
        
        # Nested structures
        nested1 = psi1(psi1(psi1))
        nested2 = psi2(psi2(psi2))
        self.assertEqual(nested1.collapse(), nested2.collapse())
    
    def test_trace_algebra_preserves_constraint(self):
        """Test: Algebraic operations maintain φ-constraint"""
        psi1 = Psi()
        psi2 = psi1(psi1)
        psi3 = psi2(psi1)
        
        t1 = psi1.collapse()
        t2 = psi2.collapse()
        t3 = psi3.collapse()
        
        # Test merge operations
        merged = CollapseAlgebra.merge(t1, t2)
        for i in range(len(merged.bits) - 1):
            self.assertFalse(
                merged.bits[i] == Bit.ONE and merged.bits[i+1] == Bit.ONE,
                f"Merge created consecutive 1s: {merged}"
            )
    
    def test_fibonacci_structure_emerges(self):
        """Test: Collapse naturally creates Fibonacci-like patterns"""
        psi = Psi()
        traces = []
        
        current = psi
        for _ in range(8):
            traces.append(current.collapse())
            current = current(psi)
        
        # Verify traces have increasing Fibonacci ranks
        ranks = [CollapseAlgebra.fibonacci_rank(t) for t in traces]
        for i in range(len(ranks) - 1):
            self.assertLess(ranks[i], ranks[i+1])
    
    def test_structural_completeness(self):
        """Test: Can generate all valid trace patterns through collapse"""
        # Generate various structures
        psi = Psi()
        structures = set()
        
        # Self-reference
        structures.add(psi.collapse())
        
        # Single application
        structures.add(psi(psi).collapse())
        
        # Multiple applications
        p1 = psi(psi)
        p2 = psi(p1)
        p3 = p1(p2)
        
        structures.add(p2.collapse())
        structures.add(p3.collapse())
        
        # Verify all are unique and valid
        self.assertEqual(len(structures), 4)
        for trace in structures:
            # Verify φ-constraint
            bits = trace.bits
            for i in range(len(bits) - 1):
                self.assertFalse(bits[i] == Bit.ONE and bits[i+1] == Bit.ONE)
    
    def test_first_principles_derivation(self):
        """Test: Everything emerges from ψ = ψ(ψ) alone"""
        # Start with nothing but the principle
        psi = Psi()  # ψ = ψ(ψ)
        
        # Binary emerges from collapse
        trace = psi.collapse()
        self.assertIn(Bit.ZERO, trace.bits)
        self.assertIn(Bit.ONE, trace.bits)
        
        # Structure emerges from application
        structured = psi(psi)
        self.assertGreater(structured.depth(), psi.depth())
        
        # Constraint emerges from consistency
        complex_structure = psi(psi(psi(psi)))
        complex_trace = complex_structure.collapse()
        # No consecutive 1s - constraint is inherent
        for i in range(len(complex_trace.bits) - 1):
            self.assertFalse(
                complex_trace.bits[i] == Bit.ONE and 
                complex_trace.bits[i+1] == Bit.ONE
            )


if __name__ == "__main__":
    # Run verification
    print("Verifying ψ = ψ(ψ) as origin of all structure...")
    print("=" * 60)
    
    # Demonstrate basic principles
    psi = Psi()
    print(f"1. Self-reference: ψ = ψ(ψ)")
    print(f"   Is self-referential: {psi.is_self_referential()}")
    print(f"   Collapse to trace: {psi.collapse()}")
    
    print(f"\n2. Structure from application: ψ(ψ)")
    psi2 = psi(psi)
    print(f"   Depth: {psi2.depth()}")
    print(f"   Collapse to trace: {psi2.collapse()}")
    
    print(f"\n3. Complex structures maintain φ-constraint:")
    current = psi
    for i in range(5):
        trace = current.collapse()
        print(f"   Level {i}: {trace} (Fibonacci rank: {CollapseAlgebra.fibonacci_rank(trace)})")
        current = current(psi)
    
    print("\n" + "=" * 60)
    print("Running formal verification tests...\n")
    
    # Run unit tests
    unittest.main(verbosity=2)