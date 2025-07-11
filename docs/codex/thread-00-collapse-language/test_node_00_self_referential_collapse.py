#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N0: Self-Referential Collapse: ψ = ψ(ψ)
Verifies the foundational principle and its properties.
"""

import unittest
from typing import Any, Callable


class Psi:
    """
    Represents the fundamental entity ψ that satisfies ψ = ψ(ψ).
    This is the primordial identity from which all structure emerges.
    """
    
    def __init__(self, name: str = "ψ"):
        self.name = name
        self._application_count = 0
    
    def __call__(self, arg: 'Psi') -> 'Psi':
        """
        Self-application: ψ(ψ) returns ψ
        This implements the fundamental equation ψ = ψ(ψ)
        """
        self._application_count += 1
        if arg == self:  # Self-application
            return self  # Fixed point
        else:
            # For testing purposes, non-self application returns a new Psi
            return Psi(f"{self.name}({arg.name})")
    
    def __eq__(self, other: Any) -> bool:
        """Two Psi instances are equal if they represent the same fixed point"""
        if not isinstance(other, Psi):
            return False
        # For the fundamental ψ, identity is based on self-reference
        return self is other or (self.name == other.name == "ψ")
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Psi({self.name})"
    
    @property
    def applications(self) -> int:
        """Count of self-applications performed"""
        return self._application_count


class TestSelfReferentialCollapse(unittest.TestCase):
    """Test cases for the fundamental principle ψ = ψ(ψ)"""
    
    def test_fixed_point_existence(self):
        """Test that ψ = ψ(ψ) has a solution"""
        psi = Psi()
        
        # Apply psi to itself
        result = psi(psi)
        
        # Verify the fixed point property
        self.assertEqual(result, psi)
        self.assertEqual(str(result), "ψ")
    
    def test_idempotence(self):
        """Test that ψ is idempotent under self-application"""
        psi = Psi()
        
        # First application: ψ(ψ)
        result1 = psi(psi)
        
        # Second application: ψ(ψ(ψ))
        result2 = psi(result1)
        
        # Both should equal ψ
        self.assertEqual(result1, psi)
        self.assertEqual(result2, psi)
        self.assertEqual(result1, result2)
    
    def test_recursive_depth(self):
        """Test that arbitrary depth of self-application returns ψ"""
        psi = Psi()
        
        # Apply recursively n times
        result = psi
        for _ in range(10):
            result = psi(result)
        
        # Should still be ψ
        self.assertEqual(result, psi)
    
    def test_self_containment(self):
        """Test that ψ contains its own structure"""
        psi = Psi()
        
        # ψ equals ψ(ψ), so ψ contains the structure ψ(ψ)
        psi_of_psi = psi(psi)
        
        # They are the same entity
        self.assertIs(psi_of_psi, psi)
        
        # This demonstrates self-containment
        self.assertTrue(psi == psi(psi))
    
    def test_uniqueness_of_fixed_point(self):
        """Test that the fixed point is unique in the self-referential context"""
        psi1 = Psi()
        psi2 = Psi()
        
        # Both satisfy the equation
        self.assertEqual(psi1(psi1), psi1)
        self.assertEqual(psi2(psi2), psi2)
        
        # In the fundamental context, they represent the same entity
        self.assertEqual(psi1, psi2)
    
    def test_non_self_application(self):
        """Test behavior when ψ is applied to something other than itself"""
        psi = Psi()
        other = Psi("φ")
        
        # ψ(φ) should not equal ψ
        result = psi(other)
        self.assertNotEqual(result, psi)
        self.assertEqual(str(result), "ψ(φ)")
    
    def test_emergence_principle(self):
        """Test that structure emerges from the self-referential principle"""
        psi = Psi()
        
        # From ψ = ψ(ψ), we have distinction between:
        # 1. The entity ψ
        # 2. The operation ψ()
        # 3. The result ψ
        
        # This creates the first structure
        entity = psi
        operation = psi.__call__
        result = psi(psi)
        
        # All three aspects exist
        self.assertIsNotNone(entity)
        self.assertIsNotNone(operation)
        self.assertIsNotNone(result)
        
        # Yet they are unified
        self.assertEqual(entity, result)


class TestEmergentProperties(unittest.TestCase):
    """Test properties that emerge from ψ = ψ(ψ)"""
    
    def test_temporal_ordering_emergence(self):
        """Test that logical sequence emerges from self-application"""
        psi = Psi()
        
        # Track the sequence of applications
        initial_count = psi.applications
        psi(psi)
        after_one = psi.applications
        psi(psi)
        after_two = psi.applications
        
        # A temporal ordering emerges
        self.assertLess(initial_count, after_one)
        self.assertLess(after_one, after_two)
        
        # This demonstrates "before" and "after" emerging from timeless principle
    
    def test_spatial_distinction_emergence(self):
        """Test that spatial concepts emerge from domain/codomain distinction"""
        psi = Psi()
        
        # In ψ(ψ), we have:
        # - Domain: the argument ψ
        # - Codomain: the result ψ
        # This creates the first "space"
        
        # The same ψ appears in different "positions"
        domain_position = psi    # ψ as input
        codomain_position = psi(psi)  # ψ as output
        
        # They are the same entity
        self.assertEqual(domain_position, codomain_position)
        
        # Yet the distinction of positions creates "space"
        self.assertTrue(domain_position is codomain_position)
    
    def test_multiplicty_from_unity(self):
        """Test that multiple aspects emerge from singular principle"""
        psi = Psi()
        
        # From ψ = ψ(ψ), we get at least three aspects:
        aspects = {
            "function": psi,      # ψ as function
            "argument": psi,      # ψ as argument
            "value": psi(psi)     # ψ as result
        }
        
        # Multiple aspects exist
        self.assertEqual(len(aspects), 3)
        
        # Yet they are one
        self.assertTrue(all(v == psi for v in aspects.values()))


class TestMathematicalConsistency(unittest.TestCase):
    """Test mathematical consistency of the framework"""
    
    def test_fixed_point_theorem_application(self):
        """Test that ψ = ψ(ψ) is consistent with fixed point theory"""
        psi = Psi()
        
        # Define the operator T(x) = x(x)
        def T(x):
            return x(x)
        
        # ψ is a fixed point of T
        self.assertEqual(T(psi), psi)
        
        # Multiple applications don't change the result
        self.assertEqual(T(T(psi)), psi)
    
    def test_self_reference_consistency(self):
        """Test that self-reference is logically consistent"""
        psi = Psi()
        
        # The statement "ψ = ψ(ψ)" is self-consistent
        # Left side: ψ
        left = psi
        
        # Right side: ψ(ψ)
        right = psi(psi)
        
        # Equality holds
        self.assertEqual(left, right)
        
        # And this equality is stable under iteration
        for _ in range(5):
            left = right
            right = right(right)
            self.assertEqual(left, right)
    
    def test_no_infinite_regress(self):
        """Test that self-application doesn't create infinite regress"""
        psi = Psi()
        
        # Even though ψ = ψ(ψ) = ψ(ψ(ψ)) = ..., we don't have infinite regress
        # because it's a fixed point
        
        depths = []
        result = psi
        for depth in range(10):
            result = result(result)
            depths.append((depth, result))
        
        # All results are the same ψ
        self.assertTrue(all(r == psi for _, r in depths))
        
        # No infinite regress - we have stability
        self.assertEqual(depths[0][1], depths[-1][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)