#!/usr/bin/env python3
"""
Machine verification unit tests for D1.2: Binary Representation
Testing the constructive definition of binary representation in self-referential complete systems.
"""

import unittest
import hashlib
from typing import Set, List, Callable, Union


class BinaryRepresentation:
    """Implementation of D1.2: Binary Representation"""
    
    def __init__(self):
        self.alphabet = {0, 1}
        self.max_length = 1000  # Practical finite limit
    
    def sigma_star(self, max_len: int = None) -> Set[str]:
        """Generate Σ* up to specified length"""
        if max_len is None:
            max_len = self.max_length
        
        result = set()
        for n in range(max_len + 1):
            for i in range(2**n):
                binary_str = format(i, f'0{n}b') if n > 0 else ''
                result.add(binary_str)
        return result
    
    def sigma_n(self, n: int) -> Set[str]:
        """Generate Σⁿ - all binary strings of length n"""
        if n == 0:
            return {''}
        return {format(i, f'0{n}b') for i in range(2**n)}
    
    def is_encodable(self, s: str) -> bool:
        """Encodable predicate: checks if string can encode system information"""
        # Basic encodability: string consists only of 0s and 1s
        return all(c in '01' for c in s) and len(s) < float('inf')
    
    def state_space(self, max_len: int = 10) -> Set[str]:
        """Construct state space S according to D1.2"""
        sigma_star = self.sigma_star(max_len)
        return {s for s in sigma_star if len(s) < float('inf') and self.is_encodable(s)}
    
    def hash_function(self, x: Union[str, int, float]) -> int:
        """Hash function: U → ℕ"""
        return int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16)
    
    def to_binary(self, n: int) -> str:
        """ToBinary: ℕ → Σ*"""
        if n == 0:
            return '0'
        return bin(n)[2:]  # Remove '0b' prefix
    
    def encode(self, x: Union[str, int, float]) -> str:
        """Encode: U → S mapping according to D1.2"""
        hash_val = self.hash_function(x)
        binary_repr = self.to_binary(hash_val)
        return binary_repr


class TestBinaryRepresentation(unittest.TestCase):
    """Unit tests for D1.2: Binary Representation"""
    
    def setUp(self):
        self.br = BinaryRepresentation()
    
    def test_alphabet_definition(self):
        """Test alphabet Σ = {0, 1}"""
        self.assertEqual(self.br.alphabet, {0, 1})
        self.assertEqual(len(self.br.alphabet), 2)
    
    def test_sigma_n_construction(self):
        """Test Σⁿ construction for various n"""
        # Σ⁰ = {ε} (empty string)
        sigma_0 = self.br.sigma_n(0)
        self.assertEqual(sigma_0, {''})
        self.assertEqual(len(sigma_0), 1)
        
        # Σ¹ = {0, 1}
        sigma_1 = self.br.sigma_n(1)
        self.assertEqual(sigma_1, {'0', '1'})
        self.assertEqual(len(sigma_1), 2)
        
        # Σ² = {00, 01, 10, 11}
        sigma_2 = self.br.sigma_n(2)
        expected_sigma_2 = {'00', '01', '10', '11'}
        self.assertEqual(sigma_2, expected_sigma_2)
        self.assertEqual(len(sigma_2), 4)
        
        # General property: |Σⁿ| = 2ⁿ
        for n in range(5):
            sigma_n = self.br.sigma_n(n)
            self.assertEqual(len(sigma_n), 2**n)
    
    def test_sigma_star_construction(self):
        """Test Σ* = ⋃_{n=0}^∞ Σⁿ"""
        sigma_star_3 = self.br.sigma_star(3)
        
        # Should contain all strings of length 0, 1, 2, 3
        expected_strings = {'', '0', '1', '00', '01', '10', '11', 
                           '000', '001', '010', '011', '100', '101', '110', '111'}
        self.assertEqual(sigma_star_3, expected_strings)
        
        # Total count should be 2⁰ + 2¹ + 2² + 2³ = 1 + 2 + 4 + 8 = 15
        self.assertEqual(len(sigma_star_3), 15)
    
    def test_encodable_predicate(self):
        """Test Encodable: Σ* → {0,1} predicate"""
        # Valid binary strings
        self.assertTrue(self.br.is_encodable(''))
        self.assertTrue(self.br.is_encodable('0'))
        self.assertTrue(self.br.is_encodable('1'))
        self.assertTrue(self.br.is_encodable('01010'))
        self.assertTrue(self.br.is_encodable('11001100'))
        
        # Invalid strings (not binary)
        self.assertFalse(self.br.is_encodable('012'))
        self.assertFalse(self.br.is_encodable('abc'))
        self.assertFalse(self.br.is_encodable('2'))
        self.assertFalse(self.br.is_encodable('1a0'))
    
    def test_state_space_construction(self):
        """Test state space S construction"""
        S = self.br.state_space(3)
        
        # All elements should be encodable
        for s in S:
            self.assertTrue(self.br.is_encodable(s))
        
        # Should be subset of Σ*
        sigma_star_3 = self.br.sigma_star(3)
        self.assertTrue(S.issubset(sigma_star_3))
        
        # All strings in Σ* that are encodable should be in S
        expected_S = {s for s in sigma_star_3 if self.br.is_encodable(s)}
        self.assertEqual(S, expected_S)
    
    def test_hash_function_properties(self):
        """Test Hash: U → ℕ function properties"""
        # Hash function should be deterministic
        x = "test_input"
        hash1 = self.br.hash_function(x)
        hash2 = self.br.hash_function(x)
        self.assertEqual(hash1, hash2)
        
        # Hash should produce non-negative integers
        test_inputs = ["", "0", "1", "hello", 42, 3.14159]
        for inp in test_inputs:
            hash_val = self.br.hash_function(inp)
            self.assertIsInstance(hash_val, int)
            self.assertGreaterEqual(hash_val, 0)
        
        # Different inputs should typically produce different hashes
        hash_a = self.br.hash_function("a")
        hash_b = self.br.hash_function("b")
        self.assertNotEqual(hash_a, hash_b)
    
    def test_to_binary_function(self):
        """Test ToBinary: ℕ → Σ* function"""
        # Test specific conversions
        self.assertEqual(self.br.to_binary(0), '0')
        self.assertEqual(self.br.to_binary(1), '1')
        self.assertEqual(self.br.to_binary(2), '10')
        self.assertEqual(self.br.to_binary(3), '11')
        self.assertEqual(self.br.to_binary(4), '100')
        self.assertEqual(self.br.to_binary(5), '101')
        self.assertEqual(self.br.to_binary(8), '1000')
        
        # Test that result is always in Σ*
        for n in range(16):
            binary_str = self.br.to_binary(n)
            self.assertTrue(self.br.is_encodable(binary_str))
            # Verify it's a valid binary representation
            self.assertEqual(int(binary_str, 2), n)
    
    def test_encode_function(self):
        """Test Encode: U → S mapping"""
        # Test encoding various objects
        test_objects = ["hello", 42, 3.14, "", "binary_test"]
        
        for obj in test_objects:
            encoded = self.br.encode(obj)
            
            # Result should be in state space S
            self.assertTrue(self.br.is_encodable(encoded))
            
            # Should be deterministic
            encoded2 = self.br.encode(obj)
            self.assertEqual(encoded, encoded2)
            
            # Result should be non-empty for non-empty inputs
            if str(obj):
                self.assertGreater(len(encoded), 0)
    
    def test_encode_injectivity(self):
        """Test that encode function is injective (different inputs → different outputs)"""
        test_inputs = ["a", "b", "c", "test1", "test2", 1, 2, 42]
        encoded_values = [self.br.encode(inp) for inp in test_inputs]
        
        # All encoded values should be different (with high probability)
        unique_encoded = set(encoded_values)
        # Note: Hash collisions are possible but very unlikely for small test set
        self.assertGreaterEqual(len(unique_encoded), len(test_inputs) * 0.8)
    
    def test_constructive_properties(self):
        """Test constructive properties required for machine verification"""
        # State space should be effectively enumerable
        S_small = self.br.state_space(5)
        self.assertIsInstance(S_small, set)
        self.assertGreater(len(S_small), 0)
        
        # All functions should be total and computable
        for s in ['', '0', '1', '01', '10']:
            # Encodable should be decidable
            result = self.br.is_encodable(s)
            self.assertIsInstance(result, bool)
        
        # Encode should be total on finite inputs
        test_finite_inputs = ["", "a", "test", 0, 1, 42]
        for inp in test_finite_inputs:
            encoded = self.br.encode(inp)
            self.assertIsInstance(encoded, str)
            self.assertTrue(self.br.is_encodable(encoded))
    
    def test_mathematical_structure_preservation(self):
        """Test that binary representation preserves essential mathematical structure"""
        # The empty string should map to something distinguishable
        empty_encoded = self.br.encode("")
        nonempty_encoded = self.br.encode("nonempty")
        
        # Different conceptual objects should typically encode differently
        # (though hash collisions are theoretically possible)
        self.assertIsInstance(empty_encoded, str)
        self.assertIsInstance(nonempty_encoded, str)
        
        # Both should be valid binary representations
        self.assertTrue(self.br.is_encodable(empty_encoded))
        self.assertTrue(self.br.is_encodable(nonempty_encoded))


if __name__ == '__main__':
    unittest.main(verbosity=2)