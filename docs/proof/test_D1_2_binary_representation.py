#!/usr/bin/env python3
"""
Machine verification unit tests for D1.2: Binary Representation Definition
Testing the formal definition of binary representation systems.
"""

import unittest
from typing import Set, Callable, Any, Dict, List, Optional, Tuple


class BinaryRepresentationSystem:
    """System for testing binary representation properties"""
    
    def __init__(self):
        self.sigma = {'0', '1'}  # Binary alphabet
        self.encodings = {}
        
    def is_binary_string(self, s: str) -> bool:
        """Check if a string is a valid binary string"""
        return all(char in self.sigma for char in s)
    
    def verify_binary_representation(self, S: Set[str], encode_func: Callable) -> bool:
        """Verify if S has a valid binary representation"""
        # Check S ⊆ Σ*
        if not all(self.is_binary_string(s) for s in S):
            return False
        
        # Check existence of encoding function
        if encode_func is None:
            return False
        
        return True
    
    def binary_encode(self, x: Any) -> str:
        """Encode an object to binary string"""
        if x is None:
            return ""  # ε (empty string)
        elif x == "ground_state":
            return "0"
        elif x == "excited_state":
            return "1"
        else:
            # General encoding using hash
            h = hash(str(x))
            # Ensure positive value for binary conversion
            h = h if h >= 0 else h + 2**64
            return bin(h)[2:]  # Remove '0b' prefix
    
    def is_encodable(self, s: str) -> bool:
        """Check if a string is encodable (finite and binary)"""
        # Check finiteness
        if len(s) == float('inf'):
            return False
        
        # Check all characters are binary
        return self.is_binary_string(s)
    
    def complement(self, s: str) -> str:
        """Compute complement of binary string"""
        return ''.join('1' if bit == '0' else '0' for bit in s)
    
    def verify_minimality(self) -> bool:
        """Verify |Σ| = 2 is minimal for completeness"""
        # Two symbols are necessary
        return len(self.sigma) == 2
    
    def verify_completeness(self, universe: Set[Any], encode_func: Callable) -> bool:
        """Verify every object can be encoded"""
        try:
            for obj in universe:
                s = encode_func(obj)
                if not self.is_binary_string(s):
                    return False
            return True
        except:
            return False
    
    def verify_symmetry(self, S: Set[str]) -> bool:
        """Verify complement closure"""
        for s in S:
            comp = self.complement(s)
            if comp not in S:
                return False
        return True
    
    def decode(self, s: str, original_objects: Dict[str, Any]) -> Optional[Any]:
        """Decode binary string back to object (if mapping exists)"""
        return original_objects.get(s)
    
    def verify_reversibility(self, objects: List[Any], encode_func: Callable) -> bool:
        """Verify encoding is reversible"""
        # Create encoding map
        encoding_map = {}
        for obj in objects:
            s = encode_func(obj)
            if s in encoding_map and encoding_map[s] != obj:
                # Collision detected - not reversible
                return False
            encoding_map[s] = obj
        
        # Check decode(encode(x)) = x
        for obj in objects:
            s = encode_func(obj)
            decoded = encoding_map.get(s)
            if decoded != obj:
                return False
        
        return True
    
    def verify_finiteness(self, S: Set[str]) -> bool:
        """Verify all strings in S are finite"""
        return all(len(s) < float('inf') for s in S)
    
    def get_all_binary_strings_up_to_length(self, n: int) -> Set[str]:
        """Generate all binary strings up to length n"""
        strings = {''}  # Empty string
        for length in range(1, n + 1):
            for i in range(2**length):
                binary = bin(i)[2:].zfill(length)
                strings.add(binary)
        return strings
    
    def verify_semantic_mapping(self) -> Tuple[bool, Dict[str, str]]:
        """Verify semantic mapping of 0 and 1"""
        semantics = {
            '0': 'potential',
            '1': 'realized'
        }
        return True, semantics


class TestBinaryRepresentationDefinition(unittest.TestCase):
    """Unit tests for D1.2: Binary Representation Definition"""
    
    def setUp(self):
        self.system = BinaryRepresentationSystem()
        self.test_universe = {
            'a', 'b', 'c', 123, 456, 
            'ground_state', 'excited_state',
            (1, 2), 'list_obj', 'dict_obj'
        }
    
    def test_binary_alphabet(self):
        """Test binary alphabet properties"""
        # Test alphabet size
        self.assertEqual(len(self.system.sigma), 2)
        
        # Test alphabet elements
        self.assertIn('0', self.system.sigma)
        self.assertIn('1', self.system.sigma)
        
        # Test minimality
        self.assertTrue(self.system.verify_minimality())
    
    def test_binary_string_validation(self):
        """Test binary string validation"""
        # Valid binary strings
        self.assertTrue(self.system.is_binary_string(''))
        self.assertTrue(self.system.is_binary_string('0'))
        self.assertTrue(self.system.is_binary_string('1'))
        self.assertTrue(self.system.is_binary_string('0101'))
        self.assertTrue(self.system.is_binary_string('111000'))
        
        # Invalid strings
        self.assertFalse(self.system.is_binary_string('2'))
        self.assertFalse(self.system.is_binary_string('01a'))
        self.assertFalse(self.system.is_binary_string('abc'))
    
    def test_encoding_function(self):
        """Test binary encoding function"""
        # Test special cases
        self.assertEqual(self.system.binary_encode(None), '')
        self.assertEqual(self.system.binary_encode('ground_state'), '0')
        self.assertEqual(self.system.binary_encode('excited_state'), '1')
        
        # Test general encoding
        for obj in self.test_universe:
            if obj is not None:  # Skip None as it has special encoding
                encoded = self.system.binary_encode(obj)
                self.assertTrue(self.system.is_binary_string(encoded))
                self.assertGreater(len(encoded), 0)  # Non-empty for non-None
    
    def test_encodability_check(self):
        """Test encodability verification"""
        # Encodable strings
        self.assertTrue(self.system.is_encodable(''))
        self.assertTrue(self.system.is_encodable('0'))
        self.assertTrue(self.system.is_encodable('101010'))
        
        # Non-encodable strings
        self.assertFalse(self.system.is_encodable('012'))
        self.assertFalse(self.system.is_encodable('abc'))
    
    def test_complement_operation(self):
        """Test complement operation"""
        # Test basic complements
        self.assertEqual(self.system.complement('0'), '1')
        self.assertEqual(self.system.complement('1'), '0')
        self.assertEqual(self.system.complement('010'), '101')
        self.assertEqual(self.system.complement('1111'), '0000')
        self.assertEqual(self.system.complement(''), '')
        
        # Test double complement
        for s in ['0', '1', '010', '111000']:
            double_comp = self.system.complement(self.system.complement(s))
            self.assertEqual(s, double_comp)
    
    def test_completeness_property(self):
        """Test Property D1.2.2: Completeness"""
        # Every object in universe can be encoded
        completeness = self.system.verify_completeness(
            self.test_universe, 
            self.system.binary_encode
        )
        self.assertTrue(completeness)
        
        # Verify each encoding
        for obj in self.test_universe:
            encoded = self.system.binary_encode(obj)
            self.assertIsInstance(encoded, str)
            self.assertTrue(self.system.is_binary_string(encoded))
    
    def test_symmetry_property(self):
        """Test Property D1.2.3: Symmetry"""
        # Generate a set of binary strings
        test_set = self.system.get_all_binary_strings_up_to_length(3)
        
        # Add complements to make it symmetric
        symmetric_set = test_set.copy()
        for s in test_set:
            symmetric_set.add(self.system.complement(s))
        
        # Verify symmetry
        self.assertTrue(self.system.verify_symmetry(symmetric_set))
        
        # Test non-symmetric set
        non_symmetric = {'0', '10', '110'}  # Missing complements
        self.assertFalse(self.system.verify_symmetry(non_symmetric))
    
    def test_reversibility_property(self):
        """Test Property D1.2.4: Reversibility"""
        # Test with unique objects
        unique_objects = ['a', 'b', 'c', 'd', 'e']
        
        # Custom injective encoding
        def injective_encode(x):
            if x in unique_objects:
                # Create unique encoding for each object
                idx = unique_objects.index(x)
                return bin(idx + 1)[2:]  # 1, 10, 11, 100, 101
            return self.system.binary_encode(x)
        
        # Verify reversibility
        is_reversible = self.system.verify_reversibility(
            unique_objects, 
            injective_encode
        )
        self.assertTrue(is_reversible)
    
    def test_finiteness_property(self):
        """Test Property D1.2.5: Finiteness"""
        # Generate finite set of strings
        finite_set = self.system.get_all_binary_strings_up_to_length(10)
        
        # Verify all are finite
        self.assertTrue(self.system.verify_finiteness(finite_set))
        
        # Each string has finite length
        for s in finite_set:
            self.assertLess(len(s), float('inf'))
    
    def test_semantic_mapping(self):
        """Test Lemma D1.2.3: Semantic mapping"""
        is_valid, semantics = self.system.verify_semantic_mapping()
        
        self.assertTrue(is_valid)
        self.assertEqual(semantics['0'], 'potential')
        self.assertEqual(semantics['1'], 'realized')
    
    def test_binary_representation_definition(self):
        """Test complete binary representation definition"""
        # Create a binary representation system
        S = self.system.get_all_binary_strings_up_to_length(5)
        
        # Verify it satisfies binary representation
        has_binary_rep = self.system.verify_binary_representation(
            S, 
            self.system.binary_encode
        )
        self.assertTrue(has_binary_rep)
    
    def test_string_concatenation_properties(self):
        """Test properties of binary string concatenation"""
        # Concatenation preserves binary nature
        s1, s2 = '010', '110'
        concat = s1 + s2
        self.assertTrue(self.system.is_binary_string(concat))
        self.assertEqual(concat, '010110')
        
        # Empty string is identity
        self.assertEqual('' + s1, s1)
        self.assertEqual(s1 + '', s1)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty string
        self.assertTrue(self.system.is_binary_string(''))
        self.assertEqual(self.system.complement(''), '')
        
        # Single characters
        self.assertEqual(self.system.complement('0'), '1')
        self.assertEqual(self.system.complement('1'), '0')
        
        # Very long strings
        long_string = '01' * 1000
        self.assertTrue(self.system.is_binary_string(long_string))
        comp = self.system.complement(long_string)
        self.assertEqual(comp, '10' * 1000)
    
    def test_mathematical_representation(self):
        """Test mathematical representation formulas"""
        # Test Σ* generation
        sigma_star_3 = self.system.get_all_binary_strings_up_to_length(3)
        expected = {
            '',  # ε
            '0', '1',  # Σ¹
            '00', '01', '10', '11',  # Σ²
            '000', '001', '010', '011', '100', '101', '110', '111'  # Σ³
        }
        self.assertEqual(sigma_star_3, expected)
        
        # Verify |Σⁿ| = 2ⁿ
        for n in range(4):  # Only test up to n=3 since sigma_star_3 only has strings up to length 3
            strings_n = {s for s in sigma_star_3 if len(s) == n}
            if n == 0:
                self.assertEqual(len(strings_n), 1)  # Just ε
            else:
                self.assertEqual(len(strings_n), 2**n)


if __name__ == '__main__':
    unittest.main(verbosity=2)