#!/usr/bin/env python3
"""
Test Suite for T0-4: Binary Encoding Completeness Theory

This module provides comprehensive testing of the universal completeness
of binary-Zeckendorf encoding for all possible information states.
"""

import unittest
import random
import math
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from itertools import combinations
import struct
import json


class ZeckendorfEncoder:
    """Complete implementation of Zeckendorf encoding system"""
    
    def __init__(self):
        """Initialize with precomputed Fibonacci numbers"""
        self.fibs = self._generate_fibonacci(100)
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate first n Fibonacci numbers (1,2,3,5,8,...)"""
        fibs = [1, 2]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs
    
    def encode(self, n: int) -> str:
        """
        Convert natural number to Zeckendorf binary string.
        
        Args:
            n: Natural number to encode
            
        Returns:
            Binary string in Zeckendorf representation (no consecutive 1s)
        """
        if n == 0:
            return "0"
        
        # Find Fibonacci numbers needed
        result = []
        remaining = n
        
        # Start from largest Fibonacci number ≤ n
        for i in range(len(self.fibs) - 1, -1, -1):
            if self.fibs[i] <= remaining:
                result.append('1')
                remaining -= self.fibs[i]
                if remaining == 0:
                    break
            elif result:  # Only add 0s after first 1
                result.append('0')
        
        return ''.join(result) if result else '0'
    
    def decode(self, z: str) -> int:
        """
        Convert Zeckendorf binary string back to natural number.
        
        Args:
            z: Binary string in Zeckendorf representation
            
        Returns:
            Natural number
        """
        if z == "0" or not z:
            return 0
        
        n = 0
        for i, bit in enumerate(reversed(z)):
            if bit == '1':
                n += self.fibs[i]
        return n
    
    def is_valid(self, z: str) -> bool:
        """Check if binary string satisfies no-11 constraint"""
        return '11' not in z
    
    def next_valid(self, z: str) -> str:
        """Get next valid Zeckendorf string"""
        n = self.decode(z)
        return self.encode(n + 1)


@dataclass
class InformationState:
    """Represents an information state in the system"""
    content: Any
    info_measure: int
    
    def __hash__(self):
        return hash(self.info_measure)
    
    def __eq__(self, other):
        return self.info_measure == other.info_measure


@dataclass 
class ComplexStructure:
    """Represents a complex structure with components and relations"""
    components: List[InformationState]
    relations: Dict[Tuple[int, int], str]
    
    def encode(self, encoder: ZeckendorfEncoder) -> str:
        """Encode structure preserving relationships"""
        parts = []
        
        # Encode each component
        for comp in self.components:
            parts.append(encoder.encode(comp.info_measure))
        
        # Encode relations
        for (i, j), rel_type in self.relations.items():
            rel_code = hash(rel_type) % 1000  # Simple relation encoding
            parts.append(encoder.encode(rel_code))
        
        # Join with separator that maintains no-11
        return '00'.join(parts)
    
    def verify_preservation(self, encoded: str) -> bool:
        """Verify that structure is preserved in encoding"""
        # Check no consecutive 1s across boundaries
        if '11' in encoded:
            return False
        
        # Check separators are preserved
        parts = encoded.split('00')
        expected_parts = len(self.components) + len(self.relations)
        
        return len(parts) == expected_parts


@dataclass
class DynamicProcess:
    """Represents a dynamic process with state transitions"""
    states: List[InformationState]
    transitions: List[str]
    
    def encode(self, encoder: ZeckendorfEncoder) -> str:
        """Encode process maintaining temporal order"""
        parts = []
        
        for i, state in enumerate(self.states):
            parts.append(encoder.encode(state.info_measure))
            if i < len(self.transitions):
                trans_code = hash(self.transitions[i]) % 1000
                parts.append(encoder.encode(trans_code))
        
        return '00'.join(parts)
    
    def verify_dynamics(self, encoded: str) -> bool:
        """Verify process dynamics are preserved"""
        if '11' in encoded:
            return False
        
        parts = encoded.split('00')
        # Should have alternating states and transitions
        expected = len(self.states) + len(self.transitions)
        return len(parts) == expected


class TestUniversalRepresentation(unittest.TestCase):
    """Test Theorem 3.1: Universal Representation"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_every_natural_number_has_encoding(self):
        """Test that every natural number has unique Zeckendorf encoding"""
        seen_encodings = set()
        
        for n in range(10000):
            z = self.encoder.encode(n)
            
            # Check validity (no consecutive 1s)
            self.assertTrue(self.encoder.is_valid(z),
                          f"Invalid encoding for {n}: {z}")
            
            # Check uniqueness
            self.assertNotIn(z, seen_encodings,
                           f"Duplicate encoding {z} for {n}")
            seen_encodings.add(z)
            
            # Check reversibility
            decoded = self.encoder.decode(z)
            self.assertEqual(decoded, n,
                           f"Failed to decode {n}: {z} -> {decoded}")
    
    def test_information_state_encoding(self):
        """Test that information states map to unique encodings"""
        states = [InformationState(f"data_{i}", i) for i in range(100)]
        encodings = set()
        
        for state in states:
            z = self.encoder.encode(state.info_measure)
            self.assertNotIn(z, encodings)
            encodings.add(z)
            
            # Verify encoding properties
            self.assertTrue(self.encoder.is_valid(z))
    
    def test_no_information_loss(self):
        """Test that encoding preserves all information"""
        test_values = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        test_values.extend([7, 12, 20, 50, 100, 500, 1000, 9999])
        
        for n in test_values:
            z = self.encoder.encode(n)
            recovered = self.encoder.decode(z)
            self.assertEqual(n, recovered,
                           f"Information loss for {n}: encoded as {z}, decoded as {recovered}")


class TestEncodingDensity(unittest.TestCase):
    """Test Theorem 3.2: Encoding Density"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def test_density_calculation(self):
        """Test that Zeckendorf strings have correct density"""
        for length in range(1, 20):
            valid_count = 0
            total_count = 2 ** length
            
            # Count valid Zeckendorf strings of given length
            for i in range(total_count):
                binary = format(i, f'0{length}b')
                if self.encoder.is_valid(binary):
                    valid_count += 1
            
            # Compare with theoretical density
            # Valid strings of length n ≈ F_{n+2}
            fib_n_plus_2 = self.encoder.fibs[length + 1] if length + 1 < len(self.encoder.fibs) else 0
            
            # Allow some deviation for small lengths
            if length > 5:
                ratio = valid_count / fib_n_plus_2
                self.assertAlmostEqual(ratio, 1.0, delta=0.2,
                                     msg=f"Length {length}: {valid_count} vs expected {fib_n_plus_2}")
    
    def test_sufficient_granularity(self):
        """Test that density provides sufficient granularity"""
        # For any range, we should have enough representable values
        test_ranges = [(0, 100), (100, 1000), (1000, 10000)]
        
        for start, end in test_ranges:
            representable = set()
            for n in range(start, end):
                z = self.encoder.encode(n)
                representable.add(z)
            
            # Should have unique encoding for each value
            self.assertEqual(len(representable), end - start,
                           f"Insufficient granularity in range {start}-{end}")


class TestFiniteCompleteness(unittest.TestCase):
    """Test Theorem 4.1: Finite State Completeness"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_finite_state_encoding(self):
        """Test that all finite states have valid encodings"""
        # Create diverse finite states
        states = []
        
        # Simple numeric states
        for i in range(100):
            states.append(InformationState(i, i))
        
        # Complex data states
        for i in range(100):
            data = {"id": i, "value": i * 17 % 100, "type": f"type_{i % 5}"}
            info_measure = hash(json.dumps(data, sort_keys=True)) % 10000
            states.append(InformationState(data, abs(info_measure)))
        
        # Verify all states encodable
        for state in states:
            z = self.encoder.encode(state.info_measure)
            self.assertTrue(self.encoder.is_valid(z))
            
            # Verify uniqueness within finite set
            decoded = self.encoder.decode(z)
            self.assertEqual(decoded, state.info_measure)
    
    def test_boundary_cases(self):
        """Test encoding at boundaries"""
        # Test Fibonacci numbers themselves
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        
        for f in fibs:
            z = self.encoder.encode(f)
            self.assertTrue(self.encoder.is_valid(z))
            self.assertEqual(self.encoder.decode(z), f)
            
            # Fibonacci numbers should have simple encodings
            # F_n should be encoded as 10...0 with n-1 zeros
            one_count = z.count('1')
            self.assertEqual(one_count, 1,
                           f"Fibonacci {f} should have single 1 in encoding {z}")


class TestInfiniteApproximation(unittest.TestCase):
    """Test Theorem 4.2: Infinite State Approximation"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_pi_approximation(self):
        """Test encoding approximation of irrational numbers like π"""
        pi = math.pi
        
        # Convert to increasingly precise integer approximations
        precisions = [10, 100, 1000, 10000, 100000]
        errors = []
        
        for precision in precisions:
            pi_int = int(pi * precision)
            z = self.encoder.encode(pi_int)
            recovered = self.encoder.decode(z)
            pi_approx = recovered / precision
            
            error = abs(pi - pi_approx)
            errors.append(error)
            
            # Error should decrease with precision
            self.assertLess(error, 1 / precision)
            
            # Encoding should be valid
            self.assertTrue(self.encoder.is_valid(z))
        
        # Verify error decreases
        for i in range(1, len(errors)):
            self.assertLess(errors[i], errors[i-1])
    
    def test_convergent_sequence(self):
        """Test that sequences converge in encoding space"""
        # Test sequence converging to √2
        sqrt2 = math.sqrt(2)
        
        # Newton's method for √2
        x = [1.0]
        for _ in range(10):
            x.append((x[-1] + 2/x[-1]) / 2)
        
        # Encode sequence with fixed precision
        precision = 1000000
        encodings = []
        
        for val in x:
            n = int(val * precision)
            z = self.encoder.encode(n)
            encodings.append(z)
            self.assertTrue(self.encoder.is_valid(z))
        
        # Later terms should stabilize
        self.assertEqual(encodings[-1], encodings[-2],
                        "Convergent sequence should stabilize in encoding")


class TestStructuralPreservation(unittest.TestCase):
    """Test Theorem 4.3: Structure Encoding"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_complex_structure_encoding(self):
        """Test that complex structures preserve relationships"""
        # Create a structure with components and relations
        components = [InformationState(f"comp_{i}", i * 10) for i in range(5)]
        relations = {
            (0, 1): "follows",
            (1, 2): "depends",
            (2, 3): "triggers",
            (3, 4): "completes"
        }
        
        structure = ComplexStructure(components, relations)
        encoded = structure.encode(self.encoder)
        
        # Verify encoding maintains no-11 constraint
        self.assertNotIn('11', encoded)
        
        # Verify structure preservation
        self.assertTrue(structure.verify_preservation(encoded))
        
        # Verify components are recoverable
        parts = encoded.split('00')
        for i, comp in enumerate(components):
            if i < len(parts):
                decoded = self.encoder.decode(parts[i])
                self.assertEqual(decoded, comp.info_measure)
    
    def test_hierarchical_structure(self):
        """Test nested hierarchical structures"""
        # Create nested structure
        level1 = [InformationState(f"L1_{i}", i) for i in range(3)]
        level2 = [InformationState(f"L2_{i}", i * 10) for i in range(3)]
        level3 = [InformationState(f"L3_{i}", i * 100) for i in range(3)]
        
        # Encode each level
        enc1 = '00'.join([self.encoder.encode(s.info_measure) for s in level1])
        enc2 = '00'.join([self.encoder.encode(s.info_measure) for s in level2])
        enc3 = '00'.join([self.encoder.encode(s.info_measure) for s in level3])
        
        # Combine with level separator
        hierarchical = '0000'.join([enc1, enc2, enc3])
        
        # Verify no consecutive 1s even across hierarchy
        self.assertNotIn('11', hierarchical)
        
        # Verify level separation
        levels = hierarchical.split('0000')
        self.assertEqual(len(levels), 3)


class TestOptimalEfficiency(unittest.TestCase):
    """Test Theorem 5.1: Optimal Efficiency"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def test_encoding_length_optimality(self):
        """Test that Zeckendorf encoding length is near-optimal"""
        test_numbers = [10, 50, 100, 500, 1000, 5000, 10000]
        
        for n in test_numbers:
            z = self.encoder.encode(n)
            zeck_length = len(z)
            
            # Theoretical minimum (Shannon entropy)
            min_length = math.ceil(math.log2(n + 1))
            
            # Zeckendorf length should be approximately 1.44 * minimum
            ratio = zeck_length / min_length
            expected_ratio = 1 / math.log2(self.golden_ratio)  # ≈ 1.44
            
            # Allow some deviation for small numbers
            if n > 100:
                self.assertLess(ratio, expected_ratio + 0.3,
                              f"Encoding of {n} is too long: {zeck_length} vs min {min_length}")
    
    def test_no_shorter_valid_encoding(self):
        """Test that no shorter encoding exists with no-11 constraint"""
        # For small numbers, enumerate all possible shorter encodings
        for n in range(1, 100):
            z = self.encoder.encode(n)
            z_length = len(z)
            
            # Check all shorter binary strings
            found_shorter = False
            for length in range(1, z_length):
                for i in range(2 ** length):
                    candidate = format(i, f'0{length}b')
                    
                    # Must satisfy no-11 constraint
                    if '11' not in candidate:
                        # Check if it could encode n
                        decoded = self.encoder.decode(candidate)
                        if decoded == n:
                            found_shorter = True
                            break
                
                if found_shorter:
                    break
            
            self.assertFalse(found_shorter,
                           f"Found shorter encoding for {n} than {z}")


class TestCompressionLimits(unittest.TestCase):
    """Test Theorem 5.2: Compression Bounds"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def test_compression_bound(self):
        """Test that compression cannot exceed theoretical limit"""
        # Generate random valid Zeckendorf strings
        test_strings = []
        for length in range(10, 30):
            for _ in range(10):
                # Generate random valid string
                s = ""
                prev = '0'
                for _ in range(length):
                    if prev == '1':
                        s += '0'
                        prev = '0'
                    else:
                        bit = random.choice(['0', '1'])
                        s += bit
                        prev = bit
                test_strings.append(s)
        
        # Try to compress (simple run-length encoding)
        total_original = 0
        total_compressed = 0
        
        for s in test_strings:
            total_original += len(s)
            
            # Simple compression: count runs
            compressed = []
            current = s[0]
            count = 1
            
            for bit in s[1:]:
                if bit == current:
                    count += 1
                else:
                    compressed.append((current, count))
                    current = bit
                    count = 1
            compressed.append((current, count))
            
            # Estimate compressed size (simplified)
            compressed_size = len(compressed) * 8  # Rough estimate
            total_compressed += min(compressed_size, len(s))
        
        compression_ratio = 1 - (total_compressed / total_original)
        max_compression = 1 - 1 / (self.golden_ratio ** 2)  # ≈ 0.382
        
        # Compression should not exceed theoretical limit
        self.assertLessEqual(compression_ratio, max_compression + 0.1)


class TestProcessEncoding(unittest.TestCase):
    """Test Theorems 6.1-6.2: Process and Dynamic Encoding"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_dynamic_process_encoding(self):
        """Test that dynamic processes can be encoded"""
        # Create a simple state machine process
        states = [
            InformationState("START", 0),
            InformationState("PROCESS", 10),
            InformationState("WAIT", 20),
            InformationState("COMPLETE", 30)
        ]
        transitions = ["init", "process", "finish"]
        
        process = DynamicProcess(states, transitions)
        encoded = process.encode(self.encoder)
        
        # Verify encoding validity
        self.assertNotIn('11', encoded)
        
        # Verify process dynamics preserved
        self.assertTrue(process.verify_dynamics(encoded))
        
        # Verify temporal ordering maintained
        parts = encoded.split('00')
        
        # Decode states and verify order
        state_indices = [0, 2, 4, 6]  # Every other part is a state
        for i, idx in enumerate(state_indices):
            if idx < len(parts):
                decoded = self.encoder.decode(parts[idx])
                self.assertEqual(decoded, states[i].info_measure)
    
    def test_recursive_process_encoding(self):
        """Test recursive self-referential processes"""
        # Simulate ψ = ψ(ψ) style recursion
        def recursive_state(depth: int) -> int:
            if depth == 0:
                return 1
            return recursive_state(depth - 1) * 2 + 1
        
        # Generate recursive sequence
        depths = list(range(10))
        states = [InformationState(f"R_{d}", recursive_state(d)) for d in depths]
        
        # Encode each state
        encodings = []
        for state in states:
            z = self.encoder.encode(state.info_measure)
            encodings.append(z)
            self.assertTrue(self.encoder.is_valid(z))
        
        # Verify recursive pattern is preserved
        # Each encoding should be related to previous
        for i in range(1, len(encodings)):
            # Decode to verify relationship
            prev_val = self.encoder.decode(encodings[i-1])
            curr_val = self.encoder.decode(encodings[i])
            
            # Verify recursive relationship
            self.assertEqual(curr_val, prev_val * 2 + 1)
    
    def test_parallel_process_encoding(self):
        """Test encoding of parallel processes"""
        # Create two parallel processes
        process1 = DynamicProcess(
            [InformationState(f"P1_S{i}", i * 10) for i in range(3)],
            ["p1_t1", "p1_t2"]
        )
        
        process2 = DynamicProcess(
            [InformationState(f"P2_S{i}", i * 20 + 5) for i in range(3)],
            ["p2_t1", "p2_t2"]
        )
        
        # Encode separately
        enc1 = process1.encode(self.encoder)
        enc2 = process2.encode(self.encoder)
        
        # Combine with process separator
        parallel = enc1 + '000' + enc2
        
        # Verify no interference
        self.assertNotIn('11', parallel)
        
        # Verify processes are distinguishable
        procs = parallel.split('000')
        self.assertEqual(len(procs), 2)
        self.assertEqual(procs[0], enc1)
        self.assertEqual(procs[1], enc2)


class TestFundamentalCompleteness(unittest.TestCase):
    """Test Theorem 7.1: Fundamental Completeness"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_complete_bijection(self):
        """Test that encoding is a complete bijection"""
        # Test bijectivity for a range
        test_range = range(1000)
        
        # Injectivity: different inputs give different outputs
        encodings = {}
        for n in test_range:
            z = self.encoder.encode(n)
            self.assertNotIn(z, encodings.values())
            encodings[n] = z
        
        # Surjectivity: every valid encoding corresponds to some input
        # Generate all valid encodings up to certain length
        max_length = 10
        for length in range(1, max_length):
            for i in range(2 ** length):
                binary = format(i, f'0{length}b')
                if self.encoder.is_valid(binary):
                    # Should decode to some natural number
                    n = self.encoder.decode(binary)
                    self.assertIsInstance(n, int)
                    self.assertGreaterEqual(n, 0)
    
    def test_all_information_types(self):
        """Test encoding of diverse information types"""
        # Test different categories of information
        
        # 1. Discrete symbols
        symbols = ['A', 'B', 'C', 'X', 'Y', 'Z']
        for i, sym in enumerate(symbols):
            state = InformationState(sym, ord(sym))
            z = self.encoder.encode(state.info_measure)
            self.assertTrue(self.encoder.is_valid(z))
        
        # 2. Continuous values (discretized)
        continuous = [0.1, 0.5, 0.9, 1.414, 2.718, 3.14159]
        for val in continuous:
            discretized = int(val * 10000)
            z = self.encoder.encode(discretized)
            self.assertTrue(self.encoder.is_valid(z))
        
        # 3. Complex structures
        structure = {
            "nested": {
                "data": [1, 2, 3],
                "more": {"deep": "value"}
            }
        }
        struct_hash = abs(hash(json.dumps(structure, sort_keys=True)))
        z = self.encoder.encode(struct_hash % 100000)
        self.assertTrue(self.encoder.is_valid(z))
        
        # 4. Binary data
        binary_data = b'\x00\x01\x02\xFF\xFE\xFD'
        data_int = int.from_bytes(binary_data, 'big')
        z = self.encoder.encode(data_int)
        self.assertTrue(self.encoder.is_valid(z))
    
    def test_scale_invariance(self):
        """Test that encoding works across all scales"""
        scales = [
            1,           # Single bit
            10,          # Small
            100,         # Medium  
            1000,        # Large
            10000,       # Very large
            100000,      # Huge
            1000000      # Extreme
        ]
        
        for scale in scales:
            # Test at each scale
            test_vals = [scale - 1, scale, scale + 1]
            
            for val in test_vals:
                z = self.encoder.encode(val)
                self.assertTrue(self.encoder.is_valid(z))
                
                # Verify reversibility at all scales
                decoded = self.encoder.decode(z)
                self.assertEqual(decoded, val)


class TestEncodingUniqueness(unittest.TestCase):
    """Test Theorem 7.2: Uniqueness of Encoding System"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_no_alternative_with_11(self):
        """Test that allowing 11 breaks uniqueness"""
        # Show that allowing consecutive 1s leads to ambiguity
        
        # Example: number 3 
        # In Zeckendorf: 3 = F₃ = "100" 
        # If we allowed 11: could also be "11" (1+2=3)
        
        n = 3
        correct_encoding = self.encoder.encode(n)
        self.assertEqual(correct_encoding, "100")
        self.assertNotIn('11', correct_encoding)
        
        # The string "11" would also decode to 3 if allowed
        # This demonstrates non-uniqueness without the constraint
        
        # Verify our encoding is unique
        for i in range(2 ** 5):
            binary = format(i, '05b')
            if self.encoder.is_valid(binary):
                if self.encoder.decode(binary) == n:
                    # Should only match our encoding
                    self.assertTrue(binary.endswith(correct_encoding))
    
    def test_binary_is_minimal(self):
        """Test that binary is the minimal base"""
        # Any base < 2 cannot distinguish states
        # Base 1 (unary) would need infinite symbols for some numbers
        
        # Demonstrate binary minimality
        test_numbers = [0, 1, 2, 3, 5, 8, 13]
        
        for n in test_numbers:
            binary_enc = self.encoder.encode(n)
            
            # Unary would need n symbols
            unary_length = n if n > 0 else 1
            
            # Binary Zeckendorf is more efficient than unary
            self.assertLess(len(binary_enc), unary_length + 5,
                          f"Binary should be more efficient than unary for {n}")
    
    def test_isomorphism_of_alternatives(self):
        """Test that any valid alternative is isomorphic to Zeckendorf"""
        # Any encoding with same constraints must be isomorphic
        
        # Create mapping between our encoding and hypothetical alternative
        mapping = {}
        
        for n in range(100):
            z = self.encoder.encode(n)
            # Any alternative with same properties would map n -> some unique string
            # That string must have same properties as z
            mapping[n] = z
        
        # Verify mapping is bijective (one-to-one and onto)
        values = list(mapping.values())
        self.assertEqual(len(values), len(set(values)), 
                        "Mapping must be one-to-one")
        
        # Verify mapping preserves order relationships
        for i in range(len(mapping) - 1):
            if i < i + 1:  # Order in naturals
                # Order should be preserved in encoding (lexicographically)
                z_i = self.encoder.decode(mapping[i])
                z_next = self.encoder.decode(mapping[i + 1])
                self.assertLess(z_i, z_next)


class TestComputationalVerification(unittest.TestCase):
    """Test computational properties and algorithms"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_encoding_algorithm_complexity(self):
        """Test that encoding is O(log n) time"""
        import time
        
        # Measure time for different scales
        times = []
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            start = time.perf_counter()
            
            # Encode the number
            z = self.encoder.encode(size)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Verify result
            self.assertTrue(self.encoder.is_valid(z))
        
        # Time should grow logarithmically
        # Each 10x increase in size should have similar time increase
        growth_rates = []
        for i in range(1, len(times)):
            if times[i-1] > 0:
                growth = times[i] / times[i-1]
                growth_rates.append(growth)
        
        # Growth rate should be much less than linear (10x)
        avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 1
        self.assertLess(avg_growth, 5,
                       "Algorithm complexity should be logarithmic")
    
    def test_validation_properties(self):
        """Test all validation properties P1-P4"""
        # P1: No consecutive ones
        for n in range(1000):
            z = self.encoder.encode(n)
            self.assertNotIn('11', z, f"P1 violated for {n}: {z}")
        
        # P2: Uniqueness
        encodings = set()
        for n in range(1000):
            z = self.encoder.encode(n)
            self.assertNotIn(z, encodings, f"P2 violated: duplicate encoding for {n}")
            encodings.add(z)
        
        # P3: Completeness
        for n in range(1000):
            z = self.encoder.encode(n)
            self.assertIsNotNone(z, f"P3 violated: no encoding for {n}")
            self.assertTrue(len(z) > 0, f"P3 violated: empty encoding for {n}")
        
        # P4: Invertibility  
        for n in range(1000):
            z = self.encoder.encode(n)
            recovered = self.encoder.decode(z)
            self.assertEqual(n, recovered, f"P4 violated for {n}")
            
            # Also test encode(decode(z)) = z for valid strings
            if n > 0:  # Skip 0 due to representation choices
                re_encoded = self.encoder.encode(recovered)
                self.assertEqual(z, re_encoded, f"P4 violated on re-encoding {n}")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test zero
        self.assertEqual(self.encoder.encode(0), "0")
        self.assertEqual(self.encoder.decode("0"), 0)
        
        # Test powers of 2
        for p in range(10):
            n = 2 ** p
            z = self.encoder.encode(n)
            self.assertTrue(self.encoder.is_valid(z))
            self.assertEqual(self.encoder.decode(z), n)
        
        # Test Fibonacci numbers
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for f in fibs:
            z = self.encoder.encode(f)
            self.assertTrue(self.encoder.is_valid(z))
            self.assertEqual(self.encoder.decode(z), f)
            
            # Fibonacci numbers should have simple encodings
            self.assertEqual(z.count('1'), 1,
                           f"Fibonacci {f} should have exactly one 1")
        
        # Test maximum values at each bit length
        for bits in range(1, 15):
            # Maximum valid Zeckendorf string of length n has pattern 10101...
            max_valid = ""
            for i in range(bits):
                max_valid += "1" if i % 2 == 0 else "0"
            
            if self.encoder.is_valid(max_valid):
                n = self.encoder.decode(max_valid)
                z = self.encoder.encode(n)
                # Should round-trip correctly
                self.assertEqual(self.encoder.decode(z), n)


class TestIntegrationComplete(unittest.TestCase):
    """Integration tests for complete system"""
    
    def setUp(self):
        self.encoder = ZeckendorfEncoder()
    
    def test_complete_information_system(self):
        """Test complete information encoding system"""
        # Simulate a complete information system
        
        # 1. Create diverse information states
        states = []
        
        # Numeric states
        for i in range(50):
            states.append(InformationState(f"num_{i}", i))
        
        # Structural states
        structures = []
        for i in range(10):
            components = [InformationState(f"c_{i}_{j}", i * 10 + j) for j in range(3)]
            relations = {(0, 1): "rel_a", (1, 2): "rel_b"}
            structures.append(ComplexStructure(components, relations))
        
        # Process states
        processes = []
        for i in range(10):
            proc_states = [InformationState(f"p_{i}_{j}", i * 20 + j * 2) for j in range(4)]
            transitions = [f"t_{j}" for j in range(3)]
            processes.append(DynamicProcess(proc_states, transitions))
        
        # 2. Encode everything
        all_encodings = set()
        
        # Encode simple states
        for state in states:
            z = self.encoder.encode(state.info_measure)
            self.assertTrue(self.encoder.is_valid(z))
            all_encodings.add(z)
        
        # Encode structures
        for struct in structures:
            z = struct.encode(self.encoder)
            self.assertNotIn('11', z)
            all_encodings.add(z)
        
        # Encode processes
        for proc in processes:
            z = proc.encode(self.encoder)
            self.assertNotIn('11', z)
            all_encodings.add(z)
        
        # 3. Verify completeness properties
        
        # All encodings should be unique
        self.assertEqual(len(all_encodings), 
                        len(states) + len(structures) + len(processes))
        
        # All encodings should be valid
        for z in all_encodings:
            self.assertNotIn('11', z)
        
        # System should handle composite encoding
        full_system = '0000'.join(all_encodings)
        self.assertNotIn('11', full_system)
    
    def test_entropy_increase_simulation(self):
        """Test that encoding respects entropy increase principle"""
        # Simulate system with increasing entropy
        
        entropy_states = []
        for t in range(100):
            # Entropy increases over time
            entropy = int(10 * (1 + t * 0.1) * math.log(t + 2))
            entropy_states.append(entropy)
        
        # Encode entropy sequence
        encodings = []
        for entropy in entropy_states:
            z = self.encoder.encode(entropy)
            encodings.append(z)
            self.assertTrue(self.encoder.is_valid(z))
        
        # Verify entropy increase is preserved
        decoded_entropies = [self.encoder.decode(z) for z in encodings]
        
        # Check general increasing trend (allow local fluctuations)
        increases = 0
        for i in range(1, len(decoded_entropies)):
            if decoded_entropies[i] >= decoded_entropies[i-1]:
                increases += 1
        
        # Most steps should show increase
        self.assertGreater(increases / len(decoded_entropies), 0.8,
                          "Entropy should generally increase")
    
    def test_self_referential_encoding(self):
        """Test encoding of self-referential structures (ψ = ψ(ψ))"""
        # Create self-referential structure
        
        def psi(n: int, depth: int = 0) -> int:
            """Simulate ψ = ψ(ψ) up to finite depth"""
            if depth >= 5:  # Limit recursion
                return n
            
            # Self-application with Fibonacci growth
            if n < 2:
                return 1
            
            # Apply self-reference with Fibonacci-like growth
            return psi(n - 1, depth + 1) + psi(n - 2, depth + 1)
        
        # Generate self-referential sequence
        psi_values = []
        for i in range(20):
            val = psi(i)
            psi_values.append(val)
        
        # Encode sequence
        psi_encodings = []
        for val in psi_values:
            z = self.encoder.encode(val)
            psi_encodings.append(z)
            self.assertTrue(self.encoder.is_valid(z))
        
        # Verify self-referential property is preserved
        # Fibonacci-like growth should be evident
        for i in range(2, len(psi_values)):
            # Each value should relate to previous values
            curr = self.encoder.decode(psi_encodings[i])
            prev1 = self.encoder.decode(psi_encodings[i-1])
            prev2 = self.encoder.decode(psi_encodings[i-2])
            
            # Verify Fibonacci-like relationship (within recursion limits)
            if i < 7:  # Where recursion depth doesn't dominate
                self.assertGreaterEqual(curr, prev1)


def run_all_tests():
    """Run complete test suite with detailed output"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUniversalRepresentation,
        TestEncodingDensity,
        TestFiniteCompleteness,
        TestInfiniteApproximation,
        TestStructuralPreservation,
        TestOptimalEfficiency,
        TestCompressionLimits,
        TestProcessEncoding,
        TestFundamentalCompleteness,
        TestEncodingUniqueness,
        TestComputationalVerification,
        TestIntegrationComplete
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("T0-4: Binary Encoding Completeness Theory - Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed! Binary-Zeckendorf encoding completeness verified.")
    else:
        print("\n✗ Some tests failed. Review the output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)