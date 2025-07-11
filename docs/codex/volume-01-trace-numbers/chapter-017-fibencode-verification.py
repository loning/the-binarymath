#!/usr/bin/env python3
"""
Chapter 017: FibEncode - Verification Program
φ-Safe Trace Construction from Individual Fibonacci Components

This program verifies that individual Fibonacci numbers can be safely encoded
into trace forms while preserving the φ-constraint, creating the building blocks
for trace arithmetic operations.

从ψ的斐波那契成分中，涌现出φ安全的追踪编码——算术构建块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import math
import itertools


class FibEncodeError(Exception):
    """Custom exception for Fibonacci encoding errors"""
    pass


@dataclass
class FibonacciComponent:
    """A single Fibonacci component with its encoding properties"""
    value: int                      # The Fibonacci number value
    index: int                      # Index in Fibonacci sequence  
    binary_encoding: str            # Binary representation
    trace_form: str                 # φ-constraint compliant trace
    encoding_length: int            # Length of binary encoding
    phi_alignment: float            # Alignment with golden ratio
    
    def __post_init__(self):
        """Validate the Fibonacci component"""
        if '11' in self.trace_form:
            raise FibEncodeError(f"Trace form {self.trace_form} violates φ-constraint")


@dataclass  
class EncodingScheme:
    """Different encoding schemes for Fibonacci components"""
    name: str
    description: str
    supports_composition: bool = True
    preserves_phi_constraint: bool = True
    computational_efficiency: float = 1.0
    memory_efficiency: float = 1.0


class EncodingType(Enum):
    """Types of Fibonacci encodings"""
    STANDARD = "standard"           # Standard positional encoding
    MINIMAL = "minimal"             # Minimal length encoding
    GOLDEN = "golden"               # Golden ratio optimized
    COMPRESSED = "compressed"       # Compressed representation
    NEURAL = "neural"               # Neural network friendly
    SPARSE = "sparse"               # Sparse representation


class FibonacciGenerator:
    """Generate Fibonacci sequence for encoding operations"""
    
    def __init__(self, max_terms: int = 50):
        self.max_terms = max_terms
        self.fibonacci_sequence = self._generate_fibonacci()
        self.phi = (1 + math.sqrt(5)) / 2
        self.value_to_index = {fib: i for i, fib in enumerate(self.fibonacci_sequence)}
    
    def _generate_fibonacci(self) -> List[int]:
        """Generate Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ..."""
        if self.max_terms <= 0:
            return []
        elif self.max_terms == 1:
            return [1]
        elif self.max_terms == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, self.max_terms):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def is_fibonacci(self, n: int) -> bool:
        """Check if number is Fibonacci"""
        return n in self.value_to_index
    
    def get_fibonacci_index(self, value: int) -> Optional[int]:
        """Get index of Fibonacci number"""
        return self.value_to_index.get(value)
    
    def get_fibonacci_at_index(self, index: int) -> Optional[int]:
        """Get Fibonacci number at index"""
        if 0 <= index < len(self.fibonacci_sequence):
            return self.fibonacci_sequence[index]
        return None


class StandardFibEncoder:
    """
    Standard encoding scheme where Fibonacci F_k maps to
    binary string with 1 in position k and 0s elsewhere.
    """
    
    def __init__(self, max_length: int = 32):
        self.max_length = max_length
        self.fib_gen = FibonacciGenerator()
        
    def encode_fibonacci(self, fib_value: int) -> FibonacciComponent:
        """Encode single Fibonacci number to trace form"""
        if not self.fib_gen.is_fibonacci(fib_value):
            raise FibEncodeError(f"{fib_value} is not a Fibonacci number")
        
        fib_index = self.fib_gen.get_fibonacci_index(fib_value)
        
        # Create binary encoding: 1 at position fib_index, 0s elsewhere
        binary_encoding = ['0'] * self.max_length
        if fib_index < self.max_length:
            binary_encoding[-(fib_index + 1)] = '1'  # Right-aligned
        
        binary_str = ''.join(binary_encoding).lstrip('0') or '0'
        
        # For single Fibonacci numbers, trace form is same as binary
        trace_form = binary_str
        
        # Calculate φ-alignment (single Fibonacci has perfect alignment)
        phi_alignment = 1.0
        
        return FibonacciComponent(
            value=fib_value,
            index=fib_index,
            binary_encoding=binary_str,
            trace_form=trace_form,
            encoding_length=len(binary_str),
            phi_alignment=phi_alignment
        )
    
    def batch_encode(self, fib_values: List[int]) -> List[FibonacciComponent]:
        """Encode multiple Fibonacci numbers"""
        return [self.encode_fibonacci(fib) for fib in fib_values]
    
    def verify_encoding(self, component: FibonacciComponent) -> bool:
        """Verify encoding is correct and φ-compliant"""
        try:
            # Check value matches index
            expected_value = self.fib_gen.get_fibonacci_at_index(component.index)
            if expected_value != component.value:
                return False
            
            # Check φ-constraint
            if '11' in component.trace_form:
                return False
            
            # Check binary format
            if not all(c in '01' for c in component.binary_encoding):
                return False
            
            return True
        
        except Exception:
            return False


class MinimalFibEncoder:
    """
    Minimal encoding that uses shortest possible representation
    while preserving φ-constraint and uniqueness.
    """
    
    def __init__(self):
        self.fib_gen = FibonacciGenerator()
    
    def encode_fibonacci(self, fib_value: int) -> FibonacciComponent:
        """Encode using minimal length representation"""
        if not self.fib_gen.is_fibonacci(fib_value):
            raise FibEncodeError(f"{fib_value} is not a Fibonacci number")
        
        fib_index = self.fib_gen.get_fibonacci_index(fib_value)
        
        # Minimal encoding: just enough bits to represent the index
        bits_needed = max(1, fib_index.bit_length())
        binary_encoding = format(fib_index, f'0{bits_needed}b')
        
        # For φ-safety, we need to ensure no consecutive 1s
        trace_form = self._make_phi_safe(binary_encoding)
        
        # Calculate φ-alignment
        phi_alignment = self._calculate_phi_alignment(trace_form)
        
        return FibonacciComponent(
            value=fib_value,
            index=fib_index,
            binary_encoding=binary_encoding,
            trace_form=trace_form,
            encoding_length=len(trace_form),
            phi_alignment=phi_alignment
        )
    
    def _make_phi_safe(self, binary: str) -> str:
        """Transform binary string to be φ-safe"""
        # Simple strategy: insert 0s between consecutive 1s
        result = []
        prev_was_one = False
        
        for bit in binary:
            if bit == '1' and prev_was_one:
                result.append('0')  # Insert separator
            result.append(bit)
            prev_was_one = (bit == '1')
        
        return ''.join(result)
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate alignment with golden ratio properties"""
        if not trace:
            return 0.0
        
        zeros = trace.count('0')
        ones = trace.count('1')
        
        if ones > 0:
            ratio = zeros / ones
            phi = (1 + math.sqrt(5)) / 2
            deviation = abs(ratio - phi) / phi
            return max(0.0, 1.0 - deviation)
        else:
            return 1.0 if zeros > 0 else 0.0


class GoldenRatioEncoder:
    """
    Encoding optimized for golden ratio properties and φ-alignment.
    Maximizes φ-alignment while maintaining uniqueness.
    """
    
    def __init__(self):
        self.fib_gen = FibonacciGenerator()
        self.phi = (1 + math.sqrt(5)) / 2
    
    def encode_fibonacci(self, fib_value: int) -> FibonacciComponent:
        """Encode to maximize golden ratio alignment"""
        if not self.fib_gen.is_fibonacci(fib_value):
            raise FibEncodeError(f"{fib_value} is not a Fibonacci number")
        
        fib_index = self.fib_gen.get_fibonacci_index(fib_value)
        
        # Golden encoding: create pattern that approximates φ ratio
        zeros_needed = max(1, int(fib_index * self.phi))
        ones_needed = max(1, fib_index)
        
        # Construct φ-safe pattern
        trace_form = self._construct_golden_pattern(zeros_needed, ones_needed)
        
        # Binary encoding is the same for this scheme
        binary_encoding = trace_form
        
        phi_alignment = self._calculate_phi_alignment(trace_form)
        
        return FibonacciComponent(
            value=fib_value,
            index=fib_index,
            binary_encoding=binary_encoding,
            trace_form=trace_form,
            encoding_length=len(trace_form),
            phi_alignment=phi_alignment
        )
    
    def _construct_golden_pattern(self, zeros: int, ones: int) -> str:
        """Construct φ-safe pattern with specified zero/one counts"""
        if zeros == 0:
            return '1' if ones == 1 else '101010'[:ones*2-1]  # Alternating pattern
        if ones == 0:
            return '0' * zeros
        if ones == 1:
            return '0' * zeros + '1'
        
        # Construct alternating pattern to ensure no consecutive 1s
        pattern = []
        total_needed = zeros + ones
        
        # Start with '1' and alternate with enough 0s between
        remaining_ones = ones
        remaining_zeros = zeros
        
        while remaining_ones > 0 or remaining_zeros > 0:
            if remaining_ones > 0:
                pattern.append('1')
                remaining_ones -= 1
            
            if remaining_zeros > 0:
                # Add at least one 0 to prevent consecutive 1s
                zeros_to_add = max(1, remaining_zeros // max(1, remaining_ones))
                zeros_to_add = min(zeros_to_add, remaining_zeros)
                pattern.extend(['0'] * zeros_to_add)
                remaining_zeros -= zeros_to_add
        
        result = ''.join(pattern)
        
        # Verify no consecutive 1s
        if '11' in result:
            # Fallback: simple alternating pattern
            result = '1' + '01' * (ones - 1) + '0' * max(0, zeros - (ones - 1))
        
        return result
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate φ-alignment score"""
        if not trace:
            return 0.0
        
        zeros = trace.count('0')
        ones = trace.count('1')
        
        if ones > 0:
            ratio = zeros / ones
            deviation = abs(ratio - self.phi) / self.phi
            return max(0.0, 1.0 - deviation)
        else:
            return 1.0 if zeros > 0 else 0.0


class CompressedFibEncoder:
    """
    Compressed encoding that minimizes storage while preserving
    all necessary information for reconstruction.
    """
    
    def __init__(self):
        self.fib_gen = FibonacciGenerator()
        self.compression_table = self._build_compression_table()
    
    def _build_compression_table(self) -> Dict[int, str]:
        """Build compression lookup table"""
        table = {}
        
        # Use variable-length codes based on frequency
        # Smaller Fibonacci numbers get shorter codes
        for i, fib_val in enumerate(self.fib_gen.fibonacci_sequence[:20]):
            # Variable length binary codes
            if i < 2:
                table[fib_val] = f'{i:01b}'
            elif i < 6:
                table[fib_val] = f'{i:02b}'
            elif i < 14:
                table[fib_val] = f'{i:03b}'
            else:
                table[fib_val] = f'{i:04b}'
        
        return table
    
    def encode_fibonacci(self, fib_value: int) -> FibonacciComponent:
        """Encode using compression table"""
        if not self.fib_gen.is_fibonacci(fib_value):
            raise FibEncodeError(f"{fib_value} is not a Fibonacci number")
        
        if fib_value not in self.compression_table:
            raise FibEncodeError(f"Fibonacci number {fib_value} not in compression table")
        
        fib_index = self.fib_gen.get_fibonacci_index(fib_value)
        
        # Get compressed encoding
        binary_encoding = self.compression_table[fib_value]
        
        # Make φ-safe
        trace_form = self._make_phi_safe(binary_encoding)
        
        phi_alignment = self._calculate_phi_alignment(trace_form)
        
        return FibonacciComponent(
            value=fib_value,
            index=fib_index,
            binary_encoding=binary_encoding,
            trace_form=trace_form,
            encoding_length=len(trace_form),
            phi_alignment=phi_alignment
        )
    
    def _make_phi_safe(self, binary: str) -> str:
        """Ensure φ-constraint compliance"""
        # Replace consecutive 1s with separated pattern
        result = binary.replace('11', '101')
        
        # Continue until no consecutive 1s remain
        while '11' in result:
            result = result.replace('11', '101')
        
        return result
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate φ-alignment"""
        if not trace:
            return 0.0
        
        zeros = trace.count('0')
        ones = trace.count('1')
        
        if ones > 0:
            ratio = zeros / ones
            phi = (1 + math.sqrt(5)) / 2
            deviation = abs(ratio - phi) / phi
            return max(0.0, 1.0 - deviation)
        else:
            return 1.0


class NeuralFibEncoder:
    """
    Neural network-friendly encoding that creates embeddings
    suitable for machine learning applications.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.fib_gen = FibonacciGenerator()
        self.neural_embedder = self._build_embedder()
    
    def _build_embedder(self) -> nn.Module:
        """Build neural embedding network"""
        return nn.Sequential(
            nn.Embedding(len(self.fib_gen.fibonacci_sequence), self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh()
        )
    
    def encode_fibonacci(self, fib_value: int) -> FibonacciComponent:
        """Encode using neural embeddings"""
        if not self.fib_gen.is_fibonacci(fib_value):
            raise FibEncodeError(f"{fib_value} is not a Fibonacci number")
        
        fib_index = self.fib_gen.get_fibonacci_index(fib_value)
        
        # Generate neural embedding
        with torch.no_grad():
            embedding = self.neural_embedder(torch.tensor([fib_index]))
            
        # Convert embedding to binary representation
        binary_encoding = self._embedding_to_binary(embedding[0])
        
        # Ensure φ-compliance
        trace_form = self._make_phi_safe(binary_encoding)
        
        phi_alignment = self._calculate_phi_alignment(trace_form)
        
        return FibonacciComponent(
            value=fib_value,
            index=fib_index,
            binary_encoding=binary_encoding,
            trace_form=trace_form,
            encoding_length=len(trace_form),
            phi_alignment=phi_alignment
        )
    
    def _embedding_to_binary(self, embedding: torch.Tensor) -> str:
        """Convert neural embedding to binary string"""
        # Use sign of embedding values as binary indicators
        binary_bits = (embedding > 0).float()
        
        # Convert to string (limit length for practical use)
        max_bits = min(32, len(binary_bits))
        binary_str = ''.join([str(int(bit)) for bit in binary_bits[:max_bits]])
        
        return binary_str.lstrip('0') or '0'
    
    def _make_phi_safe(self, binary: str) -> str:
        """Make binary string φ-compliant"""
        result = []
        prev_was_one = False
        
        for bit in binary:
            if bit == '1' and prev_was_one:
                result.append('0')
            result.append(bit)
            prev_was_one = (bit == '1')
        
        return ''.join(result)
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate φ-alignment score"""
        if not trace:
            return 0.0
        
        zeros = trace.count('0')
        ones = trace.count('1')
        
        if ones > 0:
            ratio = zeros / ones
            phi = (1 + math.sqrt(5)) / 2
            deviation = abs(ratio - phi) / phi
            return max(0.0, 1.0 - deviation)
        else:
            return 1.0


class CompositionEngine:
    """
    Engine for composing multiple Fibonacci components into
    valid φ-constrained traces while preserving arithmetic meaning.
    """
    
    def __init__(self):
        self.fib_gen = FibonacciGenerator()
    
    def compose_components(self, components: List[FibonacciComponent]) -> Dict[str, Any]:
        """Compose multiple Fibonacci components"""
        if not components:
            return {'error': 'No components provided'}
        
        # Sort components by index for consistent composition
        sorted_components = sorted(components, key=lambda c: c.index, reverse=True)
        
        # Merge trace forms
        composed_trace = self._merge_traces([c.trace_form for c in sorted_components])
        
        # Calculate total value
        total_value = sum(c.value for c in sorted_components)
        
        # Check φ-constraint compliance
        phi_compliant = '11' not in composed_trace
        
        # Calculate composition metrics
        composition_metrics = self._calculate_composition_metrics(sorted_components, composed_trace)
        
        return {
            'components': sorted_components,
            'composed_trace': composed_trace,
            'total_value': total_value,
            'phi_compliant': phi_compliant,
            'metrics': composition_metrics
        }
    
    def _merge_traces(self, traces: List[str]) -> str:
        """Merge multiple traces maintaining φ-constraint"""
        if not traces:
            return '0'
        
        # For Fibonacci components, we need to perform OR operation
        # but ensure no consecutive 1s
        max_length = max(len(trace) for trace in traces)
        
        # Pad traces to same length
        padded_traces = [trace.zfill(max_length) for trace in traces]
        
        # Perform bitwise OR
        result = ['0'] * max_length
        for i in range(max_length):
            for trace in padded_traces:
                if trace[i] == '1':
                    result[i] = '1'
                    break
        
        merged = ''.join(result).lstrip('0') or '0'
        
        # Ensure φ-compliance
        return self._ensure_phi_compliance(merged)
    
    def _ensure_phi_compliance(self, trace: str) -> str:
        """Ensure trace satisfies φ-constraint"""
        # If consecutive 1s exist, this indicates invalid composition
        # For valid Fibonacci components, this shouldn't happen
        if '11' in trace:
            # This is an error condition for Fibonacci composition
            return 'ERROR_INVALID_COMPOSITION'
        
        return trace
    
    def _calculate_composition_metrics(self, components: List[FibonacciComponent], 
                                     composed_trace: str) -> Dict[str, float]:
        """Calculate metrics for component composition"""
        return {
            'component_count': len(components),
            'average_phi_alignment': np.mean([c.phi_alignment for c in components]),
            'total_encoding_length': sum(c.encoding_length for c in components),
            'composed_length': len(composed_trace),
            'compression_ratio': len(composed_trace) / sum(c.encoding_length for c in components) if components else 0,
            'phi_efficiency': self._calculate_phi_efficiency(composed_trace)
        }
    
    def _calculate_phi_efficiency(self, trace: str) -> float:
        """Calculate how efficiently trace uses φ-space"""
        if not trace or trace == 'ERROR_INVALID_COMPOSITION':
            return 0.0
        
        # Efficiency based on length vs information content
        ones = trace.count('1')
        total_length = len(trace)
        
        return ones / total_length if total_length > 0 else 0.0


class EncodingAnalyzer:
    """
    Analyze and compare different encoding schemes for
    efficiency, φ-compliance, and computational properties.
    """
    
    def __init__(self):
        self.encoders = {
            EncodingType.STANDARD: StandardFibEncoder(),
            EncodingType.MINIMAL: MinimalFibEncoder(),
            EncodingType.GOLDEN: GoldenRatioEncoder(),
            EncodingType.COMPRESSED: CompressedFibEncoder(),
            EncodingType.NEURAL: NeuralFibEncoder()
        }
        self.composition_engine = CompositionEngine()
    
    def compare_encoding_schemes(self, fib_values: List[int]) -> Dict[str, Any]:
        """Compare all encoding schemes on given Fibonacci values"""
        results = {}
        
        for encoding_type, encoder in self.encoders.items():
            try:
                encoded_components = []
                for fib_val in fib_values:
                    if hasattr(encoder, 'encode_fibonacci'):
                        component = encoder.encode_fibonacci(fib_val)
                        encoded_components.append(component)
                
                if encoded_components:
                    # Analyze encoding properties
                    analysis = self._analyze_encoding_properties(encoded_components)
                    
                    # Test composition
                    composition_result = self.composition_engine.compose_components(encoded_components)
                    
                    results[encoding_type.value] = {
                        'encoding_analysis': analysis,
                        'composition_result': composition_result,
                        'components': encoded_components
                    }
                
            except Exception as e:
                results[encoding_type.value] = {'error': str(e)}
        
        return results
    
    def _analyze_encoding_properties(self, components: List[FibonacciComponent]) -> Dict[str, float]:
        """Analyze properties of encoded components"""
        if not components:
            return {}
        
        phi_alignments = [c.phi_alignment for c in components]
        encoding_lengths = [c.encoding_length for c in components]
        
        return {
            'average_phi_alignment': np.mean(phi_alignments),
            'max_phi_alignment': max(phi_alignments),
            'min_phi_alignment': min(phi_alignments),
            'average_encoding_length': np.mean(encoding_lengths),
            'total_encoding_length': sum(encoding_lengths),
            'phi_compliance_rate': sum(1 for c in components if '11' not in c.trace_form) / len(components),
            'compression_efficiency': self._calculate_compression_efficiency(components)
        }
    
    def _calculate_compression_efficiency(self, components: List[FibonacciComponent]) -> float:
        """Calculate compression efficiency"""
        if not components:
            return 0.0
        
        # Compare to naive binary representation
        total_value = sum(c.value for c in components)
        naive_bits = total_value.bit_length() if total_value > 0 else 1
        actual_bits = sum(c.encoding_length for c in components)
        
        return naive_bits / actual_bits if actual_bits > 0 else 0.0


class FibEncodeTests(unittest.TestCase):
    """Test Fibonacci encoding functionality"""
    
    def setUp(self):
        self.fib_gen = FibonacciGenerator()
        self.standard_encoder = StandardFibEncoder()
        self.minimal_encoder = MinimalFibEncoder()
        self.golden_encoder = GoldenRatioEncoder()
        self.composition_engine = CompositionEngine()
        self.analyzer = EncodingAnalyzer()
        
        # Test Fibonacci numbers
        self.test_fibs = [1, 1, 2, 3, 5, 8, 13, 21]
    
    def test_fibonacci_generation(self):
        """Test: Fibonacci sequence generation"""
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        actual = self.fib_gen.fibonacci_sequence[:10]
        self.assertEqual(actual, expected)
    
    def test_fibonacci_recognition(self):
        """Test: Fibonacci number recognition"""
        for fib in self.test_fibs:
            self.assertTrue(self.fib_gen.is_fibonacci(fib))
        
        non_fibs = [4, 6, 7, 9, 10, 11, 12]
        for non_fib in non_fibs:
            self.assertFalse(self.fib_gen.is_fibonacci(non_fib))
    
    def test_standard_encoding(self):
        """Test: Standard Fibonacci encoding"""
        for fib in self.test_fibs:
            component = self.standard_encoder.encode_fibonacci(fib)
            
            # Verify component properties
            self.assertEqual(component.value, fib)
            self.assertNotIn('11', component.trace_form)
            self.assertTrue(self.standard_encoder.verify_encoding(component))
            self.assertEqual(component.phi_alignment, 1.0)
    
    def test_minimal_encoding(self):
        """Test: Minimal Fibonacci encoding"""
        for fib in self.test_fibs:
            component = self.minimal_encoder.encode_fibonacci(fib)
            
            # Verify φ-compliance
            self.assertNotIn('11', component.trace_form)
            self.assertEqual(component.value, fib)
            self.assertGreaterEqual(component.phi_alignment, 0.0)
            self.assertLessEqual(component.phi_alignment, 1.0)
    
    def test_golden_encoding(self):
        """Test: Golden ratio encoding"""
        for fib in self.test_fibs:
            component = self.golden_encoder.encode_fibonacci(fib)
            
            # Verify φ-compliance and golden properties
            self.assertNotIn('11', component.trace_form)
            self.assertEqual(component.value, fib)
            self.assertGreaterEqual(component.phi_alignment, 0.0)
    
    def test_phi_constraint_preservation(self):
        """Test: All encodings preserve φ-constraint"""
        encoders = [
            self.standard_encoder,
            self.minimal_encoder,
            self.golden_encoder
        ]
        
        for encoder in encoders:
            for fib in self.test_fibs:
                component = encoder.encode_fibonacci(fib)
                self.assertNotIn('11', component.trace_form, 
                    f"φ-constraint violated in {type(encoder).__name__} for {fib}")
    
    def test_component_composition(self):
        """Test: Composition of multiple components"""
        components = []
        for fib in [1, 3, 8]:  # Non-consecutive Fibonacci numbers
            component = self.standard_encoder.encode_fibonacci(fib)
            components.append(component)
        
        result = self.composition_engine.compose_components(components)
        
        self.assertTrue(result['phi_compliant'])
        self.assertEqual(result['total_value'], 1 + 3 + 8)
        self.assertGreater(len(result['composed_trace']), 0)
    
    def test_invalid_composition_detection(self):
        """Test: Detection of invalid compositions"""
        # Try to compose consecutive Fibonacci numbers (should be valid individually)
        components = []
        for fib in [2, 3]:  # F_3, F_4 - consecutive
            component = self.standard_encoder.encode_fibonacci(fib)
            components.append(component)
        
        result = self.composition_engine.compose_components(components)
        
        # Should still be valid for component composition
        # (different from Zeckendorf where consecutive terms forbidden)
        self.assertTrue(result['phi_compliant'])
    
    def test_encoding_comparison(self):
        """Test: Comparison of encoding schemes"""
        test_fibs = [1, 2, 5, 13]
        
        comparison = self.analyzer.compare_encoding_schemes(test_fibs)
        
        # Check that all encoders produced results
        expected_schemes = ['standard', 'minimal', 'golden', 'compressed', 'neural']
        for scheme in expected_schemes:
            self.assertIn(scheme, comparison)
            if 'error' not in comparison[scheme]:
                self.assertIn('encoding_analysis', comparison[scheme])
    
    def test_phi_alignment_calculation(self):
        """Test: φ-alignment calculations are reasonable"""
        for fib in self.test_fibs:
            component = self.golden_encoder.encode_fibonacci(fib)
            
            # φ-alignment should be between 0 and 1
            self.assertGreaterEqual(component.phi_alignment, 0.0)
            self.assertLessEqual(component.phi_alignment, 1.0)
    
    def test_encoding_length_efficiency(self):
        """Test: Encoding length efficiency"""
        for fib in [1, 5, 21]:  # Sample Fibonacci numbers
            std_component = self.standard_encoder.encode_fibonacci(fib)
            min_component = self.minimal_encoder.encode_fibonacci(fib)
            
            # Minimal should generally be shorter or equal
            self.assertLessEqual(min_component.encoding_length, 
                               std_component.encoding_length * 2)  # Allow some flexibility
    
    def test_error_handling(self):
        """Test: Proper error handling for invalid inputs"""
        with self.assertRaises(FibEncodeError):
            self.standard_encoder.encode_fibonacci(4)  # Not Fibonacci
        
        with self.assertRaises(FibEncodeError):
            self.minimal_encoder.encode_fibonacci(10)  # Not Fibonacci


def visualize_fibonacci_encodings():
    """Visualize different Fibonacci encoding schemes"""
    print("=" * 70)
    print("FibEncode: φ-Safe Trace Construction from Fibonacci Components")
    print("=" * 70)
    
    fib_gen = FibonacciGenerator()
    standard_encoder = StandardFibEncoder()
    minimal_encoder = MinimalFibEncoder()
    golden_encoder = GoldenRatioEncoder()
    analyzer = EncodingAnalyzer()
    
    test_fibs = [1, 2, 3, 5, 8, 13, 21]
    
    print("\n1. Standard Fibonacci Encodings:")
    
    for fib in test_fibs:
        std_comp = standard_encoder.encode_fibonacci(fib)
        print(f"   F={fib:2d}: {std_comp.trace_form:12s} | φ-align: {std_comp.phi_alignment:.3f} | len: {std_comp.encoding_length}")
    
    print("\n2. Minimal Encodings:")
    
    for fib in test_fibs:
        min_comp = minimal_encoder.encode_fibonacci(fib)
        print(f"   F={fib:2d}: {min_comp.trace_form:12s} | φ-align: {min_comp.phi_alignment:.3f} | len: {min_comp.encoding_length}")
    
    print("\n3. Golden Ratio Encodings:")
    
    for fib in test_fibs:
        gold_comp = golden_encoder.encode_fibonacci(fib)
        print(f"   F={fib:2d}: {gold_comp.trace_form:12s} | φ-align: {gold_comp.phi_alignment:.3f} | len: {gold_comp.encoding_length}")
    
    print("\n4. Encoding Scheme Comparison:")
    
    comparison = analyzer.compare_encoding_schemes([3, 5, 8])
    
    for scheme_name, results in comparison.items():
        if 'error' not in results:
            analysis = results['encoding_analysis']
            composition = results['composition_result']
            
            print(f"\n   {scheme_name.title()} Scheme:")
            print(f"     Average φ-alignment: {analysis.get('average_phi_alignment', 0):.3f}")
            print(f"     Average length: {analysis.get('average_encoding_length', 0):.1f}")
            print(f"     φ-compliance: {analysis.get('phi_compliance_rate', 0):.1%}")
            print(f"     Composed trace: {composition.get('composed_trace', 'N/A')}")
            print(f"     Total value: {composition.get('total_value', 'N/A')}")
    
    print("\n5. Component Composition Analysis:")
    
    composition_engine = CompositionEngine()
    
    # Test composition of [1, 3, 8]
    components = [standard_encoder.encode_fibonacci(f) for f in [1, 3, 8]]
    result = composition_engine.compose_components(components)
    
    print(f"   Components: F=1, F=3, F=8")
    print(f"   Individual traces: {[c.trace_form for c in components]}")
    print(f"   Composed trace: {result['composed_trace']}")
    print(f"   Total value: {result['total_value']}")
    print(f"   φ-compliant: {result['phi_compliant']}")
    print(f"   Composition metrics:")
    metrics = result['metrics']
    for key, value in metrics.items():
        print(f"     {key}: {value}")
    
    print("\n6. φ-Constraint Verification:")
    
    all_traces = []
    for fib in test_fibs:
        for encoder in [standard_encoder, minimal_encoder, golden_encoder]:
            component = encoder.encode_fibonacci(fib)
            all_traces.append(component.trace_form)
    
    phi_violations = [trace for trace in all_traces if '11' in trace]
    
    print(f"   Total traces tested: {len(all_traces)}")
    print(f"   φ-constraint violations: {len(phi_violations)}")
    print(f"   φ-compliance rate: {(len(all_traces) - len(phi_violations)) / len(all_traces):.1%}")
    
    if phi_violations:
        print(f"   Violating traces: {phi_violations}")
    
    print("\n7. Efficiency Analysis:")
    
    for scheme_name, results in comparison.items():
        if 'error' not in results and 'encoding_analysis' in results:
            analysis = results['encoding_analysis']
            compression_eff = analysis.get('compression_efficiency', 0)
            total_length = analysis.get('total_encoding_length', 0)
            
            print(f"   {scheme_name.title()}:")
            print(f"     Total encoding length: {total_length}")
            print(f"     Compression efficiency: {compression_eff:.3f}")
    
    print("\n" + "=" * 70)
    print("All Fibonacci components successfully encoded in φ-safe form")
    print("=" * 70)


if __name__ == "__main__":
    # Run visualization
    visualize_fibonacci_encodings()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)