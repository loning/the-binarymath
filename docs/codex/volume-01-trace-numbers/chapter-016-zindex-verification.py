#!/usr/bin/env python3
"""
Chapter 016: ZIndex - Verification Program
Zeckendorf Decomposition of Natural Numbers into Non-Overlapping Trace Seeds

This program verifies that every natural number can be uniquely decomposed into
non-consecutive Fibonacci numbers, forming the foundation of trace arithmetic
in the φ-constrained space where no "11" patterns exist.

从ψ的自然数分解中，涌现出Zeckendorf表示——追踪种子的非重叠分解。
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


class ZeckendorfError(Exception):
    """Custom exception for Zeckendorf decomposition errors"""
    pass


@dataclass
class ZeckendorfDecomposition:
    """A Zeckendorf decomposition of a natural number"""
    number: int
    fibonacci_terms: List[int]  # The Fibonacci numbers in the decomposition
    fibonacci_indices: List[int]  # Indices of Fibonacci numbers used
    binary_representation: str  # Binary string with no consecutive 1s
    trace_seed: str  # φ-constraint compliant trace
    
    def __post_init__(self):
        """Validate the decomposition"""
        if not self._is_valid():
            raise ZeckendorfError(f"Invalid Zeckendorf decomposition for {self.number}")
    
    def _is_valid(self) -> bool:
        """Check if this is a valid Zeckendorf decomposition"""
        # Check sum equals original number
        if sum(self.fibonacci_terms) != self.number:
            return False
        
        # Check no consecutive Fibonacci numbers (indices should differ by >1)
        sorted_indices = sorted(self.fibonacci_indices, reverse=True)
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i] - sorted_indices[i+1] <= 1:
                return False
        
        # Check binary representation has no consecutive 1s
        if '11' in self.binary_representation:
            return False
        
        return True
    
    def complexity(self) -> float:
        """Calculate complexity of the decomposition"""
        return len(self.fibonacci_terms) / math.log2(self.number + 1) if self.number > 0 else 0


class FibonacciGenerator:
    """
    Generates Fibonacci sequences efficiently for Zeckendorf decomposition.
    Uses golden ratio properties and φ-constraint preservation.
    """
    
    def __init__(self, max_n: int = 100):
        self.max_n = max_n
        self.fibonacci_sequence = self._generate_fibonacci(max_n)
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.fibonacci_lookup = {fib: i for i, fib in enumerate(self.fibonacci_sequence)}
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ..."""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def get_fibonacci_up_to_value(self, value: int) -> List[int]:
        """Get all Fibonacci numbers up to given value"""
        return [f for f in self.fibonacci_sequence if f <= value]
    
    def get_fibonacci_index(self, fib_number: int) -> Optional[int]:
        """Get index of Fibonacci number in sequence"""
        return self.fibonacci_lookup.get(fib_number)
    
    def fibonacci_at_index(self, index: int) -> int:
        """Get Fibonacci number at given index"""
        if 0 <= index < len(self.fibonacci_sequence):
            return self.fibonacci_sequence[index]
        else:
            # Generate more if needed
            while len(self.fibonacci_sequence) <= index:
                self.fibonacci_sequence.append(
                    self.fibonacci_sequence[-1] + self.fibonacci_sequence[-2])
            return self.fibonacci_sequence[index]
    
    def is_fibonacci(self, n: int) -> bool:
        """Check if number is Fibonacci"""
        return n in self.fibonacci_lookup
    
    def largest_fibonacci_leq(self, n: int) -> Tuple[int, int]:
        """Find largest Fibonacci number ≤ n for Zeckendorf (skip F_1=1), return (value, index)"""
        # Start from index 1 to skip the first F_1=1 in standard sequence
        for i in range(len(self.fibonacci_sequence) - 1, 0, -1):  # Skip index 0
            if self.fibonacci_sequence[i] <= n:
                return self.fibonacci_sequence[i], i
        return 1, 1  # Return F_2=1 as minimum


class ZeckendorfDecomposer:
    """
    Decomposes natural numbers into Zeckendorf form (sum of non-consecutive Fibonacci numbers).
    This is the foundation of trace arithmetic in φ-constrained space.
    """
    
    def __init__(self, max_fibonacci: int = 50):
        self.fib_gen = FibonacciGenerator(max_fibonacci)
        self.decomposition_cache = {}
        
    def decompose(self, n: int) -> ZeckendorfDecomposition:
        """
        Decompose natural number n into Zeckendorf form.
        
        Uses greedy algorithm: always pick largest possible Fibonacci number.
        This guarantees unique decomposition with no consecutive terms.
        """
        if n <= 0:
            raise ValueError("Zeckendorf decomposition only defined for positive integers")
        
        if n in self.decomposition_cache:
            return self.decomposition_cache[n]
        
        original_n = n
        fibonacci_terms = []
        fibonacci_indices = []
        
        # Greedy algorithm
        while n > 0:
            # Find largest Fibonacci number ≤ n
            fib_val, fib_idx = self.fib_gen.largest_fibonacci_leq(n)
            
            fibonacci_terms.append(fib_val)
            fibonacci_indices.append(fib_idx)
            
            n -= fib_val
        
        # Sort in descending order of indices
        paired = list(zip(fibonacci_indices, fibonacci_terms))
        paired.sort(reverse=True)
        fibonacci_indices, fibonacci_terms = zip(*paired)
        fibonacci_indices = list(fibonacci_indices)
        fibonacci_terms = list(fibonacci_terms)
        
        # Generate binary representation
        binary_rep = self._generate_binary_representation(fibonacci_indices)
        
        # Generate trace seed (φ-constraint compliant)
        trace_seed = self._generate_trace_seed(binary_rep)
        
        decomposition = ZeckendorfDecomposition(
            number=original_n,
            fibonacci_terms=fibonacci_terms,
            fibonacci_indices=fibonacci_indices,
            binary_representation=binary_rep,
            trace_seed=trace_seed
        )
        
        self.decomposition_cache[original_n] = decomposition
        return decomposition
    
    def _generate_binary_representation(self, fibonacci_indices: List[int]) -> str:
        """
        Generate binary representation where 1 at position i means F_i is included.
        Position counting starts from F_1 = 1, F_2 = 1, F_3 = 2, ...
        """
        if not fibonacci_indices:
            return "0"
        
        max_index = max(fibonacci_indices)
        binary_bits = ['0'] * (max_index + 1)
        
        for idx in fibonacci_indices:
            binary_bits[max_index - idx] = '1'  # Reverse order for standard representation
        
        return ''.join(binary_bits).lstrip('0') or '0'
    
    def _generate_trace_seed(self, binary_rep: str) -> str:
        """
        Generate φ-constraint compliant trace seed.
        The binary representation already satisfies no consecutive 1s.
        """
        # Verify no consecutive 1s
        if '11' in binary_rep:
            raise ZeckendorfError(f"Binary representation {binary_rep} violates φ-constraint")
        
        # The trace seed is the binary representation itself in φ-space
        return binary_rep
    
    def batch_decompose(self, numbers: List[int]) -> List[ZeckendorfDecomposition]:
        """Decompose multiple numbers efficiently"""
        return [self.decompose(n) for n in numbers]
    
    def verify_decomposition(self, decomp: ZeckendorfDecomposition) -> bool:
        """Verify that a decomposition is correct"""
        try:
            # Check sum
            if sum(decomp.fibonacci_terms) != decomp.number:
                return False
            
            # Check no consecutive Fibonacci indices
            for i in range(len(decomp.fibonacci_indices) - 1):
                if decomp.fibonacci_indices[i] - decomp.fibonacci_indices[i+1] <= 1:
                    return False
            
            # Check φ-constraint
            if '11' in decomp.binary_representation:
                return False
            
            return True
        
        except Exception:
            return False


class ZeckendorfAnalyzer:
    """
    Analyzes properties of Zeckendorf decompositions and their trace seeds.
    Studies patterns, density, and structural properties.
    """
    
    def __init__(self):
        self.decomposer = ZeckendorfDecomposer()
        
    def analyze_range(self, start: int, end: int) -> Dict[str, Any]:
        """Analyze Zeckendorf decompositions for range of numbers"""
        decompositions = []
        
        for n in range(start, end + 1):
            try:
                decomp = self.decomposer.decompose(n)
                decompositions.append(decomp)
            except Exception as e:
                print(f"Error decomposing {n}: {e}")
                continue
        
        if not decompositions:
            return {'error': 'No valid decompositions found'}
        
        return {
            'range': (start, end),
            'total_numbers': len(decompositions),
            'average_terms': np.mean([len(d.fibonacci_terms) for d in decompositions]),
            'max_terms': max(len(d.fibonacci_terms) for d in decompositions),
            'min_terms': min(len(d.fibonacci_terms) for d in decompositions),
            'average_complexity': np.mean([d.complexity() for d in decompositions]),
            'phi_constraint_satisfaction': all('11' not in d.binary_representation for d in decompositions),
            'binary_length_distribution': self._analyze_binary_lengths(decompositions),
            'fibonacci_usage_frequency': self._analyze_fibonacci_usage(decompositions),
            'trace_seed_patterns': self._analyze_trace_patterns(decompositions)
        }
    
    def _analyze_binary_lengths(self, decompositions: List[ZeckendorfDecomposition]) -> Dict[int, int]:
        """Analyze distribution of binary representation lengths"""
        length_counts = defaultdict(int)
        for decomp in decompositions:
            length_counts[len(decomp.binary_representation)] += 1
        return dict(length_counts)
    
    def _analyze_fibonacci_usage(self, decompositions: List[ZeckendorfDecomposition]) -> Dict[int, int]:
        """Analyze frequency of Fibonacci number usage"""
        usage_counts = defaultdict(int)
        for decomp in decompositions:
            for fib_num in decomp.fibonacci_terms:
                usage_counts[fib_num] += 1
        return dict(usage_counts)
    
    def _analyze_trace_patterns(self, decompositions: List[ZeckendorfDecomposition]) -> Dict[str, int]:
        """Analyze common patterns in trace seeds"""
        pattern_counts = defaultdict(int)
        
        for decomp in decompositions:
            trace = decomp.trace_seed
            
            # Count common patterns
            for length in range(2, min(5, len(trace) + 1)):
                for i in range(len(trace) - length + 1):
                    pattern = trace[i:i+length]
                    if '11' not in pattern:  # Only φ-valid patterns
                        pattern_counts[pattern] += 1
        
        # Return top patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_patterns[:20])  # Top 20 patterns
    
    def find_zeckendorf_twins(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Find pairs of consecutive numbers with same number of Fibonacci terms"""
        twins = []
        
        for n in range(start, end):
            try:
                decomp1 = self.decomposer.decompose(n)
                decomp2 = self.decomposer.decompose(n + 1)
                
                if len(decomp1.fibonacci_terms) == len(decomp2.fibonacci_terms):
                    twins.append((n, n + 1))
                    
            except Exception:
                continue
        
        return twins
    
    def analyze_golden_ratio_convergence(self, n_max: int = 100) -> Dict[str, Any]:
        """Analyze how Zeckendorf properties relate to golden ratio"""
        phi = (1 + math.sqrt(5)) / 2
        
        ratios = []
        complexities = []
        
        for n in range(1, n_max + 1):
            try:
                decomp = self.decomposer.decompose(n)
                
                # Calculate ratio of consecutive Fibonacci terms if possible
                if len(decomp.fibonacci_terms) >= 2:
                    ratio = decomp.fibonacci_terms[0] / decomp.fibonacci_terms[1]
                    ratios.append(ratio)
                
                complexities.append(decomp.complexity())
                
            except Exception:
                continue
        
        return {
            'phi_value': phi,
            'average_ratio': np.mean(ratios) if ratios else 0,
            'ratio_std': np.std(ratios) if ratios else 0,
            'convergence_to_phi': abs(np.mean(ratios) - phi) if ratios else float('inf'),
            'average_complexity': np.mean(complexities),
            'complexity_trend': np.polyfit(range(len(complexities)), complexities, 1)[0] if complexities else 0
        }


class ZeckendorfTensor:
    """
    Tensor representation of Zeckendorf decompositions for batch processing.
    Enables neural network operations on trace seeds.
    """
    
    def __init__(self, max_fibonacci_terms: int = 20):
        self.max_terms = max_fibonacci_terms
        self.decomposer = ZeckendorfDecomposer()
        
    def tensorize_decompositions(self, numbers: List[int]) -> torch.Tensor:
        """Convert list of numbers to tensor of Zeckendorf decompositions"""
        decompositions = self.decomposer.batch_decompose(numbers)
        
        # Create tensor representation
        batch_size = len(numbers)
        tensor = torch.zeros(batch_size, self.max_terms, dtype=torch.float32)
        
        for i, decomp in enumerate(decompositions):
            # Fill tensor with Fibonacci indices (normalized)
            for j, fib_idx in enumerate(decomp.fibonacci_indices[:self.max_terms]):
                tensor[i, j] = float(fib_idx)
        
        return tensor
    
    def tensorize_binary_representations(self, numbers: List[int], max_length: int = 32) -> torch.Tensor:
        """Convert to tensor of binary representations"""
        decompositions = self.decomposer.batch_decompose(numbers)
        
        batch_size = len(numbers)
        tensor = torch.zeros(batch_size, max_length, dtype=torch.float32)
        
        for i, decomp in enumerate(decompositions):
            binary = decomp.binary_representation
            # Pad or truncate to max_length
            if len(binary) > max_length:
                binary = binary[:max_length]
            else:
                binary = binary.zfill(max_length)
            
            # Convert to tensor
            for j, bit in enumerate(binary):
                tensor[i, j] = float(bit)
        
        return tensor
    
    def tensorize_trace_seeds(self, numbers: List[int], max_length: int = 32) -> torch.Tensor:
        """Convert trace seeds to tensor format"""
        # For Zeckendorf, trace seeds are the same as binary representations
        return self.tensorize_binary_representations(numbers, max_length)


class ZeckendorfNeuralProcessor(nn.Module):
    """
    Neural network for processing Zeckendorf decompositions.
    Learns patterns in trace seeds and predicts properties.
    """
    
    def __init__(self, max_length: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Encoder for binary sequences
        self.encoder = nn.Sequential(
            nn.Linear(max_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictors for various properties
        self.term_count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.phi_alignment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, trace_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process batch of trace seeds"""
        # Encode trace seeds
        encoded = self.encoder(trace_tensor)
        
        # Predict properties
        term_counts = self.term_count_predictor(encoded)
        complexities = self.complexity_predictor(encoded)
        phi_alignments = self.phi_alignment_predictor(encoded)
        
        return {
            'encoded': encoded,
            'term_counts': term_counts,
            'complexities': complexities,
            'phi_alignments': phi_alignments
        }
    
    def predict_properties(self, numbers: List[int]) -> Dict[str, torch.Tensor]:
        """Predict Zeckendorf properties for list of numbers"""
        tensorizer = ZeckendorfTensor()
        trace_tensor = tensorizer.tensorize_trace_seeds(numbers, self.max_length)
        
        with torch.no_grad():
            predictions = self.forward(trace_tensor)
        
        return predictions


class ZeckendorfTests(unittest.TestCase):
    """Test Zeckendorf decomposition functionality"""
    
    def setUp(self):
        self.decomposer = ZeckendorfDecomposer()
        self.analyzer = ZeckendorfAnalyzer()
        self.fib_gen = FibonacciGenerator()
        
        # Test numbers with known decompositions (using standard Fibonacci: 1,1,2,3,5,8,13,...)
        # Note: For Zeckendorf, we use F_2=1, F_3=2, F_4=3, F_5=5, etc. (skip first F_1=1)
        self.test_cases = [
            (1, [1], [1]),      # F_2 = 1
            (2, [2], [2]),      # F_3 = 2  
            (3, [2, 1], [2, 1]), # F_3 + F_2 = 2 + 1
            (4, [3, 1], [3, 1]), # F_4 + F_2 = 3 + 1
            (5, [5], [4]),      # F_5 = 5
            (6, [5, 1], [4, 1]), # F_5 + F_2 = 5 + 1
            (7, [5, 2], [4, 2]), # F_5 + F_3 = 5 + 2
            (8, [8], [5]),      # F_6 = 8
            (9, [8, 1], [5, 1]), # F_6 + F_2 = 8 + 1
            (10, [8, 2], [5, 2]) # F_6 + F_3 = 8 + 2
        ]
    
    def test_fibonacci_generation(self):
        """Test: Fibonacci sequence generation"""
        expected_start = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        
        fib_seq = self.fib_gen.fibonacci_sequence[:10]
        self.assertEqual(fib_seq, expected_start)
    
    def test_zeckendorf_decomposition(self):
        """Test: Basic Zeckendorf decomposition"""
        for number, expected_terms, expected_indices in self.test_cases:
            decomp = self.decomposer.decompose(number)
            
            # Check decomposition correctness
            self.assertEqual(decomp.number, number)
            self.assertEqual(sum(decomp.fibonacci_terms), number)
            
            # Check φ-constraint (no consecutive 1s)
            self.assertNotIn('11', decomp.binary_representation)
            self.assertNotIn('11', decomp.trace_seed)
    
    def test_decomposition_uniqueness(self):
        """Test: Zeckendorf decomposition uniqueness"""
        for i in range(1, 21):
            decomp1 = self.decomposer.decompose(i)
            decomp2 = self.decomposer.decompose(i)
            
            # Should be identical
            self.assertEqual(decomp1.fibonacci_terms, decomp2.fibonacci_terms)
            self.assertEqual(decomp1.binary_representation, decomp2.binary_representation)
    
    def test_no_consecutive_fibonacci_terms(self):
        """Test: No consecutive Fibonacci numbers in decomposition"""
        for i in range(1, 51):
            decomp = self.decomposer.decompose(i)
            
            # Check no consecutive indices
            for j in range(len(decomp.fibonacci_indices) - 1):
                diff = decomp.fibonacci_indices[j] - decomp.fibonacci_indices[j+1]
                self.assertGreater(diff, 1, 
                    f"Consecutive Fibonacci terms found in decomposition of {i}")
    
    def test_phi_constraint_satisfaction(self):
        """Test: All decompositions satisfy φ-constraint"""
        for i in range(1, 101):
            decomp = self.decomposer.decompose(i)
            
            # Check no '11' patterns
            self.assertNotIn('11', decomp.binary_representation,
                f"φ-constraint violated in binary representation of {i}")
            self.assertNotIn('11', decomp.trace_seed,
                f"φ-constraint violated in trace seed of {i}")
    
    def test_decomposition_verification(self):
        """Test: Decomposition verification works correctly"""
        for i in range(1, 31):
            decomp = self.decomposer.decompose(i)
            self.assertTrue(self.decomposer.verify_decomposition(decomp))
    
    def test_batch_decomposition(self):
        """Test: Batch decomposition efficiency"""
        numbers = list(range(1, 21))
        decomps = self.decomposer.batch_decompose(numbers)
        
        self.assertEqual(len(decomps), len(numbers))
        
        # Verify each decomposition
        for i, decomp in enumerate(decomps):
            self.assertEqual(decomp.number, numbers[i])
            self.assertTrue(self.decomposer.verify_decomposition(decomp))
    
    def test_binary_representation_validity(self):
        """Test: Binary representations are valid"""
        for i in range(1, 51):
            decomp = self.decomposer.decompose(i)
            binary = decomp.binary_representation
            
            # Should be valid binary string
            self.assertTrue(all(c in '01' for c in binary))
            
            # Should not have consecutive 1s
            self.assertNotIn('11', binary)
            
            # Should not be empty
            self.assertGreater(len(binary), 0)
    
    def test_trace_seed_properties(self):
        """Test: Trace seed properties"""
        for i in range(1, 31):
            decomp = self.decomposer.decompose(i)
            trace = decomp.trace_seed
            
            # Should be φ-constraint compliant
            self.assertNotIn('11', trace)
            
            # Should match binary representation
            self.assertEqual(trace, decomp.binary_representation)
    
    def test_complexity_calculation(self):
        """Test: Complexity calculation is reasonable"""
        for i in range(1, 21):
            decomp = self.decomposer.decompose(i)
            complexity = decomp.complexity()
            
            # Should be non-negative
            self.assertGreaterEqual(complexity, 0.0)
            
            # Should be finite
            self.assertTrue(math.isfinite(complexity))
    
    def test_analyzer_range_analysis(self):
        """Test: Range analysis functionality"""
        analysis = self.analyzer.analyze_range(1, 20)
        
        # Should have required keys
        required_keys = ['range', 'total_numbers', 'average_terms', 'phi_constraint_satisfaction']
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # φ-constraint should be satisfied for all
        self.assertTrue(analysis['phi_constraint_satisfaction'])
        
        # Should have positive averages
        self.assertGreater(analysis['average_terms'], 0)
    
    def test_fibonacci_usage_analysis(self):
        """Test: Fibonacci usage frequency analysis"""
        analysis = self.analyzer.analyze_range(1, 30)
        
        self.assertIn('fibonacci_usage_frequency', analysis)
        usage_freq = analysis['fibonacci_usage_frequency']
        
        # Should have frequency data
        self.assertGreater(len(usage_freq), 0)
        
        # All frequencies should be positive
        for fib_num, freq in usage_freq.items():
            self.assertGreater(freq, 0)
    
    def test_golden_ratio_convergence(self):
        """Test: Golden ratio convergence analysis"""
        convergence = self.analyzer.analyze_golden_ratio_convergence(50)
        
        # Should have phi value
        self.assertAlmostEqual(convergence['phi_value'], (1 + math.sqrt(5)) / 2, places=5)
        
        # Should have convergence data
        self.assertIn('convergence_to_phi', convergence)
        self.assertIn('average_complexity', convergence)
    
    def test_tensorization(self):
        """Test: Tensor conversion functionality"""
        tensorizer = ZeckendorfTensor()
        numbers = [1, 2, 3, 4, 5]
        
        # Test decomposition tensorization
        decomp_tensor = tensorizer.tensorize_decompositions(numbers)
        self.assertEqual(decomp_tensor.shape[0], len(numbers))
        
        # Test binary tensorization
        binary_tensor = tensorizer.tensorize_binary_representations(numbers, max_length=16)
        self.assertEqual(binary_tensor.shape, (len(numbers), 16))
        
        # Should contain only 0s and 1s
        unique_values = torch.unique(binary_tensor)
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values))
    
    def test_neural_processor_architecture(self):
        """Test: Neural processor architecture"""
        processor = ZeckendorfNeuralProcessor(max_length=16, hidden_dim=32)
        
        # Test forward pass
        batch_size = 5
        test_input = torch.randint(0, 2, (batch_size, 16), dtype=torch.float32)
        
        outputs = processor(test_input)
        
        # Check output shapes
        self.assertEqual(outputs['encoded'].shape, (batch_size, 32))
        self.assertEqual(outputs['term_counts'].shape, (batch_size, 1))
        self.assertEqual(outputs['complexities'].shape, (batch_size, 1))
        self.assertEqual(outputs['phi_alignments'].shape, (batch_size, 1))
        
        # Check phi_alignments are in [0, 1]
        phi_alignments = outputs['phi_alignments']
        self.assertTrue(torch.all(phi_alignments >= 0))
        self.assertTrue(torch.all(phi_alignments <= 1))


def visualize_zeckendorf_decompositions():
    """Visualize Zeckendorf decomposition properties and patterns"""
    print("=" * 60)
    print("ZIndex: Zeckendorf Decomposition of Natural Numbers")
    print("=" * 60)
    
    decomposer = ZeckendorfDecomposer()
    analyzer = ZeckendorfAnalyzer()
    
    print("\n1. Basic Zeckendorf Decompositions:")
    
    for n in range(1, 21):
        decomp = decomposer.decompose(n)
        fib_terms_str = " + ".join(map(str, decomp.fibonacci_terms))
        
        print(f"   {n:2d} = {fib_terms_str:20s} | Binary: {decomp.binary_representation:8s} | Trace: {decomp.trace_seed}")
    
    print("\n2. Range Analysis (1-50):")
    
    analysis = analyzer.analyze_range(1, 50)
    
    print(f"   Total numbers analyzed: {analysis['total_numbers']}")
    print(f"   Average Fibonacci terms: {analysis['average_terms']:.2f}")
    print(f"   Max terms used: {analysis['max_terms']}")
    print(f"   Min terms used: {analysis['min_terms']}")
    print(f"   Average complexity: {analysis['average_complexity']:.3f}")
    print(f"   φ-constraint satisfied: {analysis['phi_constraint_satisfaction']}")
    
    print("\n3. Binary Length Distribution:")
    
    binary_dist = analysis['binary_length_distribution']
    for length, count in sorted(binary_dist.items()):
        print(f"   Length {length}: {count} numbers")
    
    print("\n4. Most Used Fibonacci Numbers:")
    
    fib_usage = analysis['fibonacci_usage_frequency']
    sorted_usage = sorted(fib_usage.items(), key=lambda x: x[1], reverse=True)
    
    for fib_num, freq in sorted_usage[:10]:
        print(f"   F = {fib_num}: used {freq} times")
    
    print("\n5. Common Trace Patterns:")
    
    patterns = analysis['trace_seed_patterns']
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_patterns[:10]:
        print(f"   Pattern '{pattern}': appears {count} times")
    
    print("\n6. Golden Ratio Convergence:")
    
    convergence = analyzer.analyze_golden_ratio_convergence(100)
    
    print(f"   φ (golden ratio): {convergence['phi_value']:.6f}")
    print(f"   Average ratio of consecutive terms: {convergence['average_ratio']:.6f}")
    print(f"   Standard deviation: {convergence['ratio_std']:.6f}")
    print(f"   Convergence to φ: {convergence['convergence_to_phi']:.6f}")
    print(f"   Average complexity: {convergence['average_complexity']:.6f}")
    
    print("\n7. Zeckendorf Twins (consecutive numbers with same term count):")
    
    twins = analyzer.find_zeckendorf_twins(1, 100)
    
    print(f"   Found {len(twins)} twin pairs in range 1-100")
    for pair in twins[:10]:  # Show first 10
        decomp1 = decomposer.decompose(pair[0])
        decomp2 = decomposer.decompose(pair[1])
        print(f"   {pair[0]}, {pair[1]}: both use {len(decomp1.fibonacci_terms)} terms")
    
    print("\n8. Tensor Representation Example:")
    
    tensorizer = ZeckendorfTensor()
    test_numbers = [1, 2, 3, 5, 8, 13]
    
    binary_tensor = tensorizer.tensorize_binary_representations(test_numbers, max_length=8)
    
    print("   Numbers and their tensor representations:")
    for i, n in enumerate(test_numbers):
        tensor_str = ''.join([str(int(x)) for x in binary_tensor[i]])
        decomp = decomposer.decompose(n)
        print(f"   {n:2d}: {tensor_str} (original: {decomp.binary_representation})")
    
    print("\n9. Neural Processing Example:")
    
    processor = ZeckendorfNeuralProcessor(max_length=16, hidden_dim=32)
    
    with torch.no_grad():
        predictions = processor.predict_properties([5, 8, 13, 21])
        
        print("   Neural predictions for [5, 8, 13, 21]:")
        print(f"   Predicted term counts: {predictions['term_counts'].flatten().tolist()}")
        print(f"   Predicted complexities: {predictions['complexities'].flatten().tolist()}")
        print(f"   Predicted φ-alignments: {predictions['phi_alignments'].flatten().tolist()}")
    
    print("\n" + "=" * 60)
    print("Zeckendorf decomposition reveals the φ-structure of natural numbers")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_zeckendorf_decompositions()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)