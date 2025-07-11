#!/usr/bin/env python3
"""
Chapter 002: PhiAlphabet - Verification Program
Defining Σφ = {00, 01, 10} as the Collapse-Safe Language

This program verifies that the φ-alphabet emerges necessarily from
the constraint against consecutive 1s, creating the foundation for
all safe trace construction.

从φ约束中，涌现出三个基本符号：00, 01, 10
这不是选择，而是必然。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from itertools import product


@dataclass
class PhiSymbol:
    """A symbol in the φ-alphabet"""
    bits: Tuple[int, int]
    
    def __str__(self):
        return f"{self.bits[0]}{self.bits[1]}"
    
    def __hash__(self):
        return hash(self.bits)
    
    def __eq__(self, other):
        return self.bits == other.bits
    
    @property
    def is_valid(self) -> bool:
        """Check if symbol respects φ-constraint"""
        return self.bits != (1, 1)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation"""
        return torch.tensor(self.bits, dtype=torch.float32)


class PhiAlphabet:
    """
    The complete φ-alphabet: Σφ = {00, 01, 10}
    This is the minimal complete alphabet for φ-constrained traces.
    """
    
    def __init__(self):
        # Generate all 2-bit combinations
        all_symbols = [PhiSymbol(bits) for bits in product([0, 1], repeat=2)]
        
        # Filter by φ-constraint
        self.symbols = [s for s in all_symbols if s.is_valid]
        
        # Create symbol set for fast lookup
        self.symbol_set = set(self.symbols)
        
        # Map symbols to indices
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.idx_to_symbol = {i: s for s, i in self.symbol_to_idx.items()}
    
    def __len__(self):
        return len(self.symbols)
    
    def __iter__(self):
        return iter(self.symbols)
    
    def __contains__(self, item):
        if isinstance(item, PhiSymbol):
            return item in self.symbol_set
        elif isinstance(item, tuple):
            return PhiSymbol(item) in self.symbol_set
        elif isinstance(item, str) and len(item) == 2:
            bits = (int(item[0]), int(item[1]))
            return PhiSymbol(bits) in self.symbol_set
        return False
    
    @staticmethod
    def verify_completeness() -> Dict[str, any]:
        """Verify that Σφ is complete for trace construction"""
        alphabet = PhiAlphabet()
        
        # Check all valid 2-bit patterns are included
        valid_patterns = []
        invalid_patterns = []
        
        for bits in product([0, 1], repeat=2):
            symbol = PhiSymbol(bits)
            if symbol.is_valid:
                valid_patterns.append(str(symbol))
            else:
                invalid_patterns.append(str(symbol))
        
        return {
            "alphabet_size": len(alphabet),
            "symbols": [str(s) for s in alphabet.symbols],
            "valid_patterns": valid_patterns,
            "invalid_patterns": invalid_patterns,
            "is_complete": set(valid_patterns) == set(str(s) for s in alphabet.symbols)
        }


class TraceBuilder:
    """
    Builds valid traces using only symbols from Σφ.
    Demonstrates that any valid trace can be constructed from the alphabet.
    """
    
    def __init__(self, alphabet: PhiAlphabet):
        self.alphabet = alphabet
    
    def can_build_trace(self, trace: str) -> Tuple[bool, List[PhiSymbol]]:
        """Check if trace can be built from φ-alphabet symbols"""
        if len(trace) == 0:
            return True, []
        
        if len(trace) == 1:
            # Single bit traces need special handling
            # We can represent them by considering overlap
            if trace == "0":
                return True, [PhiSymbol((0, 0))]
            else:
                return True, [PhiSymbol((1, 0))]
        
        # For longer traces, check overlapping pairs
        symbols = []
        for i in range(len(trace) - 1):
            pair = (int(trace[i]), int(trace[i+1]))
            symbol = PhiSymbol(pair)
            
            if symbol not in self.alphabet:
                return False, []
            
            symbols.append(symbol)
        
        return True, symbols
    
    def generate_all_traces(self, length: int) -> List[str]:
        """Generate all valid traces of given length using Σφ"""
        if length == 0:
            return [""]
        
        if length == 1:
            return ["0", "1"]
        
        # Start with all possible first symbols
        traces = []
        
        def extend_trace(current: str):
            if len(current) == length:
                traces.append(current)
                return
            
            # Try to extend with each symbol
            last_bit = int(current[-1])
            for symbol in self.alphabet:
                if symbol.bits[0] == last_bit:
                    # Symbol can connect
                    new_trace = current + str(symbol.bits[1])
                    extend_trace(new_trace)
        
        # Start with 0 and 1
        extend_trace("0")
        extend_trace("1")
        
        return traces


class PhiTransitions(nn.Module):
    """
    Neural model of transitions in φ-alphabet.
    Shows how symbols naturally connect while preserving constraint.
    """
    
    def __init__(self, alphabet: PhiAlphabet):
        super().__init__()
        self.alphabet = alphabet
        self.num_symbols = len(alphabet)
        
        # Transition matrix between symbols
        self.transition_matrix = nn.Parameter(
            torch.zeros(self.num_symbols, self.num_symbols)
        )
        
        # Initialize valid transitions
        self._init_transitions()
    
    def _init_transitions(self):
        """Initialize transition matrix based on φ-constraint"""
        with torch.no_grad():
            for i, sym1 in enumerate(self.alphabet.symbols):
                for j, sym2 in enumerate(self.alphabet.symbols):
                    # Check if sym2 can follow sym1
                    if sym1.bits[1] == sym2.bits[0]:
                        # Valid transition
                        self.transition_matrix[i, j] = 1.0
                    else:
                        # Invalid transition
                        self.transition_matrix[i, j] = -float('inf')
    
    def forward(self, symbol_idx: torch.Tensor) -> torch.Tensor:
        """Get transition probabilities from current symbol"""
        # Get row from transition matrix
        logits = self.transition_matrix[symbol_idx]
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_valid_transitions(self, symbol: PhiSymbol) -> List[PhiSymbol]:
        """Get all symbols that can follow the given symbol"""
        idx = self.alphabet.symbol_to_idx[symbol]
        probs = self.forward(torch.tensor(idx))
        
        valid = []
        for j, prob in enumerate(probs):
            if prob > 0:
                valid.append(self.alphabet.idx_to_symbol[j])
        
        return valid


class AlphabetProperties:
    """
    Analyzes mathematical properties of the φ-alphabet.
    """
    
    @staticmethod
    def analyze_symmetries(alphabet: PhiAlphabet) -> Dict[str, any]:
        """Analyze symmetries in the alphabet"""
        symbols = alphabet.symbols
        
        # Check bit flip symmetry
        bit_flip_pairs = []
        for s in symbols:
            flipped = PhiSymbol((1 - s.bits[0], 1 - s.bits[1]))
            if flipped in alphabet:
                bit_flip_pairs.append((str(s), str(flipped)))
        
        # Check reversal symmetry
        reversal_pairs = []
        for s in symbols:
            reversed_sym = PhiSymbol((s.bits[1], s.bits[0]))
            if reversed_sym in alphabet:
                reversal_pairs.append((str(s), str(reversed_sym)))
        
        # Information content
        info_content = {}
        for s in symbols:
            # Count 1s as "information"
            info = s.bits[0] + s.bits[1]
            info_content[str(s)] = info
        
        return {
            "bit_flip_pairs": bit_flip_pairs,
            "reversal_pairs": reversal_pairs,
            "information_content": info_content,
            "total_information": sum(info_content.values())
        }
    
    @staticmethod
    def connection_graph(alphabet: PhiAlphabet) -> Dict[str, List[str]]:
        """Build connection graph of symbols"""
        graph = {}
        
        for sym1 in alphabet.symbols:
            connections = []
            for sym2 in alphabet.symbols:
                # Check if sym2 can follow sym1
                if sym1.bits[1] == sym2.bits[0]:
                    connections.append(str(sym2))
            
            graph[str(sym1)] = connections
        
        return graph


class PhiAlphabetTests(unittest.TestCase):
    """Verify properties of the φ-alphabet"""
    
    def setUp(self):
        self.alphabet = PhiAlphabet()
        self.builder = TraceBuilder(self.alphabet)
    
    def test_alphabet_size(self):
        """Test: Σφ contains exactly 3 symbols"""
        self.assertEqual(len(self.alphabet), 3)
    
    def test_alphabet_members(self):
        """Test: Σφ = {00, 01, 10}"""
        expected = {"00", "01", "10"}
        actual = {str(s) for s in self.alphabet.symbols}
        self.assertEqual(actual, expected)
    
    def test_excludes_11(self):
        """Test: 11 is not in Σφ"""
        self.assertNotIn("11", self.alphabet)
        self.assertNotIn((1, 1), self.alphabet)
        self.assertNotIn(PhiSymbol((1, 1)), self.alphabet)
    
    def test_completeness(self):
        """Test: Σφ is complete for valid patterns"""
        result = PhiAlphabet.verify_completeness()
        
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["alphabet_size"], 3)
        self.assertEqual(set(result["valid_patterns"]), {"00", "01", "10"})
        self.assertEqual(result["invalid_patterns"], ["11"])
    
    def test_trace_construction(self):
        """Test: Any valid trace can be built from Σφ"""
        # Test various valid traces
        test_traces = ["0101", "1010", "0010", "1001", "00100"]
        
        for trace in test_traces:
            can_build, symbols = self.builder.can_build_trace(trace)
            self.assertTrue(can_build, f"Should be able to build {trace}")
            
            # Verify reconstruction
            if len(trace) > 1:
                reconstructed = trace[0]
                for sym in symbols:
                    reconstructed += str(sym.bits[1])
                self.assertEqual(reconstructed[:len(trace)], trace)
    
    def test_invalid_trace_detection(self):
        """Test: Traces with 11 cannot be built from Σφ"""
        invalid_traces = ["11", "0110", "1100", "0111"]
        
        for trace in invalid_traces:
            can_build, _ = self.builder.can_build_trace(trace)
            self.assertFalse(can_build, f"Should not be able to build {trace}")
    
    def test_trace_generation_count(self):
        """Test: Generated traces follow Fibonacci pattern"""
        expected_counts = [
            (0, 1),   # ""
            (1, 2),   # "0", "1"
            (2, 3),   # "00", "01", "10"
            (3, 5),   # 5 valid 3-bit traces
            (4, 8),   # 8 valid 4-bit traces
            (5, 13),  # 13 valid 5-bit traces
        ]
        
        for length, expected in expected_counts:
            traces = self.builder.generate_all_traces(length)
            self.assertEqual(len(traces), expected,
                           f"Length {length} should have {expected} traces")
    
    def test_transitions(self):
        """Test: Symbol transitions preserve φ-constraint"""
        model = PhiTransitions(self.alphabet)
        
        # Test each symbol
        for symbol in self.alphabet.symbols:
            valid_next = model.get_valid_transitions(symbol)
            
            # Check all transitions are valid
            for next_sym in valid_next:
                # Last bit of current must equal first bit of next
                self.assertEqual(symbol.bits[1], next_sym.bits[0])
                
                # Result must not create 11
                if symbol.bits[1] == 1:
                    self.assertNotEqual(next_sym.bits[1], 1)
    
    def test_alphabet_properties(self):
        """Test: Mathematical properties of Σφ"""
        props = AlphabetProperties.analyze_symmetries(self.alphabet)
        
        # Check symmetries
        self.assertEqual(len(props["bit_flip_pairs"]), 2)  # 01↔10 (00↔11 excluded)
        self.assertEqual(len(props["reversal_pairs"]), 3)  # 00↔00, 01↔10, 10↔01
        
        # Check information content
        self.assertEqual(props["information_content"]["00"], 0)
        self.assertEqual(props["information_content"]["01"], 1)
        self.assertEqual(props["information_content"]["10"], 1)
        self.assertEqual(props["total_information"], 2)
    
    def test_connection_graph(self):
        """Test: Symbol connection graph structure"""
        graph = AlphabetProperties.connection_graph(self.alphabet)
        
        # Expected connections
        expected = {
            "00": ["00", "01"],  # 0 can be followed by 0 or 1
            "01": ["10"],        # 1 can only be followed by 0
            "10": ["00", "01"]   # 0 can be followed by 0 or 1
        }
        
        self.assertEqual(graph, expected)


def visualize_phi_alphabet():
    """Visualize the φ-alphabet and its properties"""
    print("=" * 60)
    print("The φ-Alphabet: Σφ = {00, 01, 10}")
    print("=" * 60)
    
    alphabet = PhiAlphabet()
    
    # 1. Show alphabet members
    print("\n1. Alphabet Members:")
    for i, symbol in enumerate(alphabet.symbols):
        print(f"   Symbol {i}: {symbol}")
    
    # 2. Show why 11 is excluded
    print("\n2. Why 11 is Excluded:")
    print("   11 represents 'existence of existence'")
    print("   In ψ = ψ(ψ), this is redundant")
    print("   Therefore: Σφ = {00, 01, 10}")
    
    # 3. Show connection graph
    print("\n3. Symbol Connection Graph:")
    graph = AlphabetProperties.connection_graph(alphabet)
    for symbol, connections in graph.items():
        print(f"   {symbol} → {', '.join(connections)}")
    
    # 4. Show trace construction
    print("\n4. Example Trace Construction:")
    builder = TraceBuilder(alphabet)
    trace = "10010"
    can_build, symbols = builder.can_build_trace(trace)
    print(f"   Trace: {trace}")
    print(f"   Can build: {can_build}")
    if can_build:
        print(f"   Symbols: {' → '.join(str(s) for s in symbols)}")
    
    # 5. Show Fibonacci pattern
    print("\n5. Trace Count Pattern (Fibonacci):")
    print("   Length | Valid Traces | Count")
    print("   -------|--------------|-------")
    for length in range(6):
        traces = builder.generate_all_traces(length)
        print(f"   {length:6} | {len(traces):12} | F({length+1})")
    
    # 6. Show properties
    print("\n6. Alphabet Properties:")
    props = AlphabetProperties.analyze_symmetries(alphabet)
    print(f"   Bit-flip pairs: {props['bit_flip_pairs']}")
    print(f"   Reversal pairs: {props['reversal_pairs']}")
    print(f"   Total information: {props['total_information']} bits")
    
    print("\n" + "=" * 60)
    print("Conclusion: Σφ is the minimal complete alphabet")
    print("=" * 60)


if __name__ == "__main__":
    # Visualize first
    visualize_phi_alphabet()
    
    # Then run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)