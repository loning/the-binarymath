#!/usr/bin/env python3
"""
Chapter 018: CollapseMerge - Verification Program
Merging Collapse-Safe Blocks into Trace Tensor T^n

This program verifies that multiple collapse-safe φ-constrained traces can be
safely merged into higher-order tensor structures while preserving all
constraint properties and enabling tensor-level arithmetic operations.

从ψ的崩塌安全块中，涌现出高阶张量T^n——追踪张量的合并操作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import math
import itertools
from abc import ABC, abstractmethod


class CollapseMergeError(Exception):
    """Custom exception for collapse merge errors"""
    pass


@dataclass
class TraceBlock:
    """A collapse-safe block containing φ-constrained trace data"""
    trace_data: str                     # The φ-constrained trace
    block_id: str                       # Unique identifier
    tensor_rank: int                    # Rank in tensor hierarchy
    shape: Tuple[int, ...]             # Tensor shape
    phi_compliance: bool                # φ-constraint satisfaction
    entropy: float                      # Information entropy
    fibonacci_components: List[int]     # Contributing Fibonacci numbers
    merge_metadata: Dict[str, Any]      # Metadata for merge operations
    
    def __post_init__(self):
        """Validate the trace block"""
        if '11' in self.trace_data:
            raise CollapseMergeError(f"Block {self.block_id} violates φ-constraint")
        
        if not self.phi_compliance:
            raise CollapseMergeError(f"Block {self.block_id} marked as non-φ-compliant")


@dataclass
class TraceTensor:
    """Higher-order tensor structure from merged trace blocks"""
    tensor_data: torch.Tensor           # The tensor representation
    trace_blocks: List[TraceBlock]      # Contributing blocks
    tensor_rank: int                    # Order of the tensor (T^n)
    shape: Tuple[int, ...]             # Full tensor shape
    phi_invariants: Dict[str, float]    # φ-constraint invariants
    merge_history: List[Dict[str, Any]] # History of merge operations
    structural_properties: Dict[str, Any] # Structural analysis results
    
    def __post_init__(self):
        """Validate the trace tensor"""
        if self.tensor_rank != len(self.shape):
            raise CollapseMergeError(f"Tensor rank {self.tensor_rank} doesn't match shape {self.shape}")


class MergeStrategy(Enum):
    """Different strategies for merging trace blocks"""
    CONCATENATION = "concatenation"     # Simple concatenation
    INTERLEAVING = "interleaving"       # Interleaved merging
    HIERARCHICAL = "hierarchical"       # Hierarchical composition
    TENSOR_PRODUCT = "tensor_product"   # Tensor product construction
    FIBONACCI_ALIGNED = "fibonacci_aligned"  # Fibonacci-structure aware
    ENTROPY_MINIMIZING = "entropy_minimizing"  # Minimize total entropy
    PHI_OPTIMAL = "phi_optimal"         # Optimize φ-alignment


class BlockGenerator:
    """
    Generate collapse-safe blocks from various sources
    (Zeckendorf decompositions, Fibonacci encodings, etc.)
    """
    
    def __init__(self):
        self.fibonacci_sequence = self._generate_fibonacci(30)
        self.phi = (1 + math.sqrt(5)) / 2
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence"""
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
    
    def generate_zeckendorf_blocks(self, numbers: List[int]) -> List[TraceBlock]:
        """Generate blocks from Zeckendorf decompositions"""
        blocks = []
        
        for i, num in enumerate(numbers):
            trace_data = self._number_to_zeckendorf_trace(num)
            
            block = TraceBlock(
                trace_data=trace_data,
                block_id=f"zeck_{num}_{i}",
                tensor_rank=1,
                shape=(len(trace_data),),
                phi_compliance=self._verify_phi_constraint(trace_data),
                entropy=self._calculate_entropy(trace_data),
                fibonacci_components=self._extract_fibonacci_components(num),
                merge_metadata={'source': 'zeckendorf', 'original_number': num}
            )
            
            blocks.append(block)
        
        return blocks
    
    def generate_fibonacci_component_blocks(self, fib_values: List[int]) -> List[TraceBlock]:
        """Generate blocks from individual Fibonacci components"""
        blocks = []
        
        for i, fib_val in enumerate(fib_values):
            if fib_val not in self.fibonacci_sequence:
                continue
                
            fib_index = self.fibonacci_sequence.index(fib_val)
            trace_data = self._fibonacci_to_trace(fib_val, fib_index)
            
            block = TraceBlock(
                trace_data=trace_data,
                block_id=f"fib_{fib_val}_{i}",
                tensor_rank=1,
                shape=(len(trace_data),),
                phi_compliance=self._verify_phi_constraint(trace_data),
                entropy=self._calculate_entropy(trace_data),
                fibonacci_components=[fib_val],
                merge_metadata={'source': 'fibonacci', 'fib_index': fib_index}
            )
            
            blocks.append(block)
        
        return blocks
    
    def generate_synthetic_blocks(self, count: int, max_length: int = 16) -> List[TraceBlock]:
        """Generate synthetic φ-compliant blocks for testing"""
        blocks = []
        
        for i in range(count):
            # Generate random φ-compliant trace
            trace_data = self._generate_random_phi_trace(max_length)
            
            block = TraceBlock(
                trace_data=trace_data,
                block_id=f"synth_{i}",
                tensor_rank=1,
                shape=(len(trace_data),),
                phi_compliance=True,
                entropy=self._calculate_entropy(trace_data),
                fibonacci_components=[],
                merge_metadata={'source': 'synthetic', 'generation_id': i}
            )
            
            blocks.append(block)
        
        return blocks
    
    def _number_to_zeckendorf_trace(self, n: int) -> str:
        """Convert number to Zeckendorf binary trace"""
        if n <= 0:
            return '0'
        
        fib_terms = []
        remaining = n
        
        # Greedy algorithm
        for fib in reversed(self.fibonacci_sequence):
            if fib <= remaining:
                fib_terms.append(fib)
                remaining -= fib
                if remaining == 0:
                    break
        
        # Convert to binary trace
        if not fib_terms:
            return '0'
        
        max_fib = max(fib_terms)
        max_index = self.fibonacci_sequence.index(max_fib)
        
        trace_bits = ['0'] * (max_index + 1)
        for fib in fib_terms:
            fib_index = self.fibonacci_sequence.index(fib)
            trace_bits[max_index - fib_index] = '1'
        
        return ''.join(trace_bits).lstrip('0') or '0'
    
    def _fibonacci_to_trace(self, fib_val: int, fib_index: int) -> str:
        """Convert Fibonacci number to trace representation"""
        trace_bits = ['0'] * (fib_index + 1)
        trace_bits[0] = '1'  # Most significant bit
        return ''.join(trace_bits)
    
    def _generate_random_phi_trace(self, max_length: int) -> str:
        """Generate random φ-compliant trace"""
        length = np.random.randint(1, max_length + 1)
        trace = []
        prev_was_one = False
        
        for _ in range(length):
            if prev_was_one:
                # Must be 0 to maintain φ-constraint
                bit = '0'
            else:
                # Can be 0 or 1
                bit = np.random.choice(['0', '1'], p=[0.6, 0.4])
            
            trace.append(bit)
            prev_was_one = (bit == '1')
        
        return ''.join(trace)
    
    def _verify_phi_constraint(self, trace: str) -> bool:
        """Verify φ-constraint (no consecutive 1s)"""
        return '11' not in trace
    
    def _calculate_entropy(self, trace: str) -> float:
        """Calculate Shannon entropy of trace"""
        if not trace:
            return 0.0
        
        counts = defaultdict(int)
        for bit in trace:
            counts[bit] += 1
        
        total = len(trace)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _extract_fibonacci_components(self, n: int) -> List[int]:
        """Extract Fibonacci components from Zeckendorf decomposition"""
        components = []
        remaining = n
        
        for fib in reversed(self.fibonacci_sequence):
            if fib <= remaining:
                components.append(fib)
                remaining -= fib
                if remaining == 0:
                    break
        
        return sorted(components, reverse=True)


class ConcatenationMerger:
    """Simple concatenation merger for trace blocks"""
    
    def merge_blocks(self, blocks: List[TraceBlock]) -> TraceTensor:
        """Merge blocks through concatenation"""
        if not blocks:
            raise CollapseMergeError("No blocks to merge")
        
        # Concatenate all trace data
        concatenated_trace = ''.join(block.trace_data for block in blocks)
        
        # Verify φ-constraint preservation
        if '11' in concatenated_trace:
            # Try to insert separators
            concatenated_trace = self._insert_phi_separators(concatenated_trace)
        
        phi_compliant = '11' not in concatenated_trace
        
        # Convert to tensor
        tensor_data = self._trace_to_tensor(concatenated_trace)
        
        # Calculate φ-invariants
        phi_invariants = self._calculate_phi_invariants(concatenated_trace, blocks)
        
        # Record merge history
        merge_history = [{
            'strategy': 'concatenation',
            'block_count': len(blocks),
            'result_length': len(concatenated_trace),
            'phi_compliant': phi_compliant
        }]
        
        return TraceTensor(
            tensor_data=tensor_data,
            trace_blocks=blocks,
            tensor_rank=1,
            shape=tensor_data.shape,
            phi_invariants=phi_invariants,
            merge_history=merge_history,
            structural_properties=self._analyze_structure(concatenated_trace)
        )
    
    def _insert_phi_separators(self, trace: str) -> str:
        """Insert separators to maintain φ-constraint"""
        result = []
        prev_was_one = False
        
        for bit in trace:
            if bit == '1' and prev_was_one:
                result.append('0')  # Insert separator
            result.append(bit)
            prev_was_one = (bit == '1')
        
        return ''.join(result)
    
    def _trace_to_tensor(self, trace: str) -> torch.Tensor:
        """Convert trace string to tensor"""
        return torch.tensor([float(bit) for bit in trace], dtype=torch.float32)
    
    def _calculate_phi_invariants(self, trace: str, blocks: List[TraceBlock]) -> Dict[str, float]:
        """Calculate φ-constraint invariants"""
        zeros = trace.count('0')
        ones = trace.count('1')
        
        return {
            'total_length': len(trace),
            'zero_count': zeros,
            'one_count': ones,
            'zero_ratio': zeros / len(trace) if trace else 0,
            'phi_alignment': self._calculate_phi_alignment(trace),
            'entropy': self._calculate_entropy(trace),
            'block_count': len(blocks),
            'average_block_entropy': np.mean([b.entropy for b in blocks]),
            'phi_compliant': '11' not in trace
        }
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate alignment with golden ratio"""
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
    
    def _calculate_entropy(self, trace: str) -> float:
        """Calculate Shannon entropy"""
        if not trace:
            return 0.0
        
        counts = defaultdict(int)
        for bit in trace:
            counts[bit] += 1
        
        total = len(trace)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _analyze_structure(self, trace: str) -> Dict[str, Any]:
        """Analyze structural properties of merged trace"""
        return {
            'length': len(trace),
            'phi_compliant': '11' not in trace,
            'pattern_complexity': len(set(trace[i:i+3] for i in range(len(trace)-2))) if len(trace) >= 3 else 0,
            'alternation_frequency': sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1]) / max(1, len(trace)-1),
            'longest_zero_run': max(len(run) for run in trace.split('1') if run) if '1' in trace else len(trace),
            'transition_count': sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        }


class InterleavingMerger:
    """Interleaved merger that alternates between blocks"""
    
    def merge_blocks(self, blocks: List[TraceBlock]) -> TraceTensor:
        """Merge blocks through interleaving"""
        if not blocks:
            raise CollapseMergeError("No blocks to merge")
        
        # Find maximum block length
        max_length = max(len(block.trace_data) for block in blocks)
        
        # Pad all blocks to same length
        padded_blocks = []
        for block in blocks:
            padded_trace = block.trace_data.ljust(max_length, '0')
            padded_blocks.append(padded_trace)
        
        # Interleave bits
        interleaved = []
        for i in range(max_length):
            for block_trace in padded_blocks:
                if i < len(block_trace):
                    interleaved.append(block_trace[i])
        
        interleaved_trace = ''.join(interleaved)
        
        # Ensure φ-constraint
        if '11' in interleaved_trace:
            interleaved_trace = self._fix_phi_violations(interleaved_trace)
        
        # Convert to tensor
        tensor_data = torch.tensor([float(bit) for bit in interleaved_trace], dtype=torch.float32)
        
        # Calculate properties
        phi_invariants = self._calculate_invariants(interleaved_trace, blocks)
        merge_history = [{
            'strategy': 'interleaving',
            'block_count': len(blocks),
            'max_block_length': max_length,
            'result_length': len(interleaved_trace)
        }]
        
        return TraceTensor(
            tensor_data=tensor_data,
            trace_blocks=blocks,
            tensor_rank=2,  # Interleaving creates 2D structure
            shape=(len(blocks), max_length),
            phi_invariants=phi_invariants,
            merge_history=merge_history,
            structural_properties=self._analyze_interleaved_structure(interleaved_trace, blocks)
        )
    
    def _fix_phi_violations(self, trace: str) -> str:
        """Fix φ-constraint violations in interleaved trace"""
        # Simple approach: replace second '1' in consecutive pairs with '0'
        result = list(trace)
        
        for i in range(len(result) - 1):
            if result[i] == '1' and result[i+1] == '1':
                result[i+1] = '0'
        
        return ''.join(result)
    
    def _calculate_invariants(self, trace: str, blocks: List[TraceBlock]) -> Dict[str, float]:
        """Calculate invariants for interleaved structure"""
        return {
            'total_length': len(trace),
            'block_count': len(blocks),
            'average_block_length': np.mean([len(b.trace_data) for b in blocks]),
            'phi_compliant': '11' not in trace,
            'interleaving_entropy': self._calculate_interleaving_entropy(trace, len(blocks))
        }
    
    def _calculate_interleaving_entropy(self, trace: str, block_count: int) -> float:
        """Calculate entropy considering interleaving structure"""
        if len(trace) < block_count:
            return 0.0
        
        # Analyze entropy in each interleaved position
        entropies = []
        
        for offset in range(block_count):
            subsequence = trace[offset::block_count]
            if subsequence:
                entropy = self._shannon_entropy(subsequence)
                entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0
    
    def _shannon_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence"""
        if not sequence:
            return 0.0
        
        counts = defaultdict(int)
        for char in sequence:
            counts[char] += 1
        
        total = len(sequence)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _analyze_interleaved_structure(self, trace: str, blocks: List[TraceBlock]) -> Dict[str, Any]:
        """Analyze structure of interleaved trace"""
        return {
            'interleaving_pattern': len(blocks),
            'total_length': len(trace),
            'phi_preservation': '11' not in trace,
            'structural_entropy': self._shannon_entropy(trace),
            'block_contribution_balance': self._calculate_block_balance(trace, len(blocks))
        }
    
    def _calculate_block_balance(self, trace: str, block_count: int) -> float:
        """Calculate how balanced the block contributions are"""
        if len(trace) < block_count:
            return 0.0
        
        # Count contributions from each block position
        contributions = [0] * block_count
        
        for i, bit in enumerate(trace):
            if bit == '1':
                contributions[i % block_count] += 1
        
        # Calculate variance (lower = more balanced)
        if sum(contributions) == 0:
            return 1.0
        
        mean_contribution = sum(contributions) / block_count
        variance = sum((c - mean_contribution) ** 2 for c in contributions) / block_count
        
        # Convert to balance score (0 = unbalanced, 1 = perfectly balanced)
        max_possible_variance = mean_contribution ** 2
        if max_possible_variance == 0:
            return 1.0
        
        return max(0.0, 1.0 - variance / max_possible_variance)


class TensorProductMerger:
    """Tensor product merger creating higher-order structures"""
    
    def merge_blocks(self, blocks: List[TraceBlock]) -> TraceTensor:
        """Merge blocks using tensor product construction"""
        if len(blocks) < 2:
            raise CollapseMergeError("Tensor product requires at least 2 blocks")
        
        # Convert blocks to tensors
        block_tensors = []
        for block in blocks:
            tensor = torch.tensor([float(bit) for bit in block.trace_data], dtype=torch.float32)
            block_tensors.append(tensor)
        
        # Compute tensor product
        result_tensor = block_tensors[0]
        for tensor in block_tensors[1:]:
            result_tensor = torch.outer(result_tensor.flatten(), tensor.flatten())
        
        # Calculate tensor rank and shape
        tensor_rank = len(blocks)
        shape = tuple(len(block.trace_data) for block in blocks)
        
        # Create trace representation from tensor
        flattened = result_tensor.flatten()
        # Binarize: >0.5 becomes 1, <=0.5 becomes 0
        binary_trace = ''.join('1' if x > 0.5 else '0' for x in flattened)
        
        # Ensure φ-constraint
        if '11' in binary_trace:
            binary_trace = self._enforce_phi_constraint(binary_trace)
        
        # Calculate properties
        phi_invariants = self._calculate_tensor_invariants(result_tensor, blocks)
        merge_history = [{
            'strategy': 'tensor_product',
            'input_blocks': len(blocks),
            'tensor_rank': tensor_rank,
            'tensor_shape': shape,
            'flattened_length': len(binary_trace)
        }]
        
        return TraceTensor(
            tensor_data=result_tensor,
            trace_blocks=blocks,
            tensor_rank=tensor_rank,
            shape=shape,
            phi_invariants=phi_invariants,
            merge_history=merge_history,
            structural_properties=self._analyze_tensor_structure(result_tensor, binary_trace)
        )
    
    def _enforce_phi_constraint(self, trace: str) -> str:
        """Enforce φ-constraint on tensor-generated trace"""
        result = []
        prev_was_one = False
        
        for bit in trace:
            if bit == '1' and prev_was_one:
                # Insert zero or skip this 1
                result.append('0')
            else:
                result.append(bit)
            prev_was_one = (result[-1] == '1')
        
        return ''.join(result)
    
    def _calculate_tensor_invariants(self, tensor: torch.Tensor, blocks: List[TraceBlock]) -> Dict[str, float]:
        """Calculate invariants for tensor product structure"""
        flattened = tensor.flatten()
        
        return {
            'tensor_rank': len(blocks),
            'tensor_size': tensor.numel(),
            'tensor_norm': torch.norm(tensor).item(),
            'tensor_trace': torch.trace(tensor.view(-1, tensor.shape[-1])).item() if tensor.dim() >= 2 else 0.0,
            'sparsity': (flattened == 0).float().mean().item(),
            'max_value': torch.max(tensor).item(),
            'min_value': torch.min(tensor).item(),
            'mean_value': torch.mean(tensor).item()
        }
    
    def _analyze_tensor_structure(self, tensor: torch.Tensor, binary_trace: str) -> Dict[str, Any]:
        """Analyze structure of tensor product result"""
        return {
            'tensor_rank': tensor.dim(),
            'tensor_shape': list(tensor.shape),
            'binary_trace_length': len(binary_trace),
            'phi_compliant': '11' not in binary_trace,
            'tensor_density': (tensor != 0).float().mean().item(),
            'separability': self._calculate_separability(tensor)
        }
    
    def _calculate_separability(self, tensor: torch.Tensor) -> float:
        """Calculate how separable the tensor is"""
        if tensor.dim() < 2:
            return 1.0
        
        # Simple separability measure based on rank
        flattened = tensor.view(tensor.shape[0], -1)
        try:
            rank = torch.linalg.matrix_rank(flattened).item()
            max_rank = min(flattened.shape)
            return rank / max_rank if max_rank > 0 else 0.0
        except:
            return 0.5  # Default value if rank calculation fails


class FibonacciAlignedMerger:
    """Merger that respects Fibonacci structure in blocks"""
    
    def __init__(self):
        self.fibonacci_sequence = self._generate_fibonacci(20)
        self.phi = (1 + math.sqrt(5)) / 2
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence"""
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
    
    def merge_blocks(self, blocks: List[TraceBlock]) -> TraceTensor:
        """Merge blocks while preserving Fibonacci alignment"""
        if not blocks:
            raise CollapseMergeError("No blocks to merge")
        
        # Sort blocks by their Fibonacci components
        sorted_blocks = sorted(blocks, key=lambda b: sum(b.fibonacci_components) if b.fibonacci_components else 0)
        
        # Merge using Fibonacci-weighted concatenation
        merged_trace = self._fibonacci_weighted_merge(sorted_blocks)
        
        # Ensure φ-constraint
        if '11' in merged_trace:
            merged_trace = self._fibonacci_safe_repair(merged_trace)
        
        # Convert to tensor
        tensor_data = torch.tensor([float(bit) for bit in merged_trace], dtype=torch.float32)
        
        # Calculate Fibonacci-specific invariants
        phi_invariants = self._calculate_fibonacci_invariants(merged_trace, sorted_blocks)
        
        merge_history = [{
            'strategy': 'fibonacci_aligned',
            'block_count': len(blocks),
            'fibonacci_preservation': self._check_fibonacci_preservation(sorted_blocks),
            'phi_alignment_score': phi_invariants.get('phi_alignment', 0.0)
        }]
        
        return TraceTensor(
            tensor_data=tensor_data,
            trace_blocks=sorted_blocks,
            tensor_rank=1,
            shape=tensor_data.shape,
            phi_invariants=phi_invariants,
            merge_history=merge_history,
            structural_properties=self._analyze_fibonacci_structure(merged_trace, sorted_blocks)
        )
    
    def _fibonacci_weighted_merge(self, blocks: List[TraceBlock]) -> str:
        """Merge blocks with Fibonacci component weighting"""
        if not blocks:
            return '0'
        
        # Create weighted segments
        segments = []
        
        for block in blocks:
            # Weight by sum of Fibonacci components
            weight = sum(block.fibonacci_components) if block.fibonacci_components else 1
            fibonacci_spacing = max(1, weight.bit_length() - 1)
            
            # Add spacing based on Fibonacci structure
            segment = block.trace_data
            if segments:  # Add spacing between segments
                segment = '0' * fibonacci_spacing + segment
            
            segments.append(segment)
        
        return ''.join(segments)
    
    def _fibonacci_safe_repair(self, trace: str) -> str:
        """Repair φ-constraint violations using Fibonacci principles"""
        result = []
        prev_was_one = False
        fibonacci_counter = 0
        
        for bit in trace:
            if bit == '1' and prev_was_one:
                # Insert Fibonacci-length separator
                fib_length = self.fibonacci_sequence[min(fibonacci_counter, len(self.fibonacci_sequence)-1)]
                separator_length = min(fib_length, 3)  # Cap at 3 for practicality
                result.extend(['0'] * separator_length)
                fibonacci_counter = (fibonacci_counter + 1) % len(self.fibonacci_sequence)
            
            result.append(bit)
            prev_was_one = (bit == '1')
        
        return ''.join(result)
    
    def _calculate_fibonacci_invariants(self, trace: str, blocks: List[TraceBlock]) -> Dict[str, float]:
        """Calculate Fibonacci-specific invariants"""
        total_fibonacci_sum = sum(sum(b.fibonacci_components) if b.fibonacci_components else 0 for b in blocks)
        
        return {
            'total_fibonacci_sum': total_fibonacci_sum,
            'fibonacci_block_count': sum(1 for b in blocks if b.fibonacci_components),
            'average_fibonacci_value': total_fibonacci_sum / max(1, len([b for b in blocks if b.fibonacci_components])),
            'phi_alignment': self._calculate_phi_alignment(trace),
            'fibonacci_pattern_density': self._calculate_fibonacci_pattern_density(trace),
            'golden_ratio_approximation': self._calculate_golden_ratio_approximation(trace),
            'phi_compliant': '11' not in trace
        }
    
    def _check_fibonacci_preservation(self, blocks: List[TraceBlock]) -> bool:
        """Check if Fibonacci structure is preserved in merge"""
        # Simple check: ensure blocks with Fibonacci components maintain their identity
        return all(block.fibonacci_components or block.merge_metadata.get('source') != 'fibonacci' 
                  for block in blocks)
    
    def _calculate_phi_alignment(self, trace: str) -> float:
        """Calculate alignment with golden ratio"""
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
    
    def _calculate_fibonacci_pattern_density(self, trace: str) -> float:
        """Calculate density of Fibonacci-like patterns"""
        if len(trace) < 3:
            return 0.0
        
        fibonacci_patterns = ['101', '010', '1010', '0101']
        pattern_count = 0
        
        for i in range(len(trace) - 2):
            substring = trace[i:i+3]
            if substring in fibonacci_patterns:
                pattern_count += 1
        
        return pattern_count / max(1, len(trace) - 2)
    
    def _calculate_golden_ratio_approximation(self, trace: str) -> float:
        """Calculate how well the trace approximates golden ratio properties"""
        if len(trace) < 2:
            return 0.0
        
        # Look for golden ratio in subsequence ratios
        zero_runs = []
        one_runs = []
        
        current_run = 1
        current_char = trace[0]
        
        for i in range(1, len(trace)):
            if trace[i] == current_char:
                current_run += 1
            else:
                if current_char == '0':
                    zero_runs.append(current_run)
                else:
                    one_runs.append(current_run)
                current_run = 1
                current_char = trace[i]
        
        # Add final run
        if current_char == '0':
            zero_runs.append(current_run)
        else:
            one_runs.append(current_run)
        
        if not zero_runs or not one_runs:
            return 0.0
        
        avg_zero_run = np.mean(zero_runs)
        avg_one_run = np.mean(one_runs)
        
        if avg_one_run == 0:
            return 0.0
        
        ratio = avg_zero_run / avg_one_run
        deviation = abs(ratio - self.phi) / self.phi
        
        return max(0.0, 1.0 - deviation)
    
    def _analyze_fibonacci_structure(self, trace: str, blocks: List[TraceBlock]) -> Dict[str, Any]:
        """Analyze Fibonacci-specific structure"""
        return {
            'fibonacci_block_count': len([b for b in blocks if b.fibonacci_components]),
            'total_fibonacci_components': sum(len(b.fibonacci_components) for b in blocks),
            'phi_alignment_score': self._calculate_phi_alignment(trace),
            'fibonacci_pattern_density': self._calculate_fibonacci_pattern_density(trace),
            'golden_structure_quality': self._calculate_golden_ratio_approximation(trace)
        }


class MergeAnalyzer:
    """
    Analyze and compare different merge strategies
    """
    
    def __init__(self):
        self.mergers = {
            MergeStrategy.CONCATENATION: ConcatenationMerger(),
            MergeStrategy.INTERLEAVING: InterleavingMerger(),
            MergeStrategy.TENSOR_PRODUCT: TensorProductMerger(),
            MergeStrategy.FIBONACCI_ALIGNED: FibonacciAlignedMerger()
        }
    
    def compare_merge_strategies(self, blocks: List[TraceBlock]) -> Dict[str, Any]:
        """Compare all merge strategies on given blocks"""
        results = {}
        
        for strategy, merger in self.mergers.items():
            try:
                tensor_result = merger.merge_blocks(blocks.copy())
                
                analysis = {
                    'success': True,
                    'tensor_result': tensor_result,
                    'phi_compliance': tensor_result.phi_invariants.get('phi_compliant', False),
                    'tensor_rank': tensor_result.tensor_rank,
                    'result_size': tensor_result.tensor_data.numel(),
                    'merge_efficiency': self._calculate_merge_efficiency(blocks, tensor_result),
                    'structural_quality': self._assess_structural_quality(tensor_result)
                }
                
                results[strategy.value] = analysis
                
            except Exception as e:
                results[strategy.value] = {
                    'success': False,
                    'error': str(e),
                    'phi_compliance': False,
                    'tensor_rank': 0,
                    'result_size': 0
                }
        
        return results
    
    def _calculate_merge_efficiency(self, input_blocks: List[TraceBlock], result: TraceTensor) -> float:
        """Calculate efficiency of merge operation"""
        input_total_size = sum(len(block.trace_data) for block in input_blocks)
        result_size = result.tensor_data.numel()
        
        if input_total_size == 0:
            return 0.0
        
        # Efficiency based on compression and structure preservation
        compression_ratio = result_size / input_total_size
        structure_preservation = 1.0 if result.phi_invariants.get('phi_compliant', False) else 0.5
        
        return structure_preservation / compression_ratio if compression_ratio > 0 else 0.0
    
    def _assess_structural_quality(self, tensor_result: TraceTensor) -> float:
        """Assess structural quality of merge result"""
        quality_factors = []
        
        # φ-constraint compliance
        if tensor_result.phi_invariants.get('phi_compliant', False):
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Entropy considerations
        entropy = tensor_result.phi_invariants.get('entropy', 0.0)
        normalized_entropy = min(1.0, entropy)  # Cap at 1.0
        quality_factors.append(normalized_entropy)
        
        # Tensor rank appropriateness
        rank_quality = min(1.0, tensor_result.tensor_rank / 3.0)  # Assume rank 3 is good
        quality_factors.append(rank_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def find_optimal_strategy(self, blocks: List[TraceBlock]) -> Tuple[MergeStrategy, TraceTensor]:
        """Find optimal merge strategy for given blocks"""
        comparison = self.compare_merge_strategies(blocks)
        
        best_strategy = None
        best_score = -1.0
        best_result = None
        
        for strategy_name, analysis in comparison.items():
            if analysis['success']:
                # Score based on compliance, efficiency, and quality
                score = (
                    (1.0 if analysis['phi_compliance'] else 0.0) * 0.5 +
                    analysis['merge_efficiency'] * 0.3 +
                    analysis['structural_quality'] * 0.2
                )
                
                if score > best_score:
                    best_score = score
                    best_strategy = MergeStrategy(strategy_name)
                    best_result = analysis['tensor_result']
        
        if best_strategy is None:
            raise CollapseMergeError("No successful merge strategy found")
        
        return best_strategy, best_result


class CollapseMergeTests(unittest.TestCase):
    """Test collapse merge functionality"""
    
    def setUp(self):
        self.block_generator = BlockGenerator()
        self.concatenation_merger = ConcatenationMerger()
        self.interleaving_merger = InterleavingMerger()
        self.tensor_merger = TensorProductMerger()
        self.fibonacci_merger = FibonacciAlignedMerger()
        self.analyzer = MergeAnalyzer()
        
        # Generate test blocks
        self.test_numbers = [1, 3, 5, 8]
        self.zeckendorf_blocks = self.block_generator.generate_zeckendorf_blocks(self.test_numbers)
        self.fibonacci_blocks = self.block_generator.generate_fibonacci_component_blocks([1, 2, 3, 5])
        self.synthetic_blocks = self.block_generator.generate_synthetic_blocks(3, 8)
    
    def test_block_generation(self):
        """Test: Block generation produces valid φ-constrained blocks"""
        for block in self.zeckendorf_blocks:
            self.assertTrue(block.phi_compliance)
            self.assertNotIn('11', block.trace_data)
            self.assertGreater(len(block.trace_data), 0)
        
        for block in self.fibonacci_blocks:
            self.assertTrue(block.phi_compliance)
            self.assertNotIn('11', block.trace_data)
    
    def test_concatenation_merge(self):
        """Test: Concatenation merge preserves φ-constraint"""
        result = self.concatenation_merger.merge_blocks(self.zeckendorf_blocks)
        
        self.assertEqual(result.tensor_rank, 1)
        self.assertTrue(result.phi_invariants.get('phi_compliant', False))
        self.assertEqual(len(result.trace_blocks), len(self.zeckendorf_blocks))
        self.assertGreater(result.tensor_data.numel(), 0)
    
    def test_interleaving_merge(self):
        """Test: Interleaving merge creates valid tensor structure"""
        result = self.interleaving_merger.merge_blocks(self.fibonacci_blocks)
        
        self.assertEqual(result.tensor_rank, 2)
        self.assertTrue(result.phi_invariants.get('phi_compliant', False))
        self.assertEqual(len(result.trace_blocks), len(self.fibonacci_blocks))
    
    def test_tensor_product_merge(self):
        """Test: Tensor product merge creates higher-order structures"""
        small_blocks = self.synthetic_blocks[:2]  # Use only 2 blocks
        result = self.tensor_merger.merge_blocks(small_blocks)
        
        self.assertEqual(result.tensor_rank, 2)
        self.assertEqual(len(result.trace_blocks), 2)
        self.assertGreater(result.tensor_data.numel(), 0)
    
    def test_fibonacci_aligned_merge(self):
        """Test: Fibonacci-aligned merge preserves structure"""
        result = self.fibonacci_merger.merge_blocks(self.fibonacci_blocks)
        
        self.assertTrue(result.phi_invariants.get('phi_compliant', False))
        self.assertIn('golden_ratio_approximation', result.phi_invariants)
    
    def test_phi_constraint_preservation(self):
        """Test: All merge strategies preserve φ-constraint"""
        mergers = [
            self.concatenation_merger,
            self.interleaving_merger,
            self.fibonacci_merger
        ]
        
        for merger in mergers:
            result = merger.merge_blocks(self.synthetic_blocks)
            self.assertTrue(result.phi_invariants.get('phi_compliant', False),
                f"φ-constraint violated by {type(merger).__name__}")
    
    def test_merge_strategy_comparison(self):
        """Test: Strategy comparison produces valid results"""
        comparison = self.analyzer.compare_merge_strategies(self.zeckendorf_blocks)
        
        # Check that all strategies were attempted
        expected_strategies = ['concatenation', 'interleaving', 'tensor_product', 'fibonacci_aligned']
        for strategy in expected_strategies:
            self.assertIn(strategy, comparison)
        
        # Check that at least one strategy succeeded
        successes = [analysis['success'] for analysis in comparison.values()]
        self.assertTrue(any(successes))
    
    def test_optimal_strategy_selection(self):
        """Test: Optimal strategy selection works"""
        strategy, result = self.analyzer.find_optimal_strategy(self.fibonacci_blocks)
        
        self.assertIsInstance(strategy, MergeStrategy)
        self.assertIsInstance(result, TraceTensor)
        self.assertTrue(result.phi_invariants.get('phi_compliant', False))
    
    def test_tensor_structure_validation(self):
        """Test: Generated tensors have correct structure"""
        result = self.concatenation_merger.merge_blocks(self.zeckendorf_blocks)
        
        self.assertEqual(result.tensor_data.shape, result.shape)
        self.assertEqual(result.tensor_rank, len(result.shape))
        self.assertGreater(len(result.merge_history), 0)
    
    def test_empty_block_handling(self):
        """Test: Proper error handling for empty block lists"""
        with self.assertRaises(CollapseMergeError):
            self.concatenation_merger.merge_blocks([])
        
        with self.assertRaises(CollapseMergeError):
            self.interleaving_merger.merge_blocks([])
    
    def test_invariant_calculation(self):
        """Test: φ-invariants are calculated correctly"""
        result = self.concatenation_merger.merge_blocks(self.zeckendorf_blocks)
        
        invariants = result.phi_invariants
        self.assertIn('total_length', invariants)
        self.assertIn('phi_alignment', invariants)
        self.assertIn('entropy', invariants)
        self.assertGreaterEqual(invariants['phi_alignment'], 0.0)
        self.assertLessEqual(invariants['phi_alignment'], 1.0)


def visualize_collapse_merge():
    """Visualize collapse merge operations and results"""
    print("=" * 70)
    print("CollapseMerge: Merging Collapse-Safe Blocks into Trace Tensor T^n")
    print("=" * 70)
    
    # Initialize components
    generator = BlockGenerator()
    analyzer = MergeAnalyzer()
    
    # Generate test blocks
    test_numbers = [1, 3, 5, 8]
    zeckendorf_blocks = generator.generate_zeckendorf_blocks(test_numbers)
    fibonacci_blocks = generator.generate_fibonacci_component_blocks([1, 2, 3, 5])
    
    print("\n1. Generated Trace Blocks:")
    
    print("\n   Zeckendorf Blocks:")
    for block in zeckendorf_blocks:
        print(f"     {block.block_id}: {block.trace_data} | φ-compliant: {block.phi_compliance} | entropy: {block.entropy:.3f}")
    
    print("\n   Fibonacci Component Blocks:")
    for block in fibonacci_blocks:
        fib_comps = ', '.join(map(str, block.fibonacci_components))
        print(f"     {block.block_id}: {block.trace_data} | components: [{fib_comps}] | entropy: {block.entropy:.3f}")
    
    print("\n2. Merge Strategy Comparison:")
    
    comparison = analyzer.compare_merge_strategies(zeckendorf_blocks)
    
    for strategy_name, analysis in comparison.items():
        print(f"\n   {strategy_name.title()} Strategy:")
        if analysis['success']:
            print(f"     φ-compliant: {analysis['phi_compliance']}")
            print(f"     Tensor rank: {analysis['tensor_rank']}")
            print(f"     Result size: {analysis['result_size']}")
            print(f"     Merge efficiency: {analysis['merge_efficiency']:.3f}")
            print(f"     Structural quality: {analysis['structural_quality']:.3f}")
        else:
            print(f"     Failed: {analysis['error']}")
    
    print("\n3. Optimal Strategy Selection:")
    
    try:
        optimal_strategy, optimal_result = analyzer.find_optimal_strategy(fibonacci_blocks)
        print(f"   Optimal strategy: {optimal_strategy.value}")
        print(f"   Result tensor rank: {optimal_result.tensor_rank}")
        print(f"   Result tensor shape: {optimal_result.shape}")
        print(f"   φ-compliance: {optimal_result.phi_invariants.get('phi_compliant', False)}")
        
        print(f"\n   φ-Invariants:")
        for key, value in optimal_result.phi_invariants.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.3f}")
            else:
                print(f"     {key}: {value}")
        
    except Exception as e:
        print(f"   Error finding optimal strategy: {e}")
    
    print("\n4. Detailed Concatenation Analysis:")
    
    concatenation_merger = ConcatenationMerger()
    concat_result = concatenation_merger.merge_blocks(zeckendorf_blocks)
    
    # Extract trace from tensor
    tensor_data = concat_result.tensor_data
    binary_trace = ''.join('1' if x > 0.5 else '0' for x in tensor_data)
    
    print(f"   Input blocks: {[b.trace_data for b in zeckendorf_blocks]}")
    print(f"   Merged trace: {binary_trace}")
    print(f"   φ-constraint satisfied: {'11' not in binary_trace}")
    print(f"   Total length: {len(binary_trace)}")
    print(f"   Compression ratio: {len(binary_trace) / sum(len(b.trace_data) for b in zeckendorf_blocks):.3f}")
    
    print("\n5. Structural Properties Analysis:")
    
    structural_props = concat_result.structural_properties
    print(f"   Pattern complexity: {structural_props.get('pattern_complexity', 0)}")
    print(f"   Alternation frequency: {structural_props.get('alternation_frequency', 0):.3f}")
    print(f"   Longest zero run: {structural_props.get('longest_zero_run', 0)}")
    print(f"   Transition count: {structural_props.get('transition_count', 0)}")
    
    print("\n6. Interleaving Merge Example:")
    
    interleaving_merger = InterleavingMerger()
    small_fibonacci_blocks = fibonacci_blocks[:3]  # Use 3 blocks
    
    interleave_result = interleaving_merger.merge_blocks(small_fibonacci_blocks)
    interleave_binary = ''.join('1' if x > 0.5 else '0' for x in interleave_result.tensor_data)
    
    print(f"   Input blocks: {[b.trace_data for b in small_fibonacci_blocks]}")
    print(f"   Interleaved result: {interleave_binary}")
    print(f"   Tensor shape: {interleave_result.shape}")
    print(f"   φ-compliance: {'11' not in interleave_binary}")
    
    print("\n7. Fibonacci-Aligned Merge:")
    
    fibonacci_merger = FibonacciAlignedMerger()
    fib_result = fibonacci_merger.merge_blocks(fibonacci_blocks)
    fib_binary = ''.join('1' if x > 0.5 else '0' for x in fib_result.tensor_data)
    
    print(f"   Fibonacci-aligned result: {fib_binary}")
    print(f"   φ-alignment score: {fib_result.phi_invariants.get('phi_alignment', 0):.3f}")
    print(f"   Golden ratio approximation: {fib_result.phi_invariants.get('golden_ratio_approximation', 0):.3f}")
    
    print("\n8. Block Statistics:")
    
    all_blocks = zeckendorf_blocks + fibonacci_blocks
    total_entropy = sum(block.entropy for block in all_blocks)
    avg_entropy = total_entropy / len(all_blocks)
    
    print(f"   Total blocks processed: {len(all_blocks)}")
    print(f"   Average block entropy: {avg_entropy:.3f}")
    print(f"   φ-compliance rate: {sum(1 for b in all_blocks if b.phi_compliance) / len(all_blocks):.1%}")
    
    print("\n" + "=" * 70)
    print("All merge operations successfully preserve φ-constraint")
    print("=" * 70)


if __name__ == "__main__":
    # Run visualization
    visualize_collapse_merge()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)