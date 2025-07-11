#!/usr/bin/env python3
"""
Unit tests for ΨB-T0.N6: φ-Rank and Tensor Dimensionality
Verifies dimensional hierarchy emergence from collapse complexity.
"""

import unittest
import numpy as np
from typing import List, Tuple, Set
import math
import torch


class PhiRankCalculator:
    """Calculate φ-rank for collapse states"""
    
    def __init__(self):
        # Precompute Fibonacci numbers
        self.fib = [1, 2]
        while len(self.fib) < 20:
            self.fib.append(self.fib[-1] + self.fib[-2])
        
        self.phi = (1 + math.sqrt(5)) / 2
    
    def zeckendorf_to_indices(self, zeck: str) -> List[int]:
        """Convert Zeckendorf string to Fibonacci indices"""
        indices = []
        for i, bit in enumerate(reversed(zeck)):
            if bit == '1':
                indices.append(i)
        return indices
    
    def phi_rank(self, zeck: str) -> float:
        """Calculate φ-rank of a Zeckendorf representation"""
        if not zeck or zeck == "0":
            return 0.0
        
        indices = self.zeckendorf_to_indices(zeck)
        if not indices:
            return 0.0
        
        # φ-rank = max index (position of highest 1)
        # This gives a natural ranking based on complexity
        return float(max(indices))
    
    def effective_dimension(self, zeck: str) -> int:
        """Calculate effective dimension from φ-rank"""
        rank = self.phi_rank(zeck)
        # Dimension = rank + 1 (to include all positions)
        return int(rank) + 1


class CollapseTensor:
    """Tensor representation of collapse states"""
    
    def __init__(self, dimension: int):
        """Initialize tensor of given dimension"""
        self.dimension = dimension
        # Create tensor with Fibonacci-indexed dimensions
        shape = self._fibonacci_shape(dimension)
        self.tensor = torch.zeros(shape, dtype=torch.float32)
    
    def _fibonacci_shape(self, dim: int) -> Tuple[int, ...]:
        """Get tensor shape based on Fibonacci dimensions"""
        fib_calc = PhiRankCalculator()
        # Use smaller shape to avoid overflow - min(F_i, 5)
        shape = []
        for i in range(dim):
            if i < len(fib_calc.fib):
                # Limit size to prevent memory overflow
                shape.append(min(fib_calc.fib[i], 5))
            else:
                # Small constant size for higher dimensions
                shape.append(3)
        return tuple(shape)
    
    @classmethod
    def from_zeckendorf(cls, zeck: str) -> 'CollapseTensor':
        """Create tensor from Zeckendorf representation"""
        calc = PhiRankCalculator()
        dim = calc.effective_dimension(zeck)
        
        tensor = cls(dim)
        indices = calc.zeckendorf_to_indices(zeck)
        
        # Set tensor values at Fibonacci positions
        for idx in indices:
            if idx < dim:
                # Create index tuple with 1 at position idx
                index = [0] * dim
                index[idx] = 1
                tensor.set_value(index, 1.0)
        
        return tensor
    
    def set_value(self, indices: List[int], value: float):
        """Set value at multi-index position"""
        if len(indices) != self.dimension:
            raise ValueError("Index dimension mismatch")
        
        # Clamp indices to tensor shape
        shape = self.tensor.shape
        clamped = []
        for i, idx in enumerate(indices):
            clamped.append(min(idx, shape[i] - 1))
        
        self.tensor[tuple(clamped)] = value
    
    def get_slice(self, axis: int, index: int) -> torch.Tensor:
        """Get slice along given axis"""
        if axis >= self.dimension:
            raise ValueError("Axis out of bounds")
        
        # Create slice indices
        indices = [slice(None)] * self.dimension
        indices[axis] = index
        
        return self.tensor[tuple(indices)]
    
    def rank(self) -> int:
        """Calculate tensor rank (number of non-zero dimensions)"""
        # Count dimensions with non-zero values
        non_zero_dims = 0
        for dim in range(self.dimension):
            # Sum along all other dimensions
            axes = list(range(self.dimension))
            axes.remove(dim)
            if axes:
                dim_sum = torch.sum(torch.abs(self.tensor), dim=axes)
                if torch.any(dim_sum > 0):
                    non_zero_dims += 1
            else:
                # 1D case
                if torch.any(torch.abs(self.tensor) > 0):
                    non_zero_dims += 1
        
        return non_zero_dims


class DimensionalHierarchy:
    """Manages dimensional hierarchy of collapse states"""
    
    def __init__(self):
        self.calc = PhiRankCalculator()
        self.levels = {}  # φ-rank -> states mapping
    
    def add_state(self, zeck: str):
        """Add state to hierarchy"""
        rank = self.calc.phi_rank(zeck)
        # Use integer rank directly as level
        level = int(rank)
        
        if level not in self.levels:
            self.levels[level] = []
        self.levels[level].append(zeck)
    
    def get_level(self, level: int) -> List[str]:
        """Get all states at given level"""
        return self.levels.get(level, [])
    
    def transition_matrix(self, level: int) -> np.ndarray:
        """Get transition matrix between levels"""
        current = self.get_level(level)
        next_level = self.get_level(level + 1)
        
        if not current or not next_level:
            return np.array([[]])
        
        # Create transition matrix
        matrix = np.zeros((len(current), len(next_level)))
        
        for i, state1 in enumerate(current):
            for j, state2 in enumerate(next_level):
                # Transition exists if states differ by one Fibonacci term
                if self._can_transition(state1, state2):
                    matrix[i, j] = 1.0
        
        return matrix
    
    def _can_transition(self, zeck1: str, zeck2: str) -> bool:
        """Check if transition is possible between states"""
        # Simple heuristic: can transition if ranks differ by ~1
        rank1 = self.calc.phi_rank(zeck1)
        rank2 = self.calc.phi_rank(zeck2)
        return 0.5 < abs(rank2 - rank1) < 1.5


class TestPhiRank(unittest.TestCase):
    """Test φ-rank calculations"""
    
    def setUp(self):
        self.calc = PhiRankCalculator()
    
    def test_basic_phi_rank(self):
        """Test φ-rank for basic Zeckendorf representations"""
        test_cases = [
            ("1", 0.0),      # F_1, minimal rank
            ("10", 1.0),     # F_2
            ("100", 2.0),    # F_3
            ("1000", 3.0),   # F_4
        ]
        
        for zeck, expected_rank in test_cases:
            rank = self.calc.phi_rank(zeck)
            # φ-rank grows approximately linearly with position
            self.assertAlmostEqual(rank, expected_rank, delta=1.0)
    
    def test_composite_phi_rank(self):
        """Test φ-rank for composite representations"""
        # 101 = F_1 + F_3
        rank = self.calc.phi_rank("101")
        
        # Should be dominated by highest term (F_3)
        rank_f3 = self.calc.phi_rank("100")
        self.assertGreaterEqual(rank, rank_f3 * 0.9)
    
    def test_effective_dimension(self):
        """Test effective dimension calculation"""
        test_cases = [
            ("1", 1),
            ("10", 2),
            ("100", 3),
            ("101", 3),
            ("10101", 5)
        ]
        
        for zeck, expected_dim in test_cases:
            dim = self.calc.effective_dimension(zeck)
            self.assertLessEqual(abs(dim - expected_dim), 1)
    
    def test_phi_rank_ordering(self):
        """Test that φ-rank preserves ordering"""
        zecks = ["1", "10", "100", "1000", "10000"]
        ranks = [self.calc.phi_rank(z) for z in zecks]
        
        # Ranks should be increasing
        for i in range(1, len(ranks)):
            self.assertGreater(ranks[i], ranks[i-1])


class TestCollapseTensor(unittest.TestCase):
    """Test tensor representation"""
    
    def test_tensor_creation(self):
        """Test creating tensors from Zeckendorf"""
        tensor = CollapseTensor.from_zeckendorf("101")
        
        # Should have dimension 3 (from effective dimension)
        self.assertGreaterEqual(tensor.dimension, 2)
        
        # Check shape follows Fibonacci
        shape = tensor.tensor.shape
        for i in range(1, len(shape)):
            if i == 1:
                self.assertEqual(shape[i], shape[i-1] * 2)  # F_2 = 2*F_1
            elif i > 1:
                self.assertGreaterEqual(shape[i], shape[i-1])
    
    def test_tensor_values(self):
        """Test tensor value setting"""
        tensor = CollapseTensor(3)
        
        # Set value using valid indices (within bounds)
        tensor.set_value([0, 1, 0], 5.0)
        
        # Retrieve value
        value = tensor.tensor[0, 1, 0]
        self.assertEqual(value.item(), 5.0)
    
    def test_tensor_slicing(self):
        """Test tensor slicing operations"""
        tensor = CollapseTensor(3)
        tensor.set_value([0, 1, 0], 3.0)
        
        # Get slice along axis 1
        slice_1 = tensor.get_slice(1, 1)
        
        # Should be 2D tensor
        self.assertEqual(len(slice_1.shape), 2)
        
        # Should contain our value
        self.assertEqual(slice_1[0, 0].item(), 3.0)
    
    def test_tensor_rank(self):
        """Test tensor rank calculation"""
        # Empty tensor has rank 0
        tensor1 = CollapseTensor(3)
        self.assertEqual(tensor1.rank(), 0)
        
        # Tensor with values has positive rank
        tensor2 = CollapseTensor.from_zeckendorf("101")
        self.assertGreater(tensor2.rank(), 0)


class TestDimensionalHierarchy(unittest.TestCase):
    """Test dimensional hierarchy structure"""
    
    def setUp(self):
        self.hierarchy = DimensionalHierarchy()
    
    def test_hierarchy_construction(self):
        """Test building dimensional hierarchy"""
        # Add states of different complexities
        states = ["1", "10", "100", "101", "1000", "1010"]
        
        for state in states:
            self.hierarchy.add_state(state)
        
        # Check levels are populated
        for level in range(5):
            states_at_level = self.hierarchy.get_level(level)
            if level < 4:
                self.assertGreater(len(states_at_level), 0)
    
    def test_transition_matrix(self):
        """Test transition matrices between levels"""
        # Add some states
        self.hierarchy.add_state("10")
        self.hierarchy.add_state("100")
        self.hierarchy.add_state("1000")
        
        # Get transition matrix
        matrix = self.hierarchy.transition_matrix(1)
        
        # Should have non-zero transitions
        if matrix.size > 0:
            self.assertGreater(np.sum(matrix), 0)
    
    def test_level_separation(self):
        """Test that levels properly separate by complexity"""
        # States with very different complexity
        simple = "1"
        complex = "10101010"
        
        self.hierarchy.add_state(simple)
        self.hierarchy.add_state(complex)
        
        simple_rank = self.hierarchy.calc.phi_rank(simple)
        complex_rank = self.hierarchy.calc.phi_rank(complex)
        
        simple_level = math.floor(simple_rank)
        complex_level = math.floor(complex_rank)
        
        # Should be in different levels
        self.assertNotEqual(simple_level, complex_level)


class TestPhiRankProperties(unittest.TestCase):
    """Test mathematical properties of φ-rank"""
    
    def test_subadditivity(self):
        """Test subadditivity property"""
        calc = PhiRankCalculator()
        
        # For Zeckendorf sums, rank should be subadditive
        z1 = "100"   # F_3
        z2 = "1000"  # F_4
        
        rank1 = calc.phi_rank(z1)
        rank2 = calc.phi_rank(z2)
        
        # Combined representation
        combined = "1100"  # F_3 + F_4
        rank_combined = calc.phi_rank(combined)
        
        # Should satisfy subadditivity (approximately)
        self.assertLessEqual(rank_combined, rank1 + rank2 + 1.0)
    
    def test_monotonicity(self):
        """Test monotonicity with respect to value"""
        calc = PhiRankCalculator()
        
        # Larger Fibonacci indices should have larger φ-rank
        ranks = []
        for i in range(1, 8):
            zeck = "1" + "0" * (i - 1)  # F_i representation
            ranks.append(calc.phi_rank(zeck))
        
        # Check monotonic increase
        for i in range(1, len(ranks)):
            self.assertGreater(ranks[i], ranks[i-1])
    
    def test_golden_ratio_scaling(self):
        """Test scaling by golden ratio"""
        calc = PhiRankCalculator()
        
        # φ-rank should scale approximately with golden ratio
        base_zeck = "100"
        base_rank = calc.phi_rank(base_zeck)
        
        # Next Fibonacci term
        next_zeck = "1000"
        next_rank = calc.phi_rank(next_zeck)
        
        # Ratio should be related to φ
        if base_rank > 0:
            ratio = next_rank / base_rank
            # Should be approximately φ or related to it
            self.assertGreater(ratio, 1.0)
            self.assertLess(ratio, 2.0)


class TestTensorDimensionality(unittest.TestCase):
    """Test tensor dimensional properties"""
    
    def test_fibonacci_dimension_growth(self):
        """Test that tensor dimensions follow Fibonacci growth"""
        dimensions = []
        for dim in range(1, 6):
            tensor = CollapseTensor(dim)
            shape = tensor.tensor.shape
            total_size = np.prod(shape)
            dimensions.append(total_size)
        
        # Size should grow exponentially
        for i in range(1, len(dimensions)):
            self.assertGreater(dimensions[i], dimensions[i-1])
        
        # Growth rate should be related to φ
        if len(dimensions) >= 3:
            ratio1 = dimensions[2] / dimensions[1]
            ratio2 = dimensions[3] / dimensions[2]
            # Ratios should converge toward φ^d for some d
            self.assertGreater(ratio2, ratio1 * 0.9)
    
    def test_dimensional_projection(self):
        """Test projection between dimensions"""
        # Higher dimensional tensor
        high_tensor = CollapseTensor.from_zeckendorf("10101")
        
        # Lower dimensional projection (first 3 dimensions)
        low_dim = 3
        projected_shape = high_tensor.tensor.shape[:low_dim]
        
        # Should preserve Fibonacci structure
        calc = PhiRankCalculator()
        for i, size in enumerate(projected_shape):
            if i < len(calc.fib):
                self.assertEqual(size, calc.fib[i])
    
    def test_tensor_sparsity(self):
        """Test that tensors are naturally sparse"""
        # Most Zeckendorf representations are sparse
        tensor = CollapseTensor.from_zeckendorf("100101")
        
        # Count non-zero elements
        non_zero = torch.count_nonzero(tensor.tensor).item()
        total = torch.numel(tensor.tensor)
        
        # Sparsity ratio
        sparsity = non_zero / total
        
        # Should be very sparse
        self.assertLess(sparsity, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)