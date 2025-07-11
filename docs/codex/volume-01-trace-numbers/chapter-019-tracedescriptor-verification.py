#!/usr/bin/env python3
"""
Chapter 019: TraceDescriptor - Verification Program
High-Level Descriptor Functions for Trace T^n and Tensor Property Analysis

This program verifies that trace tensors from CollapseMerge can be analyzed
through high-level descriptor functions that capture essential properties,
enabling efficient computation and pattern recognition in φ-space.

从ψ的张量形式中，涌现出描述符函数——追踪张量的高级分析工具。
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


class TraceDescriptorError(Exception):
    """Custom exception for trace descriptor errors"""
    pass


@dataclass
class TraceTensor:
    """Higher-order tensor structure from merged trace blocks"""
    tensor_data: torch.Tensor           # The tensor representation
    trace_blocks: List[str]             # Contributing trace blocks
    tensor_rank: int                    # Order of the tensor (T^n)
    shape: Tuple[int, ...]             # Full tensor shape
    phi_invariants: Dict[str, float]    # φ-constraint invariants
    merge_history: List[Dict[str, Any]] # History of merge operations
    structural_properties: Dict[str, Any] # Structural analysis results


class DescriptorType(Enum):
    """Types of trace descriptors"""
    STATISTICAL = "statistical"         # Mean, variance, entropy
    STRUCTURAL = "structural"           # Rank, sparsity, connectivity
    ALGEBRAIC = "algebraic"             # Eigenvalues, determinant
    TOPOLOGICAL = "topological"         # Holes, components, genus
    SPECTRAL = "spectral"               # Fourier coefficients
    ENTROPIC = "entropic"               # Shannon, Renyi, Tsallis
    GEOMETRIC = "geometric"             # Curvature, dimension
    SEMANTIC = "semantic"               # Meaning vectors


class StatisticalDescriptor:
    """
    Compute statistical descriptors for trace tensors
    capturing distributional properties and moments.
    """
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, float]:
        """Compute comprehensive statistical descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float for statistics
        
        descriptors = {
            # Basic statistics
            'mean': float(tensor.mean()) if tensor.numel() > 0 else 0.0,
            'variance': float(tensor.var()) if tensor.numel() > 0 else 0.0,
            'std': float(tensor.std()) if tensor.numel() > 0 else 0.0,
            'skewness': self._compute_skewness(tensor) if tensor.numel() > 0 else 0.0,
            'kurtosis': self._compute_kurtosis(tensor) if tensor.numel() > 0 else 0.0,
            
            # φ-related statistics
            'phi_alignment': self._compute_phi_alignment(tensor) if tensor.numel() > 0 else 0.0,
            'golden_ratio_score': self._compute_golden_score(tensor) if tensor.numel() > 0 else 0.0,
            
            # Distribution properties
            'entropy': self._compute_entropy(tensor) if tensor.numel() > 0 else 0.0,
            'sparsity': self._compute_sparsity(tensor),
            'dynamic_range': self._compute_dynamic_range(tensor) if tensor.numel() > 0 else 0.0,
            
            # Higher-order moments
            'third_moment': self._compute_moment(tensor, 3) if tensor.numel() > 0 else 0.0,
            'fourth_moment': self._compute_moment(tensor, 4) if tensor.numel() > 0 else 0.0,
            
            # Quantile statistics
            'median': float(tensor.median()) if tensor.numel() > 0 else 0.0,
            'q1': float(tensor.quantile(0.25)) if tensor.numel() > 0 else 0.0,
            'q3': float(tensor.quantile(0.75)) if tensor.numel() > 0 else 0.0,
            'iqr': float(tensor.quantile(0.75) - tensor.quantile(0.25)) if tensor.numel() > 0 else 0.0
        }
        
        return descriptors
    
    def _compute_skewness(self, tensor: torch.Tensor) -> float:
        """Compute skewness (third standardized moment)"""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        return float(((tensor - mean) ** 3).mean() / (std ** 3))
    
    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis (fourth standardized moment)"""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        return float(((tensor - mean) ** 4).mean() / (std ** 4))
    
    def _compute_phi_alignment(self, tensor: torch.Tensor) -> float:
        """Compute alignment with golden ratio properties"""
        # Analyze ratio of consecutive elements
        if tensor.numel() < 2:
            return 0.0
        
        flat = tensor.flatten()
        ratios = []
        
        for i in range(len(flat) - 1):
            if flat[i] != 0:
                ratio = abs(flat[i+1] / flat[i])
                ratios.append(ratio)
        
        if not ratios:
            return 0.0
        
        # Measure deviation from φ
        deviations = [abs(r - self.phi) / self.phi for r in ratios]
        avg_deviation = np.mean(deviations)
        
        return max(0.0, 1.0 - avg_deviation)
    
    def _compute_golden_score(self, tensor: torch.Tensor) -> float:
        """Compute overall golden ratio alignment score"""
        zeros = (tensor == 0).sum().float()
        ones = (tensor == 1).sum().float()
        
        if ones > 0:
            ratio = zeros / ones
            deviation = abs(ratio - self.phi) / self.phi
            return max(0.0, 1.0 - deviation)
        
        return 0.0
    
    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute Shannon entropy"""
        # Normalize to probability distribution
        flat = tensor.flatten()
        if flat.sum() == 0:
            return 0.0
        
        probs = flat / flat.sum()
        probs = probs[probs > 0]  # Remove zeros
        
        entropy = -float((probs * torch.log2(probs)).sum())
        return entropy
    
    def _compute_sparsity(self, tensor: torch.Tensor) -> float:
        """Compute sparsity (fraction of zeros)"""
        if tensor.numel() == 0:
            return 0.0
        return float((tensor == 0).sum()) / tensor.numel()
    
    def _compute_dynamic_range(self, tensor: torch.Tensor) -> float:
        """Compute dynamic range"""
        return float(tensor.max() - tensor.min())
    
    def _compute_moment(self, tensor: torch.Tensor, order: int) -> float:
        """Compute raw moment of given order"""
        return float((tensor ** order).mean())


class StructuralDescriptor:
    """
    Analyze structural properties of trace tensors
    including rank, connectivity, and hierarchical features.
    """
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute structural descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float
        
        descriptors = {
            # Rank properties
            'effective_rank': self._compute_effective_rank(tensor),
            'numerical_rank': self._compute_numerical_rank(tensor),
            'rank_ratio': self._compute_rank_ratio(tensor),
            
            # Connectivity analysis
            'connectivity_score': self._compute_connectivity(trace_tensor),
            'clustering_coefficient': self._compute_clustering(tensor),
            
            # Hierarchical properties
            'depth': trace_tensor.tensor_rank,
            'branching_factor': self._compute_branching_factor(trace_tensor),
            'complexity': self._compute_structural_complexity(trace_tensor),
            
            # Pattern analysis
            'pattern_density': self._compute_pattern_density(tensor),
            'symmetry_score': self._compute_symmetry(tensor),
            'regularity': self._compute_regularity(tensor)
        }
        
        return descriptors
    
    def _compute_effective_rank(self, tensor: torch.Tensor) -> int:
        """Compute effective rank via SVD"""
        if tensor.dim() < 2:
            return 1
        
        # Reshape to matrix for SVD
        matrix = tensor.view(tensor.shape[0], -1)
        
        try:
            _, s, _ = torch.svd(matrix)
            # Count significant singular values
            threshold = s[0] * 1e-10 if s[0] > 0 else 1e-10
            effective_rank = (s > threshold).sum().item()
            return effective_rank
        except:
            return min(matrix.shape)
    
    def _compute_numerical_rank(self, tensor: torch.Tensor) -> float:
        """Compute numerical rank (entropy of singular values)"""
        if tensor.dim() < 2:
            return 1.0
        
        matrix = tensor.view(tensor.shape[0], -1)
        
        try:
            _, s, _ = torch.svd(matrix)
            s = s[s > 0]  # Keep positive singular values
            
            if len(s) == 0:
                return 0.0
            
            # Normalize
            s_norm = s / s.sum()
            
            # Compute entropy
            entropy = -float((s_norm * torch.log2(s_norm)).sum())
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(len(s))
            if max_entropy > 0:
                return math.exp(entropy) / len(s)
            else:
                return 1.0
        except:
            return 1.0
    
    def _compute_rank_ratio(self, tensor: torch.Tensor) -> float:
        """Compute ratio of effective rank to maximum possible rank"""
        if tensor.dim() < 2:
            return 1.0
        
        effective = self._compute_effective_rank(tensor)
        max_rank = min(tensor.shape[0], tensor.view(tensor.shape[0], -1).shape[1])
        
        return effective / max_rank if max_rank > 0 else 0.0
    
    def _compute_connectivity(self, trace_tensor: TraceTensor) -> float:
        """Compute connectivity between trace blocks"""
        if len(trace_tensor.trace_blocks) < 2:
            return 0.0
        
        # Analyze connections in merge history
        connections = 0
        possible_connections = len(trace_tensor.trace_blocks) * (len(trace_tensor.trace_blocks) - 1) / 2
        
        for merge_op in trace_tensor.merge_history:
            if 'connected_blocks' in merge_op:
                connections += len(merge_op['connected_blocks'])
        
        return connections / possible_connections if possible_connections > 0 else 0.0
    
    def _compute_clustering(self, tensor: torch.Tensor) -> float:
        """Compute clustering coefficient"""
        # Treat tensor as adjacency matrix of graph
        if tensor.dim() < 2:
            return 0.0
        
        # Threshold to create adjacency
        adj = (tensor.abs() > tensor.abs().mean()).float()
        
        # Compute clustering coefficient
        if adj.shape[0] != adj.shape[1]:
            # Make square by padding or truncating
            size = min(adj.shape[0], adj.shape[1])
            adj = adj[:size, :size]
        
        # Count triangles
        adj_squared = torch.matmul(adj, adj)
        adj_cubed = torch.matmul(adj_squared, adj)
        triangles = torch.trace(adj_cubed) / 6
        
        # Count connected triples
        degrees = adj.sum(dim=1)
        triples = (degrees * (degrees - 1)).sum() / 2
        
        return float(triangles / triples) if triples > 0 else 0.0
    
    def _compute_branching_factor(self, trace_tensor: TraceTensor) -> float:
        """Compute average branching factor in tensor structure"""
        if not trace_tensor.merge_history:
            return 1.0
        
        branching_factors = []
        for merge_op in trace_tensor.merge_history:
            if 'input_count' in merge_op:
                branching_factors.append(merge_op['input_count'])
        
        return np.mean(branching_factors) if branching_factors else 1.0
    
    def _compute_structural_complexity(self, trace_tensor: TraceTensor) -> float:
        """Compute overall structural complexity"""
        # Combine multiple factors
        rank_complexity = trace_tensor.tensor_rank / 10.0  # Normalize
        size_complexity = math.log2(trace_tensor.tensor_data.numel() + 1) / 20.0
        history_complexity = len(trace_tensor.merge_history) / 10.0
        
        return min(1.0, (rank_complexity + size_complexity + history_complexity) / 3)
    
    def _compute_pattern_density(self, tensor: torch.Tensor) -> float:
        """Compute density of non-trivial patterns"""
        # Look for repeated subsequences
        flat = tensor.flatten()
        if len(flat) < 4:
            return 0.0
        
        patterns = set()
        for length in range(2, min(5, len(flat))):
            for i in range(len(flat) - length + 1):
                pattern = tuple(flat[i:i+length].tolist())
                if len(set(pattern)) > 1:  # Non-trivial pattern
                    patterns.add(pattern)
        
        return len(patterns) / len(flat) if len(flat) > 0 else 0.0
    
    def _compute_symmetry(self, tensor: torch.Tensor) -> float:
        """Compute symmetry score"""
        if tensor.dim() < 2:
            return 1.0
        
        # Check various symmetries
        symmetries = []
        
        # Transpose symmetry
        if tensor.shape[0] == tensor.shape[1]:
            transpose_diff = (tensor - tensor.T).abs().mean()
            symmetries.append(1.0 - float(transpose_diff))
        
        # Flip symmetries
        for dim in range(tensor.dim()):
            flipped = torch.flip(tensor, dims=[dim])
            flip_diff = (tensor - flipped).abs().mean()
            symmetries.append(1.0 - float(flip_diff))
        
        return np.mean(symmetries) if symmetries else 0.0
    
    def _compute_regularity(self, tensor: torch.Tensor) -> float:
        """Compute regularity of tensor structure"""
        # Analyze variance in local regions
        if tensor.numel() < 4:
            return 1.0
        
        # Compute local variances
        kernel_size = min(3, min(tensor.shape))
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        
        # Use average pooling to get local means
        local_means = F.avg_pool1d(tensor.float(), kernel_size=kernel_size, stride=1)
        
        # Compute variance of local means
        variance = local_means.var()
        
        # Lower variance means more regular
        return float(torch.exp(-variance))


class AlgebraicDescriptor:
    """
    Compute algebraic descriptors including eigenvalues,
    determinants, and other matrix properties.
    """
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute algebraic descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float
        
        descriptors = {}
        
        # Matrix-based descriptors
        if tensor.dim() >= 2:
            matrix = self._to_square_matrix(tensor)
            
            descriptors.update({
                'eigenvalue_spectrum': self._compute_eigenvalues(matrix),
                'spectral_radius': self._compute_spectral_radius(matrix),
                'condition_number': self._compute_condition_number(matrix),
                'determinant_log': self._compute_log_determinant(matrix),
                'trace': float(torch.trace(matrix)),
                'frobenius_norm': float(torch.norm(matrix, 'fro'))
            })
        
        # General tensor properties
        if tensor.numel() > 0:
            descriptors.update({
                'nuclear_norm': self._compute_nuclear_norm(tensor),
                'operator_norm': self._compute_operator_norm(tensor),
                'tensor_rank_decomposition': self._estimate_tensor_rank(tensor)
            })
        else:
            descriptors.update({
                'nuclear_norm': 0.0,
                'operator_norm': 0.0,
                'tensor_rank_decomposition': 0
            })
        
        return descriptors
    
    def _to_square_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to square matrix for analysis"""
        if tensor.dim() == 2 and tensor.shape[0] == tensor.shape[1]:
            return tensor
        
        # Reshape to closest square matrix
        n = int(math.sqrt(tensor.numel()))
        if n * n < tensor.numel():
            n += 1
        
        flat = tensor.flatten()
        # Pad if necessary
        if flat.numel() < n * n:
            padding = torch.zeros(n * n - flat.numel())
            flat = torch.cat([flat, padding])
        
        return flat[:n*n].view(n, n)
    
    def _compute_eigenvalues(self, matrix: torch.Tensor) -> Dict[str, float]:
        """Compute eigenvalue spectrum properties"""
        try:
            eigenvalues = torch.linalg.eigvals(matrix.float())
            eigenvalues_real = eigenvalues.real
            eigenvalues_imag = eigenvalues.imag
            
            return {
                'largest_real': float(eigenvalues_real.max()),
                'smallest_real': float(eigenvalues_real.min()),
                'largest_magnitude': float(eigenvalues.abs().max()),
                'eigenvalue_gap': float(eigenvalues.abs().max() - eigenvalues.abs().min()),
                'imaginary_component': float(eigenvalues_imag.abs().mean())
            }
        except:
            return {
                'largest_real': 0.0,
                'smallest_real': 0.0,
                'largest_magnitude': 0.0,
                'eigenvalue_gap': 0.0,
                'imaginary_component': 0.0
            }
    
    def _compute_spectral_radius(self, matrix: torch.Tensor) -> float:
        """Compute spectral radius (largest eigenvalue magnitude)"""
        try:
            eigenvalues = torch.linalg.eigvals(matrix.float())
            return float(eigenvalues.abs().max())
        except:
            return 0.0
    
    def _compute_condition_number(self, matrix: torch.Tensor) -> float:
        """Compute condition number"""
        try:
            return float(torch.linalg.cond(matrix.float()))
        except:
            return float('inf')
    
    def _compute_log_determinant(self, matrix: torch.Tensor) -> float:
        """Compute log determinant for numerical stability"""
        try:
            sign, logdet = torch.linalg.slogdet(matrix.float())
            return float(sign * logdet)
        except:
            return float('-inf')
    
    def _compute_nuclear_norm(self, tensor: torch.Tensor) -> float:
        """Compute nuclear norm (sum of singular values)"""
        if tensor.dim() < 2:
            return float(tensor.abs().sum())
        
        matrix = tensor.view(tensor.shape[0], -1)
        try:
            _, s, _ = torch.svd(matrix.float())
            return float(s.sum())
        except:
            return float(tensor.abs().sum())
    
    def _compute_operator_norm(self, tensor: torch.Tensor) -> float:
        """Compute operator norm (largest singular value)"""
        if tensor.dim() < 2:
            return float(tensor.abs().max())
        
        matrix = tensor.view(tensor.shape[0], -1)
        try:
            _, s, _ = torch.svd(matrix.float())
            return float(s.max())
        except:
            return float(tensor.abs().max())
    
    def _estimate_tensor_rank(self, tensor: torch.Tensor) -> int:
        """Estimate tensor rank via CP decomposition approximation"""
        # Simplified estimation based on unfolding
        if tensor.dim() < 3:
            return self._compute_effective_rank(tensor.view(tensor.shape[0], -1))
        
        # For higher-order tensors, estimate via unfolding ranks
        ranks = []
        for mode in range(tensor.dim()):
            unfolding = self._unfold_tensor(tensor, mode)
            rank = self._compute_effective_rank(unfolding)
            ranks.append(rank)
        
        return min(ranks)
    
    def _unfold_tensor(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """Unfold tensor along specified mode"""
        shape = list(tensor.shape)
        n = shape[mode]
        
        # Move mode to front and flatten rest
        perm = [mode] + [i for i in range(len(shape)) if i != mode]
        return tensor.permute(perm).contiguous().view(n, -1)
    
    def _compute_effective_rank(self, matrix: torch.Tensor) -> int:
        """Compute effective rank of matrix"""
        try:
            _, s, _ = torch.svd(matrix.float())
            threshold = s[0] * 1e-10 if s[0] > 0 else 1e-10
            return int((s > threshold).sum())
        except:
            return min(matrix.shape)


class SpectralDescriptor:
    """
    Compute spectral descriptors using Fourier analysis
    and frequency domain properties.
    """
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute spectral descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float
        
        # Handle empty tensor case
        if tensor.numel() == 0:
            return {
                'spectral_energy': 0.0,
                'dominant_frequency': {'frequency': 0.0, 'magnitude': 0.0, 'relative_power': 0.0},
                'spectral_entropy': 0.0,
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'low_freq_energy': 0.0,
                'mid_freq_energy': 0.0,
                'high_freq_energy': 0.0,
                'spectral_flatness': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_flux': 0.0
            }
        
        # Compute FFT along each dimension
        spectral_data = self._compute_multidimensional_fft(tensor)
        
        descriptors = {
            # Frequency domain statistics
            'spectral_energy': self._compute_spectral_energy(spectral_data),
            'dominant_frequency': self._find_dominant_frequency(spectral_data),
            'spectral_entropy': self._compute_spectral_entropy(spectral_data),
            'spectral_centroid': self._compute_spectral_centroid(spectral_data),
            'spectral_spread': self._compute_spectral_spread(spectral_data),
            
            # Band analysis
            'low_freq_energy': self._compute_band_energy(spectral_data, 'low'),
            'mid_freq_energy': self._compute_band_energy(spectral_data, 'mid'),
            'high_freq_energy': self._compute_band_energy(spectral_data, 'high'),
            
            # Spectral features
            'spectral_flatness': self._compute_spectral_flatness(spectral_data),
            'spectral_rolloff': self._compute_spectral_rolloff(spectral_data),
            'spectral_flux': self._compute_spectral_flux(spectral_data)
        }
        
        return descriptors
    
    def _compute_multidimensional_fft(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute FFT along all dimensions"""
        # Convert to complex for FFT
        tensor_complex = tensor.to(torch.complex64)
        
        # Apply FFT along each dimension
        fft_result = tensor_complex
        for dim in range(tensor.dim()):
            fft_result = torch.fft.fft(fft_result, dim=dim)
        
        return fft_result.abs()  # Return magnitude spectrum
    
    def _compute_spectral_energy(self, spectral_data: torch.Tensor) -> float:
        """Compute total spectral energy"""
        return float((spectral_data ** 2).sum())
    
    def _find_dominant_frequency(self, spectral_data: torch.Tensor) -> Dict[str, float]:
        """Find dominant frequency components"""
        flat_spectrum = spectral_data.flatten()
        
        # Find peak
        peak_idx = torch.argmax(flat_spectrum)
        peak_value = flat_spectrum[peak_idx]
        
        # Convert index to normalized frequency
        normalized_freq = float(peak_idx) / len(flat_spectrum)
        
        return {
            'frequency': normalized_freq,
            'magnitude': float(peak_value),
            'relative_power': float(peak_value ** 2 / (flat_spectrum ** 2).sum())
        }
    
    def _compute_spectral_entropy(self, spectral_data: torch.Tensor) -> float:
        """Compute entropy of spectrum"""
        # Normalize to probability distribution
        power_spectrum = spectral_data ** 2
        total_power = power_spectrum.sum()
        
        if total_power == 0:
            return 0.0
        
        prob_spectrum = power_spectrum / total_power
        prob_spectrum = prob_spectrum[prob_spectrum > 0]
        
        entropy = -float((prob_spectrum * torch.log2(prob_spectrum)).sum())
        return entropy
    
    def _compute_spectral_centroid(self, spectral_data: torch.Tensor) -> float:
        """Compute spectral centroid (center of mass)"""
        flat_spectrum = spectral_data.flatten()
        frequencies = torch.arange(len(flat_spectrum), dtype=torch.float32)
        
        total_energy = flat_spectrum.sum()
        if total_energy == 0:
            return 0.0
        
        centroid = (frequencies * flat_spectrum).sum() / total_energy
        return float(centroid) / len(flat_spectrum)  # Normalize
    
    def _compute_spectral_spread(self, spectral_data: torch.Tensor) -> float:
        """Compute spectral spread (standard deviation around centroid)"""
        flat_spectrum = spectral_data.flatten()
        frequencies = torch.arange(len(flat_spectrum), dtype=torch.float32)
        
        centroid = self._compute_spectral_centroid(spectral_data) * len(flat_spectrum)
        
        total_energy = flat_spectrum.sum()
        if total_energy == 0:
            return 0.0
        
        variance = ((frequencies - centroid) ** 2 * flat_spectrum).sum() / total_energy
        return float(torch.sqrt(variance)) / len(flat_spectrum)  # Normalize
    
    def _compute_band_energy(self, spectral_data: torch.Tensor, band: str) -> float:
        """Compute energy in frequency band"""
        flat_spectrum = spectral_data.flatten()
        n = len(flat_spectrum)
        
        if band == 'low':
            band_spectrum = flat_spectrum[:n//3]
        elif band == 'mid':
            band_spectrum = flat_spectrum[n//3:2*n//3]
        else:  # high
            band_spectrum = flat_spectrum[2*n//3:]
        
        band_energy = (band_spectrum ** 2).sum()
        total_energy = (flat_spectrum ** 2).sum()
        
        return float(band_energy / total_energy) if total_energy > 0 else 0.0
    
    def _compute_spectral_flatness(self, spectral_data: torch.Tensor) -> float:
        """Compute spectral flatness (geometric mean / arithmetic mean)"""
        flat_spectrum = spectral_data.flatten()
        flat_spectrum = flat_spectrum[flat_spectrum > 0]  # Remove zeros
        
        if len(flat_spectrum) == 0:
            return 0.0
        
        geometric_mean = torch.exp(torch.log(flat_spectrum).mean())
        arithmetic_mean = flat_spectrum.mean()
        
        return float(geometric_mean / arithmetic_mean) if arithmetic_mean > 0 else 0.0
    
    def _compute_spectral_rolloff(self, spectral_data: torch.Tensor) -> float:
        """Compute spectral rolloff (frequency below which 85% of energy is contained)"""
        flat_spectrum = spectral_data.flatten()
        power_spectrum = flat_spectrum ** 2
        
        cumsum = torch.cumsum(power_spectrum, dim=0)
        total_energy = cumsum[-1]
        
        if total_energy == 0:
            return 0.0
        
        rolloff_point = 0.85 * total_energy
        rolloff_idx = torch.searchsorted(cumsum, rolloff_point)
        
        return float(rolloff_idx) / len(flat_spectrum)
    
    def _compute_spectral_flux(self, spectral_data: torch.Tensor) -> float:
        """Compute spectral flux (rate of change)"""
        if spectral_data.dim() < 2:
            return 0.0
        
        # Compute difference along first dimension
        diff = spectral_data[1:] - spectral_data[:-1]
        flux = (diff ** 2).mean()
        
        return float(flux)


class TopologicalDescriptor:
    """
    Compute topological descriptors capturing
    holes, components, and persistent features.
    """
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute topological descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float
        
        # Handle empty tensor case
        if tensor.numel() == 0:
            return {
                'betti_0': 0,
                'betti_1': 0,
                'persistence_entropy': 0.0,
                'total_persistence': 0.0,
                'euler_characteristic': 0,
                'genus': 0,
                'void_fraction': 0.0,
                'hole_complexity': 0.0
            }
        
        descriptors = {
            # Betti numbers (simplified)
            'betti_0': self._compute_connected_components(tensor),
            'betti_1': self._estimate_loops(tensor),
            
            # Persistent homology features
            'persistence_entropy': self._compute_persistence_entropy(tensor),
            'total_persistence': self._compute_total_persistence(tensor),
            
            # Graph topology (if applicable)
            'euler_characteristic': self._compute_euler_characteristic(tensor),
            'genus': self._estimate_genus(tensor),
            
            # Structural holes
            'void_fraction': self._compute_void_fraction(tensor),
            'hole_complexity': self._compute_hole_complexity(tensor)
        }
        
        return descriptors
    
    def _compute_connected_components(self, tensor: torch.Tensor) -> int:
        """Compute number of connected components (Betti 0)"""
        # Threshold tensor to create binary structure
        binary = (tensor.abs() > tensor.abs().mean()).float()
        
        # Simple connected component analysis
        if binary.dim() == 1:
            # Count runs of 1s
            diff = torch.diff(torch.cat([torch.zeros(1), binary, torch.zeros(1)]))
            starts = (diff == 1).sum()
            return int(starts)
        
        # For higher dimensions, use simplified analysis
        return self._count_regions(binary)
    
    def _estimate_loops(self, tensor: torch.Tensor) -> int:
        """Estimate number of loops (Betti 1)"""
        if tensor.dim() < 2:
            return 0
        
        # Create adjacency structure
        adj = (tensor.abs() > tensor.abs().mean()).float()
        
        # Estimate cycles using Euler characteristic
        n_vertices = adj.shape[0]
        n_edges = int(adj.sum() / 2)
        
        # For connected graph: loops = edges - vertices + 1
        # This is simplified estimation
        loops = max(0, n_edges - n_vertices + 1)
        
        return loops
    
    def _compute_persistence_entropy(self, tensor: torch.Tensor) -> float:
        """Compute entropy of persistence diagram"""
        # Simplified persistence analysis
        values = tensor.abs().flatten()
        values = values[values > 0]
        
        if len(values) == 0:
            return 0.0
        
        # Use values as persistence lifetimes
        lifetimes = values / values.sum()
        
        entropy = -float((lifetimes * torch.log2(lifetimes)).sum())
        return entropy
    
    def _compute_total_persistence(self, tensor: torch.Tensor) -> float:
        """Compute total persistence"""
        # Sum of all persistence lifetimes
        return float(tensor.abs().sum())
    
    def _compute_euler_characteristic(self, tensor: torch.Tensor) -> int:
        """Compute Euler characteristic"""
        if tensor.dim() < 2:
            return 1
        
        # V - E + F for 2D
        binary = (tensor.abs() > tensor.abs().mean()).float()
        
        vertices = int((binary > 0).sum())
        edges = int(self._count_edges(binary))
        faces = int(self._count_faces(binary))
        
        return vertices - edges + faces
    
    def _estimate_genus(self, tensor: torch.Tensor) -> int:
        """Estimate genus of surface"""
        euler_char = self._compute_euler_characteristic(tensor)
        
        # For closed surface: χ = 2 - 2g
        # So g = (2 - χ) / 2
        genus = max(0, (2 - euler_char) // 2)
        
        return genus
    
    def _compute_void_fraction(self, tensor: torch.Tensor) -> float:
        """Compute fraction of void space"""
        threshold = tensor.abs().mean()
        voids = (tensor.abs() < threshold * 0.1).float()
        
        return float(voids.sum() / tensor.numel())
    
    def _compute_hole_complexity(self, tensor: torch.Tensor) -> float:
        """Compute complexity of hole structure"""
        # Analyze distribution of void sizes
        binary = (tensor.abs() < tensor.abs().mean() * 0.1).float()
        
        if binary.sum() == 0:
            return 0.0
        
        # Count different void sizes (simplified)
        void_sizes = []
        flat = binary.flatten()
        
        current_size = 0
        for val in flat:
            if val == 1:
                current_size += 1
            elif current_size > 0:
                void_sizes.append(current_size)
                current_size = 0
        
        if not void_sizes:
            return 0.0
        
        # Complexity based on size distribution entropy
        void_tensor = torch.tensor(void_sizes, dtype=torch.float32)
        void_probs = void_tensor / void_tensor.sum()
        
        complexity = -float((void_probs * torch.log2(void_probs)).sum())
        return complexity / math.log2(len(void_sizes)) if len(void_sizes) > 1 else 0.0
    
    def _count_regions(self, binary: torch.Tensor) -> int:
        """Count connected regions in binary tensor"""
        # Simplified region counting
        return max(1, int(binary.sum() / 10))  # Rough estimate
    
    def _count_edges(self, binary: torch.Tensor) -> int:
        """Count edges in binary structure"""
        # Count adjacent pairs
        edges = 0
        
        for dim in range(binary.dim()):
            shifted = torch.roll(binary, shifts=1, dims=dim)
            edges += int((binary * shifted).sum())
        
        return edges // 2  # Each edge counted twice
    
    def _count_faces(self, binary: torch.Tensor) -> int:
        """Count faces in binary structure"""
        # Simplified face counting
        if binary.dim() < 2:
            return 0
        
        # Count 2x2 blocks
        faces = 0
        if binary.dim() >= 2:
            for i in range(binary.shape[0] - 1):
                for j in range(binary.shape[1] - 1):
                    if binary[i, j] and binary[i+1, j] and binary[i, j+1] and binary[i+1, j+1]:
                        faces += 1
        
        return faces


class SemanticDescriptor:
    """
    Compute semantic descriptors using neural embeddings
    and learned representations.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.encoder = self._build_encoder()
        self.phi = (1 + math.sqrt(5)) / 2
    
    def _build_encoder(self) -> nn.Module:
        """Build neural encoder for semantic embedding"""
        return nn.Sequential(
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh()
        )
    
    def compute_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute semantic descriptors"""
        tensor = trace_tensor.tensor_data.float()  # Convert to float
        
        # Generate embedding
        embedding = self._generate_embedding(tensor)
        
        descriptors = {
            # Embedding properties
            'embedding_norm': float(torch.norm(embedding)),
            'embedding_sparsity': float((embedding.abs() < 0.1).sum()) / embedding.numel(),
            
            # Semantic scores
            'complexity_score': self._compute_complexity_score(embedding),
            'regularity_score': self._compute_regularity_score(embedding),
            'uniqueness_score': self._compute_uniqueness_score(embedding),
            
            # Interpretability
            'dominant_features': self._extract_dominant_features(embedding),
            'feature_entropy': self._compute_feature_entropy(embedding),
            
            # φ-semantic alignment
            'phi_semantic_score': self._compute_phi_semantic_alignment(embedding)
        }
        
        return descriptors
    
    def _generate_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate semantic embedding from tensor"""
        # Prepare input
        flat = tensor.flatten()
        
        # Pad or truncate to fixed size
        if len(flat) < 128:
            padded = F.pad(flat, (0, 128 - len(flat)))
        else:
            padded = flat[:128]
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.encoder(padded.float())
        
        return embedding
    
    def _compute_complexity_score(self, embedding: torch.Tensor) -> float:
        """Compute semantic complexity from embedding"""
        # Complexity based on embedding entropy
        probs = F.softmax(embedding.abs(), dim=0)
        entropy = -float((probs * torch.log2(probs + 1e-10)).sum())
        
        # Normalize
        max_entropy = math.log2(len(embedding))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_regularity_score(self, embedding: torch.Tensor) -> float:
        """Compute semantic regularity"""
        # Regularity based on autocorrelation
        autocorr = torch.nn.functional.conv1d(
            embedding.unsqueeze(0).unsqueeze(0),
            embedding.unsqueeze(0).unsqueeze(0),
            padding=len(embedding)//2
        )
        
        # Peak ratio
        autocorr_flat = autocorr.flatten()
        peak = autocorr_flat.max()
        mean = autocorr_flat.mean()
        
        return float(mean / peak) if peak > 0 else 0.0
    
    def _compute_uniqueness_score(self, embedding: torch.Tensor) -> float:
        """Compute uniqueness score"""
        # Uniqueness based on distance from mean embedding
        mean_embedding = torch.zeros_like(embedding)  # Placeholder for mean
        distance = torch.norm(embedding - mean_embedding)
        
        # Normalize by expected distance
        expected_distance = math.sqrt(self.embedding_dim)
        
        return float(distance / expected_distance)
    
    def _extract_dominant_features(self, embedding: torch.Tensor) -> List[int]:
        """Extract indices of dominant features"""
        # Get top-k features
        k = min(5, len(embedding))
        _, indices = torch.topk(embedding.abs(), k)
        
        return indices.tolist()
    
    def _compute_feature_entropy(self, embedding: torch.Tensor) -> float:
        """Compute entropy of feature distribution"""
        # Treat embedding as feature weights
        weights = F.softmax(embedding.abs(), dim=0)
        entropy = -float((weights * torch.log2(weights + 1e-10)).sum())
        
        return entropy
    
    def _compute_phi_semantic_alignment(self, embedding: torch.Tensor) -> float:
        """Compute alignment with golden ratio in semantic space"""
        # Analyze ratios in embedding
        ratios = []
        for i in range(len(embedding) - 1):
            if embedding[i].abs() > 0.01:
                ratio = abs(embedding[i+1] / embedding[i])
                ratios.append(float(ratio))
        
        if not ratios:
            return 0.0
        
        # Measure deviation from φ
        deviations = [abs(r - self.phi) / self.phi for r in ratios]
        avg_deviation = np.mean(deviations)
        
        return max(0.0, 1.0 - avg_deviation)


class DescriptorAggregator:
    """
    Aggregate all descriptor types into comprehensive
    feature vector for trace tensors.
    """
    
    def __init__(self):
        self.statistical = StatisticalDescriptor()
        self.structural = StructuralDescriptor()
        self.algebraic = AlgebraicDescriptor()
        self.spectral = SpectralDescriptor()
        self.topological = TopologicalDescriptor()
        self.semantic = SemanticDescriptor()
    
    def compute_all_descriptors(self, trace_tensor: TraceTensor) -> Dict[str, Any]:
        """Compute all descriptor types"""
        all_descriptors = {
            'statistical': self.statistical.compute_descriptors(trace_tensor),
            'structural': self.structural.compute_descriptors(trace_tensor),
            'algebraic': self.algebraic.compute_descriptors(trace_tensor),
            'spectral': self.spectral.compute_descriptors(trace_tensor),
            'topological': self.topological.compute_descriptors(trace_tensor),
            'semantic': self.semantic.compute_descriptors(trace_tensor)
        }
        
        # Add meta-descriptors
        all_descriptors['meta'] = self._compute_meta_descriptors(all_descriptors)
        
        return all_descriptors
    
    def _compute_meta_descriptors(self, descriptors: Dict[str, Any]) -> Dict[str, float]:
        """Compute meta-level descriptors from all categories"""
        meta = {}
        
        # Complexity index (average of complexity measures)
        complexity_scores = []
        if 'statistical' in descriptors:
            complexity_scores.append(descriptors['statistical'].get('entropy', 0))
        if 'structural' in descriptors:
            complexity_scores.append(descriptors['structural'].get('complexity', 0))
        if 'semantic' in descriptors:
            complexity_scores.append(descriptors['semantic'].get('complexity_score', 0))
        
        meta['overall_complexity'] = np.mean(complexity_scores) if complexity_scores else 0.0
        
        # φ-alignment index
        phi_scores = []
        if 'statistical' in descriptors:
            phi_scores.append(descriptors['statistical'].get('phi_alignment', 0))
            phi_scores.append(descriptors['statistical'].get('golden_ratio_score', 0))
        if 'semantic' in descriptors:
            phi_scores.append(descriptors['semantic'].get('phi_semantic_score', 0))
        
        meta['overall_phi_alignment'] = np.mean(phi_scores) if phi_scores else 0.0
        
        # Regularity index
        regularity_scores = []
        if 'structural' in descriptors:
            regularity_scores.append(descriptors['structural'].get('regularity', 0))
        if 'semantic' in descriptors:
            regularity_scores.append(descriptors['semantic'].get('regularity_score', 0))
        
        meta['overall_regularity'] = np.mean(regularity_scores) if regularity_scores else 0.0
        
        # Information content
        info_scores = []
        if 'statistical' in descriptors:
            info_scores.append(descriptors['statistical'].get('entropy', 0))
        if 'spectral' in descriptors:
            info_scores.append(descriptors['spectral'].get('spectral_entropy', 0))
        if 'topological' in descriptors:
            info_scores.append(descriptors['topological'].get('persistence_entropy', 0))
        
        meta['overall_information'] = np.mean(info_scores) if info_scores else 0.0
        
        return meta


class TraceDescriptorTests(unittest.TestCase):
    """Test trace descriptor functionality"""
    
    def setUp(self):
        self.aggregator = DescriptorAggregator()
        
        # Create test trace tensors
        self.simple_tensor = TraceTensor(
            tensor_data=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            trace_blocks=['10', '10'],
            tensor_rank=1,
            shape=(4,),
            phi_invariants={'phi_compliant': True},
            merge_history=[],
            structural_properties={}
        )
        
        self.complex_tensor = TraceTensor(
            tensor_data=torch.randn(8, 8),
            trace_blocks=['1010', '1000', '10100', '10001'],
            tensor_rank=2,
            shape=(8, 8),
            phi_invariants={'phi_compliant': True, 'golden_score': 0.8},
            merge_history=[
                {'operation': 'merge', 'input_count': 2},
                {'operation': 'merge', 'input_count': 2}
            ],
            structural_properties={'depth': 3}
        )
    
    def test_statistical_descriptors(self):
        """Test: Statistical descriptor computation"""
        stats = self.aggregator.statistical.compute_descriptors(self.simple_tensor)
        
        # Check basic statistics
        self.assertAlmostEqual(stats['mean'], 0.5, places=3)
        self.assertAlmostEqual(stats['variance'], 0.3333, places=3)  # corrected variance for [1,0,1,0]
        self.assertIn('entropy', stats)
        self.assertIn('phi_alignment', stats)
    
    def test_structural_descriptors(self):
        """Test: Structural descriptor computation"""
        struct = self.aggregator.structural.compute_descriptors(self.complex_tensor)
        
        # Check structural properties
        self.assertIn('effective_rank', struct)
        self.assertIn('connectivity_score', struct)
        self.assertEqual(struct['depth'], 2)  # tensor_rank
        self.assertIn('pattern_density', struct)
    
    def test_algebraic_descriptors(self):
        """Test: Algebraic descriptor computation"""
        alg = self.aggregator.algebraic.compute_descriptors(self.complex_tensor)
        
        # Check algebraic properties
        self.assertIn('eigenvalue_spectrum', alg)
        self.assertIn('spectral_radius', alg)
        self.assertIn('trace', alg)
        self.assertIn('nuclear_norm', alg)
    
    def test_spectral_descriptors(self):
        """Test: Spectral descriptor computation"""
        spec = self.aggregator.spectral.compute_descriptors(self.simple_tensor)
        
        # Check spectral properties
        self.assertIn('spectral_energy', spec)
        self.assertIn('dominant_frequency', spec)
        self.assertIn('spectral_entropy', spec)
        self.assertIn('low_freq_energy', spec)
    
    def test_topological_descriptors(self):
        """Test: Topological descriptor computation"""
        topo = self.aggregator.topological.compute_descriptors(self.complex_tensor)
        
        # Check topological properties
        self.assertIn('betti_0', topo)
        self.assertIn('betti_1', topo)
        self.assertIn('euler_characteristic', topo)
        self.assertIn('void_fraction', topo)
    
    def test_semantic_descriptors(self):
        """Test: Semantic descriptor computation"""
        sem = self.aggregator.semantic.compute_descriptors(self.simple_tensor)
        
        # Check semantic properties
        self.assertIn('embedding_norm', sem)
        self.assertIn('complexity_score', sem)
        self.assertIn('dominant_features', sem)
        self.assertIn('phi_semantic_score', sem)
    
    def test_descriptor_aggregation(self):
        """Test: Full descriptor aggregation"""
        all_desc = self.aggregator.compute_all_descriptors(self.complex_tensor)
        
        # Check all categories present
        self.assertIn('statistical', all_desc)
        self.assertIn('structural', all_desc)
        self.assertIn('algebraic', all_desc)
        self.assertIn('spectral', all_desc)
        self.assertIn('topological', all_desc)
        self.assertIn('semantic', all_desc)
        self.assertIn('meta', all_desc)
        
        # Check meta descriptors
        meta = all_desc['meta']
        self.assertIn('overall_complexity', meta)
        self.assertIn('overall_phi_alignment', meta)
    
    def test_edge_cases(self):
        """Test: Edge cases and error handling"""
        # Empty tensor
        empty_tensor = TraceTensor(
            tensor_data=torch.zeros(0),
            trace_blocks=[],
            tensor_rank=0,
            shape=(0,),
            phi_invariants={},
            merge_history=[],
            structural_properties={}
        )
        
        # Should not raise errors
        desc = self.aggregator.compute_all_descriptors(empty_tensor)
        self.assertIsInstance(desc, dict)
        
        # Single element tensor
        single_tensor = TraceTensor(
            tensor_data=torch.tensor([1.0]),
            trace_blocks=['1'],
            tensor_rank=1,
            shape=(1,),
            phi_invariants={'phi_compliant': True},
            merge_history=[],
            structural_properties={}
        )
        
        desc = self.aggregator.compute_all_descriptors(single_tensor)
        self.assertIsInstance(desc, dict)
    
    def test_descriptor_consistency(self):
        """Test: Descriptor consistency across multiple runs"""
        desc1 = self.aggregator.compute_all_descriptors(self.simple_tensor)
        desc2 = self.aggregator.compute_all_descriptors(self.simple_tensor)
        
        # Statistical descriptors should be identical
        for key in desc1['statistical']:
            if isinstance(desc1['statistical'][key], (int, float)):
                self.assertAlmostEqual(
                    desc1['statistical'][key],
                    desc2['statistical'][key],
                    places=5
                )
    
    def test_phi_alignment_computation(self):
        """Test: φ-alignment computation accuracy"""
        # Create tensor with golden ratio pattern
        golden_tensor = TraceTensor(
            tensor_data=torch.tensor([1.0, 1.618, 2.618, 4.236]),
            trace_blocks=['1', '10', '100', '1000'],
            tensor_rank=1,
            shape=(4,),
            phi_invariants={'phi_compliant': True},
            merge_history=[],
            structural_properties={}
        )
        
        stats = self.aggregator.statistical.compute_descriptors(golden_tensor)
        
        # Should have high φ-alignment
        self.assertGreater(stats['phi_alignment'], 0.8)


def visualize_trace_descriptors():
    """Visualize trace descriptor analysis"""
    print("=" * 70)
    print("TraceDescriptor: High-Level Analysis of Trace Tensors")
    print("=" * 70)
    
    aggregator = DescriptorAggregator()
    
    # Create example tensors
    examples = {
        'Simple Binary': TraceTensor(
            tensor_data=torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]),
            trace_blocks=['1010', '0101', '1010', '0101'],
            tensor_rank=2,
            shape=(4, 4),
            phi_invariants={'phi_compliant': True},
            merge_history=[],
            structural_properties={}
        ),
        'Golden Spiral': TraceTensor(
            tensor_data=torch.tensor([1, 1, 2, 3, 5, 8, 13, 21], dtype=torch.float32),
            trace_blocks=['1', '1', '10', '100', '1000', '10000', '100000', '1000000'],
            tensor_rank=1,
            shape=(8,),
            phi_invariants={'phi_compliant': True, 'golden_score': 1.0},
            merge_history=[],
            structural_properties={'type': 'fibonacci'}
        ),
        'Random Complex': TraceTensor(
            tensor_data=torch.randn(6, 6),
            trace_blocks=['10100', '10010', '10001', '100001', '1000001', '10000001'],
            tensor_rank=2,
            shape=(6, 6),
            phi_invariants={'phi_compliant': True},
            merge_history=[
                {'operation': 'concatenate', 'input_count': 3},
                {'operation': 'interleave', 'input_count': 2}
            ],
            structural_properties={'complexity': 'high'}
        )
    }
    
    for name, tensor in examples.items():
        print(f"\n{name} Tensor Analysis:")
        print("-" * 50)
        
        descriptors = aggregator.compute_all_descriptors(tensor)
        
        # Display key descriptors from each category
        print("\nStatistical Descriptors:")
        stats = descriptors['statistical']
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Variance: {stats['variance']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  φ-Alignment: {stats['phi_alignment']:.4f}")
        print(f"  Golden Score: {stats['golden_ratio_score']:.4f}")
        
        print("\nStructural Descriptors:")
        struct = descriptors['structural']
        print(f"  Effective Rank: {struct['effective_rank']}")
        print(f"  Connectivity: {struct['connectivity_score']:.4f}")
        print(f"  Pattern Density: {struct['pattern_density']:.4f}")
        print(f"  Symmetry Score: {struct['symmetry_score']:.4f}")
        
        print("\nAlgebraic Descriptors:")
        alg = descriptors['algebraic']
        if 'spectral_radius' in alg:
            print(f"  Spectral Radius: {alg['spectral_radius']:.4f}")
        if 'trace' in alg:
            print(f"  Trace: {alg['trace']:.4f}")
        print(f"  Nuclear Norm: {alg['nuclear_norm']:.4f}")
        
        print("\nSpectral Descriptors:")
        spec = descriptors['spectral']
        print(f"  Spectral Energy: {spec['spectral_energy']:.4f}")
        print(f"  Spectral Entropy: {spec['spectral_entropy']:.4f}")
        print(f"  Low Freq Energy: {spec['low_freq_energy']:.4f}")
        print(f"  High Freq Energy: {spec['high_freq_energy']:.4f}")
        
        print("\nTopological Descriptors:")
        topo = descriptors['topological']
        print(f"  Connected Components: {topo['betti_0']}")
        print(f"  Loops: {topo['betti_1']}")
        print(f"  Euler Characteristic: {topo['euler_characteristic']}")
        print(f"  Void Fraction: {topo['void_fraction']:.4f}")
        
        print("\nSemantic Descriptors:")
        sem = descriptors['semantic']
        print(f"  Complexity Score: {sem['complexity_score']:.4f}")
        print(f"  Regularity Score: {sem['regularity_score']:.4f}")
        print(f"  Uniqueness Score: {sem['uniqueness_score']:.4f}")
        print(f"  φ-Semantic Score: {sem['phi_semantic_score']:.4f}")
        
        print("\nMeta Descriptors:")
        meta = descriptors['meta']
        print(f"  Overall Complexity: {meta['overall_complexity']:.4f}")
        print(f"  Overall φ-Alignment: {meta['overall_phi_alignment']:.4f}")
        print(f"  Overall Regularity: {meta['overall_regularity']:.4f}")
        print(f"  Overall Information: {meta['overall_information']:.4f}")
    
    print("\n" + "=" * 70)
    print("Descriptor Categories Summary:")
    print("=" * 70)
    
    print("\n1. Statistical: Distribution properties, moments, entropy")
    print("2. Structural: Rank, connectivity, patterns, symmetry")
    print("3. Algebraic: Eigenvalues, determinants, matrix norms")
    print("4. Spectral: Frequency domain, Fourier analysis")
    print("5. Topological: Holes, components, persistent features")
    print("6. Semantic: Neural embeddings, learned representations")
    print("7. Meta: Cross-category aggregations and indices")
    
    print("\n" + "=" * 70)
    print("All trace tensors successfully analyzed with comprehensive descriptors")
    print("=" * 70)


if __name__ == "__main__":
    # Run visualization
    visualize_trace_descriptors()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)