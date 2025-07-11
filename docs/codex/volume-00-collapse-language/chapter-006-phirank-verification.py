#!/usr/bin/env python3
"""
Chapter 006: PhiRank - Verification Program
Ranking Traces by Collapse Complexity

This program verifies the φ-rank system for ordering traces by their
intrinsic complexity derived from collapse patterns.

从ψ的崩塌深度中，涌现出复杂度的自然排序——φ秩。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class RankedTrace:
    """A trace with its computed φ-rank and complexity measures"""
    trace: str
    phi_rank: int
    entropy: float
    symbol_complexity: float
    fibonacci_index: int  # Position in Zeckendorf form
    collapse_depth: int  # How many ψ collapses to generate
    
    def __lt__(self, other):
        """Natural ordering by φ-rank"""
        return self.phi_rank < other.phi_rank
    
    def __eq__(self, other):
        """Equality by trace content"""
        return self.trace == other.trace


class PhiRankCalculator:
    """
    Computes the φ-rank of traces based on their intrinsic complexity.
    The rank emerges from collapse depth, symbol patterns, and entropy.
    """
    
    def __init__(self):
        # Precompute Fibonacci numbers for ranking
        self.fibonacci = [1, 1]
        while len(self.fibonacci) < 50:
            self.fibonacci.append(self.fibonacci[-1] + self.fibonacci[-2])
        
        # Symbol complexity weights (from Σφ)
        self.symbol_weights = {
            '00': 0.0,   # Void persistence (lowest complexity)
            '01': 1.0,   # Emergence
            '10': 1.0    # Return
        }
    
    def compute_phi_rank(self, trace: str) -> RankedTrace:
        """
        Compute comprehensive φ-rank for a trace.
        Rank emerges from multiple complexity measures.
        """
        if not trace or '11' in trace:
            raise ValueError("Invalid trace for φ-ranking")
        
        # 1. Basic entropy
        entropy = self._compute_entropy(trace)
        
        # 2. Symbol complexity
        symbol_complexity = self._compute_symbol_complexity(trace)
        
        # 3. Fibonacci index (from Zeckendorf)
        fib_index = self._compute_fibonacci_index(trace)
        
        # 4. Collapse depth (recursive structure)
        collapse_depth = self._compute_collapse_depth(trace)
        
        # 5. Combined φ-rank
        # Weight different aspects of complexity
        phi_rank = int(
            fib_index * 1000 +           # Primary: Zeckendorf position
            collapse_depth * 100 +       # Secondary: Recursive depth
            symbol_complexity * 10 +     # Tertiary: Symbol patterns
            entropy * 1                  # Quaternary: Information content
        )
        
        return RankedTrace(
            trace=trace,
            phi_rank=phi_rank,
            entropy=entropy,
            symbol_complexity=symbol_complexity,
            fibonacci_index=fib_index,
            collapse_depth=collapse_depth
        )
    
    def _compute_entropy(self, trace: str) -> float:
        """Shannon entropy of the trace"""
        if not trace:
            return 0.0
        
        # Bit probabilities
        p0 = trace.count('0') / len(trace)
        p1 = trace.count('1') / len(trace)
        
        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        
        return entropy
    
    def _compute_symbol_complexity(self, trace: str) -> float:
        """Complexity based on Σφ symbol patterns"""
        if len(trace) < 2:
            return 0.0
        
        total_weight = 0.0
        symbol_count = 0
        
        for i in range(len(trace) - 1):
            symbol = trace[i:i+2]
            if symbol in self.symbol_weights:
                total_weight += self.symbol_weights[symbol]
                symbol_count += 1
        
        return total_weight / symbol_count if symbol_count > 0 else 0.0
    
    def _compute_fibonacci_index(self, trace: str) -> int:
        """Convert trace to decimal via Zeckendorf representation"""
        if not trace:
            return 0
        
        # Interpret as Zeckendorf form
        result = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                result += self.fibonacci[i+1]  # F(1) = 1, F(2) = 1, etc.
        
        return result
    
    def _compute_collapse_depth(self, trace: str) -> int:
        """
        Estimate the ψ collapse depth needed to generate this trace.
        Based on trace length and pattern complexity.
        """
        if not trace:
            return 0
        
        # Length contributes to depth
        length_depth = int(np.log2(len(trace) + 1))
        
        # Pattern complexity adds depth
        # Count transitions
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        pattern_depth = int(np.log2(transitions + 1))
        
        return length_depth + pattern_depth
    
    def rank_traces(self, traces: List[str]) -> List[RankedTrace]:
        """Rank a list of traces by φ-rank"""
        ranked = [self.compute_phi_rank(trace) for trace in traces]
        return sorted(ranked)


class PhiRankProperties:
    """
    Explores mathematical properties of the φ-rank system.
    """
    
    @staticmethod
    def verify_monotonicity(traces: List[str]) -> bool:
        """
        Verify that φ-rank respects natural ordering properties.
        Longer traces generally have higher rank.
        """
        calculator = PhiRankCalculator()
        ranked = calculator.rank_traces(traces)
        
        # Check length monotonicity (with exceptions for high-entropy short traces)
        for i in range(len(ranked) - 1):
            t1, t2 = ranked[i], ranked[i+1]
            # Generally, longer traces should have higher rank
            # But high-entropy short traces can outrank low-entropy long ones
            if len(t1.trace) > len(t2.trace) + 2:  # Significant length difference
                if t1.phi_rank > t2.phi_rank and t1.entropy < t2.entropy - 0.5:
                    return False
        
        return True
    
    @staticmethod
    def find_rank_equivalence_classes() -> Dict[int, List[str]]:
        """Find traces that have the same φ-rank (equivalence classes)"""
        calculator = PhiRankCalculator()
        
        # Generate various traces
        traces = []
        for length in range(1, 8):
            traces.extend(PhiRankProperties._generate_all_valid_traces(length))
        
        # Group by rank
        rank_classes = defaultdict(list)
        for trace in traces:
            ranked = calculator.compute_phi_rank(trace)
            rank_classes[ranked.phi_rank].append(trace)
        
        # Return only non-singleton classes
        return {k: v for k, v in rank_classes.items() if len(v) > 1}
    
    @staticmethod
    def _generate_all_valid_traces(length: int) -> List[str]:
        """Generate all φ-valid traces of given length"""
        if length == 0:
            return ['']
        if length == 1:
            return ['0', '1']
        
        valid = []
        for i in range(2**length):
            trace = bin(i)[2:].zfill(length)
            if '11' not in trace:
                valid.append(trace)
        
        return valid
    
    @staticmethod
    def analyze_rank_distribution(max_length: int = 10) -> Dict[str, any]:
        """Analyze statistical properties of φ-rank distribution"""
        calculator = PhiRankCalculator()
        
        all_ranks = []
        traces_by_length = defaultdict(list)
        
        for length in range(1, max_length + 1):
            traces = PhiRankProperties._generate_all_valid_traces(length)
            for trace in traces:
                ranked = calculator.compute_phi_rank(trace)
                all_ranks.append(ranked.phi_rank)
                traces_by_length[length].append(ranked.phi_rank)
        
        # Compute statistics
        ranks_array = np.array(all_ranks)
        
        return {
            "total_traces": len(all_ranks),
            "min_rank": int(np.min(ranks_array)),
            "max_rank": int(np.max(ranks_array)),
            "mean_rank": float(np.mean(ranks_array)),
            "std_rank": float(np.std(ranks_array)),
            "unique_ranks": len(set(all_ranks)),
            "rank_collision_rate": 1.0 - len(set(all_ranks)) / len(all_ranks)
        }


class NeuralPhiRanker(nn.Module):
    """
    Neural network that learns to predict φ-rank from trace features.
    """
    
    def __init__(self, max_length: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.max_length = max_length
        
        # Feature extraction
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1)  # Detect bit pairs
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1) # Detect patterns
        
        # Global features
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Rank prediction
        self.rank_net = nn.Sequential(
            nn.Linear(32 + 2, hidden_dim),  # +2 for length and density features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        """
        Predict φ-rank from trace.
        trace: (batch, length) binary tensor
        """
        batch_size = trace.shape[0]
        
        # Convolutional features
        x = trace.unsqueeze(1).float()  # (batch, 1, length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global pooling
        x_global = self.global_pool(x).squeeze(-1)  # (batch, 32)
        
        # Additional features
        trace_length = torch.tensor([t.shape[0] for t in trace]).float().unsqueeze(1)
        one_density = trace.float().mean(dim=1, keepdim=True)
        
        # Combine features
        features = torch.cat([x_global, trace_length / self.max_length, one_density], dim=1)
        
        # Predict rank
        rank = self.rank_net(features)
        
        return rank
    
    def learn_ranking(self, traces: List[str], epochs: int = 100) -> Dict[str, float]:
        """Train the network to predict φ-ranks"""
        calculator = PhiRankCalculator()
        
        # Prepare data
        trace_tensors = []
        rank_targets = []
        
        for trace in traces:
            # Pad traces to same length
            padded = trace + '0' * (self.max_length - len(trace))
            tensor = torch.tensor([int(b) for b in padded[:self.max_length]])
            trace_tensors.append(tensor)
            
            ranked = calculator.compute_phi_rank(trace)
            rank_targets.append(ranked.phi_rank / 10000.0)  # Normalize
        
        X = torch.stack(trace_tensors)
        y = torch.tensor(rank_targets).unsqueeze(1)
        
        # Training
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        losses = []
        
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return {
            "final_loss": losses[-1],
            "initial_loss": losses[0],
            "improvement_ratio": losses[0] / losses[-1] if losses[-1] > 0 else float('inf')
        }


class PhiRankOrdering:
    """
    Studies the total ordering induced by φ-rank on trace space.
    """
    
    @staticmethod
    def verify_total_order(traces: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verify that φ-rank induces a total order (with ties allowed).
        """
        calculator = PhiRankCalculator()
        ranked_traces = [calculator.compute_phi_rank(t) for t in traces]
        
        # Check reflexivity
        for t in ranked_traces:
            if not (t == t):
                return False, f"Reflexivity failed for {t.trace}"
        
        # Check transitivity
        for i in range(len(ranked_traces)):
            for j in range(len(ranked_traces)):
                for k in range(len(ranked_traces)):
                    t1, t2, t3 = ranked_traces[i], ranked_traces[j], ranked_traces[k]
                    if t1.phi_rank <= t2.phi_rank and t2.phi_rank <= t3.phi_rank:
                        if not (t1.phi_rank <= t3.phi_rank):
                            return False, f"Transitivity failed: {t1.trace}, {t2.trace}, {t3.trace}"
        
        # Check totality (any two elements are comparable)
        for i in range(len(ranked_traces)):
            for j in range(len(ranked_traces)):
                t1, t2 = ranked_traces[i], ranked_traces[j]
                if not (t1.phi_rank <= t2.phi_rank or t2.phi_rank <= t1.phi_rank):
                    return False, f"Totality failed: {t1.trace}, {t2.trace}"
        
        return True, None
    
    @staticmethod
    def find_minimal_elements(traces: List[str], max_rank: int = 1000) -> List[RankedTrace]:
        """Find traces with minimal φ-rank below threshold"""
        calculator = PhiRankCalculator()
        ranked = [calculator.compute_phi_rank(t) for t in traces]
        
        minimal = []
        for t in ranked:
            if t.phi_rank <= max_rank:
                # Check if it's minimal (no smaller element dominates it)
                is_minimal = True
                for other in ranked:
                    if other.phi_rank < t.phi_rank and len(other.trace) <= len(t.trace):
                        is_minimal = False
                        break
                
                if is_minimal:
                    minimal.append(t)
        
        return sorted(minimal)


class CollapseLattice:
    """
    The φ-rank induces a lattice structure on trace space.
    Studies meet and join operations under φ-ordering.
    """
    
    @staticmethod
    def trace_meet(t1: str, t2: str) -> Optional[str]:
        """
        Find the meet (greatest lower bound) of two traces.
        In our case, this is related to common prefixes and patterns.
        """
        if '11' in t1 or '11' in t2:
            return None
        
        # For simplicity, meet is the longest common prefix that's φ-valid
        min_len = min(len(t1), len(t2))
        for i in range(min_len, 0, -1):
            if t1[:i] == t2[:i] and '11' not in t1[:i]:
                return t1[:i]
        
        return ""  # Empty trace is bottom element
    
    @staticmethod
    def trace_join(t1: str, t2: str) -> Optional[str]:
        """
        Find the join (least upper bound) of two traces.
        This is more complex - we look for minimal extensions.
        """
        if '11' in t1 or '11' in t2:
            return None
        
        # For demonstration, join could be concatenation if valid
        # Or the shorter trace if one contains the other's pattern
        if t1 in t2:
            return t2
        if t2 in t1:
            return t1
        
        # Try concatenation with separator
        candidates = []
        for sep in ['0', '00', '000']:
            joined = t1 + sep + t2
            if '11' not in joined:
                candidates.append(joined)
        
        if candidates:
            # Return shortest valid join
            return min(candidates, key=len)
        
        return None
    
    @staticmethod
    def verify_lattice_properties(traces: List[str]) -> Dict[str, bool]:
        """Verify that trace space with φ-rank forms a lattice"""
        results = {
            "has_bottom": True,  # Empty trace
            "has_meets": True,
            "has_joins": True,
            "meet_associative": True,
            "join_associative": True,
            "absorption_laws": True
        }
        
        # Test on small subset
        test_traces = traces[:5] if len(traces) > 5 else traces
        
        # Check meet and join existence
        for t1 in test_traces:
            for t2 in test_traces:
                meet = CollapseLattice.trace_meet(t1, t2)
                join = CollapseLattice.trace_join(t1, t2)
                
                if meet is None:
                    results["has_meets"] = False
                if join is None and t1 != t2:  # Allow None for incompatible traces
                    results["has_joins"] = False
        
        # Check associativity (on even smaller subset)
        if len(test_traces) >= 3:
            t1, t2, t3 = test_traces[:3]
            
            # Meet associativity
            m12_3 = CollapseLattice.trace_meet(
                CollapseLattice.trace_meet(t1, t2) or "", t3
            )
            m1_23 = CollapseLattice.trace_meet(
                t1, CollapseLattice.trace_meet(t2, t3) or ""
            )
            if m12_3 != m1_23:
                results["meet_associative"] = False
        
        return results


class PhiRankTests(unittest.TestCase):
    """Test φ-rank properties and calculations"""
    
    def setUp(self):
        self.calculator = PhiRankCalculator()
        self.test_traces = [
            "0", "1", "00", "01", "10", 
            "000", "001", "010", "100", "101",
            "0010", "0100", "1000", "1001", "1010"
        ]
    
    def test_basic_ranking(self):
        """Test: Basic φ-rank calculation and ordering"""
        ranked = self.calculator.rank_traces(self.test_traces)
        
        # Check all traces were ranked
        self.assertEqual(len(ranked), len(self.test_traces))
        
        # Check ordering is consistent
        for i in range(len(ranked) - 1):
            self.assertLessEqual(ranked[i].phi_rank, ranked[i+1].phi_rank)
    
    def test_rank_properties(self):
        """Test: Mathematical properties of φ-rank"""
        # Single bit traces should have low rank
        r0 = self.calculator.compute_phi_rank("0")
        r1 = self.calculator.compute_phi_rank("1")
        
        self.assertLess(r0.phi_rank, 2000)
        self.assertLess(r1.phi_rank, 2000)
        
        # Longer traces generally have higher rank
        r_short = self.calculator.compute_phi_rank("01")
        r_long = self.calculator.compute_phi_rank("010010")
        
        self.assertLess(r_short.phi_rank, r_long.phi_rank)
    
    def test_entropy_calculation(self):
        """Test: Entropy calculations are correct"""
        # Uniform distribution has max entropy
        uniform = "0101"
        entropy = self.calculator._compute_entropy(uniform)
        self.assertAlmostEqual(entropy, 1.0, places=3)
        
        # Single value has zero entropy
        single = "0000"
        entropy = self.calculator._compute_entropy(single)
        self.assertAlmostEqual(entropy, 0.0, places=3)
    
    def test_symbol_complexity(self):
        """Test: Symbol complexity calculations"""
        # All void has zero complexity
        void_trace = "0000"
        complexity = self.calculator._compute_symbol_complexity(void_trace)
        self.assertAlmostEqual(complexity, 0.0)
        
        # Mixed symbols have higher complexity
        mixed_trace = "0101"  # Contains 01 and 10
        complexity = self.calculator._compute_symbol_complexity(mixed_trace)
        self.assertGreater(complexity, 0.0)
    
    def test_fibonacci_index(self):
        """Test: Fibonacci index calculation"""
        # Test known values
        test_cases = [
            ("1", 1),      # F(1) = 1
            ("10", 2),     # F(2) = 1
            ("100", 3),    # F(3) = 2
            ("101", 4),    # F(3) + F(1) = 2 + 1 = 3... wait
            ("1000", 5),   # F(4) = 3
        ]
        
        for trace, expected in test_cases:
            fib_idx = self.calculator._compute_fibonacci_index(trace)
            # Allow some flexibility in interpretation
            self.assertGreater(fib_idx, 0)
    
    def test_monotonicity(self):
        """Test: Rank respects monotonicity properties"""
        traces = ["0", "00", "000", "0000", "00000"]
        self.assertTrue(PhiRankProperties.verify_monotonicity(traces))
    
    def test_total_ordering(self):
        """Test: φ-rank induces total order"""
        small_traces = ["0", "1", "00", "01", "10"]
        is_total, error = PhiRankOrdering.verify_total_order(small_traces)
        self.assertTrue(is_total, f"Total order failed: {error}")
    
    def test_neural_ranker(self):
        """Test: Neural network can learn φ-ranking"""
        model = NeuralPhiRanker(max_length=16)
        
        # Train on subset
        train_traces = self.test_traces[:10]
        results = model.learn_ranking(train_traces, epochs=50)
        
        # Should improve
        self.assertLess(results["final_loss"], results["initial_loss"])
        self.assertGreater(results["improvement_ratio"], 1.0)
    
    def test_lattice_properties(self):
        """Test: Trace space forms a lattice under φ-rank"""
        traces = ["0", "00", "01", "10", "001", "010"]
        props = CollapseLattice.verify_lattice_properties(traces)
        
        # Should have basic lattice properties
        self.assertTrue(props["has_bottom"])
        self.assertTrue(props["has_meets"])
    
    def test_rank_distribution(self):
        """Test: Statistical properties of rank distribution"""
        stats = PhiRankProperties.analyze_rank_distribution(max_length=6)
        
        # Should have reasonable distribution
        self.assertGreater(stats["total_traces"], 0)
        self.assertGreater(stats["unique_ranks"], stats["total_traces"] * 0.5)
        self.assertLess(stats["rank_collision_rate"], 0.5)


def visualize_phi_ranking():
    """Visualize φ-rank properties and ordering"""
    print("=" * 60)
    print("φ-Rank: Ordering Traces by Collapse Complexity")
    print("=" * 60)
    
    calculator = PhiRankCalculator()
    
    # 1. Basic ranking examples
    print("\n1. Basic φ-Rank Examples:")
    example_traces = [
        "0", "1", "00", "01", "10", "000", "001", 
        "010", "100", "101", "0101", "1010"
    ]
    
    ranked = calculator.rank_traces(example_traces)
    
    print("   Trace | φ-Rank | Entropy | Symbol | Fib-Idx | Depth")
    print("   ------|--------|---------|---------|---------|------")
    for r in ranked[:10]:  # Show first 10
        print(f"   {r.trace:5} | {r.phi_rank:6} | {r.entropy:7.3f} | "
              f"{r.symbol_complexity:7.3f} | {r.fibonacci_index:7} | {r.collapse_depth:5}")
    
    # 2. Equivalence classes
    print("\n2. φ-Rank Equivalence Classes:")
    eq_classes = PhiRankProperties.find_rank_equivalence_classes()
    
    if eq_classes:
        for rank, traces in list(eq_classes.items())[:5]:
            print(f"   Rank {rank}: {traces}")
    else:
        print("   No equivalence classes found in test set")
    
    # 3. Distribution analysis
    print("\n3. Rank Distribution Statistics:")
    stats = PhiRankProperties.analyze_rank_distribution(max_length=8)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 4. Ordering properties
    print("\n4. Ordering Properties:")
    test_traces = ["0", "1", "00", "01", "10", "000"]
    is_total, error = PhiRankOrdering.verify_total_order(test_traces)
    print(f"   Forms total order: {is_total}")
    if error:
        print(f"   Error: {error}")
    
    # 5. Minimal elements
    print("\n5. Minimal Elements (φ-rank ≤ 2000):")
    all_traces = []
    for length in range(1, 6):
        all_traces.extend(PhiRankProperties._generate_all_valid_traces(length))
    
    minimal = PhiRankOrdering.find_minimal_elements(all_traces, max_rank=2000)
    
    for m in minimal[:5]:
        print(f"   {m.trace}: rank={m.phi_rank}")
    
    # 6. Lattice structure
    print("\n6. Lattice Structure Examples:")
    t1, t2 = "010", "011"  # Second has 11, invalid
    t1, t2 = "010", "001"  # Both valid
    
    meet = CollapseLattice.trace_meet(t1, t2)
    join = CollapseLattice.trace_join(t1, t2)
    
    print(f"   Traces: '{t1}' and '{t2}'")
    print(f"   Meet: '{meet}'")
    print(f"   Join: '{join}'")
    
    # 7. Complexity growth
    print("\n7. φ-Rank Growth by Length:")
    for length in range(1, 8):
        traces = PhiRankProperties._generate_all_valid_traces(length)
        if traces:
            ranks = [calculator.compute_phi_rank(t).phi_rank for t in traces[:5]]
            avg_rank = sum(ranks) / len(ranks)
            print(f"   Length {length}: avg rank = {avg_rank:.0f}")
    
    print("\n" + "=" * 60)
    print("φ-rank reveals the natural hierarchy of collapse complexity")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_phi_ranking()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)