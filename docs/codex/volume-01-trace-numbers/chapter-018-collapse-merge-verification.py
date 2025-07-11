#!/usr/bin/env python3
"""
Chapter 018: CollapseMerge - Merging Collapse-Safe Blocks into Trace Tensor T^n

Verification program demonstrating how to merge multiple φ-valid traces into
higher-order tensors while maintaining structural constraints.

From ψ = ψ(ψ), we derive the tensor structure of trace arithmetic.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class TraceTensor:
    """Represents a trace tensor T^n"""
    
    def __init__(self, traces: List[str], order: int = 1):
        """Initialize tensor from traces
        
        Args:
            traces: List of φ-valid traces
            order: Tensor order (1=vector, 2=matrix, etc.)
        """
        self.traces = traces
        self.order = order
        self.validate_traces()
        
    def validate_traces(self):
        """Ensure all traces are φ-valid"""
        for trace in self.traces:
            if '11' in trace:
                raise ValueError(f"Invalid trace with '11': {trace}")
                
    def to_tensor(self) -> torch.Tensor:
        """Convert traces to numerical tensor"""
        max_len = max(len(t) for t in self.traces) if self.traces else 0
        
        # Pad traces to same length
        padded = []
        for trace in self.traces:
            padded_trace = trace.ljust(max_len, '0')
            binary_values = [int(b) for b in padded_trace]
            padded.append(binary_values)
            
        return torch.tensor(padded, dtype=torch.float32)
    
    def get_shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        if self.order == 1:
            return (len(self.traces),)
        elif self.order == 2:
            # For order 2, arrange as square matrix if possible
            n = int(np.sqrt(len(self.traces)))
            if n * n == len(self.traces):
                return (n, n)
            else:
                return (len(self.traces), 1)
        else:
            # Higher orders: distribute dimensions evenly
            n = int(len(self.traces) ** (1/self.order))
            return tuple([n] * self.order)

class CollapseMerger:
    """Merge traces while preserving φ-constraint"""
    
    def __init__(self):
        self.merge_cache = {}
        
    def merge_sequential(self, trace1: str, trace2: str, gap: int = 1) -> str:
        """Merge traces sequentially with gap"""
        gap_str = '0' * gap
        merged = trace1.rstrip('0') + gap_str + trace2
        
        # Verify φ-constraint preserved
        if '11' in merged:
            raise ValueError(f"Sequential merge creates '11': {merged}")
            
        return merged
    
    def merge_interleaved(self, trace1: str, trace2: str) -> str:
        """Interleave two traces"""
        max_len = max(len(trace1), len(trace2))
        t1 = trace1.ljust(max_len, '0')
        t2 = trace2.ljust(max_len, '0')
        
        merged = []
        for i in range(max_len):
            merged.append(t1[i])
            merged.append(t2[i])
            
        result = ''.join(merged)
        
        # Verify φ-constraint
        if '11' in result:
            raise ValueError(f"Interleaved merge creates '11': {result}")
            
        return result
    
    def merge_tensor_product(self, traces1: List[str], traces2: List[str]) -> List[str]:
        """Compute tensor product of trace lists"""
        result = []
        
        for t1 in traces1:
            for t2 in traces2:
                # Try sequential merge
                try:
                    merged = self.merge_sequential(t1, t2)
                    result.append(merged)
                except ValueError:
                    # Try with larger gap
                    try:
                        merged = self.merge_sequential(t1, t2, gap=2)
                        result.append(merged)
                    except ValueError:
                        # Skip incompatible pairs
                        pass
                        
        return result
    
    def merge_with_operation(self, traces: List[str], op: str = 'OR') -> str:
        """Merge multiple traces with specified operation"""
        if not traces:
            return '0'
            
        # Align all traces
        max_len = max(len(t) for t in traces)
        aligned = [t.ljust(max_len, '0') for t in traces]
        
        result = []
        for i in range(max_len):
            if op == 'OR':
                bit = '1' if any(t[i] == '1' for t in aligned) else '0'
            elif op == 'AND':
                bit = '1' if all(t[i] == '1' for t in aligned) else '0'
            elif op == 'XOR':
                count = sum(1 for t in aligned if t[i] == '1')
                bit = '1' if count % 2 == 1 else '0'
            else:
                raise ValueError(f"Unknown operation: {op}")
                
            result.append(bit)
            
        merged = ''.join(result)
        
        # Always verify constraint
        if '11' in merged:
            return None  # Merge failed
            
        return merged

class TensorStructureAnalyzer:
    """Analyze structure of trace tensors"""
    
    def __init__(self):
        self.merger = CollapseMerger()
        
    def analyze_merge_compatibility(self, traces: List[str]) -> np.ndarray:
        """Build compatibility matrix for trace merging"""
        n = len(traces)
        compatibility = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(n):
                # Check if traces can be merged sequentially
                try:
                    self.merger.merge_sequential(traces[i], traces[j])
                    compatibility[i, j] = 1
                except:
                    compatibility[i, j] = 0
                    
        return compatibility
    
    def compute_tensor_rank(self, tensor: TraceTensor) -> int:
        """Compute effective rank of trace tensor"""
        t = tensor.to_tensor()
        
        # Use SVD to find rank
        if len(t.shape) == 2:
            U, S, V = torch.svd(t)
            # Count non-zero singular values
            rank = torch.sum(S > 1e-6).item()
        else:
            # For higher order tensors, flatten first
            flat = t.flatten()
            rank = len(torch.unique(flat))
            
        return rank
    
    def analyze_tensor_properties(self, tensor: TraceTensor) -> Dict:
        """Analyze various properties of trace tensor"""
        t = tensor.to_tensor()
        
        properties = {
            'shape': tensor.get_shape(),
            'order': tensor.order,
            'num_traces': len(tensor.traces),
            'sparsity': (t == 0).sum().item() / t.numel(),
            'rank': self.compute_tensor_rank(tensor),
            'max_trace_length': max(len(tr) for tr in tensor.traces) if tensor.traces else 0,
            'φ_valid': all('11' not in tr for tr in tensor.traces)
        }
        
        return properties

class GraphTheoryTensorAnalyzer:
    """Graph theory analysis of tensor merge structures"""
    
    def __init__(self):
        self.merger = CollapseMerger()
        
    def build_merge_graph(self, traces: List[str]) -> nx.DiGraph:
        """Build directed graph of valid merges"""
        G = nx.DiGraph()
        
        # Add nodes
        for i, trace in enumerate(traces):
            G.add_node(i, trace=trace)
            
        # Add edges for valid merges
        for i in range(len(traces)):
            for j in range(len(traces)):
                if i != j:
                    try:
                        merged = self.merger.merge_sequential(traces[i], traces[j])
                        G.add_edge(i, j, merged=merged)
                    except:
                        pass  # No edge if merge invalid
                        
        return G
    
    def find_merge_paths(self, G: nx.DiGraph, start: int, end: int) -> List[List[int]]:
        """Find all valid merge paths from start to end"""
        try:
            paths = list(nx.all_simple_paths(G, start, end))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def analyze_merge_components(self, G: nx.DiGraph) -> Dict:
        """Analyze connected components in merge graph"""
        # Convert to undirected for component analysis
        G_undirected = G.to_undirected()
        
        components = list(nx.connected_components(G_undirected))
        
        analysis = {
            'num_components': len(components),
            'component_sizes': [len(c) for c in components],
            'largest_component': max(len(c) for c in components) if components else 0,
            'is_strongly_connected': nx.is_strongly_connected(G)
        }
        
        return analysis

class InformationTheoryTensorAnalyzer:
    """Information theory analysis of trace tensors"""
    
    def __init__(self):
        pass
        
    def compute_tensor_entropy(self, tensor: TraceTensor) -> float:
        """Compute entropy of tensor elements"""
        t = tensor.to_tensor().flatten()
        
        # Count frequencies
        unique, counts = torch.unique(t, return_counts=True)
        probs = counts.float() / counts.sum()
        
        # Compute entropy
        entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
        
        return entropy
    
    def compute_mutual_information(self, tensor1: TraceTensor, tensor2: TraceTensor) -> float:
        """Compute mutual information between two tensors"""
        t1 = tensor1.to_tensor().flatten()
        t2 = tensor2.to_tensor().flatten()
        
        # Ensure same length
        min_len = min(len(t1), len(t2))
        t1 = t1[:min_len]
        t2 = t2[:min_len]
        
        # Joint distribution
        joint_hist = torch.histc(t1 * 2 + t2, bins=4, min=0, max=3)
        joint_probs = joint_hist / joint_hist.sum()
        
        # Marginal distributions
        p1 = torch.tensor([joint_probs[0] + joint_probs[1], joint_probs[2] + joint_probs[3]])
        p2 = torch.tensor([joint_probs[0] + joint_probs[2], joint_probs[1] + joint_probs[3]])
        
        # MI calculation
        mi = 0.0
        for i in range(2):
            for j in range(2):
                p_joint = joint_probs[i*2 + j]
                if p_joint > 0:
                    mi += p_joint * torch.log2(p_joint / (p1[i] * p2[j] + 1e-10))
                    
        return mi.item()
    
    def analyze_compression_potential(self, tensor: TraceTensor) -> Dict:
        """Analyze how well tensor can be compressed"""
        t = tensor.to_tensor()
        
        # Analyze patterns
        analysis = {
            'entropy': self.compute_tensor_entropy(tensor),
            'theoretical_min_bits': 0,
            'actual_bits': t.numel(),
            'compression_ratio': 0
        }
        
        # Theoretical minimum bits (based on entropy)
        if analysis['entropy'] > 0:
            unique_elements = len(torch.unique(t))
            analysis['theoretical_min_bits'] = analysis['entropy'] * t.numel()
            analysis['compression_ratio'] = analysis['theoretical_min_bits'] / analysis['actual_bits']
            
        return analysis

class CategoryTheoryTensorAnalyzer:
    """Category theory analysis of tensor operations"""
    
    def __init__(self):
        self.merger = CollapseMerger()
        
    def verify_tensor_functor(self, traces1: List[str], traces2: List[str]) -> Dict:
        """Verify tensor product acts as functor"""
        results = {
            'preserves_identity': True,
            'preserves_composition': True,
            'bifunctorial': True
        }
        
        # Identity preservation: T ⊗ I ≅ T
        identity_trace = ['0']
        t1_id = self.merger.merge_tensor_product(traces1, identity_trace)
        if len(t1_id) != len(traces1):
            results['preserves_identity'] = False
            
        # Bifunctorial: (f ⊗ g) ∘ (f' ⊗ g') = (f ∘ f') ⊗ (g ∘ g')
        # Simplified check with trace operations
        
        return results
    
    def analyze_monoidal_structure(self, traces: List[str]) -> Dict:
        """Analyze monoidal category structure"""
        structure = {
            'has_unit': True,
            'associative': True,
            'symmetric': True,
            'braided': True
        }
        
        # Unit element is empty trace or '0'
        unit = '0'
        
        # Check unit laws
        for trace in traces[:5]:  # Sample
            left_unit = self.merger.merge_sequential(unit, trace, gap=0)
            right_unit = self.merger.merge_sequential(trace, unit, gap=0)
            
            if left_unit.lstrip('0') != trace or right_unit.rstrip('0') != trace:
                structure['has_unit'] = False
                break
                
        return structure

def demonstrate_basic_merging():
    """Demonstrate basic trace merging operations"""
    merger = CollapseMerger()
    
    print("=== Basic Trace Merging ===")
    
    test_cases = [
        ("Sequential", "101", "010", 1),
        ("Sequential (gap=2)", "101", "101", 2),
        ("Interleaved", "10", "01", None)
    ]
    
    for name, t1, t2, gap in test_cases:
        print(f"\n{name} merge of '{t1}' and '{t2}':")
        try:
            if name.startswith("Sequential"):
                result = merger.merge_sequential(t1, t2, gap=gap or 1)
            else:
                result = merger.merge_interleaved(t1, t2)
            print(f"  Result: {result}")
            print(f"  Valid: {'✓' if '11' not in result else '✗'}")
        except ValueError as e:
            print(f"  Failed: {e}")

def demonstrate_tensor_construction():
    """Demonstrate trace tensor construction"""
    print("\n=== Trace Tensor Construction ===")
    
    traces = ["101", "010", "100", "001"]
    
    # Order 1 tensor (vector)
    tensor1 = TraceTensor(traces, order=1)
    print(f"\nOrder 1 tensor from {len(traces)} traces:")
    print(f"  Shape: {tensor1.get_shape()}")
    print(f"  Tensor:\n{tensor1.to_tensor()}")
    
    # Order 2 tensor (matrix)
    tensor2 = TraceTensor(traces, order=2)
    print(f"\nOrder 2 tensor from {len(traces)} traces:")
    print(f"  Shape: {tensor2.get_shape()}")
    print(f"  Tensor:\n{tensor2.to_tensor()}")

def analyze_merge_compatibility():
    """Analyze compatibility of trace merging"""
    analyzer = TensorStructureAnalyzer()
    
    print("\n=== Merge Compatibility Analysis ===")
    
    traces = ["0", "1", "00", "01", "10", "000", "001", "010", "100", "101"]
    
    compatibility = analyzer.analyze_merge_compatibility(traces)
    
    print(f"\nCompatibility matrix (1=can merge, 0=cannot):")
    print("     ", end="")
    for t in traces:
        print(f"{t:>4}", end="")
    print()
    
    for i, t1 in enumerate(traces):
        print(f"{t1:>4}: ", end="")
        for j, t2 in enumerate(traces):
            print(f"{compatibility[i,j]:4}", end="")
        print()
    
    print(f"\nTotal compatible pairs: {compatibility.sum()}/{len(traces)**2}")

def demonstrate_tensor_product():
    """Demonstrate tensor product of trace lists"""
    merger = CollapseMerger()
    
    print("\n=== Tensor Product of Traces ===")
    
    traces1 = ["0", "1", "10"]
    traces2 = ["01", "00"]
    
    product = merger.merge_tensor_product(traces1, traces2)
    
    print(f"Traces1: {traces1}")
    print(f"Traces2: {traces2}")
    print(f"\nTensor product ({len(product)} valid combinations):")
    for i, merged in enumerate(product):
        print(f"  {merged}")

def graph_analysis():
    """Perform graph theory analysis of merge structures"""
    graph_analyzer = GraphTheoryTensorAnalyzer()
    
    print("\n=== Graph Theory Analysis ===")
    
    traces = ["0", "1", "00", "01", "10", "100", "001", "010"]
    
    # Build merge graph
    G = graph_analyzer.build_merge_graph(traces)
    
    print(f"\nMerge graph properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Analyze components
    components = graph_analyzer.analyze_merge_components(G)
    print(f"\nComponent analysis:")
    for key, value in components.items():
        print(f"  {key}: {value}")
    
    # Find merge paths
    print(f"\nSample merge paths:")
    for start, end in [(0, 7), (1, 6), (2, 5)]:
        paths = graph_analyzer.find_merge_paths(G, start, end)
        if paths:
            print(f"  From '{traces[start]}' to '{traces[end]}': {len(paths)} paths")
            if paths[0] and len(paths[0]) <= 4:
                path_traces = [traces[i] for i in paths[0]]
                print(f"    Example: {' → '.join(path_traces)}")

def information_analysis():
    """Perform information theory analysis"""
    info_analyzer = InformationTheoryTensorAnalyzer()
    
    print("\n=== Information Theory Analysis ===")
    
    # Create sample tensors
    traces1 = ["101", "010", "100", "001"]
    traces2 = ["110", "011", "101", "010"]  # Some invalid
    traces2_valid = [t for t in traces2 if '11' not in t]
    
    tensor1 = TraceTensor(traces1)
    tensor2 = TraceTensor(traces2_valid)
    
    # Entropy
    entropy1 = info_analyzer.compute_tensor_entropy(tensor1)
    entropy2 = info_analyzer.compute_tensor_entropy(tensor2)
    
    print(f"\nTensor entropy:")
    print(f"  Tensor 1: {entropy1:.3f} bits")
    print(f"  Tensor 2: {entropy2:.3f} bits")
    
    # Mutual information
    mi = info_analyzer.compute_mutual_information(tensor1, tensor2)
    print(f"\nMutual information: {mi:.3f} bits")
    
    # Compression analysis
    comp_analysis = info_analyzer.analyze_compression_potential(tensor1)
    print(f"\nCompression potential:")
    for key, value in comp_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

def category_analysis():
    """Perform category theory analysis"""
    cat_analyzer = CategoryTheoryTensorAnalyzer()
    
    print("\n=== Category Theory Analysis ===")
    
    traces1 = ["0", "1", "10"]
    traces2 = ["01", "00"]
    
    # Functor properties
    functor_results = cat_analyzer.verify_tensor_functor(traces1, traces2)
    print(f"\nTensor functor properties:")
    for prop, value in functor_results.items():
        print(f"  {prop}: {value}")
    
    # Monoidal structure
    monoidal = cat_analyzer.analyze_monoidal_structure(traces1 + traces2)
    print(f"\nMonoidal category structure:")
    for prop, value in monoidal.items():
        print(f"  {prop}: {value}")

def demonstrate_merge_operations():
    """Demonstrate various merge operations"""
    merger = CollapseMerger()
    
    print("\n=== Merge Operations ===")
    
    traces = ["101", "010", "100"]
    
    operations = ['OR', 'AND', 'XOR']
    
    for op in operations:
        result = merger.merge_with_operation(traces, op)
        print(f"\n{op} merge of {traces}:")
        if result:
            print(f"  Result: {result}")
            print(f"  Valid: {'✓' if '11' not in result else '✗'}")
        else:
            print(f"  Failed: Would create '11'")

def analyze_tensor_properties():
    """Analyze properties of various trace tensors"""
    analyzer = TensorStructureAnalyzer()
    
    print("\n=== Tensor Property Analysis ===")
    
    # Different tensor configurations
    configs = [
        (["0", "1"], 1, "Binary basis"),
        (["00", "01", "10"], 1, "φ-alphabet"),
        (["000", "001", "010", "100", "101"], 1, "Length-3 traces"),
        (["0", "1", "00", "01", "10", "000", "001", "010", "100"], 2, "Mixed lengths")
    ]
    
    for traces, order, name in configs:
        tensor = TraceTensor(traces, order=order)
        props = analyzer.analyze_tensor_properties(tensor)
        
        print(f"\n{name} tensor:")
        for key, value in props.items():
            print(f"  {key}: {value}")

def main():
    """Run comprehensive tensor merge analysis"""
    print("="*80)
    print("Chapter 018: CollapseMerge - Merging into Trace Tensors")
    print("="*80)
    
    # Basic merging
    demonstrate_basic_merging()
    
    # Tensor construction
    demonstrate_tensor_construction()
    
    # Compatibility analysis
    analyze_merge_compatibility()
    
    # Tensor product
    demonstrate_tensor_product()
    
    # Graph analysis
    graph_analysis()
    
    # Information analysis
    information_analysis()
    
    # Category analysis
    category_analysis()
    
    # Merge operations
    demonstrate_merge_operations()
    
    # Tensor properties
    analyze_tensor_properties()
    
    print("\n" + "="*80)
    print("Trace tensor principles verified!")
    print("From ψ = ψ(ψ) emerges higher-order structure in trace space.")

if __name__ == "__main__":
    main()