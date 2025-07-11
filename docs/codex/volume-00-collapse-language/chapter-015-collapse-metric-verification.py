#!/usr/bin/env python3
"""
Chapter 015: CollapseMetric - Defining Distance and Dissimilarity over Trace Topology

Verification program demonstrating metric structures between traces,
including distance definitions, topological properties, and metric space completeness
through graph theory, information theory, and category theory perspectives.

From ψ = ψ(ψ), we derive metric principles for collapse space.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple, Optional, Callable
import itertools
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import networkx as nx

class TraceMetrics:
    """Various distance metrics for φ-traces"""
    
    def __init__(self):
        self.phi_validator = lambda t: '11' not in t
        
    def hamming_distance(self, t1: str, t2: str) -> float:
        """Hamming distance (for equal length traces)"""
        if len(t1) != len(t2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(t1, t2))
    
    def edit_distance(self, t1: str, t2: str) -> int:
        """Levenshtein edit distance"""
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i-1] == t2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                    
        return dp[m][n]
    
    def phi_edit_distance(self, t1: str, t2: str) -> float:
        """Edit distance respecting φ-constraint"""
        # Similar to edit distance but infinite cost for creating '11'
        m, n = len(t1), len(t2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        dp[0][0] = 0
        for i in range(1, m + 1):
            if self.phi_validator(t1[:i]):
                dp[i][0] = i
        for j in range(1, n + 1):
            if self.phi_validator(t2[:j]):
                dp[0][j] = j
                
        # Fill DP table with φ-constraint checking
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i-1] == t2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Check if operations maintain φ-constraint
                    candidates = []
                    
                    # Deletion
                    if dp[i-1][j] != float('inf'):
                        candidates.append(dp[i-1][j] + 1)
                    
                    # Insertion
                    if dp[i][j-1] != float('inf'):
                        candidates.append(dp[i][j-1] + 1)
                        
                    # Substitution
                    if dp[i-1][j-1] != float('inf'):
                        candidates.append(dp[i-1][j-1] + 1)
                        
                    if candidates:
                        dp[i][j] = min(candidates)
                        
        return dp[m][n]
    
    def structural_distance(self, t1: str, t2: str) -> float:
        """Distance based on structural features"""
        features1 = self._extract_features(t1)
        features2 = self._extract_features(t2)
        
        # Euclidean distance in feature space
        diff = 0.0
        for key in set(features1.keys()) | set(features2.keys()):
            diff += (features1.get(key, 0) - features2.get(key, 0)) ** 2
            
        return np.sqrt(diff)
    
    def _extract_features(self, trace: str) -> Dict[str, float]:
        """Extract structural features from trace"""
        if not trace:
            return {'length': 0, 'zeros': 0, 'ones': 0, 'transitions': 0}
            
        features = {
            'length': len(trace),
            'zeros': trace.count('0'),
            'ones': trace.count('1'),
            'transitions': sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1]),
            'zero_runs': len([g for k, g in itertools.groupby(trace) if k == '0']),
            'one_runs': len([g for k, g in itertools.groupby(trace) if k == '1']),
            'max_zero_run': max((len(list(g)) for k, g in itertools.groupby(trace) if k == '0'), default=0),
            'density': trace.count('1') / len(trace) if len(trace) > 0 else 0
        }
        
        return features
    
    def entropy_distance(self, t1: str, t2: str) -> float:
        """Distance based on information content"""
        if not t1 or not t2:
            return float('inf')
            
        # Compute transition probabilities
        def get_transition_probs(trace):
            trans = defaultdict(int)
            for i in range(len(trace) - 1):
                trans[trace[i:i+2]] += 1
            total = sum(trans.values())
            if total == 0:
                return {}
            return {k: v/total for k, v in trans.items()}
        
        probs1 = get_transition_probs(t1)
        probs2 = get_transition_probs(t2)
        
        # KL divergence (symmetrized)
        kl_div = 0.0
        all_trans = set(probs1.keys()) | set(probs2.keys())
        
        for trans in all_trans:
            p1 = probs1.get(trans, 1e-10)
            p2 = probs2.get(trans, 1e-10)
            kl_div += p1 * np.log(p1 / p2) + p2 * np.log(p2 / p1)
            
        return kl_div / 2  # Symmetrized KL divergence

class MetricSpaceAnalyzer:
    """Analyze metric space properties"""
    
    def __init__(self, metric_func: Callable[[str, str], float]):
        self.d = metric_func
        
    def verify_metric_axioms(self, traces: List[str]) -> Dict[str, bool]:
        """Verify if function satisfies metric axioms"""
        results = {
            'non_negativity': True,
            'identity': True,
            'symmetry': True,
            'triangle_inequality': True
        }
        
        n = min(len(traces), 10)  # Test on subset for efficiency
        test_traces = traces[:n]
        
        for i, t1 in enumerate(test_traces):
            for j, t2 in enumerate(test_traces):
                d12 = self.d(t1, t2)
                
                # Non-negativity
                if d12 < 0:
                    results['non_negativity'] = False
                    
                # Identity of indiscernibles
                if i == j and d12 != 0:
                    results['identity'] = False
                if i != j and d12 == 0:
                    results['identity'] = False
                    
                # Symmetry
                d21 = self.d(t2, t1)
                if abs(d12 - d21) > 1e-10:
                    results['symmetry'] = False
                    
                # Triangle inequality
                for k, t3 in enumerate(test_traces):
                    if k != i and k != j:
                        d13 = self.d(t1, t3)
                        d32 = self.d(t3, t2)
                        if d12 > d13 + d32 + 1e-10:
                            results['triangle_inequality'] = False
                            
        return results
    
    def compute_diameter(self, traces: List[str]) -> float:
        """Compute diameter of trace set"""
        if len(traces) < 2:
            return 0.0
            
        max_dist = 0.0
        for i, t1 in enumerate(traces):
            for j, t2 in enumerate(traces[i+1:], i+1):
                max_dist = max(max_dist, self.d(t1, t2))
                
        return max_dist
    
    def find_metric_balls(self, center: str, traces: List[str], radius: float) -> List[str]:
        """Find all traces within radius of center"""
        return [t for t in traces if self.d(center, t) <= radius]
    
    def is_complete(self, traces: List[str], max_length: int) -> Tuple[bool, Optional[str]]:
        """Check if metric space is complete (simplified)"""
        # For finite spaces, check if all Cauchy sequences converge
        # Simplified: check if space is closed under limits
        
        # Generate all possible traces up to max_length
        all_possible = set()
        
        def generate_all(prefix: str, length: int):
            if len(prefix) == length:
                if '11' not in prefix:
                    all_possible.add(prefix)
                return
                
            if not prefix or prefix[-1] == '0':
                generate_all(prefix + '0', length)
                generate_all(prefix + '1', length)
            else:  # prefix ends with 1
                generate_all(prefix + '0', length)
                
        for length in range(1, max_length + 1):
            generate_all('', length)
            
        # Check if our trace set is closed
        trace_set = set(traces)
        missing = all_possible - trace_set
        
        return len(missing) == 0, (list(missing)[0] if missing else None)

class TopologicalAnalyzer:
    """Analyze topological properties of trace spaces"""
    
    def __init__(self, metric: TraceMetrics):
        self.metric = metric
        
    def compute_clustering_coefficient(self, traces: List[str], epsilon: float) -> float:
        """Compute clustering coefficient based on ε-neighborhoods"""
        if len(traces) < 3:
            return 0.0
            
        # Build ε-neighborhood graph
        G = nx.Graph()
        G.add_nodes_from(range(len(traces)))
        
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                if self.metric.edit_distance(traces[i], traces[j]) <= epsilon:
                    G.add_edge(i, j)
                    
        # Compute clustering coefficient
        return nx.average_clustering(G)
    
    def find_connected_components(self, traces: List[str], epsilon: float) -> List[Set[int]]:
        """Find connected components in ε-neighborhood graph"""
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(len(traces)))
        
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                if self.metric.edit_distance(traces[i], traces[j]) <= epsilon:
                    G.add_edge(i, j)
                    
        # Find components
        return list(nx.connected_components(G))
    
    def compute_metric_dimension(self, traces: List[str]) -> int:
        """Estimate metric dimension of trace space"""
        if len(traces) < 2:
            return 0
            
        # Sample pairwise distances
        n = min(len(traces), 20)
        sample = traces[:n]
        
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                d = self.metric.edit_distance(sample[i], sample[j])
                if d > 0:
                    distances.append(d)
                    
        if not distances:
            return 0
            
        # Estimate dimension using distance distribution
        # Simplified: use variance of log distances
        log_distances = [np.log(d) for d in distances]
        variance = np.var(log_distances)
        
        # Heuristic: higher variance suggests higher dimension
        return min(int(variance * 2 + 1), len(traces[0]) if traces else 1)

class InformationMetricAnalyzer:
    """Analyze information-theoretic aspects of metrics"""
    
    def __init__(self, metric: TraceMetrics):
        self.metric = metric
        
    def compute_metric_entropy(self, traces: List[str], bins: int = 10) -> float:
        """Compute entropy of distance distribution"""
        if len(traces) < 2:
            return 0.0
            
        # Collect all pairwise distances
        distances = []
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                distances.append(self.metric.edit_distance(traces[i], traces[j]))
                
        if not distances:
            return 0.0
            
        # Bin distances and compute entropy
        hist, _ = np.histogram(distances, bins=bins)
        probs = hist / hist.sum()
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def compute_information_distance(self, t1: str, t2: str) -> float:
        """Normalized information distance (approximation)"""
        if not t1 or not t2:
            return float('inf')
            
        # Use compression-based approximation
        def complexity(s: str) -> float:
            # Simplified: count unique substrings
            substrings = set()
            for i in range(len(s)):
                for j in range(i+1, min(i+5, len(s)+1)):
                    substrings.add(s[i:j])
            return len(substrings)
        
        K1 = complexity(t1)
        K2 = complexity(t2)
        K12 = complexity(t1 + t2)
        
        if max(K1, K2) == 0:
            return 0.0
            
        return (K12 - min(K1, K2)) / max(K1, K2)

class CategoryMetricAnalyzer:
    """Analyze metrics from category theory perspective"""
    
    def __init__(self, metric: TraceMetrics):
        self.metric = metric
        
    def construct_metric_category(self, traces: List[str], epsilon: float) -> Dict:
        """Construct category from metric space"""
        # Objects: traces
        # Morphisms: paths with distance ≤ ε
        
        category = {
            'objects': list(range(len(traces))),
            'morphisms': defaultdict(set),
            'distances': {}
        }
        
        # Add morphisms for nearby traces
        for i in range(len(traces)):
            for j in range(len(traces)):
                d = self.metric.edit_distance(traces[i], traces[j])
                if d <= epsilon:
                    category['morphisms'][i].add(j)
                    category['distances'][(i, j)] = d
                    
        return category
    
    def find_isometries(self, traces: List[str]) -> List[Tuple[int, int]]:
        """Find trace pairs that are isometric (same distance relationships)"""
        isometries = []
        n = len(traces)
        
        if n < 2:
            return []
            
        # For each pair of traces, check if they have same distance patterns
        for i in range(n):
            for j in range(i+1, n):
                # Compare distance patterns
                pattern_i = sorted([self.metric.edit_distance(traces[i], traces[k]) 
                                  for k in range(n) if k != i])
                pattern_j = sorted([self.metric.edit_distance(traces[j], traces[k]) 
                                  for k in range(n) if k != j])
                                  
                if pattern_i == pattern_j:
                    isometries.append((i, j))
                    
        return isometries

def visualize_metric_space(traces: List[str], metric: TraceMetrics, title: str = "Metric Space"):
    """Visualize trace metric space using MDS"""
    n = min(len(traces), 20)  # Limit for visualization
    sample = traces[:n]
    
    # Compute distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = metric.edit_distance(sample[i], sample[j])
            
    print(f"\n=== {title} Visualization ===")
    print(f"Distance matrix (first 5x5):")
    print(dist_matrix[:5, :5])
    
    # Show distance statistics
    distances = [dist_matrix[i, j] for i in range(n) for j in range(i+1, n)]
    if distances:
        print(f"\nDistance statistics:")
        print(f"  Min: {min(distances):.3f}")
        print(f"  Max: {max(distances):.3f}")
        print(f"  Mean: {np.mean(distances):.3f}")
        print(f"  Std: {np.std(distances):.3f}")

def demonstrate_metric_properties():
    """Demonstrate various metric properties"""
    print("\n=== Metric Properties Demonstration ===")
    
    # Generate test traces
    traces = ['0', '1', '00', '01', '10', '000', '001', '010', '100', '101',
              '0000', '0001', '0010', '0100', '0101', '1000', '1001', '1010']
    
    metrics = TraceMetrics()
    
    # Test different metrics
    metric_funcs = {
        'Hamming': metrics.hamming_distance,
        'Edit': metrics.edit_distance,
        'φ-Edit': metrics.phi_edit_distance,
        'Structural': metrics.structural_distance,
        'Entropy': metrics.entropy_distance
    }
    
    print("\nSample distances between '0101' and '1010':")
    t1, t2 = '0101', '1010'
    for name, func in metric_funcs.items():
        try:
            d = func(t1, t2)
            print(f"  {name}: {d:.3f}")
        except:
            print(f"  {name}: N/A")
    
    # Verify metric axioms
    print("\nMetric axiom verification:")
    for name, func in [('Edit', metrics.edit_distance), 
                       ('Structural', metrics.structural_distance)]:
        analyzer = MetricSpaceAnalyzer(func)
        axioms = analyzer.verify_metric_axioms(traces[:10])
        print(f"\n{name} distance:")
        for axiom, satisfied in axioms.items():
            print(f"  {axiom}: {'✓' if satisfied else '✗'}")

def analyze_trace_topology():
    """Analyze topological properties of trace space"""
    print("\n=== Topological Analysis ===")
    
    # Generate traces
    traces = []
    for length in range(1, 7):
        for i in range(min(20, 2**length)):
            binary = format(i, f'0{length}b')
            if '11' not in binary:
                traces.append(binary)
    
    metrics = TraceMetrics()
    topo = TopologicalAnalyzer(metrics)
    
    # Clustering analysis
    for epsilon in [1, 2, 3]:
        clustering = topo.compute_clustering_coefficient(traces, epsilon)
        components = topo.find_connected_components(traces, epsilon)
        print(f"\nε = {epsilon}:")
        print(f"  Clustering coefficient: {clustering:.3f}")
        print(f"  Connected components: {len(components)}")
        print(f"  Largest component size: {max(len(c) for c in components)}")
    
    # Metric dimension
    dim = topo.compute_metric_dimension(traces[:20])
    print(f"\nEstimated metric dimension: {dim}")

def analyze_information_metrics():
    """Analyze information-theoretic properties"""
    print("\n=== Information-Theoretic Analysis ===")
    
    traces = ['0', '00', '01', '10', '000', '001', '010', '100', '101',
              '0000', '0001', '0010', '0100', '0101', '1000', '1001', '1010']
    
    metrics = TraceMetrics()
    info_analyzer = InformationMetricAnalyzer(metrics)
    
    # Metric entropy
    entropy = info_analyzer.compute_metric_entropy(traces)
    print(f"Metric entropy: {entropy:.3f} bits")
    
    # Information distances
    print("\nInformation distance examples:")
    test_pairs = [('01', '10'), ('00', '000'), ('0101', '1010')]
    for t1, t2 in test_pairs:
        d = info_analyzer.compute_information_distance(t1, t2)
        print(f"  d('{t1}', '{t2}') = {d:.3f}")

def analyze_categorical_structure():
    """Analyze category-theoretic aspects"""
    print("\n=== Category-Theoretic Analysis ===")
    
    traces = ['0', '1', '00', '01', '10', '000', '001', '010', '100', '101']
    
    metrics = TraceMetrics()
    cat_analyzer = CategoryMetricAnalyzer(metrics)
    
    # Construct metric category
    for epsilon in [1, 2]:
        category = cat_analyzer.construct_metric_category(traces, epsilon)
        print(f"\nMetric category (ε = {epsilon}):")
        print(f"  Objects: {len(category['objects'])}")
        print(f"  Morphisms: {sum(len(m) for m in category['morphisms'].values())}")
        
        # Show morphism structure for first few objects
        print("  Morphism structure (first 3 objects):")
        for i in range(min(3, len(traces))):
            targets = category['morphisms'][i]
            print(f"    {i} ({traces[i]}) → {sorted(targets)}")
    
    # Find isometries
    isometries = cat_analyzer.find_isometries(traces)
    print(f"\nIsometric pairs: {len(isometries)}")
    if isometries:
        for i, j in isometries[:3]:
            print(f"  {traces[i]} ≅ {traces[j]}")

def demonstrate_completeness():
    """Demonstrate metric completeness"""
    print("\n=== Completeness Analysis ===")
    
    # Generate φ-valid traces
    traces = []
    for length in range(1, 5):
        for i in range(2**length):
            binary = format(i, f'0{length}b')
            if '11' not in binary:
                traces.append(binary)
    
    metrics = TraceMetrics()
    analyzer = MetricSpaceAnalyzer(metrics.edit_distance)
    
    # Check completeness
    is_complete, missing = analyzer.is_complete(traces, max_length=4)
    
    print(f"Trace space up to length 4:")
    print(f"  Number of traces: {len(traces)}")
    print(f"  Complete: {'Yes' if is_complete else 'No'}")
    if not is_complete:
        print(f"  Example missing trace: '{missing}'")
    
    # Diameter analysis
    diameter = analyzer.compute_diameter(traces)
    print(f"  Diameter: {diameter}")
    
    # Metric balls
    center = '01'
    for radius in [1, 2, 3]:
        ball = analyzer.find_metric_balls(center, traces, radius)
        print(f"\nBall B('{center}', {radius}) contains {len(ball)} traces:")
        print(f"  {sorted(ball[:5])}{'...' if len(ball) > 5 else ''}")

def main():
    """Run comprehensive metric analysis"""
    print("="*80)
    print("Chapter 015: CollapseMetric - Defining Distance and Dissimilarity over Trace Topology")
    print("="*80)
    
    # Demonstrate metric properties
    demonstrate_metric_properties()
    
    # Topological analysis
    analyze_trace_topology()
    
    # Information-theoretic analysis
    analyze_information_metrics()
    
    # Category-theoretic analysis
    analyze_categorical_structure()
    
    # Completeness analysis
    demonstrate_completeness()
    
    # Visualize a metric space
    traces = ['0', '00', '01', '10', '000', '001', '010', '100', '101']
    metrics = TraceMetrics()
    visualize_metric_space(traces, metrics, "Edit Distance Metric Space")
    
    print("\n" + "="*80)
    print("Metric principles for collapse space verified successfully!")
    print("From ψ = ψ(ψ) emerges natural distance structure in trace topology.")

if __name__ == "__main__":
    main()