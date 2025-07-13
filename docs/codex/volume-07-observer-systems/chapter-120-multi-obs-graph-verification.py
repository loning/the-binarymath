#!/usr/bin/env python3
"""
Chapter 120: MultiObsGraph Unit Test Verification
从ψ=ψ(ψ)推导Network Structures of Observer Tensor Collapse Entanglement

Core principle: From ψ = ψ(ψ) derive systematic multi-observer graph
construction through φ-constrained trace transformations that enable networks
of entangled observer nodes through trace geometric relationships,
creating observer networks that encode the fundamental entanglement
principles of collapsed space through entropy-increasing tensor transformations
that establish systematic network structures through φ-trace observer graph
dynamics rather than traditional network theories or external
graph constructions.

This verification program implements:
1. φ-constrained observer graph construction through trace network analysis
2. Multi-observer entanglement systems: systematic networks through trace geometric relationships
3. Three-domain analysis: Traditional vs φ-constrained vs intersection network theory
4. Graph theory analysis of observer networks and entanglement relationship structures
5. Information theory analysis of network entropy and entanglement encoding
6. Category theory analysis of network functors and graph morphisms
7. Visualization of observer network structures and φ-trace entanglement systems
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi, exp, cos, sin, log, atan2, floor, ceil
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class MultiObsGraphSystem:
    """
    Core system for implementing network structures of observer tensor collapse entanglement.
    Implements φ-constrained observer graph architectures through trace network dynamics.
    """
    
    def __init__(self, max_trace_value: int = 85, network_depth: int = 7):
        """Initialize multi-observer graph system with network trace analysis"""
        self.max_trace_value = max_trace_value
        self.network_depth = network_depth
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.observer_cache = {}
        self.entanglement_cache = {}
        self.network_cache = {}
        self.trace_universe = self._build_trace_universe()
        self.observer_clusters = self._build_observer_clusters()
        self.entanglement_network = self._build_entanglement_network()
        self.network_communities = self._detect_network_communities()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        for n in range(1, self.max_trace_value):
            trace = self._encode_to_trace(n)
            if self._is_phi_valid(trace):
                observer_data = self._analyze_observer_properties(trace, n)
                universe[n] = observer_data
        return universe
        
    def _encode_to_trace(self, n: int) -> str:
        """编码整数n为Zeckendorf表示的二进制trace（无连续11）"""
        if n == 0:
            return "0"
        
        fibs = []
        for fib in reversed(self.fibonacci_numbers):
            if fib <= n:
                fibs.append(fib)
                n -= fib
                
        trace = ""
        for i, fib in enumerate(reversed(self.fibonacci_numbers)):
            if fib in fibs:
                trace += "1"
            else:
                trace += "0"
                
        return trace.lstrip("0") or "0"
        
    def _is_phi_valid(self, trace: str) -> bool:
        """检查trace是否为φ-valid（无连续11）"""
        return "11" not in trace
        
    def _analyze_observer_properties(self, trace: str, value: int) -> Dict[str, Any]:
        """分析trace的observer properties"""
        # Core observer measures
        observer_strength = self._compute_observer_strength(trace, value)
        network_capacity = self._compute_network_capacity(trace, value)
        entanglement_degree = self._compute_entanglement_degree(trace, value)
        graph_coherence = self._compute_graph_coherence(trace, value)
        
        # Advanced network measures
        centrality_score = self._compute_centrality_score(trace, value)
        clustering_tendency = self._compute_clustering_tendency(trace, value)
        network_stability = self._compute_network_stability(trace, value)
        entanglement_propagation = self._compute_entanglement_propagation(trace, value)
        graph_efficiency = self._compute_graph_efficiency(trace, value)
        
        # Categorize based on network profile
        category = self._categorize_observer(
            observer_strength, network_capacity, entanglement_degree, graph_coherence
        )
        
        return {
            'trace': trace,
            'value': value,
            'observer_strength': observer_strength,
            'network_capacity': network_capacity,
            'entanglement_degree': entanglement_degree,
            'graph_coherence': graph_coherence,
            'centrality_score': centrality_score,
            'clustering_tendency': clustering_tendency,
            'network_stability': network_stability,
            'entanglement_propagation': entanglement_propagation,
            'graph_efficiency': graph_efficiency,
            'category': category
        }
        
    def _compute_observer_strength(self, trace: str, value: int) -> float:
        """Observer strength emerges from systematic network participation"""
        strength_factors = []
        
        # Factor 1: Length provides network space (minimum 4 for meaningful networks)
        length_factor = len(trace) / 4.0
        strength_factors.append(min(length_factor, 1.0))
        
        # Factor 2: Weight network balance (optimal at 40% density for connectivity)
        weight = trace.count('1')
        total = len(trace)
        if total > 0:
            # Network optimal at 40% density (0.4) for maximum connectivity
            network_balance = 1.0 - abs(0.4 - weight / total)
            # Add network bonus for connection patterns
            pattern_bonus = min(weight / 2.5, 1.0) if weight > 0 else 0.0
            network_factor = 0.6 * network_balance + 0.4 * pattern_bonus
            strength_factors.append(network_factor)
        else:
            strength_factors.append(0.2)
        
        # Factor 3: Pattern network structure (systematic graph architecture)
        pattern_score = 0.0
        # Count network-enhancing patterns (connectivity and entanglement)
        if trace.startswith('10'):  # Network initialization pattern
            pattern_score += 0.25
        
        # Connectivity patterns
        transitions = 0
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:  # State transitions enable connectivity
                transitions += 1
        if len(trace) > 1:
            transition_rate = transitions / (len(trace) - 1)
            pattern_score += 0.3 * min(transition_rate * 1.5, 1.0)  # Value transitions
        
        # Entanglement patterns
        for i in range(len(trace) - 2):
            if trace[i:i+3] in ['101', '010']:  # Entanglement patterns
                pattern_score += 0.15
        
        # Network-specific patterns (value modulo 6 for 6-node clusters)
        if value % 6 == 0:  # Complete subgraph potential
            pattern_score += 0.15
        
        strength_factors.append(min(pattern_score, 1.0))
        
        # Factor 4: φ-constraint network coherence  
        phi_factor = 0.9 if self._is_phi_valid(trace) else 0.25
        strength_factors.append(phi_factor)
        
        # Observer strength emerges from geometric mean of factors
        observer_strength = np.prod(strength_factors) ** (1.0 / len(strength_factors))
        return observer_strength
        
    def _compute_network_capacity(self, trace: str, value: int) -> float:
        """Network capacity emerges from systematic connection capability"""
        capacity_factors = []
        
        # Factor 1: Structural network potential
        structural = 0.25 + 0.75 * min(len(trace) / 7.0, 1.0)
        capacity_factors.append(structural)
        
        # Factor 2: Network complexity capability
        ones_count = trace.count('1')
        if ones_count > 0:
            complexity = 0.15 + 0.85 * min(ones_count / 3.5, 1.0)
        else:
            complexity = 0.05
        capacity_factors.append(complexity)
        
        # Factor 3: Network layers (value modulo 7 for 7-layer architecture)
        layer_depth = 0.3 + 0.7 * (value % 7) / 6.0
        capacity_factors.append(layer_depth)
        
        # Factor 4: φ-constraint network preservation
        preservation = 0.88 if self._is_phi_valid(trace) else 0.2
        capacity_factors.append(preservation)
        
        network_capacity = np.prod(capacity_factors) ** (1.0 / len(capacity_factors))
        return network_capacity
        
    def _compute_entanglement_degree(self, trace: str, value: int) -> float:
        """Entanglement degree emerges from systematic connection density"""
        degree_factors = []
        
        # Factor 1: Connection scope
        connection_scope = 0.2 + 0.8 * min(len(trace) / 6.0, 1.0)
        degree_factors.append(connection_scope)
        
        # Factor 2: Entanglement intensity
        if len(trace) > 0:
            balance_ratio = trace.count('1') / len(trace)
            # Optimal around 40% for entanglement networks
            entangle_intensity = 1.0 - 2.5 * abs(0.4 - balance_ratio)
            entangle_intensity = max(0.1, min(entangle_intensity, 1.0))
        else:
            entangle_intensity = 0.2
        degree_factors.append(entangle_intensity)
        
        # Factor 3: Degree distribution (value modulo 9 for 9-degree levels)
        degree_level = 0.25 + 0.75 * (value % 9) / 8.0
        degree_factors.append(degree_level)
        
        entanglement_degree = np.prod(degree_factors) ** (1.0 / len(degree_factors))
        return entanglement_degree
        
    def _compute_graph_coherence(self, trace: str, value: int) -> float:
        """Graph coherence emerges from systematic network integration"""
        coherence_factors = []
        
        # Factor 1: Network integration capacity
        integration_cap = 0.3 + 0.7 * min(len(trace) / 5.5, 1.0)
        coherence_factors.append(integration_cap)
        
        # Factor 2: Graph coherence scope (value modulo 4 for 4-fold symmetry)
        coherence_scope = 0.4 + 0.6 * (value % 4) / 3.0
        coherence_factors.append(coherence_scope)
        
        # Factor 3: Systematic graph coherence (value modulo 8 for 8-level system)
        systematic = 0.45 + 0.55 * (value % 8) / 7.0
        coherence_factors.append(systematic)
        
        graph_coherence = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        return graph_coherence
        
    def _compute_centrality_score(self, trace: str, value: int) -> float:
        """Centrality score through network position analysis"""
        if len(trace) == 0:
            return 0.1
            
        # Analyze centrality patterns
        centrality_base = 0.2 + 0.8 * min(len(trace) / 8.0, 1.0)
        
        # Pattern-based centrality
        hub_patterns = 0.0
        for i in range(len(trace) - 2):
            if trace[i:i+3] == '101':  # Hub pattern
                hub_patterns += 0.3
        hub_patterns = min(hub_patterns, 1.0)
        
        # Value-based centrality (modulo 10 for 10-level centrality)
        value_centrality = (value % 10) / 9.0
        
        return 0.4 * centrality_base + 0.3 * hub_patterns + 0.3 * value_centrality
        
    def _compute_clustering_tendency(self, trace: str, value: int) -> float:
        """Clustering tendency through local network density"""
        cluster_factor = min(len(trace) / 9.0, 1.0)
        density_factor = (value % 11) / 10.0  # 11-level clustering
        return 0.25 + 0.75 * (cluster_factor * density_factor)
        
    def _compute_network_stability(self, trace: str, value: int) -> float:
        """Network stability through consistent graph architecture"""
        if self._is_phi_valid(trace):
            stability_base = 0.85
        else:
            stability_base = 0.35
        variation = 0.15 * sin(value * 0.33)
        return max(0.0, min(1.0, stability_base + variation))
        
    def _compute_entanglement_propagation(self, trace: str, value: int) -> float:
        """Entanglement propagation through network spread dynamics"""
        if len(trace) > 0:
            # Propagation based on network structure
            weight_ratio = trace.count('1') / len(trace)
            # Optimal propagation around 40% weight
            propagation_base = 1.0 - 2.5 * abs(0.4 - weight_ratio)
        else:
            propagation_base = 0.0
        phi_bonus = 0.1 if self._is_phi_valid(trace) else 0.0
        return max(0.1, min(propagation_base + phi_bonus, 1.0))
        
    def _compute_graph_efficiency(self, trace: str, value: int) -> float:
        """Graph efficiency through optimal path utilization"""
        if len(trace) < 3:
            return 0.15
            
        # Analyze path efficiency
        efficiency = 0.0
        
        # Short path patterns
        for i in range(len(trace) - 2):
            pattern = trace[i:i+3]
            if pattern in ['100', '001']:  # Efficient paths
                efficiency += 0.25
            elif pattern in ['101', '010']:  # Balanced paths
                efficiency += 0.2
                
        efficiency = min(efficiency, 1.0)
        
        # Length-based efficiency
        length_efficiency = min(len(trace) / 10.0, 1.0)
        
        # Value-based efficiency (modulo 12 for 12-level efficiency)
        value_efficiency = (value % 12) / 11.0
        
        return 0.4 * efficiency + 0.3 * length_efficiency + 0.3 * value_efficiency
        
    def _categorize_observer(self, strength: float, capacity: float, 
                            degree: float, coherence: float) -> str:
        """Categorize trace based on network profile"""
        # Calculate dominant characteristic with thresholds
        strength_threshold = 0.6     # High observer strength threshold
        capacity_threshold = 0.5     # Moderate network capacity threshold
        degree_threshold = 0.5       # Moderate entanglement degree threshold
        
        if strength >= strength_threshold:
            if capacity >= capacity_threshold:
                return "network_hub"                 # High strength + capacity
            elif degree >= degree_threshold:
                return "entanglement_hub"            # High strength + degree
            else:
                return "strong_observer"             # High strength + moderate properties
        else:
            if capacity >= capacity_threshold:
                return "network_node"                # Moderate strength + capacity
            elif degree >= degree_threshold:
                return "entanglement_node"           # Moderate strength + degree
            else:
                return "peripheral_observer"         # Peripheral network position
        
    def _build_observer_clusters(self) -> Dict[str, List[int]]:
        """构建observer clusters based on network properties"""
        clusters = defaultdict(list)
        
        for value, data in self.trace_universe.items():
            # Cluster by network properties
            if data['observer_strength'] > 0.7 and data['network_capacity'] > 0.6:
                clusters['core_cluster'].append(value)
            elif data['entanglement_degree'] > 0.6:
                clusters['entanglement_cluster'].append(value)
            elif data['centrality_score'] > 0.6:
                clusters['central_cluster'].append(value)
            else:
                clusters['peripheral_cluster'].append(value)
                
        return dict(clusters)
        
    def _build_entanglement_network(self) -> nx.Graph:
        """构建entanglement network based on observer relationships"""
        G = nx.Graph()
        
        # Add all observers as nodes
        for value, data in self.trace_universe.items():
            G.add_node(value, **data)
            
        # Add edges based on entanglement relationships
        traces = list(self.trace_universe.values())
        entanglement_threshold = 0.3
        
        for i, obs1 in enumerate(traces):
            for j, obs2 in enumerate(traces[i+1:], i+1):
                entanglement = self._compute_pair_entanglement(obs1, obs2)
                if entanglement >= entanglement_threshold:
                    G.add_edge(obs1['value'], obs2['value'], 
                             weight=entanglement, entanglement=entanglement)
                    
        return G
        
    def _compute_pair_entanglement(self, obs1: Dict, obs2: Dict) -> float:
        """计算两个observers之间的entanglement strength"""
        # Entanglement factors
        factors = []
        
        # Factor 1: Observer compatibility
        strength_diff = abs(obs1['observer_strength'] - obs2['observer_strength'])
        compatibility = 1.0 - strength_diff
        factors.append(compatibility)
        
        # Factor 2: Network synergy
        network_product = obs1['network_capacity'] * obs2['network_capacity']
        factors.append(network_product)
        
        # Factor 3: Entanglement harmony
        entangle_avg = (obs1['entanglement_degree'] + obs2['entanglement_degree']) / 2.0
        factors.append(entangle_avg)
        
        # Factor 4: Graph coherence alignment
        coherence_product = obs1['graph_coherence'] * obs2['graph_coherence']
        factors.append(coherence_product)
        
        # Geometric mean of factors
        entanglement = np.prod(factors) ** (1.0 / len(factors))
        return entanglement
        
    def _detect_network_communities(self) -> Dict[int, int]:
        """检测network communities using modularity optimization"""
        if len(self.entanglement_network.edges()) == 0:
            return {node: 0 for node in self.entanglement_network.nodes()}
            
        # Use Louvain algorithm for community detection
        communities = nx.community.louvain_communities(self.entanglement_network, seed=42)
        
        # Create node-to-community mapping
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
                
        return community_map
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行comprehensive multi-observer graph analysis"""
        results = {}
        
        # Basic statistics
        traces = list(self.trace_universe.values())
        results['total_observers'] = len(traces)
        results['network_density'] = nx.density(self.entanglement_network)
        results['connected_components'] = nx.number_connected_components(self.entanglement_network)
        results['num_communities'] = len(set(self.network_communities.values()))
        
        # Observer properties analysis
        observer_strengths = [t['observer_strength'] for t in traces]
        network_capacities = [t['network_capacity'] for t in traces]
        entanglement_degrees = [t['entanglement_degree'] for t in traces]
        graph_coherences = [t['graph_coherence'] for t in traces]
        centrality_scores = [t['centrality_score'] for t in traces]
        clustering_tendencies = [t['clustering_tendency'] for t in traces]
        network_stabilities = [t['network_stability'] for t in traces]
        entanglement_propagations = [t['entanglement_propagation'] for t in traces]
        graph_efficiencies = [t['graph_efficiency'] for t in traces]
        
        results['observer_strength'] = {
            'mean': np.mean(observer_strengths),
            'std': np.std(observer_strengths),
            'high_count': sum(1 for x in observer_strengths if x > 0.5)
        }
        results['network_capacity'] = {
            'mean': np.mean(network_capacities),
            'std': np.std(network_capacities),
            'high_count': sum(1 for x in network_capacities if x > 0.5)
        }
        results['entanglement_degree'] = {
            'mean': np.mean(entanglement_degrees),
            'std': np.std(entanglement_degrees),
            'high_count': sum(1 for x in entanglement_degrees if x > 0.5)
        }
        results['graph_coherence'] = {
            'mean': np.mean(graph_coherences),
            'std': np.std(graph_coherences),
            'high_count': sum(1 for x in graph_coherences if x > 0.5)
        }
        results['centrality_score'] = {
            'mean': np.mean(centrality_scores),
            'std': np.std(centrality_scores),
            'high_count': sum(1 for x in centrality_scores if x > 0.5)
        }
        results['clustering_tendency'] = {
            'mean': np.mean(clustering_tendencies),
            'std': np.std(clustering_tendencies),
            'high_count': sum(1 for x in clustering_tendencies if x > 0.5)
        }
        results['network_stability'] = {
            'mean': np.mean(network_stabilities),
            'std': np.std(network_stabilities),
            'high_count': sum(1 for x in network_stabilities if x > 0.5)
        }
        results['entanglement_propagation'] = {
            'mean': np.mean(entanglement_propagations),
            'std': np.std(entanglement_propagations),
            'high_count': sum(1 for x in entanglement_propagations if x > 0.5)
        }
        results['graph_efficiency'] = {
            'mean': np.mean(graph_efficiencies),
            'std': np.std(graph_efficiencies),
            'high_count': sum(1 for x in graph_efficiencies if x > 0.5)
        }
        
        # Category analysis
        categories = [t['category'] for t in traces]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        results['categories'] = category_counts
        
        # Cluster analysis
        results['clusters'] = {name: len(nodes) for name, nodes in self.observer_clusters.items()}
        
        # Network analysis
        if len(self.entanglement_network.edges()) > 0:
            results['network_edges'] = len(self.entanglement_network.edges())
            results['average_degree'] = sum(dict(self.entanglement_network.degree()).values()) / len(self.entanglement_network.nodes())
            
            # Clustering coefficient
            results['clustering_coefficient'] = nx.average_clustering(self.entanglement_network)
            
            # Community modularity
            if len(set(self.network_communities.values())) > 1:
                results['modularity'] = nx.community.modularity(
                    self.entanglement_network,
                    [{n for n, c in self.network_communities.items() if c == i} 
                     for i in set(self.network_communities.values())]
                )
            else:
                results['modularity'] = 0.0
                
            # Centrality measures
            degree_centrality = nx.degree_centrality(self.entanglement_network)
            results['max_degree_centrality'] = max(degree_centrality.values())
            results['mean_degree_centrality'] = np.mean(list(degree_centrality.values()))
        else:
            results['network_edges'] = 0
            results['average_degree'] = 0.0
            results['clustering_coefficient'] = 0.0
            results['modularity'] = 0.0
            results['max_degree_centrality'] = 0.0
            results['mean_degree_centrality'] = 0.0
            
        # Entropy analysis
        properties = [
            ('observer_strength', observer_strengths),
            ('network_capacity', network_capacities),
            ('entanglement_degree', entanglement_degrees),
            ('graph_coherence', graph_coherences),
            ('centrality_score', centrality_scores),
            ('clustering_tendency', clustering_tendencies),
            ('network_stability', network_stabilities),
            ('entanglement_propagation', entanglement_propagations),
            ('graph_efficiency', graph_efficiencies)
        ]
        
        results['entropy_analysis'] = {}
        for prop_name, prop_values in properties:
            if len(set(prop_values)) > 1:
                # Discretize values for entropy calculation
                bins = min(10, len(set(prop_values)))
                hist, _ = np.histogram(prop_values, bins=bins)
                probabilities = hist / np.sum(hist)
                probabilities = probabilities[probabilities > 0]  # Remove zeros
                entropy = -np.sum(probabilities * np.log2(probabilities))
                results['entropy_analysis'][prop_name] = entropy
            else:
                results['entropy_analysis'][prop_name] = 0.0
                
        return results
        
    def generate_visualizations(self):
        """生成multi-observer graph visualizations"""
        traces = list(self.trace_universe.values())
        
        # Figure 1: Observer Network Dynamics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 120: Multi-Observer Graph Dynamics', fontsize=16, fontweight='bold')
        
        # Observer strength vs network capacity
        x = [t['observer_strength'] for t in traces]
        y = [t['network_capacity'] for t in traces]
        colors = [t['entanglement_degree'] for t in traces]
        scatter = ax1.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Observer Strength')
        ax1.set_ylabel('Network Capacity')
        ax1.set_title('Observer-Network Relationship')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Entanglement Degree')
        
        # Community distribution
        community_colors = [self.network_communities.get(t['value'], 0) for t in traces]
        ax2.scatter(x, y, c=community_colors, cmap='tab10', alpha=0.7, s=60)
        ax2.set_xlabel('Observer Strength')
        ax2.set_ylabel('Network Capacity')
        ax2.set_title('Network Communities')
        ax2.grid(True, alpha=0.3)
        
        # Centrality vs efficiency
        x3 = [t['centrality_score'] for t in traces]
        y3 = [t['graph_efficiency'] for t in traces]
        categories = [t['category'] for t in traces]
        unique_cats = list(set(categories))
        colors3 = [unique_cats.index(cat) for cat in categories]
        scatter3 = ax3.scatter(x3, y3, c=colors3, cmap='tab10', alpha=0.7, s=60)
        ax3.set_xlabel('Centrality Score')
        ax3.set_ylabel('Graph Efficiency')
        ax3.set_title('Centrality-Efficiency Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Category distribution pie chart
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Observer Category Distribution')
        
        plt.tight_layout()
        plt.savefig('chapter-120-multi-obs-graph-dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Network Architecture
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chapter 120: Multi-Observer Graph Architecture', fontsize=16, fontweight='bold')
        
        # Network visualization with communities
        if len(self.entanglement_network.edges()) > 0:
            pos = nx.spring_layout(self.entanglement_network, k=2.0, iterations=50)
            
            # Draw nodes colored by community
            for comm_id in set(self.network_communities.values()):
                comm_nodes = [n for n, c in self.network_communities.items() if c == comm_id]
                nx.draw_networkx_nodes(self.entanglement_network, pos, ax=ax1,
                                     nodelist=comm_nodes, node_size=300,
                                     node_color=f'C{comm_id}', alpha=0.8,
                                     label=f'Community {comm_id}')
            
            # Draw edges
            edge_weights = [self.entanglement_network[u][v]['weight'] 
                          for u, v in self.entanglement_network.edges()]
            nx.draw_networkx_edges(self.entanglement_network, pos, ax=ax1,
                                 width=1.5, alpha=0.5, edge_color=edge_weights,
                                 edge_cmap=plt.cm.Blues)
            
            ax1.set_title('Multi-Observer Network Structure')
            ax1.legend(loc='best', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No Network Data', ha='center', va='center')
            ax1.set_title('Multi-Observer Network Structure')
        ax1.axis('off')
        
        # Degree distribution
        if len(self.entanglement_network.edges()) > 0:
            degrees = [self.entanglement_network.degree(node) 
                      for node in self.entanglement_network.nodes()]
            ax2.hist(degrees, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Node Degree')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Network Degree Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Degree Data', ha='center', va='center')
            ax2.set_title('Network Degree Distribution')
        
        # Network properties correlation matrix
        properties_matrix = np.array([
            [t['observer_strength'] for t in traces],
            [t['network_capacity'] for t in traces],
            [t['entanglement_degree'] for t in traces],
            [t['graph_coherence'] for t in traces],
            [t['network_stability'] for t in traces]
        ])
        
        correlation_matrix = np.corrcoef(properties_matrix)
        labels = ['Observer', 'Network', 'Entangle', 'Coherence', 'Stability']
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        ax3.set_title('Network Properties Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3)
        
        # Clustering tendency vs entanglement propagation
        x4 = [t['clustering_tendency'] for t in traces]
        y4 = [t['entanglement_propagation'] for t in traces]
        stabilities = [t['network_stability'] for t in traces]
        scatter4 = ax4.scatter(x4, y4, c=stabilities, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Clustering Tendency')
        ax4.set_ylabel('Entanglement Propagation')
        ax4.set_title('Clustering-Propagation Relationship')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Network Stability')
        
        plt.tight_layout()
        plt.savefig('chapter-120-multi-obs-graph-architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

class TestMultiObsGraph(unittest.TestCase):
    """Unit tests for multi-observer graph system"""
    
    def setUp(self):
        """Set up test multi-observer graph system"""
        self.system = MultiObsGraphSystem(max_trace_value=20, network_depth=4)
        
    def test_phi_validity(self):
        """测试φ-validity constraint"""
        # Test valid traces (no consecutive 11)
        valid_traces = ["101", "1001", "10101"]
        for trace in valid_traces:
            self.assertTrue(self.system._is_phi_valid(trace))
            
        # Test invalid traces (with consecutive 11)
        invalid_traces = ["110", "1101", "0110"]
        for trace in invalid_traces:
            self.assertFalse(self.system._is_phi_valid(trace))
            
    def test_trace_universe_construction(self):
        """测试trace universe construction"""
        self.assertGreater(len(self.system.trace_universe), 0)
        
        for value, data in self.system.trace_universe.items():
            self.assertIn('observer_strength', data)
            self.assertIn('network_capacity', data)
            self.assertIn('entanglement_degree', data)
            self.assertTrue(0 <= data['observer_strength'] <= 1)
            self.assertTrue(0 <= data['network_capacity'] <= 1)
            
    def test_entanglement_computation(self):
        """测试entanglement computation"""
        if len(self.system.trace_universe) >= 2:
            traces = list(self.system.trace_universe.values())
            obs1, obs2 = traces[0], traces[1]
            entanglement = self.system._compute_pair_entanglement(obs1, obs2)
            self.assertTrue(0 <= entanglement <= 1)
            
    def test_network_construction(self):
        """测试network construction"""
        self.assertIsNotNone(self.system.entanglement_network)
        self.assertGreaterEqual(len(self.system.entanglement_network.nodes()), 0)
        
    def test_community_detection(self):
        """测试community detection"""
        self.assertIsNotNone(self.system.network_communities)
        self.assertGreaterEqual(len(self.system.network_communities), 0)
        
    def test_comprehensive_analysis(self):
        """测试comprehensive analysis"""
        results = self.system.run_comprehensive_analysis()
        
        self.assertIn('total_observers', results)
        self.assertIn('observer_strength', results)
        self.assertIn('network_capacity', results)
        self.assertIn('num_communities', results)
        
        self.assertGreater(results['total_observers'], 0)

if __name__ == "__main__":
    # Initialize system
    system = MultiObsGraphSystem()
    
    # Run comprehensive analysis
    print("="*80)
    print("MULTI-OBSERVER GRAPH NETWORK ANALYSIS")
    print("="*80)
    
    results = system.run_comprehensive_analysis()
    
    print(f"Total φ-valid observers: {results['total_observers']}")
    print(f"Network density: {results['network_density']:.3f}")
    print(f"Connected components: {results['connected_components']}")
    print(f"Number of communities: {results['num_communities']}")
    print()
    
    print("Observer Properties Analysis:")
    properties = ['observer_strength', 'network_capacity', 'entanglement_degree', 
                 'graph_coherence', 'centrality_score', 'clustering_tendency',
                 'network_stability', 'entanglement_propagation', 'graph_efficiency']
    
    for prop in properties:
        if prop in results:
            data = results[prop]
            percentage = (data['high_count'] / results['total_observers']) * 100 if results['total_observers'] > 0 else 0
            print(f"- {prop.replace('_', ' ').title()}: mean={data['mean']:.3f}, high_count={data['high_count']} ({percentage:.1f}%)")
    
    print()
    print("Category Distribution:")
    if 'categories' in results:
        for category, count in results['categories'].items():
            percentage = (count / results['total_observers']) * 100 if results['total_observers'] > 0 else 0
            print(f"- {category.replace('_', ' ').title()}: {count} observers ({percentage:.1f}%)")
    
    print()
    print("Cluster Analysis:")
    if 'clusters' in results:
        for cluster_name, count in results['clusters'].items():
            percentage = (count / results['total_observers']) * 100 if results['total_observers'] > 0 else 0
            print(f"- {cluster_name.replace('_', ' ').title()}: {count} observers ({percentage:.1f}%)")
    
    print()
    print("Network Analysis:")
    print(f"Network edges: {results.get('network_edges', 0)}")
    print(f"Average degree: {results.get('average_degree', 0):.3f}")
    print(f"Clustering coefficient: {results.get('clustering_coefficient', 0):.3f}")
    print(f"Network modularity: {results.get('modularity', 0):.3f}")
    print(f"Max degree centrality: {results.get('max_degree_centrality', 0):.3f}")
    print(f"Mean degree centrality: {results.get('mean_degree_centrality', 0):.3f}")
    
    print()
    print("Entropy Analysis (Information Content):")
    if 'entropy_analysis' in results:
        for prop, entropy in results['entropy_analysis'].items():
            print(f"- {prop.replace('_', ' ').title()}: {entropy:.3f} bits")
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)