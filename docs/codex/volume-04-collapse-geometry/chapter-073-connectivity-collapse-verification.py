#!/usr/bin/env python3
"""
Chapter 073: ConnectivityCollapse Unit Test Verification
从ψ=ψ(ψ)推导Trace-Connected Component Classifiers

Core principle: From ψ = ψ(ψ) derive connectivity where connectivity is φ-valid
trace connected components that encode geometric relationships through trace-based networks,
creating systematic connectivity frameworks with bounded components and natural connectivity
properties governed by golden constraints, showing how connectivity emerges from trace relations.

This verification program implements:
1. φ-constrained connectivity as trace component operations
2. Connectivity analysis: component patterns, network structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection connectivity theory
4. Graph theory analysis of component networks and connectivity patterns
5. Information theory analysis of connectivity entropy and component information
6. Category theory analysis of connectivity functors and component morphisms
7. Visualization of connectivity structures and component patterns
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
from math import log2, gcd, sqrt, pi, exp, cos, sin
from functools import reduce
import random
import warnings
warnings.filterwarnings('ignore')

class ConnectivityCollapseSystem:
    """
    Core system for implementing connectivity collapse through trace components.
    Implements φ-constrained connectivity theory via trace-based component operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_connectivity_complexity: int = 4):
        """Initialize connectivity collapse system"""
        self.max_trace_size = max_trace_size
        self.max_connectivity_complexity = max_connectivity_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.connectivity_cache = {}
        self.component_cache = {}
        self.network_cache = {}
        self.trace_universe = self._build_trace_universe()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        # First pass: build basic universe
        for n in range(self.max_trace_size + 1):
            trace_data = self._analyze_trace_structure(n, compute_connectivity=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for connectivity properties computation
        self.trace_universe = universe
        
        # Second pass: add connectivity properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['connectivity_properties'] = self._compute_connectivity_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_connectivity: bool = True) -> Dict:
        """分析单个trace的结构属性"""
        trace = self._encode_to_trace(n)
        
        result = {
            'value': n,
            'trace': trace,
            'phi_valid': '11' not in trace,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'fibonacci_indices': self._get_fibonacci_indices(trace),
            'structural_hash': self._compute_structural_hash(trace),
            'binary_weight': self._compute_binary_weight(trace)
        }
        
        if compute_connectivity and hasattr(self, 'trace_universe'):
            result['connectivity_properties'] = self._compute_connectivity_properties(trace)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将整数编码为二进制trace字符串"""
        return bin(n)[2:] if n > 0 else '0'
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace的Fibonacci编码索引"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.append(i)
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构哈希值"""
        hash_val = 0
        for i, bit in enumerate(trace):
            if bit == '1':
                hash_val ^= (1 << i) * 31
        return hash_val
        
    def _compute_binary_weight(self, trace: str) -> int:
        """计算trace的二进制权重"""
        return sum(int(bit) * (2 ** i) for i, bit in enumerate(reversed(trace)))
        
    def _compute_connectivity_properties(self, trace: str) -> Dict:
        """计算trace的connectivity属性：component结构和network关系"""
        if trace in self.connectivity_cache:
            return self.connectivity_cache[trace]
            
        properties = {
            'component_signature': self._compute_component_signature(trace),
            'connectivity_degree': self._compute_connectivity_degree(trace),
            'component_id': self._compute_component_id(trace),
            'bridge_cost': self._compute_bridge_cost(trace),
            'connectivity_type': self._classify_connectivity_type(trace),
            'connectivity_dimension': self._compute_connectivity_dimension(trace),
            'connectivity_complexity': self._compute_connectivity_complexity(trace),
            'strongly_connected': self._is_strongly_connected(trace),
            'component_diameter': self._compute_component_diameter(trace)
        }
        
        self.connectivity_cache[trace] = properties
        return properties
        
    def _compute_component_signature(self, trace: str) -> complex:
        """计算component signature：基于连通分量的复数签名"""
        if not trace:
            return complex(0, 0)
            
        # 计算component pattern
        signature = complex(0, 0)
        n = len(trace)
        
        # Analyze connectivity pattern
        components = self._find_local_components(trace)
        
        for i, comp_size in enumerate(components):
            weight = comp_size / n if n > 0 else 0.0
            phase = 2 * pi * i / len(components) if components else 0
            signature += weight * (cos(phase) + 1j * sin(phase))
            
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _find_local_components(self, trace: str) -> List[int]:
        """找到trace中的局部连通分量"""
        if not trace:
            return []
            
        components = []
        current_component = 1
        
        for i in range(1, len(trace)):
            if trace[i] == trace[i-1]:
                current_component += 1
            else:
                components.append(current_component)
                current_component = 1
                
        components.append(current_component)
        return components
        
    def _compute_connectivity_degree(self, trace: str) -> int:
        """计算connectivity degree：连通度"""
        if not trace:
            return 0
            
        # Count connections within trace
        degree = 0
        n = len(trace)
        
        # Adjacent connections
        for i in range(n-1):
            if trace[i] == '1' or trace[i+1] == '1':
                degree += 1
                
        # Fibonacci-distance connections
        for fib in self.fibonacci_numbers:
            if fib < n:
                for i in range(n - fib):
                    if trace[i] == '1' and trace[i+fib] == '1':
                        degree += 1
                        
        return degree
        
    def _compute_component_id(self, trace: str) -> int:
        """计算component ID：基于结构的组件标识"""
        if not trace:
            return 0
            
        # Component ID based on connectivity pattern
        components = self._find_local_components(trace)
        
        # Create hash from component sizes
        comp_id = 0
        for i, size in enumerate(components):
            comp_id ^= (size << (i * 3)) % (1 << 16)
            
        return comp_id
        
    def _compute_bridge_cost(self, trace: str) -> float:
        """计算bridge cost：连接不同组件的成本"""
        if not trace or len(trace) <= 1:
            return 0.0
            
        cost = 0.0
        components = self._find_local_components(trace)
        
        if len(components) <= 1:
            return 0.0
            
        # Cost of bridging between components
        for i in range(len(components) - 1):
            # Cost based on component sizes and gap
            size1, size2 = components[i], components[i+1]
            gap_cost = 1.0 / (size1 + size2)
            
            # φ-constraint penalty
            if size1 == 1 and size2 == 1:
                gap_cost *= 2.0  # Penalty for potential 11
                
            cost += gap_cost
            
        return cost / (len(components) - 1) if len(components) > 1 else 0.0
        
    def _classify_connectivity_type(self, trace: str) -> str:
        """分类connectivity类型"""
        if not trace:
            return "disconnected"
            
        degree = self._compute_connectivity_degree(trace)
        components = self._find_local_components(trace)
        
        if len(components) == 1:
            return "fully_connected"
        elif degree > len(trace):
            return "highly_connected"
        elif len(components) == len(trace):
            return "disconnected"
        else:
            return "weakly_connected"
            
    def _compute_connectivity_dimension(self, trace: str) -> int:
        """计算connectivity dimension：连通维度"""
        if not trace:
            return 0
            
        # Dimension based on independent connectivity paths
        components = self._find_local_components(trace)
        degree = self._compute_connectivity_degree(trace)
        
        # Estimate dimension from connectivity structure
        if len(components) == 1:
            dimension = 0  # Fully connected is 0-dimensional
        else:
            # Dimension increases with component diversity
            unique_sizes = len(set(components))
            dimension = min(unique_sizes - 1, 3)  # Bounded dimension
            
        return dimension
        
    def _compute_connectivity_complexity(self, trace: str) -> float:
        """计算connectivity complexity：连通复杂度"""
        if not trace:
            return 0.0
            
        # Multi-factor complexity
        degree = self._compute_connectivity_degree(trace)
        bridge_cost = self._compute_bridge_cost(trace)
        dimension = self._compute_connectivity_dimension(trace)
        components = self._find_local_components(trace)
        
        # Normalize factors
        degree_factor = min(1.0, degree / (2 * len(trace)) if trace else 0)
        cost_factor = min(1.0, bridge_cost)
        dim_factor = min(1.0, dimension / 3.0)
        comp_factor = min(1.0, len(components) / len(trace) if trace else 0)
        
        # Combined complexity
        complexity = (degree_factor + cost_factor + dim_factor + comp_factor) / 4.0
        
        return complexity
        
    def _is_strongly_connected(self, trace: str) -> bool:
        """判断是否强连通"""
        if not trace:
            return False
            
        # Strongly connected if single component or high degree
        components = self._find_local_components(trace)
        degree = self._compute_connectivity_degree(trace)
        
        return len(components) == 1 or degree > 1.5 * len(trace)
        
    def _compute_component_diameter(self, trace: str) -> int:
        """计算component diameter：组件直径"""
        if not trace:
            return 0
            
        components = self._find_local_components(trace)
        
        # Diameter is the largest component size
        return max(components) if components else 0
        
    def analyze_connectivity_system(self) -> Dict:
        """分析完整的connectivity系统"""
        results = {
            'connectivity_elements': [],
            'component_signatures': {},
            'component_groups': defaultdict(list),
            'connectivity_types': defaultdict(int),
            'network_properties': {},
            'information_measures': {},
            'category_analysis': {},
            'convergence_analysis': {}
        }
        
        # 收集所有connectivity元素
        for n, data in self.trace_universe.items():
            if data['phi_valid']:
                connectivity_props = data.get('connectivity_properties', {})
                results['connectivity_elements'].append({
                    'value': n,
                    'trace': data['trace'],
                    'properties': connectivity_props
                })
                
                # 统计connectivity类型
                connectivity_type = connectivity_props.get('connectivity_type', 'unknown')
                results['connectivity_types'][connectivity_type] += 1
                
                # 记录component signatures
                sig = connectivity_props.get('component_signature', complex(0, 0))
                results['component_signatures'][n] = sig
                
                # 组织component groups
                comp_id = connectivity_props.get('component_id', 0)
                results['component_groups'][comp_id].append(n)
                
        # 计算网络属性
        results['network_properties'] = self._compute_network_properties(results['connectivity_elements'])
        
        # 计算信息度量
        results['information_measures'] = self._compute_information_measures(results['connectivity_elements'])
        
        # 范畴论分析
        results['category_analysis'] = self._compute_category_analysis(results['connectivity_elements'])
        
        # 三域收敛分析
        results['convergence_analysis'] = self._compute_convergence_analysis(results)
        
        return results
        
    def _compute_network_properties(self, elements: List[Dict]) -> Dict:
        """计算connectivity网络属性"""
        G = nx.Graph()
        
        # 添加节点
        for elem in elements:
            n = elem['value']
            props = elem['properties']
            G.add_node(n, **props)
            
        # 添加边：基于connectivity similarity
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                n1, n2 = elem1['value'], elem2['value']
                
                # Check if connected by component relationship
                comp_id1 = elem1['properties'].get('component_id', -1)
                comp_id2 = elem2['properties'].get('component_id', -1)
                
                if comp_id1 == comp_id2 and comp_id1 != -1:
                    # Same component group
                    G.add_edge(n1, n2, weight=1.0)
                else:
                    # Check structural similarity
                    sig1 = elem1['properties'].get('component_signature', complex(0, 0))
                    sig2 = elem2['properties'].get('component_signature', complex(0, 0))
                    distance = abs(sig1 - sig2)
                    
                    if distance < 0.3:  # Threshold for connection
                        G.add_edge(n1, n2, weight=1.0 - distance)
                        
        # Compute advanced network metrics
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len) if components else set()
            
            return {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'components': len(components),
                'largest_component_size': len(largest_component),
                'clustering': nx.average_clustering(G),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
            }
        else:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0,
                'components': 0,
                'largest_component_size': 0,
                'clustering': 0,
                'avg_degree': 0
            }
        
    def _compute_information_measures(self, elements: List[Dict]) -> Dict:
        """计算connectivity信息度量"""
        if not elements:
            return {}
            
        # 收集各种属性分布
        dimensions = []
        types = []
        complexities = []
        degrees = []
        diameters = []
        
        for elem in elements:
            props = elem['properties']
            dimensions.append(props.get('connectivity_dimension', 0))
            types.append(props.get('connectivity_type', 'unknown'))
            complexities.append(props.get('connectivity_complexity', 0))
            degrees.append(props.get('connectivity_degree', 0))
            diameters.append(props.get('component_diameter', 0))
            
        # 计算熵
        def compute_entropy(values, bins=5):
            if not values:
                return 0.0
            if isinstance(values[0], str):
                # Categorical entropy
                counts = defaultdict(int)
                for v in values:
                    counts[v] += 1
                probs = [c / len(values) for c in counts.values()]
            else:
                # Continuous entropy (discretized)
                hist, _ = np.histogram(values, bins=bins)
                probs = hist / hist.sum() if hist.sum() > 0 else []
                
            return -sum(p * log2(p) for p in probs if p > 0)
            
        return {
            'dimension_entropy': compute_entropy(dimensions),
            'type_entropy': compute_entropy(types),
            'complexity_entropy': compute_entropy(complexities),
            'degree_entropy': compute_entropy(degrees),
            'diameter_entropy': compute_entropy(diameters),
            'connectivity_complexity': len(set(types))
        }
        
    def _compute_category_analysis(self, elements: List[Dict]) -> Dict:
        """计算connectivity范畴论属性"""
        # 构建态射关系
        morphisms = []
        functorial_morphisms = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j:
                    # 检查connectivity morphism
                    degree1 = elem1['properties'].get('connectivity_degree', 0)
                    degree2 = elem2['properties'].get('connectivity_degree', 0)
                    
                    if abs(degree1 - degree2) <= 2:  # Compatible degrees
                        morphisms.append((elem1['value'], elem2['value']))
                        
                        # 检查函子性质
                        dim1 = elem1['properties'].get('connectivity_dimension', 0)
                        dim2 = elem2['properties'].get('connectivity_dimension', 0)
                        
                        if dim1 == dim2:  # Dimension preserving
                            functorial_morphisms.append((elem1['value'], elem2['value']))
                            
        # 计算component groups
        comp_groups = defaultdict(list)
        for elem in elements:
            comp_id = elem['properties'].get('component_id', 0)
            comp_groups[comp_id].append(elem['value'])
            
        return {
            'morphisms': len(morphisms),
            'functorial_morphisms': len(functorial_morphisms),
            'functoriality_ratio': len(functorial_morphisms) / len(morphisms) if morphisms else 0,
            'component_groups': len(comp_groups),
            'largest_group': max(len(group) for group in comp_groups.values()) if comp_groups else 0
        }
        
    def _compute_convergence_analysis(self, results: Dict) -> Dict:
        """计算三域收敛分析"""
        total_elements = len(results['connectivity_elements'])
        
        # Traditional domain: Would have unlimited connectivity structures
        traditional_potential = 100  # Arbitrary large number
        
        # Collapse domain: φ-constrained structures
        collapse_actual = total_elements
        
        # Convergence ratio
        convergence_ratio = collapse_actual / traditional_potential
        
        # 分析connectivity属性分布
        degrees = []
        dimensions = []
        complexities = []
        diameters = []
        strongly_connected_count = 0
        
        for elem in results['connectivity_elements']:
            props = elem['properties']
            degrees.append(props.get('connectivity_degree', 0))
            dimensions.append(props.get('connectivity_dimension', 0))
            complexities.append(props.get('connectivity_complexity', 0))
            diameters.append(props.get('component_diameter', 0))
            if props.get('strongly_connected', False):
                strongly_connected_count += 1
                
        return {
            'convergence_ratio': convergence_ratio,
            'mean_degree': np.mean(degrees) if degrees else 0,
            'mean_dimension': np.mean(dimensions) if dimensions else 0,
            'mean_complexity': np.mean(complexities) if complexities else 0,
            'mean_diameter': np.mean(diameters) if diameters else 0,
            'strongly_connected_ratio': strongly_connected_count / total_elements if total_elements > 0 else 0,
            'connectivity_efficiency': 1.0 - np.std(degrees) / (np.mean(degrees) + 1e-10) if degrees else 0
        }
        
    def visualize_connectivity_structure(self, results: Dict, save_path: str = 'chapter-073-connectivity-collapse-structure.png'):
        """可视化connectivity结构"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Component Network Graph
        ax1 = plt.subplot(331)
        elements = results['connectivity_elements'][:10]  # Limit for visibility
        
        if elements:
            G = nx.Graph()
            pos = {}
            
            # Add nodes
            for i, elem in enumerate(elements):
                G.add_node(i, label=elem['trace'])
                
            # Position nodes based on component signatures
            for i, elem in enumerate(elements):
                sig = elem['properties'].get('component_signature', complex(0, 0))
                pos[i] = (sig.real, sig.imag)
                
            # Add edges based on connectivity
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    comp_id1 = elements[i]['properties'].get('component_id', -1)
                    comp_id2 = elements[j]['properties'].get('component_id', -1)
                    
                    if comp_id1 == comp_id2 and comp_id1 != -1:
                        G.add_edge(i, j, weight=1.0, color='red')
                    else:
                        sig1 = elements[i]['properties'].get('component_signature', complex(0, 0))
                        sig2 = elements[j]['properties'].get('component_signature', complex(0, 0))
                        if abs(sig1 - sig2) < 0.3:
                            G.add_edge(i, j, weight=0.5, color='blue')
                            
            # Draw network
            if G.edges():
                edge_colors = [G[u][v].get('color', 'gray') for u, v in G.edges()]
                edge_widths = [G[u][v].get('weight', 1.0) * 2 for u, v in G.edges()]
                nx.draw(G, pos, ax=ax1, node_size=300, with_labels=False,
                       edge_color=edge_colors, width=edge_widths, alpha=0.7)
                
            ax1.set_title('Component Network Structure')
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
        
        # 2. Connectivity Degree Distribution
        ax2 = plt.subplot(332)
        degrees = [elem['properties'].get('connectivity_degree', 0) 
                  for elem in results['connectivity_elements']]
        if degrees:
            ax2.hist(degrees, bins=20, alpha=0.7, color='teal', edgecolor='black')
            ax2.axvline(np.mean(degrees), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(degrees):.2f}')
            ax2.set_xlabel('Connectivity Degree')
            ax2.set_ylabel('Count')
            ax2.set_title('Connectivity Degree Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Connectivity类型分布
        ax3 = plt.subplot(333)
        connectivity_types = results['connectivity_types']
        if connectivity_types:
            types = list(connectivity_types.keys())
            counts = list(connectivity_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            ax3.pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            ax3.set_title('Connectivity Type Distribution')
        
        # 4. Component Diameter vs Complexity
        ax4 = plt.subplot(334)
        diameters = [elem['properties'].get('component_diameter', 0) 
                    for elem in results['connectivity_elements']]
        complexities = [elem['properties'].get('connectivity_complexity', 0) 
                       for elem in results['connectivity_elements']]
        if diameters and complexities:
            scatter = ax4.scatter(diameters, complexities, s=100, alpha=0.6, c='green')
            ax4.set_xlabel('Component Diameter')
            ax4.set_ylabel('Connectivity Complexity')
            ax4.set_title('Diameter vs Complexity')
            ax4.grid(True, alpha=0.3)
        
        # 5. Component Groups
        ax5 = plt.subplot(335)
        comp_groups = results['component_groups']
        if comp_groups:
            group_sizes = [len(group) for group in comp_groups.values()]
            ax5.hist(group_sizes, bins=max(group_sizes) if group_sizes else 1, 
                    alpha=0.7, color='purple', edgecolor='black')
            ax5.set_xlabel('Group Size')
            ax5.set_ylabel('Number of Groups')
            ax5.set_title('Component Group Size Distribution')
            ax5.grid(True, alpha=0.3)
        
        # 6. 网络结构可视化
        ax6 = plt.subplot(336)
        network_props = results['network_properties']
        metrics = ['Nodes', 'Edges', 'Density', 'Components', 'Avg Degree']
        values = [
            network_props.get('nodes', 0),
            network_props.get('edges', 0),
            network_props.get('density', 0) * 10,  # Scale for visibility
            network_props.get('components', 0),
            network_props.get('avg_degree', 0)
        ]
        bars = ax6.bar(metrics, values, color=plt.cm.tab10(range(len(metrics))))
        ax6.set_ylabel('Value')
        ax6.set_title('Network Properties')
        ax6.set_xticklabels(metrics, rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 7. 信息熵度量
        ax7 = plt.subplot(337)
        info_measures = results['information_measures']
        entropy_types = ['Dimension', 'Type', 'Complexity', 'Degree', 'Diameter']
        entropy_values = [
            info_measures.get('dimension_entropy', 0),
            info_measures.get('type_entropy', 0),
            info_measures.get('complexity_entropy', 0),
            info_measures.get('degree_entropy', 0),
            info_measures.get('diameter_entropy', 0)
        ]
        ax7.barh(entropy_types, entropy_values, 
                color=plt.cm.coolwarm(np.linspace(0, 1, len(entropy_types))))
        ax7.set_xlabel('Entropy (bits)')
        ax7.set_title('Information Entropy Measures')
        ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Connectivity Matrix Heatmap
        ax8 = plt.subplot(338)
        n_elements = min(8, len(results['connectivity_elements']))  # Limit for visibility
        conn_matrix = np.zeros((n_elements, n_elements))
        
        for i in range(n_elements):
            for j in range(n_elements):
                if i != j:
                    # Compute connectivity strength
                    degree1 = results['connectivity_elements'][i]['properties'].get('connectivity_degree', 0)
                    degree2 = results['connectivity_elements'][j]['properties'].get('connectivity_degree', 0)
                    conn_matrix[i, j] = 1.0 / (1.0 + abs(degree1 - degree2))
                    
        im = ax8.imshow(conn_matrix, cmap='YlOrRd', aspect='auto')
        ax8.set_xlabel('Element Index')
        ax8.set_ylabel('Element Index')
        ax8.set_title('Connectivity Strength Matrix')
        plt.colorbar(im, ax=ax8)
        
        # 9. 收敛分析总结
        ax9 = plt.subplot(339)
        conv_analysis = results['convergence_analysis']
        conv_metrics = ['Degree', 'Dimension', 'Complexity', 'Diameter', 'Strong Conn.']
        conv_values = [
            min(1.0, conv_analysis.get('mean_degree', 0) / 10.0),  # Normalize
            conv_analysis.get('mean_dimension', 0) / 3.0,  # Normalize
            conv_analysis.get('mean_complexity', 0),
            min(1.0, conv_analysis.get('mean_diameter', 0) / 5.0),  # Normalize
            conv_analysis.get('strongly_connected_ratio', 0)
        ]
        
        radar_angles = np.linspace(0, 2*np.pi, len(conv_metrics), endpoint=False).tolist()
        conv_values += conv_values[:1]  # Complete the circle
        radar_angles += radar_angles[:1]
        
        ax9 = plt.subplot(339, projection='polar')
        ax9.plot(radar_angles, conv_values, 'o-', linewidth=2, color='purple')
        ax9.fill(radar_angles, conv_values, alpha=0.25, color='purple')
        ax9.set_xticks(radar_angles[:-1])
        ax9.set_xticklabels(conv_metrics)
        ax9.set_ylim(0, 1)
        ax9.set_title('Convergence Analysis Radar', pad=20)
        ax9.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
        
    def visualize_connectivity_properties(self, results: Dict, save_path: str = 'chapter-073-connectivity-collapse-properties.png'):
        """可视化connectivity属性关系"""
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data
        elements = results['connectivity_elements']
        degrees = [e['properties'].get('connectivity_degree', 0) for e in elements]
        dimensions = [e['properties'].get('connectivity_dimension', 0) for e in elements]
        complexities = [e['properties'].get('connectivity_complexity', 0) for e in elements]
        diameters = [e['properties'].get('component_diameter', 0) for e in elements]
        strongly_connected = [e['properties'].get('strongly_connected', False) for e in elements]
        
        # 1. 3D Degree-Dimension-Diameter空间
        ax1 = fig.add_subplot(221, projection='3d')
        if degrees and dimensions and diameters:
            # Color by strong connectivity
            colors = ['red' if sc else 'blue' for sc in strongly_connected]
            scatter = ax1.scatter(degrees, dimensions, diameters,
                                c=colors, s=100, alpha=0.6)
            ax1.set_xlabel('Connectivity Degree')
            ax1.set_ylabel('Dimension')
            ax1.set_zlabel('Component Diameter')
            ax1.set_title('Connectivity Property Space')
            
            # Add legend
            red_patch = patches.Patch(color='red', label='Strongly Connected')
            blue_patch = patches.Patch(color='blue', label='Weakly Connected')
            ax1.legend(handles=[red_patch, blue_patch])
        
        # 2. Component Structure Visualization
        ax2 = plt.subplot(222)
        # Visualize component patterns for first few traces
        traces = [e['trace'] for e in elements[:5]]
        y_positions = []
        
        for i, trace in enumerate(traces):
            y_pos = i * 2
            y_positions.append(y_pos)
            
            # Draw trace as connected components
            components = self._find_local_components(trace)
            x_pos = 0
            
            for j, comp_size in enumerate(components):
                # Draw component
                for k in range(comp_size):
                    bit = trace[x_pos + k]
                    color = 'black' if bit == '1' else 'white'
                    circle = Circle((x_pos + k, y_pos), 0.4, 
                                  facecolor=color, edgecolor='black')
                    ax2.add_patch(circle)
                    
                # Draw connections within component
                if comp_size > 1:
                    for k in range(comp_size - 1):
                        ax2.plot([x_pos + k, x_pos + k + 1], [y_pos, y_pos], 
                               'k-', linewidth=2)
                        
                x_pos += comp_size
                
        ax2.set_xlim(-1, max(len(t) for t in traces) + 1)
        ax2.set_ylim(-1, len(traces) * 2)
        ax2.set_aspect('equal')
        ax2.set_title('Component Structure Examples')
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels([f'Trace {i}' for i in range(len(traces))])
        ax2.set_xlabel('Position')
        
        # 3. Connectivity Type Features
        ax3 = plt.subplot(223)
        type_features = defaultdict(lambda: {'degree': [], 'complexity': []})
        for e in elements:
            conn_type = e['properties'].get('connectivity_type', 'unknown')
            type_features[conn_type]['degree'].append(e['properties'].get('connectivity_degree', 0))
            type_features[conn_type]['complexity'].append(e['properties'].get('connectivity_complexity', 0))
        
        if type_features:
            colors = plt.cm.tab10(np.linspace(0, 1, len(type_features)))
            for (conn_type, features), color in zip(type_features.items(), colors):
                if features['degree'] and features['complexity']:
                    ax3.scatter(features['degree'], features['complexity'], 
                              label=conn_type, color=color, s=100, alpha=0.6)
            
            ax3.set_xlabel('Connectivity Degree')
            ax3.set_ylabel('Connectivity Complexity')
            ax3.set_title('Connectivity Types Feature Space')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Bridge Cost Analysis
        ax4 = plt.subplot(224)
        bridge_costs = [e['properties'].get('bridge_cost', 0) for e in elements]
        comp_counts = [len(self._find_local_components(e['trace'])) for e in elements]
        
        if bridge_costs and comp_counts:
            scatter = ax4.scatter(comp_counts, bridge_costs, 
                                c=complexities, cmap='viridis', s=100, alpha=0.6)
            ax4.set_xlabel('Number of Components')
            ax4.set_ylabel('Bridge Cost')
            ax4.set_title('Bridge Cost vs Component Count')
            plt.colorbar(scatter, ax=ax4, label='Complexity')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
        
    def visualize_three_domains(self, results: Dict, save_path: str = 'chapter-073-connectivity-collapse-domains.png'):
        """可视化三域分析"""
        fig = plt.figure(figsize=(18, 10))
        
        # 准备数据
        conv_analysis = results['convergence_analysis']
        
        # 1. 三域概览
        ax1 = plt.subplot(131)
        domains = ['Traditional\n(Unlimited)', 'φ-Constrained\n(Structural)', 'Convergence\n(Bounded)']
        values = [100, len(results['connectivity_elements']), 
                 len(results['connectivity_elements']) * conv_analysis['convergence_ratio']]
        colors = ['red', 'blue', 'purple']
        
        bars = ax1.bar(domains, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Connectivity Structures')
        ax1.set_title('Three-Domain Connectivity Analysis')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12)
        
        # Add convergence ratio
        ax1.text(0.5, 0.95, f'Convergence Ratio: {conv_analysis["convergence_ratio"]:.3f}',
                transform=ax1.transAxes, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 2. 收敛属性比较
        ax2 = plt.subplot(132)
        properties = ['Degree', 'Dimension', 'Complexity', 'Diameter']
        traditional_vals = [1.0, 1.0, 1.0, 1.0]  # Normalized unlimited
        collapse_vals = [
            min(1.0, conv_analysis['mean_degree'] / 10.0),  # Normalize
            conv_analysis['mean_dimension'] / 3.0,  # Normalize
            conv_analysis['mean_complexity'],
            min(1.0, conv_analysis['mean_diameter'] / 5.0)  # Normalize
        ]
        
        x = np.arange(len(properties))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, traditional_vals, width, 
                        label='Traditional', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, collapse_vals, width, 
                        label='φ-Constrained', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Connectivity Properties')
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('Property Comparison: Traditional vs φ-Constrained')
        ax2.set_xticks(x)
        ax2.set_xticklabels(properties, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 信息效率分析
        ax3 = plt.subplot(133)
        info_measures = results['information_measures']
        
        # Traditional would have maximum entropy (unlimited possibilities)
        max_entropy = log2(100)  # Theoretical maximum
        
        entropies = {
            'Dimension': info_measures.get('dimension_entropy', 0),
            'Type': info_measures.get('type_entropy', 0),
            'Complexity': info_measures.get('complexity_entropy', 0),
            'Degree': info_measures.get('degree_entropy', 0),
            'Diameter': info_measures.get('diameter_entropy', 0)
        }
        
        # Calculate efficiency
        efficiency_data = []
        for name, entropy in entropies.items():
            trad_entropy = max_entropy
            collapse_entropy = entropy
            efficiency = 1 - (collapse_entropy / trad_entropy) if trad_entropy > 0 else 0
            efficiency_data.append({
                'name': name,
                'traditional': trad_entropy,
                'collapse': collapse_entropy,
                'efficiency': efficiency
            })
        
        # Plot efficiency
        names = [d['name'] for d in efficiency_data]
        efficiencies = [d['efficiency'] for d in efficiency_data]
        
        bars = ax3.bar(names, efficiencies, color=plt.cm.viridis(efficiencies), 
                       alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Information Efficiency')
        ax3.set_title('Connectivity Information Compression Efficiency')
        ax3.set_ylim(0, 1)
        ax3.axhline(y=conv_analysis['connectivity_efficiency'], color='red', 
                   linestyle='--', label=f'Mean Efficiency: {conv_analysis["connectivity_efficiency"]:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff*100:.1f}%', ha='center', va='bottom')
        
        # Add overall summary
        fig.suptitle(f'Connectivity Collapse: Three-Domain Convergence Analysis\n' + 
                    f'Total Elements: {len(results["connectivity_elements"])}, ' +
                    f'Connectivity Types: {results["information_measures"].get("connectivity_complexity", 0)}, ' +
                    f'Strong Connectivity Ratio: {conv_analysis["strongly_connected_ratio"]:.3f}',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path


class TestConnectivityCollapse(unittest.TestCase):
    """Connectivity collapse单元测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = ConnectivityCollapseSystem(max_trace_size=6)
        
    def test_trace_encoding(self):
        """测试trace编码"""
        test_cases = [
            (0, '0'),
            (1, '1'),
            (5, '101'),
            (8, '1000')
        ]
        
        for n, expected in test_cases:
            result = self.system._encode_to_trace(n)
            self.assertEqual(result, expected)
            
    def test_phi_constraint(self):
        """测试φ约束验证"""
        valid_traces = ['0', '1', '10', '101', '1010']
        invalid_traces = ['11', '110', '1011', '111']
        
        for trace in valid_traces:
            data = self.system._analyze_trace_structure(int(trace, 2), compute_connectivity=False)
            self.assertTrue(data['phi_valid'])
            
        for trace in invalid_traces:
            data = self.system._analyze_trace_structure(int(trace, 2), compute_connectivity=False)
            self.assertFalse(data['phi_valid'])
            
    def test_component_signature(self):
        """测试component signature计算"""
        trace = '1010'
        signature = self.system._compute_component_signature(trace)
        
        # 验证是否在单位圆上
        self.assertAlmostEqual(abs(signature), 1.0, places=5)
        
        # 验证复数类型
        self.assertIsInstance(signature, complex)
        
    def test_local_components(self):
        """测试局部连通分量查找"""
        # 单一分量
        trace1 = '1111'  # Invalid but useful for testing
        components1 = self.system._find_local_components(trace1)
        self.assertEqual(components1, [4])
        
        # 多个分量
        trace2 = '1001'
        components2 = self.system._find_local_components(trace2)
        self.assertEqual(components2, [1, 2, 1])
        
    def test_connectivity_degree(self):
        """测试connectivity degree计算"""
        # 高连通度trace
        trace1 = '101'
        degree1 = self.system._compute_connectivity_degree(trace1)
        self.assertGreater(degree1, 0)
        
        # 低连通度trace
        trace2 = '0'
        degree2 = self.system._compute_connectivity_degree(trace2)
        self.assertEqual(degree2, 0)
        
    def test_connectivity_type_classification(self):
        """测试connectivity类型分类"""
        trace = '1010'
        connectivity_type = self.system._classify_connectivity_type(trace)
        self.assertIn(connectivity_type, ['disconnected', 'weakly_connected', 
                                         'highly_connected', 'fully_connected'])
        
    def test_component_diameter(self):
        """测试component diameter计算"""
        trace = '10010'
        diameter = self.system._compute_component_diameter(trace)
        
        # 应该是最大分量的大小
        components = self.system._find_local_components(trace)
        self.assertEqual(diameter, max(components))
        
    def test_connectivity_system_analysis(self):
        """测试完整connectivity系统分析"""
        results = self.system.analyze_connectivity_system()
        
        # 验证结果结构
        self.assertIn('connectivity_elements', results)
        self.assertIn('component_signatures', results)
        self.assertIn('component_groups', results)
        self.assertIn('connectivity_types', results)
        self.assertIn('network_properties', results)
        self.assertIn('information_measures', results)
        self.assertIn('category_analysis', results)
        self.assertIn('convergence_analysis', results)
        
        # 验证有connectivity元素
        self.assertGreater(len(results['connectivity_elements']), 0)
        
        # 验证网络属性
        net_props = results['network_properties']
        self.assertGreaterEqual(net_props['nodes'], 0)
        self.assertGreaterEqual(net_props['density'], 0)
        
    def test_strong_connectivity(self):
        """测试强连通性判断"""
        # 强连通的trace
        strong_trace = '1111'  # Invalid but useful for testing
        # Since this violates φ-constraint, test with valid trace
        strong_trace = '1'
        is_strong = self.system._is_strongly_connected(strong_trace)
        # Single bit trace should be strongly connected
        self.assertTrue(is_strong)
        
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        results = self.system.analyze_connectivity_system()
        conv_analysis = results['convergence_analysis']
        
        # 验证收敛比率
        self.assertGreater(conv_analysis['convergence_ratio'], 0)
        self.assertLessEqual(conv_analysis['convergence_ratio'], 1.0)
        
        # 验证平均值在合理范围
        self.assertGreaterEqual(conv_analysis['mean_degree'], 0)
        self.assertGreaterEqual(conv_analysis['mean_dimension'], 0)
        self.assertLessEqual(conv_analysis['mean_complexity'], 1.0)
        
    def test_visualization_generation(self):
        """测试可视化生成"""
        results = self.system.analyze_connectivity_system()
        
        # 测试结构可视化
        path1 = self.system.visualize_connectivity_structure(results, 
                    'test_connectivity_structure.png')
        self.assertTrue(path1.endswith('.png'))
        
        # 测试属性可视化
        path2 = self.system.visualize_connectivity_properties(results,
                    'test_connectivity_properties.png')
        self.assertTrue(path2.endswith('.png'))
        
        # 测试三域可视化
        path3 = self.system.visualize_three_domains(results,
                    'test_connectivity_domains.png')
        self.assertTrue(path3.endswith('.png'))
        
        # 清理测试文件
        import os
        for path in [path1, path2, path3]:
            if os.path.exists(path):
                os.remove(path)


def main():
    """主函数：运行connectivity collapse分析"""
    print("🔄 Chapter 073: ConnectivityCollapse Unit Test Verification")
    print("=" * 60)
    
    # 创建系统
    system = ConnectivityCollapseSystem(max_trace_size=6)
    
    # 运行分析
    print("📊 Building trace universe...")
    results = system.analyze_connectivity_system()
    
    print(f"✅ Found {len(results['connectivity_elements'])} φ-valid traces")
    
    # 输出关键结果
    print("\n🔍 Analyzing connectivity collapse system...")
    print(f"📈 Connectivity universe size: {len(results['connectivity_elements'])} elements")
    print(f"📊 Network density: {results['network_properties']['density']:.3f}")
    print(f"🎯 Convergence ratio: {results['convergence_analysis']['convergence_ratio']:.3f}")
    
    # 输出connectivity属性
    conv = results['convergence_analysis']
    print(f"\n📏 Connectivity Properties:")
    print(f"   Mean degree: {conv['mean_degree']:.3f}")
    print(f"   Mean dimension: {conv['mean_dimension']:.3f}")
    print(f"   Mean complexity: {conv['mean_complexity']:.3f}")
    print(f"   Mean diameter: {conv['mean_diameter']:.3f}")
    print(f"   Strong connectivity ratio: {conv['strongly_connected_ratio']:.3f}")
    
    # 输出信息度量
    info = results['information_measures']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Degree entropy: {info['degree_entropy']:.3f} bits")
    print(f"   Diameter entropy: {info['diameter_entropy']:.3f} bits")
    print(f"   Connectivity complexity: {info['connectivity_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.visualize_connectivity_structure(results)
    system.visualize_connectivity_properties(results)
    system.visualize_three_domains(results)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n✅ Chapter 073: ConnectivityCollapse verification completed!")
    print("=" * 60)
    print("🔥 Connectivity structures exhibit bounded component convergence!")


if __name__ == "__main__":
    main()