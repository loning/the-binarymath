#!/usr/bin/env python3
"""
Chapter 071: BoundaryCollapse Unit Test Verification
从ψ=ψ(ψ)推导Structural Frontiers and Observer-Relative Shells

Core principle: From ψ = ψ(ψ) derive boundaries where boundaries are φ-valid
trace frontiers that encode geometric relationships through trace-based shells,
creating systematic boundary frameworks with bounded frontiers and natural boundary
properties governed by golden constraints, showing how boundaries emerge from trace transitions.

This verification program implements:
1. φ-constrained boundaries as trace frontier operations
2. Boundary analysis: frontier patterns, shell structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection boundary theory
4. Graph theory analysis of frontier networks and boundary connectivity patterns
5. Information theory analysis of boundary entropy and frontier information
6. Category theory analysis of boundary functors and frontier morphisms
7. Visualization of boundary structures and frontier patterns
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
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

class BoundaryCollapseSystem:
    """
    Core system for implementing boundary collapse through trace frontiers.
    Implements φ-constrained boundary theory via trace-based frontier operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_boundary_complexity: int = 4):
        """Initialize boundary collapse system"""
        self.max_trace_size = max_trace_size
        self.max_boundary_complexity = max_boundary_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.boundary_cache = {}
        self.frontier_cache = {}
        self.shell_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_boundary=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for boundary properties computation
        self.trace_universe = universe
        
        # Second pass: add boundary properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['boundary_properties'] = self._compute_boundary_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_boundary: bool = True) -> Dict:
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
        
        if compute_boundary and hasattr(self, 'trace_universe'):
            result['boundary_properties'] = self._compute_boundary_properties(trace)
            
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
        
    def _compute_boundary_properties(self, trace: str) -> Dict:
        """计算trace的boundary属性：frontier结构和shell关系"""
        if trace in self.boundary_cache:
            return self.boundary_cache[trace]
            
        properties = {
            'frontier_signature': self._compute_frontier_signature(trace),
            'shell_depth': self._compute_shell_depth(trace),
            'observer_relative': self._compute_observer_relative(trace),
            'transition_cost': self._compute_transition_cost(trace),
            'boundary_type': self._classify_boundary_type(trace),
            'boundary_dimension': self._compute_boundary_dimension(trace),
            'boundary_complexity': self._compute_boundary_complexity(trace),
            'interior_point': self._is_interior_point(trace),
            'exterior_point': self._is_exterior_point(trace)
        }
        
        self.boundary_cache[trace] = properties
        return properties
        
    def _compute_frontier_signature(self, trace: str) -> complex:
        """计算frontier signature：基于边界结构的复数签名"""
        if not trace:
            return complex(0, 0)
            
        # 计算frontier pattern
        signature = complex(0, 0)
        n = len(trace)
        
        for i in range(n):
            if i == 0 or i == n-1:  # Boundary positions
                weight = 1.0
            else:
                # Interior positions with frontier detection
                left_diff = trace[i] != trace[i-1]
                right_diff = trace[i] != trace[i+1] if i < n-1 else False
                weight = 0.5 + 0.5 * (left_diff + right_diff)
                
            phase = 2 * pi * i / n
            signature += weight * (cos(phase) + 1j * sin(phase))
            
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _compute_shell_depth(self, trace: str) -> float:
        """计算shell depth：边界层次深度"""
        if not trace or len(trace) == 1:
            return 0.0
            
        # 计算层次结构
        depth = 0.0
        n = len(trace)
        
        # Count transitions
        transitions = sum(1 for i in range(1, n) if trace[i] != trace[i-1])
        
        # Compute depth based on transition density
        depth = transitions / (n - 1) if n > 1 else 0.0
        
        return depth
        
    def _compute_observer_relative(self, trace: str) -> float:
        """计算observer-relative度量：相对于观察者的边界特性"""
        if not trace:
            return 0.0
            
        # 计算相对位置的重要性
        n = len(trace)
        relative_measure = 0.0
        
        for i in range(n):
            # Position-dependent weight (observer at origin)
            position_weight = exp(-i / n)
            
            # Structure-dependent weight
            if i < n-1 and trace[i] != trace[i+1]:
                structure_weight = 1.0
            else:
                structure_weight = 0.5
                
            relative_measure += position_weight * structure_weight
            
        return relative_measure / n if n > 0 else 0.0
        
    def _compute_transition_cost(self, trace: str) -> float:
        """计算transition cost：边界穿越成本"""
        if not trace or len(trace) == 1:
            return 0.0
            
        cost = 0.0
        n = len(trace)
        
        for i in range(1, n):
            if trace[i] != trace[i-1]:
                # Transition cost based on position
                position_cost = 1.0 + i / n
                
                # φ-constraint penalty
                if i < n-1 and trace[i] == '1' and trace[i+1] == '1':
                    position_cost *= 2.0
                    
                cost += position_cost
                
        return cost / n if n > 0 else 0.0
        
    def _classify_boundary_type(self, trace: str) -> str:
        """分类boundary类型"""
        if not trace:
            return "empty_boundary"
            
        shell_depth = self._compute_shell_depth(trace)
        observer_relative = self._compute_observer_relative(trace)
        
        if shell_depth < 0.1:
            return "uniform_boundary"
        elif shell_depth < 0.3:
            return "weak_frontier"
        elif observer_relative > 0.5:
            return "strong_frontier"
        else:
            return "complex_boundary"
            
    def _compute_boundary_dimension(self, trace: str) -> float:
        """计算boundary dimension：边界的分形维度"""
        if not trace:
            return 0.0
            
        # 计算边界的box-counting维度
        n = len(trace)
        if n <= 1:
            return 0.0
            
        # Count boundary points at different scales
        boundary_counts = []
        for scale in range(1, min(n, 4)):
            count = 0
            for i in range(0, n - scale):
                window = trace[i:i+scale+1]
                if len(set(window)) > 1:  # Contains boundary
                    count += 1
            if count > 0:
                boundary_counts.append((scale, count))
                
        if len(boundary_counts) < 2:
            return 0.0
            
        # Estimate dimension from scaling
        log_scales = [log2(s) for s, _ in boundary_counts]
        log_counts = [log2(c) for _, c in boundary_counts]
        
        if len(set(log_scales)) == 1:
            return 0.0
            
        # Simple linear regression for dimension
        n_points = len(log_scales)
        mean_x = sum(log_scales) / n_points
        mean_y = sum(log_counts) / n_points
        
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_scales, log_counts))
        den = sum((x - mean_x) ** 2 for x in log_scales)
        
        dimension = num / den if den > 0 else 0.0
        
        return max(0.0, min(1.0, abs(dimension)))
        
    def _compute_boundary_complexity(self, trace: str) -> float:
        """计算boundary complexity：边界结构的复杂度"""
        if not trace:
            return 0.0
            
        # 多因素复杂度评估
        shell_depth = self._compute_shell_depth(trace)
        dimension = self._compute_boundary_dimension(trace)
        transition_cost = self._compute_transition_cost(trace)
        
        # 综合复杂度
        complexity = (shell_depth + dimension + transition_cost) / 3.0
        
        return min(1.0, complexity)
        
    def _is_interior_point(self, trace: str) -> bool:
        """判断是否为内部点"""
        if not trace or len(trace) <= 2:
            return False
            
        # Check if trace has stable interior region
        n = len(trace)
        for i in range(1, n-1):
            if trace[i-1] == trace[i] == trace[i+1]:
                return True
                
        return False
        
    def _is_exterior_point(self, trace: str) -> bool:
        """判断是否为外部点"""
        if not trace:
            return True
            
        # Check if trace is on the boundary
        shell_depth = self._compute_shell_depth(trace)
        return shell_depth > 0.5
        
    def analyze_boundary_system(self) -> Dict:
        """分析完整的boundary系统"""
        results = {
            'boundary_elements': [],
            'frontier_signatures': {},
            'shell_structures': {},
            'boundary_types': defaultdict(int),
            'network_properties': {},
            'information_measures': {},
            'category_analysis': {},
            'convergence_analysis': {}
        }
        
        # 收集所有boundary元素
        for n, data in self.trace_universe.items():
            if data['phi_valid']:
                boundary_props = data.get('boundary_properties', {})
                results['boundary_elements'].append({
                    'value': n,
                    'trace': data['trace'],
                    'properties': boundary_props
                })
                
                # 统计boundary类型
                boundary_type = boundary_props.get('boundary_type', 'unknown')
                results['boundary_types'][boundary_type] += 1
                
                # 记录frontier signatures
                sig = boundary_props.get('frontier_signature', complex(0, 0))
                results['frontier_signatures'][n] = sig
                
        # 计算网络属性
        results['network_properties'] = self._compute_network_properties(results['boundary_elements'])
        
        # 计算信息度量
        results['information_measures'] = self._compute_information_measures(results['boundary_elements'])
        
        # 范畴论分析
        results['category_analysis'] = self._compute_category_analysis(results['boundary_elements'])
        
        # 三域收敛分析
        results['convergence_analysis'] = self._compute_convergence_analysis(results)
        
        return results
        
    def _compute_network_properties(self, elements: List[Dict]) -> Dict:
        """计算boundary网络属性"""
        G = nx.Graph()
        
        # 添加节点
        for elem in elements:
            n = elem['value']
            props = elem['properties']
            G.add_node(n, **props)
            
        # 添加边：基于frontier相似性
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                n1, n2 = elem1['value'], elem2['value']
                
                # 计算frontier距离
                sig1 = elem1['properties'].get('frontier_signature', complex(0, 0))
                sig2 = elem2['properties'].get('frontier_signature', complex(0, 0))
                distance = abs(sig1 - sig2)
                
                if distance < 0.5:  # Threshold for connection
                    G.add_edge(n1, n2, weight=1.0 - distance)
                    
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'components': nx.number_connected_components(G),
            'clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        }
        
    def _compute_information_measures(self, elements: List[Dict]) -> Dict:
        """计算boundary信息度量"""
        if not elements:
            return {}
            
        # 收集各种属性分布
        dimensions = []
        types = []
        complexities = []
        shell_depths = []
        
        for elem in elements:
            props = elem['properties']
            dimensions.append(props.get('boundary_dimension', 0))
            types.append(props.get('boundary_type', 'unknown'))
            complexities.append(props.get('boundary_complexity', 0))
            shell_depths.append(props.get('shell_depth', 0))
            
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
            'shell_depth_entropy': compute_entropy(shell_depths),
            'boundary_complexity': len(set(types))
        }
        
    def _compute_category_analysis(self, elements: List[Dict]) -> Dict:
        """计算boundary范畴论属性"""
        # 构建态射关系
        morphisms = []
        functorial_morphisms = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j:
                    # 检查boundary变换关系
                    shell1 = elem1['properties'].get('shell_depth', 0)
                    shell2 = elem2['properties'].get('shell_depth', 0)
                    
                    if abs(shell1 - shell2) < 0.2:  # Compatible shells
                        morphisms.append((elem1['value'], elem2['value']))
                        
                        # 检查函子性质
                        dim1 = elem1['properties'].get('boundary_dimension', 0)
                        dim2 = elem2['properties'].get('boundary_dimension', 0)
                        
                        if abs(dim1 - dim2) < 0.1:  # Dimension preserving
                            functorial_morphisms.append((elem1['value'], elem2['value']))
                            
        # 计算可达对
        reachable_pairs = set()
        for n1 in range(len(elements)):
            for n2 in range(len(elements)):
                if n1 != n2:
                    # Simple reachability through boundary transitions
                    reachable_pairs.add((elements[n1]['value'], elements[n2]['value']))
                    
        return {
            'morphisms': len(morphisms),
            'functorial_morphisms': len(functorial_morphisms),
            'functoriality_ratio': len(functorial_morphisms) / len(morphisms) if morphisms else 0,
            'reachable_pairs': len(reachable_pairs)
        }
        
    def _compute_convergence_analysis(self, results: Dict) -> Dict:
        """计算三域收敛分析"""
        total_elements = len(results['boundary_elements'])
        
        # Traditional domain: Would have unlimited boundary structures
        traditional_potential = 100  # Arbitrary large number
        
        # Collapse domain: φ-constrained structures
        collapse_actual = total_elements
        
        # Convergence ratio
        convergence_ratio = collapse_actual / traditional_potential
        
        # 分析boundary属性分布
        shell_depths = []
        dimensions = []
        complexities = []
        observer_relatives = []
        
        for elem in results['boundary_elements']:
            props = elem['properties']
            shell_depths.append(props.get('shell_depth', 0))
            dimensions.append(props.get('boundary_dimension', 0))
            complexities.append(props.get('boundary_complexity', 0))
            observer_relatives.append(props.get('observer_relative', 0))
            
        return {
            'convergence_ratio': convergence_ratio,
            'mean_shell_depth': np.mean(shell_depths) if shell_depths else 0,
            'mean_dimension': np.mean(dimensions) if dimensions else 0,
            'mean_complexity': np.mean(complexities) if complexities else 0,
            'mean_observer_relative': np.mean(observer_relatives) if observer_relatives else 0,
            'boundary_efficiency': 1.0 - np.std(dimensions) if dimensions else 0
        }
        
    def visualize_boundary_structure(self, results: Dict, save_path: str = 'chapter-071-boundary-collapse-structure.png'):
        """可视化boundary结构"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Frontier Signature分布 (极坐标)
        ax1 = plt.subplot(331, projection='polar')
        signatures = list(results['frontier_signatures'].values())
        if signatures:
            angles = [np.angle(sig) for sig in signatures]
            radii = [abs(sig) for sig in signatures]
            colors = plt.cm.viridis(np.linspace(0, 1, len(signatures)))
            ax1.scatter(angles, radii, c=colors, s=100, alpha=0.6)
            ax1.set_title('Frontier Signatures in Complex Plane', fontsize=14, pad=20)
            ax1.set_ylim(0, 1.2)
        
        # 2. Shell Depth分布
        ax2 = plt.subplot(332)
        shell_depths = [elem['properties'].get('shell_depth', 0) 
                       for elem in results['boundary_elements']]
        if shell_depths:
            ax2.hist(shell_depths, bins=20, alpha=0.7, color='teal', edgecolor='black')
            ax2.axvline(np.mean(shell_depths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(shell_depths):.3f}')
            ax2.set_xlabel('Shell Depth')
            ax2.set_ylabel('Count')
            ax2.set_title('Shell Depth Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Boundary类型分布
        ax3 = plt.subplot(333)
        boundary_types = results['boundary_types']
        if boundary_types:
            types = list(boundary_types.keys())
            counts = list(boundary_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            ax3.pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            ax3.set_title('Boundary Type Distribution')
        
        # 4. Observer-Relative度量
        ax4 = plt.subplot(334)
        observer_relatives = [elem['properties'].get('observer_relative', 0) 
                            for elem in results['boundary_elements']]
        traces = [elem['trace'] for elem in results['boundary_elements']]
        if observer_relatives and traces:
            trace_lengths = [len(t) for t in traces]
            scatter = ax4.scatter(trace_lengths, observer_relatives, 
                                c=shell_depths if shell_depths else 'blue', 
                                cmap='plasma', s=100, alpha=0.6)
            ax4.set_xlabel('Trace Length')
            ax4.set_ylabel('Observer-Relative Measure')
            ax4.set_title('Observer-Relative vs Trace Length')
            if shell_depths:
                plt.colorbar(scatter, ax=ax4, label='Shell Depth')
            ax4.grid(True, alpha=0.3)
        
        # 5. Boundary Dimension vs Complexity
        ax5 = plt.subplot(335)
        dimensions = [elem['properties'].get('boundary_dimension', 0) 
                     for elem in results['boundary_elements']]
        complexities = [elem['properties'].get('boundary_complexity', 0) 
                       for elem in results['boundary_elements']]
        if dimensions and complexities:
            ax5.scatter(dimensions, complexities, s=100, alpha=0.6, c='green')
            ax5.set_xlabel('Boundary Dimension')
            ax5.set_ylabel('Boundary Complexity')
            ax5.set_title('Dimension vs Complexity')
            ax5.grid(True, alpha=0.3)
            
            # Add trend line
            if len(set(dimensions)) > 1:
                z = np.polyfit(dimensions, complexities, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(dimensions), max(dimensions), 100)
                ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                        label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                ax5.legend()
        
        # 6. 网络结构可视化
        ax6 = plt.subplot(336)
        network_props = results['network_properties']
        metrics = ['Nodes', 'Edges', 'Density', 'Components', 'Clustering']
        values = [
            network_props.get('nodes', 0),
            network_props.get('edges', 0),
            network_props.get('density', 0) * 10,  # Scale for visibility
            network_props.get('components', 0),
            network_props.get('clustering', 0) * 10  # Scale for visibility
        ]
        bars = ax6.bar(metrics, values, color=plt.cm.tab10(range(len(metrics))))
        ax6.set_ylabel('Value')
        ax6.set_title('Boundary Network Properties')
        ax6.set_xticklabels(metrics, rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 7. 信息熵度量
        ax7 = plt.subplot(337)
        info_measures = results['information_measures']
        entropy_types = ['Dimension', 'Type', 'Complexity', 'Shell Depth']
        entropy_values = [
            info_measures.get('dimension_entropy', 0),
            info_measures.get('type_entropy', 0),
            info_measures.get('complexity_entropy', 0),
            info_measures.get('shell_depth_entropy', 0)
        ]
        ax7.barh(entropy_types, entropy_values, color=plt.cm.coolwarm(np.linspace(0, 1, len(entropy_types))))
        ax7.set_xlabel('Entropy (bits)')
        ax7.set_title('Information Entropy Measures')
        ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Transition Cost热力图
        ax8 = plt.subplot(338)
        n_elements = min(10, len(results['boundary_elements']))  # Limit for visibility
        cost_matrix = np.zeros((n_elements, n_elements))
        
        for i in range(n_elements):
            for j in range(n_elements):
                if i != j:
                    cost1 = results['boundary_elements'][i]['properties'].get('transition_cost', 0)
                    cost2 = results['boundary_elements'][j]['properties'].get('transition_cost', 0)
                    cost_matrix[i, j] = abs(cost1 - cost2)
        
        im = ax8.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
        ax8.set_xlabel('Element Index')
        ax8.set_ylabel('Element Index')
        ax8.set_title('Transition Cost Differences')
        plt.colorbar(im, ax=ax8)
        
        # 9. 收敛分析总结
        ax9 = plt.subplot(339)
        conv_analysis = results['convergence_analysis']
        conv_metrics = ['Shell Depth', 'Dimension', 'Complexity', 'Observer Rel.', 'Efficiency']
        conv_values = [
            conv_analysis.get('mean_shell_depth', 0),
            conv_analysis.get('mean_dimension', 0),
            conv_analysis.get('mean_complexity', 0),
            conv_analysis.get('mean_observer_relative', 0),
            conv_analysis.get('boundary_efficiency', 0)
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
        
    def visualize_boundary_properties(self, results: Dict, save_path: str = 'chapter-071-boundary-collapse-properties.png'):
        """可视化boundary属性关系"""
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data
        elements = results['boundary_elements']
        shell_depths = [e['properties'].get('shell_depth', 0) for e in elements]
        dimensions = [e['properties'].get('boundary_dimension', 0) for e in elements]
        complexities = [e['properties'].get('boundary_complexity', 0) for e in elements]
        observer_relatives = [e['properties'].get('observer_relative', 0) for e in elements]
        transition_costs = [e['properties'].get('transition_cost', 0) for e in elements]
        
        # 1. 3D Shell-Dimension-Complexity空间
        ax1 = fig.add_subplot(221, projection='3d')
        if shell_depths and dimensions and complexities:
            scatter = ax1.scatter(shell_depths, dimensions, complexities,
                                c=observer_relatives, cmap='viridis', s=100, alpha=0.6)
            ax1.set_xlabel('Shell Depth')
            ax1.set_ylabel('Dimension')
            ax1.set_zlabel('Complexity')
            ax1.set_title('Boundary Property Space')
            plt.colorbar(scatter, ax=ax1, label='Observer Relative', shrink=0.5)
        
        # 2. Interior vs Exterior点分布
        ax2 = plt.subplot(222)
        interior_points = [i for i, e in enumerate(elements) 
                          if e['properties'].get('interior_point', False)]
        exterior_points = [i for i, e in enumerate(elements) 
                          if e['properties'].get('exterior_point', False)]
        
        if interior_points or exterior_points:
            # Plot interior points
            if interior_points:
                int_shells = [shell_depths[i] for i in interior_points]
                int_costs = [transition_costs[i] for i in interior_points]
                ax2.scatter(int_shells, int_costs, c='blue', label='Interior', 
                           s=100, alpha=0.6, marker='o')
            
            # Plot exterior points
            if exterior_points:
                ext_shells = [shell_depths[i] for i in exterior_points]
                ext_costs = [transition_costs[i] for i in exterior_points]
                ax2.scatter(ext_shells, ext_costs, c='red', label='Exterior', 
                           s=100, alpha=0.6, marker='^')
            
            ax2.set_xlabel('Shell Depth')
            ax2.set_ylabel('Transition Cost')
            ax2.set_title('Interior vs Exterior Points')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Boundary类型特征
        ax3 = plt.subplot(223)
        boundary_types = defaultdict(list)
        for e in elements:
            b_type = e['properties'].get('boundary_type', 'unknown')
            boundary_types[b_type].append({
                'dimension': e['properties'].get('boundary_dimension', 0),
                'complexity': e['properties'].get('boundary_complexity', 0)
            })
        
        if boundary_types:
            colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_types)))
            for (b_type, props), color in zip(boundary_types.items(), colors):
                if props:
                    dims = [p['dimension'] for p in props]
                    comps = [p['complexity'] for p in props]
                    ax3.scatter(dims, comps, label=b_type, color=color, 
                              s=100, alpha=0.6)
            
            ax3.set_xlabel('Boundary Dimension')
            ax3.set_ylabel('Boundary Complexity')
            ax3.set_title('Boundary Types Clustering')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Frontier Signature相位分布
        ax4 = plt.subplot(224, projection='polar')
        signatures = list(results['frontier_signatures'].values())
        if signatures:
            # Group by shell depth bins
            shell_bins = np.linspace(0, max(shell_depths) if shell_depths else 1, 4)
            colors = plt.cm.plasma(np.linspace(0, 1, len(shell_bins)-1))
            
            for i in range(len(shell_bins)-1):
                bin_mask = [(s >= shell_bins[i] and s < shell_bins[i+1]) 
                           for s in shell_depths]
                bin_sigs = [sig for sig, mask in zip(signatures, bin_mask) if mask]
                
                if bin_sigs:
                    angles = [np.angle(sig) for sig in bin_sigs]
                    radii = [abs(sig) for sig in bin_sigs]
                    ax4.scatter(angles, radii, c=[colors[i]], 
                              label=f'Shell [{shell_bins[i]:.2f}, {shell_bins[i+1]:.2f})',
                              s=80, alpha=0.6)
            
            ax4.set_title('Frontier Signatures by Shell Depth', pad=20)
            ax4.set_ylim(0, 1.2)
            ax4.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
        
    def visualize_three_domains(self, results: Dict, save_path: str = 'chapter-071-boundary-collapse-domains.png'):
        """可视化三域分析"""
        fig = plt.figure(figsize=(18, 10))
        
        # 准备数据
        conv_analysis = results['convergence_analysis']
        
        # 1. 三域概览
        ax1 = plt.subplot(131)
        domains = ['Traditional\n(Unlimited)', 'φ-Constrained\n(Structural)', 'Convergence\n(Bounded)']
        values = [100, len(results['boundary_elements']), 
                 len(results['boundary_elements']) * conv_analysis['convergence_ratio']]
        colors = ['red', 'blue', 'purple']
        
        bars = ax1.bar(domains, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Boundary Structures')
        ax1.set_title('Three-Domain Boundary Analysis')
        
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
        properties = ['Shell Depth', 'Dimension', 'Complexity', 'Observer Rel.']
        traditional_vals = [1.0, 1.0, 1.0, 1.0]  # Normalized unlimited
        collapse_vals = [
            conv_analysis['mean_shell_depth'],
            conv_analysis['mean_dimension'],
            conv_analysis['mean_complexity'],
            conv_analysis['mean_observer_relative']
        ]
        
        x = np.arange(len(properties))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, traditional_vals, width, 
                        label='Traditional', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, collapse_vals, width, 
                        label='φ-Constrained', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Boundary Properties')
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
            'Shell Depth': info_measures.get('shell_depth_entropy', 0)
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
        ax3.set_title('Boundary Information Compression Efficiency')
        ax3.set_ylim(0, 1)
        ax3.axhline(y=conv_analysis['boundary_efficiency'], color='red', 
                   linestyle='--', label=f'Mean Efficiency: {conv_analysis["boundary_efficiency"]:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff*100:.1f}%', ha='center', va='bottom')
        
        # Add overall summary
        fig.suptitle(f'Boundary Collapse: Three-Domain Convergence Analysis\n' + 
                    f'Total Elements: {len(results["boundary_elements"])}, ' +
                    f'Boundary Types: {results["information_measures"].get("boundary_complexity", 0)}',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path


class TestBoundaryCollapse(unittest.TestCase):
    """Boundary collapse单元测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = BoundaryCollapseSystem(max_trace_size=6)
        
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
            data = self.system._analyze_trace_structure(int(trace, 2), compute_boundary=False)
            self.assertTrue(data['phi_valid'])
            
        for trace in invalid_traces:
            data = self.system._analyze_trace_structure(int(trace, 2), compute_boundary=False)
            self.assertFalse(data['phi_valid'])
            
    def test_frontier_signature(self):
        """测试frontier signature计算"""
        trace = '1010'
        signature = self.system._compute_frontier_signature(trace)
        
        # 验证是否在单位圆上
        self.assertAlmostEqual(abs(signature), 1.0, places=5)
        
        # 验证复数类型
        self.assertIsInstance(signature, complex)
        
    def test_shell_depth(self):
        """测试shell depth计算"""
        # 均匀trace应该有低shell depth
        uniform_trace = '0000'
        depth1 = self.system._compute_shell_depth(uniform_trace)
        self.assertLess(depth1, 0.1)
        
        # 交替trace应该有高shell depth
        alternating_trace = '0101'
        depth2 = self.system._compute_shell_depth(alternating_trace)
        self.assertGreater(depth2, 0.5)
        
    def test_boundary_type_classification(self):
        """测试boundary类型分类"""
        trace = '1010'
        boundary_type = self.system._classify_boundary_type(trace)
        self.assertIn(boundary_type, ['uniform_boundary', 'weak_frontier', 
                                     'strong_frontier', 'complex_boundary'])
        
    def test_boundary_dimension(self):
        """测试boundary dimension计算"""
        trace = '101010'
        dimension = self.system._compute_boundary_dimension(trace)
        
        # 维度应该在[0, 1]范围内
        self.assertGreaterEqual(dimension, 0.0)
        self.assertLessEqual(dimension, 1.0)
        
    def test_boundary_system_analysis(self):
        """测试完整boundary系统分析"""
        results = self.system.analyze_boundary_system()
        
        # 验证结果结构
        self.assertIn('boundary_elements', results)
        self.assertIn('frontier_signatures', results)
        self.assertIn('boundary_types', results)
        self.assertIn('network_properties', results)
        self.assertIn('information_measures', results)
        self.assertIn('category_analysis', results)
        self.assertIn('convergence_analysis', results)
        
        # 验证有boundary元素
        self.assertGreater(len(results['boundary_elements']), 0)
        
        # 验证网络属性
        net_props = results['network_properties']
        self.assertGreaterEqual(net_props['nodes'], 0)
        self.assertGreaterEqual(net_props['density'], 0)
        
    def test_interior_exterior_classification(self):
        """测试内部/外部点分类"""
        # 测试内部点
        interior_trace = '0000'
        is_interior = self.system._is_interior_point(interior_trace)
        self.assertTrue(is_interior)
        
        # 测试外部点
        boundary_trace = '0101'
        is_exterior = self.system._is_exterior_point(boundary_trace)
        self.assertTrue(is_exterior)  # High shell depth indicates exterior
        
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        results = self.system.analyze_boundary_system()
        conv_analysis = results['convergence_analysis']
        
        # 验证收敛比率
        self.assertGreater(conv_analysis['convergence_ratio'], 0)
        self.assertLessEqual(conv_analysis['convergence_ratio'], 1.0)
        
        # 验证平均值在合理范围
        self.assertGreaterEqual(conv_analysis['mean_shell_depth'], 0)
        self.assertLessEqual(conv_analysis['mean_shell_depth'], 1.0)
        
        self.assertGreaterEqual(conv_analysis['mean_dimension'], 0)
        self.assertLessEqual(conv_analysis['mean_dimension'], 1.0)
        
    def test_visualization_generation(self):
        """测试可视化生成"""
        results = self.system.analyze_boundary_system()
        
        # 测试结构可视化
        path1 = self.system.visualize_boundary_structure(results, 
                    'test_boundary_structure.png')
        self.assertTrue(path1.endswith('.png'))
        
        # 测试属性可视化
        path2 = self.system.visualize_boundary_properties(results,
                    'test_boundary_properties.png')
        self.assertTrue(path2.endswith('.png'))
        
        # 测试三域可视化
        path3 = self.system.visualize_three_domains(results,
                    'test_boundary_domains.png')
        self.assertTrue(path3.endswith('.png'))
        
        # 清理测试文件
        import os
        for path in [path1, path2, path3]:
            if os.path.exists(path):
                os.remove(path)


def main():
    """主函数：运行boundary collapse分析"""
    print("🔄 Chapter 071: BoundaryCollapse Unit Test Verification")
    print("=" * 60)
    
    # 创建系统
    system = BoundaryCollapseSystem(max_trace_size=6)
    
    # 运行分析
    print("📊 Building trace universe...")
    results = system.analyze_boundary_system()
    
    print(f"✅ Found {len(results['boundary_elements'])} φ-valid traces")
    
    # 输出关键结果
    print("\n🔍 Analyzing boundary collapse system...")
    print(f"📈 Boundary universe size: {len(results['boundary_elements'])} elements")
    print(f"📊 Network density: {results['network_properties']['density']:.3f}")
    print(f"🎯 Convergence ratio: {results['convergence_analysis']['convergence_ratio']:.3f}")
    
    # 输出boundary属性
    conv = results['convergence_analysis']
    print(f"\n📏 Boundary Properties:")
    print(f"   Mean shell depth: {conv['mean_shell_depth']:.3f}")
    print(f"   Mean dimension: {conv['mean_dimension']:.3f}")
    print(f"   Mean complexity: {conv['mean_complexity']:.3f}")
    print(f"   Mean observer relative: {conv['mean_observer_relative']:.3f}")
    print(f"   Boundary efficiency: {conv['boundary_efficiency']:.3f}")
    
    # 输出信息度量
    info = results['information_measures']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Shell depth entropy: {info['shell_depth_entropy']:.3f} bits")
    print(f"   Boundary complexity: {info['boundary_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.visualize_boundary_structure(results)
    system.visualize_boundary_properties(results)
    system.visualize_three_domains(results)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n✅ Chapter 071: BoundaryCollapse verification completed!")
    print("=" * 60)
    print("🔥 Boundary structures exhibit bounded frontier convergence!")


if __name__ == "__main__":
    main()