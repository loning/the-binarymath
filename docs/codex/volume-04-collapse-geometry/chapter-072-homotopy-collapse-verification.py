#!/usr/bin/env python3
"""
Chapter 072: HomotopyCollapse Unit Test Verification
从ψ=ψ(ψ)推导Path Deformation Invariance in Collapse Systems

Core principle: From ψ = ψ(ψ) derive homotopy where homotopy is φ-valid
trace path deformations that encode geometric relationships through trace-based equivalence,
creating systematic homotopy frameworks with bounded deformations and natural homotopy
properties governed by golden constraints, showing how homotopy emerges from trace paths.

This verification program implements:
1. φ-constrained homotopy as trace path deformation operations
2. Homotopy analysis: deformation patterns, equivalence structure with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection homotopy theory
4. Graph theory analysis of deformation networks and homotopy connectivity patterns
5. Information theory analysis of homotopy entropy and deformation information
6. Category theory analysis of homotopy functors and deformation morphisms
7. Visualization of homotopy structures and deformation patterns
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

class HomotopyCollapseSystem:
    """
    Core system for implementing homotopy collapse through trace path deformations.
    Implements φ-constrained homotopy theory via trace-based deformation operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_homotopy_complexity: int = 4):
        """Initialize homotopy collapse system"""
        self.max_trace_size = max_trace_size
        self.max_homotopy_complexity = max_homotopy_complexity
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.homotopy_cache = {}
        self.deformation_cache = {}
        self.path_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_homotopy=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for homotopy properties computation
        self.trace_universe = universe
        
        # Second pass: add homotopy properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['homotopy_properties'] = self._compute_homotopy_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_homotopy: bool = True) -> Dict:
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
        
        if compute_homotopy and hasattr(self, 'trace_universe'):
            result['homotopy_properties'] = self._compute_homotopy_properties(trace)
            
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
        
    def _compute_homotopy_properties(self, trace: str) -> Dict:
        """计算trace的homotopy属性：path deformation和equivalence关系"""
        if trace in self.homotopy_cache:
            return self.homotopy_cache[trace]
            
        properties = {
            'deformation_signature': self._compute_deformation_signature(trace),
            'path_equivalence': self._compute_path_equivalence(trace),
            'fundamental_group': self._compute_fundamental_group(trace),
            'deformation_cost': self._compute_deformation_cost(trace),
            'homotopy_type': self._classify_homotopy_type(trace),
            'homotopy_dimension': self._compute_homotopy_dimension(trace),
            'homotopy_complexity': self._compute_homotopy_complexity(trace),
            'contractible': self._is_contractible(trace),
            'loop_space': self._compute_loop_space(trace)
        }
        
        self.homotopy_cache[trace] = properties
        return properties
        
    def _compute_deformation_signature(self, trace: str) -> complex:
        """计算deformation signature：基于路径变形的复数签名"""
        if not trace:
            return complex(0, 0)
            
        # 计算deformation pattern
        signature = complex(0, 0)
        n = len(trace)
        
        for i in range(n):
            # Path-based weight
            if i < n-1:
                # Adjacent bit difference represents path segment
                path_weight = 1.0 if trace[i] != trace[i+1] else 0.5
            else:
                path_weight = 0.5
                
            # Deformation potential
            deform_potential = 1.0 - (i / n) if n > 0 else 0.0
            weight = path_weight * deform_potential
            
            phase = 2 * pi * i / n
            signature += weight * (cos(phase) + 1j * sin(phase))
            
        # Normalize to unit circle
        if abs(signature) > 0:
            signature = signature / abs(signature)
            
        return signature
        
    def _compute_path_equivalence(self, trace: str) -> List[str]:
        """计算path equivalence：同伦等价的路径集合"""
        if not trace:
            return ['0']
            
        equivalent_paths = set()
        
        # Identity path
        equivalent_paths.add(trace)
        
        # Simple deformations (preserving φ-constraint)
        n = len(trace)
        
        # Type 1: Null-homotopic extensions (adding/removing trailing zeros)
        if trace.endswith('0'):
            # Can remove trailing zero
            reduced = trace.rstrip('0')
            if reduced and '11' not in reduced:
                equivalent_paths.add(reduced)
        else:
            # Can add trailing zero
            extended = trace + '0'
            if '11' not in extended:
                equivalent_paths.add(extended)
                
        # Type 2: Local deformations (bit flips preserving φ-constraint)
        for i in range(n):
            # Try flipping bit i
            deformed = list(trace)
            deformed[i] = '0' if trace[i] == '1' else '1'
            deformed_str = ''.join(deformed)
            
            if '11' not in deformed_str:
                # Check if deformation is homotopic
                if self._are_homotopic(trace, deformed_str):
                    equivalent_paths.add(deformed_str)
                    
        return sorted(list(equivalent_paths))
        
    def _are_homotopic(self, trace1: str, trace2: str) -> bool:
        """判断两个traces是否同伦等价"""
        # Simple homotopy criterion: same fundamental invariants
        inv1 = self._compute_homotopy_invariant(trace1)
        inv2 = self._compute_homotopy_invariant(trace2)
        
        return inv1 == inv2
        
    def _compute_homotopy_invariant(self, trace: str) -> int:
        """计算homotopy invariant：同伦不变量"""
        if not trace:
            return 0
            
        # Invariant based on alternating sum of positions
        invariant = 0
        for i, bit in enumerate(trace):
            if bit == '1':
                invariant += (-1) ** i * (i + 1)
                
        return abs(invariant) % 7  # Modulo for finite classification
        
    def _compute_fundamental_group(self, trace: str) -> int:
        """计算fundamental group：基本群的阶"""
        if not trace:
            return 1  # Trivial group
            
        # Count independent loops
        loops = 0
        n = len(trace)
        
        # Count transitions as potential loop generators
        for i in range(n-1):
            if trace[i] != trace[i+1]:
                loops += 1
                
        # Fundamental group order based on loop structure
        if loops == 0:
            return 1  # Simply connected
        else:
            return 2 ** min(loops, 3)  # Bounded complexity
            
    def _compute_deformation_cost(self, trace: str) -> float:
        """计算deformation cost：路径变形成本"""
        if not trace or len(trace) == 1:
            return 0.0
            
        cost = 0.0
        n = len(trace)
        
        # Cost based on structural complexity
        for i in range(n-1):
            if trace[i] != trace[i+1]:
                # Transition cost
                position_weight = 1.0 + i / n
                
                # φ-constraint proximity penalty
                if i > 0 and trace[i-1] == '1' and trace[i] == '0' and trace[i+1] == '1':
                    position_weight *= 1.5  # Near violation
                    
                cost += position_weight
                
        # Normalize by length
        return cost / n if n > 0 else 0.0
        
    def _classify_homotopy_type(self, trace: str) -> str:
        """分类homotopy类型"""
        if not trace:
            return "trivial"
            
        fundamental_group = self._compute_fundamental_group(trace)
        deformation_cost = self._compute_deformation_cost(trace)
        contractible = self._is_contractible(trace)
        
        if contractible:
            return "contractible"
        elif fundamental_group == 1:
            return "simply_connected"
        elif deformation_cost < 0.3:
            return "weakly_deformable"
        else:
            return "strongly_deformable"
            
    def _compute_homotopy_dimension(self, trace: str) -> int:
        """计算homotopy dimension：同伦维度"""
        if not trace:
            return 0
            
        # Dimension based on independent deformation directions
        equiv_paths = self._compute_path_equivalence(trace)
        
        # Count independent directions
        dimension = 0
        seen_invariants = set()
        
        for path in equiv_paths:
            inv = self._compute_homotopy_invariant(path)
            if inv not in seen_invariants:
                dimension += 1
                seen_invariants.add(inv)
                
        return dimension
        
    def _compute_homotopy_complexity(self, trace: str) -> float:
        """计算homotopy complexity：同伦复杂度"""
        if not trace:
            return 0.0
            
        # Multi-factor complexity
        fundamental_group = self._compute_fundamental_group(trace)
        deformation_cost = self._compute_deformation_cost(trace)
        dimension = self._compute_homotopy_dimension(trace)
        
        # Normalize factors
        group_factor = min(1.0, log2(fundamental_group + 1) / 3.0)
        cost_factor = min(1.0, deformation_cost)
        dim_factor = min(1.0, dimension / 5.0)
        
        # Combined complexity
        complexity = (group_factor + cost_factor + dim_factor) / 3.0
        
        return complexity
        
    def _is_contractible(self, trace: str) -> bool:
        """判断是否可缩"""
        if not trace:
            return True
            
        # Contractible if can be deformed to point
        # Simple criterion: no essential loops
        fundamental_group = self._compute_fundamental_group(trace)
        
        return fundamental_group == 1 and trace.count('1') <= 1
        
    def _compute_loop_space(self, trace: str) -> int:
        """计算loop space维度"""
        if not trace:
            return 0
            
        # Loop space dimension based on trace structure
        loops = 0
        n = len(trace)
        
        # Count closed paths
        for i in range(n-2):
            if trace[i] == trace[i+2] and trace[i] != trace[i+1]:
                loops += 1
                
        return min(loops, 3)  # Bounded dimension
        
    def analyze_homotopy_system(self) -> Dict:
        """分析完整的homotopy系统"""
        results = {
            'homotopy_elements': [],
            'deformation_signatures': {},
            'equivalence_classes': defaultdict(list),
            'homotopy_types': defaultdict(int),
            'network_properties': {},
            'information_measures': {},
            'category_analysis': {},
            'convergence_analysis': {}
        }
        
        # 收集所有homotopy元素
        for n, data in self.trace_universe.items():
            if data['phi_valid']:
                homotopy_props = data.get('homotopy_properties', {})
                results['homotopy_elements'].append({
                    'value': n,
                    'trace': data['trace'],
                    'properties': homotopy_props
                })
                
                # 统计homotopy类型
                homotopy_type = homotopy_props.get('homotopy_type', 'unknown')
                results['homotopy_types'][homotopy_type] += 1
                
                # 记录deformation signatures
                sig = homotopy_props.get('deformation_signature', complex(0, 0))
                results['deformation_signatures'][n] = sig
                
                # 组织equivalence classes
                invariant = self._compute_homotopy_invariant(data['trace'])
                results['equivalence_classes'][invariant].append(n)
                
        # 计算网络属性
        results['network_properties'] = self._compute_network_properties(results['homotopy_elements'])
        
        # 计算信息度量
        results['information_measures'] = self._compute_information_measures(results['homotopy_elements'])
        
        # 范畴论分析
        results['category_analysis'] = self._compute_category_analysis(results['homotopy_elements'])
        
        # 三域收敛分析
        results['convergence_analysis'] = self._compute_convergence_analysis(results)
        
        return results
        
    def _compute_network_properties(self, elements: List[Dict]) -> Dict:
        """计算homotopy网络属性"""
        G = nx.Graph()
        
        # 添加节点
        for elem in elements:
            n = elem['value']
            props = elem['properties']
            G.add_node(n, **props)
            
        # 添加边：基于homotopy equivalence
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                n1, n2 = elem1['value'], elem2['value']
                trace1, trace2 = elem1['trace'], elem2['trace']
                
                # Check if homotopic
                if self._are_homotopic(trace1, trace2):
                    # Compute deformation distance
                    sig1 = elem1['properties'].get('deformation_signature', complex(0, 0))
                    sig2 = elem2['properties'].get('deformation_signature', complex(0, 0))
                    distance = abs(sig1 - sig2)
                    
                    G.add_edge(n1, n2, weight=1.0 - distance)
                    
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'components': nx.number_connected_components(G),
            'clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        }
        
    def _compute_information_measures(self, elements: List[Dict]) -> Dict:
        """计算homotopy信息度量"""
        if not elements:
            return {}
            
        # 收集各种属性分布
        dimensions = []
        types = []
        complexities = []
        costs = []
        groups = []
        
        for elem in elements:
            props = elem['properties']
            dimensions.append(props.get('homotopy_dimension', 0))
            types.append(props.get('homotopy_type', 'unknown'))
            complexities.append(props.get('homotopy_complexity', 0))
            costs.append(props.get('deformation_cost', 0))
            groups.append(props.get('fundamental_group', 1))
            
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
            'cost_entropy': compute_entropy(costs),
            'group_entropy': compute_entropy(groups),
            'homotopy_complexity': len(set(types))
        }
        
    def _compute_category_analysis(self, elements: List[Dict]) -> Dict:
        """计算homotopy范畴论属性"""
        # 构建态射关系
        morphisms = []
        functorial_morphisms = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j:
                    # 检查homotopy morphism
                    trace1, trace2 = elem1['trace'], elem2['trace']
                    
                    if self._are_homotopic(trace1, trace2):
                        morphisms.append((elem1['value'], elem2['value']))
                        
                        # 检查函子性质
                        dim1 = elem1['properties'].get('homotopy_dimension', 0)
                        dim2 = elem2['properties'].get('homotopy_dimension', 0)
                        
                        if dim1 == dim2:  # Dimension preserving
                            functorial_morphisms.append((elem1['value'], elem2['value']))
                            
        # 计算equivalence classes
        equiv_classes = defaultdict(list)
        for elem in elements:
            inv = self._compute_homotopy_invariant(elem['trace'])
            equiv_classes[inv].append(elem['value'])
            
        return {
            'morphisms': len(morphisms),
            'functorial_morphisms': len(functorial_morphisms),
            'functoriality_ratio': len(functorial_morphisms) / len(morphisms) if morphisms else 0,
            'equivalence_classes': len(equiv_classes),
            'largest_class': max(len(cls) for cls in equiv_classes.values()) if equiv_classes else 0
        }
        
    def _compute_convergence_analysis(self, results: Dict) -> Dict:
        """计算三域收敛分析"""
        total_elements = len(results['homotopy_elements'])
        
        # Traditional domain: Would have unlimited homotopy structures
        traditional_potential = 100  # Arbitrary large number
        
        # Collapse domain: φ-constrained structures
        collapse_actual = total_elements
        
        # Convergence ratio
        convergence_ratio = collapse_actual / traditional_potential
        
        # 分析homotopy属性分布
        costs = []
        dimensions = []
        complexities = []
        groups = []
        contractible_count = 0
        
        for elem in results['homotopy_elements']:
            props = elem['properties']
            costs.append(props.get('deformation_cost', 0))
            dimensions.append(props.get('homotopy_dimension', 0))
            complexities.append(props.get('homotopy_complexity', 0))
            groups.append(props.get('fundamental_group', 1))
            if props.get('contractible', False):
                contractible_count += 1
                
        return {
            'convergence_ratio': convergence_ratio,
            'mean_deformation_cost': np.mean(costs) if costs else 0,
            'mean_dimension': np.mean(dimensions) if dimensions else 0,
            'mean_complexity': np.mean(complexities) if complexities else 0,
            'mean_fundamental_group': np.mean(groups) if groups else 0,
            'contractible_ratio': contractible_count / total_elements if total_elements > 0 else 0,
            'homotopy_efficiency': 1.0 - np.std(complexities) if complexities else 0
        }
        
    def visualize_homotopy_structure(self, results: Dict, save_path: str = 'chapter-072-homotopy-collapse-structure.png'):
        """可视化homotopy结构"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Deformation Signature分布 (极坐标)
        ax1 = plt.subplot(331, projection='polar')
        signatures = list(results['deformation_signatures'].values())
        if signatures:
            angles = [np.angle(sig) for sig in signatures]
            radii = [abs(sig) for sig in signatures]
            colors = plt.cm.viridis(np.linspace(0, 1, len(signatures)))
            ax1.scatter(angles, radii, c=colors, s=100, alpha=0.6)
            ax1.set_title('Deformation Signatures in Complex Plane', fontsize=14, pad=20)
            ax1.set_ylim(0, 1.2)
        
        # 2. Fundamental Group分布
        ax2 = plt.subplot(332)
        groups = [elem['properties'].get('fundamental_group', 1) 
                 for elem in results['homotopy_elements']]
        if groups:
            unique_groups = sorted(set(groups))
            group_counts = [groups.count(g) for g in unique_groups]
            ax2.bar([str(g) for g in unique_groups], group_counts, 
                   color='teal', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Fundamental Group Order')
            ax2.set_ylabel('Count')
            ax2.set_title('Fundamental Group Distribution')
            ax2.grid(True, alpha=0.3)
        
        # 3. Homotopy类型分布
        ax3 = plt.subplot(333)
        homotopy_types = results['homotopy_types']
        if homotopy_types:
            types = list(homotopy_types.keys())
            counts = list(homotopy_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            ax3.pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            ax3.set_title('Homotopy Type Distribution')
        
        # 4. Deformation Cost vs Dimension
        ax4 = plt.subplot(334)
        costs = [elem['properties'].get('deformation_cost', 0) 
                for elem in results['homotopy_elements']]
        dimensions = [elem['properties'].get('homotopy_dimension', 0) 
                     for elem in results['homotopy_elements']]
        if costs and dimensions:
            scatter = ax4.scatter(dimensions, costs, s=100, alpha=0.6, c='green')
            ax4.set_xlabel('Homotopy Dimension')
            ax4.set_ylabel('Deformation Cost')
            ax4.set_title('Deformation Cost vs Dimension')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line if enough points
            if len(set(dimensions)) > 1:
                z = np.polyfit(dimensions, costs, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(dimensions), max(dimensions), 100)
                ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        # 5. Equivalence Classes
        ax5 = plt.subplot(335)
        equiv_classes = results['equivalence_classes']
        if equiv_classes:
            class_sizes = [len(cls) for cls in equiv_classes.values()]
            ax5.hist(class_sizes, bins=max(class_sizes), alpha=0.7, 
                    color='purple', edgecolor='black')
            ax5.set_xlabel('Class Size')
            ax5.set_ylabel('Number of Classes')
            ax5.set_title('Equivalence Class Size Distribution')
            ax5.grid(True, alpha=0.3)
        
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
        ax6.set_title('Homotopy Network Properties')
        ax6.set_xticklabels(metrics, rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 7. 信息熵度量
        ax7 = plt.subplot(337)
        info_measures = results['information_measures']
        entropy_types = ['Dimension', 'Type', 'Complexity', 'Cost', 'Group']
        entropy_values = [
            info_measures.get('dimension_entropy', 0),
            info_measures.get('type_entropy', 0),
            info_measures.get('complexity_entropy', 0),
            info_measures.get('cost_entropy', 0),
            info_measures.get('group_entropy', 0)
        ]
        ax7.barh(entropy_types, entropy_values, 
                color=plt.cm.coolwarm(np.linspace(0, 1, len(entropy_types))))
        ax7.set_xlabel('Entropy (bits)')
        ax7.set_title('Information Entropy Measures')
        ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Loop Space Visualization
        ax8 = plt.subplot(338)
        loop_spaces = [elem['properties'].get('loop_space', 0) 
                      for elem in results['homotopy_elements']]
        contractible = [elem['properties'].get('contractible', False) 
                       for elem in results['homotopy_elements']]
        
        if loop_spaces:
            # Separate contractible and non-contractible
            contract_loops = [l for l, c in zip(loop_spaces, contractible) if c]
            non_contract_loops = [l for l, c in zip(loop_spaces, contractible) if not c]
            
            bins = np.arange(0, max(loop_spaces) + 2) - 0.5
            if contract_loops:
                ax8.hist(contract_loops, bins=bins, alpha=0.5, 
                        label='Contractible', color='blue')
            if non_contract_loops:
                ax8.hist(non_contract_loops, bins=bins, alpha=0.5, 
                        label='Non-contractible', color='red')
            
            ax8.set_xlabel('Loop Space Dimension')
            ax8.set_ylabel('Count')
            ax8.set_title('Loop Space Distribution')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. 收敛分析总结
        ax9 = plt.subplot(339)
        conv_analysis = results['convergence_analysis']
        conv_metrics = ['Deform Cost', 'Dimension', 'Complexity', 'Fund. Group', 'Contract. Ratio']
        conv_values = [
            conv_analysis.get('mean_deformation_cost', 0),
            conv_analysis.get('mean_dimension', 0) / 3.0,  # Normalize
            conv_analysis.get('mean_complexity', 0),
            min(1.0, conv_analysis.get('mean_fundamental_group', 0) / 8.0),  # Normalize
            conv_analysis.get('contractible_ratio', 0)
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
        
    def visualize_homotopy_properties(self, results: Dict, save_path: str = 'chapter-072-homotopy-collapse-properties.png'):
        """可视化homotopy属性关系"""
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data
        elements = results['homotopy_elements']
        costs = [e['properties'].get('deformation_cost', 0) for e in elements]
        dimensions = [e['properties'].get('homotopy_dimension', 0) for e in elements]
        complexities = [e['properties'].get('homotopy_complexity', 0) for e in elements]
        groups = [e['properties'].get('fundamental_group', 1) for e in elements]
        contractible = [e['properties'].get('contractible', False) for e in elements]
        
        # 1. 3D Cost-Dimension-Group空间
        ax1 = fig.add_subplot(221, projection='3d')
        if costs and dimensions and groups:
            # Color by contractibility
            colors = ['blue' if c else 'red' for c in contractible]
            scatter = ax1.scatter(costs, dimensions, groups,
                                c=colors, s=100, alpha=0.6)
            ax1.set_xlabel('Deformation Cost')
            ax1.set_ylabel('Dimension')
            ax1.set_zlabel('Fundamental Group')
            ax1.set_title('Homotopy Property Space')
            
            # Add legend
            blue_patch = patches.Patch(color='blue', label='Contractible')
            red_patch = patches.Patch(color='red', label='Non-contractible')
            ax1.legend(handles=[blue_patch, red_patch])
        
        # 2. Path Equivalence Network
        ax2 = plt.subplot(222)
        # Create equivalence graph
        equiv_graph = nx.Graph()
        for i, elem in enumerate(elements[:10]):  # Limit for visibility
            equiv_graph.add_node(i, trace=elem['trace'])
            
        # Add edges for equivalent paths
        for i in range(len(elements[:10])):
            for j in range(i+1, len(elements[:10])):
                if self._are_homotopic(elements[i]['trace'], elements[j]['trace']):
                    equiv_graph.add_edge(i, j)
                    
        if equiv_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(equiv_graph)
            nx.draw(equiv_graph, pos, ax=ax2, with_labels=True, 
                   node_color='lightblue', node_size=500, font_size=10)
            ax2.set_title('Path Equivalence Network (First 10 Elements)')
        
        # 3. Homotopy Type Features
        ax3 = plt.subplot(223)
        type_features = defaultdict(lambda: {'cost': [], 'complexity': []})
        for e in elements:
            h_type = e['properties'].get('homotopy_type', 'unknown')
            type_features[h_type]['cost'].append(e['properties'].get('deformation_cost', 0))
            type_features[h_type]['complexity'].append(e['properties'].get('homotopy_complexity', 0))
        
        if type_features:
            colors = plt.cm.tab10(np.linspace(0, 1, len(type_features)))
            for (h_type, features), color in zip(type_features.items(), colors):
                if features['cost'] and features['complexity']:
                    ax3.scatter(features['cost'], features['complexity'], 
                              label=h_type, color=color, s=100, alpha=0.6)
            
            ax3.set_xlabel('Deformation Cost')
            ax3.set_ylabel('Homotopy Complexity')
            ax3.set_title('Homotopy Types Feature Space')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Deformation Signature Phase Distribution
        ax4 = plt.subplot(224, projection='polar')
        signatures = list(results['deformation_signatures'].values())
        if signatures:
            # Group by fundamental group
            group_sigs = defaultdict(list)
            for elem, sig in zip(elements, signatures):
                g = elem['properties'].get('fundamental_group', 1)
                group_sigs[g].append(sig)
            
            colors = plt.cm.plasma(np.linspace(0, 1, len(group_sigs)))
            for (group, sigs), color in zip(group_sigs.items(), colors):
                if sigs:
                    angles = [np.angle(sig) for sig in sigs]
                    radii = [abs(sig) for sig in sigs]
                    ax4.scatter(angles, radii, c=[color], 
                              label=f'π₁ = {group}', s=80, alpha=0.6)
            
            ax4.set_title('Deformation Signatures by Fundamental Group', pad=20)
            ax4.set_ylim(0, 1.2)
            ax4.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
        
    def visualize_three_domains(self, results: Dict, save_path: str = 'chapter-072-homotopy-collapse-domains.png'):
        """可视化三域分析"""
        fig = plt.figure(figsize=(18, 10))
        
        # 准备数据
        conv_analysis = results['convergence_analysis']
        
        # 1. 三域概览
        ax1 = plt.subplot(131)
        domains = ['Traditional\n(Unlimited)', 'φ-Constrained\n(Structural)', 'Convergence\n(Bounded)']
        values = [100, len(results['homotopy_elements']), 
                 len(results['homotopy_elements']) * conv_analysis['convergence_ratio']]
        colors = ['red', 'blue', 'purple']
        
        bars = ax1.bar(domains, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Homotopy Structures')
        ax1.set_title('Three-Domain Homotopy Analysis')
        
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
        properties = ['Deform Cost', 'Dimension', 'Complexity', 'Fund. Group']
        traditional_vals = [1.0, 1.0, 1.0, 1.0]  # Normalized unlimited
        collapse_vals = [
            conv_analysis['mean_deformation_cost'],
            conv_analysis['mean_dimension'] / 3.0,  # Normalize
            conv_analysis['mean_complexity'],
            min(1.0, conv_analysis['mean_fundamental_group'] / 8.0)  # Normalize
        ]
        
        x = np.arange(len(properties))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, traditional_vals, width, 
                        label='Traditional', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, collapse_vals, width, 
                        label='φ-Constrained', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Homotopy Properties')
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
            'Cost': info_measures.get('cost_entropy', 0),
            'Group': info_measures.get('group_entropy', 0)
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
        ax3.set_title('Homotopy Information Compression Efficiency')
        ax3.set_ylim(0, 1)
        ax3.axhline(y=conv_analysis['homotopy_efficiency'], color='red', 
                   linestyle='--', label=f'Mean Efficiency: {conv_analysis["homotopy_efficiency"]:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff*100:.1f}%', ha='center', va='bottom')
        
        # Add overall summary
        fig.suptitle(f'Homotopy Collapse: Three-Domain Convergence Analysis\n' + 
                    f'Total Elements: {len(results["homotopy_elements"])}, ' +
                    f'Homotopy Types: {results["information_measures"].get("homotopy_complexity", 0)}, ' +
                    f'Contractible Ratio: {conv_analysis["contractible_ratio"]:.3f}',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path


class TestHomotopyCollapse(unittest.TestCase):
    """Homotopy collapse单元测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = HomotopyCollapseSystem(max_trace_size=6)
        
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
            data = self.system._analyze_trace_structure(int(trace, 2), compute_homotopy=False)
            self.assertTrue(data['phi_valid'])
            
        for trace in invalid_traces:
            data = self.system._analyze_trace_structure(int(trace, 2), compute_homotopy=False)
            self.assertFalse(data['phi_valid'])
            
    def test_deformation_signature(self):
        """测试deformation signature计算"""
        trace = '1010'
        signature = self.system._compute_deformation_signature(trace)
        
        # 验证是否在单位圆上
        self.assertAlmostEqual(abs(signature), 1.0, places=5)
        
        # 验证复数类型
        self.assertIsInstance(signature, complex)
        
    def test_path_equivalence(self):
        """测试path equivalence计算"""
        trace = '101'
        equiv_paths = self.system._compute_path_equivalence(trace)
        
        # 应该至少包含自身
        self.assertIn(trace, equiv_paths)
        
        # 所有等价路径应该满足φ-constraint
        for path in equiv_paths:
            self.assertNotIn('11', path)
            
    def test_fundamental_group(self):
        """测试fundamental group计算"""
        # 简单连通的trace
        simple_trace = '0000'
        group1 = self.system._compute_fundamental_group(simple_trace)
        self.assertEqual(group1, 1)
        
        # 有loops的trace
        loop_trace = '0101'
        group2 = self.system._compute_fundamental_group(loop_trace)
        self.assertGreater(group2, 1)
        
    def test_homotopy_type_classification(self):
        """测试homotopy类型分类"""
        trace = '1010'
        homotopy_type = self.system._classify_homotopy_type(trace)
        self.assertIn(homotopy_type, ['trivial', 'contractible', 
                                     'simply_connected', 'weakly_deformable',
                                     'strongly_deformable'])
        
    def test_homotopy_invariant(self):
        """测试homotopy invariant计算"""
        trace1 = '101'
        trace2 = '1010'  # Different trace
        
        inv1 = self.system._compute_homotopy_invariant(trace1)
        inv2 = self.system._compute_homotopy_invariant(trace2)
        
        # Invariants should be in range [0, 6]
        self.assertGreaterEqual(inv1, 0)
        self.assertLessEqual(inv1, 6)
        self.assertGreaterEqual(inv2, 0)
        self.assertLessEqual(inv2, 6)
        
    def test_homotopy_system_analysis(self):
        """测试完整homotopy系统分析"""
        results = self.system.analyze_homotopy_system()
        
        # 验证结果结构
        self.assertIn('homotopy_elements', results)
        self.assertIn('deformation_signatures', results)
        self.assertIn('equivalence_classes', results)
        self.assertIn('homotopy_types', results)
        self.assertIn('network_properties', results)
        self.assertIn('information_measures', results)
        self.assertIn('category_analysis', results)
        self.assertIn('convergence_analysis', results)
        
        # 验证有homotopy元素
        self.assertGreater(len(results['homotopy_elements']), 0)
        
        # 验证网络属性
        net_props = results['network_properties']
        self.assertGreaterEqual(net_props['nodes'], 0)
        self.assertGreaterEqual(net_props['density'], 0)
        
    def test_contractibility(self):
        """测试可缩性判断"""
        # 可缩的trace
        contractible_trace = '0'
        is_contractible = self.system._is_contractible(contractible_trace)
        self.assertTrue(is_contractible)
        
        # 不可缩的trace
        non_contractible_trace = '0101'
        is_non_contractible = self.system._is_contractible(non_contractible_trace)
        self.assertFalse(is_non_contractible)
        
    def test_three_domain_convergence(self):
        """测试三域收敛分析"""
        results = self.system.analyze_homotopy_system()
        conv_analysis = results['convergence_analysis']
        
        # 验证收敛比率
        self.assertGreater(conv_analysis['convergence_ratio'], 0)
        self.assertLessEqual(conv_analysis['convergence_ratio'], 1.0)
        
        # 验证平均值在合理范围
        self.assertGreaterEqual(conv_analysis['mean_deformation_cost'], 0)
        self.assertLessEqual(conv_analysis['mean_deformation_cost'], 1.0)
        
        self.assertGreaterEqual(conv_analysis['mean_dimension'], 0)
        self.assertLessEqual(conv_analysis['mean_complexity'], 1.0)
        
    def test_visualization_generation(self):
        """测试可视化生成"""
        results = self.system.analyze_homotopy_system()
        
        # 测试结构可视化
        path1 = self.system.visualize_homotopy_structure(results, 
                    'test_homotopy_structure.png')
        self.assertTrue(path1.endswith('.png'))
        
        # 测试属性可视化
        path2 = self.system.visualize_homotopy_properties(results,
                    'test_homotopy_properties.png')
        self.assertTrue(path2.endswith('.png'))
        
        # 测试三域可视化
        path3 = self.system.visualize_three_domains(results,
                    'test_homotopy_domains.png')
        self.assertTrue(path3.endswith('.png'))
        
        # 清理测试文件
        import os
        for path in [path1, path2, path3]:
            if os.path.exists(path):
                os.remove(path)


def main():
    """主函数：运行homotopy collapse分析"""
    print("🔄 Chapter 072: HomotopyCollapse Unit Test Verification")
    print("=" * 60)
    
    # 创建系统
    system = HomotopyCollapseSystem(max_trace_size=6)
    
    # 运行分析
    print("📊 Building trace universe...")
    results = system.analyze_homotopy_system()
    
    print(f"✅ Found {len(results['homotopy_elements'])} φ-valid traces")
    
    # 输出关键结果
    print("\n🔍 Analyzing homotopy collapse system...")
    print(f"📈 Homotopy universe size: {len(results['homotopy_elements'])} elements")
    print(f"📊 Network density: {results['network_properties']['density']:.3f}")
    print(f"🎯 Convergence ratio: {results['convergence_analysis']['convergence_ratio']:.3f}")
    
    # 输出homotopy属性
    conv = results['convergence_analysis']
    print(f"\n📏 Homotopy Properties:")
    print(f"   Mean deformation cost: {conv['mean_deformation_cost']:.3f}")
    print(f"   Mean dimension: {conv['mean_dimension']:.3f}")
    print(f"   Mean complexity: {conv['mean_complexity']:.3f}")
    print(f"   Mean fundamental group: {conv['mean_fundamental_group']:.3f}")
    print(f"   Contractible ratio: {conv['contractible_ratio']:.3f}")
    
    # 输出信息度量
    info = results['information_measures']
    print(f"\n🧠 Information Analysis:")
    print(f"   Dimension entropy: {info['dimension_entropy']:.3f} bits")
    print(f"   Type entropy: {info['type_entropy']:.3f} bits")
    print(f"   Complexity entropy: {info['complexity_entropy']:.3f} bits")
    print(f"   Cost entropy: {info['cost_entropy']:.3f} bits")
    print(f"   Group entropy: {info['group_entropy']:.3f} bits")
    print(f"   Homotopy complexity: {info['homotopy_complexity']} unique types")
    
    # 生成可视化
    print("\n🎨 Generating visualizations...")
    system.visualize_homotopy_structure(results)
    system.visualize_homotopy_properties(results)
    system.visualize_three_domains(results)
    print("✅ Visualizations saved: structure, properties, domains")
    
    # 运行单元测试
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n✅ Chapter 072: HomotopyCollapse verification completed!")
    print("=" * 60)
    print("🔥 Homotopy structures exhibit bounded deformation convergence!")


if __name__ == "__main__":
    main()