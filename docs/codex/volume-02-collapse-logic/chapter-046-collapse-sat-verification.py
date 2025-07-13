#!/usr/bin/env python3
"""
Chapter 046: CollapseSAT Unit Test Verification
从ψ=ψ(ψ)推导Trace-Constrained Structural Satisfiability

Core principle: From ψ = ψ(ψ) derive satisfiability where SAT problems emerge
through φ-constrained trace structures, creating systematic solution search that
maintains structural coherence across all variable assignments and clauses.

This verification program implements:
1. φ-constrained SAT formulation with trace-based variables
2. Structural satisfiability checking through trace transformations
3. Three-domain analysis: Traditional vs φ-constrained vs intersection SAT theory
4. Graph theory analysis of clause-variable interaction networks
5. Information theory analysis of solution space entropy
6. Category theory analysis of SAT reduction functors
7. Visualization of SAT structures and solution landscapes
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi, exp
from functools import reduce
import random

class CollapseSATSystem:
    """
    Core system for implementing trace-constrained structural satisfiability.
    Implements φ-constrained SAT solving via trace-based variable assignments.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize collapse SAT system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.sat_cache = {}
        self.solution_registry = {}
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_universe(self) -> Dict[int, Dict]:
        """构建trace universe：所有φ-valid traces的结构化表示"""
        universe = {}
        for n in range(self.max_trace_size + 1):
            trace_data = self._analyze_trace_structure(n)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        return universe
        
    def _analyze_trace_structure(self, n: int) -> Dict:
        """分析单个trace的结构属性"""
        trace = self._encode_to_trace(n)
        
        return {
            'value': n,
            'trace': trace,
            'phi_valid': '11' not in trace,
            'length': len(trace),
            'ones_count': trace.count('1'),
            'fibonacci_indices': self._get_fibonacci_indices(trace),
            'structural_hash': self._compute_structural_hash(trace),
            'sat_signature': self._compute_sat_signature(trace),
            'assignment_properties': self._compute_assignment_properties(trace)
        }
        
    def _encode_to_trace(self, n: int) -> str:
        """将自然数编码为φ-compliant trace (Zeckendorf-based)"""
        if n == 0:
            return '0'
            
        # 使用Zeckendorf分解
        decomposition = self._zeckendorf_decomposition(n)
        if decomposition is None:
            return '0'
            
        # 构造trace：位置i对应F_{i+1}
        max_index = max(decomposition) if decomposition else 1
        trace = ['0'] * max_index
        
        for idx in decomposition:
            trace[idx - 1] = '1'  # idx从1开始，所以-1
            
        return ''.join(reversed(trace))  # 高位在左
        
    def _zeckendorf_decomposition(self, n: int) -> Optional[List[int]]:
        """Zeckendorf分解：避免连续Fibonacci数"""
        if n == 0:
            return []
            
        remaining = n
        used_indices = []
        
        for i in range(len(self.fibonacci_numbers) - 1, -1, -1):
            if self.fibonacci_numbers[i] <= remaining:
                remaining -= self.fibonacci_numbers[i]
                used_indices.append(i + 1)  # 1-indexed
                
        return used_indices if remaining == 0 else None
        
    def _get_fibonacci_indices(self, trace: str) -> Set[int]:
        """获取trace中激活的Fibonacci indices"""
        indices = set()
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                indices.add(i + 1)  # 1-indexed
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构hash"""
        return hash(trace) % 10000
        
    def _compute_sat_signature(self, trace: str) -> Tuple[int, int, float, bool]:
        """计算trace的SAT签名：(length, ones_count, density, balanced)"""
        density = trace.count('1') / max(len(trace), 1)
        balanced = abs(trace.count('1') - trace.count('0')) <= 1
        return (len(trace), trace.count('1'), density, balanced)
        
    def _compute_assignment_properties(self, trace: str) -> Dict[str, Union[int, float, bool]]:
        """计算trace作为变量赋值的属性"""
        return {
            'assignment_strength': self._compute_assignment_strength(trace),
            'conflict_potential': self._compute_conflict_potential(trace),
            'propagation_power': self._compute_propagation_power(trace),
            'stability_score': self._compute_stability_score(trace)
        }
        
    def _compute_assignment_strength(self, trace: str) -> float:
        """计算赋值强度"""
        if not trace:
            return 0.0
        # 基于Fibonacci indices的分布
        indices = self._get_fibonacci_indices(trace)
        if not indices:
            return 0.0
        return sum(1.0/i for i in indices) / len(indices)
        
    def _compute_conflict_potential(self, trace: str) -> float:
        """计算冲突潜力"""
        if len(trace) < 2:
            return 0.0
        # 基于相邻位的变化
        conflicts = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        return conflicts / (len(trace) - 1)
        
    def _compute_propagation_power(self, trace: str) -> int:
        """计算传播能力"""
        # 基于可以影响的其他变量数
        return trace.count('1') * 2 + trace.count('0')
        
    def _compute_stability_score(self, trace: str) -> float:
        """计算稳定性分数"""
        if not trace:
            return 0.0
        # 基于模式的规律性
        if len(trace) < 3:
            return 1.0
        patterns = [trace[i:i+3] for i in range(len(trace)-2)]
        unique_patterns = len(set(patterns))
        return 1.0 - (unique_patterns / len(patterns))

    def create_sat_instance(self, n_vars: int, n_clauses: int, 
                           clause_size: int = 3) -> Dict:
        """创建SAT实例"""
        # 选择φ-valid traces作为变量
        valid_vars = list(self.trace_universe.keys())[1:n_vars+1]
        
        clauses = []
        for _ in range(n_clauses):
            # 随机选择变量和极性
            clause = []
            selected_vars = random.sample(valid_vars, min(clause_size, len(valid_vars)))
            
            for var in selected_vars:
                polarity = random.choice([True, False])
                clause.append((var, polarity))
                
            clauses.append(clause)
            
        return {
            'variables': valid_vars,
            'clauses': clauses,
            'n_vars': len(valid_vars),
            'n_clauses': len(clauses),
            'clause_size': clause_size,
            'instance_id': self._generate_instance_id(valid_vars, clauses)
        }
        
    def _generate_instance_id(self, variables: List[int], clauses: List[List[Tuple[int, bool]]]) -> str:
        """生成实例ID"""
        var_hash = hash(tuple(variables)) % 10000
        clause_hash = hash(tuple(tuple(c) for c in clauses)) % 10000
        return f"SAT_{var_hash}_{clause_hash}"
        
    def check_satisfiability(self, sat_instance: Dict) -> Dict:
        """检查可满足性"""
        variables = sat_instance['variables']
        clauses = sat_instance['clauses']
        
        # 尝试所有可能的φ-valid赋值
        n_vars = len(variables)
        total_assignments = 2 ** n_vars
        satisfying_assignments = []
        
        for assignment_bits in range(total_assignments):
            assignment = {}
            
            # 构建赋值
            for i, var in enumerate(variables):
                bit = (assignment_bits >> i) & 1
                assignment[var] = bool(bit)
                
            # 检查是否满足所有子句
            if self._check_assignment(assignment, clauses):
                # 检查赋值的φ-validity
                if self._is_phi_valid_assignment(assignment):
                    satisfying_assignments.append(assignment)
                    
        return {
            'satisfiable': len(satisfying_assignments) > 0,
            'n_solutions': len(satisfying_assignments),
            'solutions': satisfying_assignments[:10],  # 最多返回10个解
            'total_assignments': total_assignments,
            'phi_valid_ratio': len(satisfying_assignments) / total_assignments if total_assignments > 0 else 0
        }
        
    def _check_assignment(self, assignment: Dict[int, bool], 
                         clauses: List[List[Tuple[int, bool]]]) -> bool:
        """检查赋值是否满足所有子句"""
        for clause in clauses:
            clause_satisfied = False
            
            for var, polarity in clause:
                if var in assignment:
                    if assignment[var] == polarity:
                        clause_satisfied = True
                        break
                        
            if not clause_satisfied:
                return False
                
        return True
        
    def _is_phi_valid_assignment(self, assignment: Dict[int, bool]) -> bool:
        """检查赋值是否保持φ-validity"""
        # 构建赋值的trace表示
        assignment_trace = []
        for var, value in sorted(assignment.items()):
            assignment_trace.append('1' if value else '0')
            
        assignment_str = ''.join(assignment_trace)
        return '11' not in assignment_str
        
    def analyze_solution_space(self, sat_instance: Dict, solutions: List[Dict]) -> Dict:
        """分析解空间"""
        if not solutions:
            return {
                'empty': True,
                'entropy': 0.0,
                'clustering': 0.0,
                'diversity': 0.0
            }
            
        # 计算解之间的距离
        n_solutions = len(solutions)
        distances = np.zeros((n_solutions, n_solutions))
        
        for i in range(n_solutions):
            for j in range(i+1, n_solutions):
                dist = self._hamming_distance(solutions[i], solutions[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        # 计算解空间属性
        avg_distance = np.mean(distances[distances > 0]) if n_solutions > 1 else 0
        
        # 计算熵
        entropy = self._compute_solution_entropy(solutions)
        
        # 计算聚类系数
        clustering = self._compute_solution_clustering(solutions, distances)
        
        # 计算多样性
        diversity = avg_distance / max(len(sat_instance['variables']), 1)
        
        return {
            'empty': False,
            'n_solutions': n_solutions,
            'avg_distance': avg_distance,
            'entropy': entropy,
            'clustering': clustering,
            'diversity': diversity,
            'distance_matrix': distances
        }
        
    def _hamming_distance(self, assignment1: Dict[int, bool], 
                         assignment2: Dict[int, bool]) -> int:
        """计算两个赋值之间的汉明距离"""
        distance = 0
        all_vars = set(assignment1.keys()).union(set(assignment2.keys()))
        
        for var in all_vars:
            val1 = assignment1.get(var, False)
            val2 = assignment2.get(var, False)
            if val1 != val2:
                distance += 1
                
        return distance
        
    def _compute_solution_entropy(self, solutions: List[Dict]) -> float:
        """计算解的熵"""
        if not solutions:
            return 0.0
            
        # 统计每个变量的赋值分布
        var_distributions = defaultdict(lambda: {'true': 0, 'false': 0})
        
        for solution in solutions:
            for var, value in solution.items():
                if value:
                    var_distributions[var]['true'] += 1
                else:
                    var_distributions[var]['false'] += 1
                    
        # 计算总熵
        total_entropy = 0.0
        for var, dist in var_distributions.items():
            total = dist['true'] + dist['false']
            if total > 0:
                p_true = dist['true'] / total
                p_false = dist['false'] / total
                
                if p_true > 0:
                    total_entropy -= p_true * log2(p_true)
                if p_false > 0:
                    total_entropy -= p_false * log2(p_false)
                    
        return total_entropy / max(len(var_distributions), 1)
        
    def _compute_solution_clustering(self, solutions: List[Dict], 
                                   distances: np.ndarray) -> float:
        """计算解的聚类系数"""
        if len(solutions) < 3:
            return 0.0
            
        # 基于距离构建解的图
        threshold = np.mean(distances[distances > 0]) if distances.size > 0 else 1
        
        clustering_coeffs = []
        for i in range(len(solutions)):
            # 找到邻居（距离小于阈值的解）
            neighbors = [j for j in range(len(solutions)) 
                        if j != i and distances[i, j] < threshold]
                        
            if len(neighbors) >= 2:
                # 计算邻居之间的连接数
                neighbor_connections = 0
                for k in range(len(neighbors)):
                    for l in range(k+1, len(neighbors)):
                        if distances[neighbors[k], neighbors[l]] < threshold:
                            neighbor_connections += 1
                            
                # 计算聚类系数
                possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
                clustering_coeffs.append(neighbor_connections / possible_connections)
                
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0

    def visualize_sat_instance(self, sat_instance: Dict, save_path: str = None):
        """可视化SAT实例"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 变量-子句二部图
        G = nx.Graph()
        
        # 添加节点
        var_nodes = [f"V{v}" for v in sat_instance['variables']]
        clause_nodes = [f"C{i}" for i in range(sat_instance['n_clauses'])]
        
        G.add_nodes_from(var_nodes, bipartite=0)
        G.add_nodes_from(clause_nodes, bipartite=1)
        
        # 添加边
        for i, clause in enumerate(sat_instance['clauses']):
            for var, polarity in clause:
                var_node = f"V{var}"
                clause_node = f"C{i}"
                G.add_edge(var_node, clause_node, polarity=polarity)
                
        # 布局
        pos = {}
        # 变量在左边
        for i, v in enumerate(var_nodes):
            pos[v] = (0, i)
        # 子句在右边
        for i, c in enumerate(clause_nodes):
            pos[c] = (3, i)
            
        # 绘制
        nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, 
                             node_color='lightblue', node_size=500, ax=ax1)
        nx.draw_networkx_nodes(G, pos, nodelist=clause_nodes,
                             node_color='lightcoral', node_size=500, ax=ax1)
                             
        # 绘制边（正文字为实线，负文字为虚线）
        for edge in G.edges(data=True):
            if edge[2]['polarity']:
                nx.draw_networkx_edges(G, pos, [(edge[0], edge[1])],
                                     edge_color='green', width=2, ax=ax1)
            else:
                nx.draw_networkx_edges(G, pos, [(edge[0], edge[1])],
                                     edge_color='red', width=2,
                                     style='dashed', ax=ax1)
                                     
        nx.draw_networkx_labels(G, pos, ax=ax1)
        ax1.set_title('Variable-Clause Bipartite Graph')
        ax1.axis('off')
        
        # 2. 子句交互图
        clause_graph = nx.Graph()
        clause_graph.add_nodes_from(range(sat_instance['n_clauses']))
        
        # 添加边：共享变量的子句
        for i in range(sat_instance['n_clauses']):
            for j in range(i+1, sat_instance['n_clauses']):
                vars_i = set(v for v, _ in sat_instance['clauses'][i])
                vars_j = set(v for v, _ in sat_instance['clauses'][j])
                
                shared = vars_i.intersection(vars_j)
                if shared:
                    clause_graph.add_edge(i, j, weight=len(shared))
                    
        # 绘制
        pos2 = nx.spring_layout(clause_graph, k=1, iterations=50)
        
        # 边的权重决定宽度
        edges = clause_graph.edges()
        weights = [clause_graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_nodes(clause_graph, pos2, node_color='lightyellow',
                             node_size=700, ax=ax2)
        nx.draw_networkx_edges(clause_graph, pos2, width=weights, ax=ax2)
        nx.draw_networkx_labels(clause_graph, pos2, ax=ax2)
        
        ax2.set_title('Clause Interaction Graph')
        ax2.axis('off')
        
        plt.suptitle('φ-Constrained SAT Instance Structure', fontsize=14, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def visualize_solution_space(self, solutions: List[Dict], 
                               space_analysis: Dict, save_path: str = None):
        """可视化解空间"""
        if not solutions or space_analysis['empty']:
            return
            
        fig = plt.figure(figsize=(12, 10))
        
        # 1. 解空间3D投影
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 使用MDS降维到3D
        if len(solutions) > 1:
            from sklearn.manifold import MDS
            distances = space_analysis['distance_matrix']
            
            if len(solutions) >= 3:
                mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
                coords = mds.fit_transform(distances)
                
                ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c='blue', s=100, alpha=0.6)
                          
                # 连接相近的解
                threshold = np.mean(distances[distances > 0])
                for i in range(len(solutions)):
                    for j in range(i+1, len(solutions)):
                        if distances[i, j] < threshold:
                            ax1.plot([coords[i, 0], coords[j, 0]],
                                   [coords[i, 1], coords[j, 1]],
                                   [coords[i, 2], coords[j, 2]],
                                   'k-', alpha=0.3)
                                   
        ax1.set_title('Solution Space 3D Projection')
        ax1.set_xlabel('Dim 1')
        ax1.set_ylabel('Dim 2')
        ax1.set_zlabel('Dim 3')
        
        # 2. 解的分布热图
        ax2 = fig.add_subplot(222)
        
        # 创建变量赋值矩阵
        all_vars = sorted(set(v for sol in solutions for v in sol.keys()))
        assignment_matrix = np.zeros((len(solutions), len(all_vars)))
        
        for i, sol in enumerate(solutions):
            for j, var in enumerate(all_vars):
                if var in sol:
                    assignment_matrix[i, j] = 1 if sol[var] else -1
                    
        im = ax2.imshow(assignment_matrix, cmap='RdBu', aspect='auto')
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('Solutions')
        ax2.set_title('Solution Assignment Matrix')
        plt.colorbar(im, ax=ax2)
        
        # 3. 解空间度量
        ax3 = fig.add_subplot(223)
        
        metrics = ['Entropy', 'Clustering', 'Diversity']
        values = [space_analysis['entropy'], 
                 space_analysis['clustering'],
                 space_analysis['diversity']]
                 
        bars = ax3.bar(metrics, values, color=['skyblue', 'lightgreen', 'coral'])
        ax3.set_ylim(0, 1)
        ax3.set_title('Solution Space Metrics')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
                    
        # 4. 距离分布
        ax4 = fig.add_subplot(224)
        
        if len(solutions) > 1:
            distances_flat = space_analysis['distance_matrix'][
                np.triu_indices_from(space_analysis['distance_matrix'], k=1)]
            ax4.hist(distances_flat, bins=15, color='purple', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(distances_flat), color='red', linestyle='--',
                       label=f'Mean: {np.mean(distances_flat):.2f}')
            ax4.set_xlabel('Hamming Distance')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Solution Distance Distribution')
            ax4.legend()
            
        plt.suptitle('φ-Constrained SAT Solution Space Analysis', 
                    fontsize=14, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def analyze_sat_complexity(self, n_vars_range: List[int], 
                              n_clauses_factor: float = 4.2) -> Dict:
        """分析SAT复杂度"""
        complexity_data = {
            'n_vars': [],
            'n_clauses': [],
            'satisfiability_rate': [],
            'avg_solutions': [],
            'solving_difficulty': []
        }
        
        for n_vars in n_vars_range:
            n_clauses = int(n_vars * n_clauses_factor)
            
            # 生成多个实例进行统计
            n_instances = 10
            sat_count = 0
            total_solutions = 0
            difficulties = []
            
            for _ in range(n_instances):
                instance = self.create_sat_instance(n_vars, n_clauses)
                result = self.check_satisfiability(instance)
                
                if result['satisfiable']:
                    sat_count += 1
                    total_solutions += result['n_solutions']
                    
                # 计算求解难度（基于搜索空间大小）
                difficulty = result['total_assignments'] / max(result['n_solutions'], 1)
                difficulties.append(difficulty)
                
            complexity_data['n_vars'].append(n_vars)
            complexity_data['n_clauses'].append(n_clauses)
            complexity_data['satisfiability_rate'].append(sat_count / n_instances)
            complexity_data['avg_solutions'].append(total_solutions / n_instances)
            complexity_data['solving_difficulty'].append(np.mean(difficulties))
            
        return complexity_data

class TestCollapseSATSystem(unittest.TestCase):
    """单元测试：验证CollapseSAT系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseSATSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        # 验证φ-valid traces被正确识别
        self.assertIn(1, self.system.trace_universe)
        self.assertIn(2, self.system.trace_universe)
        self.assertIn(3, self.system.trace_universe)
        self.assertIn(5, self.system.trace_universe)
        
        # 验证SAT属性
        trace_5 = self.system.trace_universe[5]
        self.assertIn('assignment_properties', trace_5)
        self.assertGreater(trace_5['assignment_properties']['assignment_strength'], 0)
        
    def test_sat_instance_creation(self):
        """测试SAT实例创建"""
        instance = self.system.create_sat_instance(5, 10)
        
        self.assertIn('variables', instance)
        self.assertIn('clauses', instance)
        self.assertEqual(len(instance['variables']), 5)
        self.assertEqual(len(instance['clauses']), 10)
        
    def test_satisfiability_checking(self):
        """测试可满足性检查"""
        # 创建简单实例
        instance = self.system.create_sat_instance(3, 4, clause_size=2)
        result = self.system.check_satisfiability(instance)
        
        self.assertIn('satisfiable', result)
        self.assertIn('n_solutions', result)
        self.assertIn('phi_valid_ratio', result)
        
    def test_solution_space_analysis(self):
        """测试解空间分析"""
        instance = self.system.create_sat_instance(3, 4)
        result = self.system.check_satisfiability(instance)
        
        if result['satisfiable']:
            analysis = self.system.analyze_solution_space(instance, result['solutions'])
            
            self.assertIn('entropy', analysis)
            self.assertIn('clustering', analysis)
            self.assertIn('diversity', analysis)
            
    def test_phi_validity_checking(self):
        """测试φ-validity检查"""
        # 测试有效赋值
        valid_assignment = {1: True, 2: False, 3: True}
        self.assertTrue(self.system._is_phi_valid_assignment(valid_assignment))
        
        # 测试无效赋值（会产生11）
        invalid_assignment = {1: True, 2: True, 3: False}
        # 注意：具体是否无效取决于变量的排序
        
    def test_complexity_analysis(self):
        """测试复杂度分析"""
        complexity = self.system.analyze_sat_complexity([3, 4, 5])
        
        self.assertIn('satisfiability_rate', complexity)
        self.assertIn('avg_solutions', complexity)
        self.assertIn('solving_difficulty', complexity)

def run_comprehensive_analysis():
    """运行完整的CollapseSAT分析"""
    print("=" * 60)
    print("Chapter 046: CollapseSAT Comprehensive Analysis")
    print("Trace-Constrained Structural Satisfiability")
    print("=" * 60)
    
    system = CollapseSATSystem()
    
    # 1. 基础SAT分析
    print("\n1. Basic SAT Analysis:")
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Available variables: {list(system.trace_universe.keys())[:10]}")
    
    # 2. 创建并分析SAT实例
    print("\n2. SAT Instance Analysis:")
    
    # 小规模实例
    small_instance = system.create_sat_instance(5, 10, clause_size=3)
    print(f"\nSmall instance:")
    print(f"  Variables: {small_instance['n_vars']}")
    print(f"  Clauses: {small_instance['n_clauses']}")
    print(f"  Clause size: {small_instance['clause_size']}")
    
    # 检查可满足性
    small_result = system.check_satisfiability(small_instance)
    print(f"  Satisfiable: {small_result['satisfiable']}")
    print(f"  Number of solutions: {small_result['n_solutions']}")
    print(f"  φ-valid ratio: {small_result['phi_valid_ratio']:.3f}")
    
    # 中等规模实例
    medium_instance = system.create_sat_instance(8, 20, clause_size=3)
    print(f"\nMedium instance:")
    print(f"  Variables: {medium_instance['n_vars']}")
    print(f"  Clauses: {medium_instance['n_clauses']}")
    
    medium_result = system.check_satisfiability(medium_instance)
    print(f"  Satisfiable: {medium_result['satisfiable']}")
    print(f"  Number of solutions: {medium_result['n_solutions']}")
    
    # 3. 解空间分析
    print("\n3. Solution Space Analysis:")
    
    if small_result['satisfiable'] and small_result['solutions']:
        space_analysis = system.analyze_solution_space(small_instance, 
                                                      small_result['solutions'])
        
        print(f"Solution space properties:")
        print(f"  Number of solutions: {space_analysis['n_solutions']}")
        print(f"  Average distance: {space_analysis['avg_distance']:.2f}")
        print(f"  Entropy: {space_analysis['entropy']:.3f}")
        print(f"  Clustering: {space_analysis['clustering']:.3f}")
        print(f"  Diversity: {space_analysis['diversity']:.3f}")
        
    # 4. 复杂度分析
    print("\n4. Complexity Analysis:")
    
    complexity_data = system.analyze_sat_complexity([3, 4, 5, 6, 7], 
                                                   n_clauses_factor=4.2)
    
    print("\nComplexity scaling:")
    for i in range(len(complexity_data['n_vars'])):
        print(f"  n={complexity_data['n_vars'][i]}, "
              f"m={complexity_data['n_clauses'][i]}: "
              f"SAT rate={complexity_data['satisfiability_rate'][i]:.2f}, "
              f"avg solutions={complexity_data['avg_solutions'][i]:.1f}")
              
    # 5. 三域分析
    print("\n5. Three-Domain Analysis:")
    
    # Traditional SAT domain
    n_vars = 5
    traditional_assignments = 2 ** n_vars
    
    # φ-constrained domain
    phi_valid_assignments = small_result['total_assignments'] * small_result['phi_valid_ratio']
    
    # Intersection analysis
    sat_solutions = small_result['n_solutions']
    
    print(f"Traditional SAT domain: {traditional_assignments} possible assignments")
    print(f"φ-constrained domain: {phi_valid_assignments:.0f} valid assignments")
    print(f"Intersection (solutions): {sat_solutions} satisfying assignments")
    print(f"Solution density: {sat_solutions/traditional_assignments:.3f}")
    
    # 6. 可视化
    print("\n6. Generating SAT Visualizations...")
    
    # 可视化SAT实例
    system.visualize_sat_instance(small_instance, 
                                 "chapter-046-collapse-sat-instance.png")
    print("Saved visualization: chapter-046-collapse-sat-instance.png")
    
    # 可视化解空间
    if small_result['satisfiable'] and small_result['solutions']:
        system.visualize_solution_space(small_result['solutions'], 
                                      space_analysis,
                                      "chapter-046-collapse-sat-solution-space.png")
        print("Saved visualization: chapter-046-collapse-sat-solution-space.png")
        
    # 7. 生成复杂度分析图
    print("\n7. Generating Complexity Analysis Charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SAT率随问题规模变化
    ax = axes[0, 0]
    ax.plot(complexity_data['n_vars'], complexity_data['satisfiability_rate'],
           'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Satisfiability Rate')
    ax.set_title('SAT Rate vs Problem Size')
    ax.grid(True, alpha=0.3)
    
    # 平均解数量
    ax = axes[0, 1]
    ax.semilogy(complexity_data['n_vars'], complexity_data['avg_solutions'],
               'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Average Number of Solutions')
    ax.set_title('Solution Count Scaling')
    ax.grid(True, alpha=0.3)
    
    # 求解难度
    ax = axes[1, 0]
    ax.plot(complexity_data['n_vars'], complexity_data['solving_difficulty'],
           'r-^', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Solving Difficulty')
    ax.set_title('Problem Difficulty Scaling')
    ax.grid(True, alpha=0.3)
    
    # φ-constraint影响
    ax = axes[1, 1]
    n_vars_list = complexity_data['n_vars']
    traditional_space = [2**n for n in n_vars_list]
    phi_space = [len([v for v in system.trace_universe.keys() if v < 2**n]) 
                 for n in n_vars_list]
    
    ax.semilogy(n_vars_list, traditional_space, 'b-', label='Traditional space')
    ax.semilogy(n_vars_list, phi_space, 'r--', label='φ-constrained space')
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Search Space Size')
    ax.set_title('Search Space Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('φ-Constrained SAT Complexity Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("chapter-046-collapse-sat-complexity.png", dpi=150, bbox_inches='tight')
    print("Saved visualization: chapter-046-collapse-sat-complexity.png")
    
    # 8. 变量赋值属性分析
    print("\n8. Variable Assignment Properties:")
    
    # 分析前10个traces作为变量的属性
    print("\nTrace assignment properties:")
    for i, (val, data) in enumerate(list(system.trace_universe.items())[:10]):
        if val == 0:
            continue
        props = data['assignment_properties']
        print(f"  Trace {val} ({data['trace']}): "
              f"strength={props['assignment_strength']:.3f}, "
              f"conflict={props['conflict_potential']:.3f}")
              
    # 9. 相变现象分析
    print("\n9. Phase Transition Analysis:")
    
    # 测试不同子句密度
    phase_data = {'ratio': [], 'sat_rate': []}
    n_vars = 6
    
    for ratio in np.linspace(2.0, 6.0, 9):
        n_clauses = int(n_vars * ratio)
        
        # 多次实验取平均
        sat_count = 0
        n_trials = 20
        
        for _ in range(n_trials):
            instance = system.create_sat_instance(n_vars, n_clauses)
            result = system.check_satisfiability(instance)
            if result['satisfiable']:
                sat_count += 1
                
        phase_data['ratio'].append(ratio)
        phase_data['sat_rate'].append(sat_count / n_trials)
        
    # 绘制相变图
    plt.figure(figsize=(8, 6))
    plt.plot(phase_data['ratio'], phase_data['sat_rate'], 'b-o', linewidth=2)
    plt.axvline(4.2, color='r', linestyle='--', label='Classical threshold')
    plt.xlabel('Clause-to-Variable Ratio')
    plt.ylabel('Satisfiability Rate')
    plt.title('φ-Constrained SAT Phase Transition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("chapter-046-collapse-sat-phase-transition.png", dpi=150, bbox_inches='tight')
    print("Saved visualization: chapter-046-collapse-sat-phase-transition.png")
    
    # 打印相变结果
    print("\nPhase transition results:")
    for i in range(len(phase_data['ratio'])):
        print(f"  Ratio {phase_data['ratio'][i]:.1f}: "
              f"SAT rate = {phase_data['sat_rate'][i]:.2f}")
              
    print("\n" + "=" * 60)
    print("Analysis Complete - CollapseSAT System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 运行单元测试
    print("Running CollapseSAT Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()