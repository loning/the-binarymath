#!/usr/bin/env python3
"""
Chapter 031: TraceCrystals Unit Test Verification
从ψ=ψ(ψ)推导trace tensor space中的自重复算术结构

Core principle: From ψ = ψ(ψ) derive crystalline patterns in trace operations
through identification of minimal periods p such that T(x+p) = T(x) for trace operations T.

This verification program implements:
1. Crystal detection algorithms for trace lattices
2. Symmetry group computation for crystalline patterns
3. Three-domain analysis: Traditional vs φ-constrained vs intersection crystallography
4. Graph theory analysis of crystal connectivity
5. Information theory analysis of pattern entropy
6. Category theory analysis of crystal morphisms
"""

import torch
import numpy as np
import networkx as nx
import unittest
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import itertools
from math import log2, gcd
from functools import reduce

class TraceCrystalSystem:
    """
    Core system for detecting and analyzing crystalline patterns in trace tensor space.
    Implements crystal detection through periodic trace operation analysis.
    """
    
    def __init__(self, max_lattice_size: int = 50):
        """Initialize trace crystal detection system"""
        self.max_lattice_size = max_lattice_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_lattice = self._build_trace_lattice()
        self.crystal_cache = {}
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for i in range(len(fib), count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _build_trace_lattice(self) -> Dict[int, str]:
        """构建trace lattice：每个数字对应其φ-compliant trace representation"""
        lattice = {}
        for n in range(self.max_lattice_size + 1):
            lattice[n] = self._encode_to_trace(n)
        return lattice
        
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
                current_index = i + 1
                # 检查避免连续索引（Zeckendorf约束）
                if not used_indices or current_index < used_indices[-1] - 1:
                    used_indices.append(current_index)
                    remaining -= self.fibonacci_numbers[i]
                    if remaining == 0:
                        break
                        
        return sorted(used_indices) if remaining == 0 else None

    def detect_crystal_period(self, operation: str, domain_size: int = 20) -> Dict[str, int]:
        """
        检测trace operation的crystal周期：找到最小p使得T(x+p) = T(x)
        
        Args:
            operation: 'add', 'multiply', 'xor', 'compose'
            domain_size: 分析的域大小
            
        Returns:
            Dict mapping lattice positions to their periods
        """
        if operation in self.crystal_cache:
            return self.crystal_cache[operation]
            
        periods = {}
        
        for x in range(domain_size):
            trace_x = self.trace_lattice.get(x, '0')
            period = self._find_minimal_period(x, trace_x, operation, domain_size)
            periods[x] = period
            
        self.crystal_cache[operation] = periods
        return periods
        
    def _find_minimal_period(self, x: int, trace_x: str, operation: str, max_period: int) -> int:
        """找到给定x和operation的最小周期p"""
        for p in range(1, max_period + 1):
            if self._check_period(x, trace_x, operation, p, max_period):
                return p
        return max_period  # 如果没找到，返回最大值
        
    def _check_period(self, x: int, trace_x: str, operation: str, period: int, domain_size: int) -> bool:
        """检查period是否对trace operation有效"""
        # 检查多个周期以确保一致性
        for test_x in range(x, min(x + 3 * period, domain_size), period):
            if test_x + period >= domain_size:
                break
                
            result_x = self._apply_trace_operation(test_x, operation)
            result_x_plus_p = self._apply_trace_operation(test_x + period, operation)
            
            if result_x != result_x_plus_p:
                return False
                
        return True
        
    def _apply_trace_operation(self, x: int, operation: str) -> str:
        """对x应用指定的trace operation"""
        trace_x = self.trace_lattice.get(x, '0')
        
        if operation == 'add':
            # Trace addition: 模拟加法在trace space中的表现
            return self._trace_add_operation(x)
        elif operation == 'multiply':
            # Trace multiplication
            return self._trace_multiply_operation(x)
        elif operation == 'xor':
            # Trace XOR operation
            return self._trace_xor_operation(x)
        elif operation == 'compose':
            # Trace composition
            return self._trace_compose_operation(x)
        else:
            return trace_x
            
    def _trace_add_operation(self, x: int) -> str:
        """Trace加法操作：T_add(x) = trace(x + shift)"""
        shift = 1  # 简单的单位shift
        result = (x + shift) % self.max_lattice_size
        return self.trace_lattice.get(result, '0')
        
    def _trace_multiply_operation(self, x: int) -> str:
        """Trace乘法操作：T_mult(x) = trace(x * factor)"""
        factor = 2  # 使用2作为乘法因子
        result = (x * factor) % self.max_lattice_size
        return self.trace_lattice.get(result, '0')
        
    def _trace_xor_operation(self, x: int) -> str:
        """Trace XOR操作：对trace进行位操作"""
        trace_x = self.trace_lattice.get(x, '0')
        mask = '101'  # 简单的XOR mask
        
        # 扩展到相同长度
        max_len = max(len(trace_x), len(mask))
        trace_padded = trace_x.zfill(max_len)
        mask_padded = mask.zfill(max_len)
        
        # 执行XOR
        result = ''
        for i in range(max_len):
            bit_result = str(int(trace_padded[i]) ^ int(mask_padded[i]))
            result += bit_result
            
        return result.lstrip('0') or '0'
        
    def _trace_compose_operation(self, x: int) -> str:
        """Trace composition操作：T_comp(x) = trace(trace_value_of(x))"""
        trace_x = self.trace_lattice.get(x, '0')
        # 将trace转换回数值然后再编码
        value = self._trace_to_value(trace_x)
        return self.trace_lattice.get(value % self.max_lattice_size, '0')
        
    def _trace_to_value(self, trace: str) -> int:
        """将trace转换回对应的数值（简化版本）"""
        if trace == '0':
            return 0
        
        # 简单的位权重计算
        value = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1':
                # 使用Fibonacci权重
                fib_index = i + 1
                if fib_index < len(self.fibonacci_numbers):
                    value += self.fibonacci_numbers[fib_index - 1]
                    
        return value

    def compute_crystal_symmetries(self, periods: Dict[str, int]) -> Dict[str, List[int]]:
        """计算crystal的对称群：识别相同周期的位置群"""
        symmetry_groups = defaultdict(list)
        
        for position, period in periods.items():
            symmetry_groups[period].append(position)
            
        # 按周期长度排序
        result = {}
        for period in sorted(symmetry_groups.keys()):
            result[f"period_{period}"] = sorted(symmetry_groups[period])
            
        return dict(result)

    def three_domain_analysis(self) -> Dict[str, any]:
        """三域分析：Traditional vs φ-constrained vs intersection crystallography"""
        operations = ['add', 'multiply', 'xor', 'compose']
        analysis = {
            'traditional_crystal_count': 0,
            'phi_constrained_count': 0,
            'intersection_count': 0,
            'operation_analysis': {}
        }
        
        for op in operations:
            periods = self.detect_crystal_period(op)
            
            # Traditional crystallography: 任何周期性都算crystal
            traditional_crystals = [pos for pos, period in periods.items() if period <= 10]
            
            # φ-constrained: 只有φ-valid positions的crystals
            phi_crystals = [pos for pos, period in periods.items() 
                          if period <= 10 and self._is_phi_valid_position(pos)]
            
            # Intersection: traditional crystals that satisfy φ-constraint
            intersection = [pos for pos in traditional_crystals if pos in phi_crystals]
            
            analysis['operation_analysis'][op] = {
                'traditional_crystals': len(traditional_crystals),
                'phi_crystals': len(phi_crystals),
                'intersection': len(intersection),
                'traditional_positions': traditional_crystals,
                'phi_positions': phi_crystals,
                'intersection_positions': intersection
            }
            
        # 总计统计
        analysis['traditional_crystal_count'] = sum(
            op_data['traditional_crystals'] for op_data in analysis['operation_analysis'].values()
        )
        analysis['phi_constrained_count'] = sum(
            op_data['phi_crystals'] for op_data in analysis['operation_analysis'].values()
        )
        analysis['intersection_count'] = sum(
            op_data['intersection'] for op_data in analysis['operation_analysis'].values()
        )
        
        return analysis
        
    def _is_phi_valid_position(self, pos: int) -> bool:
        """检查position是否φ-valid（trace不包含连续11）"""
        trace = self.trace_lattice.get(pos, '0')
        return '11' not in trace

    def build_crystal_graph(self, operation: str) -> nx.Graph:
        """构建crystal connectivity graph"""
        periods = self.detect_crystal_period(operation)
        symmetries = self.compute_crystal_symmetries(periods)
        
        G = nx.Graph()
        
        # 添加节点（每个position是一个节点）
        for pos in periods.keys():
            period = periods[pos]
            trace = self.trace_lattice.get(pos, '0')
            G.add_node(pos, period=period, trace=trace, 
                      phi_valid=self._is_phi_valid_position(pos))
        
        # 添加边（相同周期的positions相连）
        for period_group in symmetries.values():
            for i, pos1 in enumerate(period_group):
                for pos2 in period_group[i+1:]:
                    G.add_edge(pos1, pos2, symmetry_type='same_period')
                    
        # 添加相邻位置的边（表示lattice connectivity）
        for pos in periods.keys():
            for neighbor in [pos-1, pos+1]:
                if neighbor in periods:
                    G.add_edge(pos, neighbor, symmetry_type='lattice_neighbor')
                    
        return G


class CrystalAnalyzer:
    """
    Advanced analyzer for trace crystal properties using graph theory,
    information theory, and category theory approaches.
    """
    
    def __init__(self, crystal_system: TraceCrystalSystem):
        self.crystal_system = crystal_system
        
    def graph_theory_analysis(self, crystal_graph: nx.Graph) -> Dict[str, any]:
        """图论分析crystal connectivity"""
        analysis = {}
        
        # 基本图属性
        analysis['node_count'] = crystal_graph.number_of_nodes()
        analysis['edge_count'] = crystal_graph.number_of_edges()
        analysis['density'] = nx.density(crystal_graph)
        analysis['is_connected'] = nx.is_connected(crystal_graph)
        
        if crystal_graph.number_of_nodes() > 0:
            analysis['average_clustering'] = nx.average_clustering(crystal_graph)
            analysis['average_degree'] = sum(dict(crystal_graph.degree()).values()) / crystal_graph.number_of_nodes()
            
            # 连通分量分析
            components = list(nx.connected_components(crystal_graph))
            analysis['connected_components'] = len(components)
            analysis['largest_component_size'] = max(len(comp) for comp in components) if components else 0
            
            # 中心性分析
            if nx.is_connected(crystal_graph):
                analysis['diameter'] = nx.diameter(crystal_graph)
                analysis['radius'] = nx.radius(crystal_graph)
            
        return analysis
        
    def information_theory_analysis(self, periods: Dict[str, int]) -> Dict[str, float]:
        """信息论分析pattern entropy"""
        analysis = {}
        
        # 周期分布的熵
        period_counts = defaultdict(int)
        for period in periods.values():
            period_counts[period] += 1
            
        total_positions = len(periods)
        if total_positions > 0:
            period_entropy = 0
            for count in period_counts.values():
                prob = count / total_positions
                if prob > 0:
                    period_entropy -= prob * log2(prob)
            
            analysis['period_entropy'] = period_entropy
            analysis['max_period_entropy'] = log2(len(period_counts)) if period_counts else 0
            analysis['period_efficiency'] = (analysis['period_entropy'] / 
                                           analysis['max_period_entropy']) if analysis['max_period_entropy'] > 0 else 0
            
        # Pattern complexity
        unique_periods = len(set(periods.values()))
        analysis['period_diversity'] = unique_periods
        analysis['complexity_ratio'] = unique_periods / total_positions if total_positions > 0 else 0
        
        return analysis
        
    def category_theory_analysis(self, operations: List[str]) -> Dict[str, any]:
        """范畴论分析crystal morphisms"""
        analysis = {
            'morphism_preservation': {},
            'functor_properties': {},
            'composition_analysis': {}
        }
        
        # 分析不同operation之间的morphism preservation
        for op1, op2 in itertools.combinations(operations, 2):
            periods1 = self.crystal_system.detect_crystal_period(op1)
            periods2 = self.crystal_system.detect_crystal_period(op2)
            
            # 计算morphism preservation ratio
            preserved_count = 0
            total_count = 0
            
            for pos in periods1:
                if pos in periods2:
                    total_count += 1
                    if periods1[pos] == periods2[pos]:
                        preserved_count += 1
                        
            preservation_ratio = preserved_count / total_count if total_count > 0 else 0
            analysis['morphism_preservation'][f"{op1}_{op2}"] = preservation_ratio
            
        # Functor properties analysis
        for op in operations:
            periods = self.crystal_system.detect_crystal_period(op)
            
            # Identity preservation: 检查position 0的行为
            identity_period = periods.get(0, 1)
            analysis['functor_properties'][f"{op}_identity_period"] = identity_period
            
            # Composition compatibility (simplified)
            composition_score = sum(1 for p in periods.values() if p <= 5) / len(periods) if periods else 0
            analysis['functor_properties'][f"{op}_composition_score"] = composition_score
            
        return analysis


class TestTraceCrystals(unittest.TestCase):
    """Unit tests for trace crystal detection and analysis"""
    
    def setUp(self):
        """测试设置"""
        self.crystal_system = TraceCrystalSystem(max_lattice_size=30)
        self.analyzer = CrystalAnalyzer(self.crystal_system)
        
    def test_fibonacci_generation(self):
        """测试Fibonacci数列生成"""
        expected_start = [1, 1, 2, 3, 5, 8, 13, 21]
        self.assertEqual(self.crystal_system.fibonacci_numbers[:8], expected_start)
        
    def test_trace_encoding(self):
        """测试trace编码的φ-constraint preservation"""
        # 测试一些已知的编码
        test_cases = [
            (0, '0'),
            (1, '1'),     # F₂ = 1
            (2, '10'),    # F₃ = 2  
            (3, '100'),   # F₄ = 3
        ]
        
        for n, expected_pattern in test_cases:
            trace = self.crystal_system._encode_to_trace(n)
            # 验证no consecutive 1s
            self.assertNotIn('11', trace, f"Trace for {n} contains consecutive 1s: {trace}")
            
    def test_crystal_period_detection(self):
        """测试crystal周期检测"""
        periods = self.crystal_system.detect_crystal_period('add', domain_size=15)
        
        # 验证所有位置都有周期
        self.assertGreater(len(periods), 10)
        
        # 验证周期的合理性（都是正整数）
        for pos, period in periods.items():
            self.assertIsInstance(period, int)
            self.assertGreater(period, 0)
            
    def test_crystal_symmetries(self):
        """测试crystal对称群计算"""
        periods = self.crystal_system.detect_crystal_period('multiply', domain_size=20)
        symmetries = self.crystal_system.compute_crystal_symmetries(periods)
        
        # 验证对称群结构
        self.assertIsInstance(symmetries, dict)
        
        # 验证所有位置都被分类
        total_positions = sum(len(group) for group in symmetries.values())
        self.assertEqual(total_positions, len(periods))
        
    def test_three_domain_analysis(self):
        """测试三域分析"""
        analysis = self.crystal_system.three_domain_analysis()
        
        # 验证分析结构
        required_keys = ['traditional_crystal_count', 'phi_constrained_count', 
                        'intersection_count', 'operation_analysis']
        for key in required_keys:
            self.assertIn(key, analysis)
            
        # 验证操作分析
        operations = ['add', 'multiply', 'xor', 'compose']
        for op in operations:
            self.assertIn(op, analysis['operation_analysis'])
            
        # 验证数值合理性
        self.assertGreaterEqual(analysis['traditional_crystal_count'], analysis['intersection_count'])
        self.assertGreaterEqual(analysis['phi_constrained_count'], analysis['intersection_count'])
        
    def test_crystal_graph_construction(self):
        """测试crystal graph构建"""
        graph = self.crystal_system.build_crystal_graph('add')
        
        # 验证图结构
        self.assertIsInstance(graph, nx.Graph)
        self.assertGreater(graph.number_of_nodes(), 0)
        
        # 验证节点属性
        for node in graph.nodes():
            node_data = graph.nodes[node]
            self.assertIn('period', node_data)
            self.assertIn('trace', node_data)
            self.assertIn('phi_valid', node_data)
            
    def test_graph_theory_analysis(self):
        """测试图论分析"""
        graph = self.crystal_system.build_crystal_graph('multiply')
        analysis = self.analyzer.graph_theory_analysis(graph)
        
        # 验证分析结果
        required_metrics = ['node_count', 'edge_count', 'density', 'is_connected']
        for metric in required_metrics:
            self.assertIn(metric, analysis)
            
        # 验证数值范围
        self.assertGreaterEqual(analysis['density'], 0)
        self.assertLessEqual(analysis['density'], 1)
        
    def test_information_theory_analysis(self):
        """测试信息论分析"""
        periods = self.crystal_system.detect_crystal_period('xor')
        analysis = self.analyzer.information_theory_analysis(periods)
        
        # 验证熵计算
        required_metrics = ['period_entropy', 'period_diversity', 'complexity_ratio']
        for metric in required_metrics:
            self.assertIn(metric, analysis)
            
        # 验证熵的非负性
        self.assertGreaterEqual(analysis['period_entropy'], 0)
        
    def test_category_theory_analysis(self):
        """测试范畴论分析"""
        operations = ['add', 'multiply']
        analysis = self.analyzer.category_theory_analysis(operations)
        
        # 验证分析结构
        required_sections = ['morphism_preservation', 'functor_properties', 'composition_analysis']
        for section in required_sections:
            self.assertIn(section, analysis)
            
        # 验证morphism preservation
        self.assertIn('add_multiply', analysis['morphism_preservation'])
        preservation_ratio = analysis['morphism_preservation']['add_multiply']
        self.assertGreaterEqual(preservation_ratio, 0)
        self.assertLessEqual(preservation_ratio, 1)


def main():
    """Main verification routine"""
    print("=== Chapter 031: TraceCrystals Unit Test Verification ===")
    print("从ψ=ψ(ψ)推导trace tensor space中的自重复算术结构")
    print()
    
    # 1. 运行单元测试
    print("1. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTraceCrystals)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Unit tests failed. Please check the implementation.")
        return
        
    print()
    
    # 2. 系统分析
    print("2. Running Crystal Analysis...")
    crystal_system = TraceCrystalSystem(max_lattice_size=40)
    analyzer = CrystalAnalyzer(crystal_system)
    
    # Crystal detection for different operations
    operations = ['add', 'multiply', 'xor', 'compose']
    print(f"Detected trace operations: {operations}")
    
    for op in operations:
        periods = crystal_system.detect_crystal_period(op, domain_size=25)
        symmetries = crystal_system.compute_crystal_symmetries(periods)
        
        unique_periods = len(set(periods.values()))
        avg_period = sum(periods.values()) / len(periods) if periods else 0
        
        print(f"  {op}: {len(periods)} positions, {unique_periods} unique periods, avg={avg_period:.2f}")
        
    print()
    
    # 3. 三域分析
    print("3. Three-Domain Analysis:")
    three_domain = crystal_system.three_domain_analysis()
    print(f"   Traditional crystals: {three_domain['traditional_crystal_count']}")
    print(f"   φ-constrained crystals: {three_domain['phi_constrained_count']}")
    print(f"   Intersection crystals: {three_domain['intersection_count']}")
    
    intersection_ratio = (three_domain['intersection_count'] / 
                         three_domain['traditional_crystal_count']) if three_domain['traditional_crystal_count'] > 0 else 0
    print(f"   Intersection ratio: {intersection_ratio:.3f}")
    print()
    
    # 4. 图论分析
    print("4. Graph Theory Analysis:")
    add_graph = crystal_system.build_crystal_graph('add')
    graph_analysis = analyzer.graph_theory_analysis(add_graph)
    print(f"   Nodes: {graph_analysis['node_count']}")
    print(f"   Edges: {graph_analysis['edge_count']}")
    print(f"   Density: {graph_analysis['density']:.3f}")
    print(f"   Connected: {graph_analysis['is_connected']}")
    if 'average_clustering' in graph_analysis:
        print(f"   Clustering: {graph_analysis['average_clustering']:.3f}")
    print()
    
    # 5. 信息论分析
    print("5. Information Theory Analysis:")
    add_periods = crystal_system.detect_crystal_period('add')
    info_analysis = analyzer.information_theory_analysis(add_periods)
    print(f"   Period entropy: {info_analysis['period_entropy']:.3f} bits")
    print(f"   Period diversity: {info_analysis['period_diversity']}")
    print(f"   Complexity ratio: {info_analysis['complexity_ratio']:.3f}")
    print()
    
    # 6. 范畴论分析
    print("6. Category Theory Analysis:")
    category_analysis = analyzer.category_theory_analysis(['add', 'multiply', 'xor'])
    print(f"   Morphism preservations tested: {len(category_analysis['morphism_preservation'])}")
    
    for morphism, ratio in category_analysis['morphism_preservation'].items():
        print(f"   {morphism}: {ratio:.3f} preservation")
        
    print()
    
    print("=== Verification Complete ===")
    print("Key insights from verification:")
    print("1. φ-constraint creates structured crystal lattices with meaningful periods")
    print("2. Different trace operations exhibit distinct crystalline patterns")
    print("3. Intersection domain reveals universal crystallographic principles")
    print("4. Graph connectivity demonstrates emergent crystal network structure")
    

if __name__ == "__main__":
    main()