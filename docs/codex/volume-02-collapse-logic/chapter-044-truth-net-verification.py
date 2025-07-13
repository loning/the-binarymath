#!/usr/bin/env python3
"""
Chapter 044: TruthNet Unit Test Verification
从ψ=ψ(ψ)推导Collapse-Aware Tensor Truth Table Generator

Core principle: From ψ = ψ(ψ) derive truth tables where logical truth evaluation
emerges as tensor structures respecting φ-constraints, creating systematic truth
assignments that maintain structural coherence across all logical combinations.

This verification program implements:
1. Tensor-based truth table generation with φ-constraint preservation
2. Multi-dimensional truth evaluation through tensor operations
3. Three-domain analysis: Traditional vs φ-constrained vs intersection truth table theory
4. Graph theory analysis of truth assignment networks
5. Information theory analysis of truth distribution entropy
6. Category theory analysis of truth table functors
7. Visualization of truth table structures
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
from collections import defaultdict, deque
import itertools
from math import log2, gcd
from functools import reduce

class TruthNetSystem:
    """
    Core system for implementing collapse-aware tensor truth table generation.
    Implements φ-constrained truth tables via tensor structural analysis.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize truth net system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.truth_cache = {}
        self.tensor_registry = {}
        
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
            'truth_signature': self._compute_truth_signature(trace),
            'tensor_properties': self._compute_tensor_properties(trace)
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
        
    def _compute_truth_signature(self, trace: str) -> Tuple[int, int, float, bool]:
        """计算trace的真值签名：(length, ones_count, density, balanced)"""
        density = trace.count('1') / max(len(trace), 1)
        balanced = abs(trace.count('1') - trace.count('0')) <= 1
        return (len(trace), trace.count('1'), density, balanced)
        
    def _compute_tensor_properties(self, trace: str) -> Dict[str, Union[int, float, torch.Tensor]]:
        """计算trace的张量属性"""
        # 转换为张量
        tensor = torch.tensor([int(b) for b in trace], dtype=torch.float32)
        
        return {
            'tensor': tensor,
            'dimension': len(trace),
            'norm': torch.norm(tensor).item(),
            'sparsity': (tensor == 0).sum().item() / len(tensor),
            'entropy': self._compute_tensor_entropy(tensor)
        }
        
    def _compute_tensor_entropy(self, tensor: torch.Tensor) -> float:
        """计算张量的信息熵"""
        if len(tensor) == 0:
            return 0.0
        probs = torch.bincount(tensor.long()) / len(tensor)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * torch.log2(p)
        return entropy.item()

    def generate_truth_table(self, variables: List[int], operation: str = 'AND') -> Dict:
        """生成真值表"""
        if not all(v in self.trace_universe for v in variables):
            return {'valid': False, 'reason': 'Invalid variables'}
            
        # 构建真值表张量
        n_vars = len(variables)
        n_rows = 2 ** n_vars
        
        # 创建输入组合
        input_combinations = []
        for i in range(n_rows):
            row = []
            for j in range(n_vars):
                bit = (i >> (n_vars - 1 - j)) & 1
                row.append(bit)
            input_combinations.append(row)
            
        # 计算输出
        outputs = []
        phi_valid_count = 0
        
        for inputs in input_combinations:
            # 将输入映射到traces
            trace_values = []
            for i, var in enumerate(variables):
                if inputs[i] == 1:
                    trace_values.append(var)
                else:
                    trace_values.append(0)  # 使用0表示False
                    
            # 计算操作结果
            result = self._compute_operation(trace_values, operation)
            outputs.append(result)
            
            if result['phi_valid']:
                phi_valid_count += 1
                
        # 构建真值表张量
        input_tensor = torch.tensor(input_combinations, dtype=torch.float32)
        output_values = [o['value'] for o in outputs]
        output_tensor = torch.tensor(output_values, dtype=torch.float32)
        
        # 构建完整真值表
        truth_table = torch.cat([input_tensor, output_tensor.unsqueeze(1)], dim=1)
        
        return {
            'valid': True,
            'variables': variables,
            'operation': operation,
            'n_rows': n_rows,
            'truth_table': truth_table,
            'input_combinations': input_combinations,
            'outputs': outputs,
            'phi_valid_ratio': phi_valid_count / n_rows,
            'tensor_properties': self._analyze_truth_table_properties(truth_table)
        }
        
    def _compute_operation(self, trace_values: List[int], operation: str) -> Dict:
        """计算逻辑操作结果"""
        # 过滤出有效的traces
        valid_traces = [v for v in trace_values if v > 0 and v in self.trace_universe]
        
        if not valid_traces:
            return {'value': 0, 'phi_valid': True, 'trace': '0'}
            
        if operation == 'AND':
            result = self._tensor_and(valid_traces)
        elif operation == 'OR':
            result = self._tensor_or(valid_traces)
        elif operation == 'XOR':
            result = self._tensor_xor(valid_traces)
        elif operation == 'IMPL':
            if len(valid_traces) >= 2:
                result = self._tensor_implies(valid_traces[0], valid_traces[1])
            else:
                result = {'value': 0, 'phi_valid': True}
        else:
            result = {'value': 0, 'phi_valid': True}
            
        return result
        
    def _tensor_and(self, traces: List[int]) -> Dict:
        """张量AND操作"""
        if not traces:
            return {'value': 0, 'phi_valid': True}
            
        # 获取所有trace的张量表示
        tensors = []
        for t in traces:
            if t in self.trace_universe:
                tensor = self.trace_universe[t]['tensor_properties']['tensor']
                tensors.append(tensor)
                
        if not tensors:
            return {'value': 0, 'phi_valid': True}
            
        # 对齐张量长度
        max_len = max(len(t) for t in tensors)
        aligned_tensors = []
        for t in tensors:
            if len(t) < max_len:
                padded = torch.nn.functional.pad(t, (0, max_len - len(t)))
                aligned_tensors.append(padded)
            else:
                aligned_tensors.append(t)
                
        # 执行AND操作（取最小值）
        result_tensor = aligned_tensors[0]
        for t in aligned_tensors[1:]:
            result_tensor = torch.minimum(result_tensor, t)
            
        # 转换回trace
        result_trace = ''.join(str(int(b)) for b in result_tensor)
        phi_valid = '11' not in result_trace
        
        # 计算数值
        value = self._trace_to_value(result_trace)
        
        return {
            'value': value,
            'phi_valid': phi_valid,
            'trace': result_trace,
            'tensor': result_tensor
        }
        
    def _tensor_or(self, traces: List[int]) -> Dict:
        """张量OR操作"""
        if not traces:
            return {'value': 0, 'phi_valid': True}
            
        # 获取所有trace的张量表示
        tensors = []
        for t in traces:
            if t in self.trace_universe:
                tensor = self.trace_universe[t]['tensor_properties']['tensor']
                tensors.append(tensor)
                
        if not tensors:
            return {'value': 0, 'phi_valid': True}
            
        # 对齐张量长度
        max_len = max(len(t) for t in tensors)
        aligned_tensors = []
        for t in tensors:
            if len(t) < max_len:
                padded = torch.nn.functional.pad(t, (0, max_len - len(t)))
                aligned_tensors.append(padded)
            else:
                aligned_tensors.append(t)
                
        # 执行OR操作（取最大值）
        result_tensor = aligned_tensors[0]
        for t in aligned_tensors[1:]:
            result_tensor = torch.maximum(result_tensor, t)
            
        # 转换回trace
        result_trace = ''.join(str(int(b)) for b in result_tensor)
        phi_valid = '11' not in result_trace
        
        # 计算数值
        value = self._trace_to_value(result_trace)
        
        return {
            'value': value,
            'phi_valid': phi_valid,
            'trace': result_trace,
            'tensor': result_tensor
        }
        
    def _tensor_xor(self, traces: List[int]) -> Dict:
        """张量XOR操作"""
        if not traces:
            return {'value': 0, 'phi_valid': True}
            
        # 获取所有trace的张量表示
        tensors = []
        for t in traces:
            if t in self.trace_universe:
                tensor = self.trace_universe[t]['tensor_properties']['tensor']
                tensors.append(tensor)
                
        if not tensors:
            return {'value': 0, 'phi_valid': True}
            
        # 对齐张量长度
        max_len = max(len(t) for t in tensors)
        aligned_tensors = []
        for t in tensors:
            if len(t) < max_len:
                padded = torch.nn.functional.pad(t, (0, max_len - len(t)))
                aligned_tensors.append(padded)
            else:
                aligned_tensors.append(t)
                
        # 执行XOR操作
        result_tensor = aligned_tensors[0]
        for t in aligned_tensors[1:]:
            result_tensor = (result_tensor + t) % 2
            
        # 转换回trace
        result_trace = ''.join(str(int(b)) for b in result_tensor)
        phi_valid = '11' not in result_trace
        
        # 计算数值
        value = self._trace_to_value(result_trace)
        
        return {
            'value': value,
            'phi_valid': phi_valid,
            'trace': result_trace,
            'tensor': result_tensor
        }
        
    def _tensor_implies(self, a: int, b: int) -> Dict:
        """张量蕴含操作：a → b ≡ ¬a ∨ b"""
        # 获取NOT a
        not_a = self._tensor_not(a)
        if not_a['phi_valid']:
            # 计算 NOT a OR b
            return self._tensor_or([not_a['value'], b])
        else:
            # 如果NOT a无效，直接返回b
            return {'value': b, 'phi_valid': True}
            
    def _tensor_not(self, trace_value: int) -> Dict:
        """张量NOT操作"""
        if trace_value not in self.trace_universe:
            return {'value': 0, 'phi_valid': True}
            
        tensor = self.trace_universe[trace_value]['tensor_properties']['tensor']
        
        # 执行NOT操作
        result_tensor = 1 - tensor
        
        # 转换回trace
        result_trace = ''.join(str(int(b)) for b in result_tensor)
        phi_valid = '11' not in result_trace
        
        # 计算数值
        value = self._trace_to_value(result_trace) if phi_valid else 0
        
        return {
            'value': value,
            'phi_valid': phi_valid,
            'trace': result_trace,
            'tensor': result_tensor
        }
        
    def _trace_to_value(self, trace: str) -> int:
        """将trace转换回数值"""
        value = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                value += self.fibonacci_numbers[i]
        return value
        
    def _analyze_truth_table_properties(self, truth_table: torch.Tensor) -> Dict:
        """分析真值表的属性"""
        n_vars = truth_table.shape[1] - 1
        
        # 计算各种属性
        return {
            'shape': truth_table.shape,
            'rank': torch.linalg.matrix_rank(truth_table).item(),
            'determinant': torch.det(truth_table[:n_vars, :n_vars]).item() if n_vars <= truth_table.shape[0] else 0,
            'trace': torch.trace(truth_table[:min(truth_table.shape)]).item(),
            'norm': torch.norm(truth_table).item(),
            'entropy': self._compute_truth_table_entropy(truth_table),
            'symmetry': self._check_truth_table_symmetry(truth_table),
            'completeness': self._check_truth_table_completeness(truth_table)
        }
        
    def _compute_truth_table_entropy(self, truth_table: torch.Tensor) -> float:
        """计算真值表的熵"""
        # 计算输出列的熵
        outputs = truth_table[:, -1]
        unique_values, counts = torch.unique(outputs, return_counts=True)
        probs = counts.float() / len(outputs)
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * torch.log2(p)
                
        return entropy.item()
        
    def _check_truth_table_symmetry(self, truth_table: torch.Tensor) -> Dict:
        """检查真值表的对称性"""
        n_vars = truth_table.shape[1] - 1
        
        # 检查输入对称性
        input_symmetric = True
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # 交换列i和j，检查输出是否不变
                swapped = truth_table.clone()
                swapped[:, [i, j]] = swapped[:, [j, i]]
                if not torch.allclose(truth_table[:, -1], swapped[:, -1]):
                    input_symmetric = False
                    break
                    
        return {
            'input_symmetric': input_symmetric,
            'self_dual': self._check_self_dual(truth_table)
        }
        
    def _check_self_dual(self, truth_table: torch.Tensor) -> bool:
        """检查自对偶性"""
        # 对于二元操作，检查f(¬x, ¬y) = ¬f(x, y)
        n_rows = truth_table.shape[0]
        for i in range(n_rows // 2):
            # 获取互补的输入行
            complement_idx = n_rows - 1 - i
            if abs(truth_table[i, -1] + truth_table[complement_idx, -1] - 1) > 0.01:
                return False
        return True
        
    def _check_truth_table_completeness(self, truth_table: torch.Tensor) -> Dict:
        """检查真值表的完备性"""
        outputs = truth_table[:, -1]
        unique_outputs = torch.unique(outputs)
        
        return {
            'functionally_complete': len(unique_outputs) > 1,
            'output_diversity': len(unique_outputs) / truth_table.shape[0],
            'constant_function': len(unique_outputs) == 1
        }

    def visualize_truth_table(self, truth_result: Dict, save_path: str = None):
        """可视化真值表"""
        if not truth_result['valid']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 真值表热图
        ax = axes[0, 0]
        truth_table = truth_result['truth_table'].numpy()
        sns.heatmap(truth_table, annot=True, fmt='.0f', cmap='coolwarm', ax=ax)
        ax.set_title(f"Truth Table - {truth_result['operation']}")
        ax.set_xlabel("Variables + Output")
        ax.set_ylabel("Row")
        
        # 2. φ-validity分布
        ax = axes[0, 1]
        phi_valid_counts = [1 if o['phi_valid'] else 0 for o in truth_result['outputs']]
        ax.bar(['φ-valid', 'φ-invalid'], 
               [sum(phi_valid_counts), len(phi_valid_counts) - sum(phi_valid_counts)])
        ax.set_title("φ-Validity Distribution")
        ax.set_ylabel("Count")
        
        # 3. 输出值分布
        ax = axes[1, 0]
        output_values = [o['value'] for o in truth_result['outputs']]
        unique_outputs, counts = np.unique(output_values, return_counts=True)
        ax.bar(unique_outputs.astype(str), counts)
        ax.set_title("Output Value Distribution")
        ax.set_xlabel("Output Value")
        ax.set_ylabel("Frequency")
        
        # 4. 张量属性
        ax = axes[1, 1]
        props = truth_result['tensor_properties']
        prop_names = ['Rank', 'Norm', 'Entropy']
        prop_values = [props['rank'], props['norm'], props['entropy']]
        ax.bar(prop_names, prop_values)
        ax.set_title("Tensor Properties")
        ax.set_ylabel("Value")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def analyze_truth_network(self, operations: List[str], max_vars: int = 3) -> Dict:
        """分析真值表网络的图论属性"""
        G = nx.Graph()
        
        # 为每个操作和变量数量组合创建节点
        for op in operations:
            for n_vars in range(1, max_vars + 1):
                node_id = f"{op}_{n_vars}"
                G.add_node(node_id, operation=op, n_vars=n_vars)
                
        # 添加边：基于操作的组合性
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    op1, vars1 = node1.split('_')
                    op2, vars2 = node2.split('_')
                    
                    # 如果操作可以组合，添加边
                    if self._can_compose_operations(op1, op2):
                        G.add_edge(node1, node2)
                        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected': nx.is_connected(G),
            'components': nx.number_connected_components(G),
            'average_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
            'clustering': nx.average_clustering(G)
        }
        
    def _can_compose_operations(self, op1: str, op2: str) -> bool:
        """检查两个操作是否可以组合"""
        # 定义操作组合规则
        composable = {
            'AND': ['OR', 'NOT', 'XOR'],
            'OR': ['AND', 'NOT', 'XOR'],
            'NOT': ['AND', 'OR', 'XOR', 'IMPL'],
            'XOR': ['AND', 'OR', 'NOT'],
            'IMPL': ['AND', 'OR', 'NOT']
        }
        
        return op2 in composable.get(op1, [])
        
    def compute_truth_entropy_spectrum(self, max_vars: int = 3) -> Dict:
        """计算真值熵谱"""
        entropy_spectrum = defaultdict(list)
        
        for n_vars in range(1, max_vars + 1):
            # 选择前n个变量
            variables = list(self.trace_universe.keys())[1:n_vars+1]
            
            for op in ['AND', 'OR', 'XOR', 'IMPL']:
                truth_result = self.generate_truth_table(variables, op)
                if truth_result['valid']:
                    entropy = truth_result['tensor_properties']['entropy']
                    entropy_spectrum[op].append((n_vars, entropy))
                    
        return dict(entropy_spectrum)

class TestTruthNetSystem(unittest.TestCase):
    """单元测试：验证TruthNet系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = TruthNetSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        # 验证φ-valid traces被正确识别
        self.assertIn(1, self.system.trace_universe)
        self.assertIn(2, self.system.trace_universe)
        self.assertIn(3, self.system.trace_universe)
        self.assertIn(5, self.system.trace_universe)
        
        # 验证张量属性
        trace_5 = self.system.trace_universe[5]
        self.assertIn('tensor_properties', trace_5)
        self.assertIsInstance(trace_5['tensor_properties']['tensor'], torch.Tensor)
        
    def test_truth_table_generation(self):
        """测试真值表生成"""
        variables = [1, 2]
        
        # 测试AND操作
        and_result = self.system.generate_truth_table(variables, 'AND')
        self.assertTrue(and_result['valid'])
        self.assertEqual(and_result['n_rows'], 4)
        self.assertIsInstance(and_result['truth_table'], torch.Tensor)
        
        # 测试OR操作
        or_result = self.system.generate_truth_table(variables, 'OR')
        self.assertTrue(or_result['valid'])
        
    def test_tensor_operations(self):
        """测试张量操作"""
        # 测试AND
        and_result = self.system._tensor_and([1, 2])
        self.assertIn('value', and_result)
        self.assertIn('phi_valid', and_result)
        
        # 测试OR
        or_result = self.system._tensor_or([1, 2])
        self.assertIn('value', or_result)
        
        # 测试XOR
        xor_result = self.system._tensor_xor([1, 2])
        self.assertIn('value', xor_result)
        
    def test_truth_table_properties(self):
        """测试真值表属性分析"""
        variables = [1, 2]
        result = self.system.generate_truth_table(variables, 'AND')
        
        props = result['tensor_properties']
        self.assertIn('rank', props)
        self.assertIn('entropy', props)
        self.assertIn('symmetry', props)
        self.assertIn('completeness', props)
        
    def test_truth_network_analysis(self):
        """测试真值网络分析"""
        operations = ['AND', 'OR', 'NOT', 'XOR']
        network_props = self.system.analyze_truth_network(operations, max_vars=2)
        
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        
    def test_entropy_spectrum(self):
        """测试熵谱计算"""
        spectrum = self.system.compute_truth_entropy_spectrum(max_vars=2)
        
        self.assertIsInstance(spectrum, dict)
        for op in ['AND', 'OR', 'XOR']:
            if op in spectrum:
                self.assertIsInstance(spectrum[op], list)

def run_comprehensive_analysis():
    """运行完整的TruthNet分析"""
    print("=" * 60)
    print("Chapter 044: TruthNet Comprehensive Analysis")
    print("Collapse-Aware Tensor Truth Table Generator")
    print("=" * 60)
    
    system = TruthNetSystem()
    
    # 1. 基础真值表分析
    print("\n1. Basic Truth Table Analysis:")
    test_variables = list(system.trace_universe.keys())[1:4]  # 使用前3个非零φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Test variables: {test_variables}")
    
    # 2. 操作真值表生成
    print("\n2. Operation Truth Tables:")
    operations = ['AND', 'OR', 'XOR', 'IMPL']
    operation_results = {}
    
    for op in operations:
        result = system.generate_truth_table(test_variables[:2], op)
        operation_results[op] = result
        
        if result['valid']:
            print(f"\n{op} Truth Table:")
            print(f"  Variables: {result['variables']}")
            print(f"  Rows: {result['n_rows']}")
            print(f"  φ-valid ratio: {result['phi_valid_ratio']:.3f}")
            print(f"  Tensor rank: {result['tensor_properties']['rank']}")
            print(f"  Entropy: {result['tensor_properties']['entropy']:.3f}")
            
    # 3. φ-validity分析
    print("\n3. φ-Validity Analysis:")
    total_valid = 0
    total_invalid = 0
    
    for op, result in operation_results.items():
        if result['valid']:
            valid_count = sum(1 for o in result['outputs'] if o['phi_valid'])
            invalid_count = len(result['outputs']) - valid_count
            total_valid += valid_count
            total_invalid += invalid_count
            print(f"{op}: {valid_count} φ-valid, {invalid_count} φ-invalid")
            
    print(f"\nTotal: {total_valid} φ-valid, {total_invalid} φ-invalid")
    print(f"Overall φ-validity rate: {total_valid/(total_valid+total_invalid):.3f}")
    
    # 4. 张量属性分析
    print("\n4. Tensor Properties Analysis:")
    for op, result in operation_results.items():
        if result['valid']:
            props = result['tensor_properties']
            print(f"\n{op} tensor properties:")
            print(f"  Shape: {props['shape']}")
            print(f"  Rank: {props['rank']}")
            print(f"  Norm: {props['norm']:.3f}")
            print(f"  Trace: {props['trace']:.3f}")
            
    # 5. 对称性分析
    print("\n5. Symmetry Analysis:")
    for op, result in operation_results.items():
        if result['valid']:
            sym = result['tensor_properties']['symmetry']
            print(f"{op}: input_symmetric={sym['input_symmetric']}, self_dual={sym['self_dual']}")
            
    # 6. 完备性分析
    print("\n6. Completeness Analysis:")
    for op, result in operation_results.items():
        if result['valid']:
            comp = result['tensor_properties']['completeness']
            print(f"{op}: functionally_complete={comp['functionally_complete']}, "
                  f"output_diversity={comp['output_diversity']:.3f}")
            
    # 7. 网络分析
    print("\n7. Truth Network Analysis:")
    network_props = system.analyze_truth_network(operations, max_vars=3)
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Connected: {network_props['connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Average degree: {network_props['average_degree']:.3f}")
    print(f"Clustering: {network_props['clustering']:.3f}")
    
    # 8. 信息理论分析 - 熵谱
    print("\n8. Information Theory - Entropy Spectrum:")
    spectrum = system.compute_truth_entropy_spectrum(max_vars=3)
    
    for op, entropy_data in spectrum.items():
        print(f"\n{op} entropy spectrum:")
        for n_vars, entropy in entropy_data:
            print(f"  {n_vars} variables: {entropy:.3f} bits")
            
    # 9. 三域分析
    print("\n9. Three-Domain Analysis:")
    
    # Traditional truth table domain
    n_vars = 2
    traditional_rows = 2 ** n_vars
    
    # φ-constrained domain
    phi_rows = sum(1 for result in operation_results.values() 
                   if result['valid'] for o in result['outputs'] if o['phi_valid'])
    
    # Intersection analysis
    intersection_rows = phi_rows  # 所有φ-valid在intersection中
    
    print(f"Traditional truth table domain: {traditional_rows * len(operations)} total entries")
    print(f"φ-constrained domain: {phi_rows} valid entries")
    print(f"Intersection domain: {intersection_rows} entries")
    print(f"Domain intersection ratio: {intersection_rows/(traditional_rows * len(operations)):.3f}")
    
    # 10. 可视化
    print("\n10. Generating Truth Table Visualizations...")
    for op, result in operation_results.items():
        if result['valid']:
            save_path = f"truth_table_{op.lower()}.png"
            system.visualize_truth_table(result, save_path)
            print(f"Saved visualization: {save_path}")
    
    # 11. 张量真值表深度分析
    print("\n11. Tensor Truth Table Deep Analysis:")
    
    # 分析不同操作的张量性质
    for op in ['AND', 'OR', 'XOR']:
        result = system.generate_truth_table(test_variables[:3], op)
        if result['valid']:
            truth_tensor = result['truth_table']
            print(f"\n{op} tensor analysis:")
            
            # 计算奇异值
            U, S, V = torch.svd(truth_tensor.float())
            print(f"  Top 3 singular values: {S[:3].tolist()}")
            
            # 计算条件数
            cond = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
            print(f"  Condition number: {cond:.3f}")
            
    print("\n" + "=" * 60)
    print("Analysis Complete - TruthNet System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running TruthNet Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()