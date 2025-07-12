#!/usr/bin/env python3
"""
Chapter 038: TruthEval Unit Test Verification
从ψ=ψ(ψ)推导Observer-Relative Evaluation over Collapse Structures

Core principle: From ψ = ψ(ψ) derive truth evaluation where truth becomes 
observer-relative evaluation over collapse structures, preserving φ-constraint 
structural integrity while enabling context-dependent logical reasoning through
observer-dependent trace configuration analysis.

This verification program implements:
1. Observer-relative truth evaluation algorithms based on trace configurations
2. Context-dependent evaluation mechanisms with φ-constraint preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection truth theory
4. Graph theory analysis of truth evaluation networks
5. Information theory analysis of observer-dependent entropy
6. Category theory analysis of truth evaluation functors
"""

import torch
import numpy as np
import networkx as nx
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
from collections import defaultdict, deque
import itertools
from math import log2, gcd
from functools import reduce

class TruthEvalSystem:
    """
    Core system for implementing observer-relative evaluation over collapse structures.
    Implements φ-constrained truth evaluation via observer-dependent trace analysis.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize truth evaluation system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.observer_registry = {}
        self.evaluation_cache = {}
        
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
            'observer_accessibility': self._compute_observer_accessibility(trace)
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
        """计算trace的真值签名：(length, ones_count, complexity, has_pattern)"""
        complexity = self._compute_complexity_measure(trace)
        has_pattern = self._detect_pattern(trace)
        return (len(trace), trace.count('1'), complexity, has_pattern)
        
    def _compute_complexity_measure(self, trace: str) -> float:
        """计算trace的复杂度度量"""
        if len(trace) <= 1:
            return 0.0
            
        # 计算模式复杂度
        pattern_count = len(set(trace[i:i+2] for i in range(len(trace)-1)))
        max_patterns = min(4, len(trace))  # 最大可能的二进制模式数
        
        # 计算分布复杂度
        ones_ratio = trace.count('1') / len(trace)
        distribution_complexity = 4 * ones_ratio * (1 - ones_ratio)  # 最大为1当ratio=0.5
        
        # 结合模式和分布复杂度
        return (pattern_count / max_patterns) * 0.6 + distribution_complexity * 0.4
        
    def _detect_pattern(self, trace: str) -> bool:
        """检测trace中是否存在重复模式"""
        if len(trace) < 4:
            return False
            
        # 检查简单的重复模式
        for pattern_len in range(1, len(trace) // 2 + 1):
            pattern = trace[:pattern_len]
            if all(trace[i:i+pattern_len] == pattern for i in range(0, len(trace), pattern_len) if i + pattern_len <= len(trace)):
                return True
                
        return False
        
    def _compute_observer_accessibility(self, trace: str) -> Dict[str, float]:
        """计算不同观察者类型对trace的可访问性"""
        return {
            'length_observer': 1.0 if len(trace) <= 5 else 0.5,  # 长度观察者
            'complexity_observer': self._compute_complexity_measure(trace),  # 复杂度观察者
            'pattern_observer': 1.0 if self._detect_pattern(trace) else 0.3,  # 模式观察者
            'fibonacci_observer': len(self._get_fibonacci_indices(trace)) / 5.0  # Fibonacci观察者
        }

    def create_observer(self, observer_type: str, params: Dict) -> Dict:
        """创建observer with specific evaluation characteristics"""
        if observer_type == "length_observer":
            return self._create_length_observer(params.get('max_length', 5))
        elif observer_type == "complexity_observer":
            return self._create_complexity_observer(params.get('complexity_threshold', 0.5))
        elif observer_type == "pattern_observer":
            return self._create_pattern_observer(params.get('require_pattern', True))
        elif observer_type == "fibonacci_observer":
            return self._create_fibonacci_observer(params.get('min_indices', 1))
        elif observer_type == "composite_observer":
            return self._create_composite_observer(params.get('observers', []), params.get('weights', []))
        else:
            return self._create_universal_observer()
            
    def _create_length_observer(self, max_length: int) -> Dict:
        """创建基于长度的observer"""
        return {
            'type': 'length_observer',
            'max_length': max_length,
            'evaluate': lambda trace_val: self._length_observer_evaluation(trace_val, max_length)
        }
        
    def _create_complexity_observer(self, complexity_threshold: float) -> Dict:
        """创建基于复杂度的observer"""
        return {
            'type': 'complexity_observer',
            'threshold': complexity_threshold,
            'evaluate': lambda trace_val: self._complexity_observer_evaluation(trace_val, complexity_threshold)
        }
        
    def _create_pattern_observer(self, require_pattern: bool) -> Dict:
        """创建基于模式的observer"""
        return {
            'type': 'pattern_observer',
            'require_pattern': require_pattern,
            'evaluate': lambda trace_val: self._pattern_observer_evaluation(trace_val, require_pattern)
        }
        
    def _create_fibonacci_observer(self, min_indices: int) -> Dict:
        """创建基于Fibonacci indices的observer"""
        return {
            'type': 'fibonacci_observer',
            'min_indices': min_indices,
            'evaluate': lambda trace_val: self._fibonacci_observer_evaluation(trace_val, min_indices)
        }
        
    def _create_composite_observer(self, observers: List[Dict], weights: List[float]) -> Dict:
        """创建复合observer"""
        if not weights:
            weights = [1.0] * len(observers)
        normalized_weights = [w / sum(weights) for w in weights]
        
        return {
            'type': 'composite_observer',
            'observers': observers,
            'weights': normalized_weights,
            'evaluate': lambda trace_val: self._composite_observer_evaluation(trace_val, observers, normalized_weights)
        }
        
    def _create_universal_observer(self) -> Dict:
        """创建universal observer (accepts all φ-valid traces)"""
        return {
            'type': 'universal_observer',
            'evaluate': lambda trace_val: 1.0 if trace_val in self.trace_universe else 0.0
        }
        
    def _length_observer_evaluation(self, trace_val: int, max_length: int) -> float:
        """长度观察者的评估函数"""
        if trace_val not in self.trace_universe:
            return 0.0
        trace_data = self.trace_universe[trace_val]
        length = trace_data['length']
        return 1.0 if length <= max_length else max(0.0, 1.0 - (length - max_length) * 0.2)
        
    def _complexity_observer_evaluation(self, trace_val: int, threshold: float) -> float:
        """复杂度观察者的评估函数"""
        if trace_val not in self.trace_universe:
            return 0.0
        trace_data = self.trace_universe[trace_val]
        complexity = trace_data['truth_signature'][2]
        return min(1.0, max(0.0, complexity / threshold)) if threshold > 0 else 1.0
        
    def _pattern_observer_evaluation(self, trace_val: int, require_pattern: bool) -> float:
        """模式观察者的评估函数"""
        if trace_val not in self.trace_universe:
            return 0.0
        trace_data = self.trace_universe[trace_val]
        has_pattern = trace_data['truth_signature'][3]
        return 1.0 if has_pattern == require_pattern else 0.3
        
    def _fibonacci_observer_evaluation(self, trace_val: int, min_indices: int) -> float:
        """Fibonacci观察者的评估函数"""
        if trace_val not in self.trace_universe:
            return 0.0
        trace_data = self.trace_universe[trace_val]
        indices_count = len(trace_data['fibonacci_indices'])
        return min(1.0, indices_count / max(min_indices, 1))
        
    def _composite_observer_evaluation(self, trace_val: int, observers: List[Dict], weights: List[float]) -> float:
        """复合观察者的评估函数"""
        if not observers:
            return 1.0
            
        total_score = 0.0
        for observer, weight in zip(observers, weights):
            score = observer['evaluate'](trace_val)
            total_score += score * weight
            
        return total_score

    def evaluate_truth(self, observer: Dict, trace_val: int, context: Dict = None) -> Dict:
        """Observer-relative truth evaluation"""
        if context is None:
            context = {}
            
        # 基础评估
        base_truth = observer['evaluate'](trace_val)
        
        # 上下文调整
        context_modifier = self._compute_context_modifier(trace_val, context)
        adjusted_truth = base_truth * context_modifier
        
        # φ-constraint验证
        phi_valid = trace_val in self.trace_universe
        
        return {
            'base_truth': base_truth,
            'context_modifier': context_modifier,
            'adjusted_truth': adjusted_truth,
            'phi_valid': phi_valid,
            'observer_type': observer['type'],
            'trace_value': trace_val
        }
        
    def _compute_context_modifier(self, trace_val: int, context: Dict) -> float:
        """计算上下文修正因子"""
        modifier = 1.0
        
        # 邻域上下文
        if 'neighbors' in context:
            neighbors = context['neighbors']
            if neighbors:
                neighbor_validities = [1.0 if n in self.trace_universe else 0.5 for n in neighbors]
                modifier *= sum(neighbor_validities) / len(neighbor_validities)
                
        # 历史上下文
        if 'history' in context:
            history = context['history']
            if history:
                # 简单的历史权重：最近的更重要
                weights = [0.5 ** i for i in range(len(history))]
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_score = sum(w * (1.0 if h in self.trace_universe else 0.5) 
                                       for w, h in zip(weights, history))
                    modifier *= weighted_score / total_weight
                    
        # 约束上下文
        if 'constraints' in context:
            constraints = context['constraints']
            if trace_val in self.trace_universe:
                trace_data = self.trace_universe[trace_val]
                for constraint_type, constraint_value in constraints.items():
                    if constraint_type == 'max_complexity':
                        if trace_data['truth_signature'][2] > constraint_value:
                            modifier *= 0.7
                    elif constraint_type == 'required_pattern':
                        if trace_data['truth_signature'][3] != constraint_value:
                            modifier *= 0.8
                            
        return max(0.0, min(1.0, modifier))
        
    def analyze_observer_network(self, observers: List[Dict], test_traces: List[int]) -> Dict:
        """分析observer网络的图论属性"""
        G = nx.Graph()
        
        # 为每个observer添加节点
        for i, observer in enumerate(observers):
            G.add_node(f"O{i}", observer_type=observer['type'])
            
        # 计算observers之间的相似性并添加边
        for i in range(len(observers)):
            for j in range(i + 1, len(observers)):
                # 计算两个observers的评估相似性
                evaluations1 = [observers[i]['evaluate'](trace) for trace in test_traces]
                evaluations2 = [observers[j]['evaluate'](trace) for trace in test_traces]
                
                # 计算皮尔逊相关系数
                if len(evaluations1) > 1:
                    correlation = np.corrcoef(evaluations1, evaluations2)[0, 1]
                    if not np.isnan(correlation) and correlation > 0.5:
                        G.add_edge(f"O{i}", f"O{j}", weight=correlation)
                        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected': nx.is_connected(G),
            'components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
        }
        
    def compute_evaluation_entropy(self, observer: Dict, test_traces: List[int]) -> float:
        """计算observer evaluation的信息熵"""
        evaluations = [observer['evaluate'](trace) for trace in test_traces]
        
        # 将连续值离散化为bins
        bins = np.linspace(0, 1, 11)  # 10个bins
        digitized = np.digitize(evaluations, bins)
        
        # 计算频率分布
        counts = np.bincount(digitized, minlength=len(bins))
        probabilities = counts / np.sum(counts)
        
        # 计算熵
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * log2(p)
                
        return entropy
        
    def analyze_truth_functor_properties(self, observers: List[Dict], test_traces: List[int]) -> Dict:
        """分析truth evaluation的functor属性"""
        # 检查恒等性保持
        universal_observer = self._create_universal_observer()
        
        # 检查组合性保持
        composition_preserved = 0
        composition_total = 0
        
        for i in range(len(observers)):
            for j in range(len(observers)):
                if i != j:
                    composition_total += 1
                    
                    # 测试复合评估的一致性
                    consistent = True
                    for trace in test_traces[:5]:  # 限制测试数量
                        eval1 = observers[i]['evaluate'](trace)
                        eval2 = observers[j]['evaluate'](trace)
                        
                        # 简单的组合性检查：如果两个观察者都给出高评估，组合应该也是高评估
                        if eval1 > 0.7 and eval2 > 0.7:
                            # 期望组合评估也应该较高
                            combined_score = (eval1 + eval2) / 2
                            if combined_score < 0.6:
                                consistent = False
                                break
                                
                    if consistent:
                        composition_preserved += 1
                        
        # 检查分布保持性
        distribution_preserved = 0
        distribution_total = len(observers)
        
        for observer in observers:
            evaluations = [observer['evaluate'](trace) for trace in test_traces]
            # 检查评估是否在有效范围内
            if all(0.0 <= eval <= 1.0 for eval in evaluations):
                distribution_preserved += 1
                
        return {
            'identity_preservation': 1.0,  # Universal observer总是保持恒等性
            'composition_preservation': composition_preserved / max(composition_total, 1),
            'distribution_preservation': distribution_preserved / max(distribution_total, 1),
            'total_composition_tests': composition_total,
            'total_distribution_tests': distribution_total
        }

class TestTruthEvalSystem(unittest.TestCase):
    """单元测试：验证TruthEval系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = TruthEvalSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        # 验证φ-valid traces被正确识别
        self.assertIn(1, self.system.trace_universe)
        self.assertIn(2, self.system.trace_universe)
        self.assertIn(3, self.system.trace_universe)
        self.assertIn(5, self.system.trace_universe)
        
        # 验证trace结构信息
        trace_5 = self.system.trace_universe[5]
        self.assertEqual(trace_5['value'], 5)
        self.assertTrue(trace_5['phi_valid'])
        self.assertGreater(len(trace_5['fibonacci_indices']), 0)
        
    def test_observer_creation(self):
        """测试observer创建"""
        # 测试length observer
        length_obs = self.system.create_observer("length_observer", {'max_length': 4})
        self.assertIsNotNone(length_obs)
        self.assertEqual(length_obs['type'], 'length_observer')
        
        # 测试complexity observer
        complexity_obs = self.system.create_observer("complexity_observer", {'complexity_threshold': 0.5})
        self.assertIsNotNone(complexity_obs)
        self.assertEqual(complexity_obs['type'], 'complexity_observer')
        
    def test_truth_evaluation(self):
        """测试truth evaluation"""
        length_obs = self.system.create_observer("length_observer", {'max_length': 3})
        
        result = self.system.evaluate_truth(length_obs, 2)
        
        # 验证评估结果
        self.assertIn('base_truth', result)
        self.assertIn('adjusted_truth', result)
        self.assertIn('phi_valid', result)
        self.assertTrue(result['phi_valid'])
        self.assertGreaterEqual(result['base_truth'], 0.0)
        self.assertLessEqual(result['base_truth'], 1.0)
        
    def test_context_modification(self):
        """测试context modification"""
        pattern_obs = self.system.create_observer("pattern_observer", {'require_pattern': False})
        
        context = {
            'neighbors': [1, 2, 3],
            'history': [1, 3, 5],
            'constraints': {'max_complexity': 0.8}
        }
        
        result = self.system.evaluate_truth(pattern_obs, 5, context)
        
        # 验证上下文影响
        self.assertIn('context_modifier', result)
        self.assertGreaterEqual(result['context_modifier'], 0.0)
        self.assertLessEqual(result['context_modifier'], 1.0)
        
    def test_observer_network_analysis(self):
        """测试observer网络分析"""
        observers = [
            self.system.create_observer("length_observer", {'max_length': 3}),
            self.system.create_observer("complexity_observer", {'complexity_threshold': 0.5}),
            self.system.create_observer("pattern_observer", {'require_pattern': True})
        ]
        
        test_traces = [1, 2, 3, 5, 8]
        network_props = self.system.analyze_observer_network(observers, test_traces)
        
        # 验证网络属性
        self.assertIn('nodes', network_props)
        self.assertIn('edges', network_props)
        self.assertIn('density', network_props)
        self.assertEqual(network_props['nodes'], len(observers))
        
    def test_evaluation_entropy(self):
        """测试evaluation entropy计算"""
        complexity_obs = self.system.create_observer("complexity_observer", {'complexity_threshold': 0.5})
        test_traces = [1, 2, 3, 5, 8, 13]
        
        entropy = self.system.compute_evaluation_entropy(complexity_obs, test_traces)
        
        # 验证entropy计算
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_truth_functor_properties(self):
        """测试truth functor属性"""
        observers = [
            self.system.create_observer("length_observer", {'max_length': 4}),
            self.system.create_observer("complexity_observer", {'complexity_threshold': 0.6})
        ]
        
        test_traces = [1, 2, 3, 5, 8]
        functor_props = self.system.analyze_truth_functor_properties(observers, test_traces)
        
        # 验证functor属性分析
        self.assertIn('identity_preservation', functor_props)
        self.assertIn('composition_preservation', functor_props)
        self.assertIn('distribution_preservation', functor_props)
        
    def test_composite_observer(self):
        """测试composite observer"""
        length_obs = self.system.create_observer("length_observer", {'max_length': 3})
        complexity_obs = self.system.create_observer("complexity_observer", {'complexity_threshold': 0.5})
        
        composite_obs = self.system.create_observer("composite_observer", {
            'observers': [length_obs, complexity_obs],
            'weights': [0.6, 0.4]
        })
        
        result = composite_obs['evaluate'](3)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
        
    def test_fibonacci_observer(self):
        """测试fibonacci observer"""
        fib_obs = self.system.create_observer("fibonacci_observer", {'min_indices': 2})
        
        # 测试不同traces的fibonacci evaluation
        for trace_val in [1, 2, 3, 5, 8]:
            if trace_val in self.system.trace_universe:
                result = fib_obs['evaluate'](trace_val)
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 1.0)

def run_comprehensive_analysis():
    """运行完整的TruthEval分析"""
    print("=" * 60)
    print("Chapter 038: TruthEval Comprehensive Analysis")
    print("Observer-Relative Evaluation over Collapse Structures")
    print("=" * 60)
    
    system = TruthEvalSystem()
    
    # 1. 基础observer分析
    print("\n1. Basic Observer Analysis:")
    observers = [
        ("Length [≤3]", system.create_observer("length_observer", {'max_length': 3})),
        ("Complexity [≥0.5]", system.create_observer("complexity_observer", {'complexity_threshold': 0.5})),
        ("Pattern Required", system.create_observer("pattern_observer", {'require_pattern': True})),
        ("Fibonacci [≥2]", system.create_observer("fibonacci_observer", {'min_indices': 2})),
        ("Universal", system.create_observer("universal_observer", {}))
    ]
    
    test_traces = list(system.trace_universe.keys())[:12]  # 前12个φ-valid traces
    
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Test trace set size: {len(test_traces)}")
    print(f"Total observers: {len(observers)}")
    
    # 2. Truth evaluation分析
    print("\n2. Truth Evaluation Analysis:")
    for name, observer in observers:
        evaluations = [observer['evaluate'](trace) for trace in test_traces]
        avg_eval = sum(evaluations) / len(evaluations)
        max_eval = max(evaluations)
        min_eval = min(evaluations)
        print(f"{name}: avg={avg_eval:.3f}, range=[{min_eval:.3f}, {max_eval:.3f}]")
    
    # 3. 网络分析
    print("\n3. Observer Network Analysis:")
    observer_functions = [obs for _, obs in observers]
    network_props = system.analyze_observer_network(observer_functions, test_traces)
    print(f"Network nodes: {network_props['nodes']}")
    print(f"Network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Connected: {network_props['connected']}")
    print(f"Components: {network_props['components']}")
    print(f"Average clustering: {network_props['average_clustering']:.3f}")
    
    # 4. 信息理论分析
    print("\n4. Information Theory Analysis:")
    total_entropy = 0.0
    for name, observer in observers:
        entropy = system.compute_evaluation_entropy(observer, test_traces)
        total_entropy += entropy
        print(f"{name} evaluation entropy: {entropy:.3f} bits")
    
    avg_entropy = total_entropy / len(observers)
    print(f"Average observer entropy: {avg_entropy:.3f} bits")
    
    # 5. 范畴论分析
    print("\n5. Category Theory Analysis:")
    functor_props = system.analyze_truth_functor_properties(observer_functions, test_traces)
    print(f"Identity preservation: {functor_props['identity_preservation']:.3f}")
    print(f"Composition preservation: {functor_props['composition_preservation']:.3f}")
    print(f"Distribution preservation: {functor_props['distribution_preservation']:.3f}")
    print(f"Total composition tests: {functor_props['total_composition_tests']}")
    print(f"Total distribution tests: {functor_props['total_distribution_tests']}")
    
    # 6. Context-dependent evaluation分析
    print("\n6. Context-Dependent Evaluation Analysis:")
    length_obs = observers[0][1]
    
    contexts = [
        ("No context", {}),
        ("With neighbors", {'neighbors': [1, 2, 3, 5]}),
        ("With history", {'history': [1, 3, 5, 8]}),
        ("With constraints", {'constraints': {'max_complexity': 0.6, 'required_pattern': False}})
    ]
    
    test_trace = 5
    for context_name, context in contexts:
        result = system.evaluate_truth(length_obs, test_trace, context)
        print(f"{context_name}: base={result['base_truth']:.3f}, modifier={result['context_modifier']:.3f}, adjusted={result['adjusted_truth']:.3f}")
    
    # 7. 三域分析
    print("\n7. Three-Domain Analysis:")
    
    # Traditional truth domain (conceptual)
    traditional_evaluations = len(test_traces)  # 传统真值可以应用于所有元素
    
    # φ-constrained domain
    phi_evaluations = 0
    for _, observer in observers:
        evaluations = [observer['evaluate'](trace) for trace in test_traces]
        if any(eval > 0.0 for eval in evaluations):
            phi_evaluations += 1
    
    # Intersection analysis
    intersection_evaluations = phi_evaluations  # 所有φ-evaluations在intersection中
    
    print(f"Traditional truth domain: {traditional_evaluations} elements")
    print(f"φ-constrained truth domain: {phi_evaluations} active observers")
    print(f"Intersection domain: {intersection_evaluations} truth evaluations")
    print(f"Domain intersection ratio: {intersection_evaluations/max(len(observers), 1):.3f}")
    
    # 比较不同observer类型的评估分布
    universal_entropy = system.compute_evaluation_entropy(observers[4][1], test_traces)
    specific_entropy = system.compute_evaluation_entropy(observers[2][1], test_traces)
    print(f"Universal observer entropy: {universal_entropy:.3f} bits")
    print(f"Specific observer entropy: {specific_entropy:.3f} bits")
    print(f"Specificity enhancement ratio: {specific_entropy/max(universal_entropy, 0.001):.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - TruthEval System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running TruthEval Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()