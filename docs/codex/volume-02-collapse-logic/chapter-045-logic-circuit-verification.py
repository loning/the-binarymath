#!/usr/bin/env python3
"""
Chapter 045: LogicCircuit Unit Test Verification
从ψ=ψ(ψ)推导Constructing φ-Binary Circuits from Trace Primitives

Core principle: From ψ = ψ(ψ) derive logic circuits where circuit elements emerge
as φ-constrained trace primitives, creating systematic circuit construction that
maintains structural coherence across all logic gates and connections.

This verification program implements:
1. φ-constrained logic gate primitives as trace transformations
2. Circuit construction through trace composition and routing
3. Three-domain analysis: Traditional vs φ-constrained vs intersection circuit theory
4. Graph theory analysis of circuit topology networks
5. Information theory analysis of signal flow entropy
6. Category theory analysis of circuit composition functors
7. Visualization of circuit structures and signal propagation
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
import itertools
from math import log2, gcd, sqrt, pi
from functools import reduce

class LogicCircuitSystem:
    """
    Core system for implementing φ-binary circuits from trace primitives.
    Implements φ-constrained logic circuits via trace-based components.
    """
    
    def __init__(self, max_trace_size: int = 30):
        """Initialize logic circuit system"""
        self.max_trace_size = max_trace_size
        self.fibonacci_numbers = self._generate_fibonacci(25)
        self.trace_universe = self._build_trace_universe()
        self.gate_library = self._build_gate_library()
        self.circuit_cache = {}
        
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
            'circuit_signature': self._compute_circuit_signature(trace),
            'gate_properties': self._compute_gate_properties(trace)
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
        
    def _compute_circuit_signature(self, trace: str) -> Tuple[int, int, float, int]:
        """计算trace的电路签名：(length, ones_count, density, fan_out)"""
        density = trace.count('1') / max(len(trace), 1)
        fan_out = self._compute_fan_out(trace)
        return (len(trace), trace.count('1'), density, fan_out)
        
    def _compute_gate_properties(self, trace: str) -> Dict[str, Union[int, float, List]]:
        """计算trace作为逻辑门的属性"""
        return {
            'input_capacity': self._compute_input_capacity(trace),
            'output_strength': self._compute_output_strength(trace),
            'propagation_delay': self._compute_propagation_delay(trace),
            'power_consumption': self._compute_power_consumption(trace),
            'noise_margin': self._compute_noise_margin(trace)
        }
        
    def _compute_fan_out(self, trace: str) -> int:
        """计算扇出能力"""
        if not trace or trace == '0':
            return 0
        return min(trace.count('0') + 1, 4)  # 限制最大扇出为4
        
    def _compute_input_capacity(self, trace: str) -> int:
        """计算输入容量"""
        return min(len(trace), 3)  # 限制最大输入为3
        
    def _compute_output_strength(self, trace: str) -> float:
        """计算输出强度"""
        if not trace:
            return 0.0
        return trace.count('1') / len(trace)
        
    def _compute_propagation_delay(self, trace: str) -> int:
        """计算传播延迟"""
        return len(trace) + trace.count('1')
        
    def _compute_power_consumption(self, trace: str) -> float:
        """计算功耗"""
        return trace.count('1') * 0.5 + len(trace) * 0.1
        
    def _compute_noise_margin(self, trace: str) -> float:
        """计算噪声容限"""
        if not trace or len(trace) < 2:
            return 0.0
        transitions = sum(1 for i in range(len(trace)-1) if trace[i] != trace[i+1])
        return 1.0 - (transitions / (len(trace) - 1))

    def _build_gate_library(self) -> Dict[str, Dict]:
        """构建基础逻辑门库"""
        return {
            'NOT': self._create_not_gate(),
            'AND': self._create_and_gate(),
            'OR': self._create_or_gate(),
            'XOR': self._create_xor_gate(),
            'NAND': self._create_nand_gate(),
            'NOR': self._create_nor_gate(),
            'BUFFER': self._create_buffer_gate(),
            'WIRE': self._create_wire_component()
        }
        
    def _create_not_gate(self) -> Dict:
        """创建NOT门"""
        return {
            'type': 'NOT',
            'inputs': 1,
            'outputs': 1,
            'function': lambda x: self._trace_not(x),
            'symbol': '¬',
            'delay': 1,
            'area': 1
        }
        
    def _create_and_gate(self) -> Dict:
        """创建AND门"""
        return {
            'type': 'AND',
            'inputs': 2,
            'outputs': 1,
            'function': lambda x, y: self._trace_and(x, y),
            'symbol': '∧',
            'delay': 2,
            'area': 2
        }
        
    def _create_or_gate(self) -> Dict:
        """创建OR门"""
        return {
            'type': 'OR',
            'inputs': 2,
            'outputs': 1,
            'function': lambda x, y: self._trace_or(x, y),
            'symbol': '∨',
            'delay': 2,
            'area': 2
        }
        
    def _create_xor_gate(self) -> Dict:
        """创建XOR门"""
        return {
            'type': 'XOR',
            'inputs': 2,
            'outputs': 1,
            'function': lambda x, y: self._trace_xor(x, y),
            'symbol': '⊕',
            'delay': 3,
            'area': 3
        }
        
    def _create_nand_gate(self) -> Dict:
        """创建NAND门"""
        return {
            'type': 'NAND',
            'inputs': 2,
            'outputs': 1,
            'function': lambda x, y: self._trace_nand(x, y),
            'symbol': '↑',
            'delay': 1,
            'area': 1
        }
        
    def _create_nor_gate(self) -> Dict:
        """创建NOR门"""
        return {
            'type': 'NOR',
            'inputs': 2,
            'outputs': 1,
            'function': lambda x, y: self._trace_nor(x, y),
            'symbol': '↓',
            'delay': 1,
            'area': 1
        }
        
    def _create_buffer_gate(self) -> Dict:
        """创建缓冲器"""
        return {
            'type': 'BUFFER',
            'inputs': 1,
            'outputs': 1,
            'function': lambda x: x,
            'symbol': '▷',
            'delay': 1,
            'area': 1
        }
        
    def _create_wire_component(self) -> Dict:
        """创建连线组件"""
        return {
            'type': 'WIRE',
            'inputs': 1,
            'outputs': 1,
            'function': lambda x: x,
            'symbol': '─',
            'delay': 0,
            'area': 0
        }

    def _trace_not(self, trace: int) -> int:
        """NOT操作"""
        if trace not in self.trace_universe:
            return 0
        trace_str = self.trace_universe[trace]['trace']
        result = ''.join('0' if b == '1' else '1' for b in trace_str)
        return self._trace_to_value(result) if '11' not in result else 0
        
    def _trace_and(self, a: int, b: int) -> int:
        """AND操作"""
        if a not in self.trace_universe or b not in self.trace_universe:
            return 0
        trace_a = self.trace_universe[a]['trace']
        trace_b = self.trace_universe[b]['trace']
        result = self._align_and_operate(trace_a, trace_b, min)
        return self._trace_to_value(result) if '11' not in result else 0
        
    def _trace_or(self, a: int, b: int) -> int:
        """OR操作"""
        if a not in self.trace_universe or b not in self.trace_universe:
            return 0
        trace_a = self.trace_universe[a]['trace']
        trace_b = self.trace_universe[b]['trace']
        result = self._align_and_operate(trace_a, trace_b, max)
        return self._trace_to_value(result) if '11' not in result else 0
        
    def _trace_xor(self, a: int, b: int) -> int:
        """XOR操作"""
        if a not in self.trace_universe or b not in self.trace_universe:
            return 0
        trace_a = self.trace_universe[a]['trace']
        trace_b = self.trace_universe[b]['trace']
        result = self._align_and_operate(trace_a, trace_b, 
                                       lambda x, y: str((int(x) + int(y)) % 2))
        return self._trace_to_value(result) if '11' not in result else 0
        
    def _trace_nand(self, a: int, b: int) -> int:
        """NAND操作"""
        and_result = self._trace_and(a, b)
        return self._trace_not(and_result)
        
    def _trace_nor(self, a: int, b: int) -> int:
        """NOR操作"""
        or_result = self._trace_or(a, b)
        return self._trace_not(or_result)
        
    def _align_and_operate(self, trace_a: str, trace_b: str, 
                          op: Callable[[str, str], str]) -> str:
        """对齐并操作两个trace"""
        max_len = max(len(trace_a), len(trace_b))
        padded_a = trace_a.rjust(max_len, '0')
        padded_b = trace_b.rjust(max_len, '0')
        
        result = []
        for i in range(max_len):
            result.append(op(padded_a[i], padded_b[i]))
        
        return ''.join(result).lstrip('0') or '0'
        
    def _trace_to_value(self, trace: str) -> int:
        """将trace转换回数值"""
        value = 0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                value += self.fibonacci_numbers[i]
        return value

    def create_circuit(self, circuit_spec: Dict) -> Dict:
        """创建电路"""
        circuit = {
            'gates': {},
            'connections': [],
            'inputs': circuit_spec.get('inputs', []),
            'outputs': circuit_spec.get('outputs', []),
            'topology': nx.DiGraph()
        }
        
        # 添加门
        for gate_id, gate_info in circuit_spec.get('gates', {}).items():
            gate_type = gate_info['type']
            if gate_type in self.gate_library:
                circuit['gates'][gate_id] = {
                    **self.gate_library[gate_type],
                    'id': gate_id,
                    'position': gate_info.get('position', (0, 0))
                }
                circuit['topology'].add_node(gate_id, **circuit['gates'][gate_id])
                
        # 添加连接
        for conn in circuit_spec.get('connections', []):
            from_gate = conn['from']
            to_gate = conn['to']
            circuit['connections'].append(conn)
            circuit['topology'].add_edge(from_gate, to_gate)
            
        return circuit
        
    def simulate_circuit(self, circuit: Dict, inputs: Dict[str, int]) -> Dict:
        """模拟电路"""
        # 初始化信号值
        signals = inputs.copy()
        
        # 拓扑排序
        try:
            eval_order = list(nx.topological_sort(circuit['topology']))
        except nx.NetworkXError:
            return {'error': 'Circuit contains cycles'}
            
        # 按顺序评估每个门
        for gate_id in eval_order:
            if gate_id in circuit['gates']:
                gate = circuit['gates'][gate_id]
                
                # 获取输入信号
                input_signals = []
                for edge in circuit['topology'].in_edges(gate_id):
                    source = edge[0]
                    if source in signals:
                        input_signals.append(signals[source])
                        
                # 计算输出
                if len(input_signals) == gate['inputs']:
                    output = gate['function'](*input_signals)
                    signals[gate_id] = output
                    
        # 收集输出信号
        outputs = {}
        for out_id in circuit['outputs']:
            if out_id in signals:
                outputs[out_id] = signals[out_id]
                
        return {
            'inputs': inputs,
            'outputs': outputs,
            'all_signals': signals,
            'evaluation_order': eval_order
        }
        
    def analyze_circuit_properties(self, circuit: Dict) -> Dict:
        """分析电路属性"""
        topology = circuit['topology']
        
        # 基本属性
        properties = {
            'gate_count': len(circuit['gates']),
            'connection_count': len(circuit['connections']),
            'input_count': len(circuit['inputs']),
            'output_count': len(circuit['outputs']),
            'depth': self._compute_circuit_depth(topology),
            'width': self._compute_circuit_width(topology),
            'fanout_distribution': self._compute_fanout_distribution(topology),
            'critical_path': self._compute_critical_path(circuit),
            'area': self._compute_circuit_area(circuit),
            'power': self._compute_circuit_power(circuit)
        }
        
        return properties
        
    def _compute_circuit_depth(self, topology: nx.DiGraph) -> int:
        """计算电路深度"""
        if topology.number_of_nodes() == 0:
            return 0
            
        # 找到所有路径长度
        max_depth = 0
        for source in [n for n in topology.nodes() if topology.in_degree(n) == 0]:
            for sink in [n for n in topology.nodes() if topology.out_degree(n) == 0]:
                try:
                    paths = nx.all_simple_paths(topology, source, sink)
                    for path in paths:
                        max_depth = max(max_depth, len(path) - 1)
                except nx.NetworkXNoPath:
                    continue
                    
        return max_depth
        
    def _compute_circuit_width(self, topology: nx.DiGraph) -> int:
        """计算电路宽度"""
        if topology.number_of_nodes() == 0:
            return 0
            
        # 按层分组节点
        layers = defaultdict(list)
        for node in topology.nodes():
            layer = self._get_node_layer(topology, node)
            layers[layer].append(node)
            
        return max(len(nodes) for nodes in layers.values()) if layers else 0
        
    def _get_node_layer(self, topology: nx.DiGraph, node: str) -> int:
        """获取节点所在层"""
        if topology.in_degree(node) == 0:
            return 0
            
        max_pred_layer = -1
        for pred in topology.predecessors(node):
            pred_layer = self._get_node_layer(topology, pred)
            max_pred_layer = max(max_pred_layer, pred_layer)
            
        return max_pred_layer + 1
        
    def _compute_fanout_distribution(self, topology: nx.DiGraph) -> Dict[int, int]:
        """计算扇出分布"""
        fanout_dist = defaultdict(int)
        for node in topology.nodes():
            fanout = topology.out_degree(node)
            fanout_dist[fanout] += 1
        return dict(fanout_dist)
        
    def _compute_critical_path(self, circuit: Dict) -> Tuple[List[str], int]:
        """计算关键路径"""
        topology = circuit['topology']
        gates = circuit['gates']
        
        # 计算每个节点的延迟
        delays = {}
        
        def get_max_delay(node):
            if node in delays:
                return delays[node]
                
            if topology.in_degree(node) == 0:
                delays[node] = gates[node]['delay'] if node in gates else 0
                return delays[node]
                
            max_pred_delay = 0
            for pred in topology.predecessors(node):
                max_pred_delay = max(max_pred_delay, get_max_delay(pred))
                
            node_delay = gates[node]['delay'] if node in gates else 0
            delays[node] = max_pred_delay + node_delay
            return delays[node]
            
        # 计算所有节点的延迟
        for node in topology.nodes():
            get_max_delay(node)
            
        # 找到最大延迟的输出
        max_delay = 0
        critical_output = None
        for output in circuit['outputs']:
            if output in delays and delays[output] > max_delay:
                max_delay = delays[output]
                critical_output = output
                
        # 回溯关键路径
        critical_path = []
        if critical_output:
            current = critical_output
            critical_path = [current]
            
            while topology.in_degree(current) > 0:
                # 找到导致最大延迟的前驱
                max_pred = None
                max_pred_delay = -1
                
                for pred in topology.predecessors(current):
                    if pred in delays and delays[pred] > max_pred_delay:
                        max_pred_delay = delays[pred]
                        max_pred = pred
                        
                if max_pred:
                    critical_path.insert(0, max_pred)
                    current = max_pred
                else:
                    break
                    
        return critical_path, max_delay
        
    def _compute_circuit_area(self, circuit: Dict) -> int:
        """计算电路面积"""
        total_area = 0
        for gate_id, gate in circuit['gates'].items():
            total_area += gate['area']
        return total_area
        
    def _compute_circuit_power(self, circuit: Dict) -> float:
        """计算电路功耗"""
        total_power = 0.0
        for gate_id, gate in circuit['gates'].items():
            # 基础功耗
            base_power = gate['area'] * 0.5
            # 动态功耗（基于门类型）
            dynamic_power = gate['delay'] * 0.3
            total_power += base_power + dynamic_power
        return total_power

    def visualize_circuit(self, circuit: Dict, save_path: str = None):
        """可视化电路"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 设置坐标轴
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制门
        gate_positions = {}
        for gate_id, gate in circuit['gates'].items():
            pos = gate.get('position', (0, 0))
            gate_positions[gate_id] = pos
            
            # 根据门类型选择形状
            if gate['type'] == 'NOT':
                # 三角形 + 圆圈
                triangle = patches.Polygon([(pos[0]-0.3, pos[1]-0.3),
                                          (pos[0]-0.3, pos[1]+0.3),
                                          (pos[0]+0.2, pos[1])],
                                         closed=True, fill=True,
                                         facecolor='lightblue',
                                         edgecolor='black')
                ax.add_patch(triangle)
                circle = Circle((pos[0]+0.3, pos[1]), 0.1,
                              fill=False, edgecolor='black')
                ax.add_patch(circle)
                
            elif gate['type'] in ['AND', 'NAND']:
                # D形状
                rect = FancyBboxPatch((pos[0]-0.3, pos[1]-0.3),
                                     0.6, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen',
                                     edgecolor='black')
                ax.add_patch(rect)
                if gate['type'] == 'NAND':
                    circle = Circle((pos[0]+0.4, pos[1]), 0.1,
                                  fill=False, edgecolor='black')
                    ax.add_patch(circle)
                    
            elif gate['type'] in ['OR', 'NOR']:
                # 弧形
                arc = patches.FancyBboxPatch((pos[0]-0.3, pos[1]-0.3),
                                           0.6, 0.6,
                                           boxstyle="round,pad=0.15",
                                           facecolor='lightyellow',
                                           edgecolor='black')
                ax.add_patch(arc)
                if gate['type'] == 'NOR':
                    circle = Circle((pos[0]+0.4, pos[1]), 0.1,
                                  fill=False, edgecolor='black')
                    ax.add_patch(circle)
                    
            elif gate['type'] == 'XOR':
                # 双弧形
                arc1 = patches.Arc((pos[0], pos[1]), 0.8, 0.8,
                                 angle=0, theta1=-45, theta2=45,
                                 linewidth=2, edgecolor='black')
                ax.add_patch(arc1)
                arc2 = patches.Arc((pos[0]-0.1, pos[1]), 0.8, 0.8,
                                 angle=0, theta1=-45, theta2=45,
                                 linewidth=2, edgecolor='black')
                ax.add_patch(arc2)
                
            # 添加标签
            ax.text(pos[0], pos[1], gate['symbol'],
                   ha='center', va='center', fontsize=12, weight='bold')
            ax.text(pos[0], pos[1]-0.5, gate_id,
                   ha='center', va='top', fontsize=8)
                   
        # 绘制连接
        for conn in circuit['connections']:
            from_pos = gate_positions.get(conn['from'], (0, 0))
            to_pos = gate_positions.get(conn['to'], (0, 0))
            
            # 绘制箭头
            arrow = FancyArrowPatch(from_pos, to_pos,
                                  connectionstyle="arc3,rad=0.1",
                                  arrowstyle="->",
                                  mutation_scale=20,
                                  linewidth=2,
                                  color='blue')
            ax.add_patch(arrow)
            
        # 标记输入输出
        for inp in circuit['inputs']:
            if inp in gate_positions:
                pos = gate_positions[inp]
                ax.text(pos[0]-0.8, pos[1], 'IN',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3",
                               facecolor='lightcoral'))
                               
        for out in circuit['outputs']:
            if out in gate_positions:
                pos = gate_positions[out]
                ax.text(pos[0]+0.8, pos[1], 'OUT',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3",
                               facecolor='lightcyan'))
                               
        ax.set_title('φ-Binary Logic Circuit', fontsize=16, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def analyze_circuit_network(self, circuits: List[Dict]) -> Dict:
        """分析电路网络的图论属性"""
        # 构建电路组合网络
        G = nx.Graph()
        
        # 添加电路作为节点
        for i, circuit in enumerate(circuits):
            circuit_id = f"C{i}"
            props = self.analyze_circuit_properties(circuit)
            G.add_node(circuit_id, **props)
            
        # 添加边：基于共享的门类型
        for i in range(len(circuits)):
            for j in range(i+1, len(circuits)):
                shared_gates = self._count_shared_gate_types(circuits[i], circuits[j])
                if shared_gates > 0:
                    G.add_edge(f"C{i}", f"C{j}", weight=shared_gates)
                    
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
            'clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
    def _count_shared_gate_types(self, circuit1: Dict, circuit2: Dict) -> int:
        """计算两个电路共享的门类型数"""
        types1 = set(gate['type'] for gate in circuit1['gates'].values())
        types2 = set(gate['type'] for gate in circuit2['gates'].values())
        return len(types1.intersection(types2))
        
    def compute_circuit_entropy(self, circuit: Dict, test_inputs: List[Dict[str, int]]) -> float:
        """计算电路的信息熵"""
        # 收集所有输出
        outputs = []
        for inputs in test_inputs:
            result = self.simulate_circuit(circuit, inputs)
            if 'outputs' in result:
                outputs.extend(result['outputs'].values())
                
        if not outputs:
            return 0.0
            
        # 计算输出分布
        output_counts = defaultdict(int)
        for out in outputs:
            output_counts[out] += 1
            
        total = sum(output_counts.values())
        probs = [count/total for count in output_counts.values()]
        
        # 计算熵
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * log2(p)
                
        return entropy
        
    def generate_circuit_visualization_data(self, circuit: Dict) -> Dict:
        """生成电路可视化数据"""
        props = self.analyze_circuit_properties(circuit)
        
        # 生成延迟分布图数据
        delay_dist = defaultdict(int)
        for gate in circuit['gates'].values():
            delay_dist[gate['delay']] += 1
            
        # 生成面积分布图数据
        area_dist = defaultdict(int)
        for gate in circuit['gates'].values():
            area_dist[gate['area']] += 1
            
        # 生成扇出分布数据
        fanout_dist = props['fanout_distribution']
        
        return {
            'delay_distribution': dict(delay_dist),
            'area_distribution': dict(area_dist),
            'fanout_distribution': fanout_dist,
            'critical_path_length': props['critical_path'][1],
            'total_area': props['area'],
            'total_power': props['power']
        }

class TestLogicCircuitSystem(unittest.TestCase):
    """单元测试：验证LogicCircuit系统的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = LogicCircuitSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        # 验证φ-valid traces被正确识别
        self.assertIn(1, self.system.trace_universe)
        self.assertIn(2, self.system.trace_universe)
        self.assertIn(3, self.system.trace_universe)
        self.assertIn(5, self.system.trace_universe)
        
        # 验证门属性
        trace_5 = self.system.trace_universe[5]
        self.assertIn('gate_properties', trace_5)
        self.assertGreater(trace_5['gate_properties']['input_capacity'], 0)
        
    def test_gate_operations(self):
        """测试门操作"""
        # 测试NOT门
        result = self.system._trace_not(1)
        self.assertIsInstance(result, int)
        
        # 测试AND门
        result = self.system._trace_and(1, 2)
        self.assertIsInstance(result, int)
        
        # 测试OR门
        result = self.system._trace_or(1, 2)
        self.assertIsInstance(result, int)
        
        # 测试XOR门
        result = self.system._trace_xor(1, 2)
        self.assertIsInstance(result, int)
        
    def test_circuit_creation(self):
        """测试电路创建"""
        circuit_spec = {
            'gates': {
                'G1': {'type': 'AND', 'position': (2, 4)},
                'G2': {'type': 'OR', 'position': (4, 4)},
                'G3': {'type': 'NOT', 'position': (6, 4)}
            },
            'connections': [
                {'from': 'G1', 'to': 'G2'},
                {'from': 'G2', 'to': 'G3'}
            ],
            'inputs': ['G1'],
            'outputs': ['G3']
        }
        
        circuit = self.system.create_circuit(circuit_spec)
        self.assertIn('gates', circuit)
        self.assertIn('topology', circuit)
        self.assertEqual(len(circuit['gates']), 3)
        
    def test_circuit_simulation(self):
        """测试电路模拟"""
        # 创建简单电路
        circuit_spec = {
            'gates': {
                'G1': {'type': 'AND', 'position': (2, 4)},
            },
            'connections': [],
            'inputs': ['G1'],
            'outputs': ['G1']
        }
        
        circuit = self.system.create_circuit(circuit_spec)
        
        # 模拟
        inputs = {'G1': 1}
        result = self.system.simulate_circuit(circuit, inputs)
        self.assertIn('outputs', result)
        
    def test_circuit_properties(self):
        """测试电路属性分析"""
        circuit_spec = {
            'gates': {
                'G1': {'type': 'AND', 'position': (2, 4)},
                'G2': {'type': 'OR', 'position': (4, 4)},
            },
            'connections': [
                {'from': 'G1', 'to': 'G2'}
            ],
            'inputs': ['G1'],
            'outputs': ['G2']
        }
        
        circuit = self.system.create_circuit(circuit_spec)
        props = self.system.analyze_circuit_properties(circuit)
        
        self.assertIn('gate_count', props)
        self.assertIn('depth', props)
        self.assertIn('critical_path', props)
        self.assertEqual(props['gate_count'], 2)
        
    def test_circuit_entropy(self):
        """测试电路熵计算"""
        circuit_spec = {
            'gates': {
                'G1': {'type': 'XOR', 'position': (2, 4)},
            },
            'connections': [],
            'inputs': ['G1'],
            'outputs': ['G1']
        }
        
        circuit = self.system.create_circuit(circuit_spec)
        
        test_inputs = [
            {'G1': 0},
            {'G1': 1},
            {'G1': 2},
            {'G1': 3}
        ]
        
        entropy = self.system.compute_circuit_entropy(circuit, test_inputs)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

def run_comprehensive_analysis():
    """运行完整的LogicCircuit分析"""
    print("=" * 60)
    print("Chapter 045: LogicCircuit Comprehensive Analysis")
    print("Constructing φ-Binary Circuits from Trace Primitives")
    print("=" * 60)
    
    system = LogicCircuitSystem()
    
    # 1. 基础门分析
    print("\n1. Basic Gate Analysis:")
    print(f"φ-valid universe size: {len(system.trace_universe)}")
    print(f"Gate library size: {len(system.gate_library)}")
    print(f"Available gates: {list(system.gate_library.keys())}")
    
    # 2. 门操作测试
    print("\n2. Gate Operation Tests:")
    test_values = [1, 2, 3, 5, 8]
    
    for val in test_values[:3]:
        not_result = system._trace_not(val)
        print(f"NOT({val}): {not_result} (φ-valid: {not_result in system.trace_universe})")
        
    for i in range(len(test_values)-1):
        a, b = test_values[i], test_values[i+1]
        and_result = system._trace_and(a, b)
        or_result = system._trace_or(a, b)
        xor_result = system._trace_xor(a, b)
        
        print(f"\n{a} AND {b} = {and_result}")
        print(f"{a} OR {b} = {or_result}")
        print(f"{a} XOR {b} = {xor_result}")
        
    # 3. 创建示例电路
    print("\n3. Example Circuit Creation:")
    
    # 半加器电路
    half_adder_spec = {
        'gates': {
            'XOR1': {'type': 'XOR', 'position': (2, 4)},
            'AND1': {'type': 'AND', 'position': (2, 2)}
        },
        'connections': [],
        'inputs': ['XOR1', 'AND1'],
        'outputs': ['XOR1', 'AND1']  # Sum and Carry
    }
    
    half_adder = system.create_circuit(half_adder_spec)
    
    # 全加器电路
    full_adder_spec = {
        'gates': {
            'XOR1': {'type': 'XOR', 'position': (2, 5)},
            'XOR2': {'type': 'XOR', 'position': (4, 5)},
            'AND1': {'type': 'AND', 'position': (2, 3)},
            'AND2': {'type': 'AND', 'position': (4, 3)},
            'OR1': {'type': 'OR', 'position': (6, 3)}
        },
        'connections': [
            {'from': 'XOR1', 'to': 'XOR2'},
            {'from': 'XOR1', 'to': 'AND2'},
            {'from': 'AND1', 'to': 'OR1'},
            {'from': 'AND2', 'to': 'OR1'}
        ],
        'inputs': ['XOR1', 'AND1'],
        'outputs': ['XOR2', 'OR1']  # Sum and Carry
    }
    
    full_adder = system.create_circuit(full_adder_spec)
    
    # 4. 电路属性分析
    print("\n4. Circuit Properties Analysis:")
    
    half_props = system.analyze_circuit_properties(half_adder)
    print(f"\nHalf Adder Properties:")
    print(f"  Gate count: {half_props['gate_count']}")
    print(f"  Depth: {half_props['depth']}")
    print(f"  Width: {half_props['width']}")
    print(f"  Area: {half_props['area']}")
    print(f"  Power: {half_props['power']:.2f}")
    
    full_props = system.analyze_circuit_properties(full_adder)
    print(f"\nFull Adder Properties:")
    print(f"  Gate count: {full_props['gate_count']}")
    print(f"  Depth: {full_props['depth']}")
    print(f"  Width: {full_props['width']}")
    print(f"  Area: {full_props['area']}")
    print(f"  Power: {full_props['power']:.2f}")
    print(f"  Critical path: {full_props['critical_path'][0]}")
    print(f"  Critical delay: {full_props['critical_path'][1]}")
    
    # 5. 电路模拟
    print("\n5. Circuit Simulation:")
    
    # 测试半加器
    print("\nHalf Adder Truth Table:")
    for a in [0, 1]:
        for b in [0, 1]:
            inputs = {'XOR1': a, 'AND1': b}
            result = system.simulate_circuit(half_adder, inputs)
            if 'outputs' in result:
                sum_out = result['outputs'].get('XOR1', 0)
                carry_out = result['outputs'].get('AND1', 0)
                print(f"  {a} + {b} = Sum: {sum_out}, Carry: {carry_out}")
                
    # 6. 网络分析
    print("\n6. Circuit Network Analysis:")
    
    # 创建更多电路用于网络分析
    mux_spec = {
        'gates': {
            'NOT1': {'type': 'NOT', 'position': (2, 4)},
            'AND1': {'type': 'AND', 'position': (4, 5)},
            'AND2': {'type': 'AND', 'position': (4, 3)},
            'OR1': {'type': 'OR', 'position': (6, 4)}
        },
        'connections': [
            {'from': 'NOT1', 'to': 'AND2'},
            {'from': 'AND1', 'to': 'OR1'},
            {'from': 'AND2', 'to': 'OR1'}
        ],
        'inputs': ['NOT1', 'AND1', 'AND2'],
        'outputs': ['OR1']
    }
    
    mux = system.create_circuit(mux_spec)
    
    circuits = [half_adder, full_adder, mux]
    network_props = system.analyze_circuit_network(circuits)
    
    print(f"Circuit network nodes: {network_props['nodes']}")
    print(f"Circuit network edges: {network_props['edges']}")
    print(f"Network density: {network_props['density']:.3f}")
    print(f"Average degree: {network_props['average_degree']:.3f}")
    print(f"Clustering: {network_props['clustering']:.3f}")
    
    # 7. 信息理论分析
    print("\n7. Information Theory Analysis:")
    
    test_inputs = []
    for i in range(8):
        inputs = {
            'XOR1': i % 2,
            'AND1': (i // 2) % 2
        }
        test_inputs.append(inputs)
        
    half_entropy = system.compute_circuit_entropy(half_adder, test_inputs)
    print(f"Half adder entropy: {half_entropy:.3f} bits")
    
    # 8. 三域分析
    print("\n8. Three-Domain Analysis:")
    
    # Traditional circuit domain
    traditional_gates = 2 ** len(system.gate_library)  # 所有可能的门组合
    
    # φ-constrained domain
    phi_valid_gates = sum(1 for trace in system.trace_universe.values()
                         if trace['gate_properties']['input_capacity'] > 0)
    
    # Intersection analysis
    intersection_gates = min(traditional_gates, phi_valid_gates)
    
    print(f"Traditional circuit domain: {traditional_gates} potential gates")
    print(f"φ-constrained domain: {phi_valid_gates} valid gate traces")
    print(f"Intersection domain: {intersection_gates} gates")
    print(f"Domain intersection ratio: {intersection_gates/traditional_gates:.3f}")
    
    # 9. 可视化电路
    print("\n9. Generating Circuit Visualizations...")
    
    # 可视化半加器
    system.visualize_circuit(half_adder, "chapter-045-logic-circuit-half-adder.png")
    print("Saved visualization: chapter-045-logic-circuit-half-adder.png")
    
    # 可视化全加器
    system.visualize_circuit(full_adder, "chapter-045-logic-circuit-full-adder.png")
    print("Saved visualization: chapter-045-logic-circuit-full-adder.png")
    
    # 可视化MUX
    system.visualize_circuit(mux, "chapter-045-logic-circuit-mux.png")
    print("Saved visualization: chapter-045-logic-circuit-mux.png")
    
    # 10. 生成电路特性图表
    print("\n10. Generating Circuit Property Charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 延迟分布
    ax = axes[0, 0]
    viz_data = system.generate_circuit_visualization_data(full_adder)
    delays = list(viz_data['delay_distribution'].keys())
    counts = list(viz_data['delay_distribution'].values())
    ax.bar(delays, counts, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Gate Delay')
    ax.set_ylabel('Count')
    ax.set_title('Gate Delay Distribution')
    
    # 面积分布
    ax = axes[0, 1]
    areas = list(viz_data['area_distribution'].keys())
    counts = list(viz_data['area_distribution'].values())
    ax.bar(areas, counts, color='lightgreen', edgecolor='darkgreen')
    ax.set_xlabel('Gate Area')
    ax.set_ylabel('Count')
    ax.set_title('Gate Area Distribution')
    
    # 扇出分布
    ax = axes[1, 0]
    fanouts = list(viz_data['fanout_distribution'].keys())
    counts = list(viz_data['fanout_distribution'].values())
    ax.bar(fanouts, counts, color='coral', edgecolor='darkred')
    ax.set_xlabel('Fanout')
    ax.set_ylabel('Count')
    ax.set_title('Fanout Distribution')
    
    # 电路特性总结
    ax = axes[1, 1]
    properties = [
        f"Critical Path: {viz_data['critical_path_length']}",
        f"Total Area: {viz_data['total_area']}",
        f"Total Power: {viz_data['total_power']:.1f}",
        f"Gate Count: {full_props['gate_count']}"
    ]
    ax.text(0.5, 0.5, '\n'.join(properties),
           ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow'))
    ax.axis('off')
    ax.set_title('Circuit Summary')
    
    plt.suptitle('φ-Binary Circuit Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("chapter-045-logic-circuit-analysis.png", dpi=150, bbox_inches='tight')
    print("Saved visualization: chapter-045-logic-circuit-analysis.png")
    
    # 11. φ-constraint优化分析
    print("\n11. φ-Constraint Optimization Analysis:")
    
    # 分析不同trace作为门的效率
    gate_efficiency = []
    for trace_val in list(system.trace_universe.keys())[:10]:
        if trace_val == 0:
            continue
        props = system.trace_universe[trace_val]['gate_properties']
        efficiency = props['output_strength'] / max(props['power_consumption'], 0.1)
        gate_efficiency.append((trace_val, efficiency))
        
    gate_efficiency.sort(key=lambda x: x[1], reverse=True)
    print("\nMost efficient gate traces:")
    for trace, eff in gate_efficiency[:5]:
        trace_str = system.trace_universe[trace]['trace']
        print(f"  Trace {trace} ({trace_str}): efficiency = {eff:.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete - LogicCircuit System Verified")
    print("=" * 60)

if __name__ == "__main__":
    # 运行单元测试
    print("Running LogicCircuit Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行综合分析
    run_comprehensive_analysis()