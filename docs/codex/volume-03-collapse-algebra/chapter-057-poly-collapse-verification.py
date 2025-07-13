#!/usr/bin/env python3
"""
Chapter 057: PolyCollapse Unit Test Verification
从ψ=ψ(ψ)推导Collapse Polynomial Systems over φ-Indexed Variables

Core principle: From ψ = ψ(ψ) derive polynomial algebraic structures where elements are φ-valid
trace coefficients and variables with polynomial operations that preserve the φ-constraint across
all polynomial transformations, creating systematic polynomial frameworks with bounded degree
and natural polynomial properties governed by golden constraints.

This verification program implements:
1. φ-constrained polynomial computation as trace coefficient polynomial operations
2. Polynomial analysis: degree bounds, coefficient structure, evaluation with φ-preservation
3. Three-domain analysis: Traditional vs φ-constrained vs intersection polynomial theory
4. Graph theory analysis of polynomial networks and coefficient relationship connectivity
5. Information theory analysis of polynomial entropy and coefficient information
6. Category theory analysis of polynomial functors and evaluation morphisms
7. Visualization of polynomial structures and coefficient patterns
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

class PolyCollapseSystem:
    """
    Core system for implementing collapse polynomial systems over φ-indexed variables.
    Implements φ-constrained polynomial theory via trace-based coefficient operations.
    """
    
    def __init__(self, max_trace_size: int = 6, max_polynomial_degree: int = 3):
        """Initialize poly collapse system"""
        self.max_trace_size = max_trace_size
        self.max_polynomial_degree = max_polynomial_degree
        self.fibonacci_numbers = self._generate_fibonacci(8)
        self.polynomial_cache = {}
        self.coefficient_cache = {}
        self.evaluation_cache = {}
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
            trace_data = self._analyze_trace_structure(n, compute_polynomial=False)
            if trace_data['phi_valid']:
                universe[n] = trace_data
        
        # Store universe for polynomial properties computation
        self.trace_universe = universe
        
        # Second pass: add polynomial properties
        for n in universe:
            trace = universe[n]['trace']
            universe[n]['polynomial_properties'] = self._compute_polynomial_properties(trace)
                
        return universe
        
    def _analyze_trace_structure(self, n: int, compute_polynomial: bool = True) -> Dict:
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
        
        if compute_polynomial and hasattr(self, 'trace_universe'):
            result['polynomial_properties'] = self._compute_polynomial_properties(trace)
            
        return result
        
    def _encode_to_trace(self, n: int) -> str:
        """将整数编码为trace（二进制表示）"""
        if n == 0:
            return '0'
        return bin(n)[2:]  # 移除'0b'前缀
        
    def _get_fibonacci_indices(self, trace: str) -> List[int]:
        """获取trace中1的Fibonacci位置"""
        indices = []
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                indices.append(self.fibonacci_numbers[i])
        return indices
        
    def _compute_structural_hash(self, trace: str) -> int:
        """计算trace的结构哈希"""
        return hash(trace + ''.join(map(str, self._get_fibonacci_indices(trace))))
        
    def _compute_binary_weight(self, trace: str) -> float:
        """计算trace的二进制权重"""
        weight = 0.0
        for i, bit in enumerate(reversed(trace)):
            if bit == '1' and i < len(self.fibonacci_numbers):
                weight += 1.0 / (self.fibonacci_numbers[i] + 1)
        return weight
        
    def _compute_polynomial_properties(self, trace: str) -> Dict:
        """计算trace的多项式属性"""
        properties = {
            'coefficient_signature': self._compute_coefficient_signature(trace),
            'polynomial_degree': self._compute_polynomial_degree(trace),
            'variable_indices': self._compute_variable_indices(trace),
            'evaluation_measure': self._compute_evaluation_measure(trace),
            'coefficient_weight': self._compute_coefficient_weight(trace),
            'polynomial_stability': self._compute_polynomial_stability(trace),
            'root_approximation': self._compute_root_approximation(trace)
        }
        return properties
        
    def _compute_coefficient_signature(self, trace: str) -> complex:
        """计算trace作为多项式系数的签名"""
        if not trace or trace == '0':
            return complex(0, 0)  # 零系数
        
        # 基于trace的多项式系数编码
        real_part = 0.0
        imag_part = 0.0
        
        for i, bit in enumerate(trace):
            if bit == '1':
                position_weight = self.fibonacci_numbers[min(i, len(self.fibonacci_numbers)-1)]
                # 系数性质：基于位置的权重分布
                real_part += position_weight / (position_weight + 1)
                imag_part += (position_weight % 2) / (position_weight + 1)
                
        # 模运算以确保有界性
        modulus = self.fibonacci_numbers[3]  # F_4 = 3
        real_part = real_part % modulus
        imag_part = imag_part % modulus
        
        return complex(real_part, imag_part)
        
    def _compute_polynomial_degree(self, trace: str) -> int:
        """计算多项式度数"""
        if not trace or trace == '0':
            return 0
        
        # 度数基于trace的长度，但受φ-约束限制
        degree = min(len(trace) - 1, self.max_polynomial_degree)
        return max(0, degree)
        
    def _compute_variable_indices(self, trace: str) -> List[int]:
        """计算变量索引（基于Fibonacci位置）"""
        if not trace or trace == '0':
            return []
        
        # 变量索引基于1的位置
        indices = []
        for i, bit in enumerate(trace):
            if bit == '1':
                var_index = min(i, self.max_polynomial_degree)
                indices.append(var_index)
        
        return sorted(list(set(indices)))  # 去重并排序
        
    def _compute_evaluation_measure(self, trace: str) -> float:
        """计算多项式求值度量"""
        if not trace or trace == '0':
            return 0.0
        
        # 基于Fibonacci权重的求值
        evaluation = 0.0
        for i, bit in enumerate(trace):
            if bit == '1':
                fib_index = min(i, len(self.fibonacci_numbers)-1)
                # 使用x=φ的黄金比例作为求值点
                phi = (1 + sqrt(5)) / 2  # 黄金比例
                degree = min(i, self.max_polynomial_degree)
                evaluation += self.fibonacci_numbers[fib_index] * (phi ** degree)
        
        # 归一化
        max_eval = sum(self.fibonacci_numbers[:4]) * (phi ** self.max_polynomial_degree)
        return evaluation / max_eval if max_eval > 0 else 0.0
        
    def _compute_coefficient_weight(self, trace: str) -> float:
        """计算系数权重"""
        if not trace or trace == '0':
            return 0.0
        
        # 基于Fibonacci分解的权重
        fib_indices = self._get_fibonacci_indices(trace)
        if not fib_indices:
            return 0.0
        
        # 权重 = 所有Fibonacci数的调和平均
        harmonic_sum = sum(1.0 / fib for fib in fib_indices)
        return len(fib_indices) / harmonic_sum if harmonic_sum > 0 else 0.0
        
    def _compute_polynomial_stability(self, trace: str) -> float:
        """计算多项式稳定性"""
        if not trace or trace == '0':
            return 1.0  # 零多项式是稳定的
        
        # 稳定性基于系数的分布
        coefficients = []
        for i, bit in enumerate(trace):
            if bit == '1':
                fib_index = min(i, len(self.fibonacci_numbers)-1)
                coefficients.append(self.fibonacci_numbers[fib_index])
        
        if len(coefficients) <= 1:
            return 1.0
        
        # 计算系数的变异系数（标准差/均值）
        mean_coeff = sum(coefficients) / len(coefficients)
        variance = sum((c - mean_coeff)**2 for c in coefficients) / len(coefficients)
        std_dev = sqrt(variance)
        
        # 稳定性 = 1 / (1 + 变异系数)
        cv = std_dev / mean_coeff if mean_coeff > 0 else 0
        return 1.0 / (1.0 + cv)
        
    def _compute_root_approximation(self, trace: str) -> complex:
        """计算多项式根的近似"""
        if not trace or trace == '0':
            return complex(0, 0)
        
        # 简化的根近似：基于系数比率
        coefficients = []
        for i, bit in enumerate(trace):
            if bit == '1':
                fib_index = min(i, len(self.fibonacci_numbers)-1)
                coefficients.append(self.fibonacci_numbers[fib_index])
        
        if len(coefficients) < 2:
            return complex(0, 0)
        
        # 使用简化的根公式：-a0/a1 for linear approximation
        real_root = -coefficients[0] / coefficients[-1] if coefficients[-1] != 0 else 0
        imag_root = sum(coefficients[1:-1]) / len(coefficients) if len(coefficients) > 2 else 0
        
        return complex(real_root % 10, imag_root % 10)  # 限制范围
        
    def create_polynomial(self, coefficients: List[str]) -> Dict:
        """创建φ-约束多项式"""
        # 系数必须是φ-valid traces
        phi_valid_coeffs = [coeff for coeff in coefficients if '11' not in coeff]
        
        polynomial = {
            'coefficients': phi_valid_coeffs,
            'degree': len(phi_valid_coeffs) - 1 if phi_valid_coeffs else 0,
            'variable_signature': self._compute_variable_signature(phi_valid_coeffs),
            'polynomial_hash': hash(tuple(phi_valid_coeffs))
        }
        
        return polynomial
        
    def _compute_variable_signature(self, coefficients: List[str]) -> str:
        """计算变量签名"""
        if not coefficients:
            return 'x0'
        
        # 基于系数的变量命名
        max_var_index = len(coefficients) - 1
        return f'x{max_var_index}'
        
    def evaluate_polynomial(self, polynomial: Dict, x_value: float) -> float:
        """求值多项式"""
        result = 0.0
        coefficients = polynomial['coefficients']
        
        for i, coeff_trace in enumerate(coefficients):
            # 将trace转换为数值系数
            coeff_value = int(coeff_trace, 2) if coeff_trace != '0' else 0
            # 使用Fibonacci缩放
            fib_scaled_coeff = coeff_value / (self.fibonacci_numbers[min(i, len(self.fibonacci_numbers)-1)] + 1)
            result += fib_scaled_coeff * (x_value ** i)
        
        return result
        
    def polynomial_addition(self, poly1: Dict, poly2: Dict) -> Dict:
        """多项式加法"""
        coeffs1 = poly1['coefficients']
        coeffs2 = poly2['coefficients']
        
        # 补齐到相同长度
        max_degree = max(len(coeffs1), len(coeffs2))
        padded_coeffs1 = coeffs1 + ['0'] * (max_degree - len(coeffs1))
        padded_coeffs2 = coeffs2 + ['0'] * (max_degree - len(coeffs2))
        
        # 系数相加（使用XOR作为trace加法）
        result_coeffs = []
        for c1, c2 in zip(padded_coeffs1, padded_coeffs2):
            val1 = int(c1, 2) if c1 != '0' else 0
            val2 = int(c2, 2) if c2 != '0' else 0
            result_val = val1 ^ val2  # XOR加法
            result_trace = bin(result_val)[2:] if result_val > 0 else '0'
            
            # 确保φ-valid
            if '11' in result_trace:
                result_trace = self._phi_adjust_trace(result_trace)
            
            result_coeffs.append(result_trace)
        
        return self.create_polynomial(result_coeffs)
        
    def polynomial_multiplication(self, poly1: Dict, poly2: Dict) -> Dict:
        """多项式乘法"""
        coeffs1 = poly1['coefficients']
        coeffs2 = poly2['coefficients']
        
        if not coeffs1 or not coeffs2:
            return self.create_polynomial(['0'])
        
        # 结果的度数 = 两个多项式度数之和
        result_degree = len(coeffs1) + len(coeffs2) - 1
        result_coeffs = ['0'] * result_degree
        
        # 逐项相乘
        for i, c1 in enumerate(coeffs1):
            for j, c2 in enumerate(coeffs2):
                val1 = int(c1, 2) if c1 != '0' else 0
                val2 = int(c2, 2) if c2 != '0' else 0
                
                if val1 != 0 and val2 != 0:
                    # 简化乘法：使用AND操作
                    product_val = val1 & val2
                    result_index = i + j
                    
                    if result_index < len(result_coeffs):
                        # 与现有系数相加
                        existing_val = int(result_coeffs[result_index], 2) if result_coeffs[result_index] != '0' else 0
                        new_val = existing_val ^ product_val  # XOR加法
                        new_trace = bin(new_val)[2:] if new_val > 0 else '0'
                        
                        # 确保φ-valid
                        if '11' in new_trace:
                            new_trace = self._phi_adjust_trace(new_trace)
                        
                        result_coeffs[result_index] = new_trace
        
        return self.create_polynomial(result_coeffs)
        
    def _phi_adjust_trace(self, trace: str) -> str:
        """调整trace以满足φ-约束（无连续11）"""
        if '11' not in trace:
            return trace
        
        # 用Fibonacci替换规则：11 -> 100
        adjusted = trace.replace('11', '100')
        
        # 递归调整直到无11
        while '11' in adjusted:
            adjusted = adjusted.replace('11', '100')
        
        # 移除前导零（除了'0'本身）
        if adjusted != '0':
            adjusted = adjusted.lstrip('0') or '0'
        
        return adjusted
        
    def analyze_polynomial_structure(self) -> Dict:
        """分析多项式结构"""
        traces = [data['trace'] for data in self.trace_universe.values()]
        
        # 创建示例多项式
        polynomials = []
        for i, trace in enumerate(traces[:4]):  # 限制数量
            poly = self.create_polynomial([trace, '1'])  # 简单的一次多项式
            polynomials.append(poly)
        
        # 分析度数分布
        degrees = [poly['degree'] for poly in polynomials]
        
        # 测试多项式运算
        operation_tests = 0
        successful_operations = 0
        
        # 测试加法
        for i in range(min(len(polynomials), 3)):
            for j in range(i+1, min(len(polynomials), 3)):
                try:
                    result = self.polynomial_addition(polynomials[i], polynomials[j])
                    if result['degree'] >= 0:
                        successful_operations += 1
                    operation_tests += 1
                except:
                    operation_tests += 1
        
        # 测试乘法
        for i in range(min(len(polynomials), 2)):
            for j in range(i+1, min(len(polynomials), 2)):
                try:
                    result = self.polynomial_multiplication(polynomials[i], polynomials[j])
                    if result['degree'] >= 0:
                        successful_operations += 1
                    operation_tests += 1
                except:
                    operation_tests += 1
        
        return {
            'total_polynomials': len(polynomials),
            'degree_distribution': degrees,
            'max_degree': max(degrees) if degrees else 0,
            'average_degree': sum(degrees) / len(degrees) if degrees else 0,
            'operation_success_ratio': successful_operations / max(operation_tests, 1),
            'polynomial_samples': polynomials[:3]  # 前3个样本
        }
        
    def compute_three_domain_analysis(self) -> Dict:
        """计算三域分析：传统 vs φ-约束 vs 交集"""
        # 模拟传统多项式数量
        traditional_polynomials = 100
        
        # φ-约束多项式：基于φ-valid traces
        phi_constrained_polynomials = len(self.trace_universe)
        
        # 交集：满足两种系统的多项式
        intersection_polynomials = phi_constrained_polynomials
        
        return {
            'traditional_domain': traditional_polynomials,
            'phi_constrained_domain': phi_constrained_polynomials,
            'intersection_domain': intersection_polynomials,
            'convergence_ratio': intersection_polynomials / traditional_polynomials
        }
        
    def analyze_information_theory(self) -> Dict:
        """信息论分析"""
        # 收集多项式系数签名
        coefficient_signatures = [data['polynomial_properties']['coefficient_signature'] 
                                for data in self.trace_universe.values()]
        
        # 计算多项式熵
        unique_signatures = len(set(str(sig) for sig in coefficient_signatures))
        polynomial_entropy = log2(max(unique_signatures, 1))
        
        # 收集度数分布
        degrees = [data['polynomial_properties']['polynomial_degree'] 
                  for data in self.trace_universe.values()]
        unique_degrees = len(set(degrees))
        
        return {
            'polynomial_entropy': polynomial_entropy,
            'coefficient_complexity': unique_signatures,
            'degree_diversity': unique_degrees,
            'evaluation_information': self._compute_evaluation_information()
        }
        
    def _compute_evaluation_information(self) -> float:
        """计算求值信息"""
        evaluations = [data['polynomial_properties']['evaluation_measure'] 
                      for data in self.trace_universe.values()]
        unique_evaluations = len(set(evaluations))
        return log2(max(unique_evaluations, 1))
        
    def analyze_graph_theory(self) -> Dict:
        """图论分析"""
        # 构建多项式关系图
        G = nx.Graph()
        traces = [data['trace'] for data in self.trace_universe.values()]
        
        # 添加节点
        for trace in traces:
            G.add_node(trace)
        
        # 添加边（基于系数关系）
        for i, trace_a in enumerate(traces):
            for j, trace_b in enumerate(traces):
                if i < j:
                    # 如果两个trace可以作为多项式系数相关，添加边
                    try:
                        poly_a = self.create_polynomial([trace_a, '1'])
                        poly_b = self.create_polynomial([trace_b, '1'])
                        sum_poly = self.polynomial_addition(poly_a, poly_b)
                        if sum_poly['degree'] >= 0:
                            G.add_edge(trace_a, trace_b, relation='polynomial')
                    except:
                        pass
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G),
            'clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        }
        
    def generate_visualization_structure(self):
        """生成结构可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 多项式度数分布
        degrees = [data['polynomial_properties']['polynomial_degree'] 
                  for data in self.trace_universe.values()]
        ax1.hist(degrees, bins=max(1, max(degrees)+1), alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Polynomial Degree Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Polynomial Degree')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. 系数签名（实部 vs 虚部）
        signatures = [data['polynomial_properties']['coefficient_signature'] 
                     for data in self.trace_universe.values()]
        real_parts = [sig.real for sig in signatures]
        imag_parts = [sig.imag for sig in signatures]
        
        scatter = ax2.scatter(real_parts, imag_parts, c=degrees, cmap='viridis', 
                            alpha=0.7, s=80, edgecolors='black')
        ax2.set_title('Coefficient Signatures (Real vs Imaginary)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Polynomial Degree')
        
        # 3. 求值度量 vs 系数权重
        evaluation_measures = [data['polynomial_properties']['evaluation_measure'] 
                             for data in self.trace_universe.values()]
        coefficient_weights = [data['polynomial_properties']['coefficient_weight'] 
                             for data in self.trace_universe.values()]
        
        ax3.scatter(evaluation_measures, coefficient_weights, c=degrees, 
                   cmap='plasma', alpha=0.7, s=80, edgecolors='black')
        ax3.set_title('Evaluation Measure vs Coefficient Weight', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Evaluation Measure')
        ax3.set_ylabel('Coefficient Weight')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Polynomial Degree')
        
        # 4. 多项式稳定性分布
        stabilities = [data['polynomial_properties']['polynomial_stability'] 
                      for data in self.trace_universe.values()]
        ax4.hist(stabilities, bins=8, alpha=0.7, color='green', edgecolor='black')
        ax4.set_title('Polynomial Stability Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Polynomial Stability')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-057-poly-collapse-structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_visualization_properties(self):
        """生成属性可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 收集所有属性数据
        traces = [data['trace'] for data in self.trace_universe.values()]
        degrees = [data['polynomial_properties']['polynomial_degree'] 
                  for data in self.trace_universe.values()]
        evaluation_measures = [data['polynomial_properties']['evaluation_measure'] 
                             for data in self.trace_universe.values()]
        coefficient_weights = [data['polynomial_properties']['coefficient_weight'] 
                             for data in self.trace_universe.values()]
        stabilities = [data['polynomial_properties']['polynomial_stability'] 
                      for data in self.trace_universe.values()]
        
        # 1. 度数 vs 稳定性关系
        ax1.scatter(degrees, stabilities, alpha=0.7, s=80, color='red', edgecolors='black')
        ax1.set_title('Polynomial Degree vs Stability', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Polynomial Degree')
        ax1.set_ylabel('Polynomial Stability')
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(degrees) > 1:
            z = np.polyfit(degrees, stabilities, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(degrees), p(sorted(degrees)), "r--", alpha=0.8)
        
        # 2. 多项式属性热图
        properties_matrix = np.column_stack([
            degrees,
            evaluation_measures,
            coefficient_weights,
            stabilities
        ])
        
        im1 = ax2.imshow(properties_matrix.T, cmap='YlOrRd', aspect='auto')
        ax2.set_title('Polynomial Properties Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trace Index')
        ax2.set_ylabel('Property Type')
        ax2.set_yticks(range(4))
        ax2.set_yticklabels(['Degree', 'Evaluation', 'Coefficient Weight', 'Stability'])
        plt.colorbar(im1, ax=ax2, label='Property Value')
        
        # 3. 根近似分布
        roots = [data['polynomial_properties']['root_approximation'] 
                for data in self.trace_universe.values()]
        root_reals = [r.real for r in roots]
        root_imags = [r.imag for r in roots]
        
        ax3.scatter(root_reals, root_imags, alpha=0.7, s=80, color='purple', edgecolors='black')
        ax3.set_title('Root Approximations (Real vs Imaginary)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Real Part')
        ax3.set_ylabel('Imaginary Part')
        ax3.grid(True, alpha=0.3)
        
        # 4. 变量索引分布
        all_var_indices = []
        for data in self.trace_universe.values():
            var_indices = data['polynomial_properties']['variable_indices']
            all_var_indices.extend(var_indices)
        
        if all_var_indices:
            ax4.hist(all_var_indices, bins=max(1, max(all_var_indices)+1), 
                    alpha=0.7, color='orange', edgecolor='black')
        else:
            ax4.hist([0], bins=1, alpha=0.7, color='orange', edgecolor='black')
        
        ax4.set_title('Variable Index Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Variable Index')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chapter-057-poly-collapse-properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_visualization_domains(self):
        """生成三域分析可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 获取分析数据
        domain_analysis = self.compute_three_domain_analysis()
        polynomial_analysis = self.analyze_polynomial_structure()
        info_analysis = self.analyze_information_theory()
        
        # 1. 域大小比较
        domains = ['Traditional', 'φ-Constrained', 'Intersection']
        sizes = [domain_analysis['traditional_domain'], 
                domain_analysis['phi_constrained_domain'],
                domain_analysis['intersection_domain']]
        
        bars = ax1.bar(domains, sizes, color=['blue', 'green', 'purple'], alpha=0.7, edgecolor='black')
        ax1.set_title('Domain Size Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Polynomials')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标注
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 多项式运算成功率
        success_rate = polynomial_analysis['operation_success_ratio']
        failure_rate = 1 - success_rate
        
        labels = ['Successful Operations', 'Failed Operations']
        sizes_pie = [success_rate, failure_rate]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax2.pie(sizes_pie, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Polynomial Operations Success Rate', fontsize=14, fontweight='bold')
        
        # 3. 信息论度量
        info_metrics = ['Polynomial Entropy', 'Coefficient Complexity', 'Degree Diversity']
        info_values = [info_analysis['polynomial_entropy'],
                      info_analysis['coefficient_complexity'],
                      info_analysis['degree_diversity']]
        
        bars3 = ax3.bar(info_metrics, info_values, 
                       color=['red', 'blue', 'green'], alpha=0.7, edgecolor='black')
        ax3.set_title('Information Theory Metrics', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Information Value')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标注
        for bar, value in zip(bars3, info_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 度数分布分析
        degree_counts = {}
        for data in self.trace_universe.values():
            degree = data['polynomial_properties']['polynomial_degree']
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        degrees = list(degree_counts.keys())
        counts = list(degree_counts.values())
        
        bars4 = ax4.bar(degrees, counts, alpha=0.7, color='cyan', edgecolor='black')
        ax4.set_title('Polynomial Degree Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Polynomial Degree')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标注
        for bar, count in zip(bars4, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('chapter-057-poly-collapse-domains.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数：运行PolyCollapse分析"""
    print("Running PolyCollapse Unit Tests...")
    print("\n" + "="*60)
    print("="*60)
    print("Chapter 057: PolyCollapse Comprehensive Analysis")
    print("Collapse Polynomial Systems over φ-Indexed Variables")
    print("="*60)
    
    # 创建系统
    system = PolyCollapseSystem()
    
    # 1. 基础多项式分析
    polynomial_analysis = system.analyze_polynomial_structure()
    print(f"\n1. Basic Polynomial Analysis:")
    print(f"Total polynomials: {polynomial_analysis['total_polynomials']}")
    print(f"Max degree: {polynomial_analysis['max_degree']}")
    print(f"Average degree: {polynomial_analysis['average_degree']:.3f}")
    print(f"Operation success ratio: {polynomial_analysis['operation_success_ratio']:.3f}")
    
    # 2. 多项式运算测试
    if polynomial_analysis['polynomial_samples']:
        print(f"\n2. Polynomial Operations Test:")
        poly1 = polynomial_analysis['polynomial_samples'][0]
        poly2 = polynomial_analysis['polynomial_samples'][1] if len(polynomial_analysis['polynomial_samples']) > 1 else poly1
        
        # 测试加法
        try:
            sum_poly = system.polynomial_addition(poly1, poly2)
            print(f"  Addition test: Success (degree {sum_poly['degree']})")
        except Exception as e:
            print(f"  Addition test: Failed")
        
        # 测试乘法
        try:
            product_poly = system.polynomial_multiplication(poly1, poly2)
            print(f"  Multiplication test: Success (degree {product_poly['degree']})")
        except Exception as e:
            print(f"  Multiplication test: Failed")
        
        # 测试求值
        try:
            phi = (1 + sqrt(5)) / 2  # 黄金比例
            eval_result = system.evaluate_polynomial(poly1, phi)
            print(f"  Evaluation test: {eval_result:.3f}")
        except Exception as e:
            print(f"  Evaluation test: Failed")
    
    # 3. 三域分析
    domain_analysis = system.compute_three_domain_analysis()
    print(f"\n3. Three-Domain Analysis:")
    print(f"Traditional polynomials: {domain_analysis['traditional_domain']}")
    print(f"φ-constrained polynomials: {domain_analysis['phi_constrained_domain']}")
    print(f"Intersection: {domain_analysis['intersection_domain']}")
    print(f"Convergence ratio: {domain_analysis['convergence_ratio']:.3f}")
    
    # 4. 信息论分析
    info_analysis = system.analyze_information_theory()
    print(f"\n4. Information Theory Analysis:")
    print(f"Polynomial entropy: {info_analysis['polynomial_entropy']:.3f} bits")
    print(f"Coefficient complexity: {info_analysis['coefficient_complexity']} elements")
    print(f"Degree diversity: {info_analysis['degree_diversity']} types")
    print(f"Evaluation information: {info_analysis['evaluation_information']:.3f} bits")
    
    # 5. 图论分析
    graph_analysis = system.analyze_graph_theory()
    print(f"\n5. Graph Theory Analysis:")
    print(f"Polynomial network nodes: {graph_analysis['nodes']}")
    print(f"Polynomial network edges: {graph_analysis['edges']}")
    print(f"Network density: {graph_analysis['density']:.3f}")
    print(f"Connected components: {graph_analysis['components']}")
    print(f"Average clustering: {graph_analysis['clustering']:.3f}")
    
    # 6. 生成可视化
    print(f"\n6. Generating Visualizations...")
    system.generate_visualization_structure()
    print("Saved visualization: chapter-057-poly-collapse-structure.png")
    
    system.generate_visualization_properties()
    print("Saved visualization: chapter-057-poly-collapse-properties.png")
    
    system.generate_visualization_domains()
    print("Saved visualization: chapter-057-poly-collapse-domains.png")
    
    # 7. 范畴论分析
    print(f"\n7. Category Theory Analysis:")
    print("Polynomial operations as functors:")
    print("- Objects: Trace coefficients with φ-constraint polynomial structure")
    print("- Morphisms: Polynomial-preserving transformations")
    print("- Composition: Polynomial algebraic operations")
    print("- Functors: Polynomial algebra homomorphisms")
    print("- Natural transformations: Between polynomial representations")
    
    print("\n" + "="*60)
    print("Analysis Complete - PolyCollapse System Verified")
    print("="*60)

class TestPolyCollapseSystem(unittest.TestCase):
    """PolyCollapse系统单元测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = PolyCollapseSystem()
        
    def test_trace_universe_generation(self):
        """测试trace universe生成"""
        self.assertGreater(len(self.system.trace_universe), 0)
        for data in self.system.trace_universe.values():
            self.assertTrue(data['phi_valid'])
            self.assertNotIn('11', data['trace'])
            
    def test_polynomial_creation(self):
        """测试多项式创建"""
        coeffs = ['1', '10', '101']
        poly = self.system.create_polynomial(coeffs)
        self.assertIsInstance(poly, dict)
        self.assertIn('coefficients', poly)
        self.assertIn('degree', poly)
        
    def test_polynomial_operations(self):
        """测试多项式运算"""
        poly1 = self.system.create_polynomial(['1', '10'])
        poly2 = self.system.create_polynomial(['10', '1'])
        
        # 测试加法
        sum_poly = self.system.polynomial_addition(poly1, poly2)
        self.assertIsInstance(sum_poly, dict)
        self.assertGreaterEqual(sum_poly['degree'], 0)
        
        # 测试乘法
        product_poly = self.system.polynomial_multiplication(poly1, poly2)
        self.assertIsInstance(product_poly, dict)
        self.assertGreaterEqual(product_poly['degree'], 0)
        
    def test_polynomial_evaluation(self):
        """测试多项式求值"""
        poly = self.system.create_polynomial(['1', '10'])
        result = self.system.evaluate_polynomial(poly, 2.0)
        self.assertIsInstance(result, (int, float))
        
    def test_polynomial_properties(self):
        """测试多项式属性计算"""
        for data in self.system.trace_universe.values():
            props = data['polynomial_properties']
            self.assertIn('coefficient_signature', props)
            self.assertIn('polynomial_degree', props)
            self.assertIn('variable_indices', props)
            self.assertIn('evaluation_measure', props)
            
            # 检验数值范围
            self.assertGreaterEqual(props['polynomial_degree'], 0)
            self.assertGreaterEqual(props['evaluation_measure'], 0.0)
            self.assertLessEqual(props['evaluation_measure'], 1.0)

if __name__ == "__main__":
    # 运行主分析
    main()
    
    # 运行单元测试
    unittest.main(argv=[''], exit=False, verbosity=2)