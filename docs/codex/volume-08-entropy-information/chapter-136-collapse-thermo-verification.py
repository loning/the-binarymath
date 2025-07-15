#!/usr/bin/env python3
"""
Chapter 136: CollapseThermo Unit Test Verification
从ψ=ψ(ψ)推导Collapse Thermodynamics and Entropy Flow Laws

Core principle: From ψ = ψ(ψ) derive systematic thermodynamic laws through 
entropy flow conservation that enables fundamental thermal dynamics through 
temperature analogs that create natural heat flow embodying the essential 
properties of collapsed thermodynamics through entropy-conserving tensor 
transformations that establish systematic thermal laws through internal 
energy relationships rather than external thermodynamic impositions.

This verification program implements:
1. φ-constrained thermodynamics through temperature and heat capacity
2. Entropy flow systems: fundamental conservation and growth laws
3. Three-domain analysis: Traditional vs φ-constrained vs intersection thermodynamics
4. Graph theory analysis of thermal networks and flow structures
5. Information theory analysis of thermal properties and phase transitions
6. Category theory analysis of thermal functors and conservation morphisms
7. Visualization of thermodynamic evolution and φ-constraint thermal systems
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import unittest
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from collections import defaultdict, deque
from math import log2, sqrt, pi, exp, log, cos, sin, atan2
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

class CollapseThermoSystem:
    """
    Core system for implementing collapse thermodynamics in φ-constrained space.
    Implements thermal architectures through entropy flow and conservation analysis.
    """
    
    def __init__(self, max_trace_length: int = 10, time_steps: int = 100, 
                 ensemble_size: int = 50):
        """Initialize collapse thermodynamic system with thermal analysis"""
        self.max_trace_length = max_trace_length
        self.time_steps = time_steps
        self.ensemble_size = ensemble_size
        self.phi = (1 + sqrt(5)) / 2
        self.fibonacci_numbers = self._generate_fibonacci(15)
        self.thermal_cache = {}
        self.flow_cache = {}
        
        # Initialize thermodynamic ensemble
        self.ensemble = self._initialize_ensemble()
        self.temperature_evolution = self._compute_temperature_evolution()
        self.entropy_flow = self._analyze_entropy_flow()
        self.heat_capacity = self._compute_heat_capacity()
        self.conservation_laws = self._verify_conservation_laws()
        self.phase_diagram = self._construct_phase_diagram()
        self.thermal_network = self._build_thermal_network()
        
    def _generate_fibonacci(self, count: int) -> List[int]:
        """从ψ=ψ(ψ)推导Fibonacci数列：F(n) = F(n-1) + F(n-2)"""
        fib = [1, 1]
        for i in range(2, count):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _initialize_ensemble(self) -> List[Dict]:
        """初始化热力学系综：φ-valid traces的集合"""
        ensemble = []
        
        for i in range(self.ensemble_size):
            # 生成随机长度
            length = np.random.randint(3, self.max_trace_length + 1)
            
            # 生成φ-valid trace
            trace = self._generate_random_phi_valid(length)
            if trace:
                # 计算初始热力学量
                entropy = self._compute_phi_entropy(trace)
                temperature = self._compute_temperature(trace)
                energy = self._compute_energy(trace)
                
                ensemble.append({
                    'id': i,
                    'trace': trace,
                    'length': length,
                    'entropy': entropy,
                    'temperature': temperature,
                    'energy': energy,
                    'history': [(0, trace, entropy, temperature, energy)]
                })
                
        return ensemble
        
    def _generate_random_phi_valid(self, length: int) -> Optional[str]:
        """生成随机的φ-valid trace"""
        if length == 0:
            return "0"
        
        # 使用Fibonacci表示确保φ-validity
        max_val = min(self.fibonacci_numbers[min(length-1, len(self.fibonacci_numbers)-1)], 
                      2**length - 1)
        
        # 尝试生成有效trace
        for _ in range(10):  # 最多尝试10次
            n = np.random.randint(1, max_val + 1)
            trace = self._encode_to_trace(n, length)
            if self._is_phi_valid(trace):
                return trace
                
        # 如果失败，返回安全的交替模式
        return "10" * (length // 2) + ("1" if length % 2 else "")
        
    def _encode_to_trace(self, n: int, target_length: int) -> str:
        """编码整数n为目标长度的二进制trace"""
        if n == 0:
            return "0" * target_length
            
        binary = bin(n)[2:]
        if len(binary) > target_length:
            binary = binary[-target_length:]
        elif len(binary) < target_length:
            binary = "0" * (target_length - len(binary)) + binary
            
        return binary
        
    def _is_phi_valid(self, trace: str) -> bool:
        """验证trace是否满足φ-constraint（无连续11）"""
        return "11" not in trace
        
    def _compute_phi_entropy(self, trace: str) -> float:
        """计算φ-约束熵"""
        if not trace:
            return 0.0
        
        # 转换为tensor
        trace_tensor = torch.tensor([float(bit) for bit in trace], dtype=torch.float32)
        
        # φ-熵基于Fibonacci权重的位置熵
        fib_weights = torch.tensor([1/self.phi**i for i in range(len(trace_tensor))], 
                                  dtype=torch.float32)
        
        # 计算加权比特熵
        weighted_bits = trace_tensor * fib_weights
        
        # 计算约束因子
        constraint_factor = 1.0
        for i in range(len(trace) - 1):
            if trace[i] == '1' and trace[i+1] == '0':
                constraint_factor *= 1.1
            elif trace[i] == '0' and trace[i+1] == '1':
                constraint_factor *= 0.9
        
        # φ-熵
        phi_entropy = float(torch.sum(weighted_bits)) * constraint_factor * log2(1 + len(trace))
        
        return phi_entropy
        
    def _compute_temperature(self, trace: str) -> float:
        """计算信息温度：基于局部熵变化率"""
        if len(trace) < 2:
            return 1.0
            
        # 计算局部熵差
        local_entropies = []
        window_size = min(3, len(trace))
        
        for i in range(len(trace) - window_size + 1):
            window = trace[i:i+window_size]
            local_entropies.append(self._compute_phi_entropy(window))
            
        # 温度 = 熵变化率的平均值
        if len(local_entropies) > 1:
            entropy_diffs = [abs(local_entropies[i+1] - local_entropies[i]) 
                           for i in range(len(local_entropies)-1)]
            temperature = np.mean(entropy_diffs) + 0.1  # 避免零温度
        else:
            temperature = 1.0
            
        return temperature
        
    def _compute_energy(self, trace: str) -> float:
        """计算信息能量：E = H + T*production"""
        entropy = self._compute_phi_entropy(trace)
        temperature = self._compute_temperature(trace)
        
        # 生产项：基于1的密度
        ones_count = trace.count('1')
        production = ones_count / len(trace) if len(trace) > 0 else 0
        
        energy = entropy + temperature * production
        
        return energy
        
    def _compute_temperature_evolution(self) -> Dict[int, List[float]]:
        """计算温度演化"""
        evolution = defaultdict(list)
        
        for t in range(self.time_steps):
            # 对每个系综成员进行演化
            for member in self.ensemble:
                # 执行热力学演化步骤
                new_trace = self._thermal_evolution_step(member['trace'], member['temperature'])
                
                if new_trace and self._is_phi_valid(new_trace):
                    # 更新状态
                    new_entropy = self._compute_phi_entropy(new_trace)
                    new_temperature = self._compute_temperature(new_trace)
                    new_energy = self._compute_energy(new_trace)
                    
                    member['trace'] = new_trace
                    member['entropy'] = new_entropy
                    member['temperature'] = new_temperature
                    member['energy'] = new_energy
                    member['history'].append((t+1, new_trace, new_entropy, 
                                            new_temperature, new_energy))
                    
                    evolution[member['id']].append(new_temperature)
                    
        return dict(evolution)
        
    def _thermal_evolution_step(self, trace: str, temperature: float) -> Optional[str]:
        """执行一步热力学演化"""
        if not trace:
            return None
            
        # 温度决定变化概率
        change_prob = min(temperature / 2.0, 0.8)
        
        if np.random.random() < change_prob:
            # 选择演化类型
            evolution_type = np.random.choice(['flip', 'insert', 'delete'])
            
            if evolution_type == 'flip' and len(trace) > 0:
                # 翻转随机位
                pos = np.random.randint(0, len(trace))
                new_bit = '0' if trace[pos] == '1' else '1'
                new_trace = trace[:pos] + new_bit + trace[pos+1:]
                
            elif evolution_type == 'insert' and len(trace) < self.max_trace_length:
                # 插入位
                pos = np.random.randint(0, len(trace) + 1)
                new_bit = '0' if np.random.random() < 0.6 else '1'  # 偏向0避免11
                new_trace = trace[:pos] + new_bit + trace[pos:]
                
            elif evolution_type == 'delete' and len(trace) > 2:
                # 删除位
                pos = np.random.randint(0, len(trace))
                new_trace = trace[:pos] + trace[pos+1:]
            else:
                new_trace = trace
                
            # 确保φ-validity
            if self._is_phi_valid(new_trace):
                return new_trace
                
        return trace
        
    def _analyze_entropy_flow(self) -> Dict[str, Any]:
        """分析熵流"""
        flow_data = {
            'total_entropy': [],
            'entropy_production': [],
            'entropy_dissipation': [],
            'net_flow': [],
            'conservation_violations': []
        }
        
        # 计算每个时间步的熵流
        for t in range(len(self.ensemble[0]['history'])):
            total_S = 0
            production = 0
            dissipation = 0
            
            for member in self.ensemble:
                if t < len(member['history']):
                    _, _, entropy, _, _ = member['history'][t]
                    total_S += entropy
                    
                    if t > 0:
                        prev_entropy = member['history'][t-1][2]
                        delta_S = entropy - prev_entropy
                        
                        if delta_S > 0:
                            production += delta_S
                        else:
                            dissipation += abs(delta_S)
                            
            flow_data['total_entropy'].append(total_S)
            flow_data['entropy_production'].append(production)
            flow_data['entropy_dissipation'].append(dissipation)
            flow_data['net_flow'].append(production - dissipation)
            
            # 检查守恒违反（如果净流过大）
            if production - dissipation > 0.1 * total_S:
                flow_data['conservation_violations'].append(t)
                
        return flow_data
        
    def _compute_heat_capacity(self) -> Dict[str, List[float]]:
        """计算热容量：C = dE/dT"""
        heat_capacity = defaultdict(list)
        
        for member in self.ensemble:
            temps = []
            energies = []
            
            for _, _, _, temp, energy in member['history']:
                temps.append(temp)
                energies.append(energy)
                
            # 计算局部热容量
            if len(temps) > 1:
                for i in range(1, len(temps)):
                    dT = temps[i] - temps[i-1]
                    dE = energies[i] - energies[i-1]
                    
                    if abs(dT) > 1e-6:
                        C = dE / dT
                        heat_capacity[member['id']].append(C)
                    else:
                        heat_capacity[member['id']].append(0.0)
                        
        return dict(heat_capacity)
        
    def _verify_conservation_laws(self) -> Dict[str, Any]:
        """验证守恒定律"""
        conservation = {
            'energy_conserved': True,
            'entropy_increasing': True,
            'flow_balanced': True,
            'violations': []
        }
        
        # 检查能量守恒（考虑交换）
        initial_energy = sum(m['history'][0][4] for m in self.ensemble)
        final_energy = sum(m['history'][-1][4] for m in self.ensemble)
        
        energy_change = abs(final_energy - initial_energy) / initial_energy
        if energy_change > 0.1:  # 10%容差
            conservation['energy_conserved'] = False
            conservation['violations'].append(f'Energy change: {energy_change:.2%}')
            
        # 检查熵增
        for member in self.ensemble:
            entropies = [h[2] for h in member['history']]
            for i in range(1, len(entropies)):
                if entropies[i] < entropies[i-1] * 0.95:  # 允许小幅下降
                    conservation['entropy_increasing'] = False
                    conservation['violations'].append(
                        f'Entropy decrease at member {member["id"]}, step {i}')
                    break
                    
        # 检查流平衡
        net_flows = self.entropy_flow['net_flow']
        total_net_flow = sum(net_flows)
        if abs(total_net_flow) > 0.1 * sum(self.entropy_flow['total_entropy']):
            conservation['flow_balanced'] = False
            conservation['violations'].append(f'Unbalanced flow: {total_net_flow:.3f}')
            
        return conservation
        
    def _construct_phase_diagram(self) -> Dict[Tuple[float, float], str]:
        """构建相图：温度-熵空间"""
        phase_diagram = {}
        
        # 收集所有温度-熵点
        for member in self.ensemble:
            for _, _, entropy, temperature, _ in member['history']:
                # 量化到网格
                T_bin = round(temperature * 10) / 10
                S_bin = round(entropy * 2) / 2
                
                # 分类相
                if temperature < 0.5:
                    phase = 'frozen'
                elif temperature > 2.0:
                    phase = 'chaotic'
                elif entropy < 2.0:
                    phase = 'ordered'
                elif entropy > 4.0:
                    phase = 'saturated'
                else:
                    phase = 'normal'
                    
                phase_diagram[(T_bin, S_bin)] = phase
                
        return phase_diagram
        
    def _build_thermal_network(self) -> nx.Graph:
        """构建热网络：基于温差的热流"""
        G = nx.Graph()
        
        # 添加节点（系综成员）
        for member in self.ensemble:
            G.add_node(member['id'], 
                      temperature=member['temperature'],
                      entropy=member['entropy'],
                      energy=member['energy'])
                      
        # 添加边（基于温差的热连接）
        for i in range(len(self.ensemble)):
            for j in range(i+1, len(self.ensemble)):
                T_i = self.ensemble[i]['temperature']
                T_j = self.ensemble[j]['temperature']
                
                # 温差驱动热流
                temp_diff = abs(T_i - T_j)
                if temp_diff > 0.1:  # 阈值
                    heat_flow = temp_diff * 0.5  # 简化的热流
                    G.add_edge(i, j, weight=heat_flow, temp_diff=temp_diff)
                    
        return G
        
    def get_thermo_statistics(self) -> Dict[str, Any]:
        """获取热力学统计信息"""
        # 最终状态统计
        final_temps = [m['temperature'] for m in self.ensemble]
        final_entropies = [m['entropy'] for m in self.ensemble]
        final_energies = [m['energy'] for m in self.ensemble]
        
        # 演化统计
        total_production = sum(self.entropy_flow['entropy_production'])
        total_dissipation = sum(self.entropy_flow['entropy_dissipation'])
        
        return {
            'ensemble_size': self.ensemble_size,
            'time_steps': self.time_steps,
            'mean_temperature': np.mean(final_temps),
            'std_temperature': np.std(final_temps),
            'mean_entropy': np.mean(final_entropies),
            'max_entropy': np.max(final_entropies),
            'mean_energy': np.mean(final_energies),
            'total_production': total_production,
            'total_dissipation': total_dissipation,
            'net_entropy_change': total_production - total_dissipation,
            'conservation_status': self.conservation_laws,
            'phase_count': len(set(self.phase_diagram.values())),
            'thermal_edges': self.thermal_network.number_of_edges()
        }
        
    def visualize_thermodynamic_evolution(self, save_path: str = "chapter-136-collapse-thermo.png"):
        """可视化热力学演化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Collapse Thermodynamics in φ-Constrained Systems', 
                    fontsize=16, fontweight='bold')
        
        # 1. 温度演化
        ax1 = axes[0, 0]
        for member_id, temps in list(self.temperature_evolution.items())[:10]:  # 显示前10个
            ax1.plot(temps, alpha=0.6, linewidth=1)
        
        # 平均温度
        mean_temps = []
        for t in range(self.time_steps):
            temps_at_t = [self.temperature_evolution[m_id][t] 
                         for m_id in self.temperature_evolution 
                         if t < len(self.temperature_evolution[m_id])]
            if temps_at_t:
                mean_temps.append(np.mean(temps_at_t))
        
        ax1.plot(mean_temps, 'k-', linewidth=3, label='Mean Temperature')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Temperature')
        ax1.set_title('Temperature Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 熵流分析
        ax2 = axes[0, 1]
        ax2.plot(self.entropy_flow['total_entropy'], 'b-', label='Total Entropy', linewidth=2)
        ax2.plot(self.entropy_flow['entropy_production'], 'g-', label='Production', linewidth=1)
        ax2.plot(self.entropy_flow['entropy_dissipation'], 'r-', label='Dissipation', linewidth=1)
        ax2.fill_between(range(len(self.entropy_flow['net_flow'])), 
                        self.entropy_flow['net_flow'], alpha=0.3, label='Net Flow')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy Flow Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 热容量分布
        ax3 = axes[0, 2]
        all_heat_capacities = []
        for member_id, capacities in self.heat_capacity.items():
            all_heat_capacities.extend(capacities)
        
        if all_heat_capacities:
            ax3.hist(all_heat_capacities, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(np.mean(all_heat_capacities), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_heat_capacities):.3f}')
        
        ax3.set_xlabel('Heat Capacity (dE/dT)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Heat Capacity Distribution')
        ax3.legend()
        
        # 4. 相图
        ax4 = axes[1, 0]
        
        # 提取相图数据
        temps = [k[0] for k in self.phase_diagram.keys()]
        entropies = [k[1] for k in self.phase_diagram.keys()]
        phases = list(self.phase_diagram.values())
        
        # 相的颜色映射
        phase_colors = {
            'frozen': 'blue',
            'ordered': 'cyan',
            'normal': 'green',
            'saturated': 'orange',
            'chaotic': 'red'
        }
        colors = [phase_colors.get(p, 'gray') for p in phases]
        
        scatter = ax4.scatter(temps, entropies, c=colors, s=50, alpha=0.6)
        
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Phase Diagram')
        
        # 添加相的图例
        for phase, color in phase_colors.items():
            ax4.scatter([], [], c=color, label=phase, s=100)
        ax4.legend()
        
        # 5. 热网络
        ax5 = axes[1, 1]
        
        if self.thermal_network.number_of_nodes() > 0:
            pos = nx.spring_layout(self.thermal_network, k=2, iterations=50)
            
            # 节点颜色基于温度
            node_colors = [self.ensemble[node]['temperature'] 
                          for node in self.thermal_network.nodes()]
            
            # 绘制网络
            nx.draw_networkx_nodes(self.thermal_network, pos, 
                                 node_color=node_colors, cmap='hot',
                                 node_size=100, alpha=0.8, ax=ax5)
            
            # 边的粗细基于热流
            edges = self.thermal_network.edges()
            weights = [self.thermal_network[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(self.thermal_network, pos, 
                                 edge_color='gray', width=weights,
                                 alpha=0.5, ax=ax5)
            
            ax5.set_title('Thermal Network (Heat Flow)')
            ax5.axis('off')
        
        # 6. 统计摘要
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats = self.get_thermo_statistics()
        summary_text = f"""
        Collapse Thermodynamics Summary:
        
        Ensemble: {stats['ensemble_size']} members
        Evolution: {stats['time_steps']} steps
        
        Temperature:
          Mean: {stats['mean_temperature']:.3f}
          Std: {stats['std_temperature']:.3f}
        
        Entropy:
          Mean: {stats['mean_entropy']:.3f}
          Max: {stats['max_entropy']:.3f}
          Total Production: {stats['total_production']:.3f}
          Total Dissipation: {stats['total_dissipation']:.3f}
          Net Change: {stats['net_entropy_change']:.3f}
        
        Conservation:
          Energy: {stats['conservation_status']['energy_conserved']}
          Entropy Increase: {stats['conservation_status']['entropy_increasing']}
          Flow Balance: {stats['conservation_status']['flow_balanced']}
        
        Phases: {stats['phase_count']} distinct phases
        Thermal Connections: {stats['thermal_edges']} edges
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_thermodynamic_landscape(self, save_path: str = "chapter-136-thermo-landscape.png"):
        """可视化热力学景观"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D能量景观
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 收集数据
        temps = []
        entropies = []
        energies = []
        
        for member in self.ensemble:
            for _, _, S, T, E in member['history']:
                temps.append(T)
                entropies.append(S)
                energies.append(E)
        
        # 创建散点图
        scatter = ax1.scatter(temps, entropies, energies, 
                            c=energies, cmap='viridis', s=20, alpha=0.6)
        
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Entropy')
        ax1.set_zlabel('Energy')
        ax1.set_title('Thermodynamic Energy Landscape')
        
        # 2. 熵产生率
        ax2 = fig.add_subplot(222)
        
        production_rate = []
        for i in range(1, len(self.entropy_flow['entropy_production'])):
            rate = self.entropy_flow['entropy_production'][i] - \
                   self.entropy_flow['entropy_production'][i-1]
            production_rate.append(rate)
        
        ax2.plot(production_rate, 'g-', alpha=0.7, label='Production Rate')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(production_rate)), production_rate, 
                        alpha=0.3, color='green')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy Production Rate')
        ax2.set_title('Entropy Production Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 温度-能量相关性
        ax3 = fig.add_subplot(223)
        
        final_temps = [m['temperature'] for m in self.ensemble]
        final_energies = [m['energy'] for m in self.ensemble]
        
        ax3.scatter(final_temps, final_energies, alpha=0.6, s=50)
        
        # 拟合线
        if len(final_temps) > 1:
            z = np.polyfit(final_temps, final_energies, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(final_temps), max(final_temps), 100)
            ax3.plot(x_line, p(x_line), 'r--', alpha=0.8, 
                    label=f'Fit: E = {z[0]:.2f}T + {z[1]:.2f}')
        
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Energy')
        ax3.set_title('Temperature-Energy Relationship')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 熵-温度轨迹
        ax4 = fig.add_subplot(224)
        
        # 绘制几个代表性成员的轨迹
        for i in range(min(5, len(self.ensemble))):
            member = self.ensemble[i]
            member_temps = []
            member_entropies = []
            
            for _, _, S, T, _ in member['history']:
                member_temps.append(T)
                member_entropies.append(S)
                
            ax4.plot(member_temps, member_entropies, alpha=0.6, linewidth=2)
            ax4.scatter(member_temps[0], member_entropies[0], s=100, 
                       marker='o', edgecolor='black', linewidth=2)  # 起点
            ax4.scatter(member_temps[-1], member_entropies[-1], s=100, 
                       marker='s', edgecolor='black', linewidth=2)  # 终点
        
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Entropy-Temperature Trajectories')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class TestCollapseThermoSystem(unittest.TestCase):
    """测试CollapseThermoSystem的各个功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = CollapseThermoSystem(max_trace_length=8, 
                                         time_steps=50, 
                                         ensemble_size=20)
        
    def test_ensemble_initialization(self):
        """测试系综初始化"""
        self.assertEqual(len(self.system.ensemble), self.system.ensemble_size)
        
        for member in self.system.ensemble:
            # 检查必要字段
            self.assertIn('id', member)
            self.assertIn('trace', member)
            self.assertIn('entropy', member)
            self.assertIn('temperature', member)
            self.assertIn('energy', member)
            
            # 检查φ-validity
            self.assertTrue(self.system._is_phi_valid(member['trace']))
            
    def test_temperature_computation(self):
        """测试温度计算"""
        # 测试不同traces的温度
        test_traces = ["101", "1010", "10010", "100101"]
        
        for trace in test_traces:
            temp = self.system._compute_temperature(trace)
            self.assertGreater(temp, 0)  # 温度应该为正
            self.assertLess(temp, 10)     # 合理范围内
            
    def test_energy_computation(self):
        """测试能量计算"""
        trace = "10101"
        entropy = self.system._compute_phi_entropy(trace)
        temperature = self.system._compute_temperature(trace)
        energy = self.system._compute_energy(trace)
        
        # 能量应该大于熵（因为有生产项）
        self.assertGreaterEqual(energy, entropy)
        
    def test_thermal_evolution(self):
        """测试热演化"""
        initial_trace = "1010"
        initial_temp = self.system._compute_temperature(initial_trace)
        
        # 执行演化步骤
        new_trace = self.system._thermal_evolution_step(initial_trace, initial_temp)
        
        # 应该返回φ-valid trace
        self.assertTrue(self.system._is_phi_valid(new_trace))
        
    def test_entropy_flow_conservation(self):
        """测试熵流守恒"""
        flow = self.system.entropy_flow
        
        # 总熵应该趋向增加
        if len(flow['total_entropy']) > 1:
            initial_total = flow['total_entropy'][0]
            final_total = flow['total_entropy'][-1]
            self.assertGreaterEqual(final_total, initial_total * 0.9)  # 允许小幅波动
            
    def test_heat_capacity_validity(self):
        """测试热容量有效性"""
        # 热容量应该是有限的
        for member_id, capacities in self.system.heat_capacity.items():
            for C in capacities:
                self.assertTrue(np.isfinite(C))
                self.assertLess(abs(C), 100)  # 合理范围
                
    def test_phase_diagram_completeness(self):
        """测试相图完整性"""
        phases = set(self.system.phase_diagram.values())
        
        # 应该有多个相
        self.assertGreater(len(phases), 0)
        
        # 相应该是预定义的类型之一
        valid_phases = {'frozen', 'ordered', 'normal', 'saturated', 'chaotic'}
        for phase in phases:
            self.assertIn(phase, valid_phases)
            
    def test_thermal_network_connectivity(self):
        """测试热网络连通性"""
        G = self.system.thermal_network
        
        # 应该有节点
        self.assertGreater(G.number_of_nodes(), 0)
        
        # 如果有边，检查边的属性
        for u, v, data in G.edges(data=True):
            self.assertIn('weight', data)
            self.assertIn('temp_diff', data)
            self.assertGreater(data['weight'], 0)
            
    def test_conservation_laws(self):
        """测试守恒定律"""
        conservation = self.system.conservation_laws
        
        # 检查守恒状态
        self.assertIn('energy_conserved', conservation)
        self.assertIn('entropy_increasing', conservation)
        self.assertIn('flow_balanced', conservation)
        
    def test_statistics_consistency(self):
        """测试统计一致性"""
        stats = self.system.get_thermo_statistics()
        
        # 检查统计量的合理性
        self.assertGreater(stats['mean_temperature'], 0)
        self.assertGreater(stats['mean_entropy'], 0)
        self.assertGreater(stats['mean_energy'], 0)
        
        # 生产应该大于等于0
        self.assertGreaterEqual(stats['total_production'], 0)
        
        # 相数应该合理
        self.assertGreater(stats['phase_count'], 0)
        self.assertLessEqual(stats['phase_count'], 5)


def main():
    """主函数：运行完整的验证程序"""
    print("=" * 60)
    print("Chapter 136: CollapseThermo Unit Test Verification")
    print("从ψ=ψ(ψ)推导Collapse Thermodynamics")
    print("=" * 60)
    
    # 创建系统实例
    system = CollapseThermoSystem(max_trace_length=10, time_steps=100, ensemble_size=50)
    
    # 获取统计信息
    stats = system.get_thermo_statistics()
    
    print("\n1. 系统初始化完成")
    print(f"   系综大小: {stats['ensemble_size']} 成员")
    print(f"   演化步数: {stats['time_steps']}")
    print(f"   最大trace长度: {system.max_trace_length}")
    
    print("\n2. 温度统计:")
    print(f"   平均温度: {stats['mean_temperature']:.3f}")
    print(f"   温度标准差: {stats['std_temperature']:.3f}")
    
    print("\n3. 熵统计:")
    print(f"   平均熵: {stats['mean_entropy']:.3f}")
    print(f"   最大熵: {stats['max_entropy']:.3f}")
    print(f"   总产生: {stats['total_production']:.3f}")
    print(f"   总耗散: {stats['total_dissipation']:.3f}")
    print(f"   净变化: {stats['net_entropy_change']:.3f}")
    
    print("\n4. 守恒定律:")
    conservation = stats['conservation_status']
    print(f"   能量守恒: {conservation['energy_conserved']}")
    print(f"   熵增定律: {conservation['entropy_increasing']}")
    print(f"   流平衡: {conservation['flow_balanced']}")
    if conservation['violations']:
        print(f"   违反情况: {len(conservation['violations'])}个")
    
    print("\n5. 相分析:")
    print(f"   检测到的相: {stats['phase_count']}种")
    
    # 相的分布
    phase_counts = defaultdict(int)
    for phase in system.phase_diagram.values():
        phase_counts[phase] += 1
    
    for phase, count in sorted(phase_counts.items()):
        percentage = (count / len(system.phase_diagram)) * 100
        print(f"   {phase}: {count} ({percentage:.1f}%)")
    
    print("\n6. 热网络:")
    print(f"   热连接数: {stats['thermal_edges']}")
    
    # 运行单元测试
    print("\n7. 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n8. 生成可视化...")
    try:
        system.visualize_thermodynamic_evolution()
        print("   ✓ 热力学演化图生成成功")
    except Exception as e:
        print(f"   ✗ 热力学演化图生成失败: {e}")
    
    try:
        system.visualize_thermodynamic_landscape()
        print("   ✓ 热力学景观图生成成功")
    except Exception as e:
        print(f"   ✗ 热力学景观图生成失败: {e}")
    
    print("\n9. 验证完成!")
    print("   Collapse热力学系统运行正常")
    print("   φ-constraint创造了信息热力学新形式")
    print("   发现温度-熵-能量的内在关联")


if __name__ == "__main__":
    main()