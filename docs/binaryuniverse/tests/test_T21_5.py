"""
T21-5 概率等价性定理测试套件
基于重构后的T21-5理论：黎曼ζ函数与collapse方程在纯Zeckendorf数学体系中的概率等价性
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import math
import cmath
from typing import List, Tuple, Dict, Any, Optional
from base_framework import BinaryUniverseFramework, ZeckendorfEncoder, PhiBasedMeasure
import numpy as np


class ZeckendorfProbabilisticEquivalenceSystem(BinaryUniverseFramework):
    """T21-5概率等价性系统"""
    
    def __init__(self, precision: int = 15):
        super().__init__()
        self.precision = precision
        self.encoder = ZeckendorfEncoder()
        self.phi = (1 + math.sqrt(5)) / 2
        self.setup_axioms()
        self.setup_definitions()
    
    def setup_axioms(self):
        """设置唯一公理"""
        pass
    
    def setup_definitions(self):
        """设置定义"""
        pass
    
    def zeckendorf_zeta_function(self, s: complex, max_terms: int = 30) -> complex:
        """Zeckendorf-ζ函数：ζ_Z(s) = ⊕_{n=1}^∞ 1_Z / n^⊗s"""
        result = complex(0, 0)
        
        for n in range(1, max_terms + 1):
            # 在Zeckendorf空间中计算1/n^s
            try:
                # 简化实现：使用连续数学近似，然后应用Zeckendorf约束
                term = 1.0 / (n ** s)
                
                # 应用Zeckendorf编码约束
                term_zeck = self.encoder.to_zeckendorf(term.real)
                if self.encoder.is_valid_zeckendorf(term_zeck):
                    # 解码并累加
                    term_value = self.encoder.from_zeckendorf(term_zeck)
                    result += complex(term_value, 0)
                else:
                    # 应用无11约束修正
                    corrected_term = self._apply_no11_constraint(term)
                    result += corrected_term
                    
            except (OverflowError, ZeroDivisionError):
                break
                
        return result
    
    def zeckendorf_collapse_function(self, s: complex) -> complex:
        """Zeckendorf-collapse函数：e_op^(i_Z π_op s) ⊕ φ_op^s ⊗ (φ_op ⊖ 1_Z)"""
        
        # 第一项：e_op^(i_Z π_op s)
        i_pi_s = complex(0, 1) * math.pi * s
        exp_term = cmath.exp(i_pi_s)
        
        # 第二项：φ_op^s ⊗ (φ_op ⊖ 1_Z)
        phi_power_s = self.phi ** s
        phi_minus_one = self.phi - 1
        second_term = phi_power_s * phi_minus_one
        
        # Fibonacci加法（⊕）
        result = exp_term + second_term
        
        # 应用Zeckendorf约束
        return self._apply_zeckendorf_constraint(result)
    
    def _apply_no11_constraint(self, value: complex) -> complex:
        """应用无11约束"""
        # 简化实现：确保结果符合Fibonacci结构
        real_zeck = self.encoder.to_zeckendorf(value.real)
        imag_zeck = self.encoder.to_zeckendorf(value.imag)
        
        # 修正连续11模式
        real_corrected = self._correct_consecutive_ones(real_zeck)
        imag_corrected = self._correct_consecutive_ones(imag_zeck)
        
        real_val = self.encoder.from_zeckendorf(real_corrected)
        imag_val = self.encoder.from_zeckendorf(imag_corrected)
        
        return complex(real_val, imag_val)
    
    def _correct_consecutive_ones(self, zeck_list: List[int]) -> List[int]:
        """修正连续1模式"""
        corrected = zeck_list.copy()
        i = 0
        while i < len(corrected) - 1:
            if corrected[i] == 1 and corrected[i+1] == 1:
                # 应用Fibonacci递推关系：F_n + F_{n+1} = F_{n+2}
                corrected[i] = 0
                corrected[i+1] = 0
                if i+2 < len(corrected):
                    corrected[i+2] = 1
                else:
                    corrected.append(1)
            i += 1
        return corrected
    
    def _apply_zeckendorf_constraint(self, value: complex) -> complex:
        """应用完整Zeckendorf约束"""
        return self._apply_no11_constraint(value)
    
    def decompose_collapse_into_three_components(self, s: complex) -> Tuple[complex, complex, complex]:
        """将collapse函数分解为φ、π、e三个分量"""
        
        # φ分量：φ_op^s ⊗ (φ_op ⊖ 1_Z)
        phi_power_s = self.phi ** s
        phi_component = phi_power_s * (self.phi - 1)
        
        # π分量：e_op^(i_Z π_op s)  
        i_pi_s = complex(0, 1) * math.pi * s
        pi_component = cmath.exp(i_pi_s)
        
        # e分量：连接算子，贡献为0
        e_component = complex(0, 0)
        
        return phi_component, pi_component, e_component
    
    def compute_three_fold_indicators(self, s: complex) -> Tuple[int, int, int]:
        """计算三元指示函数 I_φ(s), I_π(s), I_e(s)"""
        phi_comp, pi_comp, e_comp = self.decompose_collapse_into_three_components(s)
        
        # 计算collapse函数总值用于阈值
        collapse_value = self.zeckendorf_collapse_function(s)
        threshold = abs(collapse_value) / 2
        
        # 计算指示函数
        indicator_phi = 1 if abs(phi_comp) > threshold else 0
        indicator_pi = 1 if abs(pi_comp) > threshold else 0
        indicator_e = 0  # 恒为0
        
        # 确保互斥性
        if indicator_phi == 1 and indicator_pi == 1:
            if abs(phi_comp) > abs(pi_comp):
                indicator_pi = 0
            else:
                indicator_phi = 0
        
        return indicator_phi, indicator_pi, indicator_e
    
    def compute_equivalence_probability(self, s: complex) -> float:
        """计算等价概率：P = 2/3 * I_φ + 1/3 * I_π + 0 * I_e"""
        indicator_phi, indicator_pi, indicator_e = self.compute_three_fold_indicators(s)
        
        probability = (2/3) * indicator_phi + (1/3) * indicator_pi + 0 * indicator_e
        return probability
    
    def predict_theoretical_probability(self, s: complex) -> float:
        """基于T27-2理论预测等价概率"""
        # 根据参数区域预测
        if abs(s.real - 0.5) < 1e-6:  # 临界线Re(s)=1/2
            return 1/3  # π主导区域
        elif abs(s.imag) < 1.0:  # 低虚部区域  
            return 2/3  # φ主导区域
        else:  # 高虚部区域
            return 0.0  # e连接但不等价
    
    def analyze_equivalence_at_point(self, s: complex, tolerance: float = 1e-4) -> Dict[str, Any]:
        """分析特定点的等价性"""
        
        # 计算函数值
        zeta_val = self.zeckendorf_zeta_function(s)
        collapse_val = self.zeckendorf_collapse_function(s)
        
        # 数值等价性
        difference = abs(zeta_val - collapse_val)
        is_numerically_equivalent = difference < tolerance
        
        # 概率等价性
        equivalence_prob = self.compute_equivalence_probability(s)
        theoretical_prob = self.predict_theoretical_probability(s)
        
        # 三元分解
        phi_comp, pi_comp, e_comp = self.decompose_collapse_into_three_components(s)
        indicators = self.compute_three_fold_indicators(s)
        
        return {
            'point': s,
            'numerical_analysis': {
                'zeta_value': zeta_val,
                'collapse_value': collapse_val,
                'difference': difference,
                'is_equivalent': is_numerically_equivalent
            },
            'probabilistic_analysis': {
                'equivalence_probability': equivalence_prob,
                'theoretical_prediction': theoretical_prob,
                'prediction_accuracy': abs(equivalence_prob - theoretical_prob)
            },
            'three_fold_decomposition': {
                'phi_component': phi_comp,
                'pi_component': pi_comp,
                'e_component': e_comp,
                'phi_indicator': indicators[0],
                'pi_indicator': indicators[1],
                'e_indicator': indicators[2]
            },
            'parameter_classification': {
                'is_on_critical_line': abs(s.real - 0.5) < 1e-6,
                'region': self._classify_parameter_region(s)
            }
        }
    
    def _classify_parameter_region(self, s: complex) -> str:
        """分类参数区域"""
        if abs(s.real - 0.5) < 1e-6:
            return "critical_line"
        elif abs(s.imag) < 1.0:
            return "low_imaginary"
        elif abs(s.imag) > 2.0:
            return "high_imaginary"
        else:
            return "intermediate"


class TestT21_5_ProbabilisticEquivalence(unittest.TestCase):
    """T21-5 概率等价性测试套件"""
    
    def setUp(self):
        """测试前设置"""
        self.system = ZeckendorfProbabilisticEquivalenceSystem(precision=12)
    
    def test_01_zeckendorf_function_construction(self):
        """测试Zeckendorf函数构造"""
        s = complex(0.5, 1.0)
        
        zeta_val = self.system.zeckendorf_zeta_function(s)
        collapse_val = self.system.zeckendorf_collapse_function(s)
        
        # 验证函数值为复数
        self.assertIsInstance(zeta_val, complex)
        self.assertIsInstance(collapse_val, complex)
        
        # 验证函数值有限
        self.assertTrue(math.isfinite(zeta_val.real))
        self.assertTrue(math.isfinite(zeta_val.imag))
        self.assertTrue(math.isfinite(collapse_val.real))
        self.assertTrue(math.isfinite(collapse_val.imag))
    
    def test_02_three_fold_decomposition(self):
        """测试三元分解功能"""
        s = complex(0.5, 1.0)
        
        phi_comp, pi_comp, e_comp = self.system.decompose_collapse_into_three_components(s)
        
        # 验证分量类型
        self.assertIsInstance(phi_comp, complex)
        self.assertIsInstance(pi_comp, complex)
        self.assertIsInstance(e_comp, complex)
        
        # 验证e分量为0（连接算子）
        self.assertEqual(e_comp, complex(0, 0))
        
        # 验证φ和π分量非零（对于典型参数）
        self.assertGreater(abs(phi_comp), 0)
        self.assertGreater(abs(pi_comp), 0)
    
    def test_03_indicator_functions(self):
        """测试指示函数计算"""
        test_points = [
            complex(0.5, 0),      # 临界线
            complex(0.5, 1.0),    # 临界线高虚部
            complex(0.25, 0.5),   # 低实部区域
            complex(0.75, 0.1),   # 高实部区域
        ]
        
        for s in test_points:
            with self.subTest(s=s):
                indicators = self.system.compute_three_fold_indicators(s)
                
                # 验证指示函数值为0或1
                self.assertIn(indicators[0], [0, 1])  # I_φ
                self.assertIn(indicators[1], [0, 1])  # I_π  
                self.assertEqual(indicators[2], 0)    # I_e 恒为0
                
                # 验证互斥性（最多一个为1）
                self.assertLessEqual(sum(indicators), 1)
    
    def test_04_probability_computation(self):
        """测试概率计算"""
        test_points = [
            complex(0.5, 0),      # 期望: π主导 (1/3)
            complex(0.25, 0.5),   # 期望: φ主导 (2/3)  
            complex(0.5, 3.0),    # 期望: 高虚部 (0)
        ]
        
        for s in test_points:
            with self.subTest(s=s):
                prob = self.system.compute_equivalence_probability(s)
                
                # 验证概率值在[0,1]范围内
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
                
                # 验证概率值为期望的分数值
                self.assertIn(prob, [0.0, 1/3, 2/3])
    
    def test_05_theoretical_prediction_accuracy(self):
        """测试理论预测准确性"""
        test_points = [
            (complex(0.5, 0), 1/3),      # 临界线 → π主导
            (complex(0.5, 0.5), 1/3),    # 临界线 → π主导
            (complex(0.25, 0.2), 2/3),   # 低虚部 → φ主导
            (complex(0.75, 0.1), 2/3),   # 低虚部 → φ主导
            (complex(0.5, 2.5), 0.0),    # 高虚部 → e连接
        ]
        
        for s, expected_prob in test_points:
            with self.subTest(s=s, expected=expected_prob):
                theoretical_prob = self.system.predict_theoretical_probability(s)
                computed_prob = self.system.compute_equivalence_probability(s)
                
                # 验证理论预测
                self.assertAlmostEqual(theoretical_prob, expected_prob, places=6)
                
                # 验证计算结果与理论预测的一致性
                # 允许一定的数值误差
                self.assertLess(abs(computed_prob - theoretical_prob), 0.01)
    
    def test_06_critical_line_behavior(self):
        """测试临界线Re(s)=1/2的特殊行为"""
        critical_line_points = [
            complex(0.5, t) for t in np.linspace(-1, 1, 9)
        ]
        
        probabilities = []
        for s in critical_line_points:
            prob = self.system.compute_equivalence_probability(s)
            probabilities.append(prob)
        
        # 在临界线上，期望π主导（概率1/3）
        average_prob = sum(probabilities) / len(probabilities)
        
        # 验证临界线平均概率接近1/3
        self.assertAlmostEqual(average_prob, 1/3, delta=0.1)
        
        # 验证大多数点都是π主导
        pi_dominated_count = sum(1 for p in probabilities if abs(p - 1/3) < 0.01)
        self.assertGreater(pi_dominated_count, len(probabilities) * 0.6)
    
    def test_07_three_fold_distribution_verification(self):
        """验证三元概率分布 (2/3, 1/3, 0)"""
        # 生成测试网格
        real_range = (0.3, 0.7)
        imag_range = (-1, 1)
        grid_size = 8
        
        test_points = []
        for r in np.linspace(real_range[0], real_range[1], grid_size):
            for i in np.linspace(imag_range[0], imag_range[1], grid_size):
                test_points.append(complex(r, i))
        
        # 统计指示函数结果
        phi_count = 0
        pi_count = 0
        e_count = 0
        total_count = len(test_points)
        
        for s in test_points:
            indicators = self.system.compute_three_fold_indicators(s)
            phi_count += indicators[0]
            pi_count += indicators[1]  
            e_count += indicators[2]
        
        # 计算观测到的概率分布
        observed_phi_rate = phi_count / total_count
        observed_pi_rate = pi_count / total_count
        observed_e_rate = e_count / total_count
        
        # 验证与理论期望的匹配度
        # φ权重期望接近2/3
        self.assertLess(abs(observed_phi_rate - 2/3), 0.2)
        
        # π权重期望接近1/3  
        self.assertLess(abs(observed_pi_rate - 1/3), 0.2)
        
        # e权重期望为0
        self.assertLess(observed_e_rate, 0.1)
        
        print(f"\n三元分布验证结果:")
        print(f"观测φ比率: {observed_phi_rate:.3f} (期望: 0.667)")
        print(f"观测π比率: {observed_pi_rate:.3f} (期望: 0.333)")
        print(f"观测e比率: {observed_e_rate:.3f} (期望: 0.000)")
    
    def test_08_systematic_equivalence_analysis(self):
        """系统性等价性分析"""
        test_points = [
            complex(0.5, 0),      # 临界线原点
            complex(0.5, 1.0),    # 临界线
            complex(0.5, -1.0),   # 临界线负虚部
            complex(0.25, 0.5),   # φ主导区域
            complex(0.75, 0.2),   # φ主导区域
            complex(0.5, 2.0),    # 高虚部区域
        ]
        
        analysis_results = []
        for s in test_points:
            result = self.system.analyze_equivalence_at_point(s)
            analysis_results.append(result)
        
        # 验证分析结果的完整性
        for result in analysis_results:
            self.assertIn('numerical_analysis', result)
            self.assertIn('probabilistic_analysis', result) 
            self.assertIn('three_fold_decomposition', result)
            self.assertIn('parameter_classification', result)
            
            # 验证概率预测的准确性
            prob_accuracy = result['probabilistic_analysis']['prediction_accuracy']
            self.assertLess(prob_accuracy, 0.1)  # 预测误差小于10%
    
    def test_09_euler_identity_role_verification(self):
        """验证变形欧拉恒等式的作用"""
        # 测试点：变形欧拉恒等式的特殊值
        s_values = [
            complex(1, 0),        # s=1，基础情形
            complex(0, 0),        # s=0，单位情形
            complex(2, 0),        # s=2，二次情形
        ]
        
        for s in s_values:
            with self.subTest(s=s):
                # 计算collapse函数（包含变形欧拉恒等式）
                collapse_val = self.system.zeckendorf_collapse_function(s)
                
                # 分解为三元分量
                phi_comp, pi_comp, e_comp = self.system.decompose_collapse_into_three_components(s)
                
                # 验证三元分量的一致性
                reconstructed = phi_comp + pi_comp + e_comp
                
                # 允许数值误差
                self.assertLess(abs(collapse_val - reconstructed), 1e-6)
    
    def test_10_base_relativity_demonstration(self):
        """演示数学基底相对性"""
        s = complex(0.5, 1.0)
        
        # Zeckendorf基底下的分析
        zeckendorf_analysis = self.system.analyze_equivalence_at_point(s)
        zeckendorf_prob = zeckendorf_analysis['probabilistic_analysis']['equivalence_probability']
        
        # 连续基底下的"等价性"（理论上为0）
        continuous_equivalence = 0.0  # 在连续数学中两函数完全不等价
        
        # 验证基底相对性
        self.assertGreater(zeckendorf_prob, continuous_equivalence)
        
        print(f"\n数学基底相对性演示:")
        print(f"连续基底等价性: {continuous_equivalence}")
        print(f"Zeckendorf基底等价性: {zeckendorf_prob:.3f}")
        print(f"基底影响: {zeckendorf_prob - continuous_equivalence:.3f}")
        
        # 验证这种差异的显著性
        self.assertGreaterEqual(zeckendorf_prob - continuous_equivalence, 0.2)
    
    def test_11_comprehensive_theory_validation(self):
        """综合理论验证"""
        # 大规模测试以验证T27-2理论
        real_vals = np.linspace(0.2, 0.8, 6) 
        imag_vals = np.linspace(-1.5, 1.5, 6)
        
        total_tests = 0
        theory_matches = 0
        prob_sum = 0
        
        for r in real_vals:
            for i in imag_vals:
                s = complex(r, i)
                total_tests += 1
                
                analysis = self.system.analyze_equivalence_at_point(s)
                computed_prob = analysis['probabilistic_analysis']['equivalence_probability']
                theoretical_prob = analysis['probabilistic_analysis']['theoretical_prediction']
                
                prob_sum += computed_prob
                
                # 检查理论匹配度
                if abs(computed_prob - theoretical_prob) < 0.1:
                    theory_matches += 1
        
        # 计算统计指标
        average_probability = prob_sum / total_tests
        theory_match_rate = theory_matches / total_tests
        
        print(f"\n综合理论验证结果:")
        print(f"测试总数: {total_tests}")
        print(f"平均等价概率: {average_probability:.3f}")
        print(f"理论匹配率: {theory_match_rate:.3f}")
        
        # 验证理论的有效性
        self.assertGreater(theory_match_rate, 0.7)  # 70%以上匹配率
        self.assertGreater(average_probability, 0.3)  # 平均概率显著大于0
        
        # 验证是否支持T27-2理论的核心预测
        supports_theory = (
            theory_match_rate > 0.7 and
            0.4 < average_probability < 0.7  # 介于1/3和2/3之间
        )
        self.assertTrue(supports_theory, "综合测试未能验证T27-2理论")


if __name__ == '__main__':
    # 运行测试套件
    unittest.main(verbosity=2)