#!/usr/bin/env python3
"""
Quick test for D1.6: Entropy Definition
"""

import unittest
import math
from typing import Set


class QuickEntropyTest(unittest.TestCase):
    """Quick tests for entropy formula H = ln(|S_t|)"""
    
    def test_basic_formula(self):
        """Test H = ln(|S_t|)"""
        # Empty set
        self.assertEqual(math.log(1) if 1 > 0 else 0, 0.0)
        
        # Various sizes
        for n in [1, 2, 3, 5, 8, 13]:
            S = {str(i) for i in range(n)}
            H = math.log(n) if n > 0 else 0
            self.assertAlmostEqual(H, math.log(len(S)))
    
    def test_entropy_increase_formula(self):
        """Test ΔH = ln(growth rate)"""
        test_cases = [
            (2, 3),    # Growth rate 1.5
            (3, 5),    # Growth rate 5/3
            (5, 8),    # Growth rate 8/5
            (8, 13),   # Growth rate 13/8
        ]
        
        for n1, n2 in test_cases:
            H1 = math.log(n1)
            H2 = math.log(n2)
            delta_H = H2 - H1
            growth_rate = n2 / n1
            
            self.assertAlmostEqual(delta_H, math.log(growth_rate))
    
    def test_golden_ratio_limit(self):
        """Test that growth rate approaches φ"""
        phi = (1 + math.sqrt(5)) / 2
        
        # Fibonacci ratios approach φ
        fib = [1, 1]
        for _ in range(20):
            fib.append(fib[-1] + fib[-2])
        
        # Check last few ratios
        for i in range(-5, -1):
            ratio = fib[i] / fib[i-1]
            self.assertAlmostEqual(ratio, phi, places=4)
        
        # Therefore entropy increase approaches ln(φ)
        ln_phi = math.log(phi)
        self.assertAlmostEqual(ln_phi, 0.4812, places=4)
    
    def test_shannon_relation(self):
        """Test relation to Shannon entropy"""
        n = 16  # 16 states
        
        # Our entropy (nats)
        H_nats = math.log(n)
        
        # Shannon entropy (bits)
        H_bits = math.log2(n)
        
        # Conversion factor
        self.assertAlmostEqual(H_nats, H_bits * math.log(2))
        self.assertAlmostEqual(H_bits, H_nats / math.log(2))


if __name__ == '__main__':
    unittest.main(verbosity=2)