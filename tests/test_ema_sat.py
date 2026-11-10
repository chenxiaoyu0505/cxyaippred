#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA-SAT 教师状态与非线性硬标签权重计划的行为测试。"""
import unittest

import numpy as np

from src.ema_sat import EMASAT


class TestEMASAT(unittest.TestCase):
    def test_first_update_uses_probability_directly(self):
        """首次命中应直接写入当前概率，而不是和 0.5 先验做衰减。"""
        sat = EMASAT(3, alpha_min=0.7, decay=0.9, start_epoch=0)
        sat.update(np.array([0]), np.array([0.9]))
        self.assertAlmostEqual(float(sat.ema[0]), 0.9, places=6)

    def test_subsequent_update_uses_decay(self):
        """第二次命中才使用 decay 滑动平均。"""
        sat = EMASAT(1, alpha_min=0.7, decay=0.9, start_epoch=0)
        sat.update([0], [0.9])
        sat.update([0], [0.1])
        self.assertAlmostEqual(float(sat.ema[0]), 0.82, places=6)

    def test_disabled_update_is_noop(self):
        """关闭时不应改写 EMA 状态。"""
        sat = EMASAT(1, alpha_min=0.7, decay=0.9, start_epoch=0, enabled=False)
        sat.update([0], [0.9])
        self.assertAlmostEqual(float(sat.ema[0]), 0.5, places=6)
        self.assertFalse(bool(sat.seen[0]))

    def test_before_start_uses_hard_labels_and_does_not_update(self):
        """start_epoch 之前只给硬标签；start_epoch-1 之前不更新教师。"""
        sat = EMASAT(1, alpha_min=0.7, decay=0.65, start_epoch=2)
        sat.set_epoch(0)
        sat.update([0], [0.9])
        self.assertFalse(bool(sat.seen[0]))
        self.assertAlmostEqual(float(sat.targets([0], [1.0])[0]), 1.0, places=6)
        sat.set_epoch(1)
        sat.update([0], [0.9])
        self.assertTrue(bool(sat.seen[0]))

    def test_halving_alpha_schedule(self):
        """α 应从 1 开始每轮减半地逼近 alpha_min；β=1−α。"""
        sat = EMASAT(1, alpha_min=0.7, decay=0.65, start_epoch=2, tau=1.0)
        sat.set_epoch(1)
        sat.update([0], [0.5])
        sat.set_epoch(2)
        alpha2 = 0.85
        beta2 = 0.15
        self.assertAlmostEqual(sat.alpha_at(), alpha2, places=6)
        self.assertAlmostEqual(sat.beta_at(), beta2, places=6)
        soft = sat.targets([0], np.array([1.0], dtype=np.float32))
        self.assertAlmostEqual(float(soft[0]), alpha2 * 1.0 + (1.0 - alpha2) * 0.5,
                               places=6)
        sat.set_epoch(3)
        alpha3 = 0.775
        self.assertAlmostEqual(sat.alpha_at(), alpha3, places=6)
        self.assertGreater(alpha2, alpha3)
        self.assertGreater(alpha3, 0.7)

    def test_alpha_decreases_beta_increases_to_limits(self):
        sat = EMASAT(1, alpha_min=0.7, decay=0.65, start_epoch=2, tau=1.0)
        alphas, betas = [], []
        for epoch in range(0, 30):
            sat.set_epoch(epoch)
            alphas.append(sat.alpha_at())
            betas.append(sat.beta_at())
        self.assertTrue(all(a == 1.0 for a in alphas[:2]))
        self.assertTrue(all(b == 0.0 for b in betas[:2]))
        self.assertTrue(all(alphas[i] >= alphas[i + 1] - 1e-12
                            for i in range(len(alphas) - 1)))
        self.assertTrue(all(betas[i] <= betas[i + 1] + 1e-12
                            for i in range(len(betas) - 1)))
        self.assertTrue(all(a > 0.7 - 1e-12 for a in alphas))
        self.assertTrue(all(b < 0.3 + 1e-12 for b in betas))
        self.assertGreater(betas[10], 0.299)


if __name__ == '__main__':
    unittest.main()
