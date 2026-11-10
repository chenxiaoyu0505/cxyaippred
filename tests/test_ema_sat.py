#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA-SAT 教师状态的行为测试。"""
import unittest

import numpy as np

from src.ema_sat import EMASAT


class TestEMASAT(unittest.TestCase):
    def test_first_update_uses_probability_directly(self):
        """首次命中应直接写入当前概率，而不是和 0.5 先验做衰减。"""
        sat = EMASAT(3, alpha=0.7, decay=0.9, warmup_epochs=0, enabled=True)
        sat.update(np.array([0]), np.array([0.9]))
        self.assertAlmostEqual(float(sat.ema[0]), 0.9, places=6)

    def test_subsequent_update_uses_decay(self):
        """第二次命中才使用 decay 滑动平均。"""
        sat = EMASAT(1, alpha=0.7, decay=0.9, warmup_epochs=0, enabled=True)
        sat.update([0], [0.9])
        sat.update([0], [0.1])
        self.assertAlmostEqual(float(sat.ema[0]), 0.82, places=6)

    def test_disabled_update_is_noop(self):
        """关闭时不应改写 EMA 状态。"""
        sat = EMASAT(1, alpha=0.7, decay=0.9, warmup_epochs=0, enabled=False)
        sat.update([0], [0.9])
        self.assertAlmostEqual(float(sat.ema[0]), 0.5, places=6)
        self.assertFalse(bool(sat.seen[0]))

    def test_targets_use_updated_ema_after_warmup(self):
        """warmup 结束后教师目标应为 alpha*硬标签 + (1-alpha)*EMA。"""
        sat = EMASAT(1, alpha=0.7, decay=0.9, warmup_epochs=0, enabled=True)
        sat.set_epoch(1)
        sat.update([0], [0.9])
        soft = sat.targets([0], np.array([1.0], dtype=np.float32))
        self.assertAlmostEqual(float(soft[0]), 0.7 * 1.0 + 0.3 * 0.9, places=6)


if __name__ == '__main__':
    unittest.main()
