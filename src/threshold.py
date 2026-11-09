#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值搜索。

为什么单独一个文件
------------------
旧稿最脆弱的地方就在这里：PIP 的 MCC 0.3497 是在验证集上搜到 0.689 才得到的，
同一批概率在 0.5 阈值下只有 0.2543（基线 0.246）。换句话说"+0.10"是阈值挑出来的。

所以本项目的规矩是：
* **主指标**用与阈值无关的 AUROC / AUPR（train 的早停也用它）。
* F1 / MCC / Acc 这类阈值相关指标，必须**同时**报三个口径：
  `@0.5`、`@val`（验证集搜出）、以及搜索得到的阈值本身。
* 阈值只在**验证集**上搜，测试集只用来评估——绝不在 test 上调。

用法
----
    r = search_best_mcc(y_val, p_val, lo=0.10, hi=0.99, n=199)
    r['threshold'], r['mcc']
"""
import numpy as np


def _mcc(tp, tn, fp, fn):
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def search_best_mcc(y_true, prob, lo=0.10, hi=0.99, n=199, return_curve=False):
    """
    在 [lo, hi] 上等距取 n 个阈值，返回 MCC 最大的那个。
    与旧实现一致（np.linspace(0.10, 0.99, 199)），便于对照。
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError('y 与 prob 长度不一致: %s vs %s' % (y.shape, p.shape))

    grid = np.linspace(lo, hi, n)
    best_t, best_mcc = 0.5, -2.0
    curve = np.zeros((n, 5))          # t, mcc, tp, fp, fn
    for k, t in enumerate(grid):
        pred = (p >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        m = _mcc(tp, tn, fp, fn)
        curve[k] = (t, m, tp, fp, fn)
        if m > best_mcc:
            best_t, best_mcc = float(t), m

    out = {'threshold': best_t, 'mcc': best_mcc, 'n_grid': n}
    if return_curve:
        out['curve'] = curve
    return out


def apply_threshold(prob, t):
    return (np.asarray(prob, dtype=np.float64) >= t).astype(int)
