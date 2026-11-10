#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMA-SAT（自自适应训练）的教师目标。

目标定义
--------
    q_i^(e) = ρ · q_i^(e−1) + (1−ρ) · p_i^(e)        # 逐样本历史预测
    t_i^(e) = α_e · y_i + (1 − α_e) · q_i^(e−1)      # 教师标签

α_e 是**硬标签权重**，不是教师权重。它从 1 开始，在 start_epoch
之后快速下降并渐近逼近 α_min（默认 0.7）；等价地，教师权重
β_e = 1 − α_e 从 0 快速上升并渐近逼近 0.3。这里采用指数饱和：

    α_e = α_min + (1 − α_min) · exp(−(e − start_epoch + 1) / τ)

例如 α_min=0.7、start_epoch=2、τ=1.0 时：
    e=2: α=0.810, e=3: 0.741, e=4: 0.715, e=5: 0.706, ... → 0.700
    β:   0.190,      0.259,      0.285,      0.294,     ... → 0.300

注意：α_min=0.7 表示最终 70% 的目标来自硬标签、30% 来自教师；
不要把它误设成教师权重 0.7。若想要更强的教师，可显式降低
`--ema-alpha-min`（例如 0.5 表示教师上限 0.5）。

实现要点
--------
* 缓冲区按**全局行号**索引（4,686 个），不是按 batch 内的位置。
* `start_epoch` 之前只用硬标签；`start_epoch-1` 这一轮末才初始化教师。
* 完全可关闭：enabled=False 时 targets() 原样返回硬标签。
* 推荐在 epoch 末用 eval 模式对 train 全集预测一次，再统一更新。
"""
import math

import numpy as np
import torch


class EMASAT:
    def __init__(self, n_samples, alpha_min=0.7, decay=0.65, start_epoch=2,
                 tau=1.0, enabled=True, dtype=np.float32):
        self.alpha_min = float(np.clip(alpha_min, 0.0, 1.0)) if enabled else 1.0
        self.decay = float(decay)
        self.start_epoch = int(start_epoch) if enabled else 0
        self.tau = max(float(tau), 1e-6)
        self.enabled = bool(enabled)
        self.n = int(n_samples)
        self.ema = np.full(self.n, 0.5, dtype=dtype)
        self.seen = np.zeros(self.n, dtype=bool)
        self.epoch = 0

    # ------------------------------------------------------------ 状态
    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def active(self):
        """当前是否真的在用教师目标（关闭或仍在预热时返回 False）"""
        return self.enabled and self.epoch >= self.start_epoch

    def alpha_at(self, epoch=None):
        """当前 epoch 的硬标签权重；从 1 指数下降到 alpha_min。"""
        if not self.enabled:
            return 1.0
        e = self.epoch if epoch is None else int(epoch)
        if e < self.start_epoch:
            return 1.0
        delta = e - self.start_epoch + 1
        return self.alpha_min + (1.0 - self.alpha_min) * math.exp(-delta / self.tau)

    def beta_at(self, epoch=None):
        """当前 epoch 的教师权重；= 1 - alpha，渐近 1-alpha_min。"""
        if not self.enabled:
            return 0.0
        e = self.epoch if epoch is None else int(epoch)
        if e < self.start_epoch:
            return 0.0
        return 1.0 - self.alpha_at(e)

    def reset(self):
        self.ema[:] = 0.5
        self.seen[:] = False
        self.epoch = 0

    # ------------------------------------------------------------ 前向之前
    def targets(self, idx, hard):
        """
        idx : [B] int64  全局行号
        hard: [B] float  0/1 硬标签（tensor 或 ndarray）
        return: [B] float tensor，教师标签
        """
        if isinstance(hard, torch.Tensor):
            y = hard.detach().float().cpu().numpy()
        else:
            y = np.asarray(hard, dtype=np.float32)
        if not self.active:
            return torch.as_tensor(y, dtype=torch.float32)

        i = np.asarray(idx, dtype=np.int64)
        alpha = self.alpha_at()
        soft = alpha * y + (1.0 - alpha) * self.ema[i]
        # 没见过的样本（理论上不会发生，除非新数据）保持硬标签
        soft = np.where(self.seen[i], soft, y)
        return torch.as_tensor(soft.astype(np.float32), dtype=torch.float32)

    # ------------------------------------------------------------ 更新（推荐 epoch 末）
    def update(self, idx, probs):
        if not self.enabled:
            return
        # start_epoch-1 这一轮末才初始化教师；更早的轮次不写。
        if self.epoch < self.start_epoch - 1:
            return
        i = np.asarray(idx, dtype=np.int64)
        p = probs.detach().float().cpu().numpy() if isinstance(probs, torch.Tensor) \
            else np.asarray(probs, dtype=np.float32)
        p = np.clip(p, 1e-3, 1.0 - 1e-3)

        # 首次命中直接写入当前预测；否则 0.5 先验会一直稀释 EMA。
        fresh = ~self.seen[i]
        self.ema[i[fresh]] = p[fresh]
        old = ~fresh
        self.ema[i[old]] = (self.decay * self.ema[i[old]]
                            + (1.0 - self.decay) * p[old])
        self.seen[i] = True

    # ------------------------------------------------------------ 诊断
    def stats(self):
        m = self.seen
        if not m.any():
            return {'ema_mean': 0.0, 'ema_std': 0.0, 'seen': 0,
                    'alpha': self.alpha_at(), 'beta': self.beta_at()}
        return {'ema_mean': float(self.ema[m].mean()),
                'ema_std': float(self.ema[m].std()),
                'seen': int(m.sum()),
                'alpha': self.alpha_at(),
                'beta': self.beta_at()}
