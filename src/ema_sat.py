#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMA-SAT（自自适应训练）的教师目标。

原始定义（旧代码 EMA_SATManager）
--------------------------------
    ema_i  ← decay · ema_i + (1 − decay) · p_i        # 逐样本概率的滑动平均
    soft_i = alpha · y_i + (1 − alpha) · ema_i        # 教师标签
默认 decay = 0.99、alpha = 0.7 —— 注意 **alpha=0.7 意味着 70% 仍是硬标签**，
只有 30% 来自历史预测，不要把"自蒸馏"想得太强。

这里的实现要点
--------------
* 缓冲区按**全局行号**索引（4,686 个），不是按 batch 内的位置。
  否则同一肽在不同 fold 里会被当成不同样本，EMA 记忆全错。
* 前 warmup_epochs 轮只用硬标签（EMA 尚未收敛，掺进去是噪声）。
* 完全可关闭：enabled=False 时 targets() 原样返回硬标签，等价于普通训练。

用法
----
    sat = EMASAT(n_samples, alpha=0.7, decay=0.99, warmup_epochs=3, enabled=True)
    sat.set_epoch(epoch)
    soft = sat.targets(idx, y_batch)      # 前向之前
    ... loss = BCE(logits, soft) ...
    sat.update(idx, probs.detach())       # 反向之后
"""
import numpy as np
import torch


class EMASAT:
    def __init__(self, n_samples, alpha=0.7, decay=0.99, warmup_epochs=3,
                 enabled=True, dtype=np.float32):
        self.alpha = float(alpha) if enabled else 1.0
        self.decay = float(decay)
        self.warmup_epochs = int(warmup_epochs) if enabled else 0
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
        """当前是否真的在用 EMA（关闭或仍在 warmup 时返回 False）"""
        return self.enabled and self.epoch >= self.warmup_epochs

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
        soft = self.alpha * y + (1.0 - self.alpha) * self.ema[i]
        # 没见过的样本（理论上不会发生，除非新数据）保持硬标签
        soft = np.where(self.seen[i], soft, y)
        return torch.as_tensor(soft.astype(np.float32), dtype=torch.float32)

    # ------------------------------------------------------------ 反向之后
    def update(self, idx, probs):
        i = np.asarray(idx, dtype=np.int64)
        p = probs.detach().float().cpu().numpy() if isinstance(probs, torch.Tensor) \
            else np.asarray(probs, dtype=np.float32)
        self.ema[i] = self.decay * self.ema[i] + (1.0 - self.decay) * p
        self.seen[i] = True

    # ------------------------------------------------------------ 诊断
    def stats(self):
        m = self.seen
        if not m.any():
            return {'ema_mean': 0.0, 'ema_std': 0.0, 'seen': 0}
        return {'ema_mean': float(self.ema[m].mean()),
                'ema_std': float(self.ema[m].std()),
                'seen': int(m.sum())}
