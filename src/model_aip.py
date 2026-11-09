#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单任务 AIP 模型头（不含 ESM-2）。

设计要点
--------
* ESM-2 是冻结的、所有 fold 共用同一份，因此**不进入权重文件**。
  权重文件里只保存本文件定义的 AIPHead（约 21M 参数 / 85 MB）。
* 5 折 = 5 个 AIPHead（各 ~85 MB），推理时共享一次 ESM-2 前向。
* 结构沿用原框架：ResCNN(1280) → 掩码均值池化 → 拼接 694 维手工特征 → MLP。

用法
----
    head = AIPHead(embed_dim=1280, hand_dim=694, hidden=512)
    logits = head(hidden_states, attention_mask, hand_feats)   # [B]
    probs = torch.sigmoid(logits)
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D 残差块（与 legacy_multitask/models.py 一致）"""

    def __init__(self, ch, ks=3, dil=1, drop=0.1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        return self.relu(out + x)


class AIPHead(nn.Module):
    """ESM 表示 + 手工特征 → AIP 概率（不含 ESM 本体）"""

    def __init__(self, embed_dim=1280, hand_dim=694, hidden=512,
                 n_res=2, cnn_drop=0.2, mlp_drop=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hand_dim = hand_dim
        self.hidden = hidden
        self.n_res = n_res
        self.cnn_drop = cnn_drop
        self.mlp_drop = mlp_drop

        self.cnn = nn.Sequential(*[ResBlock1D(embed_dim, drop=cnn_drop) for _ in range(n_res)])
        self.shared = nn.Sequential(
            nn.Linear(embed_dim + hand_dim, hidden), nn.ReLU(), nn.Dropout(mlp_drop)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(mlp_drop), nn.Linear(hidden, 1)
        )

    def forward(self, hidden_states, attention_mask, hand_feats=None):
        """
        hidden_states : [B, L, D]  ESM-2 最后一层隐状态
        attention_mask: [B, L]     1 = 有效残基
        hand_feats    : [B, hand_dim] 或 None（hand_dim > 0 时必需）
        """
        mask = attention_mask.bool()
        x = self.cnn(hidden_states.transpose(1, 2))                    # [B, D, L]
        cnt = mask.unsqueeze(1).float().sum(dim=2).clamp(min=1.0)      # [B, 1]
        pooled = (x * mask.unsqueeze(1).float()).sum(dim=2) / cnt      # [B, D]

        if self.hand_dim > 0 and hand_feats is not None:
            pooled = torch.cat([pooled, hand_feats], dim=1)

        return self.out(self.shared(pooled)).squeeze(1)                # [B]

    @torch.no_grad()
    def predict_proba(self, hidden_states, attention_mask, hand_feats=None):
        self.eval()
        return torch.sigmoid(self.forward(hidden_states, attention_mask, hand_feats))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def size_mb(self):
        n = sum(p.numel() * p.element_size() for p in self.parameters())
        n += sum(b.numel() * b.element_size() for b in self.buffers())
        return n / 1024 ** 2
