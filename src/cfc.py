#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
闭式连续时间网络（CfC）—— 逐残基递归的序列支路。

为什么不装 ncps
--------------
`ncps` 依赖 torchdiffeq，而 CfC 的卖点恰恰是"不需要 ODE solver"。这里只用到
它的闭式解，三行数学，没必要为它引入两个依赖。公式（Hasani et al., 2022,
Nature Machine Intelligence）：

    z_t     = [x_t ; h_{t-1}]
    decay_t = sigmoid(-f(z_t) * dt)                     # 逐位置、逐通道，∈ (0,1)
    h_t     = decay_t ⊙ g(z_t) + (1 - decay_t) ⊙ u(z_t)

`decay` 才是 "liquid" 二字的来源：它不是一个可学的常数，而是让**有效**时间
常数 1/(1/τ + f) 随输入变化。残基 t 对 t-k 的记忆权重以 decay 连乘衰减，
等效半衰期

    t½ = ln2 / (-ln decay)

decay=0.5 → 1.0 个残基，0.9 → 6.6，0.99 → 69。所以这条支路能表达"隔多远还算
邻居"这种边界模糊的距离约束 —— ks=3 的 CNN 看不到，全序列均值池化的 ESM-2
也表达不了。

成本
----
状态必须逐位置递推（闭式解省掉的是 ODE 求解器，不是递推）。为了不让 Python
循环成为瓶颈，把 z 里只依赖 x_t 的那一半（W_f x_t 等）按整批预算掉，循环里
只剩 128×128 的小矩阵乘。

注意：这是**逐肽独立**的递归，每条肽从零状态开始；训练与推理都不跨肽传递状态，
所以预测与 batch 划分无关（有测试守着）。
"""
import torch
import torch.nn as nn


class CfCBlock(nn.Module):
    """输入 [B, L, in_dim] → 隐状态 [B, L, hidden] 与门控 decay [B, L, hidden]"""

    def __init__(self, in_dim=20, hidden=128, dt=1.0, dt_learnable=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)

        # x 侧（整批一次算完）
        self.x_f = nn.Linear(in_dim, hidden)
        self.x_g = nn.Linear(in_dim, hidden)
        self.x_u = nn.Linear(in_dim, hidden)
        # h 侧（每步才算）
        self.h_f = nn.Linear(hidden, hidden, bias=False)
        self.h_g = nn.Linear(hidden, hidden, bias=False)
        self.h_u = nn.Linear(hidden, hidden, bias=False)

        dt = float(dt)
        if dt_learnable:
            self.dt = nn.Parameter(torch.tensor(dt))
        else:
            self.register_buffer('dt', torch.tensor(dt))

    def forward(self, x, mask):
        """
        x    : [B, L, in_dim]  实值（one-hot 或已嵌入）
        mask : [B, L]          1 = 真实残基；0 的位置**不更新状态**
        返回 (h [B, L, hidden], decay [B, L, hidden])
        """
        b, l, _ = x.shape
        # 只递推到本 batch 里最长的那条肽：后面全是 padding，状态不会更新，
        # 算出来也没人用。中位长度 15aa 而 lmax=75，这一步省掉大量空转。
        # 注意取的是"最后一个有效位置的下标 + 1"，不是有效位置计数：
        # predict.py 的 token 网格残基从 1 开始，用计数会少递推最后一位。
        if l:
            pos = torch.arange(l, device=x.device, dtype=torch.float32)
            l_eff = int((mask.to(torch.float32) * pos).max().item()) + 1
            l_eff = max(1, min(int(l), l_eff))
        else:
            l_eff = 1
        h = x.new_zeros(b, self.hidden)
        xf = self.x_f(x[:, :l_eff]); xg = self.x_g(x[:, :l_eff]); xu = self.x_u(x[:, :l_eff])
        outs, decs = [], []
        for t in range(l_eff):
            f = xf[:, t] + self.h_f(h)
            g = xg[:, t] + self.h_g(h)
            u = xu[:, t] + self.h_u(h)
            decay = torch.sigmoid(-f * self.dt)
            h_new = decay * torch.tanh(g) + (1.0 - decay) * torch.tanh(u)
            m = mask[:, t].unsqueeze(1).to(h_new.dtype)
            h = h_new * m + h * (1.0 - m)          # padding 保持原状态
            outs.append(h)
            decs.append(decay)
        return torch.stack(outs, dim=1), torch.stack(decs, dim=1)

    @staticmethod
    def half_life(decay, eps=1e-6):
        """按位点换算半衰期（单位：残基）。decay→1 时返回一个大数而非 inf。"""
        d = decay.clamp(min=eps, max=1.0 - eps)
        ln2 = torch.log(torch.tensor(2.0, dtype=d.dtype, device=d.device))
        return ln2 / (-torch.log(d))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
