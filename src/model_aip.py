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

padding 处理（2026-11 修正）
---------------------------
旧实现把补到 batch 最长的整条张量直接喂进卷积，归一化用 BatchNorm1d：

1. 卷积核 ks=3、两层共 4 次卷积，首尾各约 2 个真实残基会"看见" <pad> 的隐状态；
2. BatchNorm1d 在 [B, C, L] 上统计均值方差，把 padding 位置也算进去，
   补零长度一变（换 batch 划分、单条推理 vs 批量推理），归一化结果跟着变。

后果是同一条肽的预测依赖它和谁同批：实测同一 15 残基肽的 logit
单独推理 -0.002885、与长肽同批 -0.002574，差 10.75%。

现在改为：卷积前后都把 pad 位置置零（CNN 里的标准做法），归一化默认换成
逐位置的 LayerNorm —— 只看该位置自己的通道，与 padding 完全无关。

归一化是可选的（`norm=`），因为"修 bug"和"换归一化轴"是两个变量：

    layer         逐位置 LayerNorm（默认，与 padding 无关）
    peptide_bn    masked BN + 按肽等权（掩掉 padding，每条肽投一票）
    masked_batch  masked BN，按残基位置等权（长肽权重更大）
    batch         原生 BatchNorm1d（padding 进统计量，仅供 v1 权重/消融）

用法
----
    head = AIPHead(embed_dim=1280, hand_dim=694, hidden=512)
    logits = head(hidden_states, attention_mask, hand_feats)   # [B]
    probs = torch.sigmoid(logits)
"""
import torch
import torch.nn as nn


class ChannelNorm(nn.Module):
    """在 [B, C, L] 上做"逐位置、跨通道"归一化。

    LayerNorm 归一化最后一维，而 Conv1d 输出是 [B, C, L]，直接套
    nn.LayerNorm(C) 会去归一化 L 轴，所以先转置成 [B, L, C]。
    """

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x, m=None):                                  # [B, C, L]
        # m 只是为了和 MaskedBatchNorm1d 用同一套签名；LayerNorm 与 padding 无关
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MaskedBatchNorm1d(nn.Module):
    """[B, C, L] 上的 BatchNorm，但统计量只由**真实残基**贡献。

    两种口径
    --------
    per_peptide=False（norm='masked_batch'）
        分母是有效残基总数：长肽在统计量里权重更大（73 残基 vs 5 残基差 15 倍）。
    per_peptide=True（norm='peptide_bn'）
        每条肽先算自己的均值/方差，再按肽等权汇总 —— 一条 73 残基的肽和
        一条 5 残基的肽各投一票。方差按

            E_b[Var_l(x)] + Var_b(Mean_l(x))

        分解，两种口径都成立，只是 b 上的权重不同。

    注意：训练阶段统计量仍来自当前 batch（这是 BN 的定义属性，不打算改）。
    换来的是梯度里的随机性 —— 既是噪声也是正则。
    """

    def __init__(self, ch, per_peptide=True, momentum=0.1, eps=1e-5):
        super().__init__()
        self.ch = ch
        self.per_peptide = per_peptide
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.register_buffer('running_mean', torch.zeros(ch))
        self.register_buffer('running_var', torch.ones(ch))

    def forward(self, x, m=None):                                  # [B, C, L]
        if m is None:
            m = torch.ones(x.shape[0], 1, x.shape[2],
                           dtype=x.dtype, device=x.device)
        mb = m.sum(dim=2).clamp(min=1.0)                           # [B, 1]

        if self.training:
            mu_b = (x * m).sum(dim=2) / mb                          # [B, C]
            if self.per_peptide:
                mean = mu_b.mean(dim=0)                             # [C]
                var_b = ((x - mu_b.unsqueeze(2)) ** 2 * m).sum(dim=2) / mb
                var = var_b.mean(dim=0) + (mu_b - mean).pow(2).mean(dim=0)
                n = x.new_tensor(float(x.shape[0]))          # 肽数
            else:
                tot = m.sum()
                mean = (x * m).sum(dim=(0, 2)) / tot
                var = ((x - mean.view(1, -1, 1)) ** 2 * m).sum(dim=(0, 2)) / tot
                n = tot
            with torch.no_grad():
                # 不用 `if n > 1`：n 是 tensor 时那会触发一次 host 同步（MPS 上很贵）
                unbiased = var * (n / torch.clamp(n - 1, min=1.0))
                mom = self.momentum
                self.running_mean.mul_(1.0 - mom).add_(mean.detach(), alpha=mom)
                self.running_var.mul_(1.0 - mom).add_(unbiased.detach(), alpha=mom)
        else:
            mean, var = self.running_mean, self.running_var

        y = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        y = y * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return y * m


def _norm_fwd(layer, x, m):
    """ChannelNorm / MaskedBatchNorm1d 吃掩码；nn.BatchNorm1d 不吃。"""
    if isinstance(layer, (ChannelNorm, MaskedBatchNorm1d)):
        return layer(x, m)
    return layer(x)


class ResBlock1D(nn.Module):
    """1D 残差块（卷积核/膨胀率与 legacy_multitask/models.py 一致）

    norm 取值
    ---------
    'layer'        （默认）逐位置 LayerNorm：只看该位置自己的通道，与 padding 无关。
    'peptide_bn'   masked BN + 按肽等权：掩掉 padding，每条肽在统计量里投一票。
    'masked_batch' masked BN，但按残基位置等权（长肽权重更大）。
    'batch'        原生 nn.BatchNorm1d：padding 位置也进统计量，只为加载 v1 权重 /
                   做"归一化轴"消融保留。
    """

    VALUES = ('layer', 'peptide_bn', 'masked_batch', 'batch')

    def __init__(self, ch, ks=3, dil=1, drop=0.1, norm='layer'):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(ch)
            self.norm2 = nn.BatchNorm1d(ch)
        elif norm == 'layer':
            self.norm1 = ChannelNorm(ch)
            self.norm2 = ChannelNorm(ch)
        elif norm in ('peptide_bn', 'masked_batch'):
            per_peptide = (norm == 'peptide_bn')
            self.norm1 = MaskedBatchNorm1d(ch, per_peptide=per_peptide)
            self.norm2 = MaskedBatchNorm1d(ch, per_peptide=per_peptide)
        else:
            raise ValueError("norm 只能是 %s 之一，收到 %r"
                             % ('/'.join(ResBlock1D.VALUES), norm))
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x, m=None):
        """x: [B, C, L]；m: [B, 1, L] 的 0/1 掩码（1 = 真实残基），可为 None"""
        y = self.relu(_norm_fwd(self.norm1, self.conv1(x), m))
        if m is not None:
            y = y * m
        y = self.drop(_norm_fwd(self.norm2, self.conv2(y), m))
        if m is not None:
            y = y * m
        out = self.relu(y + x)
        return out * m if m is not None else out


class AIPHead(nn.Module):
    """ESM 表示 + 手工特征 → AIP 概率（不含 ESM 本体）"""

    def __init__(self, embed_dim=1280, hand_dim=694, hidden=512,
                 n_res=2, cnn_drop=0.2, mlp_drop=0.3, norm='layer'):
        super().__init__()
        self.embed_dim = embed_dim
        self.hand_dim = hand_dim
        self.hidden = hidden
        self.n_res = n_res
        self.cnn_drop = cnn_drop
        self.mlp_drop = mlp_drop
        self.norm = norm

        # 用 ModuleList 而非 Sequential：每个 block 要把掩码透传下去
        self.cnn = nn.ModuleList([ResBlock1D(embed_dim, drop=cnn_drop, norm=norm)
                                  for _ in range(n_res)])
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
        if attention_mask is None:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool,
                              device=hidden_states.device)
        else:
            mask = attention_mask.bool()
        m = mask.unsqueeze(1).to(hidden_states.dtype)               # [B, 1, L]

        x = hidden_states.transpose(1, 2) * m                       # pad 隐状态先清零
        for block in self.cnn:
            x = block(x, m)                                         # 每次卷积后重新置零

        cnt = m.sum(dim=2).clamp(min=1.0)                           # [B, 1]
        pooled = (x * m).sum(dim=2) / cnt                           # [B, D]

        if self.hand_dim > 0 and hand_feats is not None:
            pooled = torch.cat([pooled, hand_feats], dim=1)

        return self.out(self.shared(pooled)).squeeze(1)             # [B]

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
