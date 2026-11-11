#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单任务 AIP 模型头（不含 ESM-2）。

设计要点
--------
* ESM-2 是冻结的、所有 fold 共用同一份，因此**不进入权重文件**。
  权重文件里只保存本文件定义的 AIPHead（约 20.96M 参数 / 80 MB）。
* 5 折 = 5 个 AIPHead（各 ~80 MB），推理时共享一次 ESM-2 前向。
* 结构沿用原框架：ResCNN(1280) → 掩码均值池化 → 拼接 704 维手工特征 → MLP。

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

    layer         逐位置 LayerNorm（ResBlock 单用时的默认，与 padding 无关）
    peptide_bn    masked BN + 按肽等权（训练默认；掩掉 padding，每条肽投一票）
    masked_batch  masked BN，按残基位置等权（长肽权重更大）
    batch         原生 BatchNorm1d（padding 进统计量，仅供 v1 权重/消融）

用法
----
    head = AIPHead(embed_dim=1280, hand_dim=704, hidden=512,
                   norm='peptide_bn')
    logits = head(hidden_states, attention_mask, hand_feats)   # [B]
    probs = torch.sigmoid(logits)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfc import CfCBlock


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


class ProtoBranch(nn.Module):
    """阳性/阴性多原型旁路：拼接特征 → 小投影 → 每类 K 个原型 → 分数 B。

    B = logsumexp(cos(z, P_pos)/τ) - logsumexp(cos(z, P_neg)/τ)

    它和主分支吃同一份特征，因此**不引入新信息**；作用是把特征空间往
    "类内多峰"的结构上推，并给出可按原型拆解的解释。α 固定为常量，
    避免 α 与 B 的尺度不可辨识，也避免 α→0 让这条支路免费搭车。
    """

    def __init__(self, in_dim, n_pos=2, n_neg=2, proj_dim=128, tau=0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.proj_dim = int(proj_dim)
        self.tau = float(tau)
        if self.n_pos < 1 or self.n_neg < 1:
            raise ValueError('ProtoBranch 要求每类至少 1 个原型')
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim),
                                  nn.LayerNorm(proj_dim))
        self.p_pos = nn.Parameter(torch.randn(n_pos, proj_dim) * 0.02)
        self.p_neg = nn.Parameter(torch.randn(n_neg, proj_dim) * 0.02)

    def forward(self, feat):
        z = F.normalize(self.proj(feat), dim=1)
        sp = z @ F.normalize(self.p_pos, dim=1).t() / self.tau   # [B, K_pos]
        sn = z @ F.normalize(self.p_neg, dim=1).t() / self.tau   # [B, K_neg]
        b = torch.logsumexp(sp, dim=1) - torch.logsumexp(sn, dim=1)
        return b, sp, sn

    def diversity(self):
        """类内原型两两余弦的平方和；越接近 0，原型越分散（防塌缩）。"""
        loss = 0.0
        for P in (self.p_pos, self.p_neg):
            if P.shape[0] < 2:
                continue
            pn = F.normalize(P, dim=1)
            gram = pn @ pn.t()
            off = gram - torch.diag_embed(torch.diagonal(gram))
            loss = loss + (off ** 2).sum()
        return loss


class AIPHead(nn.Module):
    """ESM 表示 [+ CfC 序列支路] + 手工特征 → AIP 概率（不含 ESM 本体）

    CfC 支路（lnn=True）
    -------------------
        one-hot [B, L, 20] → CfCBlock(hidden=lnn_dim) → 掩码均值池化
                          → LayerNorm → l [B, lnn_dim]

    拼接顺序固定为 [ESM 池化 1280] + [l lnn_dim] + [手工特征 704]：手工特征永远
    在最后，`hand_dim` 的切片语义不变。

    为什么不加 router：静态的 (w_c, w_l, w_f) 会被下一层 Linear 完全吸收
    （W(w·x) = (wW)x），权重不随样本变化就等于没加。先把支路接上、确认它带的
    信息超不超噪声，再谈门控。
    """

    def __init__(self, embed_dim=1280, hand_dim=704, hidden=512,
                 n_res=2, cnn_drop=0.2, mlp_drop=0.3, norm='peptide_bn',
                 lnn=False, lnn_dim=128, lnn_dt=1.0,
                 use_esm=True, use_hand=True,
                 esm_layers=1, esm_fuse='weighted',
                 proto_pos=0, proto_neg=0, proto_dim=128, proto_tau=0.1,
                 proto_alpha=1.0, fuse='early', fused_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hand_dim = hand_dim
        self.hidden = hidden
        self.n_res = n_res
        self.cnn_drop = cnn_drop
        self.mlp_drop = mlp_drop
        self.norm = norm
        self.lnn = bool(lnn)
        self.lnn_dim = int(lnn_dim)
        self.lnn_dt = float(lnn_dt)
        self.seq_dim = 20
        # ---- 多层 ESM 融合 ----
        # 输入可以是 [B, L, D]（单层）或 [B, K, L, D]（K 层，先融合再进卷积）。
        # 'weighted' 只增加 K 个参数（softmax 权重），是最便宜的层选择方式。
        self.esm_layers = int(esm_layers)
        self.esm_fuse = str(esm_fuse)
        if self.esm_layers > 1:
            if self.esm_fuse not in ('weighted', 'mean', 'last'):
                raise ValueError('esm_fuse 只支持 weighted/mean/last，收到 %r'
                                 % (self.esm_fuse,))
            if self.esm_fuse == 'weighted':
                self.layer_w = nn.Parameter(torch.zeros(self.esm_layers))
        # 支路开关：--lnn-only 时关掉 ESM 与手工特征，只留 CfC（探针实验用）
        self.use_esm = bool(use_esm)
        self.use_hand = bool(use_hand)
        if not (self.use_esm or self.use_hand or self.lnn):
            raise ValueError('AIPHead 至少要启用一条支路')

        # 用 ModuleList 而非 Sequential：每个 block 要把掩码透传下去
        self.cnn = nn.ModuleList([ResBlock1D(embed_dim, drop=cnn_drop, norm=norm)
                                  for _ in range(n_res if self.use_esm else 0)])
        if self.lnn:
            self.cfc = CfCBlock(in_dim=self.seq_dim, hidden=self.lnn_dim,
                                dt=self.lnn_dt)
            self.seq_norm = nn.LayerNorm(self.lnn_dim)

        feat_dim = ((embed_dim if self.use_esm else 0)
                    + (hand_dim if self.use_hand else 0)
                    + (self.lnn_dim if self.lnn else 0))
        # ---- 融合方式 ----
        # 'early'：直接把各支路原始特征拼起来（旧行为，权重兼容）
        # 'late' ：两条支路先各自投影到 fused_dim，再做显式交互
        #          [h_e, h_h, h_e*h_h, |h_e-h_h|]。乘积项随样本变化，不会被
        #          下一层 Linear 吸收；参数只多两个投影层。
        self.fuse = str(fuse)
        self.fused_dim = int(fused_dim)
        if self.fuse not in ('early', 'late'):
            raise ValueError('fuse 只支持 early/late，收到 %r' % (self.fuse,))
        if self.fuse == 'late':
            if not (self.use_esm and self.use_hand):
                raise ValueError('fuse=late 需要 ESM 与手工两条支路同时启用'
                                 '（当前 esm=%s hand=%s）' % (self.use_esm, self.use_hand))
            if self.lnn:
                raise ValueError('fuse=late 暂不支持 CfC 支路，请先关掉 --lnn')
            self.esm_proj = nn.Sequential(
                nn.Linear(embed_dim, self.fused_dim), nn.LayerNorm(self.fused_dim))
            self.hand_proj = nn.Sequential(
                nn.Linear(hand_dim, self.fused_dim), nn.LayerNorm(self.fused_dim))
            feat_dim = 4 * self.fused_dim
        self.feat_dim = int(feat_dim)
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(mlp_drop)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(mlp_drop), nn.Linear(hidden, 1)
        )

        # ---- 原型旁路（默认关闭）----
        # 关键约定：它接在**拼接特征**上，和主分支同源，所以不会带来新信息。
        # 训练时必须额外监督 B（train_aip 里的 proto_w_aux），否则主分支会把 B 压没。
        self.proto_enabled = (int(proto_pos) + int(proto_neg)) > 0
        self.proto_alpha = float(proto_alpha)
        if self.proto_enabled:
            self.proto = ProtoBranch(feat_dim, n_pos=proto_pos, n_neg=proto_neg,
                                     proj_dim=proto_dim, tau=proto_tau)

    def fuse_layers(self, hidden_states):
        """[B, K, L, D] → [B, L, D]。

        'weighted' 的权重是 softmax 出来的 K 个标量（初始化均匀 1/K）；
        训练完把 `head.layer_w.softmax(0)` 打出来，就能说清"哪一层在起作用"。
        """
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 4:
            raise ValueError('hidden_states 应为 [B,L,D] 或 [B,K,L,D]，收到 %s'
                             % (tuple(hidden_states.shape),))
        if self.esm_layers <= 1:
            raise ValueError('缓存给了 %d 层，但这个头是单层（esm_layers=1）；'
                             '要么用 --esm-fuse 训练多层头，要么加 --esm-cache single'
                             % hidden_states.shape[1])
        if self.esm_fuse == 'last':
            return hidden_states[:, -1]
        if self.esm_fuse == 'mean':
            return hidden_states.mean(dim=1)
        w = torch.softmax(self.layer_w, dim=0).to(hidden_states.dtype)
        return (hidden_states * w.view(1, -1, 1, 1)).sum(dim=1)

    def encode(self, hidden_states, attention_mask, hand_feats=None, seq=None,
               return_decay=False):
        """把各支路拼成 [B, feat_dim] 的共享特征（原型旁路也吃这一份）。

        hidden_states : [B, L, D] 或 [B, K, L, D]  ESM-2 隐状态
        attention_mask: [B, L]     1 = 有效残基
        hand_feats    : [B, hand_dim] 或 None（hand_dim > 0 时必需）
        seq           : [B, L, 20]   CfC 用的 one-hot（lnn=True 时必需）
        return_decay  : True 时额外返回 decay [B, L, lnn_dim]，供可解释性用
        """
        if hidden_states is not None and hidden_states.dim() == 4:
            hidden_states = self.fuse_layers(hidden_states)
        ref = hidden_states if hidden_states is not None else seq
        if attention_mask is None:
            if ref is None:
                raise ValueError('必须给出 attention_mask 或 seq')
            mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)
        else:
            mask = attention_mask.bool()

        esm_vec = None
        cfc_vec = None
        hand_vec = None
        if self.use_esm:
            if hidden_states is None:
                raise ValueError('AIPHead(use_esm=True) 需要 hidden_states')
            m = mask.unsqueeze(1).to(hidden_states.dtype)           # [B, 1, L]
            x = hidden_states.transpose(1, 2) * m                   # pad 隐状态先清零
            for block in self.cnn:
                x = block(x, m)                                     # 每次卷积后重新置零
            cnt = m.sum(dim=2).clamp(min=1.0)                       # [B, 1]
            esm_vec = (x * m).sum(dim=2) / cnt                      # [B, D]

        decay = None
        if self.lnn:
            if seq is None:
                raise ValueError('AIPHead(lnn=True) 需要 seq=[B, L, 20]')
            hs, decay = self.cfc(seq, mask.to(seq.dtype))
            # CfC 只递推到 batch 内最长的那条肽，掩码要跟着截断
            mf = mask[:, :hs.shape[1]].unsqueeze(2).to(hs.dtype)     # [B, l_eff, 1]
            cnt_l = mf.sum(dim=1).clamp(min=1.0)
            cfc_vec = self.seq_norm((hs * mf).sum(dim=1) / cnt_l)

        if self.use_hand and self.hand_dim > 0 and hand_feats is not None:
            hand_vec = hand_feats

        if self.fuse == 'late':
            if esm_vec is None or hand_vec is None:
                raise ValueError('fuse=late 需要同时给出 hidden_states 与 hand_feats')
            he = self.esm_proj(esm_vec)
            hh = self.hand_proj(hand_vec)
            feat = torch.cat([he, hh, he * hh, torch.abs(he - hh)], dim=1)
        else:
            parts = [v for v in (esm_vec, cfc_vec, hand_vec) if v is not None]
            if not parts:
                raise ValueError('没有任何支路输出可用（esm/hand/lnn 全空）')
            feat = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return (feat, decay) if return_decay else feat

    def forward(self, hidden_states, attention_mask, hand_feats=None, seq=None,
                return_decay=False, return_proto=False):
        """logits = A + alpha * B；return_proto=True 时返回 (logits, 分量字典)。

        分量字典：A（主分支 logit）、B（原型 logit）、s_pos / s_neg（相似度矩阵），
        供训练时的辅助损失与事后解释使用。原型关闭时分量字典为 None。
        """
        feat, decay = self.encode(hidden_states, attention_mask, hand_feats, seq,
                                  return_decay=True)
        a = self.out(self.shared(feat)).squeeze(1)                  # [B]
        aux = None
        if self.proto_enabled:
            b, sp, sn = self.proto(feat)
            logits = a + self.proto_alpha * b
            aux = {'A': a, 'B': b, 's_pos': sp, 's_neg': sn}
        else:
            logits = a
        if return_proto:
            return logits, aux
        return (logits, decay) if return_decay else logits

    @torch.no_grad()
    def predict_proba(self, hidden_states, attention_mask, hand_feats=None, seq=None):
        self.eval()
        return torch.sigmoid(self.forward(hidden_states, attention_mask,
                                          hand_feats, seq))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def size_mb(self):
        n = sum(p.numel() * p.element_size() for p in self.parameters())
        n += sum(b.numel() * b.element_size() for b in self.buffers())
        return n / 1024 ** 2
