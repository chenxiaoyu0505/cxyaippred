#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""masked BN / peptide-BN 的单元测试。

两条要守住的性质：
1. padding 位置的数值不参与统计量（改 pad 里的数，真实残基的输出与
   running stats 都不能变）；
2. peptide_bn 按肽等权（一条 6 残基的肽和一条 2 残基的肽各投一票），
   masked_batch 按残基位置等权。
"""
import torch

from src.model_aip import AIPHead, MaskedBatchNorm1d, ResBlock1D


def _make(ch=3, per_peptide=True, momentum=1.0):
    torch.manual_seed(0)
    bn = MaskedBatchNorm1d(ch, per_peptide=per_peptide, momentum=momentum)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.0, 2.0, 0.5])[:ch])
        bn.bias.copy_(torch.tensor([0.0, 1.0, -1.0])[:ch])
    return bn


def test_pad_content_does_not_change_stats_or_output():
    """同一批真实残基，pad 里填 0 还是填 1e3，结果必须一致。"""
    torch.manual_seed(1)
    m = torch.tensor([[1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(1)   # [2,1,6]
    real = torch.randn(2, 3, 6)

    a = real * m
    b = real * m + (1 - m) * 1e3          # 只在 pad 位置塞垃圾

    out = []
    stats = []
    for x in (a, b):
        bn = _make(momentum=1.0)
        bn.train()
        out.append(bn(x, m))
        stats.append((bn.running_mean.clone(), bn.running_var.clone()))

    assert torch.allclose(out[0], out[1], atol=1e-6)
    assert torch.allclose(stats[0][0], stats[1][0], atol=1e-6)
    assert torch.allclose(stats[0][1], stats[1][1], atol=1e-6)
    # pad 位置必须被清零
    assert out[0][0, :, 4:].abs().max().item() == 0.0


def test_peptide_bn_is_per_peptide():
    """一条 6 残基全 1 的肽 + 一条 2 残基全 3 的肽：按肽等权 → 均值 2、方差 1。"""
    bn = _make(ch=1, per_peptide=True, momentum=1.0)
    bn.train()
    x = torch.zeros(2, 1, 6)
    m = torch.tensor([[1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(1)
    x = x + m * torch.tensor([1.0, 3.0]).view(2, 1, 1)
    bn(x, m)
    assert abs(bn.running_mean.item() - 2.0) < 1e-5, bn.running_mean
    # 无偏修正用肽数 n=2（2/(2-1)=2）：方差 1 → running_var 2
    assert abs(bn.running_var.item() - 2.0) < 1e-5, bn.running_var


def test_masked_batch_is_position_weighted():
    """同样输入：按残基位置等权 → 均值 1.5、方差 0.75。"""
    bn = _make(ch=1, per_peptide=False, momentum=1.0)
    bn.train()
    m = torch.tensor([[1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(1)
    x = m * torch.tensor([1.0, 3.0]).view(2, 1, 1)
    bn(x, m)
    assert abs(bn.running_mean.item() - 1.5) < 1e-5, bn.running_mean
    # 无偏修正用位置数 n=8（8/7）：方差 0.75 → running_var 0.857
    assert abs(bn.running_var.item() - 0.75 * 8 / 7) < 1e-5, bn.running_var


def test_eval_uses_running_stats():
    bn = _make(ch=1, per_peptide=True, momentum=1.0)
    bn.train()
    m = torch.ones(1, 1, 4)
    bn(torch.full((1, 1, 4), 5.0), m)
    bn.eval()
    y = bn(torch.full((1, 1, 4), 7.0), m)
    # running_mean=5, running_var=0 → (7-5)/sqrt(0+eps)
    assert abs(y.mean().item() - 2.0 / (1e-5 ** 0.5)) < 1e-3


def test_invalid_norm_rejected():
    try:
        ResBlock1D(4, norm='nope')
    except ValueError:
        return
    raise AssertionError("非法 norm 应该报错")


def test_head_runs_with_every_norm():
    torch.manual_seed(0)
    b, l, d, h = 3, 9, 16, 8
    hs = torch.randn(b, l, d)
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4, [1] * 2 + [0] * 7])
    hand = torch.randn(b, 7)
    for norm in ('layer', 'peptide_bn', 'masked_batch', 'batch'):
        head = AIPHead(embed_dim=d, hand_dim=7, hidden=h, n_res=2, norm=norm)
        head.train()
        out = head(hs, mask, hand)
        assert out.shape == (b,), (norm, out.shape)
        assert torch.isfinite(out).all(), norm


def test_padding_does_not_change_prediction_in_eval():
    """推理阶段：同一条肽单独 vs 与长肽同批，概率必须一致（四种归一化都要满足）。"""
    torch.manual_seed(0)
    d, h = 16, 8
    short = torch.randn(1, 6, d)
    long_ = torch.randn(1, 20, d)
    hand_s = torch.randn(1, 7)
    hand_l = torch.randn(1, 7)
    for norm in ('layer', 'peptide_bn', 'masked_batch', 'batch'):
        head = AIPHead(embed_dim=d, hand_dim=7, hidden=h, n_res=2, norm=norm)
        head.eval()
        m1 = torch.ones(1, 6)
        alone = head(short, m1, hand_s)
        # 拼成一批，短肽补到 20
        hs = torch.cat([torch.cat([short, torch.zeros(1, 14, d)], 1), long_], 0)
        mask = torch.tensor([[1] * 6 + [0] * 14, [1] * 20])
        hand = torch.cat([hand_s, hand_l], 0)
        batched = head(hs, mask, hand)[0]
        assert torch.allclose(alone[0], batched, atol=1e-5), (norm, alone[0], batched)
