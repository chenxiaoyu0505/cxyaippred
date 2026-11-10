#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锁定两个"预测必须可复现"的性质：

1. padding 不变量 —— 同一条肽的 logit 不能因为同 batch 里有更长的肽而改变
   （旧实现：BatchNorm1d 把 padding 算进统计量 + 卷积看见 <pad> 隐状态，
   实测差 10.75%）。
2. 手工特征标准化的拟合只能用训练行 —— 否则 val/test 统计量泄漏。
"""
import numpy as np
import pytest
import torch

from src.checkpoint import FORMAT, load_head, save_head
from src.data_io import apply_hand_scaler, fit_hand_scaler
from src.model_aip import AIPHead


def _head(norm='layer', seed=0):
    torch.manual_seed(seed)
    h = AIPHead(embed_dim=8, hand_dim=4, hidden=16, n_res=2, cnn_drop=0.0,
                mlp_drop=0.0, norm=norm).eval()
    return h


def _logits(head, hs, mask, hand):
    with torch.no_grad():
        return head(hs, mask, hand)


def test_single_vs_padded_batch_identical():
    """同一条 6 残基肽：单独推理 vs 与 20 残基肽同批，logit 必须一致。"""
    head = _head()
    g = torch.Generator().manual_seed(1)
    short = torch.randn(1, 6, 8, generator=g)
    hand_short = torch.randn(1, 4, generator=g)

    alone = _logits(head, short, torch.ones(1, 6), hand_short)[0]

    # 批次化：第 0 行是短肽补到 20，第 1 行是长肽
    hs = torch.randn(2, 20, 8, generator=g)
    hs[0, :6] = short[0]
    mask = torch.zeros(2, 20)
    mask[0, :6] = 1.0
    mask[1, :] = 1.0
    hand = torch.stack([hand_short[0], torch.randn(4, generator=g)])
    mixed = _logits(head, hs, mask, hand)[0]

    assert torch.allclose(alone, mixed, atol=1e-6), (
        'logit 随 batch 补零长度变化：单独 %.8f vs 同批 %.8f（差 %.4f%%）'
        % (alone.item(), mixed.item(),
           100 * abs(alone.item() - mixed.item()) / max(abs(alone.item()), 1e-12)))


def test_prediction_independent_of_companions():
    """换个"同批伙伴"，目标样本的 logit 不能动。"""
    head = _head(seed=3)
    g = torch.Generator().manual_seed(7)
    target = torch.randn(1, 9, 8, generator=g)
    hand_t = torch.randn(1, 4, generator=g)

    outs = []
    for _ in range(3):
        hs = torch.randn(3, 25, 8, generator=g)
        hs[0, :9] = target[0]
        mask = torch.zeros(3, 25)
        mask[0, :9] = 1.0
        mask[1, :15] = 1.0
        mask[2, :] = 1.0
        hand = torch.stack([hand_t[0]] + [torch.randn(4, generator=g) for _ in range(2)])
        outs.append(_logits(head, hs, mask, hand)[0].item())

    assert max(outs) - min(outs) < 1e-6, '伴侣样本改变了目标预测：%s' % outs


def test_pad_positions_do_not_reach_pooling():
    """pad 位置塞进任意大的隐状态，池化结果也不能变。"""
    head = _head(seed=5)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 7, 8, generator=g)
    mask = torch.zeros(1, 12)
    mask[0, :7] = 1.0
    hand = torch.randn(1, 4, generator=g)

    hs = torch.zeros(1, 12, 8)
    hs[0, :7] = x[0]
    a = _logits(head, hs, mask, hand)[0]

    hs2 = hs.clone()
    hs2[0, 7:] = 1e3                       # 污染 pad 位置
    b = _logits(head, hs2, mask, hand)[0]
    assert torch.allclose(a, b, atol=1e-6), 'pad 位置的隐状态泄漏进了池化'


def test_layer_norm_has_no_running_stats():
    head = _head(norm='layer')
    names = dict(head.named_buffers())
    assert not any('running_' in n for n in names), \
        'layer 版不应含 BatchNorm 的 running_mean/running_var'


def test_hand_scaler_fit_uses_train_rows_only():
    rng = np.random.default_rng(0)
    hand = rng.normal(size=(20, 6)).astype(np.float32) * 100
    train = np.arange(12)
    mu, sd = fit_hand_scaler(hand, train)

    assert np.allclose(mu, hand[train].mean(0), atol=1e-5)
    assert np.allclose(sd, hand[train].std(0), atol=1e-5)

    # 把非训练行改得面目全非，参数不能变 —— 这就是"不泄漏"的定义
    hand2 = hand.copy()
    hand2[12:] += 1e6
    mu2, sd2 = fit_hand_scaler(hand2, train)
    assert np.array_equal(mu, mu2) and np.array_equal(sd, sd2)


def test_hand_scaler_standardizes_and_handles_constant_dim():
    rng = np.random.default_rng(1)
    hand = rng.normal(size=(50, 4)).astype(np.float32)
    hand[:, 2] = 3.0                       # 常数维：std=0
    mu, sd = fit_hand_scaler(hand, np.arange(40))
    z = apply_hand_scaler(hand, mu, sd)

    assert np.isfinite(z).all(), '常数维产生了 inf/NaN'
    assert abs(z[:, 2]).max() < 1e-6, '常数维应该被压成 0'
    assert abs(z[:40].mean(0)[[0, 1, 3]]).max() < 1e-4, '训练行均值应≈0'
    assert abs(z[:40].std(0)[[0, 1, 3]] - 1).max() < 1e-3, '训练行 std 应≈1'


def test_checkpoint_roundtrip_and_v1_rejected(tmp_path):
    head = _head()
    p = str(tmp_path / 'best.pt')
    save_head(head, p, {'fold': 0, 'hand_norm': 'standard'})
    got, meta = load_head(p)
    assert meta['hand_norm'] == 'standard'
    assert meta['norm'] == 'layer' and meta['hand_dim'] == 4
    assert got.n_params() == head.n_params()

    # 旧格式必须被明确拒绝，而不是悄悄加载后带着 bug 跑
    ck = torch.load(p, map_location='cpu', weights_only=False)
    ck['format'] = 'cxyaippred-aip-head-v1'
    p1 = str(tmp_path / 'old.pt')
    torch.save(ck, p1)
    with pytest.raises(SystemExit):
        load_head(p1)
    assert FORMAT.endswith('v3')   # v3 = v2 + 可选 CfC 支路
