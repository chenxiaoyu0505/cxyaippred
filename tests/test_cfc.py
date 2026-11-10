#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CfC 支路的单元测试：形状、padding 不变性、batch 独立性、参数量、checkpoint。"""
import os
import tempfile

import pytest
import torch

from src.checkpoint import load_head, save_head
from src.cfc import CfCBlock
from src.data_io import seq_onehot
from src.model_aip import AIPHead


def _onehot_batch(n, l, seed=0):
    torch.manual_seed(seed)
    x = torch.zeros(n, l, 20)
    for i in range(n):
        for t in range(l):
            x[i, t, (i + t) % 20] = 1.0
    return x


def test_cfc_shapes_and_range():
    cfc = CfCBlock(in_dim=20, hidden=128)
    x = _onehot_batch(2, 7)
    m = torch.ones(2, 7)
    m[0, 5:] = 0
    h, d = cfc(x, m)
    assert h.shape == (2, 7, 128) and d.shape == (2, 7, 128)
    assert (d > 0).all() and (d < 1).all(), 'decay 必须严格落在 (0, 1)'
    assert h.abs().max() <= 1.0 + 1e-6, 'h 是 tanh 的凸组合，必须落在 [-1, 1]'


def test_padding_does_not_touch_state():
    """在尾部补任意内容，真实位置的 h/decay 都不能变。"""
    torch.manual_seed(0)
    cfc = CfCBlock(in_dim=20, hidden=16)
    x = torch.randn(1, 6, 20)
    x2 = torch.cat([x, torch.randn(1, 5, 20)], dim=1)
    m1 = torch.ones(1, 6)
    m2 = torch.cat([m1, torch.zeros(1, 5)], dim=1)
    h1, d1 = cfc(x, m1)
    h2, d2 = cfc(x2, m2)
    assert torch.allclose(h1, h2[:, :6], atol=1e-6)
    assert torch.allclose(d1, d2[:, :6], atol=1e-6)


def test_batch_independence():
    """逐肽独立递归：拼进一个 batch 不能改变任何一条的预测。"""
    torch.manual_seed(1)
    cfc = CfCBlock(in_dim=20, hidden=16)
    a = torch.randn(1, 5, 20)
    b = torch.randn(1, 9, 20)
    xa = torch.zeros(1, 9, 20)
    xa[:, :5] = a
    ma = torch.zeros(1, 9)
    ma[:, :5] = 1
    xb = b.clone()
    mb = torch.ones(1, 9)
    ha, _ = cfc(xa, ma)
    hb, _ = cfc(xb, mb)
    hh, _ = cfc(torch.cat([xa, xb], 0), torch.cat([ma, mb], 0))
    # CfC 只递推到 batch 内最后一个有效位置，短肽的输出要按实际长度截取
    assert ha.shape[1] == 5 and hh.shape[1] == 9
    assert torch.allclose(ha[0], hh[0, :5], atol=1e-6)
    assert torch.allclose(hb[0], hh[1], atol=1e-6)


def test_half_life():
    hl = CfCBlock.half_life(torch.tensor([0.5, 0.9, 0.99]))
    assert abs(float(hl[0]) - 1.0) < 1e-4
    assert abs(float(hl[1]) - 6.5788) < 1e-3
    assert abs(float(hl[2]) - 69.0) < 0.1


def test_head_param_delta():
    base = AIPHead()
    full = AIPHead(lnn=True, lnn_dim=128)
    # 首层 1974→2102：+512*128；CfC 3*(20*128+128) + 3*(128*128)；LayerNorm 2*128
    expect = 512 * 128 + 3 * (20 * 128 + 128) + 3 * (128 * 128) + 2 * 128
    assert full.n_params() - base.n_params() == expect
    assert full.feat_dim == 1280 + 128 + 694


def test_head_forward_backward():
    head = AIPHead(lnn=True, lnn_dim=128)
    hs = torch.randn(3, 9, 1280)
    m = torch.ones(3, 9, dtype=torch.bool)
    hand = torch.randn(3, 694)
    seq = _onehot_batch(3, 9)
    logits, decay = head(hs, m, hand, seq, return_decay=True)
    assert logits.shape == (3,) and decay.shape == (3, 9, 128)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert head.cfc.x_f.weight.grad is not None
    assert head.cfc.h_f.weight.grad is not None


def test_lnn_requires_seq():
    head = AIPHead(lnn=True, lnn_dim=8)
    with pytest.raises(ValueError):
        head(torch.randn(2, 5, 1280), torch.ones(2, 5, dtype=torch.bool),
             torch.randn(2, 694))


def test_head_ignores_padding():
    """同一条肽补零到更长，logit 必须完全一致。"""
    torch.manual_seed(2)
    head = AIPHead(lnn=True, lnn_dim=16).eval()
    hs = torch.randn(1, 6, 1280)
    seq = _onehot_batch(1, 6)
    hand = torch.randn(1, 694)
    a = head(hs, torch.ones(1, 6, dtype=torch.bool), hand, seq)
    hs2 = torch.cat([hs, torch.randn(1, 4, 1280)], 1)
    seq2 = torch.cat([seq, torch.zeros(1, 4, 20)], 1)
    m2 = torch.cat([torch.ones(1, 6), torch.zeros(1, 4)], 1).bool()
    b = head(hs2, m2, hand, seq2)
    assert torch.allclose(a, b, atol=1e-5)


def test_seq_onehot():
    oh, lens = seq_onehot(['ACD', 'ACDEF'], lmax=6)
    assert oh.shape == (2, 6, 20) and lens.tolist() == [3, 5]
    assert oh[0, 0, 0] == 1 and oh[0, 1, 1] == 1 and oh[0, 2, 2] == 1
    assert oh[0, 3:].sum() == 0
    with pytest.raises(SystemExit):
        seq_onehot(['ACX'])


def test_checkpoint_roundtrip_and_v2_compat():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'best.pt')
        head = AIPHead(lnn=True, lnn_dim=128)
        save_head(head, p, {'fold': 0})
        h2, meta = load_head(p)
        assert h2.lnn is True and h2.lnn_dim == 128
        assert meta['format'] if 'format' in meta else True

        # v2（没有 lnn 字段）必须还能读
        p2 = os.path.join(d, 'v2.pt')
        old = AIPHead(lnn=False)
        save_head(old, p2, {'fold': 1})
        ck = torch.load(p2, map_location='cpu', weights_only=False)
        ck['format'] = 'cxyaippred-aip-head-v2'
        ck['meta'].pop('lnn', None)
        ck['meta'].pop('lnn_dim', None)
        torch.save(ck, p2)
        h3, _ = load_head(p2)
        assert h3.lnn is False and h3.feat_dim == 1974


def test_predict_offset_matches_cache_layout():
    """predict.py 的 token 网格（offset=1）必须与 cache 布局逐位等价。"""
    cfc = CfCBlock(in_dim=20, hidden=16)
    seq = 'ACDEF'
    oh0, _ = seq_onehot([seq], lmax=5, offset=0)
    m0 = torch.ones(1, 5)
    oh1, _ = seq_onehot([seq], lmax=8, offset=1)
    m1 = torch.zeros(1, 8)
    m1[:, 1:6] = 1                                   # <cls>=0, <eos>=6, pad=7
    h0, d0 = cfc(torch.from_numpy(oh0), m0)
    h1, d1 = cfc(torch.from_numpy(oh1), m1)
    assert torch.allclose(h0[0], h1[0, 1:6], atol=1e-6)
    assert torch.allclose(d0[0], d1[0, 1:6], atol=1e-6)


def test_lnn_only_head_ignores_esm_and_hand():
    """--lnn-only：关掉两条支路后，ESM 隐状态与手工特征都不能再影响预测。"""
    head = AIPHead(embed_dim=8, hand_dim=4, hidden=16, n_res=1,
                   lnn=True, lnn_dim=16, use_esm=False, use_hand=False)
    head.eval()
    assert head.feat_dim == 16 and len(head.cnn) == 0
    seq = _onehot_batch(2, 5)
    m = torch.ones(2, 5)
    with torch.no_grad():
        a = head(None, m, None, seq)
        b = head(torch.randn(2, 5, 8), m, torch.randn(2, 4), seq)
    assert torch.allclose(a, b, atol=1e-6)


def test_lnn_only_still_needs_seq():
    head = AIPHead(embed_dim=8, hand_dim=4, hidden=16, n_res=1,
                   lnn=True, lnn_dim=16, use_esm=False, use_hand=False)
    with pytest.raises(ValueError):
        head(None, torch.ones(1, 5), None, None)
