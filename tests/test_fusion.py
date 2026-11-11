#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""late 融合与手工特征筛选的单元测试。

覆盖：默认 early 的行为不变；late 的特征维度/形状；late 的约束报错；
筛选索引的确定性、落盘/读回、按列取特征；checkpoint 往返。
"""
import numpy as np
import pytest
import torch

from src.checkpoint import load_head, save_head
from src.hand_select import (apply_hand_selector, load_hand_selector,
                             save_hand_selector, select_hand_features)
from src.model_aip import AIPHead


def _batch(n=6, l=16, d=32, hand_dim=12, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(n, l, d)
    m = torch.ones(n, l)
    m[:, -4:] = 0
    hf = torch.randn(n, hand_dim)
    return h, m, hf


def test_early_is_default_and_feat_is_concat():
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer')
    assert head.fuse == 'early'
    assert head.feat_dim == 32 + 12
    head.eval()
    assert head(h, m, hf).shape == (6,)


def test_late_feat_dim_and_forward():
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   fuse='late', fused_dim=8)
    assert head.fuse == 'late'
    assert head.feat_dim == 4 * 8
    head.eval()
    feat = head.encode(h, m, hf)
    assert feat.shape == (6, 4 * 8)
    assert head(h, m, hf).shape == (6,)


def test_late_needs_both_branches():
    with pytest.raises(ValueError):
        AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                fuse='late', use_hand=False)
    with pytest.raises(ValueError):
        AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                fuse='late', lnn=True)


def test_late_checkpoint_roundtrip(tmp_path):
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   fuse='late', fused_dim=8, proto_pos=2, proto_neg=1, proto_dim=4)
    head.eval()
    p = str(tmp_path / 'best.pt')
    save_head(head, p, {'fold': 0})
    back, meta = load_head(p)
    back.eval()
    assert meta['fuse'] == 'late' and meta['fused_dim'] == 8
    assert torch.allclose(head(h, m, hf), back(h, m, hf))


def test_old_meta_without_fuse_still_loads(tmp_path):
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer')
    p = str(tmp_path / 'old.pt')
    save_head(head, p, {'fold': 0})
    # 模拟旧权重：meta 里没有 fuse/fused_dim 两个键
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    ckpt['meta'].pop('fuse', None)
    ckpt['meta'].pop('fused_dim', None)
    torch.save(ckpt, p)
    back, meta = load_head(p)
    assert back.fuse == 'early'          # meta 不回填，构造时用 ARCH_DEFAULTS


def test_select_hand_features_is_deterministic_and_keeps_signal():
    rng = np.random.RandomState(0)
    n, d = 300, 40
    X = rng.randn(n, d)
    y = (X[:, 3] + X[:, 7] > 0).astype(int)     # 只有两列有信号
    idx1 = select_hand_features(X, y, seed=42, top_mi=10, top_k=5)
    idx2 = select_hand_features(X, y, seed=42, top_mi=10, top_k=5)
    assert np.array_equal(idx1, idx2)
    assert len(idx1) <= 5
    assert 3 in idx1 and 7 in idx1


def test_selector_roundtrip_and_apply(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.randn(20, 10).astype(np.float32)
    idx = np.array([1, 4, 7], dtype=np.int64)
    p = str(tmp_path / 'hand_selector.npz')
    save_hand_selector(p, idx)
    back = load_hand_selector(p)
    assert np.array_equal(back, idx)
    out = apply_hand_selector(X, back)
    assert out.shape == (20, 3)
    assert np.allclose(out, X[:, idx])


def test_selector_rejects_out_of_range_index():
    X = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_hand_selector(X, np.array([9]))
