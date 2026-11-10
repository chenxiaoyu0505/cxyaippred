#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锁定两个新增特征入口的性质：

1. 两亲性特征（`src/features.py:compute_amphipath`）—— 10 维、处处有限，
   且 `amph_helix_power` 真的在看 3.6 残基周期（否则它只是又一个组成统计量）。
2. 多层 ESM 融合（`AIPHead.fuse_layers`）—— last/mean/weighted 三种模式
   的输出必须与"先手工融合、再喂 3D 张量"逐位等价，权重为 0 时退化到 mean。
   这条很重要：融合发生在 forward 里，一旦顺序写错（比如融合晚于 mask 对齐），
   形状会静默错位而不报错。
"""
import numpy as np
import pytest
import torch

from src.checkpoint import load_head, save_head
from src.features import (AMPHIPATH_DIM, amphipath_feature_names, compute_amphipath)
from src.model_aip import AIPHead


# ---------------------------------------------------------------- 两亲性特征

def test_amphipath_shape_and_names():
    assert AMPHIPATH_DIM == 10
    names = amphipath_feature_names()
    assert len(names) == AMPHIPATH_DIM and len(set(names)) == AMPHIPATH_DIM
    assert names[0] == 'amph_mu7_max' and names[-2] == 'amph_helix_power'


@pytest.mark.parametrize('seq', ['A', 'LK', 'LKK', 'LKKLLKKLLKKLLKKL',
                                 'ACDEFGHIKLMNPQRSTVWY' * 3 + 'A'])
def test_amphipath_is_finite_and_bounded(seq):
    v = compute_amphipath(seq)
    assert v.shape == (AMPHIPATH_DIM,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all(), v
    # μH 是单位归一化的疏水矩，不可能超过 1；argmax 是位置（按窗口长度归一化）
    assert ((v[[0, 1, 2, 4, 5, 6]] >= 0) & (v[[0, 1, 2, 4, 5, 6]] <= 1.0 + 1e-5)).all(), v
    assert ((v[[3, 7]] >= 0) & (v[[3, 7]] <= 1.0 + 1e-5)).all(), v
    assert 0.0 <= v[8] <= 1.0 + 1e-5, v


def test_amphipath_helix_power_sees_periodicity():
    """理想两亲性螺旋（疏水/亲水交替，周期 3.5）的 3.6 周期功率必须高于同组成乱序。"""
    helix = 'LKKLLKKLLKKLLKKL'
    scrambled = 'LKLLKKLKLLKKLLKK'
    assert sorted(helix) == sorted(scrambled)          # 组成完全相同
    p_helix = compute_amphipath(helix)[8]
    p_scram = compute_amphipath(scrambled)[8]
    assert p_helix > p_scram + 0.1, (p_helix, p_scram)
    assert p_helix > 0.95, p_helix


def test_amphipath_not_constant_across_lengths():
    """长度不同的肽不能给出同一组值（否则只是个常数特征）。"""
    a = compute_amphipath('LKKLLKKLLKKLLKKL')
    b = compute_amphipath('LKKLLKKLLKKLLKKLLKKLLKKLLKKL')
    assert not np.allclose(a, b)


# ---------------------------------------------------------------- 多层 ESM 融合

def _multi_head(fuse='weighted', layers=3, seed=0):
    torch.manual_seed(seed)
    return AIPHead(embed_dim=8, hand_dim=4, hidden=16, n_res=1, cnn_drop=0.0,
                   mlp_drop=0.0, norm='layer', esm_layers=layers,
                   esm_fuse=fuse).eval()


def _run(head, hs, mask=None, hand=None):
    b, l = hs.shape[0], hs.shape[-2]
    if mask is None:
        mask = torch.ones(b, l)
    if hand is None:
        hand = torch.zeros(b, 4)
    with torch.no_grad():
        return head(hs, mask, hand)


def _hs4(seed=1, b=2, k=3, l=6, d=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, k, l, d, generator=g)


@pytest.mark.parametrize('fuse', ['last', 'mean'])
def test_fuse_matches_manual(fuse):
    hs4 = _hs4()
    head = _multi_head(fuse)
    manual = hs4[:, -1] if fuse == 'last' else hs4.mean(dim=1)
    assert torch.allclose(_run(head, hs4), _run(head, manual), atol=1e-6)


def test_weighted_zero_init_equals_mean():
    """layer_w 初始化为 0 ⇒ softmax 是均匀分布 ⇒ 等价于 mean。"""
    hs4 = _hs4()
    head = _multi_head('weighted')
    assert torch.equal(head.layer_w, torch.zeros(3))
    assert torch.allclose(_run(head, hs4), _run(head, hs4.mean(dim=1)), atol=1e-6)


def test_weighted_is_learnable():
    head = _multi_head('weighted')
    assert isinstance(head.layer_w, torch.nn.Parameter)
    assert head.layer_w.requires_grad
    assert head.layer_w.shape == (3,)


def test_single_layer_head_accepts_3d():
    """单层头（esm_layers=1）必须照旧吃 3D 张量，不受多层改造影响。"""
    head = _multi_head('weighted', layers=1)
    assert _run(head, _hs4()[:, 0]).shape == (2,)


def test_bad_fuse_name_raises():
    with pytest.raises(ValueError):
        _multi_head('nope')


def test_multilayer_checkpoint_roundtrip(tmp_path):
    hs4 = _hs4()
    head = _multi_head('weighted')
    p = save_head(head, str(tmp_path / 'best.pt'), {'note': 'test'})
    back, meta = load_head(p, map_location='cpu')
    assert back.esm_layers == 3 and back.esm_fuse == 'weighted'
    assert meta['note'] == 'test'
    assert torch.allclose(_run(back, hs4), _run(head, hs4), atol=1e-6)


def test_multilayer_input_to_single_layer_head_raises():
    """缓存变 3 层但权重是单层头：必须报错，不能静默取某一层或崩在 layer_w 上。"""
    head = _multi_head('weighted', layers=1)
    with pytest.raises(ValueError, match='单层'):
        _run(head, _hs4())
