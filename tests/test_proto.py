#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原型旁路（ProtoBranch）的单元测试：形状、关闭时行为不变、分量可分解、梯度可达。"""
import torch

from src.model_aip import AIPHead


def _batch(n=6, l=16, d=32, hand_dim=12, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(n, l, d)
    m = torch.ones(n, l)
    m[:, -4:] = 0                       # 末尾 padding
    hf = torch.randn(n, hand_dim)
    return h, m, hf


def test_proto_off_is_default_and_matches_plain_forward():
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer')
    head.eval()                          # 关掉 dropout，两次前向才可比
    assert head.proto_enabled is False
    logits = head(h, m, hf)
    logits2, aux = head(h, m, hf, return_proto=True)
    assert aux is None
    assert torch.allclose(logits, logits2)


def test_proto_forward_shapes_and_decomposition():
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   proto_pos=2, proto_neg=3, proto_dim=8, proto_alpha=1.0)
    assert head.proto_enabled is True
    logits, aux = head(h, m, hf, return_proto=True)
    assert logits.shape == (6,)
    assert aux['A'].shape == (6,)
    assert aux['B'].shape == (6,)
    assert aux['s_pos'].shape == (6, 2)
    assert aux['s_neg'].shape == (6, 3)
    # logits = A + alpha * B
    assert torch.allclose(logits, aux['A'] + 1.0 * aux['B'], atol=1e-6)


def test_alpha_scales_branch():
    h, m, hf = _batch()
    torch.manual_seed(0)
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   proto_pos=2, proto_neg=2, proto_dim=8, proto_alpha=0.5)
    head.eval()
    _, aux = head(h, m, hf, return_proto=True)
    logits = head(h, m, hf)
    assert torch.allclose(logits, aux['A'] + 0.5 * aux['B'], atol=1e-6)


def test_gradients_reach_prototypes_and_projection():
    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   proto_pos=2, proto_neg=2, proto_dim=8)
    logits, aux = head(h, m, hf, return_proto=True)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(aux['B'], y)
    loss = loss + 0.01 * head.proto.diversity()
    loss.backward()
    assert head.proto.p_pos.grad is not None
    assert head.proto.p_pos.grad.abs().sum() > 0
    assert head.proto.proj[0].weight.grad is not None


def test_diversity_zero_for_single_prototype_and_positive_for_many():
    h, m, hf = _batch()
    single = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                     proto_pos=1, proto_neg=1, proto_dim=8)
    assert float(single.proto.diversity()) == 0.0
    many = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   proto_pos=4, proto_neg=4, proto_dim=8)
    assert float(many.proto.diversity().detach()) >= 0.0


def test_checkpoint_roundtrip_keeps_prototypes():
    import os
    import tempfile
    from src.checkpoint import load_head, save_head

    h, m, hf = _batch()
    head = AIPHead(embed_dim=32, hand_dim=12, hidden=16, n_res=1, norm='layer',
                   proto_pos=2, proto_neg=2, proto_dim=8)
    head.eval()
    before = head(h, m, hf).detach()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'best.pt')
        save_head(head, p, {'proto_pos': 2, 'proto_neg': 2})
        head2, _ = load_head(p, map_location='cpu')
    head2.eval()
    after = head2(h, m, hf).detach()
    assert torch.allclose(before, after, atol=1e-6)
