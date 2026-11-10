#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自检：不需要 ESM、不需要数据，验证 AIPHead 与权重格式。

    python scripts/selftest.py

检查项
------
1. 前向形状与数值范围正确
2. 保存/加载 round-trip 后参数完全一致
3. 权重文件大小符合预期（~80 MB，而不是 2.69 GB）
4. 加载时能拒绝格式不对的文件
"""
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.model_aip import AIPHead                    # noqa: E402
from src.checkpoint import save_head, load_head      # noqa: E402

B, L, D = 3, 17, 1280
HAND = 694


def main():
    torch.manual_seed(0)
    head = AIPHead(embed_dim=D, hand_dim=HAND, hidden=512).eval()
    print('参数量: %.2f M | 内存占用: %.1f MB' % (head.n_params() / 1e6, head.size_mb()))

    h = torch.randn(B, L, D)
    m = torch.ones(B, L, dtype=torch.long)
    hand = torch.rand(B, HAND)

    logits = head(h, m, hand)
    assert logits.shape == (B,), logits.shape
    p = torch.sigmoid(logits)
    assert torch.all((p >= 0) & (p <= 1))
    print('前向 OK: logits %s' % (tuple(logits.shape),))

    # 变化序列长度 + 掩码，确认池化不炸
    h2 = torch.randn(B, 9, D)
    m2 = torch.ones(B, 9, dtype=torch.long)
    m2[0, 5:] = 0
    out2 = head(h2, m2, hand)
    assert out2.shape == (B,)
    print('变长 + 掩码 OK')

    # 不用系统 temp：受限沙箱下 %TEMP% 往往不可写（torch 报 error code 5）
    tmp = os.path.join(ROOT, '_work', 'selftest_tmp')
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, 'fold0_best.pt')
    save_head(head, path, {'fold': 0, 'seed': 42, 'threshold': 0.5})
    size_mb = os.path.getsize(path) / 1024 ** 2
    print('权重文件: %.1f MB' % size_mb)
    assert size_mb < 150, '权重文件过大，可能把 ESM 也存进去了'

    head2, meta = load_head(path)
    head2.eval()
    for (k1, v1), (k2, v2) in zip(head.state_dict().items(), head2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), '参数不一致: %s' % k1
    print('round-trip OK, meta=%s' % meta)

    p1 = torch.sigmoid(head(h, m, hand))
    p2 = torch.sigmoid(head2(h, m, hand))
    assert torch.allclose(p1, p2, atol=1e-6)
    print('同一输入两次推理一致 OK')

    # 拒绝错误格式
    bad = os.path.join(tmp, 'bad.pt')
    torch.save({'foo': 1}, bad)
    try:
        load_head(bad)
    except ValueError as e:
        print('拒绝非法格式 OK: %s' % e)
    else:
        raise AssertionError('应当拒绝非法格式')

    # 手工特征必须是 694 维且无 NaN（与旧权重兼容的硬约束）
    import numpy as np
    from src.features import compute_handcraft
    f = compute_handcraft('ACDEFGHIKLMNPQRSTVWY')
    assert len(f) == 694, '手工特征维度变了：%d（应为 694）' % len(f)
    assert not np.isnan(f).any(), '手工特征出现 NaN'
    print('手工特征 OK: %d 维, 无 NaN' % len(f))

    # 自检产物（80 MB 权重 + 坏格式文件）用完即删
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print('临时文件已清理：%s' % os.path.relpath(tmp, ROOT))

    print('\n全部自检通过')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
