#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重文件格式（v2）

一个 fold 一个文件，只装 AIPHead（约 85 MB），不含 ESM-2：

    {
      'format': 'cxyaippred-aip-head-v2',
      'meta': {
          'embed_dim': 1280, 'hand_dim': 694, 'hidden': 512,
          'n_res': 2, 'cnn_drop': 0.2, 'mlp_drop': 0.3,
          'norm': 'layer',              # v2 起：残差块用逐位置 LayerNorm
          'esm_name': 'facebook/esm2_t33_650M_UR50D',
          'esm_layer': -1,
          'threshold': 0.487,          # 该 fold 的验证集选出的阈值，仅作参考
          'fold': 0, 'seed': 42,
          'train_data': 'data/folds_5cv_seed42.csv',
          'train_sha256': '...',        # 训练集文件指纹，便于追溯
          'hand_norm': 'standard',     # 手工特征是否按本折训练集标准化
          'note': '...'
      },
      'state': head.state_dict()
    }

为什么不存 ESM：ESM-2 权重 2.43 GB、完全冻结、所有 fold 相同；
存 5 份等于把同一份 12 GB 数据复制 5 遍。

手工特征的标准化参数（mean/std，各 694 个 float）不写进 meta，而是和权重
放同一目录的 hand_scaler.npz：它属于"这一折的训练集"，不属于网络结构，
单独放便于事后审计"标准化是用哪些行算的"。
"""
import hashlib
import os

import torch

from .model_aip import AIPHead

FORMAT = 'cxyaippred-aip-head-v2'

# v1 → v2 的唯一区别是残差块的归一化：BatchNorm1d（把 padding 算进统计量，
# 预测随 batch 补零长度变化）→ 逐位置 LayerNorm（norm='layer'）。
# v1 权重能加载进 norm='batch' 的新结构，但那样等于把 bug 一起带回来，
# 所以直接拒绝，要求用修好的代码重训。
LEGACY_FORMATS = ('cxyaippred-aip-head-v1',)

ARCH_KEYS = ('embed_dim', 'hand_dim', 'hidden', 'n_res', 'cnn_drop', 'mlp_drop', 'norm')


def save_head(head, path, meta):
    meta = dict(meta)
    meta.setdefault('esm_name', 'facebook/esm2_t33_650M_UR50D')
    meta.setdefault('esm_layer', -1)
    for k in ARCH_KEYS:
        meta.setdefault(k, getattr(head, k))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'format': FORMAT, 'meta': meta, 'state': head.state_dict()}, path)
    return path


def load_head(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    fmt = ckpt.get('format')
    if fmt != FORMAT:
        if fmt in LEGACY_FORMATS:
            raise SystemExit(
                '%s 是旧格式 %s：残差块用 BatchNorm1d 且卷积不吃掩码，\n'
                '同一条肽换一个 batch 补零长度预测就变（实测 logit 差 10.75%%）。\n'
                '这类权重不可用，请用当前代码重训（--exp 换一个新名字）。' % (path, fmt))
        raise ValueError('%s 不是 %s 格式（format=%r）' % (path, FORMAT, fmt))
    meta = ckpt['meta']
    head = AIPHead(**{k: meta[k] for k in ARCH_KEYS})
    head.load_state_dict(ckpt['state'])
    return head, meta


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_from_legacy(ckpt_path, out_path, threshold=0.487, fold=None, seed=None,
                        train_sha256=None, note=''):
    """
    把旧双任务 checkpoint（含 ESM，2.69 GB）里的 AIP 头抽成新格式（85 MB）。

    旧结构：cnn.* / head.shared.* / head.head_aip.{0,3}.* / head.head_pip.*
    新结构：cnn.* / shared.* / out.{0,3}.*
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if any(k.startswith('esm.') for k in sd):
        print('检测到 ESM 参数，将被丢弃（%d 个张量）'
              % sum(1 for k in sd if k.startswith('esm.')))
    new = {}
    for k, v in sd.items():
        if k.startswith('esm.') or k.startswith('head.head_pip.'):
            continue
        if k.startswith('head.head_aip.'):
            new['out.' + k[len('head.head_aip.'):]] = v
        elif k.startswith('head.shared.'):
            new['shared.' + k[len('head.shared.'):]] = v
        else:
            new[k] = v

    # 旧权重是 BatchNorm1d，所以显式构造 norm='batch' 的版本才能对上 bn1/bn2 键名
    head = AIPHead(embed_dim=1280, hand_dim=694, hidden=512, norm='batch')
    missing, unexpected = head.load_state_dict(new, strict=False)
    if missing or unexpected:
        raise SystemExit('状态字典不匹配: missing=%s unexpected=%s' % (missing, unexpected))

    meta = {
        'threshold': threshold, 'fold': fold, 'seed': seed,
        'train_sha256': train_sha256,
        'esm_name': 'facebook/esm2_t33_650M_UR50D',
        'source_checkpoint': os.path.basename(ckpt_path),
        'source_sha256': sha256_file(ckpt_path),
        'note': note or '由旧双任务 checkpoint 抽取 AIP 头（仅作初始化/对照，非 5 折重训结果）',
    }
    save_head(head, out_path, meta)
    print('已写出 %s（%.1f MB, %.2fM 参数）' % (out_path, head.size_mb(), head.n_params() / 1e6))
    return out_path
