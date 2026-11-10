#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM 缓存文件的管理（清理 / 存在性检查）。

单独放一个模块，是为了让 `run_5fold.py` 这类统筹脚本能在**不 import torch** 的
情况下判断/清理缓存。cache_esm.py 需要 GPU，但清理缓存不需要。
"""

ESM_FILES = ('esm_emb_fp16.npy', 'esm_mask.npy', 'esm_index.csv', 'esm_meta.json',
             'esm_emb_layers_fp16.npy')

# 单层缓存 esm_emb_fp16.npy 与多层缓存 esm_emb_layers_fp16.npy 二选一；两者同时
# 存在时 load_esm_cache 优先用多层（meta 里 layout 会说明是哪一种）。
EMB_ALIASES = ('esm_emb_layers_fp16.npy', 'esm_emb_fp16.npy')

import json  # noqa: E402
import os  # noqa: E402


def _read_meta(cache_dir):
    p = os.path.join(cache_dir, 'esm_meta.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def has_cache(cache_dir, expected_n=None):
    """mask / index / meta 语义齐了、且两种 emb 至少有一个，才算有缓存。

    ``expected_n`` 给出时还会校验 meta 里的 ``n_rows``，避免 4,686 行旧缓存
    被拿去配 4,634 行新表。
    """
    ok = all(os.path.exists(os.path.join(cache_dir, fn))
             for fn in ('esm_mask.npy', 'esm_index.csv', 'esm_meta.json'))
    emb = any(os.path.exists(os.path.join(cache_dir, fn)) for fn in EMB_ALIASES)
    meta = _read_meta(cache_dir)
    if not (ok and emb and meta):
        return False
    if meta.get('mask') != 'residues_only_v1':
        return False
    if expected_n is not None and meta.get('n_rows') != expected_n:
        return False
    return True


def cache_size_mb(cache_dir):
    tot = 0
    for fn in ESM_FILES:
        p = os.path.join(cache_dir, fn)
        if os.path.exists(p):
            tot += os.path.getsize(p)
    return tot / 1e6


def clean_cache(cache_dir):
    """删掉 ESM 缓存（约 0.9 GB）；保留 hand_feats.npy（12 MB、确定性、可复用）"""
    removed = []
    for fn in ESM_FILES:
        p = os.path.join(cache_dir, fn)
        if os.path.exists(p):
            size = os.path.getsize(p)
            os.remove(p)
            removed.append((fn, size))
    return removed
