#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM 缓存文件的管理（清理 / 存在性检查）。

单独放一个模块，是为了让 `run_5fold.py` 这类统筹脚本能在**不 import torch** 的
情况下判断/清理缓存。cache_esm.py 需要 GPU，但清理缓存不需要。
"""

ESM_FILES = ('esm_emb_fp16.npy', 'esm_mask.npy', 'esm_index.csv', 'esm_meta.json')

import os  # noqa: E402


def has_cache(cache_dir):
    """emb / mask / index 三个文件齐了才算有缓存"""
    return all(os.path.exists(os.path.join(cache_dir, fn)) for fn in ESM_FILES[:3])


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
