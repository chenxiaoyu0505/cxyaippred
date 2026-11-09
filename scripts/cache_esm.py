#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性把 4,686 条肽的残基级 ESM-2 隐状态算好存盘。

为什么必须先做这一步
--------------------
不缓存：每折每 epoch 都要跑 3,000 条 ESM-2 650M 前向 ≈ 数分钟 → 50 epoch × 5 折 = 数小时。
缓存后：训练只跑 21M 参数的卷积头，单折 50 epoch 约 1–3 分钟。

产物（cache/ 下）
----------------
    esm_emb_fp16.npy   [N, Lmax, 1280] float16   约 0.9 GB（Lmax = 数据最长序列 + 2）
    esm_mask.npy       [N, Lmax] uint8
    esm_index.csv      row, sha1, sequence       防止行序漂移
    esm_meta.json      esm 名称、层号、维度、行数、耗时

用法
----
    python scripts/cache_esm.py                    # 默认 GPU、fp16 存储
    python scripts/cache_esm.py --device cpu --batch-size 4
    python scripts/cache_esm.py --force            # 覆盖已有缓存
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.cache_util import ESM_FILES, clean_cache, has_cache  # noqa: E402,F401
from src.data_io import load_table, sha1_text          # noqa: E402
from src.esm_backbone import ESMBackbone               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'aip_all.csv'))
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--esm-name', default='facebook/esm2_t33_650M_UR50D')
    ap.add_argument('--layer', type=int, default=-1)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--clean', action='store_true', help='只删除缓存，不生成')
    args = ap.parse_args()

    if args.clean:
        removed = clean_cache(args.cache_dir)
        if not removed:
            print('没有可删的 ESM 缓存：%s' % args.cache_dir)
        else:
            tot = sum(s for _, s in removed) / 1e6
            print('已删除 %d 个文件，释放 %.1f MB：%s'
                  % (len(removed), tot, ', '.join(fn for fn, _ in removed)))
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    p_emb = os.path.join(args.cache_dir, 'esm_emb_fp16.npy')
    p_mask = os.path.join(args.cache_dir, 'esm_mask.npy')
    if os.path.exists(p_emb) and not args.force:
        print('已存在 %s（%.1f MB），加 --force 才重算' % (p_emb, os.path.getsize(p_emb) / 1e6))
        return

    table = load_table(args.data)
    seqs = table['sequence'].tolist()
    n = len(seqs)
    print('数据 %d 条 | 平均长度 %.1f | 最长 %d'
          % (n, np.mean([len(s) for s in seqs]), max(len(s) for s in seqs)))

    # 全局最长（含 <cls>/<eos>），一次性分配，避免边算边扩容
    lmax = max(len(s) for s in seqs) + 2
    lmax = min(lmax, 1024)

    bb = ESMBackbone(args.esm_name, device=args.device, layer=args.layer)
    dim = bb.dim
    print('ESM %s | dim=%d | layer=%d | Lmax=%d | device=%s'
          % (args.esm_name, dim, args.layer, lmax, bb.device))

    emb = np.lib.format.open_memmap(p_emb, mode='w+', dtype=np.float16, shape=(n, lmax, dim))
    msk = np.lib.format.open_memmap(p_mask, mode='w+', dtype=np.uint8, shape=(n, lmax))

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            chunk = seqs[start:start + args.batch_size]
            h, m = bb.encode(chunk)                     # [b, L, D], [b, L]
            b, L, _ = h.shape
            if L > lmax:
                raise SystemExit('实测长度 %d > 预算 %d' % (L, lmax))
            emb[start:start + b, :L, :] = h.astype(np.float16)
            emb[start:start + b, L:, :] = 0
            msk[start:start + b, :L] = m
            msk[start:start + b, L:] = 0
            done = min(start + args.batch_size, n)
            print('  %d/%d  %.0fs' % (done, n, time.time() - t0), end='\r')

    emb.flush()
    msk.flush()
    del emb, msk

    idx = pd.DataFrame({
        'row': np.arange(n, dtype=np.int64),
        'sha1': [sha1_text(s) for s in seqs],
        'sequence': seqs,
    })
    idx.to_csv(os.path.join(args.cache_dir, 'esm_index.csv'), index=False)
    with open(os.path.join(args.cache_dir, 'esm_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'esm_name': args.esm_name, 'layer': args.layer, 'dim': dim,
            'n_rows': n, 'lmax': lmax, 'dtype': 'float16',
            'seconds': round(time.time() - t0, 1),
            'data_csv': os.path.relpath(args.data, ROOT),
        }, f, indent=2, ensure_ascii=False)

    size_mb = (os.path.getsize(p_emb) + os.path.getsize(p_mask)) / 1e6
    print('\n完成：%d 行，耗时 %.0fs，占用 %.1f MB' % (n, time.time() - t0, size_mb))
    print('产物：%s' % args.cache_dir)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
