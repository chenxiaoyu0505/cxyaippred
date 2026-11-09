#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对外可比实验：只用原 4,224 训练 + 在 462 上测试，与外部基线同一批测试样本。

为什么需要单独一条线
-------------------
5 折 CV 的测试集是 4,686 里轮换的 937 条，和外部基线用的 462 条**不是同一批样本**，
两边的数字不能直接摆在一张表里。审稿人会问"你比 XX 高多少"，
那张表必须建立在同一测试集上。

这条线做什么
-----------
    train = aip_train.csv 的 80%（3,379）   → 梯度更新
    val   = aip_train.csv 的 20%（  845）   → 早停 / 选阈值
    test  = aip_test.csv  （  462）         → 只评一次，与基线同一批

产物
----
    data/splits/legacy_seed42/folds.csv            fold 恒为 0 的划分
    experiments/legacy/runs/single_m0/             8 个文件
    experiments/legacy/results/                    metrics_detail.csv / metrics.json / summary.md
    experiments/legacy/manifest.json

用法
----
    python scripts/run_legacy.py
    python scripts/run_legacy.py --ema --pos-weight 3.0
    python scripts/run_legacy.py --exp legacy_ema
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.cache_util import clean_cache, has_cache   # noqa: E402
from src.manifest import write_manifest            # noqa: E402

PY = sys.executable


def build_legacy_split(split_id='legacy_seed42', val_ratio=0.20, seed=42):
    """aip_train 里切 80/20 作 train/val，aip_test 全部作 test"""
    p_tr = os.path.join(ROOT, 'data', 'aip_train.csv')
    p_te = os.path.join(ROOT, 'data', 'aip_test.csv')
    p_all = os.path.join(ROOT, 'data', 'aip_all.csv')
    tr, te, allp = (pd.read_csv(p) for p in (p_tr, p_te, p_all))
    for d in (tr, te):
        d['sequence'] = d['sequence'].str.upper().str.strip()

    s_tr, s_te, s_all = set(tr['sequence']), set(te['sequence']), set(allp['sequence'])
    if s_tr & s_te:
        raise SystemExit('原训练集与测试集有 %d 条重叠，不能这样建基线对比' % len(s_tr & s_te))
    if not (s_tr | s_te) <= s_all:
        raise SystemExit('原 train/test 有序列不在 aip_all 中（%d 条）'
                         % len((s_tr | s_te) - s_all))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_tr, idx_val = next(sss.split(tr['sequence'], tr['label']))

    rows = []
    for idx, split in ((idx_tr, 'train'), (idx_val, 'val')):
        d = tr.iloc[idx]
        rows.append(pd.DataFrame({'sequence': d['sequence'].values,
                                  'label': d['label'].astype(int).values,
                                  'fold': 0, 'split': split}))
    rows.append(pd.DataFrame({'sequence': te['sequence'].values,
                              'label': te['label'].astype(int).values,
                              'fold': 0, 'split': 'test'}))
    long = pd.concat(rows, ignore_index=True).sort_values(['split', 'sequence'])

    if len(long) != len(allp):
        raise SystemExit('划分 %d 行 ≠ 主表 %d 行' % (len(long), len(allp)))
    if long['sequence'].duplicated().any():
        raise SystemExit('划分里有重复序列')

    out_dir = os.path.join(ROOT, 'data', 'splits', split_id)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'folds.csv')
    long.to_csv(p, index=False)

    stats = {}
    for sp in ('train', 'val', 'test'):
        d = long[long['split'] == sp]
        stats[sp] = {'n': int(len(d)), 'pos': int(d['label'].sum())}
        print('  %-5s %4d 条（阳 %d）' % (sp, stats[sp]['n'], stats[sp]['pos']))

    write_manifest(out_dir, exp=split_id, data_csv=p_all, split_csv=p, root=ROOT,
                   status='ready',
                   extra={'kind': 'split_legacy', 'split_seed': seed,
                          'val_ratio_within_train': val_ratio, 'per_split': stats})
    print('划分：%s' % os.path.relpath(p, ROOT))
    return p


def run(cmd, what):
    print('\n>>> %s' % what)
    print('    %s' % ' '.join(cmd[1:]))
    t0 = time.time()
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise SystemExit('失败：%s' % what)
    print('<<< 完成，用时 %.0fs' % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='legacy')
    ap.add_argument('--split-id', default='legacy_seed42')
    ap.add_argument('--run-name', default='single_m0')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--pos-weight', type=float, default=0.0)
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--ema-alpha', type=float, default=0.7)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--ema-warmup', type=int, default=3)
    ap.add_argument('--device', default='')
    ap.add_argument('--force-split', action='store_true', help='重建 legacy 划分')
    ap.add_argument('--keep-cache', action='store_true')
    args = ap.parse_args()

    exp_dir = os.path.join(ROOT, 'experiments', args.exp)
    split_csv = os.path.join(ROOT, 'data', 'splits', args.split_id, 'folds.csv')
    data_csv = os.path.join(ROOT, 'data', 'aip_all.csv')

    print('=' * 70)
    print('对外可比实验 %s | 原训练/测试口径 | 解释器 %s' % (args.exp, PY))
    print('=' * 70)

    # ---- 1. 划分 ----
    if args.force_split or not os.path.exists(split_csv):
        print('\n[1/4] 生成 legacy 划分（aip_train 80/20 + aip_test）')
        build_legacy_split(args.split_id)
    else:
        print('\n[1/4] 划分已存在：%s' % os.path.relpath(split_csv, ROOT))

    # ---- 2. 缓存 ----
    cache_dir = os.path.join(ROOT, 'cache')
    if not has_cache(cache_dir):
        run([PY, os.path.join(HERE, 'cache_esm.py')], '生成 ESM 缓存（约 0.9 GB）')
    else:
        print('\n[2/4] ESM 缓存已存在')

    # ---- 3. 训练（单折） ----
    cmd = [PY, os.path.join(HERE, 'train_aip.py'),
           '--fold', '0', '--seed', '0',
           '--exp', args.exp, '--split-id', args.split_id,
           '--run-name', args.run_name,
           '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
           '--lr', str(args.lr), '--patience', str(args.patience),
           '--pos-weight', str(args.pos_weight),
           '--ema-alpha', str(args.ema_alpha), '--ema-decay', str(args.ema_decay),
           '--ema-warmup', str(args.ema_warmup)]
    if args.ema:
        cmd.append('--ema')
    if args.device:
        cmd += ['--device', args.device]
    run(cmd, '训练（原训练/测试口径）')

    # ---- 4. 评估 ----
    run([PY, os.path.join(HERE, 'eval_aip.py'), '--exp', args.exp, '--no-ensemble'],
        '汇总评估')

    write_manifest(exp_dir, exp=args.exp, data_csv=data_csv, split_csv=split_csv,
                   root=ROOT, status='finished',
                   extra={
                       'kind': 'train_legacy',
                       'split_id': args.split_id,
                       'folds': [0], 'model_seeds': [0], 'n_runs': 1,
                       'note': '原 4,224 训练 / 462 测试，用于与外部基线同一测试集比较',
                       'settings': {'epochs': args.epochs, 'batch_size': args.batch_size,
                                    'lr': args.lr, 'patience': args.patience,
                                    'pos_weight': args.pos_weight, 'ema': args.ema},
                   })

    if not args.keep_cache:
        removed = clean_cache(cache_dir)
        if removed:
            print('\n[清理] 已删除 ESM 缓存 %.1f MB' % (sum(s for _, s in removed) / 1e6))

    print('\n实验目录：%s' % os.path.relpath(exp_dir, ROOT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
