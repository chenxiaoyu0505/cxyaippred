#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定独立 test，再在 train 上做 5 折；每折内 80% train / 20% val。

两种模式
--------
1. 不传 --test-data：先留出 holdout（默认 10%），再在剩余 90% 上做 5 折。
2. 传 --test-data：--data 全部作为 CV pool，--test-data 作为每折共用的固定 test；
   适合保留原始 train/test 口径（例如 4,178 / 452）。

输出
----
    data/splits/<split_id>/folds.csv      长表 sequence,label,fold,split
    data/splits/<split_id>/holdout.csv    内部 holdout（模式 1）
    data/splits/<split_id>/test.csv       外部固定 test（模式 2）
    data/splits/<split_id>/manifest.json  数据/划分/环境/每折统计

用法
----
    # 模式 1：从总数据抽 10% holdout
    python scripts/make_holdout_cv.py \
        --data data/aip_all_nobip.csv \
        --split-id split_holdout10_cv5_seed42_nobip \
        --seed 42 --folds 5 --holdout-ratio 0.10

    # 模式 2：固定原始 test
    python scripts/make_holdout_cv.py \
        --data data/aip_train_nobip.csv \
        --test-data data/aip_test_nobip.csv \
        --split-id split_origtest_cv5_seed42_nobip \
        --seed 42 --folds 5
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.manifest import sha256_file, write_manifest  # noqa: E402


def _stats(df):
    return {'n': int(len(df)), 'pos': int(df['label'].sum()),
            'neg': int((df['label'] == 0).sum())}


def _read_pairs(path, what):
    df = pd.read_csv(path)
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise SystemExit('%s 需要 sequence,label 两列：%s' % (what, path))
    df['sequence'] = df['sequence'].str.upper().str.strip()
    df['label'] = df['label'].astype(int)
    n, u = len(df), df['sequence'].nunique()
    if n != u:
        raise SystemExit('%s 存在重复序列（%d 行 / %d 唯一）' % (what, n, u))
    if not set(df['label'].unique()) <= {0, 1}:
        raise SystemExit('%s 的 label 只能是 0/1' % what)
    if df['sequence'].isna().any() or (df['sequence'] == '').any():
        raise SystemExit('%s 存在空序列' % what)
    return df


def build_split(allp, n_folds, holdout_ratio, seed, external_test=None):
    if external_test is None:
        idx_all = list(range(len(allp)))
        idx_pool, idx_hold = train_test_split(
            idx_all, test_size=holdout_ratio, random_state=seed,
            stratify=allp['label'].values,
        )
        pool = allp.iloc[idx_pool].reset_index(drop=True)
        hold = allp.iloc[idx_hold].reset_index(drop=True)
    else:
        # 外部固定 test：allp 全部作为 CV pool，test 单独给出。
        pool = allp.reset_index(drop=True)
        hold = external_test.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    frames = []
    for k, (idx_train, idx_val) in enumerate(skf.split(pool['sequence'], pool['label'])):
        tr = pool.iloc[idx_train]
        va = pool.iloc[idx_val]
        frames.append(pd.DataFrame({
            'sequence': tr['sequence'].values,
            'label': tr['label'].values,
            'fold': k,
            'split': 'train',
        }))
        frames.append(pd.DataFrame({
            'sequence': va['sequence'].values,
            'label': va['label'].values,
            'fold': k,
            'split': 'val',
        }))
        # 每一折都用同一份 holdout 作为 test；模型没有见过它。
        frames.append(pd.DataFrame({
            'sequence': hold['sequence'].values,
            'label': hold['label'].values,
            'fold': k,
            'split': 'test',
        }))

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(['fold', 'split', 'sequence']).reset_index(drop=True)
    return long, pool, hold


def check_split(long, pool, hold, n_folds):
    n = len(pool) + len(hold)
    assert len(long) == n_folds * n, '行数应为 折数 × 总样本数'
    hold_set = set(hold['sequence'])
    pool_set = set(pool['sequence'])
    assert not (hold_set & pool_set), 'holdout 与 CV pool 有重叠'
    assert len(hold_set) + len(pool_set) == n

    for k in range(n_folds):
        d = long[long['fold'] == k]
        sets = {s: set(d[d['split'] == s]['sequence'])
                for s in ('train', 'val', 'test')}
        assert not (sets['train'] & sets['val'])
        assert not (sets['train'] & sets['test'])
        assert not (sets['val'] & sets['test'])
        assert sets['train'] | sets['val'] == pool_set, 'train+val 应等于 CV pool'
        assert sets['test'] == hold_set, '每折 test 应等于固定 holdout'
        assert len(sets['train']) + len(sets['val']) + len(sets['test']) == n
    return True


def fold_stats(long):
    out = []
    for k in sorted(long['fold'].unique()):
        d = long[long['fold'] == k]
        row = {'fold': int(k)}
        for sp in ('train', 'val', 'test'):
            row[sp] = _stats(d[d['split'] == sp])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='输入 CSV，需含 sequence,label')
    ap.add_argument('--test-data', default='',
                    help='可选：固定独立 test CSV；给出后不再从 --data 抽 holdout')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'data', 'splits'))
    ap.add_argument('--split-id', required=True)
    ap.add_argument('--seed', type=int, default=42, help='split seed')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--holdout-ratio', type=float, default=0.10)
    ap.add_argument('--force', action='store_true', help='覆盖已存在的同 id 划分')
    args = ap.parse_args()

    external = bool(args.test_data)
    if not external and not (0.0 < args.holdout_ratio < 1.0):
        raise SystemExit('--holdout-ratio 必须在 (0,1) 内')
    if args.folds < 2:
        raise SystemExit('--folds 至少为 2')

    out_dir = os.path.join(args.out_dir, args.split_id)
    p_folds = os.path.join(out_dir, 'folds.csv')
    if os.path.exists(p_folds) and not args.force:
        print('已存在 %s（%d 字节），加 --force 才重算'
              % (os.path.relpath(p_folds, ROOT), os.path.getsize(p_folds)))
        return

    allp = _read_pairs(args.data, '--data')
    test = _read_pairs(args.test_data, '--test-data') if external else None
    if test is not None:
        overlap = set(allp['sequence']) & set(test['sequence'])
        if overlap:
            raise SystemExit('--data 与 --test-data 有 %d 条重叠，例如 %s'
                             % (len(overlap), sorted(overlap)[0]))
        total = len(allp) + len(test)
        print('数据 %d = train %d + test %d | 阳性 %d / %d (%.1f%%) | %d 折'
              % (total, len(allp), len(test),
                 int(allp['label'].sum()) + int(test['label'].sum()), total,
                 100.0 * (allp['label'].sum() + test['label'].sum()) / total,
                 args.folds))
    else:
        print('数据 %d 条 | 阳性 %d (%.1f%%) | holdout %.0f%% | %d 折'
              % (len(allp), int(allp['label'].sum()),
                 100.0 * allp['label'].mean(),
                 100.0 * args.holdout_ratio, args.folds))

    long, pool, hold = build_split(
        allp, args.folds, args.holdout_ratio, args.seed, external_test=test)
    check_split(long, pool, hold, args.folds)
    print('检查通过：test 独立；每折 train/val 互斥且覆盖 CV pool；test 每折固定')

    os.makedirs(out_dir, exist_ok=True)
    p_hold = os.path.join(out_dir, 'test.csv' if external else 'holdout.csv')
    p_table = os.path.join(out_dir, 'table.csv')
    # 特征/cache 需要覆盖 train+test 的全部序列；单独存一份，避免 run_5fold
    # 误把 train-only CSV 当主表而导致 test 序列找不到。
    table_all = pd.concat([pool[['sequence', 'label']],
                           hold[['sequence', 'label']]], ignore_index=True)
    if table_all['sequence'].duplicated().any():
        raise SystemExit('table.csv 出现重复序列')
    long.to_csv(p_folds, index=False)
    hold[['sequence', 'label']].to_csv(p_hold, index=False)
    table_all.to_csv(p_table, index=False)

    print('\n折  train                val                  test')
    for st in fold_stats(long):
        fmt = lambda d: '%d (%d/%d)' % (d['n'], d['pos'], d['neg'])  # noqa: E731
        print(' %d  %-20s %-20s %-20s'
              % (st['fold'], fmt(st['train']), fmt(st['val']), fmt(st['test'])))

    total = len(pool) + len(hold)
    extra = {
        'kind': 'split',
        'design': 'external_test_cv' if external else 'holdout_then_cv',
        'split_seed': args.seed,
        'n_folds': args.folds,
        'holdout_ratio': None if external else args.holdout_ratio,
        'cv_ratio': round(len(pool) / total, 4),
        'val_ratio_within_cv': round(1.0 / args.folds, 4),
        'holdout': _stats(hold),
        'cv_pool': _stats(pool),
        'per_fold': fold_stats(long),
        'table_csv': os.path.relpath(p_table, ROOT).replace('\\', '/'),
        'table_sha256': sha256_file(p_table),
        'train_csv': os.path.relpath(args.data, ROOT).replace('\\', '/'),
        'holdout_csv': os.path.relpath(p_hold, ROOT).replace('\\', '/'),
        'holdout_sha256': sha256_file(p_hold),
        'note': ('--test-data 作为固定 test；train/val 仅在 --data 内划分。'
                 if external else
                 '每折 test 都是同一份固定 holdout；train/val 仅在 CV pool 内划分。'),
    }
    if external:
        extra['test_csv'] = extra['holdout_csv']
        extra['test_sha256'] = extra['holdout_sha256']
        extra['test_data_csv'] = os.path.relpath(args.test_data, ROOT).replace('\\', '/')

    write_manifest(
        out_dir, exp=args.split_id,
        data_csv=p_table, split_csv=p_folds, root=ROOT, status='ready',
        extra=extra)

    print('\n产物：')
    print('  %s' % os.path.relpath(p_folds, ROOT))
    print('  %s' % os.path.relpath(p_hold, ROOT))
    print('  %s' % os.path.relpath(p_table, ROOT))
    print('  %s' % os.path.relpath(os.path.join(out_dir, 'manifest.json'), ROOT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
