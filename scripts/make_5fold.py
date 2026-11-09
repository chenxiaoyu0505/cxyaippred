#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 aip_all.csv 生成固定 5 折划分（折内再切 train / val / test），并登记 manifest。

唯一输入
--------
    data/aip_all.csv        4,686 行，两列 sequence,label

输出（登记制：一份划分 = 一个目录 + 一张身份证）
----------------------------------------------
    data/splits/<split_id>/folds.csv        长表 sequence,label,fold,split
    data/splits/<split_id>/manifest.json    seed / 比例 / 行数 / sha256 / 生成时间

为什么不再写死在 data/ 根目录
-----------------------------
文件名里带 seed、只此一份，容易被误当成"唯一划分"，也说不清某次训练到底用了哪份。
现在划分按 id 归档，训练时把 split_id 与 sha256 写进 run 的 config.json，
任何一个数字都能倒查回当时那份划分文件。

结构（与原稿"开发集 80/20 + 独立测试集"对齐）
    每折：test = 该折留出的 20%
          剩余 80% 中：train = 80%（整体 64%）、val = 20%（整体 16%）

用法
----
    python scripts/make_5fold.py                 # seed=42, 5 折 → split_seed42_n5
    python scripts/make_5fold.py --seed 43       # 另一份划分，独立归档
    python scripts/make_5fold.py --force         # 覆盖同 id 的旧文件
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.manifest import write_manifest  # noqa: E402


def build_folds(allp, n_folds, val_ratio, seed):
    """返回长表 DataFrame；allp 需含 sequence,label，行序即主表行序"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    frames = []
    for k, (idx_pool, idx_test) in enumerate(skf.split(allp['sequence'], allp['label'])):
        pool = allp.iloc[idx_pool]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        _, idx_val = next(sss.split(pool['sequence'], pool['label']))

        role = pd.Series('train', index=pool.index)
        role.iloc[idx_val] = 'val'

        frames.append(pd.DataFrame({
            'sequence': pool['sequence'].values,
            'label': pool['label'].values,
            'fold': k,
            'split': role.values,
        }))
        frames.append(pd.DataFrame({
            'sequence': allp['sequence'].values[idx_test],
            'label': allp['label'].values[idx_test],
            'fold': k,
            'split': 'test',
        }))

    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(['fold', 'split', 'sequence']).reset_index(drop=True)


def check_folds(long, allp, n_folds):
    n = len(allp)
    assert len(long) == n_folds * n, '行数应为 折数 × 样本数'
    for k in range(n_folds):
        d = long[long['fold'] == k]
        sets = {s: set(d[d['split'] == s]['sequence']) for s in ('train', 'val', 'test')}
        assert not (sets['train'] & sets['val'])
        assert not (sets['train'] & sets['test'])
        assert not (sets['val'] & sets['test'])
        assert len(sets['train']) + len(sets['val']) + len(sets['test']) == n
    tests = [set(long[(long.fold == k) & (long.split == 'test')]['sequence'])
             for k in range(n_folds)]
    assert sum(len(s) for s in tests) == len(set().union(*tests)) == n
    return True


def fold_stats(long):
    out = []
    for k in sorted(long['fold'].unique()):
        d = long[long['fold'] == k]
        row = {'fold': int(k)}
        for sp in ('train', 'val', 'test'):
            s = d[d['split'] == sp]
            row[sp] = {'n': int(len(s)), 'pos': int(s['label'].sum())}
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'aip_all.csv'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'data', 'splits'))
    ap.add_argument('--split-id', default='', help='默认 split_seed{seed}_n{folds}')
    ap.add_argument('--seed', type=int, default=42, help='split seed（只管划分，不管训练）')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--val-ratio', type=float, default=0.20, help='训练部分里切给 val 的比例')
    ap.add_argument('--force', action='store_true', help='覆盖已存在的同 id 划分')
    ap.add_argument('--write-doc', action='store_true', help='同时刷新 data/FOLDS.md')
    args = ap.parse_args()

    split_id = args.split_id or 'split_seed%d_n%d' % (args.seed, args.folds)
    out_dir = os.path.join(args.out_dir, split_id)
    p_folds = os.path.join(out_dir, 'folds.csv')

    if os.path.exists(p_folds) and not args.force:
        print('已存在 %s（%d 字节），加 --force 才重算'
              % (os.path.relpath(p_folds, ROOT), os.path.getsize(p_folds)))
        return

    allp = pd.read_csv(args.data)
    allp['sequence'] = allp['sequence'].str.upper().str.strip()
    allp['label'] = allp['label'].astype(int)
    n, u = len(allp), allp['sequence'].nunique()
    if n != u:
        raise SystemExit('主表存在重复序列（%d 行 / %d 唯一），行号会对不上' % (n, u))
    print('数据 %d 条 | 阳性 %d (%.1f%%) | 最长 %d'
          % (n, int(allp['label'].sum()), 100.0 * allp['label'].mean(),
             max(len(s) for s in allp['sequence'])))

    long = build_folds(allp, args.folds, args.val_ratio, args.seed)
    check_folds(long, allp, args.folds)
    print('检查通过：折内三层互斥；%d 折的 test 恰好覆盖全部样本、互不重复' % args.folds)

    os.makedirs(out_dir, exist_ok=True)
    long.to_csv(p_folds, index=False)

    print('\n折  train                val                  test')
    for st in fold_stats(long):
        fmt = lambda d: '%d (%d/%d)' % (d['n'], d['pos'], d['n'] - d['pos'])  # noqa: E731
        print(' %d  %-20s %-20s %-20s'
              % (st['fold'], fmt(st['train']), fmt(st['val']), fmt(st['test'])))

    write_manifest(
        out_dir, exp=split_id,
        data_csv=args.data, split_csv=p_folds, root=ROOT, status='ready',
        extra={
            'kind': 'split',
            'split_seed': args.seed,
            'n_folds': args.folds,
            'val_ratio_within_train': args.val_ratio,
            'ratios': {'train': round(1 - args.val_ratio, 4) * (1 - 1.0 / args.folds),
                       'val': round(args.val_ratio, 4) * (1 - 1.0 / args.folds),
                       'test': round(1.0 / args.folds, 4)},
            'per_fold': fold_stats(long),
        })

    print('\n产物：')
    print('  %s' % os.path.relpath(p_folds, ROOT))
    print('  %s' % os.path.relpath(os.path.join(out_dir, 'manifest.json'), ROOT))

    if args.write_doc:
        doc = _fold_doc(split_id, args, allp, long)
        p_doc = os.path.join(ROOT, 'data', 'FOLDS.md')
        with open(p_doc, 'w', encoding='utf-8') as f:
            f.write(doc)
        print('  %s' % os.path.relpath(p_doc, ROOT))


def _fold_doc(split_id, args, allp, long):
    lines = []
    for st in fold_stats(long):
        fmt = lambda d: '%d (%d/%d)' % (d['n'], d['pos'], d['n'] - d['pos'])  # noqa: E731
        lines.append('| %d | %s | %s | %s |' % (st['fold'], fmt(st['train']),
                                                fmt(st['val']), fmt(st['test'])))
    return """# 5 折交叉验证划分（折内再分 train / val / test）

生成脚本：`scripts/make_5fold.py`
文件：`data/splits/{sid}/folds.csv`，**长表**（{n} 序列 × {k} 折 = {rows} 行），列：
`sequence, label, fold, split`。

| 层 | 占单折 | 用途 |
|---|---|---|
| `test` | 20% | 只评一次，不得用于任何选择 |
| `val` | 16%（训练部分的 20%） | 早停、选阈值、调类权重 |
| `train` | 64% | 梯度更新、EMA |

同一条序列在自己那一折里是 `test`，在另外四折里是 `train` 或 `val` —— 所以是长表。

## 数据

| 项目 | 值 |
|---|---|
| 主表 | `data/aip_all.csv` |
| 行数 | {n}（唯一序列 {n}） |
| 阳性 / 阴性 | {pos} / {neg} |
| split seed | {seed}（只管划分） |

## 每折分布

| 折 | train（阳/阴） | val（阳/阴） | test（阳/阴） |
|---|---|---|---|
{lines}

## 使用约定

1. 训练脚本读 `folds.csv`，**不要**在脚本里重新 split。
2. 换划分方式就换 split_id（如 `split_cluster40_n5`），不要覆盖。
3. 报告 5 折 mean ± sd，并附 5 个单折数值。
4. 阈值每折用该折 val 选、应用到该折 test；同时并列报固定 0.5 的结果。
5. `test` 只用于最后打分。
""".format(sid=split_id, n=len(allp), k=args.folds, rows=len(long),
           pos=int(allp['label'].sum()), neg=int(len(allp) - allp['label'].sum()),
           seed=args.seed, lines='\n'.join(lines))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
