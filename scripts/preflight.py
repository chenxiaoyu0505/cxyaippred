#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开跑前自检（只读，不训练、不下载）。

检查项
------
1. 解释器 / torch / GPU（含一次真卷积，识别 sm_120 不匹配）
2. 固定协议主表（4,182 train + 452 test）
3. 划分 `data/splits/<split_id>/folds.csv`（5 折 train/val、固定 test、sha256）
4. 缓存状态（hand_feats 是否有、ESM 缓存是否有 → 决定 run_5fold 会不会现建）
5. ESM-2 权重是否已在本地 HF 缓存（决定会不会联网下载）
6. experiments/ 现有实验、磁盘剩余空间

退出码 0 = 可以开跑；1 = 有硬性阻塞。
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLIT_ID = 'split_origtest_cv5_seed42_nobip'
ESM_REPO = 'models--facebook--esm2_t33_650M_UR50D'

ok, warn, fail = [], [], []


def check(name, cond, detail='', hard=True):
    if cond:
        ok.append(f'  [OK]   {name}' + (f' — {detail}' if detail else ''))
    elif hard:
        fail.append(f'  [FAIL] {name}' + (f' — {detail}' if detail else ''))
    else:
        warn.append(f'  [WARN] {name}' + (f' — {detail}' if detail else ''))


def main():
    print('=' * 78)
    print('cxyaippred preflight')
    print('=' * 78)

    # ---------- 1. 环境 ----------
    import torch
    import numpy as np
    import pandas as pd

    print('\n[1] 环境')
    print(f'  python  : {sys.version.split()[0]}  ({sys.executable})')
    print(f'  torch   : {torch.__version__}')
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, 'mps', None)
                  and torch.backends.mps.is_available())
    print(f'  cuda    : {cuda_ok}')
    print(f'  mps     : {mps_ok}')
    if cuda_ok:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f'  gpu     : {name}  cap={cap}  {vram:.1f} GB')
        # 真跑一次卷积：is_available() 在坏环境里也会 True
        try:
            x = torch.randn(2, 4, 16, device='cuda')
            y = torch.nn.functional.conv1d(x, torch.randn(4, 4, 3, device='cuda'))
            torch.cuda.synchronize()
            check('GPU 卷积', tuple(y.shape) == (2, 4, 14), f'输出 {tuple(y.shape)}')
        except Exception as e:  # noqa: BLE001
            check('GPU 卷积', False, str(e).split('\n')[0][:120])
    elif mps_ok:
        check('GPU 可用', True, 'MPS（Apple Silicon；CUDA 不可用）')
    else:
        check('GPU 可用', False, 'CUDA 与 MPS 都不可用；CPU 训练会很慢', hard=False)

    # ---------- 2. 数据 ----------
    print('\n[2] 数据')
    data_csv = ROOT / 'data' / 'splits' / SPLIT_ID / 'table.csv'
    train_csv = ROOT / 'data' / 'aip_train_nobip.csv'
    test_csv = ROOT / 'data' / 'aip_test_nobip.csv'
    check('固定协议主表存在', data_csv.exists(), str(data_csv))
    check('train/test 原始文件存在', train_csv.exists() and test_csv.exists())
    n = pos = 0
    if data_csv.exists():
        df = pd.read_csv(data_csv)
        cols = list(df.columns)
        check('列名 = sequence,label', cols == ['sequence', 'label'], str(cols))
        n = len(df)
        pos = int(df['label'].sum())
        dup = int(df['sequence'].duplicated().sum())
        maxlen = int(df['sequence'].str.len().max())
        print(f'  行数 {n}  阳性 {pos} ({pos / n:.1%})  重复序列 {dup}  最长 {maxlen}')
        check('无重复序列', dup == 0, f'{dup} 条重复')
        check('固定协议总数', n == 4634 and pos == 1788, f'{n} 行 / {pos} 阳性')
        if train_csv.exists() and test_csv.exists():
            tr = pd.read_csv(train_csv)
            te = pd.read_csv(test_csv)
            check('train/test 数量', len(tr) == 4182 and len(te) == 452,
                  f'{len(tr)} / {len(te)}')
            check('train/test 序列互斥',
                  not (set(tr['sequence']) & set(te['sequence'])))
            check('主表等于 train + test',
                  set(df['sequence']) == set(tr['sequence']) | set(te['sequence']))

    # ---------- 3. 划分 ----------
    print('\n[3] 划分')
    split_dir = ROOT / 'data' / 'splits' / SPLIT_ID
    folds_csv = split_dir / 'folds.csv'
    check('folds.csv 存在', folds_csv.exists(), str(folds_csv))
    if folds_csv.exists() and data_csv.exists():
        f = pd.read_csv(folds_csv)
        check('列齐全', {'sequence', 'label', 'fold', 'split'} <= set(f.columns), str(list(f.columns)))
        folds = sorted(f['fold'].unique().tolist())
        check('5 折', folds == [0, 1, 2, 3, 4], str(folds))
        fixed_test = set(pd.read_csv(test_csv)['sequence']) if test_csv.exists() else set()
        check('每折使用相同的固定 test',
              all(set(f[(f['fold'] == k) & (f['split'] == 'test')]['sequence']) == fixed_test
                  for k in folds), f'固定 test {len(fixed_test)} 条')
        train_pool = set(pd.read_csv(train_csv)['sequence']) if train_csv.exists() else set()
        for k in folds:
            sub = f[(f['fold'] == k) & (f['split'] == 'train')]
            val = f[(f['fold'] == k) & (f['split'] == 'val')]
            tst = f[(f['fold'] == k) & (f['split'] == 'test')]
            print(f'  fold{k}: train {len(sub)} (pos {int(sub["label"].sum())}) / '
                  f'val {len(val)} ({int(val["label"].sum())}) / test {len(tst)} ({int(tst["label"].sum())})')
            check(f'fold{k} train/val 划分',
                  len(sub) > 0 and len(val) > 0 and
                  set(sub['sequence']).isdisjoint(val['sequence']) and
                  set(sub['sequence']) | set(val['sequence']) == train_pool)
        val_counts = f[f['split'] == 'val']['sequence'].value_counts()
        check('train pool 的 5 折 val 各覆盖一次',
              len(val_counts) == len(train_pool) and val_counts.eq(1).all())
        man = split_dir / 'manifest.json'
        if man.exists():
            from src.data_io import sha256_crlf
            sha = sha256_crlf(folds_csv)
            rec = json.loads(man.read_text(encoding='utf-8'))
            check('sha256 与 manifest 一致（CRLF 归一化）',
                  rec.get('split', {}).get('sha256') == sha, f'{sha[:16]}…')
        else:
            check('manifest.json 存在', False, str(man), hard=False)

    # ---------- 4. 缓存 ----------
    print('\n[4] 缓存')
    hand = ROOT / 'cache' / 'hand_feats_amphipath.npy'
    if hand.exists():
        arr = np.load(hand, mmap_mode='r')
        shape_ok = tuple(arr.shape) == (n, 704)
        check('hand_feats 形状', shape_ok,
              f'{tuple(arr.shape)} {arr.dtype}' +
              ('' if shape_ok else '（运行时会按主表重算，不阻塞）'),
              hard=False)
    else:
        warn.append('  [WARN] 704 维手工特征缓存不存在 — 首次运行会现算')
    from src.cache_util import ESM_FILES, cache_size_mb, has_cache
    if has_cache(ROOT / 'cache', expected_n=n,
                 esm_name='facebook/esm2_t33_650M_UR50D'):
        print(f'  ESM 缓存：已有 {cache_size_mb(ROOT / "cache"):.0f} MB → 直接复用')
    else:
        miss = [x for x in ESM_FILES if not (ROOT / 'cache' / x).exists()]
        print(f'  ESM 缓存：无（缺 {len(miss)} 个文件）→ run_5fold 会先建缓存（约 5–10 分钟）')

    # ---------- 5. ESM 权重 ----------
    print('\n[5] ESM-2 权重')
    hub = Path(os.path.expanduser('~')) / '.cache' / 'huggingface' / 'hub' / ESM_REPO
    w = list(hub.glob('snapshots/*/model.safetensors')) if hub.exists() else []
    if w:
        size = w[0].stat().st_size / 1024 ** 2
        check('本地已有权重', size > 1000, f'{w[0].parent.name[:12]}… model.safetensors {size:.0f} MB')
    else:
        check('本地已有权重', False,
              '未找到 → 首次加载会联网从 huggingface.co 下载 2.43 GB', hard=False)
        warn.append('  [WARN] 若无网络，加载会失败；可先手动 `hf download facebook/esm2_t33_650M_UR50D`')

    # ---------- 6. 输出与磁盘 ----------
    print('\n[6] 输出与磁盘')
    exp_root = ROOT / 'experiments'
    existing = sorted([p.name for p in exp_root.iterdir() if p.is_dir()]) if exp_root.exists() else []
    print(f'  experiments/ 现有：{existing or "（空）"}')
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    check('磁盘剩余 ≥ 5 GB', free > 5, f'{free:.1f} GB')

    # ---------- 汇总 ----------
    print('\n' + '=' * 78)
    for line in ok:
        print(line)
    for line in warn:
        print(line)
    for line in fail:
        print(line)
    print('=' * 78)
    if fail:
        print(f'结论：NOT READY（{len(fail)} 项硬性阻塞）')
        return 1
    print('结论：READY — 可以开跑')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
