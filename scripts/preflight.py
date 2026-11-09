#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开跑前自检（只读，不训练、不下载）。

检查项
------
1. 解释器 / torch / GPU（含一次真卷积，识别 sm_120 不匹配）
2. 数据 `data/aip_all.csv`
3. 划分 `data/splits/<split_id>/folds.csv`（存在性、5 折覆盖、sha256 与 manifest 一致）
4. 缓存状态（hand_feats 是否有、ESM 缓存是否有 → 决定 run_5fold 会不会现建）
5. ESM-2 权重是否已在本地 HF 缓存（决定会不会联网下载）
6. experiments/ 现有实验、D 盘剩余空间

退出码 0 = 可以开跑；1 = 有硬性阻塞。
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLIT_ID = 'split_seed42_n5'
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
    print(f'  cuda    : {cuda_ok}')
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
    else:
        check('CUDA 可用', False, 'torch.cuda.is_available() = False')

    # ---------- 2. 数据 ----------
    print('\n[2] 数据')
    data_csv = ROOT / 'data' / 'aip_all.csv'
    check('aip_all.csv 存在', data_csv.exists(), str(data_csv))
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
        check('阳性比例正常', 0.2 < pos / n < 0.6, f'{pos / n:.1%}')

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
        test_rows = f[f['split'] == 'test']
        check('5 折 test 互斥且覆盖全集',
              len(test_rows) == n and test_rows['sequence'].nunique() == n,
              f'test 共 {len(test_rows)} 行 / {test_rows["sequence"].nunique()} 唯一')
        for k in folds:
            sub = f[(f['fold'] == k) & (f['split'] == 'train')]
            val = f[(f['fold'] == k) & (f['split'] == 'val')]
            tst = f[(f['fold'] == k) & (f['split'] == 'test')]
            print(f'  fold{k}: train {len(sub)} (pos {int(sub["label"].sum())}) / '
                  f'val {len(val)} ({int(val["label"].sum())}) / test {len(tst)} ({int(tst["label"].sum())})')
            if len(val) == 0 or len(tst) == 0:
                check(f'fold{k} 三层齐全', False, 'val 或 test 为空')
        man = split_dir / 'manifest.json'
        if man.exists():
            import hashlib
            sha = hashlib.sha256(folds_csv.read_bytes()).hexdigest()
            rec = json.loads(man.read_text(encoding='utf-8'))
            check('sha256 与 manifest 一致', rec.get('split', {}).get('sha256') == sha,
                  f'{sha[:16]}…')
        else:
            check('manifest.json 存在', False, str(man), hard=False)

    # ---------- 4. 缓存 ----------
    print('\n[4] 缓存')
    hand = ROOT / 'cache' / 'hand_feats.npy'
    if hand.exists():
        arr = np.load(hand, mmap_mode='r')
        check('hand_feats 形状', tuple(arr.shape) == (n, 694), f'{tuple(arr.shape)} {arr.dtype}')
    else:
        warn.append('  [WARN] hand_feats.npy 不存在 — 首次运行会现算（约 1–2 分钟）')
    from src.cache_util import ESM_FILES, cache_size_mb, has_cache
    if has_cache(ROOT / 'cache'):
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
    check('D 盘剩余 ≥ 5 GB', free > 5, f'{free:.1f} GB')

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
