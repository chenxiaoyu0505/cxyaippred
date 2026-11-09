#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据读取与分批。

数据契约
--------
1. `data/aip_all.csv`：唯一数据源，两列 `sequence,label`（4,686 行，无重复序列）。
2. `data/folds_5cv_seed42.csv`：长表 `sequence,label,fold,split`（4,686 × 5 行）。
   同一序列在自己那一折里是 test，在其余四折里是 train 或 val。
3. 所有特征按**行号**对齐：第 i 行就是 aip_all.csv 的第 i 行。
   fold 文件里的序列被翻译成行号后再用，绝不重新排序——排序一变，全部错位。

缓存
----
* `cache/hand_feats.npy`  [N, 694] float32   手工特征（确定性，算一次即可）
* `cache/esm_emb_fp16.npy` [N, L, 1280] fp16 残基级隐状态（由 scripts/cache_esm.py 生成）
* `cache/esm_mask.npy`     [N, L] uint8      1 = 真实残基
* `cache/esm_index.csv`    row, sha1(sequence) 防止行序漂移
"""
import hashlib
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基础

def clean_sequence(s):
    """大写、去空白、只保留 20 种标准氨基酸与常见非标准字符 X/U/B/Z/O"""
    s = str(s).strip().upper()
    s = ''.join(ch for ch in s if ch.isalpha())
    return s


def read_fasta(path):
    seqs, cur = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur:
                    seqs.append(''.join(cur))
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append(''.join(cur))
    return [clean_sequence(s) for s in seqs]


def sha1_text(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- 主表

def load_table(path):
    """
    读 aip_all.csv，返回 DataFrame[sequence, label]，行序 = 文件行序 = 特征行序。
    不做去重，但会检查是否有重复序列（有则报错，因为行号会歧义）。
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if 'sequence' not in cols and 'squence' not in cols:
        raise SystemExit('%s 缺少 sequence 列：%s' % (path, list(df.columns)))
    seq_col = cols.get('sequence', cols.get('squence'))
    if 'label' not in cols:
        raise SystemExit('%s 缺少 label 列：%s' % (path, list(df.columns)))

    out = pd.DataFrame({
        'sequence': df[seq_col].map(clean_sequence),
        'label': df[cols['label']].astype(int),
    })
    if (out['sequence'] == '').any():
        raise SystemExit('存在空序列，行号 %s' % np.where(out['sequence'] == '')[0][:5])
    dup = out['sequence'].duplicated()
    if dup.any():
        raise SystemExit('存在 %d 条重复序列（行号 %s），行号对齐会歧义'
                         % (int(dup.sum()), np.where(dup)[0][:5]))
    if not set(out['label'].unique()) <= {0, 1}:
        raise SystemExit('label 只能是 0/1，实际 %s' % sorted(out['label'].unique()))
    return out


def load_folds(folds_csv, fold, table):
    """
    返回 {'train': idx, 'val': idx, 'test': idx}，idx 是 table 的行号。
    会校验：三集合互斥、并集 = 全体、test 恰好等于该 fold 的 test。
    """
    f = pd.read_csv(folds_csv)
    need = {'sequence', 'fold', 'split'}
    if not need <= set(f.columns):
        raise SystemExit('%s 缺少列 %s（实际 %s）' % (folds_csv, need - set(f.columns), list(f.columns)))

    f = f[f['fold'] == fold]
    if f.empty:
        raise SystemExit('%s 里没有 fold=%d' % (folds_csv, fold))

    pos = {s: i for i, s in enumerate(table['sequence'])}
    missing = [s for s in f['sequence'] if s not in pos]
    if missing:
        raise SystemExit('%d 条划分里的序列不在主表中，例如 %s' % (len(missing), missing[:3]))

    out = {}
    for split in ('train', 'val', 'test'):
        idx = np.array([pos[s] for s in f.loc[f['split'] == split, 'sequence']], dtype=np.int64)
        out[split] = np.unique(idx)

    n = len(table)
    union = np.concatenate([out['train'], out['val'], out['test']])
    if len(np.unique(union)) != len(union):
        raise SystemExit('fold=%d 的 train/val/test 有重叠' % fold)
    if len(union) != n:
        raise SystemExit('fold=%d 只覆盖 %d/%d 行' % (fold, len(union), n))
    return out


def check_folds_complete(folds_csv, table):
    """5 折的 test 必须正好把主表切成 5 份、互不重叠"""
    f = pd.read_csv(folds_csv)
    tests = [set(f[(f.fold == k) & (f.split == 'test')]['sequence']) for k in sorted(f.fold.unique())]
    allseq = set(table['sequence'])
    for i, a in enumerate(tests):
        for b in tests[i + 1:]:
            if a & b:
                raise SystemExit('两折 test 有交集：%d 条' % len(a & b))
    if set().union(*tests) != allseq:
        raise SystemExit('各折 test 的并集不等于主表（差 %d 条）'
                         % len(allseq - set().union(*tests)))
    return True


# ---------------------------------------------------------------- 特征缓存

def hand_feats(table, cache_dir='cache'):
    """
    694 维手工特征。第一次算完存盘，之后直接读，避免每次训练重算（也避免版本漂移）。
    """
    from .features import compute_handcraft, handcraft_feature_names

    path = os.path.join(cache_dir, 'hand_feats.npy')
    seqs = table['sequence'].tolist()
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape == (len(seqs), len(handcraft_feature_names())):
            return arr.astype(np.float32)
        print('[warn] %s 形状不符（%s），重算' % (path, arr.shape))

    arr = np.stack([compute_handcraft(s) for s in seqs], axis=0).astype(np.float32)
    if arr.shape[1] != 694:
        raise SystemExit('手工特征维度 %d ≠ 694' % arr.shape[1])
    if np.isnan(arr).any():
        raise SystemExit('手工特征出现 NaN')
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, arr)
    return arr


def load_esm_cache(cache_dir, table):
    """
    返回 (emb [N,L,D] fp16, mask [N,L] uint8)。
    用 memmap 打开，不占内存；并用 esm_index.csv 校验行序未变。
    """
    p_emb = os.path.join(cache_dir, 'esm_emb_fp16.npy')
    p_mask = os.path.join(cache_dir, 'esm_mask.npy')
    p_idx = os.path.join(cache_dir, 'esm_index.csv')
    for p in (p_emb, p_mask, p_idx):
        if not os.path.exists(p):
            raise SystemExit('缺少 ESM 缓存 %s，先运行 scripts/cache_esm.py' % p)

    idx = pd.read_csv(p_idx)
    if len(idx) != len(table):
        raise SystemExit('缓存 %d 行，主表 %d 行，对不上' % (len(idx), len(table)))
    want = [sha1_text(s) for s in table['sequence']]
    got = idx['sha1'].tolist()
    if got != want:
        bad = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise SystemExit('ESM 缓存行序与主表不一致（首个不符在第 %d 行），请重建缓存' % bad)

    emb = np.load(p_emb, mmap_mode='r')
    mask = np.load(p_mask, mmap_mode='r')
    if emb.shape[0] != len(table):
        raise SystemExit('缓存 %d 行，主表 %d 行' % (emb.shape[0], len(table)))
    return emb, mask


# ---------------------------------------------------------------- 分批

def iter_batches(indices, batch_size, shuffle=False, seed=0):
    """
    按行号分批。因为特征预先把好，这里只做索引搬运，不需要 DataLoader。
    shuffle 用独立的 RandomState，保证同一 seed 顺序可复现。
    """
    idx = np.asarray(indices, dtype=np.int64)
    if shuffle:
        rng = np.random.RandomState(seed)
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), batch_size):
        yield idx[i:i + batch_size]
