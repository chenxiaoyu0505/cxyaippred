#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据读取与分批。

数据契约
--------
1. 默认主表是 `data/splits/split_origtest_cv5_seed42_nobip/table.csv`，
   共 4,634 行（4,182 train pool + 452 固定独立 test）。
2. 同目录 `folds.csv` 是长表；每折的 test 都是同一份 452 条，
   训练池内的 5 个 val 互斥并覆盖训练池一次。
3. 所有特征按**行号**对齐：第 i 行就是本次指定主表的第 i 行。
   fold 文件里的序列被翻译成行号后再用，绝不重新排序——排序一变，全部错位。

缓存
----
* `cache/hand_feats.npy`  [N, 694] float32   手工特征（确定性，算一次即可）
* `cache/esm_emb_fp16.npy` [N, L, 1280] fp16 残基级隐状态（由 scripts/cache_esm.py 生成）
* `cache/esm_mask.npy`     [N, L] uint8      1 = 真实残基
* `cache/esm_index.csv`    row, sha1(sequence) 防止行序漂移
"""
import hashlib
import json
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


def sha256_crlf(path):
    """按 CRLF 归一化后算 sha256。

    仓库里的 manifest 是在 Windows 上生成的，记录的是 CRLF 字节的哈希；
    在 macOS/Linux 上 checkout 出来是 LF，直接对原始字节做哈希会误报不一致。
    """
    data = open(path, 'rb').read().replace(b'\r\n', b'\n')
    return hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest()


# ---------------------------------------------------------------- 主表

def load_table(path):
    """
    读指定主表，返回 DataFrame[sequence, label]，行序 = 文件行序 = 特征行序。
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
    会校验：该 fold 的 train/val/test 互斥，且并集 = 主表。
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
    """校验 folds.csv 与主表的覆盖关系，返回 'fixed' 或 'partition'。

    * ``fixed``    —— 每折的 test 完全相同（固定独立测试协议）；
    * ``partition``—— 各折 test 互不重叠且并集等于主表（随机五折协议）。

    两种协议都要求每折的 train/val/test 互斥、并集等于主表。旧版只认
    ``partition``，会把固定 test 的划分误判成"两折 test 有交集"。
    """
    f = pd.read_csv(folds_csv)
    allseq = set(table['sequence'])
    folds = sorted(f['fold'].unique())
    parts = {}
    for k in folds:
        sub = f[f['fold'] == k]
        tr = set(sub.loc[sub['split'] == 'train', 'sequence'])
        va = set(sub.loc[sub['split'] == 'val', 'sequence'])
        te = set(sub.loc[sub['split'] == 'test', 'sequence'])
        parts[k] = (tr, va, te)
        if (tr & va) or (tr & te) or (va & te):
            raise SystemExit('fold=%s 的 train/val/test 有重叠' % k)
        if (tr | va | te) != allseq:
            raise SystemExit('fold=%s 只覆盖 %d/%d 行'
                             % (k, len(tr | va | te), len(allseq)))
    tests = [parts[k][2] for k in folds]
    if all(t == tests[0] for t in tests):
        return 'fixed'
    for i, a in enumerate(tests):
        for b in tests[i + 1:]:
            if a & b:
                raise SystemExit('两折 test 有交集：%d 条' % len(a & b))
    if set().union(*tests) != allseq:
        raise SystemExit('各折 test 的并集不等于主表（差 %d 条）'
                         % len(allseq - set().union(*tests)))
    return 'partition'


# ---------------------------------------------------------------- 特征缓存

def hand_feats(table, cache_dir='cache', with_amphipath=False):
    """
    694 维手工特征（`with_amphipath=True` 时再追加 10 维两亲性特征 = 704 维）。

    第一次算完存盘，之后直接读，避免每次训练重算（也避免版本漂移）。
    两亲性特征单独存一个文件，这样加不加 `--amphipath` 的两份缓存互不覆盖。
    """
    from .features import (AMPHIPATH_DIM, amphipath_feature_names,
                           compute_amphipath, compute_handcraft,
                           handcraft_feature_names)

    fname = 'hand_feats_amphipath.npy' if with_amphipath else 'hand_feats.npy'
    path = os.path.join(cache_dir, fname)
    seqs = table['sequence'].tolist()
    dim = len(handcraft_feature_names()) + (AMPHIPATH_DIM if with_amphipath else 0)
    if with_amphipath:
        names = handcraft_feature_names() + amphipath_feature_names()
        if len(names) != dim:
            raise AssertionError('特征名 %d ≠ 维度 %d' % (len(names), dim))
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape == (len(seqs), dim):
            return arr.astype(np.float32)
        print('[warn] %s 形状不符（%s），重算' % (path, arr.shape))

    base = np.stack([compute_handcraft(s) for s in seqs], axis=0).astype(np.float32)
    if base.shape[1] != 694:
        raise SystemExit('手工特征维度 %d ≠ 694' % base.shape[1])
    if with_amphipath:
        extra = np.stack([compute_amphipath(s) for s in seqs], axis=0).astype(np.float32)
        arr = np.concatenate([base, extra], axis=1)
    else:
        arr = base
    if arr.shape[1] != dim:
        raise SystemExit('手工特征维度 %d ≠ %d' % (arr.shape[1], dim))
    if np.isnan(arr).any():
        raise SystemExit('手工特征出现 NaN')
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, arr)
    return arr


# ------------------------------------------------------------ 手工特征标准化
#
# 为什么必须做：694 维的尺度差 3.2×10⁶（paac 的 std 1.18e-4 vs mw 的 379.2）。
# 首层权重初始化在 ±1/√(1974)≈±0.0225 量级，乘以各维 std 后 694 维的贡献总和
# 只有 5.01，其中 mw 一项占 4.20（83.7%），前两维占 92.4% —— 所谓"694 维手工
# 特征"在数值上其实只有 2 维在起作用。标准化后每维贡献相当。
#
# 为什么按折、只用训练行：val/test 的统计量如果参与均值方差，就是泄漏；
# 每折各拟一份，因为只能使用本折 train 行；全局统计量会混入 val/test。

HAND_SCALER_EPS = 1e-8


def fit_hand_scaler(hand, train_idx):
    """只用训练行拟合标准化参数，返回 (mean, std) 均为 float32 [D]。"""
    h = np.asarray(hand, dtype=np.float64)[np.asarray(train_idx)]
    mean = h.mean(axis=0)
    std = h.std(axis=0)
    # 常数维（std≈0）不缩放，否则除以 0 变成 inf/NaN
    std = np.where(std < HAND_SCALER_EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_hand_scaler(hand, mean, std):
    """把同一份参数作用到全部行（train/val/test 一起）。"""
    return ((np.asarray(hand, dtype=np.float32) - mean) / std).astype(np.float32)


def save_hand_scaler(path, mean, std):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, mean=mean, std=std)
    return path


def load_hand_scaler(path):
    z = np.load(path)
    return z['mean'], z['std']


def esm_meta(cache_dir):
    """读 esm_meta.json；没有就返回 {}（旧缓存可能没有这个文件）。"""
    p = os.path.join(cache_dir, 'esm_meta.json')
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_esm_cache(cache_dir, table, prefer='auto'):
    """
    返回 (emb, mask)。
    emb 是 memmap：单层缓存为 [N, L, D]，多层缓存为 [N, K, L, D]；
    mask 恒为 [N, L] uint8。用 esm_index.csv 校验行序未变。

    prefer: 'auto'   —— 有分层缓存就用它（默认，跑多层实验省事）
            'single' —— 强制读 esm_emb_fp16.npy（复现历史基线时必须显式指定，
                        否则分层缓存一建好，所有 1 层实验都会悄悄变成 3 层）
            'layers' —— 强制读分层缓存，没有就报错
    """
    p_layers = os.path.join(cache_dir, 'esm_emb_layers_fp16.npy')
    p_single = os.path.join(cache_dir, 'esm_emb_fp16.npy')
    if prefer == 'single':
        p_emb = p_single
    elif prefer == 'layers':
        p_emb = p_layers
    elif prefer == 'auto':
        p_emb = p_layers if os.path.exists(p_layers) else p_single
    else:
        raise ValueError('prefer 只支持 auto/single/layers，收到 %r' % (prefer,))
    p_mask = os.path.join(cache_dir, 'esm_mask.npy')
    p_idx = os.path.join(cache_dir, 'esm_index.csv')
    for p in (p_emb, p_mask, p_idx):
        if not os.path.exists(p):
            raise SystemExit('缺少 ESM 缓存 %s，先运行 scripts/cache_esm.py' % p)

    # mask 语义校验：旧缓存的 1 也覆盖 <cls>/<eos>，会让池化分母多 2 并混入特殊
    # token 特征。宁可报错要求重建，也不让它在暗中污染所有指标。
    p_meta = os.path.join(cache_dir, 'esm_meta.json')
    if os.path.exists(p_meta):
        with open(p_meta, encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('mask') != 'residues_only_v1':
            raise SystemExit(
                'ESM 缓存的 mask 是旧语义（把 <cls>/<eos> 当残基），会使池化分母多 2。\n'
                '重建：python scripts/cache_esm.py --force')

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


# ---------------------------------------------------------------- 序列 one-hot

# 本数据集里恰好只有 20 种标准氨基酸（没有 X/U/O/B/Z），所以顺序即字典
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}


def seq_onehot(sequences, lmax=None, offset=0, dtype=np.float32):
    """把序列编码成 [N, L, 20] 的 one-hot，供 CfC 支路使用。

    offset 用来对齐 ESM 的 token 网格：
      * ESM 缓存里没有特殊 token（`cache_esm.py` 已把 <cls>/<eos> 切掉）→ offset=0
      * `predict.py` 的 hidden_states 第 0 位是 <cls> → offset=1

    返回 (onehot, lengths)。遇到非标准氨基酸直接报错，而不是悄悄置零 ——
    置零等于把"未知残基"和"没有残基"混为一谈。
    """
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    L = int(lmax) if lmax else int(lengths.max()) + int(offset)
    if lengths.max() + offset > L:
        raise SystemExit('序列最长 %d（+offset %d）超过 lmax=%d'
                         % (lengths.max(), offset, L))
    out = np.zeros((len(sequences), L, len(AA_ORDER)), dtype=dtype)
    bad = []
    for i, s in enumerate(sequences):
        for j, ch in enumerate(s):
            k = AA_INDEX.get(ch)
            if k is None:
                bad.append((i, ch))
                continue
            out[i, j + offset, k] = 1.0
    if bad:
        raise SystemExit('序列里有非标准氨基酸 %r（只支持 %s）'
                         % (bad[:5], AA_ORDER))
    return out, lengths
