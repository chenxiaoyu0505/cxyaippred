#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 折集成推理：共享一次 ESM-2 前向，分别过每个 fold 的 AIPHead，输出概率与折间标准差。

用法
----
    # 输入 CSV（必须有 sequence 列）或 FASTA
    python scripts/predict.py --input candidates.csv --out preds.csv
    python scripts/predict.py --input cand.fasta --weights runs/fold*_seed42/best.pt

输出列
------
    sequence, prob_mean, prob_std, prob_fold0, prob_fold1, ..., n_folds

    prob_mean : 各 fold 的平均概率（推荐作为最终打分）
    prob_std  : 折间标准差；越大说明模型分歧越大，通常意味着该序列接近训练分布边界

注意
----
* 不输出 0/1，除非显式给 --threshold；概率本身在不同应用场景下应配不同阈值。
* ESM 版本必须与训练时一致（meta['esm_name']），否则权重不兼容。
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.checkpoint import load_head            # noqa: E402
from src.features import compute_handcraft      # noqa: E402
from src.utils import pick_device               # noqa: E402

MAX_LEN = 1024


def read_sequences(path):
    low = path.lower()
    if low.endswith(('.fa', '.fasta', '.faa')):
        seqs = []
        cur = []
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
        return [s.upper() for s in seqs]

    df = pd.read_csv(path)
    col = 'sequence' if 'sequence' in df.columns else ('squence' if 'squence' in df.columns else None)
    if col is None:
        raise SystemExit('CSV 里找不到 sequence 列：%s' % list(df.columns))
    return df[col].astype(str).str.upper().str.strip().tolist()


def collect_weights(pattern, weights_arg):
    if weights_arg:
        paths = []
        for p in weights_arg.split(','):
            p = p.strip()
            if os.path.isdir(p):
                paths += sorted(glob.glob(os.path.join(p, '*.pt')))
            else:
                paths += sorted(glob.glob(p))
        return paths
    return sorted(glob.glob(os.path.join(pattern, 'fold*', '*.pt')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV（含 sequence 列）或 FASTA')
    ap.add_argument('--out', default='predictions.csv')
    ap.add_argument('--weights', default='', help='逗号分隔的 .pt / 目录 / glob；默认 runs/fold*/*.pt')
    ap.add_argument('--runs-dir', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='auto',
                    help='auto/cuda/mps/cpu，默认自动选择')
    ap.add_argument('--threshold', type=float, default=None, help='给了就额外输出 pred 列')
    ap.add_argument('--esm-name', default='facebook/esm2_t33_650M_UR50D')
    args = ap.parse_args()
    args.device = pick_device(args.device)

    paths = collect_weights(os.path.join(args.runs_dir, '*'), args.weights)
    if not paths:
        raise SystemExit('没找到权重文件。先用 train 脚本产出 runs/fold*/best.pt，或用 --weights 指定。')
    print('权重文件 %d 个：' % len(paths))
    for p in paths:
        print('  ', os.path.relpath(p, ROOT))

    heads, metas = [], []
    for p in paths:
        h, m = load_head(p)
        heads.append(h.to(args.device).eval())
        metas.append(m)
    esm_names = {m.get('esm_name') for m in metas}
    if len(esm_names) > 1:
        raise SystemExit('权重之间的 ESM 版本不一致：%s' % esm_names)
    esm_name = esm_names.pop() if esm_names else args.esm_name
    hand_dims = {m['hand_dim'] for m in metas}
    if len(hand_dims) > 1:
        raise SystemExit('权重之间的 hand_dim 不一致：%s' % hand_dims)
    hand_dim = hand_dims.pop()
    print('ESM: %s | hand_dim=%d | 每个头 %.1f MB'
          % (esm_name, hand_dim, heads[0].size_mb()))

    seqs = read_sequences(args.input)
    print('待预测序列：%d 条' % len(seqs))
    if not seqs:
        raise SystemExit('没有读入任何序列')

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(esm_name)
    esm = AutoModel.from_pretrained(esm_name).to(args.device).eval()
    for p in esm.parameters():
        p.requires_grad = False

    probs = np.zeros((len(seqs), len(heads)), dtype=np.float64)

    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            chunk = seqs[i:i + args.batch_size]
            enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                      max_length=MAX_LEN)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            h = esm(**enc, output_hidden_states=True).hidden_states[-1]
            hand = torch.tensor(
                np.stack([compute_handcraft(s) for s in chunk], axis=0),
                dtype=torch.float32,
            ).to(args.device)
            if hand.shape[1] != hand_dim:
                raise SystemExit('手工特征维度 %d 与权重要求的 %d 不一致'
                                 % (hand.shape[1], hand_dim))
            for j, head in enumerate(heads):
                probs[i:i + len(chunk), j] = torch.sigmoid(
                    head(h, enc['attention_mask'], hand)).cpu().numpy()
            print('  %d/%d' % (min(i + args.batch_size, len(seqs)), len(seqs)), end='\r')

    out = pd.DataFrame({'sequence': seqs, 'prob_mean': probs.mean(axis=1),
                        'prob_std': probs.std(axis=1), 'n_folds': probs.shape[1]})
    for j in range(probs.shape[1]):
        name = metas[j].get('fold')
        out['prob_fold%s' % ('%d' % name if name is not None else j)] = probs[:, j]
    if args.threshold is not None:
        out['pred'] = (out['prob_mean'] >= args.threshold).astype(int)

    out.to_csv(args.out, index=False)
    print('\n已写出 %s（%d 行）' % (args.out, len(out)))
    print('概率分布: mean %.3f | std %.3f | 最小 %.3f | 最大 %.3f'
          % (out['prob_mean'].mean(), out['prob_mean'].std(),
             out['prob_mean'].min(), out['prob_mean'].max()))
    print('折间分歧: 平均 %.3f，最大的 5 条：' % out['prob_std'].mean())
    for _, r in out.nlargest(5, 'prob_std').iterrows():
        print('   %.3f ± %.3f  %s' % (r['prob_mean'], r['prob_std'], r['sequence']))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
