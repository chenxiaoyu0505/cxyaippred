# -*- coding: utf-8 -*-
"""Remove the old row-level BIPs from the single-task AIP dataset.

The 56 BIPs are the peptides with both ``label-aip == 1`` and
``label-pip == 1`` in the same source row (46 train + 10 test in the old
dual-task manuscript).  They are a deliberate overlap between the two tasks,
so a pure AIP benchmark must exclude them.

This script is intentionally one-way and non-destructive: it writes a new
CSV and never edits ``data/aip_all.csv``.

The canonical ``data/aip_all_nobip.csv`` is now 4,634 rows: this script's
4,630-row output plus four HD-AIP-only positives restored by the user.
Re-running with ``--force`` would drop those four; use only for the
historical 4,630 split.

Usage (from the repository root):
    python scripts/drop_bips.py
    python scripts/drop_bips.py --out data/aip_all_nobip.csv --force
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(ROOT, 'data', 'aip_all.csv')
DEFAULT_BIPS = r'D:\BIP\data\BIPs.csv'
DEFAULT_OUT = os.path.join(ROOT, 'data', 'aip_all_nobip.csv')
EXPECTED_BIPS = 56


def norm(series):
    return series.astype(str).str.upper().str.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--bips', default=DEFAULT_BIPS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    bips = pd.read_csv(args.bips)

    for col in ('sequence', 'label'):
        if col not in data.columns:
            raise SystemExit('AIP 数据缺少列 %s：%s' % (col, list(data.columns)))
    if 'squence' not in bips.columns:
        raise SystemExit('BIP 文件缺少 squence 列：%s' % list(bips.columns))

    data['sequence'] = norm(data['sequence'])
    data['label'] = data['label'].astype(int)
    bip_seqs = set(norm(bips['squence']))

    if len(data) != data['sequence'].nunique():
        raise SystemExit('AIP 数据存在重复序列，请先处理')
    if len(bip_seqs) != len(bips):
        raise SystemExit('BIP 文件存在重复序列')
    if len(bip_seqs) != EXPECTED_BIPS:
        raise SystemExit('BIP 数量不是 %d，而是 %d；请确认口径'
                         % (EXPECTED_BIPS, len(bip_seqs)))

    present = bip_seqs & set(data['sequence'])
    missing = bip_seqs - present
    if missing:
        raise SystemExit('有 %d 条 BIP 不在 AIP 数据中，例：%s'
                         % (len(missing), sorted(missing)[:3]))

    out = data[~data['sequence'].isin(bip_seqs)].copy()
    n_drop = len(data) - len(out)
    dropped_pos = int(data.loc[data['sequence'].isin(bip_seqs), 'label'].sum())
    if n_drop != EXPECTED_BIPS:
        raise SystemExit('实际删除 %d 行，预期 %d 行' % (n_drop, EXPECTED_BIPS))
    if len(out) != out['sequence'].nunique():
        raise SystemExit('删除后仍有重复序列')

    if os.path.exists(args.out) and not args.force:
        raise SystemExit('输出已存在：%s（加 --force 覆盖）' % args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    meta = {
        'source': os.path.abspath(args.data),
        'source_sha256': sha256(args.data),
        'bips': os.path.abspath(args.bips),
        'bips_sha256': sha256(args.bips),
        'n_bips': len(bip_seqs),
        'n_before': int(len(data)),
        'n_after': int(len(out)),
        'pos_before': int(data['label'].sum()),
        'pos_after': int(out['label'].sum()),
        'neg_before': int((data['label'] == 0).sum()),
        'neg_after': int((out['label'] == 0).sum()),
        'dropped_pos': dropped_pos,
        'dropped_neg': int(n_drop - dropped_pos),
        'positive_rate_after': round(float(out['label'].mean()), 6),
    }
    meta_path = os.path.splitext(args.out)[0] + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('BIP 删除完成：%d -> %d 行（删除 %d：阳 %d / 阴 %d）'
          % (len(data), len(out), n_drop, dropped_pos, n_drop - dropped_pos))
    print('删除后：阳性 %d / 阴性 %d（%.2f%%）'
          % (meta['pos_after'], meta['neg_after'], 100 * meta['positive_rate_after']))
    print('已写入：%s' % args.out)
    print('已写入：%s' % meta_path)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
