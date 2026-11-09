#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把旧双任务 checkpoint（2.69 GB，含冻结 ESM）里的 AIP 头抽成新格式（约 85 MB）。

用途：给 5 折重训一个"不差的初始化点"，或者只是想拿旧模型的 AIP 头做对照。

    python scripts/convert_legacy_head.py \
        --ckpt D:\\BIP\\module4\\model9_2\\esm2_mode2_best.pt \
        --out heads/legacy_aip_head.pt --threshold 0.487

注意：抽出来的头来自**随机划分、单次运行**的旧模型，只作初始化/对照，
不能当成本次 5 折实验的结果。
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.checkpoint import extract_from_legacy   # noqa: E402

LEGACY = {
    'mode0': (r'D:\BIP\module4\model9_2\esm2_mode0_best.pt', None),
    'mode2': (r'D:\BIP\module4\model9_2\esm2_mode2_best.pt', 0.487),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='', help='旧 checkpoint 路径；缺省用 --which')
    ap.add_argument('--which', default='mode2', choices=sorted(LEGACY))
    ap.add_argument('--out', default='')
    ap.add_argument('--threshold', type=float, default=None)
    args = ap.parse_args()

    ckpt = args.ckpt or LEGACY[args.which][0]
    thr = args.threshold if args.threshold is not None else (LEGACY[args.which][1] or 0.5)
    out = args.out or os.path.join(ROOT, 'heads', 'legacy_%s_aip_head.pt' % args.which)

    if not os.path.exists(ckpt):
        raise SystemExit('找不到 %s' % ckpt)
    print('读取 %s（%.2f GB）...' % (ckpt, os.path.getsize(ckpt) / 1024 ** 3))
    extract_from_legacy(ckpt, out, threshold=thr,
                        note='从 %s 抽取的 AIP 头（旧随机划分模型，仅供初始化/对照）'
                             % os.path.basename(ckpt))
    print('完成。后续 5 折训练可以在此权重上初始化，也可以从零训练。')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
