#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把已经跑完的实验补登记到 experiments/_index.csv（不重新训练）。

    python scripts/register_exp.py --exp cv5_mac
    python scripts/register_exp.py --exp cv5_mac --note "mac 复现，MPS"

背景
----
    experiments/_index.csv 是被 git 跟踪的总表。run_5fold.py 默认**不再**自动写它
    （否则每跑一次实验工作区就脏一次，远端也跑过实验时 git pull 会报
    "Your local changes ... would be overwritten by merge"）。
    需要发布结果时用本脚本补登记；它只读 <exp>/manifest.json 与
    <exp>/results/metrics.json，不碰训练产物。

登记规则与 run_5fold 的 --register 完全一致：同一个 exp 覆盖原行，其余行保持不动。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from run_5fold import _update_index                     # noqa: E402


def _flat(j, key, default=None):
    """manifest 由 write_manifest 写成扁平结构；兼容早期嵌在 extra 里的写法。"""
    if key in j:
        return j[key]
    return (j.get('extra') or {}).get(key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiments/ 下的实验名')
    ap.add_argument('--note', default='', help='覆盖 manifest 里的说明（默认沿用）')
    args = ap.parse_args()

    exp_dir = os.path.join(ROOT, 'experiments', args.exp)
    man_path = os.path.join(exp_dir, 'manifest.json')
    if not os.path.exists(man_path):
        raise SystemExit('没有 %s。先跑： python scripts/run_5fold.py --exp %s'
                         % (man_path, args.exp))

    j = json.load(open(man_path, encoding='utf-8'))
    st = _flat(j, 'settings', {}) or {}
    folds = _flat(j, 'folds', []) or []
    seeds = _flat(j, 'model_seeds', []) or []
    if not folds:
        # 老 manifest 可能没记 folds，至少别把 runs 列写错
        n = _flat(j, 'n_runs', 0) or 0
        folds = list(range(5)) if n and n % max(len(seeds), 1) == 0 else []
        if folds:
            seeds = seeds or [0]

    if not os.path.exists(os.path.join(exp_dir, 'results', 'metrics.json')):
        print('[警告] 没有 results/metrics.json，总表里该实验的指标会留空')

    ns = argparse.Namespace(
        exp=args.exp,
        note=args.note or _flat(j, 'note', '') or '',
        ema=bool(st.get('ema')),
        ema_alpha_min=st.get('ema_alpha_min', 0.7),
        ema_decay=st.get('ema_decay', 0.65),
        ema_start_epoch=st.get('ema_start_epoch', 2),
        ema_tau=st.get('ema_tau', 1.0),
        epochs=st.get('epochs', ''),
    )
    _update_index(ns, folds, seeds, exp_dir)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
