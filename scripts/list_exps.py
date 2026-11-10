#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出所有实验：开了什么、跑了几个 run、主指标多少。

    python scripts/list_exps.py
    python scripts/list_exps.py --exp cv5_ema     # 看单个实验的明细

数据来源：experiments/_index.csv（run_5fold 自动登记）+ 各实验的 results/metrics.json。
若 _index.csv 缺失，会直接扫描 experiments/*/manifest.json 现算。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flat(j, key, default=None):
    """manifest 是扁平结构（write_manifest 里 man.update(extra)）；兼容早期 extra 嵌套。"""
    if key in j:
        return j[key]
    return (j.get('extra') or {}).get(key, default)


def load_index():
    import csv
    p = os.path.join(ROOT, 'experiments', '_index.csv')
    if os.path.exists(p):
        with open(p, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    rows = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'experiments'))):
        man = os.path.join(ROOT, 'experiments', name, 'manifest.json')
        if not os.path.exists(man):
            continue
        j = json.load(open(man, encoding='utf-8'))
        st = _flat(j, 'settings', {}) or {}
        rows.append({
            'exp': name, 'status': j.get('status', ''),
            'note': _flat(j, 'note', '') or '',
            'ema': int(bool(st.get('ema'))),
            'epochs': st.get('epochs', ''), 'folds': '', 'seeds': '', 'n_runs': '',
            'auroc': '', 'aupr': '', 'mcc_val': '', 'created': j.get('created', ''),
            'path': 'experiments/%s' % name,
        })
    return rows


def detail(exp):
    d = os.path.join(ROOT, 'experiments', exp)
    if not os.path.isdir(d):
        raise SystemExit('没有这个实验：%s' % exp)
    md = os.path.join(d, 'results', 'summary.md')
    if os.path.exists(md):
        print(open(md, encoding='utf-8').read())
    else:
        print('（还没有 results/summary.md）')
    dp = os.path.join(d, 'results', 'metrics_detail.csv')
    if os.path.exists(dp):
        import pandas as pd
        df = pd.read_csv(dp)
        print('\n明细（threshold_mode=free）：')
        print(df[df['threshold_mode'] == 'free']
              [['tag', 'fold', 'threshold_mode', 'auroc', 'aupr', 'mcc']]
              .to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='', help='只看某个实验的明细')
    args = ap.parse_args()

    if args.exp:
        detail(args.exp)
        return

    rows = load_index()
    if not rows:
        print('还没有任何实验。先跑： python scripts/run_5fold.py --exp cv5')
        return

    head = ('%-16s %-6s %-4s %-6s %-18s %-18s %-14s %s'
            % ('实验', 'EMA', 'ep', 'runs', 'AUROC(free)', 'AUPR(free)', 'MCC(val)', '说明'))
    print(head)
    print('-' * len(head))
    for r in rows:
        print('%-16s %-6s %-4s %-6s %-18s %-18s %-14s %s' % (
            r.get('exp', ''), '开' if r.get('ema') in ('1', 1) else '关',
            r.get('epochs', ''), r.get('n_runs', ''),
            r.get('auroc', '') or '-', r.get('aupr', '') or '-',
            r.get('mcc_val', '') or '-', r.get('note', '') or ''))
    print('\n明细：python scripts/list_exps.py --exp <名字>')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
