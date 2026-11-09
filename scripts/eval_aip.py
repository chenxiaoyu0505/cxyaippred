#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估：从落盘概率算指标。**唯一算指标的地方**。

三个口径同时报（回应 R2-4 与旧稿的阈值问题）
--------------------------------------------
    @0.5     固定阈值，不受任何调参影响，最保守
    @val     每折在验证集上搜出的阈值（旧稿用的就是这个，必须公开）
    free     AUROC / AUPR，与阈值无关，**主指标**

汇总方式
--------
    单折     每个 fold 的模型评自己那一折的 test → 5 个独立估计，报 mean ± sd。
             这是 5 折主结果；不再计算 "5 个 fold 模型集成"。
             （同一批数据里每条 test 都在另外 4 个模型的训练集中，集成评估
               必然泄漏；要估集成只能用 external test 或同 fold 多 seed。）

用法
----
    python scripts/eval_aip.py --exp cv5            # 读 experiments/cv5/runs，写 experiments/cv5/results
    python scripts/eval_aip.py --exp cv5 --n-boot 1000
    python scripts/eval_aip.py --runs <dir> --out <dir>   # 自定义路径（一般不用）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.metrics import bootstrap_ci, full_metrics, summarize   # noqa: E402
from src.utils import ensure_dir, load_json, save_json          # noqa: E402

KEYS = ('auroc', 'aupr', 'mcc', 'f1', 'acc', 'precision', 'recall', 'specificity')


def read_run(d):
    need = [os.path.join(d, f) for f in ('config.json', 'test_probs.csv', 'all_probs.csv')]
    for p in need:
        if not os.path.exists(p):
            return None
    cfg = load_json(os.path.join(d, 'config.json'))
    test = pd.read_csv(os.path.join(d, 'test_probs.csv'))
    allp = pd.read_csv(os.path.join(d, 'all_probs.csv'))
    thr = 0.5
    p_thr = os.path.join(d, 'threshold.json')
    if os.path.exists(p_thr):
        thr = float(load_json(p_thr).get('threshold', 0.5))
    return {
        'dir': d,
        'fold': int(cfg.get('fold', -1)),
        'seed': int(cfg.get('train', {}).get('seed', 0)),
        'split_id': cfg.get('data', {}).get('split_id', ''),
        'test': test,
        'all': allp,
        'threshold': thr,
    }


def summary_markdown(df, summary, args, runs):
    """人读版结果表，可直接贴进论文/补充材料"""
    n_runs = len(runs)
    lines = ['# 结果汇总', '',
             '- 实验：`%s`' % (args.exp or '(未命名)'),
             '- 划分：`%s`' % runs[0]['split_id'],
             '- run 数：%d（fold %s）' % (n_runs, ', '.join(str(r['fold']) for r in runs)),
             '- 主指标：AUROC / AUPR（阈值无关）；MCC / F1 另报 `@0.5` 与 `@val` 两种口径', '',
             '## 汇总（折间 mean ± sd，sd 只由各折均值计算）', '',
             '| 口径 | n | AUROC | AUPR | MCC | F1 |',
             '|---|---|---|---|---|---|']
    for key, s in summary.items():
        g = lambda k: ('%.4f ± %.4f' % (s[k]['mean'], s[k]['sd'])) if k in s else '—'  # noqa: E731
        lines.append('| %s | %d | %s | %s | %s | %s |'
                     % (key, s.get('auroc', {}).get('n', 0),
                        g('auroc'), g('aupr'), g('mcc'), g('f1')))

    single = df[(df['tag'] == 'single') & (df['threshold_mode'] == 'free')]
    if len(single) > 1:
        lines += ['', '## 单折明细（free 口径）', '',
                  '| fold | n | n_pos | AUROC | AUPR |', '|---|---|---|---|---|']
        for _, r in single.sort_values('fold').iterrows():
            lines.append('| %d | %d | %d | %.4f | %.4f |'
                         % (int(r['fold']), int(r['n']), int(r['n_pos']),
                            float(r['auroc']), float(r['aupr'])))

    lines += ['', '## 口径说明', '',
              '- `free`：只用概率，不设阈值（主指标）。',
              '- `@0.5`：固定阈值 0.5，不受任何调参影响。',
              '- `@val`：每折在验证集上网格搜索 MCC 最大点得到阈值（旧稿口径，须与 `free` 并列报告）。']
    return '\n'.join(lines) + '\n'


def rows_for(run, prob, y, tag, n_boot, seed=0):
    out = []
    for name, t in (('free', None), ('@0.5', 0.5), ('@val', run['threshold'])):
        if t is None:
            m = full_metrics(y, prob, threshold=0.5)
            m['mcc'] = float('nan')          # free 口径不报 MCC
            ci = {k: bootstrap_ci(y, prob, k, n_boot=n_boot, seed=seed)
                  for k in ('auroc', 'aupr')}
        else:
            m = full_metrics(y, prob, threshold=t)
            ci = {k: bootstrap_ci(y, prob, k, threshold=t, n_boot=n_boot, seed=seed)
                  for k in ('mcc', 'f1')}
        out.append({
            'tag': tag, 'fold': run['fold'], 'model_seed': run['seed'],
            'threshold_mode': name, 'threshold': t,
            'n': m['n'], 'n_pos': m['n_pos'], 'pred_pos': m['pred_pos'],
            **{k: m[k] for k in KEYS if k in m},
            'ci95': {k: [round(v['lo'], 4), round(v['hi'], 4)] for k, v in ci.items()},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='', help='实验名：读 experiments/<exp>/runs，写 experiments/<exp>/results')
    ap.add_argument('--runs', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--no-ensemble', action='store_true',
                    help='兼容旧命令；当前始终只报单折，不计算集成')
    args = ap.parse_args()

    if args.exp:
        args.runs = args.runs or os.path.join(ROOT, 'experiments', args.exp, 'runs')
        args.out = args.out or os.path.join(ROOT, 'experiments', args.exp, 'results')
    else:
        args.runs = args.runs or os.path.join(ROOT, 'runs')
        args.out = args.out or os.path.join(ROOT, 'results')

    dirs = sorted(glob.glob(os.path.join(args.runs, 'fold*')))
    runs = [r for r in (read_run(d) for d in dirs) if r is not None]
    if not runs:
        raise SystemExit('%s 下没有可用的 run（需要 config.json + test_probs.csv + all_probs.csv）'
                         % args.runs)

    print('找到 %d 个 run：%s' % (len(runs), ', '.join(os.path.basename(r['dir']) for r in runs)))
    out_dir = ensure_dir(args.out)
    all_rows = []

    # ---- 单折 ----
    for r in runs:
        y = r['test']['label'].values.astype(int)
        p = r['test']['prob'].values.astype(float)
        all_rows += rows_for(r, p, y, 'single', args.n_boot, seed=r['fold'] * 100 + r['seed'])

    # ---- 落盘 ----
    # 只报单折。5 个 fold 模型在彼此 test 上的集成无法无偏估计（见文件头说明）。
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, 'metrics_detail.csv'), index=False)

    summary = {}
    for tag in df['tag'].unique():
        for mode in df['threshold_mode'].unique():
            sub = df[(df.tag == tag) & (df.threshold_mode == mode)]
            if sub.empty:
                continue
            summary['%s|%s' % (tag, mode)] = summarize(sub.to_dict('records'))

    single = df[(df.tag == 'single') & (df.threshold_mode == 'free')]
    saved = {
        'exp': args.exp,
        'split_id': runs[0]['split_id'],
        'n_runs': len(runs),
        'folds': sorted(int(f) for f in df['fold'].unique() if f >= 0),
        'summary': summary,
        'note': 'free = 阈值无关（主指标）；@0.5 = 固定阈值；@val = 验证集搜出阈值（旧稿口径）',
    }
    save_json(saved, os.path.join(out_dir, 'metrics.json'))

    # ---- 人读版 ----
    md = summary_markdown(df, summary, args, runs)
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(md)

    # ---- 打印 ----
    print('\n%-30s %-7s %-7s %-7s' % ('口径', 'AUROC', 'AUPR', 'MCC'))
    for key, s in summary.items():
        print('%-30s %-7.4f %-7.4f %-7s'
              % (key,
                 s.get('auroc', {}).get('mean', float('nan')),
                 s.get('aupr', {}).get('mean', float('nan')),
                 ('%.4f' % s['mcc']['mean']) if 'mcc' in s else '  -'))
    if len(single) > 1:
        print('\nsingle/free 的折间 sd：AUROC %.4f, AUPR %.4f'
              % (summary['single|free']['auroc']['sd'], summary['single|free']['aupr']['sd']))
    print('\n明细：%s' % os.path.join(out_dir, 'metrics_detail.csv'))
    print('汇总：%s' % os.path.join(out_dir, 'metrics.json'))
    print('人读：%s' % os.path.join(out_dir, 'summary.md'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
