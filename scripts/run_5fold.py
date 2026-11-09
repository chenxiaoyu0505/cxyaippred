#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一条命令跑完 5 折：建缓存 → 逐折训练 → 汇总评估 → 登记 manifest。

产物（全部落在 experiments/<exp>/ 下，不同实验互不覆盖）
------------------------------------------------------
    experiments/<exp>/runs/fold{k}_m{seed}/   每折 8 个文件
    experiments/<exp>/results/metrics_detail.csv
    experiments/<exp>/results/metrics.json
    experiments/<exp>/results/summary.md      人读表
    experiments/<exp>/manifest.json           数据/划分/代码/环境 的 sha256

缓存策略（ESM 特征是用时生成、用完清理）
--------------------------------------
    默认：开始时若 cache/ 缺 ESM 缓存就建一份（约 0.9 GB），全部折跑完自动删除；
          --keep-cache 保留，--rebuild-cache 强制重建。
    手工特征 hand_feats.npy（12 MB、确定性）始终保留。

用法
----
    python scripts/run_5fold.py                      # exp=cv5, seed 0, 5 折
    python scripts/run_5fold.py --exp cv5_ema --ema  # EMA 消融
    python scripts/run_5fold.py --seeds 0,1,2        # 3 个 model seed
    python scripts/run_5fold.py --skip-existing      # 断了接着跑
    python scripts/run_5fold.py --keep-cache         # 保留 ESM 缓存
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.manifest import write_manifest            # noqa: E402

PY = sys.executable


def run(cmd, what):
    print('\n>>> %s' % what)
    print('    %s' % ' '.join(cmd[1:]))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit('失败（exit %d）：%s' % (rc, what))
    print('<<< 完成，用时 %.0fs' % (time.time() - t0))


def _update_index(args, folds, seeds, exp_dir):
    """把本次实验登记到 experiments/_index.csv（一行一个实验，重跑则覆盖该行）。"""
    import csv
    import json as _json

    idx_path = os.path.join(ROOT, 'experiments', '_index.csv')
    cols = ['exp', 'created', 'status', 'note', 'ema', 'ema_alpha',
            'mcc_adaptive', 'epochs', 'folds', 'seeds', 'n_runs',
            'auroc', 'aupr', 'mcc_val', 'path']

    row = {
        'exp': args.exp,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'finished',
        'note': args.note,
        'ema': int(bool(args.ema)),
        'ema_alpha': args.ema_alpha if args.ema else '',
        'mcc_adaptive': int(bool(args.mcc_adaptive)),
        'epochs': args.epochs,
        'folds': '%d' % len(folds),
        'seeds': '%d' % len(seeds),
        'n_runs': len(folds) * len(seeds),
        'auroc': '', 'aupr': '', 'mcc_val': '',
        'path': 'experiments/%s' % args.exp,
    }
    mp = os.path.join(exp_dir, 'results', 'metrics.json')
    if os.path.exists(mp):
        try:
            j = _json.load(open(mp, encoding='utf-8'))
            s = j.get('summary', {})
            for key, col, metric in (('single|free', 'auroc', 'auroc'),
                                     ('single|free', 'aupr', 'aupr'),
                                     ('single|@val', 'mcc_val', 'mcc')):
                cell = s.get(key, {}).get(metric)
                if cell:
                    row[col] = '%.4f±%.4f' % (cell['mean'], cell['sd'])
        except Exception as e:  # noqa: BLE001
            print('[警告] 读取 metrics.json 失败，总表里该实验的指标留空：%s' % e)

    raw = []
    if os.path.exists(idx_path):
        with open(idx_path, encoding='utf-8-sig', newline='') as f:
            raw = list(csv.DictReader(f))
    if not row['note']:
        # 重跑/续跑时没带 --note，就保留原来那句说明
        prev = [r for r in raw if r.get('exp') == args.exp]
        if prev:
            row['note'] = prev[0].get('note', '')
    old = [r for r in raw if r.get('exp') != args.exp]
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    with open(idx_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in old:
            w.writerow({c: r.get(c, '') for c in cols})
        w.writerow(row)
    print('\n[总表] 已登记 experiments/_index.csv（共 %d 个实验）' % (len(old) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    help='实验名（必填，决定产物目录 experiments/<exp>/）。'
                         '建议：cv5 / cv5_full / cv5_ema / cv5_mcc / legacy；'
                         '只能用小写字母、数字、下划线和短横线')
    ap.add_argument('--note', default='', help='一句话说明这次为什么跑，写进 manifest 与总表')
    ap.add_argument('--resume', action='store_true',
                    help='实验目录已存在时继续（跳过已完成的 run）')
    ap.add_argument('--split-id', default='split_seed42_n5')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default='0', help='逗号分隔，例如 0,1,2')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--pos-weight', type=float, default=0.0)
    ap.add_argument('--label-noise', type=float, default=0.0, help='标签噪声注入比例（只翻训练行）')
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--ema-alpha', type=float, default=0.7)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--ema-warmup', type=int, default=3)
    ap.add_argument('--mcc-adaptive', action='store_true', help='启用 MCC 类权重自适应')
    ap.add_argument('--mcc-interval', type=int, default=3)
    ap.add_argument('--mcc-rate', type=float, default=0.15)
    ap.add_argument('--mcc-ratio', type=float, default=1.2)
    ap.add_argument('--mcc-cooldown', type=int, default=3)
    ap.add_argument('--class-weight-min', type=float, default=0.3)
    ap.add_argument('--class-weight-max', type=float, default=3.0)
    ap.add_argument('--device', default='')
    ap.add_argument('--cache-dir', default='', help='ESM 缓存目录，默认 <root>/cache；'
                                                   '调试时可指向 _work\\fakecache')
    ap.add_argument('--keep-cache', action='store_true', help='跑完保留 ESM 缓存')
    ap.add_argument('--rebuild-cache', action='store_true', help='强制重建 ESM 缓存')
    ap.add_argument('--skip-existing', action='store_true',
                    help='（同 --resume）已有 test_probs.csv 的 run 跳过')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()

    # ---- 实验名校验与重复保护 ----
    import re as _re
    exp = args.exp.strip()
    if not _re.fullmatch(r'[a-z0-9][a-z0-9_\-]{1,40}', exp):
        raise SystemExit('实验名不合法：%r\n规则：小写字母/数字开头，只能含 a-z 0-9 _ -，2–41 个字符。'
                         '\n例：--exp cv5   /   --exp cv5_full   /   --exp cv5_ema' % args.exp)
    args.exp = exp
    exp_dir_probe = os.path.join(ROOT, 'experiments', exp)
    runs_probe = os.path.join(exp_dir_probe, 'runs')
    existing_runs = ([d for d in os.listdir(runs_probe)
                      if os.path.isdir(os.path.join(runs_probe, d))]
                     if os.path.isdir(runs_probe) else [])
    if existing_runs and not (args.resume or args.skip_existing):
        raise SystemExit(
            '实验 %r 已存在（%d 个 run）。\n'
            '要么换个名字（例如 --exp %s_v2），要么加 --resume 继续跑完剩下的。'
            % (exp, len(existing_runs), exp))
    if args.resume:
        args.skip_existing = True

    folds = [int(x) for x in args.folds.split(',') if x.strip() != '']
    seeds = [int(x) for x in args.seeds.split(',') if x.strip() != '']
    exp_dir = os.path.join(ROOT, 'experiments', args.exp)
    runs_dir = os.path.join(exp_dir, 'runs')
    split_id = args.split_id
    split_csv = os.path.join(ROOT, 'data', 'splits', split_id, 'folds.csv')
    data_csv = os.path.join(ROOT, 'data', 'aip_all.csv')

    print('=' * 70)
    print('实验 %s | 折 %s | model seed %s | 共 %d 次训练'
          % (args.exp, folds, seeds, len(folds) * len(seeds)))
    print('EMA-SAT   : %s%s' % ('开' if args.ema else '关',
                                ' (alpha=%.2f, decay=%.2f, warmup=%d)'
                                % (args.ema_alpha, args.ema_decay, args.ema_warmup)
                                if args.ema else ''))
    print('MCC 自适应: %s%s' % ('开' if args.mcc_adaptive else '关',
                                ' (interval=%d, rate=%.2f, ratio=%.2f, cooldown=%d, bounds=[%.2f, %.2f])'
                                % (args.mcc_interval, args.mcc_rate, args.mcc_ratio,
                                   args.mcc_cooldown, args.class_weight_min, args.class_weight_max)
                                if args.mcc_adaptive else ''))
    print('超参      : epochs=%d batch=%d lr=%g patience=%d pos_weight=%g'
          % (args.epochs, args.batch_size, args.lr, args.patience, args.pos_weight))
    if args.note:
        print('说明      : %s' % args.note)
    print('解释器 %s' % PY)
    print('=' * 70)

    # ---- 1. 划分 ----
    if not os.path.exists(split_csv):
        run([PY, os.path.join(HERE, 'make_5fold.py'), '--split-id', split_id],
            '生成划分 %s' % split_id)
    else:
        print('\n[1/5] 划分已存在：%s' % os.path.relpath(split_csv, ROOT))

    # ---- 2. 缓存 ----
    from src.cache_util import clean_cache, has_cache   # noqa: E402
    cache_dir = args.cache_dir or os.path.join(ROOT, 'cache')
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(ROOT, cache_dir)
    need_build = args.rebuild_cache or not has_cache(cache_dir)
    if need_build:
        cmd = [PY, os.path.join(HERE, 'cache_esm.py'), '--cache-dir', cache_dir]
        if args.rebuild_cache:
            cmd.append('--force')
        run(cmd, '生成 ESM 缓存（约 0.9 GB，5–10 分钟）')
    else:
        print('\n[2/5] ESM 缓存已存在，直接使用')

    # ---- 3. 训练 ----
    t_all = time.time()
    done, skipped = [], []
    for seed in seeds:
        for k in folds:
            run_dir = os.path.join(runs_dir, 'fold%d_m%d' % (k, seed))
            if args.skip_existing and os.path.exists(os.path.join(run_dir, 'test_probs.csv')):
                print('\n[skip] fold%d seed%d 已完成' % (k, seed))
                skipped.append(run_dir)
                continue
            cmd = [PY, os.path.join(HERE, 'train_aip.py'),
                   '--fold', str(k), '--seed', str(seed),
                   '--exp', args.exp, '--split-id', split_id,
                   '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
                   '--lr', str(args.lr), '--patience', str(args.patience),
                   '--pos-weight', str(args.pos_weight),
                '--label-noise', str(args.label_noise),
                   '--ema-alpha', str(args.ema_alpha),
                   '--ema-decay', str(args.ema_decay),
                   '--ema-warmup', str(args.ema_warmup),
                   '--cache-dir', cache_dir,
                   '--mcc-interval', str(args.mcc_interval),
                   '--mcc-rate', str(args.mcc_rate),
                   '--mcc-ratio', str(args.mcc_ratio),
                   '--mcc-cooldown', str(args.mcc_cooldown),
                   '--class-weight-min', str(args.class_weight_min),
                   '--class-weight-max', str(args.class_weight_max)]
            if args.ema:
                cmd.append('--ema')
            if args.mcc_adaptive:
                cmd.append('--mcc-adaptive')
            if args.device:
                cmd += ['--device', args.device]
            run(cmd, '训练 fold%d / seed%d' % (k, seed))
            done.append(run_dir)

    # ---- 4. 完整性检查 ----
    missing = [d for d in
               (os.path.join(runs_dir, 'fold%d_m%d' % (k, s)) for s in seeds for k in folds)
               if not os.path.exists(os.path.join(d, 'test_probs.csv'))]
    if missing:
        raise SystemExit('有 %d 个 run 缺 test_probs.csv：%s'
                         % (len(missing), [os.path.basename(m) for m in missing]))
    print('\n[4/5] 全部 %d 个 run 齐全' % (len(folds) * len(seeds)))

    # ---- 5. 评估 ----
    if not args.skip_eval:
        run([PY, os.path.join(HERE, 'eval_aip.py'), '--exp', args.exp], '汇总评估')

    # ---- manifest ----
    cfgs = [os.path.join(runs_dir, os.path.basename(d), 'config.json') for d in done]
    write_manifest(
        exp_dir, exp=args.exp, data_csv=data_csv, split_csv=split_csv,
        root=ROOT, status='finished',
        extra={
            'kind': 'train_5fold',
            'split_id': split_id,
            'folds': folds, 'model_seeds': seeds,
            'n_runs': len(folds) * len(seeds),
            'skipped_runs': len(skipped),
            'settings': {'epochs': args.epochs, 'batch_size': args.batch_size,
                         'lr': args.lr, 'patience': args.patience,
                         'pos_weight': args.pos_weight, 'label_noise': args.label_noise, 'ema': args.ema,
                         'ema_alpha': args.ema_alpha, 'ema_decay': args.ema_decay,
                         'ema_warmup': args.ema_warmup,
                         'mcc_adaptive': args.mcc_adaptive,
                         'mcc_interval': args.mcc_interval, 'mcc_rate': args.mcc_rate,
                         'mcc_ratio': args.mcc_ratio, 'mcc_cooldown': args.mcc_cooldown,
                         'class_weight_min': args.class_weight_min,
                         'class_weight_max': args.class_weight_max},
            'note': args.note,
            'configs': [os.path.relpath(c, ROOT).replace('\\', '/') for c in cfgs],
            'seconds': round(time.time() - t_all, 1),
        })

    # ---- 全局实验总表 ----
    _update_index(args, folds, seeds, exp_dir)

    # ---- 清理 ----
    if not args.keep_cache:
        removed = clean_cache(cache_dir)
        if removed:
            print('\n[清理] 已删除 ESM 缓存 %.1f MB（hand_feats.npy 保留）'
                  % (sum(s for _, s in removed) / 1e6))
    else:
        print('\n[保留] ESM 缓存按 --keep-cache 保留')

    print('\n实验目录：%s' % os.path.relpath(exp_dir, ROOT))
    print('  runs\\       %d 个 run' % (len(folds) * len(seeds)))
    print('  results\\    metrics_detail.csv / metrics.json / summary.md')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
