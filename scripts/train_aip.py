#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单折训练：训练 AIPHead，**只产出概率，不产出指标**。

为什么不算指标
--------------
指标一律由 `scripts/eval_aip.py` 从落盘的 `test_probs.csv` 重算。
这样"同一个指标两个数"在物理上不可能发生（旧稿 Fig 3 vs Table 2 的教训）。

产物（runs/<run_name>/）
-----------------------
    config.json        全部超参快照
    split.csv          本次用到的 train/val/test 行（自带证据）
    best.pt            80 MB 头
    val_probs.csv      sequence,label,prob      ← 选阈值用
    test_probs.csv     sequence,label,prob      ← 评估用
    all_probs.csv      全部 4,686 行的概率       ← 供误差分析 / 外部评估
    train_log.csv      每 epoch 的 loss / val AUROC / AUPR / EMA 诊断
    threshold.json     验证集搜出的阈值

用法
----
    python scripts/train_aip.py --fold 0 --seed 0
    python scripts/train_aip.py --fold 0 --seed 0 --ema --pos-weight 3.0
    python scripts/train_aip.py --fold 2 --seed 1 --epochs 60 --out runs
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.checkpoint import save_head                       # noqa: E402
from src.config import RunCfg                              # noqa: E402
from src.data_io import (apply_hand_scaler, esm_meta, fit_hand_scaler,  # noqa: E402
                         hand_feats, iter_batches, load_esm_cache,
                         load_folds, load_table, save_hand_scaler, seq_onehot)
from src.ema_sat import EMASAT                             # noqa: E402
from src.hand_select import save_hand_selector, select_hand_features  # noqa: E402
from src.manifest import sha256_file                       # noqa: E402
from src.metrics import threshold_free                       # noqa: E402
from src.model_aip import AIPHead                          # noqa: E402
from src.threshold import search_best_mcc                  # noqa: E402
from src.utils import ensure_dir, pick_device, save_json, seed_everything  # noqa: E402


# ---------------------------------------------------------------- 预测

@torch.no_grad()
def predict_ordered(head, idx, emb, mask, hand, device, batch_size=64, seq=None):
    """按 idx 的原始顺序返回概率（不重排）"""
    head.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        h = torch.from_numpy(np.asarray(emb[b], dtype=np.float32)).to(device)
        m = torch.from_numpy(np.asarray(mask[b])).to(device).float()
        hf = torch.from_numpy(hand[b]).to(device)
        sq = None
        if seq is not None:
            sq = torch.from_numpy(np.asarray(seq[b], dtype=np.float32)).to(device)
        out[s:s + len(b)] = torch.sigmoid(head(h, m, hf, sq)).cpu().numpy()
    return out


def predict_components(head, idx, emb, mask, hand, device, batch_size=64, seq=None):
    """分别返回主分支 A 与原型分支 B 的 sigmoid 概率（仅原型开启时有意义）。"""
    head.eval()
    pa = np.zeros(len(idx), dtype=np.float64)
    pb = np.zeros(len(idx), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            h = torch.from_numpy(np.asarray(emb[b], dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(mask[b])).to(device).float()
            hf = torch.from_numpy(hand[b]).to(device)
            sq = None
            if seq is not None:
                sq = torch.from_numpy(np.asarray(seq[b], dtype=np.float32)).to(device)
            _, aux = head(h, m, hf, sq, return_proto=True)
            pa[s:s + len(b)] = torch.sigmoid(aux['A']).cpu().numpy()
            pb[s:s + len(b)] = torch.sigmoid(aux['B']).cpu().numpy()
    return pa, pb


# ---------------------------------------------------------------- 原型初始化

def init_prototypes(head, tr_idx, emb, mask, hand, y, cfg, device, seq=None):
    """用本折训练行的拼接特征做类内 k-means，给原型一个贴近数据的起点。

    只在本折训练行上聚类（不看 val/test），代价是一次额外前向。
    """
    if not head.proto_enabled or cfg.model.proto_init != 'kmeans':
        return
    from sklearn.cluster import KMeans

    bs = max(cfg.train.batch_size, 64)
    head.eval()
    feats, labels = [], []
    with torch.no_grad():
        for s in range(0, len(tr_idx), bs):
            b = tr_idx[s:s + bs]
            h = torch.from_numpy(np.asarray(emb[b], dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(mask[b])).to(device).float()
            hf = torch.from_numpy(hand[b]).to(device)
            sq = None
            if seq is not None:
                sq = torch.from_numpy(np.asarray(seq[b], dtype=np.float32)).to(device)
            feats.append(head.encode(h, m, hf, sq).cpu().numpy())
            labels.append(np.asarray(y[b]))
    feats = np.concatenate(feats).astype(np.float64)
    labels = np.concatenate(labels)

    # 在**投影后的 128 维空间**聚类：原型就活在这里，直接拿 1984 维中心会形状对不上
    with torch.no_grad():
        z = []
        for s in range(0, len(feats), 512):
            ft = torch.from_numpy(feats[s:s + 512].astype(np.float32)).to(device)
            z.append(head.proto.proj(ft).cpu().numpy())
        z = np.concatenate(z).astype(np.float64)

    with torch.no_grad():
        for P, cls in ((head.proto.p_pos, 1), (head.proto.p_neg, 0)):
            X = z[labels > 0.5] if cls else z[labels <= 0.5]
            k = int(P.shape[0])
            if len(X) < k:
                continue
            km = KMeans(n_clusters=k, n_init=10,
                        random_state=cfg.train.seed).fit(X)
            P.copy_(torch.from_numpy(km.cluster_centers_).to(P.dtype).to(P.device))
    head.train()


# ---------------------------------------------------------------- 训练一步

def train_epoch(head, opt, sat, tr_idx, emb, mask, hand, y, cfg, device,
                pos_weight=None, seq=None):
    head.train()
    perm_rng = np.random.RandomState(cfg.train.seed * 1000 + sat.epoch)
    order = tr_idx[perm_rng.permutation(len(tr_idx))]
    total, n_seen = 0.0, 0

    for b in iter_batches(order, cfg.train.batch_size, shuffle=False):
        h = torch.from_numpy(np.asarray(emb[b], dtype=np.float32)).to(device)
        m = torch.from_numpy(np.asarray(mask[b])).to(device).float()
        hf = torch.from_numpy(hand[b]).to(device)
        yb = torch.from_numpy(y[b]).to(device)
        sq = None
        if seq is not None:
            sq = torch.from_numpy(np.asarray(seq[b], dtype=np.float32)).to(device)

        target = sat.targets(b, yb).to(device)
        if head.proto_enabled:
            logits, aux = head(h, m, hf, sq, return_proto=True)
            loss = F.binary_cross_entropy_with_logits(logits, target,
                                                      pos_weight=pos_weight)
            # 辅助损失：逼 B 单独也具备判别力，否则主分支会把 alpha*B 一路压没
            loss = loss + cfg.model.proto_w_aux * F.binary_cross_entropy_with_logits(
                aux['B'], target, pos_weight=pos_weight)
            if cfg.model.proto_w_div > 0:
                loss = loss + cfg.model.proto_w_div * head.proto.diversity()
        else:
            logits = head(h, m, hf, sq)
            loss = F.binary_cross_entropy_with_logits(logits, target,
                                                      pos_weight=pos_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.train.grad_clip)
        opt.step()

        total += float(loss.item()) * len(b)
        n_seen += len(b)

    return total / max(n_seen, 1)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'splits',
                                                  'split_origtest_cv5_seed42_nobip', 'table.csv'))
    ap.add_argument('--folds', default='', help='默认 data/splits/<split-id>/folds.csv')
    ap.add_argument('--split-id', default='split_origtest_cv5_seed42_nobip')
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--exp', default='', help='实验名：产物写 experiments/<exp>/runs/')
    ap.add_argument('--out', default='', help='显式指定 run 的父目录（优先于 --exp）')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0, help='model seed（初始化/打乱/dropout）')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--monitor', default='', choices=['', 'auroc', 'aupr'],
                    help='早停 / 选最优 epoch 用的验证指标，默认 auroc（AUPR 仍照常记录）')
    ap.add_argument('--proto-pos', type=int, default=0,
                    help='阳性原型数；与 --proto-neg 都为 0 时关闭原型旁路（默认）')
    ap.add_argument('--proto-neg', type=int, default=0, help='阴性原型数')
    ap.add_argument('--proto-dim', type=int, default=128, help='原型支路的投影维度')
    ap.add_argument('--proto-tau', type=float, default=0.1, help='cosine 温度')
    ap.add_argument('--proto-alpha', type=float, default=1.0,
                    help='最终分数 = A + alpha*B；先固定，避免 alpha 与 B 尺度不可辨识')
    ap.add_argument('--proto-w-aux', type=float, default=0.5,
                    help='辅助损失 BCE(B) 的权重；不加的话主分支会把 B 压没')
    ap.add_argument('--proto-w-div', type=float, default=0.01,
                    help='类内原型多样性正则权重（防塌缩）')
    ap.add_argument('--proto-init', default='kmeans', choices=['kmeans', 'random'],
                    help='kmeans = 用本折训练特征做类内聚类初始化（多一次前向）')
    ap.add_argument('--pos-weight', type=float, default=0.0)
    ap.add_argument('--label-noise', type=float, default=0.0,
                    help='标签噪声注入比例（只翻训练行，对称翻转；默认 0=不动）')
    ap.add_argument('--ema', action='store_true', help='启用 EMA-SAT')
    ap.add_argument('--ema-alpha-min', type=float, default=0.7,
                    help='硬标签权重下限 α_min；α→0.7 即教师权重上限 0.3')
    ap.add_argument('--ema-decay', type=float, default=0.65,
                    help='教师状态 q 的滑动平均系数')
    ap.add_argument('--ema-start-epoch', type=int, default=2,
                    help='从该 epoch 起使用教师；上一轮末初始化 q')
    ap.add_argument('--ema-tau', type=float, default=1.0,
                    help='α 半衰期（epoch）；1.0 即每轮减半')
    ap.add_argument('--norm', default='', choices=['', 'layer', 'peptide_bn', 'masked_batch', 'batch'],
                    help="残差块归一化。默认 peptide_bn（掩码 BN、按肽等权）；"
                         "layer = 逐位置 LayerNorm；masked_batch = masked BN 按残基等权；"
                         "batch = 原生 BN（padding 进统计量，仅消融/复现旧行为）")
    ap.add_argument('--hand-norm', default='', choices=['', 'standard', 'none'],
                    help='手工特征标准化。默认 standard（每折只用训练行拟合 mean/std）；'
                         'none 为旧行为，694 维里基本只有 mw/charg 起作用')
    ap.add_argument('--hand-select', default='', choices=['', 'none', 'mi_rf'],
                    help='手工特征筛选：mi_rf = 方差 + 相关性 + 互信息 + 随机森林'
                         '选到 ≤80 维，只用本折训练行拟合（索引写进 run 目录）')
    ap.add_argument('--fuse', default='', choices=['', 'early', 'late'],
                    help='融合方式：early = 拼原始特征（旧行为）；late = 两路各投影到 '
                         'fused_dim 后做 [h_e,h_h,h_e*h_h,|h_e-h_h|] 交互')
    ap.add_argument('--fused-dim', type=int, default=0,
                    help='late 融合的投影维度（默认沿用配置里的 128）')
    ap.add_argument('--amphipath', action='store_true',
                    help='手工块追加 10 维两亲性特征（疏水矩 μH，滑窗 7/11、转角 100°）。'
                         '**这是默认行为**，本开关只为显式表达意图')
    ap.add_argument('--no-amphipath', action='store_true',
                    help='关掉两亲性特征，退回 694 维基础块（消融用）')
    ap.add_argument('--esm-cache', default='', choices=['', 'auto', 'single', 'layers'],
                    help='读哪份 ESM 缓存。建好分层缓存后 auto 会优先用它，'
                         '复现 1 层历史基线请显式给 single')
    ap.add_argument('--esm-fuse', default='', choices=['', 'weighted', 'mean', 'last'],
                    help='多层 ESM 缓存的融合方式（默认 weighted：K 个 softmax 标量权重）。'
                         '只在 cache_esm.py --layers 生成的缓存下起作用')
    ap.add_argument('--lnn', action='store_true',
                    help='启用 CfC(LNN) 序列支路：one-hot → 闭式 CfC → 池化 → concat')
    ap.add_argument('--lnn-dim', type=int, default=128, help='CfC 隐藏维度（默认 128）')
    ap.add_argument('--lnn-dt', type=float, default=1.0,
                    help='CfC 的 Δt（默认 1.0 = 一个残基一个时间步）')
    ap.add_argument('--device', default='auto',
                    help='auto/cuda/mps/cpu，默认自动选择')
    ap.add_argument('--run-name', default='', help='默认 fold{k}_{split_id}_m{seed}')
    ap.add_argument('--overwrite', action='store_true', help='run 目录已存在时覆盖')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--lnn-only', action='store_true',
                    help='探针：只训练 CfC 一条支路（关掉 ESM 与手工特征）')
    args = ap.parse_args()

    cfg = RunCfg(fold=args.fold, exp=args.exp)
    cfg.data.data_csv = args.data
    cfg.data.split_id = args.split_id
    cfg.data.folds_csv = args.folds or os.path.join(
        ROOT, 'data', 'splits', args.split_id, 'folds.csv')
    cfg.data.cache_dir = args.cache_dir
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.patience = args.patience
    if args.monitor:
        cfg.train.monitor = args.monitor
    cfg.model.proto_pos = args.proto_pos
    cfg.model.proto_neg = args.proto_neg
    cfg.model.proto_dim = args.proto_dim
    cfg.model.proto_tau = args.proto_tau
    cfg.model.proto_alpha = args.proto_alpha
    cfg.model.proto_w_aux = args.proto_w_aux
    cfg.model.proto_w_div = args.proto_w_div
    cfg.model.proto_init = args.proto_init
    cfg.train.pos_weight = args.pos_weight
    cfg.train.label_noise = args.label_noise
    cfg.train.device = args.device
    cfg.ema.enabled = args.ema
    cfg.ema.alpha_min = args.ema_alpha_min
    cfg.ema.decay = args.ema_decay
    cfg.ema.start_epoch = args.ema_start_epoch
    cfg.ema.tau = args.ema_tau
    if args.norm:
        cfg.model.norm = args.norm
    if args.hand_norm:
        cfg.model.hand_norm = args.hand_norm
    if args.hand_select:
        cfg.model.hand_select = args.hand_select
    if args.fuse:
        cfg.model.fuse = args.fuse
    if args.fused_dim:
        cfg.model.fused_dim = args.fused_dim
    if args.amphipath:
        cfg.model.hand_extra = 'amphipath'
    if args.no_amphipath:
        cfg.model.hand_extra = ''
    if args.esm_fuse:
        cfg.model.esm_fuse = args.esm_fuse
    cfg.model.lnn = bool(args.lnn)
    cfg.model.lnn_dim = args.lnn_dim
    cfg.model.lnn_dt = args.lnn_dt
    if args.lnn_only:
        # 探针实验：回答"这条支路自己有没有信号"，先于任何融合尝试
        cfg.model.lnn = True
        cfg.model.use_esm = False
        cfg.model.use_hand = False

    run_name = args.run_name or cfg.run_name()
    if args.out:
        out_dir_base = args.out
    elif args.exp:
        out_dir_base = os.path.join(ROOT, 'experiments', args.exp, 'runs')
    else:
        # 不再往顶层 runs/ 写：产物必须归属某个实验，否则又回到"谁覆盖谁说不清"
        raise SystemExit('必须给 --exp（写到 experiments/<exp>/runs/）或 --out（显式目录）')
    run_dir = os.path.join(out_dir_base, run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.overwrite:
        raise SystemExit(
            'run 目录已存在且非空：%s\n'
            '换一个 --exp / --run-name，或加 --overwrite 明确覆盖。' % run_dir)
    ensure_dir(run_dir)
    device = pick_device(cfg.train.device)
    seed_everything(cfg.train.seed)

    # ---- 数据 ----
    table = load_table(cfg.data.data_csv)
    cfg.data.split_sha256 = sha256_file(cfg.data.folds_csv)
    y = table['label'].values.astype(np.float32)
    folds = load_folds(cfg.data.folds_csv, cfg.fold, table)
    hand = hand_feats(table, cfg.data.cache_dir,
                      with_amphipath=(cfg.model.hand_extra == 'amphipath'))
    # 维度由数据决定：694（基础）或 704（+两亲性），写进 cfg 与 checkpoint
    hand_raw_dim = int(hand.shape[1])
    if cfg.model.hand_extra == 'amphipath' and hand_raw_dim != 704:
        raise SystemExit('--amphipath 要求 hand_dim=704，实际 %d' % hand_raw_dim)
    cfg.model.hand_dim = hand_raw_dim
    # 特征筛选只能在**本折训练行**上拟合；索引写进 run 目录，推理时要按它取列
    h_idx = None
    if cfg.model.use_hand and cfg.model.hand_select != 'none':
        h_idx = select_hand_features(hand[folds['train']], y[folds['train']],
                                     seed=cfg.train.seed)
        hand = hand[:, h_idx]
        cfg.model.hand_dim = int(hand.shape[1])
        save_hand_selector(os.path.join(run_dir, 'hand_selector.npz'), h_idx)
        print('[%s] 手工特征筛选：%d → %d 维（只用 %d 条训练行）'
              % (run_name, hand_raw_dim, hand.shape[1], len(folds['train'])))
    elif cfg.model.use_hand:
        cfg.model.hand_dim = hand_raw_dim
    if not cfg.model.use_hand:
        h_mean = h_std = None
        print('[%s] 手工特征支路已关闭（--lnn-only），不做标准化' % run_name)
    elif cfg.model.hand_norm == 'standard':
        # 只在本折训练行上拟合：val/test 的统计量参与进来就是泄漏
        h_mean, h_std = fit_hand_scaler(hand, folds['train'])
        hand = apply_hand_scaler(hand, h_mean, h_std)
        save_hand_scaler(os.path.join(run_dir, 'hand_scaler.npz'), h_mean, h_std)
        print('[%s] 手工特征标准化：%d 维，统计量取自 %d 条训练样本'
              % (run_name, hand.shape[1], len(folds['train'])))
    else:
        h_mean = h_std = None
        print('[%s] [warn] hand_norm=%s：手工特征未标准化，'
              '手工特征里基本只有 mw/charg 两项在起作用' % (run_name, cfg.model.hand_norm))
    emb, mask = load_esm_cache(cfg.data.cache_dir, table,
                               prefer=args.esm_cache or 'auto')
    # 嵌入维度以缓存为准：换 ESM-2 3B(2560) / 小模型(320/480) 时不用改配置
    cfg.model.embed_dim = int(emb.shape[-1])
    if emb.ndim == 4:
        # 多层缓存：[N, K, L, D]。层号写进 cfg 与权重，推理时必须原样复现
        meta_esm = esm_meta(cfg.data.cache_dir)
        cfg.model.esm_layers = int(emb.shape[1])
        cfg.model.esm_layer_ids = ','.join(str(x) for x in meta_esm.get('layers', []))
        print('[%s] 多层 ESM：K=%d 层 [%s]，融合=%s'
              % (run_name, cfg.model.esm_layers, cfg.model.esm_layer_ids or '?',
                 cfg.model.esm_fuse))
    elif emb.ndim != 3:
        raise SystemExit('ESM 缓存维度异常：%s' % (emb.shape,))

    # ---- CfC 支路的输入：与 ESM 缓存逐行对齐的 one-hot ----
    seq = None
    if cfg.model.lnn:
        seq, lens = seq_onehot(table['sequence'].values, lmax=mask.shape[1])
        cached = np.asarray(mask, dtype=np.int64).sum(axis=1)
        bad = np.where(cached != lens)[0]
        if len(bad):
            raise SystemExit(
                '序列长度与 ESM 掩码对不上（首个第 %d 行：序列 %d vs 掩码 %d），'
                '请重建缓存：python scripts/cache_esm.py --force'
                % (bad[0], lens[bad[0]], cached[bad[0]]))
        print('[%s] CfC 支路：one-hot %s，hidden=%d，Δt=%.2f'
              % (run_name, seq.shape, cfg.model.lnn_dim, cfg.model.lnn_dt))

    # ---- 标签噪声注入（可审计：翻转了哪些行全部落盘）----
    y_fit = y
    if cfg.train.label_noise > 0:
        rng = np.random.default_rng(cfg.train.seed * 100003 + cfg.fold)
        tr = np.asarray(folds['train'])
        flip = rng.random(len(tr)) < cfg.train.label_noise
        y_fit = y.copy()
        y_fit[tr[flip]] = 1.0 - y_fit[tr[flip]]
        rec = {'enabled': True, 'rate': cfg.train.label_noise,
               'n_train': int(len(tr)), 'n_flipped': int(flip.sum()),
               'seed': cfg.train.seed, 'fold': cfg.fold,
               'rows': [int(r) for r in tr[flip]]}
        save_json(rec, os.path.join(run_dir, 'label_noise.json'))
        print('[%s] 标签噪声 %.1f%%：翻转 %d / %d 条训练样本'
              % (run_name, 100 * cfg.train.label_noise, rec['n_flipped'], rec['n_train']))

    n = len(table)
    print('[%s] 数据 %d | train %d / val %d / test %d | 阳性率 %.3f'
          % (run_name, n, len(folds['train']), len(folds['val']), len(folds['test']),
             float(y.mean())))
    print('[%s] %s' % (run_name, cfg.diff()))

    cfg.save(os.path.join(run_dir, 'config.json'))
    split_rows = pd.concat([
        pd.DataFrame({'sequence': table['sequence'].values[idx],
                      'label': table['label'].values[idx],
                      'row': idx, 'split': sp})
        for sp, idx in folds.items()
    ], ignore_index=True)
    split_rows.to_csv(os.path.join(run_dir, 'split.csv'), index=False)

    # ---- 模型 ----
    head = AIPHead(embed_dim=cfg.model.embed_dim, hand_dim=cfg.model.hand_dim,
                   hidden=cfg.model.hidden, n_res=cfg.model.n_res,
                   cnn_drop=cfg.model.cnn_drop, mlp_drop=cfg.model.mlp_drop,
                   norm=cfg.model.norm,
                   lnn=cfg.model.lnn, lnn_dim=cfg.model.lnn_dim,
                   lnn_dt=cfg.model.lnn_dt,
                   use_esm=cfg.model.use_esm, use_hand=cfg.model.use_hand,
                   esm_layers=cfg.model.esm_layers,
                   esm_fuse=cfg.model.esm_fuse,
                   proto_pos=cfg.model.proto_pos, proto_neg=cfg.model.proto_neg,
                   proto_dim=cfg.model.proto_dim, proto_tau=cfg.model.proto_tau,
                   proto_alpha=cfg.model.proto_alpha,
                   fuse=cfg.model.fuse, fused_dim=cfg.model.fused_dim).to(device)
    print('[%s] 模型 %.2fM 参数 / %.1f MB（lnn=%s esm=%s hand=%s feat=%d layers=%d/%s）'
          % (run_name, head.n_params() / 1e6, head.size_mb(), head.lnn,
             head.use_esm, head.use_hand, head.feat_dim,
             head.esm_layers, head.esm_fuse))
    if head.proto_enabled:
        init_prototypes(head, folds['train'], emb, mask, hand, y, cfg, device, seq=seq)
        print('[%s] 原型旁路：%d 阳 + %d 阴，投影 %d，tau %.2f，alpha %.2f，'
              'init=%s，w_aux %.2f，w_div %.3f'
              % (run_name, cfg.model.proto_pos, cfg.model.proto_neg,
                 cfg.model.proto_dim, cfg.model.proto_tau, cfg.model.proto_alpha,
                 cfg.model.proto_init, cfg.model.proto_w_aux, cfg.model.proto_w_div))
    opt = torch.optim.Adam(head.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    sat = EMASAT(n, alpha_min=cfg.ema.alpha_min, decay=cfg.ema.decay,
                 start_epoch=cfg.ema.start_epoch, tau=cfg.ema.tau,
                 enabled=cfg.ema.enabled)
    pos_weight = torch.tensor([cfg.train.pos_weight], device=device) \
        if cfg.train.pos_weight > 0 else None

    best_score, best_epoch, best_state = -1.0, -1, None
    best_aupr = best_auc = float('nan')
    monitor = cfg.train.monitor
    log_rows = []
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        sat.set_epoch(epoch)
        loss = train_epoch(head, opt, sat, folds['train'], emb, mask, hand, y_fit,
                           cfg, device, pos_weight, seq=seq)

        # 教师用 epoch 末、eval 模式的预测更新一次；避免 drop-out / 更新前 logits 噪声。
        if cfg.ema.enabled:
            train_p = predict_ordered(head, folds['train'], emb, mask, hand, device,
                                      cfg.train.batch_size, seq=seq)
            sat.update(folds['train'], train_p)

        val_p = predict_ordered(head, folds['val'], emb, mask, hand, device,
                                cfg.train.batch_size, seq=seq)
        val_aupr = threshold_free(y[folds['val']], val_p)['aupr']
        val_auc = threshold_free(y[folds['val']], val_p)['auroc']
        st = sat.stats()

        row = {'epoch': epoch, 'train_loss': loss, 'val_aupr': val_aupr,
               'val_auroc': val_auc, 'ema_mean': st['ema_mean'],
               'ema_std': st['ema_std'], 'ema_seen': st['seen'],
               'ema_alpha': st['alpha'], 'ema_beta': st['beta'],
               'ema_active': int(sat.active),
               'seconds': round(time.time() - t0, 1)}

        if cfg.ema.enabled:
            seen = sat.seen
            alpha = st['alpha']
            soft = alpha * y_fit + (1.0 - alpha) * sat.ema
            pos = seen & (y_fit > 0.5)
            neg = seen & (y_fit <= 0.5)
            row['soft_pos_mean'] = float(soft[pos].mean()) if pos.any() else 0.0
            row['soft_neg_mean'] = float(soft[neg].mean()) if neg.any() else 0.0

        log_rows.append(row)
        if not args.quiet:
            print('  ep%02d loss %.4f | val AUROC %.4f AUPR %.4f | ema %.3f±%.3f α %.2f β %.2f%s'
                  % (epoch, loss, val_auc, val_aupr, st['ema_mean'], st['ema_std'],
                     st['alpha'], st['beta'],
                     '' if sat.active else ' (warmup)'), end='\r')

        score = val_auc if monitor == 'auroc' else val_aupr
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_aupr, best_auc = val_aupr, val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        elif epoch - best_epoch >= cfg.train.patience:
            print('\n[%s] 早停于 ep%d（最优 ep%d, val %s %.4f）'
                  % (run_name, epoch, best_epoch,
                     'AUROC' if monitor == 'auroc' else 'AUPR', best_score))
            break

    head.load_state_dict(best_state)
    print('\n[%s] 训练完毕：最优 ep%d, val AUROC %.4f / AUPR %.4f（按 %s 选优）, 用时 %.0fs'
          % (run_name, best_epoch, best_auc, best_aupr,
             cfg.train.monitor.upper(), time.time() - t0))

    # ---- 原型旁路：把 A / B 分开落盘，方便判断"提升来自主分支还是原型" ----
    if head.proto_enabled:
        pa, pb = predict_components(head, np.arange(n), emb, mask, hand, device,
                                    max(cfg.train.batch_size, 64), seq=seq)
        pd.DataFrame({'row': np.arange(n), 'label': table['label'].values,
                      'prob': pa}).to_csv(os.path.join(run_dir, 'probs_A.csv'), index=False)
        pd.DataFrame({'row': np.arange(n), 'label': table['label'].values,
                      'prob': pb}).to_csv(os.path.join(run_dir, 'probs_B.csv'), index=False)
        np.savez(os.path.join(run_dir, 'prototypes.npz'),
                 p_pos=head.proto.p_pos.detach().cpu().numpy(),
                 p_neg=head.proto.p_neg.detach().cpu().numpy(),
                 alpha=cfg.model.proto_alpha)
        for name, p in (('A', pa), ('B', pb)):
            m = threshold_free(y[folds['test']], p[folds['test']])
            print('[%s] test 分量 %s：AUROC %.4f / AUPR %.4f'
                  % (run_name, name, m['auroc'], m['aupr']))

    # ---- 落盘：概率 ----
    all_p = predict_ordered(head, np.arange(n), emb, mask, hand, device,
                            max(cfg.train.batch_size, 64), seq=seq)
    val_p = all_p[folds['val']]
    test_p = all_p[folds['test']]

    thr = search_best_mcc(y[folds['val']], val_p, cfg.threshold.lo,
                          cfg.threshold.hi, cfg.threshold.n)
    save_json({'split_id': cfg.data.split_id, 'fold': cfg.fold,
               'model_seed': cfg.train.seed,
               'split_sha256': cfg.data.split_sha256,
               'ema': bool(cfg.ema.enabled),
               'best_epoch': best_epoch, 'monitor': cfg.train.monitor,
               'val_auroc': best_auc, 'val_aupr': best_aupr, **thr},
              os.path.join(run_dir, 'threshold.json'))

    pd.DataFrame({'sequence': table['sequence'].values[folds['val']],
                  'label': table['label'].values[folds['val']],
                  'row': folds['val'], 'prob': val_p}
                 ).to_csv(os.path.join(run_dir, 'val_probs.csv'), index=False)
    pd.DataFrame({'sequence': table['sequence'].values[folds['test']],
                  'label': table['label'].values[folds['test']],
                  'row': folds['test'], 'prob': test_p}
                 ).to_csv(os.path.join(run_dir, 'test_probs.csv'), index=False)
    pd.DataFrame({'row': np.arange(n), 'label': table['label'].values, 'prob': all_p}
                 ).to_csv(os.path.join(run_dir, 'all_probs.csv'), index=False)
    pd.DataFrame(log_rows).to_csv(os.path.join(run_dir, 'train_log.csv'), index=False)

    save_head(head, os.path.join(run_dir, 'best.pt'), {
        'fold': cfg.fold, 'seed': cfg.train.seed,
        'threshold': thr['threshold'], 'monitor': cfg.train.monitor,
        'val_auroc': best_auc, 'val_aupr': best_aupr,
        'split_id': cfg.data.split_id,
        'hand_norm': cfg.model.hand_norm,
        'hand_extra': cfg.model.hand_extra,
        'hand_select': cfg.model.hand_select,
        'hand_raw_dim': hand_raw_dim,
        'fuse': cfg.model.fuse,
        'fused_dim': cfg.model.fused_dim,
        'esm_layers': cfg.model.esm_layers,
        'esm_fuse': cfg.model.esm_fuse,
        'esm_layer_ids': cfg.model.esm_layer_ids,
        'lnn': cfg.model.lnn, 'lnn_dim': cfg.model.lnn_dim,
        'proto_pos': cfg.model.proto_pos, 'proto_neg': cfg.model.proto_neg,
        'proto_dim': cfg.model.proto_dim, 'proto_tau': cfg.model.proto_tau,
        'proto_alpha': cfg.model.proto_alpha,
        'use_esm': cfg.model.use_esm, 'use_hand': cfg.model.use_hand,
        'train_data': os.path.basename(cfg.data.folds_csv),
        'note': 'single-task AIP head; metrics computed by eval_aip.py',
    })

    print('[%s] val 阈值 %.3f (val MCC %.4f) | 产物：%s'
          % (run_name, thr['threshold'], thr['mcc'], run_dir))
    nxt = ('--exp %s' % args.exp) if args.exp else ('--runs "%s"' % out_dir_base)
    print('[%s] 下一步：python scripts/eval_aip.py %s' % (run_name, nxt))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
