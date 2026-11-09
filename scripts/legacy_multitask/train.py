#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三模态自适应训练框架 (Fig 5 热图数据记录版)
- Mode 2: EMA-SAT
- 记录每轮预测概率分布（Panel B 热图）
- 记录 FN/FP（Panel D）
- 记录 EMA 统计（Panel A）
- 记录 Class weights（Panel C）
"""
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from extract_handcraft_features import compute_handcraft, clean_sequence, AA20

# ---------------- 配置区 ----------------
SAT_MODE = 2  # 固定 Mode 2 (EMA-SAT)
# 【CSV 列名定义】
LABEL_AIP_COL = "label-aip"
LABEL_PIP_COL = "label-pip"
MASK_AIP_COL = "mask-aip"
MASK_PIP_COL = "mask-pip"
SEQ_COL = "squence"

TRAIN_CSV = r"D:\BIP\data\train2.csv"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # ← 缺失的
OUT_DIR = r"D:\BIP\module4\model9_2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "histograms"), exist_ok=True)  # 新增：热图数据目录

MODEL_PATH = os.path.join(OUT_DIR, f"esm2_mode{SAT_MODE}_best.pt")
LOG_PATH = os.path.join(OUT_DIR, f"train_log_mode{SAT_MODE}.csv")
THR_PATH = os.path.join(OUT_DIR, f"best_thresholds_mode{SAT_MODE}.csv")
HIST_PATH = os.path.join(OUT_DIR, f"pred_histogram_mode{SAT_MODE}.csv")  # Panel B 数据

RANDOM_STATE = 42
VAL_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 20  # 短周期 20 轮
LR = 1e-4

# 模型架构
HIDDEN_DIM = 512
MLP_DROPOUT = 0.3
NUM_RES_BLOCKS = 2
CNN_DROPOUT = 0.2

# SAT参数
SAT_ALPHA = 0.7
SAT_WARMUP_EPOCHS = 3

# EMA-SAT参数
EMA_DECAY = 0.99
ADAPTIVE_EMA = True
MIN_EMA_DECAY = 0.95
EMA_CLIP_MIN = 0.01
EMA_CLIP_MAX = 0.99
MAX_EMA_CHANGE = 0.3

# MCC类权重参数
MCC_ADJUST_INTERVAL = 3
MCC_FN_FP_RATIO_THR = 1.3
CLASS_ADJUST_RATE = 0.15
CLASS_WEIGHT_MAX = 3.0
CLASS_WEIGHT_MIN = 0.3

EARLY_STOP_PATIENCE = 10
SAMPLE_WEIGHT_MAP = {0: 1.0, 1: 1.5, 2: 1.5, 3: 2}

# 热图分 bin 设置
HIST_BINS = 30  # 0-1 分 20 格，每格 0.05


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=None)
def cached_compute_handcraft(seq_tuple):
    seq = ''.join(seq_tuple)
    return compute_handcraft(seq)


def combo_code_from_y(y_2col):
    return y_2col[:, 0] * 2 + y_2col[:, 1]


class SeqDataset(Dataset):
    def __init__(self, seqs, y2, mask2, sample_weight, indices=None):
        self.seqs = seqs
        self.y2 = y2.astype(np.float32)
        self.mask2 = mask2.astype(np.float32)
        self.sample_weight = sample_weight.astype(np.float32)
        self.indices = indices if indices is not None else np.arange(len(seqs))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (self.seqs[idx], self.y2[idx], self.mask2[idx],
                self.sample_weight[idx], self.indices[idx])


class ResBlock1D(nn.Module):
    def __init__(self, ch, ks=3, dil=1, drop=0.1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.drop(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        return self.relu(out + res)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden=256, drop=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(drop)
        )
        self.head_aip = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )
        self.head_pip = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        return torch.cat([self.head_aip(shared_feat), self.head_pip(shared_feat)], dim=1)


class ESMResCNNHandMLP(nn.Module):
    def __init__(self, esm_model, hidden=256, mlp_drop=0.3, n_res=2, cnn_drop=0.1, hand_dim=0):
        super().__init__()
        self.esm = esm_model
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()
        self.embed_dim = self.esm.config.hidden_size
        self.hand_dim = hand_dim
        self.cnn = nn.Sequential(*[ResBlock1D(self.embed_dim, drop=cnn_drop) for _ in range(n_res)])
        self.head = MLPHead(self.embed_dim + hand_dim, hidden, mlp_drop)

    def forward(self, input_ids, attention_mask, hand_feats=None):
        with torch.no_grad():
            h = self.esm(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True).hidden_states[-1]
        mask_valid = attention_mask.bool()
        x = h.transpose(1, 2)
        x = self.cnn(x)
        mask_cnt = mask_valid.unsqueeze(1).float().sum(dim=2).clamp(min=1.0)
        pooled = (x * mask_valid.unsqueeze(1).float()).sum(dim=2) / mask_cnt
        if hand_feats is not None and self.hand_dim > 0:
            pooled = torch.cat([pooled, hand_feats], dim=1)
        return self.head(pooled)


class EMA_SATManager:
    def __init__(self, n_samples, n_classes=2, device='cpu'):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.device = device
        self.register_buffer = torch.zeros(n_samples, n_classes, device=device)
        self.register_buffer[:, :] = 0.5
        self.initialized = torch.zeros(n_samples, dtype=torch.bool, device=device)
        self.update_count = torch.zeros(n_samples, device=device)

    def update(self, indices, current_probs, decay):
        with torch.no_grad():
            current_probs = torch.clamp(current_probs, EMA_CLIP_MIN, EMA_CLIP_MAX)
            for i, idx in enumerate(indices):
                idx = int(idx)
                old_val = self.register_buffer[idx]
                new_val = current_probs[i]
                if not self.initialized[idx]:
                    self.register_buffer[idx] = new_val
                    self.initialized[idx] = True
                else:
                    updated = decay * old_val + (1 - decay) * new_val
                    change = torch.abs(updated - old_val)
                    if (change > MAX_EMA_CHANGE).any():
                        direction = torch.sign(updated - old_val)
                        updated = old_val + direction * MAX_EMA_CHANGE
                    self.register_buffer[idx] = updated
                self.update_count[idx] += 1
            self.register_buffer = torch.clamp(self.register_buffer, EMA_CLIP_MIN, EMA_CLIP_MAX)

    def get_soft_targets(self, indices, y_true, alpha=0.7):
        batch_size = len(indices)
        ema_probs = torch.zeros(batch_size, self.n_classes, device=self.device)
        init_mask = self.initialized[indices]
        if init_mask.any():
            ema_probs[init_mask] = self.register_buffer[indices[init_mask]]
        if (~init_mask).any():
            ema_probs[~init_mask] = 0.5
        soft_targets = alpha * y_true + (1 - alpha) * ema_probs
        soft_targets[~init_mask] = y_true[~init_mask]
        return torch.clamp(soft_targets, 0.0, 1.0), ema_probs

    # 添加缺失的方法
    def check_health(self):
        if not self.initialized.any():
            return True, "No samples initialized yet"
        valid_probs = self.register_buffer[self.initialized]
        mean_val = valid_probs.mean().item()
        std_val = valid_probs.std().item()
        if mean_val < 0.05 or mean_val > 0.95:
            return False, f"EMA collapse detected: mean={mean_val:.3f}"
        if std_val < 0.05 and self.update_count[self.initialized].mean() > 10:
            return False, f"EMA stuck: std={std_val:.3f}"
        return True, f"EMA healthy: mean={mean_val:.3f}, std={std_val:.3f}"



def compute_metrics_pr_auc(logits, y2_true, mask2_true):
    y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y2_true.cpu().numpy()
    mask = mask2_true.cpu().numpy()
    res = {}
    for idx, name in enumerate(["AIP", "PIP"]):
        m = mask[:, idx] > 0.5
        if m.sum() == 0:
            res[f"{name}_roc_auc"] = np.nan
            res[f"{name}_pr_auc"] = np.nan
            continue
        y_t, y_p = y_true[m, idx], y_prob[m, idx]
        try:
            res[f"{name}_roc_auc"] = roc_auc_score(y_t, y_p)
        except ValueError:
            res[f"{name}_roc_auc"] = np.nan
        res[f"{name}_pr_auc"] = average_precision_score(y_t, y_p)
    res["macro_pr_auc"] = np.nanmean([res["AIP_pr_auc"], res["PIP_pr_auc"]])
    return res


def find_best_mcc_threshold(y_true, y_prob, mask, min_thr=0.10):
    m = mask == 1
    if m.sum() == 0:
        return 0.5, np.nan
    y_t, y_p = y_true[m], y_prob[m]
    if len(set(y_t)) < 2:
        return 0.5, np.nan
    thrs = np.linspace(min_thr, 0.99, 199)
    best_t, best_mcc = 0.5, -np.inf
    for t in thrs:
        pred = (y_p >= t).astype(int)
        try:
            mcc = matthews_corrcoef(y_t, pred)
        except Exception:
            continue
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t, best_mcc


def init_log_file():
    if not os.path.exists(LOG_PATH):
        cols = ["epoch", "train_loss", "val_macro_pr_auc",
                "AIP_roc_auc", "AIP_pr_auc", "PIP_roc_auc", "PIP_pr_auc",
                "AIP_mcc", "PIP_mcc",
                "ema_health", "ema_mean", "ema_std",
                "w_aip_pos", "w_aip_neg", "w_pip_pos", "w_pip_neg",
                "fn_rate_aip", "fp_rate_aip", "fn_rate_pip", "fp_rate_pip",
                "sat_mode"]
        pd.DataFrame(columns=cols).to_csv(LOG_PATH, index=False)


def append_log(epoch, train_loss, metrics, sat_mode, ema_health="N/A", ema_stats=(0, 0),
               mccs=(0, 0), class_weights=None, fn_fp_rates=None):
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_macro_pr_auc": metrics["macro_pr_auc"],
        "AIP_roc_auc": metrics["AIP_roc_auc"],
        "AIP_pr_auc": metrics["AIP_pr_auc"],
        "PIP_roc_auc": metrics["PIP_roc_auc"],
        "PIP_pr_auc": metrics["PIP_pr_auc"],
        "AIP_mcc": mccs[0],
        "PIP_mcc": mccs[1],
        "ema_health": ema_health,
        "ema_mean": ema_stats[0],
        "ema_std": ema_stats[1],
        "w_aip_pos": class_weights.get('AIP_pos', 1.0) if class_weights else 1.0,
        "w_aip_neg": class_weights.get('AIP_neg', 1.0) if class_weights else 1.0,
        "w_pip_pos": class_weights.get('PIP_pos', 1.0) if class_weights else 1.0,
        "w_pip_neg": class_weights.get('PIP_neg', 1.0) if class_weights else 1.0,
        "fn_rate_aip": fn_fp_rates.get('fn_rate_aip', 0.0) if fn_fp_rates else 0.0,
        "fp_rate_aip": fn_fp_rates.get('fp_rate_aip', 0.0) if fn_fp_rates else 0.0,
        "fn_rate_pip": fn_fp_rates.get('fn_rate_pip', 0.0) if fn_fp_rates else 0.0,
        "fp_rate_pip": fn_fp_rates.get('fp_rate_pip', 0.0) if fn_fp_rates else 0.0,
        "sat_mode": sat_mode
    }
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=False, index=False)


def append_histogram(epoch, y_prob_aip, y_prob_pip, mask_aip, mask_pip):
    """
    记录每轮预测概率的直方图（用于 Panel B 热图）
    只保存 bin counts，不保存原始值，节省空间
    """
    # 计算 AIP 直方图（只算有标签的样本）
    if mask_aip.sum() > 0:
        probs_aip = y_prob_aip[mask_aip]
        hist_aip, _ = np.histogram(probs_aip, bins=HIST_BINS, range=(0, 1), density=False)
        hist_aip = hist_aip / len(probs_aip)  # 归一化为比例
    else:
        hist_aip = np.zeros(HIST_BINS)

    # 计算 PIP 直方图
    if mask_pip.sum() > 0:
        probs_pip = y_prob_pip[mask_pip]
        hist_pip, _ = np.histogram(probs_pip, bins=HIST_BINS, range=(0, 1), density=False)
        hist_pip = hist_pip / len(probs_pip)
    else:
        hist_pip = np.zeros(HIST_BINS)

    # 合并保存
    row = {'epoch': epoch}
    for i in range(HIST_BINS):
        row[f'aip_bin_{i}'] = hist_aip[i]
        row[f'pip_bin_{i}'] = hist_pip[i]

    pd.DataFrame([row]).to_csv(HIST_PATH, mode='a',
                               header=(epoch == 1 and not os.path.exists(HIST_PATH)),
                               index=False)


def build_collate_fn(tokenizer):
    def collate(batch):
        seqs, y2, m2, ws, idxs = zip(*batch)
        hand_feats = []
        for seq in seqs:
            clean_seq = clean_sequence(seq)
            feat = cached_compute_handcraft(tuple(clean_seq))
            hand_feats.append(feat)
        hand_feats = np.stack(hand_feats, axis=0).astype(np.float32)
        enc = tokenizer(list(map(clean_sequence, seqs)), return_tensors="pt",
                        padding=True, truncation=True, max_length=1024)
        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor(np.stack(y2), dtype=torch.float32),
                torch.tensor(np.stack(m2), dtype=torch.float32),
                torch.tensor(np.array(ws), dtype=torch.float32),
                torch.tensor(hand_feats, dtype=torch.float32),
                torch.tensor(np.array(idxs), dtype=torch.long))

    return collate


def main():
    set_seed(RANDOM_STATE)
    init_log_file()

    print(f"Training Mode: {SAT_MODE} (EMA-SAT)")
    print(f"Epochs: {EPOCHS}, Recording histogram with {HIST_BINS} bins")

    df = pd.read_csv(TRAIN_CSV)
    df[LABEL_AIP_COL] = pd.to_numeric(df[LABEL_AIP_COL], errors='coerce').fillna(0).astype(int)
    df[LABEL_PIP_COL] = pd.to_numeric(df[LABEL_PIP_COL], errors='coerce').fillna(0).astype(int)
    df[MASK_AIP_COL] = df[MASK_AIP_COL].astype(int)
    df[MASK_PIP_COL] = df[MASK_PIP_COL].astype(int)

    seqs_all = df[SEQ_COL].astype(str).tolist()
    y2_all = df[[LABEL_AIP_COL, LABEL_PIP_COL]].values.astype(int)
    mask2_all = df[[MASK_AIP_COL, MASK_PIP_COL]].values.astype(int)

    sample_hand = compute_handcraft("ACDEFGHIKLMNPQRSTVWY")
    hand_dim = len(sample_hand)

    combo = combo_code_from_y(y2_all)
    sample_w = np.array([SAMPLE_WEIGHT_MAP.get(int(c), 1.0) for c in combo], dtype=np.float32)
    sample_w[(mask2_all.sum(1) == 0)] = 0.0

    idx = np.arange(len(df))
    tr_idx, val_idx = train_test_split(idx, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=combo)

    seqs_train = [seqs_all[i] for i in tr_idx]
    seqs_val = [seqs_all[i] for i in val_idx]
    y2_tr, y2_val = y2_all[tr_idx], y2_all[val_idx]
    m2_tr, m2_val = mask2_all[tr_idx], mask2_all[val_idx]
    w_tr, w_val = sample_w[tr_idx], sample_w[val_idx]

    train_ds = SeqDataset(seqs_train, y2_tr, m2_tr, w_tr, indices=tr_idx)
    val_ds = SeqDataset(seqs_val, y2_val, m2_val, w_val, indices=val_idx)

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME, do_lower_case=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(device)

    collate = build_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=RandomSampler(train_ds),
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

    model = ESMResCNNHandMLP(esm_model, hidden=HIDDEN_DIM, mlp_drop=MLP_DROPOUT,
                             n_res=NUM_RES_BLOCKS, cnn_drop=CNN_DROPOUT, hand_dim=hand_dim).to(device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    sat_manager = EMA_SATManager(len(df), n_classes=2, device=device)

    class_weights = {'AIP_pos': 1.0, 'AIP_neg': 1.0, 'PIP_pos': 1.0, 'PIP_neg': 1.0}
    adjust_cooldown = 0

    best_macro_pr = -1.0
    best_epoch = -1
    no_imp = 0

    for epoch in range(1, EPOCHS + 1):
        current_decay = EMA_DECAY
        if ADAPTIVE_EMA and epoch > 30:
            current_decay = max(MIN_EMA_DECAY, EMA_DECAY - 0.001 * (epoch - 30))

        # Training
        model.train()
        tot_loss, steps = 0.0, 0

        for input_ids, attn_mask, yb2, mb2, wb, hb, indices in train_loader:
            input_ids, attn_mask, yb2, mb2 = input_ids.to(device), attn_mask.to(device), yb2.to(device), mb2.to(device)
            wb, hb, indices = wb.to(device), hb.to(device), indices.to(device)

            opt.zero_grad()
            logits = model(input_ids, attn_mask, hb)

            if epoch > SAT_WARMUP_EPOCHS:
                sat_tgt, _ = sat_manager.get_soft_targets(indices, yb2, alpha=SAT_ALPHA)
                use_soft_mask = sat_manager.initialized[indices]
                final_targets = yb2.clone()
                final_targets[use_soft_mask] = sat_tgt[use_soft_mask]
            else:
                final_targets = yb2

            loss_per_axis = bce(logits, final_targets)
            sup_mask = (mb2 > 0.5).float()

            class_weight_matrix = torch.ones_like(loss_per_axis)
            aip_pos_mask = (yb2[:, 0] == 1) & (mb2[:, 0] > 0.5)
            aip_neg_mask = (yb2[:, 0] == 0) & (mb2[:, 0] > 0.5)
            pip_pos_mask = (yb2[:, 1] == 1) & (mb2[:, 1] > 0.5)
            pip_neg_mask = (yb2[:, 1] == 0) & (mb2[:, 1] > 0.5)

            class_weight_matrix[aip_pos_mask, 0] = class_weights['AIP_pos']
            class_weight_matrix[aip_neg_mask, 0] = class_weights['AIP_neg']
            class_weight_matrix[pip_pos_mask, 1] = class_weights['PIP_pos']
            class_weight_matrix[pip_neg_mask, 1] = class_weights['PIP_neg']

            loss_weighted = loss_per_axis * sup_mask * class_weight_matrix
            sup_den = sup_mask.sum(dim=1).clamp(min=1e-6).unsqueeze(1)
            per_sample_loss = (loss_weighted / sup_den).sum(dim=1)

            valid = wb > 0
            if valid.any():
                loss = (per_sample_loss * wb).sum() / wb[valid].sum().clamp(min=1e-6)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                with torch.no_grad():
                    logits_new = model(input_ids, attn_mask, hb)
                    probs_new = torch.sigmoid(logits_new)
                    sat_manager.update(indices, probs_new, current_decay)

                tot_loss += loss.item()
                steps += 1

        avg_loss = tot_loss / max(1, steps)

        # Validation
        model.eval()
        all_logits, all_y2, all_m2 = [], [], []

        with torch.no_grad():
            for input_ids, attn_mask, yb2, mb2, wb, hb, _ in val_loader:
                input_ids, attn_mask, hb = input_ids.to(device), attn_mask.to(device), hb.to(device)
                logits = model(input_ids, attn_mask, hb)
                all_logits.append(logits.cpu())
                all_y2.append(yb2.cpu())
                all_m2.append(mb2.cpu())

        all_logits = torch.cat(all_logits)
        all_y2 = torch.cat(all_y2)
        all_m2 = torch.cat(all_m2)

        # 计算指标
        metrics = compute_metrics_pr_auc(all_logits, all_y2, all_m2)
        y_prob_val = torch.sigmoid(all_logits).numpy()
        y_true_val = all_y2.numpy()
        mask_val = all_m2.numpy()

        # 【关键】记录 Panel B 热图数据：AIP 和 PIP 的预测概率分布
        mask_aip = mask_val[:, 0] == 1
        mask_pip = mask_val[:, 1] == 1
        y_prob_aip = y_prob_val[:, 0]
        y_prob_pip = y_prob_val[:, 1]
        append_histogram(epoch, y_prob_aip, y_prob_pip, mask_aip, mask_pip)

        # EMA 健康检查
        ema_health_str, ema_stats = "N/A", (0, 0)
        if epoch > SAT_WARMUP_EPOCHS:
            is_healthy, health_msg = sat_manager.check_health()
            ema_health_str = "OK" if is_healthy else "WARN"
            valid_probs = sat_manager.register_buffer[sat_manager.initialized]
            if valid_probs.numel() > 0:
                ema_stats = (valid_probs.mean().item(), valid_probs.std().item())

        # 计算 MCC 和 FN/FP（Panel D 数据）
        current_mcc = {}
        cm_dict = {}
        fn_fp_rates = {}

        for i, name in enumerate(["AIP", "PIP"]):
            t, mcc = find_best_mcc_threshold(y_true_val[:, i], y_prob_val[:, i], mask_val[:, i])
            current_mcc[name] = mcc if not np.isnan(mcc) else 0.0

            m = mask_val[:, i] == 1
            if m.sum() > 0 and len(set(y_true_val[m, i])) == 2:
                y_t, y_p_bin = y_true_val[m, i], (y_prob_val[m, i] >= t).astype(int)
                cm = confusion_matrix(y_t, y_p_bin, labels=[0, 1])
                cm_dict[name] = {'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]}

                total_pos = cm[1, 0] + cm[1, 1]
                total_neg = cm[0, 0] + cm[0, 1]
                fn_fp_rates[f'fn_rate_{name.lower()}'] = cm[1, 0] / total_pos if total_pos > 0 else 0.0
                fn_fp_rates[f'fp_rate_{name.lower()}'] = cm[0, 1] / total_neg if total_neg > 0 else 0.0
            else:
                cm_dict[name] = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
                fn_fp_rates[f'fn_rate_{name.lower()}'] = 0.0
                fn_fp_rates[f'fp_rate_{name.lower()}'] = 0.0

        # 保存日志（Panel A, C, D 数据）
        append_log(epoch, avg_loss, metrics, SAT_MODE, ema_health_str, ema_stats,
                   mccs=(current_mcc['AIP'], current_mcc['PIP']),
                   class_weights=class_weights, fn_fp_rates=fn_fp_rates)

        # Class-level 权重调整
        if epoch % MCC_ADJUST_INTERVAL == 0 and adjust_cooldown == 0:
            for task in ['AIP', 'PIP']:
                cm = cm_dict[task]
                total_pos = cm['TP'] + cm['FN']
                total_neg = cm['TN'] + cm['FP']

                if total_pos > 0 and total_neg > 0:
                    fn_rate = cm['FN'] / total_pos
                    fp_rate = cm['FP'] / total_neg
                    pos_key, neg_key = f'{task}_pos', f'{task}_neg'

                    if fn_rate > fp_rate * MCC_FN_FP_RATIO_THR:
                        class_weights[pos_key] = min(CLASS_WEIGHT_MAX, class_weights[pos_key] * (1 + CLASS_ADJUST_RATE))
                        class_weights[neg_key] = max(CLASS_WEIGHT_MIN,
                                                     class_weights[neg_key] * (1 - CLASS_ADJUST_RATE * 0.3))
                        print(
                            f"[Class-level] {task} FN({fn_rate:.3f}) > FP({fp_rate:.3f}), {pos_key}={class_weights[pos_key]:.2f}")
                    elif fp_rate > fn_rate * MCC_FN_FP_RATIO_THR:
                        class_weights[neg_key] = min(CLASS_WEIGHT_MAX, class_weights[neg_key] * (1 + CLASS_ADJUST_RATE))
                        class_weights[pos_key] = max(CLASS_WEIGHT_MIN,
                                                     class_weights[pos_key] * (1 - CLASS_ADJUST_RATE * 0.3))
                        print(
                            f"[Class-level] {task} FP({fp_rate:.3f}) > FN({fn_rate:.3f}), {neg_key}={class_weights[neg_key]:.2f}")

            adjust_cooldown = MCC_ADJUST_INTERVAL

        if adjust_cooldown > 0:
            adjust_cooldown -= 1

        # Early stopping
        if metrics["macro_pr_auc"] > best_macro_pr:
            best_macro_pr = metrics["macro_pr_auc"]
            best_epoch = epoch
            no_imp = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE:
                break

    print(f"Finished. Best PR-AUC={best_macro_pr:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
