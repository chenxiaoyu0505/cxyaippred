#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本：兼容三种训练模式 (Hard/SAT/EMA-SAT)
- Mode 0: 纯硬标签训练模型
- Mode 1: 普通SAT训练模型
- Mode 2: EMA-SAT训练模型
- 自动根据SAT_MODE加载对应路径的模型和阈值
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, matthews_corrcoef, confusion_matrix,
    balanced_accuracy_score, roc_curve, precision_recall_curve
)
from functools import lru_cache
from extract_handcraft_features import compute_handcraft, clean_sequence
from transformers import AutoTokenizer, AutoModel

# ---------------- 配置区（与 train_multi_mode.py 保持一致）----------------
# 【核心选择】必须与训练时使用的模式一致！0=硬标签, 1=普通SAT, 2=EMA-SAT
SAT_MODE = 0  # 默认评估EMA-SAT模型，可改为0或1

TEST_CSV = r"D:\BIP\data\test.csv"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LABEL_AIP_COL = "label-aip"
LABEL_PIP_COL = "label-pip"
MASK_AIP_COL = "mask-aip"
MASK_PIP_COL = "mask-pip"
SEQ_COL = "squence"

# 模型架构参数（必须与训练时一致）
HIDDEN_DIM = 512
MLP_DROPOUT = 0.3
NUM_RES_BLOCKS = 2
CNN_DROPOUT = 0.2

# 路径配置（根据SAT_MODE自动调整）
OUT_DIR = r"D:\BIP\module4\model9_2"
MODEL_PATH = os.path.join(OUT_DIR, f"esm2_mode{SAT_MODE}_best.pt")
THR_PATH = os.path.join(OUT_DIR, f"best_thresholds_mode{SAT_MODE}.csv")

# 评估输出（自动带模式后缀，避免覆盖）
ROC_PLOT = os.path.join(OUT_DIR, f"test_ROC_mode{SAT_MODE}.png")
PR_PLOT = os.path.join(OUT_DIR, f"test_PR_mode{SAT_MODE}.png")
CM_PLOT = os.path.join(OUT_DIR, f"test_CM_mode{SAT_MODE}.png")
METRICS_CSV = os.path.join(OUT_DIR, f"test_metrics_mode{SAT_MODE}.csv")
PREDICTIONS_CSV = os.path.join(OUT_DIR, f"test_predictions_mode{SAT_MODE}.csv")
BATCH_SIZE = 16


# -------------------------------------

@lru_cache(maxsize=None)
def cached_compute_handcraft(seq_tuple):
    """缓存版本的手工特征计算"""
    seq = ''.join(seq_tuple)
    return compute_handcraft(seq)


class ResBlock1D(nn.Module):
    """1D残差块（与训练脚本一致）"""

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
    """MLP分类头（与训练脚本一致）"""

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
        return torch.cat([
            self.head_aip(shared_feat),
            self.head_pip(shared_feat)
        ], dim=1)


class ESMResCNNHandMLP(nn.Module):
    """主模型（与训练脚本完全一致）"""

    def __init__(self, esm_model, hidden=256, mlp_drop=0.3, n_res=2, cnn_drop=0.1, hand_dim=0):
        super().__init__()
        self.esm = esm_model
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

        self.embed_dim = self.esm.config.hidden_size
        self.hand_dim = hand_dim

        self.cnn = nn.Sequential(
            *[ResBlock1D(self.embed_dim, drop=cnn_drop) for _ in range(n_res)]
        )

        self.head = MLPHead(self.embed_dim + hand_dim, hidden, mlp_drop)

    def forward(self, input_ids, attention_mask, hand_feats=None):
        with torch.no_grad():
            h = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]

        mask_valid = attention_mask.bool()
        x = h.transpose(1, 2)
        x = self.cnn(x)

        mask_cnt = mask_valid.unsqueeze(1).float().sum(dim=2).clamp(min=1.0)
        pooled = (x * mask_valid.unsqueeze(1).float()).sum(dim=2) / mask_cnt

        if hand_feats is not None and self.hand_dim > 0:
            pooled = torch.cat([pooled, hand_feats], dim=1)

        return self.head(pooled)


class SeqDataset(torch.utils.data.Dataset):
    """数据集（与训练脚本兼容）"""

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


def build_eval_collate_fn(tokenizer):
    """评估用的collate_fn"""

    def collate(batch):
        seqs, y2, m2, ws, idxs = zip(*batch)

        # 实时计算手工特征（带缓存）
        hand_feats = []
        for seq in seqs:
            clean_seq = clean_sequence(seq)
            seq_tuple = tuple(clean_seq)
            feat = cached_compute_handcraft(seq_tuple)
            hand_feats.append(feat)

        hand_feats = np.stack(hand_feats, axis=0).astype(np.float32)

        enc = tokenizer(list(map(clean_sequence, seqs)),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024)

        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor(np.stack(y2), dtype=torch.float32),
                torch.tensor(np.stack(m2), dtype=torch.float32),
                torch.tensor(np.array(ws), dtype=torch.float32),
                torch.tensor(hand_feats, dtype=torch.float32))

    return collate


# --------------- 指标计算函数 ---------------
def compute_metrics_from_preds(y_true, y_prob, mask, preds):
    names = ["AIP", "PIP"]
    metrics = {}
    for i, name in enumerate(names):
        m = mask[:, i] == 1
        if m.sum() == 0:
            metrics[name] = dict.fromkeys([
                "auc", "pr_auc", "acc", "precision", "recall", "specificity",
                "f1", "balanced_acc", "mcc", "tn", "fp", "fn", "tp"
            ], np.nan)
            continue

        y_t, y_p, y_h = y_true[m, i], y_prob[m, i], preds[m, i]
        try:
            auc = roc_auc_score(y_t, y_p)
            pr_auc = average_precision_score(y_t, y_p)
        except:
            auc = pr_auc = np.nan

        cm = confusion_matrix(y_t, y_h, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = np.nan

        acc = accuracy_score(y_t, y_h) if len(set(y_t)) > 1 else np.nan
        prec = precision_score(y_t, y_h, zero_division=0)
        rec = recall_score(y_t, y_h, zero_division=0)
        f1v = f1_score(y_t, y_h, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        bal = balanced_accuracy_score(y_t, y_h) if len(set(y_t)) > 1 else np.nan
        try:
            mcc = matthews_corrcoef(y_t, y_h)
        except:
            mcc = np.nan

        metrics[name] = {
            "auc": auc, "pr_auc": pr_auc, "acc": acc, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1v, "balanced_acc": bal, "mcc": mcc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    macro = {}
    for k in metrics["AIP"].keys():
        vals = [metrics["AIP"][k], metrics["PIP"][k]]
        vals = [v for v in vals if not np.isnan(v)]
        macro["macro_" + k] = np.mean(vals) if vals else np.nan
    return metrics, macro


def plot_curves(y_true, y_prob, mask, out_dir, mode):
    """绘制ROC和PR曲线"""
    names = ["AIP", "PIP"]

    # ROC曲线
    plt.figure(figsize=(6, 6))
    for i, name in enumerate(names):
        m = mask[:, i] == 1
        if m.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[m, i], y_prob[m, i])
        auc = roc_auc_score(y_true[m, i], y_prob[m, i])
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve (Mode {mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"test_ROC_mode{mode}.png"), dpi=300)
    plt.close()

    # PR曲线
    plt.figure(figsize=(6, 6))
    for i, name in enumerate(names):
        m = mask[:, i] == 1
        if m.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true[m, i], y_prob[m, i])
        pr_auc = average_precision_score(y_true[m, i], y_prob[m, i])
        plt.plot(rec, prec, label=f"{name} PR-AUC={pr_auc:.3f}")
    plt.legend()
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Mode {mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"test_PR_mode{mode}.png"), dpi=300)
    plt.close()


def plot_confusion(y_true, mask, preds, out_dir, mode):
    """绘制混淆矩阵"""
    names = ["AIP", "PIP"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(names):
        ax = axes[i]
        m = mask[:, i] == 1
        if m.sum() == 0:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        cm = confusion_matrix(y_true[m, i], preds[m, i], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Pred", ylabel="True", title=f"{name}")
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"Confusion Matrix (Mode {mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"test_CM_mode{mode}.png"), dpi=300)
    plt.close()


# ---------------- 主函数 ----------------
def main():
    mode_names = {0: "Hard Label", 1: "Standard SAT", 2: "EMA-SAT"}

    print(f"=" * 60)
    print(f"评估模式: {SAT_MODE} ({mode_names[SAT_MODE]})")
    print(f"模型路径: {MODEL_PATH}")
    print(f"阈值路径: {THR_PATH}")
    print(f"=" * 60)

    # 检查文件存在性
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}\n请检查SAT_MODE设置是否与训练时一致！")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"测试数据不存在: {TEST_CSV}")

    print(f"\nLoading test data from: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded TEST CSV shape={df.shape}")

    # 数据预处理（与训练脚本一致）
    df[LABEL_AIP_COL] = (df[LABEL_AIP_COL] > 0).astype(int)
    df[LABEL_PIP_COL] = (df[LABEL_PIP_COL] > 0).astype(int)
    df[MASK_AIP_COL] = df[MASK_AIP_COL].astype(int)
    df[MASK_PIP_COL] = df[MASK_PIP_COL].astype(int)

    seqs = df[SEQ_COL].astype(str).tolist()
    y2_true = df[[LABEL_AIP_COL, LABEL_PIP_COL]].values.astype(int)
    mask2 = df[[MASK_AIP_COL, MASK_PIP_COL]].values.astype(int)

    # 动态获取手工特征维度
    sample_hand = compute_handcraft("ACDEFGHIKLMNPQRSTVWY")
    hand_dim = len(sample_hand)
    print(f"Handcraft dim={hand_dim} (实时计算，已启用缓存)")

    # 2. 模型加载
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME, do_lower_case=False)
    esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device)

    model = ESMResCNNHandMLP(
        esm_model,
        hidden=HIDDEN_DIM,
        mlp_drop=MLP_DROPOUT,
        n_res=NUM_RES_BLOCKS,
        cnn_drop=CNN_DROPOUT,
        hand_dim=hand_dim
    ).to(device)

    # 加载训练好的权重
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 处理可能的多层dict格式
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully")

    # 3. 数据加载
    sample_weight = np.ones(len(df), dtype=np.float32)
    test_ds = SeqDataset(seqs, y2_true, mask2, sample_weight)
    collate_fn = build_eval_collate_fn(tokenizer)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. 推理
    print("\nRunning inference...")
    all_probs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids, attn_mask, yb2, mb2, wb, hb = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            hb = hb.to(device)

            logits = model(input_ids, attn_mask, hb)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE}/{len(test_ds)} samples")

    y_prob = np.concatenate(all_probs, axis=0)
    print(f"Inference completed. Predictions shape: {y_prob.shape}")

    # 5. 加载阈值 + 评估
    if not os.path.exists(THR_PATH):
        print(f"警告: 阈值文件不存在: {THR_PATH}")
        print("使用默认阈值 0.5 进行评估")
        thresholds = [0.5, 0.5]
    else:
        thr_df = pd.read_csv(THR_PATH)
        t_aip = float(thr_df.loc[thr_df["label"] == "AIP", "threshold"].values[0])
        t_pip = float(thr_df.loc[thr_df["label"] == "PIP", "threshold"].values[0])
        thresholds = [t_aip, t_pip]
        print(f"\n加载最佳阈值: AIP={t_aip:.3f}, PIP={t_pip:.3f}")

    # 生成预测
    preds = np.zeros_like(y2_true)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(int)

    # 保存详细预测结果
    pred_df = pd.DataFrame({
        'sequence': seqs,
        'label_aip_true': y2_true[:, 0],
        'label_pip_true': y2_true[:, 1],
        'mask_aip': mask2[:, 0],
        'mask_pip': mask2[:, 1],
        'prob_aip': y_prob[:, 0],
        'prob_pip': y_prob[:, 1],
        'pred_aip': preds[:, 0],
        'pred_pip': preds[:, 1]
    })
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\n预测结果已保存 -> {PREDICTIONS_CSV}")

    # 计算指标
    metrics, macro = compute_metrics_from_preds(y2_true, y_prob, mask2, preds)

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Test Results - Mode {SAT_MODE} ({mode_names[SAT_MODE]})")
    print(f"{'=' * 60}")

    for name in ["AIP", "PIP"]:
        print(f"\n[{name}]")
        print(f"  AUC-ROC    : {metrics[name]['auc']:.4f}")
        print(f"  AUC-PR     : {metrics[name]['pr_auc']:.4f}")
        print(f"  MCC        : {metrics[name]['mcc']:.4f}")
        print(f"  F1-Score   : {metrics[name]['f1']:.4f}")
        print(f"  Recall     : {metrics[name]['recall']:.4f}")
        print(f"  Precision  : {metrics[name]['precision']:.4f}")
        print(f"  Specificity: {metrics[name]['specificity']:.4f}")
        print(f"  Accuracy   : {metrics[name]['acc']:.4f}")
        print(f"  TP={metrics[name]['tp']:.0f}, FP={metrics[name]['fp']:.0f}, "
              f"TN={metrics[name]['tn']:.0f}, FN={metrics[name]['fn']:.0f}")

    print(f"\n[MACRO Average]")
    print(f"  Macro AUC-ROC: {macro.get('macro_auc', np.nan):.4f}")
    print(f"  Macro AUC-PR : {macro.get('macro_pr_auc', np.nan):.4f}")
    print(f"  Macro MCC    : {macro.get('macro_mcc', np.nan):.4f}")
    print(f"  Macro F1     : {macro.get('macro_f1', np.nan):.4f}")
    print(f"{'=' * 60}")

    # 保存指标到CSV
    rows = []
    for name in ["AIP", "PIP"]:
        r = {"label": name, "mode": SAT_MODE}
        r.update(metrics[name])
        rows.append(r)
    macro_row = {"label": "MACRO", "mode": SAT_MODE}
    macro_row.update(macro)
    rows.append(macro_row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved -> {METRICS_CSV}")

    # 绘制图表
    plot_curves(y2_true, y_prob, mask2, OUT_DIR, SAT_MODE)
    plot_confusion(y2_true, mask2, preds, OUT_DIR, SAT_MODE)
    print(f"Plots saved: ROC, PR, Confusion Matrix (Mode {SAT_MODE})")

    # 打印缓存统计
    print(f"\n特征缓存统计: {cached_compute_handcraft.cache_info()}")

    # 内存清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("内存已清理")


if __name__ == "__main__":
    main()