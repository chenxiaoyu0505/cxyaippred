#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多肽 AIP/PIP 预测脚本（修正版）
基于 ESM2 + ResCNN + HandCraft Features 模型
手工特征维度: 694 (与训练时完全一致)
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from collections import Counter

# ---------------- 配置区 ----------------
MODEL_PATH = r"D:\BIP\module4\model9_2\esm2_mode2_best.pt"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_FILE = r"D:\BIP\module4\model9_2\peps\peps.txt"
OUTPUT_FILE = r"D:\BIP\module4\model9_2\peps\predictions.csv"

HIDDEN_DIM = 512
MLP_DROPOUT = 0.3
NUM_RES_BLOCKS = 2
CNN_DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------- 手工特征模块（完整版，与训练时完全一致） ----------------
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}

GAAC_GROUPS = {
    "aliphatic": set("GAVLMI"),
    "aromatic": set("FYW"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "uncharged": set("STCPNQ"),
}

CKSAAGP_GROUPS = {
    "aliphatic": set("GAVLMI"),
    "aromatic": set("FYW"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQ"),
    "special": set("CP")
}
CKSAAGP_NAMES = list(CKSAAGP_GROUPS.keys())
CKSAAGP_K_VALUES = [0, 1, 2, 3]

PAAC_PROPERTIES = {
    'A': [0.62, -0.5, 15.0, 2.35, 9.87, 6.00],
    'C': [0.29, -1.0, 47.0, 1.71, 10.78, 5.07],
    'D': [-0.90, 3.0, 59.0, 1.88, 9.60, 2.77],
    'E': [-0.74, 3.0, 73.0, 2.19, 9.67, 3.22],
    'F': [1.19, -2.5, 91.0, 2.58, 9.24, 5.48],
    'G': [0.48, 0.0, 1.0, 2.34, 9.60, 5.97],
    'H': [-0.40, -0.5, 82.0, 1.78, 8.97, 7.59],
    'I': [1.38, -1.8, 57.0, 2.32, 9.76, 6.02],
    'K': [-1.50, 3.0, 73.0, 2.20, 8.90, 9.74],
    'L': [1.06, -1.8, 57.0, 2.36, 9.60, 5.98],
    'M': [0.64, -1.3, 75.0, 2.28, 9.21, 5.74],
    'N': [-0.78, 2.0, 58.0, 2.18, 9.09, 5.41],
    'P': [0.12, 0.0, 42.0, 1.99, 10.60, 6.30],
    'Q': [-0.85, 0.2, 72.0, 2.17, 9.13, 5.65],
    'R': [-2.53, 3.0, 101.0, 2.18, 9.09, 10.76],
    'S': [-0.18, 0.3, 31.0, 2.21, 9.15, 5.68],
    'T': [-0.05, -0.4, 45.0, 2.15, 9.12, 5.66],
    'V': [1.08, -1.5, 43.0, 2.29, 9.74, 5.96],
    'W': [0.81, -3.4, 130.0, 2.38, 9.39, 5.89],
    'Y': [0.26, -2.3, 107.0, 2.20, 9.11, 5.66]
}
PAAC_LAMBDA = 30
PAAC_WEIGHT = 0.05

KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

RES_MASS = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}
WATER_MASS = 18.01528

HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")

PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDE = {"C": 8.33, "D": 3.86, "E": 4.25, "H": 6.00, "K": 10.53, "R": 12.48, "Y": 10.07}
POS_SIDE = set(["H", "K", "R"])
NEG_SIDE = set(["C", "D", "E", "Y"])

CTD_PROPERTIES = {
    "hydrophobicity": {"g1": set("RKEDQN"), "g2": set("GASTPHY"), "g3": set("CLVIMFW")},
    "polarity": {"g1": set("LIFWCMVY"), "g2": set("PATGS"), "g3": set("HQRKNED")},
    "charge": {"g1": set("KRH"), "g2": set("DE"), "g3": set("ACFGILMNPQSTVWY")},
}

HANDCRAFT_DIM = 694  # 20+5+400+144+50+12+63


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    cleaned = []
    for ch in seq:
        if ch in AA_SET:
            cleaned.append(ch)
        elif ch.isalpha():
            cleaned.append("X")
    if not cleaned:
        cleaned = ["X"]
    return "".join(cleaned)


def _net_charge_at_pH(seq, pH: float) -> float:
    counts = Counter([c for c in seq if c in AA_SET])
    if not counts:
        return 0.0
    n_term = 1.0 / (1.0 + 10 ** (pH - PKA_NTERM))
    c_term = -1.0 / (1.0 + 10 ** (PKA_CTERM - pH))
    charge = n_term + c_term
    for aa in POS_SIDE:
        if counts.get(aa, 0) == 0:
            continue
        pka = PKA_SIDE[aa]
        charge += counts[aa] * (1.0 / (1.0 + 10 ** (pH - pka)))
    for aa in NEG_SIDE:
        if counts.get(aa, 0) == 0:
            continue
        pka = PKA_SIDE[aa]
        charge += counts[aa] * (-1.0 / (1.0 + 10 ** (pka - pH)))
    return charge


def compute_pI(seq: str, iters: int = 40) -> float:
    s = "".join([c for c in seq if c in AA_SET])
    if len(s) == 0:
        return 7.0
    lo, hi = 0.0, 14.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ch = _net_charge_at_pH(s, mid)
        if ch > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_aac(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    x = np.zeros(20, dtype=np.float32)
    if L == 0:
        return x
    cnt = Counter(s)
    for aa, i in AA_TO_IDX.items():
        x[i] = cnt.get(aa, 0) / L
    return x


def compute_gaac(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    x = np.zeros(len(GAAC_GROUPS), dtype=np.float32)
    if L == 0:
        return x
    for gi, (gname, gset) in enumerate(GAAC_GROUPS.items()):
        x[gi] = sum(1 for c in s if c in gset) / L
    return x


def compute_dpc(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    x = np.zeros(400, dtype=np.float32)
    if L < 2:
        return x
    denom = (L - 1)
    for i in range(L - 1):
        a = s[i];
        b = s[i + 1]
        x[AA_TO_IDX[a] * 20 + AA_TO_IDX[b]] += 1.0
    x /= denom
    return x


def compute_ksaagp(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    n_groups = len(CKSAAGP_NAMES)
    features = []
    for k in CKSAAGP_K_VALUES:
        count = np.zeros((n_groups, n_groups), dtype=np.float32)
        if L > k + 1:
            denom = L - k - 1
            for i in range(L - k - 1):
                aa1, aa2 = s[i], s[i + k + 1]
                g1 = next((idx for idx, gname in enumerate(CKSAAGP_NAMES)
                           if aa1 in CKSAAGP_GROUPS[gname]), n_groups - 1)
                g2 = next((idx for idx, gname in enumerate(CKSAAGP_NAMES)
                           if aa2 in CKSAAGP_GROUPS[gname]), n_groups - 1)
                count[g1][g2] += 1.0
            count = count / denom
        features.extend(count.flatten())
    return np.array(features, dtype=np.float32)


def compute_paac(seq: str, lambda_val: int = PAAC_LAMBDA, weight: float = PAAC_WEIGHT) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(20 + lambda_val, dtype=np.float32)
    aac = np.zeros(20, dtype=np.float32)
    for i, aa in enumerate(AA20):
        aac[i] = s.count(aa) / L
    actual_lambda = min(lambda_val, L - 1)
    if actual_lambda < 1:
        return np.concatenate([aac, np.zeros(lambda_val, dtype=np.float32)])
    theta = []
    for j in range(1, actual_lambda + 1):
        corr_sum = 0.0
        for i in range(L - j):
            aa1, aa2 = s[i], s[i + j]
            props1 = PAAC_PROPERTIES.get(aa1, [0] * 6)
            props2 = PAAC_PROPERTIES.get(aa2, [0] * 6)
            diff_sq_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(props1, props2))
            corr_sum += diff_sq_sum / 6.0
        theta_j = corr_sum / (L - j)
        theta.append(theta_j)
    while len(theta) < lambda_val:
        theta.append(0.0)
    theta = np.array(theta, dtype=np.float32)
    denominator = 1 + weight * np.sum(theta)
    paac_tier1 = aac / denominator
    paac_tier2 = (weight * theta) / denominator
    return np.concatenate([paac_tier1, paac_tier2])


def compute_physchem(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(12, dtype=np.float32)
    cnt = Counter(s)
    length = float(L)
    mw = sum(RES_MASS[c] for c in s) + WATER_MASS
    gravy = float(np.mean([KD_SCALE[c] for c in s]))
    aromaticity = (cnt.get("F", 0) + cnt.get("Y", 0) + cnt.get("W", 0)) / L
    frac_hydrophobic = sum(cnt.get(a, 0) for a in HYDROPHOBIC) / L
    frac_polar = sum(cnt.get(a, 0) for a in POLAR) / L
    frac_pos = (cnt.get("K", 0) + cnt.get("R", 0) + cnt.get("H", 0)) / L
    frac_neg = (cnt.get("D", 0) + cnt.get("E", 0)) / L
    net_charge_approx = (cnt.get("K", 0) + cnt.get("R", 0) + 0.1 * cnt.get("H", 0)) - (
            cnt.get("D", 0) + cnt.get("E", 0))
    a_frac = cnt.get("A", 0) / L
    v_frac = cnt.get("V", 0) / L
    i_frac = cnt.get("I", 0) / L
    l_frac = cnt.get("L", 0) / L
    aliphatic_index = 100.0 * (a_frac + 2.9 * v_frac + 3.9 * (i_frac + l_frac))
    pI = compute_pI("".join(s))
    feats = np.array([
        length, mw, gravy, aromaticity, frac_hydrophobic, frac_polar,
        frac_pos, frac_neg, net_charge_approx, aliphatic_index, pI, math.log1p(length),
    ], dtype=np.float32)
    return feats


def _ctd_group_id(aa: str, prop_map) -> int:
    if aa in prop_map["g1"]: return 0
    if aa in prop_map["g2"]: return 1
    if aa in prop_map["g3"]: return 2
    return -1


def compute_ctd(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(63, dtype=np.float32)
    feats = []
    for prop, gmap in CTD_PROPERTIES.items():
        gids = [_ctd_group_id(a, gmap) for a in s]
        comp = np.zeros(3, dtype=np.float32)
        for g in gids:
            if g >= 0: comp[g] += 1.0
        comp /= L
        trans = np.zeros(3, dtype=np.float32)
        if L >= 2:
            denom = L - 1
            for i in range(L - 1):
                a, b = gids[i], gids[i + 1]
                if a == -1 or b == -1 or a == b: continue
                x, y = sorted((a, b))
                if (x, y) == (0, 1):
                    trans[0] += 1.0
                elif (x, y) == (0, 2):
                    trans[1] += 1.0
                elif (x, y) == (1, 2):
                    trans[2] += 1.0
            trans /= denom
        dist = np.zeros(15, dtype=np.float32)
        percentiles = [0.0, 0.25, 0.50, 0.75, 1.0]
        for g in [0, 1, 2]:
            positions = [i + 1 for i, gg in enumerate(gids) if gg == g]
            if len(positions) == 0:
                vals = [0.0] * 5
            else:
                vals = []
                m = len(positions)
                for p in percentiles:
                    k = int(math.ceil(p * m)) - 1
                    k = max(0, min(k, m - 1))
                    vals.append(positions[k] / L)
            dist[g * 5:(g + 1) * 5] = np.array(vals, dtype=np.float32)
        feats.append(comp)
        feats.append(trans)
        feats.append(dist)
    return np.concatenate(feats, axis=0).astype(np.float32)


def compute_handcraft(seq: str) -> np.ndarray:
    """计算完整手工特征，共694维"""
    return np.concatenate([
        compute_aac(seq),  # 20
        compute_gaac(seq),  # 5
        compute_dpc(seq),  # 400
        compute_ksaagp(seq),  # 144
        compute_paac(seq),  # 50
        compute_physchem(seq),  # 12
        compute_ctd(seq)  # 63
    ], axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def cached_compute_handcraft(seq_tuple):
    seq = ''.join(seq_tuple)
    return compute_handcraft(seq)


# ---------------- 模型定义 ----------------
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
    def __init__(self, esm_model, hidden=256, mlp_drop=0.3, n_res=2, cnn_drop=0.1, hand_dim=HANDCRAFT_DIM):
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


# ---------------- 数据集定义 ----------------
class PepDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        clean_seq = clean_sequence(seq)
        hand_feat = cached_compute_handcraft(tuple(clean_seq))
        return seq, clean_seq, torch.tensor(hand_feat, dtype=torch.float32)


def collate_fn(tokenizer):
    def inner_collate(batch):
        raw_seqs, clean_seqs, hand_feats = zip(*batch)
        enc = tokenizer(list(clean_seqs), return_tensors="pt",
                        padding=True, truncation=True, max_length=1024)
        return (raw_seqs, enc["input_ids"], enc["attention_mask"], torch.stack(hand_feats))

    return inner_collate


# ---------------- 预测主函数 ----------------
def predict():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件不存在: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        sequences = [line.strip() for line in f if line.strip()]

    print(f"\n加载了 {len(sequences)} 条多肽序列")

    print(f"\n正在加载 ESM2 模型: {ESM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME, do_lower_case=False)
    esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(device)

    model = ESMResCNNHandMLP(
        esm_model, hidden=HIDDEN_DIM, mlp_drop=MLP_DROPOUT,
        n_res=NUM_RES_BLOCKS, cnn_drop=CNN_DROPOUT, hand_dim=HANDCRAFT_DIM
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return

    print(f"\n加载模型权重: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功！")

    dataset = PepDataset(sequences)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn(tokenizer), num_workers=0)

    print("\n正在进行预测...")
    all_results = []

    with torch.no_grad():
        for batch_idx, (raw_seqs, input_ids, attn_mask, hand_feats) in enumerate(loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            hand_feats = hand_feats.to(device)

            logits = model(input_ids, attn_mask, hand_feats)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, seq in enumerate(raw_seqs):
                all_results.append({
                    'sequence': seq,
                    'AIP_probability': round(probs[i][0], 6),
                    'PIP_probability': round(probs[i][1], 6),
                    'AIP_prediction': 'Positive' if probs[i][0] > 0.5 else 'Negative',
                    'PIP_prediction': 'Positive' if probs[i][1] > 0.5 else 'Negative'
                })

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  已处理 {min((batch_idx + 1) * 8, len(sequences))} / {len(sequences)} 条序列")

    df_results = pd.DataFrame(all_results)

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    print("预测完成！")
    print(f"{'=' * 60}")
    print(f"\n结果摘要:")
    print(f"  - 总序列数: {len(df_results)}")
    print(f"  - AIP 阳性预测: {(df_results['AIP_prediction'] == 'Positive').sum()} 条")
    print(f"  - PIP 阳性预测: {(df_results['PIP_prediction'] == 'Positive').sum()} 条")
    print(f"  - AIP 平均概率: {df_results['AIP_probability'].mean():.4f}")
    print(f"  - PIP 平均概率: {df_results['PIP_probability'].mean():.4f}")

    print(f"\n详细结果已保存至: {OUTPUT_FILE}")
    print("\n所有预测结果:")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    predict()
