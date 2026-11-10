#!/usr/bin/env python3
"""
Handcrafted feature extraction for protein sequences
"""

import math
import numpy as np
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

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
    'A': [0.62, -0.5, 15.0, 2.35, 9.87, 6.00], 'C': [0.29, -1.0, 47.0, 1.71, 10.78, 5.07],
    'D': [-0.90, 3.0, 59.0, 1.88, 9.60, 2.77], 'E': [-0.74, 3.0, 73.0, 2.19, 9.67, 3.22],
    'F': [1.19, -2.5, 91.0, 2.58, 9.24, 5.48], 'G': [0.48, 0.0, 1.0, 2.34, 9.60, 5.97],
    'H': [-0.40, -0.5, 82.0, 1.78, 8.97, 7.59], 'I': [1.38, -1.8, 57.0, 2.32, 9.76, 6.02],
    'K': [-1.50, 3.0, 73.0, 2.20, 8.90, 9.74], 'L': [1.06, -1.8, 57.0, 2.36, 9.60, 5.98],
    'M': [0.64, -1.3, 75.0, 2.28, 9.21, 5.74], 'N': [-0.78, 2.0, 58.0, 2.18, 9.09, 5.41],
    'P': [0.12, 0.0, 42.0, 1.99, 10.60, 6.30], 'Q': [-0.85, 0.2, 72.0, 2.17, 9.13, 5.65],
    'R': [-2.53, 3.0, 101.0, 2.18, 9.09, 10.76], 'S': [-0.18, 0.3, 31.0, 2.21, 9.15, 5.68],
    'T': [-0.05, -0.4, 45.0, 2.15, 9.12, 5.66], 'V': [1.08, -1.5, 43.0, 2.29, 9.74, 5.96],
    'W': [0.81, -3.4, 130.0, 2.38, 9.39, 5.89], 'Y': [0.26, -2.3, 107.0, 2.20, 9.11, 5.66]
}
PAAC_LAMBDA = 30
PAAC_WEIGHT = 0.05

KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
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


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    cleaned = [ch if ch in AA_SET else "X" for ch in seq if ch.isalpha()]
    return "".join(cleaned) if cleaned else "X"


def _net_charge_at_pH(seq, pH: float) -> float:
    counts = Counter([c for c in seq if c in AA_SET])
    if not counts:
        return 0.0
    n_term = 1.0 / (1.0 + 10 ** (pH - PKA_NTERM))
    c_term = -1.0 / (1.0 + 10 ** (PKA_CTERM - pH))
    charge = n_term + c_term
    for aa in POS_SIDE:
        if counts.get(aa, 0):
            charge += counts[aa] * (1.0 / (1.0 + 10 ** (pH - PKA_SIDE[aa])))
    for aa in NEG_SIDE:
        if counts.get(aa, 0):
            charge += counts[aa] * (-1.0 / (1.0 + 10 ** (PKA_SIDE[aa] - pH)))
    return charge


def compute_pI(seq: str, iters: int = 40) -> float:
    s = "".join([c for c in seq if c in AA_SET])
    if not s:
        return 7.0
    lo, hi = 0.0, 14.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ch = _net_charge_at_pH(s, mid)
        lo, hi = (mid, hi) if ch > 0 else (lo, mid)
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
    for i in range(L - 1):
        a, b = s[i], s[i + 1]
        x[AA_TO_IDX[a] * 20 + AA_TO_IDX[b]] += 1.0
    x /= (L - 1)
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
                g1 = next((idx for idx, g in enumerate(CKSAAGP_NAMES) if aa1 in CKSAAGP_GROUPS[g]), n_groups - 1)
                g2 = next((idx for idx, g in enumerate(CKSAAGP_NAMES) if aa2 in CKSAAGP_GROUPS[g]), n_groups - 1)
                count[g1][g2] += 1.0
            count /= denom
        features.extend(count.flatten())
    return np.array(features, dtype=np.float32)


def compute_paac(seq: str, lambda_val: int = PAAC_LAMBDA, weight: float = PAAC_WEIGHT) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(20 + lambda_val, dtype=np.float32)

    aac = np.array([s.count(aa) / L for aa in AA20], dtype=np.float32)
    actual_lambda = min(lambda_val, L - 1)

    if actual_lambda < 1:
        return np.concatenate([aac, np.zeros(lambda_val, dtype=np.float32)])

    theta = []
    for j in range(1, actual_lambda + 1):
        corr_sum = sum(
            sum((p1 - p2) ** 2 for p1, p2 in zip(PAAC_PROPERTIES[s[i]], PAAC_PROPERTIES[s[i + j]])) / 6.0
            for i in range(L - j)
        ) / (L - j)
        theta.append(corr_sum)

    theta += [0.0] * (lambda_val - len(theta))
    theta = np.array(theta, dtype=np.float32)
    denominator = 1 + weight * np.sum(theta)

    return np.concatenate([aac / denominator, (weight * theta) / denominator])


def compute_physchem(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(12, dtype=np.float32)

    cnt = Counter(s)
    length = float(L)
    mw = sum(RES_MASS[c] for c in s) + WATER_MASS
    gravy = np.mean([KD_SCALE[c] for c in s])

    return np.array([
        length, mw, gravy,
        (cnt.get("F", 0) + cnt.get("Y", 0) + cnt.get("W", 0)) / L,
        sum(cnt.get(a, 0) for a in HYDROPHOBIC) / L,
        sum(cnt.get(a, 0) for a in POLAR) / L,
        (cnt.get("K", 0) + cnt.get("R", 0) + cnt.get("H", 0)) / L,
        (cnt.get("D", 0) + cnt.get("E", 0)) / L,
        (cnt.get("K", 0) + cnt.get("R", 0) + 0.1 * cnt.get("H", 0)) - (cnt.get("D", 0) + cnt.get("E", 0)),
        100.0 * (cnt.get("A", 0) / L + 2.9 * cnt.get("V", 0) / L + 3.9 * (cnt.get("I", 0) / L + cnt.get("L", 0) / L)),
        compute_pI("".join(s)),
        math.log1p(length),
    ], dtype=np.float32)


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

        # Composition
        comp = np.array([sum(1 for g in gids if g == i) / L for i in range(3)], dtype=np.float32)

        # Transition
        trans = np.zeros(3, dtype=np.float32)
        if L >= 2:
            pairs = [(0, 1), (0, 2), (1, 2)]
            for i in range(L - 1):
                a, b = gids[i], gids[i + 1]
                if a != b and a >= 0 and b >= 0:
                    x, y = sorted((a, b))
                    for pi, (px, py) in enumerate(pairs):
                        if (x, y) == (px, py):
                            trans[pi] += 1.0
            trans /= (L - 1)

        # Distribution
        dist = np.zeros(15, dtype=np.float32)
        for g in range(3):
            positions = [i + 1 for i, gg in enumerate(gids) if gg == g]
            if positions:
                m = len(positions)
                for pi, p in enumerate([0.0, 0.25, 0.50, 0.75, 1.0]):
                    k = max(0, min(int(math.ceil(p * m)) - 1, m - 1))
                    dist[g * 5 + pi] = positions[k] / L

        feats.extend([comp, trans, dist])

    return np.concatenate(feats).astype(np.float32)


def compute_handcraft(seq: str) -> np.ndarray:
    """Compute all handcrafted features (694 dimensions)"""
    seq = clean_sequence(seq)
    return np.concatenate([
        compute_aac(seq),  # 20
        compute_gaac(seq),  # 5
        compute_dpc(seq),  # 400
        compute_ksaagp(seq),  # 144
        compute_paac(seq),  # 50
        compute_physchem(seq),  # 12
        compute_ctd(seq)  # 63
    ], axis=0).astype(np.float32)


def handcraft_feature_names():
    names = [f"aac_{a}" for a in AA20]
    names += [f"gaac_{k}" for k in GAAC_GROUPS.keys()]
    names += [f"dpc_{a}{b}" for a in AA20 for b in AA20]
    for k in CKSAAGP_K_VALUES:
        for g1 in CKSAAGP_NAMES:
            for g2 in CKSAAGP_NAMES:
                names.append(f"cksaagp_k{k}_{g1}_{g2}")
    names += [f"paac_tier1_{a}" for a in AA20]
    names += [f"paac_tier2_theta{i + 1}" for i in range(PAAC_LAMBDA)]
    names += ["len", "mw", "gravy", "aromaticity", "frac_hydrophobic", "frac_polar",
              "frac_pos", "frac_neg", "net_charge_approx", "aliphatic_index", "pI", "log1p_len"]
    for prop in CTD_PROPERTIES.keys():
        names += [f"ctd_{prop}_comp_g{i}" for i in [1, 2, 3]]
        names += [f"ctd_{prop}_trans_g12", f"ctd_{prop}_trans_g13", f"ctd_{prop}_trans_g23"]
        for g in [1, 2, 3]:
            for p in ["0", "25", "50", "75", "100"]:
                names.append(f"ctd_{prop}_dist_g{g}_p{p}")
    return names


# ------------------------------------------------------------------ 两亲性（疏水矩）
#
# 为什么单独算这一组
# ------------------
# 现有 694 维里，AAC 只有组成、没有相位；DPC 只看相邻一对；CTD 是分段统计。
# 没有任何一维能表达"疏水残基每隔 3~4 个出现一次"，而这正是 α-螺旋两亲性
# （AIP 的主要作用机制之一）的几何特征。ESM-2 理论上能学到，但让 2,998 条
# 样本从零学出周期结构不现实，显式算给它更稳。
#
# 疏水矩 μH = |Σ h_n · exp(i·δ·n)| / N，δ=100°（理想 α-螺旋的残基转角）。
# 按窗口长度归一化，不同长度的肽可比。

EISENBERG_SCALE = {
    "A": 0.25, "R": -1.76, "N": -0.64, "D": -0.72, "C": 0.04, "Q": -0.69,
    "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73, "L": 0.53, "K": -1.10,
    "M": 0.26, "F": 0.61, "P": -0.07, "S": -0.26, "T": -0.18, "W": 0.37,
    "Y": 0.02, "V": 0.54,
}

AMPHIPATH_WINDOWS = (7, 11)     # 约 2 圈与 3 圈 α-螺旋
AMPHIPATH_DELTA_DEG = 100.0
AMPHIPATH_HELIX_PERIOD = 3.6
AMPHIPATH_DIM = 10              # 4×2 + 2，见 amphipath_feature_names()


def _hydrophobic_moment_profile(h, w):
    """返回每个滑动窗的 μH（已按窗长归一化）与生效窗长。"""
    L = int(h.size)
    if L == 0:
        return np.zeros(1), 0
    w = int(min(w, L))
    if w < 2:
        return np.zeros(1), 0
    win = sliding_window_view(h, w)                 # [n_windows, w]
    ang = np.deg2rad(AMPHIPATH_DELTA_DEG) * np.arange(w)
    real = (win * np.cos(ang)).sum(axis=1)
    imag = (win * np.sin(ang)).sum(axis=1)
    return np.sqrt(real ** 2 + imag ** 2) / w, w


def compute_amphipath(seq: str) -> np.ndarray:
    """10 维两亲性特征（走 --amphipath 时追加在 694 维之后）。"""
    seq = clean_sequence(seq)
    h = np.array([EISENBERG_SCALE.get(ch, 0.0) for ch in seq], dtype=np.float64)
    L = h.size

    feats = []
    best_win_h = 0.0
    for w in AMPHIPATH_WINDOWS:
        mu, we = _hydrophobic_moment_profile(h, w)
        if we < 2 or mu.size == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        k = int(np.argmax(mu))
        # 峰位置归一化到 [0,1]，消掉长度量纲
        pos = k / max(mu.size - 1, 1)
        feats.extend([float(mu.max()), float(mu.mean()), float(mu.std()), float(pos)])
        if w == max(AMPHIPATH_WINDOWS):
            best_win_h = float(h[k:k + we].mean())

    # 3.6 残基周期的功率占比：把疏水性序列当信号做 FFT，看螺旋周期占多少能量
    if L >= 6:
        hc = h - h.mean()
        power = np.abs(np.fft.rfft(hc)) ** 2
        power = power[1:]                            # 去掉直流项
        freqs = np.fft.rfftfreq(L)[1:]
        if power.sum() > 0 and freqs.size and np.all(freqs > 0):
            k = int(np.argmin(np.abs(1.0 / freqs - AMPHIPATH_HELIX_PERIOD)))
            helix_ratio = float(power[k] / power.sum())
        else:
            helix_ratio = 0.0
    else:
        helix_ratio = 0.0
    feats.extend([helix_ratio, best_win_h])

    arr = np.asarray(feats, dtype=np.float32)
    if arr.size != AMPHIPATH_DIM:
        raise AssertionError('两亲性特征维度 %d ≠ %d' % (arr.size, AMPHIPATH_DIM))
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def amphipath_feature_names():
    names = []
    for w in AMPHIPATH_WINDOWS:
        names += [f"amph_mu{w}_max", f"amph_mu{w}_mean", f"amph_mu{w}_std",
                  f"amph_mu{w}_argmax"]
    names += ["amph_helix_power", "amph_best_window_hydrophobicity"]
    return names
