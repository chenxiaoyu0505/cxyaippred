import os
import json
import math
import numpy as np
import pandas as pd
from collections import Counter

# ===================== CONFIG =====================
INPUT_CSV = r"D:\BIP\data\train.csv"  # 要提取的 csv（现在是 test.csv）
SEQ_COL = "squence"  # 你的序列列名
OUT_DIR = r"/module2/hand/handtrain"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_NPY = os.path.join(OUT_DIR, "handcraft_feats.npy")
OUT_CSV = os.path.join(OUT_DIR, "handcraft_feats.csv")
OUT_MERGE = os.path.join(OUT_DIR, "test_with_handcraft.csv")
OUT_NAMES = os.path.join(OUT_DIR, "handcraft_feature_names.json")
# ================================================

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}

# ---- GAAC 5 组 ----
GAAC_GROUPS = {
    "aliphatic": set("GAVLMI"),
    "aromatic": set("FYW"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "uncharged": set("STCPNQ"),
}

# ---- CKSAAGP 6 组（基于理化性质分组，与StackPIP一致）----
CKSAAGP_GROUPS = {
    "aliphatic": set("GAVLMI"),  # 脂肪族
    "aromatic": set("FYW"),  # 芳香族
    "positive": set("KRH"),  # 正电荷
    "negative": set("DE"),  # 负电荷
    "polar": set("STNQ"),  # 极性不带电
    "special": set("CP")  # 特殊（半胱氨酸、脯氨酸）
}
CKSAAGP_NAMES = list(CKSAAGP_GROUPS.keys())
CKSAAGP_K_VALUES = [0, 1, 2, 3]  # k=0,1,2,3，覆盖短程到长程相互作用

# ---- PAAC 理化性质表（标准化后的值，Chou et al.）----
# 6种理化性质：疏水性(H1)、亲水性(H2)、侧链质量(Mass)、pK1、pK2、pI
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
PAAC_LAMBDA = 30  # PAAC的lambda参数（序列相关层数）
PAAC_WEIGHT = 0.05  # 权重因子

# ---- Kyte-Doolittle hydropathy (GRAVY)
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# ---- residue mass (Da) for MW
RES_MASS = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}
WATER_MASS = 18.01528

HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")

# ---- pI 计算（常用 pKa）
PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDE = {"C": 8.33, "D": 3.86, "E": 4.25, "H": 6.00, "K": 10.53, "R": 12.48, "Y": 10.07}
POS_SIDE = set(["H", "K", "R"])
NEG_SIDE = set(["C", "D", "E", "Y"])

# ---- CTD：3 个性质 × (3 comp + 3 trans + 15 dist) = 63
CTD_PROPERTIES = {
    "hydrophobicity": {"g1": set("RKEDQN"), "g2": set("GASTPHY"), "g3": set("CLVIMFW")},
    "polarity": {"g1": set("LIFWCMVY"), "g2": set("PATGS"), "g3": set("HQRKNED")},
    "charge": {"g1": set("KRH"), "g2": set("DE"), "g3": set("ACFGILMNPQSTVWY")},
}


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

    # N-term (+), C-term (-)
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
    """
    CKSAAGP: Composition of k-spaced Amino Acid Group Pairs
    基于6类分组的k-间隔对特征，k=0,1,2,3
    维度：6组 × 6组 × 4个k值 = 144维
    """
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    n_groups = len(CKSAAGP_NAMES)

    features = []

    for k in CKSAAGP_K_VALUES:
        # 初始化计数矩阵
        count = np.zeros((n_groups, n_groups), dtype=np.float32)

        if L > k + 1:
            denom = L - k - 1
            for i in range(L - k - 1):
                aa1, aa2 = s[i], s[i + k + 1]

                # 确定氨基酸所属组别（第一个匹配的组）
                g1 = next((idx for idx, gname in enumerate(CKSAAGP_NAMES)
                           if aa1 in CKSAAGP_GROUPS[gname]), n_groups - 1)
                g2 = next((idx for idx, gname in enumerate(CKSAAGP_NAMES)
                           if aa2 in CKSAAGP_GROUPS[gname]), n_groups - 1)

                count[g1][g2] += 1.0

            # 归一化
            count = count / denom

        features.extend(count.flatten())

    return np.array(features, dtype=np.float32)


def compute_paac(seq: str, lambda_val: int = PAAC_LAMBDA, weight: float = PAAC_WEIGHT) -> np.ndarray:
    """
    PAAC: Pseudo Amino Acid Composition (Chou et al.)
    维度：20 + lambda (通常lambda=30)
    包含：1) AAC (20维)  2) 序列顺序相关特征 (lambda维)
    """
    s = [c for c in seq if c in AA_SET]
    L = len(s)

    if L == 0:
        return np.zeros(20 + lambda_val, dtype=np.float32)

    # 1. 计算AAC (Tier 1)
    aac = np.zeros(20, dtype=np.float32)
    for i, aa in enumerate(AA20):
        aac[i] = s.count(aa) / L

    # 如果序列太短，无法计算高阶lambda，调整lambda
    actual_lambda = min(lambda_val, L - 1)
    if actual_lambda < 1:
        return np.concatenate([aac, np.zeros(lambda_val, dtype=np.float32)])

    # 2. 计算序列相关系数 theta (Tier 2)
    # theta_j = (1/(L-j)) * sum(H(i,j)^2)
    # 其中H(i,j)是第i和i+j位点的归一化理化性质距离
    theta = []

    for j in range(1, actual_lambda + 1):  # j = 1, 2, ..., lambda
        corr_sum = 0.0
        for i in range(L - j):
            aa1, aa2 = s[i], s[i + j]
            props1 = PAAC_PROPERTIES.get(aa1, [0] * 6)
            props2 = PAAC_PROPERTIES.get(aa2, [0] * 6)

            # 计算欧氏距离平方（6种性质）
            diff_sq_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(props1, props2))
            corr_sum += diff_sq_sum / 6.0

        theta_j = corr_sum / (L - j)
        theta.append(theta_j)

    # 填充剩余的lambda（如果序列太短）
    while len(theta) < lambda_val:
        theta.append(0.0)

    theta = np.array(theta, dtype=np.float32)

    # 3. 归一化因子和最终PAAC
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

    # Aliphatic index (Ikai)
    a_frac = cnt.get("A", 0) / L
    v_frac = cnt.get("V", 0) / L
    i_frac = cnt.get("I", 0) / L
    l_frac = cnt.get("L", 0) / L
    aliphatic_index = 100.0 * (a_frac + 2.9 * v_frac + 3.9 * (i_frac + l_frac))

    pI = compute_pI("".join(s))

    feats = np.array([
        length,
        mw,
        gravy,
        aromaticity,
        frac_hydrophobic,
        frac_polar,
        frac_pos,
        frac_neg,
        net_charge_approx,
        aliphatic_index,
        pI,
        math.log1p(length),
    ], dtype=np.float32)
    return feats


def _ctd_group_id(aa: str, prop_map) -> int:
    if aa in prop_map["g1"]:
        return 0
    if aa in prop_map["g2"]:
        return 1
    if aa in prop_map["g3"]:
        return 2
    return -1


def compute_ctd(seq: str) -> np.ndarray:
    s = [c for c in seq if c in AA_SET]
    L = len(s)
    if L == 0:
        return np.zeros(63, dtype=np.float32)

    feats = []
    for prop, gmap in CTD_PROPERTIES.items():
        gids = [_ctd_group_id(a, gmap) for a in s]

        # composition
        comp = np.zeros(3, dtype=np.float32)
        for g in gids:
            if g >= 0:
                comp[g] += 1.0
        comp /= L

        # transition
        trans = np.zeros(3, dtype=np.float32)
        if L >= 2:
            denom = L - 1
            for i in range(L - 1):
                a, b = gids[i], gids[i + 1]
                if a == -1 or b == -1 or a == b:
                    continue
                x, y = sorted((a, b))
                if (x, y) == (0, 1):
                    trans[0] += 1.0
                elif (x, y) == (0, 2):
                    trans[1] += 1.0
                elif (x, y) == (1, 2):
                    trans[2] += 1.0
            trans /= denom

        # distribution
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


def handcraft_feature_names():
    names = []
    # AAC (20)
    names += [f"aac_{a}" for a in AA20]
    # GAAC (5)
    names += [f"gaac_{k}" for k in GAAC_GROUPS.keys()]
    # DPC (400)
    names += [f"dpc_{a}{b}" for a in AA20 for b in AA20]
    # CKSAAGP (6x6x4=144)
    for k in CKSAAGP_K_VALUES:
        for g1 in CKSAAGP_NAMES:
            for g2 in CKSAAGP_NAMES:
                names.append(f"cksaagp_k{k}_{g1}_{g2}")
    # PAAC (20 + 30 = 50)
    names += [f"paac_tier1_{a}" for a in AA20]  # 前20是AAC部分
    names += [f"paac_tier2_theta{i + 1}" for i in range(PAAC_LAMBDA)]  # 后30是序列相关
    # physchem (12)
    names += [
        "len", "mw", "gravy", "aromaticity", "frac_hydrophobic", "frac_polar",
        "frac_pos", "frac_neg", "net_charge_approx", "aliphatic_index", "pI", "log1p_len"
    ]
    # CTD (63)
    for prop in CTD_PROPERTIES.keys():
        names += [f"ctd_{prop}_comp_g{i}" for i in [1, 2, 3]]
        names += [f"ctd_{prop}_trans_g12", f"ctd_{prop}_trans_g13", f"ctd_{prop}_trans_g23"]
        for g in [1, 2, 3]:
            for p in ["0", "25", "50", "75", "100"]:
                names.append(f"ctd_{prop}_dist_g{g}_p{p}")
    return names


def compute_handcraft(seq: str) -> np.ndarray:
    return np.concatenate(
        [
            compute_aac(seq),  # 20维
            compute_gaac(seq),  # 5维
            compute_dpc(seq),  # 400维
            compute_ksaagp(seq),  # 144维 (NEW)
            compute_paac(seq),  # 50维 (NEW: 20+30)
            compute_physchem(seq),  # 12维
            compute_ctd(seq)  # 63维
        ],
        axis=0
    ).astype(np.float32)


def main():
    df = pd.read_csv(INPUT_CSV)
    if SEQ_COL not in df.columns:
        raise ValueError(f"CSV 中找不到序列列 '{SEQ_COL}'，实际列：{list(df.columns)}")

    seqs = df[SEQ_COL].astype(str).tolist()
    seqs = [clean_sequence(s) for s in seqs]

    names = handcraft_feature_names()
    with open(OUT_NAMES, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    feats = np.stack([compute_handcraft(s) for s in seqs], axis=0)
    print("Input:", INPUT_CSV)
    print("N samples:", feats.shape[0], "Feature dim:", feats.shape[1])
    print("Feature breakdown:")
    print(f"  - AAC: 20")
    print(f"  - GAAC: 5")
    print(f"  - DPC: 400")
    print(f"  - CKSAAGP: 144 (k=0,1,2,3 × 6×6 groups)")
    print(f"  - PAAC: 50 (20 AAC + 30 sequence correlation)")
    print(f"  - Physchem: 12")
    print(f"  - CTD: 63")
    print(f"  - Total: {feats.shape[1]}")

    np.save(OUT_NPY, feats)
    print("Saved NPY ->", OUT_NPY)

    feat_df = pd.DataFrame(feats, columns=names)
    feat_df.to_csv(OUT_CSV, index=False)
    print("Saved CSV ->", OUT_CSV)

    merged = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    merged.to_csv(OUT_MERGE, index=False)
    print("Saved merged ->", OUT_MERGE)


if __name__ == "__main__":
    main()