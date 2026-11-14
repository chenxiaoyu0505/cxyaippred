# -*- coding: utf-8 -*-
"""手工特征三视角：从原始序列**重训**，并与发布会里的冻结概率逐位对账。

视角 = TF-IDF → TruncatedSVD(300) → StandardScaler，再喂 ExtraTrees：
  gap3  间隔 3-mer（窗口 4，保留 3 位，3 种放置，'X' 代表间隔）
  k4    连续 4-mer
  hand  704 维手工理化特征（发布包直接提供矩阵 `feats/hand_feats_amphipath.npy`，
        4480×704，行序 = data/table.csv）

防泄漏：TF-IDF 的 IDF、SVD 基、Scaler **只用每折 train 行拟合**，再 transform 全部
4480 行；脚本内置断言（train 行上的均值≈0、标准差≈1）。

用法：
  python scripts/run_ml_views.py --phase feats                       # 造 gap3/k4 特征（几分钟）
  python scripts/run_ml_views.py --phase probs                       # 训 ET（较慢，见 README）
  python scripts/run_ml_views.py --phase probs --folds 0             # 只跑 fold0（冒烟）
  python scripts/run_ml_views.py --phase verify                      # 与 repro/probs 对账

产物：`repro/out/feats/`、`repro/out/probs/`（与 repro/probs 同名同格式）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # repro/
SPLIT_DIR = os.path.join(ROOT, "data")
HAND704 = os.path.join(ROOT, "feats", "hand_feats_amphipath.npy")
FEAT_DIR = os.path.join(ROOT, "out", "feats")
PROB_DIR = os.path.join(ROOT, "out", "probs")
SHIPPED = os.path.join(ROOT, "probs")

SVD_DIM = 300
ET_TREES = 1500
ET_SEEDS = 3            # 与发布会口径一致（3 个种子取平均）
BASE_VIEWS = ["gap3", "k4"]
ALL_VIEWS = ["hand"] + BASE_VIEWS

# 与 scripts/run_gap3.py 一致的 3 种放置（窗口 4 保留 3 位，间隔位置标 'X'）
W1_PATTERNS = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1)]


def gap3_tokens(seq):
    toks = []
    for i in range(len(seq) - 3):
        w = seq[i:i + 4]
        for k0, k1, k2, skip in W1_PATTERNS:
            t = [w[k0], w[k1], w[k2]]
            t.insert(skip, "X")
            toks.append("".join(t))
    return toks


def k4_tokens(seq):
    return [seq[i:i + 4] for i in range(len(seq) - 3)]


TOKENIZERS = {"gap3": gap3_tokens, "k4": k4_tokens}


def load_split():
    table = pd.read_csv(os.path.join(SPLIT_DIR, "table.csv"))
    folds = pd.read_csv(os.path.join(SPLIT_DIR, "folds.csv"))
    pos = {s: i for i, s in enumerate(table["sequence"])}
    y_all = table["label"].values.astype(int)
    return table, folds, pos, y_all


def phase_feats(folds_arg, views):
    os.makedirs(FEAT_DIR, exist_ok=True)
    table, folds, pos, y_all = load_split()
    seqs = table["sequence"].tolist()
    fset = [int(x) for x in folds_arg.split(",")] if folds_arg else [0, 1, 2, 3, 4]
    for k in fset:
        f = folds[folds["fold"] == k]
        tr_seqs = f.loc[f["split"] == "train", "sequence"].tolist()
        for v in [x for x in views if x != "hand"]:
            out = os.path.join(FEAT_DIR, "fold%d_%s.npy" % (k, v))
            if os.path.exists(out):
                print("[skip] %s" % out, flush=True)
                continue
            t0 = time.time()
            tok = TOKENIZERS[v]
            vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                                  sublinear_tf=True, norm="l2", lowercase=False)
            Xtr = vec.fit_transform(" ".join(tok(s)) for s in tr_seqs)
            Xall = vec.transform(" ".join(tok(s)) for s in seqs)
            svd = TruncatedSVD(n_components=SVD_DIM, random_state=0)
            Ztr = svd.fit_transform(Xtr)
            Zall = svd.transform(Xall)
            sc = StandardScaler().fit(Ztr)
            Zall = sc.transform(Zall).astype(np.float32)
            tr_idx = np.array([pos[s] for s in tr_seqs], dtype=np.int64)
            zm, zs = Zall[tr_idx].mean(), Zall[tr_idx].std()
            assert abs(zm) < 1e-5 and abs(zs - 1.0) < 1e-3, "fold%d %s 泄漏自检失败" % (k, v)
            np.save(out, Zall)
            print("[fold%d %-6s] vocab=%d svd_var=%.4f leak(mean=%.1e,std=%.4f) %.0fs"
                  % (k, v, len(vec.vocabulary_), svd.explained_variance_ratio_.sum(),
                     zm, zs, time.time() - t0), flush=True)


def phase_probs(folds_arg, views):
    os.makedirs(PROB_DIR, exist_ok=True)
    table, folds, pos, y_all = load_split()
    hand704 = np.load(HAND704).astype(np.float32)
    print("[feats] hand704 = %s" % (hand704.shape,), flush=True)
    fset = [int(x) for x in folds_arg.split(",")] if folds_arg else [0, 1, 2, 3, 4]
    for k in fset:
        f = folds[folds["fold"] == k]
        tr_idx = np.array([pos[s] for s in f.loc[f["split"] == "train", "sequence"]], dtype=np.int64)
        va_idx = np.array([pos[s] for s in f.loc[f["split"] == "val", "sequence"]], dtype=np.int64)
        # folds.csv 中 897 条 test 在每一折下都重复出现，必须按折过滤
        te_idx = np.array([pos[s] for s in f.loc[f["split"] == "test", "sequence"]], dtype=np.int64)
        ytr = y_all[tr_idx]
        for v in views:
            pv = os.path.join(PROB_DIR, "fold%d_%s_val.npy" % (k, v))
            pt = os.path.join(PROB_DIR, "fold%d_%s_test.npy" % (k, v))
            if os.path.exists(pv) and os.path.exists(pt):
                print("[skip] fold%d %s" % (k, v), flush=True)
                continue
            t0 = time.time()
            X = hand704 if v == "hand" else np.load(os.path.join(FEAT_DIR, "fold%d_%s.npy" % (k, v)))
            pvs, pts = [], []
            for seed in range(ET_SEEDS):
                et = ExtraTreesClassifier(n_estimators=ET_TREES, max_features=0.1,
                                          min_samples_leaf=1, class_weight="balanced",
                                          random_state=seed, n_jobs=-1)
                et.fit(X[tr_idx], ytr)
                pvs.append(et.predict_proba(X[va_idx])[:, 1])
                pts.append(et.predict_proba(X[te_idx])[:, 1])
            np.save(pv, np.mean(pvs, axis=0).astype(np.float32))
            np.save(pt, np.mean(pts, axis=0).astype(np.float32))
            print("[fold%d %-6s] %.0fs" % (k, v, time.time() - t0), flush=True)


def phase_verify(views):
    """把 out/probs 与 repro/probs 逐文件对比，报告 max|Δ|。

    发布包里 `*_test.npy` 存的是 4485 = 5×897 行（该折把 5 折的 test 行都列了一遍，
    融合时只取前 897 行，见 run_4view.py），重训只按该折自己的 897 行写盘，
    因此按前缀对齐比较。
    """
    worst, bad = 0.0, []
    for k in range(5):
        for v in views:
            for kind in ("val", "test"):
                a = os.path.join(PROB_DIR, "fold%d_%s_%s.npy" % (k, v, kind))
                b = os.path.join(SHIPPED, "fold%d_%s_%s.npy" % (k, v, kind))
                if not os.path.exists(a):
                    print("[miss] %s（先跑 --phase probs）" % a)
                    continue
                A, B = np.load(a), np.load(b)
                n = min(len(A), len(B))
                d = float(np.abs(A[:n] - B[:n]).max())
                worst = max(worst, d)
                flag = "OK " if d < 1e-5 else "DIFF"
                if d >= 1e-5:
                    bad.append(os.path.basename(a))
                extra = "" if len(A) == len(B) else "  (对齐前 %d 行，发布 %d / 重训 %d)" % (n, len(B), len(A))
                print("  %s fold%d %-6s %-4s max|Δp|=%.2e%s" % (flag, k, v, kind, d, extra))
    print("\n[verify] 最大偏差 %.2e %s" % (worst, "（一致）" if not bad else "→ 不一致：%s" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["feats", "probs", "verify"])
    ap.add_argument("--folds", default="", help="逗号分隔折号，默认全部 5 折")
    ap.add_argument("--views", default="hand,gap3,k4",
                    help="逗号分隔视角，默认 hand,gap3,k4")
    args = ap.parse_args()
    views = [x.strip() for x in args.views.split(",") if x.strip()]
    if args.phase == "feats":
        phase_feats(args.folds, views)
    elif args.phase == "probs":
        phase_probs(args.folds, views)
    else:
        return phase_verify(views)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
