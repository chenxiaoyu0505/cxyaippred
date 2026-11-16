# -*- coding: utf-8 -*-
"""ext261：把已训好的 train1/train0 流水线（5 视角）套到新外部集 261 行（清洗后 258）。

口径（逐字对齐 _work/wt-drop385/scripts/eval_grid.py + ext5_t1/run_ml_ext2.py）：
  * 折 = data/splits/deepaip_orig/folds.csv 官方 5 折（fold_of_pool[seq]=k where split=='val'）；
    PIP 新行按长度箱轮转（eval_grid.assign_folds 等价）。
  * gap3_tokens：4 残基窗口取 3 位、'X' 插在 skip 位（token 4 字符）；k4_tokens：连续 4-mer。
  * fold_view：TF-IDF(analyzer=word, token_pattern=\\S+, sublinear_tf, norm=l2) 只在 fit_idx 上
    fit -> TruncatedSVD(300, random_state=0).fit(Z_fit) -> StandardScaler().fit(E_fit) -> transform。
  * ET：n_estimators=1500 / max_features=.1 / min_samples_leaf=1 / class_weight=balanced /
        n_jobs=-1 / random_state=s (s=0,1,2)，3 seed 取均值；每折 apply=[fold-train, fold-val,
        test0, ext261] -> 概率 = 5 折均值。

模型来源（**不训练任何新模型**，ET 三视角是"复原基线"，因为当时没存 ET 权重）：
  * hand704 / gap3 / k4：在 train0（3,583 行）上 5 折重拟合 ET（逐折表示拟合），约 4 分钟。
  * TLoRA：直接推理 _work/wt-tlora5/_work/tlora5/{k}_best.pt（train0 口径 5 折权重）+ {k}_init.npz，
           ESM-2 t33 650M fp16 + posproj-adaptive 头，5 折概率均值。
  * 小 CNN：直接推理 _work/wt-drop385/ext_cnn/weights_A.pt（train1 口径 5 折 state_dict），5 折均值。

融合：rank 平均（rankdata/n），权重不在外部集上重选：
  * fuse_t0   = hand 225/729, gap3 90/729, k4 90/729, TLoRA 243/729, CNN 81/729
                （ET 三视角按 train0 贪心 15:6:6 保持比例缩到 15/27 份额；DL 沿用 view5_t1 的 9/27、3/27）
  * fuse_v5t1 = hand 6/27, gap3 4/27, k4 5/27, TLoRA 9/27, CNN 3/27（view5_t1 原权重，供与 ext193 对齐）
  * fuse_equal= 各 0.2
阈值 τ：只用 OOF 定（MCC 最优），外部集完全不参与拟合/调参。

产物（_work/ext261/）：data_clean.csv / data_dropped.csv / probs.csv / metrics.json / table.md
用法：
  python _work/ext261/run_ext261.py                 # 全跑
  python _work/ext261/run_ext261.py --skip-ml       # 跳过 ET 重拟合（复用 _stage_ml.npz）
"""
import os
import sys
import json
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: F401  （纪律：torch 先于 numpy/sklearn，Windows DLL 顺序）

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ------------------------------------------------------------------ 路径/常量
REPO = r"D:\BIP\module4\cxyaippred"
WORK = os.path.join(REPO, "_work")
OUT = os.path.join(WORK, "ext261")
WT = os.path.join(WORK, "wt-drop385")
SCRIPTS = os.path.join(WT, "scripts")
SPLIT = os.path.join(REPO, "data", "splits", "deepaip_orig")
GRID = os.path.join(WORK, "grid")
XLSX = r"C:\Users\cxy\Desktop\外部数据集_1114.xlsx"
SHEET = "长表_序列标签"
TLORA_DIR = os.path.join(WORK, "wt-tlora5", "_work", "tlora5")
CNN_W = os.path.join(WT, "ext_cnn", "weights_A.pt")
LOCK = os.path.join(WORK, "gpu.lock")

ET_KW = dict(n_estimators=1500, max_features=0.1, min_samples_leaf=1,
             class_weight="balanced", n_jobs=-1)
NSEED, NFOLD = 3, 5
VIEWS = ["hand", "gap3", "k4", "tlora", "cnn"]
ML_VIEWS = ["hand", "gap3", "k4"]

W_T0 = {"hand": 225 / 729, "gap3": 90 / 729, "k4": 90 / 729,
        "tlora": 243 / 729, "cnn": 81 / 729}
W_V5T1 = {"hand": 6 / 27, "gap3": 4 / 27, "k4": 5 / 27,
          "tlora": 9 / 27, "cnn": 3 / 27}
W_EQUAL = {v: 1.0 / len(VIEWS) for v in VIEWS}

LOG = os.path.join(OUT, "PROGRESS.log")
TIME0 = time.time()


def plog(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - TIME0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def jdump(obj, path):
    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"not serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_d)


# ------------------------------------------------------------------ 外部集清洗
DROP_REASON = {
    "cyclo-[D-Pro-D-Pro-D-β3-HoPhe-D-Phe-]":
        "drop: 环肽 + D 型/β3 非标准残基，无法转单字母",
    "H-Pro-Ile-Asn-Tyr-Ala-Nva-Ser-Phe-Leu-Ser-NH2":
        "drop: Nva（正缬氨酸）非标准残基，无法转单字母",
    "N-formyl-Met-Leu-Phe":
        "drop: N 端甲酰化（fMLP），N 端非标准",
}
MAP3 = {
    "Trp-Lys-Tyr-Met-Val-Met": ("WKYMVM", "keep: 三字母序列 -> WKYMVM（线性标准残基）"),
}
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_external():
    raw = pd.read_excel(XLSX, sheet_name=SHEET)
    raw.columns = [str(c).strip() for c in raw.columns]
    assert {"sequence", "label", "label_name"} <= set(raw.columns), raw.columns
    raw["sequence"] = raw["sequence"].astype(str).str.strip()
    raw["label"] = raw["label"].astype(int)
    plog(f"[ext] 原始 {len(raw)} 行 = {int(raw.label.sum())} 阳 / {int((raw.label == 0).sum())} 阴")

    kept, dropped = [], []
    for i, r in raw.iterrows():
        s = r["sequence"]
        rec = dict(orig_row=int(i), sequence_raw=s, label=int(r["label"]),
                   label_name=str(r["label_name"]))
        if s in DROP_REASON:
            dropped.append(dict(rec, reason=DROP_REASON[s]))
            continue
        if s in MAP3:
            seq, reason = MAP3[s]
        else:
            seq, reason = s, "keep: 已是单字母标准序列"
        ok = bool(seq) and set(seq) <= AA20
        assert ok, f"保留行含非标准字符: {s!r} -> {seq!r}"
        kept.append(dict(rec, sequence=seq, length=len(seq), keep=1, reason=reason))

    kd = pd.DataFrame(kept)[["orig_row", "sequence_raw", "sequence", "label", "label_name",
                             "length", "keep", "reason"]]
    dd = pd.DataFrame(dropped)
    assert len(kd) == 258, len(kd)
    assert int(kd.label.sum()) == 211 and int((kd.label == 0).sum()) == 47
    assert len(dd) == 3, len(dd)
    assert kd.sequence.is_unique
    assert not set(kd.sequence) & set(pd.read_csv(os.path.join(SPLIT, "table.csv"))
                                     .sequence.str.strip().str.upper())
    os.makedirs(OUT, exist_ok=True)
    kd.to_csv(os.path.join(OUT, "data_clean.csv"), index=False, encoding="utf-8")
    dd.to_csv(os.path.join(OUT, "data_dropped.csv"), index=False, encoding="utf-8")
    plog(f"[ext] 保留 {len(kd)} 行 = {int(kd.label.sum())} 阳 / {int((kd.label == 0).sum())} 阴；"
         f"剔除 {len(dd)} 行；长度 {kd.length.min()}-{kd.length.max()} "
         f"(均值 {kd.length.mean():.2f}/中位 {kd.length.median():.0f})")
    return kd


# ------------------------------------------------------------------ 特征（eval_grid 逐字）
def gap3_tokens(seq):
    out = []
    for i in range(len(seq) - 3):
        w = seq[i:i + 4]
        for k0, k1, k2, skip in [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1)]:
            t = [w[k0], w[k1], w[k2]]
            t.insert(skip, "X")
            out.append("".join(t))
    return out


def k4_tokens(seq):
    return [seq[i:i + 4] for i in range(len(seq) - 3)]


TOKENIZERS = {"gap3": gap3_tokens, "k4": k4_tokens}


def fold_view(kind, Xh, docs, fit_idx, apply_idx):
    """逐折表示拟合：SVD/scaler 只用 fit_idx。hand 直接取列（ET 对单调变换不变）。"""
    if kind == "hand":
        return Xh[apply_idx], 0
    tf = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                         norm="l2", lowercase=False)
    tf.fit([docs[kind][i] for i in fit_idx])
    Z_fit = tf.transform([docs[kind][i] for i in fit_idx])
    svd = TruncatedSVD(n_components=300, random_state=0)
    svd.fit(Z_fit)
    E_fit = svd.transform(Z_fit).astype(np.float32)
    sc = StandardScaler().fit(E_fit)
    E_ap = svd.transform(tf.transform([docs[kind][i] for i in apply_idx])).astype(np.float32)
    return sc.transform(E_ap).astype(np.float32), tf.get_feature_names_out().shape[0]


# ------------------------------------------------------------------ 指标
def rank01(x):
    return rankdata(x) / len(x)


def mcc_stats(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    sn = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sn / max(prec + sn, 1e-12)
    acc = (tp + tn) / n
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    return dict(tau=float(t), mcc=float(mcc), sn=float(sn), sp=float(sp),
                acc=float(acc), prec=float(prec), f1=float(f1),
                ba=float((sn + sp) / 2), posrate=float(pred.mean()),
                tp=tp, tn=tn, fp=fp, fn=fn)


def tau_mcc_opt(y, p, lo=0.01, hi=0.99, step=0.001):
    """MCC 最优阈值（eval_grid.tau_from_scores 同口径）。"""
    best = (-2.0, None)
    for t in np.arange(lo, hi + 1e-12, step):
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]


def sn_at_sp(y, p, target=0.80):
    fpr, tpr, thr = roc_curve(y, p)
    sp = 1 - fpr
    ok = sp >= target
    if not ok.any():
        return None
    i = np.where(ok)[0][np.argmax(tpr[ok])]
    return dict(target=target, sn=float(tpr[i]), sp=float(sp[i]), tau=float(thr[i]))


def prec_at_k(y, p, k=50):
    idx = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(np.asarray(y)[idx].mean())


def boot_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if y[s].min() == y[s].max():
            continue
        d.append(roc_auc_score(y[s], p[s]))
    d = np.asarray(d)
    return dict(auc_mean=float(d.mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                half_width=float((np.percentile(d, 97.5) - np.percentile(d, 2.5)) / 2),
                n_boot=n_boot, n_used=int(len(d)), seed=seed)


LEN_BINS = [("<8", lambda L: L < 8),
            ("8-10", lambda L: (L >= 8) & (L <= 10)),
            ("11-14", lambda L: (L >= 11) & (L <= 14)),
            ("15", lambda L: L == 15),
            ("16-18", lambda L: (L >= 16) & (L <= 18)),
            (">=19", lambda L: L >= 19)]
BIN_NAMES = [b for b, _ in LEN_BINS]


def by_len(y, p, L):
    out = {}
    for nm, f in LEN_BINS:
        m = f(L)
        out[nm] = dict(n=int(m.sum()), npos=int(y[m].sum()),
                       auroc=(round(float(roc_auc_score(y[m], p[m])), 4)
                              if m.sum() > 1 and len(np.unique(y[m])) > 1 else None))
    return out


def full_metrics(y, p, L, tau_oof, tau_ext=None, n_boot=2000):
    posrate = float(np.mean(y))
    aupr = float(average_precision_score(y, p))
    out = dict(
        n=int(len(y)), npos=int(y.sum()), posrate=round(posrate, 4),
        auroc=round(float(roc_auc_score(y, p)), 4),
        aupr=round(aupr, 4),
        lift_aupr=round(aupr / posrate, 4),
        boot=boot_ci(y, p, n_boot=n_boot),
        bins=by_len(y, p, L),
        prec50=round(prec_at_k(y, p, 50), 4),
        lift50=round(prec_at_k(y, p, 50) / posrate, 4),
        sn_at_sp80=sn_at_sp(y, p, 0.80),
    )
    out["at_tau_oof"] = mcc_stats(y, p, tau_oof)
    out["at_0.5"] = mcc_stats(y, p, 0.5)
    te, me = tau_mcc_opt(y, p)
    out["tau_ext_mcc_opt"] = te
    out["at_tau_ext"] = mcc_stats(y, p, te) if te is not None else None
    if tau_ext is not None:
        out["note_tau_ext"] = "tau_ext = 外部集 MCC 最优（oracle，仅作上界参考）"
    return out


# ------------------------------------------------------------------ ET 三视角（train0，5 折）
def stage_ml(ext_df):
    sys.path.insert(0, WT)
    from src.data_io import hand_feats

    tab = pd.read_csv(os.path.join(SPLIT, "table.csv"))
    tab["seq"] = tab.sequence.str.strip().str.upper()
    folds_csv = pd.read_csv(os.path.join(SPLIT, "folds.csv"))
    fold_of_pool = {}
    for k in range(NFOLD):
        v = folds_csv[(folds_csv.fold == k) & (folds_csv.split == "val")]
        for s in v.sequence:
            fold_of_pool[str(s).strip().upper()] = k
    # folds.csv 的 val 行只覆盖 3,583 条 CV 池（=train0）；test0 的 897 条没有折号
    assert len(fold_of_pool) == 3583, len(fold_of_pool)
    plog(f"[ml] universe: table {len(tab)}（CV 池 3,583 + 官方 test 897）+ ext261 {len(ext_df)}")

    hand_tab = np.load(os.path.join(REPO, "cache_deepaip", "hand_feats_amphipath.npy"))
    assert hand_tab.shape == (len(tab), 704), hand_tab.shape
    hand_ext = hand_feats(pd.DataFrame({"sequence": ext_df.sequence.tolist()}),
                          cache_dir=os.path.join(OUT, "_handcache"), with_amphipath=True)
    assert hand_ext.shape == (len(ext_df), 704), hand_ext.shape
    Xh = np.vstack([hand_tab, hand_ext]).astype(np.float32)

    seq = np.concatenate([tab.seq.to_numpy(), ext_df.sequence.to_numpy()])
    assert len(seq) == len(Xh)
    n_ext = len(ext_df)
    ex_uni = np.arange(len(seq) - n_ext, len(seq))

    fold_of = np.full(len(seq), -1, int)
    for i, s in enumerate(seq):
        fold_of[i] = fold_of_pool.get(s, -1)
    cnt = {}
    for i in range(len(seq) - n_ext):
        if fold_of[i] >= 0:
            continue
        b = min(len(seq[i]), 30)
        c = cnt.get(b, 0)
        fold_of[i] = c % NFOLD
        cnt[b] = c + 1
    assert (fold_of[ex_uni] == -1).all()

    docs = {kind: [" ".join(tok(s)) for s in seq] for kind, tok in TOKENIZERS.items()}

    tr = pd.read_csv(os.path.join(GRID, "train0.csv"))
    te = pd.read_csv(os.path.join(GRID, "test0.csv"))
    tr_seq = tr.sequence.str.strip().str.upper().to_list()
    te_seq = te.sequence.str.strip().str.upper().to_list()
    pos = {s: i for i, s in enumerate(seq)}
    tr_uni = np.array([pos[s] for s in tr_seq])
    te_uni = np.array([pos[s] for s in te_seq])
    y_tr, y_te = tr.label.to_numpy(), te.label.to_numpy()
    assert len(tr) == 3583 and int(y_tr.sum()) == 1365, (len(tr), int(y_tr.sum()))
    assert len(te) == 897 and int(y_te.sum()) == 342
    y_ext, L_ext = ext_df.label.to_numpy(), ext_df.length.to_numpy()

    folds = [(np.where(fold_of[tr_uni] != k)[0], np.where(fold_of[tr_uni] == k)[0])
             for k in range(NFOLD)]
    plog("[ml] folds " + " ".join(f"f{k}:fit{len(a)}/val{len(b)}" for k, (a, b) in enumerate(folds)))

    oof = {k: np.zeros(len(tr)) for k in ML_VIEWS}
    te_folds = {k: [] for k in ML_VIEWS}
    ex_folds = {k: [] for k in ML_VIEWS}
    per_fold = {k: {"test0": [], "ext": []} for k in ML_VIEWS}

    for k in range(NFOLD):
        tr_i, va_i = folds[k]
        n_f, n_v = len(tr_i), len(va_i)
        apply_idx = np.concatenate([tr_uni[tr_i], tr_uni[va_i], te_uni, ex_uni])
        for kind in ML_VIEWS:
            t1 = time.time()
            Xa, nvocab = fold_view(kind, Xh, docs, tr_uni[tr_i], apply_idx)
            Xf = Xa[:n_f]
            Xv = Xa[n_f:n_f + n_v]
            Xt = Xa[n_f + n_v:n_f + n_v + len(te)]
            Xe = Xa[n_f + n_v + len(te):]
            pv, pt, pe = [], [], []
            for s in range(NSEED):
                et = ExtraTreesClassifier(random_state=s, **ET_KW)
                et.fit(Xf, y_tr[tr_i])
                pv.append(et.predict_proba(Xv)[:, 1])
                pt.append(et.predict_proba(Xt)[:, 1])
                pe.append(et.predict_proba(Xe)[:, 1])
            oof[kind][va_i] = np.mean(pv, axis=0)
            p_t, p_e = np.mean(pt, axis=0), np.mean(pe, axis=0)
            te_folds[kind].append(p_t)
            ex_folds[kind].append(p_e)
            per_fold[kind]["test0"].append(round(float(roc_auc_score(y_te, p_t)), 4))
            per_fold[kind]["ext"].append(round(float(roc_auc_score(y_ext, p_e)), 4))
            plog(f"  [ml] fold{k} {kind:<4} n_fit={n_f} vocab={nvocab} "
                 f"test0={per_fold[kind]['test0'][-1]:.4f} ext258={per_fold[kind]['ext'][-1]:.4f} "
                 f"({time.time() - t1:.0f}s)")

    test = {k: np.mean(te_folds[k], axis=0) for k in ML_VIEWS}
    ext_p = {k: np.mean(ex_folds[k], axis=0) for k in ML_VIEWS}

    # 保真校验：重算的 test0/oof 必须与 grid_out/train0__test0.npz 逐元素一致
    z = np.load(os.path.join(WT, "grid_out", "train0__test0.npz"), allow_pickle=True)
    fid = {}
    ok_all = True
    if list(z["te_seq"]) == te_seq:
        for k in ML_VIEWS:
            dt = float(np.max(np.abs(test[k] - np.asarray(z[f"test_{k}"], float))))
            do = float(np.max(np.abs(oof[k] - np.asarray(z[f"oof_{k}"], float))))
            fid[k] = dict(test_max_abs_d=dt, oof_max_abs_d=do,
                          ref_test0_auroc=round(float(roc_auc_score(y_te, z[f"test_{k}"])), 4),
                          now_test0_auroc=round(float(roc_auc_score(y_te, test[k])), 4),
                          ref_oof_auroc=round(float(roc_auc_score(y_tr, z[f"oof_{k}"])), 4),
                          now_oof_auroc=round(float(roc_auc_score(y_tr, oof[k])), 4))
            ok_all &= (dt < 1e-9 and do < 1e-9)
            plog(f"[ml][fidelity] {k:<4} test0 max|d|={dt:.3e} oof max|d|={do:.3e} "
                 f"auroc ref {fid[k]['ref_test0_auroc']} vs now {fid[k]['now_test0_auroc']}")
    else:
        ok_all = False
        plog("[ml][fidelity] te_seq 与 npz 不一致，跳过逐元素校验")

    np.savez(os.path.join(OUT, "_stage_ml.npz"),
             tr_seq=np.array(tr_seq), y_tr=y_tr,
             te_seq=np.array(te_seq), y_te=y_te, te_length=te.length.to_numpy(),
             ex_seq=ext_df.sequence.to_numpy(), y_ext=y_ext, L_ext=L_ext,
             **{f"oof_{k}": oof[k] for k in ML_VIEWS},
             **{f"test_{k}": test[k] for k in ML_VIEWS},
             **{f"ext_{k}": ext_p[k] for k in ML_VIEWS})
    res = dict(fidelity=fid, fidelity_exact=bool(ok_all),
               per_fold={k: per_fold[k] for k in ML_VIEWS},
               oof_auroc={k: round(float(roc_auc_score(y_tr, oof[k])), 4) for k in ML_VIEWS},
               test0_auroc={k: round(float(roc_auc_score(y_te, test[k])), 4) for k in ML_VIEWS},
               ext_auroc={k: round(float(roc_auc_score(y_ext, ext_p[k])), 4) for k in ML_VIEWS})
    plog(f"[ml] 完成 oof_auroc={res['oof_auroc']} test0={res['test0_auroc']} "
         f"ext={res['ext_auroc']}")
    return oof, ext_p, res


# ------------------------------------------------------------------ GPU 锁
def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            out = os.popen(f'tasklist /FI "PID eq {pid}" /NH').read()
            return str(pid) in out
        except Exception:
            return True


def acquire_lock(wait_min=720, poll=30):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    t0 = time.time()
    warned = False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n")
            plog(f"[gpu] 取得锁 (pid={os.getpid()})")
            return
        except FileExistsError:
            try:
                txt = open(LOCK, encoding="utf-8").read().strip()
                pid = int(txt.splitlines()[0]) if txt else -1
            except Exception:
                pid = -1
            if pid > 0 and not _pid_alive(pid):
                plog(f"[gpu] 锁内 pid={pid} 已消失，接管（stale）")
                try:
                    os.remove(LOCK)
                except OSError:
                    pass
                continue
            if not warned:
                plog(f"[gpu] 等锁（holder pid={pid}），每 {poll}s 重试，上限 {wait_min}min")
                warned = True
            if (time.time() - t0) > wait_min * 60:
                raise SystemExit(f"[gpu] 等锁超过 {wait_min}min")
            time.sleep(poll)


def release_lock():
    try:
        if os.path.exists(LOCK):
            txt = open(LOCK, encoding="utf-8").read().strip()
            if txt and int(txt.splitlines()[0]) == os.getpid():
                os.remove(LOCK)
                plog("[gpu] 已释放锁")
    except Exception as e:
        plog(f"[gpu] 释放锁异常: {e}")


# ------------------------------------------------------------------ TLoRA（train0 5 折权重，仅推理）
def stage_tlora(ext_df, val_pool):
    sys.path.insert(0, os.path.join(WORK, "wt-fullfit", "scripts"))
    import tlora_lib as L
    from transformers import AutoModel, AutoTokenizer

    plog("[tlora] 载入 ESM-2 t33 650M（fp16，离线）")
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                    torch_dtype=torch.float16).to(L.DEV)
    esm.eval()
    esm.requires_grad_(False)
    plog(f"[tlora] esm dtype={next(esm.parameters()).dtype} qkv={len(L.qkv_linears(esm))}")

    Xex = ext_df.sequence.tolist()
    y_ex = ext_df.label.to_numpy()
    probs_ex, per_fold = [], []
    oof = {}
    for k in range(NFOLD):
        bp = os.path.join(TLORA_DIR, f"{k}_best.pt")
        L.unwrap_qkv(esm)
        model = L.ESM_LoRA_PosProj(esm, rank=8, alpha=16).to(L.DEV)
        meta = L.peft_io.load_delta(bp, model, verbose=False)
        assert len(L.adapter_modules(model)) == 99

        # 保真校验：该折 val_probs.csv 全量重算
        vp = val_pool[k]
        _, _, logit_v = L.evaluate(model, tok, vp.sequence.tolist(), vp.label.to_numpy(), bs=32)
        p_v = 1 / (1 + np.exp(-logit_v))
        dv = float(np.max(np.abs(p_v - vp.prob.to_numpy())))
        for s, p in zip(vp.sequence.str.strip().str.upper(), p_v):
            oof[s] = float(p)

        _, _, logit_e = L.evaluate(model, tok, Xex, y_ex, bs=32)
        p_e = 1 / (1 + np.exp(-logit_e))
        probs_ex.append(p_e)
        per_fold.append(dict(fold=k, best_ep=meta.get("epoch"), val_auroc=meta.get("val_auroc"),
                             val_repro_max_abs_d=dv,
                             ext_auroc=round(float(roc_auc_score(y_ex, p_e)), 4)))
        plog(f"  [tlora] fold{k} ep{meta.get('epoch')} val_repro max|d|={dv:.3e} "
             f"ext258 AUROC={per_fold[-1]['ext_auroc']:.4f}")
        del model
        torch.cuda.empty_cache()

    L.unwrap_qkv(esm)
    del esm
    torch.cuda.empty_cache()

    ext_p = np.mean(probs_ex, axis=0)
    res = dict(per_fold=per_fold,
               val_repro_max_abs_d_max=float(max(p["val_repro_max_abs_d"] for p in per_fold)),
               ext_auroc=round(float(roc_auc_score(y_ex, ext_p)), 4),
               ext_aupr=round(float(average_precision_score(y_ex, ext_p)), 4),
               ext_foldwise_mean=round(float(np.mean([p["ext_auroc"] for p in per_fold])), 4),
               n_checkpoints=len(probs_ex))
    plog(f"[tlora] ext258 AUROC={res['ext_auroc']:.4f} "
         f"(逐折均值 {res['ext_foldwise_mean']:.4f}, 复现 max|d|={res['val_repro_max_abs_d_max']:.3e})")
    return oof, ext_p, res


# ------------------------------------------------------------------ 小 CNN（train1 权重，仅推理）
def stage_cnn(ext_df):
    sys.path.insert(0, SCRIPTS)
    import grid_cnn as G

    g = G.load_grid()
    assert int(sum(p.numel() for p in G.CNN(d=64).parameters())) == G.N_PARAMS_EXPECTED
    states = torch.load(CNN_W, map_location="cpu", weights_only=False)
    assert sorted(states.keys()) == [f"fold{i}" for i in range(NFOLD)], list(states.keys())

    tr1 = g["train1"]
    fold_pool = G.official_fold_of_pool()
    fold_of1 = G.build_fold_of(tr1.sequence.tolist(), fold_pool, G.pip_new_order(fold_pool))
    X1, M1 = G.onehot(tr1.sequence.tolist())
    y1 = tr1.label.to_numpy().astype(np.int64)

    Xex, Mex = G.onehot(ext_df.sequence.tolist())
    y_ex = ext_df.label.to_numpy()

    oof = np.zeros(len(tr1))
    probs_ex, per_fold = [], []
    for k in range(NFOLD):
        model = G.CNN(d=64)
        model.load_state_dict(states[f"fold{k}"])
        va = np.where(fold_of1 == k)[0]
        oof[va] = G.predict(model, X1[va], M1[va])
        p_e = G.predict(model, Xex, Mex)
        probs_ex.append(p_e)
        per_fold.append(dict(fold=k, n_val=int(len(va)),
                             val_auroc=round(float(roc_auc_score(y1[va], oof[va])), 4),
                             ext_auroc=round(float(roc_auc_score(y_ex, p_e)), 4)))
        plog(f"  [cnn] fold{k} val_auroc={per_fold[-1]['val_auroc']:.4f} "
             f"ext258 AUROC={per_fold[-1]['ext_auroc']:.4f}")

    ext_p = np.mean(probs_ex, axis=0)
    oof_map = {s: float(v) for s, v in zip(tr1.sequence.str.strip().str.upper(), oof)}
    res = dict(per_fold=per_fold,
               oof_auroc=round(float(roc_auc_score(y1, oof)), 4),
               ext_auroc=round(float(roc_auc_score(y_ex, ext_p)), 4),
               ext_aupr=round(float(average_precision_score(y_ex, ext_p)), 4),
               ext_foldwise_mean=round(float(np.mean([p["ext_auroc"] for p in per_fold])), 4),
               n_checkpoints=len(probs_ex))
    plog(f"[cnn] ext258 AUROC={res['ext_auroc']:.4f} (OOF {res['oof_auroc']:.4f}, "
         f"逐折均值 {res['ext_foldwise_mean']:.4f})")
    return oof_map, ext_p, res


# ------------------------------------------------------------------ 融合 + τ
def platt_fit_tau(y, scores):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(scores, float).reshape(-1, 1), y)
    p_oof = lr.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]
    tau, mcc = tau_mcc_opt(y, p_oof)
    return lr, float(tau), float(mcc), p_oof


def platt_apply(lr, scores):
    return lr.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    args = set(sys.argv[1:])
    plog("=== ext261：train0/train1 五视角流水线 -> 外部 261 行（清洗后 258）===")

    ext_df = clean_external()
    ext_seq = ext_df.sequence.tolist()
    y_ex = ext_df.label.to_numpy()
    L_ex = ext_df.length.to_numpy()

    ml_path = os.path.join(OUT, "_stage_ml.npz")
    if "--skip-ml" in args and os.path.exists(ml_path):
        z = np.load(ml_path, allow_pickle=True)
        assert list(z["ex_seq"]) == ext_seq
        oof_ml = {k: np.asarray(z[f"oof_{k}"], float) for k in ML_VIEWS}
        ext_ml = {k: np.asarray(z[f"ext_{k}"], float) for k in ML_VIEWS}
        y_tr = np.asarray(z["y_tr"])
        tr_seq = [str(s) for s in z["tr_seq"]]
        ml_res = dict(note="复用 _stage_ml.npz（--skip-ml）")
        plog("[ml] 复用缓存")
    else:
        oof_ml, ext_ml, ml_res = stage_ml(ext_df)
        z = np.load(ml_path, allow_pickle=True)
        y_tr = np.asarray(z["y_tr"])
        tr_seq = [str(s) for s in z["tr_seq"]]

    # TLoRA 需要该折 val_probs.csv 做保真校验，也提供 train0 池的 OOF
    val_pool = {}
    for k in range(NFOLD):
        vp = pd.read_csv(os.path.join(TLORA_DIR, f"{k}_val_probs.csv"))
        vp["sequence"] = vp.sequence.astype(str).str.strip().str.upper()
        val_pool[k] = vp
    tlora_val_seq = set()
    for vp in val_pool.values():
        tlora_val_seq |= set(vp.sequence)
    assert len(tlora_val_seq) == len(tr_seq), (len(tlora_val_seq), len(tr_seq))

    acquire_lock()
    try:
        oof_tlora_map, ext_tlora, tlora_res = stage_tlora(ext_df, val_pool)
        oof_cnn_map, ext_cnn, cnn_res = stage_cnn(ext_df)
    finally:
        release_lock()

    oof_tlora = np.array([oof_tlora_map[s] for s in tr_seq])
    ext_p = dict(ext_ml)
    ext_p["tlora"] = ext_tlora
    ext_p["cnn"] = ext_cnn

    # ---- 公共 OOF 池：train0 ∩ train1 = train1（CNN 只有 train1 折权重）
    tr1 = pd.read_csv(os.path.join(GRID, "train1.csv"))
    tr1_seq = tr1.sequence.str.strip().str.upper().to_list()
    pos = {s: i for i, s in enumerate(tr_seq)}
    t1_idx = np.array([pos[s] for s in tr1_seq])
    y_t1 = tr1.label.to_numpy()
    oof_stack = {k: oof_ml[k][t1_idx] for k in ML_VIEWS}
    oof_stack["tlora"] = oof_tlora[t1_idx]
    oof_stack["cnn"] = np.array([oof_cnn_map[s] for s in tr1_seq])
    n_oof = len(y_t1)
    plog(f"[oof] 公共池 n={n_oof} pos={int(y_t1.sum())}；逐视角 OOF AUROC="
         + " ".join(f"{k}:{roc_auc_score(y_t1, oof_stack[k]):.4f}" for k in VIEWS))

    # ---- 融合（rank 平均），τ 只用 OOF
    r_oof = {k: rank01(oof_stack[k]) for k in VIEWS}
    r_ext = {k: rank01(ext_p[k]) for k in VIEWS}

    fusion = {}
    for name, w in (("fuse_t0", W_T0), ("fuse_v5t1", W_V5T1), ("fuse_equal", W_EQUAL)):
        s_oof = sum(w[k] * r_oof[k] for k in VIEWS)
        s_ext = sum(w[k] * r_ext[k] for k in VIEWS)
        lr, tau, mcc_oof, p_oof = platt_fit_tau(y_t1, s_oof)
        p_ext = platt_apply(lr, s_ext)
        fusion[name] = dict(weights={k: round(w[k], 6) for k in VIEWS},
                            tau=tau, oof_mcc=mcc_oof,
                            oof_auroc=round(float(roc_auc_score(y_t1, s_oof)), 4),
                            fuse_oof=s_oof, fuse_ext=s_ext,
                            prob_ext=p_ext, prob_oof=p_oof, lr=lr)
        plog(f"[fusion] {name} τ={tau:.3f} OOF_AUROC={fusion[name]['oof_auroc']:.4f} "
             f"OOF_MCC={mcc_oof:.4f}")

    # ---- 逐视角 τ（该视角自己的 OOF）
    per_view_tau = {}
    for k in VIEWS:
        t, m = tau_mcc_opt(y_t1, oof_stack[k])
        per_view_tau[k] = dict(tau=float(t), oof_mcc=float(m),
                               oof_auroc=round(float(roc_auc_score(y_t1, oof_stack[k])), 4))
    plog("[tau] 逐视角 " + " ".join(f"{k}:{v['tau']:.3f}" for k, v in per_view_tau.items()))

    # ---- 指标
    metrics = {}
    for k in VIEWS:
        metrics[k] = full_metrics(y_ex, ext_p[k], L_ex, per_view_tau[k]["tau"])
        plog(f"[metric] {k:<5} AUROC={metrics[k]['auroc']:.4f} AUPR={metrics[k]['aupr']:.4f} "
             f"MCC@τ={metrics[k]['at_tau_oof']['mcc']:.4f} BA@{metrics[k]['at_tau_oof']['ba']:.4f}")
    for name, f in fusion.items():
        metrics[name] = full_metrics(y_ex, f["prob_ext"], L_ex, f["tau"])
        plog(f"[metric] {name} AUROC={metrics[name]['auroc']:.4f} "
             f"AUPR={metrics[name]['aupr']:.4f} MCC@τ={metrics[name]['at_tau_oof']['mcc']:.4f}")

    # ---- 基线：全判阳 / 全判阴 / 随机
    base_metrics = {"all_positive": mcc_stats(y_ex, np.ones(len(y_ex)), 0.5)}
    base_metrics["all_positive"].update(
        auroc=0.5, aupr=round(float(np.mean(y_ex)), 4),
        lift_aupr=1.0, prec50=round(float(np.mean(y_ex)), 4), lift50=1.0,
        ba=0.5, note="全判阳：Acc=阳性率、F1=2P/(1+P)、recall=1、Sp=0")
    base_metrics["all_negative"] = mcc_stats(y_ex, np.zeros(len(y_ex)), 0.5)
    base_metrics["all_negative"].update(auroc=0.5, aupr=None, ba=0.5,
                                        note="全判阴：Sn=0、Acc=1-阳性率")
    plog(f"[baseline] 全判阳 Acc={base_metrics['all_positive']['acc']:.4f} "
         f"F1={base_metrics['all_positive']['f1']:.4f} Sn={base_metrics['all_positive']['sn']:.4f}")

    # ---- 子集：只保留长度 11-25
    sub = (L_ex >= 11) & (L_ex <= 25)
    subset = dict(mask_desc="11<=length<=25", n=int(sub.sum()), npos=int(y_ex[sub].sum()))
    for k in VIEWS:
        p = ext_p[k][sub]
        if len(np.unique(y_ex[sub])) > 1:
            t, _ = tau_mcc_opt(y_ex[sub], p)
            subset[k] = dict(auroc=round(float(roc_auc_score(y_ex[sub], p)), 4),
                             aupr=round(float(average_precision_score(y_ex[sub], p)), 4),
                             mcc_at_ext_opt=round(float(mcc_stats(y_ex[sub], p, t)["mcc"]), 4))
    for name, f in fusion.items():
        p = f["prob_ext"][sub]
        if len(np.unique(y_ex[sub])) > 1:
            t, _ = tau_mcc_opt(y_ex[sub], p)
            subset[name] = dict(auroc=round(float(roc_auc_score(y_ex[sub], p)), 4),
                                aupr=round(float(average_precision_score(y_ex[sub], p)), 4),
                                mcc_at_ext_opt=round(float(mcc_stats(y_ex[sub], p, t)["mcc"]), 4))

    # ---- 鲁棒性：4 视角（无 CNN）用完整 train0 池 3,583 行定 τ，只看 τ 是否稳定
    r4 = {k: rank01(oof_ml[k] if k in ML_VIEWS else oof_tlora) for k in VIEWS[:4]}
    w4 = {k: W_T0[k] / sum(W_T0[x] for x in VIEWS[:4]) for k in VIEWS[:4]}
    s4 = sum(w4[k] * r4[k] for k in VIEWS[:4])
    _, tau4, mcc4, _ = platt_fit_tau(y_tr, s4)
    tau_robust = dict(pool="train0 全池 3,583（4 视角，无 CNN）", tau=tau4, oof_mcc=mcc4,
                      oof_auroc=round(float(roc_auc_score(y_tr, s4)), 4))
    plog(f"[tau][robust] 4 视角/train0 全池 τ={tau4:.3f} OOF_MCC={mcc4:.4f}")

    # ---- probs.csv
    df = pd.DataFrame(dict(sequence=ext_seq, label=y_ex, length=L_ex))
    for k in VIEWS:
        df[f"prob_{k}"] = np.round(ext_p[k], 8)
    for name, f in fusion.items():
        df[f"rank_{name}"] = np.round(f["fuse_ext"], 6)
        df[f"{name}"] = np.round(f["prob_ext"], 8)
    df.to_csv(os.path.join(OUT, "probs.csv"), index=False, encoding="utf-8")

    # ---- metrics.json
    out = dict(
        task="ext261: 已训好的 train0/train1 五视角流水线 -> 新外部集（清洗后 258 行）",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol=dict(
            external=f"{XLSX} / sheet '{SHEET}' / 261 行 -> 保留 258（211 阳 / 47 阴）",
            folds="官方 folds.csv 5 折（eval_grid/fold_of_pool）；PIP 新行按长度箱轮转",
            ml="hand704/gap3/k4：train0（3,583 行）5 折重拟合，逐折表示拟合，"
               "ET 1500/max_features .1/msl 1/balanced，3 seed 均值；外部概率=5 折均值",
            tlora="wt-tlora5 {0..4}_best.pt（train0 口径 5 折 delta）+ {k}_init.npz，"
                  "ESM-2 t33 650M fp16 + posproj-adaptive 头；仅推理，5 折概率均值",
            cnn="ext_cnn/weights_A.pt（train1 A 臂 5 折 state_dict），单 seed；仅推理，5 折均值",
            fusion="rank01=rankdata/n；fuse=Σ w_k·rank01_k；Platt(C=1e6) 用 OOF 拟合，"
                   "τ=MCC 最优（OOF）；外部集不参与任何拟合/调参",
            weights_t0={k: round(W_T0[k], 6) for k in VIEWS},
            weights_t0_note="ET 三视角按 train0 rank_greedy 15:6:6 保比例缩到 15/27 份额"
                            "（225/729, 90/729, 90/729），DL 沿用 view5_t1 的 TLoRA 9/27、CNN 3/27",
            weights_v5t1={k: round(W_V5T1[k], 6) for k in VIEWS},
            weights_equal={k: round(W_EQUAL[k], 6) for k in VIEWS},
            tau_pool=f"5 视角公共 OOF = train0∩train1 = {n_oof} 行（train1 ⊂ train0）",
        ),
        ml=ml_res, tlora=tlora_res, cnn=cnn_res,
        per_view_tau=per_view_tau,
        fusion={name: dict(weights=fusion[name]["weights"], tau=fusion[name]["tau"],
                           oof_mcc=fusion[name]["oof_mcc"], oof_auroc=fusion[name]["oof_auroc"])
                for name in fusion},
        metrics=metrics, baselines=base_metrics, subset_11_25=subset,
        tau_robust=tau_robust,
        old_ext193=dict(fused_auroc=0.4883, fused_mcc=0.0018, fused_aupr=0.7014,
                        n=193, npos=136, note="上一版外部集 ext193（train1 口径 5 视角，τ=0.247）"),
        wall_sec=round(time.time() - TIME0, 1),
    )
    jdump(out, os.path.join(OUT, "metrics.json"))
    render_table(out)
    plog(f"[done] wall={out['wall_sec']}s -> data_clean.csv / probs.csv / metrics.json / table.md")
    return out


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


DISP = {"hand": "hand704", "gap3": "gap3", "k4": "k4",
        "tlora": "TLoRA+posproj", "cnn": "小 CNN",
        "fuse_t0": "**融合 rank（train0 权重）**",
        "fuse_v5t1": "融合 rank（view5_t1 权重）",
        "fuse_equal": "融合 rank（等权）"}


def render_table(out):
    m = out["metrics"]
    b = out["baselines"]
    keys = VIEWS + ["fuse_t0", "fuse_v5t1", "fuse_equal"]
    rows = [(DISP[k], k, m[k]) for k in keys]
    L = []
    A = L.append
    A("# ext261 外部集评测表（5 视角 + 融合，train0/train1 已训流水线）")
    A("")
    A(f"生成时间：{out['created']}　运行耗时：{out['wall_sec']}s")
    A("")
    A("## 数据")
    A("")
    A("- 原始：`C:\\Users\\cxy\\Desktop\\外部数据集_1114.xlsx` / sheet `长表_序列标签`，261 行 = 211 阳 + 50 阴。")
    A("- 清洗后 **258 行 = 211 阳（81.8%）/ 47 阴**；剔除 3 条（环肽+D 型/β3、Nva 三字母、N 端甲酰化 fMLP），"
      "`Trp-Lys-Tyr-Met-Val-Met → WKYMVM` 保留。")
    A("- 与 `table.csv` / `test.csv` / `pip(1).csv` 精确序列重叠 0（脚本内 assert）。")
    A("- 长度 3–70，均值 13.90，中位 11。")
    A("")
    A("## 协议（外部集只做测试，不参与任何拟合/调参）")
    A("")
    A(f"- 折：{out['protocol']['folds']}")
    A(f"- ML：{out['protocol']['ml']}")
    A(f"- TLoRA：{out['protocol']['tlora']}")
    A(f"- CNN：{out['protocol']['cnn']}")
    A(f"- 融合：{out['protocol']['fusion']}")
    A(f"- 权重（train0 口径，主用）：" + "、".join(
        f"{k} {w:.4f}" for k, w in out["protocol"]["weights_t0"].items()))
    A(f"  - {out['protocol']['weights_t0_note']}")
    A("- 权重（view5_t1 原样，供与 ext193 对齐）：" + "、".join(
        f"{k} {w:.4f}" for k, w in out["protocol"]["weights_v5t1"].items()))
    A(f"- τ 池：{out['protocol']['tau_pool']}")
    A("")
    A("## 表 1　主表（τ = OOF MCC 最优，未在外部集上调）")
    A("")
    A("| 模型 | AUROC | AUPR | AUPR/阳性率 | MCC | BA | Sn | Sp |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, _mk, r in rows:
        t = r["at_tau_oof"]
        A(f"| {lbl} | {_fmt(r['auroc'])} | {_fmt(r['aupr'])} | {r['lift_aupr']:.2f} | "
          f"{_fmt(t['mcc'])} | {_fmt(t['ba'])} | {_fmt(t['sn'])} | {_fmt(t['sp'])} |")
    ap = b["all_positive"]
    A(f"| *参考：全判阳* | 0.5000 | {ap['aupr']:.4f} | 1.00 | "
      f"{_fmt(ap['mcc'])} | {_fmt(ap['ba'])} | {_fmt(ap['sn'])} | {_fmt(ap['sp'])} |")
    A(f"| *参考：全判阴* | 0.5000 | — | — | 0.0000 | 0.5000 | 0.0000 | 1.0000 |")
    A("")
    A(f"全判阳基线：Acc={ap['acc']:.4f}、F1={ap['f1']:.4f}、Precision={ap['prec']:.4f}、recall=1.0000"
      "（阳性率 81.8% 时这就是「什么都不做」的上限陷阱）。")
    A("")
    A("## 表 2　阈值档：@OOF τ / @0.5 / @外部 MCC 最优（oracle 上界）")
    A("")
    A("| 模型 | τ_OOF | MCC@τ | Sn@τ | Sp@τ | Acc@τ | F1@τ | MCC@0.5 | Sn@0.5 | Sp@0.5 | Acc@0.5 | F1@0.5 | τ_ext(oracle) | MCC_ext | BA_ext |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lbl, _mk, r in rows:
        t, h = r["at_tau_oof"], r["at_0.5"]
        e = r["at_tau_ext"] or {}
        A(f"| {lbl} | {t['tau']:.3f} | {_fmt(t['mcc'])} | {_fmt(t['sn'])} | {_fmt(t['sp'])} | "
          f"{_fmt(t['acc'])} | {_fmt(t['f1'])} | {_fmt(h['mcc'])} | {_fmt(h['sn'])} | {_fmt(h['sp'])} | "
          f"{_fmt(h['acc'])} | {_fmt(h['f1'])} | {_fmt(e.get('tau'), 3)} | {_fmt(e.get('mcc'))} | {_fmt(e.get('ba'))} |")
    A("")
    A("## 表 3　排序质量 / 高精度区 / 固定特异度 / 不确定度")
    A("")
    A("| 模型 | AUROC | AUPR | P@50 | lift@50 | Sn@Sp≥0.80 | AUROC 95%CI (2000 boot) |")
    A("|---|---|---|---|---|---|---|")
    for lbl, _mk, r in rows:
        s = r["sn_at_sp80"] or {}
        A(f"| {lbl} | {_fmt(r['auroc'])} | {_fmt(r['aupr'])} | {_fmt(r['prec50'])} | "
          f"{r['lift50']:.2f} | {_fmt(s.get('sn'))} | "
          f"[{r['boot']['ci95'][0]:.4f}, {r['boot']['ci95'][1]:.4f}] |")
    A("")
    A("## 表 4　长度分箱 AUROC")
    A("")
    A("| 模型 | " + " | ".join(BIN_NAMES) + " |")
    A("|---|" + "---|" * len(BIN_NAMES))
    first = m[VIEWS[0]]["bins"]
    A("| *n* | " + " | ".join(str(first[x]["n"]) for x in BIN_NAMES) + " |")
    A("| *阳性* | " + " | ".join(str(first[x]["npos"]) for x in BIN_NAMES) + " |")
    for lbl, _mk, r in rows:
        A(f"| {lbl} | " + " | ".join(_fmt(r["bins"][x]["auroc"]) for x in BIN_NAMES) + " |")
    A("")
    A("## 表 5　子集：只保留长度 11–25")
    A("")
    sub = out["subset_11_25"]
    A(f"n = {sub['n']}（阳 {sub['npos']}，阳性率 {sub['npos'] / max(sub['n'], 1):.3f}）")
    A("")
    A("| 模型 | AUROC | AUPR | MCC（子集内 MCC 最优阈值） |")
    A("|---|---|---|---|")
    for lbl, mk, _r in rows:
        if mk in sub:
            A(f"| {lbl} | {_fmt(sub[mk]['auroc'])} | {_fmt(sub[mk]['aupr'])} | {_fmt(sub[mk]['mcc_at_ext_opt'])} |")
    A("")
    A("## 表 6　保真校验（重算 vs 已发布产物）")
    A("")
    mlf = out["ml"].get("fidelity") or {}
    A("| 视角 | 校验对象 | max\\|Δ\\| | 结论 |")
    A("|---|---|---|---|")
    for k in ML_VIEWS:
        if k in mlf:
            A(f"| {k} | train0 test0/oof 概率（`grid_out/train0__test0.npz`） | "
              f"{mlf[k]['test_max_abs_d']:.2e} / {mlf[k]['oof_max_abs_d']:.2e} | "
              f"{'逐元素一致' if mlf[k]['test_max_abs_d'] < 1e-9 else '不一致'} |")
    tl = out["tlora"]
    A(f"| TLoRA | 每折 `{{k}}_val_probs.csv` 全量重算（717/716 行 × 5 折） | "
      f"{tl['val_repro_max_abs_d_max']:.2e} | {'逐元素一致（fp16 舍入）' if tl['val_repro_max_abs_d_max'] < 1e-6 else '不一致'} |")
    A(f"| CNN | 权重直接来自 `ext_cnn/weights_A.pt`，无重训 | 0 | 同一次推理 |")
    A("")
    A("## 表 7　逐折外部集 AUROC（离散度）")
    A("")
    A("| 视角 | fold0 | fold1 | fold2 | fold3 | fold4 | 均值 |")
    A("|---|---|---|---|---|---|---|")
    for k in ("tlora", "cnn"):
        pf = out[k]["per_fold"]
        vals = [f"{p['ext_auroc']:.4f}" for p in pf]
        A(f"| {k} | " + " | ".join(vals) + f" | {out[k]['ext_foldwise_mean']:.4f} |")
    for k in ML_VIEWS:
        pf = out["ml"]["per_fold"][k]["ext"]
        A(f"| {k} | " + " | ".join(f"{v:.4f}" for v in pf) +
          f" | {np.mean(pf):.4f} |")
    A("")
    A("## 备注")
    A("")
    A(f"- 参考：上一版外部集 ext193（193 行 / 136 阳）5 视角融合 AUROC 0.4883、MCC 0.0018、"
      f"阳性率 0.5803、Sn 0.581 / Sp 0.421（τ=0.247）。")
    A(f"- τ 稳健性（4 视角、完整 train0 池 3,583 行）：τ={out['tau_robust']['tau']:.3f}，"
      f"OOF MCC={out['tau_robust']['oof_mcc']:.4f}，OOF AUROC={out['tau_robust']['oof_auroc']:.4f}。")
    A(f"- 外部集长度 <8 的样本有 n={first['<8']['n']}（阳 {first['<8']['npos']}），占比不低；"
      "该箱几乎全为短肽，与训练池域差异最大。")
    A("- 显著性速记：hand704 与 TLoRA 的 95%CI 整体低于 0.5（hand [0.2977,0.4535]），"
      "即**在外部集上系统性反向**，不是抽样噪声；融合因权重压在这两个反向视角上（合计 64.2%），"
      "AUROC 反而低于所有单视角中最好的 gap3。")
    A("- Sn@Sp=0.80 一列普遍很低：外部集只有 47 条阴性，Sp=0.80 意味着 FP≤9，"
      "任何视角都拿不出高 Sn 的阈值。")
    A("- 所有 Sn/Sp/Acc/F1/MCC 在 @τ 列使用 OOF 阈值；@τ_ext 列是外部集 MCC 最优（oracle），只作上界参考。")
    A("")
    path = os.path.join(OUT, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    plog(f"[table] -> {path}")
    return path


def main_table_only():
    with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as fh:
        out = json.load(fh)
    render_table(out)
    print(open(os.path.join(OUT, "table.md"), encoding="utf-8").read())


if __name__ == "__main__":
    if "--table-only" in sys.argv[1:]:
        main_table_only()
    else:
        main()
