# -*- coding: utf-8 -*-
"""Three handcrafted-feature views (hand704 / gap3 / k4) -- retrained from raw sequence.

This is a line-by-line port of `_work/wt-drop385/scripts/eval_grid.py`
(`gap3_tokens`, `k4_tokens`, `fold_view`, the fold loop), which is the code that
produced the released numbers.  Do not "improve" the tokenizer or move the
representation fitting outside the fold -- both mistakes change the score.

Per view, per fold
------------------
    gap3 / k4 : TfidfVectorizer(sublinear_tf, l2, analyzer=word, token_pattern=r"\\S+")
                -> TruncatedSVD(300, random_state=0) -> StandardScaler
                ALL THREE fitted on that fold's TRAINING rows only, then applied to
                (train rows, val rows, test rows).  Fitting on the full table would
                leak the test set through the IDF/SVD basis.
    hand      : raw 704-dim physicochemical/amphipath matrix (no TF-IDF, no SVD).
                ExtraTrees splits per feature, so a monotone transform such as
                StandardScaler would give the same trees; eval_grid uses the raw
                matrix and so do we.

Classifier: ExtraTrees(n_estimators=1500, max_features=0.1, min_samples_leaf=1,
class_weight="balanced"), seeds 0/1/2 averaged.  OOF probability of a val row =
mean over the 3 seeds of that fold; test probability = mean over seeds, then mean
over the 5 folds.

Output: `out/ml_views.npz` with keys tr_seq, te_seq, y_tr, y_te, te_length,
oof_{hand,gap3,k4}, test_{hand,gap3,k4}.

Usage
-----
    python scripts/run_ml_views.py              # all 5 folds (~4-6 min, CPU)
    python scripts/run_ml_views.py --folds 0    # smoke test, one fold
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)

import features as fe  # noqa: E402  (vendored verbatim from the training repo)

VIEWS = ["hand", "gap3", "k4"]
TOKEN_VIEWS = ["gap3", "k4"]
ET_KW = dict(n_estimators=1500, max_features=0.1, min_samples_leaf=1,
             class_weight="balanced", n_jobs=-1)
NSEED, NFOLD, SVD_DIM = 3, 5, 300
N_TRAIN1, N_TEST1, POS_TRAIN1, POS_TEST1 = 3278, 817, 1060, 262


# --------------------------------------------------------------------------- features
def gap3_tokens(seq):
    """Spaced 3-mer: 4-residue window, drop one position, put 'X' in that slot."""
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


def hand704(seqs):
    """704-dim handcrafted matrix (694 handcraft + 10 amphipath), cached by sequence hash."""
    key = hashlib.sha256("\n".join(seqs).encode("utf-8")).hexdigest()[:16]
    os.makedirs(OUT, exist_ok=True)
    npy = os.path.join(OUT, "hand704_%s.npy" % key)
    if os.path.exists(npy):
        X = np.load(npy)
        print("[hand] cached %s %s" % (X.shape, os.path.basename(npy)), flush=True)
        return X
    t0 = time.time()
    base = np.stack([fe.compute_handcraft(s) for s in seqs]).astype(np.float32)
    extra = np.stack([fe.compute_amphipath(s) for s in seqs]).astype(np.float32)
    X = np.concatenate([base, extra], axis=1)
    assert X.shape == (len(seqs), 704), X.shape
    assert not np.isnan(X).any()
    np.save(npy, X)
    print("[hand] computed %s in %.0fs" % (X.shape, time.time() - t0), flush=True)
    return X


def fold_view(kind, Xh, docs, fit_idx, apply_idx):
    """Fit the representation on fit_idx ONLY (per fold), return matrix for apply_idx."""
    if kind == "hand":
        return Xh[apply_idx]
    tf = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                         norm="l2", lowercase=False)
    tf.fit([docs[kind][i] for i in fit_idx])
    Z_fit = tf.transform([docs[kind][i] for i in fit_idx])
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=0)
    svd.fit(Z_fit)
    E_fit = svd.transform(Z_fit).astype(np.float32)
    sc = StandardScaler().fit(E_fit)
    E_ap = svd.transform(tf.transform([docs[kind][i] for i in apply_idx])).astype(np.float32)
    return sc.transform(E_ap).astype(np.float32)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma separated fold ids")
    args = ap.parse_args()
    folds_to_run = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    t_all = time.time()
    tr = pd.read_csv(os.path.join(DATA, "train1.csv"), dtype={"sequence": str})
    te = pd.read_csv(os.path.join(DATA, "test1.csv"), dtype={"sequence": str})
    tr["sequence"] = tr.sequence.str.strip().str.upper()
    te["sequence"] = te.sequence.str.strip().str.upper()
    assert len(tr) == N_TRAIN1 and int(tr.label.sum()) == POS_TRAIN1
    assert len(te) == N_TEST1 and int(te.label.sum()) == POS_TEST1
    assert not set(tr.sequence) & set(te.sequence), "train/test overlap"
    y_tr, y_te = tr.label.values.astype(int), te.label.values.astype(int)
    L_te = te.length.values.astype(int)
    fold_of = tr.fold.values.astype(int)

    # universe = train1 rows followed by test1 rows
    seqs = list(tr.sequence) + list(te.sequence)
    n_tr, n_te = len(tr), len(te)
    te_uni = np.arange(n_tr, n_tr + n_te)

    Xh = hand704(seqs)
    docs = {k: [" ".join(TOKENIZERS[k](s)) for s in seqs] for k in TOKEN_VIEWS}
    print("[docs] gap3/k4 token strings built (%.0fs)" % (time.time() - t_all), flush=True)

    oof = {k: np.zeros(n_tr) for k in VIEWS}
    fold_test = {k: [] for k in VIEWS}
    done = []
    for k in range(NFOLD):
        tr_i = np.where(fold_of != k)[0]
        va_i = np.where(fold_of == k)[0]
        if k not in folds_to_run:
            continue
        n_f, n_v = len(tr_i), len(va_i)
        apply_idx = np.concatenate([tr_i, va_i, te_uni])
        for kind in VIEWS:
            t0 = time.time()
            Xa = fold_view(kind, Xh, docs, tr_i, apply_idx)
            Xtr_f, Xva, Xte = Xa[:n_f], Xa[n_f:n_f + n_v], Xa[n_f + n_v:]
            ps_va, ps_te = [], []
            for s in range(NSEED):
                et = ExtraTreesClassifier(random_state=s, **ET_KW)
                et.fit(Xtr_f, y_tr[tr_i])
                ps_va.append(et.predict_proba(Xva)[:, 1])
                ps_te.append(et.predict_proba(Xte)[:, 1])
            oof[kind][va_i] = np.mean(ps_va, axis=0)
            p = np.mean(ps_te, axis=0)
            fold_test[kind].append(p)
            print("  fold%d %-5s n_fit=%d n_val=%d test_auroc=%.4f (%.0fs)"
                  % (k, kind, n_f, n_v, roc_auc_score(y_te, p), time.time() - t0), flush=True)
        done.append(k)

    if len(done) != NFOLD:
        print("\n[smoke] only folds %s run; not writing out/ml_views.npz" % done)
        return 0

    tst = {k: np.mean(fold_test[k], axis=0) for k in VIEWS}
    res = {"views": {}}
    print("\n%-6s %10s %10s %10s" % ("view", "test_auroc", "test_aupr", "oof_auroc"))
    for k in VIEWS:
        row = dict(test_auroc=round(float(roc_auc_score(y_te, tst[k])), 4),
                   test_aupr=round(float(average_precision_score(y_te, tst[k])), 4),
                   oof_auroc=round(float(roc_auc_score(y_tr, oof[k])), 4))
        res["views"][k] = row
        print("%-6s %10.4f %10.4f %10.4f" % (k, row["test_auroc"], row["test_aupr"],
                                             row["oof_auroc"]))

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "ml_views.npz"),
             tr_seq=np.array(tr.sequence, dtype=str), te_seq=np.array(te.sequence, dtype=str),
             y_tr=y_tr, y_te=y_te, te_length=L_te, fold_of=fold_of,
             **{"oof_" + k: oof[k] for k in VIEWS},
             **{"test_" + k: tst[k] for k in VIEWS},
             **{"fold_test_" + k: np.array(fold_test[k]) for k in VIEWS})
    res["n_train"], res["n_test"] = int(n_tr), int(n_te)
    res["npos_train"], res["npos_test"] = int(y_tr.sum()), int(y_te.sum())
    with open(os.path.join(OUT, "ml_views.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n[saved] out/ml_views.npz  (%.1f min)" % ((time.time() - t_all) / 60))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
