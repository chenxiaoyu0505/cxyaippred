# -*- coding: utf-8 -*-
"""Late fusion: percentile ranks + Caruana forward selection on train1 OOF scores.

Two scoring modes
-----------------
batch  Ranks are computed inside the batch being scored (``rankdata(x)/n``).  This
       is what reproduces the reported test1 numbers when the whole test set is
       passed at once, and it is order-independent as long as the batch is fixed.
fixed  Each view's raw probability is mapped to a rank through the empirical CDF
       of its train1 out-of-fold scores (``weights/ecdf.json``).  Every peptide
       gets an absolute score that does not depend on what else is in the batch --
       use this for single sequences or for screening a new library.

The view weights are fitted once, on train1 OOF ranks, by 27 rounds of forward
selection with replacement (Caruana et al., 2004); ``test1`` never enters the
fitting.  The decision threshold tau is the OOF MCC-optimal value.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402

ROUNDS = 27


def rank01(x):
    x = np.asarray(x, dtype=float)
    return rankdata(x) / len(x)


def greedy_rank(y, ro, rounds=ROUNDS):
    """Caruana forward selection with replacement on OOF ranks."""
    views = list(ro)
    cnt = {v: 0 for v in views}
    ens = np.zeros(len(y))
    for r in range(rounds):
        best, bv = -1.0, None
        for v in views:
            a = roc_auc_score(y, (ens * r + ro[v]) / (r + 1))
            if a > best:
                best, bv = a, v
        cnt[bv] += 1
        ens = (ens * r + ro[bv]) / (r + 1)
    w = {v: c / rounds for v, c in cnt.items() if c}
    return ens, w


def tau_from_scores(y, scores, lo=0.05, hi=0.95, step=0.001):
    """OOF MCC-optimal threshold on the fitted probabilities."""
    best = (-2.0, lo)
    for t in np.arange(lo, hi + 1e-9, step):
        m = matthews_corrcoef(y, (scores >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1]


def fit_platt(y, oof):
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(np.asarray(oof).reshape(-1, 1), y)
    p_oof = lr.predict_proba(np.asarray(oof).reshape(-1, 1))[:, 1]
    return lr, tau_from_scores(y, p_oof), p_oof


def fit_weights(oof_by_view, y, save_path=None):
    """Fit ranks + greedy weights + Platt + tau on train1 OOF, optionally save."""
    ro = {v: rank01(oof_by_view[v]) for v in oof_by_view}
    ens, w = greedy_rank(y, ro)
    lr, tau, p_oof = fit_platt(y, ens)
    ecdf = {v: np.sort(np.asarray(oof_by_view[v], dtype=float)) for v in oof_by_view}
    cfg = dict(views=list(oof_by_view), weights=w, rounds=ROUNDS,
               tau=float(tau),
               platt=dict(coef=float(lr.coef_[0][0]), intercept=float(lr.intercept_[0])),
               oof_auroc=float(roc_auc_score(y, ens)),
               protocol="train1 OOF, SKF(5, shuffle, seed 42)")
    if save_path:
        np.savez(save_path.replace(".json", "_ecdf.npz"), **ecdf)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg, ens, p_oof


def load_config(path=None):
    path = path or os.path.join(C.WEIGHTS, "fusion.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    ecdf = np.load(path.replace(".json", "_ecdf.npz"))
    cfg["ecdf"] = {k: ecdf[k] for k in ecdf.files}
    return cfg


def _fixed_rank(values, reference):
    """Rank of each value through the empirical CDF of `reference` (0, 1]."""
    ref = np.asarray(reference, dtype=float)
    order = np.argsort(ref)
    ref_sorted = ref[order]
    idx = np.searchsorted(ref_sorted, values, side="right")
    return (idx + 0.5) / (len(ref) + 1.0)


def fuse(probs_by_view, cfg, mode="batch"):
    """Combine per-view probabilities into one score and a decision."""
    views = cfg["views"]
    if mode == "batch":
        ro = {v: rank01(probs_by_view[v]) for v in views}
    elif mode == "fixed":
        ro = {v: _fixed_rank(np.asarray(probs_by_view[v], dtype=float), cfg["ecdf"][v])
              for v in views}
    else:
        raise ValueError(mode)
    score = np.zeros(len(np.asarray(probs_by_view[views[0]])))
    for v in views:
        score = score + cfg["weights"].get(v, 0.0) * ro[v]
    coef, b = cfg["platt"]["coef"], cfg["platt"]["intercept"]
    prob = 1.0 / (1.0 + np.exp(-(coef * score + b)))
    return score, prob, (prob >= cfg["tau"]).astype(int)
