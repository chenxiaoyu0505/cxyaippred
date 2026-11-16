# -*- coding: utf-8 -*-
"""Metrics and rank-space fusion shared by fuse_eval.py and verify.py.

Ported from ``_work/wt-drop385/scripts/eval_grid.py``:

  * ``len_matched_auc``  -- length-stratified AUROC (weight n1*n0 per length)
  * ``boot_ci``          -- 2,000-resample bootstrap 95% CI of the AUROC
  * ``rank01``           -- percentile rank in [0, 1]
  * ``greedy_rank``      -- Caruana forward selection, 27 rounds, on OOF ranks
  * ``platt`` / ``tau_from_scores`` -- OOF MCC-optimal threshold (never 0.5)

Everything is computed on the fixed test1 rows; tau is always chosen on the OOF
predictions of train1, never on the test set.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

BINS = ["le14", "15", "16-18", "ge19"]
ROUNDS = 27


def len_bins(L):
    return np.where(L <= 14, "le14", np.where(L == 15, "15",
                     np.where(L <= 18, "16-18", "ge19")))


def len_matched_auc(y, p, L):
    num = den = 0.0
    for v in np.unique(L):
        m = L == v
        a, b = int(y[m].sum()), int((1 - y[m]).sum())
        if a and b:
            w = a * b
            num += w * roc_auc_score(y[m], p[m])
            den += w
    return float(num / den) if den else float("nan")


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def rank01(x):
    return rankdata(x) / len(x)


def tau_from_scores(y, scores, lo=0.01, hi=0.99, step=0.001):
    """OOF MCC-optimal threshold (never a hard-coded 0.5)."""
    best = (-2.0, lo)
    for t in np.arange(lo, hi + 1e-9, step):
        m = matthews_corrcoef(y, (scores >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1]


def platt(y, oof_scores):
    """Map an OOF rank score to a probability; returns (model, tau, p_oof)."""
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(oof_scores.reshape(-1, 1), y)
    p_oof = lr.predict_proba(oof_scores.reshape(-1, 1))[:, 1]
    return lr, tau_from_scores(y, p_oof), p_oof


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
    return ens, {v: c / rounds for v, c in cnt.items() if c}


def metrics(y, p, L, tau):
    out = dict(n=int(len(y)), npos=int(y.sum()),
               auroc=round(float(roc_auc_score(y, p)), 4),
               aupr=round(float(average_precision_score(y, p)), 4),
               lenmatch=round(len_matched_auc(y, p, L), 4),
               ci=boot_ci(y, p))
    lb = len_bins(L)
    out["bin"] = {b: (round(float(roc_auc_score(y[lb == b], p[lb == b])), 4)
                      if len(np.unique(y[lb == b])) > 1 else None) for b in BINS}
    pred = (p >= tau).astype(int)
    out["tau"] = round(float(tau), 4)
    out["mcc"] = round(float(matthews_corrcoef(y, pred)), 4)
    out["sn"] = round(float(((pred == 1) & (y == 1)).sum() / max(int((y == 1).sum()), 1)), 3)
    out["sp"] = round(float(((pred == 0) & (y == 0)).sum() / max(int((y == 0).sum()), 1)), 3)
    out["posrate"] = round(float(pred.mean()), 4)
    return out
