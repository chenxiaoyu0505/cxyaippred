# -*- coding: utf-8 -*-
"""Rank-average fusion of 3 / 4 / 5 views + the equal-weight control.

Fusion rule (identical to the released `view5_t1.py`)
-----------------------------------------------------
    rank_k = rankdata(p_k) / n         # computed inside each cohort separately
    score  = sum_k w_k * rank_k        # w frozen from the OOF greedy search

* OOF ranks are taken over the 3,278 training rows, test ranks over the 817 test
  rows.  Ranks are **cohort-level (transductive)**: you need the whole batch to
  score it, so this arm cannot classify a single sequence on its own.
* AUROC/AUPR are invariant to any strictly monotone map, so only the ranks matter.
* The operating threshold tau is the MCC-optimal cut on the OOF fused score
  (`matthews_corrcoef` maximised over a 0.01..0.99 grid) -- never 0.5, never tuned
  on test.  Sn/Sp/MCC/BA below are all at that tau.

The weights are **frozen**: they come from a Caruana greedy search with replacement
(27 rounds, OOF rank space, maximising AUROC) run once on the training pool, and are
hard-coded here on purpose.  Re-running the greedy search on this OOF would give
very nearly the same weights; hard-coding makes the fusion reproducible without
re-fitting anything.

Two sources are supported for the deep views (`--dl`):

* `published` (default) -- the frozen released probabilities of the paper's runs
  (`out/dl_published.npz`, built by `scripts/load_published_dl.py`).
* `weights`             -- re-inference from the surviving checkpoints
  (`out/dl_views.npz`, built by `scripts/run_dl_views.py`).

Usage:  python scripts/fuse_eval.py [--dl published|weights] [--out out/fused.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")

# frozen fusion weights (OOF rank-space Caruana, 27 rounds)
WEIGHTS = {
    "3view": {"hand": 0.5926, "gap3": 0.1852, "k4": 0.2222},
    "4view": {"hand": 0.2963, "gap3": 0.1481, "k4": 0.2222, "TLoRA": 0.3333},
    "5view": {"hand": 0.2222, "gap3": 0.1481, "k4": 0.1852, "TLoRA": 0.3333,
              "CNN": 0.1111},
}


def rank01(x):
    return rankdata(x) / len(x)


def tau_from_scores(y, scores, lo=0.01, hi=0.99, step=0.001):
    """MCC-optimal threshold on the OOF fused score."""
    best = (-2.0, lo)
    for t in np.arange(lo, hi + 1e-9, step):
        m = matthews_corrcoef(y, (scores >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1]


def metrics(y, score, tau):
    pred = (score >= tau).astype(int)
    sn = float(((pred == 1) & (y == 1)).sum() / max(int((y == 1).sum()), 1))
    sp = float(((pred == 0) & (y == 0)).sum() / max(int((y == 0).sum()), 1))
    return dict(n=int(len(y)), npos=int(y.sum()),
                auroc=round(float(roc_auc_score(y, score)), 4),
                aupr=round(float(average_precision_score(y, score)), 4),
                tau=round(float(tau), 4),
                mcc=round(float(matthews_corrcoef(y, pred)), 4),
                sn=round(sn, 3), sp=round(sp, 3), ba=round((sn + sp) / 2, 4),
                posrate=round(float(pred.mean()), 4))


def load_views(dl_file):
    ml = np.load(os.path.join(OUT, "ml_views.npz"))
    dl = np.load(os.path.join(OUT, dl_file))
    oof = {"hand": ml["oof_hand"], "gap3": ml["oof_gap3"], "k4": ml["oof_k4"],
           "TLoRA": dl["oof_tlora"], "CNN": dl["oof_cnn"]}
    test = {"hand": ml["test_hand"], "gap3": ml["test_gap3"], "k4": ml["test_k4"],
            "TLoRA": dl["test_tlora"], "CNN": dl["test_cnn"]}
    y_tr, y_te = ml["y_tr"].astype(int), ml["y_te"].astype(int)
    return oof, test, y_tr, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", choices=("published", "weights"), default="published",
                    help="source of the TLoRA / CNN probabilities")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dl_file = "dl_published.npz" if args.dl == "published" else "dl_views.npz"
    out_path = args.out or os.path.join(OUT, "fused.json")

    oof, test, y_tr, y_te = load_views(dl_file)
    print("[fuse] deep views from out/%s" % dl_file)
    ro = {k: rank01(v) for k, v in oof.items()}
    rt = {k: rank01(v) for k, v in test.items()}

    res = {"dl_source": args.dl, "views": {}, "fusion": {}, "equal": {}}
    print("%-22s %10s %10s %10s %8s %6s %6s" %
          ("arm", "test_auroc", "test_aupr", "oof_auroc", "tau", "sn", "sp"))
    for k in ("hand", "gap3", "k4", "TLoRA", "CNN"):
        m = dict(test_auroc=round(float(roc_auc_score(y_te, test[k])), 4),
                 test_aupr=round(float(average_precision_score(y_te, test[k])), 4),
                 oof_auroc=round(float(roc_auc_score(y_tr, oof[k])), 4))
        res["views"][k] = m
        print("%-22s %10.4f %10.4f %10.4f" % (k, m["test_auroc"], m["test_aupr"],
                                              m["oof_auroc"]))

    for tier, w in WEIGHTS.items():
        order = [k for k in w]
        fused_oof = sum(w[k] * ro[k] for k in order)
        fused_te = sum(w[k] * rt[k] for k in order)
        tau = tau_from_scores(y_tr, fused_oof)
        m = metrics(y_te, fused_te, tau)
        m["weights"] = dict(w)
        m["oof_auroc"] = round(float(roc_auc_score(y_tr, fused_oof)), 4)
        res["fusion"][tier] = m
        print("%-22s %10.4f %10.4f %10.4f %8.3f %6.3f %6.3f  (tau from OOF, MCC %.3f)"
              % (tier, m["auroc"], m["aupr"], m["oof_auroc"], m["tau"], m["sn"],
                 m["sp"], m["mcc"]))

        eq_oof = np.mean([ro[k] for k in order], axis=0)
        eq_te = np.mean([rt[k] for k in order], axis=0)
        m2 = metrics(y_te, eq_te, tau_from_scores(y_tr, eq_oof))
        m2["weights"] = {k: round(1.0 / len(order), 4) for k in order}
        m2["oof_auroc"] = round(float(roc_auc_score(y_tr, eq_oof)), 4)
        res["equal"][tier] = m2
        print("%-22s %10.4f %10.4f %10.4f   (equal weight)"
              % ("  " + tier + " (equal)", m2["auroc"], m2["aupr"], m2["oof_auroc"]))

    res["n_train"], res["n_test"] = int(len(y_tr)), int(len(y_te))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n[saved] %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
