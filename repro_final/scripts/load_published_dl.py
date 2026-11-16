# -*- coding: utf-8 -*-
"""Load the *frozen* released DL probabilities and rebuild their OOF/test vectors.

Why this file exists
--------------------
The two deep views of the paper (TLoRA and the small CNN) were trained in
`_work/wt-drop385/grid_dl` and `_work/wt-drop385/grid_cnn`.  Neither run ever wrote
a model checkpoint -- grid_dl kept only `fold{k}.json` + `fold{k}_{val,test}_probs.npy`,
grid_cnn only `fold{k}.npz` + `probs_A.npy`.  Their weights are therefore gone; the
released *predictions* survive and are shipped here under `data/published/`.

This script is the audit trail for them:

* TLoRA -- 5-fold cross-validation over train1 with
  `StratifiedKFold(5, shuffle=True, random_state=42)` (the released arm did **not**
  use the official DeepAIP folds; that is recorded in `grid_dl/T1/results.json`).
  The fold assignment is re-derived here and every fold's OOF AUROC is asserted
  against the value recorded by the original run (0.8212 / 0.7889 / 0.8102 /
  0.7713 / 0.8243).  Test probability = mean over the 5 fold models, rows 0..816
  of the 1,002-row test2 prediction (test1 is the first 817 rows of test2).
* CNN -- 5-fold over train1 using the **official** folds (`train1.csv:fold`).
  Each `cnn_grid_A_fold{k}.npz` holds `val_probs (seeds, n_val)`, `test_probs
  (seeds, 1002)` and `va_idx`; seed 0 is the released arm.

Outputs `out/dl_published.npz` with `oof_tlora, test_tlora, oof_cnn, test_cnn`.

If you would rather trust a checkpoint than a frozen prediction, run
`scripts/run_dl_views.py` -- it re-infers from the surviving checkpoints
(`weights/tlora_fold*.pt`, `weights/cnn_folds.pt`) and gets a *different model
instance* (see `weights/README.md`).

Usage:  python scripts/load_published_dl.py [--out out/dl_published.npz]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
PUB = os.path.join(DATA, "published")
OUT = os.path.join(ROOT, "out")

NFOLD = 5
# values recorded by the released runs (grid_dl/T1/{k}.json, grid_cnn/A)
TLORA_FOLD_AUROC = [0.8212, 0.7889, 0.8102, 0.7713, 0.8243]
TOL = 6e-4


def _auroc(y, p):
    return float(roc_auc_score(y, p))


def load_tlora(tr, te):
    y_tr, y_te = tr.label.values.astype(int), te.label.values.astype(int)
    oof = np.full(len(tr), np.nan, dtype=np.float64)
    test = np.zeros(len(te), dtype=np.float64)
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)
    folds = []
    for k, (_tr_i, va_i) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr)):
        vp = np.load(os.path.join(PUB, "tlora_grid_dl_fold%d_val_probs.npy" % k))
        tp = np.load(os.path.join(PUB, "tlora_grid_dl_fold%d_test_probs.npy" % k))
        if len(vp) != len(va_i):
            raise RuntimeError("fold%d val_probs %d != %d" % (k, len(vp), len(va_i)))
        a = _auroc(y_tr[va_i], vp)
        if abs(a - TLORA_FOLD_AUROC[k]) > TOL:
            raise RuntimeError("fold%d OOF AUROC %.4f != recorded %.4f -- the frozen "
                              "probs and the re-derived fold split disagree"
                              % (k, a, TLORA_FOLD_AUROC[k]))
        oof[va_i] = vp
        test += tp[:len(te)]
        folds.append(a)
    if np.isnan(oof).any():
        raise RuntimeError("OOF assembly left holes: %d" % int(np.isnan(oof).sum()))
    test /= NFOLD
    return oof, test, folds, y_tr, y_te


def load_cnn(tr, te):
    y_tr, y_te = tr.label.values.astype(int), te.label.values.astype(int)
    oof = np.full(len(tr), np.nan, dtype=np.float64)
    test = np.zeros(len(te), dtype=np.float64)
    folds = []
    for k in range(NFOLD):
        z = np.load(os.path.join(PUB, "cnn_grid_A_fold%d.npz" % k))
        va_idx, vp, tp = z["va_idx"], z["val_probs"][0], z["test_probs"][0]
        folds.append(_auroc(y_tr[va_idx], vp))
        oof[va_idx] = vp
        test += tp[:len(te)]
    if np.isnan(oof).any():
        raise RuntimeError("OOF assembly left holes: %d" % int(np.isnan(oof).sum()))
    test /= NFOLD
    return oof, test, folds, y_tr, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "dl_published.npz"))
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(DATA, "train1.csv"))
    te = pd.read_csv(os.path.join(DATA, "test1.csv"))
    if len(tr) != 3278 or len(te) != 817:
        raise RuntimeError("unexpected cohort size %d / %d" % (len(tr), len(te)))

    oof_t, test_t, f_t, y_tr, y_te = load_tlora(tr, te)
    oof_c, test_c, f_c, _, _ = load_cnn(tr, te)

    res = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "tlora": {"test_auroc": round(_auroc(y_te, test_t), 4),
                     "test_aupr": round(float(average_precision_score(y_te, test_t)), 4),
                     "oof_auroc": round(_auroc(y_tr, oof_t), 4),
                     "fold_oof_auroc": [round(a, 4) for a in f_t],
                     "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
                     "source": "grid_dl/T1/fold{k}_{val,test}_probs.npy"},
           "cnn": {"test_auroc": round(_auroc(y_te, test_c), 4),
                   "test_aupr": round(float(average_precision_score(y_te, test_c)), 4),
                   "oof_auroc": round(_auroc(y_tr, oof_c), 4),
                   "fold_oof_auroc": [round(a, 4) for a in f_c],
                   "cv": "official DeepAIP folds (train1.csv:fold)",
                   "source": "grid_cnn/A/fold{k}.npz (seed 0)"}}

    print("%-8s %10s %10s %10s   %s" % ("arm", "test_auroc", "test_aupr",
                                        "oof_auroc", "per-fold OOF AUROC"))
    for name, m in (("TLoRA", res["tlora"]), ("CNN", res["cnn"])):
        print("%-8s %10.4f %10.4f %10.4f   %s"
              % (name, m["test_auroc"], m["test_aupr"], m["oof_auroc"],
                 " ".join("%.4f" % a for a in m["fold_oof_auroc"])))

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(args.out, oof_tlora=oof_t, test_tlora=test_t,
                        oof_cnn=oof_c, test_cnn=test_c)
    with open(os.path.join(OUT, "dl_published.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n[saved] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
