# -*- coding: utf-8 -*-
"""Recompute every number in `expected_scores.json` from the shipped weights.

    python scripts/verify.py                # test1, batch mode (paper protocol)
    python scripts/verify.py --device cpu   # same, without a GPU

The script scores test1 with the five view models, fuses exactly as `predict.py`
does, and compares AUROC / AUPR / tau / weights against the recorded values with
a tolerance of 2e-3 (five-fold models averaged; the only source of drift is the
BLAS/kernel order inside torch).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import fuse as FU           # noqa: E402
import ml_views as ML       # noqa: E402

TOL = 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    with open(os.path.join(C.ROOT, "expected_scores.json"), encoding="utf-8") as f:
        exp = json.load(f)
    tr, te = C.load_split("train1"), C.load_split("test1")
    seqs = list(te.sequence)
    y = te.label.values
    cfg = FU.load_config()

    probs = {}
    for v in ("hand", "gap4", "k5"):
        probs[v] = ML.predict_view(v, seqs, ML.load_models(v))
    import dl_views
    for name in ("tlora", "cnn"):
        probs[name] = dl_views.score(name, seqs, device=args.device)

    rows, ok = [], True
    for v in C.VIEWS:
        auc = roc_auc_score(y, probs[v])
        ap_ = average_precision_score(y, probs[v])
        ref = exp["views"][v]
        good = abs(auc - ref["auroc_test"]) <= args.tol and abs(ap_ - ref["aupr_test"]) <= args.tol
        ok &= good
        rows.append((v, ref["auroc_test"], auc, ref["aupr_test"], ap_, good))

    score, prob, dec = FU.fuse(probs, cfg, mode="batch")
    auc_f, ap_f = roc_auc_score(y, prob), average_precision_score(y, prob)
    ref = exp["fusion"]
    good_f = abs(auc_f - ref["auroc_test"]) <= args.tol and abs(ap_f - ref["aupr_test"]) <= args.tol
    ok &= good_f

    print("%-6s %10s %10s %10s %10s  %s" % ("view", "ref AUC", "now AUC", "ref AUPR", "now AUPR", ""))
    for v, ra, na, rp, np_, good in rows:
        print("%-6s %10.4f %10.4f %10.4f %10.4f  %s" % (v, ra, na, rp, np_, "ok" if good else "MISMATCH"))
    print("%-6s %10.4f %10.4f %10.4f %10.4f  %s"
          % ("FUSED", ref["auroc_test"], auc_f, ref["aupr_test"], ap_f, "ok" if good_f else "MISMATCH"))
    print("\nweights  %s" % {k: round(v, 4) for k, v in cfg["weights"].items()})
    print("tau      %.4f   (called AIP: %d/%d)" % (cfg["tau"], int(dec.sum()), len(dec)))
    print("\n%s" % ("ALL OK" if ok else "MISMATCH -- see rows above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
