# -*- coding: utf-8 -*-
"""Rebuild every weight this package ships, then refit the fusion layer.

    python scripts/train_all.py              # ML + CNN weights, fusion, expected scores
    python scripts/train_all.py --skip-cnn   # ML only (a few minutes, no GPU)

What gets written
-----------------
weights/ml/<view>.joblib.xz    one model per classical view, fitted on all of train1
weights/cnn/star_fold<k>.pt    the CNN, five fold models (0.1 MB each)
weights/fusion.json            rank weights, Platt map and tau, fitted on train1 OOF
weights/fusion_ecdf.npz        per-view empirical CDFs, for batch-independent scoring
out/test_probs.npz             per-view probabilities on test1
expected_scores.json           the numbers ``verify.py`` checks

The fusion layer is always fitted on the *cross-validated* out-of-fold scores
shipped in ``weights/oof.npz`` -- those are the scores behind every reported
number, and they are never test1 scores.  The view weights therefore describe the
five-view system as evaluated, while the ML models shipped here are refit on the
full train1 set for deployment (see README).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import fuse as FU           # noqa: E402
import ml_views as ML       # noqa: E402


def auc_ap(y, p):
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    log = C.log_stdout(os.path.join(C.OUT, "train_all.log"))
    tr, te = C.load_split("train1"), C.load_split("test1")
    y_tr, y_te = tr.label.values, te.label.values
    seqs_tr, seqs_te = list(tr.sequence), list(te.sequence)

    # ------------------------------------------------------- deployment weights
    if not args.skip_ml:
        t0 = time.time()
        ML.train_deploy(seed=args.seed, log=log)
        log("[ml] three deployment models fitted in %.0fs" % (time.time() - t0))
    import dl_views
    if not args.skip_cnn:
        t0 = time.time()
        dl_views.train_cnn(device=args.device, seed=args.seed)
        log("[cnn] five fold models fitted in %.0fs" % (time.time() - t0))

    # --------------------------------------------------- test1 probabilities
    test = {}
    Xh = ML.hand704(seqs_te)
    for v in ("hand", "gap4", "k5"):
        test[v] = ML.predict_view(v, seqs_te, ML.load_models(v))
    for name in ("tlora", "cnn"):
        test[name] = dl_views.score(name, seqs_te, device=args.device, quiet=False)

    # ------------------------------------------------------ fusion from CV OOF
    oof_npz = np.load(os.path.join(C.WEIGHTS, "oof.npz"))
    oof = {v: oof_npz["oof_" + v] for v in C.VIEWS}
    cfg, ens_oof, _ = FU.fit_weights(oof, y_tr, save_path=os.path.join(C.WEIGHTS, "fusion.json"))
    score, prob, dec = FU.fuse(test, cfg, mode="batch")
    auc, apr = auc_ap(y_te, prob)
    log("[fuse] weights=%s tau=%.4f" % ({k: round(v, 4) for k, v in cfg["weights"].items()}, cfg["tau"]))
    log("[fuse] test1 AUROC=%.4f AUPR=%.4f acc@tau=%.4f" % (auc, apr, float((dec == y_te).mean())))

    np.savez(os.path.join(C.OUT, "test_probs.npz"), y_te=y_te,
             **{"test_" + k: v for k, v in test.items()})
    exp = {"views": {}, "fusion": {}, "weights": {}, "n_test": int(len(te)),
           "n_pos_test": int(y_te.sum())}
    for v in C.VIEWS:
        a, p = auc_ap(y_te, test[v])
        exp["views"][v] = dict(auroc_test=round(a, 4), aupr_test=round(p, 4),
                               auroc_oof=round(auc_ap(y_tr, oof[v])[0], 4))
        log("[view] %-5s test AUROC=%.4f AUPR=%.4f (OOF AUROC=%.4f)"
            % (v, a, p, exp["views"][v]["auroc_oof"]))
    exp["fusion"] = dict(auroc_test=round(auc, 4), aupr_test=round(apr, 4),
                         auroc_oof=round(cfg["oof_auroc"], 4), tau=round(cfg["tau"], 4))
    exp["weights"] = {k: round(v, 4) for k, v in cfg["weights"].items()}
    with open(os.path.join(C.ROOT, "expected_scores.json"), "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2, sort_keys=True)
    log("[done] expected_scores.json written")


if __name__ == "__main__":
    main()
