# -*- coding: utf-8 -*-
"""Command-line entry point: score a peptide table with PrismAIP.

    python scripts/predict.py --input my_peptides.csv
    python scripts/predict.py --input my_peptides.csv --mode fixed --out preds.csv

Input: a csv with one peptide per row (column ``sequence``; extra columns are
carried through).  Output: the same rows plus one probability per view, the fused
score, the calibrated probability and the binary decision at the recommended
threshold.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import fuse as FU           # noqa: E402
import ml_views as ML       # noqa: E402


def read_input(path, seq_col="sequence"):
    df = pd.read_csv(path, dtype=str).fillna("")
    if seq_col not in df.columns:
        raise SystemExit("input needs a %r column (have: %s)" % (seq_col, list(df.columns)))
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    bad = df[seq_col].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)
    if bad.any():
        print("[warn] %d row(s) contain non-standard residues; they are kept, "
              "the CNN maps them to an 'unknown' token." % int(bad.sum()))
    return df, seq_col


def main():
    ap = argparse.ArgumentParser(description="PrismAIP: five-view anti-inflammatory peptide predictor")
    ap.add_argument("--input", required=True, help="csv with a 'sequence' column")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--out", default=None, help="output csv (default: <input>_prismaip.csv)")
    ap.add_argument("--mode", default="batch", choices=["batch", "fixed"],
                    help="batch = ranks within this file (reproduces the paper); "
                         "fixed = absolute ranks from train1 ECDF (batch-independent)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--views", default="hand,gap4,k5,tlora,cnn",
                    help="comma separated subset of the five views")
    ap.add_argument("--weights", default=None, help="weights directory (default: prismaip/weights)")
    ap.add_argument("--fusion", default=None, help="fusion.json (default: weights/fusion.json)")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    df, seq_col = read_input(args.input, args.seq_col)
    seqs = list(df[seq_col])
    print("[data] %d peptide(s), length %d-%d (median %d)"
          % (len(seqs), min(map(len, seqs)), max(map(len, seqs)),
             int(np.median([len(s) for s in seqs]))))

    if args.weights:
        C.WEIGHTS = os.path.abspath(args.weights)
    cfg = FU.load_config(args.fusion)
    probs = {}
    Xh = None
    if "hand" in views or "gap4" in views or "k5" in views:
        if "hand" in views:
            Xh = ML.hand704(seqs)
        for v in ("hand", "gap4", "k5"):
            if v in views:
                probs[v] = ML.predict_view(v, seqs, ML.load_models(v))
                print("[view] %-4s mean=%.3f" % (v, float(np.mean(probs[v]))))
    # torch-based views run in their own process: on Windows, importing sklearn
    # before torch in one process can fail to load c10.dll.
    import dl_views
    for name in ("tlora", "cnn"):
        if name in views:
            probs[name] = dl_views.score(name, seqs, device=args.device)
            print("[view] %-5s mean=%.3f" % (name, float(np.mean(probs[name]))))

    missing = [v for v in cfg["views"] if v not in probs]
    if missing:
        raise SystemExit("fusion expects views %s but these were not computed: %s"
                         % (cfg["views"], missing))
    score, prob, dec = FU.fuse(probs, cfg, mode=args.mode)

    out = df.copy()
    for v in cfg["views"]:
        out["p_" + v] = np.round(probs[v], 6)
    out["prismaip_score"] = np.round(score, 6)
    out["prismaip_prob"] = np.round(prob, 6)
    out["prismaip_call"] = np.where(dec == 1, "AIP", "non-AIP")
    path = args.out or os.path.splitext(args.input)[0] + "_prismaip.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print("[out] %s" % path)
    print("[out] tau=%.4f  ->  %d/%d called AIP (%.1f %%)"
          % (cfg["tau"], int(dec.sum()), len(dec), 100.0 * dec.mean()))


if __name__ == "__main__":
    main()
