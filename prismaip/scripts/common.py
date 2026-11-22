# -*- coding: utf-8 -*-
"""Shared paths, folds and small metrics helpers.

The whole package is evaluated on one fixed split:

    train1.csv   3,278 peptides (1,060 positive)   <- everything is fitted here
    test1.csv      817 peptides (262 positive)     <- scored once, never fitted

Cross-validation is ``StratifiedKFold(5, shuffle=True, random_state=42)`` applied
to the train1 rows in file order, so the five folds are reproducible from the
csv alone.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
WEIGHTS = os.path.join(ROOT, "weights")
OUT = os.path.join(ROOT, "out")

NFOLD = 5
SKF_SEED = 42
VIEWS = ["hand", "gap4", "k5", "tlora", "cnn"]


def load_split(name="train1"):
    """Read train1/test1 with sequences stripped, upper-cased and validated."""
    df = pd.read_csv(os.path.join(DATA, name + ".csv"), dtype={"sequence": str})
    df["sequence"] = df.sequence.str.strip().str.upper()
    df["label"] = df.label.astype(int)
    df["length"] = df.sequence.str.len()
    return df


def skf_folds(y, n_folds=NFOLD, seed=SKF_SEED):
    """Fold id per row (0..n_folds-1) -- identical to the published protocol."""
    fold = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[va] = k
    return fold


def ensure_dirs():
    for d in (os.path.join(WEIGHTS, "ml"), os.path.join(WEIGHTS, "tlora"),
              os.path.join(WEIGHTS, "cnn"), OUT):
        os.makedirs(d, exist_ok=True)


def log_stdout(path):
    """Mirror stdout into a log file; returns the print function to use."""
    import sys
    fh = open(path, "w", encoding="utf-8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    return log
