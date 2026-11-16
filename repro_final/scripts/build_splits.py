# -*- coding: utf-8 -*-
"""Rebuild train1/test1 from the shipped inputs and check `data/manifest.json`.

`data/train1.csv` and `data/test1.csv` are **not** independent artifacts: they are
literally `train0`/`test0` minus the 385 AIP+/PIP+ conflict positives, plus the
official DeepAIP fold id.  This script proves that, byte for byte.

Usage
-----
    python scripts/build_splits.py            # verify (default; what run_all.py calls)
    python scripts/build_splits.py --write    # regenerate train1/test1 + manifest.json

Inputs (all under `data/`)
--------------------------
    train0.csv        3,583 rows  sequence,label,length   official DeepAIP train pool
    test0.csv           897 rows  sequence,label,length   official DeepAIP test
    conflicts_385.csv   385 rows  sequence                AIP+/PIP+ overlap sequences
    pool_folds.csv    3,583 rows  sequence,fold           official 5-fold assignment

Outputs
-------
    train1.csv        3,278 rows  sequence,label,length,fold   (3,583 - 305)
    test1.csv           817 rows  sequence,label,length        (897 - 80)
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
COLS = ["sequence", "label", "length"]
N_TRAIN1, N_TEST1 = 3278, 817
POS_TRAIN1, POS_TEST1 = 1060, 262
N_CONFLICT = 385
FILES = ["train0.csv", "test0.csv", "conflicts_385.csv", "pool_folds.csv",
         "train1.csv", "test1.csv"]


def sha256(path):
    """sha256 of the file in canonical LF form.

    Git may rewrite line endings on checkout (`core.autocrlf`, `* text=auto`), which
    changes the raw bytes without changing the data.  Hashing the LF-normalised form
    makes the manifest say the same thing on every platform.
    """
    h = hashlib.sha256()
    carry = b""
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            buf = carry + chunk
            if buf.endswith(b"\r"):
                carry, buf = b"\r", buf[:-1]
            else:
                carry = b""
            h.update(buf.replace(b"\r\n", b"\n"))
    h.update(carry)
    return h.hexdigest()


def read(name):
    df = pd.read_csv(os.path.join(DATA, name), dtype={"sequence": str})
    df["sequence"] = df.sequence.str.strip().str.upper()
    return df


def build(train0, test0, conflicts, pool_folds):
    """Apply the conflict filter and attach the official fold id."""
    drop = set(conflicts.sequence)
    fold_of = dict(zip(pool_folds.sequence, pool_folds.fold.astype(int)))

    train1 = train0.loc[~train0.sequence.isin(drop), COLS].reset_index(drop=True)
    test1 = test0.loc[~test0.sequence.isin(drop), COLS].reset_index(drop=True)
    train1["fold"] = pd.Series([fold_of[s] for s in train1.sequence], dtype="int64")
    train1 = train1[COLS + ["fold"]]

    assert len(conflicts) == N_CONFLICT, len(conflicts)
    assert train1.sequence.is_unique and test1.sequence.is_unique
    assert not set(train1.sequence) & set(test1.sequence), "train/test overlap"
    assert len(train1) == N_TRAIN1, len(train1)
    assert len(test1) == N_TEST1, len(test1)
    assert int(train1.label.sum()) == POS_TRAIN1, int(train1.label.sum())
    assert int(test1.label.sum()) == POS_TEST1, int(test1.label.sum())
    assert set(train1.fold) == {0, 1, 2, 3, 4}, set(train1.fold)
    return train1, test1


def dump(df, name):
    path = os.path.join(DATA, name)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def write_manifest():
    man = {}
    for name in FILES:
        df = read(name)
        man[name] = {
            "rows": int(len(df)),
            "positives": int(df.label.sum()) if "label" in df.columns else None,
            "sha256": sha256(os.path.join(DATA, name)),
        }
    man["note"] = ("train1 = train0 - conflicts_385 (305 rows removed, 3,583 -> 3,278); "
                   "test1 = test0 - conflicts_385 (80 rows removed, 897 -> 817). "
                   "fold comes from pool_folds.csv (= DeepAIP official folds.csv val split).")
    with open(os.path.join(DATA, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return man


def verify():
    man = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    train0, test0 = read("train0.csv"), read("test0.csv")
    conflicts, pool_folds = read("conflicts_385.csv"), read("pool_folds.csv")
    train1, test1 = build(train0, test0, conflicts, pool_folds)

    ok = True
    print("%-18s %8s %8s  %-10s %s" % ("file", "rows", "pos", "sha256", "check"))
    for name in FILES:
        path = os.path.join(DATA, name)
        got = sha256(path)
        exp = man[name]
        df = read(name)
        rows = len(df)
        pos = int(df.label.sum()) if "label" in df.columns else None
        good = (got == exp["sha256"] and rows == exp["rows"] and pos == exp["positives"])
        ok &= good
        print("%-18s %8d %8s  %s  %s" % (name, rows, "-" if pos is None else pos,
                                         got[:16], "OK" if good else "MISMATCH"))

    # regenerated frames must be byte-identical to the shipped ones
    for name, df in (("train1.csv", train1), ("test1.csv", test1)):
        shipped = read(name)
        same = df.reset_index(drop=True).equals(shipped.reset_index(drop=True))
        ok &= same
        print("[rebuild] %-11s reproducible from train0/test0+conflicts: %s"
              % (name, "OK" if same else "MISMATCH"))

    print("[counts]  train1 %d rows / %d pos  (expect %d / %d)"
          % (len(train1), int(train1.label.sum()), N_TRAIN1, POS_TRAIN1))
    print("[counts]  test1  %d rows / %d pos  (expect %d / %d)"
          % (len(test1), int(test1.label.sum()), N_TEST1, POS_TEST1))
    print("[counts]  removed by conflict filter: train0 %d, test0 %d (total %d)"
          % (len(train0) - len(train1), len(test0) - len(test1),
             len(train0) - len(train1) + len(test0) - len(test1)))
    print("\n[build_splits] %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate train1/test1 + manifest.json")
    args = ap.parse_args()
    if args.write:
        train1, test1 = build(read("train0.csv"), read("test0.csv"),
                              read("conflicts_385.csv"), read("pool_folds.csv"))
        dump(train1, "train1.csv")
        dump(test1, "test1.csv")
        man = write_manifest()
        for k, v in man.items():
            if isinstance(v, dict):
                print("%-18s %6d rows  %s" % (k, v["rows"], v["sha256"][:16]))
        return 0
    return verify()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
