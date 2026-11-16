# -*- coding: utf-8 -*-
"""Check that the shipped data is the data this release is built on, and print
the cross-validation folds every trainer will derive.

This package trains everything from scratch, so the only way a run can drift is
a changed input (or, silently, a different fold assignment). The data files are
pinned by content hash:

    train1.csv         3,278 rows / 1,060 positives   (no fold column)
    test1.csv            817 rows /   262 positives
    conflicts_385.csv    385 rows, documentation only

The folds themselves are *derived*, not stored: StratifiedKFold(5, shuffle=True,
random_state=42) over the train1.csv rows in file order. They are printed here
with a short hash per fold and compared to the reference block in the manifest --
advisory only, because the hash depends on the scikit-learn/numpy release.

Writes ``out/data_check.json``; exits non-zero on any data mismatch.
"""
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(name, spec, rows=None, positives=None):
    path = os.path.join(C.DATA, name)
    assert os.path.exists(path), "missing " + path
    got = sha256(path)
    ok_sha = got == spec["sha256"]
    out = dict(file=name, rows=rows, positives=positives, sha256=got,
               sha256_expected=spec["sha256"], sha256_ok=bool(ok_sha))
    if rows is not None:
        out["rows_ok"] = rows == spec["rows"]
        out["positives_ok"] = (True if "positives" not in spec
                               else positives == spec["positives"])
        print("  %-20s %5d rows / %4s pos  sha256 %s  %s"
              % (name, rows, positives if positives is not None else "-", got[:12],
                 "ok" if ok_sha and out["rows_ok"] and out["positives_ok"] else "MISMATCH"))
    else:
        print("  %-20s %5s rows        sha256 %s  %s"
              % (name, "-", got[:12], "ok" if ok_sha else "MISMATCH"))
    return out


def main():
    C.ensure_dirs()
    log = C.log_stdout(os.path.join(C.LOGS, "splits.log"))
    try:
        C.stage("data integrity (sha256 + rows + positives) and CV folds")
        man = C.read_json(os.path.join(C.DATA, "manifest.json"))
        f = man["files"]

        tr = pd.read_csv(os.path.join(C.DATA, "train1.csv"), dtype={"sequence": str})
        te = pd.read_csv(os.path.join(C.DATA, "test1.csv"), dtype={"sequence": str})
        cf = pd.read_csv(os.path.join(C.DATA, "conflicts_385.csv"))
        print("[splits] train1=%d test1=%d conflicts=%d" % (len(tr), len(te), len(cf)), flush=True)

        checks = [
            check_file("train1.csv", f["train1.csv"], len(tr), int(tr.label.sum())),
            check_file("test1.csv", f["test1.csv"], len(te), int(te.label.sum())),
            check_file("conflicts_385.csv", f["conflicts_385.csv"], len(cf)),
        ]

        # Folds are derived, never read from the data; print them so two machines
        # can compare, and compare against the manifest's reference block.
        y_tr = tr.label.values.astype(int)
        fold_of = C.skf_folds(y_tr)
        table = C.fold_table(fold_of, y_tr)
        C.print_folds(table, prefix="  cv")
        ref = man.get("cv", {}).get("reference_folds")
        cv_match = None
        if ref:
            cv_match = all(int(ref[str(r["fold"])]["n_val"]) == r["n_val"]
                           and int(ref[str(r["fold"])]["n_pos"]) == r["n_pos"]
                           and ref[str(r["fold"])]["sha"] == r["sha"] for r in table)
            print("  %-20s %s  %s" % ("cv vs manifest", "match" if cv_match else "differs",
                                      "ok" if cv_match else
                                      "ADVISORY (different scikit-learn release reshuffled "
                                      "the folds; numbers are not comparable)"))
        assert set(fold_of.tolist()) == set(range(C.NFOLD))

        overlap = len(set(tr.sequence.str.upper()) & set(te.sequence.str.upper()))
        print("  %-20s %d  %s" % ("train/test overlap", overlap, "ok" if overlap == 0 else "MISMATCH"))

        nan = int(tr.sequence.isna().sum() + te.sequence.isna().sum())
        bad_len = int((tr.length != tr.sequence.str.len()).sum() + (te.length != te.sequence.str.len()).sum())
        print("  %-20s %d  %s" % ("NaN sequences", nan, "ok" if nan == 0 else "MISMATCH"))
        print("  %-20s %d  %s" % ("length mismatch", bad_len, "ok" if bad_len == 0 else "MISMATCH"))

        # Only the data itself is a hard failure; the fold hash is advisory.
        ok = (all(c["sha256_ok"] and c.get("rows_ok", True) and c.get("positives_ok", True)
                  for c in checks) and overlap == 0 and nan == 0 and bad_len == 0)
        res = dict(ok=bool(ok), files=checks,
                   cv=dict(scheme=man["cv"]["scheme"], folds=table, match_manifest=cv_match),
                   train_test_overlap=overlap, nan_sequences=nan,
                   length_mismatch=bad_len)
        C.write_json(os.path.join(C.OUT, "data_check.json"), res)
        print("[splits] %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            print("[splits] data does not match data/manifest.json -- refusing to train")
            return 1
    finally:
        sys.stdout = sys.__stdout__
        log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
