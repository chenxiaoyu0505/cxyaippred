# -*- coding: utf-8 -*-
"""Compare this run against `expected.json` and print a PASS/FAIL table.

Nothing here is read from the expected file: all measured values come from the
artifacts the run just produced.  Two independent groups are checked, both at the
tolerance in `meta.tolerance` (1e-3 on AUROC/AUPR):

  paper.*     the released numbers of the paper table (view5_t1.json), measured on
              `out/fused_published.json` -- ML views retrained from sequence, the two
              deep views from the frozen released probabilities.
  weights.*   what the *surviving checkpoints* produce when re-inferred from scratch
              (`out/fused_weights.json`).  These differ for TLoRA / CNN because the
              paper's deep models were never checkpointed -- see README
              "Deep views: what is actually reproducible".

Exit code: 0 = every check passed in both groups, 1 = at least one mismatch.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
SKIP = ("note", "weights", "source", "cv", "dl_source", "tolerance")

GROUPS = (
    ("paper", "fused_published.json", "PAPER TABLE -- view5_t1.json"),
    ("weights", "fused_weights.json", "SURVIVING CHECKPOINTS -- re-inferred"),
)


def walk(exp, got, path, tol, rows):
    for k, v in exp.items():
        if k in SKIP or k in ("published",):
            continue
        p = "%s.%s" % (path, k) if path else k
        if isinstance(v, dict):
            if not isinstance(got, dict) or k not in got:
                rows.append((p, "-", "MISSING", None, False))
                continue
            walk(v, got[k], p, tol, rows)
        else:
            g = got.get(k) if isinstance(got, dict) else None
            if g is None:
                rows.append((p, v, "MISSING", None, False))
                continue
            d = abs(float(g) - float(v))
            rows.append((p, v, g, d, d <= tol))


def main():
    exp = json.load(open(os.path.join(ROOT, "expected.json"), encoding="utf-8"))
    tol = float(exp.get("meta", {}).get("tolerance", 1e-3))
    print("tolerance |d| <= %.0e   (expected.json vs this run)\n" % tol)

    total = fails = 0
    for key, fname, title in GROUPS:
        if key not in exp:
            continue
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            print("[verify] MISSING artifact %s -- cannot check '%s'" % (fname, key))
            total += 1
            fails += 1
            continue
        got = json.load(open(path, encoding="utf-8"))
        gtol = float(exp[key].get("tolerance", tol))
        rows = []
        for group in ("views", "fusion", "equal"):
            if group in exp[key]:
                walk(exp[key][group], got.get(group, {}), group, gtol, rows)
        print("== %s (%s, tol %.0e) ==" % (title, fname, gtol))
        print("%-30s %10s %10s %10s  %s"
              % ("item", "expected", "measured", "|d|", "check"))
        print("-" * 74)
        for p, e, g, d, ok in rows:
            print("%-30s %10s %10s %10s  %s"
                  % (p, ("%.4f" % e) if isinstance(e, float) else e,
                     ("%.4f" % g) if isinstance(g, float) else g,
                     "-" if d is None else "%.1e" % d, "PASS" if ok else "FAIL"))
        bad = sum(0 if r[4] else 1 for r in rows)
        total += len(rows)
        fails += bad
        print("-- %d/%d passed\n" % (len(rows) - bad, len(rows)))

    print("[verify] %d/%d checks passed" % (total - fails, total))
    if fails:
        print("[verify] FAIL -- %d mismatch(es) beyond tolerance %.0e" % (fails, tol))
        return 1
    print("[verify] PASS -- paper table reproduced, and the shipped checkpoints "
          "reproduce their own recorded numbers")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
