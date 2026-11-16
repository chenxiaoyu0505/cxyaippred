# -*- coding: utf-8 -*-
"""Check out/report.json against expected_ranges.json.

Two kinds of check, both on test1.csv metrics:

    point   the three ML views are deterministic (same data, same code, same
            seed) -> AUROC must match the released number within `tol` (1e-3).
            A miss here means something in the ML path changed, so it is a FAIL.
    range   the two deep views train under SGD -> AUROC must land inside the
            documented band. A miss means the release no longer reproduces its
            own behaviour; it does not indicate a bug, and the measured value is
            printed either way.

Skipping: if ``out/report.json`` simply has no entry for a view or a fusion tier
(nothing trained it yet -- e.g. a code-only run with ``--skip-dl``), that check
is reported as ``SKIP (no probs)`` instead of FAIL, and it is *not* counted as
passed. A full ``run_all.py`` trains everything, so nothing is skipped there.

Exit code 0 = every non-skipped check passed, 1 = at least one failed.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402


def dig(obj, path):
    """Nested lookup; None when the report has no such entry."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return None
        cur = cur[key]
    return cur


def check_point(name, got, want, tol, rows):
    ok = abs(got - want) <= tol
    rows.append((name, "point", "%s+-%s" % (want, tol), got, ok))
    return ok


def check_range(name, got, rng, rows):
    lo, hi = float(rng[0]), float(rng[1])
    ok = lo <= got <= hi
    rows.append((name, "range", "[%s, %s]" % (lo, hi), got, ok))
    return ok


def skip(name, want, rows):
    rows.append((name, "?", want, None, None))


def main():
    rep = C.read_json(os.path.join(C.OUT, "report.json"))
    exp = C.read_json(os.path.join(C.ROOT, "expected_ranges.json"))
    tol = float(exp.get("meta", {}).get("tolerance", 1e-3))
    col = exp.get("meta", {}).get("fusion_column", "rank_greedy")

    rows, oks, skipped = [], [], []
    for view, spec in exp["views"].items():
        got = dig(rep, ("views", view, "test", "auroc"))
        if got is None:
            skipped.append("view/" + view)
            skip("view/" + view, spec.get("auroc", spec.get("auroc_range")), rows)
            continue
        got = float(got)
        if "auroc" in spec:
            oks.append(check_point("view/" + view, got, float(spec["auroc"]),
                                   float(spec.get("tol", tol)), rows))
        else:
            oks.append(check_range("view/" + view, got, spec["auroc_range"], rows))
    for tier, spec in exp["fusion"].items():
        got = dig(rep, ("fusion", tier, col, "auroc"))
        if got is None:
            skipped.append("fusion/" + tier)
            skip("fusion/" + tier, spec["auroc_range"], rows)
            continue
        oks.append(check_range("fusion/" + tier, float(got), spec["auroc_range"], rows))

    print("\n%-16s %-6s %-16s %9s" % ("check", "kind", "expected", "measured"))
    print("-" * 58)
    for name, kind, want, got, ok in rows:
        if ok is None:
            print("%-16s %-6s %-16s %9s  SKIP (no probs)" % (name, kind, want, "-"))
        else:
            print("%-16s %-6s %-16s %9.4f  %s" % (name, kind, want, got,
                                                  "ok" if ok else "FAIL"))
    n, good = len(oks), sum(oks)
    print("-" * 58)
    if skipped:
        print("[verify] %d/%d checks passed, %d SKIPPED (no probs: %s)"
              % (good, n, len(skipped), ", ".join(skipped)))
        print("[verify] skipped checks are not evidence -- run the full pipeline "
              "(no --skip-*) before quoting numbers")
    else:
        print("[verify] %d/%d checks passed" % (good, n))
    if n and good == n:
        print("[verify] PASS")
        return 0
    print("[verify] FAIL")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
