# -*- coding: utf-8 -*-
"""One command to reproduce the released numbers:

    python repro_final/scripts/run_all.py          # from the repository root
    python scripts/run_all.py                      # from repro_final/

Stages (each in a fresh interpreter, so a crash names the failing stage):

    0  build_splits.py        check data/manifest.json + rebuild train1/test1
    1  run_ml_views.py        hand704 / gap3 / k4 ExtraTrees, 5 folds     (~13 min, CPU)
    2  run_dl_views.py        TLoRA + CNN inference from weights/         (~1 min, GPU)
    3  load_published_dl.py   frozen released DL probabilities -> OOF/test
    4  fuse_eval.py --dl published   rank fusion of the paper table        (seconds)
    5  fuse_eval.py --dl weights     rank fusion of the re-inferred arms   (seconds)
    6  verify.py              PASS/FAIL against expected.json              (seconds)

Stage 2 needs an NVIDIA GPU (fp16 ESM-2 650M).  `--skip-dl` skips it and then only
the `paper` group of expected.json is checked (the `weights` group needs stage 2).
`--device cpu` works too but TLoRA on CPU is slow.

Exit code: the first non-zero stage aborts with that code.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (script, extra args, label, needs-dl)
STEPS = [
    ("build_splits.py", [], "data check (manifest + rebuild)", False),
    ("run_ml_views.py", [], "hand704 / gap3 / k4 ExtraTrees", False),
    ("run_dl_views.py", [], "TLoRA + CNN inference from weights/", True),
    ("load_published_dl.py", [], "frozen released DL probabilities", False),
    ("fuse_eval.py", ["--dl", "published", "--out",
                      os.path.join(ROOT, "out", "fused_published.json")],
     "rank fusion -- paper table", False),
    ("fuse_eval.py", ["--dl", "weights", "--out",
                      os.path.join(ROOT, "out", "fused_weights.json")],
     "rank fusion -- shipped checkpoints", True),
    ("verify.py", [], "verify against expected.json", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="passed to run_dl_views.py (cuda/cpu)")
    ap.add_argument("--skip-ml", action="store_true", help="reuse out/ml_views.npz")
    ap.add_argument("--skip-dl", action="store_true", help="reuse out/dl_views.npz")
    args = ap.parse_args()

    t0 = time.time()
    print("[run_all] python = %s" % sys.executable, flush=True)
    print("[run_all] root   = %s\n" % ROOT, flush=True)
    n = len(STEPS)
    for i, (script, extra, label, needs_dl) in enumerate(STEPS):
        skip = (args.skip_ml and script == "run_ml_views.py") or \
               (args.skip_dl and needs_dl)
        if skip:
            print("[run_all] (%d/%d) %-22s SKIPPED -- %s"
                  % (i + 1, n, script, label), flush=True)
            continue
        cmd = [sys.executable, os.path.join(HERE, script)] + extra
        if script == "run_dl_views.py" and args.device:
            cmd += ["--device", args.device]
        print("=" * 78)
        print("[run_all] (%d/%d) %s -- %s" % (i + 1, n, script, label))
        print("=" * 78, flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("\n[run_all] FAILED at %s (exit %d, %.1f min total)"
                  % (script, r.returncode, (time.time() - t0) / 60))
            return r.returncode
        print("[run_all] %s ok in %.1f min\n" % (script, (time.time() - t) / 60),
              flush=True)
    print("[run_all] all stages done in %.1f min" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
