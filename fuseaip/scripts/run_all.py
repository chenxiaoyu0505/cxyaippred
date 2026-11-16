# -*- coding: utf-8 -*-
"""One entry point: train all five views from scratch, fuse, score, verify.

    python fuseaip/scripts/run_all.py                 # full run (~20-25 min GPU)
    python fuseaip/scripts/run_all.py --skip-ml       # deep views only
    python fuseaip/scripts/run_all.py --reuse         # score weights/trained/, no training
    python fuseaip/scripts/run_all.py --folds 0       # smoke: 1 fold, verify will fail

Stages, in order:

    1  build_splits.py     sha256 + row/positive check of data/*.csv
    2  train_ml_views.py   hand704 / gap3 / k4, ExtraTrees, 5 folds   (~5 min CPU)
    3  train_tlora.py      ESM-2 650M + LoRA + posproj head, 5 folds  (~10 min GPU)
    4  train_cnn.py        small CNN from scratch, 5 folds            (~1 min CPU)
    5  fuse_eval.py        3/4/5-view rank fusion -> out/report.json
    6  verify.py           out/report.json vs expected_ranges.json

Every stage writes its artifacts under ``fuseaip/out`` (git-ignored) or
``fuseaip/weights/trained`` (also git-ignored: weights are not shipped).
Nothing is read from a frozen probability file: the numbers in the paper are
reproduced by retraining, and the deep-view targets are ranges rather than
points. ``--reuse`` scores the archived weights instead of retraining them.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="device for the TLoRA view")
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma separated fold ids")
    ap.add_argument("--skip-ml", action="store_true", help="reuse out/probs/ml_views.npz")
    ap.add_argument("--skip-dl", action="store_true", help="reuse the deep-view probabilities")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="load weights/trained/ instead of training the deep views")
    ap.add_argument("--no-lock", action="store_true", help="do not take _work/gpu.lock")
    args = ap.parse_args()

    full = str(args.folds).replace(" ", "") == "0,1,2,3,4"
    steps = [("build_splits.py", [], False, False)]
    if not args.skip_ml:
        steps.append(("train_ml_views.py", ["--folds", args.folds], False, True))
    else:
        print("[skip] ML views: reusing out/probs/ml_views.npz", flush=True)
    if not args.skip_dl:
        dargs = ["--folds", args.folds, "--device", args.device]
        if args.no_lock:
            dargs.append("--no-lock")
        if args.reuse:
            dargs.append("--reuse")
        steps.append(("train_tlora.py", dargs, not args.reuse, True))
        steps.append(("train_cnn.py", ["--folds", args.folds] + (["--reuse"] if args.reuse else []),
                      False, True))
    else:
        print("[skip] deep views: reusing out/probs/{tlora,cnn}_views.npz", flush=True)
    steps.append(("fuse_eval.py", [], False, False))
    if not args.skip_verify:
        steps.append(("verify.py", [], False, False))

    C.ensure_dirs()
    logpath = os.path.join(C.LOGS, "run_all.log")
    fh = C.open_log(logpath)
    fh.write("\n\n===== run_all start %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    fh.flush()
    sys.stdout = C.Tee(sys.__stdout__, fh)

    t0 = time.time()
    timings = []
    failed = False
    try:
        for script, extra, needs_gpu, is_train in steps:
            cmd = [sys.executable, "-X", "utf8", os.path.join(HERE, script)] + extra
            print("\n" + "-" * 78, flush=True)
            print(">> %s %s" % (script, " ".join(extra)), flush=True)
            print("-" * 78, flush=True)
            t = time.time()
            rc = subprocess.run(cmd, cwd=C.ROOT).returncode
            secs = time.time() - t
            timings.append(dict(script=script, seconds=round(secs, 1), rc=rc,
                                gpu=bool(needs_gpu)))
            print("[%s] %.1fs (rc=%d)" % (script, secs, rc), flush=True)
            if rc != 0:
                failed = True
                print("[run_all] stage failed: %s -- stopping" % script, flush=True)
                break
            C.write_json(os.path.join(C.OUT, "timings.json"),
                         dict(total_sec=round(time.time() - t0, 1), stages=timings,
                              folds=args.folds, device=args.device))
    finally:
        total = time.time() - t0
        C.write_json(os.path.join(C.OUT, "timings.json"),
                     dict(total_sec=round(total, 1), stages=timings,
                          folds=args.folds, device=args.device))
        print("\n%-22s %8s" % ("stage", "seconds"))
        for s in timings:
            print("%-22s %8.1f" % (s["script"], s["seconds"]))
        print("%-22s %8.1f  (%.1f min)" % ("TOTAL", total, total / 60.0))
        if failed:
            print("\n[run_all] FAILED after %.1f min -- see out/logs/" % (total / 60.0))
        else:
            print("\n[run_all] done in %.1f min -- report: out/report.json"
                  % (total / 60.0))
        sys.stdout = sys.__stdout__
        fh.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
