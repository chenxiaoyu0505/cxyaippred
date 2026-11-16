# -*- coding: utf-8 -*-
"""Shared paths, the cross-worktree GPU lock, the CV protocol and tiny IO helpers.

Import-cheap on purpose:

* the two deep-view trainers import ``torch`` themselves **before** numpy /
  scikit-learn (Windows DLL init order -- see ``train_tlora.py``);
* the ML / fusion / verify scripts must not pay for a torch import at all;
* numpy/scikit-learn are imported *inside* the fold helpers, so a script that
  only needs paths or logging never pays for them.

Artifacts (all under ``fuseaip/out``, which is git-ignored):

    out/probs/{ml,tlora,cnn}_views.npz   per-view OOF + test probabilities
    weights/trained/*.pt                 trained deep-view weights (archived)
    out/report.json                      every metric verify.py checks
    out/logs/*.log                       per-stage console logs

The GPU lock lives in the *main* checkout (``<repo>/_work/gpu.lock``), so every
worktree of this checkout shares one physical GPU. ``git rev-parse
--git-common-dir`` is used to find that checkout from inside a worktree; set
``FUSEAIP_GPU_LOCK`` to override.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # .../fuseaip
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
PROBS = os.path.join(OUT, "probs")
MODELS = os.path.join(ROOT, "weights", "trained")   # archived weights, see README
LOGS = os.path.join(OUT, "logs")


def _main_checkout():
    """Root of the *main* checkout, also when we run inside a linked worktree."""
    try:
        p = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-common-dir"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        if p:
            p = os.path.abspath(os.path.join(ROOT, p))
            return os.path.dirname(p) if os.path.basename(p) == ".git" else p
    except Exception:
        pass
    return os.path.dirname(ROOT)


REPO = _main_checkout()                         # the cxyaippred checkout
LOCK = os.environ.get("FUSEAIP_GPU_LOCK", os.path.join(REPO, "_work", "gpu.lock"))


def ensure_dirs(*extra):
    for d in (OUT, PROBS, MODELS, LOGS) + extra:
        os.makedirs(d, exist_ok=True)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def stage(name):
    print("\n" + "=" * 78, flush=True)
    print("== %s  (%s)" % (name, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------- GPU lock
def pid_alive(pid):
    """True if ``pid`` is still running. Conservative: unknown -> alive."""
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
        return str(pid) in out
    except Exception:
        return True


def lock_info():
    """(pid, age_seconds) of the current lock holder, or (None, None)."""
    try:
        txt = open(LOCK, encoding="utf-8").read().strip()
        pid = int(txt.splitlines()[0]) if txt else -1
        age = time.time() - os.path.getmtime(LOCK)
        return pid, age
    except Exception:
        return None, None


def acquire_lock(wait_min=240, poll=30, stale_h=2.0):
    """O_EXCL lock file holding our PID; wait for the holder, never skip.

    A lock is reclaimed when its PID is gone, or when the file is older than
    ``stale_h`` hours (a crashed process on a machine whose tasklist we cannot
    read would otherwise block everyone forever).
    """
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    t0 = time.time()
    warned = False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("%d\n%s\n%s\n" % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S"),
                                           " ".join(sys.argv)))
            print("[gpu] lock acquired: %s (pid=%d)" % (LOCK, os.getpid()), flush=True)
            return
        except FileExistsError:
            pid, age = lock_info()
            if age is not None and age > stale_h * 3600:
                print("[gpu] lock is %.1fh old (> %.1fh) -> treating it as stale, "
                      "taking over" % (age / 3600.0, stale_h), flush=True)
                _remove_lock()
                continue
            if pid and pid > 0 and not pid_alive(pid):
                print("[gpu] lock holder pid=%d is gone -> taking over" % pid, flush=True)
                _remove_lock()
                continue
            if not warned:
                print("[gpu] another process holds the GPU lock (pid=%s, age=%.0fs); "
                      "polling every %ds, waiting up to %d min"
                      % (pid, age or 0.0, poll, wait_min), flush=True)
                warned = True
            if (time.time() - t0) > wait_min * 60:
                raise SystemExit("[gpu] gave up waiting for %s after %d min"
                                 % (LOCK, wait_min))
            time.sleep(poll)


def _remove_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def release_lock():
    try:
        pid, _ = lock_info()
        if pid == os.getpid():
            os.remove(LOCK)
            print("[gpu] lock released: %s" % LOCK, flush=True)
    except Exception:
        pass


class Lock:
    """Context manager wrapper so a crash still releases the lock."""

    def __init__(self, enabled=True, **kw):
        self.enabled = enabled
        self.kw = kw

    def __enter__(self):
        if self.enabled:
            acquire_lock(**self.kw)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            release_lock()
        return False


class Tee:
    """Mirror stdout into a log file (console stays UTF-8 via PYTHONUTF8)."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except ValueError:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass

    def isatty(self):
        return False


def open_log(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "a", encoding="utf-8", buffering=1)


def log_stdout(path):
    """Tee sys.stdout to ``path`` (returns the file handle, call .close() at the end)."""
    fh = open_log(path)
    sys.stdout = Tee(sys.__stdout__, fh)
    return fh


# ------------------------------------------------------- cross-validation protocol
# One definition, used by every trainer. Folds are derived at runtime from the
# row order + labels of train1.csv -- the data ships no `fold` column.
NFOLD = 5
SKF_SEED = 42


def skf_folds(y, n_folds=NFOLD, seed=SKF_SEED):
    """Fold id per row: StratifiedKFold(n_folds, shuffle=True, random_state=seed).

    Rows are consumed in file order (no sorting, no reindexing); every row is a
    validation row exactly once. Deterministic across machines for a given
    scikit-learn/scipy stack.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    fold_of = np.full(len(y), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y), dtype=np.float32), y)):
        fold_of[va] = k
    assert (fold_of >= 0).all(), "every row must land in exactly one validation fold"
    return fold_of


def fold_signature(idx):
    """Short hash of a fold's row ids, so two machines can compare membership."""
    import hashlib

    import numpy as np

    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:8]


def fold_table(fold_of, y):
    import numpy as np

    y = np.asarray(y)
    return [dict(fold=k, n_val=int((fold_of == k).sum()),
                 n_pos=int(y[fold_of == k].sum()),
                 sha=fold_signature(np.where(fold_of == k)[0]))
            for k in range(int(fold_of.max()) + 1)]


def print_folds(rows, prefix="[folds]"):
    for r in rows:
        print("%s k=%d n_val=%d n_pos=%d sha=%s"
              % (prefix, r["fold"], r["n_val"], r["n_pos"], r["sha"]), flush=True)

