# -*- coding: utf-8 -*-
"""TLoRA view: train 5 folds from scratch on train1.csv and score test1.csv.

Fold protocol: ``StratifiedKFold(5, shuffle=True, random_state=42)`` over the
train1.csv rows in file order (``common.skf_folds``). For fold k the validation
rows are those the splitter assigns to k and the fit rows are the rest; the
data ships no fold column, so this is derived identically by every trainer.

Outputs
-------
    weights/trained/tlora_trained_fold{k}.pt        trainable delta only (~8 MB) + meta
    weights/trained/tlora_trained_init_fold{k}.npz  the covariance-SVD init of A (per fold)
    out/probs/tlora_fold{k}.npz   val probabilities (with row index) + test probs
    out/probs/tlora_views.npz     written once all 5 folds exist:
                                  oof_tlora (3,278), test_tlora (817, 5-fold mean),
                                  fold_test_tlora (5, 817)
    out/probs/tlora_folds.json    per-fold metrics

Usage
-----
    python fuseaip/scripts/train_tlora.py              # all 5 folds (~18 min GPU)
    python fuseaip/scripts/train_tlora.py --folds 0    # smoke test, one fold
    python fuseaip/scripts/train_tlora.py --reuse      # score archived weights only
"""
import os

# Before transformers is imported anywhere below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import sys
import time

# torch MUST be imported before numpy / sklearn on this Windows box: importing
# numpy first makes the c10.dll DllMain fail with OSError WinError 1114.
import torch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402
import tlora_lib as tl  # noqa: E402

VIEW = "tlora"
NFOLD = 5
N_TRAIN1, N_TEST1, POS_TRAIN1, POS_TEST1 = 3278, 817, 1060, 262


def load_data():
    tr = pd.read_csv(os.path.join(C.DATA, "train1.csv"), dtype={"sequence": str})
    te = pd.read_csv(os.path.join(C.DATA, "test1.csv"), dtype={"sequence": str})
    tr["sequence"] = tr.sequence.str.strip().str.upper()
    te["sequence"] = te.sequence.str.strip().str.upper()
    assert len(tr) == N_TRAIN1 and int(tr.label.sum()) == POS_TRAIN1, (len(tr), int(tr.label.sum()))
    assert len(te) == N_TEST1 and int(te.label.sum()) == POS_TEST1, (len(te), int(te.label.sum()))
    assert not set(tr.sequence) & set(te.sequence), "train/test overlap"
    fold_of = C.skf_folds(tr.label.values.astype(int))
    C.print_folds(C.fold_table(fold_of, tr.label.values.astype(int)))
    return tr, te, fold_of


def collect(folds_to_run, tr, te, y_tr, y_te, fold_of):
    """Merge the per-fold fragments into out/probs/tlora_views.npz."""
    if not set(range(NFOLD)).issubset(set(folds_to_run)):
        return None
    oof = np.full(len(tr), np.nan)
    fold_test = []
    for k in range(NFOLD):
        z = np.load(os.path.join(C.PROBS, "tlora_fold%d.npz" % k))
        oof[z["val_idx"]] = z["p_val"]
        fold_test.append(z["p_test"])
    assert not np.isnan(oof).any(), "missing val rows"
    from sklearn.metrics import average_precision_score, roc_auc_score
    test = np.mean(fold_test, axis=0)
    np.savez(os.path.join(C.PROBS, "tlora_views.npz"),
             oof_tlora=oof, test_tlora=test, fold_test_tlora=np.array(fold_test),
             y_tr=y_tr, y_te=y_te, te_length=te.length.values.astype(int),
             te_seq=np.array(te.sequence, dtype=str), fold_of=fold_of)
    out = dict(view=VIEW, oof_auroc=round(float(roc_auc_score(y_tr, oof)), 4),
               ensemble_auroc=round(float(roc_auc_score(y_te, test)), 4),
               ensemble_aupr=round(float(average_precision_score(y_te, test)), 4),
               ensemble_via="mean of the 5 fold test probabilities")
    C.write_json(os.path.join(C.PROBS, "tlora_ensemble.json"), out)
    print("[ens] tlora oof=%.4f ensemble test=%.4f/%.4f"
          % (out["oof_auroc"], out["ensemble_auroc"], out["ensemble_aupr"]), flush=True)
    return out


def reuse_fold(k, esm, tok, tr, te, va_idx, y_tr, y_te, args):
    """Score one fold from the archived weights in ``weights/trained/`` (no training)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    path = os.path.join(C.MODELS, "tlora_trained_fold%d.pt" % k)
    assert os.path.exists(path), "missing %s -- run without --reuse first" % path
    model = tl.ESM_LoRA_PosProj(esm, rank=args.rank, alpha=args.alpha).to(args.device)
    meta = tl.load_delta(path, model, verbose=False)
    Xva = [tr.sequence.values[i] for i in va_idx]
    va_auc, va_ap, pva = tl.evaluate(model, tok, Xva, y_tr[va_idx])
    te_auc, te_ap, pte = tl.evaluate(model, tok, list(te.sequence), y_te)
    del model
    n_un = tl.unwrap_qkv(esm)
    pva = 1.0 / (1.0 + np.exp(-pva))
    pte = 1.0 / (1.0 + np.exp(-pte))
    res = dict(fold=k, view=VIEW, reuse=True, weights=os.path.basename(path),
               best_ep=(int(meta.get("epoch", -1)) if isinstance(meta, dict) else -1),
               val_auroc=round(float(va_auc), 4), val_aupr=round(float(va_ap), 4),
               test_auroc=round(float(roc_auc_score(y_te, pte)), 4),
               test_aupr=round(float(average_precision_score(y_te, pte)), 4),
               meta=meta, unwrapped=n_un)
    print("[fold%d] reused %s TEST auroc=%.4f aupr=%.4f"
          % (k, res["weights"], res["test_auroc"], res["test_aupr"]), flush=True)
    return res, pva, pte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lr-lora", type=float, default=5e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cov-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lock-wait", type=int, default=240, help="minutes")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="do not train: score weights/trained/tlora_trained_fold{k}.pt")
    args = ap.parse_args()

    folds = [int(x) for x in str(args.folds).split(",") if x.strip() != ""]
    C.ensure_dirs()
    log = C.log_stdout(os.path.join(C.LOGS, "tlora.log"))
    try:
        C.stage("TLoRA view: %s folds %s"
                % ("score archived" if args.reuse else "train", folds))
        tr, te, fold_of = load_data()
        print("[data] train1=%d(+%d) test1=%d(+%d) per-fold val=%s"
              % (len(tr), int(tr.label.sum()), len(te), int(te.label.sum()),
                 [int((fold_of == k).sum()) for k in range(NFOLD)]), flush=True)

        tl.set_device(args.device)
        if args.device == "cuda":
            assert torch.cuda.is_available(), "CUDA requested but not available"
            print("[gpu] %s (%.1f GiB)"
                  % (torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 2 ** 30),
                  flush=True)

        with C.Lock(enabled=not args.no_lock, wait_min=args.lock_wait, poll=30, stale_h=2.0):
            # Same protocol as the reference: seed BEFORE from_pretrained (the
            # missing-pooler init consumes torch RNG, so B/head init depends on it).
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            from transformers import AutoModel, AutoTokenizer

            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(tl.BASE_MODEL)
            dtype = torch.float16 if args.device == "cuda" else torch.float32
            esm = AutoModel.from_pretrained(tl.BASE_MODEL, torch_dtype=dtype).to(args.device)
            esm.eval()
            esm.requires_grad_(False)
            print("[esm] %s frozen dtype=%s qkv=%d (load %.0fs)"
                  % (tl.BASE_MODEL, next(esm.parameters()).dtype,
                     len(tl.qkv_linears(esm)), time.time() - t0), flush=True)

            y_tr = tr.label.values.astype(int)
            y_te = te.label.values.astype(int)
            per_fold = []
            for k in folds:
                va_idx = np.where(fold_of == k)[0]
                tr_idx = np.where(fold_of != k)[0]
                if args.reuse:
                    res, pva, pte = reuse_fold(k, esm, tok, tr, te, va_idx, y_tr, y_te, args)
                else:
                    Xtr = [tr.sequence.values[i] for i in tr_idx]
                    Xva = [tr.sequence.values[i] for i in va_idx]
                    res, pva, pte = tl.train_fold(
                        k, esm, tok, Xtr, y_tr[tr_idx].astype(np.float32),
                        Xva, y_tr[va_idx], list(te.sequence), y_te, args, C.MODELS)
                C.write_json(os.path.join(C.PROBS, "tlora_fold%d.json" % k), res)
                np.savez(os.path.join(C.PROBS, "tlora_fold%d.npz" % k),
                         val_idx=va_idx, p_val=pva, p_test=pte, fold=k)
                per_fold.append({kk: res.get(kk) for kk in
                                 ("fold", "best_ep", "epochs_run", "val_auroc",
                                  "val_aupr", "test_auroc", "test_aupr",
                                  "cov_sec", "svd_sec", "train_sec", "wall_sec")})
                if not args.reuse:
                    print("[fold%d] TEST auroc=%.4f aupr=%.4f"
                          % (k, res["test_auroc"], res["test_aupr"]), flush=True)

            C.write_json(os.path.join(C.PROBS, "tlora_folds.json"),
                         dict(view=VIEW, folds=folds, per_fold=per_fold,
                              cfg=dict(epochs=args.epochs, batch=args.batch, rank=args.rank,
                                       alpha=args.alpha, lr_lora=args.lr_lora,
                                       lr_head=args.lr_head, patience=args.patience,
                                       seed=args.seed, cov_batch=args.cov_batch,
                                       base_model=tl.BASE_MODEL)))
            ens = collect(folds, tr, te, y_tr, y_te, fold_of)
            print("[done] tlora weights -> weights/trained/tlora_trained_fold{k}.pt", flush=True)
            if ens is None:
                print("[warn] only folds %s trained; tlora_views.npz not written" % folds,
                      flush=True)
    finally:
        sys.stdout = sys.__stdout__
        log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
