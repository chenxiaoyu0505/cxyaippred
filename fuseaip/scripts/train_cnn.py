# -*- coding: utf-8 -*-
"""Small-CNN view: train 5 folds from scratch on train1.csv, score test1.csv.

Fold protocol: ``StratifiedKFold(5, shuffle=True, random_state=42)`` over the
train1.csv rows in file order (``common.skf_folds``), same folds as every other
view; the data ships no fold column.

Outputs
-------
    weights/trained/cnn_trained_fold{k}.pt   state_dict of the best val epoch
    out/probs/cnn_fold{k}.npz     val probabilities (with row index) + test probs
    out/probs/cnn_views.npz       once all 5 folds exist (oof_cnn / test_cnn /
                                  fold_test_cnn)
    out/probs/cnn_folds.json      per-fold metrics

Usage
-----
    python fuseaip/scripts/train_cnn.py               # all 5 folds (~2 min CPU)
    python fuseaip/scripts/train_cnn.py --folds 0     # smoke test
    python fuseaip/scripts/train_cnn.py --reuse       # score archived weights only
"""
import argparse
import os
import sys
import time

import torch                      # before numpy / sklearn (Windows DLL init order)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402
import cnn_lib as cl  # noqa: E402

VIEW = "cnn"
NFOLD = 5
N_TRAIN1, N_TEST1, POS_TRAIN1, POS_TEST1 = 3278, 817, 1060, 262


def load_data():
    tr = pd.read_csv(os.path.join(C.DATA, "train1.csv"), dtype={"sequence": str})
    te = pd.read_csv(os.path.join(C.DATA, "test1.csv"), dtype={"sequence": str})
    tr["sequence"] = tr.sequence.str.strip().str.upper()
    te["sequence"] = te.sequence.str.strip().str.upper()
    assert len(tr) == N_TRAIN1 and int(tr.label.sum()) == POS_TRAIN1
    assert len(te) == N_TEST1 and int(te.label.sum()) == POS_TEST1
    assert not set(tr.sequence) & set(te.sequence), "train/test overlap"
    fold_of = C.skf_folds(tr.label.values.astype(int))
    C.print_folds(C.fold_table(fold_of, tr.label.values.astype(int)))
    return tr, te, fold_of


def collect(folds_to_run, tr, te, y_tr, y_te, fold_of):
    if not set(range(NFOLD)).issubset(set(folds_to_run)):
        return None
    oof = np.full(len(tr), np.nan)
    fold_test = []
    for k in range(NFOLD):
        z = np.load(os.path.join(C.PROBS, "cnn_fold%d.npz" % k))
        oof[z["val_idx"]] = z["p_val"]
        fold_test.append(z["p_test"])
    assert not np.isnan(oof).any(), "missing val rows"
    from sklearn.metrics import average_precision_score, roc_auc_score
    test = np.mean(fold_test, axis=0)
    np.savez(os.path.join(C.PROBS, "cnn_views.npz"),
             oof_cnn=oof, test_cnn=test, fold_test_cnn=np.array(fold_test),
             y_tr=y_tr, y_te=y_te, te_length=te.length.values.astype(int),
             te_seq=np.array(te.sequence, dtype=str), fold_of=fold_of)
    out = dict(view=VIEW, oof_auroc=round(float(roc_auc_score(y_tr, oof)), 4),
               ensemble_auroc=round(float(roc_auc_score(y_te, test)), 4),
               ensemble_aupr=round(float(average_precision_score(y_te, test)), 4),
               ensemble_via="mean of the 5 fold test probabilities")
    C.write_json(os.path.join(C.PROBS, "cnn_ensemble.json"), out)
    print("[ens] cnn oof=%.4f ensemble test=%.4f/%.4f"
          % (out["oof_auroc"], out["ensemble_auroc"], out["ensemble_aupr"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=cl.EPOCHS)
    ap.add_argument("--patience", type=int, default=cl.PATIENCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", action="store_true",
                    help="do not train: score weights/trained/cnn_trained_fold{k}.pt")
    args = ap.parse_args()

    folds = [int(x) for x in str(args.folds).split(",") if x.strip() != ""]
    C.ensure_dirs()
    torch.set_num_threads(max(4, os.cpu_count() or 4))
    log = C.log_stdout(os.path.join(C.LOGS, "cnn.log"))
    try:
        C.stage("small CNN view: %s folds %s"
                % ("score" if args.reuse else "train", folds))
        tr, te, fold_of = load_data()
        print("[data] train1=%d(+%d) test1=%d(+%d) per-fold val=%s"
              % (len(tr), int(tr.label.sum()), len(te), int(te.label.sum()),
                 [int((fold_of == k).sum()) for k in range(NFOLD)]), flush=True)

        Xtr_all, Mtr_all = cl.onehot(list(tr.sequence))
        Xte, Mte = cl.onehot(list(te.sequence))
        y_tr = torch.tensor(tr.label.values, dtype=torch.float32)
        y_te = te.label.values.astype(int)

        dev = torch.device(args.device)
        # keep every tensor on the same device (CPU default, CUDA with --device cuda)
        Xtr_all, Mtr_all = Xtr_all.to(dev), Mtr_all.to(dev)
        Xte, Mte, y_tr = Xte.to(dev), Mte.to(dev), y_tr.to(dev)
        per_fold = []
        for k in folds:
            t0 = time.time()
            va_idx = np.where(fold_of == k)[0]
            tr_idx = np.where(fold_of != k)[0]
            if args.reuse:
                ck = os.path.join(C.MODELS, "cnn_trained_fold%d.pt" % k)
                assert os.path.exists(ck), "missing %s -- run without --reuse first" % ck
                model = cl.CNN(d=64, drop=0.2).to(dev)
                blob = torch.load(ck, map_location=dev)
                if isinstance(blob, dict) and "state_dict" in blob:
                    model.load_state_dict(blob["state_dict"])
                    meta = blob
                else:                      # plain state_dict from an older archive
                    model.load_state_dict(blob)
                    meta = {}
                model.eval()
                p_va = cl.probs(model, Xtr_all[va_idx], Mtr_all[va_idx])
                p_te = cl.probs(model, Xte, Mte)
                best, best_ep = float(meta.get("val_auroc", float("nan"))), int(meta.get("best_ep", -1))
                secs = time.time() - t0
                from sklearn.metrics import average_precision_score, roc_auc_score
                res = dict(fold=k, view=VIEW, reuse=True, weights=os.path.basename(ck),
                           best_ep=best_ep, val_auroc=None if best != best else round(best, 4),
                           test_auroc=round(float(roc_auc_score(y_te, p_te)), 4),
                           test_aupr=round(float(average_precision_score(y_te, p_te)), 4),
                           n_params=int(sum(p.numel() for p in model.parameters())),
                           secs=round(secs, 1))
                print("[fold%d] reused %s TEST auroc=%.4f aupr=%.4f"
                      % (k, res["weights"], res["test_auroc"], res["test_aupr"]), flush=True)
            else:
                # Seed BEFORE building the model: nn.Module init consumes torch RNG,
                # and the old code only seeded inside train_one() -- the weight init
                # drew from the unseeded global RNG, so every run (and every --folds
                # subset) silently produced different fold models. Per-fold seeding
                # makes fold k identical whether trained alone or within all five.
                # train_one() still re-seeds with `seed`, so the epoch order /
                # dropout stream stays identical across folds given the (now
                # deterministic) initialisation.
                torch.manual_seed(args.seed + k)
                np.random.seed(args.seed + k)
                model = cl.CNN(d=64, drop=0.2).to(dev)
                model, best, best_ep = cl.train_one(
                    model, Xtr_all[tr_idx], Mtr_all[tr_idx], y_tr[tr_idx],
                    Xtr_all[va_idx], Mtr_all[va_idx], y_tr[va_idx],
                    seed=args.seed, epochs=args.epochs, patience=args.patience)
                p_va = cl.probs(model, Xtr_all[va_idx], Mtr_all[va_idx])
                p_te = cl.probs(model, Xte, Mte)
                secs = time.time() - t0
                from sklearn.metrics import average_precision_score, roc_auc_score
                res = dict(fold=k, view=VIEW, best_ep=int(best_ep), val_auroc=round(float(best), 4),
                           test_auroc=round(float(roc_auc_score(y_te, p_te)), 4),
                           test_aupr=round(float(average_precision_score(y_te, p_te)), 4),
                           n_params=int(sum(p.numel() for p in model.parameters())),
                           weights="cnn_trained_fold%d.pt" % k, secs=round(secs, 1))
                torch.save(dict(state_dict=model.state_dict(), best_ep=int(best_ep),
                                val_auroc=float(best), n_params=res["n_params"],
                                fold=k, epochs=args.epochs, patience=args.patience),
                           os.path.join(C.MODELS, res["weights"]))
                print("[fold%d] ep%d val=%.4f TEST auroc=%.4f aupr=%.4f (%.0fs)"
                      % (k, best_ep, best, res["test_auroc"], res["test_aupr"], secs),
                      flush=True)
            C.write_json(os.path.join(C.PROBS, "cnn_fold%d.json" % k), res)
            np.savez(os.path.join(C.PROBS, "cnn_fold%d.npz" % k),
                     val_idx=va_idx, p_val=p_va, p_test=p_te, fold=k)
            per_fold.append(res)

        C.write_json(os.path.join(C.PROBS, "cnn_folds.json"),
                     dict(view=VIEW, folds=folds, per_fold=per_fold,
                          cfg=dict(epochs=args.epochs, patience=args.patience,
                                   seed=args.seed, lr=cl.LR, wd=cl.WD, bs=cl.BS,
                                   lmax=cl.LMAX, n_params=per_fold[0].get("n_params"))))
        ens = collect(folds, tr, te, tr.label.values.astype(int), y_te, fold_of)
        print("[done] cnn weights -> weights/trained/cnn_trained_fold{k}.pt", flush=True)
        if ens is None:
            print("[warn] only folds %s trained; cnn_views.npz not written" % folds,
                  flush=True)
    finally:
        sys.stdout = sys.__stdout__
        log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
