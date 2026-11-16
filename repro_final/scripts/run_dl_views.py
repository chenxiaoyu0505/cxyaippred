# -*- coding: utf-8 -*-
"""Deep-learning views -- inference only, no training.

TLoRA (`weights/tlora_fold{k}.pt`)
    ESM-2 t33 650M (frozen, fp16, reloaded from HuggingFace) + q/k/v LoRA (r=8, a=16)
    + posproj-adaptive head.  The `.pt` files are PEFT *delta* files: they only carry
    the adapter + head tensors (206 tensors, ~7.9 MB), never the 650M backbone.
    OOF probability of a validation row = that fold's model; test probability =
    the mean over the 5 fold models (the released main arm).

    `weights/tlora_init_fold{k}.npz` is the SVD initialisation each fold was started
    from (A = top-8 right singular vectors of W0 @ C).  It is shipped for provenance
    only: the delta already contains the same frozen A (verified here, max|dA| = 0),
    so the model can be built from the base checkpoint + delta alone.

CNN (`weights/cnn_folds.pt`)
    `cnn_k3_k5 d=64`, 34,689 params: Embedding(22,64) -> Conv1d(k=3) + BN + GELU ->
    Conv1d(k=5) + BN + GELU -> masked mean ++ masked max -> Dropout(0.2) -> Linear(1).
    Keys are `fold0..fold4`, each a plain state dict.

Output: `out/dl_views.npz` with oof_{tlora,cnn}, test_{tlora,cnn}, fold_test_{...}.

Usage
-----
    python scripts/run_dl_views.py                 # both views, GPU (~3-5 min)
    python scripts/run_dl_views.py --views cnn     # CNN only (CPU is fine, seconds)
    python scripts/run_dl_views.py --views tlora --device cpu   # slow, but works
"""
import argparse
import json
import os
import sys
import time

import torch                     # keep torch ahead of numpy/sklearn (DLL init order)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
WEIGHTS = os.path.join(ROOT, "weights")
sys.path.insert(0, HERE)

from tlora_model import ESM_LoRA_PosProj, predict, unwrap_qkv  # noqa: E402
import peft_io  # noqa: E402

BASE = "facebook/esm2_t33_650M_UR50D"
NFOLD = 5
N_TRAIN1, N_TEST1, POS_TRAIN1, POS_TEST1 = 3278, 817, 1060, 262


# ------------------------------------------------------------------------------- CNN
AA = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {a: i + 2 for i, a in enumerate(AA)}   # 0 = pad, 1 = non-standard residue
LMAX = 40


class CNN(nn.Module):
    def __init__(self, d=64, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(22, d, padding_idx=0)
        self.c1 = nn.Conv1d(d, d, 3, padding=1)
        self.b1 = nn.BatchNorm1d(d)
        self.c2 = nn.Conv1d(d, d, 5, padding=2)
        self.b2 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(2 * d, 1)

    def forward(self, x, m):
        h = self.emb(x).transpose(1, 2)
        mk = m.unsqueeze(1)
        h = F.gelu(self.b1(self.c1(h))) * mk
        h = F.gelu(self.b2(self.c2(h))) * mk
        cnt = mk.sum(-1).clamp(min=1.0)
        mean = (h * mk).sum(-1) / cnt
        mx = h.masked_fill(mk == 0, -1e4).max(-1).values
        z = self.drop(torch.cat([mean, mx], 1))
        return self.head(z).squeeze(-1)


def onehot(seqs):
    x = np.zeros((len(seqs), LMAX), dtype=np.int64)
    m = np.zeros((len(seqs), LMAX), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s[:LMAX]
        for j, c in enumerate(s):
            x[i, j] = VOCAB.get(c, 1)
            m[i, j] = 1.0
    return torch.from_numpy(x), torch.from_numpy(m)


@torch.no_grad()
def cnn_probs(model, X, M, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs], M[i:i + bs])).numpy())
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------- TLoRA
def check_init(k, delta):
    """Sanity check: the shipped SVD init is the A that the delta carries."""
    path = os.path.join(WEIGHTS, "tlora_init_fold%d.npz" % k)
    z = np.load(path, allow_pickle=True)
    worst, n = 0.0, 0
    for name, A in zip(list(z["names"]), z["A"]):
        lay, kind = str(name).split(".")
        key = "esm.encoder.layer%s.attention.self.%s.A.weight" % (
            lay.replace("layer", "."), {"q": "query", "k": "key", "v": "value"}[kind])
        if key in delta:
            worst = max(worst, float(np.abs(A - delta[key].numpy()).max()))
            n += 1
    return n, worst, int(z["r"]), int(z["fold"])


def run_tlora(tr, te, device="cuda", bs=32, hf_offline=False):
    if hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer

    print("[tlora] loading %s (fp16, %s)" % (BASE, device), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    esm = AutoModel.from_pretrained(BASE, torch_dtype=torch.float16).to(device)
    esm.eval()
    esm.requires_grad_(False)

    fold_of = tr.fold.values.astype(int)
    oof = np.zeros(len(tr))
    fold_test, per_fold = [], []
    for k in range(NFOLD):
        t0 = time.time()
        ck = os.path.join(WEIGHTS, "tlora_fold%d.pt" % k)
        delta, meta, kind = peft_io.read_any(ck)
        n_init, dmax, rank_np, fold_np = check_init(k, delta)
        rank = int(meta.get("rank", 8))
        alpha = float(meta.get("alpha", 16))
        model = ESM_LoRA_PosProj(esm, rank=rank, alpha=alpha).to(device)
        peft_io.load_delta(ck, model, verbose=False)
        print("[tlora] fold%d  %d tensors  rank=%d alpha=%g  "
              "init-check: %d A matched, max|dA|=%.1e"
              % (k, len(delta), rank, alpha, n_init, dmax), flush=True)

        va = np.where(fold_of == k)[0]
        oof[va] = predict(model, tok, list(tr.sequence.values[va]), bs=bs, device=device)
        p = predict(model, tok, list(te.sequence.values), bs=bs, device=device)
        fold_test.append(p)
        per_fold.append(round(float(roc_auc_score(te.label.values, p)), 4))
        print("  fold%d test_auroc=%.4f  (%.0fs)" % (k, per_fold[-1], time.time() - t0), flush=True)

        del model
        unwrap_qkv(esm)
        if device == "cuda":
            torch.cuda.empty_cache()
    return oof, np.mean(fold_test, axis=0), fold_test, per_fold


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="tlora,cnn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=32, help="TLoRA batch size; keep 32")
    ap.add_argument("--hf-offline", action="store_true")
    args = ap.parse_args()
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    t_all = time.time()
    tr = pd.read_csv(os.path.join(DATA, "train1.csv"), dtype={"sequence": str})
    te = pd.read_csv(os.path.join(DATA, "test1.csv"), dtype={"sequence": str})
    tr["sequence"] = tr.sequence.str.strip().str.upper()
    te["sequence"] = te.sequence.str.strip().str.upper()
    assert len(tr) == N_TRAIN1 and int(tr.label.sum()) == POS_TRAIN1
    assert len(te) == N_TEST1 and int(te.label.sum()) == POS_TEST1
    y_tr, y_te = tr.label.values.astype(int), te.label.values.astype(int)
    print("[data] train1 %d (%d pos)  test1 %d (%d pos)  device=%s"
          % (len(tr), y_tr.sum(), len(te), y_te.sum(), args.device), flush=True)

    res, payload = {"n_train": int(len(tr)), "n_test": int(len(te))}, {}
    if "tlora" in views:
        oof, test, ft, pf = run_tlora(tr, te, device=args.device, bs=args.bs,
                                      hf_offline=args.hf_offline)
        payload.update(oof_tlora=oof, test_tlora=test, fold_test_tlora=np.array(ft))
        res["tlora"] = dict(test_auroc=round(float(roc_auc_score(y_te, test)), 4),
                            test_aupr=round(float(average_precision_score(y_te, test)), 4),
                            oof_auroc=round(float(roc_auc_score(y_tr, oof)), 4),
                            per_fold_test=pf, n_init_checked=5)

    if "cnn" in views:
        t0 = time.time()
        sd = torch.load(os.path.join(WEIGHTS, "cnn_folds.pt"), map_location="cpu",
                        weights_only=False)
        assert sorted(sd.keys()) == ["fold%d" % k for k in range(NFOLD)], sorted(sd.keys())
        Xtr, Mtr = onehot(list(tr.sequence.values))
        Xte, Mte = onehot(list(te.sequence.values))
        fold_of = tr.fold.values.astype(int)
        oof = np.zeros(len(tr))
        fold_test, per_fold = [], []
        for k in range(NFOLD):
            model = CNN()
            model.load_state_dict(sd["fold%d" % k])
            va = np.where(fold_of == k)[0]
            oof[va] = cnn_probs(model, Xtr[va], Mtr[va])
            p = cnn_probs(model, Xte, Mte)
            fold_test.append(p)
            per_fold.append(round(float(roc_auc_score(y_te, p)), 4))
            print("  cnn fold%d test_auroc=%.4f" % (k, per_fold[-1]), flush=True)
        test = np.mean(fold_test, axis=0)
        payload.update(oof_cnn=oof, test_cnn=test, fold_test_cnn=np.array(fold_test))
        res["cnn"] = dict(test_auroc=round(float(roc_auc_score(y_te, test)), 4),
                          test_aupr=round(float(average_precision_score(y_te, test)), 4),
                          oof_auroc=round(float(roc_auc_score(y_tr, oof)), 4),
                          per_fold_test=per_fold, n_params=int(sum(
                              v.numel() for v in sd["fold0"].values())),
                          sec=round(time.time() - t0, 1))

    if len(payload) == 0:
        print("[err] nothing to do")
        return 2
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "dl_views.npz"),
             tr_seq=np.array(tr.sequence, dtype=str), te_seq=np.array(te.sequence, dtype=str),
             y_tr=y_tr, y_te=y_te, **payload)
    res["ns_per_view"] = {}
    with open(os.path.join(OUT, "dl_views.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("\n%-6s %10s %10s %10s" % ("view", "test_auroc", "test_aupr", "oof_auroc"))
    for k in ("tlora", "cnn"):
        if k in res:
            print("%-6s %10.4f %10.4f %10.4f" % (k, res[k]["test_auroc"],
                                                 res[k]["test_aupr"], res[k]["oof_auroc"]))
    print("\n[saved] out/dl_views.npz  (%.1f min)" % ((time.time() - t_all) / 60))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
