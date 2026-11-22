# -*- coding: utf-8 -*-
"""The fifth view: a 36,929-parameter 1-D CNN with a StarNet-style multiplicative gate.

Architecture (d = 64)
---------------------
embedding(22, d) -> conv(k=3) + masked BN + GELU
                 -> conv(k=5) + masked BN + GELU
                 -> split channels in two, multiply elementwise   (StarNet gate)
                 -> conv(1x1) + masked BN + GELU
                 -> masked mean & max pool -> dropout 0.2 -> linear -> logit

Padding never enters the batch-norm statistics, and every activation is
re-masked after the non-linearity, so a 7-residue peptide is not padded into a
40-residue one at any layer.

Training, unchanged from the reported sweep: AdamW(lr=1e-3, wd=0.02), batch 32,
BCEWithLogits, grad-norm clip 2.0, <=8 epochs, patience 5 on the fold's val AUROC.
As in the sweep, the model is constructed *before* ``train_one`` seeds torch, so
the released fold checkpoints can be reproduced from this file alone.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                       # noqa: E402
from cnn_lib import LMAX, VOCAB, MaskedBatchNorm1d, onehot, probs   # noqa: E402,F401

EPOCHS = 8
PATIENCE = 5
BS = 32
LR = 1e-3
WD = 0.02
CLIP = 2.0


class _PoolHead(nn.Module):
    """Masked mean+max pool -> dropout -> one logit."""

    def __init__(self, d, drop=0.2):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(2 * d, 1)

    def forward(self, h, mk):
        cnt = mk.sum(-1).clamp(min=1.0)
        mean = (h * mk).sum(-1) / cnt
        mx = h.masked_fill(mk == 0, -1e4).max(-1).values
        z = self.drop(torch.cat([mean, mx], 1))
        return self.head(z).squeeze(-1)


class StarCNN(nn.Module):
    """Base convolutions plus a split-and-multiply gate; 36,929 parameters."""

    def __init__(self, d=64, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(22, d, padding_idx=0)
        self.c1 = nn.Conv1d(d, d, 3, padding=1)
        self.b1 = MaskedBatchNorm1d(d)
        self.c2 = nn.Conv1d(d, d, 5, padding=2)
        self.b2 = MaskedBatchNorm1d(d)
        self.proj = nn.Conv1d(d // 2, d, 1)
        self.bp = MaskedBatchNorm1d(d)
        self.pool = _PoolHead(d, drop)

    def forward(self, x, m):
        mk = m.unsqueeze(1)
        h = self.emb(x).transpose(1, 2)
        h = F.gelu(self.b1(self.c1(h), mk)) * mk
        h = F.gelu(self.b2(self.c2(h), mk)) * mk
        a, b = h.chunk(2, dim=1)
        h = F.gelu(self.bp(self.proj(a * b), mk)) * mk
        return self.pool(h, mk)


def train_one(model, Xtr, Mtr, ytr, Xva, Mva, yva, seed=0, epochs=EPOCHS,
              patience=PATIENCE, log=print):
    """One fold; returns the model at its best val epoch, that AUROC and epoch."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=ytr.device)
        for i in range(0, n, BS):
            b = idx[i:i + BS]
            opt.zero_grad()
            loss = lossf(model(Xtr[b], Mtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        p = probs(model, Xva, Mva)
        auc = float(roc_auc_score(yva.detach().cpu().numpy(), p))
        log("    [ep%d] val_auroc=%.4f" % (ep, auc))
        if auc > best:
            best, bad, best_ep = auc, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("    early stop at ep%d" % ep)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_ep


def train_folds(seed=0, device="cuda", out_dir=None, log=print):
    """Fit the five fold models and write ``weights/cnn/star_fold<k>.pt``."""
    out_dir = out_dir or os.path.join(C.WEIGHTS, "cnn")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    tr, te = C.load_split("train1"), C.load_split("test1")
    y = tr.label.values
    fold = C.skf_folds(y)
    seqs = list(tr.sequence) + list(te.sequence)
    n_tr = len(tr)
    X, M = onehot(seqs)
    X, M, yall = X.to(dev), M.to(dev), torch.from_numpy(np.concatenate([y, te.label.values]).astype(np.float32)).to(dev)
    oof = np.zeros(n_tr)
    fold_test = np.zeros((C.NFOLD, len(te)))
    for k in range(C.NFOLD):
        fit_idx = torch.from_numpy(np.where(fold != k)[0]).to(dev)
        val_idx = torch.from_numpy(np.where(fold == k)[0]).to(dev)
        te_idx = torch.arange(n_tr, len(seqs)).to(dev)
        log("[cnn] fold%d train=%d val=%d" % (k, len(fit_idx), len(val_idx)))
        model = StarCNN().to(dev)
        model, best, best_ep = train_one(
            model, X[fit_idx], M[fit_idx], yall[fit_idx],
            X[val_idx], M[val_idx], yall[val_idx], seed=seed, log=log)
        oof[val_idx.detach().cpu().numpy()] = probs(model, X[val_idx], M[val_idx])
        fold_test[k] = probs(model, X[te_idx], M[te_idx])
        path = os.path.join(out_dir, "star_fold%d.pt" % k)
        torch.save({"state_dict": model.state_dict(), "best_ep": best_ep,
                    "val_auroc": best, "fold": k, "seed": seed}, path)
        log("[cnn] fold%d val=%.4f test=%.4f -> %s"
            % (k, best, roc_auc_score(te.label.values, fold_test[k]), os.path.basename(path)))
    test = fold_test.mean(0)
    auc = roc_auc_score(te.label.values, test)
    log("[cnn] oof=%.4f ensemble test=%.4f" % (roc_auc_score(y, oof), auc))
    np.savez(os.path.join(out_dir, "star_probs.npz"), oof=oof, test=test,
             fold_test=fold_test, y_tr=y, y_te=te.label.values)
    return auc


def predict(seqs, device="cpu", ckpt_dir=None):
    """Mean probability over the five released fold models."""
    ckpt_dir = ckpt_dir or os.path.join(C.WEIGHTS, "cnn")
    dev = torch.device(device)
    X, M = onehot(seqs)
    X, M = X.to(dev), M.to(dev)
    ps = []
    for k in range(C.NFOLD):
        ck = torch.load(os.path.join(ckpt_dir, "star_fold%d.pt" % k), map_location=dev, weights_only=False)
        model = StarCNN().to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        ps.append(probs(model, X, M))
    return np.mean(ps, axis=0)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", default=None, help="csv with a sequence column")
    ap.add_argument("--out", default=None, help="npz to write the probabilities into")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.train:
        train_folds(seed=a.seed, device=a.device)
    if a.score:
        seqs = pd.read_csv(a.score, dtype=str).sequence.str.strip().str.upper().tolist()
        p = predict(seqs, device=a.device)
        np.savez(a.out, cnn=p)
        print("[cnn] scored %d peptide(s) -> %s" % (len(seqs), a.out))
