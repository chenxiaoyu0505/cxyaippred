# -*- coding: utf-8 -*-
"""Small 1-D CNN view (34,689 parameters) -- model and training loop.

Ported from the training repo's `_work/scratch_pilot/run_scratch.py` (the
`CNN` model and `train_one` used for the released grid): two convolutions
(k=3 and k=5, 64 channels), masked mean+max pooling, dropout 0.2, one logit.

Training config, unchanged: AdamW(lr=2e-3, wd=0.02), batch 32, BCEWithLogits,
grad-norm clip 2.0, <=12 epochs, patience 4 on val AUROC, seed 0, sequences
truncated/padded to LMAX=40 over the 20 standard amino acids (index 1 = any
other character, 0 = pad).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

AA = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {a: i + 2 for i, a in enumerate(AA)}          # 0=pad, 1=non-standard, 2..21=AA
LMAX = 40
EPOCHS = 12
BS = 32
LR = 2e-3
WD = 0.02
PATIENCE = 4
CLIP = 2.0


class CNN(nn.Module):
    """Two conv layers (k=3 + k=5, receptive field 7), ~35k parameters."""

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
        h = self.emb(x).transpose(1, 2)                   # [B,d,L]
        mk = m.unsqueeze(1)
        h = F.gelu(self.b1(self.c1(h))) * mk
        h = F.gelu(self.b2(self.c2(h))) * mk
        cnt = mk.sum(-1).clamp(min=1.0)
        mean = (h * mk).sum(-1) / cnt
        mx = h.masked_fill(mk == 0, -1e4).max(-1).values
        z = self.drop(torch.cat([mean, mx], 1))
        return self.head(z).squeeze(-1)


def onehot(seqs):
    import numpy as np
    x = np.zeros((len(seqs), LMAX), dtype=np.int64)
    m = np.zeros((len(seqs), LMAX), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s[:LMAX]
        for j, c in enumerate(s):
            x[i, j] = VOCAB.get(c, 1)
            m[i, j] = 1.0
    return torch.from_numpy(x), torch.from_numpy(m)


@torch.no_grad()
def probs(model, X, M, bs=256):
    """Sigmoid probabilities as float64 (evaluation never shuffles)."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs], M[i:i + bs])).numpy())
    import numpy as np
    return np.concatenate(out).astype(np.float64)


def train_one(model, Xtr, Mtr, ytr, Xva, Mva, yva, seed=0, epochs=EPOCHS,
              patience=PATIENCE, log=print):
    """One fold; returns (model at its best val epoch, best val AUROC, best epoch)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, BS):
            b = idx[i:i + BS]
            opt.zero_grad()
            loss = lossf(model(Xtr[b], Mtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        p = probs(model, Xva, Mva)
        auc = float(roc_auc_score(yva.numpy(), p))
        log("  [ep%d] val_auroc=%.4f" % (ep, auc))
        if auc > best:
            best, bad, best_ep = auc, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("  early stop at ep%d" % ep)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_ep
