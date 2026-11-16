# -*- coding: utf-8 -*-
"""TLoRA adapter + posproj-adaptive head -- inference-side model definition.

Vendored (and trimmed to the inference path) from the training repo's
`_work/wt-fullfit/scripts/tlora_lib.py`; identical module names and forward pass,
so the released `weights/tlora_fold{k}.pt` state dicts load as-is.

Structure
---------
1. Frozen ESM-2 (650M, 33 layers); every layer's q/k/v `nn.Linear` is replaced by a
   `LoRALinear` (99 modules).  "TLoRA" = A is initialised to the top-r right singular
   vectors of `W0 @ C`, where C is the input-activation covariance on that fold's
   training rows; A is frozen, only B is trained (alpha/r = 2).
2. Head: `Linear(1280->4)` per position -> `Conv1d(4,4,3)` -> GELU
   -> mask-weighted `F.interpolate` to 32 slots -> flatten(128) -> `Linear(256)`
   -> GELU -> Dropout(0.3) -> `Linear(1)`.

Known quirk, kept on purpose (it is how the reported numbers were produced): the
convolution happens *before* masking, so pad positions leak into the sequence tail
and the score depends on batch composition.  Always infer with `bs=32` (bs=16 moves
fold0 test AUROC from 0.7445 to 0.7365).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

D_POS, LMAX, D_MODEL = 4, 32, 1280


class LoRALinear(nn.Module):
    """Frozen base + low-rank delta.  A is frozen (SVD init), only B trains."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.A.weight.requires_grad_(False)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x.float())).to(x.dtype) * self.scale


class ESM_LoRA_PosProj(nn.Module):
    def __init__(self, esm, rank=8, alpha=16, drop=0.3):
        super().__init__()
        self.esm = esm
        self.esm.requires_grad_(False)
        for layer in self.esm.encoder.layer:
            for name in ("query", "key", "value"):
                lin = getattr(layer.attention.self, name)
                setattr(layer.attention.self, name, LoRALinear(lin, rank, alpha))
        self.posproj = nn.Linear(D_MODEL, D_POS)
        self.posconv = nn.Conv1d(D_POS, D_POS, 3, padding=1)
        self.posact = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(D_POS * LMAX, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 1))

    def forward(self, ids, attn, rmask):
        h = self.esm(input_ids=ids, attention_mask=attn).last_hidden_state
        m = rmask.unsqueeze(1).to(h.dtype)
        z = self.posact(self.posconv(self.posproj(h).transpose(1, 2)))
        zf = F.interpolate(z * m, size=LMAX, mode="linear", align_corners=False)
        mf = F.interpolate(m, size=LMAX, mode="linear", align_corners=False)
        z = zf / mf.clamp(min=1e-6)
        return self.head(z.flatten(1)).squeeze(-1)


def unwrap_qkv(esm):
    """Put the plain nn.Linear back so the next fold can wrap a fresh adapter."""
    n = 0
    for layer in esm.encoder.layer:
        for name in ("query", "key", "value"):
            mod = getattr(layer.attention.self, name)
            if isinstance(mod, LoRALinear):
                setattr(layer.attention.self, name, mod.base)
                n += 1
    return n


def residue_mask(attn):
    """Drop <cls> and <eos>: keep real residue positions only."""
    m = attn.clone()
    m[:, 0] = 0
    lens = attn.sum(1)
    m[torch.arange(len(attn), device=attn.device), lens - 1] = 0
    return m


def batchify(seqs, tok, bs):
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], padding=True, truncation=True, max_length=64,
                  return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, tok, seqs, bs=32, device="cuda"):
    """Sigmoid probabilities; `bs` must stay 32 (see the module docstring)."""
    model.eval()
    ps = []
    for ids, attn in batchify(seqs, tok, bs):
        ids, attn = ids.to(device), attn.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(ids, attn, residue_mask(attn))
        ps.append(logits.float().cpu().numpy())
    z = np.concatenate(ps)
    return 1.0 / (1.0 + np.exp(-z))
