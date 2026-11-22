# -*- coding: utf-8 -*-
"""The TLoRA view: ESM-2 650M with a low-rank adapter on q/k/v.

The adapter's A matrix is initialised from the leading right singular vectors of
``W0 @ C`` (C = covariance of the layer inputs on the fold's training rows) and
then frozen; only B is trained.  Each fold ships a delta (``tlora_fold<k>.pt``)
plus the A matrices it started from (``tlora_init_fold<k>.npz``), so the fold
model can be rebuilt without re-running the covariance pass.

Head: ESM hidden states -> Linear(1280->4) -> Conv1d(4,4,3) -> GELU ->
interpolate to 32 positions -> masked mean -> Linear(128,256) -> GELU -> dropout
-> Linear(256,1).  Scoring averages the five fold models; the adapters are peeled
off the backbone between folds so each fold starts from pristine weights.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                       # noqa: E402
import tlora_lib as L                    # noqa: E402


def _load_esm(device):
    import transformers
    from transformers import AutoModel, AutoTokenizer
    transformers.logging.set_verbosity_error()      # the pooler is unused; silence the warning
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    dtype = torch.float16 if device == "cuda" else torch.float32
    esm = AutoModel.from_pretrained(L.BASE_MODEL, torch_dtype=dtype)
    esm.eval()
    esm.requires_grad_(False)
    return esm, tok


def _build_one(esm, k, device, ckpt_dir, log=print):
    """Attach the adapters for fold k: frozen A from the npz, trained B from the delta."""
    model = L.ESM_LoRA_PosProj(esm, rank=8, alpha=16).to(device)
    mods = list(L.adapter_modules(model))
    init = np.load(os.path.join(ckpt_dir, "tlora_init_fold%d.npz" % k), allow_pickle=False)
    A = init["A"]
    assert len(mods) == A.shape[0], (len(mods), A.shape)
    with torch.no_grad():
        for m, arr in zip(mods, A):
            m.A.weight.copy_(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device))
    L.load_delta(os.path.join(ckpt_dir, "tlora_fold%d.pt" % k), model, verbose=False)
    model.eval()
    log("[tlora] fold%d ready (rank=%d, %d adapters)" % (k, int(init["r"]), len(mods)))
    return model


@torch.no_grad()
def _score_one(model, tok, seqs, device, bs=32):
    out = []
    for ids, attn in L.batchify(seqs, tok, bs):
        ids, attn = ids.to(device), attn.to(device)
        rmask = L.residue_mask(attn).float()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=(device == "cuda")):
            logit = model(ids, attn, rmask)
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def predict(seqs, device="cuda", ckpt_dir=None, bs=32, log=print):
    """Mean probability over the five fold models."""
    ckpt_dir = ckpt_dir or os.path.join(C.WEIGHTS, "tlora")
    esm, tok = _load_esm(device)
    per_fold = []
    for k in range(C.NFOLD):
        model = _build_one(esm, k, device, ckpt_dir, log=log)
        per_fold.append(_score_one(model, tok, seqs, device, bs))
        L.unwrap_qkv(esm)          # pristine backbone for the next fold
        del model
    return np.mean(per_fold, axis=0)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, help="csv with a sequence column")
    ap.add_argument("--out", required=True, help="npz to write the probabilities into")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    seqs = pd.read_csv(a.score, dtype=str).sequence.str.strip().str.upper().tolist()
    p = predict(seqs, device=a.device)
    np.savez(a.out, tlora=p)
    print("[tlora] scored %d peptide(s) -> %s" % (len(seqs), a.out))
