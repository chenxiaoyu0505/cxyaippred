# -*- coding: utf-8 -*-
"""Delta (PEFT-style) checkpoint I/O: only trainable tensors are stored.

Vendored from the training repo's `_work/lora/peft_io.py` (read path only).
`weights/tlora_fold{k}.pt` has the shape

    {"format": "peft-delta-v1",
     "delta":  {name: tensor, ...},      # A/B of the 99 LoRA modules + posproj/posconv/head
     "meta":   {"fold": k, "epoch": ..., "val_auroc": ..., "base_model": ...}}

The frozen ESM-2 backbone is NOT in the file: it is reloaded from HuggingFace
(`facebook/esm2_t33_650M_UR50D`) and the delta is written on top of it.
"""
import torch

DELTA_PREFIXES = ("posproj", "posconv", "head")
DELTA_SUFFIXES = ("A.weight", "B.weight")


def is_delta_key(k):
    if k.startswith(DELTA_PREFIXES):
        return True
    return any(k.endswith(s) for s in DELTA_SUFFIXES)


def read_any(path):
    """Read a delta file (or a legacy full state dict) -> (delta, meta, kind)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "delta" in obj:
        return obj["delta"], obj.get("meta", {}), "delta"
    if "model" in obj:
        d = {k: v.detach().to("cpu").clone()
             for k, v in obj["model"].items() if is_delta_key(k)}
        return d, {"migrated_from": "full-state-dict"}, "full"
    raise ValueError("unrecognised checkpoint %s (keys=%s)" % (path, list(obj.keys())))


def load_delta(path, model, verbose=True):
    """Load delta into an already-built model; every delta key must be matched."""
    d, meta, kind = read_any(path)
    missing, unexpected = model.load_state_dict(d, strict=False)
    bad_missing = [k for k in missing if is_delta_key(k)]
    if bad_missing or unexpected:
        raise RuntimeError("delta does not match model: missing=%s unexpected=%s"
                           % (bad_missing[:5], unexpected[:5]))
    if verbose:
        n = sum(v.numel() for v in d.values())
        print("[load_delta] %s <- %d tensors / %.2fM params (source=%s)"
              % (path.replace("\\", "/").split("/")[-1], len(d), n / 1e6, kind), flush=True)
    return meta
