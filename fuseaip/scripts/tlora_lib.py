# -*- coding: utf-8 -*-
"""TLoRA view: model, SVD data-aligned init, delta checkpoint I/O, one fold loop.

Ported item by item from the training repo's `_work/wt-fullfit/scripts/tlora_lib.py`
(the file that produced the released numbers). Nothing here is "improved":

  * ESM-2 t33 650M, fp16, frozen; 33 layers x {q,k,v} = 99 linear layers each
    replaced by ``LoRALinear`` with r=8, alpha=16 (scale 2);
  * ``A`` is initialised from the top-r right singular vectors of ``W0 @ C``
    where ``C`` is the *input-activation covariance of that fold's training
    rows only* (non-pad residues only), and is frozen; only ``B`` and the
    posproj-adaptive head are trained;
  * head: ``Linear(1280->4) -> Conv1d(4,4,k=3) -> GELU -> masked resample to
    LMAX=32 -> flatten(128) -> Linear(256) -> GELU -> Dropout(0.3) -> Linear(1)``;
  * AdamW(lr_lora=5e-4, lr_head=1e-3, wd=0.01), batch 32, bf16 autocast,
    epochs<=8, patience 5, early stop on val AUROC, seed 0;
  * after training the best epoch is restored from the delta file on disk and
    val/test are scored once.

Checkpoints store only the trainable delta (206 tensors, ~8 MB), never the
2.6 GB backbone::

    {"format": "peft-delta-v1",
     "delta":  {name: tensor, ...},
     "meta":   {"fold": k, "epoch": ..., "val_auroc": ..., "base_model": ...}}
"""
import os

# Must be set before ``transformers`` is imported anywhere in the process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gc
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
D_POS, LMAX = 4, 32
D_MODEL = 1280
BASE_MODEL = "facebook/esm2_t33_650M_UR50D"

# Only used by the "initial delta is exactly zero" check; training keeps it False.
DELTA_OFF = False

DELTA_PREFIXES = ("posproj", "posconv", "head")
DELTA_SUFFIXES = ("A.weight", "B.weight")


def set_device(dev):
    global DEV
    DEV = dev


def _ac():
    """Autocast context matching the device (bf16 everywhere, as the reference)."""
    return torch.autocast("cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16)


def is_delta_key(name):
    if name.startswith(DELTA_PREFIXES):
        return True
    return any(name.endswith(s) for s in DELTA_SUFFIXES)


# ----------------------------------------------------------------- delta I/O
def extract_delta(state_dict):
    return {k: v.detach().to("cpu").clone()
            for k, v in state_dict.items() if is_delta_key(k)}


def save_delta(path, model, meta):
    torch.save({"format": "peft-delta-v1",
                "delta": extract_delta(model.state_dict()),
                "meta": dict(meta)}, path)


def load_delta(path, model, verbose=True):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    d = obj["delta"] if "delta" in obj else obj
    missing, unexpected = model.load_state_dict(d, strict=False)
    bad = [k for k in missing if is_delta_key(k)]
    if bad or unexpected:
        raise RuntimeError("delta does not match model: missing=%s unexpected=%s"
                           % (bad[:5], unexpected[:5]))
    if verbose:
        n = sum(v.numel() for v in d.values())
        print("[delta] %s <- %d tensors / %.2fM params"
              % (os.path.basename(path), len(d), n / 1e6), flush=True)
    return obj.get("meta", {})


def verify_complete(model):
    """Every trainable tensor must be covered by the delta filter."""
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and not is_delta_key(n)]
    assert not bad, "trainable tensors outside the delta set: %s" % bad[:5]
    return True


# --------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    """A frozen base linear plus a frozen data-aligned A and a trainable B."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        # Same RNG consumption as the reference (A is overwritten by SVD anyway).
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.A.weight.requires_grad_(False)
        self.scale = alpha / r

    def forward(self, x):
        if DELTA_OFF:
            return self.base(x)
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
        m = rmask.unsqueeze(1).to(h.dtype)                       # [B,1,L] residues
        z = self.posact(self.posconv(self.posproj(h).transpose(1, 2)))
        zf = F.interpolate(z * m, size=LMAX, mode="linear", align_corners=False)
        mf = F.interpolate(m, size=LMAX, mode="linear", align_corners=False)
        z = zf / mf.clamp(min=1e-6)
        return self.head(z.flatten(1)).squeeze(-1)


# ------------------------------------------------------------------ batching
def batchify(seqs, tok, bs):
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], padding=True, truncation=True, max_length=64,
                  return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"]


def residue_mask(attn):
    """Drop <cls>/<eos> from the attention mask."""
    m = attn.clone()
    m[:, 0] = 0
    lens = attn.sum(1)
    m[torch.arange(len(attn), device=attn.device), lens - 1] = 0
    return m


@torch.no_grad()
def evaluate(model, tok, seqs, y, bs=32):
    """(auroc, aupr, logits) on ``seqs`` -- one batched pass, no shuffling."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    ps = []
    for ids, attn in batchify(seqs, tok, bs):
        ids, attn = ids.to(DEV), attn.to(DEV)
        with _ac():
            logits = model(ids, attn, residue_mask(attn))
        ps.append(logits.float().cpu().numpy())
    p = np.concatenate(ps)
    return roc_auc_score(y, p), average_precision_score(y, p), p


def adapter_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def qkv_linears(esm):
    return [getattr(layer.attention.self, name)
            for layer in esm.encoder.layer for name in ("query", "key", "value")]


def unwrap_qkv(esm):
    """Put the original nn.Linear back so the next fold sees a pristine backbone."""
    n = 0
    for layer in esm.encoder.layer:
        for name in ("query", "key", "value"):
            mod = getattr(layer.attention.self, name)
            if isinstance(mod, LoRALinear):
                setattr(layer.attention.self, name, mod.base)
                n += 1
    return n


# ------------------------------------------------------- covariance C and SVD
@torch.no_grad()
def collect_cov(esm, tok, seqs, bs):
    """C = sum x x^T over the q/k/v inputs of ``seqs``, residues only.

    Runs on the *un-adaptered* backbone; the hook keeps only non-pad positions.
    """
    mods = qkv_linears(esm)
    covs = [torch.zeros(D_MODEL, D_MODEL, device=DEV, dtype=torch.float32)
            for _ in range(len(mods))]
    holder = {"m": None}
    handles = []

    def mk(i):
        def hook(mod, inp, out):
            xf = inp[0].float()
            m = holder["m"]
            if m is not None and xf.dim() == 3:
                xf = xf * m.to(xf.dtype)[:, :, None]
            xf = xf.reshape(-1, xf.shape[-1])
            covs[i].add_(xf.t() @ xf)
        return hook

    for i, mod in enumerate(mods):
        handles.append(mod.register_forward_hook(mk(i)))
    n_tok = 0
    esm.eval()
    try:
        for ids, attn in batchify(seqs, tok, bs):
            ids, attn = ids.to(DEV), attn.to(DEV)
            holder["m"] = attn
            n_tok += int(attn.sum().item())
            with _ac():
                esm(input_ids=ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()
        holder["m"] = None
    return covs, n_tok


def svd_init_A(raw_linears, covs, r):
    """A = top-r right singular vectors of W0 @ C (V_r^T)."""
    A_list, S_list = [], []
    for lin, C in zip(raw_linears, covs):
        W0 = lin.weight.detach().float()
        _, S, Vh = torch.linalg.svd(W0 @ C.float(), full_matrices=False)
        A_list.append(Vh[:r, :].contiguous().clone())
        S_list.append(S[:r].clone())
    return A_list, S_list


# ------------------------------------------------------------------ one fold
def train_fold(k, esm, tok, Xtr, ytr, Xva, yva, Xte, yte, args, models_dir, log=print):
    global DELTA_OFF
    t_start = time.time()
    raw_linears = qkv_linears(esm)
    assert len(raw_linears) == 99 and isinstance(raw_linears[0], nn.Linear), \
        "backbone still carries adapters"

    # ---- 1. covariance of THIS fold's training rows only
    t0 = time.time()
    covs, n_tok = collect_cov(esm, tok, Xtr, args.cov_batch)
    for c in covs:
        c.div_(float(n_tok))
    cov_sec = time.time() - t0
    tr_diag = torch.stack([c.diagonal().mean() for c in covs]).mean().item()
    log("[cov] fold%d train=%d val=%d test=%d | %d modules / %d non-pad tokens / "
        "mean trace(C)/d=%.4f (%.1fs)"
        % (k, len(Xtr), len(Xva), len(Xte), len(covs), n_tok, tr_diag, cov_sec))

    # ---- 2. SVD init of A
    t0 = time.time()
    A_list, S_list = svd_init_A(raw_linears, covs, args.rank)
    svd_sec = time.time() - t0
    log("[svd] fold%d %d x torch.linalg.svd(1280x1280 fp32) (%.1fs)"
        % (k, len(A_list), svd_sec))

    # ---- 2b. archive the init: A proves it came from this fold's covariance
    init_path = os.path.join(models_dir, "tlora_trained_init_fold%d.npz" % k)
    names = np.array(["layer%d.%s" % (i // 3, "qkv"[i % 3]) for i in range(len(A_list))],
                     dtype="<U16")
    np.savez(init_path,
             names=names,
             A=torch.stack([a.detach().float().cpu() for a in A_list]).numpy(),
             S=torch.stack([s.detach().float().cpu() for s in S_list]).numpy(),
             r=args.rank, fold=k, n_tokens=n_tok, base_model=BASE_MODEL,
             cov_sec=round(cov_sec, 1), svd_sec=round(svd_sec, 1),
             cov_trace_diag_mean=tr_diag)
    log("[init] fold%d archived %s (%s)"
        % (k, os.path.basename(init_path), " / ".join(
            ["%s %s" % (names[0], tuple(A_list[0].shape))])))
    del covs

    # ---- 3. build the model and stamp A in
    model = ESM_LoRA_PosProj(esm, rank=args.rank, alpha=args.alpha).to(DEV)
    mods = adapter_modules(model)
    assert len(mods) == 99
    with torch.no_grad():
        for m, a in zip(mods, A_list):
            m.A.weight.copy_(a.to(m.A.weight.dtype))
    del A_list, S_list
    if DEV == "cuda":
        torch.cuda.empty_cache()

    lora_params = [m.B.weight for m in mods]
    head_params = (list(model.posproj.parameters())
                   + list(model.posconv.parameters())
                   + list(model.head.parameters()))
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    verify_complete(model)
    log("[tlora] fold%d A frozen %.3fM | trainable B=%.3fM + head=%.3fM | "
        "r=%d alpha=%g scale=%g"
        % (k, sum(m.A.weight.numel() for m in mods) / 1e6, n_lora / 1e6,
           n_head / 1e6, args.rank, args.alpha, args.alpha / args.rank))

    # ---- 4. sanity: initial delta is exactly zero
    probe_ids, probe_attn = next(iter(batchify(Xtr[:8], tok, 8)))
    probe_ids, probe_attn = probe_ids.to(DEV), probe_attn.to(DEV)
    dw_max = max((m.B.weight @ m.A.weight).abs().max().item() for m in mods)
    model.eval()
    with torch.no_grad():
        with _ac():
            on = model(probe_ids, probe_attn, residue_mask(probe_attn)).float()
        DELTA_OFF = True
        with _ac():
            off = model(probe_ids, probe_attn, residue_mask(probe_attn)).float()
        DELTA_OFF = False
    d_logit = (on - off).abs().max().item()
    assert dw_max == 0.0 and d_logit == 0.0, (dw_max, d_logit)
    log("[check] fold%d max|dW|=%.1e max|dlogit|=%.1e" % (k, dw_max, d_logit))

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()

    # ---- 5. sanity: gradients reach B, never A
    model.train()
    yb = torch.tensor(ytr[:8].astype(np.float32), device=DEV)
    with _ac():
        loss = bce(model(probe_ids, probe_attn, residue_mask(probe_attn)), yb)
    loss.backward()
    n_bgrad = sum(1 for m in mods
                  if m.B.weight.grad is not None and m.B.weight.grad.norm().item() > 0)
    a_none = all(m.A.weight.grad is None for m in mods)
    assert n_bgrad == 99 and a_none, (n_bgrad, a_none)
    opt.zero_grad(set_to_none=True)
    log("[check] fold%d B.grad nonzero %d/99, A.grad is None: %s"
        % (k, n_bgrad, a_none))

    # ---- 6. delta round-trip (same spot as the reference: keeps the RNG stream)
    tmp_ck = os.path.join(models_dir, "_roundtrip_fold%d.pt" % k)
    before = extract_delta(model.state_dict())
    save_delta(tmp_ck, model, meta={"probe": "roundtrip"})
    with torch.no_grad():
        for m in mods:
            m.B.weight.normal_(0, 0.02)
    load_delta(tmp_ck, model, verbose=False)
    after = extract_delta(model.state_dict())
    dp = max((before[kk].float() - after[kk].float()).abs().max().item() for kk in before)
    os.remove(tmp_ck)
    assert dp < 1e-5, dp
    log("[check] fold%d delta round-trip %d tensors max|dp|=%.1e" % (k, len(before), dp))

    # ---- 7. train
    best_val, best_ep, bad = -1.0, -1, 0
    best_path = os.path.join(models_dir, "tlora_trained_fold%d.pt" % k)
    stopped_early, nan_steps = False, 0
    t_train = time.time()
    ep = -1
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        order = np.random.permutation(len(Xtr))
        seqs = [Xtr[i] for i in order]
        yy = ytr[order]
        tot, nb = 0.0, 0
        for bi, (ids, attn) in enumerate(batchify(seqs, tok, args.batch)):
            yb = torch.tensor(yy[bi * args.batch:(bi + 1) * args.batch].astype(np.float32),
                              device=DEV)
            ids, attn = ids.to(DEV), attn.to(DEV)
            with _ac():
                logits = model(ids, attn, residue_mask(attn))
                loss = bce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not torch.isfinite(loss):
                nan_steps += 1
            tot += float(loss.item())
            nb += 1
            if args.max_steps and nb >= args.max_steps:
                break
        va_auc, va_ap, _ = evaluate(model, tok, Xva, yva)
        log("[fold%d ep%d] train_bce=%.4f val_auroc=%.4f val_aupr=%.4f (%.0fs)"
            % (k, ep, tot / max(nb, 1), va_auc, va_ap, time.time() - t0))
        if va_auc > best_val:
            best_val, bad, best_ep = va_auc, 0, ep
            save_delta(best_path, model,
                       meta={"fold": k, "epoch": ep, "val_auroc": float(va_auc),
                             "val_aupr": float(va_ap), "variant": "tlora",
                             "base_model": BASE_MODEL})
        else:
            bad += 1
        if bad >= args.patience:
            log("[fold%d] early stop (patience %d)" % (k, args.patience))
            stopped_early = True
            break
        if args.max_steps:
            break

    train_sec = time.time() - t_train
    epochs_run = ep + 1
    load_delta(best_path, model, verbose=False)
    va_auc, va_ap, pva = evaluate(model, tok, Xva, yva)
    te_auc, te_ap, pte = evaluate(model, tok, Xte, yte)
    pva = 1.0 / (1.0 + np.exp(-pva))
    pte = 1.0 / (1.0 + np.exp(-pte))
    wall = time.time() - t_start
    log("[final] fold%d ep%d val_auroc=%.4f val_aupr=%.4f TEST_auroc=%.4f "
        "test_aupr=%.4f | cov=%.0fs svd=%.0fs train=%.0fs wall=%.0fs"
        % (k, best_ep, va_auc, va_ap, te_auc, te_ap, cov_sec, svd_sec, train_sec, wall))

    res = {
        "fold": k, "variant": "tlora", "arm": "tlora_qkv_posproj",
        "init": "A = top-r right singular vectors of W0 @ C, C from this fold's train rows",
        "frozen": "A.weight", "inject": "q/k/v", "n_modules": 99,
        "rank": args.rank, "alpha": args.alpha, "scale": args.alpha / args.rank,
        "batch": args.batch, "lr_lora": args.lr_lora, "lr_head": args.lr_head,
        "patience": args.patience, "seed": args.seed,
        "trainable_params": n_lora + n_head, "trainable_lora_params": n_lora,
        "trainable_head_params": n_head,
        "train_n": len(Xtr), "val_n": len(Xva), "test_n": len(Xte),
        "train_pos": int(ytr.sum()), "val_pos": int(yva.sum()), "test_pos": int(yte.sum()),
        "best_ep": best_ep, "epochs_run": epochs_run,
        "early_stopped": stopped_early, "nan_steps": nan_steps,
        "val_auroc": float(va_auc), "val_aupr": float(va_ap),
        "test_auroc": float(te_auc), "test_aupr": float(te_ap),
        "cov_n_tokens": int(n_tok), "cov_trace_diag_mean": float(tr_diag),
        "cov_sec": round(cov_sec, 1), "svd_sec": round(svd_sec, 1),
        "train_sec": round(train_sec, 1), "wall_sec": round(wall, 1),
        "weights": os.path.basename(best_path),
        "init_npz": os.path.basename(init_path),
    }

    del opt, mods, lora_params, head_params, model
    n_un = unwrap_qkv(esm)
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    log("[cleanup] fold%d removed %d adapters" % (k, n_un))
    return res, pva, pte
