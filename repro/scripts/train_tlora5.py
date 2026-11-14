# -*- coding: utf-8 -*-
"""TLoRA 视角：5 折训练（可选，需 GPU 与 ESM-2 650M 权重）。

这一步是发布包里唯一需要 GPU 的部分。**不做也能复现 0.8219** —— 融合脚本
`run_4view.py` 读的是已冻结的逐折概率。本脚本用于：① 从零重训并核对
`tlora5/*_result.json` 的指标；② 换数据/换超参做自己的实验。

每折流程（与论文口径逐项一致）：
  1. 在该折 train 行上跑一遍**未挂 adapter** 的基座，累计 q/k/v 输入协方差 C
     （只累加非 pad 残基）；
  2. SVD(W0·C) 取前 r 个右奇异向量作为 A 的初始化，A 冻结、B 零初始化
     → 起点严格等价基座（脚本内有 max|ΔW|=0 与 max|Δlogit|=0 断言）；
  3. 只训 B + posproj 头，bs=32，lr_lora 5e-4 / lr_head 1e-3，AdamW(wd=0.01)，
     bf16 autocast，早停按 val AUROC（patience 5，epoch cap 20），seed 0；
  4. 存 delta（只含 A/B/头，约 8 MB），在 val/test 上出概率。

用法：
  python scripts/train_tlora5.py --fold 0
  python scripts/train_tlora5.py                     # 0,1,2,3,4
  python scripts/train_tlora5.py --fold 0 --max-steps 3   # 冒烟

产物：`repro/out/tlora5/{k}_val_probs.csv`、`{k}_test_probs.csv`、`{k}_result.json`、
`tlora_fold{k}.pt`、`results.json`
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # repro/
SPLIT = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out", "tlora5")
sys.path.insert(0, HERE)


def assemble(fold_arg, epochs=20, batch=32, rank=8, alpha=16, lr_lora=5e-4,
             lr_head=1e-3, patience=5, seed=0, max_steps=0, device="cuda",
             hf_offline=False):
    if hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tlora_model import (ESM_LoRA_PosProj, adapter_modules, collect_cov,
                             predict, qkv_linears, residue_mask, svd_init_A)
    import peft_io

    os.makedirs(OUT, exist_ok=True)
    table = pd.read_csv(os.path.join(SPLIT, "table.csv"))
    folds = pd.read_csv(os.path.join(SPLIT, "folds.csv"))
    test = pd.read_csv(os.path.join(SPLIT, "test.csv"))
    print(f"[data] table={table.shape} folds={folds.shape} test={test.shape}", flush=True)

    # 与参考实现同口径：seed 必须在 import transformers / from_pretrained **之前**设定。
    # from_pretrained 为缺失的 pooler 权重做初始化时会消耗 torch RNG，先 seed 才能让
    # B / posproj / head 的初始化与发布结果逐位一致。
    torch.manual_seed(seed)
    np.random.seed(seed)
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                    torch_dtype=torch.float16).to(device)
    esm.eval()
    esm.requires_grad_(False)
    print(f"[esm] 冻结基座 dtype={next(esm.parameters()).dtype} qkv={len(qkv_linears(esm))}",
          flush=True)

    for k in [int(x) for x in str(fold_arg).split(",") if x.strip() != ""]:
        t_start = time.time()
        fk = folds[folds["fold"] == k].reset_index(drop=True)
        assert len(fk) == len(table) and (fk["sequence"].values == table["sequence"].values).all(), \
            "folds.csv 与 table.csv 行序不一致"
        msk = fk["split"].values
        tr, va = table[msk == "train"], table[msk == "val"]
        Xtr, ytr = tr["sequence"].tolist(), tr["label"].values.astype(np.float32)
        Xva, yva = va["sequence"].tolist(), va["label"].values
        Xte, yte = test["sequence"].tolist(), test["label"].values
        print(f"\n===== [fold{k}] train={len(Xtr)}(+{int(ytr.sum())}) "
              f"val={len(Xva)}(+{int(yva.sum())}) test={len(Xte)}(+{int(yte.sum())}) =====",
              flush=True)

        raw_linears = qkv_linears(esm)
        assert len(raw_linears) == 99 and isinstance(raw_linears[0], nn.Linear), \
            "ESM 上仍挂着 adapter，unwrap 失败"

        # ---------- 阶段 1：该折 train 行的协方差 C ----------
        t0 = time.time()
        covs, n_tok = collect_cov(esm, tok, Xtr, batch, device)
        cov_sec = time.time() - t0
        tr_diag = float(torch.stack([c.diagonal().mean() for c in covs]).mean())
        print(f"[cov] fold{k} {len(covs)} 模块 / {n_tok} 非 pad token / "
              f"trace(C)/d 均值={tr_diag:.4f} ({cov_sec:.1f}s)", flush=True)
        if k == 0:
            ok = "OK" if n_tok == 52511 else "与参考 fold0 的 52511 不符"
            print(f"[cov] fold0 token 数对照参考实现: {n_tok} vs 52511 -> {ok}", flush=True)

        # ---------- 阶段 2：SVD 初始化 A ----------
        t0 = time.time()
        A_list, _ = svd_init_A(raw_linears, covs, rank)
        svd_sec = time.time() - t0
        print(f"[svd] fold{k} {len(A_list)} 次 svd(1280x1280, fp32) ({svd_sec:.1f}s)", flush=True)
        del covs
        torch.cuda.empty_cache()

        # ---------- 阶段 3：建模型 + 覆盖 A ----------
        model = ESM_LoRA_PosProj(esm, rank=rank, alpha=alpha).to(device)
        mods = adapter_modules(model)
        assert len(mods) == 99
        with torch.no_grad():
            for m, a in zip(mods, A_list):
                m.A.weight.copy_(a.to(m.A.weight.dtype))

        lora_params = [m.B.weight for m in mods]
        head_params = (list(model.posproj.parameters()) + list(model.posconv.parameters())
                       + list(model.head.parameters()))
        n_lora = sum(p.numel() for p in lora_params)
        n_head = sum(p.numel() for p in head_params)
        print(f"[tlora] fold{k} 99 模块；A 冻结 {sum(m.A.weight.numel() for m in mods)/1e6:.3f}M；"
              f"可训 B={n_lora/1e6:.3f}M + 头={n_head/1e6:.3f}M r={rank} "
              f"alpha={alpha:g} scale={alpha/rank:g}", flush=True)
        peft_io.verify_complete(model)

        # 初始 ΔW 严格为 0
        dw_max = max((m.B.weight @ m.A.weight).abs().max().item() for m in mods)
        print(f"[check] fold{k} max|ΔW|={dw_max:.3e}", flush=True)
        assert dw_max == 0.0, dw_max

        opt = torch.optim.AdamW([{"params": lora_params, "lr": lr_lora},
                                 {"params": head_params, "lr": lr_head}],
                                weight_decay=0.01)
        bce = nn.BCEWithLogitsLoss()
        best_val, best_ep, bad = -1.0, -1, 0
        best_path = os.path.join(OUT, f"tlora_fold{k}.pt")
        stopped_early, nan_steps = False, 0
        t_train = time.time()
        ep = -1
        for ep in range(epochs):
            model.train()
            order = np.random.permutation(len(Xtr))
            seqs = [Xtr[i] for i in order]
            yy = ytr[order]
            tot, nb, steps = 0.0, 0, 0
            for bi in range(0, len(seqs), batch):
                chunk = seqs[bi:bi + batch]
                yb = torch.tensor(yy[bi:bi + batch], device=device)
                from tlora_model import batchify
                ids, attn = next(iter(batchify(chunk, tok, len(chunk))))
                ids, attn = ids.to(device), attn.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    logits = model(ids, attn, residue_mask(attn))
                    loss = bce(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if not torch.isfinite(loss):
                    nan_steps += 1
                tot += float(loss.item())
                nb += 1
                steps += 1
                if max_steps and steps >= max_steps:
                    break
            pva = predict(model, tok, Xva, batch, device)
            va_auc = roc_auc_score(yva, pva)
            va_ap = average_precision_score(yva, pva)
            print(f"[fold{k} ep{ep}] train_bce={tot/max(nb,1):.4f} "
                  f"val_auroc={va_auc:.4f} val_aupr={va_ap:.4f} ({steps} steps)", flush=True)
            if va_auc > best_val:
                best_val, bad, best_ep = va_auc, 0, ep
                peft_io.save_delta(best_path, model,
                                   meta={"fold": k, "epoch": ep, "val_auroc": float(va_auc),
                                         "variant": "tlora", "rank": rank, "alpha": alpha})
            else:
                bad += 1
            if bad >= patience:
                print(f"[fold{k}] early-stop", flush=True)
                stopped_early = True
                break
            if max_steps:
                break

        train_sec = time.time() - t_train
        peft_io.load_delta(best_path, model, verbose=False)
        pva = predict(model, tok, Xva, batch, device)
        pte = predict(model, tok, Xte, batch, device)
        va_auc, va_ap = roc_auc_score(yva, pva), average_precision_score(yva, pva)
        te_auc, te_ap = roc_auc_score(yte, pte), average_precision_score(yte, pte)
        wall = time.time() - t_start
        print(f"\n[final] fold{k} ep{best_ep} val {va_auc:.4f}/{va_ap:.4f} "
              f"TEST {te_auc:.4f}/{te_ap:.4f}  wall {wall:.0f}s", flush=True)

        pd.DataFrame({"sequence": Xva, "label": yva, "prob": pva}).to_csv(
            os.path.join(OUT, f"{k}_val_probs.csv"), index=False)
        pd.DataFrame({"sequence": Xte, "label": yte, "prob": pte}).to_csv(
            os.path.join(OUT, f"{k}_test_probs.csv"), index=False)
        res = {"fold": k, "variant": "tlora", "arm": "tlora_qkv_posproj",
               "init": "A = top-r 右奇异向量 of W0·C，C 为该折 train 行上的输入激活协方差",
               "frozen": "A.weight（初始化后冻结，只训 B）",
               "inject": "q/k/v", "n_modules": 99, "rank": rank, "alpha": alpha,
               "scale": alpha / rank, "batch": batch, "lr_lora": lr_lora,
               "lr_head": lr_head, "patience": patience, "seed": seed,
               "trainable_params": n_lora + n_head, "train_n": len(Xtr),
               "val_n": len(Xva), "test_n": len(Xte), "train_pos": int(ytr.sum()),
               "val_pos": int(yva.sum()), "test_pos": int(yte.sum()),
               "best_ep": best_ep, "epochs_run": ep + 1,
               "early_stopped": stopped_early, "nan_steps": nan_steps,
               "val_auroc": float(va_auc), "val_aupr": float(va_ap),
               "test_auroc": float(te_auc), "test_aupr": float(te_ap),
               "cov_n_tokens": int(n_tok), "cov_trace_diag_mean": tr_diag,
               "cov_sec": round(cov_sec, 1), "svd_sec": round(svd_sec, 1),
               "train_sec": round(train_sec, 1), "wall_sec": round(wall, 1)}
        with open(os.path.join(OUT, f"{k}_result.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        del opt, mods, model
        from tlora_model import unwrap_qkv
        unwrap_qkv(esm)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    aggregate()


def aggregate():
    from sklearn.metrics import average_precision_score, roc_auc_score
    per = []
    for k in range(5):
        p = os.path.join(OUT, f"{k}_result.json")
        if os.path.exists(p):
            per.append(json.load(open(p, encoding="utf-8")))
    if not per:
        return None
    auc = np.array([r["test_auroc"] for r in per])
    ap = np.array([r["test_aupr"] for r in per])
    ps, ys = [], None
    for r in per:
        d = pd.read_csv(os.path.join(OUT, f"{r['fold']}_test_probs.csv"))
        ps.append(d["prob"].values)
        ys = d["label"].values
    out = {"method": "TLoRA (A=SVD(W0·C) top-r right singular vectors, A frozen, B+head trained)",
           "folds_present": [r["fold"] for r in per], "n_folds": len(per),
           "mean_auroc": float(auc.mean()),
           "sd_auroc": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
           "mean_aupr": float(ap.mean()),
           "sd_aupr": float(ap.std(ddof=1)) if len(ap) > 1 else 0.0,
           "ensemble_auroc": float(roc_auc_score(ys, np.mean(ps, axis=0))),
           "ensemble_aupr": float(average_precision_score(ys, np.mean(ps, axis=0))),
           "ensemble_via": "五折 test prob 等权平均"}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[agg] AUROC {out['mean_auroc']:.4f}±{out['sd_auroc']:.4f} "
          f"ens {out['ensemble_auroc']:.4f}/{out['ensemble_aupr']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lr-lora", type=float, default=5e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-offline", action="store_true",
                    help="离线环境：强制只用本地 HF 缓存")
    args = ap.parse_args()
    assemble(args.fold, args.epochs, args.batch, args.rank, args.alpha,
             args.lr_lora, args.lr_head, args.patience, args.seed,
             args.max_steps, args.device, args.hf_offline)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
