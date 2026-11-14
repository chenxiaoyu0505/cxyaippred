# -*- coding: utf-8 -*-
"""用发布包里的 TLoRA delta 权重做推理（需要 GPU + ESM-2 650M）。

权重 `weights/tlora_fold{0..4}.pt` 是 **peft-delta** 格式：只含 A（冻结的 SVD 初始化）、
B（可训增量）与 posproj/head，约 8 MB/折；冻结基座每次都从 HuggingFace 拉
`facebook/esm2_t33_650M_UR50D`，不在包里。

用法：
  # 1) 只验证权重能不能装回去：读 test.csv，与发布概率对账
  python scripts/infer_tlora.py --verify

  # 2) 给自己的序列打分（csv 需含 sequence 列；多折平均 = 主口径）
  python scripts/infer_tlora.py --csv my_peptides.csv --fold all --out pred.csv

  # 3) 单折
  python scripts/infer_tlora.py --csv my_peptides.csv --fold 0

注意：batch 必须保持 32。头部的卷积发生在掩码之前，pad 隐状态会被卷进序列尾部，
bs 越小结果偏移越大（bs=16 时 fold0 test AUROC 0.7365 vs bs=32 的 0.7445）。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
WEIGHTS = os.path.join(ROOT, "weights")
SHIPPED = os.path.join(ROOT, "tlora5")
sys.path.insert(0, HERE)


def run(folds, seqs, device="cuda", bs=32, hf_offline=False):
    if hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer
    from tlora_model import ESM_LoRA_PosProj, predict, unwrap_qkv
    import peft_io

    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                    torch_dtype=torch.float16).to(device)
    esm.eval()
    esm.requires_grad_(False)

    probs, metas = [], []
    for k in folds:
        ck = os.path.join(WEIGHTS, f"tlora_fold{k}.pt")
        _, meta, kind = peft_io.read_any(ck)
        rank, alpha = int(meta.get("rank", 8)), float(meta.get("alpha", 16))
        model = ESM_LoRA_PosProj(esm, rank=rank, alpha=alpha).to(device)
        peft_io.load_delta(ck, model, verbose=False)
        p = predict(model, tok, seqs, bs=bs, device=device)
        probs.append(p)
        metas.append({"fold": k, "source": kind, "rank": rank, "alpha": alpha,
                      "meta": {kk: vv for kk, vv in meta.items()
                               if isinstance(vv, (int, float, str))}})
        del model
        unwrap_qkv(esm)
        torch.cuda.empty_cache()
        print(f"[fold{k}] 已打分 {len(seqs)} 条", flush=True)
    return np.mean(probs, axis=0), probs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all", help="折号，逗号分隔，或 all")
    ap.add_argument("--csv", default="", help="含 sequence 列的 CSV")
    ap.add_argument("--out", default="", help="输出 CSV 路径")
    ap.add_argument("--verify", action="store_true", help="用 data/test.csv 与发布概率对账")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-offline", action="store_true")
    args = ap.parse_args()

    folds = [0, 1, 2, 3, 4] if args.fold == "all" else [int(x) for x in args.fold.split(",")]

    if args.verify:
        test = pd.read_csv(os.path.join(DATA, "test.csv"))
        seqs = test.sequence.tolist()
        p, per_fold, metas = run(folds, seqs, args.device, args.bs, args.hf_offline)
        from sklearn.metrics import average_precision_score, roc_auc_score
        print(f"\n[verify] 逐折 AUROC: " + ", ".join(
            f"f{k}={roc_auc_score(test.label, q):.4f}" for k, q in zip(folds, per_fold)))
        if len(folds) == 5:
            print(f"[verify] 五折集成 AUROC {roc_auc_score(test.label, p):.4f} "
                  f"AUPR {average_precision_score(test.label, p):.4f}（发布值 0.7854 / 0.6353）")
        # 逐折与发布概率对账（发布版每折一个 CSV）
        print("\n[verify] 与 repro/tlora5 发布概率逐折对比：")
        bad = []
        for k, q in zip(folds, per_fold):
            d = pd.read_csv(os.path.join(SHIPPED, f"{k}_test_probs.csv"))
            m = {s: v for s, v in zip(d.sequence, d.prob)}
            ship = np.array([m[s] for s in seqs])
            dp = float(np.abs(q - ship).max())
            auc_mine = roc_auc_score(test.label, q)
            auc_ship = roc_auc_score(test.label, ship)
            ok = dp < 5e-3 and abs(auc_mine - auc_ship) < 1e-3
            if not ok:
                bad.append(k)
            print(f"  {'OK ' if ok else 'DIFF'} fold{k}  max|Δp|={dp:.2e}  "
                  f"AUROC 重算 {auc_mine:.4f} vs 发布 {auc_ship:.4f}")
        print("\n[verify] " + ("全部一致：delta 权重可复现发布概率" if not bad
                               else f"不一致折：{bad}（换卡/换驱动会略移，先看 AUROC 是否仍一致）"))
        return 0 if not bad else 1

    if not args.csv:
        print("[err] 需要 --csv 或 --verify")
        return 2
    df = pd.read_csv(args.csv)
    if "sequence" not in df.columns:
        print("[err] CSV 需要 sequence 列")
        return 2
    p, per_fold, metas = run(folds, df.sequence.tolist(), args.device, args.bs, args.hf_offline)
    out = args.out or os.path.join(ROOT, "out", "pred.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"sequence": df.sequence, "prob": p}).to_csv(out, index=False)
    print(f"\n[saved] {out}   （prob 已 Sigmoid，>0.5 视为阳性）")
    print("[note] 该概率是单视角分数；正式判读还需与三个手工视角按 "
          "hand .370 / gap3 .222 / k4 .185 / TLoRA .222 融合，工作点 τ 要在自己的 OOF 上重算。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
