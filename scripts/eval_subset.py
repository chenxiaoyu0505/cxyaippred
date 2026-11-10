"""在指定序列子集上重算一个已跑完实验的指标。

为什么需要：`run_5fold.py` 报的是全表指标，但和外部工作对比时
必须在**相同的序列集合**上比。这个脚本只读已落盘的 `test_probs.csv`，
不重训、不改训练管线。

用法：
    # 全集（默认）
    python scripts/eval_subset.py --exp cv5_clean

    # 与 HD-AIP 可比：只算两个数据集共有的序列
    python scripts/eval_subset.py --exp cv5_clean --overlap data/HD-AIP_benchmark_4194.csv

    # 用对方的标签重算（敏感性检查）
    python scripts/eval_subset.py --exp cv5_clean --overlap data/HD-AIP_benchmark_4194.csv --labels from-subset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


def load_oof(exp_dir: Path) -> pd.DataFrame:
    files = sorted(exp_dir.glob("runs/fold*_m0/test_probs.csv"))
    if len(files) < 1:
        raise SystemExit(f"{exp_dir} 下没有 runs/fold*_m0/test_probs.csv")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["fold"] = int(f.parent.name.split("_")[0].replace("fold", ""))
        th = json.load(open(f.parent / "threshold.json"))
        d["val_threshold"] = float(th["threshold"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def report(df: pd.DataFrame, tag: str) -> dict:
    y = df["label"].to_numpy()
    p = df["prob"].to_numpy()
    yhat = (p >= df["val_threshold"].to_numpy()).astype(int)
    out = {
        "tag": tag,
        "n": int(len(df)),
        "n_pos": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "acc": float(accuracy_score(y, yhat)),
    }
    # 逐折均值（如果有多折）
    folds = sorted(df["fold"].unique())
    if len(folds) > 1:
        per = []
        for f in folds:
            d = df[df["fold"] == f]
            if d["label"].nunique() < 2:
                continue
            per.append(
                (
                    roc_auc_score(d["label"], d["prob"]),
                    average_precision_score(d["label"], d["prob"]),
                )
            )
        if per:
            a = np.array(per)
            out["fold_mean_auroc"] = float(a[:, 0].mean())
            out["fold_sd_auroc"] = float(a[:, 0].std(ddof=1)) if len(a) > 1 else 0.0
            out["fold_mean_aupr"] = float(a[:, 1].mean())
            out["fold_sd_aupr"] = float(a[:, 1].std(ddof=1)) if len(a) > 1 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="实验名，例如 cv5_clean")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--overlap", default="", help="CSV(含 sequence 列)；给了就只算两边共有的序列")
    ap.add_argument(
        "--labels",
        default="own",
        choices=["own", "from-subset"],
        help="own=用本实验的标签；from-subset=用 --overlap 文件里的 label 覆盖（敏感性检查）",
    )
    ap.add_argument("--json-out", default="", help="把结果写进这个 json")
    args = ap.parse_args()

    df = load_oof(Path(args.exp_root) / args.exp)
    print(f"[info] {args.exp}: {len(df)} 条 OOF 预测，{df['fold'].nunique()} 折")

    rows = [report(df, "all")]

    if args.overlap:
        other = pd.read_csv(args.overlap)
        keep = set(other["sequence"])
        sub = df[df["sequence"].isin(keep)].copy()
        if args.labels == "from-subset":
            m = other.set_index("sequence")["label"]
            sub["label"] = sub["sequence"].map(m).astype(int)
        dropped = len(df) - len(sub)
        print(f"[info] 子集 {len(sub)} 条（丢 {dropped} 条不在对方数据里）")
        rows.append(report(sub, f"overlap:{Path(args.overlap).name}:{args.labels}"))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(out.round(4).to_string(index=False))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n[info] 写入 {args.json_out}")


if __name__ == "__main__":
    main()
