# -*- coding: utf-8 -*-
"""四视角融合复现（纯 CPU，几秒）。

四视角 = 3 个手工特征视角（hand-704 / gap3-200 / k4-200）+ 1 个深度学习视角
（冻结 ESM-2 650M + TLoRA(q/k/v) + posproj-adaptive 头，5 折概率平均）。

本脚本只做**融合层**：读取 `repro/probs/` 与 `repro/tlora5/` 里已冻结的
逐视角、逐折概率，重跑 Caruana 带放回贪心选权（27 轮，目标 = 池内 OOF AUROC），
然后在固定 test 897 条上评估。所有选择都只用 OOF，test 不参与。

用法：
  python scripts/run_4view.py                # 复现 + 与 expected.json 逐项对账
  python scripts/run_4view.py --no-check     # 只打印，不判 PASS/FAIL
  python scripts/run_4view.py --curve        # 额外打印弃权（risk-coverage）曲线

产物：`repro/out/report.json`
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # repro/
DATA = os.path.join(ROOT, "data")
PROB = os.path.join(ROOT, "probs")
TLORA = os.path.join(ROOT, "tlora5")
OUT = os.path.join(ROOT, "out")

GREEDY_ROUNDS = 27
VIEWS_ML = ["hand", "gap3", "k4"]


# ----------------------------------------------------------------- 数据装载
def load_split():
    table = pd.read_csv(os.path.join(DATA, "table.csv"))
    folds = pd.read_csv(os.path.join(DATA, "folds.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    seq2row = {s: i for i, s in enumerate(table.sequence)}
    oof_idx = {}
    for k in range(5):
        va = folds[(folds.fold == k) & (folds.split == "val")].sequence.tolist()
        oof_idx[k] = np.array([seq2row[s] for s in va])
    return table, folds, test, seq2row, oof_idx


def view_arrays(view, table, test, oof_idx):
    """返回 (pooled OOF 向量, test 向量)。test = 5 折概率平均。"""
    N, NT = len(table), len(test)
    oof = np.full(N, np.nan)
    tests = []
    for k in range(5):
        oof[oof_idx[k]] = np.load(os.path.join(PROB, f"fold{k}_{view}_val.npy"))
        t = np.load(os.path.join(PROB, f"fold{k}_{view}_test.npy"))
        tests.append(t[:NT] if len(t) != NT else t)
    return oof, np.mean(tests, axis=0)


def tlora_arrays(table, test, folds, oof_idx):
    """TLoRA 视角：逐折 CSV → 按序列对齐。"""
    N, NT = len(table), len(test)
    oof = np.full(N, np.nan)
    tests = []
    for k in range(5):
        va = pd.read_csv(os.path.join(TLORA, f"{k}_val_probs.csv"))
        m = {s: p for s, p in zip(va.sequence, va.prob)}
        oof[oof_idx[k]] = np.array([m[s] for s in
                                    folds[(folds.fold == k) & (folds.split == "val")].sequence])
        te = pd.read_csv(os.path.join(TLORA, f"{k}_test_probs.csv"))
        mt = {s: p for s, p in zip(te.sequence, te.prob)}
        tests.append(np.array([mt[s] for s in test.sequence]))
    return oof, np.mean(tests, axis=0)


# --------------------------------------------------------------- 贪心选权
def greedy(vs, OOF, y, ok, rounds=GREEDY_ROUNDS):
    """Caruana 带放回贪心：每轮挑使 OOF AUROC 最大的视角，权重 = 被挑次数/轮数。"""
    ens = np.zeros(int(ok.sum()))
    cnt = {v: 0 for v in vs}
    trace = []
    for r in range(rounds):
        best, ba = None, -1.0
        for v in vs:
            a = roc_auc_score(y, (ens * r + OOF[v][ok]) / (r + 1))
            if a > ba:
                best, ba = v, a
        cnt[best] += 1
        ens = (ens * r + OOF[best][ok]) / (r + 1)
        trace.append({"round": r + 1, "pick": best, "oof_auroc": float(ba)})
    w = {v: c / rounds for v, c in cnt.items() if c}
    return w, ens, trace


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pick_tau(y, p):
    """工作点 τ：池内 OOF 上 MCC 最大。test 不参与。"""
    cand = np.unique(np.round(p, 6))
    best, bt = -2.0, 0.5
    for t in cand:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best:
            best, bt = m, float(t)
    return bt, best


def binmetrics(y, p, tau):
    yh = (p >= tau).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    return {"tau": round(float(tau), 4),
            "sn": round(tp / max(tp + fn, 1), 4),
            "sp": round(tn / max(tn + fp, 1), 4),
            "acc": round((tp + tn) / len(y), 4),
            "mcc": round(matthews_corrcoef(y, yh), 4),
            "errors": fn + fp}


def risk_coverage(y, p, levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """按置信度 |p-0.5| 保留前 cov 比例样本后的 acc / AUROC。"""
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    rows = []
    for cov in levels:
        n = int(round(cov * len(y)))
        idx = order[:n]
        yy, pp = y[idx], p[idx]
        rows.append({"coverage": cov, "n": n,
                     "acc": round(float(((pp >= 0.5).astype(int) == yy).mean()), 4),
                     "auroc": round(float(roc_auc_score(yy, pp)), 4),
                     "err_rate": round(float(((pp >= 0.5).astype(int) != yy).mean()), 4)})
    return rows


# --------------------------------------------------------------------- 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true", help="不与 expected.json 对账")
    ap.add_argument("--curve", action="store_true", help="打印 risk-coverage 曲线")
    args = ap.parse_args()

    table, folds, test, seq2row, oof_idx = load_split()
    y_pool, y_test = table.label.values, test.label.values
    print(f"[data] pool={len(table)}(+{int(y_pool.sum())})  test={len(test)}(+{int(y_test.sum())})")

    OOF, TST = {}, {}
    for v in VIEWS_ML:
        OOF[v], TST[v] = view_arrays(v, table, test, oof_idx)
    OOF["TLoRA"], TST["TLoRA"] = tlora_arrays(table, test, folds, oof_idx)
    ok = ~np.isnan(OOF["hand"])
    print(f"[oof] 覆盖 {int(ok.sum())}/{len(table)}")

    print("\n--- 单视角 ---")
    per_view = {}
    for v in ["hand", "gap3", "k4", "TLoRA"]:
        o, t = float(roc_auc_score(y_pool[ok], OOF[v][ok])), float(roc_auc_score(y_test, TST[v]))
        a = float(average_precision_score(y_test, TST[v]))
        per_view[v] = {"oof_auroc": o, "test_auroc": t, "test_aupr": a}
        print(f"  {v:6s} OOF {o:.4f}   test {t:.4f} / AUPR {a:.4f}")

    arms = {}
    for tag, vs in [("3view", VIEWS_ML),
                    ("4view+TLoRA", VIEWS_ML + ["TLoRA"])]:
        w, ens, trace = greedy(vs, OOF, y_pool[ok], ok)
        blend = sum(w[v] * TST[v] for v in w)
        auc = float(roc_auc_score(y_test, blend))
        aupr = float(average_precision_score(y_test, blend))
        lo, hi = boot_ci(y_test, blend)
        tau, oof_mcc = pick_tau(y_pool[ok], ens)
        arms[tag] = {"weights": {k: round(float(v), 6) for k, v in w.items()},
                     "oof_auroc": float(roc_auc_score(y_pool[ok], ens)),
                     "test_auroc": auc, "test_aupr": aupr,
                     "test_auroc_ci": [round(lo, 4), round(hi, 4)],
                     "tau_from_oof": round(tau, 4), "oof_mcc_at_tau": round(float(oof_mcc), 4),
                     "test_at_tau": binmetrics(y_test, blend, tau),
                     "test_at_0.5": binmetrics(y_test, blend, 0.5),
                     "trace": trace}
        print(f"\n--- {tag} ---")
        print("  权重 " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))
        print(f"  OOF AUROC {arms[tag]['oof_auroc']:.4f}")
        print(f"  test AUROC {auc:.4f} [95% CI {lo:.4f}-{hi:.4f}]  AUPR {aupr:.4f}")
        print(f"  τ(OOF MCC 最优) = {tau:.3f}")
        for key in ("test_at_tau", "test_at_0.5"):
            d = arms[tag][key]
            print(f"  {key:12s} Sn {d['sn']:.3f} Sp {d['sp']:.3f} "
                  f"Acc {d['acc']:.3f} MCC {d['mcc']:.3f} 错 {d['errors']}")

    curve = None
    if args.curve:
        w = arms["4view+TLoRA"]["weights"]
        blend = sum(w[v] * TST[v] for v in w)
        curve = risk_coverage(y_test, blend)
        print("\n--- 弃权曲线（按 |p-0.5| 保留）---")
        for r in curve:
            print(f"  保留 {r['coverage']*100:3.0f}%  n={r['n']:3d}  "
                  f"acc {r['acc']:.4f}  AUROC {r['auroc']:.4f}  错误率 {r['err_rate']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    rep = {"data": {"pool": len(table), "test": len(test)},
           "per_view": per_view, "arms": arms, "curve": curve,
           "note": "权重的选择只用了池内 OOF；test 897 条只在最后评估一次。"}
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {os.path.join(OUT, 'report.json')}")

    # ---- 与 expected.json 对账 ----
    exp_path = os.path.join(ROOT, "expected.json")
    if args.no_check or not os.path.exists(exp_path):
        print("[check] 跳过（--no-check 或没有 expected.json）")
        return 0
    with open(exp_path, encoding="utf-8") as f:
        exp = json.load(f)
    tol = float(exp.get("tolerance", 2e-4))
    fails, checks = [], []
    for tag, want in exp["arms"].items():
        got = arms[tag]
        for key in ("test_auroc", "test_aupr", "oof_auroc"):
            if key in want:
                delta = abs(got[key] - want[key])
                checks.append((f"{tag}.{key}", got[key], want[key], delta, delta <= tol))
        for vname, wv in want.get("weights", {}).items():
            delta = abs(got["weights"].get(vname, -1) - wv)
            checks.append((f"{tag}.w[{vname}]", got["weights"].get(vname, -1), wv, delta, delta <= 1e-6))
    for vname, want in exp.get("per_view", {}).items():
        for key in ("test_auroc", "test_aupr"):
            if key in want:
                delta = abs(per_view[vname][key] - want[key])
                checks.append((f"view[{vname}].{key}", per_view[vname][key], want[key], delta, delta <= tol))
    print("\n--- 与 expected.json 对账 ---")
    for name, got, want, delta, passed in checks:
        print(f"  {'OK ' if passed else 'FAIL'} {name:26s} got {got:.6f}  want {want:.6f}  |Δ|={delta:.1e}")
        if not passed:
            fails.append(name)
    print(f"\n[check] {len(checks) - len(fails)}/{len(checks)} 通过"
          + ("" if not fails else f"；失败项：{fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
