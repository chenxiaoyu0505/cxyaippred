# -*- coding: utf-8 -*-
"""三态输出（判阳 / 判阴 / 拒答）+ 弃权曲线，且**覆盖率不来自 test**。

这个脚本专门处理上次遗留的那个软肋：
「保留 70% 这个 p，是看着 test 曲线挑的」——那属于泄漏。

这里给两条不碰 test 的路子：

1. **α 驱动**（`--alpha`）：你先说能容忍多大错误率（比如 0.10），
   然后在**池内 OOF** 上找阈值 t_s，使「回答部分的错误率 ≤ α」且覆盖率最大。
   **覆盖率是算出来的，不是挑出来的**，换 α 就是换覆盖率，与 test 无关。
2. **整条曲线 + AURC**（`--curve`）：AURC 是 risk-coverage 曲线下面积，
   完全不需要选定任何单点 p —— 报这个数就没有 p 选择问题。

用法：
  python scripts/abstain.py                      # 曲线 + AURC（OOF 与 test 并列）
  python scripts/abstain.py --alpha 0.10         # 按 α 在 OOF 上定阈值，再在 test 上验收一次
  python scripts/abstain.py --alpha 0.05 --alpha 0.10 --alpha 0.15
  python scripts/abstain.py --nested             # 留一折：阈值不来自被评估的那折
  python scripts/abstain.py --probs my.npy       # 给自己的 n×4 概率打三态标签

拒答分：s = |p − τ|，τ = 池内 OOF 上 MCC 最优的工作点（本臂 0.396）。
s 越小越可疑，s < t_s 即拒答。判阳规则仍是 p ≥ τ。

⚠️ 诚实边界：这是**经验分位数**，不是带有限样本保证的 conformal。
「错误率 ≤ α」在 OOF 上成立、在 test 上只是**验收一次**；不要把它写成数学保证。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)

from run_4view import (load_split, view_arrays, tlora_arrays, greedy,  # noqa: E402
                       pick_tau, VIEWS_ML)


# --------------------------------------------------------------- 基本构件
def suspicious(p, tau):
    """拒答分：越大越可信。"""
    return np.abs(p - tau)


def select_threshold(p, y, tau, alpha):
    """在给定（池内）样本上，找最小阈值 t_s 使保留部分的错误率 ≤ alpha，且保留最多。

    返回 (t_s, coverage, risk)；若连全保留都 ≤ alpha，则 t_s = -inf（不拒答）。
    """
    s = suspicious(p, tau)
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    err = ((p[order] >= tau).astype(int) != y_sorted).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    ok = np.where(cum <= alpha)[0]
    if len(ok) == 0:                       # 一个都不满足（极小的 α），退化为最保守的一小撮
        n = max(1, int(0.01 * len(err)))
    else:
        n = int(ok.max()) + 1
    t_s = float(s_sorted[n - 1])
    keep = s >= t_s
    return t_s, float(keep.mean()), float(err[:n].mean())


def evaluate(p, y, tau, t_s):
    """给定拒答阈值，报覆盖率与保留部分的指标。"""
    keep = suspicious(p, tau) >= t_s
    if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y[keep], p[keep]))
    risk = float(((p[keep] >= tau).astype(int) != y[keep]).mean()) if keep.any() else float("nan")
    return {"coverage": round(float(keep.mean()), 4), "n": int(keep.sum()),
            "risk": round(risk, 4), "acc": round(1 - risk, 4) if keep.any() else float("nan"),
            "auroc_kept": round(auc, 4) if auc == auc else None}


def risk_coverage(p, y, tau, levels=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)):
    s = suspicious(p, tau)
    order = np.argsort(-s)
    rows = []
    for cov in levels:
        n = max(1, int(round(cov * len(y))))
        idx = order[:n]
        risk = float(((p[idx] >= tau).astype(int) != y[idx]).mean())
        auc = float(roc_auc_score(y[idx], p[idx])) if len(np.unique(y[idx])) > 1 else float("nan")
        rows.append({"coverage": cov, "n": n, "risk": round(risk, 4),
                     "acc": round(1 - risk, 4), "auroc_kept": round(auc, 4) if auc == auc else None})
    return rows


def aurc(p, y, tau, grid=96):
    """AURC = ∫ risk d(coverage)，覆盖率 0.05→1.0。越低越好；随机弃权 ≈ 1-acc。"""
    s = suspicious(p, tau)
    order = np.argsort(-s)
    covs = np.linspace(0.05, 1.0, grid)
    risks = []
    for c in covs:
        n = max(10, int(round(c * len(y))))
        idx = order[:n]
        risks.append(1.0 - float(((p[idx] >= tau).astype(int) == y[idx]).mean()))
    a = float(np.trapz(risks, covs))
    base = float(1.0 - ((p >= tau).astype(int) == y).mean())
    return a, base


def three_state(p, tau, t_s):
    out = np.where(suspicious(p, tau) < t_s, "ABSTAIN",
                   np.where(p >= tau, "POS", "NEG"))
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, action="append",
                    help="可容忍的回答错误率，可重复给多个（如 0.05 0.10 0.15）")
    ap.add_argument("--curve", action="store_true", help="打印 risk-coverage 曲线")
    ap.add_argument("--nested", action="store_true", help="留一折：阈值来自其余 4 折")
    ap.add_argument("--probs", help="n×4 概率文件（列序 hand,gap3,k4,TLoRA），打三态标签")
    ap.add_argument("--out", default=os.path.join(OUT, "abstain.json"))
    args = ap.parse_args()

    table, folds, test, seq2row, oof_idx = load_split()
    y_pool, y_test = table.label.values, test.label.values

    OOF, TST = {}, {}
    for v in VIEWS_ML:
        OOF[v], TST[v] = view_arrays(v, table, test, oof_idx)
    OOF["TLoRA"], TST["TLoRA"] = tlora_arrays(table, test, folds, oof_idx)
    ok = ~np.isnan(OOF["hand"])

    w, ens, _ = greedy(VIEWS_ML + ["TLoRA"], OOF, y_pool[ok], ok)
    tau, _ = pick_tau(y_pool[ok], ens)                      # τ 只用池内 OOF
    p_oof = ens                                             # 池内 OOF 混合概率
    p_test = sum(w[v] * TST[v] for v in w)
    print(f"[blend] 权重 " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))
    print(f"[tau] 池内 OOF MCC 最优 τ = {tau:.4f}（test 未参与）")
    print(f"[data] 池内 OOF n={int(ok.sum())}   test n={len(y_test)}")

    res = {"weights": {k: round(float(v), 6) for k, v in w.items()},
           "tau_from_oof": round(float(tau), 4)}

    # ---- 曲线 + AURC ----
    if args.curve or not args.alpha:
        print("\n--- risk-coverage（按 |p−τ| 从可信到可疑）---")
        print(f"{'覆盖率':>8s}{'n':>6s}{'错误率':>9s}{'acc':>9s}{'AUROC':>9s}"
              f"   {'n':>6s}{'错误率':>9s}{'acc':>9s}{'AUROC':>9s}")
        print(f"{'':>8s}{'':>6s}{'  ← 池内 OOF':>27s}{'':>9s}   {'':>6s}{'  ← test（仅报告）':>27s}{'':>9s}")
        c_oof = risk_coverage(p_oof, y_pool[ok], tau)
        c_test = risk_coverage(p_test, y_test, tau)
        for a, b in zip(c_oof, c_test):
            print(f"{a['coverage']*100:7.0f}%{a['n']:6d}{a['risk']:9.4f}{a['acc']:9.4f}"
                  f"{a['auroc_kept'] or float('nan'):9.4f}   {b['n']:6d}{b['risk']:9.4f}"
                  f"{b['acc']:9.4f}{b['auroc_kept'] or float('nan'):9.4f}")
        a_oof, b_oof = aurc(p_oof, y_pool[ok], tau)
        a_test, b_test = aurc(p_test, y_test, tau)
        print(f"\nAURC（越低越好）  OOF {a_oof:.4f}（随机弃权 {b_oof:.4f}）   "
              f"test {a_test:.4f}（随机弃权 {b_test:.4f}）")
        print("→ 报 AURC 或整条曲线，就不存在「挑哪个覆盖率」的问题。")
        res["curve"] = {"oof": c_oof, "test": c_test,
                        "aurc_oof": round(a_oof, 4), "aurc_random_oof": round(b_oof, 4),
                        "aurc_test": round(a_test, 4), "aurc_random_test": round(b_test, 4)}

    # ---- α 驱动 ----
    if args.alpha:
        print("\n--- 按 α 在池内 OOF 上定阈值（覆盖率是算出来的）---")
        res["alpha"] = []
        for al in args.alpha:
            t_s, cov_oof, risk_oof = select_threshold(p_oof, y_pool[ok], tau, al)
            ev_test = evaluate(p_test, y_test, tau, t_s)
            ev_oof = evaluate(p_oof, y_pool[ok], tau, t_s)
            print(f"  α={al:.2f}  t_s={t_s:.4f} | OOF 覆盖 {ev_oof['coverage']*100:5.1f}% "
                  f"风险 {ev_oof['risk']:.4f} | test 覆盖 {ev_test['coverage']*100:5.1f}% "
                  f"风险 {ev_test['risk']:.4f} acc {ev_test['acc']:.4f}")
            res["alpha"].append({"alpha": al, "t_s": round(t_s, 4),
                                 "oof": ev_oof, "test": ev_test})
        print("→ α 是你要先声明的成本；覆盖率是它的函数。test 只在这里被看一次。")

    # ---- 留一折稳定性 ----
    if args.nested:
        print("\n--- 留一折：阈值来自其余 4 折（nested）---")
        res["nested"] = []
        va_rows = {k: oof_idx[k] for k in range(5)}
        all_idx = np.where(ok)[0]
        for k in range(5):
            tr = np.setdiff1d(all_idx, va_rows[k])
            te = va_rows[k]
            t_k, _ = pick_tau(y_pool[tr], p_oof[tr])
            t_s, _, _ = select_threshold(p_oof[tr], y_pool[tr], t_k, 0.10)
            ev = evaluate(p_oof[te], y_pool[te], t_k, t_s)
            res["nested"].append({"fold": k, "tau": round(float(t_k), 4),
                                  "t_s": round(float(t_s), 4), **ev})
            print(f"  fold{k}: τ={t_k:.4f} t_s={t_s:.4f} → 本折覆盖 {ev['coverage']*100:5.1f}% "
                  f"风险 {ev['risk']:.4f} acc {ev['acc']:.4f}")
        covs = [d["coverage"] for d in res["nested"]]
        print(f"  → 覆盖率跨折 {min(covs)*100:.1f}%–{max(covs)*100:.1f}%，"
              f"均值 {np.mean(covs)*100:.1f}%（这就是「换一折会变成多少」的量级）")

    # ---- 给自己的概率打三态 ----
    if args.probs:
        P = np.load(args.probs)
        if P.ndim != 2 or P.shape[1] != 4:
            raise SystemExit("--probs 需要 n×4（列序 hand,gap3,k4,TLoRA）")
        pb = sum(w[v] * P[:, i] for i, v in enumerate(["hand", "gap3", "k4", "TLoRA"]))
        t_s = res["alpha"][0]["t_s"] if res.get("alpha") else \
            select_threshold(p_oof, y_pool[ok], tau, 0.10)[0]
        st = three_state(pb, tau, t_s)
        df = pd.DataFrame({"p": pb, "suspicion": np.abs(pb - tau), "state": st})
        dst = os.path.splitext(args.out)[0] + "_states.csv"
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"\n[three-state] τ={tau:.4f} 拒答阈值 t_s={t_s:.4f}")
        print(f"  判阳 {int((st=='POS').sum())}  判阴 {int((st=='NEG').sum())}  拒答 {int((st=='ABSTAIN').sum())}")
        print(f"[saved] {dst}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}")

    # ---- 与 expected.json 对账（若有 abstain 段）----
    exp_path = os.path.join(ROOT, "expected.json")
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            exp = json.load(f)
        want = exp.get("abstain")
        if want and res.get("curve"):
            tol = float(exp.get("tolerance", 2e-4))
            fails = []
            for key in ("aurc_oof", "aurc_test"):
                if key in want:
                    d = abs(res["curve"][key] - want[key])
                    okk = d <= 1e-3
                    print(f"  {'OK ' if okk else 'FAIL'} {key:12s} got {res['curve'][key]:.4f} "
                          f"want {want[key]:.4f}  |Δ|={d:.1e}")
                    if not okk:
                        fails.append(key)
            print(f"[check] {'通过' if not fails else '失败：%s' % fails}")
            return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
