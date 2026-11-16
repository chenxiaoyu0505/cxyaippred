# -*- coding: utf-8 -*-
"""Fuse the five views and score the 3 / 4 / 5-view ensembles.

Inputs (all produced by this package, no frozen artifacts):
    out/probs/ml_views.npz     oof_{hand,gap3,k4}, test_{...}
    out/probs/tlora_views.npz  oof_tlora, test_tlora, fold_test_tlora
    out/probs/cnn_views.npz    oof_cnn, test_cnn, fold_test_cnn

Fusion = percentile-rank average (``fusion.rank01``), reported twice per tier:

    rank_equal   every view weighs the same
    rank_greedy  Caruana forward selection, 27 rounds, selected on the OOF
                 AUROC of train1 -- weights are a *result*, not a constant

The decision threshold is the OOF MCC optimum (``fusion.tau_from_scores``), so a
reported tau is never 0.5. Test predictions are ranked within test1 and scored
once; no test label is ever used for fitting or selection.

Writes ``out/report.json`` (read by verify.py) and the training-side detail dump
``out/report_train.json``; prints the tables.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C  # noqa: E402
import fusion as F  # noqa: E402

TIERS = {"3view": ["hand", "gap3", "k4"],
         "4view": ["hand", "gap3", "k4", "tlora"],
         "5view": ["hand", "gap3", "k4", "tlora", "cnn"]}
VIEWS = ["hand", "gap3", "k4", "tlora", "cnn"]


def load_views():
    """ML views are required; a deep view is used if (and only if) it was trained.

    Returns (oof, test, y_tr, y_te, L_te); ``oof``/``test`` only contain the
    views that exist, so a run with ``--skip-dl`` still fuses the 3-view tier.
    """
    ml = np.load(os.path.join(C.PROBS, "ml_views.npz"), allow_pickle=False)
    y_tr, y_te = ml["y_tr"].astype(int), ml["y_te"].astype(int)
    L_te = ml["te_length"].astype(int)
    oof = {v: ml["oof_" + v] for v in ("hand", "gap3", "k4")}
    test = {v: ml["test_" + v] for v in ("hand", "gap3", "k4")}

    deep = (("tlora", "tlora_views.npz", "oof_tlora", "test_tlora"),
            ("cnn", "cnn_views.npz", "oof_cnn", "test_cnn"))
    for name, fname, k_oof, k_test in deep:
        p = os.path.join(C.PROBS, fname)
        if not os.path.exists(p):
            print("[fuse] %s view not found (%s) -> tier(s) using it are skipped"
                  % (name, fname), flush=True)
            continue
        z = np.load(p, allow_pickle=False)
        assert np.array_equal(z["y_tr"].astype(int), y_tr), name + " y_tr mismatch"
        assert np.array_equal(z["y_te"].astype(int), y_te), name + " y_te mismatch"
        assert np.array_equal(z["te_length"].astype(int), L_te), name + " length mismatch"
        if "fold_of" in z.files and "fold_of" in ml.files:
            # catches a stale artifact from a different fold protocol
            assert np.array_equal(z["fold_of"].astype(int), ml["fold_of"].astype(int)), (
                "%s was trained under a different fold assignment -- delete out/probs/%s "
                "and retrain, do not fuse across protocols" % (name, fname))
        oof[name] = z[k_oof]
        test[name] = z[k_test]

    for v in oof:
        assert oof[v].shape == (len(y_tr),), (v, oof[v].shape)
        assert test[v].shape == (len(y_te),), (v, test[v].shape)
        assert np.isfinite(oof[v]).all() and np.isfinite(test[v]).all(), v
    return oof, test, y_tr, y_te, L_te



def _folds(name):
    p = os.path.join(C.PROBS, name + "_folds.json")
    return C.read_json(p).get("per_fold", []) if os.path.exists(p) else []


def _ens(name):
    p = os.path.join(C.PROBS, name + "_ensemble.json")
    return C.read_json(p) if os.path.exists(p) else {}


def write_train_report(report):
    """Per-fold training detail next to the fusion report (not checked by verify)."""
    out = dict(
        note=("Training-side companion to out/report.json: the per-fold numbers the two "
              "deep-view trainers wrote, plus the fusion block copied over. verify.py "
              "checks out/report.json against expected_ranges.json; this file is for "
              "debugging a run that lands outside the ranges."),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol=report["protocol"],
        data=dict(n_train=report["n_train"], n_test=report["n_test"],
                  npos_train=report["npos_train"], npos_test=report["npos_test"]),
        trained_views=dict(tlora=dict(per_fold=_folds("tlora"), ensemble=_ens("tlora")),
                           cnn=dict(per_fold=_folds("cnn"), ensemble=_ens("cnn"))),
        views={v: dict(oof_auroc=report["views"][v]["oof_auroc"],
                       oof_aupr=report["views"][v]["oof_aupr"],
                       test=report["views"][v]["test"]) for v in report["views"]},
        fusion=report["fusion"],
        timing=report.get("timing", {}),
    )
    C.write_json(os.path.join(C.OUT, "report_train.json"), out)
    return out


def main():
    C.ensure_dirs()
    log = C.log_stdout(os.path.join(C.LOGS, "fuse.log"))
    try:
        C.stage("fusion: 3 / 4 / 5 views (rank average)")
        oof, test, y_tr, y_te, L_te = load_views()
        print("[data] train1=%d(+%d) test1=%d(+%d)"
              % (len(y_tr), int(y_tr.sum()), len(y_te), int(y_te.sum())), flush=True)

        report = dict(n_train=int(len(y_tr)), n_test=int(len(y_te)),
                      npos_train=int(y_tr.sum()), npos_test=int(y_te.sum()),
                      protocol=dict(folds="StratifiedKFold(5, shuffle=True, random_state=42) "
                                         "on train1.csv in file order (OOF)",
                                    fusion="percentile-rank average",
                                    greedy="Caruana forward selection, 27 rounds, on OOF AUROC",
                                    tau="OOF MCC optimum (fusion.tau_from_scores)",
                                    ci="2,000-resample bootstrap on test1"),
                      views={})

        print("\n%-7s %9s %9s %-18s %9s %8s %6s"
              % ("view", "testAUC", "testAUPR", "95% CI (AUROC)", "lenMatch", "oofAUC", "tau"))
        for v in [x for x in VIEWS if x in oof]:
            tau = F.tau_from_scores(y_tr, oof[v])
            m = F.metrics(y_te, test[v], L_te, tau)
            report["views"][v] = dict(oof_auroc=round(float(F.roc_auc_score(y_tr, oof[v])), 4),
                                      oof_aupr=round(float(F.average_precision_score(y_tr, oof[v])), 4),
                                      test=m)
            print("%-7s %9.4f %9.4f [%6.4f, %6.4f]  %9.4f %8.4f %6.4f"
                  % (v, m["auroc"], m["aupr"], m["ci"][0], m["ci"][1], m["lenmatch"],
                     report["views"][v]["oof_auroc"], tau), flush=True)

        report["fusion"] = {}
        for tier, vs in TIERS.items():
            if not set(vs).issubset(set(oof)):
                print("[fuse] %s skipped: needs %s"
                      % (tier, ", ".join([v for v in vs if v not in oof])), flush=True)
                continue
            ro = {v: F.rank01(oof[v]) for v in vs}
            rt = {v: F.rank01(test[v]) for v in vs}
            entry = {}
            # equal weight
            o_ens = np.mean(np.stack([ro[v] for v in vs]), axis=0)
            t_ens = np.mean(np.stack([rt[v] for v in vs]), axis=0)
            lr, tau, _ = F.platt(y_tr, o_ens)
            p_te = lr.predict_proba(t_ens.reshape(-1, 1))[:, 1]
            entry["rank_equal"] = dict(F.metrics(y_te, p_te, L_te, tau),
                                       oof_auroc=round(float(F.roc_auc_score(y_tr, o_ens)), 4),
                                       weights={v: round(1.0 / len(vs), 4) for v in vs})
            # greedy (Caruana)
            g_ens, w = F.greedy_rank(y_tr, ro)
            gt = sum(w[v] * rt[v] for v in w)
            lr2, tau2, _ = F.platt(y_tr, g_ens)
            p_te2 = lr2.predict_proba(gt.reshape(-1, 1))[:, 1]
            entry["rank_greedy"] = dict(F.metrics(y_te, p_te2, L_te, tau2),
                                        oof_auroc=round(float(F.roc_auc_score(y_tr, g_ens)), 4),
                                        weights={v: round(w[v], 4) for v in sorted(w)})
            report["fusion"][tier] = entry

        print("\n%-13s %9s %9s %-18s %9s %8s %6s"
              % ("fusion", "testAUC", "testAUPR", "95% CI (AUROC)", "lenMatch", "oofAUC", "tau"))
        for tier, entry in report["fusion"].items():
            for col in ("rank_equal", "rank_greedy"):
                m = entry[col]
                print("%-13s %9.4f %9.4f [%6.4f, %6.4f]  %9.4f %8.4f %6.4f"
                      % (tier + "/" + ("eq" if col == "rank_equal" else "gr"),
                         m["auroc"], m["aupr"], m["ci"][0], m["ci"][1],
                         m["lenmatch"], m["oof_auroc"], m["tau"]))
            print("              greedy weights: %s" % entry["rank_greedy"]["weights"])

        tp = os.path.join(C.OUT, "timings.json")
        if os.path.exists(tp):
            report["timing"] = C.read_json(tp)
        C.write_json(os.path.join(C.OUT, "report.json"), report)
        print("\n[saved] out/report.json")
        tr = write_train_report(report)
        print("[saved] out/report_train.json (%d + %d folds)"
              % (len(tr["trained_views"]["tlora"]["per_fold"]),
                 len(tr["trained_views"]["cnn"]["per_fold"])))
    finally:
        sys.stdout = sys.__stdout__
        log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
