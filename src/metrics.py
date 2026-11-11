#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标：**全项目唯一实现**。

铁律：train 的早停、eval 的报表、论文里的表格，全部调用本文件。
旧稿出现"Fig 3 的 PIP MCC=0.3123、Table 2 的 PIP MCC=0.3497"，
根因就是两处各写了一遍指标函数。

口径
----
* 阈值无关：AUROC（**主指标**，train 的早停也看它）、AUPR（并列报告）
* 阈值相关：MCC、F1、Acc、Precision、Recall、Specificity、混淆矩阵四元
* 不确定性：bootstrap 置信区间（样本级有放回重采样）

用法
----
    m = full_metrics(y, p, threshold=0.5)
    ci = bootstrap_ci(y, p, 'auroc', n_boot=1000, seed=0)
"""
import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:                                    # pragma: no cover
    roc_auc_score = average_precision_score = None


# ---------------------------------------------------------------- 基础

def confusion(y, p, t):
    y = np.asarray(y).astype(int).ravel()
    pred = (np.asarray(p, dtype=np.float64).ravel() >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def mcc_from_cm(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def threshold_free(y, p):
    """与阈值无关的指标；正类全 0 或全 1 时 AUC 无定义，返回 nan"""
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return {'auroc': float('nan'), 'aupr': float('nan')}
    if roc_auc_score is None:
        raise RuntimeError('需要 scikit-learn 才能算 AUROC/AUPR')
    return {'auroc': float(roc_auc_score(y, p)),
            'aupr': float(average_precision_score(y, p))}


def threshold_metrics(y, p, t):
    tp, fp, fn, tn = confusion(y, p, t)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    return {
        'threshold': float(t),
        'mcc': mcc_from_cm(tp, fp, fn, tn),
        'f1': f1, 'acc': _div(tp + tn, tp + tn + fp + fn),
        'precision': prec, 'recall': rec, 'specificity': spec,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n': int(tp + fp + fn + tn),
        'n_pos': int(tp + fn),
        'pred_pos': int(tp + fp),
    }


def full_metrics(y, p, threshold=0.5):
    """阈值无关 + 阈值相关，一次给全"""
    m = threshold_free(y, p)
    m.update(threshold_metrics(y, p, threshold))
    return m


# ---------------------------------------------------------------- 不确定性

_BOOT_METRICS = {
    'auroc': lambda y, p: threshold_free(y, p)['auroc'],
    'aupr': lambda y, p: threshold_free(y, p)['aupr'],
    'mcc': None,      # 需要阈值，见下
    'f1': None,
}


def bootstrap_ci(y, p, metric='auroc', threshold=None, n_boot=1000, alpha=0.05, seed=0):
    """
    样本级有放回重采样，返回 (lo, hi, mean)。
    metric='mcc'/'f1' 时必须给 threshold。
    """
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    n = len(y)
    rng = np.random.RandomState(seed)

    def one(yy, pp):
        if metric in ('auroc', 'aupr'):
            return threshold_free(yy, pp)[metric]
        if threshold is None:
            raise ValueError("metric=%s 需要 threshold" % metric)
        return threshold_metrics(yy, pp, threshold)[metric]

    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        try:
            v = one(y[i], p[i])
        except Exception:
            continue
        if v == v:                    # 过滤 nan
            vals.append(v)
    if not vals:
        return {'lo': float('nan'), 'hi': float('nan'), 'mean': float('nan'), 'n_boot': 0}
    vals = np.asarray(vals)
    return {
        'lo': float(np.quantile(vals, alpha / 2)),
        'hi': float(np.quantile(vals, 1 - alpha / 2)),
        'mean': float(vals.mean()),
        'n_boot': int(len(vals)),
    }


# ---------------------------------------------------------------- 汇总

def summarize(rows, keys=('auroc', 'aupr', 'mcc', 'f1', 'acc')):
    """
    rows: list[dict]，每折一行 -> 每个指标的 mean / sd / n。
    只对**折均值**求 sd，不把 3 个 seed 的 15 个数混在一起算（那会把 CI 算窄）。
    """
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] == r[k]], dtype=float)
        if len(v) == 0:
            continue
        out[k] = {'mean': float(v.mean()),
                  'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  'n': int(len(v))}
    return out
