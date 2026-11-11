"""手工特征筛选：方差过滤 → 相关性去冗余 → 互信息 → 随机森林重要性。

参考 AIPFusion-Net 的做法，但有一处关键差别：**只在当前折的训练行上拟合**。
val/test 的标签或分布如果参与筛选，就是泄漏；所以每折各选一次，索引跟着
该折的权重一起落盘（``hand_selector.npz``），推理时必须原样复现。

为什么需要它：704 维手工特征里有大量高度相关、接近常数的列，和 ESM 拼接后
会把 1984 维里的信号稀释掉（实测线性探针：纯 ESM 0.674 / 拼接 0.655）。

典型用法（train_aip.py）::

    idx = select_hand_features(hand[folds['train']], y[folds['train']], seed=42)
    hand = hand[:, idx]
    save_hand_selector(os.path.join(run_dir, 'hand_selector.npz'), idx)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# 默认参数照抄 AIPFusion-Net；改之前先想清楚"为什么我们的数据需要不同的值"
LOW_VAR = 1e-8
CORR_THR = 0.95
TOP_MI = 150
TOP_K = 80


def select_hand_features(X, y, seed=42, low_var=LOW_VAR, corr_thr=CORR_THR,
                         top_mi=TOP_MI, top_k=TOP_K):
    """返回保留下来的列索引（升序 int64）。

    X: [N, D] 原始手工特征，**只能传训练行**；y: [N] 标签。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError('X 应为 [N, D]，收到 %s' % (tuple(X.shape),))
    if len(X) != len(y):
        raise ValueError('X 与 y 行数不一致：%d vs %d' % (len(X), len(y)))
    if len(np.unique(y)) < 2:
        raise ValueError('筛选需要两类标签，当前只有 %s' % (np.unique(y),))

    # 1) 方差过滤：常数/近常数列没有信息
    std = X.std(axis=0)
    keep = np.flatnonzero(std > low_var)
    if len(keep) == 0:
        raise ValueError('所有列的方差都 ≤ %g，没有可用的手工特征' % low_var)

    # 2) 相关性去冗余：|r| > corr_thr 的列对只保留先出现的那一列
    X1 = X[:, keep]
    corr = np.corrcoef(X1, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    drop = np.zeros(len(keep), dtype=bool)
    for j in range(1, X1.shape[1]):
        if drop[j]:
            continue
        if np.any(np.abs(corr[j, :j][~drop[:j]]) > corr_thr):
            drop[j] = True
    keep2 = keep[~drop]

    # 3) 互信息排序（保留 top_mi）
    mi = mutual_info_classif(X[:, keep2], y, random_state=seed)
    order = np.argsort(mi)[::-1][:min(top_mi, len(mi))]
    keep3 = keep2[order]

    # 4) 随机森林重要性排序（保留 top_k）
    rf = RandomForestClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1,
        class_weight='balanced_subsample')
    rf.fit(X[:, keep3], y)
    order2 = np.argsort(rf.feature_importances_)[::-1][:min(top_k, len(keep3))]
    return np.sort(keep3[order2]).astype(np.int64)


def save_hand_selector(path, idx):
    """索引落盘；和 hand_scaler.npz 放同一目录，推理时成对加载。"""
    np.savez(path, idx=np.asarray(idx, dtype=np.int64))


def load_hand_selector(path):
    with np.load(path) as d:
        return d['idx'].astype(np.int64)


def apply_hand_selector(X, idx):
    """按落盘索引取列；维度对不上时直接报错，不静默截断。"""
    X = np.asarray(X, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64)
    if X.shape[1] <= int(idx.max()):
        raise ValueError('手工特征只有 %d 列，筛选索引最大 %d'
                         % (X.shape[1], int(idx.max())))
    return X[:, idx]
