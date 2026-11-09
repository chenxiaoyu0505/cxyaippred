#!/usr/bin/env python3
"""
Utility functions for ESM2-ResCNN-HandMLP
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache
from sklearn.metrics import matthews_corrcoef, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import compute_handcraft, clean_sequence


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def combo_code_from_y(y_2col):
    """Convert binary labels [aip, pip] to combo class 0-3"""
    return y_2col[:, 0] * 2 + y_2col[:, 1]


class SeqDataset(Dataset):
    def __init__(self, seqs, y2, mask2, sample_weight, indices=None):
        self.seqs = seqs
        self.y2 = y2.astype(np.float32)
        self.mask2 = mask2.astype(np.float32)
        self.sample_weight = sample_weight.astype(np.float32)
        self.indices = indices if indices is not None else np.arange(len(seqs))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (self.seqs[idx], self.y2[idx], self.mask2[idx],
                self.sample_weight[idx], self.indices[idx])


@lru_cache(maxsize=None)
def cached_compute_handcraft(seq_tuple):
    seq = ''.join(seq_tuple)
    return compute_handcraft(seq)


def build_collate_fn(tokenizer):
    def collate(batch):
        seqs, y2, m2, ws, idxs = zip(*batch)
        hand_feats = []
        for seq in seqs:
            clean_seq = clean_sequence(seq)
            feat = cached_compute_handcraft(tuple(clean_seq))
            hand_feats.append(feat)
        hand_feats = np.stack(hand_feats, axis=0).astype(np.float32)
        enc = tokenizer(list(map(clean_sequence, seqs)), return_tensors="pt",
                        padding=True, truncation=True, max_length=1024)
        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor(np.stack(y2), dtype=torch.float32),
                torch.tensor(np.stack(m2), dtype=torch.float32),
                torch.tensor(np.array(ws), dtype=torch.float32),
                torch.tensor(hand_feats, dtype=torch.float32),
                torch.tensor(np.array(idxs), dtype=torch.long))
    return collate


def compute_metrics(logits, y2_true, mask2_true):
    y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y2_true.cpu().numpy()
    mask = mask2_true.cpu().numpy()
    res = {}
    for idx, name in enumerate(["AIP", "PIP"]):
        m = mask[:, idx] > 0.5
        if m.sum() == 0:
            res[f"{name}_pr_auc"] = np.nan
            continue
        y_t, y_p = y_true[m, idx], y_prob[m, idx]
        res[f"{name}_pr_auc"] = average_precision_score(y_t, y_p)
    res["macro_pr_auc"] = np.nanmean([res["AIP_pr_auc"], res["PIP_pr_auc"]])
    return res


def find_best_mcc_threshold(y_true, y_prob, mask, min_thr=0.10):
    m = mask == 1
    if m.sum() == 0:
        return 0.5, np.nan
    y_t, y_p = y_true[m], y_prob[m]
    if len(set(y_t)) < 2:
        return 0.5, np.nan
    thrs = np.linspace(min_thr, 0.99, 199)
    best_t, best_mcc = 0.5, -np.inf
    for t in thrs:
        pred = (y_p >= t).astype(int)
        try:
            mcc = matthews_corrcoef(y_t, pred)
        except Exception:
            continue
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t, best_mcc
