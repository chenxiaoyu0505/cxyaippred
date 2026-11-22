# -*- coding: utf-8 -*-
"""The three classical-ML views: hand704 -> ExtraTrees, gap4 -> XGBoost, k5 -> LightGBM.

Feature blocks
--------------
hand704   694 handcrafted descriptors + 10 amphipathicity values, no scaling.
gap4      tolerant 4-mer: a 5-residue window keeps position 0 and blanks one of
          the remaining four positions to ``X`` -> four patterns per window.
k5        contiguous 5-mer.

gap4/k5 go through TF-IDF (word, sublinear, l2) -> TruncatedSVD(300) ->
StandardScaler.

Two fitting modes
-----------------
deploy  one model per view, fitted on all 3,278 train1 peptides.  This is what the
        released weights contain: for a user scoring new peptides a single model
        trained on everything is both smaller and better than a fold ensemble.
cv      five fold models per view, fitted only on the training part of each fold.
        This is the protocol behind every cross-validated number in the paper.

Both modes share the same hyperparameters and the same feature code, so the two
are directly comparable.
"""
import lzma
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import features as fe       # noqa: E402

SVD_DIM = 300
ET_KW = dict(n_estimators=1500, max_features=0.1, min_samples_leaf=1,
             class_weight="balanced", n_jobs=-1)
GAP_G = 4          # gap4
KMER_K = 5         # k5


# --------------------------------------------------------------------- features
def k_tokens(seq, k=KMER_K):
    """Contiguous k-mer tokens."""
    return [seq[i:i + k] for i in range(max(len(seq) - k + 1, 0))]


def gap_tokens(seq, g=GAP_G):
    """Tolerant g-mer: window of g+1, keep position 0, blank one of the next g."""
    out = []
    for i in range(max(len(seq) - g, 0)):
        w = seq[i:i + g + 1]
        for skip in range(1, g + 1):
            out.append(w[0] + "".join("X" if j == skip else w[j] for j in range(1, g + 1)))
    return out


TOK = {"gap4": lambda s: gap_tokens(s, GAP_G), "k5": lambda s: k_tokens(s, KMER_K)}


def hand704(seqs, cache_dir=None):
    """704-dim handcrafted matrix (694 handcraft + 10 amphipath)."""
    import hashlib
    key = hashlib.sha256("\n".join(seqs).encode("utf-8")).hexdigest()[:16]
    cache_dir = cache_dir or os.path.join(C.OUT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    npy = os.path.join(cache_dir, "hand704_%s.npy" % key)
    if os.path.exists(npy):
        return np.load(npy)
    base = np.stack([fe.compute_handcraft(s) for s in seqs]).astype(np.float32)
    extra = np.stack([fe.compute_amphipath(s) for s in seqs]).astype(np.float32)
    X = np.concatenate([base, extra], axis=1)
    assert X.shape == (len(seqs), 704), X.shape
    np.save(npy, X)
    return X


# ----------------------------------------------------------------------- models
def make_model(view, y_fit, seed=0):
    npos = int(y_fit.sum())
    nneg = len(y_fit) - npos
    if view == "hand":
        return ExtraTreesClassifier(random_state=seed, **ET_KW)
    if view == "gap4":
        return xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", eval_metric="logloss", n_jobs=-1,
            random_state=seed, scale_pos_weight=float(nneg) / float(npos))
    if view == "k5":
        return lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=seed, verbose=-1)
    raise ValueError(view)


def _fit_repr(view, fit_docs):
    """Fit TF-IDF -> SVD(300) -> StandardScaler on a list of token documents."""
    tf = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                         norm="l2", lowercase=False).fit(fit_docs)
    Z = tf.transform(fit_docs)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=0).fit(Z)
    svd.components_ = svd.components_.astype(np.float32)      # half the bytes, same projections to 1e-7
    sc = StandardScaler().fit(svd.transform(Z).astype(np.float32))
    return dict(tfidf=tf, svd=svd, scaler=sc)


def _apply_repr(bundle, docs):
    Z = bundle["tfidf"].transform(docs)
    E = bundle["svd"].transform(Z).astype(np.float32)
    return bundle["scaler"].transform(E).astype(np.float32)


def _docs(view, seqs):
    return [" ".join(TOK[view](s)) for s in seqs]


# -------------------------------------------------------------------- fit/load
def fit_one(view, seqs, y, seed=0):
    """Fit one model of `view` on the given rows; returns a self-contained bundle."""
    if view == "hand":
        X = hand704(seqs)
        est = make_model(view, y, seed=seed).fit(X, y)
        return dict(view=view, mode="hand", model=est)
    docs = _docs(view, seqs)
    rep = _fit_repr(view, docs)
    est = make_model(view, y, seed=seed).fit(_apply_repr(rep, docs), y)
    return dict(view=view, mode="tokens", model=est, **rep)


def save(bundle, path):
    """Write a bundle as lzma-compressed pickle (``<path>.xz``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = pickle.dumps(bundle, protocol=4)
    with open(path + ".xz", "wb") as f:
        f.write(lzma.compress(raw, preset=6))
    return path + ".xz", len(raw), os.path.getsize(path + ".xz")


def load(path):
    raw = lzma.decompress(open(path, "rb").read())
    return pickle.loads(raw)


def train_deploy(out_dir=None, seed=0, log=print):
    """Fit one model per view on the whole train1 set; write ``weights/ml/<view>.joblib.xz``."""
    out_dir = out_dir or os.path.join(C.WEIGHTS, "ml")
    os.makedirs(out_dir, exist_ok=True)
    tr = C.load_split("train1")
    seqs, y = list(tr.sequence), tr.label.values
    for view in ("hand", "gap4", "k5"):
        b = fit_one(view, seqs, y, seed=seed)
        p, raw, comp = save(b, os.path.join(out_dir, view + ".joblib"))
        log("[ml] %-5s fitted on %d rows -> %s (%.1f MB raw, %.1f MB on disk)"
            % (view, len(seqs), os.path.basename(p), raw / 1e6, comp / 1e6))
    return out_dir


def train_folds(out_dir=None, seed=0, log=print):
    """Cross-validation mode: one model per (view, fold), for reproducing the paper."""
    out_dir = out_dir or os.path.join(C.WEIGHTS, "ml_cv")
    os.makedirs(out_dir, exist_ok=True)
    tr = C.load_split("train1")
    y = tr.label.values
    fold = C.skf_folds(y)
    seqs = list(tr.sequence)
    for k in range(C.NFOLD):
        fit_idx = np.where(fold != k)[0]
        for view in ("hand", "gap4", "k5"):
            b = fit_one(view, [seqs[i] for i in fit_idx], y[fit_idx], seed=seed)
            b["fold"] = k
            p, _, _ = save(b, os.path.join(out_dir, "%s_fold%d.joblib" % (view, k)))
            log("[ml-cv] %-5s fold%d -> %s" % (view, k, os.path.basename(p)))
    return out_dir


def load_models(view, ml_dir=None):
    """Load the deployment model (`<view>.joblib.xz`) or the five CV fold models."""
    ml_dir = ml_dir or os.path.join(C.WEIGHTS, "ml")
    deploy = os.path.join(ml_dir, view + ".joblib.xz")
    if os.path.exists(deploy):
        return [load(deploy)]
    folds = [os.path.join(ml_dir, "%s_fold%d.joblib.xz" % (view, k)) for k in range(C.NFOLD)]
    if all(os.path.exists(p) for p in folds):
        return [load(p) for p in folds]
    raise FileNotFoundError("no weights for view %r in %s" % (view, ml_dir))


def predict_view(view, seqs, bundles):
    """Probability from each bundle, averaged (one bundle in deploy mode)."""
    if view == "hand":
        X = hand704(seqs)
        return np.mean([b["model"].predict_proba(X)[:, 1] for b in bundles], axis=0)
    docs = _docs(view, seqs)
    return np.mean([b["model"].predict_proba(_apply_repr(b, docs))[:, 1] for b in bundles], axis=0)
