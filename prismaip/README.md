# PrismAIP

A five-view predictor for **anti-inflammatory peptides (AIP)**. Each peptide is
scored by five independent models and the scores are combined by a learned
rank-average; nothing is trained end-to-end.

| view | what it looks at | model |
|---|---|---|
| `hand` | 694 handcrafted descriptors + 10 amphipathicity values | ExtraTrees (1,500 trees) |
| `gap4` | tolerant 4-mers (one position blanked per window) | XGBoost |
| `k5` | contiguous 5-mers | LightGBM |
| `tlora` | ESM-2 650M with a data-aligned low-rank adapter on q/k/v | ESM-2 + adapter + positional head |
| `cnn` | raw sequence, 40-residue window | 37k-parameter 1-D CNN |

The views are combined with percentile ranks and fixed weights fitted on the
training out-of-fold scores (Caruana forward selection, 27 rounds), then mapped
to a probability by a Platt fit; the decision threshold is the out-of-fold
MCC-optimal value. Weights: `hand 0.185 · gap4 0.111 · k5 0.333 · tlora 0.333 · cnn 0.037`.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. A GPU is only needed for the `tlora` and `cnn` views (~1.5 GB of
VRAM at inference); pass `--device cpu` to run without one.

The ESM-2 backbone is downloaded on first use (`facebook/esm2_t33_650M_UR50D`,
~2.6 GB). Set `HF_HOME` to control the cache location.

## Score peptides

```bash
python scripts/predict.py --input example_input.csv
```

Input: a csv with a `sequence` column (extra columns are kept). Output: the same
rows plus one probability per view and the fused result:

| column | meaning |
|---|---|
| `p_hand`, `p_gap4`, `p_k5`, `p_tlora`, `p_cnn` | per-view probability |
| `prismaip_score` | fused rank score (the quantity the weights act on) |
| `prismaip_prob` | calibrated probability |
| `prismaip_call` | `AIP` / `non-AIP` at the recommended threshold |

Useful flags: `--mode fixed` (see below), `--views hand,gap4,k5` (skip the
neural views), `--device cpu`, `--out my_scores.csv`.

### Two scoring modes

* `--mode fixed` (recommended for screening) maps each view's probability through
  the empirical CDF of its **training** scores. A peptide's score does not depend
  on what else is in the file — score 1 sequence or 100,000, the answer is the same.
* `--mode batch` computes ranks inside the file being scored. This is the
  protocol used for the reported test-set numbers and is what `verify.py` uses;
  use it only when the file is a representative sample of the domain.

## Reported performance

Held-out test set (`data/test1.csv`: 817 peptides, 262 positive), fitted on
`data/train1.csv` only:

| view | AUROC | AUPR |
|---|---|---|
| hand | 0.786 | 0.604 |
| gap4 | 0.807 | 0.635 |
| k5 | 0.802 | 0.641 |
| tlora | 0.827 | 0.647 |
| cnn | 0.710 | 0.512 |
| **fused** | **0.860** | **0.711** |

Recompute every number above from the shipped weights:

```bash
python scripts/verify.py            # prints ref vs now, per view and fused
```

`verify.py` exits non-zero if any value drifts by more than 2e-3, so it doubles as
a smoke test after installation.

## What is in the box

```
data/            train1.csv, test1.csv, manifest.json (sha256 + fold definition)
weights/ml/      one model per classical view, fitted on all of train1 (.xz)
weights/tlora/   five fold deltas + the frozen SVD initialisation of A
weights/cnn/     five fold CNN checkpoints
weights/fusion.json, fusion_ecdf.npz, oof.npz
scripts/         predict.py (entry), verify.py, train_all.py, and the view code
```

## Retrain

```bash
python scripts/train_all.py               # ML + CNN weights, fusion, expected scores
python scripts/train_all.py --skip-cnn    # classical views only, no GPU needed
```

The three classical views are refit on the whole `train1` set (this is the
shipped configuration). The fusion weights are always fitted on the
cross-validated out-of-fold scores in `weights/oof.npz`, never on test data.

The cross-validation protocol behind the paper's numbers is
`StratifiedKFold(5, shuffle=True, random_state=42)` over `train1` rows in file
order; `scripts/ml_views.py --help`-style `train_folds()` reproduces the fold
ensembles, which score within ±0.01 of the shipped deployment models.

## Notes and limitations

* Sequences are upper-cased and read as-is; non-standard residues are mapped to an
  "unknown" token for the CNN and are ignored by the k-mer views. Nothing is
  truncated for the classical views; the CNN uses the first 40 residues.
* Training data are short peptides (11–30 aa). Performance degrades quickly
  outside that range and on peptides from other annotation sources: on two
  independent sets assembled from other databases the fused AUROC was 0.41–0.49,
  i.e. no better than chance. Treat PrismAIP as a **domain-specific** model.
* The 15-residue length is over-represented among negatives in the training data.
  The model has learned this shortcut; per-length performance is in the paper and
  is one reason a positive call on a very short peptide should be read with care.
* `cnn` carries a weight of 0.037 and adds no measurable accuracy; it is included
  because it is part of the released system and it makes the "no pre-training"
  baseline reproducible. Drop it with `--views hand,gap4,k5,tlora` if you want the
  leaner four-view model.

## Citation

If you use this code, please cite the PrismAIP paper (reference to be added) and
the ESM-2 backbone (Lin et al., 2023). The adapter follows TLoRA
(Lin et al., 2026): A is initialised from the leading right singular vectors of
`W0 @ C` and frozen, only B is trained.

## License

Released for research use; see `LICENSE`.
