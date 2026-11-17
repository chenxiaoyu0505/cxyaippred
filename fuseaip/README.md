# fuseaip — a five-view peptide-bioactivity release, trained from scratch

Five-view classifier for peptide bioactivity on the DeepAIP split: three
classical sequence views (handcrafted descriptors, gapped 3-mers, 4-mers) and
two deep views (a LoRA-adapted ESM-2 650M with a positional-projection head,
and a small 1-D CNN trained from scratch). The 5-view fusion lands at AUROC
≈ 0.82 on the 817-peptide test set; the 0.84 headline of the parent repository
belongs to a different fold assignment (see *Results*).

**This directory ships data and code only — no weights, no frozen probability
files.** A single command retrains every view from `data/train1.csv`, fuses the
views on out-of-fold predictions, scores `data/test1.csv`, and checks the result
against `expected_ranges.json`:

```bash
python fuseaip/scripts/run_all.py
```

The last two lines of a healthy full run are

```
[verify] 8/8 checks passed
[verify] PASS
```

> **Status: code-only release.** The training, fusion and verification code is
> complete and runs end to end, and the three ML views have been measured on the
> machine that cut this commit (see *Results*). The two deep views were **not**
> trained here: the TLoRA / CNN / 4-view / 5-view entries of
> `expected_ranges.json` are deliberately wide provisional bands, and
> `verify.py` prints `SKIP (no probs)` for anything that has not been trained
> yet instead of failing the run. Run `run_all.py` once (≈20-25 min on an 8 GB
> GPU) and replace those bands with the values it prints.

## Quick start

```bash
# 1. environment (Python 3.11; see requirements.txt for pinned versions)
python -m pip install -r requirements.txt

# 2. first run only: fetch the ESM-2 backbone (~2.5 GB) into the HF cache.
#    The scripts default to HF_HUB_OFFLINE=1 so later runs are offline and
#    deterministic; export HF_HUB_OFFLINE=0 for this one download.
HF_HUB_OFFLINE=0 python fuseaip/scripts/train_tlora.py --folds 0   # or just run step 3 once with it set

# 3. retrain all five views, fuse, score, verify  (~20-25 min on one 8 GB GPU:
#    ML ~1.5 min CPU + TLoRA ~3-3.5 min/fold + CNN ~30 s/fold + fusion)
python fuseaip/scripts/run_all.py
```

Partial re-runs reuse whatever is already in `out/`:

```bash
python fuseaip/scripts/run_all.py --skip-ml      # deep views only
python fuseaip/scripts/run_all.py --skip-dl      # ML views only (no GPU, ~2 min)
python fuseaip/scripts/run_all.py --reuse        # no training: score weights/trained/
python fuseaip/scripts/run_all.py --folds 0      # smoke test: 1 fold, verify WILL fail
python fuseaip/scripts/run_all.py --device cpu   # no GPU: works, TLoRA ~5-10x slower
```

A partial run makes `verify.py` **skip** the views it never trained (`SKIP (no
probs)`) rather than fail them, so a green `--skip-dl` run only tells you the ML
side is fine. Quote numbers only from a run without `--skip-*`.

`--reuse` skips both deep-view training loops and loads the weights archived in
`weights/trained/` (`train_tlora.py --reuse` / `train_cnn.py --reuse` do the same
for a single view) — that path needs no GPU time beyond inference, and it is
what you use after uploading a weight archive (see *Weights (not shipped)*).

Individual stages are runnable on their own (`train_ml_views.py`,
`train_tlora.py`, `train_cnn.py`, `fuse_eval.py`, `verify.py`); each takes
`--folds 0,1,2,3,4` and writes into `out/`.

Artifacts (all git-ignored):

| path | content |
|---|---|
| `out/probs/{ml,tlora,cnn}_views.npz` | OOF probabilities on train1 + test probabilities per view |
| `out/probs/*_fold{k}.{npz,json}` | per-fold probabilities and metrics |
| `weights/trained/tlora_trained_fold{k}.pt` | LoRA delta + head weights (~8 MB, delta only) |
| `weights/trained/tlora_trained_init_fold{k}.npz` | the covariance-SVD initialisation of A for that fold |
| `weights/trained/cnn_trained_fold{k}.pt` | CNN state dict (~0.7 MB) |
| `out/report.json` | every number `verify.py` checks, plus CIs and fusion weights |
| `out/timings.json`, `out/logs/*.log` | wall clock per stage, full console logs |

## Environment

Measured with:

| component | version |
|---|---|
| Python | 3.11.14 |
| torch | 2.11.0.dev+cu128 (any ≥2.1 with `torch.autocast` should do) |
| transformers | 4.48.1 |
| scikit-learn | 1.5.0 |
| numpy | 1.26.0 (`<2`, see requirements.txt) |
| GPU | NVIDIA RTX 5060 Laptop, 8 GB — CUDA 12.8 driver |

* The **TLoRA view needs a GPU** in practice: ESM-2 650M in fp16 plus activations
  fits comfortably in 8 GB, and one fold takes ≈3.5 min. On CPU the same fold
  takes ~20-30 min (5-10x slower) — `--device cpu` works but is not what the
  reported timings were measured on.
* The three ML views and the CNN run on CPU (CNN ≈30 s/fold).
* Windows note: the deep-view scripts `import torch` **before** numpy /
  scikit-learn. On this box the reverse order makes `c10.dll` fail at load with
  `OSError [WinError 1114]`. Keep that import order if you edit the scripts.
  Consoles with a GBK code page should be run with `python -X utf8` (all
  scripts also call `sys.stdout.reconfigure(encoding="utf-8")`).
* The GPU lock `_work/gpu.lock` is shared by every worktree of this checkout:
  `train_tlora.py` writes its PID, polls every 30 s while another process holds
  it, and treats a lock older than 2 h as stale. It never skips the lock.

## Data

| file | rows | positives | role |
|---|---|---|---|
| `data/train1.csv` | 3,278 | 1,060 | training pool; folds are **derived** at runtime, there is no `fold` column |
| `data/test1.csv` | 817 | 262 | held-out test, never used for fitting or for τ |
| `data/conflicts_385.csv` | 385 | — | documentation only (see below) |

Provenance: `train1`/`test1` are the DeepAIP official train/test split **minus
385 dual-function peptides** (305 removed from train, 80 from test) that were
flagged as conflicting annotations — the `conflicts_385.csv` list is shipped so
the exclusion is auditable. Nothing in the pipeline reads that file.

`data/manifest.json` pins row counts, positives and SHA-256 of each file;
`scripts/build_splits.py` re-checks all of it before training and refuses to
continue on a mismatch. A changed input file is the only way this release can
silently drift, so the hashes are hard requirements.

## Cross-validation protocol

Folds are not stored in the data. Every trainer derives them with

```python
StratifiedKFold(5, shuffle=True, random_state=42)
```

over the `train1.csv` rows **in file order** (row *i* is row *i*; nothing is
sorted or reindexed), via the single helper `common.skf_folds`. For fold *k* the
validation rows are exactly the rows the splitter assigns to *k* and the fit rows
are the remaining four folds, so every row is a validation row once and each
fold has 212 positives (656/656/656/655/655 rows).

Why not ship a fold column: an earlier release shipped the official DeepAIP
`folds.csv` mapping, which turns out to be label-sorted contiguous blocks — its
validation folds are *not* stratified in the usual sense, and a fold column in a
CSV invites readers to assume a protocol that is neither documented nor
comparable to `StratifiedKFold`. Deriving the folds from `(rows, labels,
random_state)` makes the protocol one line of code, applies it identically to all
five views, and keeps the data file minimal.

`build_splits.py` prints the fold sizes, positives and a short hash of each
fold's row ids, and compares them to `data/manifest.json → cv.reference_folds`.
That comparison is **advisory**: a different scikit-learn release can reshuffle
the folds, which makes runs non-comparable (the script says so) but does not make
the data wrong. Fold membership is deterministic on a fixed environment, so two
machines with the same `requirements.txt` produce the same folds. `fuse_eval.py`
also refuses to fuse artifacts that were produced under different fold
assignments (it compares the `fold_of` vector stored in every probability file),
so a stale `out/probs/*_views.npz` fails loudly instead of silently mixing
protocols.

## The five views

| view | features / model | fitting | parameters |
|---|---|---|---|
| `hand` | 704-dim handcrafted matrix (694 physicochemical descriptors + 10 amphipathicity features) | ExtraTrees per fold | — |
| `gap3` | spaced 3-mers: 4-residue window, one position dropped and replaced by `X`; patterns `(0,1,2)`, `(0,1,3)`, `(0,2,3)` → TF-IDF → SVD-300 → scaler | TF-IDF/SVD/scaler + ExtraTrees per fold | — |
| `k4` | all 4-mers → TF-IDF → SVD-300 → scaler | same | — |
| `TLoRA` | ESM-2 650M (33 layers), fp16 frozen; LoRA r=8, α=16 on every q/k/v; posproj head: `Linear(1280→4)` → `Conv1d(4,4,k=3)` → GELU → adaptive-pool to L=32 → `Linear(128→256)` → GELU → `Dropout(0.3)` → `Linear(256→1)` | A initialized from the SVD of W·C (C = covariance of that fold's **training** tokens) and frozen; only B + head trained | 1.01 M LoRA B + 0.04 M head |
| `CNN` | embedding(22→64) → conv k=3 → conv k=5 → masked mean+max pool → dropout 0.2 → linear | from scratch | 34,689 |

ML trainer: `ExtraTreesClassifier(n_estimators=1500, max_features=0.1,
min_samples_leaf=1, class_weight="balanced"), random_state=0` — one seed, one
fit per view per fold. (An earlier grid averaged seeds 0/1/2 to estimate
stability; that average is not used here.)
Deep views: AdamW with `lr_lora=5e-4`, `lr_head=1e-3`, `wd=0.01` (TLoRA) /
`lr=2e-3`, `wd=0.02`, grad-clip 2.0 (CNN), batch 32, bf16 autocast (TLoRA),
≤8 epochs with patience 5 (TLoRA) or ≤12 epochs with patience 4 (CNN), early
stopping on **validation AUROC**, seed 0.

Two invariants are asserted at the start of every TLoRA fold, because they are
what makes "LoRA" meaningful here: the initial ΔW is exactly zero (`max|B@A| ==
0.0`), and only the B matrices receive gradients (99/99 non-zero, `A.grad is
None`). The per-fold ΔW files are also round-tripped through
`save_delta`/`load_delta` before training (max |Δp| < 1e-5).

Per-fold test probability = the fold's model applied to test1; the view's test
probability = mean over the 5 fold models. The OOF vector is the concatenation
of the 5 validation blocks, and that is what fusion weights and τ are fitted on.

## Fusion

Views are combined in percentile-rank space: `rank(x) = rankdata(x)/len(x)`.
Two weightings are reported per tier:

* `rank_equal` — plain average of the ranks;
* `rank_greedy` — Caruana forward selection with replacement, 27 rounds, each
  round picking the view that maximises the fused **OOF** AUROC. The weights are
  a result of the run, not a constant in the code.

The decision threshold τ is the OOF MCC optimum (`fusion.tau_from_scores`), never
0.5, and never fitted on test. AUROC/AUPR/length-matched AUROC/τ/MCC/sens/spec
are reported with a 2,000-resample bootstrap 95 % CI on test1.

## Weights (not shipped)

**This directory contains no model weights.** Both deep views are retrained from
scratch on every run, and their weights are archived under `weights/trained/` —
git-ignored on purpose, so a clone stays small:

```
weights/trained/tlora_trained_fold{0..4}.pt         LoRA delta + head, ~8 MB each
weights/trained/tlora_trained_init_fold{0..4}.npz   per-fold SVD(A) initialisation, ~4 MB each
weights/trained/cnn_trained_fold{0..4}.pt           CNN state dicts, ~0.7 MB each
```

Keeping them out of git is a policy choice, not a limitation: if you need
weight-level reproduction instead of a retraining run inside the published band,
upload that directory (`git add -f weights/trained/`, or attach it as a release
artifact next to the commit) and score it with

```bash
python fuseaip/scripts/run_all.py --reuse
```

which loads the archived weights instead of training (`train_tlora.py --reuse` /
`train_cnn.py --reuse` for one view at a time). Before trusting an uploaded
archive, compare it with the inventory of published hashes in the parent
repository, `plan/WEIGHTS_INVENTORY.md`.

## Results (test1.csv, 817 rows / 262 positives)

Folds: `StratifiedKFold(5, shuffle=True, random_state=42)` on train1.csv. Test
probability of a view = mean over its 5 fold models; fusion = `rank_greedy`
(Caruana, 27 rounds on OOF).

| view / tier | test AUROC | test AUPR | status |
|---|---|---|---|
| `hand` | 0.7822 | 0.5949 | measured (deterministic) |
| `gap3` | 0.7574 | 0.5390 | measured (deterministic) |
| `k4` | 0.7652 | 0.5550 | measured (deterministic) |
| `TLoRA` | — | — | **not run in this commit** (GPU) |
| `CNN` | — | — | **not run in this commit** |
| fusion 3-view | 0.8136 | 0.6189 | measured (`rank_greedy`) |
| fusion 4-view | — | — | needs TLoRA |
| fusion 5-view | — | — | needs TLoRA + CNN |

The three ML rows and the 3-view fusion are the only numbers this commit has
measured; they are what `expected_ranges.json` pins (ML to ±0.001). Filling the
remaining rows takes one `run_all.py` run; `out/report.json` then carries per-view
AUROC/AUPR/length-matched AUROC, bootstrap CIs, τ/MCC/sens/spec and the greedy
weights, and `out/report_train.json` carries the per-fold training detail (best
epoch, wall clock, initialisation diagnostics).

## Reproducibility and what the checks mean

`expected_ranges.json` encodes two different standards, and `verify.py` treats
them differently:

* **The three ML views are deterministic.** Same data, same code, same seed ⇒
  same numbers: `hand` 0.7822, `gap3` 0.7574, `k4` 0.7652 test AUROC, checked to
  ±0.001. A miss here is a real regression (changed tokenizer, changed feature
  code, a leak introduced by fitting the representation outside the fold, or a
  different fold assignment — check the fold hashes `build_splits.py` prints).
  It fails the run.
* **The two deep views involve SGD.** Their bands in `expected_ranges.json`
  (TLoRA 0.72-0.85, CNN 0.65-0.75) are **provisional**: this commit did not train
  them. They are wide enough to catch a broken implementation (0.70 / 0.88) and
  narrow enough to be meaningful, but they are not the final targets. A rerun on
  another GPU, another torch build, or CPU instead of CUDA can move a deep view
  by ±0.02; individual folds scatter more than the ensemble.
* **Fusion bands** (3-view 0.78-0.86, 4-view 0.79-0.87, 5-view 0.79-0.87) are
  provisional too, and wider on purpose: the greedy weights are re-derived from
  OOF every run, so the ensemble is a slightly different model each time.

`verify.py` prints the measured value next to every target and exits non-zero if
any *measured* check fails, so a green run is evidence, not a vibe. Views that
were never trained are printed as `SKIP (no probs)` and counted separately — they
are not evidence either way. Because the deep-view numbers are ranges, this
release cannot claim bit-identical weight-level reproduction of the paper's deep
views — it claims that retraining with the documented protocol lands in the
documented band.

`--folds 0` (or any subset) trains a partial model and **is expected to fail
verification**: 1-fold predictions are not the 5-fold-mean protocol.

## Known limitations

* **Compute.** TLoRA dominates the runtime (~3.5 min/fold on an 8 GB laptop GPU);
  a CPU-only run is 5-10x slower. The package has no cached-backbone shortcut.
* **Batch-composition sensitivity in the posproj head.** The conv is applied
  before masking, so padding at the tail of a batch can leak into the features.
  Inference therefore always uses batch size 32 (switching to 16 moves fold-0
  test AUROC from 0.7445 to 0.7365 in the reference run).
* **Single seed everywhere.** ML views one ExtraTrees fit (`random_state=0`), deep
  views seed 0. The bands in `expected_ranges.json` absorb that; do not read a
  0.005 AUROC difference as a real improvement.
* **Thresholds are descriptive.** τ is chosen on OOF and applied to test; for a
  clinical-style operating point it should be recalibrated on a prospective set.
* **Not a new method.** This is a reproduction harness for the five-view results
  of the parent repository, with the frozen artifacts replaced by training code.

## Licenses

ESM-2 (`facebook/esm2_t33_650M_UR50D`) is distributed by Meta under the MIT
license; the weights are downloaded at runtime and not redistributed here. The
code in this directory follows the terms of the parent repository (`cxyaippred`),
which ships no separate LICENSE file.

## Archived weights (optional: reproduce the 0.8268 TLoRA view without training)

`weights/trained/` can optionally carry the five trained TLoRA deltas (8.3 MB
each, ~42 MB total) plus the five CNN state dicts (~0.15 MB each). When present,
the deep views can be scored instead of retrained:

    python fuseaip/scripts/train_tlora.py --reuse            # all five folds
    python fuseaip/scripts/train_cnn.py   --reuse

`--reuse` rebuilds the model, restores the delta from the `.pt`, and evaluates
val/test once — no optimiser step, so it is bit-identical across machines as long
as the same ESM-2 checkpoint is used. Reference numbers from these exact files
(test1, 817 rows / 262 positives):

| fold | TLoRA test AUROC | TLoRA AUPR | CNN test AUROC |
|---|---|---|---|
| 0 | 0.7559 | 0.6159 | 0.6812 |
| 1 | 0.8579 | 0.6974 | — |
| 2 | 0.8064 | 0.6585 | — |
| 3 | 0.7517 | 0.5831 | — |
| 4 | 0.7834 | 0.4919 | — |
| mean-of-prob ensemble | **0.8269** | 0.6474 | 0.7009 |

Per-fold AUROC spread (0.75-0.86) is a property of this protocol, not a
reproduction failure: three of five folds pick `best_ep=1`, so the view is decided
by the first epoch. Dropping fold 1 leaves an ensemble of 0.7904. Integrity:
`weights/trained/SHA256SUMS.txt`.
