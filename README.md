# cxyaippred

Anti-inflammatory peptide (AIP) prediction with protein language model embeddings and
physicochemical descriptors — a small, auditable training/evaluation pipeline built for
reproducible benchmarking rather than leaderboard chasing.

**Status:** code complete and self-tested; no production training run yet in this repository.

---

## What it does

Given a peptide sequence, predict whether it is anti-inflammatory.

- **Representation** — frozen ESM-2 (`esm2_t33_650M_UR50D`, layer 33) + 694 handcrafted
  descriptors (AAC, GAAC, DPC, CKSAAGP, PAAC, CTD, 12 physicochemical properties)
- **Head** — 2 residual 1D-conv blocks over the residue embeddings, masked mean pooling,
  concatenation with the handcrafted vector, shared 512-d layer, per-task output
- **Optional training tricks** (both off by default, both ablatable):
  EMA self-distillation ("EMA-SAT") and MCC-guided class-weight adaptation
- **Evaluation** — fixed 5-fold splits, threshold-free metrics (AUROC/AUPR) reported
  alongside thresholded ones, bootstrap CIs, and a leave-one-fold-out ensemble

Two design rules make the numbers auditable:

1. **Training writes probabilities, evaluation computes metrics.** No metric is ever
   computed twice from different code paths.
2. **One experiment, one directory.** `--exp` is required and colliding names are
   rejected, so parameter sets can never silently mix inside a results folder.

---

## Install

```bash
conda env create -f environment.yml && conda activate cxyaippred
pip install torch                                                          # macOS / CPU
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128   # NVIDIA (RTX 50 series)
python scripts/check_env.py
```

`check_env.py` runs an actual convolution on the target device — `torch.cuda.is_available()`
is not trusted, because it returns `True` on machines where kernels are missing.

Platform specifics, verified versions and expected runtimes: **[docs/INSTALL.md](docs/INSTALL.md)**.

---

## Data

| File | Rows | Description |
|---|---|---|
| `data/aip_all.csv` | 4,686 | `sequence,label`; 1,840 positives (39.3%) |
| `data/splits/split_seed42_n5/folds.csv` | 23,430 | long format: `sequence,label,fold,split` |
| `data/splits/legacy_seed42/folds.csv` | 5,148 | original 4,224/462 split, for comparison with published baselines |

Positive rate 39.3%, median length 15 aa (max 73). No duplicate sequences, no conflicting
labels, and the 5 test folds partition the dataset exactly once.

The split is a fixed artifact, not something regenerated at training time: each run records
its SHA-256, and `scripts/make_5fold.py` reproduces it bit-for-bit from `data/aip_all.csv`.

---

## Usage

```bash
python scripts/preflight.py                                   # data / splits / cache / weights / disk

python scripts/run_5fold.py --exp cv5 --note "5-fold baseline"          # 5 folds, no extras
python scripts/run_5fold.py --exp cv5_ema --ema                          # + EMA self-distillation
python scripts/run_5fold.py --exp cv5_mcc --mcc-adaptive                 # + MCC class weighting
python scripts/run_5fold.py --exp cv5_full --ema --mcc-adaptive          # both

python scripts/list_exps.py                                   # all experiments, one row each
python scripts/list_exps.py --exp cv5_ema                     # per-fold detail
```

Single-fold training and evaluation, for debugging:

```bash
python scripts/train_aip.py --fold 0 --exp dbg --epochs 2
python scripts/eval_aip.py  --exp dbg --n-boot 200
```

Prediction with a 5-fold ensemble (ESM is forwarded once, heads are averaged):

```bash
python scripts/predict.py --runs-dir experiments/cv5/runs --fasta peptides.fasta
```

### Key options

| Option | Meaning |
|---|---|
| `--exp NAME` | experiment id; required, `a-z0-9_-`, collisions refused |
| `--ema` | enable EMA self-distillation (`--ema-alpha`, `--ema-decay`, `--ema-warmup`) |
| `--mcc-adaptive` | enable MCC-guided class weights (`--mcc-interval`, `--mcc-rate`, `--mcc-ratio`, `--mcc-cooldown`) |
| `--label-noise 0.1` | flip 10% of *training* labels (audited in `label_noise.json`) to probe robustness |
| `--device auto` | `auto` → CUDA → MPS → CPU |
| `--seeds 0,1,2` | model seeds (default `0`) |

---

## Outputs

```
experiments/<exp>/
├── runs/fold0_m0/          best.pt · config.json · threshold.json · train_log.csv
│                           split.csv · val_probs.csv · test_probs.csv · all_probs.csv
├── results/                metrics_detail.csv · metrics.json · summary.md
└── manifest.json           data/split/code SHA-256, environment, settings
```

`best.pt` holds only the head (80 MB); the frozen ESM backbone is never duplicated.
The ESM hidden-state cache is built on first use (~900 MB) and deleted when the run
finishes unless `--keep-cache` is passed.

---

## Repository layout

```
src/          config, data_io, features, esm_backbone, model_aip, ema_sat,
              threshold, metrics, manifest, checkpoint, utils
scripts/      preflight, make_5fold, cache_esm, train_aip, eval_aip,
              run_5fold, run_legacy, predict, list_exps, selftest,
              check_env, convert_legacy_head, make_aip_dataset
              legacy_multitask/   read-only snapshot of the previous dual-task code
data/         dataset, fixed splits, statistics
docs/         installation and platform notes
plan/         project rationale, decisions, reviewer-comment mapping
```

`train` and `eval` are deliberately separate processes: the trainer only emits probability
files, and every metric — threshold-free, at 0.5, at the validation-selected threshold,
per fold, or for the leave-one-fold-out ensemble — is produced by one implementation in
`src/metrics.py`.

---

## Reproducibility notes

- Splits are fixed files with recorded SHA-256; changing the split means creating a new
  split id, not re-seeding a splitter.
- `seed_everything` covers Python/NumPy/PyTorch RNGs and enables deterministic cuDNN.
- Thresholds are selected on the validation split only, and the chosen value is written to
  `threshold.json` next to the predictions that used it.
- Expect small numerical differences across CUDA / MPS / CPU; treat CUDA as the reference.

---

## Limitations

- Five folds give an estimate of split variance, not training variance; multi-seed runs are
  supported (`--seeds`) but not yet part of the default protocol.
- The legacy comparison table only contains binary predictions, so AUROC/AUPR cannot be
  derived from it — baselines must be re-run with probability outputs for a threshold-free
  comparison.
- Model weights are not distributed in this repository.
