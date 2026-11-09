# cxyaippred

抗炎肽（AIP, anti-inflammatory peptide）活性预测：蛋白质语言模型表征 + 理化描述符，
一条小而可审计的训练/评估管线，目标是可复现的基准对比，而不是刷新排行榜。

**状态**：代码完成并通过自检；本仓库尚未承载正式训练产物。

[English](README.md) | 中文

---

## 做什么

给一条肽序列，判断它是否具有抗炎活性。

- **表征** —— 冻结的 ESM-2（`esm2_t33_650M_UR50D`，第 33 层）+ 694 维手工描述符
  （AAC、GAAC、DPC、CKSAAGP、PAAC、CTD、12 项理化性质）
- **头部** —— 残基级嵌入上两层残差一维卷积，掩码均值池化，与手工特征拼接，
  共享 512 维层，输出单值 logit
- **两个可选训练技巧**（默认都关，都可独立消融）：
  EMA 自蒸馏（EMA-SAT）与 MCC 引导的类权重自适应
- **评估** —— 固定 5 折划分；阈值无关指标（AUROC/AUPR）与阈值相关指标并列报告；
  bootstrap 置信区间

两条设计规则保证数字可审计：

1. **训练只写概率，评估才算指标。** 同一个指标永远不会由两段代码各算一遍。
2. **一个实验一个目录。** `--exp` 必填、重名直接拒绝，参数不同的结果不可能混进同一个文件夹。

---

## 安装

```bash
conda env create -f environment.yml && conda activate cxyaippred
pip install torch                                                                    # macOS / CPU
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128   # NVIDIA（RTX 50 系列）
python scripts/check_env.py
```

`check_env.py` 会在目标设备上真跑一次卷积——不能只信 `torch.cuda.is_available()`，
它在缺少 kernel 的机器上同样返回 `True`。

平台细节、已验证版本与预期耗时见 **[docs/INSTALL.md](docs/INSTALL.md)**。

---

## 数据

| 文件 | 行数 | 说明 |
|---|---|---|
| `data/aip_all.csv` | 4,686 | `sequence,label`；阳性 1,840（39.3%） |
| `data/splits/split_seed42_n5/folds.csv` | 23,430 | 长表：`sequence,label,fold,split` |
| `data/splits/legacy_seed42/folds.csv` | 5,148 | 原 4,224/462 划分，用于与已发表基线比较 |

阳性率 39.3%，长度中位数 15 aa（最长 73）。无重复序列、无冲突标签；
5 折的测试集互不重叠且恰好覆盖全集一次。

划分是**固定产物**而不是训练时现场生成：每次运行都会记录它的 SHA-256，
`scripts/make_5fold.py` 可以从 `data/aip_all.csv` 逐位复现同一份划分。

---

## 用法

```bash
python scripts/preflight.py                                    # 数据 / 划分 / 缓存 / 权重 / 磁盘

python scripts/run_5fold.py --exp cv5 --note "5 折基线"                  # 5 折，不加技巧
python scripts/run_5fold.py --exp cv5_ema --ema                          # + EMA 自蒸馏
python scripts/run_5fold.py --exp cv5_mcc --mcc-adaptive                 # + MCC 类权重
python scripts/run_5fold.py --exp cv5_full --ema --mcc-adaptive          # 两者都开

python scripts/list_exps.py                                   # 所有实验，一行一个
python scripts/list_exps.py --exp cv5_ema                     # 单个实验的逐折明细
```

调试用的单折训练与评估：

```bash
python scripts/train_aip.py --fold 0 --exp dbg --epochs 2
python scripts/eval_aip.py  --exp dbg --n-boot 200
```

用 5 折集成做预测（ESM 只前向一次，多个头取平均）：

```bash
python scripts/predict.py --runs-dir experiments/cv5/runs --fasta peptides.fasta
```

### 关键参数

| 参数 | 含义 |
|---|---|
| `--exp NAME` | 实验名，必填，仅 `a-z0-9_-`，重名拒绝 |
| `--ema` | 开启 EMA 自蒸馏（`--ema-alpha`、`--ema-decay`、`--ema-warmup`） |
| `--mcc-adaptive` | 开启 MCC 引导类权重（`--mcc-interval`、`--mcc-rate`、`--mcc-ratio`、`--mcc-cooldown`） |
| `--label-noise 0.1` | 翻转 10% 的**训练**标签（逐条记录在 `label_noise.json`），用于检验抗噪性 |
| `--device auto` | `auto` → CUDA → MPS → CPU |
| `--seeds 0,1,2` | 模型种子（默认 `0`） |

---

## 产物

```
experiments/<exp>/
├── runs/fold0_m0/          best.pt · config.json · threshold.json · train_log.csv
│                           split.csv · val_probs.csv · test_probs.csv · all_probs.csv
├── results/                metrics_detail.csv · metrics.json · summary.md
└── manifest.json           数据/划分/代码 SHA-256、环境、超参
```

`best.pt` 只存头部（80 MB），冻结的 ESM 主干从不复制。
ESM 隐状态缓存在首次使用时建立（约 900 MB），运行结束自动删除，
除非显式传 `--keep-cache`。

---

## 目录结构

```
src/          config、data_io、features、esm_backbone、model_aip、ema_sat、
              threshold、metrics、manifest、checkpoint、utils
scripts/      preflight、make_5fold、cache_esm、train_aip、eval_aip、
              run_5fold、run_legacy、predict、list_exps、selftest、
              check_env、convert_legacy_head、make_aip_dataset
              legacy_multitask/   上一版双任务代码的只读快照
data/         数据集、固定划分、统计
docs/         安装与平台说明
plan/         项目缘由、决策记录、审稿意见对照
```

`train` 与 `eval` 是两个独立进程：训练器只产出概率文件，所有指标——
阈值无关的、0.5 阈值下的、验证集阈值下的、逐折的——
统一由 `src/metrics.py` 这一份实现产出。

---

## 可复现性说明

- 划分是带 SHA-256 的固定文件；要换划分就新建一个 split id，而不是给切分器换个种子。
- `seed_everything` 覆盖 Python / NumPy / PyTorch 的随机源，并开启 cuDNN 确定性。
- 阈值只在验证集上选，选中的值连同用到它的预测一起写进 `threshold.json`。
- CUDA / MPS / CPU 之间会有小的数值差异，以 CUDA 为准。

---

## 已知局限

- 5 折给出的是**划分方差**，不是训练方差；多 seed 已支持（`--seeds`），但尚未纳入默认协议。
- 旧对比表只有 0/1 预测，无法反推 AUROC/AUPR——要做阈值无关比较，必须重跑基线拿到概率。
- 模型权重不随仓库分发。
