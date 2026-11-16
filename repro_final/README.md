# repro_final — 5 视角 AIP 预测的可复现发布包

本目录把"DeepAIP 官方划分去掉 AIP+/PIP+ 冲突阳性"这一版数据上的 5 视角模型
（hand704 / gap3 / k4 / TLoRA / CNN）与它们的秩平均融合，打包成**别人 clone 就能跑**的形式：
数据、权重、脚本、金标准数字全在仓库里，运行时不依赖 `_work/`、不重新训练。

论文表（test1，817 行 / 262 阳性）：**5 视角融合 AUROC 0.8398 / AUPR 0.6676**。

## 快速开始

```bash
# 1) 装依赖（建议新建 conda 环境；已测 torch 2.11+cu128 / transformers 4.48.1 / sklearn 1.5.0 / numpy 1.26.0）
pip install -r repro_final/requirements.txt

# 2) 首次运行需要联网下载 ESM-2 底座（约 2.6 GB，只会下一次，之后走 HF 缓存）
#    facebook/esm2_t33_650M_UR50D

# 3) 一条命令跑完并自动核对
python repro_final/scripts/run_all.py
```

全 PASS 时最后一行是 `[verify] 54/54 checks passed` / `[verify] PASS`。
耗时实测：**总 4.9 min**（ML 5 折 3.5 min，CPU；DL 推理 0.9 min，单卡 GPU）。机器空闲时可到
这个数，CPU 被占满时 ML 段会慢到 ~13 min（ExtraTrees `n_jobs=-1`）。
`--skip-ml` / `--skip-dl` 可复用 `out/` 里已有的中间结果。

Windows 控制台是 GBK，跑之前设 `$env:PYTHONUTF8=1`；脚本自身都带 `# -*- coding: utf-8 -*-`，
且 `import torch` 一律早于 numpy / sklearn（否则 DLL 初始化会失败）。

## 环境

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | 3.9+ 应该都行，未逐一测 |
| torch | >= 2.0（实测 2.11+cu128） | 只有 TLoRA 需要；CNN 与 ML 视角不需要 GPU |
| transformers | 4.48.1 | 固定版本，`EsmModel` 的 API 变动会影响加载 |
| scikit-learn | 1.5.0 | ExtraTrees / TfidfVectorizer / StratifiedKFold |
| numpy | 1.26.0 | — |
| pandas | >= 1.5 | 只用于读 CSV |
| scipy | >= 1.9 | `rankdata` |
| GPU | 显存 >= 8 GB | TLoRA 用 fp16 + bf16 autocast 推理 |

## 目录结构

```
repro_final/
├── README.md                  # 本文件
├── requirements.txt
├── expected.json              # 金标准：两组数字 + 容差
├── data/
│   ├── train0.csv             # 官方 train 全量 3,583 行（未剔冲突）
│   ├── test0.csv              # 官方 test 全量   897 行
│   ├── train1.csv             # 本包训练集 3,278 行 / 1,060 阳性，含 fold 0-4
│   ├── test1.csv              # 本包测试集   817 行 /   262 阳性
│   ├── conflicts_385.csv      # 被剔除的 385 条冲突阳性序列
│   ├── pool_folds.csv         # 3,583 条官方 train 的折号（train1 的 fold 从这里来）
│   ├── manifest.json          # 每个文件的行数 / 阳性数 / sha256
│   └── published/             # 冻结的已发布 DL 概率（见下）
├── weights/                   # 60.6 MB，只读不训
│   ├── tlora_fold0..4.pt      # TLoRA 增量（PEFT delta）
│   ├── tlora_init_fold0..4.npz# 每折 SVD 初始化
│   ├── cnn_folds.pt           # 小 CNN 5 折 state_dict
│   └── README.md              # 结构 / 格式 / 哈希 / ⚠️ 与论文表的差异
├── scripts/
│   ├── run_all.py             # 一键
│   ├── build_splits.py        # 从 train0/test0+conflicts 重建 train1/test1 并核 manifest
│   ├── run_ml_views.py        # hand704 / gap3 / k4
│   ├── run_dl_views.py        # TLoRA + CNN 推理（只推理）
│   ├── load_published_dl.py   # 冻结的已发布概率 → OOF/test
│   ├── fuse_eval.py           # 秩平均融合 3/4/5 视角 + 等权对照
│   ├── verify.py              # 与 expected.json 逐项比对
│   ├── features.py            # 704 维手工特征（从 src/features.py 原样拷入）
│   ├── tlora_model.py         # LoRALinear / ESM_LoRA_PosProj
│   └── peft_io.py             # delta 读写
└── out/                       # 运行产物（gitignore）
```

## 数据口径

`train1` / `test1` 是 **DeepAIP 官方划分去掉 385 条 AIP+/PIP+ 冲突阳性**后的版本：

| | 官方（train0 / test0） | 本包（train1 / test1） | 剔除 |
|---|---|---|---|
| train | 3,583 / 1,365 阳性 | **3,278 / 1,060** | 305（8.51%） |
| test | 897 / 342 阳性 | **817 / 262** | 80（8.92%） |

`conflicts_385.csv` 是这批序列的清单：它等于 PIP 数据集的序列与 `deepaip_orig/table.csv`
中 label==1 行的交集（385 条）。剔除理由：这批序列在多个外部工具里**早就是阳性**——

- **383 / 385** 在 AIPpred 2018 里就是阳性；
- **358 / 385** 在 ProIn-Fuse 里是促炎阳性。

也就是说，把它们留在测试集里测出来的"性能"，很大程度上是在测模型有没有背下这几个公开阳性，
而不是测泛化。剔除后测试集才成为"模型没见过的 AIP+"。

### 剔除没有改变分布（一致性证据）

| 指标 | train0 → train1 | test0 → test1 |
|---|---|---|
| 剔除比例 | 8.51% | 8.92% |
| 阳性率 | 38.10% → **32.34%** | 38.13% → **32.07%** |
| 长度分布 | KS p = 0.916（test 侧） | — |

`build_splits.py` 会从 `train0/test0/conflicts_385.csv` **重新生成** train1/test1，
并断言行数（3,278 / 817）、阳性数（1,060 / 262）与六个文件的 sha256 都与 `manifest.json` 一致。
哈希按 **LF 规范化后的内容**计算，且 `.gitattributes` 里写了 `* -text`，
所以在 Windows（CRLF 检出）与 Linux 上校验结果相同。

### ⚠️ test1 不是文献里的 DeepAIP test

- test1 = 官方 test 的**子集**（817 / 897），阳性率 **32.1%**；
- 因此**不要**把本 README 的 0.8398 与文献里 DeepAIP 官方 test（897 行）的报数直接横比——
  样本不同、难度不同，差 0.01–0.02 完全可能是数据口径造成的；
- 要比就比同一份 test1，或者用 `data/test0.csv` 自行复算官方口径（但那需要另训一版，本包不含）。

## 五个视角

| 视角 | 输入表示 | 模型 | 说明 |
|---|---|---|---|
| **hand704** | 694 维手工特征 + 10 维两亲性特征 | ExtraTrees(1500) | 氨基酸组成、二肽、理化性质、电荷/疏水片段等；`scripts/features.py` 从序列现算，无缓存依赖 |
| **gap3** | 4-mer 窗口的"带一个空位"三元组（空位记 `X`） | TF-IDF → SVD(300) → ExtraTrees(1500) | 3 个空位位置 × 每个 4-mer；捕捉非相邻残基组合 |
| **k4** | 纯 4-mer（`s[i:i+4]`） | TF-IDF → SVD(300) → ExtraTrees(1500) | 最朴素的 k-mer 基线 |
| **TLoRA** | ESM-2 t33 650M 隐藏态 | qkv LoRA(r=8, α=16) + posproj-adaptive 头 | 只训 LoRA 的 B 与头；A 用截断 SVD 初始化 |
| **CNN** | 20 种氨基酸的 one-hot（定长 40，0=pad，1=非标准） | 2 层 Conv1d(k=3,5) + BN + GELU + mean/max 池化 | 34,689 参数的小模型，单卡秒级 |

三个 ML 视角的**严格约束**（历史上踩过坑，`run_ml_views.py` 里逐字保留了原实现）：

1. 分词窗口必须是 **4 字符**、且带 `X` 占位（`gap3` 的 `X` 是真空位，不是 padding）；
2. TF-IDF / SVD / Scaler **只能在折内 fit**，验证集与测试集只 transform。

ExtraTrees 超参：`n_estimators=1500, max_features=0.1, min_samples_leaf=1,
class_weight="balanced"`，seed 0/1/2 取概率均值；5 折用官方折（`train1.csv:fold`）。

## 融合

```
rank_k = rankdata(p_k) / n            # 每个 cohort 内单独排（OOF 3,278 行、test 817 行）
score  = Σ_k w_k · rank_k             # 权重冻结
ŷ      = 1[score ≥ τ],  τ = argmax_τ MCC(OOF)
```

| 档位 | 权重 |
|---|---|
| 3 视角 | hand 0.5926 / gap3 0.1852 / k4 0.2222 |
| 4 视角 | hand 0.2963 / gap3 0.1481 / k4 0.2222 / TLoRA 0.3333 |
| 5 视角 | hand 0.2222 / gap3 0.1481 / k4 0.1852 / TLoRA 0.3333 / CNN 0.1111 |

权重来源：**OOF 秩空间、Caruana 贪心（有放回，27 轮，每轮选 AUROC 增益最大的视角）**，
在训练池上跑一次后**冻结写死**，不在 test 上重选、也不在本包运行时重跑。
`fuse_eval.py` 同时输出每一档的**等权对照**（`equal.*`），用来证明加权不是玄学：
5 视角等权 0.8275 < 加权 0.8398。

排序融合是 **cohort 级（transductive）**的：需要整批样本才能排名，单条序列无法独立打分。
若部署需要逐条推理，应改用概率加权（本包未提供）。

## 本次运行结果

本仓库实测（`python repro_final/scripts/run_all.py`；运行后日志落在 `out/run_all.log`）。
同一份提交用 `git archive` 导出到空目录后重跑，数字与下表逐项相同（54/54 PASS），
说明结果不依赖本机的 `_work/`、缓存或旧 `out/`。

单视角 test1：

| 视角 | AUROC | AUPR | OOF AUROC |
|---|---|---|---|
| hand704 | 0.7825 | 0.5943 | 0.7934 |
| gap3 | 0.7578 | 0.5402 | 0.7366 |
| k4 | 0.7702 | 0.5644 | 0.7430 |
| TLoRA | 0.8268 | 0.6474 | 0.7738 |
| CNN | 0.6965 | 0.4913 | 0.6795 |

融合（论文表，`out/fused_published.json`）：

| 档位 | AUROC | AUPR | OOF AUROC | τ | Sn | Sp | MCC | BA |
|---|---|---|---|---|---|---|---|---|
| 3 视角 | 0.8135 | 0.6207 | 0.8172 | 0.514 | 0.813 | 0.661 | 0.443 | 0.737 |
| 4 视角 | 0.8393 | 0.6639 | 0.8283 | 0.534 | 0.836 | 0.717 | 0.518 | 0.777 |
| 5 视角 | **0.8398** | **0.6676** | 0.8305 | 0.496 | 0.897 | 0.663 | 0.523 | 0.780 |
| 3 视角（等权） | 0.8064 | 0.6084 | 0.8007 | — | — | — | — | — |
| 4 视角（等权） | 0.8354 | 0.6571 | 0.8242 | — | — | — | — | — |
| 5 视角（等权） | 0.8275 | 0.6439 | 0.8245 | — | — | — | — | — |

`verify.py`：**54/54 通过**（`paper` 27 项 @1e-3 + `weights` 27 项 @2e-3）。

## 深度视角到底复现了什么（请先读这一节）

三个 ML 视角是**从序列真跑**出来的：ExtraTrees 在这台机器上重训，数字与论文表逐位一致
（0.7825 / 0.7578 / 0.7702），所以它们是真正意义上的复现。

**两个深度视角不是。** 当年训练它们的脚本（`grid_dl.py`、`grid_cnn.py`）只保存了概率、json 和 log，
**没有 `torch.save` 模型**。现存的两个权重集合都不是产生论文表数字的那一次实例：

| | 论文表那一臂 | 本包里现存的权重 |
|---|---|---|
| TLoRA | `grid_dl/T1`：train1 + `StratifiedKFold(5, seed=42)` | `wt-tlora5`：完整官方池（3,583/897）上的 5 折 |
| CNN | `grid_cnn/A`：官方折 | `ext_cnn/weights_A.pt`：同协议重训（`torch.manual_seed(0)`） |
| 权重是否留存 | ❌ 只剩概率 | ✅ |

所以本包提供两条路径，**两条都被 `expected.json` 硬核对**：

1. **`published`（默认）**：把当年留下的概率一并发布在 `data/published/`，
   `load_published_dl.py` 重推折划分、断言每折 AUROC 与当年记录一致后拼回 OOF/test。
   这条路径逐位复现论文表：**TLoRA 0.8268、CNN 0.6965、5 视角 0.8398**。
2. **`weights`**：用现存权重重新推理（`run_dl_views.py`）：TLoRA **0.7957**、CNN **0.7145**、
   5 视角 **0.8283**。这组数字同样被核对，但它是"另一个模型实例"的成绩，不是论文表。

这不算漂亮，但它是诚实的：**深度视角的复现到这里为止是"概率级"的，不是"权重级"的**。
要把它们变成真正的权重级复现，需要用 `grid_dl` / `grid_cnn` 的原协议重训一次并重新冻结
（约 20 min GPU），那样得到的数字会在 ±0.01 内浮动，而不是逐位相同。

## 已知局限

- **长度捷径**：阴性样本 77% 挤在 15 aa（最短长度），模型很容易学到"短即阴性"。
  本包所有数字都建立在 11–30 aa 的短肽上，别外推到长蛋白。
- **外部集失效**：同一套模型在独立外部集上明显掉点——ext193 **AUROC 0.488**、ext261 **0.389**，
  接近随机。这说明 test1 上的 0.84 有相当一部分来自与 DeepAIP 同分布，而不是普适的 AIP 判别能力。
- **test1 是子集**（817/897，阳性率 32.1%），不要与文献的官方 test 报数直接横比。
- **融合是 transductive 的**（见上），且权重是在本数据 OOF 上搜出来的，属于"方法内评估"。
- 单卡、单 seed 的 CNN 方差约 ±0.014 AUROC（`ext_cnn/results_ext.json` 自己记的 Δ=+0.0136）。

## 许可证与引用

- 代码：**TBD**（待定，建议 MIT / Apache-2.0，发布前替换本行）。
- 数据：来自 DeepAIP 官方划分（`data/splits/deepaip_orig/`），请遵循其原始许可；
  `conflicts_385.csv` 来自 PIP 数据集，同样遵循原始许可。
- 权重：ESM-2 底座（`facebook/esm2_t33_650M_UR50D`）遵循 Facebook 的 ESM 许可（MIT），
  本包只发布在其上训练的 LoRA 增量与头。

引用占位（发布时替换为正式条目）：

```bibtex
@article{<TODO>,
  title   = {<TODO>},
  author  = {<TODO>},
  journal = {<TODO>},
  year    = {<TODO>}
}
```
