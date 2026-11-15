# 五视角 rank + 贪心权重（当前数值最高分支）

> 记录日期：2026-11-15　状态：**可复现、已冻结**，但增益统计上与四视角不可区分（见 §5）
> 一句话：在四视角（hand / gap3 / k4 / TLoRA）基础上加入「自训小 CNN（35k 参数）」，先做视角内 rank 归一，再按 OOF 贪心权重加权 → test 897 **AUROC 0.8308 / AUPR 0.6972**。

## 1. 配置

| 项 | 值 |
|---|---|
| 数据 | `repro/data/{table,folds,test}.csv`（`deepaip_orig`：池 3,583 训练 / 897 test） |
| 视角 1–3 | hand 704（`repro/probs/fold{k}_{hand,gap3,k4}_{val,test}.npy`，ET 1500 树 × 3 seed） |
| 视角 4 | TLoRA + posproj（`repro/tlora5/{k}_{val,test}_probs.csv`，LoRA r=8 qkv，5 折） |
| 视角 5 | 小 1D CNN（34,689 参数，从头训：emb 64 → Conv k3 → BN/GELU → Conv k5 → BN/GELU → mean‖max → Linear；AdamW lr 2e-3 wd 0.02，bs 32，≤12 epoch，早停 patience 4，**3 seed 概率平均**） |
| 融合 | 每视角在本 cohort 内 `rankdata(p)/n` → 权重加权求和 |
| 权重（27 轮 Caruana 贪心，OOF 3,583 条上选） | hand **8/27 = .2963**、k4 6/27 = .2222、gap3 5/27 = .1852、TLoRA 5/27 = .1852、**CNN 3/27 = .1111** |
| 概率映射 | 对融合 rank 分在 OOF 上做 Platt（`LogisticRegression(C=1e6)`），严格单调，不改变 AUROC |

## 2. 结果（test 897，阳性 342）

| 口径 | 值 |
|---|---|
| **AUROC** | **0.8308** |
| AUPR | 0.6972 |
| 等长 AUROC（同长度组内加权） | 0.8098 |
| OOF AUROC（3,583） | 0.8274 |

单视角 test AUROC 参照：hand 0.7942、TLoRA 0.7854、k4 0.7784、gap3 0.7710、小 CNN 0.7271。

对照分支：

| 分支 | test AUROC | AUPR |
|---|---|---|
| 三视角 rank（发布包 3view） | 0.8174 | 0.6804 |
| 四视角 rank 等权（发布包 4view） | 0.8300 | 0.6959 |
| 四视角 rank + 贪心权重 | 0.8291 | — |
| 四视角 raw 概率 + 贪心权重 | 0.8219 | 0.6904 |
| **五视角 rank + 贪心权重（本记录）** | **0.8308** | 0.6972 |

## 3. 怎么用（后面直接调）

```powershell
# 只重算权重与落盘概率，不训练（复用已冻结的 5 个视角概率）
D:\anaconda3\envs\esm2_env\python.exe -X utf8 _work\scratch_pilot\view5\freeze_5view.py
```

产物（`_work/scratch_pilot/view5/`）：

- `weights.json` —— 视角名与权重（27 分制原值也在里面）
- `test_prob_5view.csv` —— 897 行：`sequence,label,score_rank,prob`
- `oof_prob_5view.csv` —— 3,583 行（OOF，供拟合/画图）
- `metrics.json` —— 上表所有数字 + 配对 bootstrap
- `cnn_probs.npz` —— 小 CNN 的 5 折 × 3 seed val/test 概率（缓存，重跑无需训练）

上游脚本：`_work/scratch_pilot/cnn_5fold.py`（CNN 五折）、`cnn_view5.py`（融合评估）、`freeze_5view.py`（冻结）。

## 4. 已知的四个口径事实（汇报时必须一起说）

1. **rank 是 cohort-level 的**：需要整批样本才能算分位，**单条序列无法独立推断**。部署要么用 raw 概率（0.8219），要么明确写成“批次内排序”。
2. **+0.0017 不显著**：与「四视角 rank + 贪心权重」配对 bootstrap 为 **+0.0017，95% CI [−0.0026, +0.0059]**；与等权四视角（0.8300）相比只有 +0.0008。
3. **OOF 涨、test 不动**：OOF 0.8229 → 0.8274（+0.0045），test 只有 +0.0017 —— 与前几次 rank 融合的 OOF/test 脱节一致。
4. **小 CNN 的残差信息很少**：它与四视角融合分的 Spearman 只有 **0.68**（视角间最低），但对融合的残差 AUROC 仅 **0.5244**（预注册门槛 0.62）。也就是说它“不一样”，但“不一样的部分基本是噪声”，只值 1/9 权重。

## 5. 结论与建议用法

- 作为**数值最高的分支**可以用，但论文里不能写成“加入自训 CNN 带来提升”；正确写法是：五视角 0.8308 与四视角 0.8300 在统计上不可区分（Δ CI 跨 0），**主表仍报四视角**，五视角作为“自训轻量模型的价值上限”附表。
- 若审稿人问“为什么不自己训表征”：本分支正好是答案 —— 35k 参数从头训能把单视角做到 0.727（且**几乎没有长度捷径**：等长只掉 0.003，对比小 TF 掉 0.128），但并入融合只值 1/9 权重、+0.0017。
- 后续若要真正提分，方向仍是**换数据/负样本匹配**（BBP 2022 的结论），不是再加视角。

## 6. 环境

`D:\anaconda3\envs\esm2_env\python.exe`（Python 3.11.14 / torch 2.11.0.dev+cu128 / sklearn 1.5.0）。CNN 全流程纯 CPU，5 折 × 3 seed 共 **145 s**；`freeze_5view.py` 不训练，约 20 s。控制台 GBK，必须 `-X utf8` 或 `sys.stdout.reconfigure(encoding="utf-8")`。
