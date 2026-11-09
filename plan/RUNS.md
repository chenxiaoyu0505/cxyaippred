# 运行台账（RUNS）

> 这个文件是本项目的"实验记忆"。**每跑一次 `run_5fold.py` / `run_legacy.py`，不管成功失败，都要在这里补一行。**
> 目的：任何人（包括几周后的你、或者换一个 agent）只看这一个文件，就知道跑过什么、参数是什么、结果在哪、为什么失败。
>
> 只写事实与结论，不写计划。计划在 `plan/README.md`，决策在 `plan/DECISIONS.md`。
> 结果数字以 `experiments/<exp>/results/metrics.json` 为准，本文件只做索引与解读。

---

## 1. 总表

| # | 日期 | exp | 命令要点 | 状态 | 结果目录 | 一句话结论 |
|---|---|---|---|---|---|---|
| 1 | 2026-11-09 17:38 | `cv5` | 默认参数（无 EMA、无 MCC、无噪声、seed 0） | **完成** | `experiments/cv5/` | AUROC 0.7277±0.0130 / AUPR 0.6494±0.0103；单折主表 |

状态取值：`运行中` / `完成` / `失败` / `中止`。

---

## 2. 逐次记录

### #1 `cv5` —— 5 折基线

- **启动时间**：2026-11-09 17:38（用户本地终端，非本会话发起）
- **命令**：
  ```powershell
  python scripts\run_5fold.py --exp cv5 --note "5折基线"
  ```
- **实际参数**（全部取默认值）：

  | 项 | 值 |
  |---|---|
  | 划分 | `split_seed42_n5`（sha256 `4b17ee05f73e8363…`） |
  | 折 | 0–4，共 5 折 |
  | model seed | 0 |
  | epochs / batch / lr | 50 / 16 / 1e-4 |
  | patience（val AUPR 早停） | 10 |
  | EMA-SAT | 关 |
  | MCC 类权重 | 关 |
  | label-noise | 0 |
  | device | `auto` → cuda |

- **进度证据**（文件时间戳）：
  - `17:37:57` `cache/esm_emb_fp16.npy`（858 MB）、`esm_mask.npy` 写完
  - `17:38:32` `cache/esm_index.csv`、`esm_meta.json` 写完 → 缓存建立完成，约 5 分钟
  - `17:38:39` `experiments/cv5/runs/fold0_m0/{config.json,split.csv}` 写出 → fold0 开始训练
  - 之后要等到该折训练结束才会出现 `best.pt` / `train_log.csv` / `*_probs.csv`（训练中不写盘）
- **预期产物**：`experiments/cv5/runs/fold{0..4}_m0/` 各 8 个文件；`experiments/cv5/results/{metrics_detail.csv,metrics.json,summary.md}`；`experiments/cv5/manifest.json`；`experiments/_index.csv`
- **跑完第一件事**：
  ```powershell
  python scripts\list_exps.py
  type experiments\cv5\results\summary.md
  ```
- **观察点**（先写下预期，避免事后找理由）：
  1. 5 折 AUPR 的折间 sd 有多大——如果 sd 和折间均值差同量级，说明单折数字不可比。
  2. 概率校准：ECE / Brier 有多大，`@val` 阈值是否被 miscalibration 拖累（5 折主表只报单折）。
  3. `@val` 与 `@0.5` 的 MCC 差多少——差值就是"阈值调优"的贡献，旧稿 PIP 的 +0.10 正是差在这。
  4. 与旧 AIP 头（AUROC 0.7336 / AUPR 0.6089 / MCC 0.3988@0.487）比，新管线是否至少不更差。
- **状态更新（2026-11-09 17:52 完成）**：
  - 5 折全部跑完，`experiments/cv5/results/` 已生成。
  - 单折主结果：AUROC **0.7277 ± 0.0130**，AUPR **0.6494 ± 0.0103**，MCC@0.5 **0.3141 ± 0.0279**，MCC@val **0.3580 ± 0.0259**。
  - 与旧 AIP 头（AUROC 0.7336 / AUPR 0.6089 / MCC 0.3988@0.487）比：AUROC 略低、AUPR 略高、MCC 低约 0.04；先作为新管线基线，不据此下结论。
  - **异常**：早期 `summary.md` 里的 `lofo_ensemble`（AUROC 0.8932）已确认是泄漏（评 fold k 时平均了见过 fold k 的模型），已从 `eval_aip.py` 删除；该数字不得引用。

---

## 3. 待跑清单（跑完一个划一个）

| 顺序 | exp | 命令 | 为什么 |
|---|---|---|---|
| 2 | `cv5_ema` | `--ema` | EMA-SAT 有没有精度增益 |
| 3 | `cv5_mcc` | `--mcc-adaptive` | MCC 加权有没有用、会不会来回摆 |
| 4 | `cv5_full` | `--ema --mcc-adaptive` | 旧 IDPP 配方整体 |
| 5 | `noise10` / `noise10_ema` | `--label-noise 0.10`（±`--ema`） | EMA 是否真的抗噪（判定规则见 DECISIONS.md） |
| 6 | `legacy` | `run_legacy.py --exp legacy` | 与外部基线的公平对照线（4,224/462） |
| 7 | `cv5_cluster` | 换聚类划分 | 回应 R1-4 的序列相似性泄漏（尚未实现） |

---

## 4. 记录模板（复制粘贴用）

```markdown
### #N `exp名` —— 一句话目的

- **启动时间**：
- **命令**：
- **实际参数**：（只写与默认值不同的项）
- **状态**：完成 / 失败 / 中止
- **产物**：
- **关键数字**：auroc / aupr / mcc(@val) / 折间 sd
- **结论**：这一跑支持/否定了什么
- **异常**：若有，写清楚现象与排查到哪一步
```
