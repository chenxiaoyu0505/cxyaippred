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
| 2 | 2026-11-09 18:43 | `cv5_full` | `--ema --mcc-adaptive`（历史，MCC 已移除） | **完成** | `experiments/cv5_full/` | AUROC 0.7316±0.0060 / AUPR 0.6525±0.0049；但 MCC@0.5 降至 0.2505，校准变差 |
| 3 | 2026-11-10 10:51 | `cv5_fixed` | `--norm layer`（三处 bug 全修，mac/MPS） | **完成** | `experiments/cv5_fixed/` | AUPR 0.6063±0.0211 / AUROC 0.7011±0.0151；比 cv5 低 0.043，掉的是长度捷径 |
| 4 | 2026-11-10 12:0x | `cv5_fixed_ema` | `cv5_fixed` + `--ema` | **完成** | `experiments/cv5_fixed_ema/` | AUPR 0.6102±0.0187 / AUROC 0.7039±0.0120；净增益 +0.0039 在噪声内，主要作用是降方差 |
| 5 | 2026-11-10 16:09 | `cv5_peptidebn` | `--norm peptide_bn --keep-cache` | **完成** | `experiments/cv5_peptidebn/` | AUPR 0.6027±0.0115 / AUROC 0.6949±0.0093；**sd 减半，定为新基线** |
| 6 | 2026-11-10 17:0x | `cv5_peptidebn_ema` | `--norm peptide_bn --ema --keep-cache` | **完成** | `experiments/cv5_peptidebn_ema/` | AUPR 0.5998±0.0145 / AUROC 0.6955±0.0114；EMA 这条线上无增益 |
| 7 | 2026-11-10 18:5x | `probe_cfc` | `--lnn-only --keep-cache`（只训 CfC 支路） | **完成** | `experiments/probe_cfc/` | AUPR 0.4867±0.0286 / AUROC 0.5880±0.0263；**不如只用肽长**（0.5074/0.6090），CfC 否决 |

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
  | MCC 类权重（已移除） | 关（当时未启用） |
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

### #2 `cv5_full` —— EMA-SAT + MCC 自适应（旧 IDPP 配方；历史记录）

> **2026-11-10 更新**：MCC 自适应类权重已从代码中移除，下面的 `--mcc-adaptive` 命令仅作历史存档，不能再执行。当前只保留 EMA-SAT，并修正了首次命中、decay、warmup 与更新时机。

- **启动时间**：2026-11-09 18:43
- **命令**：
  ```powershell
  python scripts\run_5fold.py --exp cv5_full --ema --mcc-adaptive --note "5折+EMA+MCC"
  ```
- **实际参数**：5 折 / seed 0 / 50 epochs / batch 16 / lr 1e-4；EMA-SAT 开（alpha 0.7，decay 0.99，warmup 3）；MCC 自适应开（interval 3，rate 0.15，ratio 1.2，cooldown 3，权重 0.3–3.0）；其余默认
- **状态**：完成（2026-11-09 19:03）
- **产物**：`experiments/cv5_full/{runs,results,manifest.json}`
- **关键数字**（单折，mean ± sd）：
  - AUROC **0.7316 ± 0.0060**（基线 0.7277 ± 0.0130，Δ +0.0040）
  - AUPR **0.6525 ± 0.0049**（基线 0.6494 ± 0.0103，Δ +0.0031）
  - MCC@0.5 **0.2505 ± 0.0758**（基线 0.3141 ± 0.0279，Δ −0.0636）
  - MCC@val **0.3517 ± 0.0219**（基线 0.3580 ± 0.0259，Δ −0.0063）
  - F1@val **0.5535 ± 0.0306**（基线 0.5829 ± 0.0391，Δ −0.0294）
  - 校准：ECE **0.1823** / Brier **0.2441** / mean_p **0.5681**（基线 0.1167 / 0.2171 / 0.4182；真实阳性率 0.3927）
- **结论**：
  - 排序指标（AUROC/AUPR）略升，且折间 sd 约减半；但 paired 检验不显著（AUROC p=0.343，AUPR p=0.429），不能当成确定增益。
  - 概率整体右移：0.5 阈值下预测阳性率 **0.678**（基线 0.349），precision 0.494（基线 0.630）、recall 0.813（基线 0.521）；因此 F1@0.5 反而升，但 MCC@0.5 大幅下降。
  - `@val` 阈值自动抬高到 0.657 后，MCC 与基线基本持平（0.3517 vs 0.3580），但 F1 仍低；说明 full 的主要问题是校准/阈值，不是排序。
  - **不能把 full 当作整体更优**；下一步必须拆 `cv5_ema` 与 `cv5_mcc`，确认是 EMA 带来稳定、还是 MCC 把概率推高。

### #3 `cv5_ema_fix` —— 修正版 EMA-SAT（固定 α=0.7，教师 β=0.3）

- **启动时间**：2026-11-10 09:22
- **命令**：
  ```powershell
  D:\anaconda3\envs\esm2_env\python.exe scripts\run_5fold.py --exp cv5_ema_fix --ema --note "修正版EMA"
  ```
- **实际参数**：5 折 / seed 0 / 50 epochs / batch 16 / lr 1e-4；EMA 开，固定 `α=0.7`（教师 β=0.3）、`decay=0.9`、`start_epoch=1`；首次命中直接初始化，epoch 末 eval 模式更新。该结果来自旧 `--ema-alpha` 接口，当前 CLI 已改为 `--ema-alpha-min`。
- **状态**：完成（2026-11-10 09:40）
- **产物**：`experiments/cv5_ema_fix/{runs,results,manifest.json}`
- **关键数字**（单折，mean ± sd）：
  - AUROC **0.7294 ± 0.0100**（基线 0.7277 ± 0.0130，Δ +0.0018，p=0.621）
  - AUPR **0.6515 ± 0.0104**（基线 0.6494 ± 0.0103，Δ +0.0021，p=0.631）
  - MCC@0.5 **0.2947 ± 0.0493**（基线 0.3141 ± 0.0279，Δ −0.0195，p=0.336）
  - MCC@val **0.3267 ± 0.0057**（基线 0.3580 ± 0.0259，Δ −0.0314，p=0.043）
  - F1@val **0.5346 ± 0.0706**（基线 0.5829 ± 0.0391，Δ −0.0484，p=0.147）
  - 校准：ECE **0.1290** / Brier **0.2221**（基线 0.1167 / 0.2171）
- **结论**：实现修正后 EMA 不再卡在 0.5，但排序指标仍无显著增益，MCC@val 显著下降；SAT 不能作为主创新，只能作负结果/可选正则。
- **下一步**：`cv5_ema_ramp` 用指数饱和硬标签权重 α（α_min=0.7、start=2、tau=1.0、decay=0.65）验证渐进 SAT；不要再用“教师权重 0.7”的旧解释。

### #4 `cv5_ema_ramp` —— 指数饱和 EMA-SAT 的旧 `exp` 近似（已完成）

- **首次启动**：2026-11-10 11:00（旧 β 语义，已终止）
- **作废原因**：把 0.7 当成教师权重 β_max，实际应为硬标签权重 α_min=0.7（教师上限 0.3）。旧 run 的 fold0–fold2 产物只能当错误实现记录，不得作为结果。
- **旧命令（不可再执行）**：
  ```powershell
  # 旧接口已删除；仅作历史记录
  D:\anaconda3\envs\esm2_env\python.exe scripts\run_5fold.py --exp cv5_ema_ramp --ema --ema-beta-max 0.7 --ema-start-epoch 2 --ema-tau 1.0 --ema-decay 0.65 --note "指数饱和SAT"
  ```
- **该 run 的参数**：α_e = 0.7 + 0.3·exp(−(e−2+1)/1.0)；e=2/3/4/5 时 α=0.810/0.741/0.715/0.706，β=0.190/0.259/0.285/0.294。这是旧 `exp` 近似，**不是**用户最终指定的 `2^(−max(0,e−1))`。
- **新命令**：
  ```powershell
  D:\anaconda3\envs\esm2_env\python.exe scripts\run_5fold.py --exp cv5_ema_ramp --ema --ema-alpha-min 0.7 --ema-start-epoch 2 --ema-tau 1.0 --ema-decay 0.65 --note "渐进SAT(alpha0.7)"
  ```
- **状态**：完成（2026-11-10 11:59；旧目录已移到 `_work/cv5_ema_ramp_beta07_invalid`）
- **结果**（单折，mean ± sd）：AUROC **0.7293 ± 0.0147**（基线 0.7277 ± 0.0130，Δ +0.0016，p=0.775）；AUPR **0.6516 ± 0.0142**（基线 0.6494 ± 0.0103，Δ +0.0022，p=0.662）；MCC@0.5 **0.3410 ± 0.0276**（基线 0.3141，Δ +0.0269，p=0.026）；MCC@val **0.3417 ± 0.0221**（基线 0.3580，Δ −0.0163，p=0.385）；F1@free 0.5999；ECE **0.1211**、Brier **0.2193**（基线 0.1167/0.2171）。
- **结论**：与固定 `cv5_ema_fix`（0.7294/0.6515）几乎相同；AUROC/AUPR 无显著增益，MCC@0.5 的提高来自概率整体少右移（阈值效应），MCC@val 仍低于 `cv5`。旧 `exp` 近似不带来额外收益，SAT 不能作为主创新；最多作为“阈值/校准”消融或负结果。
- **产物**：`experiments/cv5_ema_ramp/{runs,results,manifest.json}`。
- **精确 `2^-s` 版（待跑，未启动）**：当前代码已实现 `α_e=0.7+0.3·2^(−max(0,e−start+1)/τ)`；`start=2, τ=1` 时 e=2/3/4/5 为 α=0.850/0.775/0.7375/0.71875。用户自行执行：
  ```powershell
  D:\anaconda3\envs\esm2_env\python.exe scripts\run_5fold.py --exp cv5_ema_halving --ema --ema-alpha-min 0.7 --ema-start-epoch 2 --ema-tau 1.0 --ema-decay 0.65 --note "精确2^-s渐进SAT"
  ```

---

### #5 `cv5_peptidebn` —— masked BN + 按肽等权（新基线）

- **启动时间**：2026-11-10 16:09（macOS / MPS，`--keep-cache`）
- **命令**：`python scripts/run_5fold.py --exp cv5_peptidebn --norm peptide_bn --keep-cache`
- **实际参数**：`--norm peptide_bn`；其余同 `cv5`（epochs 50 / batch 16 / lr 1e-4 / patience 10 / 无 EMA）
- **状态**：完成
- **关键数字**：折均 AUPR **0.6027±0.0115** / AUROC **0.6949±0.0093**；
  OOF（池化口径）AUPR 0.5788 / AUROC 0.6857；逐折 test AUPR 0.6169/0.6038/0.6096/0.5875/0.5958
- **与 `cv5_fixed` 比较**：OOF paired bootstrap ΔAUPR −0.0050，95% CI [−0.0221, +0.0115]（跨 0），
  均值无显著差异；但 **sd 减半**（AUPR 0.0211→0.0115，AUROC 0.0151→0.0093）
- **结论**：作为新基线。sd 减半是真实收益（选择噪声只会同时抬高两者）。跨折阈值仍很差
  （0.298/0.810/0.496/0.374/0.401），原始池化 OOF AUPR 0.5788，按折秩归一化后 0.6013（+0.023）
  ⇒ 部署前必须做 Platt/isotonic 校准
- **异常**：无

### #6 `cv5_peptidebn_ema` —— 新基线 + EMA

- **命令**：`python scripts/run_5fold.py --exp cv5_peptidebn_ema --norm peptide_bn --ema --keep-cache`
- **实际参数**：`--ema-alpha-min 0.7 --ema-start-epoch 2 --ema-tau 1.0 --ema-decay 0.65`（默认值）
- **状态**：完成
- **关键数字**：折均 AUPR 0.5998±0.0145 / AUROC 0.6955±0.0114；OOF 0.5875 / 0.6936
- **结论**：EMA 在 masked BN 这条线上没有增益（均值 −0.0029 AUPR），与 D-10b 的结论一致——
  EMA 只是降方差，而 masked BN 已经把 sd 降下来了，两者作用重叠

### #7 `probe_cfc` —— CfC 支路生死判据（负结果）

- **命令**：`python scripts/run_5fold.py --exp probe_cfc --lnn-only --keep-cache`
- **实际参数**：只训 CfC 支路，`use_esm=False use_hand=False`，hidden 128，0.39M 参数
- **状态**：完成
- **关键数字**：折均 AUPR **0.4867±0.0286** / AUROC **0.5880±0.0263**；OOF 0.4834 / 0.5873；
  逐折 OOF AUPR 0.4923/0.4555/0.4667/0.4890/0.5297
- **对照**：只用肽长当分数 → OOF AUPR **0.5074** / AUROC **0.6090**；随机 0.3927
- **结论**：**CfC 不如一个单特征基线，否决**（见 DECISIONS.md D-11）。
  best_epoch 10–25 且末轮 val 低于峰值 ⇒ 过拟合而非欠训。
  与基线的秩相关 0.372，任何非零融合权重都不超过基线

---

## 3. 待跑清单（跑完一个划一个）

| 顺序 | exp | 命令 | 为什么 |
|---|---|---|---|
| — | `cv5_ema` / `cv5_mcc` / `cv5_full` | 历史实验 | 已完成；`cv5_ema` 用的是未修正的 EMA，`cv5_mcc`/`cv5_full` 含已移除的 MCC，均只能当历史对照 |
| 2 | `cv5_ema_fix` | 旧接口固定 α=0.7（历史） | 已完成：AUROC 0.7294 / AUPR 0.6515，排序无显著增益 |
| 2b | `cv5_ema_ramp` | 旧 `exp` 近似，已完成 | AUROC 0.7293 / AUPR 0.6516，与固定 α=0.7 相同；不改善排序 |
| 2c | `cv5_ema_halving` | `--ema --ema-alpha-min 0.7 --ema-start-epoch 2 --ema-tau 1.0 --ema-decay 0.65` | 精确 `2^-s` 调度；已被 `cv5_peptidebn_ema` 覆盖，优先级降低 |
| 3 | `cv5_fix_nohand` | `--norm peptide_bn --hand-norm none --keep-cache` | **长度捷径的阳性对照**：不标准化手工特征，预期全局回升但固定长度子集不涨 ⇒ 证明回升来自捷径 |
| 3b | `cv5_nohand` | `--no-hand`（需先加 CLI flag） | 隔离 ESM 通道：若全局 AUROC 仍比固定长度子集高 ~0.08，说明 ESM 也在泄漏长度 |
| 4 | `noise10` / `noise10_ema` | `--label-noise 0.10`（±`--ema`） | EMA 是否真的抗噪（判定规则见 DECISIONS.md） |
| 5 | `legacy` | `run_legacy.py --exp legacy` | 与外部基线的公平对照线（4,224/462） |
| 6 | `cv5_cluster` | 换聚类划分 | 回应 R1-4 的序列相似性泄漏（尚未实现） |
| — | `ab_lnn` | 已取消 | CfC 探针否决，见 DECISIONS.md D-11 |

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

## 4. 4,190 口径的跑批计划（2026-11-10 起，替代上面第 3 节的旧清单）

数据集已切换为 4,190 条（DECISIONS.md D-15），**上面第 3 节的待跑项全部作废**，除标注"历史"的以外不得再跑。

| 顺序 | exp | 命令 | 为什么 |
|---|---|---|---|
| 1 | `cv5_clean` | `run_5fold.py --exp cv5_clean --note "4190 清洗版基线" --norm peptide_bn --keep-cache` | 新口径的**唯一参照线**，之后所有 Δ 都相对它 |
| 2 | `cv5_clean_amph` | 同上 + `--amphipath`（exp 改 `cv5_clean_amph`） | μH 10 维在干净数据上是否还有增益（4,686 上是 +0.010 AUROC / −0.003 AUPR，不可靠） |
| 3 | `cv5_clean_layers` | 同上 + `--esm-fuse weighted` | 三层 ESM 24/30/33（缓存已就绪，869 MB） |
| 4 | `cv5_clean_ema` | 同上 + `--ema` | EMA 是否仍主要降方差 |
| 5 | `cv5_clean_nohand` | `--no-hand`（需先加 CLI flag） | ESM 单通道的边际价值 |
| 6 | `fsel_leak` | 复现对方的 top-300 特征选择流程 | 验证 0.84 vs 0.68 的 0.15 差距是否来自选择泄漏 |

**统一报数口径**（D-12 / D-15）：

```bash
# 全集 + 与 HD-AIP 可比的 4,190 子集 + 换用对方标签
python scripts/eval_subset.py --exp cv5_clean --overlap data/HD-AIP_benchmark_4194.csv
python scripts/eval_subset.py --exp cv5_clean --overlap data/HD-AIP_benchmark_4194.csv --labels from-subset
```

**注意**：`--keep-cache` 必带（否则每折删 1.1 GB 缓存重算）；复现单层实验时若缓存 meta 是 `layers_v1`，
必须加 `--esm-cache single`。
