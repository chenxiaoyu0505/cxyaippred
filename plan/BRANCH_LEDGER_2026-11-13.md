# 三条主线台账：单视角 DL / 单视角 ML / 多视角

建立日期：2026-11-13。用途：后续整合的**唯一事实来源**。所有数字都标注了产出文件；没有文件的数字一律标"未落盘"。

## 0. 口径约定（先读这段，否则数字会被混排）

- **数据**：`deepaip_orig`。训练池 3,583（1,365 阳），固定 test 897（342 阳）。test 只评一次。
- **"5 折均值"**：5 个单折模型各自在 897 上评，再对 5 个 AUROC 取均值±SD。**不是**集成值。
- **"集成"**：5 折概率平均后再算 AUROC（固定 test 上合法）。
- **"等长口径"**：同长度组内做 Mann-Whitney 再加权（`_work/fusion/len_matched_compare.py`）。长度 alone 的对照是 0.6314 raw / 0.5000 等长，方法本身已校验。
- **fold 0 单折** ≠ 5 折。单折数字不能和 5 折数字并排放进同一列。
- 2026-11-11 之前按 val AUPR 选优，之后 `monitor='auroc'`。

## 1. 单视角 DL（输入只有序列 → ESM-2 → 网络）

### 1.1 冻结 ESM-2（650M，1280 维）

| 变体 | 头 | 折数 | AUROC | AUPR | 等长 | 产物 |
|---|---|---|---|---|---|---|
| ESM 均值池化 | MLP 512 | 5 | **0.6600±0.0088** | 0.5549 | 0.6078 | `experiments/deepaip_esmonly_mlp/` |
| posproj pad, d=4 | Linear(1280→4)+Conv3→32 槽补零 | 5 | **0.7435±0.0058** | 0.6249 | 0.6016 | `_work/wt-posproj/experiments/posproj5_d4_lr4/` |
| posproj pad, d=16 | 同上 d=16 | 5 | 0.7391±0.0103 | 0.6271 | — | `.../posproj5_d16_lr3/` |
| posproj **adaptive**, d=4 | 个体裁剪→conv→interpolate(32) | 5 | **0.7035±0.0028** | 0.5670 | — | `.../posproj5_d4_adp/` |
| ESM-only CNN 头（arch=cnn, use_hand=False） | ResBlock×2 | — | **从未在 deepaip_orig 跑过** | — | — | — |

要点：pad 与 adaptive 差 0.040，差的正是"零填充图案=长度"这条泄漏（等长口径 pad 掉到 0.6016）。

### 1.2 PEFT：ESM + posproj 头（微调 QKV）

统一结构：LoRA/DoRA/MoRA 注入每层 attention 的 q/k/v，头 = `Linear(1280→4) → Conv1d(4,4,3) → GELU → mask 加权 interpolate(32) → Linear(128→256) → GELU → Dropout(0.3) → Linear(1)`。`posproj` 用 mask 感知重采样，**没有补零图案**（与 1.1 的 pad 不同）。

| 变体 | fold 0 val | fold 0 test | ckpt | 脚本 |
|---|---|---|---|---|
| **LoRA**（r=8, α=16） | 0.8070 / 0.7012 | **0.7445 / 0.6203** | 1.3 GB | `_work/lora/run_lora_posproj_fold0.py` |
| **DoRA** | 0.7938 / 0.6958 | **0.7226 / 0.5990** | 1.3 GB | `_work/lora/run_dora_posproj_fold0.py` |
| **MoRA** | 0.7268 / 0.5836 | **0.6876 / 0.5438** | 1.3 GB | `_work/lora/run_mora_posproj_fold0.py` |
| LoRA + 均值池化头 | 0.7738 / 0.6675 | 0.7120 / 0.5878 | 1.3 GB | `run_lora_fold0.py` |
| LoRA + gap3 特征 | — | 0.7253 / 0.5920 | 1.3 GB | `run_lora_gap3_fold0.py`（非纯 DL） |
| LoRA-PepBERT + posproj | 0.7822 / 0.6739 | 0.7094 / 0.5733 | — | `run_lora_pbposproj_fold0.py` |
| LoRA-PepBERT + 池化 | — | 0.7147 / 0.5742 | — | `run_lora_pepbert_fold0.py` |

排序：**LoRA > DoRA > MoRA**，且 LoRA+posproj 的 val（0.8070）明显高于 DoRA（0.7938）与 MoRA（0.7268）——val 与 test 的排序一致，说明不是早停噪声。

正在跑：`run_lora_posproj_5fold.py`（fold 1 的 ckpt 于 09:26 落盘）。**5 折跑完前，1.2 全部只能算"单折观察"。**

### 1.3 单视角 DL 的三个已知事实

1. LoRA+posproj fold0（0.7445）≈ posproj pad 5 折均值（0.7435）→ **微调 ESM 在这份数据上的贡献尚不能证明**，要等 5 折。
2. 换编码器已被证伪三次：ProtT5 +0.0004/+0.0039、ESM-1280 进 ET 掉 0.034、PepBERT 无效。
3. 冻结 ESM 的最优无泄漏上限是 adaptive posproj 的 0.7035，不是 0.7435。

## 2. 单视角 ML（无神经网络编码器）

### 2.1 特征来源

- **hand704** = AAC20 + GAAC5 + DPC400 + CKSAAGP144 + PAAC50 + CTD63 + 理化12 + 两亲性10。raw，不筛选、不标准化。
- **gap3 200** = spaced 3-mer（3 种排布、窗 4、1 个 wildcard）→ TF-IDF → TruncatedSVD(200, rs=0) → StandardScaler，**逐折只用该折 train 拟合**。
- 其它单视角：gap5 / k3 / k4 / char24，各自 TF-IDF→SVD300→Scaler（定义见 §3 的注意事项）。

### 2.2 分类器与结果（ExtraTrees: 1000 树, sqrt, leaf=2, balanced, 10 seed）

| 特征 | 5 折均值 | 5 折集成 | AUPR | 等长 |
|---|---|---|---|---|
| **hand704 + gap3-200（"ET904"）** | **0.7996±0.0036** | **0.8073** | **0.6747** | 0.7668 |
| hand704 + gap3-200 → sel300 | 0.7918 | 0.7999 | 0.6762 | — |
| hand704 only | 0.7795 | 0.7837 | 0.6506 | — |
| ESM-1280 → ET（对照，非 ML） | 0.6952 | 0.7003 | — | — |

产物：`_work/et_battery/run_et_battery.py`、`et_battery_summary.csv:3`、`_work/et_battery/probs/full904_fold{k}_seed{0..9}.npy`。**没有 .pkl 落盘、没有 manifest。**

## 3. 多视角 MV_BLEND（`_work/mvkmer/`）

### 3.1 做法

- 11 个视角：`hand` + `gap3/gap5/k3/k4/char24`（TF-IDF→SVD300→Scaler）+ 5 个"视角+hand"拼接。
- 每折 ET(1500, mf=0.1, msl=1, balanced, 3 seed)；逐折防泄漏，`run_mv.py` 四阶段断点续跑。
- 权重：**Caruana 有放回贪心 27 轮**，在 5 折 pooled OOF 上选。
- 贪心权重：`hand 0.407 / k4 0.185 / k3 0.111 / char24 0.111 / gap5+hand 0.111 / gap5 0.037 / gap3+hand 0.037`。

### 3.2 结果（`views_eval.csv` / `analysis_summary.csv`）

| | OOF | test AUROC | test AUPR |
|---|---|---|---|
| **GREEDY_BLEND** | 0.8217 | **0.8233** | **0.6899** |
| EQUAL_WEIGHT | 0.8177 | 0.8212 | 0.6866 |
| 最好单视角 gap3+hand | 0.8129 | 0.8153 | 0.6809 |
| gap5 单视角（最弱） | 0.7485 | 0.7643 | 0.6092 |

- 等长配对 **0.7957**（vs ET904 0.7668）→ 等长 +0.029 > raw +0.016，增益不是长度伪影。
- 与 ET904：Spearman 0.9764、残差 AUROC 0.5936、ET 判错子集只救回 5.98%。
- 外部 215 条（部署版）：交叉 AUROC 0.9104（ET904 0.9058）。
- 对方（外部 17 方案报告）原值 0.8273 [0.8006–0.8541]，我们复现 0.8233 [0.7967–0.8505]，**差 0.004、CI 内**。

### 3.3 三个必须随数字一起引用的坑

1. **gap5 与 char24 的定义是推断的**（对方未公开），`weights.json` 的 `note` 已注明。定义敏感性未做消融。
2. 权重在 OOF 上贪心选 → OOF 0.8217 有轻微乐观；test 干净但"嵌套选择"没做。
3. 与 ET904 同源度 0.976 → 融合线到此饱和，不要再往上堆。

## 4. 已冻结的否定结论（别再重复试）

自注意力头 0.7360、端到端 MoE 0.7261、砍容量（h128 0.6563 / h64 0.6623）、904→sel300 掉分、概率级/rank/logit stacking 融合（5 次，最好 0.8105）、活动悬崖加权与删行（−0.0005 / −0.0017）、SMOTE（−0.0082）/mixup（+0.0021）/窄 λ（+0.0060 CI 含 0）、LoRA-PepBERT+posproj 进融合（0.7997）。

## 5. 整合路线（下一步要定的事）

三条线的分数已经收敛（0.8233 / 0.8073 / 0.7445），**再融合只会重复"特征不重叠、错误重叠"的老结论**。可行的整合只有四种：

1. **统一评估框架**（推荐先做）：把 hand、gap3、gap5、k3、k4、char24、ESM-posproj 全部当成"视图"，用同一套 OOF + 等长 + 外部三口径打分，产出一张"视图 × 口径"矩阵。贡献是**哪个视图在哪个子群有用**，不是分数叠加。
2. **按长度/子群分工**：先算各视图在 13–15 / 16–18 / 19–22 段的 AUROC，若确有分工，才谈路由（这是 posproj adapter 之外唯一还站得住的"结构"说法）。
3. **共享表示**：MV 的 SVD 特征与 ESM 表示之间做 CCA/线性映射，看是否能把 ESM 投影到"k-mer 子空间"——这是"整合"而非"融合"的正统做法。
4. **决策层整合（论文第二部分）**：不再合模型，而是把三条线的分数 + 检索特征作为 agent 的输入信号（含 OOD/新颖性），由预算约束选实验。

## 6. 缺口（按优先级）

1. LoRA+posproj 5 折（在跑）；跑完后与 posproj pad 5 折配对比较，回答"微调到底值不值"。
2. ET904 与 MV_BLEND 落盘模型 + manifest（否则两条最高分都不可复现）。
3. MV_BLEND 的等长/外部结果补进 summary，别只留在 HANDOVER。
4. gap5 / char24 定义敏感性消融（改定义重跑一折即可）。
5. 895/897 与 452 test 的 58 条重合、838/897 的 452 线泄漏，写进论文数据节。

## 7. 存储约定：ckpt 只存 delta（2026-11-13 起强制）

**冻结基座只有一个来源**：`C:\Users\cxy\.cache\huggingface\hub\models--facebook--esm2_t33_650M_UR50D\snapshots\<hash>\model.safetensors`（2,488.6 MB），路径注册在 `_work/lora/peft_io.py::BASE_MODELS`。ckpt 里只记 HF 名称，**永不再复制权重**。

| 项 | 旧 | 新 |
|---|---|---|
| `*_best.pt` | 1,313 MB（652.4M 参数，99.68% 是冻结主干） | 8–9 MB（delta，fp32） |
| `*_ckpt.pt`（resume） | 1,331 MB | 8–9 MB + optimizer 状态 |
| 缩比 | — | 约 150× |

delta = LoRA/DoRA 的 `A.weight`/`B.weight`、DoRA 的 `m`、MoRA 的 `A`/`B`（buffer）与 `M`、以及 `posproj`/`posconv`/`head`。

- 工具：`_work/lora/peft_io.py`（`save_delta` / `load_delta` / `save_resume` / `load_resume` / `verify_complete`），兼容读旧的全量 ckpt（自动过滤）。
- 已改：`run_lora_posproj_5fold.py`、`run_lora_5fold.py`、`run_lora_posproj_fold0.py`、`run_lora_fold0.py`、`run_lora_gap3_fold0.py`、`run_lora_pepbert_fold0.py`、`run_lora_pbposproj_fold0.py`、`run_dora_posproj_fold0.py`、`run_mora_posproj_fold0.py`，以及诊断脚本 `check_test_repro.py`、`predict_lora_posproj_fold0_val.py`、`recover_dora_ckpt.py`。
- 历史文件：`convert_ckpt_to_delta.py` 就地转换（逐张量 `torch.equal` 校验 + 原子替换）。fold0 的 14 个文件已转：**13,302 MB → 91.8 MB（回收 13.2 GB）**。
- 复现校验：`verify_delta.py` 重建模型并载入 delta，`lora_posproj_fold0` 的 test **AUROC 0.744494 / AUPR 0.620288 与旧 csv 完全一致**，逐条概率最大差 **7.96e-08**。
- 两个坑：
  1. **MoRA 的 `A`/`B` 是随机 buffer，不是可训参数**，但决定 ΔW；不存就无法复现。首版过滤规则漏了它们（mora delta 只有 0.2 MB），已修；`verify_complete()` 现在会在保存前拦截"可训参数未覆盖"。
  2. **别用 PowerShell 批量改写含中文的 .py**：`Get-Content -Raw` 在 GBK 代码页下把 UTF-8 读成乱码再写回（本次把两个诊断脚本的注释写花，已重写修复）。
- 收尾：fold1/fold2 的 6 个文件是**在跑进程用旧代码**写出的，跑完后执行 `python convert_ckpt_to_delta.py --only fold1`、`--only fold2` 即可再回收约 7.3 GB。

## 8. 微调实验吞吐：为什么慢、怎么提速（2026-11-13 15:25）

**结论先行**：慢的不是 GPU。单臂 90 步 21 s，20 epoch 约 7 min；真正的时间花在“每个变体重写一份 30–40 KB 脚本 + 冒烟 + 排错”和“单卡串行排队”上。

### 8.1 四个来源（实测）
| # | 来源 | 量级 | 证据 |
|---|---|---|---|
| 1 | 每个变体由新子代理从零写自包含脚本（ESM 加载/posproj/batchify/mask/eval/delta 全复制）+ 冒烟 + 修错 | **10–15 min / 变体，纯 CPU 空转** | PiSSA 首个子代理 0 产物崩溃，15 min 白费 |
| 2 | 统一跑 20 epoch / patience 5，但所有臂的 peak 都在 ≤ep7 | 单臂 **−60%** 可省 | LoRA ep2、FA ep2–3、IA³ ep2、TLoRA ep2、DoRA-FFN ep6 |
| 3 | 8.55 GB 单卡 + `.gpu.lock` 串行（DoRA-FFN peak 8.62 GB 已溢出共享内存） | 总时长 = 各臂相加 | 三代理排队 |
| 4 | 诊断（lr 探针、99 模块 SVD 谱）不出分但要装模与时间 | 3–5 min / 变体 | VeRA 3×63 s 探针；PiSSA svd_cache 16 MB |

### 8.2 提速方案（下一批执行）
1. **上限改 8 epoch / patience 3**（依据：全部已试臂的 best_ep ≤ 7）。
2. **一变体一进程 → 一进程多变体**：共享 ESM 加载与 tokenize，按 `--arms vera,pissa,milora,randlora` 依次跑完，单次装模只做一遍。
3. **诊断只在首臂做一次**（谱/梯度探针），后续臂复用结论。
4. 预计：4 个变体从 ~50 min 压到 **15–20 min**。

### 8.3 现状快照（15:07）
- 在跑：VeRA（已出 lr 探针，val 0.732 @3ep，疑似负结果）、PiSSA/MiLoRA 重派版（谱已算：top8 能量 8.55%，bottom8 1e-5%，差 4.07e6×；残差恒等式 5.96e-08）、RandLoRA（bases 已生成，pilot 概率已出）。
- 已试臂全表见 `_work/lora/RESULTS_micro_ft_2026-11-13.md`。
- 硬规则（沿用）：实验默认只跑 fold 0，扩折前必须先问；ckpt 只存 delta（第 7 节）。

## 9. 子代理隔离：worktree + 分支（2026-11-13 起强制）

**规则文档**：`plan/SUBAGENT_WORKTREE_PROTOCOL.md`；**工具**：`scripts/wt.py`（new / list / rm）。

| 项 | 约定 |
|---|---|
| 目录 | `_work/wt-<name>/` |
| 分支 | `exp-<name>`，从 `mlp-head` HEAD 起 |
| 创建 | `python scripts/wt.py new <name>` |
| 收尾 | `python scripts/wt.py rm <name>`（默认保留分支） |
| 提交 | 实验脚本必须 commit，不留未跟踪代码 |
| GPU 锁 | **跨 worktree 用同一绝对路径** `D:\BIP\module4\cxyaippred\_work\gpu.lock` |
| 缓存 | `_work/cache_*` 不复制，脚本用绝对路径引用主工作树 |

**为什么改**：本批次三个子代理共用同一工作树，产物散在未跟踪目录、不属于任何分支；GPU 软锁被反复抢占，PiSSA 的 agent 排队 **26 分钟**，锁内 PID 在 40124/8076/33524/34872 间跳变且新旧格式并存；8.55 GB 卡上三进程并发溢出到共享内存。

**迁移**：正在跑的 VeRA/RandLoRA/PiSSA 不迁移；下一批（T-CENTER、一进程四臂 runner）一律走 `wt.py`。

## 10. 两条环境坑（子代理实测，2026-11-13）

1. **huggingface.co 在本机不可达**：脚本必须在 `import transformers` 之前设 `HF_HUB_OFFLINE=1`，否则 tokenizer 加载会挂 10+ 分钟（VeRA agent 实测）。
2. **别用 pwsh `-replace` / `Set-Content` 改含中文的 UTF-8 脚本**：会把全文变 mojibake 并吞掉约 10 处换行；`_work/` 被 gitignore、无备份，只能整份重写。改文件只用 write/edit 工具或 `python -X utf8`。


## 11. 多视角：把 LoRA 换成 TLoRA（2026-11-13 19:30）

同一套 4 视角贪心（`_work/mvkmer/swap_tlora_4view.py`，OOF 3583 行选权、27 轮带放回、test 只评一次），只替换第 4 个视角：

| 组合 | 权重 | OOF | test AUROC | test AUPR |
|---|---|---|---|---|
| 3 视角（hand+gap3+k4） | .556 / .222 / .222 | 0.8210 | **0.8229** | **0.6933** |
| 4 视角 + LoRA | .444 / .222 / .222 / **.111** | 0.8229 | 0.8221 | 0.6933 |
| 4 视角 + TLoRA | .370 / .222 / .185 / **.222** | **0.8241** | 0.8219 | 0.6904 |

配对 bootstrap（TLoRA 版 − LoRA 版，2000 次）：**ΔAUROC −0.0001 [−0.0031, +0.0029]，P(>0)=0.461** → 完全等价。

**读法**：换成更强的 ESM 臂（单折均值 0.7625 vs 0.7435、集成 0.7854 vs 0.7575）后，贪心给它的权重从 11% 翻到 22%、OOF 从 0.8229 抬到 0.8241，但 **test 一动没动、AUPR 还低 0.003**。这是"ESM 视角对混合体无贡献"的第四次独立证据（前三次：12 视角 +0.0003、fold0 TLoRA −0.003、删掉 LoRA 后 test +0.0008），且这次用的是 ESM 侧最好的模型，排除了"是 LoRA 本身太弱"的解释。

**同时暴露一个方法学问题**：贪心只在 OOF 上最大化，OOF 从 0.8210→0.8229→0.8241 单调上升，test 却 0.8229→0.8221→0.8219 单调下降 —— **权重选择在过拟合 OOF**。论文若报 4 视角，必须说明"选权只用 OOF、test 未参与"，并接受 test 略低于 3 视角这一事实，不能反选。

## 12. 决策阈值 τ = 0.398（2026-11-13 23:26，四视角混合）

**决定**：四视角混合（hand .467 / gap3 .233 / k4 .2 / LoRA .1）的正式工作点定为 **τ = 0.398**，
选择准则 = **池内 OOF 上 MCC 最大**，test 未参与选择（只在选定后评一次）。

**为什么之前没有 τ**：主指标 AUROC/AUPR 是无阈值量；所有涉及阈值的数字（acc 0.7436、MCC 0.45）都是随手 `p ≥ 0.5`。ET 用了 `class_weight='balanced'`，先验从 38.6% 拉到 50%，0.5 并不对应"判阳/判阴等代价"。

**候选与落点**（`_work/fusion/pick_threshold.py`，池内 OOF n=3583 / 阳 1365，AUROC 0.8229）：

| 准则 | τ | 池内 Sn | 池内 Sp | 池内 MCC | test Sn | test Sp | test Acc | test MCC |
|---|---|---|---|---|---|---|---|---|
| 朴素 0.5 | 0.500 | 0.643 | 0.799 | 0.445 | 0.646 | 0.804 | 0.744 | 0.453 |
| **MCC 最优（采用）** | **0.398** | 0.884 | 0.622 | **0.496** | **0.883** | 0.629 | 0.726 | **0.502** |
| Sp = 0.90 | 0.215 | 0.997 | 0.100 | 0.194 | 0.991 | 0.086 | 0.431 | 0.163 |
| Sp = 0.95 | 0.174 | 0.998 | 0.050 | 0.132 | 0.994 | 0.038 | 0.402 | 0.098 |
| Sn = 0.90 | 0.389 | 0.900 | 0.589 | 0.484 | 0.906 | 0.616 | 0.727 | 0.515 |

池内概率中位：阴性 0.357 / 阳性 0.568；test 阴性 0.349 / 阳性 0.563 → 两集合校准一致，τ 可外推。

**必须一起写的三条**：

1. **MCC 不是唯一标准**，它隐含"漏阳与误阴等价"。若目标是"筛候选去合成"，应按 Sp 定（合成一条假阳性有成本）；若目标是"尽量不漏"，按 Sn 定。论文报 MCC 点时必须写明这是等代价假设。
2. **τ 只属于这个模型**。ET(904) 单模型、LoRA-ESM-posproj、TLoRA 各自的概率尺度不同，必须各自在 OOF 上定；**不能把 0.398 套到别的臂上**。
3. **报任何 Sn/Sp 都必须同时写 τ**。外部 215 条全阳性集的 `Sn@0.5 = 0.9302` 是 ET(904) @ 0.5 的结果，与本文 τ 不同口径，**不可并列**。

**落盘**：`_work/fusion/tau.json`（含 τ、准则、样本量、脚本路径、日期）。部署前还需把它与四个视角模型一起冻结（目前视角模型未落盘，`predict.py` 也不覆盖 ET 侧）。
