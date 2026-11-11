# cxyaippred —— 项目连续性文档

> 这份文件写给**下一个接手的人（或者忘光了的我）**。
> 读完这一页就该知道：为什么有这个项目、已经做了什么、现在卡在哪、下一步做什么。
> 最后更新：2026-11-09

---

## 1. 背景：为什么要重开一个项目

原来的论文是 **IDPP**：一个双任务框架，同时预测抗炎肽（AIP）和促炎肽（PIP），
投稿 *Bioinformatics* 后被拒。两位审稿人的核心意见（原文见 `plan/REVIEWS.md` 摘要）：

| 编号 | 意见 | 是否致命 |
|---|---|---|
| R1-1 | 多任务学习的收益没有证明，必须和「两个独立单任务模型」「常规 masked MTL」对比 | 致命 |
| R1-2 | 输出两个正预测 ≠ 识别双向肽，应在**双标注肽**上评估 | 致命（但数据只有 10 条） |
| R1-3 | 报告不一致（Table 1 与 Fig 6D 计数、特征维度、Fig 6H 图文不符），指标须从样本级重算 | 致命 |
| R1-4 | 随机划分不足，需序列聚类评估、重复运行、不确定性估计 | 高 |
| R1-5 | 机制结论超出证据（聚丙氨酸突变、近正交权重不证明功能解耦） | 中 |
| R2 | 超参敏感性、验证集分布、基线偏旧、缺 AUROC/AUPR 与显著性检验、EMA-SAT 加速收敛无证据…… | 高 |

**2026-11-09 的决定：放弃双任务，只做干净的单任务 AIP 预测，架构沿用原框架。**
理由：R1-1、R1-2 以及 mask/部分标注相关问题会一次性消失；而且 AIP 这一半数据本身是干净的。

代价也写在这里：多任务 + 部分标注 + 双向肽是原稿的主要"新颖性"来源，
删掉之后方法只剩 EMA-SAT + ESM-2/手工特征融合，需要重新论证它凭什么发表。

---

## 2. 数据（已定版，不要随意改）

来源：`D:\BIP\data\train2.csv`（7,096 行）与 `D:\BIP\data\test.csv`（804 行），
取 `mask-aip == 1` 的行，只保留 `sequence, label` 两列。

| 项目 | 值 |
|---|---|
| 原 AIP 训练集 | 4,224 行（1,655 阳 / 2,569 阴） |
| 原 AIP 测试集 | 462 行（185 阳 / 277 阴） |
| 合并 | **4,686 行 / 4,686 唯一序列**（1,840 阳 / 2,846 阴，39.3%） |
| 重复序列 | 0 |
| 标签冲突 | 0 |
| 原训练/测试精确重叠 | **0** |

> 旧稿统计里那 1,367 个"重复序列"是同一肽在 AIP 和 PIP 两个任务各一条记录
> （mask `10` 与 `01` 成对），**不是重复标注，也不是标签噪声**。
> 曾有一版结论说"788 对标签冲突"，已复核为错误，不再使用。

划分：`data/splits/split_seed42_n5/folds.csv`（长表 23,430 行 = 4,686 × 5 折），
同目录 `manifest.json` 记录 split_seed / n_folds / 每折人数 / 文件 sha256。
（旧的 `data/folds_5cv_seed42.csv` 已删除，内容与新版逐条一致，见 `data/FOLDS.md`。）

**2026-11-10 BIP-free 变体**：原 `aip_all.csv` 含 56 条同一行 `label-aip=1,label-pip=1`
的 BIP（46 train + 10 test）。为排除旧双任务 BIP 重叠，另生成
`data/aip_all_nobip.csv`：4,634 行（1,788 阳 / 2,846 阴，38.6%），删除这 56 条，
并补回 4 条 HD-AIP-only 阳性。旧 `data/splits/split_seed42_n5_nobip/folds.csv`
（23,150 行 = 4,630 × 5 折）只作历史，不再与当前主表对应；原文件与原 split 保留用于复现。

**2026-11-10 评估设计固定为原始 train/test**：新实验沿用 BIP-free 的原始
`aip_train_nobip.csv`（4,182 条：1,613 阳 / 2,569 阴）和
`aip_test_nobip.csv`（452 条：175 阳 / 277 阴）。5 折只在 4,182 条 train 内做，
每折 80% train / 20% val；452 条 test 固定，模型不参与任何选择。
产物：`data/splits/split_origtest_cv5_seed42_nobip/`（23,170 行），生成脚本
`scripts/make_holdout_cv.py`（`--data` + `--test-data`）。
旧 `split_seed42_n5` / `split_seed42_n5_nobip` 只用于复现；4,167/463 的 holdout 设计已废弃。

| 层 | 占总量 | 行数 | 用途 |
|---|---|---|---|
| 固定 test | 9.8% | 452 | 只在最后打分，模型不参与任何选择 |
| val | 18.0% | 836/837 | 早停、选阈值 |
| train | 72.2% | 3,345/3,346 | 梯度更新、EMA |

旧 5 折的 test/val/train 比例表（20/16/64）仅适用于 `split_seed42_n5*` 历史结果。

| 层 | 占单折 | 行数 | 用途 |
|---|---|---|---|
| test | 20% | 937 | 只评一次，不得用于任何选择 |
| val | 16% | 750 | 早停、选阈值 |
| train | 64% | 2,999 | 梯度更新、EMA |

`val : train = 20 : 80`，与原稿一致。详见 `data/FOLDS.md`。

---

## 3. 文件夹各自是干什么的

```
cxyaippred\
  plan\          本项目文档（本文件、决策记录、审稿意见摘要）
  data\          定版数据
    splits\      划分登记处：每个 split_id 一个目录（folds.csv + manifest.json）
  src\           **新**单任务代码：模型、权重格式、特征、指标、配置
  scripts\       生成划分/缓存、训练、评估、推理、自检、旧权重转换、统筹运行
    legacy_multitask\   旧双任务脚本快照，**只读参考，不要修改也不要 import**
  experiments\   一次实验一个目录（见下）
  cache\         可重建缓存（hand_feats.npy 常驻；ESM 隐状态用完即删）
  heads\         5 折的 AIPHead 权重（每份 ~80 MB），现在为空
  models\        本地 ESM-2 拷贝（可选），现在为空，默认不复制
  reference\     旧双任务模型在 AIP 上的预测，作为"不许更差"的基线
  _work\         冒烟测试用的随机向量缓存，**不是结果**，不要引用
```

实验目录统一长这样，**不同实验永不混写**：

```
experiments\<exp>\
  runs\<fold>_m<seed>\   best.pt / config.json / split.csv / val_probs.csv
                         test_probs.csv / all_probs.csv / train_log.csv / threshold.json
  results\               metrics_detail.csv / metrics.json / summary.md
  manifest.json          数据与划分的 sha256、代码哈希、环境、耗时
```

`<exp>` 命名小写无空格：`cv5`（5 折主实验）、`legacy`（原 4,224/462 对照）、
将来的 `cv5_ema`、`cv5_nomcc` 等消融。

**关于 "src"（澄清）**：`cxyaippred` 下面**只有一个** `src\`，就是上面第 67 行那个，
内容为 `model_aip.py`、`checkpoint.py`、`features.py`、`__init__.py`
（`src\__pycache__\` 是 Python 自动生成的字节码缓存，不是源码）。

容易看错的是另外两处：

| 位置 | 是什么 | 能不能用 |
|---|---|---|
| `cxyaippred\src\` | 新单任务代码包 | ✅ 只能用这个 |
| `cxyaippred\scripts\legacy_multitask\` | 旧双任务脚本的**只读快照**（7 个文件，含 models.py / features.py / utils.py） | ❌ 只看，不 import |
| `D:\BIP\module4\immudualpred\src\` | **项目外**的旧双任务包 | ❌ 不 import |

也就是说：项目内没有第二个 src，只有一份旧代码快照目录。

真正的风险在 `features.py`（704 维手工特征 = 694 维基础块 + 10 维两亲性）——它现在有三份**物理副本**
（`immudualpred\src\`、`cxyaippred\src\`、`cxyaippred\scripts\legacy_multitask\`），
内容目前相同但会漂移。新项目只准用 `cxyaippred\src\features.py`；
`scripts/selftest.py` 会断言基础块输出 694 维且无 NaN，改坏了会当场失败。

---

## 4. 权重策略（重要，别再踩）

旧代码 `torch.save(model.state_dict())` 把**冻结的 ESM-2 650M 也存进去了**，
每份 2.69 GB；5 折 × 3 seeds = 40 GB。改成只存 `AIPHead`：

| 方案 | 单份 | 5 折 | 5 折 × 3 seeds |
|---|---|---|---|
| 完整 state_dict | 2.69 GB | 13.5 GB | 40 GB |
| 只存头（现方案） | **80 MB** | 0.4 GB | 1.2 GB |

* `AIPHead`：ResCNN(1280, 2 块) + 掩码均值池化 + 拼 704 维手工特征（694 + 10 两亲性）+ MLP(512)，
  共 **20.95M 参数 / 80.0 MB**。
* ESM-2 **不进权重文件**，只在 meta 里记名字 `facebook/esm2_t33_650M_UR50D` 与层号 `-1`。
* 文件格式见 `src/checkpoint.py`（`format = cxyaippred-aip-head-v3`）。

**推理形态（已定）：5 折集成。**（部署形态；CV 评估只报单折，不能在同一批数据上估 5 模型集成。）ESM 只前向一次，然后分别过 5 个头，取平均概率；
同时输出折间标准差（分歧大 = 该序列可能落在训练分布外）。
不输出 0/1，除非显式给阈值。

---

## 5. 现状（2026-11-09 18:40）

### 已完成

- [x] 数据集抽取与核对（`scripts/make_aip_dataset.py`，`data/DATA_STATS.md`）
- [x] 5 折划分，迁到 `data/splits/split_seed42_n5/` + manifest（`scripts/make_5fold.py`）
- [x] 新单任务模型定义 `src/model_aip.py`
- [x] 权重格式与读写 `src/checkpoint.py`
- [x] 集成推理脚本 `scripts/predict.py`
- [x] 自检 `scripts/selftest.py`（前向形状、变长掩码、round-trip、体积、拒绝非法格式、694 维基础块）
- [x] 旧权重迁移工具 `scripts/convert_legacy_head.py`（2.51 GB → 80 MB，已完成一次验证）
- [x] **训练 `scripts/train_aip.py`**（EMA-SAT 可开关、按 val AUROC 早停、val 上选阈值）
- [x] **评估 `scripts/eval_aip.py`**（free/@0.5/@val 三口径 + bootstrap CI，只报单折）
- [x] **统筹 `scripts/run_5fold.py`**（自动建划分→建缓存→5 折训练→评估→写 manifest→删缓存）
- [x] **对照 `scripts/run_legacy.py`**（原 4,224 训练 / 462 测试，用于和外部基线并表）
- [x] 指标唯一实现 `src/metrics.py`（train 只产概率、eval 才算指标）
- [x] 端到端冒烟：`--exp smoketest` 跑通 1 epoch + 评估（已删除该冒烟产物）
- [x] 命令行 `--label-noise`（默认 0）：训练行对称翻转标签，逐条记录到 `label_noise.json`
- [x] 中文 README（`README.zh-CN.md`）与英文版互链
- [x] **`cv5` 5 折基线**（2026-11-09 17:52 完成）：AUROC 0.7277 ± 0.0130 / AUPR 0.6494 ± 0.0103 / MCC@val 0.3580 ± 0.0259；结果在 `experiments/cv5/results/`，详见 `plan/RUNS.md`
- [x] **`cv5_full` 5 折（EMA+MCC）**（2026-11-09 19:03 完成）：AUROC 0.7316 ± 0.0060 / AUPR 0.6525 ± 0.0049；但 MCC@0.5 降至 0.2505、ECE 升至 0.182
- [x] **EMA / MCC 单独拆分**（2026-11-09 20:14 / 20:41）：`cv5_ema` AUROC 0.7280 / AUPR 0.6513，`cv5_mcc` AUROC 0.7208 / AUPR 0.6421；两者都没有显著增益，MCC 已移除
- [x] **修正版 EMA**（首次命中、decay 0.9、warmup 1、epoch 末更新）已完成 `cv5_ema_fix`：AUROC 0.7294 ± 0.0100 / AUPR 0.6515 ± 0.0104，排序指标仍无显著增益
- [x] **非线性指数饱和 SAT 的旧 `exp` 近似**（α_min 0.7、start 2、tau 1.0、decay 0.65）：`cv5_ema_ramp` 完成，AUROC 0.7293 / AUPR 0.6516，与固定 α=0.7 相同；不改善排序
- [ ] **精确 `2^-s` 渐进 SAT**：代码已改，待跑 `cv5_ema_halving`（α=0.850/0.775/0.7375…）

### 没做

- [ ] 噪声梯度 / legacy 对照均未跑
- [ ] 模型创新（用户 2026-11-09 说"准备好了我们再看看模型如何创新"）
- [ ] 论文文本的对应修改（另一条线，见 `D:\BIP\module4\BIB\待修问题清单_1109.md`）
- [ ] `plan/RUNS.md` 之外的 run 台账尚未建立（每次跑完必须补一行）

---

## 6. 怎么跑

```bat
:: 环境
set PY=D:\anaconda3\envs\esm2_env\python.exe

:: 主实验：5 折 × 1 seed，用完自动删 ESM 缓存
%PY% scripts\run_5fold.py --exp cv5 --epochs 50

:: EMA-SAT（默认关闭，显式 --ema 才开；硬标签 α→0.7 / start 2 / tau 1.0）
%PY% scripts\run_5fold.py --exp cv5_ema_halving --ema --ema-alpha-min 0.7 --ema-start-epoch 2 --ema-tau 1.0 --epochs 50

:: 与原 462 条测试集并表的对照线
%PY% scripts\run_legacy.py --exp legacy
```

**EMA-SAT 默认关闭**，必须显式 `--ema` 才开；MCC 自适应类权重已于 2026-11-10 移除。
EMA 会在 `train_log.csv` 留下逐 epoch 证据（`ema_active`、`ema_alpha`、`ema_beta`、`ema_mean`、`ema_std`、
`soft_pos_mean`、`soft_neg_mean`），可直接检查 teacher 是否还卡在 0.5 附近、α 是否按计划下降、β 是否上升。

单折手跑（调试用）：

```bat
%PY% scripts\train_aip.py --fold 0 --exp dbg0 --epochs 2 --cache-dir _work\fakecache
%PY% scripts\eval_aip.py --exp dbg0 --n-boot 100
```

`--cache-dir _work\fakecache` 是随机向量假缓存，只用来验证代码跑通（几十秒、CPU 即可），
**产出的指标没有任何意义**，看完就删。

产物落在 `experiments\<exp>\`，人读结果看 `results\summary.md`。

### 6.1 实验命名规则（`--exp` 必填）

`run_5fold.py` 与 `run_legacy.py` 的 `--exp` 是**必填**，不再默认 `cv5`——避免"顺手一跑"覆盖掉昨天的结果。

- 规则：小写字母/数字开头，只能含 `a-z 0-9 _ -`，2–41 字符；`--exp 'CV5 x'` 会被直接拒绝。
- **已存在的实验名会被拒绝**（`实验 't_demo' 已存在（1 个 run）`），要么换名，要么加 `--resume` 续跑剩下的折。
- `--note "..."` 写一句话说明，进 `manifest.json` 与总表；续跑时留空会自动沿用上次的说明。
- `train_aip.py` 单跑时，run 目录已存在且非空也会拒绝（要覆盖必须显式 `--overwrite`）。

命名建议（前缀表示"这一组在测什么"，后缀表示"开了什么"）：

| 名字 | 含义 |
|---|---|
| `cv5` | 5 折主实验，什么都不开 |
| `cv5_ema_fix` | EMA-SAT 修正版（固定 α=0.7/教师 β=0.3，历史对照） |
| `cv5_ema_ramp` | EMA-SAT 旧 `exp` 近似（α=0.810/0.741/…，历史） |
| `cv5_ema_halving` | EMA-SAT 精确 `2^-s` 版（α=0.850/0.775/0.7375…，待跑） |
| `cv5_ema` | 旧 EMA 实现的历史对照，不再新跑 |
| `legacy` | 原 4,224/462 划分，和外部基线并表用 |
| `dbg0` / `smoke*` | 调试，看完就删 |

### 6.2 怎么知道哪个结果是哪个

三层，随便看哪层都能对上：

1. **目录**：`experiments\<exp>\runs\fold{k}_m{seed}\`——实验名 + 折号 + seed 全在路径里。
2. **单 run 内**：`config.json` 存全部超参（含 `ema`、`split_sha256`）、`threshold.json` 存阈值与实际开关、`train_log.csv` 逐 epoch 记录。
3. **总表**：`experiments\_index.csv`（每次跑完自动登记）或直接

```bat
%PY% scripts\list_exps.py                    :: 所有实验一行一个，含 EMA 开关与主指标
%PY% scripts\list_exps.py --exp cv5_ema      :: 单个实验的明细表
```

总表长这样（冒烟数据，指标无意义）：

```
实验               EMA    ep   runs   AUROC(free)        AUPR(free)         MCC(val)       说明
t_demo_ema       开     1    1      0.5962±0.0000      0.5195±0.0000      0.2018±0.0000  冒烟：EMA 开
```

---

## 7. 环境与操作须知

> 详细版见 `plan/ENV.md`；动手前先跑
> `D:\anaconda3\envs\esm2_env\python.exe scripts\check_env.py`（会在 GPU 上真跑一次卷积）。

* **训练/推理只用 conda 环境 `esm2_env`**：
  `D:\anaconda3\envs\esm2_env\python.exe` —— Python 3.11.14、torch 2.11.0.dev+cu128、
  transformers 4.48.1，支持 RTX 5060（sm_120）。
* `D:\anaconda3\python.exe`（base，torch 2.5.1+cu121）**不支持 sm_120**，
  `torch.cuda.is_available()` 也是 True，但一跑卷积就报
  `no kernel image is available`——只能用来改 Word/Excel（它有 python-docx）。
* `run_5fold.py` / `run_legacy.py` 用 `sys.executable` 传解释器：**用哪个 python 启动，
  子进程就用哪个**，所以务必用 `%PY%`（esm2_env）启动，别用裸 `python`。
* GPU：RTX 5060 Laptop，8 GB 显存，够用（ESM 冻结 + 短序列）。
* ESM-2 已在 HF 缓存：`C:\Users\cxy\.cache\huggingface\hub\models--facebook--esm2_t33_650M_UR50D`（2.43 GB），**不需要下载**。
* 写文件：`cxyaippred\` 与 `immudualpred\` 在受限沙箱下不可写，需要提权（danger-full-access）才能写。
* 控制台是 GBK，Python 输出中文要 `-X utf8` 且 `sys.stdout.reconfigure(encoding='utf-8')`。

---

## 8. 坑（按重要性排）

1. **test 折不能再被用来做任何选择**。一旦用它调过阈值或挑过 epoch，5 折就白做了。
2. **`val` 目前一集三用**（早停 + 选阈值 + 调类权重），这是旧稿被批的点（待修清单 M01）。
   现行方案沿用，但写论文时必须说明；若要更严，需要在 train 内部再切一层。
3. **手工特征没有标准化**（`src/features.py` 里没有任何 scaler，旧稿却写了"用训练集均值方差标准化"）。
   换实现时别偷偷加，否则与旧权重不兼容。
4. **ESM 版本/层号钉死**：t33 650M、`hidden_states[-1]`、1280 维。换模型=重训。
5. **不要用 `immudualpred\src` 的代码**，那是双任务旧包，mask 逻辑会让单任务训练结果不可解释。
6. 旧的 7 个 `model9*` 目录里指标互不相同（同一"论文结果"有多个版本），不要随手拿其中一个当基线。
