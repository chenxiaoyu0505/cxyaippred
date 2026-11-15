# 四视角 AIP 预测 —— 可复现发布包

本目录把项目里**当前最好的一条融合臂**（test AUROC **0.8300** / AUPR **0.6958**）
连同它的全部输入、脚本和权重一起冻结下来，任何人拿到就能把数字重跑一遍。

融合方式：**秩平均（rank average）**。每个视角先在评估队列内转成 0–1 均匀秩分，再等权平均；
秩分再用池内 OOF 上拟合的一维 logistic（Platt，严格单调）映射成概率，只为给出可解释的工作点。

```
四视角 = hand-704 (手工理化)  ┐
       + gap3-200  (间隔 3-mer)├─ 3 个 ExtraTrees 视角  ─┐
       + k4-200    (连续 4-mer)┤                        ├─ 等权秩平均 → 0.8300
       + TLoRA-ESM2 (深度学习) ┘  冻结 ESM-2 650M + TLoRA(q/k/v) + posproj 头 ─┘
```

| 臂 | 权重 | OOF AUROC | test AUROC | test AUPR |
|---|---|---|---|---|
| 3 视角（rank） | .333 × 3 | 0.8057 | 0.8174 | 0.6804 |
| **4 视角（rank，默认臂）** | .250 × 4 | 0.8229 | **0.8300** | **0.6958** |
| 3 视角（greedy，对照） | hand .556 / gap3 .222 / k4 .222 | 0.8210 | 0.8229 | 0.6933 |
| 4 视角（greedy，可部署对照） | .370 / .222 / .185 / .222 | 0.8241 | 0.8219 | 0.6904 |

四条必须同时说的话：

1. 秩平均把四视角从 0.8219 推到 **0.8300**（Δ = +0.0081，配对 bootstrap 95% CI
   **[−0.0000, +0.0157]**，p = 0.975 —— 卡在显著边缘，别说成"显著提升"）。
2. **逐折看只有 +0.0011**（4/5 折为正，fold4 为 −0.0039）。那 +0.008 里有一半来自
   "先跨折平均、再算秩分"把各视角的分布拉平，不是每个模型都变好。
3. **秩分是 cohort-level（transductive）的**：要先有整批样本才能算分位，
   **单条序列无法单独推断**。要做单条部署/推理，用 `--fuse greedy`（0.8219，固定权重）。
4. 等权秩平均给 DL 视角 25% 的固定权重（贪心只给 22.2%），所以在这个口径下
   "加 DL"是 **+0.0126**（0.8174 → 0.8300）；反过来，等权对三视角本身是 **−0.0055**
   （0.8229 → 0.8174），因为 hand 明显强于 gap3/k4。想两者兼顾用 `--fuse rankw`
   （贪心权重 × 秩分）：三视角 0.8242 / 四视角 0.8291。

---

## 1. 30 秒复现（纯 CPU，不需要 GPU，不需要 ESM-2）

```bash
pip install -r requirements.txt
python scripts/run_4view.py            # 默认 rank 融合，复现 + 与 expected.json 逐项对账
python scripts/run_4view.py --fuse greedy   # 旧口径（可部署），0.8219
python scripts/run_4view.py --fuse rankw    # 贪心权重 × 秩分，4 视角 0.8291
python scripts/run_4view.py --curve    # 额外打印弃权（risk-coverage）曲线
python scripts/abstain.py --alpha 0.10 # 三态输出：按 α 反推覆盖率（不碰 test，见 5.1）
```

期望输出（`[check] 21/21 通过` 即通过，阈值 |Δ| ≤ 2e-4）：

```
--- 4view+TLoRA (rank) ---
  权重 hand=0.250, gap3=0.250, k4=0.250, TLoRA=0.250
  OOF AUROC 0.8229
  test AUROC 0.8300 [95% CI 0.8034-0.8556]  AUPR 0.6958
  τ(OOF MCC 最优) = 0.268
  test_at_tau  Sn 0.895 Sp 0.638 Acc 0.736 MCC 0.522 错 237
  test_at_0.5  Sn 0.675 Sp 0.800 Acc 0.752 MCC 0.475 错 222
  [对照] 贪心 test AUROC 0.8219 / AUPR 0.6904
```

这一步读的是 `probs/` 与 `tlora5/` 里**已冻结的逐视角、逐折概率**，只重跑融合层。
它验证的是：我报的融合数字，能否从发布的分视角概率里原样算出来。

## 2. 目录

```
data/    table.csv(4480=池3583+test897)  folds.csv(池内5折)  test.csv(固定897)
feats/   hand_feats_amphipath.npy   4480×704 float32，行序 = data/table.csv
probs/   fold{0..4}_{hand,gap3,k4}_{val,test}.npy    三个手工视角的逐折概率
tlora5/  {0..4}_{val,test}_probs.csv  逐折概率；{0..4}_result.json 逐折指标
         results.json 五折汇总；verify.json 复现验收记录
weights/ tlora_fold{0..4}.pt   peft-delta（只含 A/B/posproj/head，约 8 MB/折）
scripts/ run_4view.py  run_ml_views.py  train_tlora5.py  infer_tlora.py
         abstain.py  tlora_model.py  peft_io.py
expected.json  对账基准
```

数据集协议：DeepAIP 原始划分 —— 训练池 **3,583**（1,365 阳 / 2,218 阴），池内 5 折；
固定 test **897**（342 阳 / 555 阴），**只在最后评估一次**。

## 3. 三个手工视角（可从原始序列重训）

每个视角：TF-IDF → TruncatedSVD(300) → StandardScaler → ExtraTrees(1500 棵,
`max_features=0.1`, `min_samples_leaf=1`, `class_weight=balanced`, 3 个种子取平均)。

```bash
python scripts/run_ml_views.py --phase feats            # gap3/k4 特征（每折单独拟合 IDF/SVD/Scaler）
python scripts/run_ml_views.py --phase probs            # 训 ET，约 11 视角×5 折的量级；只跑三视角约 20-40 min
python scripts/run_ml_views.py --phase verify           # 与 repro/probs 逐文件 max|Δp| 对账
```

`--folds 0` 可只跑一折冒烟。`hand` 视角直接用发布包里的 `feats/hand_feats_amphipath.npy`
（704 维 = 理化/组成/两亲性等，**预先算好**；从序列重建它依赖上游特征脚本，不在本包内 —— 这是
本包最大的复现缺口，见第 6 节）。

## 4. 深度学习视角（TLoRA，需要 GPU）

```bash
# 用发布权重推理（推荐先跑这条，验证权重能装回去）
python scripts/infer_tlora.py --verify
# 给自己的序列打分（5 折平均 = 主口径）
python scripts/infer_tlora.py --csv my_peptides.csv --fold all --out pred.csv
# 从零重训（约 1-2 h/折，RTX 5060 8GB 量级）
python scripts/train_tlora5.py --fold 0
```

结构（见 `scripts/tlora_model.py`）：

* 冻结 `facebook/esm2_t33_650M_UR50D`（fp16），把 33 层的 q/k/v 换成 LoRALinear（99 个模块）；
* **TLoRA**：C = 该折训练行上被适配层的输入激活协方差（只累加非 pad 残基）；
  A = SVD(`W0·C`) 的前 r=8 个右奇异向量，**A 冻结**，B 零初始化，只训 B（α=16，scale=2）；
  起点严格等价基座（脚本内有 `max|ΔW|=0`、`max|Δlogit|=0`、`B.grad≠0 且 A.grad is None` 断言）；
* 头 = `Linear(1280→4)` 逐位置投影 → `Conv1d(4,4,3)` → GELU → 掩码加权 `interpolate` 重采样到 32 槽
  → flatten(128) → `Linear(256)` → GELU → Dropout(0.3) → `Linear(1)`；
* 训练：bs=32，lr_lora 5e-4 / lr_head 1e-3，AdamW(wd=0.01)，bf16 autocast，早停按 val AUROC
  （patience 5，epoch 上限 20），seed 0。

五折单折 test AUROC 均值 0.7625±0.0168，五折概率集成 0.7854 / AUPR 0.6353。

**本机实测（发布前跑过）**：`infer_tlora.py --verify --fold all` → 五折逐折
max|Δp| ≤ 3e-08（fp32 舍入量级），逐折 AUROC 完全一致，集成 0.7854 / 0.6353。
delta 权重（只存 A/B/posproj/head，8 MB/折）能单独把发布概率装回去，不依赖任何初始化文件。

> 注：加载时有 `esm.pooler.dense.* not initialized` 的 warning，可忽略（pooler 不参与前向）。

## 5. 融合层与阈值（方法学）

* 融合：**等权秩平均**（`--fuse rank`，默认）。对每个视角 v，在评估队列内计算
  `r_v = rank(p_v)/n`（0–1 均匀），然后 `s = mean_v r_v`。
  秩变换**抹掉每个视角的概率尺度差异**，所以不需要拟合任何权重 —— 这一步零参数、零选择。
  队列 = 池内 OOF 3,583 条（算 τ）与 test 897 条（评估）各自独立算秩，互不混用。
* 概率映射：用池内 OOF 上的 `(s, y)` 拟合一维 logistic（Platt），把 `s` 单调映射成 `p`，
  **只影响工作点与弃权曲线，不影响 AUROC/AUPR**（严格单调，无并列）。
* 备选口径：`--fuse rankw`（贪心权重 × 秩分）、`--fuse greedy`（Caruana 27 轮贪心，可部署）、
  `--fuse mean`（等权概率平均）。同一份冻结概率上：greedy 0.8219 / mean 0.8226 /
  rank 0.8300 / rankw 0.8291。
* 工作点 τ：在池内 OOF 上取 **MCC 最大**的值。本臂 τ = **0.268**（Platt 后的概率尺度，
  与贪心口径的 0.396 **不可混用**）。
* 报 Sn/Sp 必须同时报 τ：τ=0.268 时 Sn .895 / Sp .638，τ=0.5 时 Sn .675 / Sp .800。
* 弃权曲线由 `scripts/abstain.py` **直接算**（不训练），见 5.1 节。
  注意：按 `|p−τ|` 还是 `|p−0.5|` 排序、以及用 0.5 还是 τ 判定，都会改变曲线上的数字，
  项目历史文档里那套（100% → .736 / 70% → .828）是**别的臂 + 别的口径**，不要与本节的表并列。

## 5.1 弃权 / 拒答（`scripts/abstain.py`）

三态输出 = 判阳 / 判阴 / 拒答。拒答分 `s = |p − τ|`（越小越可疑），
`p ≥ τ` 判阳、`s < t_s` 拒答，τ 来自池内 OOF（本包 rank 口径 0.268）。

**上次遗留的软肋是「保留 70% 这个 p 是看着 test 曲线挑的」。** 现在给了两条不碰 test 的路：

```bash
python scripts/abstain.py --curve                      # 整条曲线 + AURC（不需要挑任何 p）
python scripts/abstain.py --alpha 0.10                 # 先声明能容忍的错误率，覆盖率由它反推
python scripts/abstain.py --alpha 0.10 --nested        # 留一折，看覆盖率跨折抖多少
python scripts/abstain.py --fuse greedy --curve        # 用可部署的贪心口径再算一遍
```

**(a) 整条曲线 + AURC**（无 p 依赖的报告口径，越低越好；rank 口径）

| 覆盖率 | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|
| OOF 错误率 | .2783 | .2617 | .2519 | .2476 | .2530 | .2718 |
| test 错误率 | .2642 | .2466 | .2340 | .2325 | .2509 | .2701 |

AURC：OOF **.2495**（随机弃权 .2783）、test **.2464**（随机弃权 .2642）。

> ⚠️ 秩平均换了置信度的排序：同一模块在 `--fuse greedy` 下 AURC 是 OOF **.2126** /
> test **.2232**（更好）。也就是说 **rank 提排序、greedy 提置信度**；如果产品卖点是
> "三态拒答"，请用 greedy 口径报 AURC，别用 rank 的数字。

**(b) 按 α 定阈值 —— 结论是残酷的**

| 目标错误率 α | t_s | OOF 覆盖率 | OOF 实际风险 | test 覆盖率 | test 实际风险 |
|---|---|---|---|---|---|
| 0.05 | 0.328 | **1.3%** | .0444 | 0.3% | .3333（n=3，无意义）|
| 0.10 | 0.300 | **3.4%** | .0992 | 1.6% | .1429 |
| 0.15 | 0.289 | **5.2%** | .1497 | 3.2% | .2069 |

也就是说：**想要「回答部分的错误率 ≤ 10%，就只剩 3.4% 的样本能回答」。**
所以当时那个"保留 70%"在任何正当的 α 下都不成立 —— 它不是工作点，只是一张
"按置信度排序后的分数表"。误差信息量就这么多，置信度只能买到这么小的低风险核。

**(c) 稳定性**：留一折（阈值来自其余 4 折）时，同一条 α=0.10 规则在五折上给出
**1.9%–6.2%** 的覆盖率（均值 3.9%）。数量级稳定，绝对值抖 3 倍 —— 这本身也说明
这个低风险核太薄，不适合作论文里的一个"点"。

⚠️ 边界：这是**经验分位数**，不是有有限样本保证的 split-conformal。
「错误率 ≤ α」在 OOF 上成立、在 test 上只**验收一次**，不要写成数学保证。

## 6. 已知问题与局限（读论文数字前请先看）

1. **弃权工作点曾经是泄漏的，现在给了修法，但坦白说修完之后不剩什么**：
   文档里"保留 70%"那个覆盖率是**在 test 上看着曲线挑的**。本包改成
   "先声明 α（可容忍错误率），在池内 OOF 上反推覆盖率"——不碰 test，但结果是
   α=0.10 只剩 **3.4%** 覆盖率（第 5.1 节）。所以：**要么报整条曲线 + AURC，要么承认
   低风险核只有几个百分点**；「保留 70% 时 acc 0.82」不能再当成一个可辩护的工作点。
   另外这是经验分位数，不是严格的 conformal 保证。
2. **融合层数字对口径敏感，且 rank 的增益有一半不是"每个模型都变好"**：
   贪心只看 OOF，OOF 一路 0.8210→0.8229→0.8241 单调升、test 0.8229→0.8221→0.8219 单调降；
   换成等权秩平均后 test 变成 0.8174→0.8300（DL 看起来是 +0.0126），但**逐折对比只有 +0.0011**。
   三个数字（greedy / rank / rankw 下的三视角与四视角）都写在表里，报论文时要写清用的是哪个。
   另外秩平均是 cohort-level 的：**不能单条推断**。
3. **TLoRA 的增益不稳健**：五折里 fold2 独大（+0.070），fold4 为负，AUPR 打平、CI 含 0；
   相对 LoRA 的配对 bootstrap ΔAUROC = −0.0001 [−0.0031, +0.0029]，**完全等价**。
4. **头部的卷积发生在掩码之前**：pad 位置的隐状态会被卷进序列尾部，所以结果对 batch 组成敏感
   （fold0：bs=16 → 0.7365，bs=32 → 0.7445）。本包为复现原数字**故意保留**该实现，
   推理请固定 `--bs 32`。修法：进 posproj 之前先 `h = h * rmask.unsqueeze(-1)`。
5. **hand-704 特征矩阵是预计算产物**，本包不含从序列重建它的完整链路；
   三个手工视角的 ET 模型也未序列化，只能靠固定种子重训（`run_ml_views.py`）。
6. **视角级概率用的是 3 seed ET**（项目里另有 10 seed 的 ET904 口径，数字不同，不要混比）。

## 7. 环境

对照实现：`numpy 1.26.0 / pandas 2.1.0 / scikit-learn 1.5.0`，
TLoRA 部分 `torch 2.11.0.dev+cu128 / transformers 4.48.1`，单卡 RTX 5060 8GB。
融合层（第 1 节）在这些版本上是**确定性**的；换 sklearn 大版本后 ET 的平局打破可能微移，
届时 `expected.json` 的 2e-4 容差可能不够，可放宽到 2e-3。

TLoRA 需要能访问 HuggingFace 下载 `facebook/esm2_t33_650M_UR50D`（2.4 GB）；
离线环境加 `--hf-offline` 并预置缓存。
