# 四视角 AIP 预测 —— 可复现发布包

本目录把项目里**当前最好的一条融合臂**（test AUROC **0.8219** / AUPR **0.6904**）
连同它的全部输入、权重和脚本一起冻结下来，任何人拿到就能把数字重跑一遍。

```
四视角 = hand-704 (手工理化)  ┐
       + gap3-200  (间隔 3-mer)├─ 3 个 ExtraTrees 视角  ─┐
       + k4-200    (连续 4-mer)┤                        ├─ Caruana 贪心加权 → 0.8219
       + TLoRA-ESM2 (深度学习) ┘  冻结 ESM-2 650M + TLoRA(q/k/v) + posproj 头 ─┘
```

| 臂 | 权重 | OOF AUROC | test AUROC | test AUPR |
|---|---|---|---|---|
| 3 视角（无深度学习） | hand .556 / gap3 .222 / k4 .222 | 0.8210 | **0.8229** | **0.6933** |
| 4 视角（+ TLoRA） | hand .370 / gap3 .222 / k4 .185 / TLoRA .222 | 0.8241 | 0.8219 | 0.6904 |

**先把丑话说在前面**：加了深度学习视角之后 test 是**降**了 0.0010（0.8229 → 0.8219），
OOF 才是升的。这条臂的意义是"冻结 pLM + TLoRA 能与手工特征**等价**地融进同一框架"，
不是"提分"。别把它写成提升。

---

## 1. 30 秒复现（纯 CPU，不需要 GPU，不需要 ESM-2）

```bash
pip install -r requirements.txt
python scripts/run_4view.py            # 复现 + 与 expected.json 逐项对账
python scripts/run_4view.py --curve    # 额外打印弃权（risk-coverage）曲线
python scripts/abstain.py --alpha 0.10 # 三态输出：按 α 反推覆盖率（不碰 test，见 5.1）
```

期望输出（`[check] 21/21 通过` 即通过，阈值 |Δ| ≤ 2e-4）：

```
--- 4view+TLoRA ---
  权重 hand=0.370, gap3=0.222, k4=0.185, TLoRA=0.222
  OOF AUROC 0.8241
  test AUROC 0.8219 [95% CI 0.7955-0.8492]  AUPR 0.6904
  τ(OOF MCC 最优) = 0.396
  test_at_tau  Sn 0.860 Sp 0.665 Acc 0.739 MCC 0.511 错 234
  test_at_0.5  Sn 0.620 Sp 0.807 Acc 0.736 MCC 0.433 错 237
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

* 权重：**Caruana 带放回贪心，27 轮**，每轮挑使**池内 OOF AUROC** 最大的视角，
  权重 = 被挑次数 / 27。**test 不参与任何选择**。
* 工作点 τ：在池内 OOF 上取 **MCC 最大**的值。本臂 τ = **0.396**。
  τ 只属于这个模型 —— 换臂必须重算（旧四视角 LoRA 版的 0.398、新划分四视角的 0.465 都不能套用）。
* 报 Sn/Sp 必须同时报 τ：τ=0.396 时 Sn .860 / Sp .665，τ=0.5 时 Sn .620 / Sp .807。
* 弃权曲线由 `scripts/abstain.py` **直接算**（不训练、不拟合任何模型），见 5.1 节。
  注意：按 `|p−0.5|` 还是 `|p−τ|` 排序、以及用 0.5 还是 τ 判定，都会改变曲线上的数字，
  项目历史文档里那套（100% → .736 / 70% → .828）是**别的臂 + 别的口径**，不要与本节的表并列。

## 5.1 弃权 / 拒答（`scripts/abstain.py`）

三态输出 = 判阳 / 判阴 / 拒答。拒答分 `s = |p − τ|`（越小越可疑），
`p ≥ τ` 判阳、`s < t_s` 拒答，τ 来自池内 OOF（0.396）。

**上次遗留的软肋是「保留 70% 这个 p 是看着 test 曲线挑的」。** 现在给了两条不碰 test 的路：

```bash
python scripts/abstain.py --curve                      # 整条曲线 + AURC（不需要挑任何 p）
python scripts/abstain.py --alpha 0.10                 # 先声明能容忍的错误率，覆盖率由它反推
python scripts/abstain.py --alpha 0.10 --nested        # 留一折，看覆盖率跨折抖多少
```

**(a) 整条曲线 + AURC**（无 p 依赖的报告口径，越低越好）

| 覆盖率 | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|
| OOF 错误率 | .2674 | .2484 | .2334 | .2185 | .2191 | .2266 |
| test 错误率 | .2609 | .2354 | .2270 | .2229 | .2268 | .2277 |

AURC：OOF **.2126**（随机弃权 .2674）、test **.2232**（随机弃权 .2609）。

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
2. **融合层在过拟合 OOF**：贪心只看 OOF，OOF 一路 0.8210→0.8229→0.8241 单调升，
   对应 test 0.8229→0.8221→0.8219 单调降。四视角比三视角在 test 上更差。
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
