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
         tlora_model.py  peft_io.py
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
* 弃权曲线（`--curve`）按 `|p−0.5|` 排序保留：保留 100% → acc .736 / AUROC .822；
  70% → acc .828 / AUROC .869；50% → acc .888 / AUROC .879。**曲线本身是直接算的，不训练。**

## 6. 已知问题与局限（读论文数字前请先看）

1. **弃权工作点是一个未解决的软肋**：文档里"保留 70%"那个覆盖率是**在 test 上看着曲线挑的**。
   正确做法是在池内做 split-conformal，给定错误率上限 α 反推覆盖率。本包只提供曲线，
   **不提供**一个"凭什么取 70%"的正当化 —— 请勿把 70% 当成结论。
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
