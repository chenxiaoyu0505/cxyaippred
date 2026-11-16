# 定稿模型：5 视角 rank 融合（train1 / test1）

> 2026-11-16 决定：**以 `(train1, test1)` 为主口径，5 视角 rank 融合为主模型**。
> 本文是该模型与数据的唯一真源；与 `SPLIT_GRID.md` 冲突时以本文为准。

## 1. 数据口径

| 集合 | 行数 | 阳性 | sha256（前16） | 来源 |
|---|---|---|---|---|
| **train1** | 3,278 | 1,060 | `c52effd6214259ff` | train0 去掉 305 条 AIP+/PIP+ 冲突阳性 |
| **test1** | 817 | 262 | `7f5fd15a11210ba2` | test0 去掉 80 条同类冲突阳性 |
| train0（旧主口径） | 3,583 | 1,365 | `0ebda3347f86a030` | DeepAIP 官方训练集 |
| test0（旧主口径） | 897 | 342 | `e26e2918b2cf2442` | DeepAIP 官方测试集 |
| train2 / test2（附录） | 4,018 / 1,002 | 1,060 / 262 | `3ef0576b…` / `949b97a5…` | train1/test1 + 925 条 PIP+ 当阴性 |

**为什么去掉 385 条**：这 385 条在 DeepAIP 里标为 AIP 阳性，但同时出现在促炎肽基准 ProIn-Fuse 的阳性集里（358/385，93.0%），且 383/385（99.5%）在更上游的 AIPpred(2018) 里就已标阳性。两条证据线指向"双功能肽被单一功能标签收纳"。详见 `_work/wt-pip385/REPORT_pip385.md`。

**取舍的代价必须写明**：test1 是 897 的**子集**（817），不是官方留出集；剔除的 80 条恰好是"组成最像促炎肽"的阳性。因此 test1 上的数字**不能**与文献里 DeepAIP 官方 test 的结果直接比。

### 1.1 剔除后 train1 / test1 的一致性检验（应对"改数据"质疑）

| 检验项 | train1 | test1 | 结论 |
|---|---|---|---|
| 剔除比例（全体） | 305/3,583 = **8.51%** | 80/897 = **8.92%** | 几乎相同（比 0.95） |
| 剔除比例（仅阳性） | 305/1,365 = **22.34%** | 80/342 = **23.39%** | 几乎相同（比 0.96） |
| 阳性率 | 1,060/3,278 = 32.34% | 262/817 = 32.07% | 差 0.27 pt |
| 长度分布 | 均值 16.25 / 中位 15 | 均值 16.35 / 中位 15 | KS p = **0.916**；卡方 p = **0.481** |
| 阳性长度分布 | — | — | KS p = 0.852 |
| 阴性长度分布 | — | — | KS p = 0.997 |
| 净电荷/10aa | 0.113 ± 1.28 | 0.066 ± 1.25 | SMD = **+0.036** |
| GRAVY | −0.313 ± 0.83 | −0.350 ± 0.81 | SMD = **+0.045** |

逐长度箱阳性率（±95% CI）：

| 长度箱 | train1 | test1 | 差 |
|---|---|---|---|
| 11–14 | 0.670 ± 0.067 (n=191) | 0.605 ± 0.146 (n=43) | −0.066 |
| 15 | 0.170 ± 0.016 (n=2,078) | 0.181 ± 0.033 (n=513) | +0.011 |
| 16–18 | 0.715 ± 0.051 (n=305) | 0.632 ± 0.115 (n=68) | −0.082 |
| ≥19 | 0.513 ± 0.037 (n=704) | 0.518 ± 0.070 (n=193) | +0.005 |

所有箱内差的 95% CI 均重叠；SMD 全部 < 0.1（可忽略）。**结论：剔除是按原 80/20 比例均匀发生的，两个集合的分布同构，不构成"一边删很多、一边删很少"的问题。**

论文写法建议（一句话）：*"The 385 peptides with conflicting AIP/PIP annotations were removed from both training (305 of 3,583, 8.5%) and test (80 of 897, 8.9%) sets in their original proportions; the resulting splits remain distributionally matched (length KS p = 0.92; standardized mean differences for net charge and hydrophobicity both < 0.05)."*

仍需一并披露的两点：①官方数据自带 12 对近似重复（其中 2 对跨标签，见 `_work/grid/near_duplicates.md`）；②test1 的阳性率由 38.1% 降到 32.1%，因此**不要再与文献中 DeepAIP 官方 test 的指标直接比较**。

## 2. 模型

5 个视角，全部只用序列（+ 手工特征），共享同一套官方折：

| # | 视角 | 输入 | 模型 | 单视角 AUROC (test1) |
|---|---|---|---|---|
| 1 | hand704 | 704 维手工特征（AAC/GAAC/DPC/CKSAAGP/PAAC/CTD/理化/两亲） | ExtraTrees(1500, max_features=.1, msl=1, balanced) | 0.7825 |
| 2 | gap3 | spaced 3-mer 词袋（`(0,1,2,3)/(0,1,3,2)/(0,2,3,1)`，窗口 4，插 X） | TF-IDF → SVD(300) → Scaler → ET(同上) | 0.7578 |
| 3 | k4 | 连续 4-mer 词袋 | TF-IDF → SVD(300) → Scaler → ET(同上) | 0.7702 |
| 4 | TLoRA+posproj | ESM-2 t33 650M 第 33 层 + LoRA(qkv, r=8, α=16) + 位置投影(1280→4, conv k=3, 自适应 32) | 8 epoch, bs32, lr 5e-4/1e-3, patience 5 | **0.8268** |
| 5 | 小 CNN | one-hot(20) 从头训 | Conv1d(20→128,k3)+BN+GELU+Conv1d(128→128,k5)+均值/最大池化+MLP，34,689 参数 | 0.6965 |

**融合**：每视角 test 概率先转百分位秩（`rankdata/n`），按权重加权平均，再用 OOF 上拟合的 Platt（`LogisticRegression(C=1e6)`）映射回概率。AUROC 不受影响，只是给阈值用。

**权重**（OOF 秩空间 Caruana 27 轮贪心）：
`hand .2222 / gap3 .1481 / k4 .1852 / TLoRA .3333 / CNN .1111`

**阈值 τ**：OOF 上 MCC 最优，test1 上不调。

## 3. 结果（test1, n=817, 262 阳）

| 模型 | AUROC | 相对 3 视角 |
|---|---|---|
| 3 视角（hand+gap3+k4） | 0.8135 | — |
| 4 视角（+TLoRA） | 0.8393 | **+0.0259 [.0182, .0335]**（显著） |
| **5 视角（+CNN）** | **0.8398** | +0.0005 [−.0037, .0047]（噪声） |
| 5 视角等权（对照） | 0.8275 | — |

- **主要增益来自 TLoRA（+0.026，CI 排除 0）**；小 CNN 贡献为 0（但它是唯一"从头训"的对照臂，保留了论文里"预训练值多少分"的证据）。
- 四种融合对照（同一批概率、同一 test1）：`fixed_publish` 0.8219 / `rank_equal` 0.8174 / `rank_greedy` 0.8135（三视角）；见 `_work/wt-drop385/grid_out/`。
- CNN 只有单 seed 权重（`ext_cnn/weights_A.pt`），其波动约 ±0.03；5 视角数字的稳定区间应按 ±0.005 理解。

## 4. 已知边界（必须写进论文局限）

1. **长度捷径未解决**：训练池阴性 77% 挤在 15 aa；test1 同样继承该分布。等长口径（长度条件 AUROC）会明显低于 raw。
2. **跨来源失效**：两个外部集上 5 视角融合均为随机水平 —— ext193（train1 训练）AUROC 0.488、新 261 集（258 条清洗后）0.389；后者连"亲水性"单规则（0.601）都追不上。
3. **test1 = 官方 test 的子集**，n=817/262，AUROC 的 95% CI 半宽约 ±0.03。
4. **train2（加 PIP 阴性）是净负**：干净子集 −0.013（ML）/ −0.058（DL），不采用，只作负结果。

## 5. 复现

```bash
# 数据（六个集合，含 sha 清单）
python _work/wt-drop385/scripts/grid_splits.py     # -> _work/grid/{train0,1,2,test0,1,2}.csv + manifest.json
# 三视角 + 融合（逐折表示拟合，官方 folds.csv）
python _work/wt-drop385/scripts/eval_grid.py --train train1 --test test1
# 5 视角融合（TLoRA/CNN 已有权重）
python _work/wt-drop385/scripts/...                # 见 _work/wt-drop385/view5_t1/
# 外部集评测
python _work/ext261/run_ext261.py --table-only
```

## 6. 与旧结论的关系

- `train0/test0` 的数字（三视角 0.8174、四视角 0.8219、发布包 repro-4view 0.8300/0.8219）**保留为兼容性附表**，不再作主表。
- `plan/SPLIT_GRID.md` 里的"主表用 train0/test0"一条**已被本文取代**。
- GitHub `repro-4view` 分支的发布包对应 train0 口径，未更新；若要对外发布，应重新生成 train1 版本。
