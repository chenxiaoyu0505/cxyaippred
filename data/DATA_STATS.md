# data/ 数据核对

生成时间：2026-11-09
生成方式：`python scripts/make_aip_dataset.py`（读取 `D:\BIP\data\train2.csv` / `test.csv`）

---

## 1. 文件

| 文件 | 行数 | 列 |
|---|---|---|
| `aip_train.csv` | 4,224 | `sequence`, `label` |
| `aip_test.csv` | 462 | `sequence`, `label` |

`label`：1 = AIP 阳性，0 = AIP 阴性。

---

## 2. 分布

|  | 训练 | 测试 |
|---|---|---|
| 行数 | 4,224 | 462 |
| 唯一序列 | 4,224 | 462 |
| 阳性 / 阴性 | 1,655 / 2,569 | 185 / 277 |
| 阳性率 | 39.2% | 40.0% |
| 长度 中位数 | 15 | 15 |
| 长度 95 分位 | 20 | 26 |
| 长度 最大 | 73 | 56 |
| 含非标准氨基酸的行 | 0 | 0 |

---

## 3. 已通过的检查

- **同任务内无重复**：训练 4,224 行对应 4,224 个唯一序列，测试 462 行对应 462 个唯一序列。
- **无标签冲突**：既然同任务无重复，就不存在"同一序列两次标注相反"的情况。
- **训练/测试精确序列重叠 = 0**。

作为对照，PIP 子集（本次不用）的情况是：训练/测试共有 **109 条序列重叠**、测试集内有 **8 条重复**。也就是说原稿被质疑的"序列泄漏"主要发生在 PIP 侧。

> 附注：原双任务表里那 1,367 个"重复序列"，是同一肽在 AIP 任务和 PIP 任务各有一条记录（mask 组合 `10` 与 `01` 成对出现），并不是重复标注或标签噪声。训练集中同时有 AIP 记录和 PIP 记录的序列共 1,413 个。

---

## 4. 尚未检查（重跑之前应该补）

1. **近同源泄漏**：精确重叠为 0 不代表安全。需要对 4,224 条做序列相似性聚类（例如 identity ≥ 40% 视为同簇），检查训练/测试是否落在同一簇。
2. **长度与标签的相关性**：AIP 阳性可能偏长/偏短，若测试集长度分布不同，会影响泛化解释。
3. **与外部独立集的一致性**：目前所有数字都来自同一个 462 条测试集，没有第二套外部验证。

---

## 5. BIP-free 变体（2026-11-10）

旧双任务数据里，56 条肽在同一行同时有 `label-aip=1` 和 `label-pip=1`（46 train + 10 test），
即论文 Table 1 的 BIP。单任务 AIP 数据集 `aip_all.csv` 把这 56 条都算作 AIP 阳性。

为得到不含旧 BIP 的 AIP 基准，新增：

- 输入：`data/aip_all.csv`（4,686 行）+ `D:\BIP\data\BIPs.csv`（56 行）
- 脚本：`scripts/drop_bips.py`
- 输出：`data/aip_all_nobip.csv`（4,634 行；1,788 阳 / 2,846 阴，38.6%）
- 删除：56 条，全部为阳性（56 阳 / 0 阴）
- 补回：4 条 HD-AIP-only 阳性（`CIIHRGKPFQLEAVFEANQNTKTA`、`GEPGIAGFKGAQGPKGEPGPA`、
  `HGRGTCNYYSNSYSFWLASL`、`RIPLAVLFGIFLYMGV`），按用户 2026-11-10 要求保留
- 旧 4,630 划分：`data/splits/split_seed42_n5_nobip/folds.csv`（23,150 行）仅作历史，
  不再与当前 4,634 主表对应

注意：该口径只删除“同一行双阳”的 56 条 BIP；另有 327 条序列在原始表中以分开的
AIP 行和 PIP 行出现（序列级 AIP+/PIP+ 并集为 383 条），本变体**保留**这 327 条。
如果要做“完全不含任何 PIP 阳性序列”的严格版本，需要另建口径，不要与本文件混用。

### 5.1 BIP-free 原始 train/test

保留原 `aip_train.csv` / `aip_test.csv` 的划分，只删除其中的 BIP：

| 文件 | 行数 | 阳性 / 阴性 | 删除 BIP |
|---|---|---|---|
| `aip_train_nobip.csv` | 4,182 | 1,613 / 2,569 | 46 |
| `aip_test_nobip.csv` | 452 | 175 / 277 | 10 |

两文件并集 = `aip_all_nobip.csv`（4,634 行），训练/测试精确序列重叠为 0，
BIP 交集为 0。原始 `aip_train.csv` / `aip_test.csv` 未改动；4 条补回的 HD-AIP-only
阳性只在 train 中，因此 train 比“删 BIP 后”的 4,178 多 4 条。

### 5.2 固定原始 test + 5 折 CV

`data/splits/split_origtest_cv5_seed42_nobip/`：

- 固定 test：452 条（175 阳 / 277 阴），即 `aip_test_nobip.csv`。
- CV pool（train）：4,182 条（1,613 阳 / 2,569 阴），即 `aip_train_nobip.csv`。
- 5 折：每折 train 3,345/3,346，val 837/836；test 固定为同一份 452 条。
- 生成命令：
  `python scripts/make_holdout_cv.py --data data/aip_train_nobip.csv --test-data data/aip_test_nobip.csv --split-id split_origtest_cv5_seed42_nobip --seed 42 --folds 5`
- 5 折 test 相同，因此不能把 5 折 test 指标当作互斥测试集上的独立估计；详见 `data/FOLDS.md`。
