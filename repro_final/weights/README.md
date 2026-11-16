# 权重说明

全部 11 个文件共 60.6 MB，直接提交在仓库里（没有 Git LFS）。运行时**只读不训**：
`scripts/run_dl_views.py` 加载它们做推理，不需要 GPU 训练，也不需要原始 `_work/` 目录。

| 文件 | 大小 | sha256(前16) | 内容 |
|---|---|---|---|
| `tlora_fold0..4.pt` | 7.95 MB ×5 | `a2cc2d9edb1e6623` / `6e25a3b4a5b3b29e` / `40ce8c8da55e6955` / `a4363b8c975258f9` / `f5dcc24161052f6c` | 5 折 TLoRA **增量**（PEFT delta） |
| `tlora_init_fold0..4.npz` | 4.04 MB ×5 | `8b348276f22acfda` / `c9bda2ae59e35a2d` / `8058eea019bb2d95` / `8b2b68d6bcb960fe` / `309d841648de43f4` | 每折的 SVD 初始化（A 的起点） |
| `cnn_folds.pt` | 0.69 MB | `4243687c72adda54` | 5 折小 CNN 的 `state_dict`（键 `fold0..fold4`） |

---

## 1. TLoRA：`tlora_fold{k}.pt`

### 底座

`facebook/esm2_t33_650M_UR50D`（ESM-2 t33，650M 参数）。底座**不在**这些文件里——README 里写明
要从 HuggingFace 下载一次（约 2.6 GB），`run_dl_views.py` 会现场加载 fp16 底座再把增量叠上去。

### 结构

```
ESM-2 t33 (frozen, fp16)
  └─ 33 层 × {q, k, v} = 99 个 nn.Linear  ← 每一个都换成 LoRALinear(rank=8, alpha=16)
       W = W0 + (B @ A) * (alpha / r)      # W0 frozen；A frozen；只训 B
  └─ posproj-adaptive 头
       Linear(1280 → 4) → Conv1d(4,4,k=3) → GELU → 掩码加权的 F.interpolate 到 32 槽
       → flatten(128) → Linear(256) → GELU → Dropout(0.3) → Linear(1)
```

"TLoRA" 的 T 指 **T**runcated-SVD 初始化：`A` 取 `W0 @ C` 的前 `r` 个右奇异向量，`C` 是该折训练行上的
输入激活协方差（`n_tokens = 52,511`）。这解释了为什么每折还有一个 `*_init.npz`。

### 文件格式

```python
{"format": "peft-delta-v1",
 "delta":  {name: tensor, ...},   # 206 个张量：99×A/99×B + posproj/posconv/head 8 个
 "meta":   {"fold", "epoch", "val_auroc", "val_aupr", "variant", "base_model", "format"}}
```

只存可训练张量，所以一个折只有 7.95 MB 而不是 2.6 GB。加载方式就是
`model.load_state_dict(delta, strict=False)`，再断言"没有任何 delta 键没被消费"
（`scripts/peft_io.py:load_delta`）。`A`/`B` 的形状是 `(8,1280)` / `(1280,8)`。

### `tlora_init_fold{k}.npz` 的定位（重要）

键：`A (99,8,1280)`、`S (99,8)`、`names (99,)`（`layer{i}.{q,k,v}`）、`r`、`n_tokens`、`fold`、
`Cdiag (33,1280)`、`uid`、`mean_diag`、`cov_sec`、`svd_sec`。

它记录的是**训练开始前**的初始化，用于复现"这一折是怎么造出来的"。**推理不需要它**：
`*_best.pt` 里的 `A.weight` 与这里的 `A` 逐元素相同（实测 `max|Δ| = 0`，`run_dl_views.py`
每次都会打印这个核对结果），所以 `base + delta` 就足以重建模型。之所以仍然一起发布，
是因为任务要求保留构造链，并且 `Cdiag`/`S` 是解释 TLoRA 与随机初始化差别的唯一凭证。

## 2. CNN：`cnn_folds.pt`

```python
{"fold0": state_dict, ..., "fold4": state_dict}
```

每折 34,689 参数：`Embedding(22,64,padding_idx=0)` → `Conv1d(64,64,k=3)` + BatchNorm + GELU
→ `Conv1d(64,64,k=5)` + BatchNorm + GELU → 掩码 mean ++ 掩码 max → `Dropout(0.2)` → `Linear(128,1)`。
键名与 `scripts/run_dl_views.py:class CNN` 完全一致（`emb.weight, c1.*, b1.*, c2.*, b2.*, head.*`），
`run_dl_views.py` 会断言顶层键恰好是 `["fold0".."fold4"]`。输入编码见同文件的 `onehot()`
（20 种标准氨基酸 → 2..21，0 = pad，1 = 非标准残基，固定长 40）。

### ⚠️ 这一份是"同协议重训"

发布表里的 CNN 单视角是 0.6965 / 0.4913，但**产生那组数字的模型权重当年没有保存**
（`grid_cnn.py` 只 `np.savez_compressed` 了 val/test 概率和 `va_idx`，没有 `torch.save`）。
`cnn_folds.pt` 来自 `ext_cnn/weights_A.pt`，是用**同一协议**（同架构、同超参、同 5 折、
`torch.manual_seed(0)`）重训的一份，`ext_cnn/results_ext.json` 自己就记了
`test1_auroc_ref 0.6965 → now 0.7145`（Δ = +0.0136），并注明差异即初始化随机性量级。

### ⚠️ 这一份 TLoRA 也不是 0.8268 那一臂

`tlora_fold{k}.pt` 来自 `_work/wt-tlora5/`，是在**完整官方池**（train0 3,583 / test0 897，
每折 val 717、test 897）上训练的 5 折 TLoRA；而论文表的 0.8268 来自
`_work/wt-drop385/grid_dl/T1`（train1 上 `StratifiedKFold(5, shuffle=True, random_state=42)`，
每折 val 656/655）。`grid_dl` **同样没有保存任何 checkpoint**（只存了 json / 概率 / log）。
两个模型对 test1 的预测在折内相关 0.997，但 AUROC 差 ~0.03（0.7957 vs 0.8268），
并且 TLoRA 的推理对 batch 组成敏感（头里 conv 在 mask 之前），换 batch 顺序还会有 ~0.01 抖动。

结论：**这份权重能跑通整条链路、能复现它自己的 0.7957，但它不是论文表那一臂**。
论文表那一臂只剩概率，见 `data/published/`。

## 3. 两条路，各自都被核对

| 路径 | TLoRA | CNN | 5 视角融合 | 命令 |
|---|---|---|---|---|
| `published`（默认） | 0.8268 | 0.6965 | **0.8398** | `fuse_eval.py --dl published` |
| `weights` | 0.7957 | 0.7145 | 0.8283 | `fuse_eval.py --dl weights` |

`run_all.py` 两条都跑、两条都写进 `expected.json`；前者与论文表逐位一致（27/27 PASS），
后者是"用现存权重重新推理"的真实数字（27/27 PASS，容差放宽到 2e-3，因为 bf16 的 GPU kernel
不同设备不保证逐位相同）。3 视角与 4 视角融合在两条路径下分别是 0.8134 / 0.8272（weights）
与 0.8135 / 0.8393（published）。

## 4. 校验

```bash
cd repro_final/weights
sha256sum *.pt *.npz      # Windows: Get-FileHash * -Algorithm SHA256
```

`data/manifest.json` 只覆盖 `data/` 下的 CSV（数据是否被改动），权重完整性以上表哈希为准。
