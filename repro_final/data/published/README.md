# 冻结的已发布 DL 概率

这 15 个文件是**论文表里两个深度视角的原始预测**，不是权重：

| 文件 | 形状 | 来源 | 折内 AUROC |
|---|---|---|---|
| `tlora_grid_dl_fold{k}_val_probs.npy` | (656,) / (655,) | `_work/wt-drop385/grid_dl/T1/fold{k}_val_probs.npy` | 0.8212 / 0.7889 / 0.8102 / 0.7713 / 0.8243 |
| `tlora_grid_dl_fold{k}_test_probs.npy` | (1002,) | 同上 `fold{k}_test_probs.npy` | 前 817 行 5 折均值 = **0.8268** |
| `cnn_grid_A_fold{k}.npz` | `val_probs (3,n)`、`test_probs (3,1002)`、`va_idx (n,)` | `_work/wt-drop385/grid_cnn/A/fold{k}.npz` | 取 seed 0 行；817 行 5 折均值 = **0.6965** |

## 为什么是概率而不是权重

`grid_dl` 与 `grid_cnn` 两个训练脚本都只 `np.save` 了概率、json 和 log，**从未 `torch.save` 模型**。
仓库里现存的 TLoRA 权重是另一臂（完整官方池上训练的 5 折），CNN 权重是同协议重训，
两者都不是产生论文表数字的那一次实例。为了让论文表能被逐位核对，只能把当年留下的概率一并发布。

这不是"隐形的测试集泄漏"：这些概率只覆盖 **test1 的 817 行预测**和**训练行的 OOF 预测**，
不含 test1 的标签信息，`load_published_dl.py` 也不会拿它们去拟合任何东西——它只做
"把 OOF 拼回去 + 断言每折 AUROC 与当年记录一致"。

## 可审计性

`scripts/load_published_dl.py` 会：

1. 用 `StratifiedKFold(5, shuffle=True, random_state=42)` 在 train1 上**重新推出**折划分
   （TLoRA 当年就是这么分的，不是官方折），断言每折 OOF AUROC 与上表一致（容差 6e-4）；
2. CNN 直接用 `train1.csv:fold`（官方折）与 npz 里的 `va_idx` 拼 OOF，并断言 0.6965 / 0.4913 / 0.6795；
3. 输出 `out/dl_published.npz`，`fuse_eval.py --dl published` 从这里读。

想用权重而不是概率，跑 `scripts/run_dl_views.py`（路径 `--dl weights`）。
