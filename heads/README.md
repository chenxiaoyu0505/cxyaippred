# heads/ —— 5 折的 AIP 头权重（占位）

**当前为空文件夹，是有意的。** 等 5 折训练跑完再往里放。

## 应该放什么

每个 fold 一个文件，约 **80 MB**：

```
heads\
  fold0_seed42.pt
  fold1_seed42.pt
  fold2_seed42.pt
  fold3_seed42.pt
  fold4_seed42.pt
```

**不要放 ESM-2**（2.43 GB）。ESM-2 是冻结的、五折共用同一份，放在 `models/` 或直接用 HuggingFace 缓存。

## 文件里装了什么

```
{'format': 'cxyaippred-aip-head-v1',
 'meta': {... esm_name, hand_dim, hidden, threshold, fold, seed, train_sha256 ...},
 'state': 20.95M 参数的 ResCNN + MLP 头}
```

生成方式见 `src/checkpoint.py` 的 `save_head()`；读取用 `load_head()`。

## 旧的 2.69 GB checkpoint 怎么办

不用复制。需要时用一条命令把里面的 AIP 头抽成 80 MB：

```
python scripts/convert_legacy_head.py --which mode2
```

（抽出来的只用于初始化或对照，不是 5 折的实验结果。）
