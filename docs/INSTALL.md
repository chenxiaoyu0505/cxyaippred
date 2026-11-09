# 安装与跨平台运行

三条硬性前提：**Python ≥ 3.10**、**约 4 GB 磁盘**、**能访问 huggingface.co**（首次下载 ESM-2 权重，2.43 GB）。
之后所有实验都可离线跑。

---

## 1. 十分钟装好

### Windows + NVIDIA（本机已验证）

```bat
conda create -n cxyaippred python=3.11 -y
conda activate cxyaippred
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
python scripts\check_env.py
```

RTX 50 系（compute capability 12.0）**必须**用 cu128 及以上的 wheel；官方 stable 的 cu121 能 import 但一跑卷积就报
`no kernel image is available for execution on the device`。

### macOS（Apple Silicon / Intel）

```bash
conda create -n cxyaippred python=3.11 -y
conda activate cxyaippred
pip install torch                 # 官方 wheel，自带 MPS 支持
pip install -r requirements.txt
python scripts/check_env.py
```

### Linux + NVIDIA

```bash
conda create -n cxyaippred python=3.11 -y
conda activate cxyaippred
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128   # 或按驱动选 cu126/cu124
pip install -r requirements.txt
python scripts/check_env.py
```

也可以用 `conda env create -f environment.yml`，之后同样要补装 torch（见文件顶部注释）。

---

## 2. 设备选择：`--device auto`

所有脚本的 `--device` 默认 `auto`，按 **CUDA → MPS → CPU** 顺序挑第一个可用的，`src/utils.py::pick_device` 实现。

```bash
python scripts/train_aip.py --fold 0 --exp dbg0 --device cpu   # 强制 CPU
python scripts/train_aip.py --fold 0 --exp dbg0 --device mps   # 强制 Apple GPU
```

`check_env.py` 不看 `torch.cuda.is_available()`（那个在坏驱动上也会返回 True），而是**真跑一次 conv1d**，
所以能提前发现"看起来有 GPU、其实跑不了"的情况。

---

## 3. 三个平台各能跑到什么程度

| | Windows + RTX 5060 8 GB | Apple Silicon (M 系列) | 纯 CPU |
|---|---|---|---|
| 建 ESM 缓存（4,686 条） | 5–10 min | 20–40 min（MPS） | 1–2 h |
| 单折训练（50 epoch，早停） | 2–4 min | 10–20 min | 1–2 h |
| 完整 5 折 | 约 20–30 min | 2–3 h | 6–10 h |
| 显存/内存需求 | 峰值约 4–5 GB | 统一内存 ≥ 16 GB 建议 | ≥ 16 GB |

上表是量级估计；真实耗时取决于序列长度分布与早停轮数。

**Apple Silicon 的两个注意点**

1. MPS 对个别算子支持不全（尤其某些 conv/batchnorm 的 backward）。若报
   `NotImplementedError: The operator 'xxx' is not currently implemented for the MPS device`，
   加环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1` 让未实现的算子回退 CPU，或直接 `--device cpu`。
2. 本仓库全程 **float32** 计算（ESM 前向 + 卷积 + 头），缓存文件虽以 float16 落盘，读取时会转 float32，
   所以 MPS 上不存在 fp16 精度问题。

**Intel Mac** 没有 MPS，会落到 CPU，可用但慢（见上表 CPU 列）。

---

## 4. 已验证环境（Windows，出论文数字的那台）

| 组件 | 版本 |
|---|---|
| OS | Windows 11 |
| Python | 3.11.14 |
| torch | 2.11.0.dev20260107+**cu128** |
| transformers / tokenizers | 4.48.1 / 0.21.4 |
| numpy / pandas / scipy / scikit-learn | 1.26.0 / 2.1.0 / 1.16.3 / 1.5.0 |
| safetensors / huggingface-hub | 0.7.0 / 0.36.0 |
| GPU | NVIDIA RTX 5060 Laptop，8 GB，capability (12, 0) |

完整清单见 `plan/ENV.md`。

---

## 5. 权重放哪

`facebook/esm2_t33_650M_UR50D` 走 Hugging Face 默认缓存：

```
~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/
```

- 已有该缓存 → 直接复用，不联网；
- 没有 → 首次 `from_pretrained` 自动下载 2.43 GB；
- 想换位置：设 `HF_HOME` 或 `HF_HUB_CACHE`。

模型权重**不在本仓库里**，也不需要放仓库里——它是公开预训练权重。

---

## 6. 跑之前先自检

```bash
python scripts/preflight.py     # 数据/划分/缓存/权重/磁盘，全部 OK 才开跑
python scripts/run_5fold.py --exp cv5 --note "5折主实验"
```

`preflight.py` 退出码 0 表示可以开跑；任何硬性阻塞（GPU 卷积失败、划分缺失、磁盘不足）会返回 1。
