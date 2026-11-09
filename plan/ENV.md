# 环境（Environment）

> 一句话：**本项目只用 conda 环境 `esm2_env`。**
> 解释器全路径：`D:\anaconda3\envs\esm2_env\python.exe`

## 1. 为什么非要指定这个环境

本机有两个 Python，长得很像，但只有一个能在 RTX 5060 上跑：

| 解释器 | torch | CUDA | sm_120（RTX 5060） | 能不能训练 |
|---|---|---|---|---|
| `D:\anaconda3\envs\esm2_env\python.exe` | **2.11.0.dev20260107+cu128** | 12.8 | ✅ | **能** |
| `D:\anaconda3\python.exe` | 2.5.1+cu121 | 12.1 | ❌ | 不能 |

坑在于：base 环境里 `torch.cuda.is_available()` **返回 True**，看起来一切正常，
直到第一次卷积才炸：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

所以判据不是 `is_available()`，而是"在 GPU 上真跑一次卷积"——
`scripts/check_env.py` 做的就是这件事。

## 2. 动手前的 10 秒自检

```bat
D:\anaconda3\envs\esm2_env\python.exe scripts\check_env.py
```

- 退出码 0 + `结论：环境可用（esm2_env）` → 可以开始跑
- 退出码 1 → 按提示换解释器，别硬跑

## 3. 实测版本（2026-11-09）

| 包 | 版本 |
|---|---|
| Python | 3.11.14 |
| torch | 2.11.0.dev20260107+cu128 |
| transformers | 4.48.1 |
| tokenizers | 0.21.4 |
| numpy | 1.26.0 |
| pandas | 2.1.0 |
| scikit-learn | 1.5.0 |
| scipy | 1.16.3 |
| biopython | 1.86 |
| GPU | NVIDIA GeForce RTX 5060 Laptop，8 GB，capability (12, 0) |

注意 `python-docx` **不在**这个环境里（只有 base 有），所以改 Word 稿要用
`D:\anaconda3\python.exe`，跑模型要用 `esm2_env`，两者别混。

## 4. 重建（换机器时）

```bat
conda create -n esm2_env python=3.11 -y
conda activate esm2_env

:: torch 要 cu128 的预编译版（cu121 不支持 sm_120）
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install transformers==4.48.1 scikit-learn==1.5.0 pandas==2.1.0 ^
            numpy==1.26.0 scipy biopython
```

torch 2.11 目前是 nightly，**版本号会变**。只要 `check_env.py` 的卷积自检通过即可，
不必强求和上表完全一致；但换了 torch 主版本要重跑一次 `scripts\selftest.py`。

## 5. 子进程也会继承这个解释器

`run_5fold.py` / `run_legacy.py` 用的是 `sys.executable`，也就是说：
**你用哪个 python 启动统筹脚本，训练/评估子进程就用哪个**。

```bat
:: 正确
set PY=D:\anaconda3\envs\esm2_env\python.exe
%PY% scripts\run_5fold.py --exp cv5

:: 错误（子进程会全部继承 base 的 torch，跑几步就崩）
python scripts\run_5fold.py --exp cv5
```

## 6. 沙箱小坑

- conda CLI 在本机受限沙箱里偶发 `PermissionError: [WinError 5]`（multiprocessing 管道被拒），
  所以文档里一律写**解释器绝对路径**，不依赖 `conda activate`。
- 控制台默认 GBK：脚本里已加 `sys.stdout.reconfigure(encoding='utf-8')`，
  手工排查时用 `python -X utf8`。
