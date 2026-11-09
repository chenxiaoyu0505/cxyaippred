#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小工具：随机种子、落盘、计时、设备。

注意：torch 只在真正需要它的函数里 import。
`eval_aip.py` 这类纯统计脚本不该被 GPU 库拖累（Windows 上 torch 的 DLL
偶发 WinError 1114，一旦发生在 import 阶段会连累整个脚本）。
"""
import json
import os
import random
import time

import numpy as np


def seed_everything(seed):
    """
    固定所有随机源。注意：这只影响**模型侧**随机性；
    数据划分由 fold 文件决定，不受此影响。
    """
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 卷积用 cudnn 的确定性算法，慢一点但可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(want='auto'):
    """
    统一入口，三平台通用：

        auto → cuda > mps(Apple Silicon) > cpu
        显式给 cuda / mps / cpu 则尽量照办，不可用时退回 cpu 并打印警告

    注意 MPS 上个别算子仍不完整，遇到报错就显式 --device cpu。
    """
    import torch

    want = (want or 'auto').lower()
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, 'mps', None)
                   and torch.backends.mps.is_available())

    if want == 'auto':
        if has_cuda:
            return torch.device('cuda')
        if has_mps:
            print('[info] 未检测到 CUDA，使用 Apple MPS')
            return torch.device('mps')
        print('[info] 未检测到加速器，使用 CPU（会很慢，但能跑通）')
        return torch.device('cpu')

    if want.startswith('cuda') and not has_cuda:
        print('[warn] CUDA 不可用，退回 CPU')
        return torch.device('cpu')
    if want == 'mps' and not has_mps:
        print('[warn] MPS 不可用，退回 CPU')
        return torch.device('cpu')
    return torch.device(want)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class Timer:
    def __init__(self, name=''):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.sec = time.time() - self.t0
        if self.name:
            print('[%s] %.1fs' % (self.name, self.sec))

    @property
    def elapsed(self):
        return time.time() - self.t0
