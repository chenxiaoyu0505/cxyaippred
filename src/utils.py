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


def pick_device(want='cuda'):
    import torch

    if want == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA 不可用，退回 CPU')
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
