#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验清单：给每一次"跑完的实验"留一份可核对的身份证。

为什么需要
----------
半年后（或换一个 agent）打开 experiments/ 时，必须能回答三个问题：
  1. 这次跑的是哪份数据？—— 数据和划分的 sha256
  2. 用的是哪版代码？    —— src/ 与 scripts/ 全部文件的 sha256
  3. 在什么环境里跑的？  —— python / torch / GPU

任何一个对不上，数字就不可信。这正是审稿意见 R1-3 说的
"metrics should be regenerated from auditable sample-level labels and predictions"。

用法
----
    write_manifest(exp_dir, exp='cv5', data_csv=..., split_csv=..., extra={...})
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()


def code_hashes(root, subdirs=('src', 'scripts')):
    """项目代码快照：src/ 与 scripts/ 下全部 .py 的 sha256（含子目录）"""
    out = {}
    for sub in subdirs:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in sorted(filenames):
                if not fn.endswith('.py'):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root).replace('\\', '/')
                out[rel] = sha256_file(p)[:16]
    return out


def env_info():
    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'executable': sys.executable,
    }
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda'] = torch.version.cuda
        info['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
    except Exception as e:                      # noqa: BLE001
        info['torch'] = 'unavailable: %s' % e
    return info


def write_manifest(exp_dir, exp, data_csv=None, split_csv=None,
                   extra=None, status='finished', root=None):
    """写 <exp_dir>/manifest.json，返回路径"""
    os.makedirs(exp_dir, exist_ok=True)
    man = {
        'experiment': exp,
        'created': datetime.now().isoformat(timespec='seconds'),
        'status': status,
    }

    if data_csv and os.path.exists(data_csv):
        import pandas as pd
        df = pd.read_csv(data_csv)
        col = 'label' if 'label' in df.columns else None
        man['data'] = {
            'file': os.path.relpath(data_csv, root).replace('\\', '/') if root else data_csv,
            'n': int(len(df)),
            'pos': int(df[col].astype(int).sum()) if col else None,
            'sha256': sha256_file(data_csv),
        }

    if split_csv and os.path.exists(split_csv):
        man['split'] = {
            'file': os.path.relpath(split_csv, root).replace('\\', '/') if root else split_csv,
            'sha256': sha256_file(split_csv),
        }

    if root:
        man['code'] = code_hashes(root)

    man['env'] = env_info()
    if extra:
        man.update(extra)

    path = os.path.join(exp_dir, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    return path


def read_manifest(exp_dir):
    path = os.path.join(exp_dir, 'manifest.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
