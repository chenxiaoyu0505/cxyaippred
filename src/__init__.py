#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cxyaippred —— 单任务 AIP 预测

为什么这里不在导入时加载 torch
------------------------------
`make_5fold.py`、`manifest.py` 这类纯数据脚本只需要 pandas。
如果 `src/__init__.py` 里 eager import model_aip，那么任何 `import src.xxx`
都会先把 torch 拖进来（慢，而且 Windows 上偶发 DLL 加载失败会连累数据脚本）。
所以这里改成惰性加载：用到 AIPHead / save_head 时才真正 import。
"""
import importlib

_LAZY = {
    'AIPHead': ('src.model_aip', 'AIPHead'),
    'ResBlock1D': ('src.model_aip', 'ResBlock1D'),
    'save_head': ('src.checkpoint', 'save_head'),
    'load_head': ('src.checkpoint', 'load_head'),
    'extract_from_legacy': ('src.checkpoint', 'extract_from_legacy'),
    'FORMAT': ('src.checkpoint', 'FORMAT'),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))
