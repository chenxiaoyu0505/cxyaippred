#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境自检：确认当前解释器就是能跑 GPU 的 conda 环境 `esm2_env`。

背景：本机有两个 Python，长得像但**不一样**——
    D:\\anaconda3\\envs\\esm2_env\\python.exe  torch 2.11+cu128  → 支持 RTX 5060（sm_120）✅
    D:\\anaconda3\\python.exe                  torch 2.5.1+cu121 → 不支持 sm_120，一跑卷积就报
                                              "no kernel image is available" ❌
这个脚本会在 GPU 上真跑一次卷积，而不是只看 torch.cuda.is_available()（那项在坏环境下也是 True）。

用法：
    D:\\anaconda3\\envs\\esm2_env\\python.exe scripts\\check_env.py
退出码 0 = 可用；1 = 不可用（并说明原因）。
"""
import importlib
import os
import sys

REQUIRED = ['torch', 'transformers', 'numpy', 'pandas', 'sklearn', 'scipy']


def main():
    ok = True
    print('解释器 : %s' % sys.executable)
    print('版本   : %s' % sys.version.split()[0])
    print('前缀   : %s' % sys.prefix)

    if sys.version_info[:2] < (3, 10):
        print('[FAIL] 需要 Python >= 3.10')
        ok = False

    for m in REQUIRED:
        try:
            mod = importlib.import_module(m)
            print('  %-14s %s' % (m, getattr(mod, '__version__', '?')))
        except Exception as e:
            print('  %-14s [MISSING] %s' % (m, type(e).__name__))
            ok = False

    if not ok:
        print('\n结论：依赖不全，换 D:\\anaconda3\\envs\\esm2_env\\python.exe 再试')
        return 1

    import torch
    print('\ntorch  : %s | CUDA %s' % (torch.__version__, torch.version.cuda))
    if not torch.cuda.is_available():
        print('[FAIL] CUDA 不可用')
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print('GPU    : %s | capability %s' % (name, cap))

    # 真跑一次卷积：sm_120 不被支持时这里才会暴露
    try:
        x = torch.randn(2, 4, 16, device='cuda')
        w = torch.randn(4, 4, 3, device='cuda')
        y = torch.nn.functional.conv1d(x, w, padding=1)
        torch.cuda.synchronize()
        assert y.shape == (2, 4, 16)
        print('卷积   : OK %s' % (tuple(y.shape),))
    except Exception as e:
        print('[FAIL] GPU 卷积失败：%s' % e)
        print('       多半是 torch 版本不支持 sm_120，换 esm2_env（torch 2.11+cu128）')
        return 1

    print('\n结论：环境可用（%s）' % ('esm2_env' if 'envs' in sys.prefix else sys.prefix))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
