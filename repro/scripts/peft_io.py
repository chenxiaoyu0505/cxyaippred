# -*- coding: utf-8 -*-
"""PEFT 增量存取：只落盘可训练参数，冻结基座永不再存。

背景
----
`_work/lora` 下原有的 ckpt 用 `torch.save({'model': model.state_dict()})`，
把 652.4M 的冻结 ESM-2 主干一起写进每个文件：单文件 1.31 GB，其中 99.68%
是从未更新过的基座（与 HF 缓存逐字节相同）。18 个文件 = 17.1 GB。

约定（本文件实施）
------------------
* 基座只以「HF 名称 + revision」记录在 meta 里，权重来自共享缓存，见 BASE_MODELS。
* delta 只含可训练项：LoRA/DoRA 的 `A.weight`/`B.weight`、DoRA 的 `m`、
  MoRA 的 `M`、以及 `posproj` / `posconv` / `head`。
* 兼容旧格式：读到 `{'model': full_state_dict}` 时自动过滤成 delta，
  之后可以安全删掉旧文件（delta 能完整复现）。
* resume 文件额外存 optimizer 状态（只覆盖可训练参数，约 16 MB）。

典型体积：LoRA r=8 = 2.028M 参数，fp32 约 8.1 MB、fp16 约 4.1 MB，
相对 1.31 GB 缩小约 160–310 倍。
"""
import os

import torch

# 共享基座：唯一的权重来源，永不复制进 ckpt
BASE_MODELS = {
    'facebook/esm2_t33_650M_UR50D':
        r'C:\Users\cxy\.cache\huggingface\hub\models--facebook--esm2_t33_650M_UR50D',
}

DELTA_PREFIXES = ('posproj', 'posconv', 'head')
# A.weight/B.weight = LoRA/DoRA；.A/.B = MoRA 的固定随机投影（buffer，但决定 ΔW，
# 不存就无法复现）；.m = DoRA 幅度；.M = MoRA 可训方阵
DELTA_SUFFIXES = ('A.weight', 'B.weight', '.A', '.B', '.m', '.M')


def is_delta_key(k):
    """判断一个 state_dict key 是否属于「学到了东西」的部分。"""
    if k.startswith(DELTA_PREFIXES):
        return True
    return any(k.endswith(s) for s in DELTA_SUFFIXES)


def extract_delta(state_dict):
    return {k: v.detach().to('cpu').clone()
            for k, v in state_dict.items() if is_delta_key(k)}


def _numel(d):
    return sum(v.numel() for v in d.values())


def verify_complete(model):
    """训练前/保存前自检：任何 requires_grad 的参数都必须被 is_delta_key 认领。

    防止新增一个可训练模块却忘了更新过滤规则，导致保存时静默丢参数。
    """
    lost = [n for n, p in model.named_parameters()
            if p.requires_grad and not is_delta_key(n)]
    if lost:
        raise RuntimeError('有可训练参数不在 delta 规则内（会被丢）：%s' % lost[:5])
    return True


def save_delta(path, model, meta=None, dtype=None):
    """把可训练部分存成 delta 文件，返回 (tensor 数, 参数数, 文件 MB)。"""
    verify_complete(model)
    d = extract_delta(model.state_dict())
    if dtype is not None:
        d = {k: v.to(dtype) for k, v in d.items()}
    meta = dict(meta or {})
    meta.setdefault('base_model', 'facebook/esm2_t33_650M_UR50D')
    meta['format'] = 'peft-delta-v1'
    tmp = path + '.tmp'
    torch.save({'format': 'peft-delta-v1', 'delta': d, 'meta': meta}, tmp)
    os.replace(tmp, path)
    return len(d), _numel(d), os.path.getsize(path) / 1e6


def read_any(path):
    """读 delta 或旧全量 ckpt，统一返回 (delta_dict, meta, kind)。"""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'delta' in obj:
        return obj['delta'], obj.get('meta', {}), 'delta'
    if 'model' in obj:
        return extract_delta(obj['model']), {'migrated_from': 'full-state-dict'}, 'full'
    raise ValueError('无法识别的 checkpoint 格式: %s (keys=%s)'
                     % (path, list(obj.keys())))


def load_delta(path, model, verbose=True):
    """把 delta 装进已建好的模型；返回 meta。冻结基座不在文件里，保持原样。"""
    d, meta, kind = read_any(path)
    missing, unexpected = model.load_state_dict(d, strict=False)
    bad_missing = [k for k in missing if is_delta_key(k)]
    if bad_missing or unexpected:
        raise RuntimeError('delta 与模型不匹配: missing=%s unexpected=%s'
                           % (bad_missing[:5], unexpected[:5]))
    if verbose:
        print('[load_delta] %s <- %d tensors / %.3fM params (source=%s)'
              % (os.path.basename(path), len(d), _numel(d) / 1e6, kind), flush=True)
    return meta


def save_resume(path, model, opt, epoch, best_val, bad, meta=None, dtype=None):
    d = extract_delta(model.state_dict())
    if dtype is not None:
        d = {k: v.to(dtype) for k, v in d.items()}
    meta = dict(meta or {})
    meta.setdefault('base_model', 'facebook/esm2_t33_650M_UR50D')
    payload = {'format': 'peft-resume-v1', 'delta': d, 'meta': meta,
               'opt': opt.state_dict(), 'epoch': epoch,
               'best_val': best_val, 'bad': bad}
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path) / 1e6


def load_resume(path, model, opt):
    """恢复训练现场；旧全量 ckpt 也能读（optimizer 参数组顺序不变）。"""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'delta' in obj:
        d = obj['delta']
    elif 'model' in obj:
        d = extract_delta(obj['model'])
    else:
        raise ValueError('无法识别的 resume 文件: %s' % path)
    missing, unexpected = model.load_state_dict(d, strict=False)
    if [k for k in missing if is_delta_key(k)] or unexpected:
        raise RuntimeError('resume delta 不匹配')
    if 'opt' in obj:
        opt.load_state_dict(obj['opt'])
    return obj.get('epoch', -1) + 1, obj.get('best_val', -1.0), obj.get('bad', 0)


def describe(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'delta' in obj:
        d, mb, kind = obj['delta'], os.path.getsize(path) / 1e6, 'delta'
    else:
        d = extract_delta(obj['model'])
        mb, kind = os.path.getsize(path) / 1e6, 'full'
    print('%-52s %-6s %8.1f MB  %4d tensors  %7.3fM params'
          % (os.path.basename(path), kind, mb, len(d), _numel(d) / 1e6))


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        describe(p)
