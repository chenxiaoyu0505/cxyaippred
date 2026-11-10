#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESM-2 骨干（冻结）。

三条硬约束
----------
1. **只加载一次**：5 折 × N seeds 共用同一个 backbone，绝不每折重新加载。
2. **永远 no_grad + eval**：ESM 不参与训练，也不许更新 BN/dropout。
3. **层号固定**：取 `hidden_states[layer]`，默认 -1（最后一层），与旧权重一致。
   换层 = 换表征 = 旧权重作废，所以写进 meta 里。

用法
----
    bb = ESMBackbone('facebook/esm2_t33_650M_UR50D', device='cuda')
    h, mask = bb.encode(['ACDEFGHIK', 'MKV'])   # h: [2, Lmax, 1280] float32, mask: [2, Lmax] uint8
"""
import numpy as np
import torch


def residue_mask(enc, tok):
    """把 tokenizer 的 attention_mask 收紧成"只含真实残基"的掩码。

    为什么需要这一步
    ----------------
    `attention_mask` 的语义是"不是 padding"，而 `<cls>` 和 `<eos>` 不是 padding，
    所以它们的位也是 1。直接拿它做掩码均值池化，分母会变成 L+2 而不是 L
    （中位长度 15 的肽，有 11.8% 的池化向量来自特殊 token）。

    `special_tokens_mask` 那个开关在部分 tokenizer 上不可靠，所以这里直接按
    token id 判定：<cls>/<eos>/<pad> 一律置 0，其余保持 attention_mask 原值。

    返回：与 input_ids 同形状的 bool 张量，True = 真实氨基酸残基。
    """
    keep = enc['attention_mask'].bool()
    ids = enc['input_ids']
    for tid in (getattr(tok, 'cls_token_id', None),
                getattr(tok, 'eos_token_id', None),
                getattr(tok, 'pad_token_id', None)):
        if tid is not None:
            keep = keep & (ids != tid)
    return keep


class ESMBackbone:
    def __init__(self, name='facebook/esm2_t33_650M_UR50D', device='auto',
                 layer=-1, max_len=1024, dtype='fp32'):
        from transformers import AutoModel, AutoTokenizer

        from .utils import pick_device

        self.name = name
        self.device = pick_device(device)
        self.layer = layer
        self.max_len = max_len
        self.dtype = dtype

        # 大模型（3B）在 fp32 下要 12 GB，这台 24 GB 的机器直接 OOM；
        # fp16 只要 5.7 GB。冻结只省梯度/优化器状态，权重本身照样驻留。
        dtype_kw = {}
        if dtype in ('fp16', 'float16', 'half'):
            dtype_kw['torch_dtype'] = torch.float16
        elif dtype in ('bf16', 'bfloat16'):
            dtype_kw['torch_dtype'] = torch.bfloat16
        elif dtype not in ('fp32', 'float32'):
            raise ValueError('dtype 只支持 fp32/fp16/bf16，收到 %r' % (dtype,))

        self.tok = AutoTokenizer.from_pretrained(name)
        try:
            # low_cpu_mem_usage 必须和 device_map 一起用：只给前者时权重停在
            # meta device 上，随后 .to() 会抛 "Cannot copy out of meta tensor"
            self.model = AutoModel.from_pretrained(
                name, low_cpu_mem_usage=True, device_map=self.device, **dtype_kw)
        except (ImportError, ValueError, RuntimeError, KeyError) as exc:
            print('[warn] low_cpu_mem_usage/device_map 不可用（%s: %s），退回普通加载；'
                  '换 3B 这种大模型时峰值内存会翻倍'
                  % (type(exc).__name__, str(exc)[:120]))
            self.model = AutoModel.from_pretrained(name, **dtype_kw).to(self.device)
        self.model = self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self):
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, seqs):
        """
        seqs: list[str] -> (h [B, L, D] float32 CPU, mask [B, L] uint8 CPU)
        L 是本批最长（含 <cls>/<eos>），不是全表最长，省显存。

        mask 的语义：**1 = 真实氨基酸残基，0 = <cls>/<eos>/<pad>**。
        不能直接用 tokenizer 的 attention_mask——它把 <cls>/<eos> 也算作有效位，
        会让掩码均值池化的分母多 2，并把两个特殊 token 的特征混进池化向量。
        """
        enc = self.tok(list(seqs), return_tensors='pt', padding=True,
                       truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer]
        mask = residue_mask(enc, self.tok)
        return h.float().cpu().numpy(), mask.cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def encode_layers(self, seqs, layers):
        """取多层隐状态：layers=[24,30,33] → h [B, K, L, D]（K=len(layers)）。

        为什么要多层：只用最后一层是"默认"，不是"最优"。不同任务的最佳层不同，
        而层选择本身是零成本的超参（见 model_aip.AIPHead.fuse_layers）。
        层号允许负数（-1 = 最后一层），与 hidden_states 的索引一致。
        """
        enc = self.tok(list(seqs), return_tensors='pt', padding=True,
                       truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        n = len(hs)
        picked = []
        for L in layers:
            i = int(L)
            if i < 0:
                i += n
            if not (0 <= i < n):
                raise ValueError('层号 %d 超出范围（共 %d 层）' % (L, n))
            picked.append(hs[i])
        h = torch.stack(picked, dim=1)                 # [B, K, L, D]
        mask = residue_mask(enc, self.tok)
        return h.float().cpu().numpy(), mask.cpu().numpy().astype(np.uint8)
