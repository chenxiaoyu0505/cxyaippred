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


class ESMBackbone:
    def __init__(self, name='facebook/esm2_t33_650M_UR50D', device='auto',
                 layer=-1, max_len=1024):
        from transformers import AutoModel, AutoTokenizer

        from .utils import pick_device

        self.name = name
        self.device = pick_device(device)
        self.layer = layer
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
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
        """
        enc = self.tok(list(seqs), return_tensors='pt', padding=True,
                       truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer]
        return h.float().cpu().numpy(), enc['attention_mask'].cpu().numpy().astype(np.uint8)
