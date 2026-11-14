# -*- coding: utf-8 -*-
"""TLoRA 适配 + posproj-adaptive 分类头的模型定义。

与仓库内实验实现（`_work/lora/ablation_tlora/run_tlora_fold0.py`、
`_work/wt-tlora5/scripts/run_tlora_5fold.py`）**逐行同构**，只是剥掉了 GPU 锁、
日志 Tee、绝对路径等实验脚手架。改这里等于改论文口径，请勿随手动数字。

结构
----
1. 冻结 ESM-2 (650M, 33 层)；把每层的 q/k/v 三个 nn.Linear 换成 LoRALinear。
2. TLoRA：A 的初始化 = SVD(W0 · C) 的前 r 个右奇异向量（C 为该折训练行上的
   输入激活协方差），A 冻结，只训 B（零初始化 → 起点严格等价基座）。
3. 头 = posproj-adaptive：
   Linear(1280→4) 逐位置  ->  Conv1d(4,4,3)  ->  GELU
   ->  掩码加权 F.interpolate 重采样到 32 槽  ->  flatten(128)
   ->  Linear(256) -> GELU -> Dropout(0.3) -> Linear(1)

已知瑕疵（为复现原数字，故意保留）：卷积发生在掩码之前，pad 位置的隐状态会被
卷进序列尾部；因此推理结果对 batch 组成敏感（bs=16 时 fold0 test AUROC 0.7365，
bs=32 时 0.7445）。见 README "已知问题"。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_POS, LMAX, D_MODEL = 4, 32, 1280

# 仅用于自检"初始 ΔW 严格为 0"的开关；训练/推理恒为 False
DELTA_OFF = False


class LoRALinear(nn.Module):
    """A 冻结在数据对齐初始化上，只训 B。"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        # 与参考实现逐位一致的 RNG 消耗量（A 随后被 SVD 覆盖）
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.A.weight.requires_grad_(False)
        self.scale = alpha / r

    def forward(self, x):
        if DELTA_OFF:
            return self.base(x)
        return self.base(x) + self.B(self.A(x.float())).to(x.dtype) * self.scale


class ESM_LoRA_PosProj(nn.Module):
    def __init__(self, esm, rank=8, alpha=16, drop=0.3):
        super().__init__()
        self.esm = esm
        self.esm.requires_grad_(False)
        for layer in self.esm.encoder.layer:
            for name in ("query", "key", "value"):
                lin = getattr(layer.attention.self, name)
                setattr(layer.attention.self, name, LoRALinear(lin, rank, alpha))
        self.posproj = nn.Linear(D_MODEL, D_POS)
        self.posconv = nn.Conv1d(D_POS, D_POS, 3, padding=1)
        self.posact = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(D_POS * LMAX, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 1))

    def forward(self, ids, attn, rmask):
        h = self.esm(input_ids=ids, attention_mask=attn).last_hidden_state
        m = rmask.unsqueeze(1).to(h.dtype)                      # [B,1,L] 仅残基
        z = self.posact(self.posconv(self.posproj(h).transpose(1, 2)))
        zf = F.interpolate(z * m, size=LMAX, mode="linear", align_corners=False)
        mf = F.interpolate(m, size=LMAX, mode="linear", align_corners=False)
        z = zf / mf.clamp(min=1e-6)
        return self.head(z.flatten(1)).squeeze(-1)


def qkv_linears(esm):
    out = []
    for layer in esm.encoder.layer:
        for name in ("query", "key", "value"):
            out.append(getattr(layer.attention.self, name))
    return out


def adapter_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def unwrap_qkv(esm):
    """把 LoRALinear 换回原始 nn.Linear，供下一折重算协方差。"""
    n = 0
    for layer in esm.encoder.layer:
        for name in ("query", "key", "value"):
            mod = getattr(layer.attention.self, name)
            if isinstance(mod, LoRALinear):
                setattr(layer.attention.self, name, mod.base)
                n += 1
    return n


def residue_mask(attn):
    """去掉 <cls>/<eos>：只留真实残基位置。"""
    m = attn.clone()
    m[:, 0] = 0
    lens = attn.sum(1)
    m[torch.arange(len(attn), device=attn.device), lens - 1] = 0
    return m


def batchify(seqs, tok, bs):
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], padding=True, truncation=True, max_length=64,
                  return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, tok, seqs, bs=32, device="cuda"):
    """返回 sigmoid 后的概率（numpy）。bs 必须与训练一致（32）。"""
    model.eval()
    ps = []
    for ids, attn in batchify(seqs, tok, bs):
        ids, attn = ids.to(device), attn.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(ids, attn, residue_mask(attn))
        ps.append(logits.float().cpu().numpy())
    z = np.concatenate(ps)
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def collect_cov(esm, tok, seqs, bs=32, device="cuda"):
    """在**未挂 adapter** 的原始 ESM 上统计 q/k/v 输入的协方差 C = Σ x xᵀ。

    只累加非 pad 残基，fp32 在 GPU 累积；调用方再除以 token 数。
    """
    mods = qkv_linears(esm)
    covs = [torch.zeros(D_MODEL, D_MODEL, device=device, dtype=torch.float32)
            for _ in range(len(mods))]
    holder = {"m": None}
    handles = []

    def mk(i):
        def hook(mod, inp, out):
            x = inp[0]
            xf = x.float()
            m = holder["m"]
            if m is not None and xf.dim() == 3:
                xf = xf * m.to(xf.dtype)[:, :, None]
            xf = xf.reshape(-1, xf.shape[-1])
            covs[i].add_(xf.t() @ xf)
        return hook

    for i, mod in enumerate(mods):
        handles.append(mod.register_forward_hook(mk(i)))
    n_tok = 0
    esm.eval()
    try:
        for ids, attn in batchify(seqs, tok, bs):
            ids, attn = ids.to(device), attn.to(device)
            holder["m"] = attn
            n_tok += int(attn.sum().item())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                esm(input_ids=ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()
        holder["m"] = None
    for c in covs:
        c.div_(float(n_tok))
    return covs, n_tok


def svd_init_A(raw_linears, covs, r):
    """A = (W0 C) 的前 r 个右奇异向量 V_rᵀ。"""
    A_list, S_list = [], []
    for lin, C in zip(raw_linears, covs):
        W0 = lin.weight.detach().float()
        M = W0 @ C.float()
        _, S, Vh = torch.linalg.svd(M, full_matrices=False)
        A_list.append(Vh[:r, :].contiguous().clone())
        S_list.append(S[:r].clone())
    return A_list, S_list
