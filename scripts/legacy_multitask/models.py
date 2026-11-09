#!/usr/bin/env python3
"""
Model architectures: ESM2 + ResCNN + HandMLP
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D Residual Block"""

    def __init__(self, ch, ks=3, dil=1, drop=0.1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.drop(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        return self.relu(out + res)


class MLPHead(nn.Module):
    """Dual MLP heads for AIP and PIP"""

    def __init__(self, input_dim, hidden=256, drop=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(drop)
        )
        self.head_aip = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )
        self.head_pip = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        return torch.cat([self.head_aip(shared_feat), self.head_pip(shared_feat)], dim=1)


class EMA_SATManager:
    """EMA-SAT: Exponential Moving Average Self-Training"""

    def __init__(self, n_samples, n_classes=2, device='cpu'):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.device = device
        self.register_buffer = torch.zeros(n_samples, n_classes, device=device)
        self.register_buffer[:, :] = 0.5
        self.initialized = torch.zeros(n_samples, dtype=torch.bool, device=device)

    def update(self, indices, current_probs, decay=0.99):
        with torch.no_grad():
            current_probs = torch.clamp(current_probs, 0.01, 0.99)
            for i, idx in enumerate(indices):
                idx = int(idx)
                old_val = self.register_buffer[idx]
                new_val = current_probs[i]
                if not self.initialized[idx]:
                    self.register_buffer[idx] = new_val
                    self.initialized[idx] = True
                else:
                    updated = decay * old_val + (1 - decay) * new_val
                    self.register_buffer[idx] = updated
            self.register_buffer = torch.clamp(self.register_buffer, 0.01, 0.99)

    def get_soft_targets(self, indices, y_true, alpha=0.7):
        batch_size = len(indices)
        ema_probs = torch.zeros(batch_size, self.n_classes, device=self.device)
        init_mask = self.initialized[indices]
        if init_mask.any():
            ema_probs[init_mask] = self.register_buffer[indices[init_mask]]
        if (~init_mask).any():
            ema_probs[~init_mask] = 0.5

        soft_targets = alpha * y_true + (1 - alpha) * ema_probs
        soft_targets[~init_mask] = y_true[~init_mask]
        return torch.clamp(soft_targets, 0.0, 1.0)


class ESMResCNNHandMLP(nn.Module):
    """Main model: Frozen ESM2 + ResCNN + Handcrafted features"""

    def __init__(self, esm_model, hidden=512, mlp_drop=0.3, n_res=2, cnn_drop=0.2, hand_dim=0):
        super().__init__()
        self.esm = esm_model
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

        self.embed_dim = self.esm.config.hidden_size
        self.hand_dim = hand_dim

        self.cnn = nn.Sequential(*[ResBlock1D(self.embed_dim, drop=cnn_drop) for _ in range(n_res)])
        self.head = MLPHead(self.embed_dim + hand_dim, hidden, mlp_drop)

    def forward(self, input_ids, attention_mask, hand_feats=None):
        with torch.no_grad():
            h = self.esm(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True).hidden_states[-1]

        mask_valid = attention_mask.bool()
        x = h.transpose(1, 2)
        x = self.cnn(x)
        mask_cnt = mask_valid.unsqueeze(1).float().sum(dim=2).clamp(min=1.0)
        pooled = (x * mask_valid.unsqueeze(1).float()).sum(dim=2) / mask_cnt

        if hand_feats is not None and self.hand_dim > 0:
            pooled = torch.cat([pooled, hand_feats], dim=1)

        return self.head(pooled)
