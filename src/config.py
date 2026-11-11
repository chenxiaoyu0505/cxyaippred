#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置：所有超参的唯一来源。

为什么要这个文件
----------------
审稿意见 R1-3 要求"每个数字都能从可审计的样本级标签与预测重算"。
如果超参散落在命令行里，半年后没人说得清 0.7336 是哪组参数跑出来的。
所以训练开始时把 RunCfg 整体写进 `runs/<run>/config.json`，
预测文件与配置永远成对出现。

用法
----
    cfg = RunCfg(fold=0)
    cfg.train.seed = 1
    cfg.ema.enabled = True
    cfg.save('runs/fold0_m1/config.json')
    cfg2 = RunCfg.load('runs/fold0_m1/config.json')
"""
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class ESMCfg:
    """冻结的 ESM-2 语言模型"""
    name: str = 'facebook/esm2_t33_650M_UR50D'
    layer: int = -1          # -1 = 最后一层（hidden_states[-1]）
    max_len: int = 1024
    batch_size: int = 16     # 仅 cache 脚本使用


@dataclass
class ModelCfg:
    """AIPHead（不含 ESM）"""
    embed_dim: int = 1280
    # 占位默认；train_aip.py 会按实际手工特征矩阵覆盖（默认 amphipath=704，消融 694）
    hand_dim: int = 704
    hidden: int = 512
    n_res: int = 2
    cnn_drop: float = 0.2
    mlp_drop: float = 0.3
    # 残差块归一化：'layer'        = 逐位置 LayerNorm（ResBlock 单独使用/显式指定时）
    #               'peptide_bn'   = masked BN + 按肽等权（掩掉 padding，每条肽投一票）
    #               'masked_batch' = masked BN，按残基位置等权（长肽权重更大）
    #               'batch'        = 原生 BatchNorm1d（padding 进统计量，仅消融/旧权重）
    # 默认 peptide_bn：掩掉 padding、按肽等权，是 2026-11-10 定下的基线配置。
    norm: str = 'peptide_bn'
    # 手工特征标准化：'standard' = 每折只用训练行拟合 mean/std；'none' = 旧行为（不推荐）
    hand_norm: str = 'standard'
    # 手工块追加项：'amphipath' = 694 + 10 维两亲性（704，**默认**）；
    #               '' = 只保留 694 维基础块（消融用，CLI 加 --no-amphipath）
    hand_extra: str = 'amphipath'
    # 手工特征筛选：'none' = 全量 704 维（旧行为，默认）；'mi_rf' = 方差 + 相关性 +
    # 互信息 + 随机森林选到 ≤80 维。只在**本折训练行**上拟合，索引落盘到
    # hand_selector.npz；推理时必须用它复现（见 scripts/predict.py）。
    hand_select: str = 'none'
    # 融合方式：'early' = cat(ESM 池化, hand) 后进 MLP（旧行为，默认）；
    #           'late'  = 两条支路各自投影到 fused_dim，再做
    #                     [h_e, h_h, h_e*h_h, |h_e-h_h|] 的显式交互后进 MLP
    fuse: str = 'early'
    fused_dim: int = 128
    # ---- CfC（LNN）序列支路 ----
    # 直接把 one-hot 序列喂进闭式 CfC，池化后与 ESM/手工特征 concat。
    # 不加 router：静态权重会被下一层 Linear 吸收，等于没加。
    lnn: bool = False
    lnn_dim: int = 128        # 用户定的 128（原设计 64）
    lnn_dt: float = 1.0       # Δt；"一个残基 = 一个时间步"，做机制实验时可缩放
    # 支路开关：--lnn-only 关掉 ESM 与手工特征，只留 CfC（探针实验用）
    use_esm: bool = True
    use_hand: bool = True
    # ---- 多层 ESM 融合 ----
    # 缓存里存 K 层时（cache_esm.py --layers 24,30,33），这里决定怎么合成一层：
    #   'weighted' = softmax 出来的 K 个标量权重（只多 K 个参数，推荐）
    #   'mean'     = 等权平均；'last' = 只用最后一层（等价单层）
    esm_fuse: str = 'weighted'
    esm_layers: int = 1
    # 缓存里实际存了哪几层，由 train 脚本从 esm_meta.json 填进来，写进权重供推理复现
    esm_layer_ids: str = ''
    # ---- 原型旁路（prototype side-branch）----
    # 与主分支吃同一份拼接特征，只做"小投影 → 每类 K 个原型 → 分数 B"，
    # 最终分数 = A + alpha * B。proto_pos/proto_neg 都为 0 时完全关闭（默认）。
    # 注意：它不引入新信息，作用是结构先验 + 可解释性，别指望它替代主分支提分。
    proto_pos: int = 0        # 阳性原型数
    proto_neg: int = 0        # 阴性原型数
    proto_dim: int = 128      # 投影维度
    proto_tau: float = 0.1    # cosine 温度
    proto_alpha: float = 1.0  # B 的固定权重（先不做可学习，避免 alpha 与 B 尺度不可辨识）
    proto_w_aux: float = 0.5  # 辅助损失 BCE(B) 的权重，防止 B 免费搭车
    proto_w_div: float = 0.01 # 类内原型多样性正则权重
    proto_init: str = 'kmeans'  # 'kmeans' = 用训练折特征聚类初始化；'random' = 随机


@dataclass
class DataCfg:
    """数据与划分"""
    data_csv: str = 'data/splits/split_origtest_cv5_seed42_nobip/table.csv'
    folds_csv: str = 'data/splits/split_origtest_cv5_seed42_nobip/folds.csv'
    split_id: str = 'split_origtest_cv5_seed42_nobip'
    split_sha256: str = ''    # 由 run 脚本填入，写进 config.json 作证据
    cache_dir: str = 'cache'


@dataclass
class EMACfg:
    """EMA-SAT：教师目标 = α_e·硬标签 + (1−α_e)·历史概率 EMA

    α_e = alpha_min + (1−alpha_min)·2^(−max(0, e−start_epoch+1)/tau).
    α 是硬标签权重，不是教师权重；alpha_min=0.7 表示最终 70% 来自
    硬标签、30% 来自教师（β_max=0.3）。tau=1 时每轮减半。
    """
    enabled: bool = False
    alpha_min: float = 0.7    # 硬标签权重下限；等价教师权重上限 0.3
    decay: float = 0.65       # 教师状态 q 的滑动平均系数
    start_epoch: int = 2      # 从该 epoch 起使用教师；上一轮末初始化 q
    tau: float = 1.0          # 半衰期（epoch）；1.0 即每轮减半


@dataclass
class TrainCfg:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 10        # 早停：监控指标连续 N 轮不升
    monitor: str = 'auroc'    # 早停/选最优 epoch 的指标：auroc（默认）或 aupr
    grad_clip: float = 0.0
    pos_weight: float = 0.0   # 0 = 不加权；>0 = BCE 的 pos_weight（固定值）
    device: str = 'cuda'
    seed: int = 0             # model seed：初始化 / 打乱 / dropout
    log_every: int = 1

    # ---- 标签噪声注入（默认 0 = 不动标签；用于检验抗噪性，只翻训练行） ----
    label_noise: float = 0.0  # 对称翻转比例：随机把 label 0<->1


@dataclass
class ThresholdCfg:
    """验证集上网格搜 MCC 最大阈值（不改变概率，只用于报告 F1/MCC/Acc）"""
    lo: float = 0.10
    hi: float = 0.99
    n: int = 199


@dataclass
class RunCfg:
    esm: ESMCfg = field(default_factory=ESMCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    ema: EMACfg = field(default_factory=EMACfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    threshold: ThresholdCfg = field(default_factory=ThresholdCfg)
    fold: int = 0
    exp: str = ''            # 实验名（experiments/<exp>/）；空 = 旧的平铺模式

    # ---- 派生 ----
    def run_name(self):
        """有实验名时用 <fold>_m<seed>（实验目录已标明是哪条线），否则带 split_id"""
        if self.exp:
            return 'fold%d_m%d' % (self.fold, self.train.seed)
        return 'fold%d_%s_m%d' % (self.fold, self.data.split_id, self.train.seed)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def _sub(cls, data):
        """只接收当前 dataclass 认识的键，兼容旧实验里已删除的字段。"""
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls(
            esm=cls._sub(ESMCfg, d.get('esm', {})),
            model=cls._sub(ModelCfg, d.get('model', {})),
            data=cls._sub(DataCfg, d.get('data', {})),
            ema=cls._sub(EMACfg, d.get('ema', {})),
            train=cls._sub(TrainCfg, d.get('train', {})),
            threshold=cls._sub(ThresholdCfg, d.get('threshold', {})),
            fold=d.get('fold', 0),
            exp=d.get('exp', ''),
        )

    def diff(self):
        """给人看的开关摘要，写进日志首行"""
        return ('fold=%d seed=%d ema=%s alpha_min=%.2f start=%d tau=%.2f '
                'pos_weight=%.2f epochs=%d bs=%d lr=%g norm=%s hand_norm=%s '
                'hand_extra=%s lnn=%s lnn_dim=%d lnn_dt=%.2f use_esm=%s use_hand=%s'
                % (self.fold, self.train.seed,
                   self.ema.enabled, self.ema.alpha_min,
                   self.ema.start_epoch, self.ema.tau, self.train.pos_weight,
                   self.train.epochs, self.train.batch_size, self.train.lr,
                   self.model.norm, self.model.hand_norm,
                   self.model.hand_extra or '-',
                   self.model.lnn, self.model.lnn_dim, self.model.lnn_dt,
                   self.model.use_esm, self.model.use_hand))
