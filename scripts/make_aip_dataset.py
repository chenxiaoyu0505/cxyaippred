# -*- coding: utf-8 -*-
"""
从双任务数据集里抽出单任务 AIP 数据集。

输入（论文原始数据，位置在项目外）：
    D:\\BIP\\data\\train2.csv
    D:\\BIP\\data\\test.csv
    列：id1, id2, squence, label-aip, label-pip, mask-aip, mask-pip

输出（本目录 data/ 下）：
    aip_train.csv   列：sequence, label
    aip_test.csv    列：sequence, label

用法：
    python scripts/make_aip_dataset.py            # 生成并打印核对信息
    python scripts/make_aip_dataset.py --check    # 只核对，不覆盖

说明：
    1. 只保留 mask-aip == 1 的行，即"有 AIP 标注"的样本；PIP 列整体丢弃。
    2. 序列统一大写并去除空白；非标准氨基酸的行会被报告（默认保留，不静默丢弃）。
    3. 同一序列在 AIP 子集内不会重复（已核实），脚本仍会做断言，防止上游数据变化。
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')

SRC_TRAIN = r'D:\BIP\data\train2.csv'
SRC_TEST = r'D:\BIP\data\test.csv'

AA = set('ACDEFGHIKLMNPQRSTVWY')


def to_aip(df, name):
    if 'mask-aip' not in df.columns:
        raise SystemExit('%s 缺少 mask-aip 列：%s' % (name, list(df.columns)))
    sub = df[df['mask-aip'] == 1]
    out = pd.DataFrame({
        'sequence': sub['squence'].astype(str).str.upper().str.strip(),
        'label': sub['label-aip'].astype(int),
    }).reset_index(drop=True)
    return out


def report(df, name):
    n = len(df)
    pos = int(df['label'].sum())
    uniq = df['sequence'].nunique()
    bad = [s for s in df['sequence'] if set(s) - AA]
    lens = df['sequence'].str.len()
    print('[%s] 行数 %d | 唯一序列 %d | 阳性 %d (%.1f%%) | 长度 中位 %d / 95%% %d / 最大 %d'
          % (name, n, uniq, pos, 100.0 * pos / max(n, 1),
             lens.median(), lens.quantile(0.95), lens.max()))
    if n != uniq:
        print('  !! 有 %d 条重复序列' % (n - uniq))
    if bad:
        print('  !! %d 条含非标准氨基酸，例：%s' % (len(bad), bad[:3]))
    return {'n': n, 'uniq': uniq, 'pos': pos, 'bad': len(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只核对，不写文件')
    args = ap.parse_args()

    train = pd.read_csv(SRC_TRAIN)
    test = pd.read_csv(SRC_TEST)

    tr = to_aip(train, 'train2.csv')
    te = to_aip(test, 'test.csv')

    s_tr = report(tr, 'aip_train')
    s_te = report(te, 'aip_test')

    overlap = set(tr['sequence']) & set(te['sequence'])
    print('[检查] 训练/测试 精确序列重叠：%d' % len(overlap))
    if overlap:
        print('  !! 存在重叠，例：%s' % sorted(overlap)[:3])

    assert s_tr['n'] == s_tr['uniq'], 'AIP 训练集出现重复序列，请先排查上游数据'
    assert s_te['n'] == s_te['uniq'], 'AIP 测试集出现重复序列，请先排查上游数据'
    assert not overlap, 'AIP 训练/测试存在精确重叠序列'

    if args.check:
        print('--check：未写入文件')
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    p_tr = os.path.join(DATA_DIR, 'aip_train.csv')
    p_te = os.path.join(DATA_DIR, 'aip_test.csv')
    tr.to_csv(p_tr, index=False)
    te.to_csv(p_te, index=False)
    print('已写入：%s (%d 行)' % (p_tr, len(tr)))
    print('已写入：%s (%d 行)' % (p_te, len(te)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
