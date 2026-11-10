# experiments\ —— 一次实验一个目录

```
experiments\
  cv5\              5 折主实验（run_5fold.py --exp cv5）
    runs\
      fold0_m0\      best.pt / config.json / split.csv / val_probs.csv
                     test_probs.csv / all_probs.csv / train_log.csv / threshold.json
      fold1_m0\ ...
    results\         metrics_detail.csv / metrics.json / summary.md
    manifest.json    数据与划分的 sha256、代码哈希、环境、耗时
  legacy\           原 4,224 训练 / 462 测试 的对照线（run_legacy.py --exp legacy）
  cv5_ema\          将来的消融：加 EMA-SAT
```

规则（`plan/DECISIONS.md` D-8）：

1. **不要手工往 `runs\` 里塞文件**，只由 `train_aip.py` 写。
2. **指标不要写在 run 目录里**，只由 `eval_aip.py` 写到 `results\`——
   旧项目"同一个指标两个数"就是这么来的。
3. 换数据或换划分 = 换 `--exp`，不要覆盖旧实验。

看结果先看 `results\summary.md`（mean±sd + 单折明细 + 口径说明）；
要重算指标只需 `config.json` + `test_probs.csv` + `all_probs.csv` + `threshold.json`。
