# 子代理 worktree / 分支管理约定（2026-11-13 起强制）

> 起因：2026-11-13 的 LoRA 变体批次里，三个子代理跑在**同一个工作树**上，只靠 `_work/lora/ablation_<name>/` 目录隔离 + 一个软锁。结果：(1) 文件虽没撞，但所有产物散在未跟踪目录、不属于任何分支，`git status` 长期很脏；(2) GPU 锁被三个进程反复抢占，PiSSA 的 agent 排队等了 26 分钟，锁文件里 PID 在四个值之间跳变；(3) 卡只有 8.55 GB，三进程并发直接溢出到共享内存。

## 1. 硬规则

**每个子代理实验 = 一个 worktree + 一个分支。** 不再允许"同一个工作树里各写各的目录"。

| 项 | 约定 |
|---|---|
| 目录 | `_work/wt-<name>/`（`<name>` 与分支同名，小写、连字符） |
| 分支 | `exp-<name>`，从 `mlp-head` 的当前 HEAD 起 |
| 创建 | `python scripts/wt.py new <name>` |
| 收尾 | 结果确认后 `python scripts/wt.py rm <name>`（保留分支，除非 `--drop-branch`） |
| 提交 | 实验脚本必须在 worktree 里 `git add` + `commit`，**不允许留未跟踪的实验代码** |
| 产物 | 大文件（probs/csv/ckpt/npz）留在 `_work/`，**不进 git**；只有脚本、README、结果表 commit |

## 2. GPU 锁必须跨 worktree 共享

worktree 之间**不共享** `_work`（它是工作树里的普通目录），所以每个 worktree 会有自己的 `.gpu.lock` —— 那就等于没有锁。

**规则**：锁文件固定用主工作树的绝对路径

```
D:\BIP\module4\cxyaippred\_work\gpu.lock
```

所有 worktree 里的脚本都必须 `LOCK = r"D:\BIP\module4\cxyaippred\_work\gpu.lock"`（硬编码绝对路径，不要用相对路径或 `__file__` 推导）。同一个物理 GPU 只能有一个锁。

## 3. 数据与缓存

- 划分、`data/`、`src/` 都在 worktree 里（git 跟踪，自动一致）。
- `_work/cache_*`（ESM 缓存，每个几百 MB 到 1 GB）**不复制**：脚本里用绝对路径引用主工作树的 `_work/cache_*`。硬链接也可以，但绝对路径更不容易出错。
- 冻结基座统一从 HF 缓存读（`_work/lora/peft_io.py::BASE_MODELS`），任何 worktree 都不许存全量权重。

## 4. 已有的 worktree（2026-11-13 15:45 状态）

| 路径 | 分支 | 用途 |
|---|---|---|
| `_work/wt-posproj` | `posproj-head` | posproj 模块（已合并进主线思路） |
| `_work/wt-retrieval` | `retrieval-head` | 检索增强 A 线 |
| `_work/wt-attn` | `exp-attn` | 自注意力头（已证伪） |
| `_work/wt-hidden` | `exp-hidden` | 容量消融（已证伪） |
| `_work/wt-moe` | `exp-moe` | 端到端 MoE（已证伪） |

已证伪的三条（attn/hidden/moe）可以删 worktree，分支留着当证据。

## 5. 迁移说明

- **正在跑的** VeRA / RandLoRA / PiSSA 不迁移（迁移会打断在跑的训练），它们跑完后产物归档到 `_work/lora/ablation_<name>/`，补一条 commit 把脚本收进 `exp-lora-<name>` 分支。
- **下一批**（T-CENTER、一进程四臂 runner、任何新变体）一律走 `scripts/wt.py new`。

## 6. 与"只跑 fold 0"的关系

两条规则互不冲突、必须同时满足：

1. 实验默认**只跑 fold 0**，扩折前先问。
2. 每个实验**独立 worktree + 分支**，脚本必须 commit。

违反任一条视为实验未完成。
