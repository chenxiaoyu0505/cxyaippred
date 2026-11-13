#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""子代理 worktree 管理（对应 plan/SUBAGENT_WORKTREE_PROTOCOL.md）。

用法：
    python scripts/wt.py new <name> [--base mlp-head]
    python scripts/wt.py list
    python scripts/wt.py rm <name> [--drop-branch]

约定：目录 _work/wt-<name> ，分支 exp-<name> ，基座默认 mlp-head。
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WT_ROOT = os.path.join(REPO, "_work")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def cmd_new(args):
    name = args.name
    if not NAME_RE.match(name):
        sys.exit(f"名字不合法：{name}（只允许小写字母/数字/连字符）")
    path = os.path.join(WT_ROOT, f"wt-{name}")
    branch = f"exp-{name}"
    if os.path.exists(path):
        sys.exit(f"目录已存在：{path}")
    if git("rev-parse", "--verify", branch).returncode == 0:
        sys.exit(f"分支已存在：{branch}（先 git branch -D 或换个名字）")
    r = git("worktree", "add", path, "-b", branch, args.base)
    if r.returncode != 0:
        sys.exit(f"worktree 创建失败：\n{r.stderr}")
    print(f"[ok] worktree {path}")
    print(f"[ok] branch   {branch}  (base {args.base})")
    print()
    print("提醒：")
    print("  1. GPU 锁必须用绝对路径 D:\\BIP\\module4\\cxyaippred\\_work\\gpu.lock")
    print("  2. _work/cache_* 不复制，脚本里用绝对路径引用主工作树")
    print("  3. 实验脚本必须 git add + commit，不留未跟踪代码")
    print(f"  4. 收尾：python scripts/wt.py rm {name}")


def cmd_list(_args):
    r = git("worktree", "list")
    print(r.stdout.rstrip())


def cmd_rm(args):
    name = args.name
    path = os.path.join(WT_ROOT, f"wt-{name}")
    branch = f"exp-{name}"
    if not os.path.exists(path):
        sys.exit(f"目录不存在：{path}")
    r = git("worktree", "remove", path)
    if r.returncode != 0:
        sys.exit(f"删除失败（可能有未提交改动，先 commit 或加 --force）：\n{r.stderr}")
    print(f"[ok] removed worktree {path}")
    if args.drop_branch:
        r = git("branch", "-D", branch)
        print(f"[ok] branch {branch} dropped" if r.returncode == 0 else f"[warn] {r.stderr.strip()}")
    else:
        print(f"[keep] branch {branch} 仍在（用它保留实验记录）")


def main():
    ap = argparse.ArgumentParser(description="子代理 worktree 管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="新建 worktree + 分支")
    p.add_argument("name")
    p.add_argument("--base", default="mlp-head")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("list", help="列出所有 worktree")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rm", help="删除 worktree（默认保留分支）")
    p.add_argument("name")
    p.add_argument("--drop-branch", action="store_true")
    p.set_defaults(func=cmd_rm)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
