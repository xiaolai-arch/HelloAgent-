#!/usr/bin/env python3
"""Render {{VAR}} placeholders in prompt templates.

用法：
    python render_prompt.py --list
    python render_prompt.py 01-langgpt-role-coach.md -v ROLE_NAME=概念教练 -v DOMAIN=LLM
    python render_prompt.py 02-cot-reasoning.md -v TASK="..." -v CONTEXT="..." -o prompt.txt

设计说明：
- 只替换形如 {{UPPER_SNAKE_CASE}} 的占位符，避免误伤正文里的普通花括号。
- 默认要求所有变量都提供；缺变量直接报错，防止把 {{TASK}} 原样发给模型。
- 只用标准库，方便在课程仓库里直接运行，不引入额外依赖。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 模板固定放在脚本同级目录，避免从不同工作目录运行时找不到文件。
TEMPLATE_DIR = Path(__file__).resolve().parent
# 只匹配大写变量名；正文里的小写花括号、JSON、代码块不会被当成占位符。
VAR_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def list_templates() -> list[Path]:
    """列出模板目录下的 Markdown 文件，并排除使用说明 README。"""
    return sorted(
        path
        for path in TEMPLATE_DIR.glob("*.md")
        if path.name.lower() != "readme.md"
    )


def find_variables(text: str) -> list[str]:
    """扫描模板中的变量名，去重后按字母序返回。"""
    return sorted(set(VAR_RE.findall(text)))


def parse_values(items: list[str]) -> dict[str, str]:
    """把 CLI 传入的 KEY=VALUE 列表解析成字典。

    只按第一个等号切分，因此值里可以包含等号（例如 URL 查询参数）。
    变量名统一转大写，和模板里的 {{UPPER_SNAKE_CASE}} 保持一致。
    """
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().upper()
        if not re.fullmatch(r"[A-Z0-9_]+", key):
            raise ValueError(f"invalid variable name: {key!r}")
        values[key] = value
    return values


def render(text: str, values: dict[str, str], allow_missing: bool = False) -> str:
    """替换模板变量；默认缺变量即报错。

    allow_missing=True 时保留 {{VAR}} 原样，便于先看模板结构或分阶段填值。
    """
    missing = [name for name in find_variables(text) if name not in values]
    if missing and not allow_missing:
        raise ValueError("missing variables: " + ", ".join(missing))

    def replace(match: re.Match[str]) -> str:
        """返回变量值；未提供时保留原占位符。"""
        # 用回调而不是 str.replace，确保只替换正则命中的变量位置。
        name = match.group(1)
        return values.get(name, match.group(0))

    return VAR_RE.sub(replace, text)


def resolve_template(name: str) -> Path:
    """解析模板路径：优先脚本同级目录，其次当前目录/显式路径。"""
    candidate = TEMPLATE_DIR / name
    if candidate.is_file():
        return candidate
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"template not found: {name}")


def main() -> int:
    """解析命令行参数，执行列表或渲染流程，返回进程退出码。"""
    # 命令行入口：--list 查看模板和变量，其余参数用于渲染。
    parser = argparse.ArgumentParser(
        description="Render {{VAR}} placeholders in prompt templates."
    )
    parser.add_argument("template", nargs="?", help="template file name or path")
    parser.add_argument(
        "-v",
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="variable value; repeat for multiple variables",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list templates and their variables",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="leave unknown placeholders unchanged instead of failing",
    )
    parser.add_argument("-o", "--out", help="write rendered prompt to a file")
    args = parser.parse_args()

    # 列表模式：只读模板，不要求提供任何变量。
    if args.list:
        for path in list_templates():
            variables = find_variables(path.read_text(encoding="utf-8"))
            suffix = ", ".join(variables) if variables else "(no variables)"
            print(f"{path.name}\n  {suffix}")
        return 0

    if not args.template:
        parser.print_help()
        return 1

    try:
        path = resolve_template(args.template)
        text = path.read_text(encoding="utf-8")
        result = render(text, parse_values(args.var), args.allow_missing)
    except (OSError, ValueError) as exc:
        # 文件不存在、编码错误、变量缺失都走这里；返回非 0 便于脚本/CI 判断。
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        # 显式指定 UTF-8，避免 Windows 默认 GBK 把中文写成乱码。
        output = Path(args.out)
        output.write_text(result, encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
