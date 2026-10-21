#!/usr/bin/env python3
"""Render {{VAR}} placeholders in prompt templates.

Usage:
    python render_prompt.py --list
    python render_prompt.py 01-langgpt-role-coach.md -v ROLE_NAME=概念教练 -v DOMAIN=LLM
    python render_prompt.py 02-cot-reasoning.md -v TASK="..." -v CONTEXT="..." -o prompt.txt

The script only uses the Python standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
VAR_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def list_templates() -> list[Path]:
    return sorted(
        path
        for path in TEMPLATE_DIR.glob("*.md")
        if path.name.lower() != "readme.md"
    )


def find_variables(text: str) -> list[str]:
    return sorted(set(VAR_RE.findall(text)))


def parse_values(items: list[str]) -> dict[str, str]:
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
    missing = [name for name in find_variables(text) if name not in values]
    if missing and not allow_missing:
        raise ValueError("missing variables: " + ", ".join(missing))

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return values.get(name, match.group(0))

    return VAR_RE.sub(replace, text)


def resolve_template(name: str) -> Path:
    candidate = TEMPLATE_DIR / name
    if candidate.is_file():
        return candidate
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"template not found: {name}")


def main() -> int:
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
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        output = Path(args.out)
        output.write_text(result, encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
