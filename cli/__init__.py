from __future__ import annotations

import argparse
from collections.abc import Sequence

from .tensorboard import run_tensorboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log", help="启动 TensorBoard，带有预设参数")
    log_parser.set_defaults(func=run_tensorboard)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func()
