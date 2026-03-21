# -*- coding: utf-8 -*-
"""Entry point: run `python dreamcli.py` (see README for Conda env)."""
from __future__ import annotations

import argparse
from pathlib import Path

from dream_cli.application import DreamApplication
from dream_cli.cli import InteractiveDreamCli


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDream interactive CLI (Inception 5h).")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (default: current working directory).",
    )
    args = parser.parse_args()
    app = DreamApplication.with_defaults(args.root)
    InteractiveDreamCli(app).run()


if __name__ == "__main__":
    main()
