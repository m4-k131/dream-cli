# -*- coding: utf-8 -*-
"""Backward-compatible shim — use `dreamcli` console script or `python -m dream_cli` instead."""
from dream_cli._scripts.dreamcli import main

if __name__ == "__main__":
    main()
