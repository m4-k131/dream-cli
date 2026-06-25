# -*- coding: utf-8 -*-
"""Backward-compatible shim — use `dream` console script instead."""
from dream_cli._scripts.dream import main

if __name__ == "__main__":
    main()
