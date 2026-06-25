"""Backward-compatible shim — use `render-sequence` console script instead."""
from dream_cli._scripts.render_sequence import main

if __name__ == "__main__":
    main()
