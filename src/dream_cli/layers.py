"""Inception 5h layer names and channel counts (frozen graph tensor metadata)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

LayerInfo = Tuple[str, int]


@lru_cache(maxsize=1)
def inception_layers() -> List[LayerInfo]:
    path = Path(__file__).with_name("inception_layers.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(str(row[0]), int(row[1])) for row in raw]
