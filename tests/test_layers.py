"""Unit tests for dream_cli.layers (JSON metadata only)."""
from __future__ import annotations

from dream_cli.layers import inception_layers


def test_inception_layers_nonempty() -> None:
    layers = inception_layers()
    assert len(layers) > 0
    name, count = layers[0]
    assert isinstance(name, str)
    assert isinstance(count, int)
    assert count > 0
