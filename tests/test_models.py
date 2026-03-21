"""Unit tests for dream_cli.models (no TensorFlow)."""
from __future__ import annotations

import numpy as np

from dream_cli.models import DreamSettings, RendererConfig, legacy_default_setting_dict, legacy_default_renderer_dict


def test_dream_settings_roundtrip() -> None:
    s = DreamSettings.default_new()
    s.name = "t"
    s.iterations = 5
    restored = DreamSettings.from_mapping(s.to_stored_dict())
    assert restored.name == "t"
    assert restored.iterations == 5
    assert restored.to_stored_dict() == s.to_stored_dict()


def test_renderer_from_mapping_partial() -> None:
    r = RendererConfig.from_mapping({"name": "x", "layer": "conv2d2_pre_relu"})
    assert r.name == "x"
    assert r.layer == "conv2d2_pre_relu"
    assert r.tile_size == 300


def test_renderer_runtime_dict_empty_mask() -> None:
    r = RendererConfig()
    d = r.to_runtime_dict()
    assert d["mask"] == []


def test_legacy_default_helpers() -> None:
    s = legacy_default_setting_dict()
    assert s["name"] == "default_setting"
    assert s["renderers"] == []
    r = legacy_default_renderer_dict()
    assert r["name"] == "default_renderer"


def test_dream_settings_mask_roundtrip() -> None:
    arr = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    raw = {
        "name": "m",
        "iterations": 1,
        "octaves": 1,
        "octave_scale": 1.0,
        "iteration_descent": 0,
        "save_gradient": False,
        "background": [0, 0, 0],
        "color_correction": False,
        "cc_vars": [1, 4, 4, 4],
        "renderers": [
            {
                "name": "mr",
                "mask_name": "m.png",
                "mask": arr.tolist(),
            }
        ],
    }
    ds = DreamSettings.from_mapping(raw)
    assert ds.renderers[0].mask_name == "m.png"
    assert np.asarray(ds.renderers[0].mask).shape == arr.shape
