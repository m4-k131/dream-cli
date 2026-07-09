"""Smoke tests for dreamer2 and dream_cli.model_registry.

These tests use tiny synthetic images so they run without a GPU and without
downloading ImageNet weights (weights=None).  Full model quality is not
verified — only shapes, types, and control-flow paths.

The entire module is skipped when TensorFlow is not installed (e.g. in CI).
"""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed — skipping dreamer2 tests")

from dream_cli.model_registry import (  # noqa: E402
    default_layers_for,
    known_model_names,
    list_layer_names,
    load_feature_extractor,
)


# ---------------------------------------------------------------------------
# model_registry tests
# ---------------------------------------------------------------------------

def test_known_model_names_nonempty() -> None:
    names = known_model_names()
    assert len(names) > 0
    assert "InceptionV3" in names


def test_default_layers_for_inceptionv3() -> None:
    layers = default_layers_for("InceptionV3")
    assert isinstance(layers, list)
    assert len(layers) > 0


def test_default_layers_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        default_layers_for("NonExistentModel")


def test_list_layer_names_returns_strings() -> None:
    names = list_layer_names("InceptionV3")
    assert all(isinstance(n, str) for n in names)
    assert len(names) > 0


def test_load_feature_extractor_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        load_feature_extractor("FakeModel")


def test_load_feature_extractor_unknown_layer() -> None:
    with pytest.raises(ValueError, match="Layer"):
        load_feature_extractor("InceptionV3", layer_names=["this_layer_does_not_exist"], weights=None)


def test_load_feature_extractor_inceptionv3(tmp_path) -> None:
    model, preprocess_fn = load_feature_extractor(
        "InceptionV3",
        layer_names=["mixed3"],
        weights=None,
    )
    assert isinstance(model, tf.keras.Model)
    assert callable(preprocess_fn)
    # Model should not be trainable
    assert model.trainable is False


def test_feature_extractor_output_shape() -> None:
    """Feature model should forward-pass without error and return activation tensors."""
    model, preprocess_fn = load_feature_extractor(
        "InceptionV3",
        layer_names=["mixed3"],
        weights=None,
    )
    # InceptionV3 minimum input is 75×75; use that to keep test fast
    dummy = np.random.randint(0, 255, (1, 75, 75, 3), dtype=np.uint8).astype(np.float32)
    preprocessed = preprocess_fn(dummy)
    out = model(preprocessed)
    # Single-layer extractor may return a tensor or a list with one tensor
    if isinstance(out, list):
        assert len(out) == 1
        assert out[0].ndim == 4
    else:
        assert out.ndim == 4


# ---------------------------------------------------------------------------
# dreamer2 tests
# ---------------------------------------------------------------------------

# Minimal settings dict — valid against the existing schema
def _minimal_settings(octaves: int = 1, iterations: int = 2) -> dict:
    return {
        "name": "test",
        "iterations": iterations,
        "octaves": octaves,
        "octave_scale": 1.3,
        "iteration_descent": 0,
        "save_gradient": False,
        "background": [0, 0, 0],
        "color_correction": False,
        "cc_vars": [1, 4, 4, 4],
        "renderers": [
            {
                "name": "r1",
                "layer": "mixed3",
                "f_channel": 0,
                "l_channel": 4,
                "squared": True,
                "cropped": False,
                "boundraries": [[0.0, 1.0], [0.0, 1.0]],
                "rotate": False,
                "rotation": 0,
                "tile_size": 75,
                "render_x_iteration": 1,
                "step_size": 0.1,
                "color_correction": False,
                "cc_vars": [1, 4, 4, 4],
                "masked": False,
                "mask": [],
                "mask_name": "",
                "t_masks": [],
            }
        ],
    }


def _make_image(h: int = 80, w: int = 80) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3)).astype(np.float32)


def test_dreamer2_import() -> None:
    from dreamer2 import Dreamer2  # noqa: PLC0415
    assert Dreamer2 is not None


def test_dreamer2_init() -> None:
    from dreamer2 import Dreamer2  # noqa: PLC0415
    d = Dreamer2(model_name="InceptionV3", layer_names=["mixed3"], weights=None)
    assert d.model_name == "InceptionV3"


def test_dreamer2_dream_image_smoke(tmp_path, monkeypatch) -> None:
    """Full pipeline smoke test: 1 octave, 2 iterations, 80×80 image."""
    import utils as u  # noqa: PLC0415

    saved: list[tuple[np.ndarray, str]] = []

    def _fake_save(img: np.ndarray, filename: str) -> None:
        saved.append((img, filename))

    monkeypatch.setattr(u, "save_image", _fake_save)

    from dreamer2 import Dreamer2  # noqa: PLC0415

    d = Dreamer2(model_name="InceptionV3", layer_names=["mixed3"], weights=None)
    img = _make_image(80, 80)
    d.dream_image(img, _minimal_settings(octaves=1, iterations=2), "smoke_test")

    assert len(saved) == 1
    out_img, out_name = saved[0]
    assert out_img.shape == img.shape, "Output shape must match input shape"
    assert out_name == "smoke_test"


def test_dreamer2_save_gradient(monkeypatch) -> None:
    """save_gradient flag should produce two save calls."""
    import utils as u  # noqa: PLC0415

    saved: list[str] = []

    def _fake_save(img: np.ndarray, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(u, "save_image", _fake_save)

    from dreamer2 import Dreamer2  # noqa: PLC0415

    settings = _minimal_settings(octaves=1, iterations=1)
    settings["save_gradient"] = True

    d = Dreamer2(model_name="InceptionV3", layer_names=["mixed3"], weights=None)
    d.dream_image(_make_image(), settings, "grad_test")

    assert len(saved) == 2
    assert any("gradient_" in s for s in saved)


def test_dreamer2_module_level_dream_image(monkeypatch) -> None:
    """Module-level dream_image shim should work."""
    import utils as u  # noqa: PLC0415
    import dreamer2  # noqa: PLC0415

    saved: list[str] = []

    def _fake_save(img: np.ndarray, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(u, "save_image", _fake_save)

    # Reset module singleton so weights=None path is exercised cleanly
    dreamer2._instance = None  # noqa: SLF001

    # Patch load_feature_extractor to avoid full InceptionV3 download
    import dream_cli.model_registry as reg  # noqa: PLC0415

    _orig = reg.load_feature_extractor

    def _patched(model_name, layer_names=None, weights="imagenet", **kw):
        return _orig(model_name, layer_names=layer_names or ["mixed3"], weights=None, **kw)

    monkeypatch.setattr(reg, "load_feature_extractor", _patched)

    dreamer2.dream_image(_make_image(), _minimal_settings(octaves=1, iterations=1), "shim_test")

    assert "shim_test" in saved
