"""Tests for dream_cli.sequence: keyframe loading, interpolation, transforms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dream_cli.models import DreamSettings, RendererConfig, SequenceParams
from dream_cli.sequence import (
    apply_transforms,
    find_surrounding_keyframes,
    generate_noise_image,
    interpolate,
    load_keyframes,
    SequenceKeyframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    iterations: int = 10,
    octave_scale: float = 1.5,
    step_size: float = 1.0,
    layer: str = "conv2d1_pre_relu",
    zoom: float = 1.0,
    rotation_deg: float = 0.0,
) -> DreamSettings:
    renderer = RendererConfig(layer=layer, step_size=step_size)
    seq = SequenceParams(zoom=zoom, rotation_deg=rotation_deg)
    return DreamSettings(
        iterations=iterations,
        octave_scale=octave_scale,
        renderers=[renderer],
        sequence=seq,
    )


def _make_kf(
    frame: int,
    iterations: int = 10,
    octave_scale: float = 1.5,
    step_size: float = 1.0,
    layer: str = "conv2d1_pre_relu",
    zoom: float = 1.0,
    rotation_deg: float = 0.0,
) -> SequenceKeyframe:
    settings = _make_settings(
        iterations=iterations,
        octave_scale=octave_scale,
        step_size=step_size,
        layer=layer,
        zoom=zoom,
        rotation_deg=rotation_deg,
    )
    return SequenceKeyframe(frame=frame, settings=settings)


# ---------------------------------------------------------------------------
# SequenceParams round-trip
# ---------------------------------------------------------------------------

class TestSequenceParamsRoundTrip:
    def test_to_stored_dict_defaults(self):
        sp = SequenceParams()
        d = sp.to_stored_dict()
        assert d == {"zoom": 1.0, "rotation_deg": 0.0, "translate_x": 0.0, "translate_y": 0.0}

    def test_from_mapping_round_trip(self):
        sp = SequenceParams(zoom=1.05, rotation_deg=3.0, translate_x=2.0, translate_y=-1.0)
        sp2 = SequenceParams.from_mapping(sp.to_stored_dict())
        assert sp2.zoom == pytest.approx(1.05)
        assert sp2.rotation_deg == pytest.approx(3.0)
        assert sp2.translate_x == pytest.approx(2.0)
        assert sp2.translate_y == pytest.approx(-1.0)

    def test_from_mapping_partial_uses_defaults(self):
        sp = SequenceParams.from_mapping({"zoom": 2.0})
        assert sp.zoom == pytest.approx(2.0)
        assert sp.rotation_deg == pytest.approx(0.0)

    def test_dream_settings_sequence_round_trip(self):
        ds = _make_settings(zoom=1.02, rotation_deg=5.0)
        d = ds.to_stored_dict()
        ds2 = DreamSettings.from_mapping(d)
        assert ds2.sequence is not None
        assert ds2.sequence.zoom == pytest.approx(1.02)
        assert ds2.sequence.rotation_deg == pytest.approx(5.0)

    def test_dream_settings_no_sequence(self):
        ds = DreamSettings.default_new()
        ds.sequence = None
        d = ds.to_stored_dict()
        assert "sequence" not in d
        ds2 = DreamSettings.from_mapping(d)
        assert ds2.sequence is None


# ---------------------------------------------------------------------------
# load_keyframes
# ---------------------------------------------------------------------------

class TestLoadKeyframes:
    def test_loads_sorted_by_frame(self, tmp_path: Path):
        for frame, iters in [(30, 15), (0, 10), (50, 20)]:
            ds = _make_settings(iterations=iters)
            (tmp_path / f"{frame}.json").write_text(
                json.dumps(ds.to_stored_dict()), encoding="utf-8"
            )
        kfs = load_keyframes(tmp_path)
        assert [kf.frame for kf in kfs] == [0, 30, 50]

    def test_ignores_non_integer_stems(self, tmp_path: Path):
        for frame in [0, 10]:
            ds = _make_settings()
            (tmp_path / f"{frame}.json").write_text(
                json.dumps(ds.to_stored_dict()), encoding="utf-8"
            )
        (tmp_path / "readme.json").write_text("{}", encoding="utf-8")
        kfs = load_keyframes(tmp_path)
        assert len(kfs) == 2

    def test_raises_with_single_keyframe(self, tmp_path: Path):
        ds = _make_settings()
        (tmp_path / "0.json").write_text(json.dumps(ds.to_stored_dict()), encoding="utf-8")
        with pytest.raises(ValueError, match="At least 2"):
            load_keyframes(tmp_path)

    def test_ignores_non_json_files(self, tmp_path: Path):
        for frame in [0, 10]:
            ds = _make_settings()
            (tmp_path / f"{frame}.json").write_text(
                json.dumps(ds.to_stored_dict()), encoding="utf-8"
            )
        (tmp_path / "0.txt").write_text("ignored", encoding="utf-8")
        kfs = load_keyframes(tmp_path)
        assert len(kfs) == 2


# ---------------------------------------------------------------------------
# find_surrounding_keyframes
# ---------------------------------------------------------------------------

class TestFindSurroundingKeyframes:
    @pytest.fixture()
    def kfs(self):
        return [_make_kf(0), _make_kf(30), _make_kf(50)]

    def test_at_first_keyframe(self, kfs):
        a, b = find_surrounding_keyframes(0, kfs)
        assert a.frame == 0 and b.frame == 0

    def test_at_last_keyframe(self, kfs):
        a, b = find_surrounding_keyframes(50, kfs)
        assert a.frame == 50 and b.frame == 50

    def test_between_first_two(self, kfs):
        a, b = find_surrounding_keyframes(15, kfs)
        assert a.frame == 0 and b.frame == 30

    def test_between_last_two(self, kfs):
        a, b = find_surrounding_keyframes(40, kfs)
        assert a.frame == 30 and b.frame == 50

    def test_before_first(self, kfs):
        a, b = find_surrounding_keyframes(-5, kfs)
        assert a.frame == 0 and b.frame == 0

    def test_after_last(self, kfs):
        a, b = find_surrounding_keyframes(100, kfs)
        assert a.frame == 50 and b.frame == 50

    def test_at_middle_keyframe(self, kfs):
        a, b = find_surrounding_keyframes(30, kfs)
        assert a.frame == 0 and b.frame == 30


# ---------------------------------------------------------------------------
# interpolate — settings scalars
# ---------------------------------------------------------------------------

class TestInterpolateScalars:
    def test_t0_uses_a_settings(self):
        kf_a = _make_kf(0, iterations=10, octave_scale=1.5)
        kf_b = _make_kf(30, iterations=20, octave_scale=2.0)
        sd, _ = interpolate(kf_a, kf_b, 0.0)
        assert sd["iterations"] == 10
        assert sd["octave_scale"] == pytest.approx(1.5)

    def test_t1_uses_b_settings(self):
        kf_a = _make_kf(0, iterations=10, octave_scale=1.5)
        kf_b = _make_kf(30, iterations=20, octave_scale=2.0)
        sd, _ = interpolate(kf_a, kf_b, 1.0)
        assert sd["iterations"] == 20
        assert sd["octave_scale"] == pytest.approx(2.0)

    def test_midpoint_lerps(self):
        kf_a = _make_kf(0, iterations=10, octave_scale=1.0)
        kf_b = _make_kf(30, iterations=20, octave_scale=2.0)
        sd, _ = interpolate(kf_a, kf_b, 0.5)
        assert sd["iterations"] == 15
        assert sd["octave_scale"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# interpolate — renderer blending
# ---------------------------------------------------------------------------

class TestInterpolateRenderers:
    def test_t0_only_a_renderers(self):
        kf_a = _make_kf(0, step_size=2.0, layer="conv2d0_pre_relu")
        kf_b = _make_kf(30, step_size=4.0, layer="conv2d1_pre_relu")
        sd, _ = interpolate(kf_a, kf_b, 0.0)
        assert len(sd["renderers"]) == 1
        assert sd["renderers"][0]["step_size"] == pytest.approx(2.0)

    def test_t1_only_b_renderers(self):
        kf_a = _make_kf(0, step_size=2.0, layer="conv2d0_pre_relu")
        kf_b = _make_kf(30, step_size=4.0, layer="conv2d1_pre_relu")
        sd, _ = interpolate(kf_a, kf_b, 1.0)
        assert len(sd["renderers"]) == 1
        assert sd["renderers"][0]["step_size"] == pytest.approx(4.0)

    def test_different_layers_produce_two_renderers_midpoint(self):
        kf_a = _make_kf(0, step_size=2.0, layer="conv2d0_pre_relu")
        kf_b = _make_kf(30, step_size=4.0, layer="conv2d1_pre_relu")
        sd, _ = interpolate(kf_a, kf_b, 0.5)
        assert len(sd["renderers"]) == 2
        steps = sorted(r["step_size"] for r in sd["renderers"])
        assert steps[0] == pytest.approx(1.0)
        assert steps[1] == pytest.approx(2.0)

    def test_same_layer_produces_single_lerped_renderer(self):
        kf_a = _make_kf(0, step_size=1.0, layer="conv2d1_pre_relu")
        kf_b = _make_kf(30, step_size=3.0, layer="conv2d1_pre_relu")
        sd, _ = interpolate(kf_a, kf_b, 0.5)
        assert len(sd["renderers"]) == 1
        assert sd["renderers"][0]["step_size"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# interpolate — transforms
# ---------------------------------------------------------------------------

class TestInterpolateTransforms:
    def test_zoom_lerps(self):
        kf_a = _make_kf(0, zoom=1.0)
        kf_b = _make_kf(30, zoom=1.3)
        _, tr = interpolate(kf_a, kf_b, 0.5)
        assert tr.zoom == pytest.approx(1.15)

    def test_rotation_lerps(self):
        kf_a = _make_kf(0, rotation_deg=0.0)
        kf_b = _make_kf(30, rotation_deg=30.0)
        _, tr = interpolate(kf_a, kf_b, 1 / 30)
        assert tr.rotation_deg == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_transforms
# ---------------------------------------------------------------------------

class TestApplyTransforms:
    @pytest.fixture()
    def canvas(self):
        rng = np.random.default_rng(0)
        return np.float32(rng.integers(0, 256, size=(64, 64, 3)))

    def test_identity_no_change(self, canvas):
        out = apply_transforms(canvas, SequenceParams())
        np.testing.assert_array_equal(out, canvas)

    def test_zoom_in_preserves_shape(self, canvas):
        out = apply_transforms(canvas, SequenceParams(zoom=1.1))
        assert out.shape == canvas.shape

    def test_rotation_preserves_shape(self, canvas):
        out = apply_transforms(canvas, SequenceParams(rotation_deg=45.0))
        assert out.shape == canvas.shape

    def test_translation_preserves_shape(self, canvas):
        out = apply_transforms(canvas, SequenceParams(translate_x=5.0, translate_y=-3.0))
        assert out.shape == canvas.shape

    def test_combined_transform_preserves_shape(self, canvas):
        out = apply_transforms(
            canvas, SequenceParams(zoom=1.05, rotation_deg=10.0, translate_x=2.0)
        )
        assert out.shape == canvas.shape

    def test_zoom_changes_content(self, canvas):
        out = apply_transforms(canvas, SequenceParams(zoom=2.0))
        assert not np.array_equal(out, canvas)

    def test_rotation_changes_content(self, canvas):
        out = apply_transforms(canvas, SequenceParams(rotation_deg=90.0))
        assert not np.array_equal(out, canvas)


# ---------------------------------------------------------------------------
# generate_noise_image
# ---------------------------------------------------------------------------

class TestGenerateNoiseImage:
    def test_shape(self):
        img = generate_noise_image(32, 48)
        assert img.shape == (32, 48, 3)

    def test_dtype(self):
        img = generate_noise_image(32, 32)
        assert img.dtype == np.float32

    def test_range(self):
        img = generate_noise_image(64, 64)
        assert img.min() >= 0
        assert img.max() <= 255

    def test_reproducible_with_seed(self):
        a = generate_noise_image(32, 32, seed=7)
        b = generate_noise_image(32, 32, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_noise_image(32, 32, seed=1)
        b = generate_noise_image(32, 32, seed=2)
        assert not np.array_equal(a, b)
