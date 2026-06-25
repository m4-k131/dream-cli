"""Sequence rendering helpers: keyframe loading, interpolation, and canvas transforms."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dream_cli.models import DreamSettings, RendererConfig, SequenceParams


@dataclass
class SequenceKeyframe:
    frame: int
    settings: DreamSettings

    @property
    def sequence(self) -> SequenceParams:
        return self.settings.sequence if self.settings.sequence is not None else SequenceParams()


def load_keyframes(keyframe_dir: Path) -> list[SequenceKeyframe]:
    """Load all *.json files from *keyframe_dir* whose stem is a valid integer.

    Returns keyframes sorted ascending by frame index.
    Raises ValueError if fewer than 2 keyframes are found.
    """
    keyframes: list[SequenceKeyframe] = []
    for path in sorted(keyframe_dir.iterdir()):
        if path.suffix.lower() != ".json":
            continue
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        settings = DreamSettings.from_mapping(raw)
        keyframes.append(SequenceKeyframe(frame=frame_index, settings=settings))

    keyframes.sort(key=lambda kf: kf.frame)

    if len(keyframes) < 2:
        raise ValueError(
            f"At least 2 keyframe JSON files are required in '{keyframe_dir}', found {len(keyframes)}."
        )
    return keyframes


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_int(a: int, b: int, t: float) -> int:
    return int(round(_lerp(float(a), float(b), t)))


def _lerp_cc_vars(a: list[Any], b: list[Any], t: float) -> list[Any]:
    """Interpolate cc_vars: method (index 0) snaps to closer keyframe; RGB floats lerp."""
    method = a[0] if t < 0.5 else b[0]
    r = _lerp(float(a[1]), float(b[1]), t)
    g = _lerp(float(a[2]), float(b[2]), t)
    bv = _lerp(float(a[3]), float(b[3]), t)
    return [method, r, g, bv]


def _interpolate_sequence_params(a: SequenceParams, b: SequenceParams, t: float) -> SequenceParams:
    return SequenceParams(
        zoom=_lerp(a.zoom, b.zoom, t),
        rotation_deg=_lerp(a.rotation_deg, b.rotation_deg, t),
        translate_x=_lerp(a.translate_x, b.translate_x, t),
        translate_y=_lerp(a.translate_y, b.translate_y, t),
    )


def _scale_renderer_step(r: RendererConfig, weight: float) -> dict[str, Any]:
    """Return a runtime dict for *r* with step_size scaled by *weight*."""
    d = r.to_runtime_dict()
    d["step_size"] = float(d.get("step_size", 1.0)) * weight
    return d


def _interpolate_renderer_pair(
    ra: RendererConfig,
    rb: RendererConfig,
    t: float,
) -> dict[str, Any]:
    """Return a single merged renderer dict that lerps numeric fields and uses weight-scaled step."""
    d = ra.to_runtime_dict()
    d["f_channel"] = _lerp_int(ra.f_channel, rb.f_channel, t)
    d["l_channel"] = _lerp_int(ra.l_channel, rb.l_channel, t)
    d["tile_size"] = _lerp_int(ra.tile_size, rb.tile_size, t)
    d["render_x_iteration"] = _lerp_int(ra.render_x_iteration, rb.render_x_iteration, t)
    d["step_size"] = _lerp(ra.step_size, rb.step_size, t)
    if ra.color_correction and rb.color_correction:
        d["cc_vars"] = _lerp_cc_vars(ra.cc_vars, rb.cc_vars, t)
    return d


def interpolate(
    kf_a: SequenceKeyframe,
    kf_b: SequenceKeyframe,
    t: float,
) -> tuple[dict[str, Any], SequenceParams]:
    """Interpolate between two adjacent keyframes at blend factor *t* (0 = A, 1 = B).

    Returns:
        settings_dict: runtime settings dict ready to pass to dreamer.dream_image.
        transforms: SequenceParams for the canvas transform to apply before dreaming.
    """
    sa = kf_a.settings
    sb = kf_b.settings

    transforms = _interpolate_sequence_params(kf_a.sequence, kf_b.sequence, t)

    renderers: list[dict[str, Any]] = []

    if t <= 0.0:
        renderers = [r.to_runtime_dict() for r in sa.renderers]
    elif t >= 1.0:
        renderers = [r.to_runtime_dict() for r in sb.renderers]
    else:
        weight_a = 1.0 - t
        weight_b = t

        if len(sa.renderers) == len(sb.renderers):
            for ra, rb in zip(sa.renderers, sb.renderers):
                if ra.layer == rb.layer:
                    renderers.append(_interpolate_renderer_pair(ra, rb, t))
                else:
                    if weight_a > 0.0:
                        renderers.append(_scale_renderer_step(ra, weight_a))
                    if weight_b > 0.0:
                        renderers.append(_scale_renderer_step(rb, weight_b))
        else:
            for ra in sa.renderers:
                if weight_a > 0.0:
                    renderers.append(_scale_renderer_step(ra, weight_a))
            for rb in sb.renderers:
                if weight_b > 0.0:
                    renderers.append(_scale_renderer_step(rb, weight_b))

    settings_dict: dict[str, Any] = {
        "name": sa.name,
        "iterations": _lerp_int(sa.iterations, sb.iterations, t),
        "octaves": _lerp_int(sa.octaves, sb.octaves, t),
        "octave_scale": _lerp(sa.octave_scale, sb.octave_scale, t),
        "iteration_descent": _lerp_int(sa.iteration_descent, sb.iteration_descent, t),
        "save_gradient": sa.save_gradient,
        "background": [
            _lerp_int(sa.background[i], sb.background[i], t) for i in range(3)
        ],
        "renderers": renderers,
        "color_correction": sa.color_correction or sb.color_correction,
        "cc_vars": _lerp_cc_vars(sa.cc_vars, sb.cc_vars, t),
    }

    return settings_dict, transforms


def find_surrounding_keyframes(
    frame: int,
    keyframes: list[SequenceKeyframe],
) -> tuple[SequenceKeyframe, SequenceKeyframe]:
    """Return the (prev, next) keyframes that bracket *frame*.

    At or before the first keyframe returns (kf[0], kf[0]).
    At or after the last keyframe returns (kf[-1], kf[-1]).
    """
    if frame <= keyframes[0].frame:
        return keyframes[0], keyframes[0]
    if frame >= keyframes[-1].frame:
        return keyframes[-1], keyframes[-1]
    for i in range(len(keyframes) - 1):
        if keyframes[i].frame <= frame <= keyframes[i + 1].frame:
            return keyframes[i], keyframes[i + 1]
    return keyframes[-1], keyframes[-1]


def apply_transforms(image: np.ndarray, params: SequenceParams) -> np.ndarray:
    """Apply zoom, rotation, and translation to *image* using scipy.ndimage.

    All transforms are applied around the image centre.
    Falls back to PIL/NumPy if scipy is unavailable.
    """
    if importlib.util.find_spec("scipy") is not None:
        return _apply_transforms_scipy(image, params)
    return _apply_transforms_pil(image, params)


def _apply_transforms_scipy(image: np.ndarray, params: SequenceParams) -> np.ndarray:
    """scipy.ndimage-based implementation of zoom + rotation + translation."""
    from scipy.ndimage import affine_transform  # noqa: PLC0415

    if (
        params.zoom == 1.0
        and params.rotation_deg == 0.0
        and params.translate_x == 0.0
        and params.translate_y == 0.0
    ):
        return image

    h, w = image.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    angle_rad = np.deg2rad(-params.rotation_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    s = 1.0 / params.zoom if params.zoom != 0.0 else 1.0

    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) * s

    tx = params.translate_x
    ty = params.translate_y

    offset = np.array([cy, cx]) - rot @ np.array([cy - ty, cx - tx])

    output = np.empty_like(image)
    for c in range(image.shape[2]):
        output[:, :, c] = affine_transform(
            image[:, :, c],
            rot,
            offset=offset,
            output_shape=(h, w),
            order=1,
            mode="reflect",
        )
    return output


def _apply_transforms_pil(image: np.ndarray, params: SequenceParams) -> np.ndarray:
    """PIL fallback for zoom + rotation + translation."""
    from PIL import Image  # noqa: PLC0415

    pil = Image.fromarray(np.uint8(np.clip(image / 255.0, 0, 1) * 255))
    w, h = pil.size

    if params.zoom != 1.0 and params.zoom > 0.0:
        new_w = int(round(w * params.zoom))
        new_h = int(round(h * params.zoom))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        pil = pil.crop((left, top, left + w, top + h))

    if params.rotation_deg != 0.0:
        pil = pil.rotate(params.rotation_deg, resample=Image.BILINEAR, expand=False)

    if params.translate_x != 0.0 or params.translate_y != 0.0:
        pil = pil.transform(
            (w, h),
            Image.AFFINE,
            (1, 0, -params.translate_x, 0, 1, -params.translate_y),
            resample=Image.BILINEAR,
        )

    return np.float32(pil)


def generate_noise_image(height: int, width: int, seed: int | None = None) -> np.ndarray:
    """Generate an RGB noise image of shape (H, W, 3) in [0, 255] float32."""
    rng = np.random.default_rng(seed)
    return np.float32(rng.integers(0, 256, size=(height, width, 3)))
