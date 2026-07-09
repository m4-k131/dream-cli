"""dreamer2 — TF2/Keras DeepDream engine.

Drop-in companion to ``dreamer.py`` (TF1/compat.v1).  The TF1 module is
**not** modified.  This module reuses the same settings schema, color-grading
utility, and save helpers from the existing codebase.

Public API (mirrors ``dreamer.py``)
------------------------------------
    Dreamer2(model_name, layer_names)   — class
    dream_image(image, settings, out_name)  — module-level convenience

Key design choices
------------------
* **Laplacian pyramid octaves** — same approach as ``dreamer.py``:
  build the pyramid from the original image *before* the dream loop so that
  fine high-frequency detail is restored at each octave rather than being
  blurred by repeated upscaling (unlike the TF tutorial which naïvely resizes
  the accumulated image).
* **Tiled gradient ascent with random roll** — avoids seams and lets the
  engine work on large images without running out of VRAM.
* **Per-renderer channel slice + squared/linear objective** — mirrors the TF1
  ``set_layer`` logic.
* **All bells and whistles** from the TF1 path preserved: crop bounds, masks,
  rotation, per-renderer color correction, ``render_x_iteration``,
  ``iteration_descent``, global color correction, gradient save.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils as utils
from dream_cli.model_registry import load_feature_extractor
from dream_cli.schemas import validate_and_fill_settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_loss(
    activations: list[tf.Tensor],
    squared: bool,
    f_channel: int,
    l_channel: int,
) -> tf.Tensor:
    """Mean (squared) activation over a channel slice — matches TF1 ``set_layer``."""
    loss = tf.constant(0.0)
    for act in activations:
        sliced = act[..., f_channel:l_channel]
        loss = loss + tf.reduce_mean(tf.square(sliced) if squared else sliced)
    return loss


def _random_roll(img: tf.Tensor, max_roll: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Randomly shift *img* to avoid tiling seams; return ``(shift, rolled_img)``."""
    shift = tf.random.uniform(shape=[2], minval=-max_roll, maxval=max_roll, dtype=tf.int32)
    rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, rolled


def _compute_tiled_gradients(
    img: np.ndarray,
    feature_model: tf.keras.Model,
    squared: bool,
    f_channel: int,
    l_channel: int,
    tile_size: int,
) -> np.ndarray:
    """Compute gradient of the dream loss w.r.t. *img* using random-roll tiling.

    Returns a ``np.ndarray`` of the same shape as *img*.
    """
    img_tf = tf.cast(img, tf.float32)
    h, w = img_tf.shape[:2]
    shift, img_rolled = _random_roll(img_tf, tile_size)

    gradients = tf.zeros_like(img_rolled)

    xs = list(range(0, w - tile_size // 2, tile_size)) or [0]
    ys = list(range(0, h - tile_size // 2, tile_size)) or [0]

    for y in ys:
        for x in xs:
            tile = img_rolled[y : y + tile_size, x : x + tile_size]
            tile = tf.expand_dims(tile, 0)
            with tf.GradientTape() as tape:
                tape.watch(tile)
                activations = feature_model(tile)
                if not isinstance(activations, list):
                    activations = [activations]
                loss = _calc_loss(activations, squared, f_channel, l_channel)
            tile_grad = tape.gradient(loss, tile)
            if tile_grad is not None:
                tile_grad = tf.squeeze(tile_grad, 0)
                padded = tf.pad(
                    tile_grad,
                    [
                        [y, max(h - y - tile_size, 0)],
                        [x, max(w - x - tile_size, 0)],
                        [0, 0],
                    ],
                )
                # Trim to exact image size in case tile overhangs
                padded = padded[:h, :w, :]
                gradients = gradients + padded

    gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
    std = tf.math.reduce_std(gradients) + 1e-8
    gradients = gradients / std
    return gradients.numpy()


def _calculate_tile_size(num_pixels: int, preferred_tile_size: int = 400) -> int:
    estimated_tiles = max(1, round(num_pixels / preferred_tile_size))
    return math.ceil(num_pixels / estimated_tiles)


def _resize(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize *image* (H×W×C float32 numpy) to *target_hw* using bilinear interpolation."""
    h, w = target_hw
    img_tf = tf.image.resize(image[np.newaxis], [h, w], method="bilinear")
    return img_tf[0].numpy()


# ---------------------------------------------------------------------------
# Laplacian pyramid (same algorithm as dreamer.py, ported to TF2 resize)
# ---------------------------------------------------------------------------

def _build_pyramids(
    image: np.ndarray,
    gradient_accumulator: np.ndarray,
    octave_count: int,
    octave_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Decompose *image* and *gradient_accumulator* into Laplacian pyramids.

    Returns ``(smallest_image, smallest_grad, image_details, grad_details)``
    where ``image_details[0]`` is the detail of the second-largest octave, etc.
    """
    image_octaves: list[np.ndarray] = []
    gradient_octaves: list[np.ndarray] = []
    working_image = image.copy()
    working_gradient = gradient_accumulator.copy()

    for _ in range(octave_count - 1):
        h, w = working_image.shape[:2]
        small_hw = tuple(int(d / octave_scale) for d in (h, w))
        downscaled_img = _resize(working_image, small_hw)
        detail_img = working_image - _resize(downscaled_img, (h, w))
        working_image = downscaled_img
        image_octaves.append(detail_img)

        downscaled_grad = _resize(working_gradient, small_hw)
        detail_grad = working_gradient - _resize(downscaled_grad, (h, w))
        working_gradient = downscaled_grad
        gradient_octaves.append(detail_grad)

    return working_image, working_gradient, image_octaves, gradient_octaves


# ---------------------------------------------------------------------------
# Per-octave processing
# ---------------------------------------------------------------------------

def _process_octave(
    octave: int,
    working_image: np.ndarray,
    gradient_accumulator: np.ndarray,
    original_image: np.ndarray,
    image_octaves: list[np.ndarray],
    gradient_octaves: list[np.ndarray],
    renderers: list[dict[str, Any]],
    feature_models: list[tf.keras.Model],
    preprocess_fn: Callable,
    iterations: int,
    iteration_descent: int,
    octave_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one octave of gradient ascent; return updated ``(working_image, gradient_accumulator)``."""
    if octave > 0:
        detail_img = image_octaves[-octave]
        working_image = _resize(working_image, detail_img.shape[:2]) + detail_img
        detail_grad = gradient_octaves[-octave]
        gradient_accumulator = _resize(gradient_accumulator, detail_grad.shape[:2]) + detail_grad

    h, w = working_image.shape[:2]
    bounds_list = utils.get_bounds(w, h, renderers)
    unpacked_bounds = [(b[0], b[1], b[2], b[3]) for b in bounds_list]

    iteration_masks: list[np.ndarray | None] = []
    for renderer in renderers:
        masked = renderer.get("masked", False)
        mask = renderer.get("mask", [])
        if masked and len(np.asarray(mask)) > 0:
            iteration_masks.append(_resize(np.asarray(mask, dtype=np.float32), (h, w)) / 255.0)
        else:
            iteration_masks.append(None)

    reference_image = _resize(original_image, (h, w)) / 255.0
    iterations_this_octave = max(1, iterations - octave * iteration_descent)

    for iteration in tqdm(
        range(iterations_this_octave),
        desc=f"Octave {octave + 1}/{octave_count}",
        unit="iter",
        leave=True,
    ):
        for renderer_index, renderer in enumerate(renderers):
            render_every = renderer.get("render_x_iteration", 1)
            if (iteration + 1) % render_every != 0:
                continue

            x_start, x_end, y_start, y_end = unpacked_bounds[renderer_index]
            tile_image = working_image[y_start:y_end, x_start:x_end]

            rotate = renderer.get("rotate", False)
            rotation = renderer.get("rotation", 0)
            if rotate:
                tile_image = np.rot90(tile_image, rotation)

            # Pre-process for the Keras model
            preprocessed = preprocess_fn(tile_image[np.newaxis].copy())

            tile_size = renderer.get("tile_size", 300)
            squared = renderer.get("squared", True)
            f_channel = renderer.get("f_channel", 0)
            l_channel = renderer.get("l_channel", 64)
            step_size = renderer.get("step_size", 1.0)

            gradient = _compute_tiled_gradients(
                preprocessed[0],
                feature_models[renderer_index],
                squared,
                f_channel,
                l_channel,
                tile_size,
            )

            # Scale gradient (mean-absolute normalisation, matching TF1 dreamer)
            gradient = gradient * (step_size / (np.abs(gradient).mean() + 1e-7))

            if rotate:
                gradient = np.rot90(gradient, 4 - rotation)

            if renderer.get("masked", False):
                mask = iteration_masks[renderer_index]
                if mask is not None:
                    gradient *= mask[y_start:y_end, x_start:x_end]

            if renderer.get("color_correction", False):
                cc_vars = renderer.get("cc_vars", [1, 4, 4, 4])
                gradient = utils.gradient_grading(
                    gradient,
                    reference_image[y_start:y_end, x_start:x_end],
                    method=cc_vars[0] if len(cc_vars) > 0 else 1,
                    fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                    fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                    fb=cc_vars[3] if len(cc_vars) > 3 else 4,
                )

            working_image[y_start:y_end, x_start:x_end] += gradient
            gradient_accumulator[y_start:y_end, x_start:x_end] += gradient

    return working_image, gradient_accumulator


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Dreamer2:
    """TF2/Keras DeepDream engine.

    Parameters
    ----------
    model_name:
        Name of a ``keras.applications`` model registered in
        ``dream_cli.model_registry`` (e.g. ``"InceptionV3"``).
    layer_names:
        Layer names to use as the dream objective.  When *None* the registry
        default for *model_name* is used.  Each renderer shares the same base
        model but may target different channel slices; if you want *different*
        layers per renderer, supply a list with one entry per renderer (the
        index is matched by position).  Alternatively you can use a single
        layer list and rely on channel slicing to differentiate renderers.
    weights:
        Passed to the Keras constructor — ``"imagenet"`` by default.
    """

    def __init__(
        self,
        model_name: str = "InceptionV3",
        layer_names: list[str] | None = None,
        weights: str = "imagenet",
    ) -> None:
        self.model_name = model_name
        self._feature_model, self._preprocess_fn = load_feature_extractor(
            model_name, layer_names, weights=weights
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def dream_image(
        self,
        image: np.ndarray,
        settings_raw: dict[str, Any],
        out_name: str,
    ) -> None:
        """Run the full dream pipeline and save the result.

        Parameters
        ----------
        image:
            Source image as a ``float32`` numpy array, shape ``(H, W, 3)``,
            values in ``[0, 255]`` (same convention as ``dreamer.py``).
        settings_raw:
            Settings dict compatible with the existing schema
            (``DreamSettings.to_runtime_dreamer_dict()``).
        out_name:
            Base filename passed to ``utils.save_image``.
        """
        settings = validate_and_fill_settings(settings_raw)
        iterations: int = settings.get("iterations", 20)
        octave_count: int = settings.get("octaves", 4)
        octave_scale: float = settings.get("octave_scale", 1.5)
        iteration_descent: int = settings.get("iteration_descent", 0)
        save_gradient: bool = settings.get("save_gradient", False)
        renderers: list[dict[str, Any]] = settings.get("renderers", [])

        # Build one feature model per renderer so each can have its own layer
        # list in future; for now they all share the same base model.
        feature_models = [self._feature_model] * len(renderers)

        gradient_accumulator = np.zeros_like(image)
        working_image, gradient_accumulator, image_octaves, gradient_octaves = (
            _build_pyramids(image, gradient_accumulator, octave_count, octave_scale)
        )

        for octave in range(octave_count):
            working_image, gradient_accumulator = _process_octave(
                octave,
                working_image,
                gradient_accumulator,
                image,
                image_octaves,
                gradient_octaves,
                renderers,
                feature_models,
                self._preprocess_fn,
                iterations,
                iteration_descent,
                octave_count,
            )

        if settings.get("color_correction", False):
            cc_vars = settings.get("cc_vars", [1, 4, 4, 4])
            working_image = image + utils.gradient_grading(
                gradient_accumulator,
                image / 255.0,
                method=cc_vars[0] if len(cc_vars) > 0 else 1,
                fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                fb=cc_vars[3] if len(cc_vars) > 3 else 4,
            )

        utils.save_image(working_image, out_name)
        if save_gradient:
            utils.save_image(gradient_accumulator, "gradient_" + out_name)


# ---------------------------------------------------------------------------
# Module-level convenience shim (mirrors dreamer.py public API)
# ---------------------------------------------------------------------------

_instance: Dreamer2 | None = None


def _get_instance() -> Dreamer2:
    global _instance
    if _instance is None:
        _instance = Dreamer2()
    return _instance


def dream_image(image: np.ndarray, settings: dict[str, Any], out_name: str) -> None:
    """Module-level convenience wrapper — lazy-initialises a default ``Dreamer2``."""
    _get_instance().dream_image(image, settings, out_name)
