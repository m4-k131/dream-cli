"""Sequence renderer entry point: `render-sequence` console script.

Usage example::

    render-sequence \\
        --keyframes Sequences/my_sequence/ \\
        --seed      Images/start.jpg \\
        --output    renderedImages/my_sequence/ \\
        [--noise-seed 42] \\
        [--noise-size 512 512] \\
        [--resume 10]

Keyframe files must be named by their frame index, e.g. ``0.json``, ``30.json``,
``50.json``. The script renders every integer frame from the first to the last
keyframe index (inclusive).

The dreamer session is opened once and reused across all frames (with a session
refresh between frames to flush GPU memory, matching the existing single-image
workflow).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _save_frame(image: np.ndarray, output_dir: Path, frame: int) -> None:
    from PIL import Image  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    a = np.uint8(np.clip(image / 255.0, 0, 1) * 255)
    filename = output_dir / f"frame_{frame:05d}.jpg"
    Image.fromarray(a).save(str(filename), "JPEG")


def _load_seed(args: argparse.Namespace) -> np.ndarray:
    from dream_cli.sequence import generate_noise_image  # noqa: PLC0415

    if args.seed is not None:
        seed_path = Path(args.seed)
        if not seed_path.is_file():
            print(f"ERROR: seed image not found: {seed_path}", file=sys.stderr)
            sys.exit(1)
        from PIL import Image  # noqa: PLC0415

        return np.float32(Image.open(seed_path))

    h, w = args.noise_size
    noise_seed = args.noise_seed if args.noise_seed is not None else None
    print(f"No seed image supplied — generating {w}×{h} noise canvas.")
    return generate_noise_image(h, w, seed=noise_seed)


def _dream_frame(
    canvas: np.ndarray,
    settings_dict: dict[str, Any],
    frame: int,
    output_dir: Path,
) -> np.ndarray:
    """Run dream_image on *canvas*, save the result, and return the output array.

    The dreamer writes to disk via utils.save_image; we also save a zero-padded
    copy in *output_dir* for easy ffmpeg assembly.
    """
    import dream_cli.dreamer as dreamer_mod  # noqa: PLC0415

    dreamer_mod.close_and_reopen_session()

    tmp_name = f"seq_frame_{frame:05d}"
    dreamer_mod.dream_image(canvas, settings_dict, tmp_name)
    dreamer_mod.close_session()

    rendered_path = Path("renderedImages") / "4" / f"{tmp_name}+.jpg"

    if rendered_path.is_file():
        from PIL import Image  # noqa: PLC0415

        result = np.float32(Image.open(rendered_path))
    else:
        print(
            f"  WARNING: expected output '{rendered_path}' not found; "
            "using pre-dream canvas as next-frame input.",
            file=sys.stderr,
        )
        result = canvas

    _save_frame(result, output_dir, frame)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a DeepDream sequence from keyframe JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--keyframes",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing keyframe JSON files named by frame index (e.g. 0.json, 30.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for rendered frame JPEGs.",
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=Path,
        default=None,
        metavar="IMAGE",
        help="Seed image for frame 0 (JPEG/PNG). Mutually exclusive with --noise-seed.",
    )
    seed_group.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Integer RNG seed for a random noise starting canvas.",
    )

    parser.add_argument(
        "--noise-size",
        nargs=2,
        type=int,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
        help="Canvas size (H W) when generating a noise seed (default: 512 512).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="N",
        help="Skip rendering frames with index < N (resume a partial run).",
    )

    args = parser.parse_args()

    keyframe_dir = args.keyframes.resolve()
    if not keyframe_dir.is_dir():
        print(f"ERROR: keyframe directory not found: {keyframe_dir}", file=sys.stderr)
        sys.exit(1)

    from dream_cli.sequence import (  # noqa: PLC0415
        apply_transforms,
        find_surrounding_keyframes,
        interpolate,
        load_keyframes,
    )

    print(f"Loading keyframes from: {keyframe_dir}")
    keyframes = load_keyframes(keyframe_dir)
    first_frame = keyframes[0].frame
    last_frame = keyframes[-1].frame
    total = last_frame - first_frame + 1

    print(f"Keyframes: {[kf.frame for kf in keyframes]}")
    print(f"Rendering frames {first_frame}–{last_frame} ({total} total).")
    if args.resume > first_frame:
        print(f"Resuming from frame {args.resume}.")

    canvas = _load_seed(args)

    output_dir = args.output.resolve()

    for frame in range(first_frame, last_frame + 1):
        kf_a, kf_b = find_surrounding_keyframes(frame, keyframes)

        span = kf_b.frame - kf_a.frame
        t = (frame - kf_a.frame) / span if span > 0 else 0.0

        settings_dict, transforms = interpolate(kf_a, kf_b, t)

        canvas = apply_transforms(canvas, transforms)

        if frame < args.resume:
            print(f"  [skip] frame {frame:05d}/{last_frame}")
            continue

        print(
            f"  frame {frame:05d}/{last_frame}  "
            f"[KF{kf_a.frame}→KF{kf_b.frame} t={t:.3f}]  "
            f"zoom={transforms.zoom:.3f}  rot={transforms.rotation_deg:.2f}°"
        )

        canvas = _dream_frame(canvas, settings_dict, frame, output_dir)

    print(f"\nDone. Frames written to: {output_dir}")


if __name__ == "__main__":
    main()
