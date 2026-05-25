# -*- coding: utf-8 -*-
"""Non-interactive entry point: run `python dream.py --image <path> --renderer <path>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import utils as utils_mod
from dream_cli.models import DreamSettings, RendererConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepDream a single image with a renderer JSON file (non-interactive)."
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        help="Path to the input image (jpg/jpeg).",
    )
    parser.add_argument(
        "--renderer", "-r",
        type=Path,
        required=True,
        help="Path to a renderer JSON file (e.g. Settings/Renderer/c21_lines_r.json).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename basename (default: derived from input image name).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations (default: 20).",
    )
    parser.add_argument(
        "--octaves",
        type=int,
        default=4,
        help="Number of octaves (default: 4).",
    )
    parser.add_argument(
        "--octave-scale",
        type=float,
        default=1.5,
        help="Octave scale factor (default: 1.5).",
    )
    parser.add_argument(
        "--iteration-descent",
        type=int,
        default=0,
        help="Iteration descent per octave (default: 0).",
    )
    parser.add_argument(
        "--save-gradient",
        action="store_true",
        help="Also save the gradient image.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="0,0,0",
        help="Background color as R,G,B (default: 0,0,0).",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Override step_size in the renderer (default: use value from JSON).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Override tile_size in the renderer (default: use value from JSON).",
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")
    if not args.renderer.is_file():
        parser.error(f"Renderer file not found: {args.renderer}")

    # Load image
    print(f"Loading image: {args.image}")
    image = utils_mod.load_image(str(args.image))

    # Load renderer from JSON
    print(f"Loading renderer: {args.renderer}")
    raw = json.loads(args.renderer.read_text(encoding="utf-8"))
    renderer = RendererConfig.from_mapping(raw)

    # Apply CLI overrides to renderer
    if args.step_size is not None:
        renderer.step_size = args.step_size
    if args.tile_size is not None:
        renderer.tile_size = args.tile_size

    # Parse background
    bg_parts = args.background.split(",")
    if len(bg_parts) != 3:
        parser.error("--background must be in R,G,B format (e.g. 0,0,0)")
    background = [int(x.strip()) for x in bg_parts]

    # Build settings
    settings = DreamSettings(
        name="cli_run",
        iterations=args.iterations,
        octaves=args.octaves,
        octave_scale=args.octave_scale,
        iteration_descent=args.iteration_descent,
        save_gradient=args.save_gradient,
        background=background,
        renderers=[renderer],
    )

    # Determine output name
    output_name = args.output or args.image.stem

    # Run dream
    import dreamer as dreamer_mod

    dreamer_mod.close_and_reopen_session()
    dreamer_mod.dream_image(image, settings.to_runtime_dreamer_dict(), output_name)
    dreamer_mod.close_session()

    print(f"Done. Output saved to renderedImages/{output_name}+.jpg")


if __name__ == "__main__":
    main()
