# dream-cli

Interactive CLI for **DeepDream** on still images. It uses Google's frozen **Inception "5h"** graph (`tensorflow_inception_graph.pb`, fetched automatically from `inception5h.zip` the first time you run it). The core idea is gradient ascent on the input image — maximising activations (or squared activations) in a chosen layer and channel range — with multi-scale octaves, tiling, optional crops, masks, and colour controls.

## Requirements

- **Conda** (Miniconda or Anaconda)
- A CUDA-capable GPU is recommended but not required — CPU rendering works but is slower

Python and all library dependencies (TensorFlow, NumPy, Pillow, tqdm) are managed by the Conda environment defined in `environment.yml`.

## Installation

Clone the repo, then create the environment:

```bash
git clone https://github.com/your-username/dream-cli.git
cd dream-cli
conda env create -f environment.yml -p .venv
```

After the environment is created, activate it:

```bash
conda activate ./.venv
```

Check that TensorFlow sees your GPU (optional):

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

> **Windows note:** Recent pip-only TensorFlow on native Windows is often CPU-only. This project pins **TensorFlow 2.9.x** via Conda, which ships GPU-capable wheels. If you need a slightly newer version, **TensorFlow 2.10.1** also has `cp310-win_amd64` wheels — bump the pin in `environment.yml`.

To update the environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml -p .venv --prune
```

## Quick start — interactive CLI

1. Put your source images (`.jpg` / `.jpeg`) in the **`Images/`** folder (create it if it does not exist).
2. Run the CLI from the repo root:

```bash
conda activate ./.venv
python dreamcli.py
```

3. Follow the menus:
   - **Select image** — pick a JPEG from `Images/`.
   - **Edit Settings** — global parameters (octaves, iterations, colour correction, …).
   - **Edit Renderer** — one or more render passes (layer, channels, step size, crop, mask, …).
   - **Continue** — run DeepDream and save the result.

Outputs are written to **`renderedImages/`**. Saved presets live in **`Settings/*_s.json`** (full settings) and **`Settings/Renderer/*_r.json`** (individual renderers).

## Non-interactive usage

Dream a single image from the command line using a renderer JSON preset:

```bash
python dream.py \
    --image    Images/my_photo.jpg \
    --renderer Settings/Renderer/my_preset_r.json \
    --iterations 30 \
    --octaves 4
```

Run `python dream.py --help` for all options.

## Sequence rendering

Render a multi-frame DeepDream sequence from keyframe JSON files:

```bash
python render_sequence.py \
    --keyframes Sequences/my_sequence/ \
    --seed      Images/start.jpg \
    --output    renderedImages/my_sequence/
```

Keyframe files are named by frame index (e.g. `0.json`, `30.json`, `60.json`). Settings are interpolated between adjacent keyframes. Run `python render_sequence.py --help` for all options.

## Settings reference

### Global settings

| Setting | Description |
|---------|-------------|
| **Octaves** | Number of scaled resolutions; each octave adds finer detail. |
| **Octave scale** | Downscale factor between octaves (e.g. `1.5`). |
| **Iterations** | Gradient steps per octave. |
| **Iteration descent** | Amount subtracted from the iteration count each octave. |
| **Save gradient** | Write the accumulated gradient onto a solid background colour. |
| **Color correction** | Global gradient colour tinting. |

### Renderer settings (per pass)

Each renderer targets one **layer** and a **channel range** (`f_channel`–`l_channel`).

| Setting | Description |
|---------|-------------|
| **Layer** | Inception layer to optimise (204 options shown in the CLI). |
| **Channels** | First and last channel index within the layer. |
| **Squared** | Use squared activations as the objective (stronger, more pattern-like result). |
| **Step size** | Scales the gradient each step; negative values reverse the direction. |
| **Tile size** | Inception works best around 300 px; larger tiles use more VRAM. |
| **Cropped / boundaries** | Restrict rendering to a normalised region `[[x_min, x_max], [y_min, y_max]]` in `[0, 1]`. |
| **Mask** | Multiply the gradient by a greyscale mask image. |
| **Rotation** | Rotate the crop before the gradient step, then rotate back. |
| **Render every N iterations** | Compute this renderer only every N steps. |
| **Color correction** | Per-renderer gradient colour tinting. |

### Colour correction methods

| Method | Description |
|--------|-------------|
| 1 | Simple grayscale correction with RGB multipliers. |
| 2 | Retaining original image colours. |
| 3 | Linear correction with RGB multipliers. |

## Project layout

```
dream-cli/
├── src/dream_cli/          # Python package (all source)
│   ├── application.py      # DreamApplication: paths, session state, run_dream
│   ├── cli.py              # InteractiveDreamCli: terminal menus
│   ├── dreamer.py          # Dreamer class and TensorFlow graph logic
│   ├── layers.py           # Inception layer names and channel counts
│   ├── inception_layers.json
│   ├── models.py           # DreamSettings, RendererConfig dataclasses
│   ├── prompts.py          # Prompter: validated input() helpers
│   ├── schemas.py          # JSON schema validation for settings dicts
│   ├── sequence.py         # Keyframe loading, interpolation, canvas transforms
│   ├── utils.py            # Image I/O, download helper, colour grading
│   ├── legacy.py           # Backward-compatible setting defaults
│   └── _scripts/           # Console-script entry points
│       ├── dreamcli.py     # `dreamcli` / `python dreamcli.py`
│       ├── dream.py        # `dream`    / `python dream.py`
│       └── render_sequence.py  # `render-sequence` / `python render_sequence.py`
├── tests/                  # pytest test suite
├── Images/                 # Input JPEGs (user-provided, not committed)
├── Settings/               # Saved presets (*_s.json, Renderer/*_r.json)
├── dreamcli.py             # Thin shim — forwards to src entry point
├── dream.py                # Thin shim — forwards to src entry point
├── render_sequence.py      # Thin shim — forwards to src entry point
├── environment.yml         # Conda environment definition
└── pyproject.toml          # Build config and tool settings
```

## License and third-party software

- **This repository's own code** is under the **MIT License** — see [`LICENSE`](LICENSE).
- **The Inception model files are not in git.** `inception5h.zip`, `tensorflow_inception_graph.pb`, and `imagenet_comp_graph_label_strings.txt` are listed in `.gitignore`. They are downloaded at runtime and are covered by the Apache 2.0 licence from Google.
- Tutorial reference: [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials) — see that repo's licence. More detail: [`NOTICE`](NOTICE).
