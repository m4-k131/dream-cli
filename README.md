# dream-cli

Interactive CLI for **DeepDream** on still images. It uses Google’s frozen **Inception “5h”** graph (`tensorflow_inception_graph.pb`, fetched automatically from `inception5h.zip` the first time you run it). The implementation follows the classic gradient-ascent idea (optionally on squared activations) over chosen layers and channel ranges, with octaves, tiling, crops, masks, and simple color controls.

## Environment (project `.venv`)

The repo uses a **Conda prefix environment** at **`.venv/`** (gitignored). It is defined in **`environment.yml`**:

- **Python 3.10** — first release with **structural pattern matching** (`match` / `case`) if you modernize the code.
- **TensorFlow 2.9.3** — same line as the author’s known-good **`tf`** Conda env, and PyPI ships **`cp310` `win_amd64` wheels** for it.
- **CUDA / cuDNN** — **`cudatoolkit` 11.2.x** and **`cudnn` 8.1.0.77** (channels and versions taken from a **`conda list --export`** of **`tf`**), resolved from **conda-forge** so the solve works on current Conda indexes.

Create or refresh the env from the repo root:

```bash
conda env create -f environment.yml -p .venv
# or, after edits to environment.yml:
conda env update -f environment.yml -p .venv --prune
```

Activate and run:

```bash
conda activate ./.venv
python dreamcli.py
```

On Windows with recent Conda, you can also use `conda activate C:\full\path\to\dream-cli\.venv`.

Quick GPU check:

```bash
conda activate ./.venv
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

### Reference env `tf`

The separate Conda env **`tf`** on this machine is the **sanity reference** (GPU + TensorFlow stack). **`environment.yml`** was aligned to that stack, except **Python is 3.10** here so new code can use pattern matching while staying on **TF 2.9.x** and **`tensorflow.compat.v1`**.

## Usage

1. Put input images as **`.jpg`** / **`.JPEG`** in the **`Images/`** folder (create it if needed).
2. Run **`dreamcli.py`** from the repo root.
3. Use the menus to pick an image, adjust **settings** and **renderers**, then continue to run the dream.
4. Outputs go under **`renderedImages/`** (see `utils.save_folder`).

Saved presets: **`Settings/*_s.json`** (full settings) and **`Settings/Renderer/*_r.json`** (single renderer).

## Dependencies

- **TensorFlow** 2.9.x (with **`compat.v1`** in code)
- **NumPy** 1.26.x (pinned in `environment.yml`; avoid NumPy 2.x with this TF line unless you re-validate)
- **Pillow**
- **CUDA runtime / cuDNN** via Conda (see `environment.yml`)

## Settings (global)

- **Octaves**: How many scaled resolutions are used; each octave refines detail at a different size.
- **Octave scale**: Downscale factor between octaves (e.g. `1.5`).
- **Iterations**: Gradient steps per octave (before any per-octave reduction).
- **Iteration descent**: Subtracted from the iteration count each octave (can shorten later octaves).
- **Background / save gradient**: Optionally write the accumulated gradient onto a solid background color.

## Renderer (per pass)

Each renderer optimizes one **layer** and a **channel range** (`f_channel`–`l_channel`). Many combinations are listed in the CLI (204 layer entries matching the frozen graph).

- **Squared**: Use squared activations as the objective.
- **Step size**: Scales the gradient each step (negative reverses direction).
- **Cropped & boundaries**: Restrict rendering to a normalized crop `[[x_min, x_max], [y_min, y_max]]` in `[0, 1]`.
- **Mask**: Multiply the gradient by a mask image where enabled.
- **Rotation**: Rotate the crop before the gradient, then rotate back (useful because many patterns are directional).
- **Tile size**: Inception expects roughly ~300×300 inputs; larger tiles use more VRAM and can look less stable.
- **Color correction / multipliers**: Per-renderer or global tweaks to how the gradient is tinted before accumulation.

## Note on native Windows and TensorFlow

Recent **pip-only** TensorFlow on **native Windows** is often **CPU-only**. This project relies on **Conda-provided CUDA/cuDNN** plus a **TensorFlow build that still supports native Windows GPU** (here **2.9.x**, same family as the reference **`tf`** env). For a slightly newer line with **Python 3.10** wheels, **TensorFlow 2.10.1** also publishes **`cp310-win_amd64`** wheels; bump the pin in `environment.yml` if you want to try it.
