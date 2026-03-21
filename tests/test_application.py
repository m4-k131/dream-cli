"""Unit tests for DreamApplication (no dreamer / TensorFlow import)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from dream_cli.application import DreamApplication
from dream_cli.models import RendererConfig


def test_with_defaults_paths(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    assert app.root == tmp_path.resolve()
    assert app.image_dir == tmp_path / "Images"
    assert app.orig_image is None
    assert not app.has_image()


def test_ensure_directories(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    app.ensure_directories()
    assert app.image_dir.is_dir()
    assert app.settings_dir.is_dir()
    assert app.renderer_dir.is_dir()


def test_list_jpeg_pick_tuples(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    app.ensure_directories()
    paths, names = app.list_jpeg_pick_tuples()
    assert paths == [] and names == []
    img_path = app.image_dir / "a.jpg"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_path)
    paths, names = app.list_jpeg_pick_tuples()
    assert len(paths) == 1 and names == ["a.jpg"]


def test_load_settings_missing(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    ok = app.load_settings_from_path(tmp_path / "missing_s.json")
    assert ok is False
    assert app.settings.name == "default_setting"


def test_settings_write_read_roundtrip(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    app.ensure_directories()
    app.settings.name = "saved"
    app.settings.renderers = [RendererConfig(name="r1")]
    path = tmp_path / "Settings" / "test_s.json"
    app.write_settings_file(path)
    app2 = DreamApplication.with_defaults(tmp_path)
    assert app2.load_settings_from_path(path) is True
    assert app2.settings.name == "saved"
    assert len(app2.settings.renderers) == 1
    assert app2.settings.renderers[0].name == "r1"


def test_load_image_path(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    app.ensure_directories()
    img_path = app.image_dir / "x.jpeg"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(img_path)
    app.load_image_path(img_path)
    assert app.has_image()
    assert app.orig_image_name == "x.jpeg"
    assert app.orig_image is not None
    assert app.orig_image.shape == (2, 2, 3)


def test_load_renderer_from_path(tmp_path: Path) -> None:
    app = DreamApplication.with_defaults(tmp_path)
    app.ensure_directories()
    p = app.renderer_dir / "preset_r.json"
    payload = {"name": "p", "layer": "conv2d1_pre_relu", "f_channel": 0, "l_channel": 8}
    p.write_text(json.dumps(payload), encoding="utf-8")
    r = app.load_renderer_from_path(p)
    assert r is not None
    assert r.name == "p"
    assert r.l_channel == 8
