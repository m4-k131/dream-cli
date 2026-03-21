"""Session state, paths, persistence, and dreaming (headless core for CLI / future GUI)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

import utils as utils_mod
from dream_cli.models import DreamSettings, RendererConfig


@dataclass
class DreamApplication:
    root: Path
    image_dir: Path
    settings_dir: Path
    renderer_dir: Path
    orig_image: np.ndarray | None
    orig_image_name: str
    settings: DreamSettings

    @classmethod
    def with_defaults(cls, root: Path | None = None) -> DreamApplication:
        base = (root or Path.cwd()).resolve()
        img = base / "Images"
        st = base / "Settings"
        rd = st / "Renderer"
        return cls(
            root=base,
            image_dir=img,
            settings_dir=st,
            renderer_dir=rd,
            orig_image=None,
            orig_image_name="No image selected",
            settings=DreamSettings.default_new(),
        )

    def ensure_directories(self) -> None:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.renderer_dir.mkdir(parents=True, exist_ok=True)

    def has_image(self) -> bool:
        return self.orig_image is not None and self.orig_image.size > 0

    def list_jpeg_pick_tuples(self) -> Tuple[List[Path], List[str]]:
        if not self.image_dir.is_dir():
            return [], []
        paths: List[Path] = []
        names: List[str] = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg"):
                paths.append(p)
                names.append(p.name)
        return paths, names

    def load_image_path(self, path: Path) -> None:
        self.orig_image = utils_mod.load_image(str(path))
        self.orig_image_name = path.name

    def run_dream(self, output_basename: str) -> None:
        import dreamer as dreamer_mod
        if not self.has_image():
            raise RuntimeError("No image selected")
        if not self.settings.renderers:
            raise RuntimeError("No renderers configured")
        dreamer_mod.close_and_reopen_session()
        dreamer_mod.dream_image(
            self.orig_image,
            self.settings.to_runtime_dreamer_dict(),
            output_basename,
        )
        dreamer_mod.close_session()

    def settings_payload_for_disk(self) -> dict:
        payload = self.settings.to_stored_dict()
        for r in payload["renderers"]:
            if r.get("mask_name"):
                m = r.get("mask")
                if hasattr(m, "tolist"):
                    r["mask"] = m.tolist()
        return payload

    def write_settings_file(self, path: Path) -> None:
        self.ensure_directories()
        path.write_text(json.dumps(self.settings_payload_for_disk(), indent=4), encoding="utf-8")
        for r in self.settings.renderers:
            if r.mask_name and r.mask is not None and not (isinstance(r.mask, list) and len(r.mask) == 0):
                r.mask = np.asarray(r.mask, dtype=np.float32)

    def load_settings_from_path(self, path: Path) -> bool:
        if not path.is_file():
            self.settings = DreamSettings.default_new()
            return False
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.settings = DreamSettings.from_mapping(raw)
        return True

    def list_settings_files(self) -> List[Path]:
        if not self.settings_dir.is_dir():
            return []
        return sorted(self.settings_dir.glob("*_s.json"))

    def list_renderer_files(self) -> List[Path]:
        if not self.renderer_dir.is_dir():
            return []
        return sorted(self.renderer_dir.glob("*_r.json"))

    def load_renderer_from_path(self, path: Path) -> RendererConfig | None:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RendererConfig.from_mapping(raw)

    def renderer_payload_for_disk(self, renderer: RendererConfig) -> dict:
        d = renderer.to_stored_dict()
        if d.get("mask_name"):
            m = d.get("mask")
            if hasattr(m, "tolist"):
                d["mask"] = m.tolist()
        return d

    def write_renderer_file(self, renderer: RendererConfig, path: Path) -> RendererConfig:
        self.ensure_directories()
        path.write_text(json.dumps(self.renderer_payload_for_disk(renderer), indent=4), encoding="utf-8")
        if renderer.mask_name:
            renderer.mask = np.asarray(renderer.mask, dtype=np.float32)
        return renderer
