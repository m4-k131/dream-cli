"""Structured settings for DeepDream sessions (JSON-compatible with legacy files)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

import numpy as np


def _renderer_template_dict() -> dict[str, Any]:
    return {
        "name": "default_renderer",
        "layer": "conv2d1_pre_relu",
        "f_channel": 0,
        "l_channel": 64,
        "max_channel": 64,
        "squared": True,
        "cropped": False,
        "boundraries": [[0, 1], [0, 1]],
        "rotate": False,
        "rotation": 0,
        "tile_size": 300,
        "render_x_iteration": 1,
        "step_size": 1.0,
        "color_correction": False,
        "cc_vars": [1, 4, 4, 4],
        "masked": False,
        "mask": [],
        "mask_name": "",
        "t_masks": [],
    }


def _settings_template_dict() -> dict[str, Any]:
    return {
        "name": "default_setting",
        "iterations": 20,
        "octaves": 4,
        "octave_scale": 1.5,
        "iteration_descent": 0,
        "save_gradient": False,
        "background": [0, 0, 0],
        "renderers": [],
        "color_correction": False,
        "cc_vars": [1, 4, 4, 4],
    }


@dataclass
class RendererConfig:
    name: str = "default_renderer"
    layer: str = "conv2d1_pre_relu"
    f_channel: int = 0
    l_channel: int = 64
    max_channel: int = 64
    squared: bool = True
    cropped: bool = False
    boundraries: list[list[float]] = field(default_factory=lambda: [[0.0, 1.0], [0.0, 1.0]])
    rotate: bool = False
    rotation: int = 0
    tile_size: int = 300
    render_x_iteration: int = 1
    step_size: float = 1.0
    color_correction: bool = False
    cc_vars: list[Any] = field(default_factory=lambda: [1, 4, 4, 4])
    masked: bool = False
    mask: Any = field(default_factory=list)
    mask_name: str = ""
    t_masks: list[Any] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RendererConfig:
        merged = _renderer_template_dict()
        merged.update(dict(raw))
        names = {f.name for f in fields(cls)}
        kw = {k: merged[k] for k in names}
        return cls(**kw)

    def to_stored_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if hasattr(self.mask, "tolist"):
            d["mask"] = self.mask.tolist()
        return d

    def to_runtime_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.mask_name and self.mask is not None and not (isinstance(self.mask, list) and len(self.mask) == 0):
            d["mask"] = np.asarray(self.mask, dtype=np.float32)
        else:
            d["mask"] = []
        return d


@dataclass
class DreamSettings:
    name: str = "default_setting"
    iterations: int = 20
    octaves: int = 4
    octave_scale: float = 1.5
    iteration_descent: int = 0
    save_gradient: bool = False
    background: list[int] = field(default_factory=lambda: [0, 0, 0])
    renderers: list[RendererConfig] = field(default_factory=list)
    color_correction: bool = False
    cc_vars: list[Any] = field(default_factory=lambda: [1, 4, 4, 4])

    @classmethod
    def default_new(cls) -> DreamSettings:
        return cls(renderers=[])

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DreamSettings:
        merged = _settings_template_dict()
        merged.update(dict(raw))
        rend_raw = merged.pop("renderers", [])
        ds = cls(
            name=str(merged["name"]),
            iterations=int(merged["iterations"]),
            octaves=int(merged["octaves"]),
            octave_scale=float(merged["octave_scale"]),
            iteration_descent=int(merged["iteration_descent"]),
            save_gradient=bool(merged["save_gradient"]),
            background=[int(x) for x in merged["background"]],
            renderers=[RendererConfig.from_mapping(x) for x in rend_raw],
            color_correction=bool(merged["color_correction"]),
            cc_vars=list(merged["cc_vars"]),
        )
        for r in ds.renderers:
            if r.mask_name and r.mask is not None and not hasattr(r.mask, "shape"):
                r.mask = np.asarray(r.mask, dtype=np.float32)
        return ds

    def to_stored_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "octaves": self.octaves,
            "octave_scale": self.octave_scale,
            "iteration_descent": self.iteration_descent,
            "save_gradient": self.save_gradient,
            "background": list(self.background),
            "renderers": [r.to_stored_dict() for r in self.renderers],
            "color_correction": self.color_correction,
            "cc_vars": copy.deepcopy(self.cc_vars),
        }

    def to_runtime_dreamer_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "octaves": self.octaves,
            "octave_scale": self.octave_scale,
            "iteration_descent": self.iteration_descent,
            "save_gradient": self.save_gradient,
            "background": list(self.background),
            "renderers": [r.to_runtime_dict() for r in self.renderers],
            "color_correction": self.color_correction,
            "cc_vars": list(self.cc_vars),
        }


def legacy_default_setting_dict() -> dict[str, Any]:
    return DreamSettings.default_new().to_stored_dict()


def legacy_default_renderer_dict() -> dict[str, Any]:
    return RendererConfig().to_stored_dict()
