"""JSON schemas and validation for DeepDream settings."""
from __future__ import annotations

from typing import Any


# JSON Schema for DreamSettings (runtime version)
DREAM_SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["iterations", "octaves", "renderers"],
    "properties": {
        "name": {"type": "string", "default": "default_setting"},
        "iterations": {"type": "integer", "minimum": 0, "default": 20},
        "octaves": {"type": "integer", "minimum": 1, "default": 4},
        "octave_scale": {"type": "number", "minimum": 1.0, "default": 1.5},
        "iteration_descent": {"type": "integer", "default": 0},
        "save_gradient": {"type": "boolean", "default": False},
        "background": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 255},
            "minItems": 3,
            "maxItems": 3,
            "default": [0, 0, 0],
        },
        "renderers": {
            "type": "array",
            "items": {"type": "object"},  # Renderer schema applied per item
            "minItems": 1,
        },
        "color_correction": {"type": "boolean", "default": False},
        "cc_vars": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "default": [1, 4, 4, 4],
        },
    },
}

# JSON Schema for RendererConfig (runtime version)
RENDERER_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["layer", "step_size"],
    "properties": {
        "name": {"type": "string", "default": "default_renderer"},
        "layer": {"type": "string"},
        "f_channel": {"type": "integer", "minimum": 0, "default": 0},
        "l_channel": {"type": "integer", "minimum": 0, "default": 64},
        "max_channel": {"type": "integer", "minimum": 1, "default": 64},
        "squared": {"type": "boolean", "default": True},
        "cropped": {"type": "boolean", "default": False},
        "boundraries": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "minItems": 2,
            "maxItems": 2,
            "default": [[0.0, 1.0], [0.0, 1.0]],
        },
        "rotate": {"type": "boolean", "default": False},
        "rotation": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0},
        "tile_size": {"type": "integer", "minimum": 1, "default": 300},
        "render_x_iteration": {"type": "integer", "minimum": 1, "default": 1},
        "step_size": {"type": "number"},
        "color_correction": {"type": "boolean", "default": False},
        "cc_vars": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "default": [1, 4, 4, 4],
        },
        "masked": {"type": "boolean", "default": False},
        "mask": {"type": "array", "default": []},  # Can be list or numpy array
        "mask_name": {"type": "string", "default": ""},
        "t_masks": {"type": "array", "default": []},
    },
}


def _get_default_from_schema(schema: dict[str, Any], key: str) -> Any:
    """Extract default value for a key from schema properties."""
    props = schema.get("properties", {})
    prop = props.get(key, {})
    return prop.get("default")


def validate_and_fill_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate settings dict and fill missing values with defaults.

    Returns a new dict with all defaults applied. Raises ValueError on invalid input.
    """
    if not isinstance(settings, dict):
        raise ValueError(f"Settings must be a dict, got {type(settings).__name__}")

    # Check required fields
    for key in DREAM_SETTINGS_SCHEMA.get("required", []):
        if key not in settings:
            raise ValueError(f"Missing required setting: '{key}'")

    filled: dict[str, Any] = {}
    props = DREAM_SETTINGS_SCHEMA.get("properties", {})

    for key, schema_def in props.items():
        value = settings.get(key)
        if value is None:
            value = schema_def.get("default")
        filled[key] = value

    # Ensure renderers is a list
    renderers = filled.get("renderers", [])
    if not isinstance(renderers, list):
        raise ValueError(f"'renderers' must be a list, got {type(renderers).__name__}")
    if len(renderers) == 0:
        raise ValueError("At least one renderer is required")

    # Validate each renderer
    filled["renderers"] = [validate_and_fill_renderer(r) for r in renderers]

    return filled


def validate_and_fill_renderer(renderer: dict[str, Any]) -> dict[str, Any]:
    """Validate renderer dict and fill missing values with defaults.

    Returns a new dict with all defaults applied. Raises ValueError on invalid input.
    """
    if not isinstance(renderer, dict):
        raise ValueError(f"Renderer must be a dict, got {type(renderer).__name__}")

    # Check required fields
    for key in RENDERER_CONFIG_SCHEMA.get("required", []):
        if key not in renderer:
            raise ValueError(f"Missing required renderer field: '{key}'")

    filled: dict[str, Any] = {}
    props = RENDERER_CONFIG_SCHEMA.get("properties", {})

    for key, schema_def in props.items():
        value = renderer.get(key)
        if value is None:
            value = schema_def.get("default")
        filled[key] = value

    return filled
