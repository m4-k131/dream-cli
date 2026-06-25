# -*- coding: utf-8 -*-
"""Backward-compatible shim — use dream_cli.legacy instead."""
from dream_cli.legacy import (  # noqa: F401
    c_settings,
    get_default_renderer,
    get_default_setting,
    orig_image,
    orig_image_name,
)
