# -*- coding: utf-8 -*-
"""Legacy globals for any external scripts; new code should use dream_cli.application."""
from typing import Any

from dream_cli.models import legacy_default_renderer_dict, legacy_default_setting_dict


def get_default_setting() -> dict[str, Any]:
    return legacy_default_setting_dict()


def get_default_renderer() -> dict[str, Any]:
    return legacy_default_renderer_dict()


orig_image_name = "No image selected"
orig_image: list = []
c_settings = get_default_setting()
