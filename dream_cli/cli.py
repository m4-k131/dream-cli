"""Interactive terminal UI; delegates state and I/O to DreamApplication + Prompter."""
from __future__ import annotations

from typing import Callable, Union

import utils as utils_mod
from dream_cli.application import DreamApplication
from dream_cli.layers import inception_layers
from dream_cli.models import DreamSettings, RendererConfig
from dream_cli.prompts import Prompter

Holder = Union[DreamSettings, RendererConfig]


class InteractiveDreamCli:
    LONG = "------------------------"
    MEDIUM = "-----------"
    SHORT = "-----"

    def __init__(self, app: DreamApplication, prompter: Prompter | None = None) -> None:
        self._app = app
        self._p = prompter or Prompter()

    def run(self) -> None:
        self._app.ensure_directories()
        while True:
            self._main_menu_once()

    def _main_menu_once(self) -> None:
        print(self.LONG)
        print(self.SHORT + "Main menu")
        print(self.LONG)
        if self._app.has_image():
            print("Selected image: " + self._app.orig_image_name)
            assert self._app.orig_image is not None
            print("Image size:" + str(self._app.orig_image.shape[0]) + "x" + str(self._app.orig_image.shape[1]))
        else:
            print("No image selected")
        print("Settings: %s" % self._app.settings.name)
        print("Renderer: %d Renderer selected" % len(self._app.settings.renderers))
        print(self.LONG)
        print("0: Select image \n1: Edit Settings \n2: Edit Renderer \n3: Continue(deepdream Image)")
        print(self.LONG)
        sel = self._p.int_range(0, 3, "Enter corrosponding number to navigate:  ")
        if sel == 0:
            self._select_image_menu()
        elif sel == 1:
            self._settings_menu()
        elif sel == 2:
            self._renderer_menu()
        elif sel == 3:
            print(self.LONG)
            if not self._app.has_image():
                print("No Image selected")
            elif not self._app.settings.renderers:
                print("No renderer selected! Add a renderer to dream the image")
            else:
                out = self._p.string_no_spaces(msg="Enter output Filename. Will be overwritten if it already exists: ")
                self._app.run_dream(out)
            print(self.LONG)

    def _select_image_menu(self) -> None:
        paths, names = self._app.list_jpeg_pick_tuples()
        print("0: back")
        for i, n in enumerate(names, start=1):
            print(str(i) + ": " + n)
        sel = self._p.int_range(0, len(paths))
        if sel > 0:
            self._app.load_image_path(paths[sel - 1])

    def _settings_menu(self) -> None:
        while True:
            s = self._app.settings
            print(self.LONG)
            print(self.SHORT + "Settings menu")
            print(self.LONG)
            print("0:  Back")
            print("1:  Name: %s" % s.name)
            print("2:  Iterations: %d" % s.iterations)
            print("3:  Octaves: %d" % s.octaves)
            print("4:  Octave scale: %.2f" % s.octave_scale)
            print("5:  Iteration descent: %d" % s.iteration_descent)
            print("6:  Background menu. Currently activated: " + str(s.save_gradient))
            print("7:  Global color correction: " + str(s.color_correction))
            print("8:  Renderer menu. Current number of renderer: %d" % len(s.renderers))
            print("9:  New Settings")
            print("10: Load Settings")
            print("11: Save Settings")
            print(self.LONG)
            sel = self._p.int_range(0, 11, msg="Enter selection:")
            if sel == 0:
                return
            if sel == 1:
                self._edit_setting_name()
            elif sel == 2:
                self._edit_iterations()
            elif sel == 3:
                self._edit_octaves()
            elif sel == 4:
                self._edit_octave_scale()
            elif sel == 5:
                self._edit_iteration_descent()
            elif sel == 6:
                self._edit_background_menu()
            elif sel == 7:
                self._edit_global_cc()
            elif sel == 8:
                self._renderer_menu()
            elif sel == 9:
                self._app.settings = DreamSettings.default_new()
            elif sel == 10:
                self._load_settings_menu()
            elif sel == 11:
                self._save_settings_menu()

    def _edit_setting_name(self) -> None:
        print(self.MEDIUM)
        print("Current name: %s" % self._app.settings.name)
        print(self.MEDIUM)
        print("0: Back \n1:  Change Name")
        print(self.MEDIUM)
        if self._p.int_range(0, 1, "Enter selection:") == 1:
            self._app.settings.name = self._p.string_no_spaces("Enter Settings name:")

    def _edit_iterations(self) -> None:
        print(self.MEDIUM)
        print("Current iterations: " + str(self._app.settings.iterations))
        print(self.MEDIUM)
        print("0: Back \n1: Change iterations")
        print(self.MEDIUM)
        if self._p.int_range(0, 1, "Enter selection:") == 1:
            self._app.settings.iterations = self._p.int_range(0, 10000, "Enter new Iteration no.:")

    def _edit_octaves(self) -> None:
        print(self.MEDIUM)
        print("Current octaves: " + str(self._app.settings.octaves))
        print(self.MEDIUM)
        print("0: Back \n1: Change Octaves")
        print(self.MEDIUM)
        if self._p.int_range(0, 1) == 1:
            self._app.settings.octaves = self._p.int_range(1, 200, "Enter Octave no.:")

    def _edit_octave_scale(self) -> None:
        print(self.MEDIUM)
        print("Current octave_scale: " + str(self._app.settings.octave_scale))
        print(self.MEDIUM)
        print("0: Back \n1: Change octave_scale")
        print(self.MEDIUM)
        if self._p.int_range(0, 1, "Enter selection:") == 1:
            self._app.settings.octave_scale = self._p.float_range(
                1.0, 50.0, "Enter Octave scale (Recommended: between 1.1 and 2.0):"
            )

    def _edit_iteration_descent(self) -> None:
        print(self.MEDIUM)
        print("Current iteration_descent: " + str(self._app.settings.iteration_descent))
        print(self.MEDIUM)
        print("0: Back \n1: Change iteration_descent")
        print(self.MEDIUM)
        if self._p.int_range(0, 1) == 1:
            s = self._app.settings
            max_descent = int(s.iterations / s.octaves) if s.octaves > 0 else s.iterations
            s.iteration_descent = self._p.int_range(
                -100,
                max_descent,
                "Enter Iteration descent (Can not be higher than Iterations/octaves=%d):" % max_descent,
            )

    def _edit_background_color(self) -> None:
        print(self.SHORT)
        print("Current background color: " + str(self._app.settings.background))
        print(self.SHORT)
        print("0: Back \n1: Change background color")
        print(self.SHORT)
        if self._p.int_range(0, 1) == 1:
            r = self._p.int_range(0, 255, "Enter Red value of background:")
            g = self._p.int_range(0, 255, "Enter Green value of background:")
            b = self._p.int_range(0, 255, "Enter Blue value of background:")
            self._app.settings.background = [r, g, b]

    def _edit_background_menu(self) -> None:
        while True:
            s = self._app.settings
            print(self.MEDIUM)
            print("Write gradient onto empty background: " + str(s.save_gradient))
            print("Background color: " + str(s.background))
            print(self.MEDIUM)
            print("0: Back \n1: Activate/Deactivate if gradient is saved onto background \n2: Change background color")
            print(self.MEDIUM)
            sel = self._p.int_range(0, 2)
            if sel == 0:
                return
            if sel == 1:
                print(self.SHORT)
                print("0: Back \n1: Activate \n2: Deactivate")
                print(self.SHORT)
                sub = self._p.int_range(0, 2)
                if sub == 0:
                    continue
                if sub == 1:
                    s.save_gradient = True
                if sub == 2:
                    s.save_gradient = False
            if sel == 2:
                self._edit_background_color()

    def _edit_global_cc(self) -> None:
        while True:
            s = self._app.settings
            print(self.MEDIUM)
            print("Global color correction: " + str(s.color_correction))
            print("Method: " + str(s.cc_vars[0]))
            print("RGB multiplier: " + str(s.cc_vars[1:]))
            print(self.MEDIUM)
            print("0: Back \n1: Activate multiplier \n2: Deactivate multiplier\n3: Edit color correction values")
            sel = self._p.int_range(0, 3)
            if sel == 0:
                return
            if sel == 1:
                s.color_correction = True
            elif sel == 2:
                s.color_correction = False
            elif sel == 3:
                self._edit_cc_vars_on(s)

    def _renderer_menu(self) -> None:
        while True:
            rs = self._app.settings.renderers
            print(self.LONG)
            print(self.SHORT + "Renderer menu")
            print(self.LONG)
            if rs:
                print("Current renderer:")
                for r in rs:
                    print("-" + r.name)
            else:
                print("-No Renderer selected-")
            print(self.LONG)
            print("0: Back \n1: Edit renderer \n2: New Renderer \n3: Load Renderer \n4: Remove Renderer \n5: Save Renderer")
            print(self.LONG)
            sel = self._p.int_range(0, 5, "Enter selection: ")
            if sel == 0:
                return
            if sel == 1:
                self._pick_renderer_to_edit()
            elif sel == 2:
                self._add_renderer_flow()
            elif sel == 3:
                self._load_renderer_menu()
            elif sel == 4:
                self._remove_renderer_menu()
            elif sel == 5:
                self._save_renderer_menu()

    def _pick_renderer_to_edit(self) -> None:
        rs = self._app.settings.renderers
        if not rs:
            return
        print(self.MEDIUM)
        print("Select Renderer to edit:")
        print(self.SHORT)
        print("0: Back")
        for i, r in enumerate(rs):
            print(
                str(i + 1) + ": " + r.name + " | " + r.layer + "[" + str(r.f_channel) + ":" + str(r.l_channel) + "]"
            )
        print(self.SHORT)
        selected = self._p.int_range(0, len(rs))
        if selected > 0:
            self._edit_renderer_settings_loop(rs[selected - 1])

    def _remove_renderer_menu(self) -> None:
        rs = self._app.settings.renderers
        if not rs:
            return
        print(self.MEDIUM)
        print("Select Renderer to remove:")
        print(self.SHORT)
        print("0: Back")
        for i, r in enumerate(rs):
            print(
                str(i + 1) + ": " + r.name + " | " + r.layer + "[" + str(r.f_channel) + ":" + str(r.l_channel) + "]"
            )
        print(self.SHORT)
        selected = self._p.int_range(0, len(rs))
        if selected > 0:
            del rs[selected - 1]

    def _add_renderer_flow(self) -> None:
        nr = RendererConfig()
        self._select_layer_menu_mutate(nr)
        self._edit_renderer_settings_loop(nr)
        self._app.settings.renderers.append(nr)

    def _select_layer_menu_mutate(self, r: RendererConfig) -> None:
        ll = inception_layers()
        print(self.MEDIUM)
        print("Select layer:")
        for i, row in enumerate(ll):
            print(str(i) + ": " + row[0] + " Channels: " + str(row[1]))
        last = len(ll) - 1
        sel = self._p.int_range(0, last)
        r.layer = ll[sel][0]
        r.l_channel = ll[sel][1]
        r.max_channel = ll[sel][1]

    def _edit_layer(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current layer: " + str(r.layer))
        print(self.MEDIUM)
        print("0: Back \n1: Change layer")
        if self._p.int_range(0, 2) == 1:
            self._select_layer_menu_mutate(r)
        return r

    def _edit_channels(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current channels: First channel: " + str(r.f_channel) + " Last channel: " + str(r.l_channel))
        print(self.MEDIUM)
        print("0: Back \n1: Change channels")
        print(self.MEDIUM)
        if self._p.int_range(0, 2) == 1:
            mc = r.max_channel
            f = self._p.int_range(0, mc - 1, "Enter first channel no. (Must be between 0 and " + str(mc - 1) + "):")
            last_ch = self._p.int_range(
                f + 1,
                mc,
                "Enter last channel no. (Must be between " + str(f + 1) + " and " + str(mc) + "):",
            )
            r.f_channel = f
            r.l_channel = last_ch
        return r

    def _edit_squared(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Squared gradient activated: " + str(r.squared))
        print(self.MEDIUM)
        print("0: Back \n1: Activate squared gradient \n2: Deactivate squared gradient")
        print(self.MEDIUM)
        sel = self._p.int_range(0, 3)
        if sel == 1:
            r.squared = True
        if sel == 2:
            r.squared = False
        return r

    def _edit_step_size(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current step_size: " + str(r.step_size))
        print(self.MEDIUM)
        print("0: Back \n1: Edit step_size")
        print(self.MEDIUM)
        if self._p.int_range(0, 2) == 1:
            r.step_size = self._p.float_range(-100.0, 100.0, "Enter step size (Recommended: between 2 and 0.01): ")
        return r

    def _edit_boundaries(self, r: RendererConfig) -> RendererConfig:
        print(self.SHORT)
        print(
            "All values most be between 0 and 1 on both axes starting on the top left corner "
            "(If this renderer should e.g. only render the bottom half choose x_min=0,x_max=1, y_min=0.5,  y_max=1"
        )
        print(self.SHORT)
        x_min = self._p.float_range(0.0, 1.0, "Enter x_min:")
        x_max = self._p.float_range(x_min, 1.0, "Enter x_max:")
        y_min = self._p.float_range(0.0, 1.0, "Enter y_min:")
        y_max = self._p.float_range(y_min, 1.0, "Enter y_max:")
        r.boundraries = [[x_min, x_max], [y_min, y_max]]
        return r

    def _edit_cropped(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Cropped renderer activated: " + str(r.cropped))
        print("Boundraries: " + str(r.boundraries))
        print(self.MEDIUM)
        print("0: Back \n1: Activate cropped renderer \n2: Deactivate cropped renderer\n3: Edit boundraries")
        print(self.MEDIUM)
        sel = self._p.int_range(0, 3)
        if sel == 1:
            r.cropped = True
        if sel == 2:
            r.cropped = False
        if sel == 3:
            self._edit_boundaries(r)
            return self._edit_cropped(r)
        return r

    def _load_mask_into_renderer(self, r: RendererConfig) -> None:
        paths, names = self._app.list_jpeg_pick_tuples()
        print("0: back")
        for i, n in enumerate(names, start=1):
            print(str(i) + ": " + n)
        sel = self._p.int_range(0, len(paths))
        if sel > 0:
            r.mask = utils_mod.load_image(str(paths[sel - 1]))
            r.mask_name = names[sel - 1]

    def _edit_mask(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Mask activated: " + str(r.masked))
        print("Mask:" + str(r.mask_name))
        print(self.MEDIUM)
        print("0: Back \n1: Activate Mask \n2: Deactivate Mask \n3: Load mask \n4: Remove mask")
        print(self.MEDIUM)
        sel = self._p.int_range(0, 4)
        if sel in (1, 2):
            r.masked = not r.masked
            return self._edit_mask(r)
        if sel == 3:
            self._load_mask_into_renderer(r)
            return self._edit_mask(r)
        if sel == 4:
            r.mask = []
            r.mask_name = ""
            r.masked = False
            return self._edit_mask(r)
        if len(r.mask_name) == 0:
            r.masked = False
        return r

    def _edit_rotate(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current rotation: %d° right" % (r.rotation * 90))
        print(self.MEDIUM)
        print(
            "0: Back \n1: Rotate 0° right (deactivate rotation) \n2: Rotate 90° right\n3: Rotate 180° right \n4: Rotate 270° right"
        )
        print(self.MEDIUM)
        sel = self._p.int_range(0, 4)
        if sel > 0:
            if sel == 1:
                r.rotation = 0
                r.rotate = False
            else:
                r.rotation = sel - 1
                r.rotate = True
        return r

    def _edit_tile_size(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current tile size: " + str(r.tile_size))
        print(self.MEDIUM)
        print("0: Back \n1: Edit tile size")
        print(self.MEDIUM)
        if self._p.int_range(0, 1) == 1:
            r.tile_size = self._p.int_range(1, 10000, "Enter Tile size (should be around 300):")
        return r

    def _edit_renderer_cc(self, r: RendererConfig) -> RendererConfig:
        while True:
            print(self.MEDIUM)
            print("Renderer specific color correction: " + str(r.color_correction))
            print("Method: " + str(r.cc_vars[0]))
            print("RGB multiplier: " + str(r.cc_vars[1:]))
            print(self.MEDIUM)
            print("0: Back \n1: Activate multiplier \n2: Deactivate multiplier\n3: Edit color correction values")
            sel = self._p.int_range(0, 3)
            if sel == 0:
                return r
            if sel == 1:
                r.color_correction = True
            elif sel == 2:
                r.color_correction = False
            elif sel == 3:
                self._edit_cc_vars_on(r)

    def _edit_cc_vars_on(self, holder: Holder) -> None:
        while True:
            cv = holder.cc_vars
            print(self.SHORT)
            print("Current Method: " + str(cv[0]))
            print("Current RGB multiplier: " + str(cv[1:]))
            print(self.SHORT)
            print("0: Back \n1: Edit Method \n2: Edit RGB multiplier")
            print(self.SHORT)
            sel = self._p.int_range(0, 2)
            if sel == 0:
                return
            if sel == 1:
                print("0: Back \n1: Simple Grayscale correction \n2: Retaining original colors \n3: Linear correction")
                print(self.SHORT)
                new_sel = self._p.int_range(0, 3)
                if new_sel > 0:
                    cv[0] = new_sel
            if sel == 2:
                print(self.SHORT)
                print("Current color multiplier: " + str(cv[1:]))
                print(self.SHORT)
                print(
                    "Enter the color multiplier (mult) for the Red, Green and Blue channel (r,g,b). "
                    "Recommended values are between -5.0 and 5.0"
                )
                red = self._p.float_range(-100.0, 100.0, "Enter r_mult:")
                green = self._p.float_range(-100.0, 100.0, "Enter g_mult:")
                blue = self._p.float_range(-100.0, 100.0, "Enter b_mult:")
                holder.cc_vars = [cv[0], red, green, blue]

    def _edit_render_x_iteration(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Currently rendering every %d iteration (1=render every Iteration)" % r.render_x_iteration)
        print(self.MEDIUM)
        print("0: back \n1: Edit")
        print(self.MEDIUM)
        if self._p.int_range(0, 1) == 1:
            r.render_x_iteration = self._p.int_range(0, 10000, "Enter new value:")
        return r

    def _edit_r_name(self, r: RendererConfig) -> RendererConfig:
        print(self.MEDIUM)
        print("Current name: " + str(r.name))
        print(self.MEDIUM)
        print("0: Back \n1: Change name")
        print(self.MEDIUM)
        if self._p.int_range(0, 1) == 1:
            r.name = self._p.string_no_spaces("Enter name:")
        return r

    def _save_renderer_from_menu(self, r: RendererConfig) -> RendererConfig:
        filename = r.name + "_r.json"
        path = self._app.renderer_dir / filename
        while path.exists():
            print(filename + " already exists. Do you want to overwrite it or rename your current Renderer?")
            if self._p.overwrite_or_rename():
                break
            r.name = self._p.string_no_spaces(msg="Enter new name:")
            filename = r.name + "_r.json"
            path = self._app.renderer_dir / filename
        return self._app.write_renderer_file(r, path)

    def _edit_renderer_settings_loop(self, r: RendererConfig) -> None:
        handlers: dict[int, Callable[[RendererConfig], RendererConfig]] = {
            1: self._edit_r_name,
            2: self._edit_layer,
            3: self._edit_channels,
            4: self._edit_squared,
            5: self._edit_step_size,
            6: self._edit_cropped,
            7: self._edit_mask,
            8: self._edit_rotate,
            9: self._edit_tile_size,
            10: self._edit_renderer_cc,
            11: self._edit_render_x_iteration,
            12: self._save_renderer_from_menu,
        }
        while True:
            print(self.LONG)
            print(self.SHORT + "Renderer")
            print(self.LONG)
            print("0:  Back")
            print("1:  Name: %s" % r.name)
            print("2:  Layer: %s" % r.layer)
            print("3:  Channels: " + str(r.f_channel) + "-" + str(r.l_channel))
            print("4:  Squared: " + str(r.squared))
            print("5:  Step size: %.5f" % r.step_size)
            if r.cropped:
                print("6:  Cropped: Activated with Boundraries: " + str(r.boundraries))
            else:
                print("6:  Cropped: " + str(r.cropped))
            if r.masked:
                print("7:  Masked gradient activated ")
            else:
                print("7:  Masked gradient deactivated")
            if r.rotate:
                print("8:  Rotation. Activated with " + str(r.rotation * 90) + "° right")
            else:
                print("8:  Rotation deactivated")
            print("9:  Tile size: %d" % r.tile_size)
            if r.color_correction:
                print("10: Renderer specific color correction activated ")
            else:
                print("10: Color correction deactivated")
            print("11: Render every %d iteration" % r.render_x_iteration)
            print("12: Save current Renderer to file")
            print(self.LONG)
            sel = self._p.int_range(0, 12, "Enter selection:")
            if sel == 0:
                return
            r = handlers[sel](r)

    def _save_settings_menu(self) -> None:
        filename = self._app.settings.name + "_s.json"
        path = self._app.settings_dir / filename
        while path.exists():
            print(filename + " already exists. Do you want to overwrite it or rename your current setting?")
            if self._p.overwrite_or_rename():
                break
            self._app.settings.name = self._p.string_no_spaces(msg="Enter new name:")
            filename = self._app.settings.name + "_s.json"
            path = self._app.settings_dir / filename
        self._app.write_settings_file(path)

    def _load_settings_menu(self) -> None:
        paths = self._app.list_settings_files()
        print(self.MEDIUM)
        print("0: Back")
        for i, p in enumerate(paths, start=1):
            print(str(i) + ": " + p.name)
        print(self.MEDIUM)
        sel = self._p.int_range(0, len(paths), "Select File to load:")
        if sel > 0:
            self._app.settings.renderers = []
            ok = self._app.load_settings_from_path(paths[sel - 1])
            if not ok:
                print("File not found. Falling back to default settings")

    def _load_renderer_menu(self) -> None:
        paths = self._app.list_renderer_files()
        print(self.MEDIUM)
        print("0: Back")
        for i, p in enumerate(paths, start=1):
            print(str(i) + ": " + p.name)
        print(self.MEDIUM)
        sel = self._p.int_range(0, len(paths), "Select File to load:")
        if sel > 0:
            path = paths[sel - 1]
            print(str(path))
            loaded = self._app.load_renderer_from_path(path)
            if loaded is None:
                print("File not found. Adding default renderer")
                loaded = RendererConfig()
            self._app.settings.renderers.append(loaded)

    def _save_renderer_menu(self) -> None:
        rs = self._app.settings.renderers
        if not rs:
            return
        print(self.MEDIUM)
        print("Select Renderer to save:")
        print(self.SHORT)
        print("0: Back")
        for i, r in enumerate(rs):
            print(
                str(i + 1) + ": " + r.name + " | " + r.layer + "[" + str(r.f_channel) + ":" + str(r.l_channel) + "]"
            )
        print(self.SHORT)
        selected = self._p.int_range(0, len(rs))
        if selected > 0:
            self._save_renderer_from_menu(rs[selected - 1])
