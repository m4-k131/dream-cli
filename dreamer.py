# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:53:10 2020
TF-functions copied from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/14_DeepDream.ipynb
"""

import numpy as np
import tensorflow as tf
import utils as utils
import tensorflow.compat.v1 as tfc
import math
from typing import Any

from tqdm import tqdm

from dream_cli.schemas import validate_and_fill_settings


class Constants:
    model_fn = "tensorflow_inception_graph.pb"
    inception_download_url="http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    save_folder="./renderedImages/4"
    imagenet_mean = 117.


class Dreamer:
    def __init__(self):
        self.__device_name='CPU:0'
        ###Activate GPU rendering if possible
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            self.__device_name='GPU:0'
        with tfc.device(self.__device_name):
            self.graph = tfc.Graph()
            self.sess = tfc.InteractiveSession(graph=self.graph)
        #Load model and build Graph
        print('Ignore the following Gfile-warning:')
        with tfc.gfile.FastGFile(Constants.model_fn, 'rb') as f, tfc.device(self.__device_name):
            ###
            utils.maybe_download_and_extract(Constants.inception_download_url, ".")
            self.__graph_def = tfc.GraphDef()
            self.__graph_def.ParseFromString(f.read())
            self.__t_input = tfc.placeholder(np.float32, name = 'input')
            #default 117.0
            t_preprocessed = tf.expand_dims(self.__t_input - Constants.imagenet_mean, 0)
            tfc.import_graph_def(self.__graph_def, {'input':t_preprocessed})
        self.__resize = self.create_tf_function(np.float32, np.int32)(Dreamer.resize_image)

    def get_layer_tensor(self, layer_name: str):
        return self.graph.get_tensor_by_name("import/%s:0" % layer_name)

    @staticmethod
    def create_tf_function(*argument_types):
        placeholders = list(map(tfc.placeholder, argument_types))

        def wrap_function(user_function):
            output_tensor = user_function(*placeholders)

            def wrapper(*arguments, **kwargs):
                feed_dict = dict(zip(placeholders, arguments))
                session = kwargs.get("session")
                return output_tensor.eval(feed_dict, session=session)

            return wrapper
        return wrap_function

    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple):
        expanded = tf.expand_dims(image, 0)
        resized = tfc.image.resize_bilinear(expanded, target_size)
        return resized[0, :, :, :]

    @staticmethod
    def calculate_tile_size(num_pixels: int, preferred_tile_size: int = 400) -> int:
        estimated_tiles = max(1, round(num_pixels / preferred_tile_size))
        return math.ceil(num_pixels / estimated_tiles)

    def calculate_gradient_tiled(self, image: np.ndarray, gradient_tensor: np.ndarray, tile_size: int = 550) -> np.ndarray:
        height, width = image.shape[:2]
        shift_x, shift_y = np.random.randint(tile_size, size=2)
        shifted_image = np.roll(np.roll(image, shift_x, axis=1), shift_y, axis=0)
        gradient = np.zeros_like(image)
        half_tile = tile_size // 2
        max_y = max(height - half_tile, tile_size)
        max_x = max(width - half_tile, tile_size)
        for y in range(0, max_y, tile_size):
            for x in range(0, max_x, tile_size):
                tile = shifted_image[y:y + tile_size, x:x + tile_size]
                with tfc.device(self.__device_name):
                    tile_gradient = self.sess.run(gradient_tensor, {self.__t_input: tile})
                gradient[y:y + tile_size, x:x + tile_size] = tile_gradient
        return np.roll(np.roll(gradient, -shift_x, axis=1), -shift_y, axis=0)

    def set_layer(self, layer_name: str, squared: bool = True, first_channel: int = 0, last_channel: int = 1):
        with tfc.device(self.__device_name):
            layer_tensor = self.get_layer_tensor(layer_name)
            channel_slice = layer_tensor[:, :, :, first_channel:last_channel]
            if squared:
                objective_tensor = tfc.square(channel_slice)
            else:
                objective_tensor = channel_slice
            score_tensor = tfc.reduce_mean(objective_tensor)
            input_gradient = tfc.gradients(score_tensor, self.__t_input)[0]
        return input_gradient

    def _build_pyramids(self, image: np.ndarray, gradient_accumulator: np.ndarray, octave_count: int, octave_scale: float):
        """Build Laplacian pyramids for image and gradient accumulator."""
        image_octaves = []
        gradient_octaves = []
        working_image = image.copy()
        working_gradient = gradient_accumulator.copy()
        for _ in range(octave_count - 1):
            height, width = working_image.shape[:2]
            downscaled = self.__resize(working_image, np.int32(np.float32((height, width)) / octave_scale))
            detail = working_image - self.__resize(downscaled, (height, width))
            working_image = downscaled
            image_octaves.append(detail)
            downscaled_gradient = self.__resize(working_gradient, np.int32(np.float32((height, width)) / octave_scale))
            gradient_detail = working_gradient - self.__resize(downscaled_gradient, (height, width))
            working_gradient = downscaled_gradient
            gradient_octaves.append(gradient_detail)
        return working_image, working_gradient, image_octaves, gradient_octaves

    def _process_octave(self, octave: int, working_image: np.ndarray, gradient_accumulator: np.ndarray,
                        original_image: np.ndarray, image_octaves: list, gradient_octaves: list,
                        renderers: list, layer_gradients: list, iterations: int, iteration_descent: int, octave_count: int):
        """Process a single octave: restore from pyramid, prepare masks/bounds, run iterations."""
        if octave > 0:
            detail = image_octaves[-octave]
            working_image = self.__resize(working_image, detail.shape[:2]) + detail
            gradient_detail = gradient_octaves[-octave]
            gradient_accumulator = self.__resize(gradient_accumulator, detail.shape[:2]) + gradient_detail

        bounds_list = utils.get_bounds(working_image.shape[1], working_image.shape[0], renderers)
        unpacked_bounds = [(b[0], b[1], b[2], b[3]) for b in bounds_list]

        iteration_masks = []
        for renderer in renderers:
            masked = renderer.get("masked", False)
            mask = renderer.get("mask", [])
            if masked and mask:
                iteration_masks.append(self.__resize(mask, working_image.shape[:2]) / 255)
            else:
                iteration_masks.append(None)

        reference_image = self.__resize(original_image, working_image.shape[:2]) / 255
        iterations_this_octave = iterations - octave * iteration_descent

        for iteration in tqdm(
            range(iterations_this_octave),
            desc=f"Octave {octave + 1}/{octave_count}",
            unit="iter",
            leave=True,
        ):
            for renderer_index, renderer in enumerate(renderers):
                render_every = renderer.get("render_x_iteration", 1)
                if (iteration + 1) % render_every != 0:
                    continue

                x_start, x_end, y_start, y_end = unpacked_bounds[renderer_index]
                tile_image = working_image[y_start:y_end, x_start:x_end]

                rotate = renderer.get("rotate", False)
                rotation = renderer.get("rotation", 0)
                if rotate:
                    tile_image = np.rot90(tile_image, rotation)

                tile_size = renderer.get("tile_size", 300)
                gradient = self.calculate_gradient_tiled(tile_image, layer_gradients[renderer_index], tile_size=tile_size)
                step_size = renderer.get("step_size", 1.0)
                gradient = gradient * (step_size / (np.abs(gradient).mean() + 1e-7))

                if rotate:
                    gradient = np.rot90(gradient, 4 - rotation)

                if renderer.get("masked", False):
                    mask = iteration_masks[renderer_index]
                    if mask is not None:
                        gradient *= mask[y_start:y_end, x_start:x_end]

                if renderer.get("color_correction", False):
                    cc_vars = renderer.get("cc_vars", [1, 4, 4, 4])
                    gradient = utils.gradient_grading(
                        gradient,
                        reference_image[y_start:y_end, x_start:x_end],
                        method=cc_vars[0] if len(cc_vars) > 0 else 1,
                        fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                        fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                        fb=cc_vars[3] if len(cc_vars) > 3 else 4,
                    )

                working_image[y_start:y_end, x_start:x_end] += gradient
                gradient_accumulator[y_start:y_end, x_start:x_end] += gradient
        return working_image, gradient_accumulator

    def dream_image(self, image: np.ndarray, settings_raw: dict[str, Any], out_name: str) -> None:
        settings = validate_and_fill_settings(settings_raw)
        iterations = settings.get("iterations", 20)
        octave_count = settings.get("octaves", 4)
        octave_scale = settings.get("octave_scale", 1.5)
        iteration_descent = settings.get("iteration_descent", 0)
        save_gradient = settings.get("save_gradient", False)
        renderers = settings.get("renderers", [])
        layer_gradients = []
        for renderer in renderers:
            layer_name = renderer.get("layer", "conv2d1_pre_relu")
            squared = renderer.get("squared", True)
            first_channel = renderer.get("f_channel", 0)
            last_channel = renderer.get("l_channel", 64)
            layer_gradients.append(self.set_layer(layer_name, squared, first_channel, last_channel))
        gradient_accumulator = np.zeros_like(image)
        working_image, gradient_accumulator, image_octaves, gradient_octaves = self._build_pyramids(
            image, gradient_accumulator, octave_count, octave_scale
        )
        for octave in range(octave_count):
            working_image, gradient_accumulator = self._process_octave(
                octave, working_image, gradient_accumulator, image, image_octaves, gradient_octaves,
                renderers, layer_gradients, iterations, iteration_descent, octave_count
            )
        if settings.get("color_correction", False):
            cc_vars = settings.get("cc_vars", [1, 4, 4, 4])
            working_image = image + utils.gradient_grading(
                gradient_accumulator,
                image / 255,
                method=cc_vars[0] if len(cc_vars) > 0 else 1,
                fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                fb=cc_vars[3] if len(cc_vars) > 3 else 4,
            )
        utils.save_image(working_image, out_name)
        if save_gradient:
            utils.save_image(gradient_accumulator, "gradient_" + out_name)

_instance = Dreamer()
sess = _instance.sess
graph = _instance.graph


def close_and_reopen_session() -> None:
    global sess
    _instance.sess.close()
    _instance.sess = tfc.InteractiveSession(graph=_instance.graph)
    sess = _instance.sess


def close_session() -> None:
    global sess
    _instance.sess.close()
    sess = _instance.sess


def dream_image(image: np.ndarray, settings: dict[str, Any], out_name: str) -> None:
    _instance.dream_image(image, settings, out_name)
