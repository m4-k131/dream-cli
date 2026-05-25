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
import time as time
from typing import Any

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
        self.__resize = self.tffunc(np.float32, np.int32)(Dreamer.resize)
    
    
    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name("import/%s:0" % layer)
 

    @staticmethod
    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tfc.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap
    

    @staticmethod
    def resize(img:np.ndarray, size:tuple):
        img = tf.expand_dims(img, 0)
        return tfc.image.resize_bilinear(img, size)[0,:,:,:]
 

    @staticmethod
    def get_tile_size(num_pixels:int, tile_size:int = 400) -> int:
        num_tiles = int(round(num_pixels / tile_size))
        num_tiles = max(1, num_tiles)
        actual_tile_size = math.ceil(num_pixels / num_tiles)
        return actual_tile_size


    def calc_grad_tiled(self, img:np.ndarray, t_grad:np.ndarray, tile_size=550) -> np.ndarray:
        '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                with tfc.device(self.__device_name):
                    g = self.sess.run(t_grad, {self.__t_input:sub})
                    grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)
 
 
    def set_layer(self, layer, squared:bool = True, first_channel:int = 0, last_channel:int = 1):
        with tfc.device(self.__device_name):
            if squared:
                t_obj=tfc.square(self.T(layer)[:,:,:,first_channel:last_channel])
            else:
                t_obj=self.T(layer)[:,:,:,first_channel:last_channel]
            t_score = tfc.reduce_mean(t_obj)  # defining the optimization objective
            t_grad = tfc.gradients(t_score, self.__t_input)[0]  
        return t_grad
   

    def dream_image(self, image:np.ndarray, settings_raw: dict[str, Any], out_name: str) -> None:
        print("Processing...")

        # Validate and fill defaults from schema
        settings = validate_and_fill_settings(settings_raw)

        ####Global settings for every renderer
        img = image
        #Iterations and octaves (with safe .get() and defaults)
        iterations = settings.get("iterations", 20)
        octave_n = settings.get("octaves", 4)
        octave_scale = settings.get("octave_scale", 1.5)
        iteration_descent = settings.get("iteration_descent", 0)
        #Additional Settings
        save_gradient = settings.get("save_gradient", False)
        background_color = settings.get("background", [0, 0, 0])
        #Renderers
        renderers = settings.get("renderers", [])
        ###Set layers and channels
        t_obs = []
        for r in renderers:
            layer_name = r.get("layer", "conv2d1_pre_relu")
            squared = r.get("squared", True)
            f_channel = r.get("f_channel", 0)
            l_channel = r.get("l_channel", 64)
            t_obs.append(self.set_layer(layer_name, squared, f_channel, l_channel))
        ##Create background
        g_sum = np.zeros_like(img)
        # split the image into a number of octaves
        octaves = []
        g_sums = []
        #Prepare Image & backgrounds for every octave
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = self.__resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - self.__resize(lo, hw)
            img = lo
            octaves.append(hi)
            lo = self.__resize(g_sum, np.int32(np.float32(hw) / octave_scale))
            hi = g_sum - self.__resize(lo, hw)
            g_sum = lo
            g_sums.append(hi)
        # generate details octave by octave
        for octave in range(octave_n):
            ##Prepare current Octave
            if octave > 0:
                hi = octaves[-octave]
                img = self.__resize(img, hi.shape[:2]) + hi
                hi_g =g_sums[-octave]
                g_sum = self.__resize(g_sum, hi.shape[:2]) + hi_g
            ##More Preperations
            bounds = utils.get_bounds(img.shape[1], img.shape[0], renderers)
            iteration_masks = []
            for r in renderers:
                masked = r.get("masked", False)
                mask = r.get("mask", [])
                if masked and mask:
                    iteration_masks.append(self.__resize(mask, img.shape[:2]) / 255)  # move up, /255 just once
                else:
                    iteration_masks.append([])
            orig_img_m=self.__resize(image,img.shape[:2])/255#move up, /255 just once
            ####Iterations
            for iteration in range(iterations-octave*iteration_descent):
                print("Iteration "+str(iteration+1)+" / "+str(iterations-octave*iteration_descent) + " Octave: " +str(octave+1)+" / "+str(octave_n))
                ####Gradient
                gradients = []
                for i in range(len(renderers)):
                    r = renderers[i]
                    render_every = r.get("render_x_iteration", 1)
                    if (iteration + 1) % render_every == 0:
                        start_time = time.time()
                        # Crop the image
                        t_img = img[bounds[i][2]:bounds[i][3], bounds[i][0]:bounds[i][1]]
                        # Rotate if true
                        rotate = r.get("rotate", False)
                        rotation = r.get("rotation", 0)
                        if rotate:
                            t_img = np.rot90(t_img, rotation)
                        # Get the gradient
                        tile_size = r.get("tile_size", 300)
                        g = self.calc_grad_tiled(t_img, t_obs[i], tile_size=tile_size)
                        step_size = r.get("step_size", 1.0)
                        g = g * (step_size / (np.abs(g).mean() + 1e-7))
                        # Gradient manipulations:
                        # Rotate back if necessary
                        if rotate:
                            g = np.rot90(g, 4 - rotation)
                        # Masking the gradient
                        if r.get("masked", False):
                            g *= iteration_masks[i][bounds[i][2]:bounds[i][3], bounds[i][0]:bounds[i][1]]
                        # Color Correction
                        if r.get("color_correction", False):
                            cc_vars = r.get("cc_vars", [1, 4, 4, 4])
                            g = utils.gradient_grading(
                                g,
                                orig_img_m[bounds[i][2]:bounds[i][3], bounds[i][0]:bounds[i][1]],
                                method=cc_vars[0] if len(cc_vars) > 0 else 1,
                                fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                                fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                                fb=cc_vars[3] if len(cc_vars) > 3 else 4,
                            )
                        # Adding the finalized Gradient to list
                        gradients.append(g)
                        print("Finished computing Gradient for Renderer {} in {}s".format(i, time.time() - start_time))
                    else:
                        gradients.append(np.zeros_like(img))
                for i in range(len(bounds)):
                    img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]] += gradients[i]
                    g_sum[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]+= gradients[i]
        ####Save Image
        if settings.get("color_correction", False):
            cc_vars = settings.get("cc_vars", [1, 4, 4, 4])
            img = image + utils.gradient_grading(
                g_sum,
                image / 255,
                method=cc_vars[0] if len(cc_vars) > 0 else 1,
                fr=cc_vars[1] if len(cc_vars) > 1 else 4,
                fg=cc_vars[2] if len(cc_vars) > 2 else 4,
                fb=cc_vars[3] if len(cc_vars) > 3 else 4,
            )
        g_sum[:,:]+=background_color
        utils.save_image(img, out_name)
        if save_gradient:
            utils.save_image(g_sum, 'gradient_'+out_name)


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
