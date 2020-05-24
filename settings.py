# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:55:31 2020

"""
import os
import numpy as np

def get_default_setting():
    setting_dict={
        'name':"default_setting",
        'iterations':20,
        'octaves':4,
        'octave_scale':1.5,
        'iteration_descent':0,
        'save_gradient':False,
        'background':[0,0,0],
        'renderers':[],
        'color_correction':False,
        'cc_vars':[1, 4, 4, 4],
        
        
        }
    return setting_dict
    
def get_default_renderer():
    renderer_dict={
        'name':"default_renderer",
        'layer':"conv2d1_pre_relu",
        'f_channel':0,
        'l_channel':64,
        'max_channel':64,
        'squared':True,
        'cropped':False,
        'boundraries':[[0,1],[0,1]],
        'rotate':False,
        'rotation':0,
        'tile_size':300,
		'render_x_iteration':1,
		'step_size':1,
        'color_correction':False,
        'cc_vars':[1, 4, 4, 4],
		'masked': False,
        'mask':[],
        'mask_name':'',
        't_masks':[]
        }
    return renderer_dict

orig_image_name="No image selected"
orig_image=[]
c_settings=get_default_setting()