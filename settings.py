# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:55:31 2020

"""
import os
import numpy as np
def write_all(layername):
    rend=get_default_renderer()
    rend['layer']=layername
    for file in os.listdir():
        splitted=file.split('_')
        rend['f_channel']=int(splitted[1])
        rend['l_channel']=int(splitted[2].split('.')[0])
        save_renderer(rend, layername+'_'+str(rend['l_channel'])+'.json')

import json

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
        'layer':"layer",
        'f_channel':0,
        'l_channel':1,
        'max_channel':1,
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
def save_renderer(r, filename):
    to_file=r
    
    #r['mask']=r['mask'].tolist()
    with open(filename, "w") as fp:
        json.dump(to_file, fp, indent=4)


def save_setting(s, filename):
    to_file=s
    for renderer in to_file['renderer']:
        renderer=renderer.tolist()
    with open(filename, 'w') as fp:
        json.dump(to_file, fp, indent=4)
        

def load_renderer(filename):
    with open(filename, 'r') as fp:
        data=json.load(fp)
    return data


orig_image_name="No image selected"
orig_image=[]
c_settings=get_default_setting()