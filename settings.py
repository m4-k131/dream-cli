# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:55:31 2020

"""




class Setting():
    def __init__(self):
        self.name="default_setting"
        self.iterations=20
        self.octaves=4
        self.octave_scale=1.5
        self.iteration_descent=0
        
        self.save_gradient=False
        self.background=[0,0,0]
        
        self.renderers=[]
    
        self.r_names=[]
        self.tf_seed=1
        
        self.output_name="default"
    def set_name(self, t_name):
        self.name=t_name
        
    def set_iterations(self, t_iterations):
        if t_iterations >0:
            self.iterations=t_iterations
        else:
            print('Iterations have to be larger than 0')
            
    def set_octaves(self, t_octaves):
        if t_octaves >0:
            self.octaves=t_octaves
        else:
            print('Octaves have to be larger than 0')
    def set_iteration_descent(self, t_iteration_descent):
        if t_iteration_descent*self.octaves > 1:
            self.iteration_descent=t_iteration_descent
        else:
            print('Octaves*Iteration_descent has to be larger than 0')
            
    def disable_gradient_save(self):
        self.save_gradient=False
        
    def enable_gradient_save(self, t_background=[0,0,0]):
        self.save_gradient=True
        self.background=t_background
    def add_renderer(self):
        self.renderers.append(Renderer())
        
    def remove_renderer(self, i):
        if len(self.renderers)<i:
            print('Invalid renderer index')
        else:
            del self.renderers[i]
        
    def set_seed(self, t_seed):
        self.tf_seed=t_seed
    def print_settings(self):
        
        print('placeholder')
    
    def print_renderers(self):
        for renderer in self.renderers:
            renderer.print_settings()
            
    
class Renderer():
    name="default_renderer"
    layer="layer"
    f_channel=0
    l_channel=1
    max_channel=1
    squared=True
    ##
    step_size=1
    
    ##
    cropped=False
    boundraries=[[0,1],[0,1]]
    ###
    
    rgb_filter=[[0,256],[0,256],[0,256]]
    filtered=False
    
    
    ##
    rotate=False
    rotation=0
    
    ##
    tile_size=300
    ###
    multiply_colors=False
    immidiate_cm=True
    
    render_x_iteration=1
    color_mult=[1,1,1]
    def set_layer(self, layername, layer_length):
        self.layer=layername
        self.f_channel=0
        self.l_channel=layer_length
        self.max_channel=layer_length
    
    def set_channels(self, t_f_channel, t_l_channel):
        if t_f_channel <0 or t_l_channel < t_f_channel or t_l_channel>self.max_channel:
            print('Invalid input')
        else:
            self.f_channel=t_f_channel
            self.l_channel=t_f_channel
            
            
    def set_squared(self, t_squared=True):
        self.squared=t_squared
        
        
    def set_step_size(self, t_step_size):
        self.step_size=t_step_size
        
    
    def set_cropped(self, t_boundraries, t_cropped=True):
        self.cropped=t_cropped
        if self.cropped:
            self.boundraries=t_boundraries
            
            
        
    def set_rgb_filter(self, r=[],g=[],b=[]):
        self.rgb_filter=[]
        self.rgb_filter.append(r)
        self.rgb_filter.append(g)
        self.rgb_filter.append(b)
        self.filtered=True
        
        
    def set_rotation(self, t_rotation):
        if t_rotation in range(0, 3):
            self.rotate=True
            self.rotation=t_rotation
    def disable_rotation(self):
        self.rotate=False
        
        
    def set_tile_size(self, t_tile_size=300):
        self.tile_size=t_tile_size
            
    def set_color_mult(self, t_immidiate, t_color_mult):
        self.immidiate_cm=t_immidiate
        self.multiply_colors=True
        self.color_mult=t_color_mult
    def disable_color_mult(self):
        self.multiplay_colors=False
        
        
    def print_settings(self):
        print('placeholder')
    def init_default(self):
        print('test')
orig_image_name="No image selected"
orig_image=[]
c_settings=Setting()