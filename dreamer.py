# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:53:10 2020


TF-functions copied from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/14_DeepDream.ipynb
"""

import PIL as PIL
import numpy as np
import tensorflow as tf
import utils as utils
import tensorflow.compat.v1 as tfc
import math
 
 

model_fn = "tensorflow_inception_graph.pb"
inception_download_url="http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
save_folder="./renderedImages/4"
 
graph = tf.Graph()
sess = tfc.InteractiveSession(graph=graph)



###Activate GPU rendering if possible
device_name='CPU:0'
if len(tf.config.experimental.list_physical_devices('GPU'))>0:
    device_name='GPU:0'

 
print('Ignore the following Gfile-warning:')
with tfc.gfile.FastGFile(model_fn, 'rb') as f:
    ###
    utils.maybe_download_and_extract(inception_download_url, ".")
    graph_def = tfc.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tfc.placeholder(np.float32, name = 'input')
#default 117.0
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})
def load_image(filename):
    image = PIL.Image.open(filename)
    global img_name
    img_name=filename
    return np.float32(image)
 
def T(layer):
    #print("Processing...")
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0" % layer)
 
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
 
# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tfc.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)
 
def get_tile_size(num_pixels, tile_size=400):
   
    num_tiles = int(round(num_pixels / tile_size))
    num_tiles = max(1, num_tiles)
    actual_tile_size = math.ceil(num_pixels / num_tiles)
    return actual_tile_size
def calc_grad_tiled(img, t_grad, tile_size=550):
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
            with tf.device(device_name):
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)
 

 
def set_layer(layer, squared, int1, int2):
    if squared:
        t_obj=tf.square(T(layer)[:,:,:,int1:int2])
    else:
        t_obj=T(layer)[:,:,:,int1:int2]
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  
   
    return t_grad
   

def dream_image(image, settings, out_name):
    print("Processing...")

    
    ####Global settings for every renderer
    img = image
      #Iterations and octaves
    iterations=settings.iterations
    octave_n=settings.octaves
    octave_scale=settings.octave_scale
    iteration_descent=settings.iteration_descent
    
    #Additional Settings
    save_gradient=settings.save_gradient
    background_color=settings.background
    tf.random.set_seed(settings.tf_seed)#not working?
    
    
    #Renderers
    renderers=settings.renderers
    
    
    ###Set layers and channels
    t_obs=[]
    for r in renderers:
        t_obs.append(set_layer(r.layer, r.squared, r.f_channel, r.l_channel))
       
    ##Create background
    
    background = np.zeros_like(img)
    background[:,:]=background_color
    # split the image into a number of octaves
    
    octaves = []
    backgrounds = []
    
    #Prepare Image & backgrounds for every octave
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

        
        lo = resize(background, np.int32(np.float32(hw) / octave_scale))
        hi = background - resize(lo, hw)
        background = lo
        backgrounds.append(hi)
        

    # generate details octave by octave
    for octave in range(octave_n):

        ##Prepare current Octave
        if octave > 0:
            hi = octaves[-octave]

            img = resize(img, hi.shape[:2]) + hi
            hi_b =backgrounds[-octave]
            
            background = resize(background, hi.shape[:2]) + hi_b
            
        ##More Preperations
        bounds=get_bounds(img.shape[1], img.shape[0], renderers)
        filtered_pixel=get_filtered_pixel(img, renderers, bounds)
        
        
####Iterations
        
        for iteration in range(iterations-octave*iteration_descent):
            
            print("Iteration "+str(iteration+1)+" / "+str(iterations-octave*iteration_descent) + " Octave: " +str(octave+1)+" / "+str(octave_n))
####Gradient

            gradients=[]
            for i in range(len(renderers)):
                if (iteration+1)%renderers[i].render_x_iteration==0:
                
                    ##Pre Gradient preperations
                    
                    #Crop the image
                    t_img=img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]
                    
                    #Rotate if true
                    if renderers[i].rotate:
                        t_img=np.rot90(t_img, renderers[i].rotation)
                        
                        
                    ##Get the gradient
                    g=calc_grad_tiled(t_img,
                                      t_obs[i],
                                      tile_size=renderers[i].tile_size)
    
                    g=g * (renderers[i].step_size / (np.abs(g).mean() + 1e-7))
                    ##
                    ##Gradient manipulations:
                    ##
                    #Rotate back if necessary
                    if renderers[i].rotate:
                        g=np.rot90(g, 4-renderers[i].rotation)
                    #delete filtered pixels
                    for fp in filtered_pixel[i]:
                        g[fp[0], fp[1]]=[0,0,0]
                        
                    #Multiply Color Channels
                    if renderers[i].multiply_colors:
                        g[:,:,0] *=renderers[i].color_mult[0]
                        g[:,:,1] *=renderers[i].color_mult[1]
                        g[:,:,2] *=renderers[i].color_mult[2]
                        
                        
                    ##Adding the finalized Gradient to list
                    gradients.append(g)
            
                else:
                    gradients.append(np.zeros_like(img))
            for i in range(len(bounds)):
                
                img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]] += gradients[i]
                if save_gradient:
                    background[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]+= gradients[i]
        
                

####Save Image


    
    
    utils.save_image(img, out_name)
    if save_gradient:
        utils.save_image(background, 'background_'+out_name)


def get_bounds(x_max, y_max, renderers):
    t_bounds=[]
    for r in renderers:
        if r.cropped:
            x_lower=r.boundraries[0][0]*x_max
            x_upper=r.boundraries[0][1]*x_max
            y_lower=r.boundraries[1][0]*y_max
            y_upper=r.boundraries[1][1]*y_max
            t_bounds.append([int(x_lower), int(x_upper), int(y_lower), int(y_upper)])
        else:
            t_bounds.append([0,x_max,0,y_max])
            
    return t_bounds
    
def get_filtered_pixel(img, renderers, bounds):
    '''
    Way too slow
    '''

    all_fp=[]
    for i in range(len(renderers)):
        
        t_fp=[]
        if renderers[i].filtered:
            print('Filtering. This will take some time.')
            rgb_filter=renderers[i].rgb_filter
            t_img=img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]
            for x in range(t_img.shape[0]):
                for y in range(t_img.shape[1]):
                    current_pixel=t_img[x,y]
                    if  current_pixel[0] in range(rgb_filter[0][0],rgb_filter[0][1]):
                        if current_pixel[1] in range(rgb_filter[1][0],rgb_filter[1][1]):
                            if not current_pixel[2] in range(rgb_filter[2][0],rgb_filter[2][1]):
                                t_fp.append([x,y])
                        else:
                            t_fp.append([x,y])
                    else:
                        t_fp.append([x,y])
        all_fp.append(t_fp)
        
    return all_fp

