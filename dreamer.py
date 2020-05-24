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
import time as time
 
 

model_fn = "tensorflow_inception_graph.pb"
inception_download_url="http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
save_folder="./renderedImages/4"



###Activate GPU rendering if possible
device_name='CPU:0'
if len(tf.config.experimental.list_physical_devices('GPU'))>0:
    device_name='GPU:0'

with tfc.device(device_name):
    graph = tfc.Graph()
    sess = tfc.InteractiveSession(graph=graph)
    
    
#Load model and build Graph
print('Ignore the following Gfile-warning:')
with tfc.gfile.FastGFile(model_fn, 'rb') as f, tfc.device(device_name):
    ###
    utils.maybe_download_and_extract(inception_download_url, ".")
    graph_def = tfc.GraphDef()
    graph_def.ParseFromString(f.read())
    t_input = tfc.placeholder(np.float32, name = 'input')
#default 117.0
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tfc.import_graph_def(graph_def, {'input':t_preprocessed})
    
 
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
            with tfc.device(device_name):
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)
 

 
def set_layer(layer, squared, int1, int2):
    
    with tfc.device(device_name):

        if squared:
            t_obj=tfc.square(T(layer)[:,:,:,int1:int2])
        else:
            t_obj=T(layer)[:,:,:,int1:int2]
        t_score = tfc.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tfc.gradients(t_score, t_input)[0]  
   
    return t_grad
   

def dream_image(image, settings, out_name):
    print("Processing...")

    
    ####Global settings for every renderer
    img = image
    orig_img=image#maybe clean up


    

      #Iterations and octaves
    iterations=settings['iterations']
    octave_n=settings['octaves']
    octave_scale=settings['octave_scale']
    iteration_descent=settings['iteration_descent']
    
    #Additional Settings
    save_gradient=settings['save_gradient']
    background_color=settings['background']
    

    #tf.random.set_seed(settings.tf_seed)#not working?
    
    
    #Renderers
    renderers=settings['renderers']
    
    
    ###Set layers and channels
    t_obs=[]
    for r in renderers:
        t_obs.append(set_layer(r['layer'], r['squared'], r['f_channel'], r['l_channel']))
       
    ##Create background
    
    g_sum = np.zeros_like(img)
    
    # split the image into a number of octaves
    
    octaves = []
    g_sums = []
    
    #Prepare Image & backgrounds for every octave
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

        
        lo = resize(g_sum, np.int32(np.float32(hw) / octave_scale))
        hi = g_sum - resize(lo, hw)
        g_sum = lo
        g_sums.append(hi)
        

    # generate details octave by octave
    for octave in range(octave_n):

        ##Prepare current Octave
        if octave > 0:
            hi = octaves[-octave]

            img = resize(img, hi.shape[:2]) + hi
            
            hi_g =g_sums[-octave]
            
            g_sum = resize(g_sum, hi.shape[:2]) + hi_g
            
        ##More Preperations
        bounds=utils.get_bounds(img.shape[1], img.shape[0], renderers)
        iteration_masks=[]
        for r in renderers:
            if r['masked']:
                iteration_masks.append(resize(r['mask'], img.shape[:2])/255)#move up, /255 just once
            else:
                iteration_masks.append([])
        orig_img_m=resize(orig_img,img.shape[:2])/255#move up, /255 just once
        
####Iterations
        
        for iteration in range(iterations-octave*iteration_descent):
            
            print("Iteration "+str(iteration+1)+" / "+str(iterations-octave*iteration_descent) + " Octave: " +str(octave+1)+" / "+str(octave_n))
####Gradient

            gradients=[]
            
            
            for i in range(len(renderers)):
                if (iteration+1)%renderers[i]['render_x_iteration']==0:
                    start_time=time.time()
                    ##Pre Gradient preperations
                    
                    #Crop the image
                    t_img=img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]
                    
                    #Rotate if true
                    if renderers[i]['rotate']:
                        t_img=np.rot90(t_img, renderers[i]['rotation'])
                        
                        
                    ##Get the gradient
                    g=calc_grad_tiled(t_img,
                                      t_obs[i],
                                      tile_size=renderers[i]['tile_size'])
    
                    g=g * (renderers[i]['step_size'] / (np.abs(g).mean() + 1e-7))
                    
                    ##
                    ##Gradient manipulations:
                    ##
                    #Rotate back if necessary
                    if renderers[i]['rotate']:
                        g=np.rot90(g, 4-renderers[i]['rotation'])

                        
                    #Masking the gradient
                    if renderers[i]['masked']:
                        g*=iteration_masks[i][bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]
                        
                    #Color Correction
                    if renderers[i]['color_correction']:
                        g=utils.gradient_grading(g, orig_img_m[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]],
                                                 method=renderers[i]['cc_vars'][0],
                                                 fr=renderers[i]['cc_vars'][1],
                                                 fg=renderers[i]['cc_vars'][2],
                                                 fb=renderers[i]['cc_vars'][3])
                        
                        
                    ##Adding the finalized Gradient to list
                    gradients.append(g)
                    
                    print('Finished computing Gradient for Renderer {} in {}s'.format(i, time.time()-start_time))
                else:
                    gradients.append(np.zeros_like(img))
            for i in range(len(bounds)):
                
                img[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]] += gradients[i]
                g_sum[bounds[i][2]:bounds[i][3],bounds[i][0]:bounds[i][1]]+= gradients[i]
        
                

####Save Image
    if settings['color_correction']:
        img=image+utils.gradient_grading(g_sum, image/255,
                                                 method=settings['cc_vars'][0],
                                                 fr=settings['cc_vars'][1],
                                                 fg=settings['cc_vars'][2],
                                                 fb=settings['cc_vars'][3])

    
    g_sum[:,:]+=background_color
    utils.save_image(img, out_name)
    if save_gradient:
        utils.save_image(g_sum, 'gradient_'+out_name)



    