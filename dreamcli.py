# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:45:48 2020
@author: M.K.
"""

import os as os
import settings as settings
import utils as utils
import dreamer as d
import json
import numpy as np


image_dir='./Images'
settings_dir='./Settings'
renderer_dir='./Settings/Renderer'
longline='------------------------'
mediumline='-----------'
shortline='-----'


"""
Input Parser
"""
def parse_input(lower=0, upper=1, msg="Enter selection: ", t="int"):
    correct_input=False
    while correct_input==False:
        i=input(msg)
        if len(i)>0:
            if t=="int":
                try:
                    c_i=int(i)
                    if c_i in range(lower, upper+1):
                        correct_input=True
                    else:
                        print('Input out of bounds: Input must be between '+str(lower)+" and "+str(upper))
                except ValueError:
                    print('Invalid input: ValueError. Input must be a number e.g. 1')
            elif t=='float':
                try:
                    c_i=float(i)
                    if c_i <= upper and c_i>=lower:
                        correct_input=True
                    else:
                        print('Input out of bounds: Input must be between '+str(lower)+" and "+str(upper))
                except ValueError:
                    print('Invalid input: ValueError. Input must be a number e.g. 0.5')
            else:
                c_i=i
                if " " in c_i:
                    print('Input contains a blank space. Please use "_" instead to avert possible errors.')
                else:
                    correct_input=True
        else:
            print('Invalid input: No input')
    return c_i

def parse_yn(msg='Placeholder'):
    correct_input=False
    r_yn=False
    while correct_input==False:
        yn=input(msg)
        if yn=="y" or yn=="Y":
            correct_input=True
            r_yn=True
        elif yn=="n" or yn=="N":
            correct_input=True
        else:
            print('Invalid input: Only y/Y (yes) and n/N (no) accepted')
    return r_yn

def parse_or(msg='Enter "o" to overwrite or "r" to rename:'):
    correct_input=False
    overwrite=False
    while correct_input==False:
        yn=input(msg)
        if yn=="o" or yn=="O":
            correct_input=True
            overwrite=True
        elif yn=="r" or yn=="R":
            correct_input=True
        else:
            print('Invalid input: Only o/O (letter) and r/R  accepted')
    return overwrite

"""
Main Menu
"""

def main_menu():
    print(longline)
    print(shortline+'Main menu')
    print(longline)
    if len(settings.orig_image)>0:
        print('Selected image: '+settings.orig_image_name)
        print('Image size:'+str(settings.orig_image.shape[0])+"x"+str(settings.orig_image.shape[1]))
    else:
        print('No image selected')
    print('Settings: %s'%settings.c_settings['name'])
    print('Renderer: %d Renderer selected'%len(settings.c_settings['renderers']))
#        settings=settings.Setting()
#        image_dir='./Images'
#        settings_dir='./Settings'
    print(longline)
##Print 
    print('0: Select image \n1: Edit Settings \n2: Edit Renderer \n3: Continue(deepdream Image)')
    print(longline)
    
    selection=parse_input(0, 3, 'Enter corrosponding number to navigate:  ')

    if selection==0:
        settings.orig_img=select_image_menu()
    if selection==1:
        change_settings()
    if selection==2:
        renderer_menu()
    if selection==3:
        print(longline)
        if len (settings.orig_image)>0:
            if len(settings.c_settings['renderers'])>0:
                dream()
            else:
                print('No renderer selected! Add a renderer to dream the image')
        else:
            print('No Image selected')
        print(longline)
    
    
              
    if not selection ==4:
        main_menu()

"""
Image functions
"""
def load_jpg():
    print('0: back')
    #Lists of files
    jpgs=[]
    jpgs_names=[]
    all_files=os.listdir(image_dir)
    #Check for .jpg
    jpg_index=0
    for i in range(len(all_files)):
        
        if all_files[i].endswith(".jpg") or all_files[i].endswith(".JPEG"):
            print(str(jpg_index+1)+": "+all_files[i])
            jpgs.append(os.path.join(image_dir, all_files[i]))
            jpgs_names.append(all_files[i])
            jpg_index+=1
        
    #Input
    selection=parse_input(0, len(jpgs))

    if selection >0:
        return utils.load_image(jpgs[selection-1]), jpgs_names[selection-1]
    else:
        return [], ""
                        
def select_image_menu():

    print('0: back')
    #Lists of files
    jpgs=[]
    jpgs_names=[]
    all_files=os.listdir(image_dir)
    #Check for .jpg
    jpg_index=0
    for i in range(len(all_files)):
        
        if all_files[i].endswith(".jpg") or all_files[i].endswith(".JPEG"):
            print(str(jpg_index+1)+": "+all_files[i])
            jpgs.append(os.path.join(image_dir, all_files[i]))
            jpgs_names.append(all_files[i])
            jpg_index+=1
        
    #Input
    selection=parse_input(0, len(jpgs))

    if selection >0:
        settings.orig_image=select_image(jpgs[selection-1])
        settings.orig_image_name=jpgs_names[selection-1]

def select_image(image):
    t_img=utils.load_image(image)
    return t_img

"""
Settings Menu
"""
def change_settings():
    print(longline)
    print(shortline+'Settings menu')
    print(longline)
    print('0:  Back')
    print('1:  Name: %s'%settings.c_settings['name'])
    print('2:  Iterations: %d'%settings.c_settings['iterations'])
    print('3:  Octaves: %d'%settings.c_settings['octaves'])
    print('4:  Octave scale: %.2f'%settings.c_settings['octave_scale'])
    print('5:  Iteration descent: %d'%settings.c_settings['iteration_descent'])
    print('6:  Background menu. Currently activated: '+str(settings.c_settings['save_gradient']))
    print('7:  Global color correction: '+str(settings.c_settings['color_correction']))
    print('8:  Renderer menu. Current number of renderer: %d'%len(settings.c_settings['renderers']))
    print('9:  New Settings')
    print('10: Load Settings')
    print('11: Save Settings')
    print(longline)

    selection=parse_input(0, 11, msg="Enter selection:")
    switcher={1:edit_setting_name,
              2:edit_iterations,
              3:edit_octaves,
              4:edit_octave_scale,
              5:edit_iteration_descent,
              6:edit_background,
              7:edit_global_cc,
              8:renderer_menu,
              9:new_settings,
              10:load_settings_menu,
              11:save_settings}
    if selection >0:
        switcher[selection]()
        change_settings()

        
def edit_setting_name():
    print(mediumline)
    print('Current name: %s'%settings.c_settings['name'])
    print(mediumline)
    print('0: Back \n1:  Change Name')
    print(mediumline)
    selection=parse_input(0, 1, 'Enter selection:')


    if selection==1:
        n=parse_input(0, 2,'Enter Settings name:', t='string')
        settings.c_settings['name']=n
    


def edit_iterations():
    print(mediumline)
    print('Current iterations: '+str(settings.c_settings['iterations']))
    print(mediumline)
    print('0: Back \n1: Change iterations')
    print(mediumline)
    selection=parse_input(0, 1, 'Enter selection:')
    
    if selection==1:
            i=parse_input(0, 10000, 'Enter new Iteration no.:')
            settings.c_settings['iterations']=i

    
def edit_octaves():
    print(mediumline)
    print('Current octaves: '+str(settings.c_settings['octaves']))
    print(mediumline)
    print('0: Back \n1: Change Octaves')
    print(mediumline)
    selection=parse_input(0,1)
    if selection ==1:
        o=parse_input(1, 200, 'Enter Octave no.:')

        settings.c_settings['octaves']=o
            
def edit_octave_scale():
    print(mediumline)
    print('Current octave_scale: '+str(settings.c_settings['octave_scale']))
    print(mediumline)
    print('0: Back \n1: Change octave_scale')
    print(mediumline)
    selection=parse_input(0,1, 'Enter selection:')

    if selection ==1:
        o=parse_input(1, 50, 'Enter Octave scale (Recommended: between 1.1 and 2.0):', t='float')

        settings.c_settings['octave_scale']=o
        
def edit_iteration_descent():
    print(mediumline)
    print('Current iteration_descent: '+str(settings.c_settings['iteration_descent']))
    print(mediumline)
    print('0: Back \n1: Change iteration_descent')
    print(mediumline)
    
    selection=parse_input(0,1)

    if selection==1:
        max_descent=int(settings.c_settings['iterations']/settings.c_settings['octaves'])
        i=parse_input(-100, max_descent, "Enter Iteration descent (Can not be higher than Iterations/octaves=%d):"%max_descent )

        settings.c_settings['iteration_descent']=i
    
def edit_background_color():
    print(shortline)
    print('Current background color: '+str(settings.c_settings['background']))
    print(shortline)
    print('0: Back \n1: Change background color')
    print(shortline)
    selection=parse_input(0,1)

    if selection==1:
        r=parse_input(0,255, 'Enter Red value of background:')
        g=parse_input(0,255, 'Enter Green value of background:')
        b=parse_input(0,255, 'Enter Blue value of background:')
        settings.c_settings['background']=[r,g,b]
        

    
def edit_background():
    print(mediumline)
    
    print('Write gradient onto empty background: '+str(settings.c_settings['save_gradient']))
    print('Background color: '+str(settings.c_settings['background']))
    print(mediumline)
    print('0: Back \n1: Activate/Deactivate if gradient is saved onto background \n2: Change background color')
    print(mediumline)
    selection=parse_input(0, 2)
    if selection==1:
        print(shortline)
        print('0: Back \n1: Activate \n2: Deactivate')
        print(shortline)
        selection=parse_input(0,2)
        if selection==0:
            edit_background()
        if selection==1:
            settings.c_settings['save_gradient']=True
            edit_background()
        if selection==2:
            settings.c_c_settings['save_gradient']=False
            edit_background()
    if selection==2:
        edit_background_color()
        edit_background()
        
        
def new_settings():
    settings.c_settings=settings.get_default_setting()
    
"""
Renderer
"""
def edit_global_cc():
    
    print(mediumline)
    print('Global color correction: '+str(settings.c_settings['color_correction']))
    print('Method: '+str(settings.c_settings['cc_vars'][0]))
    print('RGB multiplier: '+str(settings.c_settings['cc_vars'][1:]))
    print(mediumline)
    print('0: Back \n1: Activate multiplier \n2: Deactivate multiplier\n3: Edit color correction values')
    selection=parse_input(0,3)
    if selection==1:
        settings.c_settings['color_correction']=True
        edit_global_cc()
    if selection==2:
        settings.c_settings['color_correction']=False
        edit_global_cc()
    if selection==3:
        settings.c_settings=edit_cc_vars(settings.c_settings)
        edit_global_cc()

    
def renderer_menu():
    print(longline)
    print(shortline+'Renderer menu')
    print(longline)

    
    if len(settings.c_settings['renderers'])>0:
        print('Current renderer:')
        for i in range(0, len(settings.c_settings['renderers'])):
            print("-"+settings.c_settings['renderers'][i]['name'])
    else:
        print('-No Renderer selected-')
    print(longline)
    print('0: Back \n1: Edit renderer \n2: New Renderer \n3: Load Renderer \n4: Remove Renderer \n5: Save Renderer')
    print(longline)
    selection=parse_input(0,5, 'Enter selection: ')
    if selection ==1:
        print(mediumline)
        print('Select Renderer to edit:')
        print(shortline)
        print('0: Back')
        for i in range(0, len(settings.c_settings['renderers'])):
            print(str(i+1)+": "+settings.c_settings['renderers'][i]['name']+' | '+
                  settings.c_settings['renderers'][i]['layer']+
                  '['+str(settings.c_settings['renderers'][i]['f_channel'])
                  +":"+str(settings.c_settings['renderers'][i]['l_channel'])+"]")
        print(shortline)
        selected=parse_input(0, len(settings.c_settings['renderers'])+1)
        
        if selected>0:
            settings.c_settings['renderers'][selected-1]=edit_renderer_settings(settings.c_settings['renderers'][selected-1])
        
    if selection==2:
        add_renderer()
    if selection==3:
        load_renderer_menu()
    if selection==4:
        remove_renderer_menu()
    if selection==5:
        save_renderer_menu()
    if not selection==0:
        renderer_menu()

def remove_renderer_menu():
        print(mediumline)
        print('Select Renderer to remove:')
        print(shortline)
        print('0: Back')
        for i in range(0, len(settings.c_settings['renderers'])):
            print(str(i+1)+": "+settings.c_settings['renderers'][i]['name']+' | '+
                  settings.c_settings['renderers'][i]['layer']+
                  '['+str(settings.c_settings['renderers'][i]['f_channel'])
                  +":"+str(settings.c_settings['renderers'][i]['l_channel'])+"]")
        print(shortline)
        selected=parse_input(0, len(settings.c_settings['renderers'])+1)
        
        if selected>0:
            del settings.c_settings['renderers'][selected-1]
def edit_r_name(r):
    print(mediumline)
    print('Current name: '+str(r['name']))
    print(mediumline)
    print('0: Back \n1: Change name')
    print(mediumline)

    selection = parse_input(0,1)
    if selection==1:
        n=parse_input(0, 0,'Enter name:', t='string')
        r['name']=n
    return r

def add_renderer():

    settings.c_settings['renderers'].append(settings.get_default_renderer())
    settings.c_settings['renderers'][-1]=select_layer_menu(settings.c_settings['renderers'][-1])
    settings.c_settings['renderers'][-1]=edit_renderer_settings(settings.c_settings['renderers'][-1])

def select_layer_menu(r):
    print(mediumline)

    print('Select layer:')
    ll=get_layer_list()
    for i in range(len(ll)):
        print(str(i)+": "+ll[i][0]+" Channels: "+str(ll[i][1]))
    selection=parse_input(0, len(ll))
    r['layer']=(ll[selection][0])
    r['l_channel']=ll[selection][1]
    r['max_channel']=ll[selection][1]
    return r

def edit_layer(r):
    print(mediumline)
    print('Current layer: '+str(r['layer']))
    print(mediumline)
    print('0: Back \n1: Change layer')
    selection=parse_input(0,2)
    if selection ==1:
        r=select_layer_menu(r)
        
    return r
def edit_channels(r):
    print(mediumline)
    print('Current channels: First channel: '+str(r['f_channel'])+" Last channel: "+str(r['l_channel']))
    print(mediumline)
    print('0: Back \n1: Change channels')
    print(mediumline)
    selection=parse_input(0, 2)
    if selection ==1:
        
        f=parse_input(0, r['max_channel']-1, "Enter first channel no. (Must be between 0 and "+str(r['max_channel']-1)+"):")
        l=parse_input(f+1, r.max_channel,"Enter last channel no. (Must be between "+str(f+1)+" and "+str(r['max_channel'])+"):")
        r['f_channel']=f
        r['l_channel']=l
    
    return r
def edit_squared(r):
    print(mediumline)
    print('Squared gradient activated: '+str(r['squared']))
    print(mediumline)
    print('0: Back \n1: Activate squared gradient \n2: Deactivate squared gradient')
    print(mediumline)
    selection=parse_input(0,3)

    if selection==1:
        r['squared']=True
    if selection==2:
        r['squared']=False
    return r
    

def edit_step_size(r):
    print(mediumline)
    print('Current step_size: '+str(r['step_size']))
    print(mediumline)
    print('0: Back \n1: Edit step_size' )
    print(mediumline)
    selection=parse_input(0,2)

    if selection ==1:
        sz=parse_input(-100, 100,'Enter step size (Recommended: between 2 and 0.01): ', t='float')
        r['step_size']=sz
    return r
    
    
    
def edit_cropped(r):
    print(mediumline)
    print('Cropped renderer activated: '+str(r['cropped']))
    print('Boundraries: '+str(r['boundraries']))
    print(mediumline)
    print('0: Back \n1: Activate cropped renderer \n2: Deactivate cropped renderer\n3: Edit boundraries')
    print(mediumline)
    selection=parse_input(0,3)

    if selection ==1:
        r['cropped']=True
    if selection ==2:
        r['cropped']=False
    if selection ==3:
        r=edit_boundraries(r)
        r=edit_cropped(r)
    return r
        
def edit_boundraries(r):
    print(shortline)
    print('All values most be between 0 and 1 on both axes starting on the top left corner (If this renderer should e.g. only render the bottom half choose x_min=0,x_max=1, y_min=0.5,  y_max=1' )
    print(shortline)
    x_min=parse_input(0, 1,'Enter x_min:', t='float')
    x_max=parse_input(x_min, 1,'Enter x_max:', t='float')
    y_min=parse_input(0, 1,'Enter y_min:', t='float')
    y_max=parse_input(y_min,1,'Enter y_max:', t='float')
    r['boundraries']=[[x_min,x_max],[y_min, y_max]]
    return r

        
    
def edit_mask(r):
    
    
    print(mediumline)
    print('Mask activated: '+str(r['masked']))
    print('Mask:' +str(r['mask_name']))
    
    print(mediumline)
    print('0: Back \n1: Activate Mask \n2: Deactivate Mask \n3: Load mask \n4: Remove mask')
    print(mediumline)
    selection=parse_input(0, 4)

    if selection==1 or selection==2:
        r['masked']=not r['masked']
        
        r=edit_mask(r)
    if selection==3:
        r['mask'],r['mask_name']=load_jpg()
        r=edit_mask(r)
    if selection ==4:
        r['mask']=[]
        r['mask_name']=''
        r['maksed']=False
        r=edit_mask(r)
        
    if len(r['mask_name'])==0:
        r['masked']=False
    ######set masked to false if mask is empty

    return r

def edit_rotate(r):
    print(mediumline)
    print('Current rotation: %d° right'%(r['rotation']*90))
    print(mediumline)
    print('0: Back \n1: Rotate 0° right (deactivate rotation) \n2: Rotate 90° right\n3: Rotate 180° right \n4: Rotate 270° right')
    print(mediumline)
    selection=parse_input(0,4)

    if selection>0:
        if selection==1:
            r['rotation']=0
            r['rotate']=False
        else:
            r['rotation']=selection-1
            r['rotate']=True
    return r
        

def edit_tile_size(r):
    print(mediumline)
    print('Current tile size: '+str(r['tile_size']))
    print(mediumline)
    print('0: Back \n1: Edit tile size')
    print(mediumline)
    selection=parse_input(0,1)
    if selection==1:
        tz=parse_input(1, 10000, 'Enter Tile size (should be around 300):')
        r['tile_size']=tz
    
    return r
    
def edit_renderer_cc(r):
    
    print(mediumline)
    print('Renderer specific color correction: '+str(r['color_correction']))
    print('Method: '+str(r['cc_vars'][0]))
    print('RGB multiplier: '+str(r['cc_vars'][1:]))
    print(mediumline)
    print('0: Back \n1: Activate multiplier \n2: Deactivate multiplier\n3: Edit color correction values')
    selection=parse_input(0,3)
    if selection==1:
        r['color_correction']=True
        r=edit_renderer_cc(r)
    if selection==2:
        r=edit_renderer_cc(r)
        r['color_correction']=False
    if selection==3:
        r=edit_cc_vars(r)
        r=edit_renderer_cc(r)

    return r
def edit_cc_vars(rs):
    print(shortline)
    print('Current Method: '+str(rs['cc_vars'][0]))
    print('Current RGB multiplier: '+str(rs['cc_vars'][1:]))
    print(shortline)
    print('0: Back \n1: Edit Method \n2: Edit RGB multiplier')
    print(shortline)
    
    selection=parse_input(0,2)
    if selection==1:
        print('0: Back \n1: Simple Grayscale correction \n2: Retaining original colors \n3: Linear correction')
        print(shortline)
        new_selection=parse_input(0,3)
        if selection>0:
            rs['cc_vars'][0]=new_selection
            rs=edit_cc_vars(rs)
    if selection ==2:
        print(shortline)
        print('Current color multiplier: '+str(rs['cc_vars'][1:]))
        print(shortline)
        print('Enter the color multiplier (mult) for the Red, Green and Blue channel (r,g,b). Recommended values are between -5.0 and 5.0')
        red=parse_input(-100,100,'Enter r_mult:', t='float')
        green=parse_input(-100,100,'Enter g_mult:', t='float')
        blue=parse_input(-100,100,'Enter b_mult:', t='float')
        rs['cc_vars'][1:]=[red,green,blue]
        rs=edit_cc_vars(rs)
            
    return rs
def edit_render_x_iteration(r):
    print(mediumline)
    print('Currently rendering every %d iteration (1=render every Iteration)'%r['render_x_iteration'])
    print(mediumline)
    print('0: back \n1: Edit')
    print(mediumline)
    selection=parse_input(0,1)
    if selection ==1:
        x=parse_input(0, 10000, "Enter new value:")
        r['render_x_iteration']=x
    return r

def edit_renderer_settings(r):
    print(longline)
    print(shortline+'Renderer')
    print(longline)
    print('0:  Back')
    print('1:  Name: %s'%r['name'])
    print('2:  Layer: %s'%r['layer'])
    print('3:  Channels: '+str(r['f_channel'])+"-"+str(r['l_channel']))
    print('4:  Squared: '+str(r['squared']))
    print('5:  Step size: %.5f'%r['step_size'])
    
    if r['cropped']:
        print('6:  Cropped: Activated with Boundraries: '+str(r.boundraries))
    else:
        print('6:  Cropped: '+str(r['cropped']))
        
    
    if r['masked']:
        print('7:  Masked gradient activated ')

    else:
        print('7:  Masked gradient deactivated')
    
    if r['rotate']:
        print('8:  Rotation. Activated with '+ str(r['rotation']*90)+'° right')
    else:
        print('8:  Rotation deactivated')
    print('9:  Tile size: %d'%r['tile_size'])
    
    
    if r['color_correction']:
        print('10: Renderer specific color correction activated ')
    else:
        print('10: Color correction deactivated')
    
    print('11: Render every %d iteration'%r['render_x_iteration'])
    print('12: Save current Renderer to file')
    print(longline)

    selection=parse_input(0,12, 'Enter selection:')
    switcher={1:edit_r_name,
              2:edit_layer,
              3:edit_channels,
              4:edit_squared,
              5:edit_step_size,
              6:edit_cropped,
              7:edit_mask,
              8:edit_rotate,
              9:edit_tile_size,
              10:edit_renderer_cc,
              11:edit_render_x_iteration,
              12:save_renderer}

    if selection>0:
        r=switcher[selection](r)
        edit_renderer_settings(r)

    return r


def save_settings():

    t_name=settings.c_settings['name']+'_s.json'
    file=os.path.join(settings_dir, t_name)
    overwrite=False
    while os.path.exists(file) and overwrite ==False:
        print(t_name+ " already exists. Do you want to overwrite it or rename your current setting?")
        overwrite=parse_or()
        if not overwrite:
            settings.c_settings['name']=parse_input(msg="Enter new name:", t='string')
            t_name=settings.c_settings['name']+'_s.json'
            file=os.path.join(settings_dir, t_name)
        for r in settings.c_settings['renderers']:
            if not r['mask_name']=='':
                r['mask']=r['mask'].tolist()
            
    with open(file, "w") as fp:
        json.dump(settings.c_settings, fp, indent=4)
        
    for r in settings.c_settings['renderers']:
        if not r['mask_name']=='':
            r['mask']=np.array(r['mask'])

def load_settings(name):
    file_path = os.path.join(settings_dir, name)
    file_path=name
    if os.path.exists(file_path):
        with open(file_path, 'r') as fp:
            t_settings=json.load(fp)
        
        """
        Backwards compatibility:
        Get default settings
        
        For key, value pair write settings to default settings
        
        For layer in t_settings:
            get default layer
            for key, value pair writte renderer-settings to default renderer
        """
        settings.c_settings=t_settings
        for r in settings.c_settings['renderers']:
            if not r['mask_name']=='':
                r['mask']=np.array(r['mask'])
    
    else:
        print('File not found. Falling back to default settings')
        t_settings=settings.get_default_setting()

    return t_settings

def load_settings_menu():
    print(mediumline)
    print("0: Back")
    ddss=[] #All .dds-files in folder
    dds_names=[]
    for file in os.listdir(settings_dir):
        if file.endswith('_s.json'):
            ddss.append(os.path.join(settings_dir, file))
            dds_names.append(file)
    for i in range(len(dds_names)):
        print(str(i+1)+": "+dds_names[i])
    print(mediumline)
    selection=parse_input(0, len(dds_names), "Select File to load:")
    if selection>0:
        settings.c_settings['renderers']=[]
        settings.c_settings['r_names']=[]
        settings.c_settings=load_settings(ddss[selection-1])

        

def load_renderer_menu():
    #change variable names2
    print(mediumline)
    print("0: Back")
    ddrs=[]
    ddr_names=[]
    for file in os.listdir(renderer_dir):
        if file.endswith('_r.json'):
            ddrs.append(os.path.join(renderer_dir, file))
            ddr_names.append(file)
    for i in range(len(ddr_names)):
        print(str(i+1)+": "+ddr_names[i])
    print(mediumline)
    selection=parse_input(0, len(ddr_names), "Select File to load:")
    if selection>0:
        settings.c_settings['renderers'].append(load_renderer(ddrs[selection-1]))

def save_renderer(t_renderer):
    t_name=t_renderer['name']+'_r.json'
    filepath=os.path.join(renderer_dir, t_name)
    overwrite=False
    while os.path.exists(filepath) and overwrite==False:
        print(t_name+ " already exists. Do you want to overwrite it or rename your current Renderer?")
        overwrite=parse_or()
        if not overwrite:
            t_renderer['name']=parse_input(msg="Enter new name:", t='string')
            t_name=t_renderer['name']+'_r.json'
            filepath=os.path.join(renderer_dir, t_name)
    if not t_renderer['mask_name'] =='':
        t_renderer['mask']=t_renderer['mask'].tolist()
    with open(filepath, "w") as fp:
        json.dump(t_renderer, fp, indent=4)
    if not t_renderer['mask_name'] =='':
        t_renderer['mask']=np.array(t_renderer['mask'])
    return t_renderer

def save_renderer_menu():
    print(mediumline)
    print('Select Renderer to save:')
    print(shortline)
    print('0: Back')
    for i in range(0, len(settings.c_settings['renderers'])):
        print(str(i+1)+": "+settings.c_settings['renderers'][i]['name'] +' | '+
              settings.c_settings['renderers'][i]['layer']+
              '['+str(settings.c_settings['renderers'][i]['f_channel'])
              +":"+str(settings.c_settings['renderers'][i]['l_channel'])+"]")
    print(shortline)
    selected=parse_input(0, len(settings.c_settings['renderers'])+1)
    
    if selected>0:
        settings.c_settings['renderers'][selected-1]=save_renderer(settings.c_settings['renderers'][selected-1])

def load_renderer(name):
    file_path =name
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path, 'r') as fp:
            t_renderer=json.load(fp)
            """
            Backwards compatibility as in load_settings()
            """
        
    else:
        print('File not found. Adding default renderer')
        t_renderer=settings.get_default_renderer()
    if not t_renderer['mask_name'] =='':
        t_renderer['mask']=np.array(t_renderer['mask'])
    return t_renderer
"""
Dream
"""

def dream():
    d.sess.close()
    out_name=parse_input(msg='Enter output Filename. Will be overwritten if it already exists: ', t='string')
    d.sess = d.tfc.InteractiveSession(graph=d.graph)
    d.dream_image(settings.orig_image, settings.c_settings, out_name)
    d.sess.close()



"""
Data
"""
def get_layer_list():
    ll=[['conv2d0_pre_relu/conv', 64],
 ['conv2d0_pre_relu', 64],
 ['conv2d0', 64],
 ['maxpool0', 64],
 ['localresponsenorm0', 64],
 ['conv2d1_pre_relu/conv', 64],
 ['conv2d1_pre_relu', 64],
 ['conv2d1', 64],
 ['conv2d2_pre_relu/conv', 192],
 ['conv2d2_pre_relu', 192],
 ['conv2d2', 192],
 ['localresponsenorm1', 192],
 ['maxpool1', 192],
 ['mixed3a_1x1_pre_relu/conv', 64],
 ['mixed3a_1x1_pre_relu', 64],
 ['mixed3a_1x1', 64],
 ['mixed3a_3x3_bottleneck_pre_relu/conv', 96],
 ['mixed3a_3x3_bottleneck_pre_relu', 96],
 ['mixed3a_3x3_bottleneck', 96],
 ['mixed3a_3x3_pre_relu/conv', 128],
 ['mixed3a_3x3_pre_relu', 128],
 ['mixed3a_3x3', 128],
 ['mixed3a_5x5_bottleneck_pre_relu/conv', 16],
 ['mixed3a_5x5_bottleneck_pre_relu', 16],
 ['mixed3a_5x5_bottleneck', 16],
 ['mixed3a_5x5_pre_relu/conv', 32],
 ['mixed3a_5x5_pre_relu', 32],
 ['mixed3a_5x5', 32],
 ['mixed3a_pool', 192],
 ['mixed3a_pool_reduce_pre_relu/conv', 32],
 ['mixed3a_pool_reduce_pre_relu', 32],
 ['mixed3a_pool_reduce', 32],
 ['mixed3a', 256],
 ['mixed3b_1x1_pre_relu/conv', 128],
 ['mixed3b_1x1_pre_relu', 128],
 ['mixed3b_1x1', 128],
 ['mixed3b_3x3_bottleneck_pre_relu/conv', 128],
 ['mixed3b_3x3_bottleneck_pre_relu', 128],
 ['mixed3b_3x3_bottleneck', 128],
 ['mixed3b_3x3_pre_relu/conv', 192],
 ['mixed3b_3x3_pre_relu', 192],
 ['mixed3b_3x3', 192],
 ['mixed3b_5x5_bottleneck_pre_relu/conv', 32],
 ['mixed3b_5x5_bottleneck_pre_relu', 32],
 ['mixed3b_5x5_bottleneck', 32],
 ['mixed3b_5x5_pre_relu/conv', 96],
 ['mixed3b_5x5_pre_relu', 96],
 ['mixed3b_5x5', 96],
 ['mixed3b_pool', 256],
 ['mixed3b_pool_reduce_pre_relu/conv', 64],
 ['mixed3b_pool_reduce_pre_relu', 64],
 ['mixed3b_pool_reduce', 64],
 ['mixed3b', 480],
 ['maxpool4', 480],
 ['mixed4a_1x1_pre_relu/conv', 192],
 ['mixed4a_1x1_pre_relu', 192],
 ['mixed4a_1x1', 192],
 ['mixed4a_3x3_bottleneck_pre_relu/conv', 96],
 ['mixed4a_3x3_bottleneck_pre_relu', 96],
 ['mixed4a_3x3_bottleneck', 96],
 ['mixed4a_3x3_pre_relu/conv', 204],
 ['mixed4a_3x3_pre_relu', 204],
 ['mixed4a_3x3', 204],
 ['mixed4a_5x5_bottleneck_pre_relu/conv', 16],
 ['mixed4a_5x5_bottleneck_pre_relu', 16],
 ['mixed4a_5x5_bottleneck', 16],
 ['mixed4a_5x5_pre_relu/conv', 48],
 ['mixed4a_5x5_pre_relu', 48],
 ['mixed4a_5x5', 48],
 ['mixed4a_pool', 480],
 ['mixed4a_pool_reduce_pre_relu/conv', 64],
 ['mixed4a_pool_reduce_pre_relu', 64],
 ['mixed4a_pool_reduce', 64],
 ['mixed4a', 508],
 ['mixed4b_1x1_pre_relu/conv', 160],
 ['mixed4b_1x1_pre_relu', 160],
 ['mixed4b_1x1', 160],
 ['mixed4b_3x3_bottleneck_pre_relu/conv', 112],
 ['mixed4b_3x3_bottleneck_pre_relu', 112],
 ['mixed4b_3x3_bottleneck', 112],
 ['mixed4b_3x3_pre_relu/conv', 224],
 ['mixed4b_3x3_pre_relu', 224],
 ['mixed4b_3x3', 224],
 ['mixed4b_5x5_bottleneck_pre_relu/conv', 24],
 ['mixed4b_5x5_bottleneck_pre_relu', 24],
 ['mixed4b_5x5_bottleneck', 24],
 ['mixed4b_5x5_pre_relu/conv', 64],
 ['mixed4b_5x5_pre_relu', 64],
 ['mixed4b_5x5', 64],
 ['mixed4b_pool', 508],
 ['mixed4b_pool_reduce_pre_relu/conv', 64],
 ['mixed4b_pool_reduce_pre_relu', 64],
 ['mixed4b_pool_reduce', 64],
 ['mixed4b', 512],
 ['mixed4c_1x1_pre_relu/conv', 128],
 ['mixed4c_1x1_pre_relu', 128],
 ['mixed4c_1x1', 128],
 ['mixed4c_3x3_bottleneck_pre_relu/conv', 128],
 ['mixed4c_3x3_bottleneck_pre_relu', 128],
 ['mixed4c_3x3_bottleneck', 128],
 ['mixed4c_3x3_pre_relu/conv', 256],
 ['mixed4c_3x3_pre_relu', 256],
 ['mixed4c_3x3', 256],
 ['mixed4c_5x5_bottleneck_pre_relu/conv', 24],
 ['mixed4c_5x5_bottleneck_pre_relu', 24],
 ['mixed4c_5x5_bottleneck', 24],
 ['mixed4c_5x5_pre_relu/conv', 64],
 ['mixed4c_5x5_pre_relu', 64],
 ['mixed4c_5x5', 64],
 ['mixed4c_pool', 512],
 ['mixed4c_pool_reduce_pre_relu/conv', 64],
 ['mixed4c_pool_reduce_pre_relu', 64],
 ['mixed4c_pool_reduce', 64],
 ['mixed4c', 512],
 ['mixed4d_1x1_pre_relu/conv', 112],
 ['mixed4d_1x1_pre_relu', 112],
 ['mixed4d_1x1', 112],
 ['mixed4d_3x3_bottleneck_pre_relu/conv', 144],
 ['mixed4d_3x3_bottleneck_pre_relu', 144],
 ['mixed4d_3x3_bottleneck', 144],
 ['mixed4d_3x3_pre_relu/conv', 288],
 ['mixed4d_3x3_pre_relu', 288],
 ['mixed4d_3x3', 288],
 ['mixed4d_5x5_bottleneck_pre_relu/conv', 32],
 ['mixed4d_5x5_bottleneck_pre_relu', 32],
 ['mixed4d_5x5_bottleneck', 32],
 ['mixed4d_5x5_pre_relu/conv', 64],
 ['mixed4d_5x5_pre_relu', 64],
 ['mixed4d_5x5', 64],
 ['mixed4d_pool', 512],
 ['mixed4d_pool_reduce_pre_relu/conv', 64],
 ['mixed4d_pool_reduce_pre_relu', 64],
 ['mixed4d_pool_reduce', 64],
 ['mixed4d', 528],
 ['mixed4e_1x1_pre_relu/conv', 256],
 ['mixed4e_1x1_pre_relu', 256],
 ['mixed4e_1x1', 256],
 ['mixed4e_3x3_bottleneck_pre_relu/conv', 160],
 ['mixed4e_3x3_bottleneck_pre_relu', 160],
 ['mixed4e_3x3_bottleneck', 160],
 ['mixed4e_3x3_pre_relu/conv', 320],
 ['mixed4e_3x3_pre_relu', 320],
 ['mixed4e_3x3', 320],
 ['mixed4e_5x5_bottleneck_pre_relu/conv', 32],
 ['mixed4e_5x5_bottleneck_pre_relu', 32],
 ['mixed4e_5x5_bottleneck', 32],
 ['mixed4e_5x5_pre_relu/conv', 128],
 ['mixed4e_5x5_pre_relu', 128],
 ['mixed4e_5x5', 128],
 ['mixed4e_pool', 528],
 ['mixed4e_pool_reduce_pre_relu/conv', 128],
 ['mixed4e_pool_reduce_pre_relu', 128],
 ['mixed4e_pool_reduce', 128],
 ['mixed4e', 832],
 ['maxpool10', 832],
 ['mixed5a_1x1_pre_relu/conv', 256],
 ['mixed5a_1x1_pre_relu', 256],
 ['mixed5a_1x1', 256],
 ['mixed5a_3x3_bottleneck_pre_relu/conv', 160],
 ['mixed5a_3x3_bottleneck_pre_relu', 160],
 ['mixed5a_3x3_bottleneck', 160],
 ['mixed5a_3x3_pre_relu/conv', 320],
 ['mixed5a_3x3_pre_relu', 320],
 ['mixed5a_3x3', 320],
 ['mixed5a_5x5_bottleneck_pre_relu/conv', 48],
 ['mixed5a_5x5_bottleneck_pre_relu', 48],
 ['mixed5a_5x5_bottleneck', 48],
 ['mixed5a_5x5_pre_relu/conv', 128],
 ['mixed5a_5x5_pre_relu', 128],
 ['mixed5a_5x5', 128],
 ['mixed5a_pool', 832],
 ['mixed5a_pool_reduce_pre_relu/conv', 128],
 ['mixed5a_pool_reduce_pre_relu', 128],
 ['mixed5a_pool_reduce', 128],
 ['mixed5a', 832],
 ['mixed5b_1x1_pre_relu/conv', 384],
 ['mixed5b_1x1_pre_relu', 384],
 ['mixed5b_1x1', 384],
 ['mixed5b_3x3_bottleneck_pre_relu/conv', 192],
 ['mixed5b_3x3_bottleneck_pre_relu', 192],
 ['mixed5b_3x3_bottleneck', 192],
 ['mixed5b_3x3_pre_relu/conv', 384],
 ['mixed5b_3x3_pre_relu', 384],
 ['mixed5b_3x3', 384],
 ['mixed5b_5x5_bottleneck_pre_relu/conv', 48],
 ['mixed5b_5x5_bottleneck_pre_relu', 48],
 ['mixed5b_5x5_bottleneck', 48],
 ['mixed5b_5x5_pre_relu/conv', 128],
 ['mixed5b_5x5_pre_relu', 128],
 ['mixed5b_5x5', 128],
 ['mixed5b_pool', 832],
 ['mixed5b_pool_reduce_pre_relu/conv', 128],
 ['mixed5b_pool_reduce_pre_relu', 128],
 ['mixed5b_pool_reduce', 128],
 ['mixed5b', 1024],
 ['avgpool0', 1024],
 ['head0_pool', 508],
 ['head0_bottleneck_pre_relu/conv', 128],
 ['head0_bottleneck_pre_relu', 128],
 ['head0_bottleneck', 128],
 ['head1_pool', 528],
 ['head1_bottleneck_pre_relu/conv', 128],
 ['head1_bottleneck_pre_relu', 128],
 ['head1_bottleneck', 128]]
    return ll


def main():
    main_menu()

if __name__ == "__main__":
    main()