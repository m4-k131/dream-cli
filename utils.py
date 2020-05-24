from PIL import Image

import urllib.request
import tarfile
import zipfile
import os
import numpy as np

save_folder="./renderedImages/"



def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/CIFAR-10/"
    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        print('Downloading Inception-model. Please wait a moment.')
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)

        
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")

    
def save_image(img, filename):
    a = img / 255.0
    a = np.uint8(np.clip(a, 0, 1) * 255)
    ##Create precise filename
    filename=filename+".jpg"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder,filename), 'wb') as file:
        Image.fromarray(a).save(file, 'jpeg')
        
def load_image(filename):
    image = Image.open(filename)
    return np.float32(image)


def gradient_grading(grad, image, method=1, fr=4, fg=4,fb=4 ):
    
    if method==1 or method==2:
        #Grayscale
        r, g, b = grad[:,:,0], grad[:,:,1], grad[:,:,2]
        gg = 0.2989 * r + 0.5870 * g + 0.1140 * b

        if method==1:
            graded=np.expand_dims(gg, 2)
            graded=grad*[fr, fg, fb]
        else:
            #Grayscale masked
            om_r, om_g, om_b =image[:,:,0], image[:,:,1], image[:,:,2]
            g_mask=0.2989 * om_r + 0.5870 * om_g + 0.1140 * om_b

            graded=np.zeros_like(grad)
            graded[:,:,0]=gg*fr*(om_g)
            graded[:,:,1]=gg*fg*(om_r)
            graded[:,:,2]=gg*g_mask*fb

    else:
        #Linear
        graded=grad*np.array(fr, fg, fb)
    return graded


def get_bounds(x_max, y_max, renderers):
    t_bounds=[]
    for r in renderers:
        if r['cropped']:
            boundraries=r['boundraries']
            x_lower=boundraries[0][0]*x_max
            x_upper=boundraries[0][1]*x_max
            y_lower=boundraries[1][0]*y_max
            y_upper=boundraries[1][1]*y_max
            t_bounds.append([int(x_lower), int(x_upper), int(y_lower), int(y_upper)])
        else:
            t_bounds.append([0,x_max,0,y_max])
            
    return t_bounds
    
def get_bounds(x_max, y_max, renderers):
    t_bounds=[]
    for r in renderers:
        if r['cropped']:
            b=r['boundraries']
            x_lower=b[0][0]*x_max
            x_upper=b[0][1]*x_max
            y_lower=b[1][0]*y_max
            y_upper=b[1][1]*y_max
            t_bounds.append([int(x_lower), int(x_upper), int(y_lower), int(y_upper)])
        else:
            t_bounds.append([0,x_max,0,y_max])
            
    return t_bounds