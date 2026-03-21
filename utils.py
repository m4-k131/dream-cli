from PIL import Image

import urllib.request
import tarfile
import zipfile
import os
import numpy as np

save_folder="./renderedImages/"


def maybe_download_and_extract(url:str, download_dir:str) -> None:
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
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        print('Downloading Inception-model. Please wait a moment.')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)      
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")

    
def save_image(img:np.ndarray, filename:str) -> None:
    a = img / 255.0
    a = np.uint8(np.clip(a, 0, 1) * 255)
    filename=f"{filename}+.jpg"
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder,filename), 'wb') as file:
        Image.fromarray(a).save(file, 'jpeg')


def load_image(filename:str) -> np.ndarray:
    image = Image.open(filename)
    return np.float32(image)


def gradient_grading(grad:np.ndarray, image:np.ndarray, method :int =1, fr:float = 4, fg:float = 4, fb:float = 4) -> np.ndarray:
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
        graded=grad*np.array(fr, fg, fb)
    return graded


def get_bounds(x_max:float, y_max:float, renderers:list) -> list:
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
    