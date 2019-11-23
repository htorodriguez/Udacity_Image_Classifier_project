# Author: Humberto Rodriguez-Alvarez
# Date: 19.11.2019
#Github htorodriguez
#############################
'''
This file contains heper functions
'''
# TODO: Process a PIL image for use in a PyTorch model
from PIL import Image
import numpy as np

## support functions
def standardize(np_2D_array, mean, std ): 
    m = np.mean(np_2D_array)
    s = np.std(np_2D_array)
    norm=(np_2D_array - m) / s   
    standard=norm*std + mean
    #print(standard.mean())
    #print(standard.std())
    return standard

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #load
    im=Image.open(image_path)
    #im=image
    # scale to 256 on the smallest dimension
    size_x, size_y= im.size
    if (min(im.size)==size_x):
        im=im.resize((256, int(size_y/(size_x/256))))
    
    elif (min(im.size)==size_y):
        im=im.resize((int(size_x/(size_y/256)), 256))
        # crop to (224, 224)    
    size_x, size_y = im.size
    left = (size_x - 224)/2
    top = (size_y - 224)/2
    right = (size_x + 224)/2
    bottom = (size_y + 224)/2
    im = im.crop((left, top, right, bottom))
    # standardize
    np_image = np.array(im).astype('float')
    np_image=np_image/255# bring from 0-255 to 0-1
    i=0
    for mean, std in zip([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]):
        np_image[:,:,i] = standardize(np_image[:,:,i], mean, std)
        i+=1
    np_image=np.transpose(np_image, [2,0,1])
    return(np_image)
    