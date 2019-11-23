# Author: Humberto Rodriguez-Alvarez
# Date: 19.11.2019
#Github htorodriguez
#############################
'''
This Script takes as an input and image path and a model checkpoint, and retruns the forward pass through the model
'''
###########################
# Imports here
import torch
from torchvision import datasets, transforms, models
import functions 
import json
import argparse
import numpy as np
###########################
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path',  default="flowers/train/43/image_02347.jpg", type=str)
parser.add_argument('checkpoint',  default="", type=str)
parser.add_argument('--topk',  default=3, type=int)
parser.add_argument('--gpu',  action='store_true')
parser.add_argument('--category_names',  default="cat_to_name.json", type=str)
args=parser.parse_args()
#########################
#####associate input arguments to variables
image_path=args.image_path
checkpoint_import=args.checkpoint+"/"+ 'checkpoint.pth'
##########################
#category name mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
#Load Checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg11(pretrained=True)
    model.classifier= checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return (model, checkpoint['Mapping_class_idx_train'], checkpoint['epochs'])
#########################
def predict(image_path, model_path, topk=1, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep            learning model.
    '''   
    model, Mapping_class_idx, epochs_old = load_checkpoint(checkpoint_import)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu==True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    ####
    Mapping_idx_class = {v: k for k, v in Mapping_class_idx.items()}
    ###
    # Get the image form the file and transform it and covert it to a pytorch tensor of shape [1,3,224,224]
    np_image=torch.from_numpy(functions.process_image(image_path))
    np_image=np_image[:,:,:,None]
    np_image=np.transpose(np_image, [3,0,1,2])
    np_image=np_image.float()
    np.image=np_image.to(device)
        #turn off drop outs
    with torch.no_grad():
        model.eval()
        log_probs= model.forward(np_image.to(device))
        probs=torch.exp(log_probs)
        top_p, top_index = probs.topk(topk, dim=1)# gets the top probabilities and the indices of those probs
        top_index=top_index.to("cpu").numpy()
        top_class=[Mapping_idx_class.get(key) for key in top_index[0]]# gets the classes based on the indices
        
    return (top_p.to("cpu").numpy()[0], top_class)

# loading the checkpoint and passing your image 
print('Loading the checkpoint from {}'.format(checkpoint_import))
print("Requested image: {}". format(image_path))
probs, classes = predict(image_path,checkpoint_import, args.topk, args.gpu)

names=[cat_to_name[label] for label in classes]
#######
# Results
print('The names and probabilities of this image are') 
print(names)
print(probs)