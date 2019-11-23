# Author: Humberto Rodriguez-Alvarez
# Date: 19.11.2019
#Github htorodriguez
#############################
'''
This file contains the main function for training the model 
'''
# Imports here
import torch
from torchvision import datasets, transforms, models
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os
###########################
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('train_folder',  default="flowers", type=str)
parser.add_argument('--save_dir',  default="", type=str)
parser.add_argument('--learning_rate',  default=0.01, type=float)
parser.add_argument('--epochs',  default=10, type=int)
parser.add_argument('--hidden_units',  default=12544, type=int)
parser.add_argument('--arch',  default="vgg11", type=str)
parser.add_argument('--gpu',  action='store_true')
args=parser.parse_args()
#########################
#####associate input arguments to variables
data_dir =args.train_folder
save_directory=args.save_dir
learning_rate=args.learning_rate
architecture=args.arch
use_gpu=args.gpu
epochs_input=args.epochs
hidden_units_input=args.hidden_units
#########################
### Data Dirs
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
##########################
#Define the tr, ansforms
train_transformation=transforms.Compose([transforms.RandomRotation(45),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
                                        ])

test_valid_transformation=transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  
                                        ])
#Load the datasets with ImageFolder 
train_data=datasets.ImageFolder(root=train_dir,transform=train_transformation)
valid_data=datasets.ImageFolder(root=valid_dir, transform=test_valid_transformation)
test_data=datasets.ImageFolder(root=test_dir, transform=test_valid_transformation)
#dataloaders 
train_generator= torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
valid_generator=torch.utils.data.DataLoader(valid_data, batch_size=20)
test_generator=torch.utils.data.DataLoader(test_data, batch_size=20)

###############################
#Load Checkpoint
# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = models.vgg11(pretrained=True)
#     model.classifier= checkpoint['classifier']
#     model.load_state_dict(checkpoint['state_dict'])
#     return (model, checkpoint['Mapping_class_idx_train'], checkpoint['epochs'])

# model, Mapping_class_idx, epochs_old = load_checkpoint(checkpoint_import)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
###############################
##model
model_str="models."+architecture+"(pretrained=True)"
#model=exec(model_str)
model=models.vgg11(pretrained=True)
#freeze the feature parameters
for param in model.parameters():
    param.requires_grad=False
#define my classifier
my_classifier= torch.nn.Sequential(torch.nn.Linear(25088, hidden_units_input),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(0.2),
                                   torch.nn.Linear(hidden_units_input, 102),
                                   torch.nn.LogSoftmax(dim=1)
                                    )

# modify the ccurrent model and assign my_calssifier to th classifer, define loss criterion and optimizer, send to device
model.classifier= my_classifier
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_gpu==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
model.to(device)
##############################
### training and validating
# Define the vectors for training and testing control
print('Lets train!... this might take a few minutes...')
loss_vector_train=[]
loss_vector_valid=[]
#The accuracy is only calculated on the validation set
accuracy_vector=[]
# Set the Epochs to train
epochs =epochs_input
##start main loop over epochs
for epoch in range(epochs):
    # Zero the training loss
    train_current_loss=0
    #start loop over training images
    for images, labels in train_generator:
        images, labels = images.to(device), labels.to(device)
        # zero the gradient matrices 
        optimizer.zero_grad()
        #forward pass
        log_probs= model.forward(images)
        # define loss based on the NLLL criterion defined above
        loss = criterion(log_probs, labels)
        # backward pass and optimze the weights
        loss.backward()
        optimizer.step()
        # sum up losses 
        train_current_loss+=loss
    #once looped over the training images we extract loss and accuracy on the validation set
    else:
        #fill the loss vector
        number_of_images_train=len(train_generator)
        loss_vector_train.append(train_current_loss/number_of_images_train) 
        #zero the validation loss and accuracy
        accuracy_valid=0
        loss_valid=0
        #work without calculating gradients
        with torch.no_grad():
            #turn off dropouts
            model.eval()
            for images, labels in valid_generator:
                images, labels = images.to(device), labels.to(device)
                log_probs_valid= model.forward(images)
                loss_valid+=criterion (log_probs_valid, labels)
                # get top class
                probs=torch.exp(log_probs_valid)
                top_p, top_class = probs.topk(1, dim=1)
                hits=top_class==labels.view(*top_class.shape)
                accuracy_valid+=torch.mean(hits.type(torch.FloatTensor)) 
            #Once we have looped over the validation set we can calculate loss and accuracy vectors for this epoch
            number_of_images_valid= len(valid_generator)
            loss_vector_valid.append(loss_valid/number_of_images_valid)
            accuracy_vector.append(accuracy_valid/number_of_images_valid)
            model.train()
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_current_loss/number_of_images_train),
                  "Validation Loss: {:.3f}.. ".format(loss_valid/number_of_images_valid),
                  "validation Accuracy: {:.3f}".format(accuracy_valid/number_of_images_valid) 
                  )
print("Training finished!")
print("Saving Model to Checkpoint")

#Save the checkpoint 
checkpoint = {'input_size': 224*224,
              'output_size': 102,
              'model template': architecture,
              'criterion':'NLLLoss',
              'classifier':model.classifier,
              #'optimizer_state':optimizer.state_dict,
              'state_dict': model.state_dict(), 
              'epochs':epochs, 
              'Mapping_class_idx_train':train_data.class_to_idx}

#Create new dir
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
torch.save(checkpoint, save_directory+'/'+'checkpoint.pth')
print("model trained and saved, epochs: {}".format(epochs))