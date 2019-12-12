# Imports here

import matplotlib.pyplot as plt
import numpy as np
import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import argparse

def get_training_input_args():
    
    model_list = ['vgg16','vgg19','vgg13','densenet','alexnet', 'densenet201', 'resnet18']
    parser = argparse.ArgumentParser(description='Train and save a model checkpoint')
    parser.add_argument('data_dir', type = str, help = 'path to dataset folder')
    parser.add_argument('--save_dir', type = str, help = 'Directory where the checkpoint will be saved', default= 'checkpoint.pth')
    parser.add_argument('--arch', type = str, help = 'Model Architecture eg vgg,densenet, alexnet, resnet18', default= 'vgg13', choices = model_list)
    parser.add_argument('--learning_rate', type = float, help = 'Learning Rate', default= 0.01)
    parser.add_argument('--hidden_units', type = int, help = 'List of number of nodes in hidden layers', nargs='+', default= [1000])
    parser.add_argument('--epochs', type = int, help = 'Number of epochs', default = 20)
    parser.add_argument('--gpu', help = 'Switch to gpu ', action= 'store_true')
    parser.add_argument('--dropout', type = float, help = 'Dropout for training', default = 0.5)

    in_arg = parser.parse_args()
    return in_arg

def get_prediction_input_args():
    
    parser = argparse.ArgumentParser(description='Predict the category of a flower')
    parser.add_argument('image_path', type = str, help = 'path to flower whose class is to be predicted')
    parser.add_argument('checkpoint', type = str, help = 'Directory where the checkpoint was saved' )
    parser.add_argument('--top_k', type = int, help = 'number of most likely classes', default= 3)
    parser.add_argument('--category_names', help = 'Enter JSON file', default= 'cat_to_name.json')
    parser.add_argument('--gpu', help = 'Switch to gpu ', action= 'store_true')

    in_arg = parser.parse_args()
    return in_arg


def check_command_line_args_prediction(in_arg):
    
    print("______________________________________")
    print("____ Arguments used for prediction ___")
    print("______________________________________")
    print("\n       image_path =", in_arg.image_path, 
          "\n       checkpoint =", in_arg.checkpoint, "\n       top_k =", in_arg.top_k, "\n       category_names =", in_arg.category_names, 
          "\n       gpu =", in_arg.gpu)

def check_command_line_args(in_arg):
    print("______________________________________")
    print("____  Arguments used for training  ___")
    print("______________________________________")
    print("\n       data_dir =", in_arg.data_dir, 
          "\n       arch =", in_arg.arch, "\n       save_dir =", in_arg.save_dir, "\n       dropout =", in_arg.dropout, 
          "\n       learning_rate =", in_arg.learning_rate, "\n       hidden_units =", in_arg.hidden_units, "\n       epochs =", in_arg.epochs, 
          "\n       gpu =", in_arg.gpu)


# Loading the pretrained Network

in_features = 0
def load_model(model_name):
    if 'vgg' in model_name:
        model = models.__dict__[model_name](pretrained=True)
        in_features = model.classifier[0].in_features
        
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
        in_features = 9216
        
        
    elif 'densenet' in model_name:
        model = models.__dict__[model_name](pretrained=True)
        
        in_features = model.classifier.in_features
        
    return model, in_features

# Building the Network

class FlowerNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()      
        self.hidden_layers=nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])          
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)  
        self.dropout = nn.Dropout(p=drop_p)
      
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)   
        return F.log_softmax(x, dim=1)


def validation(model, device, dataloaders_valid, criterion):
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    
    for images, labels in dataloaders_valid:

        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()   
        #   Calculating the accuracy
        # take exponents to get the probabilities
        ps = torch.exp(output)
        equality = (labels.data == output.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
        
    return test_loss, accuracy

def trainer(device, model, dataloaders, print_every,criterion,optimizer,epochs, dataloaders_valid):
    
    steps = 0
    running_loss = 0
    model.to(device)
    
    for e in range(epochs):
        
        model.train()


        for images, labels in dataloaders:
            steps += 1
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward step happens here
            output = model.forward(images)
            loss = criterion(output, labels)

            #backpropagation step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, device, dataloaders_valid, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders_valid)),
                      " Accuracy: {:.3f}".format(accuracy/len(dataloaders_valid)))

                running_loss = 0
                model.train()
    print("__Training successfully completed__")

# TODO: Do validation on the test set

def test_network(model, device,dataloaders_test):
    model.to(device)
    model.eval()
    accurately_classified_count = 0
    total = 0
    
    for images, labels in dataloaders_test:
        
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        
        _, prediction = torch.max(output.data,1)
        total += labels.size(0)
        accurately_classified_count += torch.sum(prediction == labels.data).item()
    testing_accuracy = 100 * (accurately_classified_count/total)
    
    return testing_accuracy



# TODO: Save the checkpoint

def save_checkpoint(model,model_name, filename, image_datasets, optimizer):
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'model_name': model_name,
                  'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'class_to_idx': model.class_to_idx
                }

    torch.save(checkpoint, filename )
    print("Checkpoint successfully saved to: ", filename)


def load_checkpoint(filename,device):
    checkpoint = torch.load(filename)
    
    model_name = checkpoint['model_name']
    
    model = models.__getattribute__(model_name)(pretrained = True) 
    model.to(device)
        
    for param in model.parameters():
        param.requires_grad = False 
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    
    model.eval()
    return (model)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
   
    if img.width > img.height:
        ratio = img.width/img.height
        img = img.resize(size=( int(round(ratio*256,0)),256))    
        
    else:
        ratio = img.height/img.width
        img = img.resize(size=(256, int(round(ratio*256,0))))  
    
    bottom = (img.height + 224)/2
    right = (img.width + 224)/2
    top = (img.height - 224)/2
    left = (img.width - 224)/2
    
    img = img.crop(box=(left,top,right,bottom))
    np_image = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = (np_image - mean)/std 
    img = img.transpose((2, 0, 1))
    return img



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
    

def predict(image_path, model, device, category_names, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0).float()
    
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
    
    probs, indices = torch.topk(ps, topk)
    Probs = np.array(probs.data[0])
    Indices = np.array(indices.data[0])
    
    # invert class_to_idx

    idx_to_class = {idx:Class for Class,idx in model.class_to_idx.items()}
    
    classes = [idx_to_class[i] for i in Indices]
    
    labels = [cat_to_name[Class] for Class in classes]
    
    print("\n ___Most likely flower class with associated Probability___ ")
    print(labels[0], " : ", Probs[0])
    
    print(":\n ___Top K classes along with associated probabilities ___: \n")
    
    for i, val in enumerate(Probs):
        print( labels[i], " : ", Probs[i])
    
    return Probs,labels

def duration(start_time, end_time):
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time/3600))) + ":" + str(int((tot_time%3600)/60)) + ":" + str(int((tot_time%3600)%60)))
