
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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


```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                       ])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                       ])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)

image_datasets_valid = datasets.ImageFolder(valid_dir, transform = data_transforms)

image_datasets_test = datasets.ImageFolder(test_dir, transform = data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size = 64, shuffle = True)

dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid,batch_size = 32)

dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size = 32)
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.


```python

in_features = 0
def load_model(model_name):
    if 'vgg' in model_name:
        model = models.__dict__[model_name](pretrained=True)
        in_features = model.classifier[0].in_features
        
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_features = 9216
        model = models.__dict__['densenet121'](pretrained=True)
        
        in_features = model.classifier.in_features
        print(in_features)
        
    elif 'densenet' in model_name:
        model = models.__dict__[model_name](pretrained=True)
        
        in_features = model.classifier.in_features
        print(in_features)
        
    return model, in_features

model, in_features = load_model('densenet201')
```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
    Downloading: "https://download.pytorch.org/models/densenet201-c1103571.pth" to /root/.torch/models/densenet201-c1103571.pth
    100%|██████████| 81131730/81131730 [00:00<00:00, 89380805.68it/s]


    1920



```python
# TODO: Build and train your network

for param in model.parameters():
    param.requires_grad = False 

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


flower_classifer = FlowerNetwork(in_features, 102, [1000], drop_p=0.5)
model.classifier = flower_classifer


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)


def validation(model, dataloaders_valid, criterion):
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

def trainer(model, dataloaders, print_every,criterion,optimizer,epochs):
    
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
                    test_loss, accuracy = validation(model, dataloaders_valid, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders_valid)),
                      " Accuracy: {:.3f}".format(accuracy/len(dataloaders_valid)))


                running_loss = 0
                model.train()

trainer(model, dataloaders, 40,criterion,optimizer,5)
```

    Epoch: 1/5..  Training Loss: 4.127..  Validation Loss: 3.182..   Accuracy: 0.347
    Epoch: 1/5..  Training Loss: 2.795..  Validation Loss: 1.782..   Accuracy: 0.617
    Epoch: 2/5..  Training Loss: 1.830..  Validation Loss: 1.056..   Accuracy: 0.771
    Epoch: 2/5..  Training Loss: 1.364..  Validation Loss: 0.757..   Accuracy: 0.854
    Epoch: 2/5..  Training Loss: 1.194..  Validation Loss: 0.572..   Accuracy: 0.879
    Epoch: 3/5..  Training Loss: 1.050..  Validation Loss: 0.512..   Accuracy: 0.887
    Epoch: 3/5..  Training Loss: 0.917..  Validation Loss: 0.444..   Accuracy: 0.894
    Epoch: 4/5..  Training Loss: 0.820..  Validation Loss: 0.380..   Accuracy: 0.922
    Epoch: 4/5..  Training Loss: 0.763..  Validation Loss: 0.343..   Accuracy: 0.921
    Epoch: 4/5..  Training Loss: 0.687..  Validation Loss: 0.317..   Accuracy: 0.927
    Epoch: 5/5..  Training Loss: 0.724..  Validation Loss: 0.309..   Accuracy: 0.932
    Epoch: 5/5..  Training Loss: 0.667..  Validation Loss: 0.271..   Accuracy: 0.938


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# TODO: Do validation on the test set
def test_network(model, dataloaders_test):
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
print('The accuracy of the model on testing data is: %d ' %test_network(model,dataloaders_test))
```

    The accuracy of the model on testing data is: 94 


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# TODO: Save the checkpoint

def save_checkpoint(model, filename, image_datasets, optimizer):
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'model_name': 'densenet201',
                  'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'class_to_idx': model.class_to_idx
                }

    torch.save(checkpoint, filename )
save_checkpoint(model,'checkpoint.pth',image_datasets, optimizer)
```

    /opt/conda/lib/python3.6/site-packages/torch/serialization.py:193: UserWarning: Couldn't retrieve source code for container of type FlowerNetwork. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

def load_checkpoint(filename, device):
    
    checkpoint = torch.load(filename)
    model = models.densenet201(pretrained = True) 
    model.to(device)
        
    for param in model.parameters():
        param.requires_grad = False 
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    
    model.eval()
    return (model)

model = load_checkpoint('checkpoint.pth',device)

```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.


# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
image_path = 'flowers/test/28/image_05230.jpg'

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


```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python

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
    
```

## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

model = load_checkpoint('checkpoint.pth',device)


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
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

    return Probs,labels

```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# TODO: Display an image along with the top 5 classes
def sanity_check(image_path, dataloaders_test, model, device, cat_to_name):
    
    name = image_path.split('/')[-2]
    imag = Image.open(image_path)
    
    
    fig,(ax1, ax2) = plt.subplots(figsize =(6,10), ncols=1, nrows=2)
    flower_name = cat_to_name[name]
    
    ax1.set_title(flower_name)
    ax1.imshow(imag)
    ax1.axis('off')
    
    probs, labs= predict(image_path, model,device)
    print(probs, labs)
    bin_edges = np.arange(len(labs))
    ax2.barh(bin_edges, probs, align='center', color='blue')
    ax2.set_yticks(bin_edges)
    ax2.set_yticklabels(labs)
    ax2.invert_yaxis()
    
    
    
image_path = 'flowers/test/28/image_05230.jpg'
sanity_check(image_path, dataloaders_test, model,device,cat_to_name)
```

    [ 0.9567042   0.01341128  0.01311425  0.0079197   0.00379168] ['stemless gentian', 'canterbury bells', 'monkshood', 'morning glory', 'bolero deep blue']



![png](readme_files/readme_23_1.png)



```python

```
