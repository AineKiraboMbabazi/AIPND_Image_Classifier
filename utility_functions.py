import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models

def load_and_preprocess_data(data_dir):

    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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

    return(dataloaders, dataloaders_valid, dataloaders_test, image_datasets)
