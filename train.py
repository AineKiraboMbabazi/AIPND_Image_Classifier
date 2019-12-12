#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/Image classifier/train.py
#                                                                             
# PROGRAMMER: Ainekirabo Mbabazi
# DATE CREATED: 11/28/2019                                  
# REVISED DATE: 12/2/2019  - Remove variable assignments
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py data_directory <directory with images> --save_dir <save directory> --arch <model> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <epochs> --gpu <gpu>
#
#   Example calls:
#       python train.py flowers  --save_dir checkpoint.pth --arch vgg13 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu
#       python train.py flowers  --save_dir checkpoint.pth --arch 'densenet201' --learning_rate 0.001 --hidden_units 1000 --epochs 5 --gpu
#       python train.py flowers  --save_dir checkpoint.pth --arch 'alexnet' --learning_rate 0.001 --hidden_units 1000 --epochs 5 --gpu

import numpy as np
from time import time
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
import helper_functions as func
from utility_functions import load_and_preprocess_data

def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = func.get_training_input_args()
    
    # Function that checks command line arguments
    func.check_command_line_args(in_arg)
        
    # Function to load and transform data
    result = load_and_preprocess_data(in_arg.data_dir)
    
    dataloaders, dataloaders_valid, dataloaders_test, image_datasets = result
    
    # Function to load the model
    model, input_size = func.load_model(in_arg.arch)

    for param in model.parameters():
        param.requires_grad = False 

    # Function to build the network
    flower_classifer = func.FlowerNetwork(input_size, 102, in_arg.hidden_units, in_arg.dropout)
    
    model.classifier = flower_classifer
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == True
                                      else "cpu")

    print_every = 40
    
    # Function to train the network

    func.trainer(device, model, dataloaders, print_every, criterion, optimizer, in_arg.epochs, dataloaders_valid)
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    
    func.duration(start_time, end_time)
    
    # Function to save the checkpoint
    func.save_checkpoint(model, in_arg.arch, in_arg.save_dir, image_datasets, optimizer)
    

if __name__ == "__main__":
    main()
