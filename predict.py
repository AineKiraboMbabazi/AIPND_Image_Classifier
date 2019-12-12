#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/home/workspace/Image classifier/predict.py
#                                                                             
# PROGRAMMER: Ainekirabo Mbabazi
# DATE CREATED: 11/29/2019                                  
# REVISED DATE: 12/2/2019  - Remove variable assignments
#
# PURPOSE: Uses a trained network to predict the class for an input image.
# Use argparse Expected Call with <> indicating expected user input:
#       python predict.py image_path < path to image whose class is to be predicted> checkpoint <path to the checkpoint file> --top_k <number of top likely classes> --category_names <mapping of categories to real names> --gpu <flag to enable the  gpu>

#   Example call:
#    python predict.py 'flowers/test/28/image_05230.jpg' checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu


# Imports here
import torch
import argparse
import json
import helper_functions as func


def main():
    in_arg = func.get_prediction_input_args()
    func.check_command_line_args_prediction(in_arg)
 
    
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == True
                                      else "cpu")

    
    model = func.load_checkpoint(in_arg.checkpoint, device)
    
        
    probs, labels = func.predict(in_arg.image_path, model, device, in_arg.category_names, in_arg.top_k)
        
    

if __name__ == "__main__":
    main()
    
