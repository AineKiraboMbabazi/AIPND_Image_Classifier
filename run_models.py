#!/bin/sh
# */home/workspace/Image classifier/run_models.sh
#                                                                             
# PROGRAMMER: Ainekirabo Mbabazi
# DATE CREATED: 06/12/2019                          
# PURPOSE: Runs all three models
#         
#
# Usage: sh run_models.sh  -- will run program from commandline
#  
python train.py flowers  --save_dir checkpoint.pth --arch vgg13 --learning_rate 0.001 --hidden_units 1000 --epochs 5 --gpu 
python predict.py 'flowers/test/28/image_05230.jpg' checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu 

python train.py flowers  --save_dir checkpoint_1.pth --arch 'densenet201' --learning_rate 0.001 --hidden_units 1000 --epochs 5 --gpu 
python predict.py 'flowers/test/28/image_05230.jpg' checkpoint_1.pth --top_k 5 --category_names cat_to_name.json --gpu 

python train.py flowers  --save_dir checkpoint_.pth --arch 'alexnet' --learning_rate 0.001 --hidden_units 1000 --epochs 5 --gpu 
python predict.py 'flowers/test/28/image_05230.jpg' checkpoint_.pth --top_k 5 --category_names cat_to_name.json --gpu
