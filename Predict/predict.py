#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:39:01 2020

@author: ubuntu
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime
from keras.applications.mobilenet import MobileNet
import utils

FC_LAYERS = [1024, 1024]

# Create directories if needed
if not os.path.isdir("%s"%("Predictions")):
    os.makedirs("%s"%("Predictions"))

# Read in your image
image = cv2.imread("/home/ubuntu/Documents/Transfer-Learning-Suite-master/data/val/n1/n102.jpg",-1)
save_image = image
image = np.float32(cv2.resize(image, (224, 224)))
image = image.reshape(1, 224, 224, 3)

class_list_file = "./checkpoints/MobileNet_Pets_class_list.txt"

class_list = utils.load_class_list(class_list_file)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
finetune_model = utils.build_finetune_model(base_model, dropout=1e-3,fc_layers=FC_LAYERS,num_classes=len(class_list))
finetune_model.load_weights("./checkpoints/MobileNet_model_weights.h5" )

# Run the classifier and print results
st = time.time()

out = finetune_model.predict(image)

confidence = out[0]
class_prediction = list(out[0]).index(max(out[0]))
class_name = class_list[class_prediction]

run_time = time.time()-st

print("Predicted class = ", class_name)
print("Confidence = ", confidence)
print("Run time = ", run_time)
cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)