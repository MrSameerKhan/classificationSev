#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:20:10 2020

@author: ubuntu
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

# Files
import utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Pets", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=0.0, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="MobileNet", help='Your pre-trained classification model of choice')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]


preprocessing_function = None
base_model = None

from keras.applications.mobilenet import preprocess_input
preprocessing_function = preprocess_input
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))


# Read in your image
test_img_path = "./test_images/"
for img in os.listdir(test_img_path):
    print(img)
    image = cv2.imread(test_img_path+img,-1)
    save_image = image
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))
    
    class_list_file = "./checkpoints/MobileNet_Pets_class_list.txt"
    
    class_list = utils.load_class_list(class_list_file)
    print(class_list)
    
    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout,fc_layers=FC_LAYERS,num_classes=len(class_list))
    finetune_model.load_weights("./checkpoints/MobileNet_model_weights.h5")
    
    # Run the classifier and print results
    st = time.time()
    
    out = finetune_model.predict(image)
    
    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]
    
    run_time = time.time()-st
    path='result/'+str(class_name)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(str(path)+str(img)+'.jpg', save_image)
    print("Predicted class = ", class_name)
    print("Confidence = ", confidence)
    print("Run time = ", run_time)
    # cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)