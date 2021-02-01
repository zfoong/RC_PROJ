# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:27:55 2021

@author: zfoong
"""

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np    # for mathematical operations
from keras.preprocessing import image   # for preprocessing the images
from keras.utils import np_utils
from keras.utils import to_categorical
from glob import glob
from tqdm import tqdm
import os
from model import *
import csv
import random

# Seed
np.random.seed(100)
random.seed(100)

# Param Init
input_size = 5400
N = 3000

# Build video label mapping
VIDEO_DIRECTORY = "C:\\Users\zfoong\Desktop\RC_proj\dataset\hmdb51_org"

label = []
for x in os.scandir(VIDEO_DIRECTORY):
    label.append(x.name)
    
integer_mapping = {x: i for i,x in enumerate(label)}
vec = [integer_mapping[word] for word in label]
encoded = to_categorical(vec)
encoded_zip = dict(zip(integer_mapping, encoded))

data = []

with open('hnmdb_label.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data.append(row)

# Shuffle data
random.shuffle(data)

# Split Data
middle_index = math.floor(len(data)*7/10)
train = data[:middle_index]
test = data[middle_index:]

# Define model
model = ESN(input_size,len(label),N)

frame_rate = 5

# Training
for i in tqdm(range(len(train))):
    model.reset_state()
    imgs = []
    video_file = train[i][0]
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        frameId = cap.get(1) 
        ret, frame = cap.read()
        if (ret == False):
            break
        if (frameId % math.floor(frame_rate) == 0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            dim = (90, 60)
            resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
            img = image.img_to_array(resized)
            img = img.flatten()
            img = img/255
            imgs.append(img)
    z = model.compute_z_all(imgs)
    model.train(z, encoded_zip[train[i][1]].reshape(1, -1))
    
# Validation


# Print selected neuron distribution

# Plot loss



# model.reset_state()
# imgs = []
# video_file = train[0][0]
# cap = cv2.VideoCapture(video_file)   # capturing the video from the given path
# while(cap.isOpened()):
#     frameId = cap.get(1) #current frame number
#     ret, frame = cap.read()
#     if (ret == False):
#         break
#     if (frameId % math.floor(frameRate) == 0):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#         dim = (90, 60)
#         resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
#         cv2.imshow('img', resized)
#         cv2.waitKey(0)  
#         img = image.img_to_array(resized)
#         img = img.flatten()
#         img = img/255
#         imgs.append(img)
# z = model.compute_z_all(imgs)
# model.train(z, encoded_zip[train[0][1]].reshape(1, -1))

# # Save result in numpy file
# np.save(TEST_NAME + '.npy', score_list)