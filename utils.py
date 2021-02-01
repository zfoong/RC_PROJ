# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:37:16 2021

@author: zfoong
"""
import cv2
import math   
import matplotlib.pyplot as plt    
from keras.preprocessing import image  
import pandas as pd
import numpy as np  
from tqdm import tqdm 
import os
import csv

def moving_average(a, n=13) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def data_label_mapping(VIDEO_DIRECTORY):
    with open('hnmdb_label.csv', mode='w', newline="\n") as csvfile:
        writer = csv.writer(csvfile)
        for x in os.scandir(VIDEO_DIRECTORY):
            for y in os.scandir(x.path):
                writer.writerow([y.path, x.name])


# storing the frames from training videos
def extract_frame(data):
    for i in tqdm(range(len(data))):
        count = 0
        videoFile = data[i][0]
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        frameRate = 5 #frame rate
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret == False):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                filename ='extracted/' + videoFile.split('\\')[-1]+"_frame%d.jpg" % count;count+=1
                dim = (180, 120)
                resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(filename, resized)
        cap.release()
        

def video_to_frames(video_file, frame_rate):
    imgs = []
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
    return imgs
            

def display_plt(result_list, name):
    avg_result = moving_average(result_list)
    plt.plot(avg_result)
    plt.ylabel("")
    plt.xlabel("")
    # plt.ylim(0,200)
    plt.savefig(name + '.png')
    plt.show()