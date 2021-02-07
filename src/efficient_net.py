
import os
import cv2
import pydicom
import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Nadam
from PIL import Image
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
import efficientnet.tfkeras as efn



def tab_data(Age, Gender, Smoking_Status):
         
    agss_table = [(Age - 30) / 30] 

    if Gender == 'Male':
       agss_table.append(0)
    else:
       agss_table.append(1)
    
    if Smoking_Status == 'Never smoked':
        agss_table.append(0)
    elif Smoking_Status == 'Ex-smoker':
        agss_table.append(1)
    elif Smoking_Status == 'Currently smokes':
        agss_table.append(2)
    else:
        agss_table.append(3)

    return np.array(agss_table)


def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize((d.pixel_array - d.RescaleIntercept) / 2**11, (128, 128))

def fetch_images(img_path):
    images = read_images_in_middle_of_scan(img_path)
    return images

def read_images_in_middle_of_scan(img_path, lower=0.15, upper=0.8):
    images = []
    for filename in os.listdir(img_path):
        if int(filename[:-4]) / len(os.listdir(img_path)) < 0.8 and int(filename[:-4]) / len(os.listdir(img_path)):
            images.append(get_img(os.path.join(img_path, f'{filename}'))) 
    return images

def create_agss_vec_mat(Age, Gender, Smoking_Status, num_rows):
    agss_vector = tab_data(Age, Gender, Smoking_Status)
    agss_matrix = np.array([agss_vector] * num_rows)
    return agss_vector, agss_matrix

def pred_fvc(x, m, c):
    return m * x + c

def pred_confidence(base_percent, m, gap_in_weeks):
    return base_percent - m * abs(gap_in_weeks)

def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
    return np.mean(metric)

def make_model_pred(Age, Gender, Smoking_Status, img_path, model):
    images = fetch_images(img_path)
    images = np.expand_dims(images, axis=-1)
    agss_vector, agss_matrix = create_agss_vec_mat(Age, Gender, Smoking_Status, num_rows=images.shape[0])
    output = model.predict([images, agss_matrix])
    output = sum(output)/len(images)
    return output


# # Prediction Function

def start_predict (patient_id, ipt_FVC, ipt_Week, Percent, Age, Gender, Smoking_Status, img_path, model):
    patient_wk_np = np.empty(146, dtype = 'object')
    fvc_np = np.empty(146)
    conf_np = np.empty(146)   
     
    # pred at week 0. Other weeks will be predicted using this as base
    gradient = make_model_pred(Age, Gender, Smoking_Status, img_path, model) 
    fvc_base = (ipt_FVC - (gradient * ipt_Week))  
    base_week = ipt_Week
   

    # this loop predicts values (FVC and confidence) for each week
    for i, k in enumerate(range(-12, 134)):
        week_no = k
        gap_from_base_week = base_week - week_no
        
        predicted_fvc = pred_fvc(week_no, m=gradient, c=fvc_base)
        predicted_conf = pred_confidence(Percent, m=gradient, gap_in_weeks=gap_from_base_week)
        
        patient_wk_np[i] = patient_id + '_' + str(k)
        fvc_np[i] = predicted_fvc
        conf_np[i] = predicted_conf
    
    return patient_wk_np, fvc_np, conf_np
