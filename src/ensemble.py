import os
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import copy
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from keras import Model
import _pickle as cPickle
from tensorflow.keras.models import load_model
from configparser import ConfigParser

from src.linear_model import NoTransformer, load_huber_models, huber_predict
from src.efficient_net import start_predict
from src.convmdlnosave import conv3dModel


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# This is the working directory.
working_dir = os.getcwd()

# read config
parser = ConfigParser()
parser.read("config.ini")

neuralnet_dir = parser.get("config", "neuralnet_dir")
linear_dir = parser.get("config", "linear_dir")
efficientnet_dir = parser.get("config", "efficientnet_dir")
conv3d_dir = parser.get("config", "conv3d_dir")
onehotencoder = parser.get("config", "onehotencoder")


print("========================= LOADING MODELS =========================")

# load models
nn_model = load_model(neuralnet_dir + 'neural_net.h5', compile = False)
lower_huber, mid_huber, upper_huber, datawrangler = load_huber_models(linear_dir + 'lower_huber.pkl',
                                                                    linear_dir +'mid_huber.pkl',
                                                                    linear_dir +'upper_huber.pkl',
                                                                    linear_dir +'datawrangler.pkl')
efficient_net_model = tf.keras.models.load_model(efficientnet_dir + 'fold-2_best.h5')

print("========================= MODELS LOADED =========================")

# Data preprocessing
feature_num = 10
with open(onehotencoder, 'rb') as f:
    onehot = cPickle.load(f)


def one_hot_encode(df):
    trf = onehot.transform(df[['Sex','SmokingStatus']])

    cat_list = []

    for item in onehot.categories_:
        temp = item.tolist()
        cat_list.extend(temp)

    trf_df = pd.DataFrame(trf, columns = cat_list)
    return pd.concat([df, trf_df], axis = 1).drop(['Sex','SmokingStatus'], axis = 1)


def group_data(df):
    temp = pd.merge(df.groupby('Patient')['Weeks'].apply(list).to_frame(),
                      df.groupby('Patient')['FVC'].apply(list).to_frame(),
                      on = 'Patient')

    temp = pd.merge(temp, df.groupby('Patient')['Percent'].apply(list).to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Age'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Male'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Female'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Currently smokes'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Ex-smoker'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Never smoked'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Height'].first().to_frame(), on = 'Patient')

    return temp.reset_index()

def group_first_data(df, one_hotted = False):
    temp = pd.merge(df.groupby('Patient')['Weeks'].first().to_frame(),
                      df.groupby('Patient')['FVC'].first().to_frame(),
                      on = 'Patient')

    temp = pd.merge(temp, df.groupby('Patient')['Percent'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Age'].first().to_frame(), on = 'Patient')
    if one_hotted:
        temp = pd.merge(temp, df.groupby('Patient')['Male'].first().to_frame(), on = 'Patient')
        temp = pd.merge(temp, df.groupby('Patient')['Female'].first().to_frame(), on = 'Patient')
        temp = pd.merge(temp, df.groupby('Patient')['Currently smokes'].first().to_frame(), on = 'Patient')
        temp = pd.merge(temp, df.groupby('Patient')['Ex-smoker'].first().to_frame(), on = 'Patient')
        temp = pd.merge(temp, df.groupby('Patient')['Never smoked'].first().to_frame(), on = 'Patient')
    else:
        temp = pd.merge(temp, df.groupby('Patient')['Sex'].first().to_frame(), on = 'Patient')
        temp = pd.merge(temp, df.groupby('Patient')['SmokingStatus'].first().to_frame(), on = 'Patient')
    temp = pd.merge(temp, df.groupby('Patient')['Height'].first().to_frame(), on = 'Patient')

    return temp.reset_index()

def preprocess_x(df):
  x = np.zeros([df.shape[0], feature_num])
  for index, row in df.iterrows():
    x[index, 0] = row['Weeks'][0] + 12
    x[index, 1] = row['FVC'][0]
    x[index, 2] = row['Percent'][0]
    x[index, 3] = row['Age']
    x[index, 4] = row['Male']
    x[index, 5] = row['Female']
    x[index, 6] = row['Currently smokes']
    x[index, 7] = row['Ex-smoker']
    x[index, 8] = row['Never smoked']
    x[index, 9] = row['Height']
  return x

def preprocess_y(df):
  y = np.zeros((df.shape[0], 146))
  for index, row in df.iterrows():
    y[index, :][np.array(row['Weeks']) + 12] = row['FVC']

  return y


def correct_NN_noise(result):
    result = copy(result)
    for idx, row in enumerate(result):
        gradient = (row[80+12]-row[0+12])/80
        result[idx,:12] = np.arange(start = row[12]-gradient*12,
                                    stop = row[12],
                                    step = gradient)[:12]
        result[idx, 80:] = np.arange(start = row[80],
                                     stop = row[80] +gradient*(146-80),
                                     step = gradient)[-66:]
    return result


def preprocess_test_NN(inp, return_x = True, first_only = True, one_hot = True):
    # These data contain DICOMs requiring GDCM. Removing them from dataset.
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    for id in BAD_ID:
        inp = inp[inp['Patient'] != id]
    inp = inp.reset_index()
    
    submission_file = copy(inp)
    height_sub = np.empty(submission_file.shape[0])
    
    for idx, row in submission_file.iterrows():
        if row["Sex"] == 'Male':
            ht = row['FVC']/(27.63- 0.112 * row['Age'])
            height_sub[idx] = ht
        elif row["Sex"] == 'Female':
            ht = row['FVC']/(21.78- 0.101 * row['Age'])
            height_sub[idx] = ht
    
    submission_file['Height'] = pd.Series(height_sub)
    
    if one_hot:
        submission_file = one_hot_encode(submission_file)
    
    # If first_only == True, return only dataframe and first data, for EN training
    if first_only:
        if one_hot:
            submission_file = group_first_data(submission_file, one_hotted = True)
        else:
            submission_file = group_first_data(submission_file)
    else:
        submission_file = group_data(submission_file)
    
    if return_x:
        submission_file = preprocess_x(submission_file)
    
    return submission_file



def effnet_predict_ensemble(inp, test_img_dir):
    '''
    Predicts efficientnet for ensemble
    '''

    test_result = np.empty((inp.shape[0], 146))
    conf_result = np.empty((inp.shape[0], 146))
    patient_wk  = np.empty((inp.shape[0], 146), dtype = 'object')
    for idx, row in tqdm(inp.iterrows()):
        patient_wk_np, fvc_np, conf_np = start_predict(row['Patient'],
                                                       row['FVC'],
                                                       row['Weeks'],
                                                       row['Percent'],
                                                       row['Age'],
                                                       row['Sex'],
                                                       row['SmokingStatus'],
                                                       test_img_dir,
                                                       efficient_net_model
                                                       )
        test_result[idx,:] = fvc_np
        conf_result[idx,:] = conf_np
        patient_wk[idx,:] = patient_wk_np
    return test_result, conf_result, patient_wk


def combined_predict(df, test_img_dir, submission = True, weight = None):
    '''
    Unified predictors for both NN & EN. submission arg determines which data (test/train) is submitted.
    If weights!= None, a weighted average of the results is returned.
    '''
    
    # NN (Darrel) prediction
    print('Starting NN_predict')
    NN_input = preprocess_test_NN(df, return_x = True, first_only = False, one_hot = True)
    NN_predict = nn_model.predict(NN_input)
    NN_predict = correct_NN_noise(NN_predict)
    NN_conf = np.full(NN_predict.shape, 200)
    print("NN_predict complete")
    
    
    # Linear (Veronica) prediction
    print('Starting LN_predict')
    LN_input = preprocess_test_NN(df, return_x = False, first_only = True, one_hot = False)
    LN_predict = np.empty(NN_predict.shape)
    LN_conf = np.empty(NN_predict.shape)
    for idx, row in LN_input.iterrows():
        LN_pred = huber_predict(lower_huber,
                                mid_huber,
                                upper_huber,
                                datawrangler,
                                row['Patient'],
                                row['Weeks'],
                                row['FVC'],
                                row['Percent'],
                                row['Age'],
                                row['Sex'],
                                row['SmokingStatus']
                               )
        LN_predict[idx,:] = LN_pred['Weeks'].values
        LN_conf[idx,:] = LN_pred['Confidence'].values
    print('LN_predict complete')
    
    
    # Efficientnet (Chunhow) prediction
    print('Starting EN_predict')
    if submission:
        EN_predict = effnet_predict_full(df)
        EN_conf = None
    else:
        temp = preprocess_test_NN(df, return_x = False, first_only = True, one_hot = False)
        EN_predict, EN_conf, patient_wk = effnet_predict_ensemble(temp, test_img_dir)
    print("EN_predict complete")
    
    
    print(EN_predict)
    # CNN (Mikhail) prediction
    print('Starting MN_predict')
    MN_predict = np.empty(NN_predict.shape)
    for idx, row in tqdm(LN_input.iterrows()):
        MN_predict[idx,:] = np.array(conv3dModel(age = row['Age'],
                                              sex = row['Sex'],
                                              week = row['Weeks'],
                                              fvc = row['FVC'],
                                              percent = row['Percent'],
                                              smokingStatus = row['SmokingStatus'],
                                              imgpath = test_img_dir,
                                              modelpath = conv3d_dir,
                                              scalerpath = conv3d_dir
                                             ))

    print(MN_predict)
    print("MN_predict complete")
    
    if type(weight) != type(None):
        weighted_predict = np.transpose(np.average(np.array([np.transpose(NN_predict),
                                                             np.transpose(LN_predict),
                                                             np.transpose(EN_predict),
                                                             np.transpose(MN_predict)]), weights = weight, axis = 0))
        
        weighted_conf = np.transpose(np.average(np.array([np.transpose(NN_conf),
                                                          np.transpose(LN_conf),
                                                          np.transpose(EN_conf),]), weights = weight[:3], axis = 0))
        
        return weighted_predict, weighted_conf, patient_wk
    else:
        return NN_predict, LN_predict, EN_predict, MN_predict, EN_conf



def predict_single(patient_name, week, FVC, percent, age, sex, smoking_status):
    '''
    Function assumes img folder name for patient is the same as the patient's name.
    '''
    submit_dict = {'Patient':[patient_name],
                   'Weeks' :[week],
                   'FVC' : [FVC],
                   'Percent' :[percent],
                   'Age' : [age],
                   'Sex' : [sex],
                   'SmokingStatus' :[smoking_status]
                  }
    submit_df = pd.DataFrame.from_dict(submit_dict)
    submit_FVC, submit_Conf, patient_wk = combined_predict(submit_df, patient_name, submission = False, weight = [4.16112198e-01, 5.22315414e-04, 5.83365487e-01,1.26314924e-03])
    
    return submit_FVC, submit_Conf