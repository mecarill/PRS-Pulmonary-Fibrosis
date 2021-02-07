import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.compose import ColumnTransformer
import zipfile
import os
import re
import pydicom 
import matplotlib.pyplot as plt
import glob
import time
import shutil
import operator

from src.linear_model import NoTransformer
from src.ensemble import predict_single

def main():

    # sidebar
    st.sidebar.subheader("Patient Details")
    patient = 'placeholder' #st.text_input("Patient Name")
    sex = st.sidebar.radio('Sex',['Male','Female'])
    st.sidebar.markdown('')
    smokingstatus = st.sidebar.radio('Smoking Status',['Never smoked','Ex-smoker','Currently smokes'])
    st.sidebar.markdown('')
    age = st.sidebar.slider('Age', 1, 100)
    fvc = st.sidebar.slider('FVC', 1, 5000)
    percent = st.sidebar.slider('Patients FVC as a percentage of Normal FVC', 1, 100)
    week = st.sidebar.slider('Number of weeks between FVC measurement and CT scan', -10, 100)

    # main page
    st.title('Predicting Pulmonary Fibrosis Progression')

    st.write('')
    st.write("To get a prediction, please indicate Patient's Details in the sidebar, and upload Patient's CT scans below.")
    st.write('')

    uploaded_file = st.file_uploader("CT scans of patient's lungs (Please upload zip folder containing dcm images)")

    if uploaded_file is not None:
        try:
            dcm_paths, datasets, image_path = read_uploaded_file(uploaded_file)
            print(image_path)
            plotted_img = os.path.join('plotted_img',image_path.split('/')[-2] + '.png')

            if os.path.exists(plotted_img):
                st.image(plotted_img,use_column_width=True)
            else:
                dir = 'plotted_img'
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.mkdir(dir)
                create_plotted_image(dcm_paths,datasets, plotted_img)
                st.image(plotted_img,use_column_width=True)
            
        except(zipfile.BadZipFile): 
            st.markdown("<font color='red'>Note: Please upload a zip file</font>", unsafe_allow_html=True)

    if (age!=1) and (fvc!=1) and (percent!=1) and (week!=-10) and (uploaded_file is not None):     
        st.subheader("FVC prediction for weeks 0 to 133")

        patient_name = image_path
        fvc, confidence = predict_single(patient_name, week,fvc,percent,age,sex,smokingstatus)
        fvc = fvc[0][12:]
        standard_deviation = confidence[0][12:]/2
        
        lower_confidence = list(map(operator.sub, fvc, standard_deviation))
        upper_confidence = list(map(operator.add, fvc, standard_deviation))
        df = pd.DataFrame({'FVC': fvc, 'Lower Confidence':lower_confidence, 'Upper Confidence': upper_confidence})
        st.line_chart(df)
        st.table(df)


 

# @st.cache(allow_output_mutation=True) 
def read_uploaded_file(uploaded_file):

    # remove previously uploaded images
    try:
        shutil.rmtree('zip')
    except(FileNotFoundError):
        pass

    with zipfile.ZipFile(uploaded_file, 'r') as z: 
        z.extractall('zip')
    patient_dir = "zip/**/"
    ext = ('*.dcm')
    
    datasets = []

    # Order the files in the dataset
    dcm_paths = []
    for dcm in glob.glob(patient_dir + ext, recursive=True): 
        dcm_path = os.path.join(os.getcwd(),dcm)
        dcm_paths.append(dcm_path) 
    dcm_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Read in the Dataset
    for dcm_path in dcm_paths:
        datasets.append(pydicom.dcmread(dcm_path))

    image_path = dcm_paths[0].replace('\\','/')
    image_path=image_path.replace(image_path.split('/')[-1],'')
    return dcm_paths, datasets, image_path


def create_plotted_image(dcm_paths, datasets, plotted_img):

    # Plot the images
    columns = 10
    rows = len(dcm_paths)//columns + 1
    fig=plt.figure(figsize=(16, rows+3))

    for i in range(1, len(dcm_paths)):
    
        img = datasets[i-1].pixel_array
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap="plasma")
        plt.title(i, fontsize = 9)
        plt.axis('off')
    plt.savefig(plotted_img, bbox_inches='tight')

    return 

main()
