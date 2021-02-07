# Pulmonary Fibrosis Progress Prediction

# Introduction

Artificial Intelligence (AI) and Machine Learning has been used in the medical field since the early 70s, when AI was only 15 years old. Since then, AI has been used to assist medical professionals to make informed decisions, particularly in areas where diagnosis or prediction is less straightforward. One such area would be in predicting the progression of fibrotic lung disease, which is what this project aims to do. Taking patient’s physiological and behavioral characteristics as well as Computerized Tomography (CT) scans of their lungs, and processing them through an ensemble of traditional machine learning techniques as well as deep learning techniques, our intelligent system then returns clinicians a prediction of the decline in patient's lung function in the form of forced vital capacity (FVC). This paper discusses the approach in building this system and delves into the mechanics behind it. Our evaluations show that the system achieves competitive performance. 

# Getting Started 

Ensure that [anaconda or miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed. Also ensure that [Git Large File Storage](https://git-lfs.github.com/) is installed.

Execute the following commands in terminal to run the application locally. The commands below clones this repository, and install all the necessary packages in a new environment named 'fibrosis', before starting the application in http://localhost:8501/.

Administrator privilages may be required to run the app. This can be done by running Anaconda prompt as an administrator on Windows.

```
# clone this repository
git clone https://github.com/VeronicaLoy/pulmonary-fibrosis-progression.git

cd pulmonary-fibrosis-progression

conda create -n fibrosis python=3.6.7
conda activate fibrosis
pip install -r requirements.txt

streamlit run app.py
```

Alternatively, you can access the app [here](http://54.186.100.151:8501/). Due to hosting limitations, the publicly hosted model is a simplied version, which contains three out of four models in the ensemble - the MLP model, EfficientNet model, and Huber Linear model (see section on About the Model below for more information). The application will be hosted on at the aforementioned link for a period of 2 months, till 31st December 2020.


# Using the App

The app requires patient details to be indicated, and a zip folder containing CT scans of patients lungs in to be uploaded before prediction. The CT scans has to be in .dcm format. A sample of this zip folder of .dcm images can be found under `sample dataset` folder.

![alt text](https://github.com/VeronicaLoy/pulmonary-fibrosis-progression/blob/master/images/front-end.gif)




# About the Model

The system consists of 4 machine learning sub-models and an ensemble model which aggregates the results and determines the final output. Using this private dataset [here](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data), the ensemble achieved a laplace log likelihood of -6.87. The four models are as follows. The architecture is illustrated below.

- MLP model
- EfficientNet model
- Huber linear model
- CNN model

![alt text](https://github.com/VeronicaLoy/pulmonary-fibrosis-progression/blob/master/images/system%20architecture.png)

Inputs by the user are accepted through a web-based GUI, which also displays the prediction results.The inputs first go through the necessary pre-processing steps before being fed into the individual models. The model outputs are then passed through the ensemble model before being returned to the user as a 133-week forecast of the patient’s FVC and the confidence in each week’s forecast.
