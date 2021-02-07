import _pickle as cPickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NoTransformer(BaseEstimator, TransformerMixin):
    """Passes through data without any change and is compatible with ColumnTransformer class"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X
        

def load_huber_models(lower_huber_path, mid_huber_path, upper_huber_path, datawrangler_path):
    
    with open(lower_huber_path, 'rb') as f:
        lower_huber = cPickle.load(f)

    with open(mid_huber_path, 'rb') as f:
        mid_huber = cPickle.load(f)

    with open(upper_huber_path, 'rb') as f:
        upper_huber = cPickle.load(f)

    with open(datawrangler_path, 'rb') as f:
        datawrangler = cPickle.load(f)

    return lower_huber, mid_huber, upper_huber, datawrangler
    

def huber_predict(lower_huber, mid_huber, upper_huber, datawrangler, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start=-12, week_end=134):
    
    MinWeek, FirstFVC, FullFVC, Height = _engineer_feature(Week, FVC, Percent, Age, Sex)

    df = _create_df_with_running_weeks(Patient,
                                        Week,
                                        FVC,
                                        Percent,
                                        Age,
                                        Sex,
                                        SmokingStatus,
                                        MinWeek,
                                        FirstFVC,
                                        FullFVC,
                                        Height,
                                        week_start,
                                        week_end)

    df, df_transformed = _wrangle_data(df, datawrangler)
    
    df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)
    
    return df
        
def _engineer_feature(Week, FVC, Percent, Age, Sex):
    '''
    function to calculate MinWeek, FullFVC, and Height from patient details
    
    '''

    MinWeek = min(Week, 0)
    FirstFVC = FVC
    FullFVC = (FVC/Percent)*100

    if Sex == 'Male':
        Height = FirstFVC / (27.63 - 0.112 * Age)
    else:
        Height = FirstFVC / (21.78 - 0.101 * Age)

    return MinWeek, FirstFVC, FullFVC, Height
    
def _create_df_with_running_weeks(Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus, MinWeek, FirstFVC, FullFVC, Height, week_start, week_end):

    '''
    function to put patient details, engineered features, and running list of weeks into DataFrame
    
    '''

    Weeks = list(range(week_start, week_end))
    df = pd.DataFrame({'Weeks':Weeks})
    df['Patient'] = Patient
    df['Sex'] = Sex
    df['Age'] = Age
    df['SmokingStatus'] = SmokingStatus
    df['MinWeek'] = MinWeek
    df['FirstFVC'] = FVC
    df['FullFVC'] = FullFVC
    df['Height'] = Height
    df['Percent'] = Percent
    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']
    df['FVC'] = 0 # dummy FVC, to be predicted

    return df

def _wrangle_data(df, datawrangler):
    '''
    function to transform patient details into suitable format for models' ingestion
    
    '''

    transformed_data_series = datawrangler.transform(df)

    ## put transformed series into dataframe

    # define which attributes shall not be transformed, are numeric or categorical
    no_transform_attribs = ['Patient','FVC']
    num_attribs = ['Percent', 'Age', 'WeeksPassed', 'FirstFVC','Height', 'Weeks', 'MinWeek', 'FullFVC']
    cat_attribs = ['Sex', 'SmokingStatus']

    # get column names for non-categorical data
    new_col_names = no_transform_attribs + num_attribs

    # extract possible values from the fitted transformer
    categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
    new_col_names += categorical_values

    # create Dataframe based on the extracted Column-Names
    df_transformed = pd.DataFrame(transformed_data_series, columns=new_col_names)

    return df, df_transformed


def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):
    
    '''
    function to predict lower, upper and mid FVC and confidence interval for patients
    
    '''
    csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']
    df_transformed = df_transformed[csv_features_list]

    preds_lower = lower_huber.predict(df_transformed)
    preds_mid = mid_huber.predict(df_transformed)
    preds_upper = upper_huber.predict(df_transformed)
    
    df['Lower'] = preds_lower
    df['Upper'] = preds_upper
    df['FVC'] = preds_mid
    df['Confidence'] = abs(preds_upper - preds_lower)

    return df
