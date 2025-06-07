'''
In insurance csv file you have to apply machine learnig model (linerRegresson , RandomForestRegressor) and these both files you have to upload on s3 bucket using ci/cd via github actions .


'''

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
df=pd.read_csv('D:\\enosh_regex\\Datasets\\insurance.csv')
from sklearn.model_selection import train_test_split

print(df.head(2))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()