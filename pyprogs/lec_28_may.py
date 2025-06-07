'''
----------------------------------------------------------------------
# HOMEWORK : study UV and install in the system, its like pip but better in every way.
------------------------------------------------------------------------
'''

# LogisticRegression ---> DecisionTreeClassifier ---> R.F.Classifier


# LogisticRegression ---> When we have categorical target column
# Backend ---> Activation function ---> sigmoid ---> Probability convert # all sub-categories ---> threshold_value = 0.5

# 0.5 <= output = 1
# 0.5 > output = 0


import numpy as np 
import pandas as pd

df = pd.read_csv('D:\\enosh_regex\\Datasets\\covid.csv')

df.isnull().sum()

from sklearn.impute import SimpleImputer

si=SimpleImputer()

df['fever'] = si.fit_transform(df[['fever']])

#df.isnull().sum()

#print(df.head(2))

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])


x = df.drop(columns = ['has_covid'])
y = df['has_covid']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

#print(lr.fit(x_train, y_train))

#y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score

#print(accuracy_score(y_test, y_pred))


# DecisionTreeClassifier ---> 
# Changes in entropy and gini index

# -p1log1-p2log2 3if we have 2 sub-categories
# -p1log1-p2log2-p3log3....-pnlogn  #if we have more than 2 sub-categories

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

#print(accuracy_score(y_test, y_pred))


# RandomForestClassifier

# Ensemble method ---> multiple decision tree works ---> 100 d.t. ---> 70 yes, 30no ---> final output ---> majority(yes)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

#print(accuracy_score(y_test, y_pred))


# LinearRegression --- DecisionTreeRegressor --->  RF.Regression

import numpy as np
import pandas as pd

df = pd.read_csv('D:\\enosh_regex\\Datasets\\insurance.csv')
df = pd.get_dummies(df, columns = ['sex', 'smoker','region'])
#print(df.head(2))

x=df.drop(columns=['charges'])
y = df['charges']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42)

# LinearRegressionModel

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score

#print(r2_score(y_test, y_pred))


# DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train, y_train)

y_pres = dt.predict(x_test)

#print(r2_score(y_test, y_pred))


# RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

#print(r2_score(y_test, y_pred))



# Naive-Bayes-classifier


import numpy as np 
import pandas as pd

df = pd.read_csv('D:\\enosh_regex\\Datasets\\social_network_ads.csv', usecols = ['Age', 'EstimatedSalary', 'Purchased'])

print(df.head())

x = df.drop(columns = ['Purchased']) #Independent column
y = df['Purchased']   #dependent column

from sklearn.model_selection import train_test_split(x,y,test_size=0.2,random_state=23)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=23)

from sklearnpreprocessing import StandardScaler

sc= StandardScaler()

x_train_new = sc.fit_transform(x_train)
x_test_new = sc.transform(x_test)


# from sklearn.naive_bayes import BurnolliNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliiNB

classifier = GaussianNB()

classifier.fit(x_train_new, y_train)

GaussianNB()

y_pred = classifier.predict(x_test_new)

print(y_pred)

















































