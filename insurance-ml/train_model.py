import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

df = pd.read_csv('insurance.csv')

df = pd.get_dummies(df, drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
joblib.dump(lr, 'linear_regression_model.pkl')

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'random_forest_model.pkl')

print("Models trained and saved successfully.")
