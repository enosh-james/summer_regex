import pandas as pd
import joblib

lr_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

new_data = pd.DataFrame({
    'age': [30],
    'bmi': [25.5],
    'children': [2],
    'sex_male': [1],
    'smoker_yes': [0],
    'region_northwest': [0],
    'region_southeast': [0],
    'region_southwest': [1]
})

lr_pred = lr_model.predict(new_data)
print(f"Linear Regression Prediction: {lr_pred[0]:.2f}")

rf_pred = rf_model.predict(new_data)
print(f"Random Forest Prediction: {rf_pred[0]:.2f}")
