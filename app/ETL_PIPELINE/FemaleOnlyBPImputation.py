#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from IPython.core.display import display, HTML 


# <b>Objective<br></b>
# • The purpose of this diabetes dataset is to predict whether a woman has diabetes or not. <br>
# • The dataset includes several medical predictors (independent variables) and one target variable (Outcome).<br>
# • The predictors used in forecasting diabetes include pregnancies, age, BMI, blood pressure, family history, and waist-to-hip ratio.

# ***Features and their description***
# 1. Pregnancies - Number of times the person conceived
# 2. Fasting Glucose - 8 hrs of fasting
# 3. Age - In years -> 21 and up
# 4. BMI - weight in kg  / Height in meters^2
# 5. Family History - 0 for no Family history in diabetes, 1 if there is family history
# 6. Waist to hip ratio = Waist / Hips in cm
# 7. Bloodpressure - Diastolic blood pressure (mm Hg)
# 8. Outcome - 0 for not diabetic and 1 if diabetic (Target Variable)

# Initial merging of data with size:

# ***Data loading with Identified Features***

# In[12]:


data = pd.read_csv("C:/diabetes1.csv")
data.info()


# Checking for missing values to avoid error when training the model
# 

# In[13]:


data.info()
#As you could see here, BP has less values than the others, it is because we have blank rows for BP


# In[14]:


correlation_matrix = data.corr()
print(correlation_matrix['BloodPressure'].sort_values(ascending=False))
# First we need to check which feature has the highest correlation with BP, we cannot include Outcome to avoid data leakage
#as this will be used to predict diabetes later


# Checking how many rows have complete values versus those that are not <br>
# Since we have identified earlier that BP has missing values, this code counts rows with no missing values and rows with at least one missing entry.<br>

# Since we have decided to keep the BP, how are we going to fill out the values of that column?<br>
# Are we going to just do average on all of them? <br>
# We are going to use Linear Regression  <br>
# <b>Why?<b>
# For each missing value, linear regression uses available data to estimate BP based on factors that are likely associated with it (e.g., age, BMI, health conditions).<br>
# Not like putting mean on all missing records, where it doesn't correlate on the individual's health record, but just as a whole on the dataset.

# In[19]:


# Feature sets for imputation of BloodPressure
feature_sets = {
    "Set 1 (Age,BMI,WaistToHipRatio)": ['Age', 'BMI', 'WaistToHipRatio'],
    "Set 2 (Age Only)": ['Age'],
    "Set 3 (All Features Except Outcome)": ['Age', 'BMI', 'WaistToHipRatio', 'Pregnancies', 'Glucose', 'FamilyHistory']
}

# Function to evaluate feature sets for imputing BloodPressure
def evaluate_feature_sets(data, feature_sets):
    results = []
    best_model = None
    best_scaler = None
    best_features = None
    best_r2_val = -float('inf')  # Initialize to negative infinity for proper comparison

    for label, features in feature_sets.items():
        BP_X = data[features]
        BP_y = data['BloodPressure']

        # First split: 80/20
        BP_X_train_80, BP_X_test_20, BP_y_train_80, BP_y_test_20 = train_test_split(
            BP_X, BP_y, test_size=0.2, random_state=42
        )

        # Second split: 60/20 from the 80% training data
        BP_X_train_60, BP_X_val_20, BP_y_train_60, BP_y_val_20 = train_test_split(
            BP_X_train_80, BP_y_train_80, test_size=0.25, random_state=42
        )

        # Scale the data
        scaler = StandardScaler()
        BP_X_train_60_scaled = scaler.fit_transform(BP_X_train_60)
        BP_X_val_20_scaled = scaler.transform(BP_X_val_20)
        BP_X_test_20_scaled = scaler.transform(BP_X_test_20)

        # Train the model on 60% training data (non-missing BP values)
        train_complete = BP_X_train_60[~BP_y_train_60.isna()]
        BP_y_train_complete = BP_y_train_60.dropna()

        if train_complete.empty:
            continue

        train_complete_scaled = scaler.transform(train_complete)
        model = LinearRegression()
        model.fit(train_complete_scaled, BP_y_train_complete)

        # Evaluate on validation set
        val_complete = BP_X_val_20[~BP_y_val_20.isna()]
        BP_y_val_actual = BP_y_val_20.dropna()
        val_complete_scaled = scaler.transform(val_complete)
        BP_y_val_predicted = model.predict(val_complete_scaled)

        # Evaluate on test set
        test_complete = BP_X_test_20[~BP_y_test_20.isna()]
        BP_y_test_actual = BP_y_test_20.dropna()
        test_complete_scaled = scaler.transform(test_complete)
        BP_y_test_predicted = model.predict(test_complete_scaled)

        # Calculate metrics
        r2_val = r2_score(BP_y_val_actual, BP_y_val_predicted)
        mse_val = mean_squared_error(BP_y_val_actual, BP_y_val_predicted)
        r2_test = r2_score(BP_y_test_actual, BP_y_test_predicted)
        mse_test = mean_squared_error(BP_y_test_actual, BP_y_test_predicted)

        results.append({
            "Feature Set": label,
            "R-squared (Validation)": r2_val,
            "MSE (Validation)": mse_val,
            "R-squared (Test)": r2_test,
            "MSE (Test)": mse_test
        })

        # Update best model
        if r2_val > best_r2_val:
            best_model = model
            best_scaler = scaler
            best_features = features
            best_r2_val = r2_val  # Update best R-squared (Validation)

    return results, best_model, best_scaler, best_features

# Function to fill missing BloodPressure values
def fill_missing_values_sequentially(data, best_model, best_scaler, best_features):
    BP_X = data[best_features]
    missing_indices = data['BloodPressure'].isna()

    if missing_indices.any():
        BP_X_scaled = best_scaler.transform(BP_X)
        missing_data = BP_X_scaled[missing_indices]
        predicted_values = best_model.predict(missing_data)
        predicted_values = np.round(predicted_values).astype(int)
        data.loc[missing_indices, 'BloodPressure'] = predicted_values
        print("Missing values successfully filled!")
    return data

# Main Execution
data = pd.read_csv(r"C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes1.csv")

# Perform imputation for BloodPressure
results, best_model, best_scaler, best_features = evaluate_feature_sets(data, feature_sets)
updated_data = fill_missing_values_sequentially(data, best_model, best_scaler, best_features)

# Display which feature set was used
print("\nChosen Feature Set for Imputation of BloodPressure:")
print(best_features)

# Display evaluation results
results_df = pd.DataFrame(results)
html_table = results_df.to_html(index=False, border=1, justify="center", classes="table table-bordered table-striped")
display(HTML(html_table))





