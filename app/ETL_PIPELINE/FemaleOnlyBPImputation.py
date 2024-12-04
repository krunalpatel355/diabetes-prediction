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




