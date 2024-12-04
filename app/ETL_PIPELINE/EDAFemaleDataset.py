#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
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

# In[23]:


# List of file paths
file_paths = [
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\questionnaire_merged_data\questionnairedata.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_blood_pressure_data.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_bmi_data.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_health_data.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_skin_thickness_data.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\ReproductiveHealth_Doc\ReproductiveHealth\ReproductiveHealth_combined.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\diabetessas\DiabetesData\DataFiles\DiabetesData_updated.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\diabetessas\DiabetesData\DataFiles\OutcomeDIQ010.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\insulinfiles\insulindata.csv",
    r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\DemographicsSAS files\demographics.csv"
]
total_size = 0
total_rows = 0
total_columns = 0

for file_path in file_paths:
    # Calculate file size in MB
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    total_size += file_size
    
    try:
        # Load data and handle parsing errors
        data = pd.read_csv(file_path, on_bad_lines='skip')
        file_rows, file_columns = data.shape
        total_rows += file_rows
        total_columns = max(total_columns, file_columns)  # Ensures the highest column count
        
        print(f"File: {file_path}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Rows: {file_rows}, Columns: {file_columns}\n")
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Summary of combined data
print("Total Combined Data:")
print(f"Total Size for All Files: {total_size:.2f} MB")
print(f"Total Rows: {total_rows}")
print(f"Total Columns: {total_columns}")


# ***Data loading with Identified Features***

# In[4]:


data = pd.read_csv(r"C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes1.csv")
data.info()


# In[3]:


data.tail()


# In[4]:


data = pd.read_csv(r"C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes1.csv")
data.describe()


# Checking for missing values to avoid error when training the model
# 

# In[5]:


data.info()
#As you could see here, BP has less values than the others, it is because we have blank rows for BP


# Exploring if there are NaN values, data shows no NaN values

# In[7]:


# Load the provided dataset
data = pd.read_csv(r'C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes1.csv')

# Check the initial number of rows
initial_rows = data.shape[0]

# Drop rows that have fewer than 1 non-null values
data_filtered = data.dropna(thresh=1)

# Check the number of rows after filtering
final_rows = data_filtered.shape[0]
rows_dropped = initial_rows - final_rows

initial_rows, final_rows, rows_dropped


# This heatmap visualizes missing values in the dataset.
# The pure black cells indicate No non-missing values, while the lighter cells indicate missing values and it is showing in the BP section

# In[8]:


sns.heatmap(data.isnull())


# Checking how many rows have complete values versus those that are not <br>
# Since we have identified earlier that BP has missing values, this code counts rows with no missing values and rows with at least one missing entry.<br>

# In[19]:


# Count of rows with all values intact (no NaNs)
rows_no_nan = data.dropna().shape[0]

# Count of rows with at least one NaN value
rows_with_nan = data.shape[0] - rows_no_nan

# Display results
print("Rows with all values intact (no NaN values):", rows_no_nan)
print("Rows with at least one NaN value:", rows_with_nan)


# Since we have decided to keep the BP, how are we going to fill out the values of that column?<br>
# Are we going to just do average on all of them? <br>
# We are going to use Linear Regression  <br>
# <b>Why?<b>
# For each missing value, linear regression uses available data to estimate BP based on factors that are likely associated with it (e.g., age, BMI, health conditions).<br>
# Not like putting mean on all missing records, where it doesn't correlate on the individual's health record, but just as a whole on the dataset.

# In[20]:


# Define features to use in the model
features = ['Glucose', 'BMI', 'WaistToHipRatio', 'FamilyHistory', 'Age']

# Only proceed if there are missing BloodPressure values
if data['BloodPressure'].isna().sum() > 0:
    # Separate rows with complete and missing BloodPressure values
    data_complete = data.dropna(subset=['BloodPressure'] + features)
    data_missing = data[data['BloodPressure'].isna()]

    # Define X (predictors) and y (target) for training
    X_train = data_complete[features]
    y_train = data_complete['BloodPressure']

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure X_missing has rows before attempting prediction
    X_missing = data_missing[features].dropna()
    if not X_missing.empty:
        # Predict missing BloodPressure values and convert them to integers
        predicted_values = model.predict(X_missing).astype(int)

        # Convert the predicted values to integers
        predicted_values = predicted_values.astype(int)

        # Fill missing BloodPressure values in the original DataFrame
        data.loc[data['BloodPressure'].isna(), 'BloodPressure'] = predicted_values
        print("Imputation complete. Missing values in BloodPressure have been filled using Linear Regression.")
    else:
        print("No valid rows found for prediction in the missing data due to missing values in predictor columns.")
else:
    print("All missing BloodPressure values have already been filled.")

# Ensure the entire BloodPressure column is converted to integer type
data['BloodPressure'] = data['BloodPressure'].astype(int)

# Track final number of rows
final_rows = data.shape[0]
rows_dropped = initial_rows - final_rows

print(f"Initial rows: {initial_rows}, Final rows after filtering: {final_rows}, Rows dropped: {rows_dropped}")


# In[11]:


missing_bp_count = data['BloodPressure'].isna().sum()
print(f"Remaining missing values in BloodPressure: {missing_bp_count}")
# Find rows that originally had missing values and have now been filled
filled_bp_data = data[data['BloodPressure'].notna()]
print("Sample of filled BloodPressure values:\n", filled_bp_data[['BloodPressure']].head())
# Find rows that originally had missing values and have now been filled
filled_bp_data = data[data['BloodPressure'].notna()]
print("Sample of filled BloodPressure values:\n", filled_bp_data[['BloodPressure']].head())

#output after training the model for Linear Reg


# In[12]:


data.describe()


# ***Data Visualization***

# In[13]:


plt.figure(figsize=(8, 4))
sns.countplot(x='Outcome', data=data, palette='Set2', hue=None, legend=False)
plt.show()
#This graph shows the distribution of the people with diabetes versus those who are healthy


# **Observing Outliers**

# In[14]:


plt.figure(figsize=(12, 12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BMI', 'WaistToHipRatio','Age']):
    plt.subplot(3, 2, i + 1)  # Adjusted to a 3x2 grid for six plots
    sns.boxplot(x=col, data=data)
    plt.title({col})  # adds titles to each plot

plt.tight_layout() 
plt.show()
# BMI: High BMI outliers may represent individuals with obesity, relevant to health-related analyses.
# WaistToHipRatio: High waist-to-hip ratio outliers may indicate central obesity, also important for understanding health risks.


# Key Observations from Each Feature on Pair Plot:
# Glucose:
# Higher glucose levels are associated with diabetic cases (orange).
# There’s a clear distinction between diabetic and non-diabetic cases, with diabetic cases concentrated at higher glucose values.
# 
# BMI (Body Mass Index):
# Higher BMI values are also associated with diabetes.
# There is a visible clustering of diabetic cases at higher BMI levels, indicating BMI as a significant factor for diabetes risk.
# Like glucose, BMI shows a clear separation between diabetic and non-diabetic cases.
# 
# Age:
# Older individuals tend to have a higher probability of diabetes.
# Diabetic cases (orange) are more frequent in the older age range, suggesting age is also an influential factor.
# However, the separation is not as distinct as with glucose or BMI.
# 
# Pregnancies:
# Higher numbers of pregnancies show a slight association with diabetes, but this relationship is weaker.
# There are diabetic cases across various pregnancy counts, but a slight increase in cases with more pregnancies is visible.
# 
# Blood Pressure:
# Blood pressure does not show a strong correlation with diabetes.
# 
# Waist-to-Hip Ratio:
# Shows some clustering of diabetic cases at higher ratios, but the separation is not as distinct as with glucose and BMI.
# 
# Family History:
# There is little visible separation between diabetic and non-diabetic cases based on family history.This is due to the data structure, where its only indicating a binary outcome.
# 
# Summary of Insights:
# Strong Predictors: Glucose and BMI are the strongest predictors of diabetes, as diabetic cases are clustered at higher values for these features.
# Moderate Predictors: Age shows some association, with older individuals more likely to be diabetic. Waist-to-Hip Ratio also shows a slight increase in diabetic cases at higher values.
# Weak Predictors: Blood Pressure, Family History, and Pregnancies show weaker associations with diabetes and may be less influential for prediction purposes.

# In[16]:


sns.pairplot(data=data_complete, hue='Outcome')
plt.show()
# Orange means diabetic, blue not diabetic


# In[18]:


plt.figure(figsize = (12,12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BMI', 'WaistToHipRatio','Age']):
    plt.subplot(3, 3, i + 1) 
    sns.histplot(x=col, data=data_complete, kde = True)
    plt.title(f'Hist Plot of {col}') 

plt.tight_layout()  
plt.show()

#for example for pregnancies, values are decreasing as number of pregnancy increases
#Glucose distribution is between 80 and 150
#Bmi is roughly normal,centered around 30 which falls in the overweight to obese range.Many individuals fall in the overweight or obese range, which is often a risk factor for diabetes. The BMI distribution aligns with expectations for a population where obesity could contribute to diabetes prevalence.
# WaistToHipRatio does not indicate a clear pattern, this feature alone may not provide strong prediction power to influence diabetes without combining with other feature
#while Age is spread out across different age groups. peaks around 20 to 30 years old and 50-60


# In[ ]:




