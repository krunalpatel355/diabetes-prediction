#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\Diabetes.csv"
df = pd.read_csv(file_path)

# 1. Replace placeholders (e.g., 77 or 99) for missing data with NaN
df.replace({77: pd.NA, 99: pd.NA}, inplace=True)

# 2. Set specific columns to NaN for missing values and then replace with mean or median where applicable
# Continuous numeric columns for mean/median replacement if NaN exists
continuous_columns = ["BloodPressure", "BMI", "Fasting Glucose"]
for column in continuous_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, setting errors to NaN
    mean_value = df[column].mean()  # Calculate mean
    df[column].fillna(mean_value, inplace=True)  # Replace NaN with mean

# Set 'Age' to NaN initially for missing values, and leave it as NaN for flexibility
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Save the cleaned DataFrame to a new file
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\Meandiabetes.csv"
df.to_csv(output_path, index=False)

output_path

