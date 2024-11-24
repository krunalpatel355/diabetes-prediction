#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os

# File paths based on provided locations
files = {
    "InsulinGlucose": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\insulincombinedupdated.csv",
    "ReproductiveHealth": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\reproductivehealthPregnancies.csv",
    "CombinedBloodPressure": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BPcombinedNew.csv",
    "CombinedBMI": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BMI_combined.csv",
    "Outcome": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\Outcome.csv",
    "FamilyHistory": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\FinalCombinedDiabetesDataWithAge.csv",
    "Demographics": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\demographics2.csv"
}

# Helper function to load CSV
def load_csv(filepath, columns):
    df = pd.read_csv(filepath, usecols=columns, low_memory=False)
    df["SEQN"] = df["SEQN"].astype("Int64")  # Ensure SEQN is integer (nullable integer type)
    return df

# Load SEQN and Age data from Demographics.csv
demographics_df = load_csv(files["Demographics"], ["SEQN", "RIDAGEYR", "RIDEXPRG"])
demographics_df.rename(columns={"RIDAGEYR": "Age"}, inplace=True)

# Load reproductive health data from ReproductiveHealth.csv and clean RHQ160
reproductive_health_df = load_csv(files["ReproductiveHealth"], ["SEQN", "RHD143", "RHQ160"])
reproductive_health_df["RHQ160"] = 0  # Set pregnancies to zero as per instruction

# Calculate pregnancies
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = pregnancy_df["RHQ160"]

# Replace any values in Number of Pregnancies greater than 11 with zero
pregnancy_df["Number of Pregnancies"] = pregnancy_df["Number of Pregnancies"].apply(lambda x: x if x <= 11 else 0)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio, calculate if not provided
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data and filter only diastolic values
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
blood_pressure_df["BloodPressure"] = blood_pressure_df[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].max(axis=1)

# Replace BloodPressure = 0 or pd.NA with a random value between 60 and 80
blood_pressure_df["BloodPressure"] = blood_pressure_df.apply(
    lambda row: np.random.randint(60, 80) if pd.isna(row["BloodPressure"]) or row["BloodPressure"] == 0 else row["BloodPressure"],
    axis=1
)
blood_pressure_df = blood_pressure_df[["SEQN", "BloodPressure"]]

# Load Insulin and Glucose data, generate appropriate random fasting glucose based on age and outcome
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU"])
insulin_glucose_df.rename(columns={"LBXGLU": "Fasting Glucose"}, inplace=True)
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: np.random.randint(70, 99) if pd.isna(row["Fasting Glucose"]) or row["Fasting Glucose"] == 0 else row["Fasting Glucose"],
    axis=1
)

# Load Outcome data, replace NaNs with 0 and move to the last column
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(
    lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0))
)
outcome_df["Outcome"] = outcome_df["Outcome"].fillna(0)  # Set missing to 0
outcome_df = outcome_df[["SEQN", "Outcome"]]

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")

# Add Outcome data as the last column
result_df = result_df.merge(outcome_df, on="SEQN", how="left")

# Specify the new output path
archive_dir = r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive"
os.makedirs(archive_dir, exist_ok=True)  # Create directory if it doesn't exist
new_file_path = os.path.join(archive_dir, "DiabetesNHANESUpdated_New.csv")

# Save the final DataFrame to the new location
result_df.to_csv(new_file_path, index=False)

print(f"Combined data saved to {new_file_path}")




# In[ ]:


import pandas as pd
import numpy as np

# File paths based on provided locations
files = {
    "InsulinGlucose": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\insulincombinedupdated.csv",
    "ReproductiveHealth": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\reproductivehealthPregnancies.csv",
    "CombinedBloodPressure": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BPcombinedNew.csv",
    "CombinedBMI": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BMI_combined.csv",
    "Outcome": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\Outcome.csv",
    "FamilyHistory": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\FinalCombinedDiabetesDataWithAge.csv",
    "Demographics": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\demographics2.csv"
}

# Helper function to load CSV
def load_csv(filepath, columns):
    df = pd.read_csv(filepath, usecols=columns, low_memory=False)
    df["SEQN"] = df["SEQN"].astype("Int64")  # Ensure SEQN is integer (nullable integer type)
    return df

# Load SEQN, Age, and Outcome data
demographics_df = load_csv(files["Demographics"], ["SEQN", "RIDAGEYR"])
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df["Outcome"] = outcome_df["Outcome"].fillna(0)  # Replace NaN values with 0

# Merge Outcome with demographics and replace any remaining NaN with 0
demographics_df = demographics_df.merge(outcome_df[["SEQN", "Outcome"]], on="SEQN", how="left")
demographics_df["Outcome"] = demographics_df["Outcome"].fillna(0)  # Ensure no NaN remains in Outcome
demographics_df.rename(columns={"RIDAGEYR": "Age"}, inplace=True)

# Load BMI data
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXBMI"])
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)

# Load Fasting Glucose data
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU"])
insulin_glucose_df.rename(columns={"LBXGLU": "Fasting Glucose"}, inplace=True)

# Merge Age, Outcome, BMI, and Fasting Glucose data into one dataframe
glucose_data = demographics_df.merge(combined_bmi_df, on="SEQN", how="left")
glucose_data = glucose_data.merge(insulin_glucose_df, on="SEQN", how="left")

# Function to assign fasting glucose based on age, BMI, and outcome
def assign_fasting_glucose(row):
    if pd.isna(row["Fasting Glucose"]) or row["Fasting Glucose"] == 0:
        if row["Outcome"] == 0:  # Non-diabetic
            if row["Age"] < 45:
                return np.random.randint(70, 90)  # Normal range for younger adults
            else:
                return np.random.randint(85, 99)  # Normal range for older adults
        elif row["Outcome"] == 1:  # Diabetic
            if row["BMI"] < 25:
                return np.random.randint(100, 125)  # Pre-diabetic range for normal weight
            else:
                return np.random.randint(125, 140)  # Diabetic range for higher BMI
    else:
        return row["Fasting Glucose"]

# Apply the function to fill missing and zero values in Fasting Glucose
glucose_data["Fasting Glucose"] = glucose_data.apply(assign_fasting_glucose, axis=1)

# Finalize the dataset
glucose_data = glucose_data[["SEQN", "Age", "BMI", "Fasting Glucose", "Outcome"]]

# Save the modified dataset
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
glucose_data.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[ ]:


# Reload the new uploaded file to ensure the latest data
file_path = 'C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv'
data = pd.read_csv(file_path)

# Remove duplicate SEQNs, keeping the row with the most non-NaN values
data = data.sort_values(by='SEQN')  # Sort by SEQN to process duplicates sequentially
data = data.loc[data.groupby('SEQN').apply(lambda x: x.count(axis=1).idxmax())]  # Keep row with most data per SEQN

# Set Outcome NaNs and blanks to zero
data['Outcome'] = data['Outcome'].fillna(0).replace('', 0).astype(int)

# Set Number of Pregnancies to zero
data['Number of Pregnancies'] = 0

# Fill in Waist to Hip Ratio with calculated values based on age and BMI, using random but controlled ranges
def calculate_waist_to_hip_ratio(row):
    if pd.isna(row['Waist to Hip Ratio']):
        if row['BMI'] >= 30 or row['Age'] >= 60:
            return round(np.random.uniform(0.95, 1.1), 2)  # Higher ratio for obesity/older age
        elif row['BMI'] >= 25:
            return round(np.random.uniform(0.85, 0.95), 2)  # Slightly higher for overweight
        else:
            return round(np.random.uniform(0.75, 0.85), 2)  # Normal range
    return row['Waist to Hip Ratio']

data['Waist to Hip Ratio'] = data.apply(calculate_waist_to_hip_ratio, axis=1)

# Fill Fasting Glucose based on age and Outcome
def generate_fasting_glucose(row):
    if pd.isna(row['Fasting Glucose']) or row['Fasting Glucose'] == 0:
        if row['Outcome'] == 0:
            if row['Age'] < 45:
                return np.random.randint(70, 90)  # Normal range for younger adults
            else:
                return np.random.randint(85, 99)  # Normal range for older adults
        else:
            if row['BMI'] < 25:
                return np.random.randint(100, 125)  # Pre-diabetic range for normal weight
            else:
                return np.random.randint(125, 140)  # Diabetic range for higher BMI
    return row['Fasting Glucose']

data['Fasting Glucose'] = data.apply(generate_fasting_glucose, axis=1).astype(int)

# Remove non-diastolic Blood Pressure values and fill with age and BMI adjusted values
def calculate_diastolic_bp(row):
    if pd.isna(row['Blood Pressure']) or row['Blood Pressure'] == 0:
        if row['BMI'] >= 30 or row['Age'] >= 60:
            return np.random.randint(80, 90)  # Higher diastolic BP for obesity/older age
        elif row['BMI'] >= 25:
            return np.random.randint(75, 85)  # Slightly elevated diastolic BP
        else:
            return np.random.randint(60, 80)  # Normal diastolic range
    return row['Blood Pressure']

data['Blood Pressure'] = data.apply(calculate_diastolic_bp, axis=1).astype(int)

# Round off all float columns to integers
data = data.round(0).astype(int, errors='ignore')

# Save the cleaned and processed dataset
cleaned_file_path = '/mnt/data/Cleaned_DiabetesNHANES.csv'
data.to_csv(cleaned_file_path, index=False)

cleaned_file_path


# In[ ]:





# In[ ]:




