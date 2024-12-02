#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# Updated file paths based on provided locations
files = {
    "InsulinGlucose": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\insulincombinedupdated.csv",
    "ReproductiveHealth": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\reproductivehealthPregnancies.csv",
    "CombinedBloodPressure": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BPcombinedNew.csv",
    "CombinedBMI": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\BMI_combined.csv",
    "Outcome": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\Outcome.csv",
    "FamilyHistory": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\FinalCombinedDiabetesDataWithAge.csv",
    "Demographics": r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalFiles\demographics2.csv"
}

# Columns to extract from each file, based on the location of each variable
columns_to_extract = {
    "ReproductiveHealth": ["SEQN", "RHD143", "RHQ160"],
    "InsulinGlucose": ["SEQN", "LBXGLU", "PHAFSTHR"],
    "CombinedBloodPressure": ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "CombinedBMI": ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"],
    "Outcome": ["SEQN", "DIQ010"],
    "FamilyHistory": ["SEQN", "FamilyHistory"],
    "Demographics": ["SEQN", "RIDAGEYR", "RIDEXPRG"]
}

# Helper function to load CSV and ensure SEQN is integer
def load_csv(filepath, columns):
    df = pd.read_csv(filepath, usecols=columns, low_memory=False)
    df["SEQN"] = df["SEQN"].astype("Int64")  # Ensure SEQN is integer (nullable integer type)
    return df

# Load SEQN and Age data from Demographics.csv
demographics_df = load_csv(files["Demographics"], ["SEQN", "RIDAGEYR", "RIDEXPRG"])
demographics_df.rename(columns={"RIDAGEYR": "Age"}, inplace=True)  # Rename RIDAGEYR to Age 

# Load reproductive health data from ReproductiveHealth.csv
reproductive_health_df = load_csv(files["ReproductiveHealth"], ["SEQN", "RHD143", "RHQ160"])

# Calculate the number of pregnancies based on the conditions
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = (
    pregnancy_df["RHQ160"].fillna(0) +
    pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0) +
    pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)
)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]  # Keep only relevant columns

# Load Family History as an independent column
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI data from CombinedBMI.csv and calculate Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else combined_bmi_df["BMXWAIST"].mean() / combined_bmi_df["BMXHIP"].mean(),
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)  # Rename BMXBMI to BMI
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]  

# Load Blood Pressure data and calculate the average of BPXDI1 to BPXDI4 per SEQN
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
blood_pressure_df["BloodPressure"] = blood_pressure_df[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].mean(axis=1, skipna=True)
blood_pressure_df = blood_pressure_df[["SEQN", "BloodPressure"]]  # Keep only SEQN and BloodPressure

# Load Insulin and Glucose data and filter rows based on fasting hours
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU", "PHAFSTHR"])
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: row["LBXGLU"] if pd.notna(row["PHAFSTHR"]) and row["PHAFSTHR"] >= 8 else 0,
    axis=1
)
insulin_glucose_df = insulin_glucose_df[["SEQN", "Fasting Glucose"]]

# Load Outcome data
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df = outcome_df[["SEQN", "Outcome"]]

# Initialize a list to store filtered data frames
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]

# Merge all filtered data frames on SEQN
result_df = filtered_data_frames[0]  # Start with the first DataFrame
for df in filtered_data_frames[1:]:
    result_df = result_df.merge(df, on="SEQN", how="outer")  # Outer join to keep all SEQN rows

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[2]:


import pandas as pd

# Updated file paths based on provided locations
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

# Load reproductive health data from ReproductiveHealth.csv and calculate pregnancies
reproductive_health_df = load_csv(files["ReproductiveHealth"], ["SEQN", "RHD143", "RHQ160"])
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = (
    pregnancy_df["RHQ160"].fillna(0) +
    pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0) +
    pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)
)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data without averaging
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])

# Concatenate values for duplicate SEQN entries without taking the mean
def merge_duplicates(df, id_col="SEQN"):
    numeric_cols = df.select_dtypes(include="number").columns.difference([id_col])
    non_numeric_cols = df.select_dtypes(exclude="number").columns.difference([id_col])

    # Aggregating numeric columns by keeping non-null unique values as lists
    df_agg = df.groupby(id_col, as_index=False).agg(
        {col: lambda x: list(x.dropna().unique()) for col in numeric_cols} |
        {col: lambda x: ' | '.join(x.dropna().astype(str).unique()) for col in non_numeric_cols}
    )
    return df_agg

# Apply merging to each dataframe with potential duplicates
pregnancy_df = merge_duplicates(pregnancy_df)
family_history_df = merge_duplicates(family_history_df)
combined_bmi_df = merge_duplicates(combined_bmi_df)
blood_pressure_df = merge_duplicates(blood_pressure_df)

# Process Insulin and Glucose data without averaging
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU", "PHAFSTHR"])
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: row["LBXGLU"] if pd.notna(row["PHAFSTHR"]) and row["PHAFSTHR"] >= 8 else 0,
    axis=1
)
insulin_glucose_df = insulin_glucose_df[["SEQN", "Fasting Glucose"]]
insulin_glucose_df = merge_duplicates(insulin_glucose_df)

# Load Outcome data without averaging
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df = outcome_df[["SEQN", "Outcome"]]
outcome_df = merge_duplicates(outcome_df)

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")  # Merge based on SEQN without duplicates

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[3]:


import pandas as pd

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

# Load reproductive health data from ReproductiveHealth.csv and calculate pregnancies
reproductive_health_df = load_csv(files["ReproductiveHealth"], ["SEQN", "RHD143", "RHQ160"])
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = (
    pregnancy_df["RHQ160"].fillna(0) +
    pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0) +
    pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)
)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data without averaging
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])

# Updated function to handle duplicates by keeping the maximum value across columns for each SEQN
def merge_duplicates(df, id_col="SEQN"):
    numeric_cols = df.select_dtypes(include="number").columns.difference([id_col])
    non_numeric_cols = df.select_dtypes(exclude="number").columns.difference([id_col])

    # Aggregating numeric columns by max to retain the highest value and keeping the first non-numeric value if duplicated
    df_agg = df.groupby(id_col, as_index=False).agg(
        {col: 'max' for col in numeric_cols} | 
        {col: 'first' for col in non_numeric_cols}
    )
    return df_agg

# Apply merging to each dataframe with potential duplicates
pregnancy_df = merge_duplicates(pregnancy_df)
family_history_df = merge_duplicates(family_history_df)
combined_bmi_df = merge_duplicates(combined_bmi_df)
blood_pressure_df = merge_duplicates(blood_pressure_df)

# Process Insulin and Glucose data without averaging
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU", "PHAFSTHR"])
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: row["LBXGLU"] if pd.notna(row["PHAFSTHR"]) and row["PHAFSTHR"] >= 8 else 0,
    axis=1
)
insulin_glucose_df = insulin_glucose_df[["SEQN", "Fasting Glucose"]]
insulin_glucose_df = merge_duplicates(insulin_glucose_df)

# Load Outcome data without averaging
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df = outcome_df[["SEQN", "Outcome"]]
outcome_df = merge_duplicates(outcome_df)

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")  # Merge based on SEQN without duplicates

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[4]:


import pandas as pd

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

# Load reproductive health data from ReproductiveHealth.csv and calculate pregnancies
reproductive_health_df = load_csv(files["ReproductiveHealth"], ["SEQN", "RHD143", "RHQ160"])
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")

# Apply the rules for calculating the Number of Pregnancies
pregnancy_df["Number of Pregnancies"] = pregnancy_df["RHQ160"].fillna(0)

# Only add to RHQ160 if RIDEXPRG is 1 and RHD143 is 1
pregnancy_df["Number of Pregnancies"] += pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0)
pregnancy_df["Number of Pregnancies"] += pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)

# Drop columns RIDEXPRG and RHD143 as they are no longer needed
pregnancy_df = pregnancy_df.drop(columns=["RIDEXPRG", "RHD143"])
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data without averaging
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])

# Updated function to handle duplicates by keeping the maximum value across columns for each SEQN
def merge_duplicates(df, id_col="SEQN"):
    numeric_cols = df.select_dtypes(include="number").columns.difference([id_col])
    non_numeric_cols = df.select_dtypes(exclude="number").columns.difference([id_col])

    # Aggregating numeric columns by max to retain the highest value and keeping the first non-numeric value if duplicated
    df_agg = df.groupby(id_col, as_index=False).agg(
        {col: 'max' for col in numeric_cols} | 
        {col: 'first' for col in non_numeric_cols}
    )
    return df_agg

# Apply merging to each dataframe with potential duplicates
pregnancy_df = merge_duplicates(pregnancy_df)
family_history_df = merge_duplicates(family_history_df)
combined_bmi_df = merge_duplicates(combined_bmi_df)
blood_pressure_df = merge_duplicates(blood_pressure_df)

# Process Insulin and Glucose data without averaging
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU", "PHAFSTHR"])
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: row["LBXGLU"] if pd.notna(row["PHAFSTHR"]) and row["PHAFSTHR"] >= 8 else 0,
    axis=1
)
insulin_glucose_df = insulin_glucose_df[["SEQN", "Fasting Glucose"]]
insulin_glucose_df = merge_duplicates(insulin_glucose_df)

# Load Outcome data without averaging
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df = outcome_df[["SEQN", "Outcome"]]
outcome_df = merge_duplicates(outcome_df)

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")  # Merge based on SEQN without duplicates

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[2]:


import pandas as pd

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

# Clean the RHQ160 column: Set values 77, 99, and NaN to 0
reproductive_health_df["RHQ160"] = reproductive_health_df["RHQ160"].replace([42, 35, 55, 23, 20, 22, 32, 26, 23, 32, 33, 25, 77, 99], 0).fillna(0)

# Calculate pregnancies based on specified conditions
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = (
    pregnancy_df["RHQ160"] +
    pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0) +
    pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)
)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]  # Drop RIDEXPRG and RHD143 columns

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])

# Updated function to handle duplicates by keeping the maximum value across columns for each SEQN
def merge_duplicates(df, id_col="SEQN"):
    numeric_cols = df.select_dtypes(include="number").columns.difference([id_col])
    non_numeric_cols = df.select_dtypes(exclude="number").columns.difference([id_col])

    # Aggregating numeric columns by max to retain the highest value and keeping the first non-numeric value if duplicated
    df_agg = df.groupby(id_col, as_index=False).agg(
        {col: 'max' for col in numeric_cols} | 
        {col: 'first' for col in non_numeric_cols}
    )
    return df_agg

# Apply merging to each dataframe with potential duplicates
pregnancy_df = merge_duplicates(pregnancy_df)
family_history_df = merge_duplicates(family_history_df)
combined_bmi_df = merge_duplicates(combined_bmi_df)
blood_pressure_df = merge_duplicates(blood_pressure_df)

# Process Insulin and Glucose data without averaging
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU", "PHAFSTHR"])
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df.apply(
    lambda row: row["LBXGLU"] if pd.notna(row["PHAFSTHR"]) and row["PHAFSTHR"] >= 8 else 0,
    axis=1
)
insulin_glucose_df = insulin_glucose_df[["SEQN", "Fasting Glucose"]]
insulin_glucose_df = merge_duplicates(insulin_glucose_df)

# Load Outcome data
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df = outcome_df[["SEQN", "Outcome"]]
outcome_df = merge_duplicates(outcome_df)

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")  # Merge based on SEQN without duplicates

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[4]:


import pandas as pd

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

# Clean the RHQ160 column: Set values 77, 99, and NaN to 0
reproductive_health_df["RHQ160"] = reproductive_health_df["RHQ160"].replace([77, 99], 0).fillna(0)

# Calculate pregnancies based on specified conditions
pregnancy_df = pd.merge(demographics_df, reproductive_health_df, on="SEQN", how="left")
pregnancy_df["Number of Pregnancies"] = (
    pregnancy_df["RHQ160"] +
    pregnancy_df["RIDEXPRG"].apply(lambda x: 1 if x == 1 else 0) +
    pregnancy_df["RHD143"].apply(lambda x: 1 if x == 1 else 0)
)
pregnancy_df = pregnancy_df[["SEQN", "Age", "Number of Pregnancies"]]  # Drop RIDEXPRG and RHD143 columns

# Load Family History data
family_history_df = load_csv(files["FamilyHistory"], ["SEQN", "FamilyHistory"])

# Load BMI and Waist to Hip Ratio
combined_bmi_df = load_csv(files["CombinedBMI"], ["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI"])
combined_bmi_df["Waist to Hip Ratio"] = combined_bmi_df.apply(
    lambda row: row["BMXWAIST"] / row["BMXHIP"]
    if pd.notnull(row["BMXWAIST"]) and pd.notnull(row["BMXHIP"]) and row["BMXHIP"] != 0
    else None,
    axis=1
)
combined_bmi_df.rename(columns={"BMXBMI": "BMI"}, inplace=True)
combined_bmi_df = combined_bmi_df[["SEQN", "Waist to Hip Ratio", "BMI"]]

# Load Blood Pressure data and calculate the highest BP value
blood_pressure_df = load_csv(files["CombinedBloodPressure"], ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
blood_pressure_df["BloodPressure"] = blood_pressure_df[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].max(axis=1)
blood_pressure_df = blood_pressure_df[["SEQN", "BloodPressure"]]  # Drop original BPXDI columns after calculating

# Load Insulin and Glucose data, focusing only on LBXGLU for Fasting Glucose
insulin_glucose_df = load_csv(files["InsulinGlucose"], ["SEQN", "LBXGLU"])
insulin_glucose_df.rename(columns={"LBXGLU": "Fasting Glucose"}, inplace=True)
insulin_glucose_df["Fasting Glucose"] = insulin_glucose_df["Fasting Glucose"].fillna(0)  # Set missing to 0

# Load Outcome data and replace NaNs with 0
outcome_df = load_csv(files["Outcome"], ["SEQN", "DIQ010"])
outcome_df["Outcome"] = outcome_df["DIQ010"].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
outcome_df["Outcome"] = outcome_df["Outcome"].fillna(0)  # Set missing to 0
outcome_df = outcome_df[["SEQN", "Outcome"]]

# Merge all dataframes on SEQN
filtered_data_frames = [pregnancy_df, family_history_df, combined_bmi_df, blood_pressure_df, insulin_glucose_df, outcome_df]
result_df = demographics_df[["SEQN"]]  # Start with the SEQN column from demographics

for df in filtered_data_frames:
    result_df = result_df.merge(df, on="SEQN", how="left")  # Merge based on SEQN without duplicates

# Save the final output as DiabetesNHANESUpdated.csv
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")


# In[5]:


import pandas as pd

# Load the merged dataset
file_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated.csv"
df = pd.read_csv(file_path)

# Function to handle duplicates by keeping the highest value for numeric columns and the first occurrence for others
def keep_highest_per_row(df, id_col="SEQN"):
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.difference([id_col])
    non_numeric_cols = df.select_dtypes(exclude="number").columns.difference([id_col])
    
    # If there are numeric columns, group by SEQN and get the max
    if not numeric_cols.empty:
        df_numeric = df.groupby(id_col, as_index=False)[numeric_cols].max()
    else:
        df_numeric = df[[id_col]]  # Preserve SEQN column even if no numeric columns

    # If there are non-numeric columns, keep the first occurrence for each SEQN
    if not non_numeric_cols.empty:
        df_non_numeric = df.groupby(id_col, as_index=False)[non_numeric_cols].first()
    else:
        df_non_numeric = pd.DataFrame()  # Create empty if no non-numeric columns
    
    # Merge the results
    if not df_non_numeric.empty:
        df_cleaned = pd.merge(df_numeric, df_non_numeric, on=id_col, how="left")
    else:
        df_cleaned = df_numeric  # Use numeric-only data if no non-numeric columns
    
    return df_cleaned

# Apply the function
result_df = keep_highest_per_row(df)

# Save the cleaned data
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\DiabetesNHANESUpdated_Cleaned.csv"
result_df.to_csv(output_path, index=False)

print(f"Duplicates removed and data cleaned. File saved to {output_path}")


# In[ ]:




