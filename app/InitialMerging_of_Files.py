#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os

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

# Initialize variables for total size, rows, and columns
total_size = 0
total_rows = 0
total_columns = 0

# Iterate over files to calculate size and load data
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


# In[10]:


import pandas as pd
import os

# List of file paths and file names
files = [
    ("questionnairedata.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\questionnaire_merged_data\questionnairedata.csv"),
    ("combined_blood_pressure_data.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_blood_pressure_data.csv"),
    ("combined_bmi_data.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_bmi_data.csv"),
    ("combined_health_data.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_health_data.csv"),
    ("combined_skin_thickness_data.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\examination_data\examination_data\examination_data\required_examination_csv_files\combined_skin_thickness_data.csv"),
    ("ReproductiveHealth_combined.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\ReproductiveHealth_Doc\ReproductiveHealth\ReproductiveHealth_combined.csv"),
    ("DiabetesData_updated.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\diabetessas\DiabetesData\DataFiles\DiabetesData_updated.csv"),
    ("OutcomeDIQ010.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\diabetessas\DiabetesData\DataFiles\OutcomeDIQ010.csv"),
    ("insulindata.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\insulinfiles\insulindata.csv"),
    ("demographics.csv", r"C:\Users\Hazel\Downloads\DiabetesDatasets\archive\DemographicsSAS files\demographics.csv")
]

# Initialize variables for total size, rows, and columns
total_size = 0
total_rows = 0
total_columns = 0

# Iterate over files to calculate size and load data
for file_name, file_path in files:
    # Calculate file size in MB with comma formatting
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    total_size += file_size
    
    try:
        # Load data and handle parsing errors
        data = pd.read_csv(file_path, on_bad_lines='skip')
        file_rows, file_columns = data.shape
        total_rows += file_rows
        total_columns = max(total_columns, file_columns)  # Ensures the highest column count
        
        print(f"File: {file_name}")
        print(f"Size: {file_size:,.2f} MB")  # Only MB size with comma format
        print(f"Rows: {file_rows:,} Columns: {file_columns:,}\n")  # Rows and Columns with comma formatting
        
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Summary of combined data
print("Total Combined Data:")
print(f"Total Size for All Files: {total_size:,.2f} MB")  # Total size with comma format
print(f"Total Rows: {total_rows:,}")
print(f"Total Columns: {total_columns:,}")



# In[ ]:




