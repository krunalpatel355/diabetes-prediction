import os
import pandas as pd
import pyreadstat

# Define the folder where the .XPT files are located
xpt_folder = 'NHANES_Laboratory_Data'  # Adjust the folder path as needed

# List of target variables to search for
target_variables = {"LBXGLU", "LBDGLUSI", "LBXIN", "LBDINSI", "LBDINLC"}

# List to store matching file names
matching_files = []

# Loop through each file and inspect columns for target variables
for file_name in os.listdir(xpt_folder):
    if file_name.endswith('.XPT'):
        xpt_path = os.path.join(xpt_folder, file_name)
        
        try:
            # Load only the metadata to avoid loading the entire data into memory
            _, meta = pyreadstat.read_xport(xpt_path)
            
            # Check if any target variable is in the column names
            columns = set(meta.column_names)
            if target_variables.intersection(columns):  # Check if any target variable exists in the file
                matching_files.append(file_name)
                print(f"Found target variable(s) in {file_name}")
        
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# Print the matching files
print("\nFiles containing at least one of the target variables:")
for file in matching_files:
    print(file)


