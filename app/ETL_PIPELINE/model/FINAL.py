# combined_health_data_processing.py

import pandas as pd

# Step 1: Define the list of columns related to each area
columns_of_interest = [
    # Blood Pressure
    'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3', 'BPXSY4', 'BPXDI4',
    'BPXCHR', 'BPAARM', 'BPACSZ', 'BPXPLS', 'BPXPULS', 'BPXPTY', 'BPXML1', 'BPXSAR', 'BPXDAR',

    # BMI
    'BMXWT', 'BMIWT', 'BMXBMI', 'BMDBMIC', 'BMXHT', 'BMIHT', 'BMXWAIST', 'BMIWAIST', 'BMXHIP', 'BMIHIP',

    # Gender
    'RIAGENDR',

    # Skin Thickness
    'BMXTHICR', 'BMITHICR', 'BMXTRI', 'BMITRI', 'BMXSUB', 'BMISUB',

    # Primary Key
    'SEQN'
]

# List of filenames
files = [
    "BMX.csv", "BMX_B.csv", "BMX_C.csv", "BMX_D.csv", "BMX_E.csv", "BMX_F.csv", "BMX_G.csv",
    "BMX_H.csv", "BMX_I.csv", "BMX_J.csv", "BMX_L.csv", "BPX.csv", "BPX_B.csv", "BPX_C.csv",
    "BPX_D.csv", "BPX_E.csv", "BPX_F.csv", "BPX_G.csv", "BPX_H.csv", "BPX_I.csv", "BPX_J.csv",
    "P_BMX.csv"
]

# Step 2: Initialize an empty list to hold each DataFrame
filtered_dfs = []

# Step 3: Loop through each file, filter for relevant columns, and append to list
for file in files:
    try:
        # Read the file and filter the columns
        df = pd.read_csv(file, usecols=lambda col: col in columns_of_interest)
        if not df.empty:
            filtered_dfs.append(df)
        else:
            print(f"File {file} has no relevant columns.")
    except ValueError as e:
        print(f"Error reading {file}: {e}")

# Step 4: Check if any DataFrames were loaded before attempting to concatenate
if filtered_dfs:
    combined_df = pd.concat(filtered_dfs, ignore_index=True).drop_duplicates()
    combined_df.to_csv('combined_health_data.csv', index=False)
    print("Combined data saved as 'combined_health_data.csv'.")
else:
    print("No data to combine. Please check the column names and files.")


# Additional tasks to log dropped columns from each file

# Dictionary to store dropped columns for each CSV file
dropped_columns_dict = {}

# Loop through each file and process dropped columns
for file in files:
    # Load the data
    df = pd.read_csv(file)
    
    # Find the columns that are not in the columns_of_interest list
    dropped_columns = df.columns.difference(columns_of_interest)
    
    # Store dropped columns in dictionary
    dropped_columns_dict[file] = dropped_columns.tolist()

# Print the dropped columns for each file
for file, dropped_columns in dropped_columns_dict.items():
    print(f"### {file}:")
    print(f"- **Dropped Columns**: {', '.join(dropped_columns)}\n")

# Blood Pressure Data Extraction

bp_columns = [
    'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3', 'BPXSY4', 'BPXDI4',
    'BPXCHR', 'BPAARM', 'BPACSZ', 'BPXPLS', 'BPXPULS', 'BPXPTY', 'BPXML1', 'BPXSAR', 'BPXDAR', 'SEQN'
]

all_data = []

for file in files:
    df = pd.read_csv(file)
    if all(col in df.columns for col in bp_columns):
        df_filtered = df[bp_columns]
        all_data.append(df_filtered)

combined_data = pd.concat(all_data, ignore_index=True)
combined_data.to_csv("combined_blood_pressure_data.csv", index=False)
print("Combined CSV file created: combined_blood_pressure_data.csv")


# BMI Data Extraction

bmi_columns = [
    'BMXWT', 'BMIWT', 'BMXBMI', 'BMDBMIC', 'BMXHT', 'BMIHT', 'BMXWAIST', 'BMIWAIST', 'BMXHIP', 'BMIHIP', 'SEQN'
]

all_data = []

for file in files:
    df = pd.read_csv(file)
    if all(col in df.columns for col in bmi_columns):
        df_filtered = df[bmi_columns]
        all_data.append(df_filtered)

combined_data = pd.concat(all_data, ignore_index=True)
combined_data.to_csv("combined_bmi_data.csv", index=False)
print("Combined CSV file created: combined_bmi_data.csv")


# Skin Thickness Data Extraction

skin_thickness_columns = [
    'BMXTHICR', 'BMITHICR', 'BMXTRI', 'BMITRI', 'BMXSUB', 'BMISUB', 'SEQN'
]

all_data = []

for file in files:
    df = pd.read_csv(file)
    if all(col in df.columns for col in skin_thickness_columns):
        df_filtered = df[skin_thickness_columns]
        all_data.append(df_filtered)

combined_data = pd.concat(all_data, ignore_index=True)
combined_data.to_csv("combined_skin_thickness_data.csv", index=False)
print("Combined CSV file created: combined_skin_thickness_data.csv")
