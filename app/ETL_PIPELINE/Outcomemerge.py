import pandas as pd

# File paths based on provided locations
files = {
    "InsulinGlucose": r"C:\Users\Hazel\Downloads\DiabetesDatasets\InsulinGlucose.csv",
    "GenderAge": r"C:\Users\Hazel\Downloads\DiabetesDatasets\GenderAge.csv",
    "ReproductiveHealth": r"C:\Users\Hazel\Downloads\DiabetesDatasets\reproductivehealthPregnancies.csv",
    "BMISkinThicknessBP": r"C:\Users\Hazel\Downloads\DiabetesDatasets\combined_skin_thickness_data.csv",
    "Outcome": r"C:\Users\Hazel\Downloads\DiabetesDatasets\Outcome.csv",
    "CombinedBloodPressure": r"C:\Users\Hazel\Downloads\DiabetesDatasets\combined_blood_pressure_data.csv",
    "CombinedSkinThickness": r"C:\Users\Hazel\Downloads\DiabetesDatasets\combined_skin_thickness_data.csv",
    "CombinedBMI": r"C:\Users\Hazel\Downloads\DiabetesDatasets\combined_bmi_data.csv"
}

# Columns to extract from each file, based on the location of each variable
columns_to_extract = {
    "GenderAge": ["SEQN", "RIDEXPRG", "RIDAGEEX"],
    "ReproductiveHealth": ["SEQN", "RHD143", "RHQ160"],
    "InsulinGlucose": ["SEQN", "LBXGLU", "LBXIN"],
    "CombinedBloodPressure": ["SEQN", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "CombinedSkinThickness": ["SEQN", "BMXTRI"],
    "BMISkinThicknessBP": ["SEQN", "BMXBMI"],
    "Outcome": ["SEQN", "DIQ175A", "DIQ010"],
    "CombinedBMI": ["SEQN", "BMXWAIST"]
}

# Load SEQN values from GenderAge.csv to filter data
gender_age_df = pd.read_csv(files["GenderAge"], usecols=["SEQN"])
seqn_values = gender_age_df["SEQN"].tolist()  # List of SEQN values to match

# Initialize a list to store filtered data frames
filtered_data_frames = []

# Loop through each file and extract matching rows with required columns
for file_key, column_names in columns_to_extract.items():
    file_path = files[file_key]
    try:
        # Load the CSV file with SEQN and specified columns
        df = pd.read_csv(file_path, usecols=column_names, low_memory=False)
        # Filter rows to include only those with matching SEQN values
        df_filtered = df[df["SEQN"].isin(seqn_values)]
        filtered_data_frames.append(df_filtered)
        print(f"Successfully extracted columns from {file_path}")
    except ValueError:
        print(f"Warning: Some columns in {column_names} were not found in {file_path}")

# Merge all filtered data frames on SEQN
result_df = filtered_data_frames[0]  # Start with the first DataFrame
for df in filtered_data_frames[1:]:
    result_df = result_df.merge(df, on="SEQN", how="outer")  # Outer join to keep all SEQN rows

# Output to a single CSV file
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\CombinedDiabetesData.csv"
result_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")
