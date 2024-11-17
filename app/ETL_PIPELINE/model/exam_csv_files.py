import os
import pandas as pd
import xport

# Define the folder paths
input_folder = r"C:\Users\sneha\OneDrive\Desktop\examination_data\required_examination_xpt_files"
output_folder = r"C:\Users\sneha\OneDrive\Desktop\examination_data\required_examination_csv_files"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each .XPT file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".XPT"):
        xpt_file_path = os.path.join(input_folder, file_name)
        csv_file_path = os.path.join(output_folder, file_name.replace(".XPT", ".csv"))

        # Read the .XPT file and convert to DataFrame
        with open(xpt_file_path, 'rb') as xpt_file:
            df = xport.to_dataframe(xpt_file)

        # Write the DataFrame to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"Converted {file_name} to CSV at {csv_file_path}")
