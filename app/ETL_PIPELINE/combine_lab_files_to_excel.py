import pandas as pd
import os

# Folder where .XPT files are located
xpt_folder = 'NHANES_Laboratory_Data'  # Adjust the path as needed

# List of files containing at least one target variable
target_files = [
    "GLU_D.XPT", "GLU_E.XPT", "GLU_F.XPT", "GLU_G.XPT", "GLU_H.XPT", 
    "GLU_I.XPT", "GLU_J.XPT", "GLU_L.XPT", "INS_H.XPT", "INS_I.XPT", 
    "INS_J.XPT", "INS_L.XPT", "L10AM_B.XPT", "L10AM_C.XPT", 
    "LAB10AM.XPT", "P_GLU.XPT"
]

# List to hold DataFrames for each file
dfs = []

# Process each file
for file_name in target_files:
    xpt_path = os.path.join(xpt_folder, file_name)
    
    # Load each file with Pandas
    try:
        pd_df = pd.read_sas(xpt_path, format='xport', encoding='utf-8')
        
        # Append the DataFrame to the list
        dfs.append(pd_df)
        print(f"Processed {file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Concatenate all DataFrames, aligning columns by name and filling missing columns with NaN
combined_df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)

# Fill NaN values with an empty string for a cleaner look
combined_df = combined_df.fillna("")

# Save the combined DataFrame as an Excel file
output_excel = 'combined_nhanes_data.xlsx'
combined_df.to_excel(output_excel, index=False)
print(f"All files have been merged and saved to {output_excel} in Excel format.")