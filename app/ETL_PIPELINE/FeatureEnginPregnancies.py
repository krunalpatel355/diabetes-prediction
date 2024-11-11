import pandas as pd

# Load the combined dataset
file_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\CombinedDiabetesData.csv"
df = pd.read_csv(file_path)

# Check if columns exist before performing operations
if 'RIDEXPRG' in df.columns and 'RHD143' in df.columns and 'RHQ160' in df.columns:
    # Update RHQ160 based on the conditions
    df.loc[df['RIDEXPRG'] == 1, 'RHQ160'] += 1
    df.loc[df['RHD143'] == 1, 'RHQ160'] += 1

    # Drop the RIDEXPRG and RHD143 columns
    df.drop(columns=['RIDEXPRG', 'RHD143'], inplace=True)
else:
    print("Required columns (RIDEXPRG, RHD143, RHQ160) are not present in the dataset.")

# Save the modified DataFrame to a new CSV file
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\ModifiedCombinedDiabetesData.csv"
df.to_csv(output_path, index=False)

print(f"Modified data saved to {output_path}")
