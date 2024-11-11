import pandas as pd

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalCombinedDiabetesDataWithAge.csv"
df = pd.read_csv(file_path)

# Check if DIQ010 column exists
if 'DIQ010' in df.columns:
    # Apply transformations based on specified rules
    df['DIQ010'] = df['DIQ010'].apply(lambda x: 1 if x == 3 else (0 if x == 2 else (1 if x == 1 else 0)))
    
    # Rename the column to Outcome
    df.rename(columns={'DIQ010': 'Outcome'}, inplace=True)
    
    # Move the Outcome column to the last position
    outcome_column = df.pop('Outcome')
    df['Outcome'] = outcome_column
else:
    print("DIQ010 column not found in the dataset.")

# Save the modified DataFrame to a new CSV file
output_path = r"C:\Users\Hazel\Downloads\DiabetesDatasets\FinalCombinedDiabetesDataWithOutcome.csv"
df.to_csv(output_path, index=False)

print(f"Updated data with Outcome column saved to {output_path}")
