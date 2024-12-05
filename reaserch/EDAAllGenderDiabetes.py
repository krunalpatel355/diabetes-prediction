#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix


# In[4]:


from IPython.core.display import display, HTML

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Features and Descriptions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h2 {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        table {
            border-collapse: collapse;
            width: 50%; /* Reduced table width */
            margin: 10px 20px; /* Left indent */
            font-size: 0.9em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 5px;
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
        }
    </style>
</head>
<body>
    <h2>Features and Their Descriptions</h2>
    <ul>
        <li><b>Pregnancies</b> - Number of times the person conceived.</li>
        <li><b>Glucose</b> - 2 hours of fasting.</li>
        <li><b>BloodPressure</b> - Diastolic blood pressure (mm Hg).</li>
        <li><b>SkinThickness</b> - Measurement of Triceps Skinfold Thickness (mm):</li>
        <li><b>Insulin</b> - Fasting insulin levels (µU/mL). Normal range: 2–25 µU/mL.</li>
        <li><b>BMI</b> - Body Mass Index (weight in kg / height in meters²):</li>
        <li><b>Family History</b> - 0 for no family history in diabetes, 1 if there is family history.</li>
        <li><b>Age</b> - In years, ranging from 1 to 85.</li>
        <li><b>Outcome</b> - 0 for not diabetic and 1 if diabetic (Target Variable).</li>
    </ul>
</body>
</html>
"""

# Display the HTML in Jupyter
display(HTML(html_content))


# Initial merging of data with size:

# In[3]:


from IPython.core.display import display, HTML

# Updated file paths and their descriptive names
file_paths = {
    "Diabetes Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\combined_diabetes_data.csv",
    "Skin Thickness Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\combined_skin_thickness_data.csv",
    "Reproductive Health Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\reproductivehealth.csv",
    "BMI Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\BMI_combined.csv",
    "Insulin Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\insulindata.csv",
    "Blood Pressure Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\BPcombinedNew.csv",
    "Demographics Data": r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetesdatasets\demographics.csv"
}

# Initialize total metrics
total_size = 0
total_rows = 0
total_columns = 0

# Store results for HTML rendering
file_stats = []

for name, file_path in file_paths.items():
    try:
        # Calculate file size in MB
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        total_size += file_size

        # Load data and handle parsing errors
        data = pd.read_csv(file_path, on_bad_lines='skip')
        file_rows, file_columns = data.shape
        total_rows += file_rows
        total_columns = max(total_columns, file_columns)  # Ensures the highest column count

        # Append file statistics
        file_stats.append({
            "Name": name,
            "Size (MB)": f"{file_size:.2f}",
            "Rows": f"{file_rows:,}",  # Format rows with commas
            "Columns": file_columns
        })
    except Exception as e:
        file_stats.append({
            "Name": name,
            "Size (MB)": "N/A",
            "Rows": "Error",
            "Columns": "Error",
            "Error": str(e)
        })

# HTML report generation
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>File Statistics Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h2 {
            text-align: left; /* Left-align the title */
            margin-left: 15%;
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 60%; /* Reduced table width */
            margin: 20px auto;
            font-size: 0.9em; /* Adjust font size */
        }
        th, td {
            border: 1px solid #ddd;
            padding: 4px; /* Compact padding for smaller spaces */
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9; /* Alternating row colors */
        }
        .summary {
            margin: 20px auto;
            width: 60%;
            font-size: 0.9em;
            text-align: left; /* Left-align the summary */
        }
    </style>
</head>
<body>
    <h2>File Statistics Report</h2>
    <table>
        <tr>
            <th>Name</th>
            <th>Size (MB)</th>
            <th>Rows</th>
            <th>Columns</th>
        </tr>
"""

# Add file details to the HTML table
for stat in file_stats:
    html_content += f"""
        <tr>
            <td>{stat['Name']}</td>
            <td>{stat['Size (MB)']}</td>
            <td>{stat['Rows']}</td>
            <td>{stat['Columns']}</td>
        </tr>
    """

# Add summary metrics
html_content += f"""
    </table>
    <div class="summary">
        <p><b>Total Combined Data:</b></p>
        <ul>
            <li><b>Total Size for All Files:</b> {total_size:.2f} MB</li>
            <li><b>Total Rows:</b> {total_rows:,}</li> <!-- Format total rows with commas -->
            <li><b>Total Columns (Max):</b> {total_columns}</li>
        </ul>
    </div>
</body>
</html>
"""

# Display the HTML in Jupyter Notebook
display(HTML(html_content))


# ***Data Size with Identified Features***

# In[4]:


data = pd.read_csv(r"C:\Users\Hazel\Downloads\Multiple disease prediction\diabetes.csv")
data.info()


# In[5]:


# Recalculate the descriptive statistics
numeric_data = data.select_dtypes(include=['float', 'int'])
formatted_describe = numeric_data.describe().round(2)

# Render the corrected table as HTML
html_output = formatted_describe.to_html(classes='table table-bordered', index=True)

# Apply the CSS for proper formatting
styled_html = f"""
<style>
    table {{
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        word-wrap: break-word;
        padding: 8px;
    }}
    th {{
        background-color: #f2f2f2;
    }}
</style>
{html_output}
"""

# Display the final HTML table
from IPython.core.display import display, HTML
display(HTML(styled_html))


# Checking for missing values to avoid error when training the model
# 

# In[6]:


data.info()
#As you could see here, BP has less values than the others, it is because we have blank rows for BP


# Checking how many rows have complete values versus those that are not <br>
# Since we have identified earlier that BP has missing values, this code counts rows with no missing values and rows with at least one missing entry.<br>

# In[7]:


data.isna().sum()
#BP has 20,623 missing values, we are deciding to keep this feature because it is important in predicting Diabetes


# This heatmap visualizes missing values in the dataset.
# The pure black cells indicate No non-missing values, while the lighter cells indicate missing values and it is showing in the BP section

# ***Data Visualization***

# In[8]:


# Generate a heatmap with a different color map
plt.figure(figsize=(6, 4))  # Adjust the size if needed
sns.heatmap(data.isnull(), cmap='Reds', cbar=False, cbar_kws={'label': 'Missing Values'}, annot=False, fmt='')

# Add labels and title
plt.title('Missing Values Heatmap', fontsize=16)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Rows', fontsize=12)

plt.show()


# In[9]:


# Count of rows with all values intact (no NaNs)
rows_no_nan = data.dropna().shape[0]

# Count of rows with at least one NaN value
rows_with_nan = data.shape[0] - rows_no_nan

# Display results
print("Rows with all values intact (no NaN values):", rows_no_nan)
print("Rows with at least one NaN value:", rows_with_nan)


# In[10]:


plt.figure(figsize=(8, 4))
sns.countplot(x='Outcome', data=data, palette='Set2', hue=None, legend=False)
plt.show()
#This graph shows the distribution of the people with diabetes versus those who are healthy


# In[11]:


corrmat = data.corr()

# Sort correlation with respect to 'Outcome' (replace 'Outcome' with your target column name)
top_corr_features = corrmat['Outcome'].sort_values(ascending=False).index

# Plotting the heatmap
plt.figure(figsize=(10, 10))
g = sns.heatmap(
    data[top_corr_features].corr(), 
    annot=True, 
    cmap="coolwarm",  # Updated heatmap color palette
    fmt=".2f"
)
plt.title("Correlation Heatmap with Outcome")
plt.show()


# In[5]:


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Extract correlations with 'Outcome' column (assuming 'Outcome' represents diabetes diagnosis)
outcome_correlation = correlation_matrix['Outcome'].sort_values(ascending=False)

# Display the correlations
print("Correlation of features with 'Outcome':")
print(outcome_correlation)

# Plot the correlations as a bar chart
plt.figure(figsize=(10, 6))
outcome_correlation.drop('Outcome').plot(kind='bar', color='teal')
plt.title("Correlation of Features with Outcome (Diabetes)", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Correlation Coefficient", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# **Observing Outliers**

# In[15]:


plt.figure(figsize=(8, 8))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BMI', 'Insulin','Age']):
    plt.subplot(3, 2, i + 1)  # Adjusted to a 3x2 grid for six plots
    sns.boxplot(x=col, data=data)
    plt.title({col})  # adds titles to each plot

plt.tight_layout() 
plt.show()
# BMI: High BMI outliers may represent individuals with obesity, relevant to health-related analyses.
# WaistToHipRatio: High waist-to-hip ratio outliers may indicate central obesity, also important for understanding health risks.


# Observations:
# 1. Pregnancies
# Most of the data points are concentrated in 0 with max data points seen at 12 pregnancies
# There few outliers for the number of pregnancies, indicates a small portion of individuals who have significantly higher pregnancies as the dataset is a combination of male and female gender.
# 
# 2. Glucose
# The tall line at approximately 100 suggests that a large proportion of glucose values are clustered near this value.
# This aligns with normal fasting glucose levels for healthy individuals, typically ranging from 70 to 100 mg/dL.
# Extreme outliers are observed above 300, indicating potential cases of diabetes.
# 
# 
# 3. BMI
# Most BMI values (20–40) represent healthy to obese categories.
# Outliers above 40 indicate severe obesity, often linked to chronic diseases like diabetes.
# Low BMI values likely reflect underweight individuals, particularly among children or the elderly.
# 
# 
# 4. Insulin
# Most insulin values are tightly packed within the normal range.
# A large number of outliers are observed above 200, which may belong to individuals who are classified as diabetics.
# 
# 
# 5. Age
# The dataset primarily consists of middle-aged individuals, with a smaller proportion of younger and elderly individuals. Outliers below 20 and above 70 represent specific age demographics (children and elderly) that may influence health trends, such as diabetes risk.

# In[16]:


sns.pairplot(data=data, hue='Outcome')
plt.show()
# Orange means diabetic, blue not diabetic

1. Age
Observation: Older individuals tend to have a higher proportion of diabetes (orange points).
Insights: The risk of diabetes increases with age, likely due to age-related factors such as declining insulin sensitivity or chronic lifestyle habits.
2. Pregnancies
Observation: Higher numbers of pregnancies show an association with diabetes cases. However, the correlation is not definitive for all cases.
Insights: Gestational diabetes might play a role in women with multiple pregnancies being more susceptible to diabetes later in life.
3. Glucose
Observation: Higher glucose levels are strongly associated with diabetic individuals (orange points). Non-diabetic individuals (blue points) cluster around lower glucose levels.
Insights: Glucose is a key determinant for diabetes prediction. The boundary between normal and diabetic glucose levels is evident in the data.
4. Blood Pressure
Observation: The relationship between blood pressure and diabetes is less clear. Both diabetic and non-diabetic individuals span a wide range of blood pressure values.
Insights: While high blood pressure might correlate with diabetes due to shared risk factors, it isn't a definitive standalone predictor.
5. Skin Thickness
Observation: Skin thickness shows overlap between diabetic and non-diabetic groups but tends to be slightly higher among diabetic individuals.
Insights: Skinfold thickness is indirectly related to insulin resistance and body composition, which might explain its moderate role in diabetes.
6. Insulin
Observation: Diabetic individuals often show either very low or very high insulin levels. Non-diabetic individuals are more evenly distributed across the middle range.
Insights: This reflects varying insulin resistance or beta-cell function in diabetic individuals, highlighting insulin's key role in diabetes.
7. BMI (Body Mass Index)
Observation: Higher BMI values are predominantly associated with diabetic individuals, while non-diabetic individuals tend to have lower BMI values.
Insights: Obesity (higher BMI) is a significant risk factor for diabetes, aligning with established medical knowledge.
8. Family History
Observation: A family history of diabetes (indicated by 1) shows a strong association with diabetes cases.
Insights: Genetics plays a role in diabetes predisposition, as reflected in the dataset.

Overall Summary
Key Predictors: Glucose, BMI, and insulin levels show the strongest visual distinction between diabetic and non-diabetic individuals.
Moderate Predictors: Age and pregnancies provide some differentiation but are not as decisive.
Weaker Predictors: Blood pressure and skin thickness show significant overlap between diabetic and non-diabetic groups.
# In[ ]:





# In[ ]:




