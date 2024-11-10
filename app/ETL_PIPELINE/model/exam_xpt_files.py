import requests
from bs4 import BeautifulSoup
import os

# Define URLs and folder paths
url = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination"
base_url = "https://wwwn.cdc.gov"
output_folder = r"C:\Users\sneha\OneDrive\Desktop\examination_data\required_examination_xpt_files"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Keywords to look for, excluding specific unwanted terms like arthritis
keywords = ["blood pressure", "body measures"]
exclude_keywords = ["arthritis", "lower extremity","oscillometric"]

# Fetch the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all table rows containing data file names
table_rows = soup.select("tr")

xpt_links = []

# Loop through each row to find relevant XPT links
for row in table_rows:
    row_text = row.get_text().strip().lower()
    # Check if any keyword is in the row text and exclude rows with unwanted terms
    if any(keyword in row_text for keyword in keywords) and not any(exclude_keyword in row_text for exclude_keyword in exclude_keywords):
        # Find 'a' tags with 'href' that ends with '.XPT'
        links = row.select("a[href$='.XPT']")
        for link in links:
            xpt_url = base_url + link['href']
            xpt_links.append(xpt_url)
            print(f"Matched row with keyword; link added: {xpt_url}")  # Debug info

# If no links matched, print a message
if not xpt_links:
    print("No relevant XPT links were found. Please check the keyword list or page structure.")
else:
    # Download the relevant XPT files
    for xpt_url in xpt_links:
        file_name = os.path.join(output_folder, os.path.basename(xpt_url))

        # Download the XPT file
        xpt_response = requests.get(xpt_url)
        with open(file_name, "wb") as file:
            file.write(xpt_response.content)
        print(f"Downloaded {file_name}")
