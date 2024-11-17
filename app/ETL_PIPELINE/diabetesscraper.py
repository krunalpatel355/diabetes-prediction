import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Define the base URL and the URL for Questionnaire Data (Diabetes-related)
base_url = "https://wwwn.cdc.gov"
questionnaire_url = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire"

# Define download folders for diabetes files (Data and Doc)
download_folder = r"C:\Users\Hazel\Downloads\NHANES_Data\DiabetesData"
data_folder = os.path.join(download_folder, "DataFiles")
doc_folder = os.path.join(download_folder, "DocFiles")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(doc_folder, exist_ok=True)

# Keywords for identifying diabetes-related data and documentation files
diabetes_data_keywords = ["DIQ_B Data", "DIQ_C Data", "DIQ_D Data", "DIQ_E Data", "DIQ_F Data", 
                          "DIQ_G Data", "DIQ_H Data", "DIQ_I Data", "DIQ_J Data", "DIQ_L Data", "P_DIQ Data"]
diabetes_doc_keywords = ["DIQ_B Doc", "DIQ_C Doc", "DIQ_D Doc", "DIQ_E Doc", "DIQ_F Doc", 
                         "DIQ_G Doc", "DIQ_H Doc", "DIQ_I Doc", "DIQ_J Doc", "DIQ_L Doc", "P_DIQ Doc"]

# Function to scrape links for both data and doc files
def get_links():
    response = requests.get(questionnaire_url)
    soup = BeautifulSoup(response.text, "html.parser")
    data_links = {}
    doc_links = {}
    
    print("Searching for diabetes-related .XPT files and documentation...")

    for link in soup.find_all("a", href=True):
        href = link['href']
        text = link.text.strip()
        
        # Check if the link is a Data file
        if ".XPT" in href and any(keyword in text for keyword in diabetes_data_keywords):
            full_url = base_url + href
            data_links[text] = full_url
            print(f"Matched data file for download: {full_url}")
        
        # Check if the link is a Doc file
        elif any(keyword in text for keyword in diabetes_doc_keywords):
            full_url = base_url + href
            doc_links[text] = full_url
            print(f"Matched doc file for download: {full_url}")

    if not data_links and not doc_links:
        print("No matching diabetes-related files found.")
    
    return data_links, doc_links

# Function to download and save files
def download_files(links, save_folder):
    for name, url in links.items():
        file_name = url.split("/")[-1]
        file_path = os.path.join(save_folder, file_name)
        
        if not os.path.exists(file_path):  # Avoid redownloading
            response = requests.get(url)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {file_name} to {file_path}")

# Combine .XPT data files into a single CSV file
def combine_xpt_to_csv():
    combined_df = pd.DataFrame()
    output_csv = os.path.join(data_folder, 'combined_diabetes_data.csv')
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.XPT'):
            xpt_path = os.path.join(data_folder, file_name)
            try:
                df = pd.read_sas(xpt_path, format='xport', encoding='utf-8')
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"Processed {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    combined_df.to_csv(output_csv, index=False)
    print(f"All files have been combined and saved to {output_csv}")

# Execute the full process
def main():
    # Step 1: Scrape links for diabetes data and doc files
    data_links, doc_links = get_links()
    
    # Step 2: Download diabetes data files
    if data_links:
        print("Downloading Diabetes Data Files...")
        download_files(data_links, data_folder)
    
    # Step 3: Download diabetes doc files
    if doc_links:
        print("Downloading Diabetes Doc Files...")
        download_files(doc_links, doc_folder)
    
    # Step 4: Combine downloaded data files into a single CSV
    print("Combining Diabetes Data Files...")
    combine_xpt_to_csv()

if __name__ == "__main__":
    main()
