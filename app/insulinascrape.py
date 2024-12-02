import requests
from bs4 import BeautifulSoup
import os

# Define the datasets to be downloaded
datasets = [
    "GLU_I", "GLU_J", "GLU_L", "GLU_E", "GLU_D", "GLU_F", 
    "GLU_G", "LAB10AM", "INS_H", "INS_I", "INS_J", "P_INS"
]

# Base URL for the NHANES laboratory data search page
base_url = "https://wwwn.cdc.gov"
search_url = base_url + "/nchs/nhanes/search/datapage.aspx?Component=Laboratory"

# Directory to save downloaded files
save_dir = "nhanes_datasets"
os.makedirs(save_dir, exist_ok=True)

def get_dataset_links():
    """ Scrape the NHANES website to find links to the specified datasets """
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    dataset_links = {}
    
    # Find all links on the page
    for link in soup.find_all("a", href=True):
        href = link['href']
        text = link.text.strip()
        
        # Check if the link corresponds to one of the dataset names
        for dataset in datasets:
            if dataset in href:
                full_url = base_url + href  # Ensure the URL is complete
                dataset_links[dataset] = full_url
                
    return dataset_links

def download_dataset(dataset_name, url):
    """ Download the dataset and save it to the specified directory """
    response = requests.get(url)
    
    # Extract the file name from the URL
    file_name = url.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)
    
    # Save the file
    with open(file_path, "wb") as file:
        file.write(response.content)
    
    print(f"Downloaded {dataset_name} to {file_path}")

def main():
    dataset_links = get_dataset_links()
    
    for dataset, link in dataset_links.items():
        download_dataset(dataset, link)

if __name__ == "__main__":
    main()
