
import requests

def download_dropbox_file(dropbox_link, save_path):
    # Ensure the link is a direct download link
    direct_link = dropbox_link.replace("?dl=0", "?dl=1")

    response = requests.get(direct_link, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Download complete: {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# Example usage
UR_FUNNY_RAW = "https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1"
#UR_FUNNY_Extracted = 'https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1'
save_location = "../ur-funny_raw.zip"

download_dropbox_file(UR_FUNNY_RAW, save_location)

import zipfile
import os

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")

# Example usage
zip_file_path = "../ur-funny_raw.zip"  # Path to the ZIP file
extract_folder = "../../../../scratch1/jiaqilu/CSCI535/CSCI535-Project/dataset_raw"  # Folder to extract files

# Ensure the extraction folder exists
os.makedirs(extract_folder, exist_ok=True)

unzip_file(save_location, extract_folder)