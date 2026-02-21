import json
from pathlib import Path
import os
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

def download_nfc(subfolder: str):
    # Load the JSON data
    with open(Path(subfolder) / 'fursuit-list.json', 'r') as file:
        data = json.load(file)

    # Create a directory to save the images
    os.makedirs(Path(subfolder) / 'fursuit_images', exist_ok=True)

    # Iterate through the list and download images with progress indicator
    for fursuit in tqdm(data['FursuitList'], desc="Downloading images"):
        image_url = fursuit['ImageUrl']
        image_name = image_url.split('/')[-1]
        save_path = os.path.join(Path(subfolder) / 'fursuit_images', image_name)
        try:
            download_image(image_url, save_path)
        except Exception as e:
            print(e)

        os.makedirs(Path(subfolder) / 'fursuit_thumbs', exist_ok=True)

    # Iterate through the list and download thumbs with progress indicator
    for fursuit in tqdm(data['FursuitList'], desc="Downloading thumbnails"):
        image_url = fursuit['ThumbnailUrl']
        if not image_url:
            continue
        image_name = image_url.split('/')[-1]
        save_path = os.path.join(Path(subfolder) / 'fursuit_thumbs', image_name)
        try:
            download_image(image_url, save_path)
        except Exception as e:
            print(e)

# Function to download an image with retries
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_image(url, save_path, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download image from {url}")

