"""
Usage:
python marin/processing/fasttext/download_dolma_classifier.py
ray job submit --working-dir . --no-wait -- python marin/processing/fasttext/download_dolma_classifier.py
"""
import os
from huggingface_hub import hf_hub_download

# Define the repository and file details
repo_id = "allenai/dolma-1_7-fasttext-quality-filter"
filename = "model.bin"
local_filename = "model.bin"
local_dir = os.path.expanduser("~/dolma_fasttext_model")

def download_file():
    # Create the local folder if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download the file
    downloaded_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)

    # Move the downloaded file to the desired location
    destination_path = os.path.join(local_dir, local_filename)
    print(f"File downloaded to: {destination_path}")
    
    return destination_path

if __name__ == "__main__":
    download_file()
