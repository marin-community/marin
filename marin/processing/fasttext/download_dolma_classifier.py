"""
Usage:
python marin/processing/fasttext/download_dolma_classifier.py
ray job submit --working-dir . --no-wait -- python marin/processing/fasttext/download_dolma_classifier.py
"""
import os
from huggingface_hub import hf_hub_download

# Define the repository and file details
REPO_ID = "allenai/dolma-1_7-fasttext-quality-filter"
FILENAME = "model.bin"

def download_file(output_dir=os.path.expanduser("~/dolma_fasttext_model")):
    destination_path = os.path.join(output_dir, FILENAME)

    # if os.path.exists(destination_path):
    #     return destination_path

    # Create the local folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the file
    downloaded_file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=output_dir)

    print(f"File downloaded to: {destination_path}")
    
    return destination_path

if __name__ == "__main__":
    download_file()
