"""
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
"""

import bz2
from io import BytesIO
import zipfile
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path

import draccus
import fsspec
import requests


fs_http = fsspec.filesystem("http")
fs_gcs = fsspec.filesystem("gcs")

@dataclass
class DownloadConfig:
    input_path: Path
    output_path: str
    chunk_size: int = 1024 * 1024 # 1MB


def get_file_size(url):
    """Get content length from headers without downloading"""
    response = requests.head(url)
    return int(response.headers.get('content-length', 0))


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    try:
        print(f"Starting transfer of FAU dataset...")
        print(f"Source: {cfg.input_path}")
        
        fs_http = fsspec.filesystem('http')
        fs_gcs = fsspec.filesystem('gcs')
        
        total_size = get_file_size(cfg.input_path)
        
        print(f"Downloading and extracting to {cfg.output_path}...")
        
        # Download zip to memory
        zip_content = BytesIO()
        with fs_http.open(cfg.input_path, 'rb') as source, \
                tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading ZIP") as pbar:
            while True:
                chunk = source.read(cfg.chunk_size)
                if not chunk:
                    break
                zip_content.write(chunk)
                pbar.update(len(chunk))
        
        zip_content.seek(0)
        
        # Process and upload files
        with zipfile.ZipFile(zip_content) as zip_ref:
            file_list = [f for f in zip_ref.filelist if not f.filename.endswith('/')]
            print(f"\nExtracting and uploading {len(file_list)} files...")
            
            for file_info in tqdm(file_list, desc="Extracting and uploading"):
                # Read each file from zip
                with zip_ref.open(file_info.filename) as file:
                    content = file.read()
                
                # Upload to GCS maintaining folder structure
                gcs_path = f"{cfg.output_path}/{file_info.filename}"
                with fs_gcs.open(gcs_path, 'wb') as destination:
                    destination.write(content)
                
                print(f"Uploaded: gs://{gcs_path}")
        
        zip_content.close()
        print("\nTransfer completed successfully!")
        
    except Exception as e:
        print(f"Error during transfer: {e}")
        raise
