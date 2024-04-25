'''Just download a single parache file right now and test on it'''
from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb",
                repo_type="dataset",
                local_dir="../../data/fineweb",
                allow_patterns="data/CC-MAIN-2024-10/000_00000.parquet") # A single parquet file