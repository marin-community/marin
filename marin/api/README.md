# Preparation
## Pip install the necessary packages
`pip3 install -r requirements.txt`

# Steps to use the models
## 1. Download the HF model
`gsutil -m cp -r gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1536000 ~/MarinModel`

The command copies from GCP to `MarinModel` folder in the user's home directory.
Note that `-m` is for multi-threading. Remove if it interfers with the operation.

## 2. Run the Flask server
`python3 marin/api/model_server.py` (binds to localhost:8000 by default)

## 3a. Query with Python script.
Sample provided in `marin/api/test_client.py`

## 3b. Try out the GUI
Launch `python3 marin/api/chat_ui.py` and then navigate to localhost:7860 (or whatever the script tells you to) 