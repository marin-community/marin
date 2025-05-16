#!/bin/bash
# setup_and_run.sh
# This script downloads the model and launches the UI

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./setup_and_run.sh [model_path]"
    exit 1
fi

MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)

# Download model files
echo "Downloading model from $MODEL_PATH..."
gsutil cp -r $MODEL_PATH .

# Launch the UI
echo "Starting chat interface with model $MODEL_NAME..."
python chat_ui_for_model.py --model $MODEL_NAME
