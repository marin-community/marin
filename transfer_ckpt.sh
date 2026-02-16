#!/bin/bash

# GCS Checkpoint Transfer Script
# Usage: ./transfer_ckpt.sh <source_gcs_path> <dest_gcs_path>
# Example: ./transfer_ckpt.sh gs://marin-us-east1/ckpt/model gs://marin-eu/ckpt/model

SOURCE_PATH=gs://marin-us-west4/checkpoints/vlm-official-qwen3-4b-stage2-correct-mixed-6-9fe39f/checkpoints/step-39000/
DEST_PATH=gs://marin-us-east1/checkpoints/vlm-official-qwen3-4b-stage2-correct-mixed-6-9fe39f/checkpoints/step-39000/  

if [ -z "$SOURCE_PATH" ] || [ -z "$DEST_PATH" ]; then
    echo "Usage: $0 <source_gcs_path> <dest_gcs_path>"
    exit 1
fi

echo "Transferring: $SOURCE_PATH -> $DEST_PATH"
gsutil -m cp -r "$SOURCE_PATH" "$DEST_PATH"
echo "Done!"
