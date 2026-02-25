#!/bin/bash

# GCS Checkpoint Transfer Script
# Usage: ./transfer_ckpt.sh <source_gcs_path> <dest_gcs_path>
# Example: ./transfer_ckpt.sh gs://marin-us-east1/ckpt/model gs://marin-eu/ckpt/model

SOURCE_PATH=gs://marin-eu-west4/checkpoints/vlm-official-qwen3-stage3_4b_4096-correct-mixed-full-0-128-705a81/checkpoints/step-92000/*
DEST_PATH=gs://marin-us-central1/checkpoints/vlm-official-qwen3-stage3_4b_4096-correct-mixed-full-0-128-705a81/checkpoints/step-92000/

if [ -z "$SOURCE_PATH" ] || [ -z "$DEST_PATH" ]; then
    echo "Usage: $0 <source_gcs_path> <dest_gcs_path>"
    exit 1
fi

echo "Transferring: $SOURCE_PATH -> $DEST_PATH"
gsutil -m cp -r "$SOURCE_PATH" "$DEST_PATH"
echo "Done!"
