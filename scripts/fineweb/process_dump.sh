#!/bin/bash

# Ensure the script exits on any error
set -e

# Check if the GCS_PATH argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <GCS_PATH>"
  exit 1
fi

# Set the GCS path from the first argument
GCS_PATH="$1"

# Ensure the GCS_PATH ends with a slash
if [[ "${GCS_PATH}" != */ ]]; then
  GCS_PATH="${GCS_PATH}/"
fi

# List all parquet files in the specified GCS path
PARQUET_FILES=$(gsutil ls "${GCS_PATH}*.parquet")

# Loop through each Parquet file and process it
for FILE in $PARQUET_FILES
do
  echo "Processing $FILE..."
  ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python scripts/fineweb/process_parquet_fw.py --input_file_path "$FILE"
done

echo "All files processed."
