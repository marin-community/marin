#!/bin/bash
# Fix double-nested validation/validation/ -> validation/ for 5 Paloma caches.
# The original copy script created validation/validation/ instead of validation/.
set -euo pipefail

CACHES=(
  "gs://marin-us-central2/tokenized/paloma/4chan-29cbdb"
  "gs://marin-us-central2/tokenized/paloma/mc4-164c1a"
  "gs://marin-us-central2/tokenized/paloma/dolma_100_programing_languages-12d009"
  "gs://marin-us-central2/tokenized/paloma/dolma-v1_5-0eedd4"
  "gs://marin-us-central2/tokenized/paloma/manosphere_meta_sep-4c7013"
)

for cache in "${CACHES[@]}"; do
  echo "Fixing: $cache"
  # Move validation/validation/* up to validation/
  gcloud storage cp -r "${cache}/validation/validation/*" "${cache}/validation/"
  # Remove the nested validation/validation/ directory
  gcloud storage rm -r "${cache}/validation/validation/"
  echo "Done: $cache"
  echo "---"
done

echo "All fixes complete."
