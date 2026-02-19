#!/bin/bash
# Copy FAILED Paloma caches from us-central1 to us-central2.
# These 5 subsets are FAILED in us-central2 but SUCCESS in us-central1,
# both tokenized with meta-llama/Meta-Llama-3.1-8B from raw/paloma-fc6827.
set -euo pipefail

declare -A COPY_MAP=(
  ["gs://marin-us-central1/tokenized/paloma/4chan-16d506"]="gs://marin-us-central2/tokenized/paloma/4chan-29cbdb"
  ["gs://marin-us-central1/tokenized/paloma/mc4-8ead3d"]="gs://marin-us-central2/tokenized/paloma/mc4-164c1a"
  ["gs://marin-us-central1/tokenized/paloma/dolma_100_programing_languages-3eb825"]="gs://marin-us-central2/tokenized/paloma/dolma_100_programing_languages-12d009"
  ["gs://marin-us-central1/tokenized/paloma/dolma-v1_5-94152e"]="gs://marin-us-central2/tokenized/paloma/dolma-v1_5-0eedd4"
  ["gs://marin-us-central1/tokenized/paloma/manosphere_meta_sep-bb4a78"]="gs://marin-us-central2/tokenized/paloma/manosphere_meta_sep-4c7013"
)

for src in "${!COPY_MAP[@]}"; do
  dst="${COPY_MAP[$src]}"

  # Clean up bad double-nested validation/validation/ from previous run
  if gcloud storage ls "${dst}/validation/validation/" &>/dev/null; then
    echo "Removing bad nested dir: ${dst}/validation/validation/"
    gcloud storage rm -r "${dst}/validation/validation/"
  fi

  echo "Copying: ${src}/validation/* -> ${dst}/validation/"
  gcloud storage cp -r "${src}/validation/*" "${dst}/validation/"

  # Append SUCCESS status so executor skips re-tokenization
  echo '{"date": "'"$(date -u +%Y-%m-%dT%H:%M:%S.%6N)"'", "status": "SUCCESS", "message": "Copied from us-central1", "ray_task_id": null}' | \
    gcloud storage cp - "${dst}/.executor_status"

  echo "Done: $dst"
  echo "---"
done

echo "All copies complete."
