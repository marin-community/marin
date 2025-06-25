TPU_NAME="post-training-v5p-8"
ACCELERATOR_TYPE="v5p-8"
VERSION="v2-alpha-tpuv5"

ZONES=(
  "us-central1-a"
  "us-east5-a"
  "us-east5-b"
  "us-east5-c"
  "us-east5-d"
  "europe-west4-a"
  "europe-west4-b"
  "europe-west4-c"
  "us-east1-b"
  "us-east1-c"
  "us-central1-b"
  "us-central1-c"
)

while true; do
  for ZONE in "${ZONES[@]}"; do
    echo "Trying zone: $ZONE"
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --zone="$ZONE" \
      --accelerator-type="$ACCELERATOR_TYPE" \
      --version="$VERSION"
    
    if [ $? -eq 0 ]; then
      echo "✅ TPU successfully created in zone: $ZONE"
      exit 0
    else
      echo "❌ Failed in zone: $ZONE, trying next..."
    fi
  done

  echo "🔁 All zones tried, none succeeded. Sleeping for 60 seconds before retrying..."
  sleep 60
done
