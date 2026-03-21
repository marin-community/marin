#!/bin/bash
TPU=${1:-exp2039-nb-sampler}
ZONE=${2:-us-central1-a}
while true; do
  echo "=== $(date) ==="
  gcloud alpha compute tpus tpu-vm ssh "$TPU" --quiet --worker=all --zone="$ZONE" \
    --command="docker ps --format '{{.Names}} {{.Status}}' | grep levanter && docker logs levanter 2>&1 | tail -5"
  echo
  sleep 60
done
