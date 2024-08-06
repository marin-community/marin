#!/bin/bash

VALID_ZONES=("us-central2-b" "us-west4-a" "europe-west4-b")
ZONE=${1:-"us-central2-b"}

if [[ ! " ${VALID_ZONES[@]} " =~ " ${ZONE} " ]]; then
    echo "Error: Invalid zone. Valid zones are: ${VALID_ZONES[*]}"
    exit 1
fi

echo "Cleaning inactive queued-resource in zone: $ZONE"

resources=$(gcloud alpha compute tpus queued-resources list --zone=$ZONE | awk '$NF == "SUSPENDED" || $NF == "FAILED" {print $1}')

if [ -z "$resources" ]; then
    echo "No SUSPENDED or FAILED resources found in zone $ZONE."
    exit 0
fi

for resource in $resources; do
    echo "Deleting resource: $resource in zone $ZONE"
    yes | gcloud compute tpus queued-resources delete "$resource" --zone=$ZONE --force --async
done

echo "Deletion process completed for zone $ZONE."
