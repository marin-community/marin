#!/bin/bash

# This script deploys the Marin Data Browser application to Google Cloud Run.
# It performs the following steps:
# 1. Creates an Artifact Registry repository if it doesn't exist
# 2. Builds the Docker image using production Dockerfile
# 3. Tags and pushes the image to Artifact Registry
# 4. Deploys the application to Cloud Run
# 5. Configures public access to the service

set -e

# Check if repository exists, create only if it doesn't
if ! gcloud artifacts repositories describe marin-data-browser \
    --location=us-central1 \
    --project=hai-gcp-models >/dev/null 2>&1; then
    gcloud artifacts repositories create marin-data-browser \
        --repository-format=docker \
        --location=us-central1 \
        --project=hai-gcp-models
fi

docker build --platform linux/amd64 -t marin-data-browser -f Dockerfile.prod .

gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag marin-data-browser us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser

docker push us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser

gcloud run deploy marin-data-browser \
    --image us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser \
    --port 80 \
    --region us-central1 \
    --platform managed \
    --ingress all \
    --service-account marin-data-browser@hai-gcp-models.iam.gserviceaccount.com

# Allow unauthenticated access to the service
gcloud run services add-iam-policy-binding marin-data-browser \
    --region=us-central1 \
    --member="allUsers" \
    --role="roles/run.invoker"
