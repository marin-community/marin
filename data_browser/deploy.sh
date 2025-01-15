#/bin/bash

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

docker build -t marin-data-browser -f Dockerfile.prod .

gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag marin-data-browser us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser

docker push us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser

gcloud run deploy marin-data-browser \
    --image us-central1-docker.pkg.dev/hai-gcp-models/marin-data-browser/marin-data-browser \
    --port 80 \
    --region us-central1 \
    --platform managed \
    --ingress internal-and-cloud-load-balancing \
    --service-account marin-data-browser@hai-gcp-models.iam.gserviceaccount.com
