# Data Browser

## Development

To start the data browser:

    docker-compose up --build

## Deployment (Google Cloud Run)

To deploy:

    ./deploy.sh

View:

    https://marin-data-browser-748532799086.us-central1.run.app

Check logs:

    https://console.cloud.google.com/run/detail/us-central1/marin-data-browser/logs?project=hai-gcp-models

TODO: https://cloud.google.com/iap/docs/enabling-cloud-run#gcloud

## Deployment (CRFM)

### One-time setup (already done for everyone, just for reference)

Create a GCP service account:

    gcloud iam service-accounts create marin-data-browser --description="Marin Data Browser"
    gcloud projects add-iam-policy-binding hai-gcp-models --member=serviceAccount:marin-data-browser@hai-gcp-models.iam.gserviceaccount.com --role=roles/storage.objectViewer

### One-time setup (for each person):

Get GCS credentials:

    gcloud iam service-accounts keys create gcs-key.json --iam-account=marin-data-browser@hai-gcp-models.iam.gserviceaccount.com

Put the ngrok credentials in `ngrok.env` (get it from https://dashboard.ngrok.com/get-started/your-authtoken):

    NGROK_AUTHTOKEN=<insert auth token>

### Every time you change the code:

Build React app into static assets to optimize for performance; these assets
will be served from the server:

    npm run build

Build the Docker image:

    docker build . -t marin/data_browser

Create a shared network:

    docker network create data_browser_net

To run the server:

    docker run --restart=always --rm --name flask -p 5000:5000 --network data_browser_net -v $PWD/gcs-key.json:/app/gcs-key.json -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json marin/data_browser

Use ngrok to make the server available publicly:

    docker run --restart=always --rm --name ngrok -p 4040:4040 --network data_browser_net --env-file ngrok.env ngrok/ngrok http --domain=marlin-subtle-barnacle.ngrok-free.app http://flask:5000 --oauth google

TODO: make this work with docker-compose
