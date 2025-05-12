## Deployment (Google Cloud Run)

To deploy, first copy the service account key file for marin-data-browser@hai-gcp-models.iam.gserviceaccount.com to `gcs-key.json` in this directory:

    gcloud iam service-accounts keys create gcs-key.json --iam-account=marin-data-browser@hai-gcp-models.iam.gserviceaccount.com

Then run:

    ./deploy.sh

And then open https://marin-data-browser-748532799086.us-central1.run.app in your browser.

To check logs, you can visit https://console.cloud.google.com/run/detail/us-central1/marin-data-browser/logs?project=hai-gcp-models or run:

    gcloud run services logs read marin-data-browser --project=hai-gcp-models --platform=managed --region=us-central1

To update the service, run:

    ./deploy.sh

To delete the service, run:

    gcloud run services delete marin-data-browser --project=hai-gcp-models --region=us-central1
