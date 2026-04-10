#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Deploy the Iris IAP proxy to Cloud Run.
#
# Prerequisites (one-time setup — see SETUP section below):
#   1. Create a service account for the proxy
#   2. Deploy the Cloud Run service (this script)
#   3. Create a Global External Application Load Balancer with serverless NEG
#   4. Enable IAP on the LB backend service
#   5. Store the proxy API key in Secret Manager
#
# Usage:
#   ./deploy.sh                    # deploy to Cloud Run
#   ./deploy.sh --setup            # print one-time setup commands

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="hai-gcp-models"
REGION="us-central1"
SERVICE="iris-iap-proxy"
SA_NAME="iris-iap-proxy"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
VPC_NETWORK="default"
VPC_SUBNET="default"

if [[ "${1:-}" == "--setup" ]]; then
    cat <<'SETUP'
=== One-Time Setup ===

# 1. Create service account
gcloud iam service-accounts create iris-iap-proxy \
  --project=hai-gcp-models \
  --display-name="Iris IAP Proxy"

# Grant compute.viewer for VM discovery
gcloud projects add-iam-policy-binding hai-gcp-models \
  --member="serviceAccount:iris-iap-proxy@hai-gcp-models.iam.gserviceaccount.com" \
  --role="roles/compute.viewer"

# Grant Secret Manager access for proxy token
gcloud projects add-iam-policy-binding hai-gcp-models \
  --member="serviceAccount:iris-iap-proxy@hai-gcp-models.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 2. Deploy the Cloud Run service (use deploy.sh without --setup)

# 3. Create a serverless NEG for the Cloud Run service
gcloud compute network-endpoint-groups create iris-iap-proxy-neg \
  --project=hai-gcp-models \
  --region=us-central1 \
  --network-endpoint-type=serverless \
  --cloud-run-service=iris-iap-proxy

# 4. Create a backend service
gcloud compute backend-services create iris-iap-proxy-backend \
  --project=hai-gcp-models \
  --global \
  --load-balancing-scheme=EXTERNAL_MANAGED \
  --protocol=HTTPS

gcloud compute backend-services add-backend iris-iap-proxy-backend \
  --project=hai-gcp-models \
  --global \
  --network-endpoint-group=iris-iap-proxy-neg \
  --network-endpoint-group-region=us-central1

# 5. Create URL map and HTTPS proxy
gcloud compute url-maps create iris-iap-proxy-urlmap \
  --project=hai-gcp-models \
  --default-service=iris-iap-proxy-backend

# Reserve a static IP
gcloud compute addresses create iris-iap-proxy-ip \
  --project=hai-gcp-models \
  --global

# Create a managed SSL certificate (replace with your domain)
gcloud compute ssl-certificates create iris-iap-proxy-cert \
  --project=hai-gcp-models \
  --domains=iris.example.com \
  --global

gcloud compute target-https-proxies create iris-iap-proxy-https \
  --project=hai-gcp-models \
  --ssl-certificates=iris-iap-proxy-cert \
  --url-map=iris-iap-proxy-urlmap

gcloud compute forwarding-rules create iris-iap-proxy-fwd \
  --project=hai-gcp-models \
  --global \
  --target-https-proxy=iris-iap-proxy-https \
  --address=iris-iap-proxy-ip \
  --ports=443

# 6. Enable IAP on the backend service
gcloud iap web enable \
  --project=hai-gcp-models \
  --resource-type=backend-services \
  --service=iris-iap-proxy-backend

# Grant team members access through IAP
gcloud iap web add-iam-policy-binding \
  --project=hai-gcp-models \
  --resource-type=backend-services \
  --service=iris-iap-proxy-backend \
  --member="user:you@example.com" \
  --role="roles/iap.httpsResourceAccessor"

# 7. Create the proxy API key and store in Secret Manager
# iris --cluster=marin key create --name iap-proxy
# echo -n "<KEY>" | gcloud secrets create iris-iap-proxy-token \
#   --project=hai-gcp-models --data-file=-

# 8. Get the IAP audience (needed for IAP_AUDIENCE env var)
# Format: /projects/<PROJECT_NUMBER>/global/backendServices/<BACKEND_SERVICE_ID>
# Find PROJECT_NUMBER:  gcloud projects describe hai-gcp-models --format='value(projectNumber)'
# Find BACKEND_SERVICE_ID: gcloud compute backend-services describe iris-iap-proxy-backend --global --format='value(id)'
# Then update the Cloud Run service env var:
# gcloud run services update iris-iap-proxy --region=us-central1 \
#   --update-env-vars="IAP_AUDIENCE=/projects/<NUMBER>/global/backendServices/<ID>"

SETUP
    exit 0
fi

echo "==> Building and deploying ${SERVICE} to Cloud Run..."

gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --source=. \
  --service-account="${SA_EMAIL}" \
  --no-allow-unauthenticated \
  --network="${VPC_NETWORK}" \
  --subnet="${VPC_SUBNET}" \
  --vpc-egress=private-ranges-only \
  --set-env-vars="GCP_PROJECT=${PROJECT},CONTROLLER_ZONE=us-central1-a,CONTROLLER_LABEL=iris-marin-controller,CONTROLLER_PORT=10000,PROXY_TOKEN_SECRET=iris-iap-proxy-token,REQUIRE_IAP=true" \
  --timeout=300 \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=1 \
  --port=8080

echo "==> Deployed. Service URL:"
gcloud run services describe "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)'
