#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Deploy the Iris IAP proxy to Cloud Run with native IAP integration.
#
# Uses Cloud Run's built-in IAP support (--iap flag) — no load balancer,
# serverless NEG, or SSL certificate required.
#
# Prerequisites (one-time setup — see SETUP section below):
#   1. Create a service account for the proxy
#   2. Configure OAuth consent screen + client (Cloud Console)
#   3. Apply OAuth client to IAP settings
#   4. Deploy the Cloud Run service with --iap (this script)
#   5. Grant the IAP service agent roles/run.invoker
#   6. Grant team members IAP access
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

# 2. Configure OAuth consent screen (Cloud Console — one-time)
#    Go to: https://console.cloud.google.com/auth/branding?project=hai-gcp-models
#    Select "External" audience type.

# 3. Create OAuth client (Cloud Console)
#    Go to: https://console.cloud.google.com/auth/clients?project=hai-gcp-models
#    Create a "Web application" client.
#    Add authorized redirect URI:
#      https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect

# 4. Apply OAuth client to IAP settings
#    Create iap_settings.yaml:
#      access_settings:
#        oauth_settings:
#          client_id: <CLIENT_ID>
#          client_secret: <CLIENT_SECRET>
#
#    Then apply:
#    gcloud iap settings set iap_settings.yaml --project=hai-gcp-models

# 5. Deploy the Cloud Run service (use deploy.sh without --setup)

# 6. Grant the IAP service agent permission to invoke the Cloud Run service
#    Find your project number:
#    gcloud projects describe hai-gcp-models --format='value(projectNumber)'
gcloud run services add-iam-policy-binding iris-iap-proxy \
  --region=us-central1 \
  --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

# 7. Grant team members access through IAP
gcloud iap web add-iam-policy-binding \
  --member="user:you@example.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --region=us-central1 \
  --resource-type=cloud-run \
  --service=iris-iap-proxy

# 8. (Optional) Create the proxy API key and store in Secret Manager
# iris --cluster=marin key create --name iap-proxy
# echo -n "<KEY>" | gcloud secrets create iris-iap-proxy-token \
#   --project=hai-gcp-models --data-file=-

# 9. Verify IAP is enabled
gcloud run services describe iris-iap-proxy --region=us-central1
# Look for: Iap Enabled: true

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
  --iap \
  --network="${VPC_NETWORK}" \
  --subnet="${VPC_SUBNET}" \
  --vpc-egress=private-ranges-only \
  --set-env-vars="GCP_PROJECT=${PROJECT},CONTROLLER_ZONE=us-central1-a,CONTROLLER_LABEL=iris-marin-controller,CONTROLLER_PORT=10000,PROXY_TOKEN_SECRET=iris-iap-proxy-token" \
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
