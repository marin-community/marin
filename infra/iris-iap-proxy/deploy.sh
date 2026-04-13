#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Deploy the Iris IAP proxy to Cloud Run with native IAP integration.
#
# Uses Cloud Run's built-in IAP support (--iap flag) — no load balancer,
# serverless NEG, or SSL certificate required.
#
# One service per cluster: the cluster name maps to both the Cloud Run
# service name and the GCE label used for controller discovery.
#
# Usage:
#   ./deploy.sh marin              # deploy iris-iap-proxy-marin
#   ./deploy.sh marin-dev          # deploy iris-iap-proxy-marin-dev
#   ./deploy.sh marin --setup      # print one-time setup commands

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="hai-gcp-models"
REGION="us-central1"
ZONE="us-central1-a"
SA_NAME="iris-iap-proxy"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
VPC_NETWORK="default"
VPC_SUBNET="default"

CLUSTER="${1:-}"
if [[ -z "${CLUSTER}" || "${CLUSTER}" == "--setup" ]]; then
    echo "Usage: $0 <cluster> [--setup]" >&2
    echo "  e.g.  $0 marin" >&2
    echo "        $0 marin-dev" >&2
    exit 1
fi

SERVICE="iris-iap-proxy-${CLUSTER}"
CONTROLLER_LABEL="iris-${CLUSTER}-controller"

if [[ "${2:-}" == "--setup" ]]; then
    cat <<SETUP
=== One-Time Setup (cluster=${CLUSTER}, service=${SERVICE}) ===

# 1. Create service account (shared across all clusters — only run once)
gcloud iam service-accounts create ${SA_NAME} \\
  --project=${PROJECT} \\
  --display-name="Iris IAP Proxy"

gcloud projects add-iam-policy-binding ${PROJECT} \\
  --member="serviceAccount:${SA_EMAIL}" \\
  --role="roles/compute.viewer"

gcloud projects add-iam-policy-binding ${PROJECT} \\
  --member="serviceAccount:${SA_EMAIL}" \\
  --role="roles/secretmanager.secretAccessor"

# 2. Configure OAuth consent screen (Cloud Console — one-time per project)
#    https://console.cloud.google.com/auth/branding?project=${PROJECT}
#    Select "External" audience type.

# 3. Create OAuth client (Cloud Console — one-time per project)
#    https://console.cloud.google.com/auth/clients?project=${PROJECT}
#    Create a "Web application" client.
#    Add authorized redirect URI:
#      https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect

# 4. Apply OAuth client to IAP settings (one-time per project)
#    Create iap_settings.yaml:
#      access_settings:
#        oauth_settings:
#          client_id: <CLIENT_ID>
#          client_secret: <CLIENT_SECRET>
#    gcloud iap settings set iap_settings.yaml --project=${PROJECT}

# 5. Deploy the Cloud Run service (use deploy.sh without --setup)

# 6. Grant the IAP service agent permission to invoke this Cloud Run service.
#    Find your project number:
#    gcloud projects describe ${PROJECT} --format='value(projectNumber)'
gcloud run services add-iam-policy-binding ${SERVICE} \\
  --region=${REGION} \\
  --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com" \\
  --role="roles/run.invoker"

# 7. Grant team members access through IAP
gcloud iap web add-iam-policy-binding \\
  --member="user:you@example.com" \\
  --role="roles/iap.httpsResourceAccessor" \\
  --region=${REGION} \\
  --resource-type=cloud-run \\
  --service=${SERVICE}

# 8. Verify IAP is enabled
gcloud run services describe ${SERVICE} --region=${REGION}
# Look for: Iap Enabled: true

SETUP
    exit 0
fi

echo "==> Building and deploying ${SERVICE} (cluster=${CLUSTER}, label=${CONTROLLER_LABEL})..."

gcloud beta run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --source=. \
  --service-account="${SA_EMAIL}" \
  --no-allow-unauthenticated \
  --iap \
  --network="${VPC_NETWORK}" \
  --subnet="${VPC_SUBNET}" \
  --vpc-egress=private-ranges-only \
  --set-env-vars="GCP_PROJECT=${PROJECT},CONTROLLER_ZONE=${ZONE},CONTROLLER_LABEL=${CONTROLLER_LABEL},CONTROLLER_PORT=10000" \
  --timeout=300 \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=1 \
  --concurrency=80 \
  --port=8080

echo "==> Deployed. Service URL:"
gcloud run services describe "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)'
