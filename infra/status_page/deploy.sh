#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Deploy the Marin status page to Cloud Run with native IAP integration.
#
# Mirrors infra/iris-iap-proxy/deploy.sh: one Cloud Run service
# (`marin-status-page`) in hai-gcp-models/us-central1, Direct VPC egress
# so the service can reach the iris controller at its internal IP,
# native IAP for auth, min/max-instances=1 to keep the TTL cache warm.
#
# Usage:
#   ./deploy.sh             # deploy
#   ./deploy.sh --setup     # print one-time setup commands

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="hai-gcp-models"
REGION="us-central1"
ZONE="us-central1-a"
SERVICE="marin-status-page"
SA_NAME="marin-status-page"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
VPC_NETWORK="default"
VPC_SUBNET="default"
CONTROLLER_LABEL="iris-marin-controller"
CONTROLLER_PORT=10000
GITHUB_TOKEN_SECRET="marin-status-page-github-token"

if [[ "${1:-}" == "--setup" ]]; then
    cat <<SETUP
=== One-Time Setup (service=${SERVICE}) ===

# 1. Create the service account
gcloud iam service-accounts create ${SA_NAME} \\
  --project=${PROJECT} \\
  --display-name="Marin Status Page"

gcloud projects add-iam-policy-binding ${PROJECT} \\
  --member="serviceAccount:${SA_EMAIL}" \\
  --role="roles/compute.viewer"

gcloud projects add-iam-policy-binding ${PROJECT} \\
  --member="serviceAccount:${SA_EMAIL}" \\
  --role="roles/secretmanager.secretAccessor"

# 2. Create the GitHub token secret.
#    The repo is public — the token exists ONLY to lift the GitHub API
#    rate limit from 60/hr (unauth, per egress IP) to 5000/hr. Either a
#    classic token with NO scopes, or a fine-grained PAT scoped to
#    "Public repositories (read-only)" works.
echo -n "<paste-github-token>" | gcloud secrets create ${GITHUB_TOKEN_SECRET} \\
  --project=${PROJECT} \\
  --data-file=-

# 3. OAuth consent screen + OAuth client are shared with the iris IAP
#    proxy — no extra setup if you already ran
#    infra/iris-iap-proxy/deploy.sh marin --setup. Otherwise see that
#    script for the one-time project-level steps.

# 4. Deploy (use deploy.sh without --setup)

# 5. Grant the IAP service agent permission to invoke this Cloud Run service.
#    Find your project number:
#    gcloud projects describe ${PROJECT} --format='value(projectNumber)'
gcloud run services add-iam-policy-binding ${SERVICE} \\
  --region=${REGION} \\
  --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com" \\
  --role="roles/run.invoker"

# 6. Grant team members access through IAP
gcloud iap web add-iam-policy-binding \\
  --member="user:you@example.com" \\
  --role="roles/iap.httpsResourceAccessor" \\
  --region=${REGION} \\
  --resource-type=cloud-run \\
  --service=${SERVICE}

# 7. Verify IAP is enabled
gcloud run services describe ${SERVICE} --region=${REGION}
# Look for: Iap Enabled: true

SETUP
    exit 0
fi

echo "==> Building and deploying ${SERVICE}..."

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
  --set-env-vars="GCP_PROJECT=${PROJECT},CONTROLLER_ZONE=${ZONE},CONTROLLER_LABEL=${CONTROLLER_LABEL},CONTROLLER_PORT=${CONTROLLER_PORT},CLUSTER_NAME=marin" \
  --set-secrets="GITHUB_TOKEN=${GITHUB_TOKEN_SECRET}:latest" \
  --timeout=60 \
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
