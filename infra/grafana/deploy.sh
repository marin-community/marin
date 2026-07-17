#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Deploy Grafana + the finelog bridge to Cloud Run, IAP-gated, with Direct VPC
# egress so it reaches both finelog VMs on their internal IPs.
#
# Mirrors infra/status-page/deploy.sh — the same substrate, the same VPC path, the
# same IAP gate. Run `./deploy.sh setup` for the one-time SA/IAM steps.
set -euo pipefail

# --source=. below sends the build context to Cloud Build, so the context has to
# be this directory no matter where the script was invoked from.
cd "$(dirname "$0")"

# Not overridable: src/config.py pins the same project for the finelog VM
# lookup, so a PROJECT override here would deploy elsewhere and still read
# hai-gcp-models. Change both together or neither.
PROJECT="hai-gcp-models"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-marin-grafana}"
SA_NAME="marin-grafana"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
VPC_NETWORK="${VPC_NETWORK:-default}"
VPC_SUBNET="${VPC_SUBNET:-default}"

if [[ "${1:-}" == "setup" ]]; then
  cat <<SETUP
One-time setup for ${SERVICE} in ${PROJECT}:

  # The bridge resolves each finelog VM's internal IP via the Compute API, so the
  # service account needs read-only instance listing and nothing else. finelog
  # itself needs no grant: its cidr auth layer admits the VPC.
  gcloud iam service-accounts create ${SA_NAME} --project=${PROJECT} \\
    --display-name="Marin Grafana (Cloud Run)"

  gcloud projects add-iam-policy-binding ${PROJECT} \\
    --member="serviceAccount:${SA_EMAIL}" --role="roles/compute.viewer"

  # The OAuth consent screen and client are project-level and shared across the
  # project's IAP-protected services, so there is nothing per-service to create
  # here; see lib/iris/docs/iap-gclb.md.

Then deploy with: ./deploy.sh

Access takes two bindings. IAP terminates the browser session and calls the
service as its own service agent, so the agent needs invoke rights and people
need IAP rights; granting a person run.invoker only admits callers who mint their
own identity token.

  # 1. Let the IAP service agent invoke the service. Project number via:
  #    gcloud projects describe ${PROJECT} --format='value(projectNumber)'
  gcloud run services add-iam-policy-binding ${SERVICE} --region=${REGION} \\
    --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com" \\
    --role="roles/run.invoker"

  # 2. Let a person (or group:) through IAP.
  gcloud iap web add-iam-policy-binding --region=${REGION} \\
    --resource-type=cloud-run --service=${SERVICE} \\
    --member="user:someone@example.com" --role="roles/iap.httpsResourceAccessor"

Verify: gcloud run services describe ${SERVICE} --region=${REGION}   # Iap Enabled: true
SETUP
  exit 0
fi

echo "==> Building and deploying ${SERVICE}..."

# min/max-instances are both 1 on purpose. Grafana's SQLite is per-instance and
# ephemeral here, so >1 instance means divergent alert state and dashboard
# versions; 0 instances means no alert rules evaluate and first paint is a cold
# start. One warm instance is the only configuration that behaves.
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
  --timeout=60 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=1

echo "==> Deployed. URL:"
gcloud run services describe "${SERVICE}" --project="${PROJECT}" --region="${REGION}" \
  --format='value(status.url)'
