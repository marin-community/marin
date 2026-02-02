#!/bin/bash
set -e

# Required environment variables:
#   RUNNER_NAME   - unique name for this runner instance
#   RUNNER_LABELS - comma-separated labels
#
# The GitHub token is fetched at runtime from GCP Secret Manager via the
# metadata server, so it never needs to be stored in systemd unit files.

REPO_URL="https://github.com/marin-community/marin"
GCP_SECRET="tpu-ci-github-token"

# Fetch GitHub PAT from Secret Manager using the VM's service account.
# Uses the metadata server to get an access token, then calls the Secret Manager REST API.
ACCESS_TOKEN=$(curl -sSf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | jq -r .access_token)

PROJECT_ID=$(curl -sSf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id")

GITHUB_TOKEN=$(curl -sSf \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  "https://secretmanager.googleapis.com/v1/projects/${PROJECT_ID}/secrets/${GCP_SECRET}/versions/latest:access" \
  | jq -r '.payload.data' | base64 -d)

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Failed to fetch GitHub token from Secret Manager"
    exit 1
fi

# Get a registration token from the PAT
REGISTRATION_TOKEN=$(curl -s -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  https://api.github.com/repos/marin-community/marin/actions/runners/registration-token \
  | jq -r .token)

if [ -z "$REGISTRATION_TOKEN" ] || [ "$REGISTRATION_TOKEN" = "null" ]; then
    echo "Failed to get registration token"
    exit 1
fi

# Clean up on exit: remove the runner from GitHub
cleanup() {
    echo "Removing runner..."
    ./config.sh remove --token "$REGISTRATION_TOKEN" 2>/dev/null || true
}
trap cleanup EXIT

# Configure as ephemeral runner (runs one job then exits)
./config.sh \
  --url "$REPO_URL" \
  --token "$REGISTRATION_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work _work \
  --unattended \
  --ephemeral \
  --replace

# Run the runner (blocks until one job completes, then exits)
./run.sh
