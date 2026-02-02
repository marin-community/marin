#!/bin/bash
set -e

# Required environment variables:
#   GITHUB_TOKEN - PAT with repo scope for runner registration
#   RUNNER_NAME  - unique name for this runner instance
#   RUNNER_LABELS - comma-separated labels

REPO_URL="https://github.com/marin-community/marin"

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
