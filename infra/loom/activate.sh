#!/usr/bin/env bash
set -euo pipefail

: "${LOOM_PROJECT:?LOOM_PROJECT is required}"
: "${LOOM_ZONE:?LOOM_ZONE is required}"
: "${LOOM_INSTANCE:?LOOM_INSTANCE is required}"
: "${LOOM_DOMAIN:?LOOM_DOMAIN is required}"

gcloud --project="$LOOM_PROJECT" compute ssh "$LOOM_INSTANCE" \
  --zone="$LOOM_ZONE" --quiet \
  --command='
    set -euo pipefail
    sudo rm -f /run/loom-startup-succeeded
    sudo systemctl restart google-startup-scripts.service
    sudo test -f /run/loom-startup-succeeded
  '

for _ in $(seq 1 90); do
  if curl -fsS "https://${LOOM_DOMAIN}/api/ready" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"'; then
    exit 0
  fi
  sleep 10
done

echo "Loom did not become publicly ready after activation" >&2
exit 1
