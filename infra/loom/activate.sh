#!/usr/bin/env bash
set -euo pipefail

: "${LOOM_PROJECT:?LOOM_PROJECT is required}"
: "${LOOM_ZONE:?LOOM_ZONE is required}"
: "${LOOM_INSTANCE:?LOOM_INSTANCE is required}"
: "${LOOM_DOMAIN:?LOOM_DOMAIN is required}"

SSH_KEY="${HOME}/.ssh/google_compute_engine"
SSH_CONNECT_TIMEOUT=15
RESTART_TIMEOUT=600
READY_ATTEMPTS=90
READY_INTERVAL=10

# gcloud --quiet generates a missing key and then blocks until metadata
# propagation lets it connect. Require an existing key so activation fails
# immediately instead of stalling the rollout.
if [ ! -r "$SSH_KEY" ] || [ ! -r "${SSH_KEY}.pub" ]; then
  echo "no Compute Engine SSH key at ${SSH_KEY}; run 'gcloud compute ssh ${LOOM_INSTANCE} --zone=${LOOM_ZONE} --project=${LOOM_PROJECT}' once to create and propagate one" >&2
  exit 1
fi

restart_status=0
timeout "$RESTART_TIMEOUT" gcloud --project="$LOOM_PROJECT" compute ssh "$LOOM_INSTANCE" \
  --zone="$LOOM_ZONE" --quiet \
  --ssh-key-file="$SSH_KEY" \
  --ssh-flag="-oBatchMode=yes" \
  --ssh-flag="-oConnectTimeout=${SSH_CONNECT_TIMEOUT}" \
  --command='
    set -euo pipefail
    sudo rm -f /run/loom-startup-succeeded
    sudo systemctl restart google-startup-scripts.service
    sudo test -f /run/loom-startup-succeeded
  ' || restart_status=$?

if [ "$restart_status" -eq 124 ]; then
  echo "startup-script restart on ${LOOM_INSTANCE} exceeded ${RESTART_TIMEOUT}s; ${SSH_KEY} is likely not propagated to the instance" >&2
  exit 1
fi
if [ "$restart_status" -ne 0 ]; then
  echo "startup-script restart on ${LOOM_INSTANCE} failed with exit ${restart_status}" >&2
  exit "$restart_status"
fi

for _ in $(seq 1 "$READY_ATTEMPTS"); do
  if curl -fsS "https://${LOOM_DOMAIN}/api/ready" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"'; then
    exit 0
  fi
  sleep "$READY_INTERVAL"
done

echo "Loom did not become publicly ready after activation" >&2
exit 1
