#!/usr/bin/env bash
# Remove the capability image publisher from envreg and verify absence.
#
#   ops/publisher/teardown.sh                  # delete the Deployment only (Secrets kept for redeploy)
#   ops/publisher/teardown.sh --delete-secrets # also delete the publisher's two Secrets
#
# Never touches the registry, cw-s3, registry-htpasswd, registry-tls, the namespace
# or any S3 data. Queue requests/returns stay in S3; construction controllers keep
# reporting pending_publication (retryable) until a publisher runs again.
set -euo pipefail
set +x

KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/coreweave-iris}"
CONTEXT="${CONTEXT:-marin-gpu_US-EAST-02A}"
NAMESPACE=envreg
S3_SECRET="${S3_SECRET:-capability-publisher-s3}"
k() { kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$CONTEXT" -n "$NAMESPACE" "$@"; }

k delete deployment capability-image-publisher --ignore-not-found=true --wait=true
resources=("deployment/capability-image-publisher")
if [ "${1:-}" = "--delete-secrets" ]; then
  k delete secret capability-registry-publisher --ignore-not-found=true --wait=true
  resources+=("secret/capability-registry-publisher")
  if [ "$S3_SECRET" != cw-s3 ]; then
    k delete secret "$S3_SECRET" --ignore-not-found=true --wait=true
    resources+=("secret/$S3_SECRET")
  fi
fi
for resource in "${resources[@]}"; do
  remaining="$(k get "$resource" --ignore-not-found=true -o name)"
  [ -z "$remaining" ] || { echo "teardown: $resource still present" >&2; exit 1; }
  echo "teardown: $resource absent"
done
pods="$(k get pods -l app.kubernetes.io/name=capability-image-publisher -o name)"
[ -z "$pods" ] || echo "teardown: publisher pods still terminating: $pods"
