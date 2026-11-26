#!/usr/bin/env bash
# Run the Marin integration pipeline against the always-on CoreWeave CI cluster.
#
# Prerequisites:
#   - ~/.kube/coreweave-iris kubeconfig exists
#   - CW_KEY_ID / CW_KEY_SECRET set (for S3-backed tests)
#
# Usage:
#   ./scripts/iris/run_smoke_cw.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAMESPACE="iris-ci"
SVC="iris-ci-controller-svc"
LOCAL_PORT=10000

# --- Preflight ---
if [ ! -f ~/.kube/coreweave-iris ]; then
    echo "ERROR: ~/.kube/coreweave-iris not found" >&2
    exit 1
fi
export KUBECONFIG=~/.kube/coreweave-iris

# --- Port-forward ---
echo "=== Port-forwarding to ${SVC} in ${NAMESPACE} ==="
kubectl port-forward -n "$NAMESPACE" "svc/${SVC}" "${LOCAL_PORT}:10000" &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT

export IRIS_CONTROLLER_URL="http://localhost:${LOCAL_PORT}"

HEALTHY=false
for i in $(seq 1 30); do
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        echo "ERROR: port-forward process died" >&2
        exit 1
    fi
    if curl -sf "${IRIS_CONTROLLER_URL}/health" > /dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    sleep 2
done
if [ "$HEALTHY" != "true" ]; then
    echo "ERROR: controller did not become healthy" >&2
    exit 1
fi
echo "Controller healthy at ${IRIS_CONTROLLER_URL}"

# --- S3 env setup ---
export AWS_ACCESS_KEY_ID="${CW_KEY_ID:?CW_KEY_ID must be set}"
export AWS_SECRET_ACCESS_KEY="${CW_KEY_SECRET:?CW_KEY_SECRET must be set}"
export AWS_ENDPOINT_URL="https://cwobject.com"
export FSSPEC_S3="{\"endpoint_url\": \"${AWS_ENDPOINT_URL}\", \"client_kwargs\": {\"region_name\": \"auto\"}, \"config_kwargs\": {\"s3\": {\"addressing_style\": \"virtual\"}}}"
export MARIN_CI_S3_PREFIX="s3://marin-us-east-08a/tmp/ttl=3d/ci"

export WANDB_MODE=disabled
export JAX_TRACEBACK_FILTERING=off

# --- Run pipeline ---
echo "=== Running Marin integration pipeline ==="
cd "$REPO_ROOT"
timeout 1800 uv run pytest tests/test_integration_test.py \
    -m integration -o "addopts=" --timeout=900 -v -s

echo "=== Done ==="
