#!/usr/bin/env bash
# Run Iris integration tests against the always-on CoreWeave CI cluster.
#
# Prerequisites:
#   - ~/.kube/coreweave-iris kubeconfig exists
#   - R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY set (for S3-backed tests)
#
# Usage:
#   ./scripts/iris/run_smoke_cw.sh              # run integration tests
#   ./scripts/iris/run_smoke_cw.sh --full       # also run the full pipeline test

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

CONTROLLER_URL="http://localhost:${LOCAL_PORT}"

HEALTHY=false
for i in $(seq 1 30); do
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        echo "ERROR: port-forward process died" >&2
        exit 1
    fi
    if curl -sf "${CONTROLLER_URL}/health" > /dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    sleep 2
done
if [ "$HEALTHY" != "true" ]; then
    echo "ERROR: controller did not become healthy" >&2
    exit 1
fi
echo "Controller healthy at ${CONTROLLER_URL}"

# --- S3 env setup ---
if [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ]; then
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    export AWS_ENDPOINT_URL="https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"
    export FSSPEC_S3="{\"endpoint_url\": \"${AWS_ENDPOINT_URL}\"}"
    export MARIN_CI_S3_PREFIX="s3://marin-na/temp/ci"
fi

export WANDB_MODE=disabled
export JAX_TRACEBACK_FILTERING=off

# --- Run tests ---
echo "=== Running integration tests ==="
cd "$REPO_ROOT"
uv run pytest tests/integration/iris/ \
    --controller-url "$CONTROLLER_URL" \
    -v --tb=short --timeout=600 \
    -o "addopts=" \
    -x

if [ "${1:-}" = "--full" ]; then
    echo "=== Running full integration pipeline ==="
    timeout 600 uv run tests/integration_test.py \
        --controller-url "$CONTROLLER_URL"
fi

echo "=== Done ==="
