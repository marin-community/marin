#!/usr/bin/env bash
# Run the CoreWeave Iris cloud smoke test locally.
#
# Prerequisites:
#   - ~/.kube/coreweave-iris kubeconfig exists
#   - GHCR auth: docker/podman login ghcr.io
#   - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set (R2 credentials)
#   - MARIN_PREFIX set (defaults to s3://marin-na)
#
# Usage:
#   ./scripts/iris/run_smoke_cw.sh              # run tests
#   ./scripts/iris/run_smoke_cw.sh --cleanup    # tear down only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IRIS_DIR="$REPO_ROOT/lib/iris"
LABEL_PREFIX="smoke-cw-local-$(whoami)-$(date +%s | tail -c 6)"
SCREENSHOT_DIR="${IRIS_SCREENSHOT_DIR:-/tmp/iris-cloud-screenshots}"
URL_FILE="/tmp/iris-controller-url-$$"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-900}"

export MARIN_PREFIX="${MARIN_PREFIX:-s3://marin-na}"

cleanup() {
    echo "--- Tearing down smoke cluster (label=$LABEL_PREFIX) ---"
    cd "$IRIS_DIR" && uv run --group dev iris -v \
        --config=examples/smoke-cw.yaml \
        cluster stop --label "$LABEL_PREFIX" || true

    if command -v kubectl &>/dev/null && [ -f ~/.kube/coreweave-iris ]; then
        echo "--- Failsafe Kubernetes cleanup ---"
        export KUBECONFIG=~/.kube/coreweave-iris
        MANAGED_LABEL="iris-${LABEL_PREFIX}-managed"

        # Namespaced resources
        kubectl delete pods,deployments,services,configmaps \
            -n iris-smoke-cw \
            -l "${MANAGED_LABEL}=true" \
            --ignore-not-found || true

        # Cluster-scoped NodePools
        kubectl delete nodepools \
            -l "${MANAGED_LABEL}=true" \
            --ignore-not-found || true
    fi

    rm -f "$URL_FILE"
}

if [[ "${1:-}" == "--cleanup" ]]; then
    # Allow overriding the label for manual cleanup
    LABEL_PREFIX="${2:-$LABEL_PREFIX}"
    cleanup
    exit 0
fi

trap cleanup EXIT

# --- Preflight checks ---
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set (needed for R2 object storage)" >&2
        exit 1
    fi
done

if [ ! -f ~/.kube/coreweave-iris ]; then
    echo "ERROR: ~/.kube/coreweave-iris not found" >&2
    exit 1
fi

echo "=== CoreWeave Smoke Test ==="
echo "Label prefix: $LABEL_PREFIX"
echo "Screenshots:  $SCREENSHOT_DIR"
echo ""

# --- Start smoke cluster ---
echo "--- Starting smoke cluster ---"
cd "$IRIS_DIR" && uv run --group dev iris -v \
    --config=examples/smoke-cw.yaml \
    cluster start-smoke \
    --label-prefix "$LABEL_PREFIX" \
    --url-file "$URL_FILE" \
    --wait-for-workers 1 \
    --worker-timeout "$WORKER_TIMEOUT" &
START_PID=$!

for i in $(seq 1 450); do
    if [ -f "$URL_FILE" ]; then
        echo "Cluster ready!"
        break
    fi
    sleep 2
done

if [ ! -f "$URL_FILE" ]; then
    echo "ERROR: Timed out waiting for cluster" >&2
    kill $START_PID 2>/dev/null || true
    exit 1
fi

IRIS_CONTROLLER_URL="$(cat "$URL_FILE")"
echo "Controller URL: $IRIS_CONTROLLER_URL"

# --- Run smoke tests ---
echo "--- Running smoke tests ---"
mkdir -p "$SCREENSHOT_DIR"

cd "$IRIS_DIR" && IRIS_SCREENSHOT_DIR="$SCREENSHOT_DIR" PYTHONASYNCIODEBUG=1 \
    uv run --group dev python -m pytest \
    tests/e2e/test_smoke.py \
    -m e2e \
    --iris-controller-url "$IRIS_CONTROLLER_URL" \
    -o "addopts=" \
    --tb=short -v \
    --timeout=1200

echo "--- Done! Screenshots in $SCREENSHOT_DIR ---"
