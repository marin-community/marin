#!/usr/bin/env bash
# Create or update the capability image publisher in envreg (cw-us-east-02a).
#
#   ops/publisher/deploy.sh            # build + upload pinned source, refresh Secrets, apply, verify
#   ops/publisher/deploy.sh --render DIR   # build + render into DIR only; no upload, no kubectl writes
#
# Steps: preflight -> build the deterministic source archive (its sha256 pins the
# pod template) -> upload it content-addressed under the queue prefix -> create or
# update the two Secrets from GCP Secret Manager through pipes (values never reach
# the terminal, argv, or a file) -> apply the Deployment -> wait for rollout and
# for a heartbeat from exactly this source.
#
# Environment overrides: MARIN (a Marin checkout with rigging; default
# ~/openathena/marin-construct), KUBECONFIG_PATH, CONTEXT, QUEUE_URI,
# S3_SECRET (default capability-publisher-s3, created from the same GCP
# cw-object-storage keys the construction jobs use; set S3_SECRET=cw-s3 to reuse
# the registry's existing storage Secret instead), CONCURRENCY, POLL_SECONDS.
set -euo pipefail
set +x
umask 077

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MARIN="${MARIN:-$HOME/openathena/marin-construct}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/coreweave-iris}"
CONTEXT="${CONTEXT:-marin-gpu_US-EAST-02A}"
NAMESPACE=envreg
PROJECT=hai-gcp-models
REGISTRY_HOST="${REGISTRY_HOST:-envreg.208261-marin-gpu.coreweave.app}"
QUEUE_URI="${QUEUE_URI:-s3://marin-us-east-02a/users/muchanem/capability-pipeline/publication}"
S3_SECRET="${S3_SECRET:-capability-publisher-s3}"
CONCURRENCY="${CONCURRENCY:-2}"
POLL_SECONDS="${POLL_SECONDS:-15}"
RENDER_ONLY=0
RENDER_DIR=""
if [ "${1:-}" = "--render" ]; then
  RENDER_ONLY=1
  RENDER_DIR="${2:?--render needs an output directory}"
fi

die() { echo "deploy: $*" >&2; exit 1; }
k() { kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$CONTEXT" -n "$NAMESPACE" "$@"; }
kit() { (cd "$MARIN" && PYTHONPATH="$ROOT" uv run --frozen "$HERE/publisher_kit.py" "$@"); }

command -v uv >/dev/null || die "uv is required"
[ -f "$MARIN/pyproject.toml" ] || die "MARIN checkout not found: $MARIN"
if [ "$RENDER_ONLY" = 0 ]; then
  command -v kubectl >/dev/null || die "kubectl is required"
  command -v gcloud >/dev/null || die "gcloud is required"
  [ -f "$KUBECONFIG_PATH" ] || die "kubeconfig not found: $KUBECONFIG_PATH"
  k get deployment registry -o name >/dev/null || die "cannot reach namespace $NAMESPACE in $CONTEXT"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/capability-publisher-deploy.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

SOURCE_SHA="$(kit build-source --root "$ROOT" --output "$WORK/publisher-source.tar.gz")"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{64}$ ]] || die "source build did not return a sha256"
SOURCE_URI="$QUEUE_URI/service/source/$SOURCE_SHA/publisher-source.tar.gz"
echo "deploy: source sha256 $SOURCE_SHA"
kit render-deployment --source-sha256 "$SOURCE_SHA" --source-uri "$SOURCE_URI" --queue-uri "$QUEUE_URI" \
  --s3-secret "$S3_SECRET" --registry-host "$REGISTRY_HOST" --concurrency "$CONCURRENCY" \
  --poll-seconds "$POLL_SECONDS" --output "$WORK/deployment.yaml" >/dev/null
if [ "$RENDER_ONLY" = 1 ]; then
  mkdir -p "$RENDER_DIR"
  cp "$WORK/deployment.yaml" "$WORK/publisher-source.tar.gz" "$RENDER_DIR/"
  echo "deploy: rendered into $RENDER_DIR (nothing uploaded or applied)"
  exit 0
fi

# Object-store keys for the upload (and for the S3 Secret below). Exported only
# into this process; never echoed.
CW_KEY_ID="$(gcloud secrets versions access latest --secret=cw-object-storage-key-id --project="$PROJECT" 2>/dev/null)" \
  || die "could not read cw-object-storage-key-id"
CW_KEY_SECRET="$(gcloud secrets versions access latest --secret=cw-object-storage-key-secret --project="$PROJECT" 2>/dev/null)" \
  || die "could not read cw-object-storage-key-secret"
export CW_KEY_ID CW_KEY_SECRET
kit upload-source --archive "$WORK/publisher-source.tar.gz" --uri "$SOURCE_URI"

# Registry credential: Secret Manager -> validated Secret manifest -> kubectl, all on pipes.
gcloud secrets versions access latest --secret=capability-registry-publisher --project="$PROJECT" 2>/dev/null \
  | kit render-registry-secret --expected-registry "$REGISTRY_HOST" \
  | k apply -f - || die "registry credential Secret was not applied"
if [ "$S3_SECRET" != cw-s3 ]; then
  kit render-s3-secret --name "$S3_SECRET" | k apply -f - || die "object-store Secret was not applied"
else
  k get secret cw-s3 -o name >/dev/null || die "Secret cw-s3 is absent"
fi

k apply -f "$WORK/deployment.yaml"
k rollout status deployment/capability-image-publisher --timeout=15m
kit health --queue "$QUEUE_URI" --expect-source "$SOURCE_SHA" --max-age 120 --wait 300
echo "deploy: capability-image-publisher is serving source $SOURCE_SHA"
