#!/usr/bin/env bash
# Build a thin, pinned mutation of the authors' Open-MOPD runtime.
# SHELLOPTS can propagate xtrace, so disable it before reading credentials.
set +x
set -euo pipefail

: "${GITSHA:?}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"

if [ "$(uname -m)" != x86_64 ]; then
  echo "Open-MOPD experiments currently target the x86_64 Iris H100 slice" >&2
  exit 2
fi
if [[ ! "$GITSHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITSHA must be the full lowercase commit SHA bundled into the Iris task" >&2
  exit 2
fi

DOCKER_CONTEXT=/app
DOCKERFILE="${DOCKER_CONTEXT}/docker/Dockerfile.open-mopd"
IMAGE_REPOSITORY="${OPEN_MOPD_IMAGE_REPOSITORY:-ghcr.io/marin-community/marinskyrl-opd-repro}"
REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/kaniko_executor_setup.sh"

if [ -z "${IRIS_TASK_ID:-}" ] || [ ! -f "$DOCKERFILE" ]; then
  echo "build_open_mopd_kaniko.sh must run inside a disposable Iris task" >&2
  exit 2
fi
if [ "$REGISTRY_HOST" = "$IMAGE_REPOSITORY" ]; then
  echo "OPEN_MOPD_IMAGE_REPOSITORY must include a registry hostname and repository path" >&2
  exit 2
fi

install_kaniko_build_packages ca-certificates curl tar
prepare_kaniko_executor_and_registry x86_64 linux/amd64 "$REGISTRY_HOST"

exec /kaniko/executor \
  --context "dir://${DOCKER_CONTEXT}" \
  --dockerfile "$DOCKERFILE" \
  --build-arg GITSHA="$GITSHA" \
  --cache=true \
  --cache-repo="${IMAGE_REPOSITORY}/cache" \
  "${KANIKO_REGISTRY_RETRY_FLAGS[@]}" \
  --destination "${IMAGE_REPOSITORY}:opd-repro-${GITSHA}"
