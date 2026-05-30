#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Deploy the probes daemon to a single Container-Optimized OS GCP VM.
#
# Subcommands:
#   build    — docker build, tag with git sha + "latest", push to Artifact Registry
#   apply    — gcloud compute instances update-container infra-probes --container-image <tag>
#   status   — show the VM state + tail container logs
#
# One-time VM creation is documented in infra/probes/README.md.

set -euo pipefail

PROJECT="${MARIN_PROBES_PROJECT:-hai-gcp-models}"
REGION="${MARIN_PROBES_REGION:-us-central1}"
ZONE="${MARIN_PROBES_ZONE:-us-central1-b}"
VM_NAME="${MARIN_PROBES_VM:-infra-probes}"
REPO="${MARIN_PROBES_REPO:-marin}"
IMAGE_NAME="infra-probes"

ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}"

usage() {
    cat >&2 <<EOF
Usage: $0 <command>

Commands:
  build    Build the image, tag with git sha and 'latest', push to ${ARTIFACT_REGISTRY}.
  apply    Roll the prod VM '${VM_NAME}' to the 'latest' image.
  status   Print VM state and last 50 lines of container logs.

Environment overrides:
  MARIN_PROBES_PROJECT  (default: ${PROJECT})
  MARIN_PROBES_REGION   (default: ${REGION})
  MARIN_PROBES_ZONE     (default: ${ZONE})
  MARIN_PROBES_VM       (default: ${VM_NAME})
  MARIN_PROBES_REPO     (default: ${REPO})
EOF
    exit 1
}

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
    usage
fi

# Build context is infra/probes/ — no marin sibling source needed.
PROBES_DIR="$(cd "$(dirname "$0")/.." && pwd)"

build() {
    local sha
    sha="$(git -C "${PROBES_DIR}" rev-parse --short HEAD)"
    local image_sha="${ARTIFACT_REGISTRY}:${sha}"
    local image_latest="${ARTIFACT_REGISTRY}:latest"

    echo "==> Building ${image_sha}"
    docker build \
        --platform=linux/amd64 \
        -f "${PROBES_DIR}/deploy/Dockerfile" \
        -t "${image_sha}" \
        -t "${image_latest}" \
        "${PROBES_DIR}"

    echo "==> Pushing ${image_sha} and :latest"
    docker push "${image_sha}"
    docker push "${image_latest}"
}

apply() {
    local image_latest="${ARTIFACT_REGISTRY}:latest"
    echo "==> Rolling VM ${VM_NAME} (${ZONE}) to ${image_latest}"
    gcloud compute instances update-container "${VM_NAME}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --container-image="${image_latest}"
}

status() {
    echo "==> VM ${VM_NAME} (${ZONE})"
    gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --format="value(status,labels,metadata.items.filter(key=gce-container-declaration).extract(value))"
    echo
    echo "==> Last 50 lines of container logs"
    gcloud compute ssh "${VM_NAME}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --command="docker ps --filter ancestor=${ARTIFACT_REGISTRY} -q | head -1 | xargs -r docker logs --tail 50"
}

case "${cmd}" in
    build)  build ;;
    apply)  apply ;;
    status) status ;;
    -h|--help|help) usage ;;
    *) echo "unknown command: ${cmd}" >&2; usage ;;
esac
