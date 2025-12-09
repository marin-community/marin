#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 REPO_ID [REVISION] [DEST_PATH]" >&2
  echo "Example: $0 google/gemma-3-27b-pt main" >&2
  exit 1
fi

REPO_ID="$1"
REVISION="${2:-main}"
DEST_PATH="${3:-}"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_TOKEN or HUGGING_FACE_HUB_TOKEN." >&2
  exit 1
fi

GCS_DEST_ROOT="${GCS_DEST_ROOT:-gs://marin-us-east1-d/gcsfuse_mount/models}"
if [[ -z "${DEST_PATH}" ]]; then
  SAFE_NAME="${REPO_ID//\//--}"
  DEST_PATH="${GCS_DEST_ROOT}/${SAFE_NAME}--${REVISION}"
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required." >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is required." >&2
  exit 1
fi

WORKDIR="$(mktemp -d /tmp/hf_download.XXXXXX)"
cleanup() {
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

echo "Downloading ${REPO_ID}@${REVISION} to ${WORKDIR}"
HF_TOKEN="${HF_TOKEN_VALUE}" huggingface-cli download "${REPO_ID}" --revision "${REVISION}" --local-dir "${WORKDIR}" --local-dir-use-symlinks False

echo "Syncing to ${DEST_PATH}"
gsutil -m rsync -r "${WORKDIR}" "${DEST_PATH}"

echo "Done. Copied ${REPO_ID}@${REVISION} to ${DEST_PATH}"
