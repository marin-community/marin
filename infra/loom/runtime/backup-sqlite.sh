#!/usr/bin/env bash
set -euo pipefail

: "${BACKUP_BUCKET:?BACKUP_BUCKET is required}"
: "${GCP_PROJECT:?GCP_PROJECT is required}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_NAME="loom-${STAMP}.sqlite"
TMP_DIR="$(mktemp -d /tmp/loom-backup.XXXXXX)"
BACKUP_PATH="${TMP_DIR}/${BACKUP_NAME}"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

sqlite3 /home/app/.weaver/weaver.db ".timeout 30000" ".backup '${BACKUP_PATH}'"
check="$(sqlite3 "$BACKUP_PATH" "PRAGMA quick_check;" | tr -d '\r')"
if [ "$check" != ok ]; then
  echo "loom backup failed integrity check: ${check}" >&2
  exit 1
fi
gzip "$BACKUP_PATH"
gcloud storage cp "${BACKUP_PATH}.gz" \
  "gs://${BACKUP_BUCKET}/sqlite/${GCP_PROJECT}/${BACKUP_NAME}.gz"
