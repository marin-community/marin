#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ARTIFACT_DIR="$ROOT/reports/spec_repair_v0_artifacts"
DEST="${1:-$ROOT}"

cd "$ROOT"
sha256sum -c "$ARTIFACT_DIR/chunks.sha256"
cat "$ARTIFACT_DIR"/chunks/spec_repair_v0.tar.zst.b64.part-* \
  | base64 -d \
  | zstd -d \
  | tar -xf - -C "$DEST"

