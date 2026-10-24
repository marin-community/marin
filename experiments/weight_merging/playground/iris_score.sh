#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-score-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" "marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging"
"$work_dir/io/bin/fsutil" cp -r "$PLAYGROUND_INPUT" "$work_dir/run"
if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends docker.io
fi
docker pull python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
"$work_dir/io/bin/python" -m experiments.weight_merging.playground.rescore \
  --run "$work_dir/run" --output "$work_dir/scored" --parity-items "${PLAYGROUND_PARITY_ITEMS:-0}"
"$work_dir/io/bin/fsutil" cp -r "$work_dir/scored" "$PLAYGROUND_OUTPUT/"
