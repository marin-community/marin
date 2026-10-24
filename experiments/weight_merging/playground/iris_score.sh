#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-score-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" "marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging"
"$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io  cp -r "$PLAYGROUND_INPUT" "$work_dir/run"
if ! command -v docker >/dev/null; then
  # The Docker-access profile drops setuid/setgid capabilities, so apt cannot run.
  # Install only the pinned upstream CLI; use the worker's existing Docker daemon.
  "$work_dir/io/bin/python" - "$work_dir" <<'PYCLI'
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path
root = Path(sys.argv[1])
archive = root/'docker-27.5.1.tgz'
urllib.request.urlretrieve('https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz', archive)
print('Docker CLI archive SHA256', hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
# /tmp is mounted noexec on these workers; place the CLI on the image filesystem.
tool = Path('/usr/local/bin/docker')
with tarfile.open(archive) as source:
    tool.write_bytes(source.extractfile('docker/docker').read())
tool.chmod(0o755)
PYCLI
  export PATH="/usr/local/bin:$PATH"
fi
docker pull python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
"$work_dir/io/bin/python" -m experiments.weight_merging.playground.rescore \
  --run "$work_dir/run" --output "$work_dir/scored" --parity-items "${PLAYGROUND_PARITY_ITEMS:-0}"
if [ "${PLAYGROUND_PARITY_ITEMS:-0}" = 0 ]; then
  "$work_dir/io/bin/python" - "$work_dir/scored" <<'PYREPORT'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = json.loads((root/'run-metadata.json').read_text())
(root/'candidates.json').write_text(json.dumps(metadata['candidates'], indent=2)+'\n')
PYREPORT
  "$work_dir/io/bin/python" -m experiments.weight_merging.playground.report \
    --run "$work_dir/scored" --candidates "$work_dir/scored/candidates.json" --output "$work_dir/scored/report.md"
fi
"$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io  cp -r "$work_dir/scored" "$PLAYGROUND_OUTPUT/"
