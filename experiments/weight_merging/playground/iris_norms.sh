#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-norms-XXXXXX)
uv --no-config venv "$work_dir/env" --python 3.12
uv --no-config pip install --python "$work_dir/env/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv --no-config pip install --python "$work_dir/env/bin/python" safetensors huggingface_hub "marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging"
"$work_dir/env/bin/python" - "$work_dir" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
anchor = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                           allow_patterns=['*.json', '*.safetensors'])
donor = snapshot_download('EphAsad/Aristaeus', revision='f6e918c83cda9aa8ec57ab8d57e9a557b8190b00',
                          allow_patterns=['*.json', '*.safetensors'])
subprocess.run([sys.executable, '-m', os.environ.get('PLAYGROUND_NORM_MODULE', 'experiments.weight_merging.playground.dose_response'),
                '--anchor', anchor, '--donor', donor, '--output', str(root/'recipes.json'),
                '--metrics', str(root/'norms.json')], check=True)
PY
"$work_dir/env/bin/python" -m experiments.weight_merging.playground.cloud_io  cp "$work_dir/recipes.json" "$work_dir/norms.json" "$PLAYGROUND_OUTPUT/"
