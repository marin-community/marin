#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-resume-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" -e lib/rigging
mkdir -p "$work_dir/code-rft"
"$work_dir/io/bin/fsutil" cp -r "$PLAYGROUND_INPUT/code-rft/checkpoint-50" "$work_dir/code-rft/"
"$work_dir/io/bin/fsutil" cp -r "$PLAYGROUND_INPUT/rft-data" "$work_dir/"
uv --no-config venv "$work_dir/train" --python 3.12
uv --no-config pip install --python "$work_dir/train/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv --no-config pip install --python "$work_dir/train/bin/python" transformers==4.57.6 datasets==4.8.5 accelerate==1.12.0 bitsandbytes==0.45.5 liger-kernel==0.6.4 numpy==2.5.4 pyarrow==26.0.0 safetensors==0.8.0 wandb==0.31.0
"$work_dir/train/bin/python" - "$work_dir" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
parent = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                          allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.json'])
(root/'parent-path.txt').write_text(parent)
p = root/'code-rft/checkpoint-50/trainer_state.json'
state = json.loads(p.read_text())
if state['global_step'] != 50:
    raise ValueError('Unexpected resume step')
original_best = state.get('best_model_checkpoint')
state['best_model_checkpoint'] = str(root/'code-rft/checkpoint-50')
p.write_text(json.dumps(state, indent=2)+'\n')
(root/'code-rft/resume-provenance.json').write_text(json.dumps({
    'original_best_checkpoint': original_best, 'resume_step': 50,
    'previous_gpu': 'RTX 3090', 'note': 'Hardware changed; bitwise training equivalence is not assumed.'}, indent=2)+'\n')
PY
parent=$(cat "$work_dir/parent-path.txt")
export WANDB_MODE=offline
export WANDB_PROJECT=weight-merging-playground
export WANDB_DIR="$work_dir/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Always publish any completed checkpoints if training exits early.
trap '"$work_dir/io/bin/fsutil" cp -r "$work_dir/code-rft" "$PLAYGROUND_OUTPUT/"' EXIT
"$work_dir/train/bin/python" -m experiments.weight_merging.playground.train \
  --parent "$parent" --data "$work_dir/rft-data" --domain code --output "$work_dir/code-rft" \
  --resume "$work_dir/code-rft/checkpoint-50" --steps 100 --learning-rate 5e-6 \
  --accumulation 8 --eval-steps 50 --revision "$PLAYGROUND_REVISION"
