#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Compare the original math donor at c=0.1/0.25 under both serving precisions.
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-precision-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" 'marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging'
fsutil() { "$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io "$@"; }
fsutil cp -r "$PLAYGROUND_MATH_SFT/final-fp32" "$work_dir/math_sft"
fsutil cp "$PLAYGROUND_MATH_SFT/training-metadata.json" "$work_dir/math-training.json"
mkdir "$work_dir/common"
fsutil cp "$PLAYGROUND_PANEL" "$work_dir/common/panel.json"
uv --no-config venv "$work_dir/eval" --python 3.12
uv --no-config pip install --python "$work_dir/eval/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv --no-config pip install --python "$work_dir/eval/bin/python" vllm==0.6.6.post1 transformers==4.48.3 numpy==1.26.4
"$work_dir/eval/bin/python" - "$work_dir" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
sources = json.loads(Path('experiments/weight_merging/playground/models.json').read_text())
sources = {key: sources[key] for key in ('anchor', 'instruct')}
paths = {key: snapshot_download(value['repo'], revision=value['revision'],
    allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.json'])
    for key, value in sources.items()}
(root/'common/model-paths.json').write_text(json.dumps(paths, indent=2)+'\n')
(root/'common/model-sources.json').write_text(json.dumps(sources, indent=2)+'\n')
candidates = [{'name': 'instruct', 'source': 'instruct'}]
for coefficient in (0.1, 0.25):
    candidates.append({'name': f'precision-math-c{coefficient}', 'source': 'instruct',
        'donors': ['math_sft'], 'recipe': {'method': 'task_arithmetic', 'coefficients': [1.0],
        'density': 1.0, 'scale': coefficient, 'seed': 42, 'ties_density': 1.0}})
(root/'recipes.json').write_text(json.dumps(candidates, indent=2)+'\n')
PY
"$work_dir/eval/bin/python" -m experiments.weight_merging.playground.register \
  --run "$work_dir/common" --name math_sft --path "$work_dir/math_sft" \
  --training-metadata "$work_dir/math-training.json"
mkdir "$work_dir/outputs"
publish_outputs() { fsutil cp -r "$work_dir/outputs" "$PLAYGROUND_OUTPUT/"; }
trap publish_outputs EXIT
export VLLM_ATTENTION_BACKEND=XFORMERS
for dtype in bfloat16 float32; do
  run_dir="$work_dir/outputs/precision-run-$dtype"
  cp -r "$work_dir/common" "$run_dir"
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.sweep \
    --run "$run_dir" --candidates "$work_dir/recipes.json" \
    --max-new-tokens 8192 --max-model-len 12288 --max-num-seqs 64 \
    --reload-check-every 2 --dtype "$dtype" --defer-code-scoring --revision "$PLAYGROUND_REVISION"
done
