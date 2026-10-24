#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Run the preregistered Aristaeus curve on one H100; score code separately.
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-generate-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" 'marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging'
fsutil() { "$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io "$@"; }
mkdir "$work_dir/run"
fsutil cp "$PLAYGROUND_PANEL" "$work_dir/run/panel.json"
fsutil cp "$PLAYGROUND_RECIPES" "$work_dir/recipes.json"
uv --no-config venv "$work_dir/eval" --python 3.12
uv --no-config pip install --python "$work_dir/eval/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv --no-config pip install --python "$work_dir/eval/bin/python" vllm==0.6.6.post1 transformers==4.48.3 numpy==1.26.4
"$work_dir/eval/bin/python" - "$work_dir/run" <<'PY'
import json
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
sources = json.loads(Path('experiments/weight_merging/playground/models.json').read_text())
family = os.environ.get('PLAYGROUND_FAMILY', 'aristaeus')
if family not in ('aristaeus', 'far'):
    raise ValueError(f'Unknown model family: {family}')
names = ('anchor', 'instruct', 'coder', 'math') if family == 'far' else ('anchor', 'instruct')
sources = {key: sources[key] for key in names}
if family == 'aristaeus':
    sources['aristaeus'] = {'repo': 'EphAsad/Aristaeus', 'revision': 'f6e918c83cda9aa8ec57ab8d57e9a557b8190b00'}
paths = {key: snapshot_download(value['repo'], revision=value['revision'],
    allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.json'])
    for key, value in sources.items()}
(root/'model-paths.json').write_text(json.dumps(paths, indent=2)+'\n')
(root/'model-sources.json').write_text(json.dumps(sources, indent=2)+'\n')
PY
publish_outputs() { fsutil cp -r "$work_dir/run" "$PLAYGROUND_OUTPUT/"; }
trap publish_outputs EXIT
export VLLM_ATTENTION_BACKEND=XFORMERS
"$work_dir/eval/bin/python" -m experiments.weight_merging.playground.sweep \
  --run "$work_dir/run" --candidates "$work_dir/recipes.json" \
  --max-new-tokens 8192 --max-model-len 12288 --max-num-seqs 64 \
  --reload-check-every 5 --dtype bfloat16 --defer-code-scoring --revision "$PLAYGROUND_REVISION"

if [ "${PLAYGROUND_BASE_CONTROL:-0}" = 1 ]; then
  uv --no-config pip install --python "$work_dir/eval/bin/python" datasets==3.2.0 pyarrow==18.1.0
  anchor=$("$work_dir/eval/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["anchor"])' "$work_dir/run/model-paths.json")
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.controls \
    --mode base_fewshot --model "$anchor" --tokenizer "$anchor" \
    --panel "$work_dir/run/panel.json" --output "$work_dir/run/base-eightshot.json" --max-new-tokens 8192
fi
