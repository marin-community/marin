#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Gate matched RFT donors in both serving precisions against fresh controls.
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-rft-gates-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" 'marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging'
fsutil() { "$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io "$@"; }
for domain in math code; do
  if [ "$domain" = math ]; then source_prefix=$PLAYGROUND_MATH_RFT; else source_prefix=$PLAYGROUND_CODE_RFT; fi
  fsutil cp -r "$source_prefix/final-fp32" "$work_dir/${domain}_rft"
  fsutil cp "$source_prefix/training-metadata.json" "$work_dir/$domain-training.json"
done
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
(root/'recipes.json').write_text(json.dumps([{'name': name, 'source': name}
    for name in ('instruct', 'math_rft', 'code_rft')], indent=2)+'\n')
PY
for domain in math code; do
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.register \
    --run "$work_dir/common" --name "${domain}_rft" --path "$work_dir/${domain}_rft" \
    --training-metadata "$work_dir/$domain-training.json"
done
mkdir "$work_dir/outputs"
publish_outputs() { fsutil cp -r "$work_dir/outputs" "$PLAYGROUND_OUTPUT/"; }
trap publish_outputs EXIT
export VLLM_ATTENTION_BACKEND=XFORMERS
for dtype in bfloat16 float32; do
  run_dir="$work_dir/outputs/rft-run-$dtype"
  cp -r "$work_dir/common" "$run_dir"
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.sweep \
    --run "$run_dir" --candidates "$work_dir/recipes.json" \
    --max-new-tokens 8192 --max-model-len 12288 --max-num-seqs 64 \
    --reload-check-every 2 --dtype "$dtype" --defer-code-scoring --revision "$PLAYGROUND_REVISION"
done
