#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Run frozen generation code one candidate at a time, committing each before the next.
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-generate-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" 'marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging'
fsutil() { "$work_dir/io/bin/python" -m experiments.weight_merging.playground.cloud_io "$@"; }
mkdir "$work_dir/common"
fsutil cp "$PLAYGROUND_PANEL" "$work_dir/common/panel.json"
fsutil cp "$PLAYGROUND_RECIPES" "$work_dir/recipes.json"
fsutil cp "$PLAYGROUND_REFERENCE" "$work_dir/reference.json"
durable() {
 "$work_dir/io/bin/python" -m experiments.weight_merging.playground.remote.durable "$@" \
  --remote "$PLAYGROUND_OUTPUT" --reference "$work_dir/reference.json" --recipes "$work_dir/recipes.json"
}
durable init
if [ -n "${PLAYGROUND_RECOVERED_RUN:-}" ]; then
 fsutil cp -r "$PLAYGROUND_RECOVERED_RUN" "$work_dir/recovered"
 durable recover --run "$work_dir/recovered"
fi
"$work_dir/io/bin/python" - "$work_dir/recipes.json" > "$work_dir/names" <<'PYNAMES'
import json
import sys
print('\n'.join(candidate['name'] for candidate in json.load(open(sys.argv[1]))))
PYNAMES
: > "$work_dir/pending"
while IFS= read -r candidate; do
 set +e
 durable check --candidate "$candidate"
 status=$?
 set -e
 case "$status" in
  0) echo "Skipping durable candidate: $candidate" ;;
  3) echo "$candidate" >> "$work_dir/pending" ;;
  *) exit "$status" ;;
 esac
done < "$work_dir/names"
if [ "${PLAYGROUND_PREFLIGHT:-0}" = 1 ]; then
 durable collect --run "$work_dir/aggregate"
 fsutil cp -r "$work_dir/aggregate" "$PLAYGROUND_OUTPUT/"
 echo "Preflight verified; missing candidates: $(wc -l < "$work_dir/pending")"
 exit 0
fi
if [ -s "$work_dir/pending" ]; then
 uv --no-config venv "$work_dir/eval" --python 3.12
 uv --no-config pip install --python "$work_dir/eval/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
 uv --no-config pip install --python "$work_dir/eval/bin/python" vllm==0.6.6.post1 transformers==4.48.3 numpy==1.26.4
 "$work_dir/eval/bin/python" - "$work_dir/common" <<'PYMODELS'
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
PYMODELS
 export VLLM_ATTENTION_BACKEND=XFORMERS
 while IFS= read -r candidate; do
  run_dir="$work_dir/candidates/$candidate"
  mkdir -p "$work_dir/candidates"
  cp -r "$work_dir/common" "$run_dir"
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.sweep \
    --run "$run_dir" --candidates "$work_dir/recipes.json" --only "$candidate" \
    --max-new-tokens 8192 --max-model-len 12288 --max-num-seqs 64 \
    --reload-check-every 5 --dtype "${PLAYGROUND_DTYPE:-bfloat16}" --defer-code-scoring --revision "$PLAYGROUND_REVISION"
  durable publish --candidate "$candidate" --run "$run_dir"
  rm -rf "$run_dir"
 done < "$work_dir/pending"
fi
durable collect --run "$work_dir/aggregate"
fsutil cp -r "$work_dir/aggregate" "$PLAYGROUND_OUTPUT/"

if [ "${PLAYGROUND_BASE_CONTROL:-0}" = 1 ]; then
 set +e
 durable check-asset --candidate base-eightshot
 status=$?
 set -e
 if [ "$status" = 3 ]; then
  if [ ! -x "$work_dir/eval/bin/python" ]; then
   uv --no-config venv "$work_dir/eval" --python 3.12
   uv --no-config pip install --python "$work_dir/eval/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   uv --no-config pip install --python "$work_dir/eval/bin/python" vllm==0.6.6.post1 transformers==4.48.3 numpy==1.26.4
  fi
  uv --no-config pip install --python "$work_dir/eval/bin/python" datasets==3.2.0 pyarrow==18.1.0
  anchor=$("$work_dir/eval/bin/python" - <<'PYBASE'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
source = json.loads(Path('experiments/weight_merging/playground/models.json').read_text())['anchor']
print(snapshot_download(source['repo'], revision=source['revision'], allow_patterns=['*.json','*.safetensors','*.jinja','merges.txt','vocab.json']))
PYBASE
)
  export VLLM_ATTENTION_BACKEND=XFORMERS
  "$work_dir/eval/bin/python" -m experiments.weight_merging.playground.controls \
   --mode base_fewshot --model "$anchor" --tokenizer "$anchor" \
   --panel "$work_dir/common/panel.json" --output "$work_dir/base-eightshot.json" --max-new-tokens 8192
  durable publish-asset --candidate base-eightshot --run "$work_dir/base-eightshot.json"
 elif [ "$status" != 0 ]; then
  exit "$status"
 fi
fi
