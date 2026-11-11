set -euo pipefail
cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app
python - <<'PY'
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
root = 's3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b'
copy(f'{root}/2026.11.08.61/routing/B', '/tmp/routes/B', recursive=True)
copy(f'{root}/2026.11.08.7/math50/B/P_R.jsonl', '/tmp/probe/P_R.jsonl')
copy(f'{root}/2026.11.08.29/pj50/B/P_J.jsonl', '/tmp/probe/P_J.jsonl')
PY
python -m experiments.post_training.phase1b.analyze_routing_overlap \
  --capture-root /tmp/routes/B \
  --math-source /tmp/probe/P_R.jsonl \
  --terminal-source /tmp/probe/P_J.jsonl \
  --model-id s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192 \
  --experts 256 \
  --out /tmp/routing-overlap.json
python - <<'PY'
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
copy('/tmp/routing-overlap.json', 's3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.62/routing-overlap.json')
PY
