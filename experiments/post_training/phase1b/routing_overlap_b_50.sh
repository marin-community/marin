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
copy('s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192', '/tmp/model/B', recursive=True)
copy('s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.7/math50/B/P_R.jsonl', '/tmp/probe/P_R.jsonl')
copy('s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.29/pj50/B/P_J.jsonl', '/tmp/probe/P_J.jsonl')
PY
python -m experiments.post_training.mismatch_probe.capture_routes \
  --model-path /tmp/model/B \
  --model-id s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192 \
  --set P_R=/tmp/probe/P_R.jsonl --set P_J=/tmp/probe/P_J.jsonl \
  --out /tmp/routes
python - <<'PY'
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
copy('/tmp/routes', 's3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.61/routing/B', recursive=True)
PY
