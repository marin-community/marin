#!/usr/bin/env bash
set -euo pipefail

panel_root=${1:?pass the native response panel root}
probe_dir=${2:?pass the frozen probe directory}
[[ "$panel_root" == s3://marin-us-east-08a/* && "$probe_dir" == s3://marin-rno2a/* ]] || {
  echo "rate-control output must be in marin-us-east-08a and frozen probes in marin-rno2a" >&2
  exit 2
}

uv run python -m experiments.post_training.phase1b.align_native_responses \
  --checkpoint s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845 \
  --probe-dir "$probe_dir" --response-dir "$panel_root/continuations" \
  --sets P_R --out "$panel_root/aligned-S38"

uv run python -m experiments.post_training.phase1b.filter_common_context \
  --named-input theta "$panel_root/continuations/P_R.jsonl" \
  --named-input S38 "$panel_root/aligned-S38/P_R.jsonl" \
  --require-ended --max-tokens 32768 --out "$panel_root/common-S38"
