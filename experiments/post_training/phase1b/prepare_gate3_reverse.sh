#!/usr/bin/env bash
set -euo pipefail

condition=${1:?pass theta_R, theta_J, or theta_RJ}
step=${2:?pass a verified export step}
case "$condition" in theta_R|theta_J|theta_RJ) ;; *) echo "unexpected condition: $condition" >&2; exit 2;; esac
case "$step" in 10|30|50|100|125) ;; *) echo "unexpected export step: $step" >&2; exit 2;; esac

base="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/reverse/${condition}/step_${step}"
probes=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/probes

uv run python -m experiments.post_training.phase1b.align_native_responses \
  --checkpoint s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845 \
  --probe-dir "$probes" --response-dir "$base/continuations" --sets P_R \
  --out "$base/aligned-S38"

uv run python -m experiments.post_training.phase1b.align_native_responses \
  --checkpoint s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699 \
  --probe-dir "$probes" --response-dir "$base/continuations" --sets P_J \
  --out "$base/aligned-T1"

uv run python -m experiments.post_training.phase1b.filter_common_context \
  --named-input theta "$base/continuations/P_R.jsonl" \
  --named-input S38 "$base/aligned-S38/P_R.jsonl" \
  --max-tokens 32768 --out "$base/common-S38"
