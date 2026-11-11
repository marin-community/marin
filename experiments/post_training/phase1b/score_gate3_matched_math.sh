#!/usr/bin/env bash
set -euo pipefail

panel=${1:?pass B, R10, R125, RJ10, or RJ125}
case "$panel" in
  B)
    model_uri=s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
    ;;
  R10|R125|RJ10|RJ125)
    if [[ "$panel" == R* && "$panel" != RJ* ]]; then condition=theta_R; step=${panel#R}; fi
    if [[ "$panel" == RJ* ]]; then condition=theta_RJ; step=${panel#RJ}; fi
    model_uri="s3://marin-rno2a/marin/users/benfeuer/checkpoints/phase1b/sft-${condition}-lr2e-06-seed0/2026.11.10.14/exports/global_step_${step}/policy"
    ;;
  *) echo "unexpected panel: $panel" >&2; exit 2 ;;
esac

root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.16/matched-P_R
input="$root/$panel"
output="$root/scores/$panel"
exec bash experiments/post_training/phase1b/score_math_panel.sh "$model_uri" "$input" "$output"
