#!/usr/bin/env bash
set -euo pipefail

variant=${1:?pass s92, s38, datakit, or antidoom}
root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8
probe_dir=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.5/inputs/math-validation-50
case "$variant" in
  s92)
    repo=open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92
    revision=fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab
    model_name=S92-fdc3c3c5
    max_model_len=65536
    ;;
  s38)
    repo=open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38
    revision=cfc1d845dae89b067cdc7250d0164abefa5a69cf
    model_name=S38-cfc1d845
    max_model_len=32768
    ;;
  datakit)
    repo=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05
    revision=8c76b22afc38f5e95360c83ca88008f00b5ae1e0
    model_name=Datakit-8c76b22a
    max_model_len=65536
    ;;
  antidoom)
    repo=open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04
    revision=fd07ef0f2b78a83099a0042e59c00de986e7b647
    model_name=Antidoom-fd07ef0f
    max_model_len=65536
    ;;
  *) echo "unexpected math candidate: $variant" >&2; exit 2 ;;
esac

model_uri=$root/models/$model_name
uv run fsutil verified-copy --workers 4 "hf://$repo@$revision" "$model_uri"
uv run python -m experiments.post_training.phase1b.verify_replacement_mirror \
  --repo "$repo" --revision "$revision" --mirror "$model_uri" \
  --out "/tmp/$variant-mirror-verification.json"
uv run fsutil cp "/tmp/$variant-mirror-verification.json" "$root/mirror-verification/$variant.json"
uv run python -m experiments.post_training.phase1b.native_math_continuations \
  --checkpoint "$model_uri" --name "opd-1b-$variant-parity-pilot" \
  --probe-dir "$probe_dir" --sets P_R --out "$root/parity/$variant/pilot" \
  --max-model-len "$max_model_len"
