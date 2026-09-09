#!/usr/bin/env bash
set -euo pipefail
for sizes in "0,100000,162043" "1,1,262041" "0,0,262043" "262043,0,0" "86153,108681,106632"; do
  echo "##### sizes $sizes"
  VARIANTS=xla-cudnn bash lib/levanter/scripts/bench/run_ragged_bench.sh --group-sizes "$sizes" --legs fwd_w13,fwd_w2,dx_w13,dw13,dw2 || echo "##### sizes $sizes: runner exited $?"
done
