#!/usr/bin/env bash
# GB200 job: install the candidate PJRT wheel, run stream_smoke.py with and without
# XLA_GPU_HOST_TRANSFER_STREAMS=1, and print each mode's async-stream assignment.
#   PJRT_WHEEL=<url> bash autoresearch/loop-260930-mfu30/a/stream_smoke.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT=/tmp/stream_smoke; mkdir -p "$OUT"
if [ -n "${PJRT_WHEEL:-}" ]; then
  UV_LINK_MODE=copy uv pip install --reinstall --no-deps "$PJRT_WHEEL" 2>&1 | tail -2
fi
python -c "import importlib.metadata as m; print('jax-cuda13-pjrt', m.version('jax-cuda13-pjrt'))"
export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=execution_stream_assignment=3
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
for mode in 0 1; do
  echo "===== XLA_GPU_HOST_TRANSFER_STREAMS=$mode"
  ref=()
  [ "$mode" = 1 ] && ref=(--reference "$OUT/grads0.npy")
  XLA_GPU_HOST_TRANSFER_STREAMS=$mode python stream_smoke.py "$OUT/grads$mode.npy" "${ref[@]}" > "$OUT/log$mode.txt" 2>&1 || { tail -40 "$OUT/log$mode.txt"; exit 1; }
  grep -E "step ms|bitwise" "$OUT/log$mode.txt"
  grep "Start new compute execution scope" "$OUT/log$mode.txt" | sed 's/.*instr=//' | sort | uniq | head -60
done
