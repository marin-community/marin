#!/usr/bin/env bash
set -euxo pipefail

run_root=/tmp/shuttle-attention-training-h100-fixed
preserved_root=/tmp/shuttle-attention-training-h100-preserved
python=/app/.venv/bin/python
nvcc=/app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
ptxas=/app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas
shuttle_revision=ec9cc46259df3ce29e7ebe95675e5f1bb63d854e

rm -rf "$run_root" "$preserved_root"
mkdir -p "$run_root" "$preserved_root"

preserve_run() {
  status=$?
  trap - EXIT
  find "$run_root" -maxdepth 1 -type f -exec cp '{}' "$preserved_root"/ \;
  if [[ -d "$run_root/header-smoke" ]]; then
    cp -a "$run_root/header-smoke" "$preserved_root"/
  fi
  if [[ -d "$run_root/build" ]]; then
    cd "$run_root"
    find build -type f \
      \( -name '*.bc' -o -name '*.c' -o -name '*.cu' -o -name '*.h' -o -name '*.json' \
      -o -name '*.mlir' -o -name '*.py' -o -name '*.so' \) \
      -exec cp --parents '{}' "$preserved_root"/ \;
    cd - >/dev/null
  fi
  tar -C "$preserved_root" -czf "$run_root/preserved.tgz" .
  sha256sum "$run_root/preserved.tgz"
  "$python" - "$run_root/preserved.tgz" <<'PY'
from __future__ import annotations

import base64
import sys
from pathlib import Path

encoded = base64.b64encode(Path(sys.argv[1]).read_bytes()).decode()
print("ATTN_TRAINING_ARTIFACT_BASE64_BEGIN")
for index in range(0, len(encoded), 12_000):
    print(f"{index // 12_000:05d}:{encoded[index:index + 12_000]}")
print("ATTN_TRAINING_ARTIFACT_BASE64_END")
PY
  exit "$status"
}
trap preserve_run EXIT

export PYTHONPATH=lib/tile_lifetime/src

"$python" - <<'PY' | tee "$run_root/environment.json"
from __future__ import annotations

import importlib.metadata
import json

import jax
import jaxlib
import torch
import triton

environment = {
    "cccl": importlib.metadata.version("nvidia-cuda-cccl"),
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
}
assert environment["cccl"] == "13.3.3.4.1"
assert environment["jax"] == "0.10.1"
assert environment["jaxlib"] == "0.10.1"
assert environment["torch"].startswith("2.11.0")
assert environment["triton"] == "3.6.0"
print(json.dumps(environment, indent=2, sort_keys=True))
PY

uv pip freeze --python "$python" | sort | tee "$run_root/packages.txt"
"$python" lib/tile_lifetime/benchmarks/cuda13_header_smoke.py \
  --nvcc "$nvcc" \
  --build-directory "$run_root/header-smoke" \
  --architecture compute_90 \
  | tee "$run_root/header-smoke.json"
"$nvcc" --version | tee "$run_root/nvcc.txt"
"$ptxas" --version | tee "$run_root/ptxas.txt"
printf '%s\n' "$shuttle_revision" | tee "$run_root/revision.txt"
nvidia-smi \
  --query-gpu=name,uuid,compute_cap,driver_version,power.limit,clocks.current.sm,clocks.current.memory \
  --format=csv,noheader,nounits \
  | tee "$run_root/nvidia-smi.txt"

"$python" lib/tile_lifetime/benchmarks/jax_streaming_attention_backward_ffi_gpu.py \
  --repository . \
  --build-directory "$run_root/build" \
  --nvcc "$nvcc" \
  --architecture sm_90a \
  --sequence 2048 \
  --query-heads 32 \
  --key-value-heads 8 \
  --head-dimension 128 \
  --block-m 32 \
  --block-n 32 \
  --num-warps 8 \
  --num-stages 3 \
  --boundary training_forward_backward \
  --oracle torch_flash \
  --warmups 5 \
  --repeats 30 \
  --iterations 5 \
  --json-output "$run_root/result.json" \
  --shuttle-revision "$shuttle_revision" \
  | tee "$run_root/stdout.log"
