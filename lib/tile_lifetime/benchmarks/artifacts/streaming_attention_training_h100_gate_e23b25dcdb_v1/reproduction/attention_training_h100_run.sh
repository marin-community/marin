#!/usr/bin/env bash
set -euxo pipefail

rm -rf /tmp/attn-venv /tmp/attn-training-h100 /tmp/attn-training-preserved
mkdir -p /tmp/attn-training-h100 /tmp/attn-training-preserved
uv venv /tmp/attn-venv --python 3.12
uv pip install --python /tmp/attn-venv/bin/python \
  'jax[cuda13]==0.10.1' \
  'torch==2.11.0' \
  'triton==3.6.0'

export PYTHONPATH=lib/tile_lifetime/src
NVCC=/tmp/attn-venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc

uv pip freeze --python /tmp/attn-venv/bin/python \
  | sort \
  | tee /tmp/attn-training-h100/packages.txt
nvidia-smi \
  --query-gpu=name,uuid,compute_cap,driver_version,power.limit,clocks.current.sm,clocks.current.memory \
  --format=csv,noheader,nounits \
  | tee /tmp/attn-training-h100/nvidia-smi.txt
"$NVCC" --version | tee /tmp/attn-training-h100/nvcc.txt

/tmp/attn-venv/bin/python \
  lib/tile_lifetime/benchmarks/jax_streaming_attention_backward_ffi_gpu.py \
  --repository . \
  --build-directory /tmp/attn-training-h100/build \
  --nvcc "$NVCC" \
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
  --json-output /tmp/attn-training-h100/result.json \
  --shuttle-revision e23b25dcdb \
  | tee /tmp/attn-training-h100/stdout.log

cp /tmp/attn-training-h100/result.json /tmp/attn-training-preserved/
cp /tmp/attn-training-h100/packages.txt /tmp/attn-training-preserved/
cp /tmp/attn-training-h100/nvidia-smi.txt /tmp/attn-training-preserved/
cp /tmp/attn-training-h100/nvcc.txt /tmp/attn-training-preserved/
mkdir -p /tmp/attn-training-preserved/build
find /tmp/attn-training-h100/build -type f \
  \( -name '*.bc' -o -name '*.c' -o -name '*.cu' -o -name '*.h' -o -name '*.py' \) \
  -exec cp --parents '{}' /tmp/attn-training-preserved/build/ \;
tar -C /tmp/attn-training-preserved -czf /tmp/attn-training-h100/preserved.tgz .
sha256sum /tmp/attn-training-h100/preserved.tgz
/tmp/attn-venv/bin/python - <<'PY'
from __future__ import annotations

import base64
from pathlib import Path

encoded = base64.b64encode(Path("/tmp/attn-training-h100/preserved.tgz").read_bytes()).decode()
print("ATTN_TRAINING_ARTIFACT_BASE64_BEGIN")
for index in range(0, len(encoded), 12000):
    print(f"{index // 12000:05d}:{encoded[index:index + 12000]}")
print("ATTN_TRAINING_ARTIFACT_BASE64_END")
PY
