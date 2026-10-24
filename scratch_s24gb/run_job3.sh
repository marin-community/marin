#!/bin/bash
# Runs inside the iris task (repo root = cwd). Two GPUs of one GB200 node: JAX/QuACK on GPU0, torch/cuSPARSELt on GPU1.
set -x
cd scratch_s24gb
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv
(CUDA_VISIBLE_DEVICES=0 /app/.venv/bin/python -u bench_jax.py 2>&1 | grep -v "^W0000\|^I0000" | sed 's/^/[jax] /') &
(
  uv venv /tmp/tv --python 3.12 -q
  uv pip install -q --python /tmp/tv/bin/python torch triton numpy 'nvmath-python[cu13]==1.0.0' nvidia-cusparselt-cu13 \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  /tmp/tv/bin/python -c "import torch; print(torch.__version__, torch.backends.cusparselt.version())"
  uv pip list --python /tmp/tv/bin/python 2>/dev/null | grep -i "cusparselt\|nvmath\|triton"
  CUDA_VISIBLE_DEVICES=1 /tmp/tv/bin/python -u bench_v2.py 2>&1 | sed 's/^/[torch] /'
) &
wait
