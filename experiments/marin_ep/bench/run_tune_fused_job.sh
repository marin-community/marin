#!/bin/bash
# Iris-job driver for tune_fused_constants.py on one GB200 tray (4 GPUs, 1 proc/GPU).
set -u
cd /app
NIGHTLY=https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry
uv sync --all-packages --extra=gpu > /tmp/uvsync.log 2>&1
uv pip install \
  "$NIGHTLY/jax/jax-0.11.1.dev20260809-py3-none-any.whl" \
  "$NIGHTLY/jaxlib/jaxlib-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "$NIGHTLY/jax-cuda13-plugin/jax_cuda13_plugin-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "$NIGHTLY/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.1.dev20260809-py3-none-manylinux_2_27_aarch64.whl" \
  >> /tmp/uvsync.log 2>&1
uv pip uninstall nvidia-nccl-cu12 >> /tmp/uvsync.log 2>&1 || true
uv pip install --no-deps --reinstall nvidia-nccl-cu13==2.30.7 >> /tmp/uvsync.log 2>&1
uv run --no-sync python -c "import jax; print('jax', jax.__version__)" || { tail -20 /tmp/uvsync.log; exit 1; }
echo SETUP_OK

export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export MARIN_EP_COLLECTIVE_MEMORY_MB=61440
export XLA_FLAGS="--xla_gpu_ragged_all_to_all_mode=symmetric --xla_gpu_enable_dynamic_slice_fusion=false"

port=9973
for cfg in 8,6,4 4,6,4 16,6,4 8,4,4 8,12,4 4,12,4 8,6,3 8,6,5; do
  port=$((port + 1))
  pids=()
  for i in 0 1 2 3; do
    MARIN_EP_TUNE_CFG=$cfg \
    MARIN_EP_COORD=127.0.0.1:$port MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=$i \
    CUDA_VISIBLE_DEVICES=$i \
      timeout 1500 uv run --no-sync python experiments/marin_ep/bench/tune_fused_constants.py \
      > /tmp/tune_${cfg//,/}_$i.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  grep -hE "sr=|FAIL" /tmp/tune_${cfg//,/}_0.log | head -2
done
echo SWEEP_COMPLETED
