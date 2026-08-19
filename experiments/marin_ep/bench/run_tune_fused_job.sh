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
export MARIN_EP_COLLECTIVE_MEMORY_MB=20480
export XLA_FLAGS="--xla_gpu_ragged_all_to_all_mode=symmetric --xla_gpu_enable_dynamic_slice_fusion=false"

pids=()
for i in 0 1 2 3; do
  MARIN_EP_COORD=127.0.0.1:9973 MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=$i \
  CUDA_VISIBLE_DEVICES=$i \
    timeout 9000 uv run --no-sync python experiments/marin_ep/bench/tune_fused_constants.py \
    > /tmp/tune_$i.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "==== rank0 output ===="
cat /tmp/tune_0.log
for i in 1 2 3; do
  grep -m1 -E "FAIL|Error" /tmp/tune_$i.log && { echo "---- rank $i tail ----"; tail -5 /tmp/tune_$i.log; }
done
exit $rc
