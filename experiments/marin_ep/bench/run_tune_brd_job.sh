#!/bin/bash
# Iris-job driver for tune_brd_i3072.py on one GB200 GPU.
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
uv run --no-sync python experiments/marin_ep/bench/tune_brd_i3072.py
echo TUNE_DONE
