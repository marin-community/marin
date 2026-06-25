#!/usr/bin/env bash
# M0 step 3 H100 validation (GFP8-026): the Mosaic FP8 hybrid through Fp8RaggedDotOp end-to-end
# on the real Grug MoE expert MLP. Mosaic = f8 fwd/dgrad + bf16 wgrad (all-E4M3 recipe). Each arm
# prints numerics-vs-bf16 (forward + dx/dw13/dw2 grads) and a result_json with steady timing;
# diff the bf16 vs mosaic steady_time_s lines for the fwd and fwd+bwd speedups.
#
# Cluster toolchain (cw-us-east-02a, jax[cuda13]): the nvidia wheel libs (cuDNN etc.) are not on
# the loader path by default, so XLA fails with `dnn_support != nullptr`. Put every nvidia/*/lib on
# LD_LIBRARY_PATH before python starts (the loader reads it at process start). `uv run --no-sync`
# uses the iris-synced gpu env (jax) without re-syncing it to the no-gpu root default.
set +e
B=lib/levanter/scripts/bench/bench_ragged_mosaic_hybrid_e2e.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
# The synced gpu env now resolves nvidia-cudnn-cu13 to 9.10.2, which jaxlib 0.10.0 (built
# against 9.12) rejects -> dnn_support null -> RET_CHECK on the first GPU op (GFP8-027).
# Upgrade cuDNN in place so the all-nvidia-libs LD_LIBRARY_PATH finds 9.12.
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
uv pip install --python "$VENV_PY" 'nvidia-cudnn-cu13==9.12.0.46'
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "### ARM 1: bf16 fwd+bwd (real Grug shapes, baseline)"
uv run --no-sync python -u "$B" --path bf16

echo "### ARM 2: mosaic f8-hybrid fwd+bwd (e4m3) — KEY numerics + fwd+bwd timing"
uv run --no-sync python -u "$B" --path mosaic --grad-dtype e4m3

echo "### ARM 3: bf16 fwd-only (baseline)"
uv run --no-sync python -u "$B" --path bf16 --forward-only

echo "### ARM 4: mosaic f8-hybrid fwd-only (e4m3) — KEY fwd timing"
uv run --no-sync python -u "$B" --path mosaic --grad-dtype e4m3 --forward-only

echo "### DONE"
