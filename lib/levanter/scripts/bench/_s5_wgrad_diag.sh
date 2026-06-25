#!/usr/bin/env bash
# GFP8-028 M2 diagnostic: split the f8-wgrad regression into transpose vs kernel cost, per wgrad GEMM.
# Toolchain (cw-us-east-02a, jax[cuda13]==0.10.0): upgrade cuDNN 9.12 in place + put nvidia/*/lib on
# LD_LIBRARY_PATH before python (GFP8-027).
set +e
B=lib/levanter/scripts/bench/bench_ragged_wgrad_diag.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
echo "### upgrade cuDNN -> 9.12 in the synced venv ($VENV_PY)"
uv pip install --python "$VENV_PY" 'nvidia-cudnn-cu13==9.12.0.46'
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "### cuDNN sanity"
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('ok', jax.jit(lambda a: (a+1).sum())(jnp.arange(8)))" || { echo "### cuDNN sanity FAILED"; exit 1; }

echo "### WGRAD DIAGNOSTIC (transpose vs kernel vs bf16 ref, real Grug shapes)"
uv run --no-sync python -u "$B"
RC=$?
echo "### DONE rc=$RC"
exit $RC
