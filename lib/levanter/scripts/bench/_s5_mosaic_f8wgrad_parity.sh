#!/usr/bin/env bash
# GFP8-033 M3: H100 parity + timing for the fused f8 cast-transpose weight-gradient.
# `--mosaic-wgrad fp8` routes the mosaic wgrad through the fused cast-transpose (in_q_transpose /
# cast_transpose -> _mosaic_wgrad_transposed, no XLA swapaxes); `bf16` keeps the shipped bf16 wgrad.
# Each mosaic arm prints dw13/dw2 rel_frob vs bf16 (parity must stay in the ~6-8% band) + result_json.
#
# Toolchain (cw-us-east-02a, jax[cuda13]==0.10.0): nvidia wheel libs are off the loader path and the
# synced env resolves cuDNN 9.10.2 which jaxlib 0.10.0 (built vs 9.12) rejects -> dnn_support null.
# Upgrade cuDNN 9.12 in place + put every nvidia/*/lib on LD_LIBRARY_PATH before python (GFP8-027).
set +e
B=lib/levanter/scripts/bench/bench_ragged_mosaic_hybrid_e2e.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
echo "### upgrade cuDNN -> 9.12 in the synced venv ($VENV_PY)"
uv pip install --python "$VENV_PY" 'nvidia-cudnn-cu13==9.12.0.46'
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "### cuDNN sanity"
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('ok', jax.jit(lambda a: (a+1).sum())(jnp.arange(8)))" || { echo "### cuDNN sanity FAILED"; exit 1; }

echo "### ARM 1: bf16 fwd+bwd (baseline, timing bar)"
uv run --no-sync python -u "$B" --path bf16

echo "### ARM 2: mosaic f8-hybrid, bf16 wgrad (--mosaic-wgrad bf16) — reference numerics + timing"
uv run --no-sync python -u "$B" --path mosaic --grad-dtype e4m3 --mosaic-wgrad bf16

echo "### ARM 3: mosaic f8-hybrid, f8 wgrad (--mosaic-wgrad fp8) — KEY parity (dw13/dw2) + timing"
uv run --no-sync python -u "$B" --path mosaic --grad-dtype e4m3 --mosaic-wgrad fp8
RC=$?

echo "### DONE rc=$RC"
exit $RC
