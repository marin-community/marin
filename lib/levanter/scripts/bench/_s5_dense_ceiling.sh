#!/usr/bin/env bash
# GFP8-030: test whether XLA fuses an f8 transpose into the producing bf16->f8 cast (decides if the
# cast-transpose lever is ~free). cuDNN 9.12 in-place upgrade + nvidia libs on LD_LIBRARY_PATH (GFP8-027).
set +e
B=lib/levanter/scripts/bench/bench_f8_dense_ceiling.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
echo "### upgrade cuDNN -> 9.12 in the synced venv ($VENV_PY)"
uv pip install --python "$VENV_PY" 'nvidia-cudnn-cu13==9.12.0.46'
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "### cuDNN sanity"
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('ok', jax.jit(lambda a: (a+1).sum())(jnp.arange(8)))" || { echo "### cuDNN sanity FAILED"; exit 1; }

echo "### F8 DENSE CEILING (cuBLAS)"
uv run --no-sync python -u "$B" "$@"
RC=$?
echo "### DONE rc=$RC"
exit $RC
