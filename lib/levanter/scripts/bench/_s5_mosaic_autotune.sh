#!/usr/bin/env bash
# M0 step 3 follow-up (GFP8-026 -> autotune): re-tune the Mosaic FP8 ragged-dot block config at
# the real Grug regime (T=8192/D=2048/F=5632/E=8) after the default config (128/128/64) lost to
# bf16 e2e. Sweeps a curated block grid over the four mosaic-served GEMMs in-process and prints
# the best config + per-GEMM speedup vs the bf16-Triton baseline.
#
# Cluster toolchain (cw-us-east-02a, jax[cuda13]==0.10.0): the synced gpu env resolves
# nvidia-cudnn-cu13 to 9.10.2, but jaxlib 0.10.0's XLA was built against cuDNN 9.12 and rejects
# the older runtime (`Loaded runtime CuDNN 9.10.2 but source compiled with 9.12.0`) -> nulls
# dnn_support -> `RET_CHECK dnn_support != nullptr` on the FIRST GPU op (jax 0.10.0 ships a
# too-loose cudnn lower bound; the e2e earlier hit a cached 9.12 wheel). Upgrade cuDNN to 9.12
# IN PLACE in the synced venv, then the all-nvidia-libs LD_LIBRARY_PATH finds the 9.12 libcudnn.
set +e
B=lib/levanter/scripts/bench/bench_ragged_mosaic_autotune.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
echo "### upgrade cuDNN -> 9.12 in the synced venv ($VENV_PY)"
uv pip install --python "$VENV_PY" 'nvidia-cudnn-cu13==9.12.0.46'

export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "cudnn .so now: $(ls -1 "$SITE"/nvidia/cudnn/lib/libcudnn.so.9* 2>/dev/null | tr '\n' ' ')"
export GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"

# Fast cuDNN sanity op: fail in seconds if the upgrade didn't take, not after the full sweep.
echo "### cuDNN sanity"
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('ok', jax.jit(lambda a: (a+1).sum())(jnp.arange(8)))" || { echo "### cuDNN sanity FAILED"; exit 1; }

echo "### MOSAIC FP8 BLOCK AUTOTUNE (real Grug shapes) args=$*"
uv run --no-sync python -u "$B" "$@"
RC=$?

echo "### DONE rc=$RC"
exit $RC
