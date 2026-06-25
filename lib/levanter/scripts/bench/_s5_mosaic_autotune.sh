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
# too-loose cudnn lower bound). Force a 9.12 overlay with `uv run --with` and compute SITE/
# LD_LIBRARY_PATH from that SAME overlay so the 9.12 libcudnn is the one the loader finds.
set +e
B=lib/levanter/scripts/bench/bench_ragged_mosaic_autotune.py

# Shared uv invocation: overlay a cuDNN >= 9.12 onto the iris-synced gpu env (no re-sync).
UVRUN=(uv run --no-sync --with 'nvidia-cudnn-cu13>=9.12,<9.13')

SITE=$("${UVRUN[@]}" python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "cudnn libs found:"; ls -1 "$SITE"/nvidia/cudnn/lib/libcudnn.so.9* 2>/dev/null
export GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"

# Fast cuDNN sanity op: fail in seconds if the 9.12 overlay didn't take, not after the full sweep.
echo "### cuDNN sanity"
"${UVRUN[@]}" python -c "import jax, jax.numpy as jnp; print('ok', jax.jit(lambda a: (a+1).sum())(jnp.arange(8)))" || { echo "### cuDNN sanity FAILED"; exit 1; }

echo "### MOSAIC FP8 BLOCK AUTOTUNE (curated grid, real Grug shapes)"
"${UVRUN[@]}" python -u "$B"
RC=$?

echo "### DONE rc=$RC"
exit $RC
