#!/usr/bin/env bash
# M0 step 3 follow-up (GFP8-026 -> autotune): re-tune the Mosaic FP8 ragged-dot block config at
# the real Grug regime (T=8192/D=2048/F=5632/E=8) after the default config (128/128/64) lost to
# bf16 e2e. Sweeps a curated block grid over the four mosaic-served GEMMs in-process and prints
# the best config + per-GEMM speedup vs the bf16-Triton baseline.
#
# Cluster toolchain (cw-us-east-02a, jax[cuda13]): put every nvidia/*/lib on LD_LIBRARY_PATH
# before python (cuDNN/etc not on the loader path -> `dnn_support != nullptr`). `uv run --no-sync`
# uses the iris-synced gpu env (jax) without re-syncing to the no-gpu root default.
set +e
B=lib/levanter/scripts/bench/bench_ragged_mosaic_autotune.py

SITE=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"

echo "### MOSAIC FP8 BLOCK AUTOTUNE (curated grid, real Grug shapes)"
uv run --no-sync python -u "$B"

echo "### DONE"
