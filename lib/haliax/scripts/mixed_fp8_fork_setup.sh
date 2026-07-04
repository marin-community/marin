#!/usr/bin/env bash
# Install the mixed-E4M3/E5M2 wgmma jax fork REBASED ONTO jax/jaxlib 0.10.1 into the
# current (cw-us-east-02a H100) container.
#
# The 0.10.0 fork wheel no longer installs against the repo's jax 0.10.1 (version
# gate), so this rebuilds from upstream v0.10.1 + the two fork patches
# (lib/haliax/scripts/mixed_fp8_0101/*.patch, cherry-picked locally from
# mcwitt/jax@mixed-fp8-wgmma-0.10.0 — nothing pushed to any jax remote).
#
# Usage (inside the Iris task container, repo synced at /app):
#   bash lib/haliax/scripts/mixed_fp8_fork_setup.sh
# Optional: JAXLIB_WHEEL=/path/to/jaxlib-*.whl skips the ~11 min jaxlib build (fetch it
# with FP8_WHEEL_URI=<0.10.1 uri> fp8_wheel_cache.py get).
set -euo pipefail

SRC=/root/jaxsrc
PATCH_DIR=/app/lib/haliax/scripts/mixed_fp8_0101

echo "== GPU =="; nvidia-smi -L 2>&1 | head -1

# 1. Upstream jax at the v0.10.1 tag + the two mixed-wgmma patches.
rm -rf "$SRC"
git clone --depth 1 --branch jax-v0.10.1 https://github.com/jax-ml/jax.git "$SRC" 2>&1 | tail -1
git -C "$SRC" -c user.name=build -c user.email=build@local am "$PATCH_DIR"/*.patch
git -C "$SRC" log --oneline -3 | cat

# 2. Forked jaxlib wheel (C++ WGMMAOp::verify relaxation). Build unless prebuilt given.
if [[ -n "${JAXLIB_WHEEL:-}" ]]; then
  WHL="$JAXLIB_WHEEL"
else
  curl -fsSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64
  chmod +x /usr/local/bin/bazel
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq clang lld >/dev/null 2>&1
  ( cd "$SRC" && python build/build.py build --wheels=jaxlib --verbose )
  WHL=$(ls "$SRC"/dist/jaxlib-*.whl | head -1)
fi
echo "JAXLIB_WHEEL=$WHL"

cd /app
# Heal a poisoned uv cache first (see gpu_mosaic_setup.sh) so the overlay below starts
# from clean 0.10.1 files.
if ! uv run --no-sync python -c "from jax._src.pallas.mosaic_gpu.primitives import barrier_test" 2>/dev/null; then
  echo "WARN: jax install corrupted (poisoned uv cache); force-reinstalling jax without cache"
  uv pip install --no-cache --force-reinstall --no-deps jax==0.10.1 2>&1 | tail -1
fi

uv pip install --no-deps --force-reinstall "$WHL" 2>&1 | tail -1
SP=$(uv run --no-sync python -c "import jaxlib,os;print(os.path.dirname(os.path.dirname(jaxlib.__file__)))")
# A from-source jaxlib reports 0.10.1.dev0+selfbuilt (< 0.10.1 under PEP440), which
# jax's version gate rejects; the ABI matches the v0.10.1 tag, so pin the string.
sed -i "s|^_release_version: str = .*|_release_version: str = '0.10.1'|" "$SP/jaxlib/version.py"
if ! grep -qE "^_release_version: str = '0\.10\.1'" "$SP/jaxlib/version.py"; then
  echo "FATAL: jaxlib version pin did not apply to $SP/jaxlib/version.py" >&2
  grep -nE "_release_version" "$SP/jaxlib/version.py" >&2 || true
  exit 1
fi

# 3. Forked jax python files (wgmma PTX emitter, pallas wgmma gate, ragged_dot dlhs
#    guard). --remove-destination: a plain cp writes through uv's hardlinks and poisons
#    the node-shared wheel cache for later jobs.
cp --remove-destination "$SRC/jax/experimental/mosaic/gpu/wgmma.py"               "$SP/jax/experimental/mosaic/gpu/wgmma.py"
cp --remove-destination "$SRC/jax/_src/pallas/mosaic_gpu/primitives.py"           "$SP/jax/_src/pallas/mosaic_gpu/primitives.py"
cp --remove-destination "$SRC/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py" "$SP/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py"

# 4. Mosaic-GPU toolchain: ptxas/libdevice (the cuda13 plugin's relative paths do not
#    resolve in the uv layout).
N="$SP/nvidia/cu13"
[[ -x "$N/bin/ptxas" ]] || { echo "FATAL: missing CUDA toolchain binary $N/bin/ptxas" >&2; exit 1; }
for t in ptxas nvlink nvdisasm fatbinary; do
  if [[ -x "$N/bin/$t" ]]; then ln -sf "$N/bin/$t" /app/.venv/bin/$t; else echo "warn: optional toolchain binary $N/bin/$t absent; skipping"; fi
done
[[ -f "$N/nvvm/libdevice/libdevice.10.bc" ]] || { echo "FATAL: missing $N/nvvm/libdevice/libdevice.10.bc" >&2; exit 1; }
ln -sf "$N/nvvm/libdevice/libdevice.10.bc" /app/libdevice.10.bc

uv run --no-sync python -c "import jax,jaxlib;print('jax',jax.__version__,'jaxlib',jaxlib.__version__,'backend',jax.default_backend())"
echo MIXED_FP8_0101_SETUP_DONE
