#!/usr/bin/env bash
# Install the mcwitt/jax mixed-E4M3/E5M2 wgmma fork into the current (cw-us-east-02a
# H100) container, so the FP8 ragged-dot hybrid runs its mixed dgrad/wgrad GEMMs on the
# genuine f8 tensor core. See MIXED_FP8_FORK.md for the full rationale.
#
# The fork (https://github.com/mcwitt/jax, branch mixed-fp8-wgmma-0.10.0) carries every
# change the mixed path needs:
#   - jaxlib (C++):  Mosaic-GPU WGMMAOp::verify accepts an e4m3/e5m2 operand pair
#   - jax (python):  the wgmma.py PTX emitter emits independent .atype/.btype, and the
#                    pallas wgmma primitive gate + ragged_dot_mgpu dlhs guard allow the mix
# so this script installs the forked jaxlib wheel + the forked jax python package, with no
# runtime monkeypatching. Contrast: the pure-python shim branch (grug-fp8-shim) runs the
# same path on STOCK jaxlib via an import-time overlay + scoped verify-disable.
#
# Usage (inside the Iris task container, repo synced at /app):
#   bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh
# Optional: set JAXLIB_WHEEL=/path/to/jaxlib-...whl to skip the ~11 min jaxlib build.
set -euo pipefail

JAX_FORK_URL="${JAX_FORK_URL:-https://github.com/mcwitt/jax.git}"
JAX_FORK_BRANCH="${JAX_FORK_BRANCH:-mixed-fp8-wgmma-0.10.0}"
SRC=/root/jaxsrc

echo "== GPU =="; nvidia-smi -L 2>&1 | head -1

# 1. Forked jax source (python package + jaxlib build tree).
rm -rf "$SRC"
git clone --depth 1 --branch "$JAX_FORK_BRANCH" "$JAX_FORK_URL" "$SRC" 2>&1 | tail -1
git -C "$SRC" log --oneline -1

# 2. Forked jaxlib wheel (C++ verifier relaxation). Build unless a prebuilt wheel is given.
#    The base jaxlib is CUDA-plugin-agnostic, so the stock jax-cuda13-plugin/-pjrt 0.10.0 stay.
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
uv pip install --no-deps --force-reinstall "$WHL" 2>&1 | tail -1
SP=$(uv run --no-sync python -c "import jaxlib,os;print(os.path.dirname(os.path.dirname(jaxlib.__file__)))")
# A from-source jaxlib reports 0.10.0.dev0+selfbuilt (< 0.10.0 under PEP440), which jax's
# version gate would reject. The ABI is identical to the 0.10.0 tag, so pin the string. Verify
# the substitution took effect — a silent no-op (renamed/reformatted var, different jax revision)
# would leave the wheel reporting the dev version and brick `import jax` for the whole container.
sed -i "s|^_release_version: str = .*|_release_version: str = '0.10.0'|" "$SP/jaxlib/version.py"
if ! grep -qE "^_release_version: str = '0\.10\.0'" "$SP/jaxlib/version.py"; then
  echo "FATAL: jaxlib version pin did not apply to $SP/jaxlib/version.py — aborting before it bricks the container" >&2
  grep -nE "_release_version" "$SP/jaxlib/version.py" >&2 || true
  exit 1
fi

# 3. Forked jax python package: the wgmma PTX emitter + the relaxed pallas wgmma gate.
cp "$SRC/jax/experimental/mosaic/gpu/wgmma.py"               "$SP/jax/experimental/mosaic/gpu/wgmma.py"
cp "$SRC/jax/_src/pallas/mosaic_gpu/primitives.py"           "$SP/jax/_src/pallas/mosaic_gpu/primitives.py"
cp "$SRC/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py" "$SP/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py"

# 3b. Relax the ragged_dot_mgpu dlhs same-dtype guard (the mixed e5m2-grad x e4m3-rhs dgrad).
#     The fork branch now carries this relaxation itself (mcwitt/jax "[Mosaic GPU] Allow mixed
#     E4M3/E5M2 FP8 operands in ragged_dot_mgpu dlhs", commit 7ee3f26) as an `_is_mixed_fp8` check,
#     which the file copy at step 3 already brings in. This block is therefore now a FALLBACK for an
#     older $SRC ref predating that commit: recognize the relaxation (the fork's form or this
#     script's itemsize form) and skip; only patch the unrelaxed stock guard.
python3 - "$SP" <<'PYEOF'
import sys, os
ragged = os.path.join(sys.argv[1], "jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py")
src = open(ragged).read()
old = "  if lhs.dtype != rhs.dtype:\n"
new = "  if lhs.dtype != rhs.dtype and not (lhs.dtype.itemsize == 1 and rhs.dtype.itemsize == 1):\n"
if "_is_mixed_fp8" in src or new in src:
    print("ragged_dot_mgpu dlhs guard already relaxed (fork carries it)")
elif src.count(old) == 1:
    open(ragged, "w").write(src.replace(old, new))
    print("relaxed ragged_dot_mgpu dlhs guard")
else:
    raise SystemExit(f"FATAL: ragged_dot_mgpu dlhs guard not found exactly once (count={src.count(old)})")
PYEOF

# 4. Mosaic-GPU toolchain: ptxas/nvlink/libdevice (the cuda13 plugin loads from the uv cache,
#    so its relative ../nvidia/cu13/bin/ptxas does not resolve) + cuDNN 9.12 (jaxlib 0.10.0 is
#    built against 9.12; the synced env ships 9.10, which leaves the dnn handle null). Fail loudly
#    if the toolchain layout is not where we expect rather than dangling-symlinking past it.
N="$SP/nvidia/cu13"
# ptxas (PTX->SASS) is required to JIT mosaic kernels; nvlink/nvdisasm/fatbinary are best-effort —
# not every nvidia-cu13 wheel ships the disassembler/packager, and mosaic execution does not need them.
[[ -x "$N/bin/ptxas" ]] || { echo "FATAL: missing CUDA toolchain binary $N/bin/ptxas" >&2; exit 1; }
for t in ptxas nvlink nvdisasm fatbinary; do
  if [[ -x "$N/bin/$t" ]]; then ln -sf "$N/bin/$t" /app/.venv/bin/$t; else echo "warn: optional toolchain binary $N/bin/$t absent; skipping"; fi
done
[[ -f "$N/nvvm/libdevice/libdevice.10.bc" ]] || { echo "FATAL: missing $N/nvvm/libdevice/libdevice.10.bc" >&2; exit 1; }
ln -sf "$N/nvvm/libdevice/libdevice.10.bc" /app/libdevice.10.bc
uv pip install --python "$(uv run --no-sync python -c 'import sys;print(sys.executable)')" \
  'nvidia-cudnn-cu13==9.12.0.46' 2>&1 | tail -1

uv run --no-sync python -c "import jax,jaxlib;print('jax',jax.__version__,'jaxlib',jaxlib.__version__,'backend',jax.default_backend())"
echo MIXED_FP8_FORK_SETUP_DONE
