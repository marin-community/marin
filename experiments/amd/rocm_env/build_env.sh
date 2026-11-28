#!/bin/bash
# Build the Marin AMD Instinct (MI300X/MI355X) training environment from scratch on a new Linux x86_64 host:
#   uv 0.13.1, CPython 3.12.3, a Marin checkout at a pinned commit, and a venv synced from that commit's uv.lock
#   with the ROCm extra (JAX 0.12.0, jax-rocm10-plugin/pjrt 0.12.0, AMD TheRock rocm-sdk 10.1.0 wheels with
#   RCCL 2.30.7, Triton 3.6.0, jax-triton 0.3.1).
# Nothing is copied from another venv. The host needs git, curl, internet access to GitHub, PyPI and
# stable.repo.amd.com, and the amdgpu driver; it does not need a system ROCm install.
#
# Usage:
#   build_env.sh --prefix DIR --repo GIT_URL_OR_PATH --commit SHA [--anp-plugin PATH] [--expect-devices N]
#
#   DIR/marin     the checkout at SHA (detached), with a COMMIT file
#   DIR/venv      the venv (DIR/venv/bin/python runs the trainer)
#   DIR/bin/uv    uv, when the host has no uv 0.13.1 on PATH
#   DIR/manifest  commit, host, uv/python versions, the full package list and the net plugin's checksum
#
# After the build, the self-test (selftest.sh) runs when the host has AMD GPUs; otherwise run it later on a GPU
# node with DIR/marin/experiments/amd/rocm_env/selftest.sh DIR/venv/bin/python --expect-devices 8.
# --anp-plugin names the host's Pensando ANP RCCL net plugin (librccl-anp.so). It is a host component tied to the
# NIC driver, so it is recorded (path and sha256), not copied. Without it the self-test uses RCCL's built-in
# transports.
set -euo pipefail

UV_VERSION=0.13.1
PYTHON_VERSION=3.12.3
RCCL_VERSION_CODE=23007  # 2.30.7, as reported by ncclGetVersion
# The JAX/ROCm stack of the rocm extra, as validated on MI355X. The build fails if uv.lock resolves anything else.
STACK_PINS=(
  jax==0.12.0
  jaxlib==0.12.0
  jax-rocm10-plugin==0.12.0
  jax-rocm10-pjrt==0.12.0
  jax-triton==0.3.1
  triton==3.6.0
  rocm==10.1.0
  rocm-sdk-core==10.1.0
  rocm-sdk-libraries==10.1.0
  rocm-sdk-device-gfx942==10.1.0
  rocm-sdk-device-gfx950==10.1.0
)

PREFIX= REPO= COMMIT= ANP_PLUGIN= EXPECT_DEVICES=8
while [ $# -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --commit) COMMIT=$2; shift 2 ;;
    --anp-plugin) ANP_PLUGIN=$2; shift 2 ;;
    --expect-devices) EXPECT_DEVICES=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$PREFIX" ] && [ -n "$REPO" ] && [ -n "$COMMIT" ] || { sed -n 2,22p "$0" >&2; exit 2; }
[ "$(uname -sm)" = "Linux x86_64" ] || { echo "the ROCm wheels exist only for Linux x86_64" >&2; exit 1; }
[ -e "$PREFIX/venv" ] && { echo "$PREFIX/venv exists; build into a new prefix" >&2; exit 1; }
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
log() { echo "=== $* ($(date -u +%FT%T))"; }

# uv, pinned. Its cache goes under the prefix unless UV_CACHE_DIR is set (home directories are often small).
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PREFIX/uv-cache}
# Cache and venv may sit on different filesystems, where hard links fail.
export UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$PREFIX/python
UV=$(command -v uv || true)
if [ -z "$UV" ] || [ "$("$UV" --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  log "installing uv $UV_VERSION into $PREFIX/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$PREFIX/bin" UV_NO_MODIFY_PATH=1 sh
  UV=$PREFIX/bin/uv
fi
"$UV" --version

log "checking out $COMMIT from $REPO"
git clone --quiet --no-checkout "$REPO" "$PREFIX/marin"
git -C "$PREFIX/marin" -c advice.detachedHead=false checkout --quiet "$COMMIT"
git -C "$PREFIX/marin" rev-parse HEAD > "$PREFIX/marin/COMMIT"

log "syncing $PREFIX/venv from uv.lock"
# --all-packages installs every workspace member editable; --no-default-groups leaves out the dev group, which
# conflicts with the rocm extra.
(cd "$PREFIX/marin" && UV_PROJECT_ENVIRONMENT=$PREFIX/venv "$UV" sync --frozen --all-packages --extra rocm \
  --no-default-groups --python "$PYTHON_VERSION" --python-preference managed)
PY=$PREFIX/venv/bin/python

log "checking pins"
FREEZE=$("$UV" pip freeze --python "$PY")
fail=0
for pin in "${STACK_PINS[@]}"; do
  grep -qx "$pin" <<<"$FREEZE" || { echo "expected $pin, got: $(grep -i "^${pin%%==*}==" <<<"$FREEZE" || echo none)"; fail=1; }
done
got_python=$("$PY" -c 'import platform; print(platform.python_version())')
[ "$got_python" = "$PYTHON_VERSION" ] || { echo "expected Python $PYTHON_VERSION, got $got_python"; fail=1; }
RCCL_LIB=$PREFIX/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1
rccl_code=$("$PY" -c 'import ctypes, sys; v = ctypes.c_int(); ctypes.CDLL(sys.argv[1]).ncclGetVersion(ctypes.byref(v)); print(v.value)' "$RCCL_LIB")
[ "$rccl_code" = "$RCCL_VERSION_CODE" ] || { echo "expected RCCL $RCCL_VERSION_CODE, got $rccl_code"; fail=1; }
[ $fail -eq 0 ] || { echo "pin check failed" >&2; exit 1; }
echo "stack pins, Python $got_python and RCCL $rccl_code ok"

anp_line="none"
if [ -n "$ANP_PLUGIN" ]; then
  if [ -r "$ANP_PLUGIN" ]; then
    anp_line="$ANP_PLUGIN sha256 $(sha256sum "$ANP_PLUGIN" | cut -d' ' -f1)"
  else
    anp_line="$ANP_PLUGIN (not present on $(hostname))"
  fi
fi

{
  echo "commit $(cat "$PREFIX/marin/COMMIT")"
  echo "repo $REPO"
  echo "built $(date -u +%FT%T) on $(hostname) ($(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r))"
  echo "uv $("$UV" --version | cut -d' ' -f2)"
  echo "python $got_python"
  echo "rccl $rccl_code"
  echo "anp_plugin $anp_line"
  echo "packages:"
  echo "$FREEZE"
} > "$PREFIX/manifest"
log "wrote $PREFIX/manifest"

if [ -e /dev/kfd ]; then
  log "self-test"
  SELFTEST_ARGS=(--expect-devices "$EXPECT_DEVICES")
  [ -n "$ANP_PLUGIN" ] && SELFTEST_ARGS+=(--anp-plugin "$ANP_PLUGIN")
  "$PREFIX/marin/experiments/amd/rocm_env/selftest.sh" "$PY" "${SELFTEST_ARGS[@]}"
else
  log "no /dev/kfd on $(hostname); run the self-test on a GPU node:"
  echo "  $PREFIX/marin/experiments/amd/rocm_env/selftest.sh $PY --expect-devices $EXPECT_DEVICES" \
    "${ANP_PLUGIN:+--anp-plugin $ANP_PLUGIN}"
fi
