#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: bootstrap_runtime.sh PROJECT_ROOT ENVIRONMENT RUNTIME_FILE PROFILE INSTALL_MODE" >&2
  exit 2
fi

project_root="$1"
environment="$2"
runtime_file="$3"
profile="$4"
install_mode="$5"

case "$profile" in
  megatron|megatron-export) ;;
  *)
    echo "unsupported runtime profile: $profile" >&2
    exit 2
    ;;
esac

case "$install_mode" in
  production) dependency_group=(--no-group dev) ;;
  development) dependency_group=(--group dev) ;;
  *)
    echo "unsupported install mode: $install_mode" >&2
    exit 2
    ;;
esac

if [[ "$profile" == *-export ]]; then
  strategy="${profile%-export}"
  runtime_extras=(--extra "$strategy")
else
  runtime_extras=(--extra "$profile" --extra vllm --extra telemetry)
fi

required_python_minor="3.12"
system_python="$(command -v "python$required_python_minor" || true)"
if [[ -z "$system_python" ]]; then
  echo "required system interpreter python$required_python_minor is unavailable ($(uv --version))" >&2
  exit 1
fi

system_python_version="$($system_python -c 'import platform; print(platform.python_version())')"
if [[ "$system_python_version" != "$required_python_minor".* ]]; then
  echo "required system interpreter Python $required_python_minor, found Python $system_python_version at $system_python ($(uv --version))" >&2
  exit 1
fi

uv_version="$(cat "$project_root/.uv-version")"
UV_PROJECT_ENVIRONMENT="$environment" uv tool run --from "uv==$uv_version" \
  --python "$system_python" --no-python-downloads uv sync --quiet \
  --project "$project_root" \
  --python "$system_python" \
  --no-python-downloads \
  --frozen \
  --link-mode symlink \
  "${dependency_group[@]}" \
  "${runtime_extras[@]}"

python="$environment/bin/python"
"$python" "$project_root/marinskyrl/environment_contract.py" write-frozen-cuda-runtime "$runtime_file"
# Native JITs invoke tools such as ninja by name after bootstrap completes.
printf 'export PATH=%q${PATH:+:$PATH}\n' "$environment/bin" >> "$runtime_file"
source "$runtime_file"
# FlashInfer's CUDA 13 JIT still links through CUDA_HOME/lib64, while the
# NVIDIA wheel installs its libraries in CUDA_HOME/lib.
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
  ln -s "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
fi
# Keep the development linker name in this environment, outside uv's package cache.
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$environment/lib/libcudart.so"
ln -sf "$CUDA_HOME/lib/libnvrtc.so.13" "$environment/lib/libnvrtc.so"
# Test-only substitution for an unreleased validated wheel. The frozen setup
# installs the released vllm from the vendored lock; when the task environment
# carries MOCHI_TEST_WHEEL_URI and MOCHI_TEST_WHEEL_SHA256, download that exact
# wheel from regional object storage, verify its SHA256, and reinstall it into
# this environment without dependency resolution, before any runtime check
# imports vllm. The submitted spec pins both values; they are unset afterwards
# so nothing downstream re-triggers the substitution.
if [[ -n "${MOCHI_TEST_WHEEL_URI:-}" && -n "${MOCHI_TEST_WHEEL_SHA256:-}" ]]; then
  test_wheels_dir="$environment/wheels"
  mkdir -p "$test_wheels_dir"
  test_wheel="$test_wheels_dir/${MOCHI_TEST_WHEEL_URI##*/}"
  "$python" - "$MOCHI_TEST_WHEEL_URI" "$MOCHI_TEST_WHEEL_SHA256" "$test_wheel" <<'MOCHI_TEST_WHEEL_PY'
import hashlib
import sys

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

uri, expected, destination = sys.argv[1], sys.argv[2], sys.argv[3]
configure_coreweave_s3()
fs, key = filesystem_for(uri)
digest = hashlib.sha256()
with fs.open(key, "rb") as stream, open(destination, "wb") as out:
    for chunk in iter(lambda: stream.read(8 << 20), b""):
        digest.update(chunk)
        out.write(chunk)
actual = digest.hexdigest()
assert actual == expected, f"test wheel sha256 mismatch: {actual} != {expected}"
print(f"[rl-iris] test wheel downloaded and sha-verified: {destination} {actual}")
MOCHI_TEST_WHEEL_PY
  uv tool run --from "uv==$uv_version" --python "$system_python" --no-python-downloads \
    uv pip install --python "$python" --no-deps --reinstall "$test_wheel"
  "$python" -c 'import importlib.metadata as md; dist = md.distribution("vllm"); print("[rl-iris] test wheel installed:", dist.version)'
  unset MOCHI_TEST_WHEEL_URI MOCHI_TEST_WHEEL_SHA256
fi
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
check_pids=()
check_logs=()

run_check() {
  local name="$1"
  shift
  "$@" >"$workdir/check-$name.log" 2>&1 &
  check_pids+=("$!")
  check_logs+=("$workdir/check-$name.log")
}

run_check flash-attention "$python" -c '
import platform

# Check the attention package selected by the locked profile before starting Ray.
if platform.machine() == "aarch64":
    from flash_attn.cute.interface import flash_attn_varlen_func
else:
    from flash_attn_interface import flash_attn_varlen_func
'
if [[ "$profile" == megatron || "$profile" == megatron-export ]]; then
  run_check megatron "$python" -c "import transformer_engine.common; from megatron.bridge import AutoBridge"
fi
if [[ "$profile" == *-export ]]; then
  run_check export "$python" -c "import ray, torch; from skyrl_train.checkpoint_exporter import CheckpointExporter"
else
  run_check agents "$python" -c "import memray; from shellbox.backends.daytona.machine import DaytonaMachineFactory; from shellbox.backends.shellsim.machine import ShellSimMachineFactory; from rolloutengine.engine import RolloutEngine"
  run_check vllm "$python" -c "import quack.activation, torch, vllm; import vllm._C_stable_libtorch, vllm.cumem_allocator; from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE; from vllm.model_executor.models import ModelRegistry; assert GRUG_MOE_ARCHITECTURE in ModelRegistry.get_supported_archs(); print('[rl-iris] frozen runtime ready:', torch.__version__, vllm.__version__)"
fi

check_status=0
for index in "${!check_pids[@]}"; do
  if ! wait "${check_pids[$index]}"; then
    cat "${check_logs[$index]}" >&2
    check_status=1
  fi
done
if ((check_status)); then
  exit 1
fi
cat "${check_logs[@]}"
