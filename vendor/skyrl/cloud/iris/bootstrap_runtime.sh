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

UV_PROJECT_ENVIRONMENT="$environment" uv sync --quiet \
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
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
check_pids=()
check_logs=()

run_check() {
  local name="$1"
  shift
  (
    started=$SECONDS
    status=0
    "$@" || status=$?
    printf '[rl-iris] bootstrap check %s: %ss\n' "$name" "$((SECONDS - started))"
    exit "$status"
  ) >"$workdir/check-$name.log" 2>&1 &
  check_pids+=("$!")
  check_logs+=("$workdir/check-$name.log")
}

run_check flash-attention "$python" -c "import flash_attn, flash_attn_2_cuda"
run_check megatron "$python" -c "import transformer_engine.common; from megatron.bridge import AutoBridge"
if [[ "$profile" == *-export ]]; then
  run_check export "$python" -c "import ray, torch; from skyrl_train.checkpoint_exporter import CheckpointExporter"
else
  run_check agents "$python" -c "import memray; from shellbox.backends.daytona.machine import DaytonaMachineFactory; from shellbox.backends.shellsim.machine import ShellSimMachineFactory; from rolloutengine.engine import RolloutEngine"
  run_check vllm "$python" -c "import sys; from pathlib import Path; import quack.activation, torch, vllm; import vllm._C_stable_libtorch, vllm.cumem_allocator; from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE; from vllm.model_executor.models import ModelRegistry; assert GRUG_MOE_ARCHITECTURE in ModelRegistry.get_supported_archs(); Path(sys.argv[1]).write_text(f'{torch.__version__} {vllm.__version__}')" "$workdir/runtime-versions"
fi

failed=0
for index in "${!check_pids[@]}"; do
  if wait "${check_pids[$index]}"; then
    cat "${check_logs[$index]}"
  else
    failed=1
    cat "${check_logs[$index]}" >&2
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
if [[ "$profile" == *-export ]]; then
  exit 0
fi
printf '[rl-iris] frozen runtime ready: %s\n' "$(cat "$workdir/runtime-versions")"
