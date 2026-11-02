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
"$python" -c "import flash_attn, flash_attn_2_cuda"
if [[ "$profile" == megatron || "$profile" == megatron-export ]]; then
  "$python" -c "import transformer_engine.common; from megatron.bridge import AutoBridge"
fi
if [[ "$profile" == *-export ]]; then
  "$python" -c "import ray, torch; from skyrl_train.checkpoint_exporter import CheckpointExporter"
  exit 0
fi
"$python" - <<'PY'
import memray
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from rolloutengine.engine import RolloutEngine
PY
"$python" -c "import quack.activation, torch, vllm; import vllm._C_stable_libtorch, vllm.cumem_allocator; from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE; from vllm.model_executor.models import ModelRegistry; assert GRUG_MOE_ARCHITECTURE in ModelRegistry.get_supported_archs(); print('[rl-iris] frozen runtime ready:', torch.__version__, vllm.__version__)"
