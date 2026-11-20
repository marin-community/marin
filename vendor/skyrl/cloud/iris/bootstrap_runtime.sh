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
  megatron|megatron-export|levanter) ;;
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

if [[ "$profile" == levanter ]]; then
  # Keep rollout engines and the model-free publisher on the existing Torch lock.
  runtime_extras=(--extra megatron --extra vllm --extra telemetry)
elif [[ "$profile" == *-export ]]; then
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
if [[ "$profile" == levanter ]]; then
  root_project="$(cd "$project_root/../.." && pwd)"
  levanter_environment="$root_project/.levanter-policy-env"
  # Only the stdlib IPC client is imported by Torch; model code stays in JAX.
  export PYTHONPATH="$root_project/lib/marin/src${PYTHONPATH:+:$PYTHONPATH}"
  printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$root_project/lib/marin/src" >> "$runtime_file"
  UV_PROJECT_ENVIRONMENT="$levanter_environment" uv tool run --from "uv==$uv_version" \
    --python "$system_python" --no-python-downloads uv sync --quiet \
    --project "$root_project" \
    --python "$system_python" \
    --no-python-downloads \
    --frozen \
    --link-mode symlink \
    --no-default-groups \
    --extra pipeline
fi
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
if [[ "$profile" == levanter ]]; then
  run_check levanter-parent "$python" -c \
    "import sys; from skyrl_train.workers.levanter.levanter_worker import LevanterRayActorGroup; assert 'jax' not in sys.modules"
  run_check levanter env -u CUDA_HOME -u NVRTC_HOME -u LIBRARY_PATH -u LD_LIBRARY_PATH \
    -u LD_PRELOAD -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
    -u JAX_PLATFORM_NAME -u JAX_CUDA_VISIBLE_DEVICES -u JAX_ROCM_VISIBLE_DEVICES -u JAX_LOCAL_DEVICE_IDS \
    JAX_PLATFORMS=cuda,cpu \
    "$levanter_environment/bin/python" -c \
    "import jax; from marin.rl.levanter_worker_runtime import main; from levanter.policy_learner import SpmdPolicyLearner; assert jax.default_backend() == 'gpu', f'Native learner precheck requires CUDA GPU backend, got {jax.default_backend()}'; print('[rl-iris] native CUDA runtime ready:', jax.__version__, [(device.platform, device.device_kind) for device in jax.local_devices()])"
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
