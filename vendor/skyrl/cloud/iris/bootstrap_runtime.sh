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

case "$install_mode" in
  production) dependency_group=(--no-group dev) ;;
  development) dependency_group=(--group dev) ;;
  *)
    echo "unsupported install mode: $install_mode" >&2
    exit 2
    ;;
esac

required_python_minor="$(cat "$project_root/.python-version")"
system_python="$(command -v "python$required_python_minor" || true)"
if [[ -z "$system_python" ]]; then
  echo "required system interpreter python$required_python_minor is unavailable ($(uv --version))" >&2
  exit 1
fi

system_python_version="$("$system_python" --version)"
if [[ "$system_python_version" != "Python $required_python_minor".* ]]; then
  echo "required system interpreter Python $required_python_minor, found $system_python_version at $system_python ($(uv --version))" >&2
  exit 1
fi

profile_settings="$("$system_python" -I "$project_root/cloud/iris/runtime_environment.py" "$profile")"
mapfile -t settings <<< "$profile_settings"
device="${settings[0]}"
read -r -a extras <<< "${settings[1]}"
read -r -a checks <<< "${settings[2]}"
runtime_link_mode=symlink
if [[ "$device" == cpu ]]; then
  runtime_link_mode="${UV_LINK_MODE:-copy}"
fi
runtime_extras=()
for extra in "${extras[@]}"; do
  runtime_extras+=(--extra "$extra")
done

uv_version="$(cat "$project_root/.uv-version")"
UV_PROJECT_ENVIRONMENT="$environment" uv tool run --from "uv==$uv_version" \
  --python "$system_python" --no-python-downloads uv sync --quiet \
  --project "$project_root" \
  --python "$system_python" \
  --no-python-downloads \
  --frozen \
  --link-mode "$runtime_link_mode" \
  "${dependency_group[@]}" \
  "${runtime_extras[@]}"

python="$environment/bin/python"
check_script="$project_root/cloud/iris/runtime_checks.py"
"$python" -I "$project_root/marinskyrl/environment_contract.py" write-task-runtime "$runtime_file" "$device" "$project_root"
if [[ "$device" == cpu ]]; then
  for check in "${checks[@]}"; do
    "$python" -I "$check_script" "$check"
  done
  exit 0
fi
source "$project_root/cloud/iris/load_runtime_environment.sh"
load_runtime_environment "$runtime_file"
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

for check in "${checks[@]}"; do
  run_check "$check" "$python" -I "$check_script" "$check"
done

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
