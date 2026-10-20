#!/usr/bin/env bash
set -euo pipefail

payload="$(cd -- "$(dirname -- "$0")" && pwd)"
environment="${IRIS_VENV:-$payload/.venv}"
runtime_extra_flags=$(python3.12 -c \
  'import json, sys; print(" ".join("--extra=" + extra for extra in json.load(open(sys.argv[1]))["runtime_extras"]))' \
  "$payload/receipt.json")
read -r -a runtime_extras <<< "$runtime_extra_flags"
UV_PROJECT_ENVIRONMENT="$environment" uv sync --frozen --no-install-project \
  --project "$payload/dependencies" --python python3.12 --no-python-downloads \
  --no-group dev "${runtime_extras[@]}" --link-mode symlink
uv pip install --python "$environment/bin/python" --no-deps "$payload/"*.whl
"$environment/bin/python" "$payload/check_dependencies.py"
"$environment/bin/python" -m marinskyrl.environment_contract write-frozen-cuda-runtime "$payload/runtime.sh"
source "$payload/runtime.sh"
export PATH="$environment/bin:$PATH"
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
  ln -s "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
fi
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$environment/lib/libcudart.so"
ln -sf "$CUDA_HOME/lib/libnvrtc.so.13" "$environment/lib/libnvrtc.so"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_USE_UVLOOP=0
uv pip install --python "$environment/bin/python" pytest==8.4.2
export PYTHONPATH="$payload/acceptance"
"$environment/bin/python" -m pytest -q -s \
  "$payload/acceptance/tests/gpu/test_megatron_worker.py::test_native_adamw_recipe_matches_torch_weight_updates" \
  "$payload/acceptance/tests/gpu/test_megatron_worker.py::test_megatron_flash_attention_cp2_forward_backward" \
  --junitxml="$IRIS_OUTPUT_DIR/megatron.xml"
"$environment/bin/python" "$payload/run.py" megatron-result
unset PYTHONPATH
exec "$environment/bin/python" "$payload/run.py" generate --limit "${1:-500}"
