#!/bin/bash
# Validate the opt-in overflow-drop patch against its pinned TE source.

set -euo pipefail

PINNED_TE_SHA=4adad4c218c115cd9af235fb3d4e13ef4cec55a8
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_PATH="$SCRIPT_DIR/transformer_engine_jax_overflow_drop.patch"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 TE_SOURCE [--check|--patched]" >&2
  exit 64
fi

TE_SOURCE=$1
MODE=${2:---check}

if ! git -C "$TE_SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FATAL: not a Transformer Engine git checkout: $TE_SOURCE" >&2
  exit 1
fi

actual_sha=$(git -C "$TE_SOURCE" rev-parse HEAD)
if [[ "$actual_sha" != "$PINNED_TE_SHA" ]]; then
  echo "FATAL: overflow-drop patch requires TE $PINNED_TE_SHA, got $actual_sha" >&2
  exit 1
fi

case "$MODE" in
  --check)
    if ! git -C "$TE_SOURCE" diff --quiet; then
      echo "FATAL: pristine patch check requires a clean TE checkout" >&2
      exit 1
    fi
    git -C "$TE_SOURCE" apply --check --whitespace=error-all "$PATCH_PATH"
    ;;
  --patched)
    git -C "$TE_SOURCE" diff --check
    git -C "$TE_SOURCE" apply --reverse --check "$PATCH_PATH"
    grep -Fq "NCCL_EP_OVERFLOW_DROP" \
      "$TE_SOURCE/3rdparty/nccl-extensions/nccl_ep/include/ep_enums.h"
    python - "$TE_SOURCE/transformer_engine/jax/ep.py" <<'PY'
import ast
import pathlib
import sys

source_path = pathlib.Path(sys.argv[1])
source = source_path.read_text()
compile(source, source_path, "exec")
module = ast.parse(source)
policy_mapping = None
bootstrap = None
for node in module.body:
    if isinstance(node, ast.Assign):
        if any(isinstance(target, ast.Name) and target.id == "_OVERFLOW_POLICIES" for target in node.targets):
            policy_mapping = ast.literal_eval(node.value)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "ep_bootstrap":
        bootstrap = node

if policy_mapping != {"auto": 0, "trap": 1, "drop": 2}:
    raise SystemExit(f"unexpected _OVERFLOW_POLICIES mapping: {policy_mapping!r}")
if bootstrap is None:
    raise SystemExit("ep_bootstrap definition not found")

positional = bootstrap.args.posonlyargs + bootstrap.args.args
defaults = dict(
    zip(
        (argument.arg for argument in positional[-len(bootstrap.args.defaults) :]),
        bootstrap.args.defaults,
        strict=True,
    )
)
overflow_default = defaults.get("overflow_policy")
if not isinstance(overflow_default, ast.Constant) or overflow_default.value != "trap":
    raise SystemExit("ep_bootstrap overflow_policy default is not 'trap'")
if "jnp.float32" not in source:
    raise SystemExit("ep_bootstrap does not accept float32 as a maximum token dtype")
PY
    ;;
  *)
    echo "FATAL: mode must be --check or --patched, got $MODE" >&2
    exit 64
    ;;
esac

echo "Transformer Engine overflow-drop patch validation passed ($MODE)"
