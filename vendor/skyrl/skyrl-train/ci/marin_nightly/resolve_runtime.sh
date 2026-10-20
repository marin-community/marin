#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: source resolve_runtime.sh REPOSITORY_ROOT NIGHTLY_RL_ENV INSTALL_MODE [PROFILE]" >&2
  return 2
fi

repository_root="$1"
NIGHTLY_RL_ENV="$2"
install_mode="$3"
runtime_profile="${4:-megatron}"
RUNTIME_ENV_FILE="$NIGHTLY_RL_ENV/marinskyrl-runtime.sh"
PYTHON="$NIGHTLY_RL_ENV/bin/python"

echo "::: resolving the frozen MarinSkyRL runtime"
bash "$repository_root/cloud/iris/bootstrap_runtime.sh" \
  "$repository_root" \
  "$NIGHTLY_RL_ENV" \
  "$RUNTIME_ENV_FILE" \
  "$runtime_profile" \
  "$install_mode"
source "$RUNTIME_ENV_FILE"
