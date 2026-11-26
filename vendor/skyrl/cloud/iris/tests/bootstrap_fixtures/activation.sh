#!/usr/bin/env bash

set -uo pipefail

source "$1"
load_runtime_environment "$2" || exit $?
set -e
command -v ninja
printf '%s\n' "$LD_LIBRARY_PATH" "$LIBRARY_PATH" "$PATH"
