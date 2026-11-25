#!/usr/bin/env bash

set -euo pipefail

source "$1"
load_runtime_environment "$2"
command -v ninja
printf '%s\n' "$LD_LIBRARY_PATH" "$LIBRARY_PATH" "$PATH"
