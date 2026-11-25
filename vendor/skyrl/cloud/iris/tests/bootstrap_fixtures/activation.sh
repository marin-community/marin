#!/usr/bin/env bash

set -euo pipefail

source "$1" "$2"
command -v ninja
printf '%s\n' "$LD_LIBRARY_PATH" "$PATH"
