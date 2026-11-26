#!/usr/bin/env bash

skyrl_task_runtime_file() {
  printf '%s/.iris-runtime-env\n' "$IRIS_VENV"
}

load_runtime_environment() {
  local record assignment name value
  while IFS= read -r -d '' record; do
    assignment="${record#*:}"
    name="${assignment%%=*}"
    value="${assignment#*=}"
    if [[ "$record" == prepend:* ]]; then
      value="$value${!name:+:${!name}}"
    fi
    export "$name=$value"
  done < "$1"
}
