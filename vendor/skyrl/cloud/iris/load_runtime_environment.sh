#!/usr/bin/env bash

skyrl_task_runtime_file() {
  printf '%s/.iris-runtime-env\n' "$IRIS_VENV"
}

load_runtime_environment() {
  local record='' mode assignment name value
  while IFS= read -r -d '' record; do
    mode="${record%%:*}"
    assignment="${record#*:}"
    name="${assignment%%=*}"
    value="${assignment#*=}"
    if [[ "$assignment" != *=* || ! "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
      echo "Invalid SkyRL runtime assignment" >&2
      return 1
    fi
    case "$mode" in
      prepend) value="$value${!name:+:${!name}}" ;;
      replace) ;;
      *)
        echo "Invalid SkyRL runtime assignment mode" >&2
        return 1
        ;;
    esac
    export "$name=$value"
  done < "$1" || return
  if [[ -n "$record" ]]; then
    echo "Unterminated SkyRL runtime assignment" >&2
    return 1
  fi
}
