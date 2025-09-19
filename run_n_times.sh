#!/bin/bash

set -o pipefail

usage() {
    echo "Usage: $0 --command \"<command>\" [--times <N>]" >&2
}

times=10
user_command=""

while [ $# -gt 0 ]; do
    case "$1" in
        --command|-c|--comand)
            shift
            user_command="${1:-}"
            ;;
        --times)
            shift
            times="${1:-}"
            ;;
        --help|-h|help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage; exit 1 ;;
    esac
    shift
done

if [ -z "$user_command" ]; then
    echo "Error: --command is required" >&2
    usage
    exit 1
fi

case "$times" in
    ''|*[!0-9]*) echo "Error: --times must be a non-negative integer" >&2; exit 1 ;;
esac

successes=0
failures=0

for i in $(seq 1 "$times"); do
    echo "[run $i/$times] $(date -Is) running: $user_command" >&2
    bash -e -o pipefail -c "$user_command"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[run $i/$times] success (rc=0)" >&2
        successes=$((successes + 1))
    else
        echo "[run $i/$times] failure (rc=$rc)" >&2
        failures=$((failures + 1))
    fi
done

echo "Completed $times runs: successes=$successes failures=$failures" >&2

if [ $successes -gt 0 ]; then
    exit 0
else
    exit 1
fi
