#!/bin/bash

set -o pipefail

usage() {
    echo "Usage: $0 [--command \"<command>\" | -c \"<command>\"] [--duration-hours <hours>] [--sleep-seconds <seconds>]" >&2
    echo "Aliases: --comand (typo alias for --command)" >&2
}

# Defaults
duration_hours=8
sleep_seconds=60
user_command=""

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --command|-c|--comand)
            shift
            user_command="${1:-}"
            if [ -z "$user_command" ]; then
                echo "Error: --command requires an argument" >&2
                usage
                exit 1
            fi
            ;;
        --duration-hours)
            shift
            duration_hours="${1:-}"
            ;;
        --sleep-seconds)
            shift
            sleep_seconds="${1:-}"
            ;;
        --help|-h|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

# Validate numeric inputs
case "$duration_hours" in
    ''|*[!0-9]*) echo "Error: --duration-hours must be a non-negative integer" >&2; exit 1 ;;
esac
case "$sleep_seconds" in
    ''|*[!0-9]*) echo "Error: --sleep-seconds must be a non-negative integer" >&2; exit 1 ;;
esac

end=$((SECONDS + duration_hours * 60 * 60))  # duration window from now

while [ $SECONDS -lt $end ]; do
    if [ -n "$user_command" ]; then
        bash -e -o pipefail -c "$user_command"
    else
        echo "No command provided; exiting." >&2
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo "Job succeeded!"
        break
    else
        echo "Job failed, retrying in ${sleep_seconds} seconds..."
        sleep "${sleep_seconds}"
    fi
done

if [ $SECONDS -ge $end ]; then
    echo "Time window expired; exiting." >&2
fi
