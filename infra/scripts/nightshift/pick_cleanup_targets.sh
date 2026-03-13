#!/usr/bin/env bash
# Selects random cleanup target directories and outputs a GitHub Actions matrix JSON.
#
# Environment variables:
#   AGENT_COUNT    - Number of agents to spawn (default: 5)
#   TARGET_FOLDER  - Override: assign all agents to this folder
#   GITHUB_OUTPUT  - GitHub Actions output file (matrix=<json> is appended)
set -euo pipefail

AGENT_COUNT="${AGENT_COUNT:-5}"
OVERRIDE="${TARGET_FOLDER:-}"

# Candidate top-level source directories (excludes scripts/, infra/, docs/)
CANDIDATES=(
  "lib/marin/src/marin"
  "lib/levanter/src/levanter"
  "lib/iris/src/iris"
  "lib/haliax/src/haliax"
  "lib/zephyr/src/zephyr"
  "lib/fray/src/fray"
  "experiments"
  "tests"
)

# Build list of all subdirectories with >=3 python files
ALL_DIRS=()
for base in "${CANDIDATES[@]}"; do
  if [ -d "$base" ]; then
    while IFS= read -r dir; do
      count=$(find "$dir" -maxdepth 1 -name '*.py' | wc -l)
      if [ "$count" -ge 3 ]; then
        ALL_DIRS+=("$dir")
      fi
    done < <(find "$base" -type d -maxdepth 3)
  fi
done

if [ "${#ALL_DIRS[@]}" -eq 0 ]; then
  echo "No candidate directories found"
  exit 1
fi

# Pick AGENT_COUNT random directories (or use override)
if [ -n "$OVERRIDE" ]; then
  SELECTED=()
  for i in $(seq 1 "$AGENT_COUNT"); do
    SELECTED+=("$OVERRIDE")
  done
else
  readarray -t SELECTED < <(printf '%s\n' "${ALL_DIRS[@]}" | shuf -n "$AGENT_COUNT")
fi

MATRIX='{"include":['
for i in $(seq 0 $(( ${#SELECTED[@]} - 1 ))); do
  FOLDER="${SELECTED[$i]}"
  AGENT_NUM=$((i + 1))

  # Generate a haiku seed for the agent
  HAIKU_SEED=$(od -An -tx4 -N4 /dev/urandom | tr -d ' ')

  if [ "$AGENT_NUM" -gt 1 ]; then MATRIX+=','; fi
  MATRIX+="{\"agent_id\":$AGENT_NUM,\"folder\":\"$FOLDER\",\"haiku_seed\":\"$HAIKU_SEED\"}"
done
MATRIX+=']}'

echo "matrix=$MATRIX" >> "$GITHUB_OUTPUT"
echo "Selected folders:"
echo "$MATRIX" | python3 -m json.tool
