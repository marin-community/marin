#!/usr/bin/env bash
# File the upstream issue bodies in this directory on marin-community/marin.
# Usage: ./file_issues.sh [--dry-run]
# Each body's first line is "# <title>"; the rest is the issue body.
set -euo pipefail

REPO="marin-community/marin"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

# file|comma-separated labels, in filing order.
ISSUES=(
  "taskcompendium/verifier-status-and-detail.md|post-training,agent-generated"
  "taskcompendium/json-submission-fence.md|bug,post-training,agent-generated"
  "verifyit/candidate-modes.md|post-training,agent-generated"
  "verifyit/judge-output-budget.md|post-training,agent-generated"
  "verifyit/judge-criterion-samples.md|post-training,agent-generated"
  "verifyit/weighted-gated-aggregation.md|post-training,agent-generated"
  "rolloutengine/grade-supplied-state.md|post-training,agent-generated"
)

for entry in "${ISSUES[@]}"; do
  file="$DIR/${entry%%|*}"
  labels="${entry#*|}"
  title="$(head -n 1 "$file" | sed 's/^# //')"
  body="$(mktemp)"
  tail -n +3 "$file" > "$body"
  label_args=()
  IFS=',' read -ra label_list <<< "$labels"
  for label in "${label_list[@]}"; do
    label_args+=(--label "$label")
  done
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "would create: $title"
    echo "  body: ${entry%%|*} ($(wc -l < "$body" | tr -d ' ') lines)"
    echo "  labels: $labels"
  else
    gh issue create --repo "$REPO" --title "$title" --body-file "$body" "${label_args[@]}"
  fi
  rm -f "$body"
done
