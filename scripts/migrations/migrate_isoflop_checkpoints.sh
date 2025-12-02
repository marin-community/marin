#!/usr/bin/env bash
# Migrate legacy ISOFlop checkpoints into the static isoflop/ output tree.

set -euo pipefail

PREFIX="gs://marin-us-central2/checkpoints"
RUN_FILTER="${RUN_NAME:-}"
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage: migrate_isoflop_checkpoints.sh [--execute] [--run-name NAME] [gs://bucket/checkpoints]
  --execute   Perform the gcloud moves instead of a dry run.
  --run-name  Only migrate checkpoints whose run name matches NAME.
  --help      Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      DRY_RUN=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --run-name)
      if [[ $# -lt 2 ]]; then
        echo "--run-name requires an argument" >&2
        exit 1
      fi
      RUN_FILTER="$2"
      shift 2
      ;;
    *)
      PREFIX="$1"
      shift
      ;;
  esac
done

PREFIX="${PREFIX%/}"

echo "Migrating ISOFlop checkpoints under ${PREFIX}"
if [[ -n "${RUN_FILTER}" ]]; then
  echo "Filtering for run name: ${RUN_FILTER}"
fi

mapfile -t LEGACY_LIST < <({ gcloud storage ls "${PREFIX}" 2>/dev/null || true; } \
  | { grep -E 'isoflop-[^/]+-[0-9a-fA-F]{6}/?$' || true; })

if [[ ${#LEGACY_LIST[@]} -eq 0 ]]; then
  echo "No legacy checkpoints found."
  exit 0
fi

echo "Found ${#LEGACY_LIST[@]} legacy checkpoint directories."

matched_paths=0
skipped_paths=0

for legacy_path in "${LEGACY_LIST[@]}"; do
  [[ -n "${legacy_path}" ]] || continue
  # Strip the trailing slash so gcloud renames the prefix instead of nesting it.
  legacy_prefix="${legacy_path%/}"
  legacy_display="${legacy_prefix}/"
  run_name="${legacy_prefix##*/}"

  if [[ ! "${run_name}" =~ ^.+-[0-9a-fA-F]{6}$ ]]; then
    echo "Skipping ${legacy_path} (unexpected name)" >&2
    continue
  fi

  clean_name="${run_name::-7}"
  experiment="${clean_name}"
  if [[ "${clean_name}" =~ ^isoflop-[^-]+-d[^-]+-L[^-]+-B[^-]+-(.+)$ ]]; then
    experiment="${BASH_REMATCH[1]}"
  fi

  if [[ -n "${RUN_FILTER}" ]]; then
    experiment_suffix="${experiment#*-}"
    if [[ "${experiment}" != *"-"* ]]; then
      experiment_suffix="${experiment}"
    fi

    if [[ "${clean_name}" != "${RUN_FILTER}" && "${experiment}" != "${RUN_FILTER}" && "${experiment_suffix}" != "${RUN_FILTER}" ]]; then
      ((skipped_paths+=1))
      continue
    fi
  fi

  dest_prefix="${PREFIX}/isoflop/${clean_name}"
  dest_display="${dest_prefix}/"
  ((matched_paths+=1))
  echo "Moving ${legacy_display} -> ${dest_display}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  (dry-run) would run: gcloud storage mv \"${legacy_prefix}\" \"${dest_prefix}\""
  else
    gcloud storage mv --no-clobber "${legacy_prefix}" "${dest_prefix}"
  fi
done

if [[ -n "${RUN_FILTER}" ]]; then
  echo "Matched ${matched_paths} directories for run filter '${RUN_FILTER}' (skipped ${skipped_paths})."
else
  echo "Planned ${matched_paths} moves."
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo
  echo "Dry run complete. Re-run with --execute to migrate."
fi
