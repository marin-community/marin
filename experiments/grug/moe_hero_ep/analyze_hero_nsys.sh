#!/usr/bin/env bash
# CPU-only Iris entrypoint: export both hero Nsight reports to SQLite, run the standard
# `nsys stats` reports, run the custom overlap/geometry reduction, and upload everything.
#
# `nsys` is baked into the Iris task image and is not available on a laptop, which is why this
# is a job rather than a local step.
set -euo pipefail

PROFILE_ROOT=${PROFILE_ROOT:?PROFILE_ROOT must be set}
ANALYSIS_ROOT=${ANALYSIS_ROOT:?ANALYSIS_ROOT must be set}
PY=${IRIS_PYTHON:?IRIS_PYTHON must be set by the Iris task}
WORK=${IRIS_WORKDIR:-/tmp}/nsyswork
mkdir -p "$WORK"

echo "=== environment ==="
hostname
uname -m
nsys --version

reports=(
  cuda_api_sum
  cuda_gpu_kern_sum
  cuda_gpu_mem_time_sum
  cuda_gpu_mem_size_sum
  nvtx_sum
  cuda_gpu_trace
)

for arm in mok wave; do
  echo "=== $arm: locate report ==="
  uv run --no-sync fsutil ls -l "$PROFILE_ROOT/$arm/"
  name=$(uv run --no-sync fsutil ls "$PROFILE_ROOT/$arm/" | grep '\.nsys-rep$' | head -1)
  if [ -z "$name" ]; then
    echo "no .nsys-rep under $PROFILE_ROOT/$arm/" >&2
    exit 1
  fi
  echo "report: $name"
  uv run --no-sync fsutil cp "$PROFILE_ROOT/$arm/$name" "$WORK/$arm.nsys-rep"
  ls -l "$WORK/$arm.nsys-rep"
  sha256sum "$WORK/$arm.nsys-rep" | tee "$WORK/${arm}_report.sha256"
  echo "$name" > "$WORK/${arm}_report.name"

  echo "=== $arm: export sqlite ==="
  nsys export --type sqlite --force-overwrite true --output "$WORK/$arm.sqlite" "$WORK/$arm.nsys-rep"
  ls -l "$WORK/$arm.sqlite"

  for report in "${reports[@]}"; do
    echo "=== $arm: nsys stats $report ==="
    nsys stats --report "$report" --format csv "$WORK/$arm.sqlite" > "$WORK/${arm}_${report}.csv" || \
      echo "report $report failed for $arm" >&2
  done

  echo "=== $arm: custom reduction ==="
  "$PY" /app/experiments/grug/moe_hero_ep/hero_nsys_analyze.py "$WORK/$arm.sqlite" \
    --output "$WORK/${arm}_analysis.json"

  gzip -9 -k -f "$WORK/$arm.sqlite"
  rm -f "$WORK/$arm.nsys-rep"
done

echo "=== upload ==="
for f in "$WORK"/*.csv "$WORK"/*_analysis.json "$WORK"/*.sqlite.gz "$WORK"/*.sha256 "$WORK"/*.name; do
  [ -e "$f" ] || continue
  uv run --no-sync fsutil cp "$f" "$ANALYSIS_ROOT/$(basename "$f")"
  ls -l "$f"
done
echo "=== done ==="
uv run --no-sync fsutil ls -l "$ANALYSIS_ROOT/"
