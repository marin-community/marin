#!/usr/bin/env bash

set -euo pipefail

WORK=/tmp/mok-vs-deepep-nsys
REMOTE_BASE=s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808
ANALYSIS_BASE="$REMOTE_BASE/analysis/r8"

mkdir -p "$WORK"

python - <<'PY'
import fsspec

fs = fsspec.filesystem("s3")
base = "s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808"
for backend in ("mok", "deepep"):
    fs.get(f"{base}/{backend}/r00000-s38vxs64.nsys-rep", f"/tmp/mok-vs-deepep-nsys/{backend}.nsys-rep")
PY

reports=(
  cuda_api_sum
  cuda_gpu_kern_sum
  cuda_gpu_mem_time_sum
  cuda_gpu_mem_size_sum
  nvtx_sum
  osrt_sum
)

for backend in mok deepep; do
  nsys export \
    --type sqlite \
    --force-overwrite true \
    --output "$WORK/$backend.sqlite" \
    "$WORK/$backend.nsys-rep"

  for report in "${reports[@]}"; do
    nsys stats \
      --report "$report" \
      --format csv \
      "$WORK/$backend.sqlite" > "$WORK/${backend}_${report}.csv"
  done

  gzip -9 -k "$WORK/$backend.sqlite"
done

python - <<'PY'
import fsspec
from pathlib import Path

fs = fsspec.filesystem("s3")
remote = "s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808/analysis/r8"
for path in Path("/tmp/mok-vs-deepep-nsys").iterdir():
    if path.suffix == ".csv" or path.name.endswith(".sqlite.gz"):
        fs.put(str(path), f"{remote}/{path.name}")
        print(path.name, path.stat().st_size)
PY
