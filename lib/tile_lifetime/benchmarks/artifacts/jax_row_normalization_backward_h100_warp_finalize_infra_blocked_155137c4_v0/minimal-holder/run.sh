#!/usr/bin/env bash

set -euo pipefail

readonly revision=155137c49565590ce09232e0a67ede303ecc7911
readonly source_archive=/app/shuttle-source.tar.gz

printf '%s  %s\n' \
  '03bb2f6a04cdc81533e398090bb2f2bba8d65f31d6278d25fa059eccf8be4643' \
  "${source_archive}" | sha256sum --check
tar -xzf "${source_archive}" -C /app
exec /app/lib/tile_lifetime/benchmarks/run_jax_row_normalization_warp_finalize_h100.sh "${revision}"
