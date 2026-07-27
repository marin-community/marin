#!/usr/bin/env bash

set -euo pipefail

readonly UBX_COMMIT="db0c814185a0415cc2e23dca387fecb9282de551"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE="${NCCL_UBX_SOURCE:?Set NCCL_UBX_SOURCE to the pinned NVIDIA/nccl checkout}"
readonly OUTPUT_DIR="${NCCL_UBX_OUTPUT_DIR:-${PWD}/nccl-ubx-direct-results}"

if [[ "$(git -C "${SOURCE}" rev-parse HEAD)" != "${UBX_COMMIT}" ]]; then
  echo "NCCL_UBX_SOURCE must be checked out at ${UBX_COMMIT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

for routing in balanced learned_skew; do
  torchrun \
    --standalone \
    --nproc-per-node=8 \
    "${SCRIPT_DIR}/benchmark_nccl_ubx.py" \
    --ubx-source "${SOURCE}" \
    --routing "${routing}" \
    "$@" \
    | tee "${OUTPUT_DIR}/${routing}.jsonl"
done

echo "PASS: balanced and learned_skew both satisfied the UB-X direct admission gate."
