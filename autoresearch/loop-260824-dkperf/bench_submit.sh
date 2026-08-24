#!/usr/bin/env bash
# Submit one 64-rank microbench cell (16-node gang, one XLA flag config).
# usage: NAME=<cell> FLAGS="--xla_..." [WHEEL=s3://...] bench_submit.sh
set -euo pipefail
NAME="${NAME:?cell name, e.g. dkp-a1-mode-symmem}"
FLAGS="${FLAGS:?}"
WHEEL="${WHEEL:-s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-mainpatch-g8x128-20260823/jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl}"

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
cd "$REPO"

uv run iris --config lib/iris/config/marin.yaml job run \
  --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority production --replicas 16 \
  --gpu GB200x4 --cpu 32 --memory 256GB --disk 64GB --timeout 3600 \
  --extra gpu \
  --job-name "${NAME}" \
  -e IRIS_USER mwittmann \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e IRIS_PORT_JAX 32703 \
  -e WHEEL "${WHEEL}" \
  -e FLAGS "${FLAGS}" \
  -e GANG 1 \
  -- bash autoresearch/loop-260824-dkperf/bench_driver.sh
echo "submitted ${NAME}"
