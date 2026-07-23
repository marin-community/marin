# H100x8 Transformer Engine NCCL_EP transport gate

This gate builds Transformer Engine
`4adad4c218c115cd9af235fb3d4e13ef4cec55a8` locally for `sm_90`, retains the
generated NCCL_EP JIT headers in the same task, and runs eight supervised JAX
processes with one H100 each. It does not fetch or upload build artifacts.

The fixed transport shape is EP8, 16,384 tokens per rank, hidden size 2560,
top-k 4, 64 experts, BF16 token payloads, and uniform routing. Transformer
Engine's NCCL_EP ABI requires float32 routing weights. Rank 0 emits one JSON
object containing forward and `value_and_grad` median/p10/p90 latency plus
effective remote wire GB/s for BF16 token and float32 routing-weight payloads.
Each timed sample is the slowest of the eight ranks.

## Run

Inspect the resolved phases without installing or launching anything:

```bash
bash experiments/ncclep_h100/run_gate.sh --dry-run
```

Launch exactly one H100x8 Iris task on RNO2A:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --gpu=H100x8 \
  --cpu=64 --memory=256G --disk=256G --extra=gpu --timeout=7200 \
  --job-name="ncclep-h100-ep8-gate-${STAMP}" -- \
  bash experiments/ncclep_h100/run_gate.sh
```

The launcher requires x86_64, exactly eight H100 SM90 devices with active
NVLink, NCCL 2.30.4 or newer, and an XLA preallocation fraction no greater than
0.70. It imports Transformer Engine before JAX distributed initialization,
disables XLA command buffers, and disables TE EP handle eviction for the fixed
benchmark shapes.

## Decision

The historical 22.9143 ms ring full routed-MLP forward-backward median came
from a one-process, eight-device harness. This gate uses eight processes and is
transport-only, so that value is **not** an apples-to-apples performance
comparison or evidence that either transport wins. It is only a hard sanity
bound.

Transport alone must leave material compute headroom against that bound,
defined here as at least 20%: the NCCL_EP transport-only forward-backward
median must be at most 18.33144 ms. The summary labels the reference as
`unpaired_historical_hard_sanity_bound`, reports `status: "pass"`, and exits
zero only when that threshold and all finite-value checks pass. A slower result
reports `status: "stop"` and exits 2; stop NCCL_EP integration work at that
point. A real winner/loser claim requires a later paired eight-process
TE-transport-plus-Marin-GEMM versus ring benchmark.
