# Recipe: Agent Research Loop (MoE EP + Int8)

## Overview

This recipe captures a repeatable workflow for benchmark-oriented MoE research in this repo, using:

- `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- TPU allocation/execution via `scripts/ray/dev_tpu.py`

It is designed for short experiment loops (parameter sweeps, implementation A/Bs, and int8 toggles) and mirrors the workflow used in issues:

- https://github.com/marin-community/marin/issues/2704
- https://github.com/marin-community/marin/issues/2710

## Prerequisites

- Authenticated gcloud session for Marin project.
- Access to Ray cluster config (for example `infra/marin-us-central1.yaml`).
- Local repo synced and dependencies installed (`uv sync`).
- TPU access in a v5p-capable cluster.

## Standard Workflow

1. Allocate a dev TPU (v5p target):

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  allocate --tpu-type v5p-8 --tpu-name "$USER-moe-int8"
```

2. Run baseline EP MoE benchmark:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  --tpu-name "$USER-moe-int8" execute -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --distribution random \
    --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 \
    --shared-expert-dim 5632 --shared-fused \
    --backend gmm --impl fused_w13 \
    --parallel-mode ep --routing-pack-strategy argsort \
    --queue-mode full --bench-pass forward_backward \
    --quant-mode none --iters 3 --warmup 1
```

3. Run int8 fake-quantized MoE benchmark (same shape for direct A/B):

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  --tpu-name "$USER-moe-int8" execute -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --distribution random \
    --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 \
    --shared-expert-dim 5632 --shared-fused \
    --backend gmm --impl fused_w13 \
    --parallel-mode ep --routing-pack-strategy argsort \
    --queue-mode full --bench-pass forward_backward \
    --quant-mode int8 --iters 3 --warmup 1
```

4. Sweep top-k with int8:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  --tpu-name "$USER-moe-int8" execute -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --distribution random \
    --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk-list 1,2,4,6,8 \
    --shared-expert-dim 5632 --shared-fused \
    --backend gmm --impl fused_w13 \
    --parallel-mode ep --routing-pack-strategy argsort \
    --queue-mode full --bench-pass forward_backward \
    --quant-mode int8 --iters 3 --warmup 1
```

## Notes

- Prefer `--routing-pack-strategy argsort` unless intentionally running a routing A/B.
- Compare `--quant-mode none` vs `--quant-mode int8` at identical shapes before broader sweeps.
- For train-like throughput, use `--bench-pass forward_backward`.
- For overlap-only timing, use `--bench-pass forward` with `--queue-mode both`.
