# Grug MoE EP vs DP (v5p-8): Research Logbook

## Scope
- Goal: Compare current Grug MoE (`MoEMLP`) throughput against Larry baseline DP-only MoE and characterize EP-vs-DP behavior by batch regime.
- Primary metric(s): forward+backward MoE block tokens/sec.
- Constraints: dev TPU workflow only (`scripts/ray/dev_tpu.py`), `v5p-8`, `us-east5-a`.
- Related issue: https://github.com/marin-community/marin/issues/2710

## Baseline
- Date: 2026-02-25
- Code refs:
  - Branch: `codex/grug-moe-ep-sensible-defaults`
  - Bench harness: `.agents/tmp/moe_impl_compare.py`
  - Baseline source snapshot: `.agents/tmp/baseline_scaled_sharded_aab2354be.py`
- Baseline case for repeated comparison:
  - `batch=192`, `seq=128`, `hidden=1024`, `experts=64`, `topk={2,8}`

## Experiment Log
### 2026-02-25 12:13 - OOM characterization + VMEM mitigation
- Hypothesis: Prior OOM was scoped TPU VMEM pressure, likely mitigated by scoped-vmem limit override.
- Command:
```bash
RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-codex-compare-1111 execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 \
  -e BENCH_HIDDEN=1024 -e BENCH_BATCH=192 -e BENCH_WARMUP=1 -e BENCH_ITERS=3 \
  -- uv run --python 3.11 .agents/tmp/moe_impl_compare.py
```
- Config:
  - Meshes:
    - baseline DP / our DP: `(data=4, model=1)`
    - our DP+EP: `(data=2, expert=2, model=1)`
- Result:
  - Run completed successfully with no OOM.
  - topk=2:
    - baseline DP: 357,643.65 tok/s
    - our DP: 788,548.50 tok/s
    - our DP+EP: 1,078,683.69 tok/s
  - topk=8:
    - baseline DP: 137,470.73 tok/s
    - our DP: 629,355.54 tok/s
    - our DP+EP: 657,279.15 tok/s
- Interpretation:
  - Scoped VMEM OOM is mitigated by `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`.
  - At this regime, EP still beats DP (modestly at topk=8).
- Next action:
  - Push batch by 10x and check whether EP scaling trend holds.

### 2026-02-25 12:21 - 10x batch scaling, EP axis size 2
- Hypothesis: At very large token load, EP communication may dominate and DP may overtake EP.
- Command:
```bash
RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-codex-compare-1111 execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 \
  -e BENCH_HIDDEN=1024 -e BENCH_BATCH=1920 -e BENCH_WARMUP=0 -e BENCH_ITERS=1 \
  -e BENCH_EP_AXIS_SIZE=2 \
  -- uv run --python 3.11 .agents/tmp/moe_impl_compare.py
```
- Config:
  - EP mesh: `(data=2, expert=2, model=1)`
- Result:
  - topk=2:
    - baseline DP: 600,291.12 tok/s
    - our DP: 4,079,560.53 tok/s
    - our DP+EP: 3,410,399.68 tok/s
  - topk=8:
    - baseline DP: 158,733.36 tok/s
    - our DP: 1,546,113.18 tok/s
    - our DP+EP: 1,090,368.58 tok/s
- Interpretation:
  - At 10x batch, our DP outperforms our EP on this 4-chip v5p-8 setup.
  - EP still strongly beats baseline DP.
- Next action:
  - Double-check `expert=4` interpretation via EP axis size 4 on the same shape.

### 2026-02-25 12:27 - 10x batch scaling, EP axis size 4
- Hypothesis: Changing EP partition to expert-axis size 4 may materially change EP result.
- Command:
```bash
RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-codex-compare-1111 execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 \
  -e BENCH_HIDDEN=1024 -e BENCH_BATCH=1920 -e BENCH_WARMUP=0 -e BENCH_ITERS=1 \
  -e BENCH_EP_AXIS_SIZE=4 \
  -- uv run --python 3.11 .agents/tmp/moe_impl_compare.py
```
- Config:
  - EP mesh: `(data=1, expert=4, model=1)`
- Result:
  - topk=2:
    - baseline DP: 600,187.55 tok/s
    - our DP: 4,097,478.03 tok/s
    - our DP+EP: 3,411,612.51 tok/s
  - topk=8:
    - baseline DP: 158,752.70 tok/s
    - our DP: 1,564,200.65 tok/s
    - our DP+EP: 1,086,899.80 tok/s
- Interpretation:
  - EP axis size 4 gives nearly the same EP throughput as axis size 2 in this setup.
  - Large-batch regime remains DP-favored for our implementation on v5p-8.
- Next action:
  - Post summary/update on #2710 and tag snapshot commit.

## Negative Results Index
- Without VMEM override, the `hidden=1024` runs hit scoped vmem OOM in pallas GMM custom-call (`RESOURCE_EXHAUSTED`, scoped limit overrun).
