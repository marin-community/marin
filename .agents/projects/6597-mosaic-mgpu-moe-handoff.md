# 6597 Mosaic MGPU MoE Dispatch-Up Handoff

## Current State

- Branch: `codex/6597-mosaic-mgpu-moe-m1-m2`
- Latest pushed commit: `dd08c1326 [levanter] Add compact ring MoE transport probe`
- Research logbook: `.agents/logbooks/pallas-mosaic-mgpu-moe.md`
- Main benchmark harness: `lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py`
- Main Mosaic implementation file: `lib/levanter/src/levanter/kernels/pallas/moe_dispatch_up/mosaic_gpu.py`

## Status Against The Spec

Requested final goal:

> Implement an overlapped Mosaic GPU MoE dispatch-up path where scratch dispatch signals row/block readiness and W13 consumes ready blocks without waiting for full dispatch completion. Validate against the reference at H=2560, I=2560, E=256, topk=4 with Grug-style weights, then tune the end-to-end steady-state runtime below 2 ms on CoreWeave H100x8.

Current status:

- Bigger target shape is runnable on CoreWeave H100x8:
  - `H=2560`, `I=2560`, `E=256`, `topk=4`, `T/rank=4096`, `EP=8`, bf16, Grug-truncated weights.
- Correct compact source/expert capacity direction is established:
  - Average source/expert rows: `4096 * 4 / 256 = 64`.
  - 0% buffer: `source_expert_capacity=64`.
  - 25% buffer: `source_expert_capacity=80`.
  - Old full source slots/source: `131072`.
  - Compact 25% source slots/source: `20480`.
- Correctness:
  - H128 correctness passes against reference for compact all-to-all + Mosaic W13.
  - H128 ordered ring transport exactly matches compact all-to-all payload/count tensors.
  - Target runs generally use `--skip-reference-checks`; full target-shape reference validation is still not done because reference/prepack is expensive and prior target perf probes were candidate-only.
- Performance:
  - Best current clean non-overlapped compact path:
    - Cap64 merged compact all-to-all + Mosaic W13: mean `2.399 ms`, min `1.864 ms`.
    - Cap80 merged compact all-to-all + Mosaic W13: mean `2.706 ms`, min `2.147 ms`.
  - Split timing:
    - Cap64: transport `0.559 ms`, W13 `1.446 ms`, sum `2.006 ms`.
    - Cap80: transport `0.635 ms`, W13 `1.742 ms`, sum `2.377 ms`.
  - Existing Grug ring+ragged baseline/helper remains faster:
    - Ring+ragged target: about `1.889 ms`.
  - Ring+Mosaic diagnostic:
    - Target: about `2.221 ms`.
- Not yet achieved:
  - No production fused/overlapped Mosaic W13 consumer yet.
  - No stable `<2 ms` target-shape Mosaic path yet.
  - No tuned table/autotune-on-miss integration yet.
  - No backward/gradient kernel path yet.

## Cleanest Happy Path Today

Use the compact source/expert all-to-all path with merged local-expert W13:

1. Build compact source/expert groups sized by `source_expert_capacity`, not by full `T*K`.
2. Use built-in `lax.all_to_all` to exchange compact `[dst, local_expert, Cse, H]` payloads into receiver-local `[src, local_expert, Cse, H]`.
3. Merge source groups per local expert into `[local_expert, EP*Cse, H]`.
4. Run Mosaic W13 over local-expert groups.

This is not the final overlap design, but it is the cleanest correct measured substrate and gives the target latency budget:

- cap64: `0.559 ms` transport + `1.446 ms` W13 = `2.006 ms`.
- cap80: `0.635 ms` transport + `1.742 ms` W13 = `2.377 ms`.

The overlap implementation only needs to hide roughly `0.4-0.6 ms` of transport to cross `<2 ms` for this deterministic target shape.

## Key Commands

Run commands from repo root:

```bash
cd /Users/dlwh/.codex/worktrees/b809/marin
```

### Local Checks

```bash
uv run --package marin-levanter python -m py_compile \
  lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  lib/levanter/src/levanter/kernels/pallas/moe_dispatch_up/mosaic_gpu.py

uv run --package marin-levanter pytest \
  lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q

./infra/pre-commit.py --changed-files --fix
```

### H128 Correctness: Compact All-To-All + Mosaic W13

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 \
  --job-name dlwh-6597-moe-compact-a2a-mosaic-h128-capmax \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 \
  --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.25 \
  --run-compact-a2a-mosaic-dispatch-up \
  --bench-iters 1 --warmup-steps 1
```

Expected previously:

- `source_expert_capacity: 4`
- `source_expert_overflow_count: 0`
- `dispatch_up_compact_a2a_mosaic_gpu_max_abs_error: 0.0078125`
- `dispatch_up/compact_a2a_mosaic_gpu_end_to_end_ms: 0.390`

### Target Shape: Compact All-To-All + Merged Mosaic W13, Cap64

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 \
  --job-name dlwh-6597-moe-compact-a2a-mosaic-target-cap64-merge \
  --env-vars JAX_OPTIMIZATION_LEVEL O1 \
  --env-vars XLA_FLAGS "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 \
  --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.0 --source-expert-capacity 64 \
  --block-m 64 --block-n 128 --block-k 64 --num-stages 4 --grid-block-n 1 \
  --run-compact-a2a-mosaic-dispatch-up \
  --compact-a2a-return-compact-output --compact-a2a-merge-source-groups \
  --skip-reference-checks --bench-iters 5 --warmup-steps 2
```

Expected previously:

- `source_expert_capacity: 64`
- `source_expert_overflow_count: 0`
- `dispatch_up/compact_a2a_mosaic_gpu/steady: mean=2.399 ms min=1.864 ms max=2.559 ms iters=5`

### Target Shape: Compact All-To-All + Merged Mosaic W13, Cap80

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 \
  --job-name dlwh-6597-moe-compact-a2a-mosaic-target-cap80-merge \
  --env-vars JAX_OPTIMIZATION_LEVEL O1 \
  --env-vars XLA_FLAGS "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 \
  --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.25 --source-expert-capacity 80 \
  --block-m 64 --block-n 128 --block-k 64 --num-stages 4 --grid-block-n 1 \
  --run-compact-a2a-mosaic-dispatch-up \
  --compact-a2a-return-compact-output --compact-a2a-merge-source-groups \
  --skip-reference-checks --bench-iters 3 --warmup-steps 1
```

Expected previously:

- `source_expert_capacity: 80`
- `source_expert_overflow_count: 0`
- `dispatch_up/compact_a2a_mosaic_gpu/steady: mean=2.706 ms min=2.147 ms max=3.013 ms iters=3`

### Target Breakdown: Transport vs W13, Cap64

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 \
  --job-name dlwh-6597-moe-compact-a2a-breakdown-target-cap64 \
  --env-vars JAX_OPTIMIZATION_LEVEL O1 \
  --env-vars XLA_FLAGS "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 \
  --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.0 --source-expert-capacity 64 \
  --block-m 64 --block-n 128 --block-k 64 --num-stages 4 --grid-block-n 1 \
  --run-compact-a2a-breakdown \
  --skip-reference-checks --bench-iters 5 --warmup-steps 2
```

Expected previously:

- `dispatch/compact_a2a_transport/steady: mean=0.559 ms`
- `w13_silu/compact_a2a_merged_mosaic_gpu/steady: mean=1.446 ms`
- `dispatch_up/compact_a2a_breakdown_sum_ms: 2.006`

### Target Breakdown: Transport vs W13, Cap80

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 \
  --job-name dlwh-6597-moe-compact-a2a-breakdown-target-cap80 \
  --env-vars JAX_OPTIMIZATION_LEVEL O1 \
  --env-vars XLA_FLAGS "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 \
  --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.25 --source-expert-capacity 80 \
  --block-m 64 --block-n 128 --block-k 64 --num-stages 4 --grid-block-n 1 \
  --run-compact-a2a-breakdown \
  --skip-reference-checks --bench-iters 5 --warmup-steps 2
```

Expected previously:

- `dispatch/compact_a2a_transport/steady: mean=0.635 ms`
- `w13_silu/compact_a2a_merged_mosaic_gpu/steady: mean=1.742 ms`
- `dispatch_up/compact_a2a_breakdown_sum_ms: 2.377`

### H128 Correctness: Ordered Compact Ring Transport

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 \
  --job-name dlwh-6597-moe-compact-ring-transport-h128-rowvec \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 \
  --hidden 128 --intermediate 64 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.25 \
  --run-compact-ring-transport \
  --bench-iters 1 --warmup-steps 1
```

Expected previously:

- `dispatch_compact_ring_transport_x_max_abs_error: 0`
- `dispatch_compact_ring_transport_count_max_abs_error: 0`
- `dispatch/compact_ring_transport_end_to_end_ms: 1.439`

### Target: Ordered Compact Ring Transport, Cap64

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run \
  --gpu H100x8 --enable-extra-resources --cpu 16 --memory 160GB --disk 80GB --extra gpu --timeout 2400 \
  --job-name dlwh-6597-moe-compact-ring-transport-target-cap64-rowvec \
  --env-vars JAX_OPTIMIZATION_LEVEL O1 \
  --env-vars XLA_FLAGS "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python lib/levanter/scripts/bench/bench_moe_dispatch_up_mosaic_gpu.py \
  --ep-size 8 --tokens-per-rank 4096 --experts-per-rank 32 --top-k 4 \
  --hidden 2560 --intermediate 2560 --dtype bf16 --weight-init grug_truncated \
  --recv-capacity-factor 1.0 --source-expert-capacity 64 \
  --run-compact-ring-transport \
  --skip-reference-checks --bench-iters 3 --warmup-steps 1
```

Expected previously:

- `dispatch_compact_ring_transport_x_max_abs_error: 0`
- `dispatch_compact_ring_transport_count_max_abs_error: 0`
- `dispatch/compact_ring_transport/steady: mean=1.792 ms min=1.752 ms max=1.848 ms iters=3`

## Important Negative Results

- Original destination-major `[dst, T*K, H]` Mosaic peer-copy prepack is the wrong direction.
- Compact source/expert peer-copy transport returns and is correct, but remains too slow:
  - dispatch-only around `2.45 ms`.
  - dispatch+W13 around `3.722 ms`.
- Compact all-to-all W13 over 256 tiny `(source, local_expert)` groups is pathological:
  - cap80 compact-output run was about `14.273 ms`.
  - W13 must consume merged local-expert groups or equivalent pipelined chunks.
- Ordered compact ring transport with row-vector remote copies is correct but slow:
  - cap64 target transport around `1.792 ms`.
  - Built-in compact all-to-all transport is only around `0.559 ms`.
- A 256-column tiled version of ordered ring transport failed at target shape with a Mosaic layout-inference recursion error. H128 compiled, target did not.

## Next Engineering Step

Implement the first single-launch overlap prototype:

1. Keep compact source/expert payloads.
2. Use ordered cumulative GMEM semaphore thresholds, matching `collective_matmul_mgpu.py`.
3. For each source/ring step:
   - copy one source chunk into destination scratch,
   - signal the cumulative step semaphore,
   - consume the received chunk with W13 before all sources have arrived.
4. On H128, compare against reference.
5. On target, compare against:
   - cap64 non-overlapped compact all-to-all merged W13 (`2.399 ms`, split sum `2.006 ms`),
   - cap80 non-overlapped compact all-to-all merged W13 (`2.706 ms`, split sum `2.377 ms`),
   - ring+ragged helper baseline (`~1.889 ms`).

If the overlapped ring path is still above target because raw transport is too slow, optimize transport separately:

- Try fewer programs with larger per-program row chunks.
- Revisit bulk copy/TMA-style copies while avoiding the target layout-inference recursion.
- Consider wrapping the built-in compact all-to-all layout as the production happy path while the Pallas ring transport matures.

## Notes For Humans

- The current work is benchmark/prototype code, not production API integration.
- The benchmark harness now has several useful modes; prefer the compact all-to-all merged W13 path for clean numbers.
- The active objective is not complete until a real overlapped Mosaic path is correct at the target shape and steady-state `<2 ms`.
- The branch was clean and pushed before this handoff was written.
