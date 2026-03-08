# MoE SparseCore Dispatch: Research Logbook

- GitHub experiment issue: [#3418](https://github.com/marin-community/marin/issues/3418)

## Scope
- Goal: Evaluate whether an opt-in SparseCore-assisted dispatch path for [`moe_mlp`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py) can improve full forward+backward throughput over the current argsort-based TC-only dispatch path on `v5p-8`.
- Primary metric(s): steady-state `tokens_per_s` for full `moe_mlp` forward+backward; compile time as a secondary metric.
- Constraints: benchmark-only implementation, dev TPU workflow only, one TPU job at a time, existing ragged-dot expert matmuls remain unchanged.
- Related context:
  - `docs/dev-guide/dev_tpu.md`
  - `docs/recipes/add_pallas_kernel.md`
  - `docs/recipes/agent_research.md`
  - historical harness snapshot: commit `dda102c4c` (`refs #2710`)

## Baseline
- Date: 2026-03-07
- Code refs:
  - [`grug_moe.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - [`grug_moe_sparsecore.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe_sparsecore.py)
  - [`bench_grug_moe_sparsecore_dispatch.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py)
- Baseline case for repeated comparison:
  - `batch=192`, `seq=128`, `hidden=1024`, `intermediate=3072`, `experts=8`, `topk=2`, `dtype=bf16`

## Experiment Log
### 2026-03-07 14:xx - Kickoff
- Hypothesis:
  - SparseCore can help on the dispatch stage if we offload MoE pack/unpack work while leaving expert matmuls on the existing ragged-dot path.
  - The main win condition is lower dispatch latency or enough XLA overlap to hide dispatch under independent TC work.
- Command:
  - Pending first TPU execution.
- Config:
  - Start with local path; EP can be revisited once local parity is understood.
- Result:
  - Implemented an opt-in `dispatch_implementation="sparsecore"` path plus a benchmark harness.
- Interpretation:
  - First pass is intentionally bounded: SC only handles activation packing gathers, not expert math.
- Next action:
  - Run sequential `v5p-8` loops on the new harness and inspect whether the SC path lowers, runs, and improves steady-state throughput.

### 2026-03-07 13:47 - TPU sanity loop: baseline xla path
- Hypothesis:
  - The new benchmark harness should run cleanly on `v5p-8` with the existing `xla` dispatch path before we try SparseCore lowering.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation xla \
    --batch 4 --seq 32 --hidden 128 --intermediate 384 --experts 8 --topk 2 \
    --warmup 0 --iters 1
```
- Config:
  - TPU: `v5p-8`
  - Devices observed by JAX: 4 TPU chips
- Result:
  - `xla` ran successfully.
  - `compile_s=1.166865`
  - `steady_s=0.000540`
  - `tokens_per_s=237145.07`
- Interpretation:
  - The benchmark harness, mesh setup, and baseline path are valid on the target TPU.
- Next action:
  - Run the matching tiny `sparsecore` case to identify lowering/runtime issues before scaling shapes.

### 2026-03-07 13:48 - TPU sanity loop: first sparsecore lowering attempts
- Hypothesis:
  - The opt-in `dispatch_implementation="sparsecore"` path would either run or fail with a concrete lowering constraint we could iterate on.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation sparsecore \
    --batch 4 --seq 32 --hidden 128 --intermediate 384 --experts 8 --topk 2 \
    --warmup 0 --iters 1
```
- Result:
  - First failure: `core_map` rejected a captured `i32[64]` constant in the SC kernel closure.
  - Fix applied: rebuild the `ones` vector inside the kernel body instead of capturing it.
- Follow-up command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation sparsecore \
    --batch 4 --seq 32 --hidden 128 --intermediate 384 --experts 8 --topk 2 \
    --warmup 0 --iters 1
```
- Result:
  - Second failure during SC verification:
    - `tpu.vector_load` in the SC kernel expected the base memref to be in `VMEM`.
    - Current kernel is reading `expert_chunks` from `HBM`.
- Interpretation:
  - This is a substantive current-API blocker for the first-pass design.
  - The local dispatch kernel logic is not enough; a working SC path appears to require an explicit HBM→VMEM staging pattern or a different SC block-mapped input formulation.
- Next action:
  - Investigate the intended `tpu_sc` VMEM staging pattern before attempting more throughput runs.

### 2026-03-07 14:00 - Redesign to tutorial-style SC gather
- Hypothesis:
  - The right bounded experiment is not SC-side prefix-count metadata, but a hybrid path where XLA still forms routing metadata and SparseCore only performs activation packing via `plsc.BlockSpec(indexed_by=...)`.
- Result:
  - Replaced the failing metadata kernel with a SparseCore row-gather helper.
  - Added bf16 support via bitcast-to-`int32` on the minor dimension, then bitcast back after gather.
  - Added an explicit custom VJP so the backward path remains correct even though the SC forward pack is custom.
  - Also threaded the same gather helper into the EP ring local path, though the main sweep below stayed on EP1.
- Interpretation:
  - This is much closer to the official SparseCore tutorial and gives us a clean experiment on pack-only offload.

### 2026-03-07 14:02 - TPU sanity loop: tiny shapes after redesign
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation xla --implementation sparsecore \
    --batch 4 --seq 32 --hidden 128 --intermediate 384 --experts 8 --topk 2 \
    --warmup 1 --iters 5 --dtype bf16
```
- Result:
  - `xla`: `tokens_per_s=257205.26`
  - `sparsecore`: `tokens_per_s=240722.59` on local/non-EP shape from the earlier single-impl run
  - `xla`: `tokens_per_s=266448.84`
  - `sparsecore`: `tokens_per_s=268262.49` on EP=2 tiny sanity
- Interpretation:
  - The SC gather path lowers and runs, including the bf16 bitcast path.
  - Small-shape performance is roughly neutral; any real value has to come from larger routing-heavy shapes.

### 2026-03-07 14:05 - Target-hyper EP1 sweep (`seq=4096`, `experts=128`, `hidden=2048`, `intermediate=1536`)
- Motivation:
  - User-relevant stress regime:
    - `hidden_dim=2048`
    - `intermediate_dim=1536`
    - `num_experts=128`
    - `num_experts_per_token in {4, 8, 16}`
    - `max_seq_len=4096`
  - This sweep focuses on the MoE MLP-relevant subset and holds EP at 1 for parity first.
- Shared command template:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation xla --implementation sparsecore \
    --batch 1 --seq 4096 --hidden 2048 --intermediate 1536 --experts 128 \
    --warmup 1 --iters 3 --dtype bf16 --expert-axis-size 1 \
    --topk <4|8|16>
```
- Results:
  - `topk=4`
    - `xla`: `steady_s=0.036279`, `tokens_per_s=112901.42`
    - `sparsecore`: `steady_s=0.036443`, `tokens_per_s=112394.87`
    - speedup vs xla: `0.9955x`
  - `topk=8`
    - `xla`: `steady_s=0.037184`, `tokens_per_s=110153.62`
    - `sparsecore`: `steady_s=0.037652`, `tokens_per_s=108785.65`
    - speedup vs xla: `0.9876x`
  - `topk=16`
    - `xla`: `steady_s=0.039366`, `tokens_per_s=104048.56`
    - `sparsecore`: `steady_s=0.039958`, `tokens_per_s=102508.30`
    - speedup vs xla: `0.9852x`
- Interpretation:
  - The pack-only SparseCore hybrid does not cross over on `v5p-8`; it stays slightly slower across `topk=4/8/16`.
  - Higher `topk` makes dispatch more important, but not enough to outweigh the added SC path overhead.

### 2026-03-07 14:12 - Compiler dump / schedule inspection on target shape (`topk=8`)
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation sparsecore \
    --batch 1 --seq 4096 --hidden 2048 --intermediate 1536 --experts 128 --topk 8 \
    --warmup 0 --iters 1 --dtype bf16 --expert-axis-size 1 \
    --xla-dump-dir /tmp/moe_sc_dump_topk8 \
    --compiler-log-path /tmp/moe_sc_dump_topk8/compile.log
```
- Evidence:
  - `module_0095.jit_loss_fn...after_codegen.txt` shows:
    - `async-start(...), async_execution_thread="sparsecore"` for the SC gather call
    - later `async-done(...)` for that same SC call
    - only after `async-done` does the first TensorCore `gmm` `custom-call("tpu_custom_call")` appear
  - Relevant artifacts:
    - `/tmp/moe_sc_dump_topk8/module_0095.jit_loss_fn.cl_813921542.after_codegen.txt`
    - `/tmp/moe_sc_dump_topk8/compile.log`
  - A profile trace also includes a distinct `sparse-core-call...` region, so the path is genuinely using SparseCore rather than silently falling back.
- Interpretation:
  - XLA does lower the SC gather asynchronously, but the main TC expert matmul does not start until after the SC gather completes.
  - There is some room for overlap with surrounding metadata prep, but not with the dominant `gmm` work.
  - This strongly explains why throughput is flat-to-negative: the current graph shape does not realize the SC+TC overlap we would need for a win.

## Current Conclusion
- On `v5p-8`, a bounded hybrid that offloads only MoE activation packing to SparseCore is not enough to beat the current TC-only `moe_mlp` path on the tested EP1 target shapes.
- The key limiting factor is not that SparseCore fails to lower; it does lower and run.
- The key limiting factor is schedule structure:
  - SC gather runs as its own async SparseCore region.
  - The dominant TensorCore `gmm` work starts after that region completes.
  - Therefore the hoped-for SC+TC overlap is not materializing in the current formulation.

## Likely Follow-ups
- If we revisit this, the next worthwhile directions are:
  - restructure the graph so SC dispatch can run in parallel with genuinely independent TC work,
  - or move more of the dispatch/unpack pipeline onto SC so the offloaded portion is large enough to matter even without deep overlap.
- EP sweeps (`EP=2/4` on `v5p-8`) are now mechanically possible in the harness, but are lower priority until the local/EP1 path shows at least neutral-to-positive directionality.

### 2026-03-07 14:20 - Explicit blockwise pipelining attempt
- Rationale:
  - The whole-buffer SC gather path has a pure dataflow chain:
    - `SC pack(all tokens) -> TC gmm(all tokens)`
  - To test real overlap, I added a benchmark-only `dispatch_implementation="sparsecore_pipeline"` variant that splits the local token dimension into fixed-size blocks and runs local MoE block by block.
  - This creates a graph where, in principle, `SC pack(block i+1)` can overlap `TC gmm(block i)`.
- Important caveat:
  - On the earlier `batch=1, seq=4096` EP1 runs, the token dimension was already sharded across 4 data devices, so each shard saw only `1024` local tokens.
  - That exactly matched the pipeline block size and meant the first `sparsecore_pipeline` measurements were effectively degenerate: they did not create multiple local blocks.
- Meaningful pipelining test:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation xla --implementation sparsecore --implementation sparsecore_pipeline \
    --batch 4 --seq 4096 --hidden 2048 --intermediate 1536 --experts 128 --topk 4 \
    --warmup 1 --iters 2 --dtype bf16 --expert-axis-size 1
```
- Result:
  - `xla`: `steady_s=0.039472`, `tokens_per_s=415076.98`
  - `sparsecore`: `steady_s=0.040300`, `tokens_per_s=406555.10`
  - `sparsecore_pipeline`: `steady_s=0.080854`, `tokens_per_s=202636.68`
  - speedup vs xla: `0.4882x`
- Interpretation:
  - The explicit blockwise pipeline does create the right independence in principle, but the straightforward implementation is much worse overall.
  - The cost of duplicating routing/sort/ragged-dot setup per block dominates any overlap benefit.
  - So: yes, a pipelined formulation was tried, and in this first benchmark-only form it regresses badly rather than unlocking a win.

### 2026-03-07 14:30 - Expert-chunk pipeline (global sort once, chunk by expert range)
- Rationale:
  - The token-block pipeline was too expensive because it repeated routing/sort/ragged-dot setup per block.
  - A better benchmark-only overlap attempt is:
    - do the global routing sort once,
    - partition the grouped dispatch by expert chunks,
    - run SC gather + TC `gmm` per expert chunk.
  - This preserves the existing grouped-by-expert semantics and gives chunk-level independence without repeating the whole routing pipeline.
- Target-shape command:

### 2026-03-07 21:55 - VLIW comparison: vanilla `xla` vs one-expert dense exact-capacity SC path
- Goal:
  - Compare compiler/codegen structure directly for the same shape:
    - `batch=1, seq=32768, hidden=2048, intermediate=1536, experts=8, topk=8`
  - Baselines:
    - vanilla grouped `xla` path
    - one-expert dense exact-capacity SC path (`chunk_experts=1`, `single_expert_dense=1`, `capacity_factor=1.0`, `capacity_pad=0`)
- Dump artifacts:
  - XLA: `/tmp/moe_xla_dump_oneexpert_compare/module_0095.jit_loss_fn.cl_813921542.after_codegen.txt`
  - SC path: `/tmp/moe_sc_dump_oneexpert_e8_k8_exact_barrier/module_0095.jit_loss_fn.cl_813921542.after_codegen.txt`
- Key differences:
  - Vanilla `xla` keeps one large grouped dispatch buffer:
    - grouped gather/scatter prep is cheap:
      - `_prepare_moe_dispatch/gather` kernels are only ~`2335-2375` estimated cycles
      - scatter-add prep is ~`2530` cycles
    - then it launches two large grouped Megablox matmuls:
      - `gmm.4`: `bf16[65536,2048] -> bf16[65536,3072]`
      - `gmm.5`: `bf16[65536,1536] -> bf16[65536,2048]`
    - surrounding fused TC/vector work is large but still batched:
      - `slice_multiply_fusion`: `855136` estimated cycles
      - `pad_maximum_fusion`: `1563991`
      - scatter multiply fusions: `936384-937664`
  - The SC path replaces that with eight per-expert streams:
    - eight async `sparse-core-call...` launches, each gathering `s32[8192,1024]`
    - then a large fan-out/fan-in staging fusion (`fusion.76`) just to materialize eight expert-specific `bf16[8192,2048]` tensors:
      - `787536` estimated cycles
    - then sixteen dense TC kernels:
      - eight first matmuls `bf16[8192,2048] -> bf16[8192,3072]`, each `421567` estimated cycles
      - eight second matmuls `bf16[8192,1536] -> bf16[8192,2048]`, each `223398`
    - plus eight separate scatter-multiply paths, each ~`118392` cycles, and many tiny per-expert gather/scatter index kernels
- Most important schedule finding:
  - Even with the benchmark-only optimization barrier enabled, all eight SC `async-done` events still occur before the first dense expert matmul becomes available to run.
  - So the SC path is not winning by overlap; it is mainly changing decomposition.
- Interpretation:
  - The vanilla `xla` VLIW confirms the user intuition:
    - pack/scatter itself is already a small part of the total schedule in this regime
    - the dominant work is still the large grouped TC path
  - Our SC-enabled path does not remove enough real work from that critical path.
  - Instead, it introduces:
    - extra SC launch/staging,
    - bitcast/copy/broadcast overhead,
    - and fragmentation from one large grouped TC program into many smaller per-expert TC programs.
  - This matches the throughput story:
    - realistic valid cases get at best to parity or near-parity
    - the structural overlap win we wanted is not present in the generated schedule.
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation xla --implementation sparsecore --implementation sparsecore_expert_pipeline \
    --batch 1 --seq 4096 --hidden 2048 --intermediate 1536 --experts 128 --topk 4 \
    --warmup 1 --iters 3 --dtype bf16 --expert-axis-size 1
```
- Result:
  - `xla`: `steady_s=0.036186`, `tokens_per_s=113192.48`
  - `sparsecore`: `steady_s=0.036477`, `tokens_per_s=112289.20`
  - `sparsecore_expert_pipeline`: `steady_s=0.042205`, `tokens_per_s=97049.05`
  - speedup vs xla: `0.8574x`
- Interpretation:
  - This expert-chunk pipeline is meaningfully better than the naive token-block pipeline, but it is still clearly slower than the baseline TC path.
  - The extra chunk orchestration, padding, and multiple `gmm` invocations outweigh the overlap benefit.

### 2026-03-07 14:35 - Schedule check for expert-chunk pipeline
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --implementation sparsecore_expert_pipeline \
    --batch 1 --seq 4096 --hidden 2048 --intermediate 1536 --experts 128 --topk 4 \
    --warmup 0 --iters 1 --dtype bf16 --expert-axis-size 1 \
    --xla-dump-dir /tmp/moe_sc_dump_expert_pipeline_topk4 \
    --compiler-log-path /tmp/moe_sc_dump_expert_pipeline_topk4/compile.log
```
- Evidence:
  - `module_0095.jit_loss_fn...after_codegen.txt` shows many `async_execution_thread="sparsecore"` chunk launches (`sparse-core-call`, `sparse-core-call.1`, ..., `sparse-core-call.7`) rather than a single monolithic SC region.
  - After the first SC `async-done`, the first chunked TC `gmm` (`moe_up_down_chunk_0/jit(gmm)`) appears while other SC chunk launches are still present in the graph.
- Interpretation:
  - Unlike the original whole-buffer SC gather, this formulation does provide a graph shape where SC/TC chunk overlap is possible.
  - So the negative result here is not "XLA refused to overlap anything"; it is "even with chunkable overlap, the added chunking overhead still loses on `v5p-8` for this benchmark-only design."

### 2026-03-07 14:50 - Larger-token sweep (`seq=32768`) at smaller expert counts
- Rationale:
  - The earlier `seq=4096` target shape only left `1024` local tokens per data shard on `v5p-8`, which is too little to fairly judge pipeline variants.
  - I reran the benchmark at `batch=1, seq=32768` so each shard sees `8192` local tokens, and reduced experts to `64` and `32` to increase per-expert arithmetic intensity.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  bash -lc '
set -euo pipefail
for experts in 64 32; do
  for topk in 4 8; do
    for impl in xla sparsecore sparsecore_pipeline sparsecore_expert_pipeline; do
      uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
        --batch 1 --seq 32768 --hidden 2048 --intermediate 1536 \
        --experts "$experts" --topk "$topk" --dtype bf16 \
        --warmup 1 --iters 2 --implementation "$impl"
    done
  done
done'
```
- Result:
  - `experts=64, topk=4`
    - `xla`: `1102999.24 tok/s`
    - `sparsecore`: `1051045.92 tok/s` (`0.953x`)
    - `sparsecore_pipeline`: `438286.59 tok/s` (`0.397x`)
    - `sparsecore_expert_pipeline`: `854826.21 tok/s` (`0.775x`)
  - `experts=64, topk=8`
    - `xla`: `799250.37 tok/s`
    - `sparsecore`: `751194.63 tok/s` (`0.940x`)
    - `sparsecore_pipeline`: `388805.09 tok/s` (`0.486x`)
    - `sparsecore_expert_pipeline`: `598356.74 tok/s` (`0.749x`)
  - `experts=32, topk=4`
    - `xla`: `1576693.92 tok/s`
    - `sparsecore`: `1460194.89 tok/s` (`0.926x`)
    - `sparsecore_pipeline`: `757348.82 tok/s` (`0.480x`)
    - `sparsecore_expert_pipeline`: `1180551.39 tok/s` (`0.749x`)
  - `experts=32, topk=8`
    - `xla`: `1016418.09 tok/s`
    - `sparsecore`: `943184.12 tok/s` (`0.928x`)
    - `sparsecore_pipeline`: `613504.44 tok/s` (`0.604x`)
    - `sparsecore_expert_pipeline`: `735385.59 tok/s` (`0.724x`)
- Interpretation:
  - Larger local token counts do help the benchmark become more representative of a real pipeline test, but they do not change the ordering.
  - The plain SC path remains modestly slower than the TC-only baseline.
  - The token-block pipeline remains decisively bad.
  - The expert-chunk pipeline benefits from larger token counts and fewer experts, but still loses by a substantial margin.
  - This weakens the hypothesis that the earlier negative results were mainly a "too few local tokens" artifact.

### 2026-03-07 14:53 - Tutorial-style standalone overlap check
- Rationale:
  - The JAX SparseCore tutorial explicitly claims SC+TC overlap is possible by putting independent work in one `jax.jit`.
  - The exact tutorial code uses newer `pl.kernel` APIs than our local `jax 0.8`, so I adapted it to an equivalent `pl.pallas_call(..., kernel_type=SC_VECTOR_SUBCORE)` microbench.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_sparsecore_tutorial_overlap.py \
    --rows 32768 --cols 128 --warmup 10 --iters 100
```
- Result:
  - `sc_add_one_s=0.000223`
  - `tc_add_one_s=0.000171`
  - `two_add_ones_s=0.000276`
  - `sum_s=0.000394`
  - overlap ratio: `1.4276x`
- Interpretation:
  - On this `v5p-8` stack, independent SC and non-SC work do overlap materially in a minimal benchmark.
  - So the MoE negative result is not a blanket platform limitation.
  - The more precise conclusion is:
    - overlap is achievable,
    - but the current MoE SC-dispatch formulations do not recover enough overlap benefit to pay for their extra orchestration overhead.

### 2026-03-07 15:15 - Expert-chunk size sweep on the larger-token case
- Rationale:
  - The `seq=32768` runs suggested the expert-chunk pipeline was less bad than at `seq=4096`, but still clearly behind `xla`.
  - I made the benchmark-only chunk size configurable via environment so I could determine whether the loss was mostly bad tuning or fundamentally bad chunk orchestration.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  bash -lc '
set -euo pipefail
for experts in 64 32; do
  for topk in 4 8; do
    for chunk in 4 8 16 32; do
      GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS="$chunk" \
      uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
        --batch 1 --seq 32768 --hidden 2048 --intermediate 1536 \
        --experts "$experts" --topk "$topk" --dtype bf16 \
        --warmup 1 --iters 2 \
        --implementation sparsecore_expert_pipeline
    done
  done
done'
```
- Result:
  - `experts=64, topk=4`
    - chunk `4`: `775541 tok/s`
    - chunk `8`: `818513 tok/s`
    - chunk `16`: `854431 tok/s`
    - chunk `32`: `873934 tok/s`
  - `experts=64, topk=8`
    - chunk `4`: `549341 tok/s`
    - chunk `8`: `582082 tok/s`
    - chunk `16`: `598285 tok/s`
    - chunk `32`: `602110 tok/s`
  - `experts=32, topk=4`
    - chunk `4`: `1081813 tok/s`
    - chunk `8`: `1143287 tok/s`
    - chunk `16`: `1180605 tok/s`
    - chunk `32`: `1229024 tok/s`
  - `experts=32, topk=8`
    - chunk `4`: `704148 tok/s`
    - chunk `8`: `728298 tok/s`
    - chunk `16`: `735057 tok/s`
    - chunk `32`: `759395 tok/s`
- Interpretation:
  - Performance improves monotonically as the chunk size gets larger.
  - That is the opposite of what we would want if chunk-level overlap were providing the win.
  - The dominant cost is chunk orchestration, padding, and repeated `gmm` setup, not insufficient overlap.

### 2026-03-07 15:24 - Single-chunk / exact-capacity limit of the expert pipeline
- Rationale:
  - Since larger chunks were always better, I checked the near-degenerate limit:
    - one chunk covering all experts,
    - and exact capacity (`factor=1.0`, `pad=0`) to remove avoidable slack.
  - If this limit converges to the plain SC path, then the pipeline only improves by collapsing back toward the non-overlapped formulation.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  bash -lc '
set -euo pipefail
while read -r experts topk chunk factor pad; do
  GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS="$chunk" \
  GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR="$factor" \
  GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD="$pad" \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --batch 1 --seq 32768 --hidden 2048 --intermediate 1536 \
    --experts "$experts" --topk "$topk" --dtype bf16 \
    --warmup 1 --iters 2 \
    --implementation sparsecore_expert_pipeline
done <<EOF
64 4 64 1.25 256
64 4 64 1.0 0
64 8 64 1.25 256
64 8 64 1.0 0
32 4 32 1.0 0
32 8 32 1.0 0
EOF'
```
- Result:
  - `experts=64, topk=4, chunk=64`
    - default capacity (`1.25`, `256`): `928119 tok/s`
    - exact capacity (`1.0`, `0`): `1044777 tok/s`
    - plain `sparsecore` reference from the larger-token sweep: `1051046 tok/s`
  - `experts=64, topk=8, chunk=64`
    - default capacity (`1.25`, `256`): `632975 tok/s`
    - exact capacity (`1.0`, `0`): `748445 tok/s`
    - plain `sparsecore` reference: `751195 tok/s`
  - `experts=32, topk=4, chunk=32, exact capacity`: `1450404 tok/s`
    - plain `sparsecore` reference: `1460195 tok/s`
  - `experts=32, topk=8, chunk=32, exact capacity`: `936524 tok/s`
    - plain `sparsecore` reference: `943184 tok/s`
- Interpretation:
  - In the best-tuned limit, the expert-chunk pipeline converges almost exactly to the plain SC path.
  - This is strong evidence that the chunked formulation does not unlock a meaningful net overlap benefit for this MoE kernel on `v5p-8`.
  - The improvement comes from removing chunk overhead, not from extracting more useful SC/TC concurrency.

### 2026-03-07 15:35 - Attempt to skip padded SC gather/scatter work at `capacity_factor=1.25`
- Rationale:
  - The expert-chunk pipeline still did full-capacity SC gather and final scatter work even when a chunk's real load was smaller.
  - I added an opt-in benchmark-only path (`GRUG_SPARSECORE_EXPERT_PIPELINE_STATIC_ROUTING=1`) that chooses a smaller per-chunk work bucket before SC gather / ragged-dot / scatter, so padded capacity work is skipped more aggressively.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  bash -lc '
set -euo pipefail
while read -r experts topk bucketed; do
  GRUG_SPARSECORE_EXPERT_PIPELINE_STATIC_ROUTING="$bucketed" \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --batch 1 --seq 32768 --hidden 2048 --intermediate 1536 \
    --experts "$experts" --topk "$topk" --dtype bf16 \
    --warmup 1 --iters 2 \
    --implementation sparsecore_expert_pipeline
done <<EOF
64 4 0
64 4 1
64 8 0
64 8 1
EOF'
```
- Result:
  - `experts=64, topk=4`
    - default path (`bucketed=0`): `854835 tok/s`
    - padded-work-skipping path (`bucketed=1`): `647459 tok/s`
  - `experts=64, topk=8`
    - default path (`bucketed=0`): `598241 tok/s`
    - padded-work-skipping path (`bucketed=1`): `448323 tok/s`
  - `experts=32, topk=4`
    - default path (`bucketed=0`): `1179713 tok/s`
    - padded-work-skipping path (`bucketed=1`): `915449 tok/s`
  - compile time also jumped sharply (`~16-18s` to `~104-117s`)
- Interpretation:
  - Structurally, this path does skip more padded SC gather / scatter / ragged-dot work.
  - Empirically, it is much worse overall because the bucketed branch structure explodes compile cost and still regresses steady-state.
  - I kept it behind an explicit env flag for future experiments, but disabled any default benchmark opt-in.

### 2026-03-07 22:20 - JF LLO comparison: vanilla `xla` vs one-expert dense SC path
- Shape / setup:
  - `batch=1, seq=32768, hidden=2048, intermediate=1536, experts=8, topk=8, dtype=bf16`
  - vanilla baseline: `--implementation xla`
  - SC path: `--implementation sparsecore_expert_pipeline`
  - SC env:
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS=1`
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_SINGLE_EXPERT_DENSE=1`
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_TOKEN_CHUNK_SIZE=0`
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR=1.0`
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD=0`
    - `GRUG_SPARSECORE_EXPERT_PIPELINE_BARRIER=1`
  - JF dump flags:
    - `--xla_jf_dump_hlo_text=true`
    - `--xla_jf_dump_llo_text=true`
    - `--xla_jf_dump_llo_static_gaps=true`
    - `--xla_jf_emit_annotations=true`
    - `--xla_jf_debug_level=2`
- Dump locations:
  - XLA: `/tmp/llo_xla_compare`
  - SC: `/tmp/llo_sc_compare`
- Key LLO findings:
  - Vanilla `xla` dispatch stays compact.
    - `_prepare_moe_dispatch/gather`: `broadcast_clamp_fusion`, `135` scheduled bundles in final bundles.
    - scatter back: `iota.83` / `broadcast_clamp_fusion.1`, `53` / `69` scheduled bundles.
    - the heavy path is still Megablox grouped kernels (`jit(gmm)` / `jit(tgmm)`), not explicit dispatch kernels from `grug_moe.py`.
  - SC one-expert path does not look like a cheap “swap gather with SC” transformation.
    - top-level routed gather: `fusion.301`, `94` scheduled bundles.
    - per-expert scatter kernels are replicated; e.g. `scatter_single_expert_7_subchunk_0/scatter-add` has `24` / `41` / `50` bundle helpers plus a `sort.25` kernel at `4304` bundles.
    - per-expert dense TC kernels are fully split out:
      - up-proj `dot_general`: `bf16[8192,3072]`, about `23816` final-bundle schedule for each expert.
      - down-proj `dot_general`: `bf16[8192,2048]`, about `22223` final-bundle schedule for each expert.
      - there are 8 copies of each, one per expert/subchunk.
- Interpretation:
  - The LLO comparison matches the earlier VLIW read.
  - Baseline `xla` is not using SparseCore; it is a compact dispatch path plus large grouped Megablox TC kernels.
  - The SC path mainly trades one grouped program for many per-expert programs and a lot of extra gather/scatter/indexing machinery.
  - This supports the conclusion that our current SC-enabled decomposition is not winning because it replaces a relatively cheap visible dispatch path with much more fragmented work, rather than exposing a clearly shorter critical path.

### 2026-03-07 22:35 - Real EP4 profile breakdown for `grug-moe-qwen3-32b-a4b-v5p64-bs320-ep4-cf1p0-topk4-matched-active-pf32-buf64-synthetic-profile-iris-main-r1`
- Artifact:
  - W&B run target `marin-community/marin/grug-moe-qwen3-32b-a4b-v5p64-bs320-ep4-cf1p0-topk4-matched-active-pf32-buf64-synthetic-profile-iris-main-r1`
  - summarized into `scratch/grug_moe_qwen3_ep4_profile_summary.json`
- Run metadata:
  - `train_batch_size=320`
  - `max_seq_len=4096`
  - `hidden_dim=2048`
  - `intermediate_dim=1536`
  - `num_experts=128`
  - `topk=4`
  - `capacity_factor=1.0`
  - mesh axes `data=-1, expert=4, model=1`
  - batch is sharded over `("data", "expert")`, so the profiled run has `10` local sequences / chip and `40960` local tokens / chip
- Exclusive MoE breakdown (`MoEMLP=>moe_mlp`, no double counting):
  - local layout-transform: `25.06M` (`34.6%`)
  - EP collectives: `11.07M` (`15.3%`)
  - visible Megablox bookkeeping: `0.34M` (`0.5%`)
  - grouped expert matmul roots (`gmm` / `tgmm`): `25.87M` (`35.7%`)
  - other MoE glue: `10.14M` (`14.0%`)
- Notable dispatch/collect hotspots:
  - `scatter=>scatter-add`: `7.92M`
  - `gather=>_take=>scatter-add`: `6.72M`
  - `gather=>_take=>gather`: `3.48M` and `2.56M`
  - `gather=>all_gather`: `3.71M`
  - `scatter=>gather`: `1.46M`
  - `scatter=>all_gather`: `1.15M`
- Interpretation:
  - The real EP4 profile does support the user intuition: dispatch/collect is a very large part of lived MoE overhead.
  - The earlier EP1 synthetic microbench understated this cost.
  - However, this makes the target more credible, not the current SC decomposition more credible.

### 2026-03-07 22:55 - Matched synthetic EP4 preset on `v5p-8`
- Harness change:
  - added `--preset qwen3-32b-ep4-profile` and `--capacity-factor` to `bench_grug_moe_sparsecore_dispatch.py`
  - preset matches the real profile's per-chip geometry instead of only matching model dims
- Preset behavior on `v5p-8`:
  - derives `batch=40, seq=4096, hidden=2048, intermediate=1536, experts=128, topk=4, expert_axis_size=4, capacity_factor=1.0`
  - batch sharding over all 4 TPU devices gives:
    - `local_batch=10`
    - `local_tokens=40960`
    - `local_assignments=163840`
  - this matches the profiled EP4 run's per-chip token / assignment load
- Benchmark:
```bash
uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --preset qwen3-32b-ep4-profile \
    --warmup 0 --iters 1 \
    --implementation xla \
    --implementation sparsecore
```
- Result:
  - `xla`: `1810546.93 tok/s`
  - `sparsecore`: `1589415.52 tok/s`
  - `speedup_vs_xla = 0.8779x`
- Interpretation:
  - Even after fixing the synthetic benchmark so it matches the real EP4 local geometry, the current `sparsecore` EP path is still materially slower than vanilla `xla`.
  - So the mismatch was not just "we benchmarked the wrong token regime"; the current SC path still does not convert the real dispatch/collect burden into a throughput win.

### 2026-03-07 23:05 - Profile the matched synthetic `xla` benchmark itself
- Goal:
  - verify that the profile-matched synthetic benchmark actually reproduces the dispatch-heavy structure from the real EP4 run, rather than merely matching shape metadata
- Profile command:
```bash
uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute -- \
  bash -lc '
    rm -rf scratch/bench_profile_match_xla scratch/bench_profile_match_xla_summary.json &&
    uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
      --preset qwen3-32b-ep4-profile \
      --implementation xla \
      --profile-implementation xla \
      --profile-dir scratch/bench_profile_match_xla \
      --warmup 1 --iters 3 &&
    uv run python lib/marin/tools/profile_summary.py summarize \
      --profile-dir scratch/bench_profile_match_xla \
      --breakdown-mode exclusive_global \
      --output scratch/bench_profile_match_xla_summary.json
  '
```
- Trace quality:
  - `num_complete_events=8735`
  - `suspected_truncation=false`
  - no `quality_warnings`
- MoE structure from `scratch/bench_profile_match_xla_summary.json`:
  - `loss_fn=>moe_mlp`: `1.876M` inclusive
  - immediate children:
    - `loss_fn=>moe_mlp=>moe_up_down`: `810.1k` inclusive (`43.2%`)
    - `loss_fn=>moe_mlp=>gather`: `554.2k` inclusive (`29.5%`)
    - `loss_fn=>moe_mlp=>scatter`: `510.0k` inclusive (`27.2%`)
  - so gather + scatter = `56.7%` of `moe_mlp` inclusive time, larger than expert compute in this synthetic benchmark
- Most expensive MoE descendants:
  - `loss_fn=>moe_mlp=>moe_up_down=>gmm`: `477.5k` inclusive
  - `loss_fn=>moe_mlp=>gather=>_take`: `428.4k` inclusive
  - `loss_fn=>moe_mlp=>scatter=>scatter-add`: `350.2k` inclusive/exclusive
  - `loss_fn=>moe_mlp=>gather=>_take=>scatter-add`: `348.3k` inclusive/exclusive
  - `loss_fn=>moe_mlp=>moe_up_down=>tgmm`: `273.1k` inclusive
  - `loss_fn=>moe_mlp=>gather=>all_gather`: `86.2k`
  - `loss_fn=>moe_mlp=>gather=>_take=>gather`: `80.0k`
  - `loss_fn=>moe_mlp=>scatter=>all_gather`: `73.3k`
  - `loss_fn=>moe_mlp=>scatter=>gather`: `59.7k`
- Interpretation:
  - The matched synthetic benchmark does reproduce the qualitative profile shape the user cares about.
  - The hot dispatch leaves are exactly the ones the user pointed at: `_take=>scatter-add`, `scatter=>scatter-add`, and related gather/scatter paths.
  - So the negative SC result is not coming from a benchmark where gather/scatter is artificially tiny; even in the matched synthetic profile, take/scatter are a huge fraction of MoE time.

### 2026-03-07 23:25 - Less-toy SC/TC overlap microbench closer to MoE dispatch
- Added `lib/levanter/scripts/bench/bench_sparsecore_dispatch_overlap.py`.
- Purpose:
  - fixed-routing benchmark for chunked `gather -> dense expert MLP -> scatter`
  - isolates the dispatch/compute overlap hypothesis without top-k sort, ragged multi-expert GMM, or EP collectives
  - benchmark-only bridge between the tutorial overlap demo and full `moe_mlp`
- Implementations:
  - `serial_xla`: XLA gather + dense MLP + scatter for each chunk
  - `serial_sc`: SparseCore gather + dense MLP + scatter for each chunk
  - `pipeline_sc --barrier`: prefetch next chunk with SparseCore while dense MLP runs on the current chunk

### 2026-03-07 23:35 - Overlap microbench timings on `v5p-8`
- Shape:
  - `tokens=40960`
  - `hidden=2048`
  - `intermediate=1536`
  - `num_chunks=32`
  - `chunk_tokens=5120`
  - total chunk assignments `= 163840`, matching the real EP4 local assignment load
- Command shape:
  - profiled each implementation separately with `--warmup 1 --iters 3`
- Timings:
  - `serial_xla`: `0.043321 s`
  - `serial_sc`: `0.050206 s`
  - `pipeline_sc --barrier`: `0.051691 s`
- Interpretation:
  - This more realistic overlap benchmark still does not beat the plain XLA local path.
  - It also does not beat the serial SparseCore version in steady state.

### 2026-03-07 23:45 - Profile and dump evidence for the overlap microbench
- Trace summaries were clean (`suspected_truncation=false`) for:
  - `scratch/sc_overlap_serial_xla_summary.json`
  - `scratch/sc_overlap_serial_sc_summary.json`
  - `scratch/sc_overlap_pipeline_sc_summary.json`
- Perfetto trace observation:
  - only the SparseCore gather scopes showed up cleanly by chunk name in the trace
  - the TC-side chunked MLP work was heavily lowered/fused and was easier to read from codegen than from the trace directly
- After-codegen dump comparison (`module_0088.jit__lambda...after_codegen.txt`):
  - serial SC:
    - first scheduled MLP launch was `mlp_chunk_0`
    - before that first MLP launch there were `31` `copy-start(%sparse-core-call...)` lines and `31` SparseCore `async-done` lines
    - in other words, essentially all SparseCore chunk materialization had completed before TC chunk compute began
  - pipelined SC:
    - first scheduled MLP launch was `mlp_chunk_1`
    - before that first MLP launch there were only `18` `copy-start(%sparse-core-call...)` lines, with `10` more such copy-starts appearing in the following local window after MLP compute had already begun
    - so the pipelined version does create partial interleaving between pending SparseCore chunk completion/copy work and TC MLP launches
- Interpretation:
  - This benchmark is the first one in this thread where the codegen evidence shows the intended structural change: TC chunk compute starts while some later SC chunk materialization is still being scheduled.
  - That said, the overlap is not sufficient to outweigh the chunking/orchestration overhead, because steady-state time is still slightly worse than serial SparseCore and materially worse than the local XLA baseline.

### 2026-03-07 23:55 - EP-ring overlap benchmark closer to the real workload
- Added `lib/levanter/scripts/bench/bench_sparsecore_ep_overlap.py`.
- Goal:
  - keep the real EP ring gather/collect structure from `#2710`
  - benchmark exact `xla` EP routing versus a benchmark-only chunked SparseCore pipeline inside the expert shard
  - make the synthetic overlap case look more like the profiled `grug-moe-qwen3-32b...ep4...` workload
- The custom `pipeline_sc_ep` path keeps:
  - `all_gather(x, selected_experts, combine_weights)`
  - local-expert filtering / capacity compaction
  - chunked SparseCore token take
  - chunked ragged local-expert GMM/TGMM
  - `psum_scatter` collection

### 2026-03-08 00:05 - Matched EP-ring overlap benchmark on `v5p-8`
- Shape:
  - `batch=40, seq=4096, hidden=2048, intermediate=1536, experts=128, topk=4, expert_axis_size=4, capacity_factor=1.0`
  - this matches the real EP4 profile's per-chip geometry (`40960` local tokens, `163840` local assignments)
- First-pass timings (`warmup=0`, `iters=1`):
  - `xla_ep`: `0.090513 s`, `1.810M tok/s`
  - `pipeline_sc_ep --barrier --chunk-experts=8`: `0.131820 s`, `1.243M tok/s`
- Interpretation:
  - the benchmark-only EP pipeline is substantially slower than the exact current EP `xla` path
  - but it does run on the real EP shape and is much closer to the actual workload than the earlier local-only overlap toy

### 2026-03-08 00:15 - Profile shape of the new EP overlap benchmark
- Profiled:
  - `xla_ep` into `scratch/sc_ep_xla_summary.json`
  - `pipeline_sc_ep` into `scratch/sc_ep_pipeline_summary.json`
- `xla_ep` profile matches the real workload structure well:
  - `loss_fn=>moe_mlp=>moe_up_down`: `540.9k` inclusive
  - `loss_fn=>moe_mlp=>gather`: `369.6k` inclusive
  - `loss_fn=>moe_mlp=>scatter`: `339.9k` inclusive
  - hot leaves look like the real run:
    - `gather=>_take`: `285.5k`
    - `scatter=>scatter-add`: `233.5k`
    - `gather=>_take=>scatter-add`: `232.2k`
    - `gather=>all_gather`: `57.6k`
    - `scatter=>all_gather`: `48.9k`
- `pipeline_sc_ep` now has a much more realistic gather bucket than the earlier local-only overlap benchmark:
  - `loss_fn=>gather`: `85.6k` inclusive
  - `loss_fn=>gather=>all_gather`: `63.3k`
  - `loss_fn=>gather_chunk_1`: `206.7k`
  - `loss_fn=>gather_chunk_0`: `142.8k`
  - `loss_fn=>mlp_chunk_0..3`: about `136k` each
  - `loss_fn=>scatter_chunk_0..3`: about `143k-151k` each
- Interpretation:
  - This new benchmark is much closer to the actual EP4 workload than `bench_sparsecore_dispatch_overlap.py`.
  - In particular, it now includes the heavy `all_gather` / local-compaction part of dispatch that made the real profile's gather bucket large.
  - The negative throughput result remains, but at least it is now being measured in a benchmark whose profile shape looks like the real workload rather than a local-only take/scatter toy.

### 2026-03-08 00:30 - Scatter-focused follow-ups
- The matched `pipeline_sc_ep` profile confirms `scatter_chunk_*` is now the dominant per-chunk cost:
  - `scatter_chunk_1/scatter-add`: about `8.52 ms`
  - `scatter_chunk_2/scatter-add`: about `8.52 ms`
  - `scatter_chunk_3/scatter-add`: about `8.51 ms`
  - `scatter_chunk_0/scatter-add`: about `7.99 ms`
- A benchmark-only `coalesced` scatter path was added to [`bench_sparsecore_ep_overlap.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_sparsecore_ep_overlap.py):
  - sort `token_ids`
  - `segment_sum` duplicate token writes
  - scatter only unique rows
- Result on the same EP4-matched `pipeline_sc_ep` config:
  - `naive`: `0.131705 s`, `1.244M tok/s`
  - `coalesced`: `0.167535 s`, `0.978M tok/s`
- Profile of the coalesced variant (`scratch/sc_ep_pipeline_coalesced_profile_20260308_summary.json`) shows why it lost:
  - the expensive final `scatter_chunk_* / scatter-add` kernels remained essentially unchanged
  - new reorder/reduction kernels were added on top (`sorted_values` / `segment_sum` path, about `3.61 ms` each in the top ops)
- I also checked whether current JAX provides a more direct SparseCore scatter route:
  - local `0.8.0` and `0.9.0.1` sources do expose low-level SC indexed scatter-add primitives (`addupdate_scatter`)
  - but indexed SC gather/scatter refs are still restricted to `int32` / `float32`; `0.9.0.1` does not add a new native bf16 indexed path for `v5p`
- I tried a benchmark-only `sparsecore_transpose` scatter path that realizes scatter-add as the transpose of the underlying SC gather lowering:
  - first attempt failed because transpose through the bf16 bitcast restore path is unsupported
  - after rewriting the helper to transpose purely in `int32` space for bf16 payloads, the run still failed because the current JAX stack has no usable transpose rule for the relevant `pallas_call` path (`AssertionError: must override`)
- Interpretation:
  - the simple “SC scatter by transposing SC gather” route is blocked by current JAX internals
  - the simple “coalesce duplicates before scatter” route loses because it adds work without shrinking the expensive final scatter kernel
  - the next real SC-scatter experiment would have to be a hand-written `addupdate_scatter` kernel, likely tiled by output rows to fit VMEM

### 2026-03-08 00:45 - Direct SC scatter feasibility check on the current stack
- I added a benchmark-only unique-index overwrite helper in [`grug_moe_sparsecore.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe_sparsecore.py) and a standalone microbench in [`bench_sparsecore_scatter_unique.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_sparsecore_scatter_unique.py).
- The goal was narrower than MoE:
  - shape matched the MoE-scale local output: `rows=163840`, `updates=40960`, `width=2048`
  - compare pure unique-index row scatter overwrite with `xla` vs the documented SparseCore indexed-copy pattern
- Baseline `xla` unique scatter is already extremely fast:
  - `steady_s=0.00323-0.00333 s` on repeated reruns
- The SC helper did not produce a usable benchmark result, but the failure is informative:
  - direct row-by-row `sync_copy` lowered to an invalid local-to-local DMA on SparseCore
  - rewriting the helper to follow the docs more closely with `emit_pipeline(... sync_copy(... indexed HBM ...))` still lowered to the same invalid local-to-local DMA on this JAX `0.8` stack
  - forcing the output `BlockSpec` to `memory_space=ANY` did not change that lowering
- Interpretation:
  - on this runtime, even the easier unique-index overwrite form of SC scatter is not straightforwardly available through the documented pattern
  - this makes it less likely that a production-worthy additive SC scatter will be easy to prototype in the repo’s current environment
  - if we keep pushing SC scatter, the next credible options are:
    - hand-written `addupdate_scatter` directly with low-level SC primitives and a custom tiling strategy
    - or a temporary newer JAX sandbox if `@pl.kernel` / newer lowering paths are materially better there

### 2026-03-08 00:50 - Isolated newer-JAX SC scatter repro
- I created a standalone repro in [`bench_sparsecore_scatter_repro.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_sparsecore_scatter_repro.py) that imports only JAX/Pallas so it can run in an isolated environment.
- I verified that a temporary TPU env with `jax[tpu]==0.9.0.1` is usable on the dev TPU:
  - `jax 0.9.0.1`
  - TPU devices visible normally
- Baseline in the isolated `0.9.0.1` env is unchanged:
  - `xla` unique scatter on `rows=163840, updates=40960, width=2048, dtype=bf16`: `steady_s=0.003285`
- I then tested two SC scatter routes in the isolated `0.9.0.1` env:
  - old-style `pallas_call` adaptation
  - newer docs-style `@pl.kernel` / `emit_pipeline` path
- Result:
  - the old-style path still did not yield a usable indexed HBM scatter
  - the docs-style `@pl.kernel` path got further, but still failed verification with a pipeline/layout mismatch on the VMEM staging buffer (`expected 1x128x2048, got 128x2048`)
- Interpretation:
  - the repo’s current `0.8` stack is not the whole story
  - even on `0.9.0.1`, a straightforward adaptation of the documented SC scatter path is not working yet on this `v5p` setup
  - this does not prove SC scatter is impossible, but it does mean the “easy” route is not there; the next attempt would need lower-level kernel work or a more exact reproduction of the docs kernel structure than the current adaptation

### 2026-03-08 01:15 - Deferred final reduction removes most of the scatter penalty
- Added two benchmark-only scatter finalization variants to `lib/levanter/scripts/bench/bench_sparsecore_ep_overlap.py`:
  - `deferred_segment_sum`
  - `deferred_coalesced_set`
- Idea:
  - stop doing `out_global.at[token_ids].add(...)` once per expert chunk
  - instead collect all chunk contributions and do one final reduction by `token_id`
- EP4-matched `pipeline_sc_ep` results on `v5p-8`:
  - previous `naive` scatter: `0.131705 s`, `1.244M tok/s`
  - `deferred_coalesced_set`: `0.134397 s`, `1.219M tok/s`
  - `deferred_segment_sum`: `0.107178 s`, `1.529M tok/s`
- Interpretation:
  - the repeated per-chunk scatter-add really was the dominant pathology in this benchmark
  - simply changing the final accumulation pattern recovers about `+22.9%` over the earlier pipelined SC EP path
  - the `set`-after-coalesce formulation still loses; the useful change is the single dense final `segment_sum`

### 2026-03-08 01:25 - Chunk sweep with deferred final reduction
- Swept `chunk_experts` for `pipeline_sc_ep` with `scatter_implementation=deferred_segment_sum`:
  - `chunk_experts=4`: `0.106828 s`, `1.534M tok/s`
  - `chunk_experts=8`: `0.107178 s`, `1.529M tok/s`
  - `chunk_experts=16`: `0.106506 s`, `1.538M tok/s`
  - `chunk_experts=32`: `0.101803 s`, `1.609M tok/s`
- Interpretation:
  - the improvement is not a narrow chunk-size sweet spot
  - larger chunks are slightly better, and the best point is actually `chunk_experts=32` (one chunk per local expert shard)
  - that means the gain is coming from the scatter reformulation, not from improved SC/TC overlap

### 2026-03-08 01:35 - With scatter fixed, SC gather is now the clear remaining regression
- Added a benchmark-only `pipeline_xla_ep` implementation that keeps the same chunked/deferred schedule but uses `_take_rows_impl(..., implementation="xla")` instead of SparseCore gather.
- Head-to-head on the same EP4-matched shape with `chunk_experts=32` and `deferred_segment_sum`:
  - `pipeline_xla_ep`: `0.090418 s`, `1.812M tok/s`
  - `pipeline_sc_ep`: `0.101803 s`, `1.609M tok/s`
- Comparison to the earlier baseline:
  - `xla_ep`: `0.090513 s`, `1.810M tok/s`
- Interpretation:
  - the deferred-final-reduction schedule can match baseline exactly when gather stays on the XLA path
  - SparseCore gather remains about `11.2%` slower in this best-found formulation
  - so the remaining evidence does **not** support “SC for token take helps” on this workload; the big recoverable win was reducing the scatter pattern, not moving gather to SparseCore

### 2026-03-08 01:50 - Profile comparison of the best deferred-scatter variants
- Captured and summarized:
  - `pipeline_sc_ep`, `chunk_experts=32`, `scatter_implementation=deferred_segment_sum`
  - `pipeline_xla_ep`, `chunk_experts=32`, `scatter_implementation=deferred_segment_sum`
- Local summaries:
  - `scratch/sc_ep_sc_deferred32_profile_20260308_summary.json`
  - `scratch/sc_ep_xla_deferred32_profile_20260308_summary.json`
- The final deferred scatter cost is effectively identical:
  - SC path: `loss_fn=>scatter_finalize=>scatter-add = 159657.659`
  - XLA path: `loss_fn=>scatter_finalize=>scatter-add = 159653.965`
- The MLP cost is also effectively identical:
  - SC path: `loss_fn=>mlp_chunk_0 = 268839.483`
  - XLA path: `loss_fn=>mlp_chunk_0 = 270150.568`
- The big difference is extra layout/conversion work that only appears on the SC gather path:
  - `loss_fn=>convert.6 = 30966.438`
  - `loss_fn=>reshape.327 = 24544.314`
  - `loss_fn=>broadcast.317 = 13511.510`
- End-to-end totals from the summaries:
  - SC path: `loss_fn = 725713.367`
  - XLA path: `loss_fn = 645781.876`
- Interpretation:
  - after fixing scatter, the remaining SC penalty is not in the final reduction and not in the expert matmuls
  - it is mostly gather-adjacent layout/convert overhead introduced by the current SparseCore gather formulation
  - this makes the overall conclusion stronger: the best local win in this thread came from changing the scatter decomposition, while SparseCore gather still loses to XLA once the scatter path is no longer pathological

### 2026-03-08 02:05 - Gather block-size tuning did not recover the remaining SC gap
- Added benchmark-only env overrides to `lib/levanter/src/levanter/grug/grug_moe_sparsecore.py`:
  - `GRUG_SPARSECORE_GATHER_BLOCK_SIZE`
  - `GRUG_SPARSECORE_GATHER_VMEM_WORD_BUDGET`
- Baseline best SC deferred-scatter point remained:
  - `chunk_experts=32`, default gather block size selection (`16`): `0.101803 s`, `1.609M tok/s`
- Tuning results on the same EP4-matched case:
  - `block_size=24`: `0.105031 s`, `1.560M tok/s`
  - `block_size=12`: `0.105162 s`, `1.558M tok/s`
  - `block_size=32`: failed VMEM allocation verification
- Failure detail for `block_size=32`:
  - `memref.alloca ... exceeds the legitimate user allocatable offset upper bound ... when allocating 65536 words`
- Interpretation:
  - the helper’s default `16` is near the useful ceiling on this workload
  - simple tile-size tuning does not close the remaining SC gap

### 2026-03-08 02:15 - Pre-bitcasting the activation buffer was a false win
- I briefly added a benchmark-only `pipeline_sc_prebitcast_ep` path that bitcast the bf16 activation buffer to `int32` once before SparseCore gather.
- It looked very fast on the large benchmark:
  - `0.081598 s`, `2.008M tok/s`
- But a TPU-side gradient check on a smaller aligned case showed the catch:
  - loss matched exactly
  - weight gradients matched exactly
  - `x` gradient norm dropped from `55893.8867` to `0.0`
- So the speedup came from silently severing the activation-grad path through the `int32` reinterpret cast, not from a real gather improvement.
- I removed this invalid benchmark path from `bench_sparsecore_ep_overlap.py`.

### 2026-03-08 02:30 - Correct reusable pre-bitcast gather is valid but not materially faster
- I added a benchmark-only helper in `lib/levanter/src/levanter/grug/grug_moe_sparsecore.py`:
  - `sparsecore_row_gather_bf16_prebitcast(x_bf16, x_i32, row_indices)`
- Idea:
  - compute the bf16->int32 view once outside the chunk loop
  - gather from the reused `int32` view in the forward pass
  - define a custom VJP so gradients still scatter back to the bf16 activation buffer
- TPU-side gradient check on a small aligned EP case:
  - loss matched exactly
  - all gradient norms matched exactly
  - grad-diff norms were `[0.0, 0.0, 0.0]`
- Full EP4-matched benchmark result (`chunk_experts=32`, `deferred_segment_sum`):
  - `pipeline_sc_ep`: `0.101803 s`, `1.609M tok/s`
  - `pipeline_sc_reusebitcast_ep`: `0.101746 s`, `1.610M tok/s`
- Interpretation:
  - reusing the bitcast once is valid
  - but it does not materially change throughput on this workload
  - so the remaining SC gather regression is not just repeated bf16<->int32 view setup inside the chunk loop

### 2026-03-08 02:45 - Forward-only benchmark shows the SC gap is already present without backward
- I ran a value-only version of the EP4-matched deferred-scatter benchmark on TPU (no gradients, same `chunk_experts=32` setup):
  - `xla`: `0.036019 s`, `4.549M tok/s`
  - `sparsecore`: `0.047829 s`, `3.426M tok/s`
  - `sparsecore_prebitcast_reuse`: `0.047758 s`, `3.431M tok/s`
- Interpretation:
  - the remaining SC-vs-XLA gap is already large in pure forward
  - so this is not primarily a training-only or backward-only problem
  - it strengthens the conclusion that the current SparseCore gather formulation itself is the bottleneck

### 2026-03-08 02:55 - `plsc.bitcast` does not lower as a normal TPU op outside SC lowering
- I tried swapping the bf16/int32 lane reinterpret in `grug_moe_sparsecore.py` from ordinary JAX `.view(...)` to `plsc.bitcast(...)`, since the extra profile noise looked like layout work around the current view-based path.
- Result:
  - compile failed on TPU with `NotImplementedError: MLIR translation rule for primitive 'bitcast' not found for platform tpu`
- Interpretation:
  - `plsc.bitcast` is not usable here as a drop-in outside the SparseCore lowering context
  - I reverted that change immediately

### 2026-03-08 03:05 - Dedup-and-expand gather is not promising, even at high top-k
- Added benchmark-only gather variants to `lib/levanter/scripts/bench/bench_sparsecore_ep_overlap.py`:
  - `pipeline_xla_dedup_ep`
  - `pipeline_sc_dedup_ep`
- Idea:
  - deduplicate repeated token ids inside a chunk
  - gather unique token rows once
  - expand back to assignment order locally
- On the matched EP4 `topk=4` case (`chunk_experts=32`, `deferred_segment_sum`):
  - `pipeline_xla_ep`: `1.812M tok/s`
  - `pipeline_xla_dedup_ep`: `1.484M tok/s`
  - `pipeline_sc_ep`: `1.609M tok/s`
  - `pipeline_sc_dedup_ep`: `1.355M tok/s`
- Since dedup should help more at higher duplicate rate, I also ran a value-only `topk=16` case:
  - `xla`: `1.474M tok/s`
  - `xla_dedup_expand`: `1.119M tok/s`
  - `sparsecore`: `1.198M tok/s`
  - `sparsecore_dedup_expand`: `0.948M tok/s`
- Interpretation:
  - duplicate token fetches are not the dominant issue in the current formulation
  - the sort/segment/inverse-map overhead of dedup-and-expand outweighs the reduced HBM fetches
  - this remains true even in a much higher-topk value-only case where the intuition should have been strongest

### 2026-03-08 03:20 - Higher top-k narrows the SC gap slightly, but does not reverse it
- I reran the full EP4-matched training benchmark with the current best benchmark formulation:
  - `chunk_experts=32`
  - `scatter_implementation=deferred_segment_sum`
  - compare `pipeline_xla_ep` vs `pipeline_sc_ep`
- Results:
  - `topk=8`
    - `pipeline_xla_ep`: `0.158236 s`, `1.035M tok/s`
    - `pipeline_sc_ep`: `0.174734 s`, `0.938M tok/s`
    - SC/XLA ratio: `0.906x`
  - `topk=16`
    - `pipeline_xla_ep`: `0.295453 s`, `0.555M tok/s`
    - `pipeline_sc_ep`: `0.320992 s`, `0.510M tok/s`
    - SC/XLA ratio: `0.920x`
- Comparison to `topk=4`:
  - `pipeline_xla_ep`: `1.812M tok/s`
  - `pipeline_sc_ep`: `1.609M tok/s`
  - SC/XLA ratio: `0.888x`
- Interpretation:
  - the SC gap does shrink slightly as top-k rises and routing gets more painful
  - but it does not close on `v5p-8` in this formulation
  - so “high top-k is where this wins” is only weakly supported by the current evidence; high top-k makes SC less bad, not actually better

### 2026-03-08 03:35 - Smaller expert count does not materially improve the SC ratio
- I swept `num_experts` at fixed EP4-matched local geometry, always using one chunk per local expert shard:
  - `experts=32 -> chunk_experts=8`
  - `experts=64 -> chunk_experts=16`
  - `experts=128 -> chunk_experts=32`
- Results for `topk=8`:
  - `experts=32`
    - `pipeline_xla_ep`: `1.053M tok/s`
    - `pipeline_sc_ep`: `0.946M tok/s`
    - ratio: `0.898x`
  - `experts=64`
    - `pipeline_xla_ep`: `1.045M tok/s`
    - `pipeline_sc_ep`: `0.946M tok/s`
    - ratio: `0.905x`
  - `experts=128`
    - `pipeline_xla_ep`: `1.035M tok/s`
    - `pipeline_sc_ep`: `0.938M tok/s`
    - ratio: `0.906x`
- Results for `topk=16`:
  - `experts=32`
    - `pipeline_xla_ep`: `0.562M tok/s`
    - `pipeline_sc_ep`: `0.512M tok/s`
    - ratio: `0.910x`
  - `experts=64`
    - `pipeline_xla_ep`: `0.557M tok/s`
    - `pipeline_sc_ep`: `0.512M tok/s`
    - ratio: `0.918x`
  - `experts=128`
    - `pipeline_xla_ep`: `0.555M tok/s`
    - `pipeline_sc_ep`: `0.510M tok/s`
    - ratio: `0.920x`
- Interpretation:
  - decreasing expert count (and thus increasing average tokens per local expert) does not materially change the SC/XLA ratio
  - expert count is a very weak lever here compared with the overall routing intensity (`topk`)
  - this further supports the view that the remaining limitation is in the current SparseCore gather path itself, not in per-expert fragmentation on the TC side
