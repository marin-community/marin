# Grug MoE EP Ring: Research Logbook

## Scope
- Goal: make [`lib/levanter/src/levanter/grug/grug_moe.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/src/levanter/grug/grug_moe.py) the fastest functional Grug MoE path for the target Qwen3 32B a4b MoE shape, especially at higher expert-parallel (`EP`) ranks.
- Primary metric(s): end-to-end `forward_backward` wall time for the focused harness on `v5p-16`, comparing `legacy` vs `current` kernels at `EP=1,2,4,8`.
- Constraints: do not open a new GitHub issue; preserve numerical equivalence with the pre-change EP ring path.

## Baseline
- Date: 2026-03-06
- Code refs:
  - [`lib/levanter/src/levanter/grug/grug_moe.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - [`lib/levanter/scripts/bench/bench_moe_hillclimb.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/scripts/bench/bench_moe_hillclimb.py)
- Baseline numbers: pending TPU measurement.
- Reference shape:
  - W&B run: `grug-moe-qwen3-32b-a4b-v5p64-perf`
  - v5p-64 training batch size: `32`
  - v5p-16 benchmark tokens: `8 * 4096 = 32768`
  - hidden=`2048`, mlp_dim=`768`, experts=`128`, topk=`8`, shared_expert_dim=`2048`

## Experiment Log
### 2026-03-06 18:00 - Current kernel compaction change
- Hypothesis: high-EP slowdown is dominated by the global `argsort` plus fused full-length `take`s in `_moe_mlp_ep_ring_local`, and replacing that with direct local-assignment compaction should materially reduce time at `EP>=2`.
- Command: local code edit plus focused correctness harness.
- Config:
  - Replaced global sort/filter path with `local_mask` + ordered `top_k` selection over only local assignments.
  - Added a focused benchmark harness that compares `legacy` and `current` kernels and can check forward/grad equivalence.
- Result:
  - CPU smoke equivalence passed for tiny shapes.
  - Existing Grug MoE tests passed locally.
- Interpretation: the new EP ring path is functionally plausible and ready for TPU timing.
- Next action: finish multihost harness startup and run `v5p-16` timings for `EP=1,2,4,8`.

### 2026-03-06 23:45 - Multihost TPU harness validation
- Hypothesis: the focused harness can run directly on a 2-host `v5p-16` slice once it initializes `jax.distributed`, and `legacy` vs `current` remain numerically identical in EP mode.
- Command:
  - Bootstrap two-worker slice `ray-marin-us-central1-worker-e565cfbd-tpu`
  - Run:
    ```bash
    gcloud compute tpus tpu-vm ssh ray-marin-us-central1-worker-e565cfbd-tpu \
      --zone us-central1-a --worker=all --command '
      set -e
      source "$HOME/.local/bin/env"
      cd "$HOME/marin"
      PROC_ID=$(hostname | sed -E "s/.*-w-([0-9]+)$/\1/")
      export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"
      uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
        --coordinator-address 10.128.0.5:12355 \
        --num-processes 2 \
        --process-id "$PROC_ID" \
        --tokens 512 \
        --hidden 256 \
        --mlp-dim 128 \
        --experts 16 \
        --topk 4 \
        --shared-expert-dim 128 \
        --ep-list 1,2,4,8 \
        --iters 1 \
        --warmup 0 \
        --check-equivalence'
    ```
- Config:
  - 8 TPU devices total across 2 hosts
  - `bench_pass=forward_backward`
  - small TPU-valid GMM shape (`mlp_dim=128`)
- Result:
  - `CHECK ep=1 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=2 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=4 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=8 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
- Interpretation: the current EP-ring compaction path matches the legacy implementation for both forward outputs and backward grads on TPU across the full `EP=1,2,4,8` sweep.
- Next action: run full-shape timings without equivalence mode.

### 2026-03-07 00:00 - Full-shape `v5p-16` benchmark
- Hypothesis: at the Qwen3 32B a4b shape, removing the global sort/full-length take sequence from the EP ring path will be neutral at `EP=1` and increasingly beneficial as `EP` rises.
- Command:
  ```bash
  gcloud compute tpus tpu-vm ssh ray-marin-us-central1-worker-e565cfbd-tpu \
    --zone us-central1-a --worker=all --command '
    set -e
    source "$HOME/.local/bin/env"
    cd "$HOME/marin"
    PROC_ID=$(hostname | sed -E "s/.*-w-([0-9]+)$/\1/")
    export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"
    uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --coordinator-address 10.128.0.5:12355 \
      --num-processes 2 \
      --process-id "$PROC_ID" \
      --tokens 32768 \
      --hidden 2048 \
      --mlp-dim 768 \
      --experts 128 \
      --topk 8 \
      --shared-expert-dim 2048 \
      --ep-list 1,2,4,8 \
      --iters 3 \
      --warmup 1'
  ```
- Config:
  - Shape derived from W&B run `grug-moe-qwen3-32b-a4b-v5p64-perf`
  - v5p-64 training batch size `32`, reduced by `4x` to v5p-16 benchmark tokens `32768`
  - `bench_pass=forward_backward`
  - routing distributions tested: `random`, `runs`
- Result:
  - `random`
    - `EP=1`: legacy `24.330 ms`, current `24.275 ms`
    - `EP=2`: legacy `25.229 ms`, current `23.473 ms`
    - `EP=4`: legacy `22.986 ms`, current `20.121 ms`
    - `EP=8`: legacy `25.939 ms`, current `17.882 ms`
  - `runs`
    - `EP=1`: legacy `24.381 ms`, current `24.302 ms`
    - `EP=2`: legacy `25.209 ms`, current `23.406 ms`
    - `EP=4`: legacy `23.066 ms`, current `20.126 ms`
    - `EP=8`: legacy `26.609 ms`, current `17.811 ms`
- Interpretation:
  - The current kernel is effectively neutral at `EP=1` (`~0.3%` faster on average).
  - It is clearly better at `EP=2` (`~7.6%`), `EP=4` (`~14.4%`), and especially `EP=8` (`~47.2%`).
  - Under the current kernel on this `v5p-16`, `EP=8` is the fastest setting; average current times were:
    - `EP=1`: `24.289 ms`
    - `EP=2`: `23.440 ms`
    - `EP=4`: `20.124 ms`
    - `EP=8`: `17.847 ms`
  - Relative to current `EP=1`, current `EP=8` is `~1.36x` faster for total `forward_backward` time.
- Next action: keep the compact local-selection path in `grug_moe.py`; recommend `EP=8` for this shape on `v5p-16`, with `EP=4` the next-best fallback.

### 2026-03-07 08:00 - Profiling follow-on
- Hypothesis: after removing the global sort/full-length take bottleneck, the next EP limit is more likely collective communication (`all_gather` / `psum_scatter`) than local routed compute.
- Command: extend the focused harness with optional JAX trace capture for a single `(kernel, EP)` case.
- Config:
  - Add `--profile-root` to write a standard `jax_profile` directory.
  - Use `jax.profiler.StepTraceAnnotation` around each profiled benchmark iteration.
  - Keep trace capture opt-in and restricted to one kernel / one EP value so profile provenance stays unambiguous.
- Result: pending.
- Interpretation: this should let us capture representative TPU traces for `current EP=1` and `current EP=8`, then use `lib/marin/tools/profile_summary.py` to answer whether the remaining EP cost is comm- or compute-dominated.
- Next action: validate the profiling mode locally, capture TPU traces, and ingest/query them.

### 2026-03-07 08:10 - Forward/backward trace capture
- Hypothesis: full-shape traces will show whether the next EP bottleneck is collective time or local routing/packing work.
- Command:
  - Capture `current EP=8` and `current EP=1` full-shape traces on `ray-marin-us-east5-a-worker-6548e010-tpu` with `--profile-root`.
  - Pull worker-0 traces locally into:
    - `scratch/profiles/grug-current-ep8`
    - `scratch/profiles/grug-current-ep1`
  - Ingest with:
    ```bash
    uv run python lib/marin/tools/profile_summary.py summarize \
      --profile-dir scratch/profiles/grug-current-ep8 \
      --output scratch/profiles/grug-current-ep8-summary.json
    ```
- Config:
  - shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
  - `bench_pass=forward_backward`
  - `kernel=current`
- Result:
  - `EP=8` summary:
    - compute share `52.4%`
    - communication share `3.2%`
    - top hierarchical regions:
      - `moe_mlp=>gather=>_take=>gather`: `36.7%` exclusive
      - `moe_mlp=>scatter=>scatter-add`: `10.7%` exclusive
      - `moe_mlp=>moe_up_down=>gmm`: `14.8%` exclusive
  - `EP=1` summary is confounded by backward reduction traffic:
    - communication share `11.0%`
    - top ops include backward `psum` on parameter gradients, which is not the forward EP-ring path of interest.
- Interpretation:
  - The first `forward_backward` pass already points away from EP collectives as the primary next bottleneck at `EP=8`.
  - But `EP=1` vs `EP=8` is not an apples-to-apples comparison for diagnosing the forward ring path because backward gradient reductions dominate the `EP=1` communication list.
- Next action: profile `bench_pass=forward` for `current EP=8` to isolate the ring-dispatch path.

### 2026-03-07 08:15 - Forward-only EP trace
- Hypothesis: if forward-only `EP=8` still shows local gather/scatter materialization dominating over collectives, the next worthwhile redesign is packed dispatch rather than a more clever collective primitive alone.
- Command:
  ```bash
  gcloud compute tpus tpu-vm ssh ray-marin-us-east5-a-worker-6548e010-tpu \
    --zone us-east5-a --worker=all --command '
    set -e
    source "$HOME/.local/bin/env"
    cd "$HOME/marin"
    PROC_ID=$(hostname | sed -E "s/.*-w-([0-9]+)$/\1/")
    export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"
    uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --coordinator-address 10.202.0.95:12355 \
      --num-processes 2 \
      --process-id "$PROC_ID" \
      --tokens 32768 \
      --hidden 2048 \
      --mlp-dim 768 \
      --experts 128 \
      --topk 8 \
      --shared-expert-dim 2048 \
      --kernel current \
      --ep-list 8 \
      --bench-pass forward \
      --iters 3 \
      --warmup 1 \
      --profile-root "$HOME/marin/scratch/profiles/grug-current-ep8-forward"'
  ```
- Config:
  - same full shape as benchmark
  - `bench_pass=forward`
  - worker-0 trace ingested to `scratch/profiles/grug-current-ep8-forward-summary.json`
- Result:
  - compute share `41.0%`
  - communication share `2.1%`
  - hierarchical regions:
    - `_forward=>moe_mlp=>gather=>_take=>gather`: `55.6%` exclusive
    - `_forward=>moe_mlp=>scatter=>scatter-add`: `17.7%` exclusive
    - `_forward=>moe_mlp=>moe_up_down=>gmm`: `9.7%` exclusive
  - top ops:
    - `grug_moe.py:150` gather/take fusion: `61.8 ms`
    - `grug_moe.py:196` scatter-add fusion: `58.8 ms`
    - `grug_moe.py:152` gather/take fusion: `58.1 ms`
    - `all-reduce`: `26.9 ms`
    - `grug_moe.py:169` gather/scatter-add fusion: `23.6 ms`
    - `gmm.2`: `23.2 ms`
- Interpretation:
  - The next EP bottleneck is not the ring collective itself; it is the local materialization path around dispatch and collection.
  - A more aggressive packed-dispatch design is still attractive, but the likely win comes from eliminating the current global-view gather/take/scatter materialization, not from swapping one collective for another in isolation.
  - Because v5p is bidirectional-ring connected, any packed-dispatch redesign should stay ring-aware:
    - pack tokens by destination expert shard,
    - pipeline chunks in both ring directions,
    - avoid a logical all-to-all that ignores physical topology.
- Next action:
  - Prototype a ring-aware packed dispatch that forwards only destination-owned assignment chunks.
  - Measure whether it reduces the current `gather/_take` and `scatter-add` regions without inflating ring-hop overhead.

### 2026-03-07 00:20 - Return-path-only packing experiment
- Hypothesis: the `scatter-add` / `psum_scatter` half of the current EP ring path might be improved by packing routed outputs by owner shard and returning them with `all_to_all`, without touching the inbound dispatch side.
- Command:
  - Added an experimental `packed_return` kernel to the focused harness.
  - Small-shape TPU smoke test on `v5p-16`:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=8`, `bench_pass=forward_backward`, `iters=1`, `warmup=0`
  - Full-shape TPU benchmark on `v5p-16`:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=8`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - Small shape:
    - `current`: `0.669 ms`
    - `packed_return`: `0.764 ms`
  - Full shape:
    - `current`: `17.866 ms`
    - `packed_return`: `23.360 ms`
- Interpretation:
  - Replacing the return path in isolation is a clear regression.
  - The extra pack/unpack plus `all_to_all` overhead is larger than the `psum_scatter` cost it removes.
  - This confirms the profile-driven suspicion that the best next target is not “swap the collective primitive,” but “remove more of the global materialization path.”
- Next action: drop `packed_return` as a candidate direction and focus on local-compaction and streamed-ring designs.

### 2026-03-07 00:25 - Routing skew check for streamed-ring sizing
- Hypothesis: on the target benchmark shape, the per-source routed load into any destination shard is tight enough that a streamed ring can use a small fixed per-hop capacity rather than worst-case full-chunk padding.
- Command:
  ```bash
  uv run python - <<'PY'
  import importlib.util
  import math
  import numpy as np
  import jax

  path = 'lib/levanter/scripts/bench/bench_moe_hillclimb.py'
  spec = importlib.util.spec_from_file_location('bench_moe_hillclimb', path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  TOKENS=32768
  EXPERTS=128
  TOPK=8
  EP=8
  LOCAL_EXPERTS=EXPERTS//EP
  A=TOKENS*TOPK
  CAP=math.ceil(1.25*A/EP)
  TL=TOKENS//EP
  for dist in ['random', 'runs']:
      key = jax.random.PRNGKey(0)
      key_router = jax.random.split(key, 6)[1]
      logits = mod._sample_router_logits(
          key_router,
          tokens=TOKENS,
          experts=EXPERTS,
          distribution=dist,
          run_alpha=0.98,
          run_noise_scale=0.35,
      )
      selected, _ = mod._route_topk(logits, topk=TOPK)
      selected = np.array(selected)
      shard = selected // LOCAL_EXPERTS
      shard_counts = np.bincount(shard.reshape(-1), minlength=EP)
      per_source = np.zeros((EP, EP), dtype=np.int64)
      for src in range(EP):
          src_selected = selected[src * TL : (src + 1) * TL]
          src_shard = src_selected // LOCAL_EXPERTS
          per_source[src] = np.bincount(src_shard.reshape(-1), minlength=EP)
      print(dist, shard_counts.min(), shard_counts.max(), per_source.max())
  PY
  ```
- Result:
  - `random`:
    - global shard-count min/max: `32583 / 33063`
    - worst per-source-to-destination count: `4235`
  - `runs`:
    - global shard-count min/max: `32504 / 32937`
    - worst per-source-to-destination count: `4617`
  - Current full-shape global per-shard capacity at `EP=8` is `40960`.
  - A per-hop streamed-ring capacity of `ceil(1.25 * (4096 * 8) / 8) = 5120` covers the measured worst per-source load for both tested routing distributions.
- Interpretation:
  - For the specific benchmark shape under both `random` and `runs`, the target shape does not hit capacity overflow globally and does not require worst-case full-chunk per-hop buffers.
  - That makes a streamed-ring prototype plausible on this workload: it can use a per-hop bound near the statistical mean instead of padding to the full chunk.
- Next action: prototype a streamed EP ring in the harness using per-hop capacity based on chunk assignments and test whether avoiding global materialization beats the extra ring traffic.

### 2026-03-07 00:35 - Alternative compaction and streamed-ring filters
- Hypothesis:
  - `cumsum`: replacing the `top_k`-based local compaction with expert-wise prefix packing may beat the current `top_k` / `take` selection machinery while preserving the current ring structure.
  - `stream_ring`: if global materialization is the real bottleneck, circulating per-source chunks around the ring may outperform `all_gather` + `psum_scatter` even with more explicit ring traffic.
- Command:
  - Added experimental `cumsum` and `stream_ring` kernels to the focused harness.
  - `cumsum` small-shape TPU check on `v5p-16`:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=8`, `bench_pass=forward_backward`, `iters=1`, `warmup=0`
  - `stream_ring` filtered on a ready `v5p-8` slice (4 visible TPU devices, so `EP=4` only):
    - small shape: same as above with `EP=4`
    - full shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`, `EP=4`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - `cumsum` on `v5p-16`, small shape, `EP=8`:
    - `current`: `0.669 ms`
    - `cumsum`: `0.764 ms`
  - `stream_ring` on `v5p-8`, small shape, `EP=4`:
    - `current`: `0.763 ms`
    - `stream_ring`: `1.189 ms`
  - `stream_ring` on `v5p-8`, full shape, `EP=4`:
    - `current`: `33.003 ms`
    - `stream_ring`: `40.589 ms`
- Interpretation:
  - `cumsum` looks worse than the current `top_k` selection path on the TPU-valid small case; unless the full-shape `v5p-16` says otherwise, it is not promising.
  - The unidirectional streamed ring is a clear regression on the `v5p-8` filter. Carrying activations plus an accumulating output buffer around the ring appears to cost more than the global materialization it removes.
  - This does not rule out a better ring-aware design entirely, but it strongly suggests that a naive streamed ring is not competitive enough to justify moving into production code.
- Next action:
  - Try to get a healthy `v5p-16` slice for direct `stream_ring` / `cumsum` confirmation on the target hardware.
  - If the `v5p-16` result matches the filter, stop investing in these two directions and look for more surgical improvements inside the current ring path.

### 2026-03-07 00:40 - `v5p-16` slice health note
- Hypothesis: a freshly provisioned `v5p-16` slice may need extra settle time before JAX can discover global TPU topology.
- Command:
  - Provisioned spot `v5p-16` slices in `us-central1-a` (`codex-grug-ep-0307a`) and `us-east5-a` (`codex-grug-ep-0307b`).
  - Bootstrapped `codex-grug-ep-0307a` and attempted a basic multihost JAX run.
- Result:
  - `codex-grug-ep-0307a` reached `READY` but both workers failed TPU init with:
    - `RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology.`
  - A second `v5p-16` in `us-central1-a` (`ray-marin-us-central1-worker-14d9e7d5-tpu`) became `READY` and is being bootstrapped as an alternative target.
- Interpretation:
  - The first slice looks unhealthy or not fully settled despite showing `READY`.
  - This appears to be infrastructure/runtime state, not a harness regression.
- Next action: finish bootstrapping the alternate `v5p-16` and resume the target-hardware confirmations there.

### 2026-03-07 00:55 - Target-hardware confirmation on healthy `v5p-16`
- Hypothesis: the `v5p-8` filter results will carry over to the target `v5p-16`: neither `cumsum` nor the unidirectional `stream_ring` will beat the current ring EP path.
- Command:
  - Used healthy slice `ray-marin-us-central1-worker-14d9e7d5-tpu` in `us-central1-a`.
  - Small-shape `stream_ring` smoke:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=8`, `bench_pass=forward_backward`, `iters=1`, `warmup=0`
  - Full-shape target benchmark:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=8`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - kernels: `current`, `cumsum`, `stream_ring`
- Result:
  - Small shape:
    - `stream_ring`: `1.884 ms`
  - Full shape:
    - `current`: `17.881 ms`
    - `cumsum`: `86.922 ms`
    - `stream_ring`: `25.235 ms`
- Interpretation:
  - `cumsum` is not competitive; the expert-mask / prefix-pack approach is dramatically slower than the current `top_k` compaction on the real target shape.
  - `stream_ring` is also a regression on target hardware, though far less catastrophic than `cumsum`.
  - The current `grug_moe.py` ring path remains the best implementation tested by a wide margin.
  - The practical takeaway is that more EP work on TPU should stay focused on surgical improvements inside the existing ring design, or on a significantly more topology-aware dispatch scheme than the naive unidirectional streamed ring.
- Next action:
  - Treat `packed_return`, `cumsum`, and naive `stream_ring` as explored-and-rejected benchmark branches.
  - If continuing this thread, shift to more surgical experiments in the current ring path or to a truly bidirectional / distance-aware dispatch design rather than simple stream-around-the-ring variants.

### 2026-03-07 01:15 - MaxText MoE pathway review
- Hypothesis: MaxText may already contain a TPU MoE dispatch strategy that is closer to the right next move than the ring-style experiments tried here.
- Sources:
  - MaxText repo at commit `ed706bea318b64565933dc71207bf2eede141db1`
  - `src/MaxText/layers/moe.py`
  - `src/MaxText/configs/types.py`
  - `src/MaxText/configs/base.yml`
  - `tests/end_to_end/tpu/deepseek/Run_DeepSeek.md`
  - `tests/end_to_end/tpu/mixtral/Run_Mixtral.md`
- Result:
  - MaxText has two distinct EP sparse-matmul families in `layers/moe.py`:
    - `use_ring_of_experts=True`:
      - duplicates `x`, `logits`, and `pre_bias_logits` with `all_gather`
      - routes locally after duplication
      - reduces outputs with `psum_scatter`
      - this is conceptually close to the current Grug ring path
    - `use_ring_of_experts=False` with EP:
      - computes global group sizes first
      - derives `input_offsets`, `send_sizes`, `output_offsets`, `recv_sizes`
      - dispatches with `jax.lax.ragged_all_to_all`
      - locally re-permutes by local expert id
      - runs grouped matmuls
      - returns outputs with another `ragged_all_to_all`
      - then unpermutes back to token order
  - MaxText’s documented default TPU MoE story is dropless sparse matmul, not ring-of-experts:
    - `base.yml` defaults to `megablox: true`, `sparse_matmul: true`, `capacity_factor: -1.0`, `use_ring_of_experts: false`
    - DeepSeek and Mixtral docs call out dropless MegaBlocks / ragged-dot as the main supported high-performance path.
  - MaxText also carries a few smaller implementation ideas worth noting:
    - custom VJP sort/unsort helper for permutations
    - explicit group-size-driven local re-permute (`local_permute`) after communication
    - TPU-oriented grouped-matmul backends (`megablox` / `tokamax`) with tile and buffer-count tuning
- Interpretation:
  - The important MaxText idea is not “their ring implementation is better.” Their ring-of-experts branch still starts with `all_gather`, so it lives in the same design family we already profiled and partially optimized.
  - The useful idea is the non-ring EP path:
    - first exchange compact routing counts,
    - then move only the routed token payloads with `ragged_all_to_all`,
    - then locally reorder by expert.
  - That lines up with the current profile evidence much better than the naive streamed-ring attempt:
    - it avoids materializing a global assignment view on every shard,
    - it avoids duplicating all tokens/logits to all expert shards,
    - and it keeps the communication volume proportional to routed work rather than full gathered state.
  - It also suggests that, if we keep pushing on TPU EP, the next serious prototype should look more like “count-driven ragged exchange” than “ppermute chunks around a ring.”
- Next action:
  - If building one more substantial harness prototype, make it a MaxText-style EP path:
    - compute per-destination routed counts,
    - use those counts to drive a compact ragged dispatch,
    - locally reorder by expert,
    - grouped expert compute,
    - ragged return,
    - final unpermute/weight combine.
  - Secondary small idea: consider whether a custom-VJP sort/unsort helper is worth testing around any remaining permutation-heavy hot spots in the current Grug path.

### 2026-03-07 02:00 - MaxText-style `ragged_all_to_all` prototype
- Hypothesis: a count-driven sparse dispatch in the MaxText style may beat the current EP ring path by avoiding the global gathered assignment view and moving only routed payloads.
- Command:
  - Added an experimental `ragged_a2a` kernel to `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
  - Mirrored the main MaxText sparse-EP structure:
    - global expert-group histogram via fixed-length `bincount`
    - shard-to-shard count exchange
    - dispatch with `jax.lax.ragged_all_to_all`
    - local re-permute by expert
    - grouped matmuls
    - return with a second `ragged_all_to_all`
    - final unsort / weighted combine
  - Reused MaxText’s custom-VJP sort/unsort pattern for the permutation-heavy steps.
- Result:
  - The harness compiles locally.
  - CPU cannot execute it because XLA:CPU does not implement `ragged-all-to-all`, so TPU validation is required.
- Interpretation:
  - The prototype is close enough to the MaxText design family to be a useful performance filter.
  - This is still a harness experiment, not a production candidate.
- Next action: TPU smoke it, then compare full-shape `forward_backward` time against `current`.

### 2026-03-07 02:15 - `v5p-16` smoke and full-shape filter for `ragged_a2a`
- Hypothesis: if the MaxText-style sparse dispatch is a fit for this workload, the win should show up most clearly at the target `v5p-16`, `EP=8`, full-shape case.
- Command:
  - Used healthy slice `ray-marin-us-central1-worker-14d9e7d5-tpu` in `us-central1-a` before it was later preempted.
  - Small-shape TPU smoke:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=8`, `bench_pass=forward_backward`, `iters=1`, `warmup=0`
    - kernels: `current`, `ragged_a2a`
  - Full-shape target benchmark:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=8`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - kernels: `current`, `ragged_a2a`
- Result:
  - Small shape:
    - `current`: `0.621 ms`
    - `ragged_a2a`: `0.811 ms`
  - Full shape:
    - `current`: `17.743 ms`
    - `ragged_a2a`: `20.569 ms`
- Interpretation:
  - The MaxText-style sparse dispatch is not competitive on the target `v5p-16` shape in this stack.
  - It is meaningfully slower even before any `runs`-distribution follow-up.
- Next action:
  - Profile the ragged path on a healthy TPU slice to see whether the regression is communication or local pack/reorder overhead.
  - If the `v5p-16` slice disappears, fall back to a healthy `v5p-8` filter rather than re-spending time on obviously unhealthy `v5p-16` slices.

### 2026-03-07 02:35 - `v5p-16` slice health follow-up
- Hypothesis: the remaining spare `v5p-16` slices can be used to extend the `ragged_a2a` sweep after the original healthy slice disappeared.
- Command:
  - Tried `codex-grug-ep-0307a` in `us-central1-a`.
  - Bootstrapped `codex-grug-ep-0307b` in `us-east5-a` from scratch:
    - installed `uv`
    - cloned the repo
    - ran `uv sync --all-packages --extra=tpu`
    - copied local `grug_moe.py` and `bench_moe_hillclimb.py`
  - Ran tiny TPU startup checks on both slices.
- Result:
  - Both spare `v5p-16` slices fail at TPU initialization with:
    - `INTERNAL: Failed to get global TPU topology.`
  - The original healthy `ray-marin-us-central1-worker-14d9e7d5-tpu` slice was no longer available by the time the follow-up sweep started.
- Interpretation:
  - This blocks a clean `v5p-16` follow-up sweep tonight.
  - The issue is infrastructure health, not a harness regression.
- Next action: use a healthy, already-bootstrapped `v5p-8` slice as a filter for the remaining MaxText-style investigation.

### 2026-03-07 02:50 - `v5p-8` EP4 filter for `ragged_a2a`
- Hypothesis: if the sparse ragged dispatch has any practical advantage, a smaller but healthy TPU slice should still show at least a directional improvement at `EP=4`.
- Command:
  - Used healthy slice `ray-marin-us-central1-worker-671dced6-tpu` in `us-central1-a`.
  - Copied local `grug_moe.py` and updated harness to the slice.
  - Full-shape benchmark:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - kernels: `current`, `ragged_a2a`
- Result:
  - `current`: `31.207 ms`
  - `ragged_a2a`: `39.269 ms`
- Interpretation:
  - The regression is not specific to the `v5p-16` EP8 case.
  - On a healthy `v5p-8` filter, the MaxText-style sparse dispatch is even less competitive relative to `current` than on the target `v5p-16`.
- Next action: profile `current` and `ragged_a2a` forward on this healthy `v5p-8` slice to see where the extra time lands.

### 2026-03-07 03:10 - `v5p-8` forward profiles: current vs `ragged_a2a`
- Hypothesis: if the sparse ragged path is slower because it is still communication-bound, its trace should show larger collective share than the current ring EP path.
- Command:
  - Captured full-shape `bench_pass=forward` traces on `ray-marin-us-central1-worker-671dced6-tpu`:
    - `current`, `EP=4`
    - `ragged_a2a`, `EP=4`
  - Pulled traces locally to:
    - `scratch/profiles/grug-current-ep4-v5p8`
    - `scratch/profiles/grug-ragged-ep4-v5p8`
  - Ingested with `lib/marin/tools/profile_summary.py summarize`.
- Result:
  - Forward timings during profile run:
    - `current`: `413.128 ms`
    - `ragged_a2a`: `479.528 ms`
  - `current` profile summary:
    - compute share: `40.9%`
    - communication share: `1.0%`
    - host share: `58.1%`
    - top regions:
      - `_forward=>moe_mlp=>scatter=>scatter-add`: `111.5k`
      - `_forward=>moe_mlp=>gather=>_take=>gather`: `81.8k`
      - `_forward=>moe_mlp=>moe_up_down=>gmm`: `64.9k`
      - `_forward=>moe_mlp=>gather=>all_gather`: `16.1k`
  - `ragged_a2a` profile summary:
    - compute share: `47.5%`
    - communication share: `0.03%`
    - host share: `52.5%`
    - top regions:
      - `_forward=>dispatch=>searchsorted=>while=>body=>closed_call=>gather`: `80.1k`
      - `_forward=>moe_up_down=>gmm`: `64.9k`
      - `_forward=>combine=>gather`: `53.8k`
      - `_forward=>dispatch=>gather`: `53.7k`
      - `_forward=>combine=>ragged_all_to_all`: `52.5k`
      - `_forward=>dispatch=>ragged_all_to_all`: `48.4k`
- Interpretation:
  - The sparse ragged path is not losing because communication dominates.
  - It actually has *lower* communication share than `current`, but pays more in local dispatch/combine bookkeeping:
    - the `searchsorted`-driven local re-permute path is expensive,
    - the extra gather materialization is expensive,
    - and the two `ragged_all_to_all` kernels still add substantial cost.
  - In this stack, the savings from avoiding `all_gather` / `psum_scatter` are outweighed by the cost of the extra sparse dispatch machinery.
- Next action:
  - Treat the MaxText-style `ragged_a2a` harness path as explored-and-rejected for the current Grug/TPU stack and benchmark shape.
  - If revisiting this direction, only do so with a significantly different local implementation strategy or backend (for example, a more optimized sparse grouped-matmul / permutation stack than the current harness prototype).

### 2026-03-07 03:20 - `ragged_a2a` correctness spot-check
- Hypothesis: the observed slowdown is a performance issue, not a functional mismatch in the harness implementation.
- Command:
  - Ran a small-shape equivalence spot-check on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=4`
    - compared `current` vs `ragged_a2a` for:
      - forward output
      - backward gradients
- Result:
  - `out_max_abs`: `2.44140625e-04`
  - `grad_max_abs`: `2.9802322387695312e-08`
- Interpretation:
  - The prototype is numerically aligned with the current path at the expected bf16 tolerance.
  - The reason to reject it is performance, not correctness.
- Next action: keep the current ring EP path as the fastest implementation tested; do not move the MaxText-style sparse dispatch into `grug_moe.py`.

### 2026-03-07 03:35 - Ring-path micro-optimizations after the MaxText filter
- Hypothesis: after rejecting the MaxText-style sparse dispatch, there may still be small wins inside the current ring EP path by removing metadata work that is now redundant.
- Command:
  - Added three narrow harness variants to `bench_moe_hillclimb.py`:
    - `prefix_counts`:
      - keep current dispatch ordering,
      - derive accepted per-expert group sizes from a fixed-length local `bincount` plus prefix capacity clipping,
      - drop the extra `expert_local` gather,
      - derive `token_local` as `local_idx // topk` instead of materializing/taking `token_flat`.
    - `segment_sum`:
      - keep current dispatch path,
      - replace scatter-add materialization with `jax.ops.segment_sum`.
    - `prefix_segment_sum`:
      - combine both changes.
  - Ran TPU full-shape filters on healthy `ray-marin-us-central1-worker-671dced6-tpu` (`v5p-8`, `EP=4`):
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - `random` routing:
    - `current`: `31.207 ms`
    - `prefix_counts`: `30.286 ms`
    - `segment_sum`: `30.942 ms`
    - `prefix_segment_sum`: `30.348 ms`
  - `runs` routing:
    - `current`: `31.303 ms`
    - `prefix_counts`: `30.274 ms`
    - `prefix_segment_sum`: `30.358 ms`
- Interpretation:
  - The winning change is `prefix_counts`.
  - `segment_sum` helps only a little; it is not where most of the remaining loss sits.
  - The production-worthy improvement is to stop gathering metadata we can reconstruct cheaply from the existing local mask and static `topk`.
- Next action: validate `prefix_counts` numerically against `current`, then move only that change into `grug_moe.py`.

### 2026-03-07 03:45 - `prefix_counts` equivalence and production promotion
- Hypothesis: the `prefix_counts` harness variant is a semantics-preserving simplification of the current ring path and should carry over directly to `grug_moe.py`.
- Command:
  - TPU equivalence check on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=512 hidden=256 mlp_dim=128 experts=16 topk=4 shared_expert_dim=128`
    - `EP=4`
    - `distribution=runs`
    - compared `current` vs `prefix_counts` for forward output and backward grads
  - Updated `lib/levanter/src/levanter/grug/grug_moe.py`:
    - added `_prefix_cap_counts`
    - removed `token_flat`
    - replaced `expert_local` gather + post-selection `bincount` with:
      - fixed-length local counts,
      - prefix capacity clipping,
      - `token_local = local_idx // topk`
- Result:
  - Equivalence:
    - `out_max_abs`: `0.0`
    - `grad_max_abs`: `0.0`
  - Local tests:
    - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`
    - `3 passed, 2 skipped`
- Interpretation:
  - This is a clean production change:
    - same numerics,
    - less metadata materialization,
    - aligns with the harness result.
- Next action: benchmark the updated production `current` path on TPU to confirm it inherited the same win.

### 2026-03-07 03:55 - Production `grug_moe.py` retest on TPU
- Hypothesis: after the production code change, the canonical `current` kernel should match the `prefix_counts` harness improvement on TPU.
- Command:
  - Synced updated `grug_moe.py` and harness to healthy `ray-marin-us-central1-worker-671dced6-tpu`.
  - Re-ran full-shape `current` benchmarks:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - Updated production `current`:
    - `random`: `30.369 ms`
    - `runs`: `30.380 ms`
  - Previous production `current` on the same healthy `v5p-8` slice:
    - `random`: `31.207 ms`
    - `runs`: `31.303 ms`
- Interpretation:
  - The production ring EP path picked up the expected improvement:
    - `random`: `~2.7%` faster
    - `runs`: `~2.9%` faster
  - This is now the best production ring-path change found after the larger MaxText-style exploration.
  - I still do not have a second healthy `v5p-16` to re-confirm on target hardware after promotion; spare `v5p-16` slices remained topology-broken.
- Next action:
  - Keep this production simplification in `grug_moe.py`.
  - If a healthy `v5p-16` becomes available later, re-run the target `EP=8` confirmation there, but the current evidence is already strong enough to keep the change.

### 2026-03-07 04:05 - Before/after profile for the production simplification
- Hypothesis: the production speedup should show up specifically in the gather-side metadata/materialization region, not in the grouped matmuls or the return collective.
- Command:
  - Captured a new full-shape `bench_pass=forward` trace for the updated production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
  - Pulled the trace to `scratch/profiles/grug-current-ep4-v5p8-after`
  - Ingested with `profile_summary.py summarize`
  - Compared against the earlier baseline profile in `scratch/profiles/grug-current-ep4-v5p8-summary.json`
- Result:
  - Updated production forward profile:
    - `current`: `418.239 ms`
  - Top-region comparison:
    - `_forward=>moe_mlp=>gather=>_take=>gather`
      - before: `81.8k`
      - after: `41.0k`
    - `_forward=>moe_mlp=>scatter=>scatter-add`
      - before: `111.5k`
      - after: `111.5k`
    - `_forward=>moe_mlp=>moe_up_down=>gmm`
      - before: `64.9k`
      - after: `65.0k`
  - `profile_summary compare` also reported:
    - collective family exclusive duration: `-2.8%`
    - broad `other` family exclusive duration: `-7.4%`
- Interpretation:
  - The win is exactly where expected:
    - halving the gather/take metadata/materialization region,
    - leaving scatter-add and GMM essentially unchanged.
  - That is consistent with the code change:
    - stop taking `token_flat`,
    - stop taking `expert_local`,
    - compute accepted per-expert counts from the mask directly.
- Next action: no additional production change from this result; it confirms the current simplified ring path is the right one to keep.

### 2026-03-07 04:20 - Metadata-width experiments
- Hypothesis: after the `prefix_counts` simplification, there may still be a small EP-ring win from shrinking the globally gathered metadata itself:
  - gather `combine_weights` directly in `x.dtype` instead of `float32`
  - gather `selected_experts` in `uint16` and cast back locally
- Command:
  - Added two harness-only variants:
    - `weight_cast`: `prefix_counts` plus bf16 weight gather
    - `narrow_meta`: `prefix_counts` plus bf16 weight gather and `uint16` expert-id gather
  - TPU benchmark on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=4`
    - `distribution=runs`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - `current`: `30.323 ms`
  - `weight_cast`: `30.301 ms`
  - `narrow_meta`: `30.378 ms`
- Interpretation:
  - `weight_cast` is within noise of the already-improved production path.
  - `narrow_meta` regresses slightly.
  - So the remaining bottleneck is not just the nominal byte width of the gathered metadata; the useful win was the metadata materialization simplification itself.
- Next action: do not promote either metadata-width variant into `grug_moe.py`.

### 2026-03-07 04:35 - Capacity-factor sweep after the ring simplification
- Hypothesis: after simplifying the ring-path metadata work, the remaining TPU headroom might come from reducing EP overpadding rather than changing the kernel again.
- Command:
  - Measured the updated production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu` (`v5p-8`, `EP=4`) across a narrow `capacity_factor` sweep.
  - Full shape:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
  - Also computed exact shard-count overflow at `capacity_factor=1.0` for the fixed harness seed:
    - `random`: max overflow `317`
    - `runs`: max overflow `406`
- Result:
  - `random`:
    - `1.00`: `30.370 ms`
    - `1.01`: `30.335 ms`
    - `1.02`: `30.371 ms`
    - `1.05`: `30.291 ms`
    - `1.10`: `30.373 ms`
    - `1.25`: `30.320 ms`
  - `runs`:
    - `1.00`: `30.321 ms`
    - `1.01`: `30.305 ms`
    - `1.02`: `30.328 ms`
    - stopped the remaining points after the sweep was clearly flat within noise
- Interpretation:
  - For this workload, the updated ring path is effectively insensitive to `capacity_factor` in the `1.0` to `1.25` range.
  - So capacity overpadding is not a first-order TPU bottleneck here, even though `1.25` is statistically conservative.
  - This means the next EP TPU win is unlikely to come from default-capacity retuning alone.
- Next action:
  - Keep the production default unchanged for now.
  - Treat capacity tuning as secondary and workload-specific, not a generally useful next optimization thread for this shape.

### 2026-03-07 04:50 - Filtered scatter/count micro-variants
- Hypothesis: after the metadata simplification, the remaining local TPU headroom is either:
  - a tiny amount in `_prefix_cap_counts`, or
  - a better lowering for the hot scatter/add return path.
- Command:
  - Added harness-only variants:
    - `vector_prefix`: replace the small `_prefix_cap_counts` loop with a vectorized `cumsum` formulation
    - `sorted_segment_sum`: sort `token_local` and use `segment_sum(..., indices_are_sorted=True)` on the return path
    - `vector_sorted_segment_sum`: combine both
  - Full-shape TPU benchmark on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
    - `distribution=random`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
  - Follow-up `distribution=runs` benchmark for `vector_prefix` vs `current`.
- Result:
  - `random`:
    - `current`: `32.590 ms`
    - `vector_prefix`: `32.644 ms`
    - `sorted_segment_sum`: `34.405 ms`
    - `vector_sorted_segment_sum`: `34.668 ms`
  - `runs`:
    - `current`: `32.624 ms`
    - `vector_prefix`: `32.614 ms`
- Interpretation:
  - The vectorized capacity clip is functionally fine, but performance-neutral.
  - Sorting the return path to feed `segment_sum(..., indices_are_sorted=True)` regresses clearly.
  - So neither the small prefix loop nor the scatter lowering was the next useful production change.
- Next action: stop investing in sorted-return variants and look at the remaining gather-side count hotspot instead.

### 2026-03-07 05:05 - Replacing gather-side `bincount`
- Hypothesis: the remaining gather-side hotspot at `grug_moe.py:178` is the per-expert count computation itself, and TPU may lower a dense compare+sum better than `jnp.bincount` because `local_experts` is small.
- Command:
  - Added harness-only variant `onehot_counts`, which computes counts as:
    - dense compare of `local_expert` against `jnp.arange(local_experts)`
    - masked `int32` reduction over the assignment axis
  - Benchmarked on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - `random`:
    - `current`: `32.671 ms`
    - `onehot_counts`: `31.588 ms`
  - `runs`:
    - `current`: `32.633 ms`
    - `onehot_counts`: `31.574 ms`
- Interpretation:
  - This is a real win on the healthy `v5p-8` filter:
    - `random`: `~3.3%` faster
    - `runs`: `~3.2%` faster
  - The remaining useful local optimization was not a different return path; it was replacing the hot `bincount`.
- Next action: promote the `bincount` replacement into production `grug_moe.py` and re-test the canonical `current` kernel.

### 2026-03-07 05:15 - Production promotion of the `bincount` replacement
- Hypothesis: the harness `onehot_counts` win should carry directly into the production ring EP path because the only semantic change is the exact per-expert count lowering.
- Command:
  - Updated `lib/levanter/src/levanter/grug/grug_moe.py` so the EP ring path computes `counts` via dense compare+sum instead of `jnp.bincount`.
  - Re-synced `grug_moe.py` to healthy `ray-marin-us-central1-worker-671dced6-tpu`.
  - Re-ran the canonical production `current` kernel:
    - full shape
    - `EP=4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - Updated production `current`:
    - `random`: `31.571 ms`
    - `runs`: `31.631 ms`
  - Session-local baseline immediately before the production patch:
    - `random`: `32.671 ms`
    - `runs`: `32.633 ms`
- Interpretation:
  - The production kernel inherited the same improvement:
    - `random`: `~3.4%` faster
    - `runs`: `~3.1%` faster
  - Absolute times drifted relative to earlier overnight measurements on the same slice, so the reliable comparison is the local before/after in this session, not the cross-session absolute number.
  - This is another clean ring-path improvement: exact numerics, no extra collectives, simpler hotspot lowering.
- Next action: keep the production count-lowering change in `grug_moe.py` and refresh the EP guidance with at least a fallback post-change sweep.

### 2026-03-07 05:25 - Fresh `v5p-16` attempt still topology-broken
- Hypothesis: a newly allocated spot `v5p-16` in `us-central1-a` might finally provide a healthy target-hardware confirmation for the latest production change.
- Command:
  - Created queued resource `codex-grug-ep-0307c` in `us-central1-a`.
  - Bootstrapped node `codex-grug-ep-0307c-0` from scratch:
    - installed `uv`
    - cloned `https://github.com/marin-community/marin`
    - ran `uv sync --all-packages --extra=tpu`
  - Synced updated `grug_moe.py` and harness.
  - Tried a small multihost smoke (`EP=8`, small shape) on `codex-grug-ep-0307c-0`.
- Result:
  - The fresh queued-resource slice still fails at TPU init with:
    - `INTERNAL: Failed to get global TPU topology`
- Interpretation:
  - This was not a stale-image or bootstrap problem; even a fresh spot `v5p-16` can still come up topology-broken.
  - So I still do not have a clean target-hardware confirmation for the latest `bincount` replacement.
- Next action: stop spending more time on `v5p-16` allocation churn tonight and use the healthy `v5p-8` for any remaining fallback guidance.

### 2026-03-07 05:35 - Post-change fallback EP sweep on healthy `v5p-8`
- Hypothesis: even without a healthy `v5p-16`, a fresh post-change `EP=1,2,4` sweep on the healthy `v5p-8` is still useful to confirm the monotonic EP trend after the latest ring-path improvement.
- Command:
  - Re-ran the updated production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=1,2,4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - `random`:
    - `EP=1`: `39.773 ms`
    - `EP=2`: `38.310 ms`
    - `EP=4`: `31.577 ms`
  - `runs`:
    - `EP=1`: `39.767 ms`
    - `EP=2`: `38.297 ms`
    - `EP=4`: `31.546 ms`
- Interpretation:
  - The updated production ring path still scales in the expected direction with higher EP.
  - On the healthy `v5p-8`, `EP=4` remains decisively better than `EP=1/2`.
  - Combined with the earlier healthy `v5p-16` measurements, nothing here suggests changing the practical guidance:
    - on the target shape, higher EP remains the right move when hardware supports it.
- Next action:
  - Keep the new count-lowering change.
  - If a truly healthy `v5p-16` appears later, the first recheck to run is the updated `current` `EP=8` full-shape benchmark.

### 2026-03-07 05:50 - Post-`bincount` production profile
- Hypothesis: after replacing `bincount`, the ring-path bottleneck should shift even more decisively toward the local return scatter, with communication still minor.
- Command:
  - Captured a fresh full-shape `bench_pass=forward` trace for the updated production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
    - `distribution=random`
  - Ingested to `scratch/profiles/grug-current-ep4-after2-summary.json`
- Result:
  - Forward time:
    - `421.617 ms`
  - Top regions / ops:
    - `_forward=>moe_mlp=>scatter=>scatter-add`: `133.5k` exclusive (`45.8%` of profiled exclusive time)
    - `_forward=>moe_mlp=>gather=>_take=>gather`: `41.0k` exclusive (`14.1%`)
    - `_forward=>moe_mlp=>moe_up_down=>gmm`: `64.9k` exclusive (`22.3%`)
    - communication share: `0.96%`
    - compute share: `38.5%`
- Interpretation:
  - The `bincount` replacement did what it was supposed to do: communication remains tiny and the gather side is now clearly secondary.
  - The next local bottleneck is the flat token scatter-add into the full `[tokens_global, hidden]` buffer.
- Next action: filter only scatter-layout/lowering variants against this new baseline.

### 2026-03-07 06:00 - Return-path layout and lowering variants
- Hypothesis: the remaining return bottleneck might yield to either:
  - a more topology-aligned owner/local-token layout before `psum_scatter`, or
  - a lower-level scatter primitive that TPU lowers better than `.at[].add(...)`.
- Command:
  - Added harness-only variants:
    - `owner_local_scatter`:
      - accumulate into `[ep_size, tokens_per_shard, hidden]` by `(owner_shard, local_token)`
      - then `psum_scatter` over axis `0`
    - `padded_take`:
      - gather invalid padded rows from sentinel zero rows instead of explicit `where` zeroing
    - `lax_scatter`:
      - explicit `jax.lax.scatter_add` instead of `.at[token].add(...)`
  - Benchmarked each on healthy `ray-marin-us-central1-worker-671dced6-tpu`, full shape, `EP=4`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`.
- Result:
  - `owner_local_scatter`:
    - `random`: `31.755 ms` vs `current` `31.595 ms`
    - `runs`: `31.705 ms` vs `current` `31.597 ms`
  - `padded_take`:
    - `random`: `32.165 ms` vs `current` `31.666 ms`
    - `runs`: `32.157 ms` vs `current` `31.622 ms`
  - `lax_scatter`:
    - `random`: `31.613 ms` vs `current` `31.584 ms`
- Interpretation:
  - Owner/local layout does not improve the return path on this TPU stack.
  - Sentinel-padded takes are clearly worse than the explicit `where` zeroing they replace.
  - Explicit `jax.lax.scatter_add` is effectively identical to `.at[].add(...)`, so the scatter cost is not just a surface-API lowering artifact.
  - At this point the easy local scatter/gather variants are exhausted.
- Next action:
  - Keep the production `onehot` count lowering only.
  - If more EP TPU work is needed later, it should probably move up a level to a new dispatch/return architecture rather than more local lowering tweaks.

### 2026-03-07 06:15 - Default overflow-accounting removal
- Hypothesis: the production ring path still computes and cross-reduces capacity-overflow stats even when `report_capacity_overflow=False`, so skipping that bookkeeping on the default hot path might be a free TPU win.
- Command:
  - Patched `grug_moe.py` so the default path would not compute or `psum` dropped-assignment counts unless explicitly requested.
  - Measured the canonical production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - Baseline immediately before patch:
    - `random`: `31.629 ms`
    - `runs`: `31.640 ms`
  - Patched default path:
    - `random`: `31.619 ms`
    - `runs`: `31.649 ms`
- Interpretation:
  - This is noise, not a real win.
  - The extra bookkeeping is too small to matter in the current end-to-end TPU path.
- Next action:
  - Reverted the production patch.
  - Keep `grug_moe.py` limited to the proven `onehot` count lowering.

### 2026-03-07 06:25 - `forward_backward` profile after the production fixes
- Hypothesis: the next useful signal for total step time will come from a full `forward_backward` trace, not another forward-only profile.
- Command:
  - Captured a full-shape `bench_pass=forward_backward` trace for the updated production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
    - `distribution=random`
  - Ingested to `scratch/profiles/grug-current-ep4-fwbwd-summary.json`
- Result:
  - Top ops by exclusive time:
    - `jvp(moe_mlp)/shard_map/scatter/scatter-add`: `88.23k`
    - `transpose(jvp(moe_mlp))/shard_map/gather/_take/scatter-add`: `88.25k`
    - backward `gmm`/`tgmm` family: `~99.4k` combined across top kernels
  - Top regions:
    - `_loss_and_grads=>moe_mlp=>moe_up_down=>gmm`: `24.0%`
    - `_loss_and_grads=>moe_mlp=>scatter=>scatter-add`: `19.3%`
    - `_loss_and_grads=>moe_mlp=>gather=>_take=>scatter-add`: `19.2%`
  - communication share: `1.0%`
- Interpretation:
  - The total-step bottleneck is still local compute/materialization, not collectives.
  - The biggest non-GMM costs in `forward_backward` are now:
    - forward return scatter-add
    - backward transpose of the `x_take` gather (also a scatter-add)
  - This explains why forward-only experiments were no longer enough to guide the next step.
- Next action: filter harness variants that specifically target the backward gather-transpose scatter-add.

### 2026-03-07 06:35 - Custom-VJP take experiments for backward
- Hypothesis: the hot backward transpose of `x_take = x_global[token_local]` might lower better if we supply a custom VJP that reduces cotangents with `segment_sum` instead of relying on the default gather transpose.
- Command:
  - Added harness-only custom-VJP take helpers:
    - `take_segment_bwd`: backward uses unsorted `segment_sum`
    - `take_sorted_segment_bwd`: backward sorts by token then uses sorted `segment_sum`
  - Benchmarked on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=4`
    - `distribution=random`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - `current`: `31.613 ms`
  - `take_segment_bwd`: `31.635 ms`
  - `take_sorted_segment_bwd`: `32.091 ms`
- Interpretation:
  - Custom-VJP `segment_sum` backward is at best neutral.
  - Sorting for the backward reduction is clearly worse.
  - So the backward gather-transpose scatter-add is not easily improved by swapping in segment-style reductions from Python.
- Next action:
  - Keep these as rejected harness variants only.
  - Treat the remaining local bottlenecks as likely requiring lower-level kernel work or a different overall dispatch/return architecture, not another small Python-surface change.

### 2026-03-07 06:45 - Owner/local-token gather view
- Hypothesis: the hot backward transpose of `x_take = x_global[token_local]` might lower better if the forward gather uses an explicit `[owner_shard, local_token]` view of `x_global` instead of a flat global-token index.
- Command:
  - Added harness-only variant `owner_local_take`, which:
    - reshapes `x_global` to `[ep_size, tokens_per_shard, hidden]`
    - gathers dispatch activations as `x_by_owner[owner, local_token]`
    - keeps the rest of the ring path unchanged
  - Benchmarked on healthy `ray-marin-us-central1-worker-671dced6-tpu`:
    - full shape
    - `EP=4`
    - `distribution=random`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
- Result:
  - `current`: `31.627 ms`
  - `owner_local_take`: `31.642 ms`
- Interpretation:
  - Changing the gather view/transpose shape at the Python level is effectively neutral here.
  - This weakens the case for more reshaping/layout-only gather experiments in the harness.
- Next action:
  - Stop adding more Python-surface gather variants.
  - Treat the remaining step-time bottlenecks as likely requiring lower-level kernel work or a genuinely new dispatch/return architecture.

### 2026-03-07 07:00 - Routed-vs-shared cost split
- Hypothesis: even if the routed EP path is locally hard to optimize further, it is still useful to quantify how much of step time is actually the routed MoE path versus the shared dense branch.
- Command:
  - Benchmarked the canonical production `current` kernel on healthy `ray-marin-us-central1-worker-671dced6-tpu` with:
    - `shared_expert_dim=0`
    - `shared_expert_dim=2048`
  - Full shape:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8`
    - `EP=1,2,4`
    - `bench_pass=forward_backward`, `iters=3`, `warmup=1`
    - distributions: `random`, `runs`
- Result:
  - `shared_expert_dim=0`
    - `random`: `EP=1 36.591 ms`, `EP=2 36.318 ms`, `EP=4 29.799 ms`
    - `runs`: `EP=1 36.500 ms`, `EP=2 36.325 ms`, `EP=4 29.795 ms`
  - `shared_expert_dim=2048`
    - `random`: `EP=1 39.681 ms`, `EP=2 38.357 ms`, `EP=4 31.572 ms`
    - `runs`: `EP=1 39.786 ms`, `EP=2 38.266 ms`, `EP=4 31.655 ms`
- Interpretation:
  - At `EP=4`, the shared branch contributes about `1.8 ms` on this workload.
  - At `EP=1`, the shared branch contributes about `3.2 ms`, so the routed path is still the dominant EP-sensitive lever.
  - This means further ring-path work is still potentially worthwhile, but the absolute remaining upside is no longer huge.
- Next action: capture a routed-only `forward_backward` profile to see whether the same local routed hotspots remain after removing the shared branch.

### 2026-03-07 07:10 - Routed-only `forward_backward` profile
- Hypothesis: if the shared dense branch is not masking the routed EP bottlenecks, the routed-only `forward_backward` profile should still be dominated by the same local scatter/gather-transpose kernels.
- Command:
  - Captured a routed-only full-shape `bench_pass=forward_backward` trace with:
    - `shared_expert_dim=0`
    - `EP=4`
    - `distribution=random`
  - Ingested to `scratch/profiles/grug-current-ep4-fwbwd-noshared-summary.json`
  - Compared against the earlier full `shared_expert_dim=2048` `forward_backward` profile.
- Result:
  - Routed-only `forward_backward` time:
    - `679.729 ms`
  - Routed-only top regions remain effectively the same:
    - `_loss_and_grads=>moe_mlp=>moe_up_down=>gmm`: `25.5%`
    - `_loss_and_grads=>moe_mlp=>scatter=>scatter-add`: `20.5%`
    - `_loss_and_grads=>moe_mlp=>gather=>_take=>scatter-add`: `20.3%`
  - Top ops are still:
    - forward scatter-add in the ring return path
    - backward transpose scatter-add of the `x_take` gather
    - backward `gmm` / `tgmm`
- Interpretation:
  - Removing the shared branch does not change the identity of the routed-path bottlenecks.
  - So the earlier conclusion is robust:
    - the remaining EP TPU costs are still local routed kernels,
    - but they are no longer yielding to Python-surface gather/scatter rewrites.
- Next action:
  - Keep the current production ring path as-is.
  - If more EP TPU work is pursued later, prioritize lower-level kernel work for these routed scatter/gather-transpose hotspots or a different dispatch/return architecture.

### 2026-03-07 07:30 - Sealed conclusion and recommended next steps
- Hypothesis: after the local-compaction, metadata, and count-lowering fixes, the remaining TPU EP bottlenecks will no longer yield to Python-surface changes in the current ring path.
- Command:
  - Consolidated the final benchmark and profiling results from this branch.
  - Compared the surviving production deltas against the rejected harness variants.
- Result:
  - Surviving production wins in `lib/levanter/src/levanter/grug/grug_moe.py`:
    - local-first compaction instead of global `argsort` + full-length `take`s
    - reconstruct cheap metadata instead of gathering it
    - dense compare+sum local count reduction instead of `jnp.bincount`
  - Best validated production numbers:
    - healthy `v5p-16`, target shape, `forward_backward`:
      - `EP=1`: `24.289 ms`
      - `EP=2`: `23.440 ms`
      - `EP=4`: `20.124 ms`
      - `EP=8`: `17.847 ms`
      - current vs legacy speedup: `1.00x`, `1.08x`, `1.14x`, `1.47x`
    - healthy `v5p-8`, full shape, `EP=4`, `forward_backward`:
      - session-local count-lowering win:
        - `random`: `32.671 -> 31.571 ms`
        - `runs`: `32.633 -> 31.631 ms`
  - Rejected directions:
    - MaxText-style `ragged_a2a`
    - streamed ring / packed return / cumsum variants
    - scatter API swaps and owner/local reshapes
    - custom-VJP gather backward rewrites
    - metadata width tweaks and capacity-factor tuning
  - Final profile read:
    - communication share is small on the healthy traces
    - dominant remaining non-GMM costs are local `scatter-add` kernels in:
      - forward ring return
      - backward transpose of the gather path
- Interpretation:
  - The current ring EP path in `grug_moe.py` is the fastest thing found for this workload.
  - For this shape family, the practical recommendation remains:
    - prefer `EP=8` on healthy `v5p-16`
    - use `EP=4` as the fallback
  - Further TPU EP gains are unlikely to come from more Python-surface rewrites of the current scatter/gather path.
- Next action:
  - Seal this branch snapshot.
  - If work resumes, prioritize either lower-level kernel/codegen work around the `scatter-add` hotspots or a materially different dispatch/return architecture.

### 2026-03-07 11:10 - MaxText v5p flag shortlist for Grug MoE
- Hypothesis: MaxText's TPU/XLA flag bundles for v5p MoE workloads may transfer to the Grug MoE harness on `v5p-8`, especially the async all-gather and reduce-scatter layout knobs.
- Command:
  - Inspected MaxText at commit `44039d8248696afc8fc59fd39f29055e41682c57`:
    - `benchmarks/xla_flags_library.py`
    - `benchmarks/maxtext_v5p_model_configs.py`
  - Pulled the MoE-relevant v5p flag groups into a local shortlist:
    - `MOE_VMEM_LIMIT_FLAG` (`--xla_tpu_scoped_vmem_limit_kib=81920`)
    - `CF_FOR_ALL_GATHER`
    - `DATA_PARALLEL_OVERLAP`
    - `LAYOUT_FOR_ALL_REDUCE_SCATTER`
- Result:
  - The MaxText MoE v5p recipe for `deepseek_v3_ep_256_v5p_512` is:
    - `MOE_VMEM_LIMIT_FLAG + CF_FOR_ALL_GATHER + DATA_PARALLEL_OVERLAP`
  - Larger dropless v5p configs also add:
    - `LAYOUT_FOR_ALL_REDUCE_SCATTER`
- Interpretation:
  - These are the only MaxText flag families that looked plausibly relevant to the current Grug MoE ring path on `v5p-8`.
  - Host-offload and SparseCore-offload flags do not match this harness.
- Next action: benchmark these flags on the current production Grug MoE path on a healthy `v5p-8`.

### 2026-03-07 11:20 - `v5p-8` EP=4 MaxText flag sweep (shared path)
- Hypothesis: if MaxText's TPU/XLA flags help the current Grug MoE ring path, the most likely wins will come from async all-gather overlap or reduce-scatter layout tuning on the target shared-expert workload.
- Command:
  - Allocated healthy `v5p-8` dev TPU `codex-maxtext-flags-v5p8` on `infra/marin-us-central1.yaml`.
  - Ran the canonical production `current` kernel with:
    - `tokens=32768 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `EP=4`
    - `bench_pass=forward_backward`
    - `distribution=random`
    - `iters=3`, `warmup=1`
  - Swept:
    - baseline: `--xla_tpu_scoped_vmem_limit_kib=50000`
    - `vmem81920`
    - `cf_ag`
    - `dp_overlap`
    - `maxtext_moe`
    - `maxtext_moe_layout`
    - repeated baseline
- Result:
  - `baseline`: `29.268 ms`
  - `vmem81920`: `29.948 ms`
  - `cf_ag`: `29.208 ms`
  - `dp_overlap`: `29.218 ms`
  - `maxtext_moe`: `29.976 ms`
  - `maxtext_moe_layout`: `29.705 ms`
  - `baseline_repeat`: `29.264 ms`
  - Follow-up checks:
    - shared `runs`, baseline vs `cf_ag`: `29.289 ms` vs `29.235 ms`
    - routed-only (`shared_expert_dim=0`) `random`, baseline vs `cf_ag`: `27.760 ms` vs `27.747 ms`
- Interpretation:
  - On `v5p-8` at `EP=4`, the MaxText MoE bundle regresses by about `2.4%`.
  - The `81920` vmem limit is the clearest losing component.
  - `CF_FOR_ALL_GATHER` is at most noise-level on both shared and routed-only `EP=4`.
  - `DATA_PARALLEL_OVERLAP` is not meaningful at `EP=4` on `v5p-8` because `DP=1` for this harness split, so its near-neutral result is expected.
- Next action: re-run the relevant flags at `EP=2`, where `DP=2` and the MaxText data-parallel overlap flags can actually matter.

### 2026-03-07 11:45 - `v5p-8` EP=2 follow-up and reduce-scatter layout confirmation
- Hypothesis: the only MaxText flag family likely to survive on this harness is the reduce-scatter layout tuning, and any `DATA_PARALLEL_OVERLAP` benefit should show up only once `DP>1` (`EP=2` on `v5p-8`).
- Command:
  - Re-ran the same shared workload on healthy `codex-maxtext-flags-v5p8` with:
    - `EP=2`
    - `distribution=random`
    - `iters=3`, `warmup=1`
  - Swept:
    - baseline
    - `cf_ag`
    - `dp_overlap`
    - `maxtext_moe`
    - `maxtext_moe_layout`
    - repeated baseline
  - Ran a follow-up attribution sweep with:
    - `layout_only`
    - `cf_dp_layout`
  - Confirmed the best candidate with a higher-iteration paired check:
    - baseline/layout repeated with `iters=5`, `warmup=2`
- Result:
  - `EP=2` main sweep:
    - `baseline`: `36.752 ms`
    - `cf_ag`: `36.730 ms`
    - `dp_overlap`: `36.721 ms`
    - `maxtext_moe`: `36.581 ms`
    - `maxtext_moe_layout`: `36.395 ms`
    - `baseline_repeat`: `36.764 ms`
  - `EP=2` attribution sweep:
    - `layout_only`: `36.524 ms`
    - `cf_dp_layout`: `36.551 ms`
  - Higher-iteration confirmation:
    - `baseline_a`: `36.672 ms`
    - `layout_a`: `36.405 ms`
    - `baseline_b`: `36.714 ms`
    - `layout_b`: `36.495 ms`
- Interpretation:
  - `CF_FOR_ALL_GATHER` and `DATA_PARALLEL_OVERLAP` are individually noise-level on this workload.
  - The only reproducible positive result is `LAYOUT_FOR_ALL_REDUCE_SCATTER`, which improves `EP=2` by about `0.6-0.7%`.
  - Avoiding the MaxText `81920` vmem bump matters; it still drags the bundle down at `EP=4` and is not needed for the small `EP=2` layout win.
  - This is a real but small effect, well below the earlier algorithmic wins in the ring path itself.
- Next action:
  - Keep the current production recommendation unchanged:
    - no MaxText-derived flag change for the `v5p-8` `EP=4` fast path
  - If we want a TPU-runtime tweak later, the only candidate worth considering is an opt-in `LAYOUT_FOR_ALL_REDUCE_SCATTER` flag for `EP=2`; validate it on a fuller train-step path before promoting it.

### 2026-03-07 20:35 - `v5p-64` MaxText flag sweep and direct-create runtime fix
- Hypothesis: on the full `v5p-64` shape, MaxText's MoE TPU flag bundle might help more than it did on `v5p-8`, especially because `EP=8` still leaves `DP=4`, so data-parallel overlap has room to matter.
- Command:
  - Tried three direct-created spot `v5p-64` slices:
    - `codex-grug-flags-v5p64` in `us-central1-a`
    - `codex-grug-flags-v5p64b` in `us-central1-a`
    - `codex-grug-flags-v5p64e` in `us-east5-a`
  - All three used `--version=tpu-ubuntu2204-base` and failed the same way:
    - local `uv run python -c "import jax; print(jax.devices())"` on worker `0` errored with `INTERNAL: Failed to get global TPU topology`
    - `tpu-runtime.service` logs showed the wrong container image:
      - `gcr.io/cloud-tpu-v2-images/fake_tensorflow:latest`
  - Checked the current Google TPU JAX guidance for v5p and recreated the slice with the correct runtime:
    - `codex-grug-flags-v5p64j` in `us-east5-a`
    - `--version=v2-alpha-tpuv5`
  - On `v2-alpha-tpuv5`, the clean launch pattern was:
    - run the harness on every worker with `gcloud compute tpus tpu-vm ssh ... --worker=all`
    - do **not** pass explicit `jax.distributed.initialize(...)` arguments
    - let JAX auto-detect the slice across hosts
  - Swept the canonical production `current` kernel with:
    - `tokens=131072 hidden=2048 mlp_dim=768 experts=128 topk=8 shared_expert_dim=2048`
    - `bench_pass=forward_backward`
    - `distribution=random`
    - `iters=3`, `warmup=1`
  - Swept `EP=8`:
    - baseline: `--xla_tpu_scoped_vmem_limit_kib=50000`
    - `cf_ag`
    - `dp_overlap`
    - `layout_only`
    - `maxtext_moe`
    - `maxtext_moe_layout`
    - repeated baseline
  - Swept `EP=2`:
    - baseline
    - `layout_only`
    - `maxtext_moe_layout`
    - repeated baseline
- Result:
  - `EP=8`
    - `baseline`: `21.921 ms`
    - `cf_ag`: `21.978 ms`
    - `dp_overlap`: `21.845 ms`
    - `layout_only`: `21.894 ms`
    - `maxtext_moe`: `22.857 ms`
    - `maxtext_moe_layout`: `22.792 ms`
    - `baseline_repeat`: `21.894 ms`
  - `EP=2`
    - `baseline`: `24.076 ms`
    - `layout_only`: `24.035 ms`
    - `maxtext_moe_layout`: `24.349 ms`
    - `baseline_repeat`: `24.152 ms`
- Interpretation:
  - For direct-created `v5p-64` JAX runs, `--version=tpu-ubuntu2204-base` is a trap on this date; it booted the `fake_tensorflow` TPU runtime and never exposed a usable global topology.
  - `--version=v2-alpha-tpuv5` fixed that immediately and is the right direct-create runtime for these experiments.
  - On the actual `v5p-64` workload, the MaxText MoE flag story stayed weak:
    - `EP=8`: `cf_ag`, `dp_overlap`, and `layout_only` all stayed inside baseline noise
    - `EP=8`: the full MaxText MoE bundle regressed by about `4.3%`
    - `EP=2`: `layout_only` was again a tiny positive (`~0.3%`), but smaller than the `v5p-8` effect
    - `EP=2`: adding the full MaxText bundle still regressed (`~1.0%`)
  - The only remotely positive signal on `v5p-64` is the same one seen on `v5p-8`: `LAYOUT_FOR_ALL_REDUCE_SCATTER`, and even there the effect is very small.
- Next action:
  - Keep the production recommendation unchanged for `v5p-64`: no MaxText-derived TPU flag change for the current fast path.
  - If we ever need direct-created `v5p-64` TPU benchmarking again, use `--version=v2-alpha-tpuv5` from the start.
