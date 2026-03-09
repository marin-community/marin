# MoE SparseCore Shared-Expert Overlap: Research Logbook

## Scope
- Goal: Test whether a shared dense expert branch gives XLA enough independent TC work to overlap a SparseCore-backed routed dispatch path in Grug MoE.
- Primary metric(s): tokens/s, step time, and direct scheduling evidence of SC/TC overlap in TPU trace/LLO.
- Constraints: benchmark-only work on public `v5p-8` stack; use Grug/Qwen-style shapes close to the target production regime.
- GitHub issue: `#3436` - <https://github.com/marin-community/marin/issues/3436>

## Baseline
- Date: 2026-03-09
- Code refs:
  - `experiments/grug/moe/model.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py`
- Baseline numbers:
  - Prior sealed result from `#3418`: SparseCore dispatch/collect overlap did not beat XLA on `v5p-8`.
  - Open question here: shared expert creates an actually independent dense branch, unlike the earlier routed-only chunking attempts.

## Experiment Log
### 2026-03-09 14:10 - Kickoff
- Hypothesis: If routed MoE dispatch runs on SparseCore while the shared expert runs as a dense TC MLP on the same `mlp_in`, XLA may schedule enough real SC/TC overlap to hide part of routed dispatch cost.
- Command:
  - Context gathering only.
- Config:
  - Target shape family:
    - `hidden_dim=2048`
    - `intermediate_dim=1536`
    - `num_experts=128`
    - `topk=4`
    - `max_seq_len=4096`
    - `capacity_factor=1.25`
    - `shared_expert_intermediate_dim=2048`
  - Also sweep wider shared expert sizes if baseline shared branch is too small to create overlap.
- Result:
  - `experiments/grug/moe/model.py` confirms the exact structure needed:
    - `mlp_in = self.rms_mlp(x)`
    - `mlp_out, router_stats = self.mlp(mlp_in)`
    - `mlp_out = mlp_out + self.shared(mlp_in, ...)` when shared expert is enabled.
- Interpretation:
  - This is a materially better overlap test than earlier routed-only pipelines because the shared branch is independent of routed dispatch until the final add.
- Next action:
  - Create experiment issue.
  - Add a benchmark harness for `routed_branch(mlp_in) + shared_branch(mlp_in)` with `xla` vs `sparsecore` dispatch implementations and configurable shared width.

### 2026-03-09 15:05 - Added benchmark harness
- Hypothesis:
  - A benchmark that mirrors `routed_moe(mlp_in) + shared_dense_mlp(mlp_in)` is the minimal test needed for the shared-expert overlap idea.
- Command:
  - Local smoke only.
- Config:
  - Added `lib/levanter/scripts/bench/bench_grug_moe_sc_shared_overlap.py`.
  - Preset uses:
    - `hidden=2048`
    - `intermediate=1536`
    - `experts=128`
    - `topk=4`
    - `seq=4096`
    - `capacity_factor=1.25`
    - target local tokens/chip `= 32768`
- Result:
  - Harness compiles and runs locally in both forward-only and train modes.
- Interpretation:
  - Ready for TPU sweeps and profile capture.
- Next action:
  - Run `v5p-8` forward-only sweeps over shared width.

### 2026-03-09 15:32 - Forward-only shared-width sweep on `v5p-8`
- Hypothesis:
  - If shared expert creates a real overlap window, wider shared widths should materially narrow the SC/XLA gap.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scshared-1530 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000' -- \
  uv run --package levanter --extra tpu python -u \
  lib/levanter/scripts/bench/bench_grug_moe_sc_shared_overlap.py \
  --preset qwen-shared-overlap \
  --shared-widths 0 2048 4096 8192 12288 24576 \
  --warmup 1 \
  --iters 2 \
  --implementation xla sparsecore \
  --forward-only
```
- Config:
  - TPU: `v5p-8`
  - JAX mesh: `{'data': 4, 'model': 1}`
  - Global batch: `32`
  - Local tokens/chip: `32768`
- Result:
  - Forward-only throughput:

| shared width | xla tok/s | sparsecore tok/s | SC/XLA |
| --- | ---: | ---: | ---: |
| 0 | 5.287M | 4.252M | 0.804x |
| 2048 | 4.890M | 4.005M | 0.819x |
| 4096 | 4.559M | 3.768M | 0.826x |
| 8192 | 4.034M | 3.412M | 0.846x |
| 12288 | 3.607M | 3.111M | 0.863x |
| 24576 | 2.746M | 2.444M | 0.890x |
- Interpretation:
  - Wider shared expert does help modestly: the SC/XLA ratio improves monotonically as shared width grows.
  - But even an extremely wide shared expert does not flip the result on `v5p-8`.
- Next action:
  - Profile the widest viable case and inspect whether the gain is coming from real SC/TC overlap.

### 2026-03-09 16:00 - Trace comparison for shared width `12288`
- Hypothesis:
  - If shared expert rescues the SC path, the trace should show `shared_dense_mlp` starting before SparseCore dispatch/permute finishes.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scshared-1530 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000' -- \
  uv run --package levanter --extra tpu python -u \
  lib/levanter/scripts/bench/bench_grug_moe_sc_shared_overlap.py \
  --preset qwen-shared-overlap \
  --shared-widths 12288 \
  --warmup 1 \
  --iters 2 \
  --implementation sparsecore \
  --forward-only \
  --profile-implementation sparsecore \
  --profile-shared-width 12288 \
  --profile-dir /home/dlwh/marin/scratch/sc_shared_sparse_profile_12288_light

RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scshared-1530 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000' -- \
  uv run --package levanter --extra tpu python -u \
  lib/levanter/scripts/bench/bench_grug_moe_sc_shared_overlap.py \
  --preset qwen-shared-overlap \
  --shared-widths 12288 \
  --warmup 1 \
  --iters 2 \
  --implementation xla \
  --forward-only \
  --profile-implementation xla \
  --profile-shared-width 12288 \
  --profile-dir /home/dlwh/marin/scratch/sc_shared_xla_profile_12288_light
```
- Config:
  - Local copied summaries:
    - `scratch/sc_shared_sparse_profile_12288_light_summary.json`
    - `scratch/sc_shared_xla_profile_12288_light_summary.json`
- Result:
  - SparseCore trace:
    - run 20 `SparseCore` `jit_forward_fn` on pid 4 ends at `146011.254 us`
    - first `shared_dense_mlp` op starts at `146149.423 us`
    - delta: `+138.169 us`
  - XLA trace:
    - run 20 last `_prepare_moe_dispatch` event ends at `164692.144 us`
    - first `shared_dense_mlp` op starts at `136397.405 us`
    - delta: `-28294.739 us`
  - In other words:
    - plain XLA starts the shared branch far before routed dispatch prep completes
    - SparseCore starts the shared branch only after the SC stage finishes
  - Region summaries also match that story:
    - SparseCore shared profile:
      - `forward_fn=>routed_moe=>moe_mlp`: `475.469 ms` inclusive
      - `forward_fn=>shared_dense_mlp=>_shared_dense_mlp=>dot_general`: `181.985 ms`
    - XLA shared profile:
      - `forward_fn=>routed_moe=>moe_mlp`: `391.677 ms` inclusive
      - `forward_fn=>shared_dense_mlp=>_shared_dense_mlp=>dot_general`: `182.132 ms`
- Interpretation:
  - Shared expert does **not** create the hoped-for SC/TC overlap window on this public `v5p-8` stack.
  - The widening shared expert helps throughput only because the dense shared branch becomes a larger fraction of total work, not because XLA overlaps the SC permute stage with that branch.
  - Plain XLA is already doing the overlap-friendly thing with its own routed-prep path, while the SparseCore path is not.
- Next action:
  - Check whether the same pattern survives forward+backward.

### 2026-03-09 16:22 - Train-mode shared-width sweep on `v5p-8`
- Hypothesis:
  - Backward might change the balance enough for shared expert to help SparseCore more than it helps XLA.
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scshared-1530 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000' -- \
  uv run --package levanter --extra tpu python -u \
  lib/levanter/scripts/bench/bench_grug_moe_sc_shared_overlap.py \
  --preset qwen-shared-overlap \
  --shared-widths 0 2048 12288 \
  --warmup 1 \
  --iters 2 \
  --implementation xla sparsecore
```
- Result:
  - Train throughput:

| shared width | xla tok/s | sparsecore tok/s | SC/XLA |
| --- | ---: | ---: | ---: |
| 0 | 1.605M | 1.490M | 0.928x |
| 2048 | 1.472M | 1.390M | 0.944x |
| 12288 | 1.106M | 1.050M | 0.949x |
- Interpretation:
  - The same pattern survives forward+backward: wider shared expert narrows the gap a bit, but it does not produce a win.
  - This reinforces the trace conclusion that the shared expert is not unlocking the desired SC overlap; it is only diluting the routed dispatch cost.
- Next action:
  - Write up the current conclusion on `#3436` and stop unless a materially different overlap formulation appears.

## Conclusion
- Outcome:
  - Shared expert is not enough to rescue the public-stack SparseCore dispatch path on `v5p-8`.
- What worked:
  - Wider shared expert consistently narrows the SC/XLA gap a bit in both forward-only and train-mode runs.
- What did not work:
  - The shared branch did not create useful SC/TC overlap for the SparseCore path.
  - Plain XLA already overlaps the shared dense branch with its own routed dispatch-prep path more effectively than the SparseCore formulation.
- Confidence:
  - replicated
- Limitations:
  - This conclusion is for the public `v5p-8` stack and the tested benchmark formulation, not a statement about possible lower-level/internal TPU kernels.
- Next steps:
  1. Do not spend more time on public-stack SparseCore/shared-expert overlap variants for this kernel on `v5p-8`.
  2. Optimize the TPU experiment regime at the model/system level instead: smaller scientifically acceptable `topk`, fewer wider experts, tight capacity factor, and careful EP choices.
  3. Revisit only with materially new help: deeper TPU kernel/compiler expertise or a genuinely different dispatch kernel idea.

## Seal
- Date: 2026-03-09
- Issue: `#3436` - <https://github.com/marin-community/marin/issues/3436>
- Branch: `codex/moe-sc-shared-expert-overlap`
