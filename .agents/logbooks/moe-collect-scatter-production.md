# MoE Collect/Scatter Production: Research Logbook

- GitHub experiment issue: [#3420](https://github.com/marin-community/marin/issues/3420)
- Prior sealed production thread: [#2710](https://github.com/marin-community/marin/issues/2710)
- Related negative-result thread: [#3418](https://github.com/marin-community/marin/issues/3418)

## Scope
- Goal: understand the remaining production bottlenecks in the ring-EP Grug MoE path after the sealed `#2710` wins, with emphasis on the forward ring-return `scatter-add` and the backward gather-transpose `scatter-add`.
- Primary metric(s): inclusive and exclusive time under `MoEMLP=>moe_mlp` for the real production profile; follow-on kernel/runtime changes will use end-to-end `forward_backward` step time.
- Constraints:
  - start from the current ring-EP production path in [`grug_moe.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - do not reopen the SparseCore dispatch path on `v5p-8`
  - do not spend time on Python-surface scatter/gather API swaps unless the profile evidence says the underlying codegen changed materially
- Initial hypotheses:
  - the remaining `scatter-add` hotspots are a combination of indexed-write cost and large layout-conversion baggage
  - communication is probably secondary on the healthy traces, matching `#2710`
  - the most plausible next production experiments will be layout/decomposition changes around collect/writeback rather than expert-math changes
- Stop criteria:
  - enough profile evidence to rank a small set of concrete production experiments, or
  - enough negative evidence to conclude the remaining bottleneck is mostly lower-level codegen not worth chasing at the Python/JAX surface

## Baseline
- Date: 2026-03-08
- Code refs:
  - [`grug_moe.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - sealed reference thread: [#2710](https://github.com/marin-community/marin/issues/2710)
  - sealed ring-EP tag: [grug-moe-ep-ring-20260307](https://github.com/marin-community/marin/tree/grug-moe-ep-ring-20260307)
- Fixed baseline profile:
  - W&B run: `marin-community/marin/grug-moe-qwen3-32b-a4b-v5p64-bs320-ep4-cf1p0-topk4-matched-active-pf32-buf64-synthetic-profile-iris-main-r1`
  - local summary: [`scratch/grug_moe_qwen3_ep4_profile_summary.json`](/Users/dlwh/.codex/worktrees/4f0b/marin/scratch/grug_moe_qwen3_ep4_profile_summary.json)
- Baseline facts imported from prior thread:
  - inside `MoEMLP=>moe_mlp`, local layout-transform plus EP collectives were about half of exclusive time
  - the hottest leaves included `gather=>_take=>scatter-add`, `scatter=>scatter-add`, `gather=>_take=>gather`, and `gather=>all_gather`
  - `#2710` sealed conclusion: remaining non-GMM bottlenecks are forward ring-return `scatter-add` and backward gather-transpose `scatter-add`

## Experiment Log
### 2026-03-08 10:xx - Kickoff after reading #2710
- Hypothesis:
  - the next production work should start from the sealed ring-EP profile bottlenecks, not from benchmark-only dispatch ideas
- Command:
  - `gh issue view -R marin-community/marin 2710 --json number,title,body,url,labels,comments`
  - `gh issue view -R marin-community/marin 2710 --comments`
- Config:
  - reference issue `#2710`
  - follow-up issue `#3420`
- Result:
  - `#2710` explicitly seals the production baseline and names the remaining hotspots:
    - forward ring return `scatter-add`
    - backward gather-transpose `scatter-add`
  - it also explicitly advises against spending more time on Python-surface scatter/gather rewrites in the current ring path
- Interpretation:
  - this follow-up needs to be a production profile decomposition thread, not another benchmark-only kernel exploration
- Next action:
  - query the real EP4 profile summary around the two `scatter-add` hotspots and separate local layout-transform cost from collective cost and grouped-GMM cost

### 2026-03-08 10:xx - Real EP4 profile decomposition confirms local scatter-add is bigger than comm
- Hypothesis:
  - on the sealed production baseline, the next largest non-GMM cost is local indexed movement, not EP communication
- Command:
```bash
python - <<'PY'
import json
from pathlib import Path
regions = json.loads(Path("scratch/grug_moe_qwen3_ep4_profile_summary.json").read_text())["hierarchical_regions"]
paths = [
    "train_step=>Transformer=>Block=>MoEMLP=>moe_mlp=>scatter=>scatter-add",
    "train_step=>Transformer=>Transformer=>Block=>MoEMLP=>moe_mlp=>gather=>_take=>scatter-add",
    "train_step=>Transformer=>Block=>MoEMLP=>moe_mlp=>gather=>all_gather",
    "train_step=>Transformer=>Transformer=>Block=>MoEMLP=>moe_mlp=>scatter=>all_gather",
    "train_step=>Transformer=>Transformer=>Block=>MoEMLP=>moe_mlp=>gather=>reduce_scatter",
    "train_step=>Transformer=>Block=>MoEMLP=>moe_mlp=>gather=>_take=>gather",
    "train_step=>Transformer=>Transformer=>rematted_computation=>Block=>MoEMLP=>moe_mlp=>gather=>_take=>gather",
    "train_step=>Transformer=>Transformer=>Block=>MoEMLP=>moe_mlp=>scatter=>gather",
]
lookup = {r["path"]: r for r in regions}
for p in paths:
    r = lookup[p]
    print(f"{p}\n  inc={r['inclusive_duration']:.1f} exc={r['exclusive_duration']:.1f} count={r['count']}")
PY
```
- Config:
  - profile summary: [`scratch/grug_moe_qwen3_ep4_profile_summary.json`](/Users/dlwh/.codex/worktrees/4f0b/marin/scratch/grug_moe_qwen3_ep4_profile_summary.json)
  - production code path:
    - gather/take at [`grug_moe.py:579`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L579)
    - forward return scatter-add at [`grug_moe.py:594`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L594)
    - `psum_scatter` collect at [`grug_moe.py:597`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L597)
- Result:
  - forward ring-return `scatter=>scatter-add`: `7.922M` exclusive
  - backward gather-transpose `gather=>_take=>scatter-add`: `6.721M` exclusive
  - forward gather collective `gather=>all_gather`: `3.708M` exclusive
  - backward gather collective `gather=>all_gather` (remat): `2.282M` exclusive
  - `gather=>reduce_scatter`: `0.330M` exclusive
  - backward/transpose-visible local gathers:
    - `gather=>_take=>gather`: `2.565M` exclusive
    - remat `gather=>_take=>gather`: `3.480M` exclusive
    - `scatter=>gather`: `1.457M` exclusive
- Interpretation:
  - the two local `scatter-add` hotspots alone sum to about `14.64M` exclusive, which is materially larger than the visible collectives in the same MoE region
  - this supports the sealed `#2710` conclusion that the next production target is local collect/writeback, not comm
  - it also means the follow-up should inspect the lowering/codegen around the indexed add and its transpose, not just high-level collective structure
- Next action:
  - inspect the production lowering around the forward return `scatter-add` and the backward transpose path to see whether the dominant cost is the scatter itself or reshape/gather/layout baggage around it

### 2026-03-08 11:xx - HLO/codegen confirms the hot paths are real scatter kernels plus fused helpers
- Hypothesis:
  - the forward return and backward transpose paths may already be in their canonical gather/scatter lowering, which would make more Python-surface rewrites unlikely to help
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute --no-sync \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 \
  -e XLA_FLAGS='--xla_dump_to=/tmp/moe_collect_prod_xla_dump2 --xla_dump_hlo_as_text' -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --preset qwen3-32b-ep4-profile \
    --implementation xla \
    --warmup 1 --iters 1
```
- Config:
  - dump module: `/tmp/moe_collect_prod_xla_dump2/module_0095.jit_loss_fn.cl_813921542.after_optimizations.txt`
- Result:
  - forward return from [`grug_moe.py:601`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L601) lowers to:
    - `reshape(idx)` -> `transpose(idx)`
    - `reshape(weighted_dispatch)` -> `transpose(weighted_dispatch)`
    - `scatter` with `indices_are_sorted=true`
    - helper `gather` and fused kernels around the same path
  - backward transpose of the take from [`grug_moe.py:197`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L197) lowers similarly to:
    - a `scatter`
    - a paired `gather`
    - large custom fusions around that transpose path
  - after-codegen for the forward return path still contains an internal sort/fusion sequence under the same metadata:
    - `sort.1`
    - `broadcast_clamp_fusion`
    - large custom fusion with ~45 MB scoped-memory usage
- Interpretation:
  - the profile hotspot is not a fake attribution to tiny wrapper ops; it really is the return scatter lowering and its transpose
  - the compiler is already in a sorted-index scatter formulation, so another surface-level API swap is unlikely to be enough
  - this pushes the next experiments toward materially different return architectures rather than more local spelling changes
- Next action:
  - test whether changing the return architecture, not just the scatter shape, can move the bottleneck

### 2026-03-08 11:xx - Owner-bucket return shape is a slight regression; all-to-all variant is in progress but blocked by TPU SSH timeouts
- Hypothesis:
  - if the flat global-token scatter shape is part of the problem, accumulating by owner shard first might help even before changing the communication pattern
- Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --preset qwen3-32b-ep4-profile \
    --implementation xla \
    --warmup 1 --iters 2

RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name dlwh-scdispatch-134559 \
  execute \
  -e LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 \
  -e GRUG_MOE_EP_RETURN_IMPL=owner_bucket_psum_scatter -- \
  uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
    --preset qwen3-32b-ep4-profile \
    --implementation xla \
    --warmup 1 --iters 2
```
- Config:
  - current default return path at [`grug_moe.py:600`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L600)
  - owner-bucketed `psum_scatter` variant at [`grug_moe.py:605`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L605)
- Result:
  - default `scatter_psum_scatter`: `2.082M-2.090M tok/s`
  - `owner_bucket_psum_scatter`: `2.059M tok/s`
  - regression: about `-1.1%`
  - also implemented a more radical benchmark-only `owner_bucket_all_to_all_local_scatter` path at [`grug_moe.py:613`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L613), but repeated retries to benchmark it were blocked by transient SSH timeouts to the dev TPU alias before the run could complete
- Interpretation:
  - changing only the scatter shape is not enough
  - the next plausible experiment remains the architecture-changing `all_to_all + local scatter` return, because it removes the full global-token return scatter instead of just reshaping it
- Next action:
  - rerun the `owner_bucket_all_to_all_local_scatter` variant once the dev TPU alias is stable again

### 2026-03-08 17:xx - Pre-sorting the return indices is staged but currently blocked by TPU alias instability
- Hypothesis:
  - if the compiler is redundantly sorting inside the forward return scatter path, explicitly pre-sorting `token_local` might change the lowering or shave off some overhead
- Command:
  - Implemented `GRUG_MOE_EP_RETURN_IMPL=sorted_scatter_psum_scatter` in [`grug_moe.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py) and validated it locally with:
    - `uv run --package levanter --group test python -m pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`
    - `./infra/pre-commit.py lib/levanter/src/levanter/grug/grug_moe.py`
  - Attempted matched TPU benchmark reruns on `dlwh-scdispatch-134559`.
- Result:
  - No benchmark number yet.
  - Repeated attempts to reach the dev TPU alias timed out even for plain `hostname` / `execute --no-sync` probes, so the TPU-side benchmark could not be completed reliably.
- Interpretation:
  - this is an infra blocker, not a kernel result
- Next action:
  - rerun `sorted_scatter_psum_scatter` once the dev TPU alias is reachable again

### 2026-03-08 21:xx - Fresh TPU rerun closes out the remaining return-path ideas as regressions
- Hypothesis:
  - two remaining benchmark-only return-path experiments were still worth checking on the matched EP4 production geometry:
    - pre-sort `token_local` before the existing `scatter -> psum_scatter` path
    - replace the full global-token return scatter with `owner_bucket_all_to_all_local_scatter`
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
     --preset qwen3-32b-ep4-profile --implementation xla --warmup 1 --iters 2 \
     > /tmp/moe_return_baseline.out 2>&1'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 \
          GRUG_MOE_EP_RETURN_IMPL=sorted_scatter_psum_scatter && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
     --preset qwen3-32b-ep4-profile --implementation xla --warmup 1 --iters 2 \
     > /tmp/moe_return_sorted.out 2>&1'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 \
          GRUG_MOE_EP_RETURN_IMPL=owner_bucket_all_to_all_local_scatter && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py \
     --preset qwen3-32b-ep4-profile --implementation xla --warmup 1 --iters 2 \
     > /tmp/moe_return_alltoall.out 2>&1'
```
- Config:
  - temporary dev-TPU hygiene fix: `sudo chmod 1777 /tmp/tpu_logs` after finding the directory left behind by another user and blocking libtpu log creation
  - fixed the sorted-return implementation in [`grug_moe.py`](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py#L609) by replacing the invalid `sort_key_val(token_local, weighted_dispatch)` with a stable `argsort` + `take`
- Result:
  - fresh baseline `scatter_psum_scatter`: `compile_s=13.438`, `steady_s=0.079190`, `2.069M tok/s`
  - `sorted_scatter_psum_scatter`: `compile_s=14.442`, `steady_s=0.094191`, `1.739M tok/s`
  - `owner_bucket_all_to_all_local_scatter`: `compile_s=19.065`, `steady_s=0.106066`, `1.545M tok/s`
- Interpretation:
  - explicit pre-sorting makes the return path materially worse; it does not unlock a better lowering
  - the `all_to_all` architecture change is even worse on TPU here, both in compile cost and steady state
  - together with the earlier `owner_bucket_psum_scatter` regression, the plausible Python/JAX-level return-path ideas are now exhausted on this stack
- Next action:
  - stop surface-level production-kernel work here and treat the remaining hotspot as a compiler/kernel-level problem rather than another high-level decomposition tweak

### 2026-03-09 00:50 - Final conclusion / seal
- Conclusion:
  - On the public `v5p` stack, the remaining production MoE collect/scatter hotspot in the ring-EP path did not yield to the Python/JAX-level return-path changes tested here.
  - Matched EP4 baseline on `v5p-8`:
    - `scatter_psum_scatter`: `2.069M tok/s`
  - Benchmark-only return-path variants all regressed:
    - `owner_bucket_psum_scatter`: about `-1.1%`
    - `sorted_scatter_psum_scatter`: `1.739M tok/s`
    - `owner_bucket_all_to_all_local_scatter`: `1.545M tok/s`
- Confidence:
  - `stable` for the scoped claim above: the real production profile isolates the hotspot, and the remaining production-adjacent return-path rewrites were directly benchmarked on the matched EP4 shape.
- Decision:
  - Stop Python/JAX-level production collect/scatter work on this path.
  - Treat the remaining bottleneck as lower-level compiler/kernel territory rather than something we should keep poking with more high-level return rewrites.
- Ordered next steps:
  1. Keep the current ring-EP production return path as the TPU baseline.
  2. Focus TPU MoE effort on architecture/regime choices or materially new kernel ideas rather than more return-path rewrites.
  3. Revisit this only with lower-level TPU/compiler help or a fundamentally different collect/writeback design.
