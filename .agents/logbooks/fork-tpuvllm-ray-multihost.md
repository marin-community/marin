# Fork TPU vLLM Ray Multi-Host: Research Logbook

## TL;DR

- This logbook is a handoff for the question: what did it actually take to get Qwen3-235B MoE running on TPU with vLLM + Ray multi-host, and how much of that work is already in upstream `tpu-inference/main`?
- The relevant model in the source material is `Qwen/Qwen3-235B-A22B-Thinking-2507`. The user said "225B", but the working logbook and patches are all for the 235B checkpoint.
- The old `fix-ray-multihost-large-models` branch is not the branch to use. It is an ancestor of `main` in the local `tpu-inference` checkout. Most early Ray fixes were merged into upstream `main`.
- The strongest evidence in the old experiment logbook is:
  - dense Llama Ray multi-host became mostly upstream by March 21, 2026;
  - Qwen3-235B MoE still needed a larger stack of local patches, especially around PP rank propagation and KV-cache registration.
- Current upstream `tpu-inference/main` on April 3, 2026 is newer than the experiment checkout and appears to have absorbed several of the early Ray fixes. It does not look obviously sufficient for Qwen3-235B Ray multi-host out of the box.
- If a future agent wants the shortest path to a fresh proof, assume:
  - use upstream `main` as the base,
  - keep the local Qwen3 PP/KV-cache patches in reserve,
  - expect at least one more validation pass before claiming stock `main` works for Qwen3-235B.

## Scope

- Goal: record the patch stack and code-path analysis needed for `tpu-inference` Ray multi-host serving, with emphasis on `Qwen/Qwen3-235B-A22B-Thinking-2507`.
- Primary question: is the working Ray multi-host path for very large models still branch-local, or is it merged into upstream `main`?
- Output for next agent:
  - exact local patch inventory,
  - which patches were part of the final working Qwen3-235B recipe,
  - which patches were intermediate dead ends,
  - what current upstream `main` appears to have fixed already,
  - what still looks missing.

## Sources

- Main experimental logbook:
  - `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/ray_multihost_vllm.md`
- Local experiment checkout with patch scripts:
  - `/Users/ahmed/code/vllm_tpu_multi/tpu-inference`
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev`
- Current upstream `tpu-inference` `main` inspected on April 3, 2026:
  - branch head from `git ls-remote`: `fe442ba25c8ea40ecabcd1fa2a04c6d918e4747d`
- Local experiment checkout state:
  - `tpu-inference` local `main`: `7f0436b5`
  - `fix-ray-multihost-large-models`: `ff6813f5`
  - `git merge-base main fix-ray-multihost-large-models` returns `ff6813f5`, so that branch is an ancestor of `main`, not a separate surviving line.

## What Was Proven In The Original Logbook

### 1. Ray multi-host worked for dense Llama before it worked for Qwen3 MoE

The old logbook shows a progression:

- `RAY-001`: first successful Ray multi-host Llama 8B run.
- `RAY-004`: Llama 70B Ray multi-host working after additional weight-sharding fixes.
- `RAY-008`: upstream `tpu-inference/main` validation for Llama 70B on March 21, 2026.

The March 21 result matters because it distinguishes "old local patch stack" from "mostly upstreamed":

- upstream `main` at `7f0436b5` worked for Llama 3.1 70B PP=4 TP=4 with Ray;
- only one runtime patch was still needed there: `patch_ray_tp_sharding.py`;
- the logbook explicitly says upstream had fixed 5 of the 6 earlier Ray multi-host issues.

Important caveat:

- that conclusion was for dense Llama 70B, not for Qwen3-235B MoE;
- it does not automatically imply Qwen3 MoE worked on stock upstream.

### 2. Qwen3-235B did run, but not on stock upstream

The logbook's `RAY-007` section records a successful run for:

- model: `Qwen/Qwen3-235B-A22B-Thinking-2507`
- topology: Ray multi-host
- configurations observed:
  - v6e-16, PP=4, TP=4
  - v5p-16, PP=2, TP=4

Key recorded outcomes:

- On v6e-16:
  - first successful bring-up after a patch stack;
  - 1,008 / 1,024 completions in one GRPO mini-batch run;
  - around 221 generated tok/s in the low-context, low-concurrency setup;
  - around 28.5 GiB HBM per chip.
- On v5p-16:
  - with `max_model_len=16384` and concurrency tuned up to 256, throughput improved substantially;
  - one clean run reported:
    - 1,024 completions,
    - 0 timeouts,
    - `1127` gen tok/s,
    - mean `3230` tokens/response.

Important caveat:

- These successful Qwen3-235B runs were not described as stock upstream runs.
- The logbook explicitly lists a multi-patch local recipe.

## Final Working Qwen3-235B Patch Stack

The final successful `RAY-007` recipe in the old logbook lists 7 required changes.
Some patch files are preserved in `vllm/marin_dev/`. Some are only described in the logbook.

### Patch 1: `patch_ray_multihost_v2.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_multihost_v2.py`
- Purpose:
  - JAX isolation env vars for Ray multi-host TPU workers.
  - `d.coords` fallback in topology ordering.
  - HBM usage fallback for Ray.
- Modified files:
  - `tpu_inference/worker/tpu_worker.py`
  - `tpu_inference/distributed/utils.py`
  - `tpu_inference/utils.py`
- Why it existed:
  - Without JAX isolation, Ray workers saw the wrong TPU topology and collective compilation broke.
  - Without the coords fallback, device ordering code could break when Ray surfaced devices without `.coords`.
  - Without the HBM fallback, some runs could not report memory usage cleanly under Ray.

### Patch 2: `patch_ray_sharding.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_sharding.py`
- Purpose:
  - make `shard_put()` safe for `None` and raw `PartitionSpec` by wrapping with `NamedSharding(mesh, spec)` before `general_device_put`.
- Modified file:
  - `tpu_inference/models/jax/utils/weight_utils.py`
- Why it existed:
  - the Ray path could crash when `general_device_put()` received `None` or a raw `PartitionSpec`.

### Patch 3: `patch_ray_sharding_v3.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_sharding_v3.py`
- Purpose:
  - hardcoded TP sharding fallback by weight name when `nnx.get_named_sharding(params, mesh)` failed.
- Modified file:
  - `tpu_inference/models/jax/utils/weight_utils.py`
- Why it existed:
  - when `nnx.get_named_sharding` failed, big weights were effectively treated as replicated;
  - that caused OOM for large models;
  - the patch inferred specs such as:
    - `gate_proj`, `up_proj` -> `P(None, 'model')`
    - `down_proj` -> `P('model', None)`
    - `q_proj`, `k_proj`, `v_proj` -> `P(None, 'model', None)`
    - `o_proj` -> `P('model', None, None)`
    - `embed_tokens` -> `P('model', None)`
    - `lm_head` -> `P(None, 'model')`
- Important caveat:
  - this patch is dense-model oriented;
  - the old logbook explicitly worried that MoE expert weights might need more cases;
  - nevertheless this patch remained part of the final Qwen3-235B stack.

### Patch 4: `patch_ray_mm.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_mm.py`
- Purpose:
  - add `supports_mm_inputs = False` to `TPUModelRunner`.
- Modified file:
  - `tpu_inference/runner/tpu_runner.py`
- Why it existed:
  - Ray executor expected `worker.model_runner.supports_mm_inputs`;
  - TPU runner did not define it yet.

### Patch 5: `patch_pp_parallel_state.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_parallel_state.py`
- Purpose:
  - override vLLM PP group state after `ensure_model_parallel_initialized`.
- Modified file:
  - `tpu_inference/worker/tpu_worker.py`
- Why it existed:
  - TPU Ray workers initialize vLLM distributed as `world_size=1, rank=0`;
  - that means every worker thinks it is PP rank 0 inside vLLM's own parallel-state helpers;
  - for evenly divisible layer counts this may go unnoticed;
  - for Qwen3-235B with 94 layers and PP=4, it breaks layer assignment.
- What it changed:
  - after vLLM single-worker initialization, it overwrote:
    - `pp_group.rank`
    - `pp_group.ranks`
    - `pp_group.rank_in_group`
    - `pp_group.world_size`
  - using the actual TPU PP rank from the worker.

### Patch 6: `patch_kv_cache_local_names.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_kv_cache_local_names.py`
- Purpose:
  - rebuild KV-cache registration using local attention layers on each PP worker;
  - allocate extra KV caches when local layer count exceeds the incoming cache count.
- Modified file:
  - `tpu_inference/runner/kv_cache_manager.py`
- Why it existed:
  - after fixing PP rank propagation, KV caches still landed on the wrong workers for non-even PP layer splits;
  - the logbook found a mismatch between:
    - local attention layers used by the worker,
    - KV-cache names/configs delivered to that worker.
- What it changed:
  - rediscovered local attention layer names from the vLLM model;
  - filtered shared KV-cache layers;
  - allocated extra caches if needed;
  - cleared and rebuilt `layer_name_to_kvcache_index` from local layer names.

### Patch 7: gcsfuse / runtime setup changes

- Not a code patch in `tpu-inference`, but part of the working recipe:
  - use `--only-dir` when mounting the model subtree;
  - set `--file-cache-max-size-mb 0` in the specific disk-constrained run;
  - do not destroy Docker containers unnecessarily, because page-cache reuse materially reduces reload time.

## Intermediate Patches And Diagnostic Attempts

These were part of the investigation but were not the final durable fix set.

### `patch_kv_cache_names.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_kv_cache_names.py`
- Purpose:
  - add `.attn` suffix aliases to KV-cache registration names.
- Result:
  - useful diagnosis, not sufficient by itself.
- Why it was not enough:
  - the later logbook shows the problem was deeper than just the `.attn` suffix;
  - local workers were receiving the wrong KV-cache layer groups.

### `patch_pp_rank.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_rank.py`
- Purpose:
  - minimal PP-rank override.
- Result:
  - partial / superseded.

### `patch_pp_rank_v2.py`

- File:
  - `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_rank_v2.py`
- Purpose:
  - pass PP rank through environment variables into vLLM's `get_pp_indices`.
- Result:
  - partial / superseded.
- Why it was superseded:
  - the final issue was not just `get_pp_indices`;
  - other vLLM PP-group consumers still saw rank 0 semantics.

### Other debugging scripts present in `vllm/marin_dev`

- `patch_pp_indices.py`
- `patch_vllm_pp_indices.py`
- `patch_make_layers_debug.py`
- `patch_flash_attn_debug.py`
- `patch_multihost.py`
- `patch_ray_coords.py`

These are useful if a future agent needs to replay the debugging path, but the final logbook evidence points to:

- `patch_pp_parallel_state.py` as the PP fix that actually mattered,
- `patch_kv_cache_local_names.py` as the KV-cache fix that actually mattered.

## Why Qwen3-235B Needed More Than Dense Llama

The logbook isolates two extra classes of failure that did not show up in the same way for Llama 70B:

### 1. Non-even PP layer counts

- Llama 70B case:
  - 80 layers over PP=4 divides evenly.
- Qwen3-235B case:
  - 94 layers over PP=4 does not divide evenly.
- Consequence:
  - any hidden assumption that all PP ranks look like rank 0 becomes visible.

### 2. KV-cache registration and worker ordering

The final Qwen3-235B diagnosis in the old logbook was:

- PP layer assignment became correct after PP-state overrides;
- but KV caches were still on the wrong workers;
- the worker's forward pass and its KV-cache index disagreed;
- this produced `KeyError` on boundary layers.

This is the key reason not to generalize the dense Llama `RAY-008` result to Qwen3 MoE.

## Current Upstream `tpu-inference/main`: What Looks Fixed

This section is based on direct inspection of upstream `main` on April 3, 2026 at:

- `fe442ba25c8ea40ecabcd1fa2a04c6d918e4747d`

### Clearly upstreamed or very likely upstreamed

#### A. Ray multi-host JAX isolation in `tpu_worker.py`

- Current upstream `tpu_worker.py` contains Ray-specific multi-host JAX env setup for PP workers.
- It sets:
  - `TPU_PROCESS_ADDRESSES`
  - `TPU_PROCESS_PORT`
  - `CLOUD_TPU_TASK_ID`
  - `TPU_PROCESS_BOUNDS`
  - `TPU_CHIPS_PER_PROCESS_BOUNDS`
  - `TPU_VISIBLE_CHIPS`
- This is the same functional class of fix as the JAX-isolation part of `patch_ray_multihost_v2.py`.

Assessment:

- Status: fixed upstream.

#### B. `shard_put()` handling of raw `PartitionSpec`

- Current upstream `weight_utils.py` does:
  - `if shardings is None: shardings = ()`
  - wraps tuple shardings as `NamedSharding(mesh, P(*shardings))`
  - wraps raw `PartitionSpec` shardings as `NamedSharding(mesh, shardings)`
- This is the same functional class as `patch_ray_sharding.py`.

Assessment:

- Status: fixed upstream.

#### C. `supports_mm_inputs`

- Current upstream `tpu_runner.py` initializes:
  - `self.supports_mm_inputs = True`
- The old patch only existed because the attribute did not exist yet.

Assessment:

- Status: fixed upstream, but the final upstream semantics differ from the old local patch.

#### D. Dense-model weight-sharding path improved after the old logbook checkpoint

Upstream contains later changes after the March 21 experiment snapshot, including:

- `472d9cb`:
  - `[Bugfix] Fix JAX/Flax path sharding not being properly configured for sharded model weights (#2045)`
- `9226299`:
  - `[Fix] Align weight sharding metadata and ensure RoPE embeddings use nnx.Param (#2042)`

The important practical point is:

- current upstream `weight_utils.py` now uses `model_sharding` in `_load_and_shard_weight` rather than relying on `model_weight.sharding` alone.

Assessment:

- Status: partially fixed upstream.
- This is a real improvement versus the March 21 `main` snapshot used in `RAY-008`.
- It does not obviously prove the Qwen3-235B weight-loading problem is gone.

## Current Upstream `tpu-inference/main`: What Still Looks Missing

### 1. PP parallel-state override for Ray workers

Current upstream `tpu_worker.py` still initializes vLLM distributed as:

- `world_size=1`
- `rank=0`
- `local_rank=0`

This is fine for the TPU JAX execution model itself, but it means vLLM parallel-state helpers still start out believing every worker is rank 0 unless additional state is overridden.

I do not see an upstream equivalent of:

- `patch_pp_parallel_state.py`

Assessment:

- Status: not obviously fixed upstream.
- Relevance to Qwen3-235B: high.

### 2. KV-cache local-name re-registration and extra-cache allocation

Current upstream `kv_cache_manager.py` does not obviously contain:

- local attention-layer rediscovery after allocation,
- extra KV-cache allocation when local layer count exceeds received cache count,
- local re-registration logic equivalent to `patch_kv_cache_local_names.py`.

I do see standard KV-cache registration and cleanup logic, but not the Qwen3-specific repair path.

Assessment:

- Status: not obviously fixed upstream.
- Relevance to Qwen3-235B: high.

### 3. Robust fallback when `nnx.get_named_sharding()` fails

Current upstream `weight_utils.py` still has:

- `try: shardings = nnx.get_named_sharding(params, mesh)`
- `except TypeError: shardings = params`

I do not see either of these Qwen-era fallback strategies in current upstream:

- fallback to `model_weight.value.sharding`,
- hardcoded TP inference by weight name.

Assessment:

- Status: not obviously fixed upstream.
- Relevance to Qwen3-235B:
  - medium to high;
  - it was central to the dense-model 70B story;
  - Qwen3-235B inherited `patch_ray_sharding_v3.py` in the old working stack.

### 4. `d.coords` fallback

Current upstream `distributed/utils.py` still assumes `local_devices` have `.coords` and only logs an error if they do not.
I do not see the explicit fallback from the local patch.

Assessment:

- Status: not obviously fixed upstream in source.
- Relevance to Qwen3-235B:
  - probably low for PP>1;
  - the old logbook already suggested this path was mainly a PP=1 issue.

### 5. HBM fallback logic

Current upstream `utils.py::hbm_usage_bytes()` still uses a simple Ray loop:

- try `memory_stats()` on devices until one works;
- no explicit local-addressable filter;
- no explicit "if nothing worked, synthesize a safe fallback" path.

Assessment:

- Status: not obviously fixed upstream in source.
- Relevance to Qwen3-235B:
  - low to medium;
  - likely not the blocking issue once the run is otherwise healthy.

## Support-Matrix Evidence Against Claiming "Out Of The Box"

The local `tpu-inference` support matrices are another reason to avoid overstating upstream readiness.

They currently show:

- `parallelism_support_matrix.csv`
  - `PP`: correctness ✅, performance ✅
  - `TP`: unverified
  - `EP`: unverified
- `kernel_support_matrix.csv`
  - `MoE`: unverified
- `text_only_model_support_matrix.csv`
  - `Qwen/Qwen3-Coder-480B-A35B-Instruct`: unverified / unverified / unverified
  - `Qwen/Qwen3-30B-A3B`: verified
  - no verified giant Qwen3 MoE entry comparable to 235B.

This does not prove the model cannot work.
It does mean the repo itself is not claiming this class of setup is fully validated.

## Interpretation

### Best current reading

- The separate Ray-fix branch is old history. Do not revive it.
- The correct base is upstream `tpu-inference/main`.
- Upstream `main` has clearly absorbed most of the early Ray multi-host work.
- Upstream `main` likely works for the dense-Llama Ray cases with far less patching than before.
- Qwen3-235B is different:
  - its successful run in the old logbook still required PP-state and KV-cache fixes that I do not see in upstream `main`.

### What I would assume going into a fresh attempt

- Assume stock `main` may get farther than the old logbook-era checkout did.
- Do not assume Qwen3-235B Ray multi-host works out of the box.
- Keep these patches ready first:
  1. `patch_pp_parallel_state.py`
  2. `patch_kv_cache_local_names.py`
  3. a weight-sharding fallback if `nnx.get_named_sharding()` still fails under the chosen configuration
- Treat `patch_ray_mm.py` as obsolete.
- Treat the JAX-isolation and `PartitionSpec` fixes as likely already upstream.

## Recommended Validation Order For A Future Agent

1. Confirm the exact upstream SHAs for both:
   - `vllm-project/tpu-inference`
   - `vllm-project/vllm`
2. Start from stock upstream `main`.
3. Reproduce a dense control case first:
   - Llama 70B, Ray PP=4 TP=4.
4. Then try Qwen3-235B with no local patches.
5. If it fails, add patches in this order:
   - `patch_pp_parallel_state.py`
   - `patch_kv_cache_local_names.py`
   - weight-sharding fallback only if the failure points back to `nnx.get_named_sharding()` or OOM during load.
6. If a fresh stock `main` run succeeds for Qwen3-235B, diff the current code against the old local patch responsibilities before declaring the issue closed. It is possible that the fix moved into upstream `vllm` rather than `tpu-inference`.

## Exact Local Patch Files Worth Preserving

- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_multihost_v2.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_sharding.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_sharding_v2.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_sharding_v3.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_ray_mm.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_parallel_state.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_kv_cache_local_names.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_kv_cache_names.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_rank.py`
- `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/patch_pp_rank_v2.py`

## Final Conclusion

The clean answer for the next agent is:

- Qwen3-235B Ray multi-host was made to work.
- That working state was not stock `tpu-inference/main`.
- The old dedicated branch is no longer the right reference point because its meaningful Ray fixes were largely merged.
- Current upstream `main` appears to have absorbed the early Ray bring-up fixes, but I do not see convincing source-level evidence that it has absorbed the Qwen3-235B PP/KV-cache fixes.
- Until revalidated, the safest assumption is:
  - dense Llama Ray multi-host is close to upstream,
  - Qwen3-235B Ray multi-host still needs local rescue patches.
