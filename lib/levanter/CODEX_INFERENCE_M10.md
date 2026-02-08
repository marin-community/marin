# CODEX Inference M10: Multi-Host Data-Parallel Inference for `128 x 2048`

Status: SOLVED (completed 2026-02-08; M10.1-M10.7 implemented and runtime-validated, plus `v5p-32` prompt-scaling through `1024 x 2048`)

## M10 Solved: `v5p-32` Prompt-Scaling Results (2026-02-08)

All runs below use:
- `inference_mode: host_data_parallel`
- ragged paged attention `ON` (`q_block=16`, `kv_block_pages=8`)
- `max_new_tokens: 2048`

`v5p-32` timing/results:

| prompts | prompts/host | config | wall clock | total generated tokens |
| --- | --- | --- | --- | --- |
| `128` | `32` | `sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_wandb_v5p32.yaml` | `real 253.95s` | `262144` |
| `256` | `64` | `sample_llama8b_multihost_real_256prompts_2048_m10_hostdp_wandb_v5p32.yaml` | `real 237.66s` | `524288` |
| `512` | `128` | `sample_llama8b_multihost_real_512prompts_2048_m10_hostdp_wandb_v5p32.yaml` | `real 271.70s` | `1048576` |
| `1024` | `256` | `sample_llama8b_multihost_real_1024prompts_2048_m10_hostdp_wandb_v5p32.yaml` | `real 366.42s` | `2097152` |

Interpretation note:
- End-to-end wall clock includes launcher/container overhead and can be non-monotonic for small prompt counts.
- Decode-side host round totals (better compute-only signal) scale as expected: about `~86s` (`128`), `~107s` (`256`), `~138s` (`512`), `~211s` (`1024`).

Special handling needed while increasing prompts (`v5p-32`):
- `128/256/512` worked with the existing host-DP ragged setup.
- `1024` initially failed with `ValueError: max_queued_tokens must be >= max_seqs`; fix was `max_queued_tokens: 256` with `max_seqs: 256`.
- `1024` initially produced only `1281` tokens/prompt (`1311744` total) with `max_tokens_per_round: 128`; fix was `max_tokens_per_round: 256` (with `max_rounds: 64`) to remove the decode-budget cap.
- `1024` also required larger engine budgets: `max_seqs: 256`, `max_seqs_in_prefill: 256`, `max_prefill_size: 4096`, `max_pages: 4608`.
- Final `1024` validation: all `1024` prompts completed at exactly `2048` generated tokens each.

## Latest Update (2026-02-07)

- M10.4 execution succeeded on `simpo_worker_2` (`v5p-16`) with ragged attention enabled:
  - `model.use_tpu_ragged_paged_attention: true`
  - `model.ragged_paged_q_block_size: 16`
  - `model.ragged_paged_kv_block_pages: 8`
- Key fix to enable this:
  - HF load path now applies ragged kernel overrides (`use_tpu_ragged_paged_attention`, `ragged_paged_q_block_size`,
    `ragged_paged_kv_block_pages`) in `src/levanter/main/sample_lm_multihost.py`.
- Per-host outputs were written and validated:
  - `/tmp/m10_host_outputs/host_0000_of_0002.jsonl` (`64` rows)
  - `/tmp/m10_host_outputs/host_0001_of_0002.jsonl` (`64` rows)
  - rows include `generated_token_count: 2048`.
- M10.5 sweep executed on `simpo_worker_2` (`v5p-16`) across scheduler/page knobs (`max_tokens_per_round`,
  `max_queued_tokens`, `page_size`, `max_pages`, `max_rounds`).
- Best M10.5 run in this sweep:
  - config: `ps128/p1088/tpr128/r64/mqt512` (`..._m105_f_...yaml`)
  - wall clock: `real 4:00.22` (`~240.22s`)
  - host decode summaries: `round_total_s ~= 106.36s` and `107.93s`.

## Performance Snapshot (`128 x 2048`, `v5p-16`)

Direct comparison (current best known runs):
- M9 global-mesh baseline (end of M9):
  - ragged paged attention: `ON`
  - fastest observed wall clock: `real 255.93s` (`OptO rerun2`)
- M10.3 host-data-parallel fallback (non-ragged):
  - ragged paged attention: `OFF`
  - host decode summaries: `round_total_s ~= 550.65s` and `552.51s`
  - end-to-end wall clock from run timestamps (`19:23:26 -> 19:34:21`): `~655s`
- M10.4 host-data-parallel with ragged restored:
  - ragged paged attention: `ON`
  - host decode summaries: `round_total_s ~= 100.90s` and `102.38s`
  - end-to-end wall clock from run timestamps (`19:46:03 -> 19:49:12`): `~189s`
- M10.5 host-data-parallel (ragged + tuning sweep best):
  - ragged paged attention: `ON`
  - host decode summaries: `round_total_s ~= 106.36s` and `107.93s`
  - wall clock: `real 240.22s` (config `ps128/p1088/tpr128/r64/mqt512`)

Observed result:
- M10.4 ragged vs M10.3 non-ragged (host-DP):
  - end-to-end speedup: `655 / 189 ~= 3.47x`
  - per-host round speedup: `~551 / ~101.6 ~= 5.4x`
- M10.5 ragged+tuned best vs M10.3 non-ragged (host-DP):
  - end-to-end speedup: `655 / 240.22 ~= 2.73x`
  - per-host round speedup: `~551 / ~107.15 ~= 5.14x`
- M10.5 ragged+tuned best vs M10.4 ragged baseline:
  - end-to-end: `189 / 240.22 ~= 0.79x` (about `27%` slower)
  - per-host round: `~101.6 / ~107.15 ~= 0.95x` (about `5.5%` slower)
- M10.4 ragged vs M9 global baseline:
  - end-to-end speedup: `255.93 / 189 ~= 1.35x` faster.

Ragged status quick map:
- M9 baseline: `ON`
- M10.3 baseline: `OFF`
- M10.4 baseline: `ON`
- M10.5 best tuned: `ON`

## Goal

Add a true multi-host data-parallel inference path for the M9 fixed-length workload:
- `128` prompts
- `max_new_tokens: 2048`
- `n_rounds: 1`

Primary objective:
- Make `v5p-32` (16 chips) materially faster than `v5p-16` (8 chips) for one end-to-end run of `128 x 2048`.

## Baseline Context (From M9)

M9 established stable fixed-length configs for `128 x 2048`, but current execution uses a global mesh path. Under that path, moving from `v5p-16` to `v5p-32` does not automatically imply near-2x lower latency for one run.

Reference:
- `CODEX_REFACTOR_KV.md` M10 section.
- `CODEX_INFERENCE_M9.md` for fixed-length and throughput results.

## Current Behavior (Global Mesh Path)

Current `sample_lm_multihost.py` behavior:
1. Runs under one global device mesh with `config.trainer.use_device_mesh()`.
2. Tokenizes prompts on leader and broadcasts token IDs to all hosts.
3. Builds the same request list on all hosts.
4. Executes one global `engine.generate(requests)` call across that shared prompt set.
5. M9 configs use `trainer.mesh.axes.model: 4` and do not introduce explicit prompt sharding by host.

Implication:
- This is globally sharded inference execution, not explicit host-level prompt partitioning.
- Additional hosts are not automatically equivalent to splitting prompt batch in half per host.

## M10 Scope

### M10.1 (Implemented)

Scope:
- Add plumbing and guardrails for a future host data-parallel path without changing current default behavior.

Implemented changes:
1. Added `inference_mode` config switch to `SampleLmMultihostConfig`:
   - `global_mesh` (default)
   - `host_data_parallel`
2. Added `_validate_inference_mode_safety(...)` fail-fast guardrails.
3. Kept existing `global_mesh` execution path unchanged.
4. Added unit tests for inference-mode guard behavior.

Guardrails in M10.1:
- `host_data_parallel` requires:
  - multi-host execution
  - `n_rounds == 1`
  - `n_generations == 1`
  - `defer_tracker_logs_until_end == true`
- In M10.2, guard validation remains pure validation; temporary `NotImplementedError` moved to explicit post-sharding stub logic in `main`.

Code references:
- `src/levanter/main/sample_lm_multihost.py`
- `tests/inference/test_sample_lm_multihost_config_guard.py`

## Validation (M10.1/M10.2/M10.3)

Validation commands:
- `uv run pytest tests/inference/test_sample_lm_multihost_config_guard.py`
- `uv run pytest tests/inference/test_sample_lm_multihost_prompt_sharding.py`
- `uv run pytest tests/inference/test_sample_lm_multihost_host_outputs.py`

Expected outcomes:
- `global_mesh` mode remains allowed and unchanged.
- Unknown `inference_mode` is rejected.
- `host_data_parallel` invalid combinations are rejected with explicit errors.
- `host_data_parallel` valid guard combination passes validation.
- `host_data_parallel` runtime executes host-local generation and writes per-host JSONL generation dumps.

### M10.2 (Implemented)

Scope:
- Implement deterministic host prompt sharding and preserve global prompt IDs in request metadata.

Implemented changes:
1. Added deterministic shard helpers:
   - `_prompt_shard_bounds(num_prompts, process_index, process_count)`
   - `_shard_prompts_for_host(prompts, prompt_ids, process_index, process_count)`
2. Extended request construction to support global prompt identity across shards:
   - `_build_requests_for_rounds(..., prompt_id_offset=..., total_num_prompts=...)`
3. Moved the `host_data_parallel` temporary `NotImplementedError` out of config validation and into explicit post-sharding stub logic in `main`.
4. Added tests for:
   - shard coverage/no-overlap/no-gap behavior
   - uneven split behavior
   - global request-ID mapping with sharded offsets

Notes:
- M10.2 established deterministic sharding and global prompt IDs.
- M10.3 removed the temporary runtime stub and now executes locally per host.

Code references:
- `src/levanter/main/sample_lm_multihost.py`
- `tests/inference/test_sample_lm_multihost_prompt_sharding.py`
- `tests/inference/test_sample_lm_multihost_config_guard.py`

### M10.3 (Implemented)

Scope:
- Execute `host_data_parallel` inference on each host’s local mesh and persist each host shard’s generations locally.

Implemented changes:
1. Added local execution path for `inference_mode=host_data_parallel`:
   - build/load model through the existing global startup path
   - replicate to host-local devices (`replicate_model_to_local_mesh`)
   - create local mesh (`create_local_mesh`) and run local `engine.generate(...)`
2. Added host-local engine validation and request construction:
   - `_validate_engine_config(...)` now runs against each host shard in this mode
   - request IDs keep global prompt identity from M10.2 sharding offsets
3. Added local output persistence:
   - new config field: `host_data_parallel_output_dir` (default: `host_data_parallel_outputs`)
   - one JSONL file per host:
     - `host_0000_of_0016.jsonl`, `host_0001_of_0016.jsonl`, etc.
   - rows include process metadata, global/local prompt IDs, request ID, generated tokens, and generated text
4. Added unit coverage for host-local output helpers:
   - deterministic host output file naming
   - row content and global prompt identity
   - mismatch guardrails on sequence count

Runtime notes from first TPU runs (`simpo_worker_2`, `v5p-16`):
1. Observed failure:
   - `NameError: name 'ResourceAxis' is not defined` from `create_local_mesh(...)` default axes path.
   - Fix: import `ResourceAxis` in `src/levanter/utils/mesh.py`.
2. Observed failure:
   - `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: CopyArrays only supports destination device list of the same size as the array device lists.`
   - This occurred in `replicate_model_to_local_mesh(...)` when copying from global-sharded arrays to per-host local mesh replicas.
   - Fix: materialize each leaf to host (`jax.device_get`) before `jax.device_put(..., replicated_sharding)` in `src/levanter/utils/jax_utils.py`.
3. Observed failure:
   - `ValueError: Received incompatible devices for jitted computation ... device ids [0,1,2,3] ... mesh ids [0,2,1,3]`.
   - Root cause: replication and JIT were using different local mesh device orders.
   - Fix: pass the exact `local_mesh` into `replicate_model_to_local_mesh(...)` and use local-mesh device order for host-local engine config.
4. Observed failure:
   - `RESOURCE_EXHAUSTED ... ragged_paged_attention ... Scoped allocation ... limit 16.00M`.
   - Even after host-local engine down-sizing (`max_seqs=64`, `max_seqs_in_prefill=64`, `max_pages=1088`), ragged prefill still failed on this setup.
   - Working run-time fix for this M10.3 `v5p-16` config: set `model.use_tpu_ragged_paged_attention: false`.
5. Validation result:
   - `sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_v5p16.yaml` completed successfully.
   - Each worker wrote one shard file:
     - `/tmp/m10_host_outputs/host_0000_of_0002.jsonl`
     - `/tmp/m10_host_outputs/host_0001_of_0002.jsonl`
   - Sample rows show `generated_token_count: 2048` for local prompts.
6. M10.4 follow-up:
   - Restored ragged operation by fixing HF override plumbing for ragged kernel knobs in `_load_model(...)`.
   - With ragged enabled (`q_block=16`, `kv_block_pages=8`), the same `128 x 2048` host-DP run completed end-to-end on `v5p-16`.

Code references:
- `src/levanter/main/sample_lm_multihost.py`
- `tests/inference/test_sample_lm_multihost_host_outputs.py`

How to inspect per-host outputs after launch:
1. Set in config:
   - `inference_mode: host_data_parallel`
   - `host_data_parallel_output_dir: /tmp/m10_host_outputs`
2. After job completion, inspect across workers:
   - `gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command='ls -lh /tmp/m10_host_outputs'`
   - `gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command='head -n 2 /tmp/m10_host_outputs/host_*.jsonl'`

## M10.4+ Plan

### M10.4 - Ragged Paged Attention Recovery for M10.3
- Goal: make `inference_mode=host_data_parallel` work with `model.use_tpu_ragged_paged_attention: true` on the `128 x 2048` path.
- Implemented progress:
  - Reproduced the scoped-vmem OOM in host-local prefill on `v5p-16`.
  - Fixed HF load overrides so ragged kernel knobs are applied in host-DP mode.
  - Validated a successful ragged-on run with per-host outputs and full-length generations.
- Remaining:
  - Repeatability checks (multiple reruns) and config hardening.
- Exit criteria:
  - M10.3 completes end-to-end with ragged attention enabled (met in current run).
  - Per-host outputs are still written and preserve `generated_token_count: 2048`.
  - No regression in M10.3 correctness tests.

### M10.5 - Host-DP Parameter Sweep (M9-style)
- Goal: run the same style of tuning sweep used in M9, but on the M10 host-data-parallel path now that ragged is restored.
- Sweep axes (at minimum):
  - `engine.max_tokens_per_round`
  - `engine.max_queued_tokens`
  - `engine.page_size`
  - `engine.max_pages`
  - `engine.max_rounds` (for decode-iteration behavior)
- Keep workload fixed:
  - `128 prompts`, `max_new_tokens=2048`, `n_rounds=1`, `n_generations=1`
  - ragged enabled (`use_tpu_ragged_paged_attention=true`) unless an explicit fallback experiment is being run.
- Deliverables:
  - pass/fail matrix with runtime signatures
  - throughput and wall-clock table for each config
  - recommended host-DP operating point for subsequent milestones
- Execution status: PARTIAL COMPLETE (2026-02-07, `v5p-16`, all runs PASS).
- Topology explainer (why M9-style tuning can regress in M10.5):
  - M9 path is one global model-parallel mesh across all chips/hosts, so each decode step uses full-cluster compute.
  - M10 host-data-parallel path runs one local model replica per host; each host serves half the prompts (`64`) but also has half
    the chips (`4` on `v5p-16` 2-host runs).
  - So "half prompts per host" does not imply near-2x faster local rounds: per-host token latency can stay similar, and scheduler/
    paging overhead can dominate if budgets are over-aggressive.
  - Practical implication: M9-optimal settings (`tpr`, `mqt`, `page_size`, `max_pages`) do not transfer 1:1; host-DP needs its own
    operating point.
- Swept configs and results:
  - A `ps128/p2304/tpr256/r64/mqt512`: `real 258.88s`, rounds `116.91s / 117.81s`
  - B `ps128/p2304/tpr256/r32/mqt512`: `real 260.93s`, rounds `118.64s / 119.12s`
  - C `ps128/p2304/tpr128/r64/mqt512`: `real 251.72s`, rounds `107.02s / 107.54s`
  - D `ps64/p4224/tpr256/r64/mqt512`: `real 258.07s`, rounds `120.14s / 120.49s`
  - E `ps128/p1088/tpr128/r64/mqt128`: `real 258.71s`, rounds `107.74s / 108.88s`
  - F `ps128/p1088/tpr128/r64/mqt512`: `real 240.22s`, rounds `106.36s / 107.93s` (**best wall clock**)
  - G `ps128/p1088/tpr256/r64/mqt512`: `real 259.56s`, rounds `118.19s / 117.98s`
- Current M10.5 recommendation:
  - keep `page_size=128`, `max_pages=1088`, `max_tokens_per_round=128`, `max_rounds=64`
  - use higher queue depth (`max_queued_tokens=512`) from config F.
- Temporary decision:
  - Revisit M10.5 later after M10.6/M10.7. For now, treat M10.4 as the latency reference and keep this M10.5 sweep as
    a documented exploration result.
- M10.5 summary snapshot:
  - best M10.5 config from sweep:
    - `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_m105_f_ps128_p1088_tpr128_r64_mqt512.yaml`
    - wall clock: `real 240.22s`
    - host round totals: `106.36s` and `107.93s`
    - all runs produced full `131072` tokens per host (`64 * 2048`)
  - requested comparisons (using existing baselines in `lib/levanter/CODEX_INFERENCE_M10.md:24`):
    1. baseline M10.3 non-ragged: `~655s`
    2. baseline M10.4 ragged-only: `~189s`
    3. M10.5 ragged+tuning (best): `240.22s`
  - speedups:
    1. ragged+tuning over M10.3 baseline: `655 / 240.22 = 2.73x` faster
    2. ragged-only over M10.3 baseline: `655 / 189 = 3.47x` faster
    3. ragged+tuning vs ragged-only: `189 / 240.22 = 0.79x` (about `27%` slower)

### M10.6 - Gather and Leader Aggregation (Implemented in Code)

Goal:
- Add explicit cross-host gather in `host_data_parallel` mode so leader can produce one merged global output.

Implemented changes:
1. Added host-row serialization transport helpers in `sample_lm_multihost.py`:
   - `_encode_host_rows_payload(...)`
   - `_decode_host_rows_payload(...)`
   - payloads are JSON + `zlib` compressed + base64-encoded for distributed KV transport.
2. Added host-row gather helper:
   - `_gather_host_rows_to_leader(...)`
   - each host publishes its local rows to JAX distributed KV store
   - process `0` fetches host payloads in deterministic host-index order.
3. Added deterministic merged ordering helper:
   - `_merge_gathered_host_rows(...)`
   - validates source identity (`row.process_index` must match host payload index)
   - sorts merged rows by `(round_index, global_prompt_index, generation_index, process_index)`.
4. Added leader merged output artifact:
   - `_merged_host_output_path(output_dir, process_count)`
   - leader writes `all_hosts_merged_of_XXXX.jsonl` next to per-host files.
5. Host-DP runtime path now performs:
   - local generation
   - local row build + optional local JSONL write
   - cross-host gather to leader
   - leader merged JSONL write.

Validation (unit tests):
- `uv run pytest tests/inference/test_sample_lm_multihost_host_outputs.py`
  - deterministic merged path naming
  - payload encode/decode roundtrip
  - merged-order determinism and source mismatch guardrail.

Runtime validation notes:
- TPU validation completed on `2026-02-07` on `simpo_worker_2` (`v5p-16`) with fixed workload
  (`128 prompts`, `max_new_tokens=2048`, ragged enabled), using two paired configs:
- M10.4-equivalent baseline (gather disabled):
  - ragged paged attention: `ON`
  - `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_m104_compare.yaml`
  - wall clock: `real 249.51s`
  - host round totals: `106.37s` and `107.10s`
- M10.6 (gather enabled):
  - ragged paged attention: `ON`
  - `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_m106_compare.yaml`
  - wall clock: `real 266.58s`
  - host round totals: `106.93s` and `107.54s`
- Measured impact of enabling gather:
  - decode/round slowdown: `~0.47%` (`+0.50s` on average host round total)
  - end-to-end wall slowdown: `~6.84%` (`+17.07s`)
- Artifact validation:
  - M10.4-equivalent output: per-host shards only (`64` rows each)
  - M10.6 output:
    - per-host shards (`64` rows each)
    - leader merged file: `/tmp/m10_host_outputs_m106_compare/all_hosts_merged_of_0002.jsonl` (`128` rows)
- Interpretation:
  - gather/merge overhead is negligible in decode-time terms for this workload.
  - most wall-clock delta appears outside decode (launcher/container/runtime overhead + run-to-run variance).
- Logs:
  - `/tmp/m10_compare_m104.log`
  - `/tmp/m10_compare_m106.log`

### M10.7 - Benchmark Matrix
Status: COMPLETE (`2026-02-07` runtime sweep on active TPUs `simpo_worker_2` `v5p-16` and `simpo_worker_3` `v5p-32`).

Target matrix:
1. `v5p-16 + global_mesh`
2. `v5p-16 + host_data_parallel`
3. `v5p-32 + global_mesh`
4. `v5p-32 + host_data_parallel`

Common workload:
- `128 prompts`, `max_new_tokens=2048`, `n_rounds=1`, `n_generations=1`.

Ragged-global failure signatures observed before fallback:
- `v5p-16 + global_mesh` (ragged ON) failed with TPU `RESOURCE_EXHAUSTED` in ragged prefill.
- `v5p-32 + global_mesh` (ragged ON) failed with TPU `RESOURCE_EXHAUSTED` in ragged prefill.
- Fail logs:
  - `/tmp/m10_7_v5p16_global.log` (`real 135.44s`, fail)
  - `/tmp/m10_7_v5p32_global.log` (`real 129.70s`, fail)
  - `/tmp/m10_7_v5p16_global_rerun.log` (`real 138.02s`, fail)
  - `/tmp/m10_7_v5p32_global_rerun.log` (`real 134.65s`, fail)

Completed comparison matrix (global fallback used `model.use_tpu_ragged_paged_attention: false`):
- `v5p-16 + global_mesh` (non-ragged fallback):
  - ragged paged attention: `OFF` (fallback from ragged OOM)
  - config: `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m107_v5p16_global.yaml`
  - wall clock: `real 1093.91s`
  - round total(s): `983.37s`, `983.37s`
  - generated tokens: `262144`
- `v5p-32 + global_mesh` (non-ragged fallback):
  - ragged paged attention: `OFF` (fallback from ragged OOM)
  - config: `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m107_v5p32_global.yaml`
  - wall clock: `real 1086.14s`
  - round total(s): `982.86s` (all hosts)
  - generated tokens: `262144`
- `v5p-16 + host_data_parallel` (M10.6 path, ragged ON):
  - ragged paged attention: `ON`
  - config: `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m107_v5p16_hostdp.yaml`
  - wall clock: `real 254.26s`
  - host round totals: `107.41s`, `107.40s`
  - generated tokens per host: `131072` (`64 * 2048`)
  - output files:
    - per-host shards: `64` rows each
    - merged leader file: `all_hosts_merged_of_0002.jsonl` (`128` rows)
- `v5p-32 + host_data_parallel` (M10.6 path, ragged ON):
  - ragged paged attention: `ON`
  - config: `lib/levanter/config/sampler/sample_llama8b_multihost_real_128prompts_2048_m107_v5p32_hostdp.yaml`
  - wall clock: `real 221.80s`
  - host round totals: `86.28s`, `86.58s`, `86.33s`, `86.96s`
  - generated tokens per host: `65536` (`32 * 2048`)
  - output files:
    - per-host shards: `32` rows each
    - merged leader file: `all_hosts_merged_of_0004.jsonl` (`128` rows)

Ragged status for M10.7 completed matrix:
1. `v5p-16 + global_mesh`: `OFF` (ragged `ON` failed with OOM).
2. `v5p-32 + global_mesh`: `OFF` (ragged `ON` failed with OOM).
3. `v5p-16 + host_data_parallel`: `ON`.
4. `v5p-32 + host_data_parallel`: `ON`.

Derived speedups from completed matrix:
1. `v5p-32` vs `v5p-16` in `global_mesh` fallback: `1093.91 / 1086.14 = 1.01x` (no material gain).
2. `v5p-32` vs `v5p-16` in `host_data_parallel`: `254.26 / 221.80 = 1.15x` faster.
3. `host_data_parallel` vs `global_mesh` on `v5p-16`: `1093.91 / 254.26 = 4.30x` faster.
4. `host_data_parallel` vs `global_mesh` on `v5p-32`: `1086.14 / 221.80 = 4.90x` faster.

Logs:
- `/tmp/m10_7_v5p16_global_nonragged.log`
- `/tmp/m10_7_v5p32_global_nonragged.log`
- `/tmp/m10_7_v5p16_hostdp.log`
- `/tmp/m10_7_v5p32_hostdp.log`

## M10 Closure

- M10 is considered complete for its stated objective: practical multi-host data-parallel inference path with runtime validation and performance measurements on `v5p-16` and `v5p-32`.
- Remaining tuning questions (for example weighted prompt balancing and tighter scaling targets) are non-blocking and should be handled as follow-on work, not M10 blockers.
