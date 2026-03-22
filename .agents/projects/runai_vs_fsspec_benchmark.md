# RunAI vs fsspec Weight Loading Benchmark

## ⚠️  CRITICAL: Model-Region Colocation Rules

**NEVER read model weights across GCS regions.** Cross-region egress is
extremely expensive and confounds all throughput measurements. Every benchmark
MUST use a model path in the same region as the compute worker.

### Model Paths by Region

Use ONLY these paths when running benchmarks. If a model is missing in the
region you need, copy it first with `gcloud storage cp -r`.

#### us-east1 (v6e-4 workers in us-east1-d)

| Model | Path |
|-------|------|
| Llama 3.1 8B Instruct | `gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f` |
| Llama 3.3 70B Instruct | `gs://marin-us-east1/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b` |
| Qwen3 235B | `gs://marin-us-east1/models/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae` |

#### us-east5 (v6e-4 workers in us-east5-b)

| Model | Path |
|-------|------|
| Llama 3.1 8B Base | `gs://marin-us-east5/gcsfuse_mount/models/meta-llama--Llama-3-1-8B--d04e592` |
| Llama 3.2 1B | `gs://marin-us-east5/models/meta-llama--Llama-3-2-1B--main` |
| Llama 3.3 70B Instruct | `gs://marin-us-east5/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b` |
| Qwen3 0.6B | `gs://marin-us-east5/models/Qwen--Qwen3-0-6B--main` |
| OLMo-2 7B | `gs://marin-us-east5/models/allenai--OLMo-2-1124-7B--7df9a82` |

#### us-central1 (no v6e-4 workers observed; v5p-8 workers here)

| Model | Path |
|-------|------|
| Llama 3.1 8B Instruct | `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f` |
| Llama 3.3 70B Instruct | `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b` |

#### us-central2 — DO NOT USE for benchmarks

Has models but prefer us-central1, us-east1, or us-east5. If none of those
have free workers, tell the user rather than using central2.

### Before every benchmark submission

1. Check which zone the TPU worker will land in (from job history or Iris)
2. Look up the model path for that region in the table above
3. If missing, copy with `gcloud storage cp -r gs://marin-{src}/... gs://marin-{dst}/...`
   and update this table
4. Use the regional path in the job command

---

## Purpose

Controlled comparison of weight loading paths on TPU to isolate where time
is actually spent. Previous analysis conflated different metrics (download
time, weight pipeline time, total cold start) and made unproven causal claims.
This logbook tracks every run with exact commands, exact metrics, and exact
code paths so we can reach defensible conclusions.

## Open Questions

1. **Where does time go in `MODEL_IMPL_TYPE=vllm` vs `auto`?** The A/B test
   shows 366s vs 151s total but RunAI download speed is similar (~1 GiB/s both).
   The slow phase is somewhere *after* download, likely in `vllm_get_model()` or
   `shard_model_to_tpu()`. Phase-timed instrumentation is pending.

2. **What was the original 53 MiB/s?** The issue body says "RunAI downloads via
   single-threaded HTTP at ~53 MiB/s." The negative results section says "RunAI
   concurrency parameter (up to 16) has no effect — throughput stays at 53 MiB/s."
   Our A/B test shows RunAI downloading at ~1 GiB/s on both paths. These are
   probably different metrics or different conditions. Not yet reconciled.

3. **Cache warmth confound.** The A/B test ran on the same worker, same model,
   back-to-back. GCS or RunAI caching could explain the ~1 GiB/s download speed.
   Need a cold-cache test to rule this out.

4. **Is the original in-process speedup still valid?** The original experiment
   used `LLM(load_format="dummy")` + Levanter fsspec + `sync_weights()`, which
   bypassed RunAI entirely. That path's value was real: fast startup + low memory.
   But the *baseline it was compared against* (53 MiB/s) may have been confounded
   by `MODEL_IMPL_TYPE=vllm` forcing the PyTorch path (commit `3f1420e8d`).

---

## Historical Metrics (from issue #3768 and comments)

These are the numbers as originally reported. Metric boundaries are not always
clear from the original text.

### Original issue body (filed ~March 12-15)

| Claimed metric | Value | What it probably measures |
|----------------|-------|--------------------------|
| "RunAI downloads via single-threaded HTTP" | 53 MiB/s | Unclear — could be download only, or end-to-end throughput including weight materialization |
| "RunAI concurrency (up to 16) no effect" | 53 MiB/s | Same as above |
| "70B weight download at 53 MiB/s" | 41 min | 131 GiB / 53 MiB/s ≈ 41 min. Matches if 53 MiB/s is sustained download |
| "Cold start exceeds 1 hour" | >60 min | Download (41 min) + XLA compilation (20-30 min) |

### Original in-process results (commit `dbeddbd45`, March 15-16)

These used `LLM(load_format="dummy")` + Levanter `read_safetensors_fsspec()` +
`sync_weights()`. The `MODEL_IMPL_TYPE` env var was NOT set by the in-process
path — tpu-inference defaults it to `"auto"` in `envs.py:19`. However, the
*baseline* these were compared against used `vllm serve` which DID set
`MODEL_IMPL_TYPE=vllm` (commit `3f1420e8d`).

**8B (Llama 3.1, v5p-8, TP=1):**

| Phase | Time | Notes |
|-------|------|-------|
| LLM skeleton | 108s | `LLM(load_format="dummy")` |
| Weight download (fsspec) | 95.1s | 15 GiB, ~162 MiB/s, Levanter async with 4 concurrent chunks |
| NNX convert | 3.4s | `levanter_state_dict_to_nnx_state_on_cpu` |
| Weight inject | 40.5s | `sync_weights()` |
| **Weight pipeline** | **139s** | Download + convert + inject |
| **Total** | ~26 min | Includes XLA compilation (~25 min) |
| Claimed baseline | "RunAI 300s" | Issue says "vs RunAI 300s" — unclear if download only or pipeline |

**70B (Llama 3.3, v5p-8, TP=4, enforce_eager):**

| Phase | Time | Notes |
|-------|------|-------|
| Bootstrap | 2.4s | Stage config.json + tokenizer to tmpdir |
| LLM skeleton | 189.0s | `LLM(load_format="dummy", tensor_parallel_size=4)` |
| Weight download (fsspec) | 876.1s (14.6 min) | 131 GiB at 154 MiB/s |
| Reshape | 40.4s | HF 2D → Levanter 3D for 320 attention projections |
| NNX convert | 53.0s | |
| Weight inject | 484.0s | `sync_weights()` |
| **Weight pipeline** | **1453.4s (24.2 min)** | |
| Generation | 94.7s | 5 prompts, 568 tokens, 6.0 tok/s |
| **Total** | **1739.6s (29 min)** | |
| Claimed baseline | "RunAI 41 min download alone" | 131 GiB / 53 MiB/s |

### Shard-streaming results (March 16, issue comment 1)

Same in-process path but loading one shard at a time instead of all at once.

**70B at 24GB memory:**

| Metric | All-at-once | Shard-streaming |
|--------|------------|-----------------|
| Iris `--memory` | 400GB | **24GB** |
| Peak host RAM | ~131 GiB + overhead | ~15 GiB (skeleton) + ~5 GiB (shard) |
| Weight pipeline | 1453.4s | **1379.8s** |
| `sync_weights` calls | 1 | 30 |

### Stress test results (March 18, issue comment 2)

**8B in-process (50 prompts, LLM.generate via queue):**
- Weight pipeline: 119.2s, 15.0 GiB at 128 MiB/s
- 50/50 successful, 6400 tokens, 34.1 tok/s aggregate

**70B in-process (10 prompts):**
- Weight pipeline: ~1230s, 131.4 GiB at 109 MiB/s
- 10/10 successful, 1280 tokens, 6.7 tok/s aggregate

### Baseline comparison (March 18, issue comment 3)

**8B subprocess baseline vs in-process:**

| Metric | Subprocess (`vllm serve`) | In-process (queue + fsspec) |
|--------|--------------------------|----------------------------|
| Server startup | 2052.6s (34 min) | ~122s (2 min) |
| Weight loading | 359s at 42.7 MiB/s | 119s at 128 MiB/s |
| XLA compilation | ~1700s | 0s (enforce_eager) |
| Generation (50 prompts) | 17.8s | 187.7s |
| Aggregate tok/s | 360.0 | 34.1 |
| Memory | 64 GB | 24 GB |

**Key:** The subprocess baseline weight loading was 359s at 42.7 MiB/s. This
is close to the original 53 MiB/s claim. The subprocess path used
`MODEL_IMPL_TYPE=vllm` (commit `3f1420e8d`) which forces the PyTorch path.
This is the closest we have to an actual measurement of "RunAI at 53 MiB/s" —
it was 42.7 MiB/s in the subprocess comparison.

### Temperature=0 comparison (March 18, issue comment 4)

| Metric | Subprocess | In-process |
|--------|-----------|------------|
| Server startup | 1331s (22 min) | ~122s (2 min) |
| Aggregate tok/s | 401.7 | 53.2 |
| p50 latency | 1.19s | 2.42s |
| Exact match (50 prompts) | — | 12/50 byte-identical |

### Throughput regression analysis (March 18, issue comments 5-7)

Root cause: queue serialized `LLM.generate([1_prompt])` one at a time.

| Method | Aggregate tok/s |
|--------|----------------|
| Subprocess baseline (`vllm serve`, 4 concurrent) | 401.7 |
| Subprocess baseline (per-request) | 105.3 |
| Single `generate(50 prompts)` — batch | 80.9 |
| Queue + HTTP (enforce_eager) | 53.2 |
| Queue + HTTP (XLA compiled) | 33.3 |

---

## New Benchmarks (March 22, this session)

### Code path context

Before this session, the **subprocess `vllm serve` path** forced Llama models
through the PyTorch `get_vllm_model()` path because Marin's `vllm_server.py`
hardcoded `MODEL_IMPL_TYPE=vllm` (commit `3f1420e8d`). Even after fixing that
to `"auto"` (commit `8f83e3692`), vLLM remapped `LlamaForCausalLM` →
`MistralForCausalLM` which wasn't in the JAX registry, causing a fallback to
the PyTorch path.

Note: the old **in-process `LLM.generate()` path** (commit `dbeddbd45`) did
NOT set `MODEL_IMPL_TYPE` — tpu-inference defaults it to `"auto"` in
`envs.py:19`. That path bypassed the routing issue entirely by using
`load_format="dummy"` + Levanter fsspec + `sync_weights()`.

This session fixed the subprocess path routing (6 fork commits). Llama now
correctly routes through the JAX `get_flax_model()` path in both modes.

### Run 1: 8B fsspec sequential (v9) — first fork-native success

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-smoke-fsspec-v9` |
| **TPU** | v6e-4, us-east5-b |
| **Worker** | `marin-tpu_v6e_4-us-east5-b-20260322-0115-34282cfd` |
| **Model** | `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f` |
| **Memory** | 32 GiB |
| **Code path** | `get_flax_model` → `abstract_load` + `fsspec_streamer` |
| **Wheel** | `marin-988ed996` |

| Phase | Time | Detail |
|-------|------|--------|
| Shard 1 | 58.0s | 82 tensors, 4.6 GiB, 82 MiB/s, RSS=12642 MB |
| Shard 2 | 29.0s | 104 tensors, 4.7 GiB, 165 MiB/s, RSS=12829 MB |
| Shard 3 | 31.6s | 100 tensors, 4.6 GiB, 149 MiB/s, RSS=12829 MB |
| Shard 4 | 7.8s | 5 tensors, 1.1 GiB, 143 MiB/s, RSS=12829 MB |
| **All shards** | **126.3s** | **291 tensors, 15.0 GiB, 121 MiB/s avg** |
| **Total runtime** | **342.2s** | |

### Run 2: 8B RunAI baseline (v7) — default path with alias fix

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-smoke-baseline-v7` |
| **TPU** | v6e-4, us-east5-b (same worker as v9) |
| **Memory** | 64 GiB |
| **Code path** | `get_flax_model` → default (RunAI iterator) |

| Phase | Time | Detail |
|-------|------|--------|
| RunAI download | 7.0s | 15.0 GiB, 2.1 GiB/s |
| HBM used | — | 14.96 GiB |
| **Total runtime** | **135.8s** | |

**Note:** Same worker as v9. Cache warmth is a confound for the RunAI 2.1 GiB/s number.

### Run 3-5: 70B comparisons (v5p-8, TP=4)

**Run 3: 70B fsspec sequential (`70b-fsspec-v3`, 64GB)**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-70b-fsspec-v3` |
| **TPU** | v5p-8, TP=4 |
| **Memory** | 64 GiB |
| **All shards** | 723 tensors, 131.4 GiB in 1173.7s (115 MiB/s avg) |
| **HBM** | 32.86 GiB/chip |
| **Peak RSS** | 15509 MB |
| **Total runtime** | 1481.6s |

**Run 4: 70B RunAI baseline (`70b-baseline-v4`, 64GB)**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-70b-baseline-v4` |
| **TPU** | v5p-8, TP=4 |
| **Memory** | 64 GiB |
| **RunAI download** | 185.08s, 131.4 GiB, 727.1 MiB/s |
| **HBM** | 32.86 GiB/chip (131.44 GiB total) |
| **Total runtime** | 481.4s |

**Run 5: 70B RunAI at 32GB (`70b-baseline-32g`) — OOM**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-70b-baseline-32g` |
| **Result** | OOM killed (container exceeded memory limit) |

**Run 6: 70B fsspec at 32GB (`70b-fsspec-32g`) — succeeded**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-70b-fsspec-32g` |
| **Memory** | 32 GiB |
| **All shards** | 723 tensors, 131.4 GiB in 1239.2s (109 MiB/s avg) |
| **Peak RSS** | 15546 MB |
| **Total runtime** | 1446.6s |

### Run 7: 8B fsspec concurrent (`fsspec-v10`)

After porting Levanter's async concurrency pattern (ThreadPoolExecutor +
asyncio.gather + Semaphore, `FSSPEC_MAX_CONCURRENT_CHUNKS=8`).

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-smoke-fsspec-v10` |
| **Wheel** | `marin-d634f760` |
| **All shards** | 291 tensors, 15.0 GiB in 90.1s (170 MiB/s avg) |
| **Peak RSS** | 17458 MB |
| **Total runtime** | 310.9s |

### Run 8: 70B fsspec concurrent (`70b-fsspec-v4`, 32GB)

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-70b-fsspec-v4` |
| **Memory** | 32 GiB |
| **All shards** | 723 tensors, 131.4 GiB in 688.3s (196 MiB/s avg) |
| **Peak RSS** | not measured (estimate ~17.5 GiB based on 8B pattern) |
| **Total runtime** | 901.4s |

### Run 9-10: Controlled A/B — MODEL_IMPL_TYPE only

Same model, same TPU worker, same `load_format=runai_streamer`, same memory
(64GB). Only `MODEL_IMPL_TYPE` differs. **Purpose: isolate whether RunAI
transport speed depends on the model implementation path.**

**Run 9: MODEL_IMPL_TYPE=vllm (`vllm-ab-impl-vllm`)**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-ab-impl-vllm` |
| **Code path** | `get_vllm_model()` (PyTorch wrapper) |
| **RunAI download** | 12.95s, 15.0 GiB, 1.2 GiB/s |
| **HBM** | 14.99 GiB |
| **Total runtime** | 366.0s |

**Run 10: MODEL_IMPL_TYPE=auto (`vllm-ab-impl-auto`)**

| Field | Value |
|-------|-------|
| **Job** | `/ahmed/vllm-ab-impl-auto` |
| **Code path** | `get_flax_model()` (JAX/flax_nnx) |
| **RunAI download** | 16.35s, 15.0 GiB, 936.7 MiB/s |
| **HBM** | 14.96 GiB |
| **Total runtime** | 150.8s |

**Finding:** In this same-worker back-to-back test, RunAI download speed was
similar on both paths (~1 GiB/s). The total runtime difference (366s vs 151s)
appears to be due to post-download processing rather than RunAI transport.
This finding is subject to the cache warmth confound noted above.

**Confound:** Both ran on the same worker back-to-back. Cache warmth could
explain the high download speeds on both. The historical 42.7 MiB/s (from
the March 18 subprocess comparison) and the original 53 MiB/s may have been
measured under different caching conditions.

---

## Summary Tables

### 8B (Llama 3.1 8B Instruct)

| Run | Path | Weight Loading | Total | Memory | Notes |
|-----|------|---------------|-------|--------|-------|
| Original in-process (Mar 15) | fsspec + sync_weights | 95.1s (162 MiB/s) | ~26 min | — | Includes XLA compilation |
| Stress test (Mar 18) | fsspec + sync_weights | 119.2s (128 MiB/s) | — | 24 GB | Queue-based HTTP |
| Subprocess baseline (Mar 18) | RunAI + PyTorch path | 359s (42.7 MiB/s) | 34 min startup | 64 GB | MODEL_IMPL_TYPE=vllm |
| fsspec-v9 (Mar 22) | fork fsspec sequential | 126.3s (121 MiB/s) | 342.2s | 32 GB | First fork-native |
| baseline-v7 (Mar 22) | RunAI + JAX path | 7.0s (2.1 GiB/s) | 135.8s | 64 GB | Same worker as v9 |
| fsspec-v10 (Mar 22) | fork fsspec concurrent | 90.1s (170 MiB/s) | 310.9s | 32 GB | 8 concurrent chunks |
| A/B vllm (Mar 22) | RunAI + PyTorch path | 13.0s (1.2 GiB/s) | 366.0s | 64 GB | Controlled A/B |
| A/B auto (Mar 22) | RunAI + JAX path | 16.4s (937 MiB/s) | 150.8s | 64 GB | Controlled A/B |

### 70B (Llama 3.3 70B Instruct)

| Run | Path | Weight Loading | Total | Memory | Notes |
|-----|------|---------------|-------|--------|-------|
| Original in-process (Mar 16) | fsspec + sync_weights | 876.1s (154 MiB/s) | 29 min | 400 GB | All-at-once |
| Shard-streaming (Mar 16) | fsspec + sync_weights | — | — | 24 GB | Pipeline 1379.8s |
| 70b-fsspec-v3 (Mar 22) | fork fsspec sequential | 1173.7s (115 MiB/s) | 1481.6s | 64 GB | |
| 70b-baseline-v4 (Mar 22) | RunAI + JAX path | 185.1s (727 MiB/s) | 481.4s | 64 GB | |
| 70b-baseline-32g (Mar 22) | RunAI + JAX path | OOM | OOM | 32 GB | Container killed |
| 70b-fsspec-32g (Mar 22) | fork fsspec sequential | 1239.2s (109 MiB/s) | 1446.6s | 32 GB | |
| 70b-fsspec-v4 (Mar 22) | fork fsspec concurrent | 688.3s (196 MiB/s) | 901.4s | 32 GB | 8 concurrent chunks |

---

## What Is Solid

- The original in-process fsspec + sync_weights path produced real startup
  improvements and real memory savings. Those results are valid.
- The fsspec streaming path enables 70B at 32GB host memory where RunAI OOMs.
  This is the strongest unique value.
- In the March 22 same-worker A/B, RunAI transport did not appear to be the
  dominant bottleneck on the PyTorch path. Download speeds were similar (~1
  GiB/s) on both paths; the runtime difference was in post-download
  processing. This finding is conditional on cache warmth (same worker,
  same model, back-to-back runs — see confound in Open Questions).
- Concurrent chunk downloads (Levanter pattern) improved fsspec throughput
  from 109-121 MiB/s to 170-196 MiB/s (1.4-1.7x).

## What Still Needs Proof

1. **Phase-timed breakdown of `get_vllm_model()` vs `get_flax_model()`.**
   Instrumentation committed (fork `4abb68f4`) but not yet run. Need to
   identify which sub-phase (vllm_get_model, shard_model_to_tpu, jit_step_func,
   etc.) accounts for the 366s vs 151s total difference.

2. **Cold-cache RunAI download speed.** All March 22 tests ran on workers that
   may have had warm GCS caches. The historical 42.7 MiB/s from March 18
   (subprocess comparison) is the closest to a cold-cache RunAI measurement.
   To control for cache without cross-region reads (which are expensive and
   confound throughput measurements), options are:
   - Use a model checkpoint that hasn't been accessed recently on that worker
   - Wait long enough between runs for any page cache to be evicted
   - Compare the RunAI streamer's own timing log (which reports download
     speed) against the end-to-end phase timing to see if download dominates

3. **Reconcile the 53 MiB/s number.** The original issue says "RunAI downloads
   at 53 MiB/s" and the stress test comparison says 42.7 MiB/s. The A/B test
   shows ~1 GiB/s. These could be:
   - Different metrics (download vs pipeline vs end-to-end)
   - Different cache states (cold vs warm)
   - Different RunAI versions or configurations
   Until isolated, the "53 MiB/s" baseline should be treated as uncertain.

4. **LLM.generate() mode test.** The original experiment used direct
   `LLM.generate()`, not `vllm serve`. Current benchmarks all go through
   `vllm serve` (subprocess). A direct LLM.generate() comparison would be
   closer to the original setup.

## Same-Region Experiment Plan (March 22, late session)

All prior benchmarks were cross-region (model in us-central1, workers in
us-east1/us-east5). This invalidates all download speed measurements. We now
have the 8B model copied to us-east1. Re-running with same-region colocation.

### Hypotheses to test

**H1: The dominant bottleneck on the PyTorch (`MODEL_IMPL_TYPE=vllm`) path is
post-download processing, not network transport.**
- Falsified if: phase timing shows RunAI download dominates total time
- Supported if: `vllm_get_model()` or `shard_model_to_tpu()` accounts for
  most of the time difference vs the JAX path

**H2: Direct `LLM.generate()` on the corrected JAX path (MODEL_IMPL_TYPE=auto,
load_format=runai_streamer) is fast — comparable to `vllm serve` JAX path.**
- Falsified if: direct LLM.generate() is much slower than vllm serve JAX path
- Supported if: total times are similar, meaning the original pain was mostly
  the old subprocess/PyTorch baseline, not the LLM.generate() surface itself

**H3: Same-region download speeds are materially different from the cross-region
numbers we measured earlier.**
- This is not a hypothesis we're testing directly, but we will observe it.
  If RunAI reports similar ~1 GiB/s same-region as it did cross-region, cache
  warmth is less of a concern. If it's much faster, the cross-region numbers
  were bandwidth-limited.

### Experiment batch 1 (priority: root-speed question)

All use model `gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
on v6e-4 workers (us-east1-d). Instrumented wheel `marin-4abb68f4`.

| # | Job name | Mode | MODEL_IMPL_TYPE | Weight loader | Memory | Tests |
|---|----------|------|-----------------|---------------|--------|-------|
| 1 | `vllm-sr-phase-vllm` | `vllm serve` | `vllm` | RunAI (PyTorch path) | 64GB | H1 |
| 2 | `vllm-sr-phase-jax` | `vllm serve` | `auto` | RunAI (JAX path) | 64GB | H1 |
| 3 | `vllm-sr-llm-direct` | `LLM.generate()` | `auto` | RunAI (JAX path) | 64GB | H2 |

Phase instrumentation in runs 1 and 2 will log:
- **PyTorch path:** VllmModelWrapper init → vllm_get_model → shard_model_to_tpu
  → jax_view → jit_step_func → jit_compute_logits
- **JAX path:** arch lookup → _get_nnx_model → eval_shape → load_weights →
  create_jit_model → nnx.split

Run 3 uses direct `LLM.generate()` with `MODEL_IMPL_TYPE=auto` and
`load_format=runai_streamer` (no fsspec, no abstract_load). This matches the
current corrected default path but in the same mode as the original complaint.

### Experiment batch 2 (after inspecting batch 1 results)

| # | Job name | Mode | Weight loader | Memory | Tests |
|---|----------|------|---------------|--------|-------|
| 4 | `vllm-sr-fsspec-conc` | `vllm serve` | fsspec concurrent | 32GB | memory fit |

This validates that fsspec still works at 32GB same-region. It is a memory
test, not a speed comparison against runs 1-2 (different memory budget).

### What we will learn

- **From runs 1 vs 2:** Which sub-phase inside `get_vllm_model()` is slow.
  This is the root-speed question.
- **From run 3 vs 2:** Whether the `vllm serve` subprocess adds meaningful
  overhead vs direct `LLM.generate()`. If they're similar, the original pain
  was the PyTorch path, not the serving surface.
- **From run 4:** Whether fsspec at 32GB still works same-region (memory story).

### Same-region batch 1 results

All use model `gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`.
Instrumented wheel `marin-4abb68f4`.

**Run 1: PyTorch path (`vllm-sr-phase-vllm`)**
Worker: `us-east1-d` (same region as model ✅)

| Phase | Time |
|-------|------|
| VllmModelWrapper init | 0.7s |
| RunAI download | 5.92s (15.0 GiB at **2.5 GiB/s**) |
| vllm_get_model (includes RunAI download) | 17.0s |
| model wrapping | 0.0s |
| shard_model_to_tpu | 0.4s |
| jax_view | 0.0s |
| **load_weights TOTAL** | **17.4s** |
| jit_step_func | 0.0s |
| jit_compute_logits | 0.0s |
| **get_vllm_model TOTAL** | **18.1s** |
| **Smoke test total** | **136.8s** |

**Run 2: JAX path (`vllm-sr-phase-jax`)**
Worker: `us-east5-b` (**cross-region from model ⚠️** — landed on different worker)

| Phase | Time |
|-------|------|
| arch lookup | 0.5s |
| eval_shape | 2.0s |
| RunAI download | 7.63s (15.0 GiB at **2.0 GiB/s**) |
| load_weights (includes RunAI download) | 9.3s |
| create_jit_model | 0.9s |
| **abstract_load TOTAL** | **12.2s** |
| nnx.split | 0.0s |
| **get_flax_model TOTAL** | **12.7s** |
| **Smoke test total** | **226.4s** |

**Run 3: Direct LLM.generate() (`vllm-sr-llm-direct`)**
Worker: `europe-west4-a` (**cross-region from model ⚠️** — Iris sent it to EU!)

| Phase | Time |
|-------|------|
| eval_shape | 2.5s |
| RunAI download | 12.58s (15.0 GiB at **1.2 GiB/s**) |
| load_weights (includes RunAI) | 13.9s |
| create_jit_model | 0.8s |
| **abstract_load TOTAL** | **17.2s** |
| **LLM() constructor TOTAL** | **321.8s** |
| generate | 2.1s (128 tokens, 59.8 tok/s) |
| **Overall TOTAL** | **324.0s** |

### Analysis of batch 1

**H1 (bottleneck is post-download):** Partially supported, partially surprising.
- On the PyTorch path, `vllm_get_model` took 17.0s. RunAI download was 5.92s
  of that. So ~11s was model construction/initialization — not trivial but also
  not the dominant bottleneck we expected.
- `shard_model_to_tpu` was only 0.4s — **not** the bottleneck.
- The total `get_vllm_model` was 18.1s. The total smoke test was 136.8s.
  **Where are the other ~119s?** This is likely XLA compilation, KV cache init,
  and vllm serve startup — outside our instrumented region.

**H2 (direct LLM.generate() is fast):** Supported with caveats.
- The `LLM()` constructor was 321.8s, but model loading (`abstract_load`) was
  only 17.2s of that. The remaining ~305s is XLA compilation + KV cache init.
- Generation itself was fast: 2.1s for 128 tokens (59.8 tok/s).
- **Confound:** This ran on europe-west4 reading from us-east1 — cross-region.
  The 1.2 GiB/s RunAI speed may be lower than same-region.

**Key finding: model loading is NOT the bottleneck for either path.**
- PyTorch: 18.1s model loading out of 136.8s total (13%)
- JAX: 12.7s model loading out of 226.4s total (6%)
- Direct: 17.2s model loading out of 324.0s total (5%)

The dominant cost in all cases is XLA compilation + KV cache initialization +
vllm serve startup overhead — NOT weight download or model materialization.

**Confound: runs 2 and 3 were cross-region.** Iris placed them on us-east5-b
and europe-west4-a respectively, not us-east1-d. We cannot control Iris worker
placement per-job without zone pinning. The RunAI download speeds (2.0 and
1.2 GiB/s) may be slower than true same-region. Run 1 was same-region (2.5 GiB/s).

### Batch 1 conclusions

1. **Model loading is fast on both paths.** PyTorch: 18s. JAX: 13s. Neither
   is the dominant cost in total runtime (137s, 226s, 324s respectively).
2. **RunAI download is fast (~2 GiB/s same-region).** Not the bottleneck.
3. **The dominant cost is XLA compilation + KV cache init + server startup** —
   all happening outside our instrumented region.
4. **Confound: runs 2 and 3 landed cross-region.** Iris placed them on
   us-east5-b and europe-west4-a. We added `regional_model_path.py` to
   auto-detect region and pick the right model bucket going forward.

### Batch 1 rerun: direct LLM.generate() with auto-regional resolution

| # | Job name | Mode | Region handling | Status |
|---|----------|------|----------------|--------|
| 3b | `vllm-sr-llm-direct-v2` | `LLM.generate()` | auto-resolve via `regional_model_path.py` | submitted |

Uses model name `meta-llama--Llama-3-1-8B-Instruct--0e9e39f` (not a full
gs:// path). The script detects region from GCE metadata and resolves to
the correct bucket. Logs will print `Detected region:` and `Resolved model
path:` for validation.

### Unresolved questions deferred to later

- Reconciling the historical 53 MiB/s / 42.7 MiB/s numbers — need to map
  each to its exact timing boundary (raw download vs weight pipeline vs total)
- 70B same-region validation — needs v5p-8 in us-central1 (where 70B model
  exists) or use gcsfuse_mount paths in us-east1/us-east5
- Run 1 (PyTorch path) was same-region ✅. Runs 2 and 3 need rerunning
  same-region if we want controlled comparison of download speeds

## Constraints

- **NEVER read model weights across GCS regions.** Cross-region egress is a
  major cost driver for this project. All benchmarks must run in the same
  region as the model checkpoint. Model and compute MUST be co-located.
- **Do not use storage transfer service** to move files between regions
  unless the user explicitly approves the cost.

## CRITICAL: All prior benchmarks were cross-region reads

**Every run in this logbook read from `gs://marin-us-central1/` but ran on
workers in `us-east5-b` or `us-east1-d`.** This is cross-region and:
1. Adds latency and reduces throughput compared to same-region
2. Incurs egress costs
3. Confounds all download speed measurements

### Available models by region

| Region | Bucket | Llama models available |
|--------|--------|----------------------|
| us-central1 | `gs://marin-us-central1/models/` | Llama 3.1 8B, Llama 3.3 70B |
| us-east1 | `gs://marin-us-east1/models/` | **Llama 3.1 8B** (copied 2026-03-22) |
| us-east5 | `gs://marin-us-east5/models/` | Llama 3.2 1B only |

### Available v6e-4 workers by region (from job history)

| Zone | Workers seen |
|------|-------------|
| us-east1-d | `marin-tpu_v6e_4-us-east1-d-*` |
| us-east5-b | `marin-tpu_v6e_4-us-east5-b-*` |
| us-central2-b | Pending (no workers matched) |

### Fix required before next benchmark

Before running any more benchmarks, copy the test model to the same region
as the compute. For example:
```bash
# Copy 8B model to us-east1 (where v6e-4 workers are)
gcloud storage cp -r \
  gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f
```
Then use `gs://marin-us-east1/models/...` in all benchmark commands.

Track which models have been copied where in the table above.
