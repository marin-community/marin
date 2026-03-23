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

## Post-Mortem: How the Original Comparison Got Confounded

### What we thought we were comparing

> RunAI weight download (53 MiB/s) vs fsspec weight download (154 MiB/s) → "3x faster"

### What we actually compared

Two completely different things with different timing boundaries, different
code paths, and different measurement methods:

**Our "fast path" (in-process, commit `dbeddbd45`):**
- Direct `LLM(load_format="dummy")` — no RunAI involved at all
- Levanter `read_safetensors_fsspec()` for downloading
- `sync_weights()` for injection
- **Explicitly timed** our own weight pipeline phases

**The "baseline" (subprocess `vllm serve`):**
- `vllm serve` launched as subprocess
- `load_format=runai_streamer`
- `MODEL_IMPL_TYPE=vllm` forced by Marin (commit `3f1420e8d`) → PyTorch path
- **Timing derived from server startup logs**, not a clean download timer

### The two confounds

**1. Different timing boundaries.**
The fast path explicitly timed `download → reshape → convert → inject`.
The baseline's "weight loading" number was derived from subprocess startup
logs — it measured `vllm_get_model()` end-to-end, which includes RunAI
download + PyTorch model construction + CPU tensor materialization + whatever
else happens inside vLLM's model loader. These are not the same measurement.

**2. Different model implementation paths.**
The fast path used `load_format="dummy"` + external injection (RunAI never
ran). The baseline used the PyTorch `get_vllm_model()` path because Marin
forced `MODEL_IMPL_TYPE=vllm`. We were not comparing RunAI vs fsspec
download speed. We were comparing:
- (fsspec download + reshape + NNX convert + sync_weights)
- vs (RunAI download + PyTorch model construction + CPU materialization +
  shard_model_to_tpu + everything else in the PyTorch startup path)

### Forensic provenance of the "42.7 MiB/s" and "53 MiB/s" numbers

Both numbers were post-hoc calculations, not metrics produced by any code or
tool. Neither RunAI nor the baseline script ever output a "MiB/s" figure.

#### "42.7 MiB/s" — traced as far as possible

**Confirmed (artifacts checked):**
1. The baseline script (`experiments/inference/exp_vllm_baseline_comparison.py`)
   does NOT compute weight-loading throughput. It records only
   `server_startup_sec`, `total_time_sec`, `aggregate_tokens_per_sec`, and
   latency percentiles. No MiB/s metric exists in the code.
2. The GCS artifact (`baseline_results_20260318-194349.json`) contained
   `server_startup_sec: 2052.6`. No weight-loading metric, no MiB/s field.
   (This artifact is no longer in the repo — it was a transient GCS upload.)
3. "42.7 MiB/s" was a post-hoc division: `15 GiB / 359s = 42.7 MiB/s`.
   No code ever computed this number.

**Unknown (evidence expired):**
4. The exact source of "359s" cannot be confirmed. The original Iris job logs
   for `/ahmed/vllm-baseline-8b-v3` have expired — only Iris tunnel metadata
   remains. The vLLM subprocess output went to temp files on the worker
   (before the tee fix), so Iris never captured it.
5. "359s" was most likely read from the vLLM subprocess stdout at the time
   by the agent that posted the issue comment. But we cannot cite the exact
   log line because those logs no longer exist.

#### "53 MiB/s" / "51.5 MiB/s" — partially traced

**Confirmed (artifacts checked):**
1. Documented in `FAST_VLLM_LOAD_TPU.md` appendix (commit `dbeddbd45`,
   deleted in commit `aa7d817f1`). Exact text:
   > Tested `--model-loader-extra-config '{"concurrency": 16, "memory_limit":
   > 5368709120}'` on 8B model. Result: 51.5 MiB/s
2. The same document's commit message (commit `dbeddbd45`) states:
   "Replace slow RunAI streamer (53 MiB/s)". The number was treated as a
   known fact from that point forward.
3. RunAI's progress bar shows `it/s` (tensors/sec), NOT MiB/s. Example from
   our own logs:
   `Loading safetensors using Runai Model Streamer: 100% | 291/291 [00:05, 50.85it/s]`
4. RunAI's summary log shows GiB/s, NOT MiB/s:
   `[RunAI Streamer] Overall time to stream 15.0 GiB: 5.92s, 2.5 GiB/s`
5. Neither RunAI output format produces a "MiB/s" number directly.

**Unknown (evidence unavailable):**
6. The exact measurement method used by the Codex agent that wrote
   `FAST_VLLM_LOAD_TPU.md`. The session logs are not available. "51.5 MiB/s"
   was most likely hand-computed: `model_size / some_phase_duration`. But we
   cannot prove which duration was used (RunAI-reported download time,
   `vllm_get_model` duration, or broader subprocess startup phase).

#### What the numbers were NOT

Both numbers were NOT:
- Direct RunAI transport speed (which is 2.1–2.5 GiB/s same-region)
- A metric produced by any code in the repo
- The RunAI progress bar rate (which shows it/s, i.e., tensors/sec)
- The RunAI summary throughput (which shows GiB/s)

#### Reconciliation with current measurements

Our same-region phase-timed measurements show:
- RunAI raw download: 5.92–7.21s for 15 GiB = **2.1–2.5 GiB/s**
- `vllm_get_model()` total: 15.0–17.0s (includes download + construction)
- If you divide model size by `vllm_get_model()` time: 15 GiB / 17s ≈ 900 MiB/s

Even the broadest measurement (full `get_vllm_model` including JIT setup)
gives ~900 MiB/s, not 53 MiB/s. The original 53 MiB/s cannot be reproduced
under current same-region conditions. The most likely explanations (none
confirmed, all inferred) are: different infrastructure at the time, cross-region
reads, cold GCS cache, or a timing window that included XLA compilation.

### Did the queue-based HTTP wrapper affect weight loading?

No. The queue wrapper made **serving throughput** worse (serialized
`LLM.generate()` calls, hurt request batching). But model loading happens
before the server becomes ready. The queue did not slow RunAI downloads.

### What was real vs what was overstated

**Real:**
- The in-process fsspec + sync_weights path produced working weight injection
- Shard-streaming reduced host RAM from 131 GiB to ~15 GiB for 70B
- The fsspec path enables 70B at 32GB where RunAI OOMs
- The old subprocess PyTorch path was genuinely slower end-to-end (but for
  reasons beyond just weight download speed)

**Overstated:**
- "RunAI is slow at 53 MiB/s" — RunAI transport is actually 2.1+ GiB/s
- "3x faster weight loading" — we compared different timing boundaries
- "fsspec download speed is the key improvement" — the speed improvement was
  mostly from bypassing the PyTorch path overhead, not from faster downloads

### Correct mental model

- RunAI transport itself is fast (~2 GiB/s same-region)
- fsspec's strongest durable value is **bounded host RAM** (15-17 GiB peak)
- The real cold-start cost is post-load: XLA compilation, KV cache init,
  startup path differences
- Model loading (either path) is 11-18s for 8B — not the bottleneck

### Forensic trail

Artifacts checked during the March 22 provenance investigation, for
reproducibility:

| Artifact | Location | Status | What it confirmed |
|----------|----------|--------|-------------------|
| `exp_vllm_baseline_comparison.py` | `experiments/inference/` | Exists | Records `server_startup_sec`, `total_time_sec`, `aggregate_tokens_per_sec`, latency percentiles. Does NOT compute MiB/s. |
| `baseline_results_20260318-194349.json` | Was a transient GCS upload | Not in repo | Contained `server_startup_sec: 2052.6`. No weight-loading metric. |
| `FAST_VLLM_LOAD_TPU.md` | Deleted in commit `aa7d817f1` | Recoverable via `git show aa7d817f1^:FAST_VLLM_LOAD_TPU.md` | Contains the "51.5 MiB/s" appendix entry. No description of measurement method. |
| Commit `dbeddbd45` message | git history | Exists | States "Replace slow RunAI streamer (53 MiB/s)" — treated the number as established fact. |
| Iris job logs for `/ahmed/vllm-baseline-8b-v3` | Iris | Expired | Would have contained the vLLM subprocess stdout with the "359s" timing. Not recoverable. |
| Codex agent session logs | External | Not available | Would have documented how "51.5 MiB/s" was computed. |
| RunAI progress bar format | Observed in current logs | Confirmed | Shows `it/s` (tensors/sec), not MiB/s. |
| RunAI summary log format | Observed in current logs | Confirmed | Shows GiB/s, not MiB/s. |
| Same-region proof runs P1/P2/A | This logbook, "Confound proven" section | Complete | RunAI transport is 2.1–2.5 GiB/s; model load is 11–18s on all paths. |

**Evidence gap:** We can confirm what the numbers were NOT (not RunAI output,
not code-computed, not reproducible under current conditions). We cannot
confirm exactly how they were computed because the Codex session logs and the
Iris job stdout are both unavailable. The most parsimonious explanation is a
hand division of model size by a broad timing window that included more than
raw network transport.

---

## Historical Metrics (from issue #3768 and comments)

These are the numbers as originally reported. Metric boundaries were not
clear at the time and are now understood to have been confounded (see
post-mortem above).

### Original issue body (filed ~March 12-15)

| Claimed metric | Value | What it actually measured |
|----------------|-------|--------------------------|
| "RunAI downloads via single-threaded HTTP" | 53 MiB/s | Documented in `FAST_VLLM_LOAD_TPU.md` appendix (commit `dbeddbd45`, deleted in `aa7d817f1`). Measurement method undocumented — not a RunAI-reported metric (RunAI reports it/s and GiB/s, not MiB/s). Most likely a post-hoc division of model size by an unspecified phase duration. Codex session logs unavailable. |
| "RunAI concurrency (up to 16) no effect" | 51.5 MiB/s | Same document, same unknown measurement method. The concurrency parameter may not have helped because download was already fast and the observed rate reflected a broader phase than raw transport. |
| "70B weight download at 53 MiB/s" | 41 min | Projected from 53 MiB/s assumption: 131 GiB / 53 MiB/s ≈ 41 min |
| "Cold start exceeds 1 hour" | >60 min | Projected: 41 min download + 20-30 min XLA compilation |

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

**Key:** The subprocess baseline weight loading was reported as 359s at
42.7 MiB/s. This was a post-hoc division (`15 GiB / 359s`). The source of
"359s" is expired Iris job logs for `/ahmed/vllm-baseline-8b-v3` — the vLLM
subprocess stdout went to temp files on the worker (before the tee fix), so
Iris never captured it. The baseline comparison script
(`exp_vllm_baseline_comparison.py`) records only `server_startup_sec` and
inference metrics, not weight-loading throughput. The subprocess path used
`MODEL_IMPL_TYPE=vllm` (commit `3f1420e8d`) which forces the PyTorch path.

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

## What Still Needs Proof: The Confound Proof Matrix

The post-mortem hypothesis is now specific:

1. The original fast path (`LLM(load_format="dummy")` + fsspec +
   `sync_weights()`) was timed with a **direct in-process weight pipeline**
   boundary.
2. The old RunAI baseline was timed with a **subprocess server-startup**
   boundary.
3. The old subprocess baseline also forced `MODEL_IMPL_TYPE=vllm`, which sent
   Llama through the PyTorch TPU wrapper path instead of the JAX/flax_nnx path.

To prove that cleanly, we do **not** need to benchmark old commits. We only
need to reproduce the old behavior on current code with current instrumentation.

### Archaeological reference (read-only, do not run)

| Commit | What to inspect | Why it matters |
|--------|-----------------|----------------|
| `dbeddbd45` | `lib/marin/src/marin/inference/vllm_inprocess.py`, `experiments/inference/exp_vllm_inprocess_direct.py` | Original Marin-side fast path using `dummy` + fsspec + `sync_weights()` |
| `3f1420e8d` | `lib/marin/src/marin/inference/vllm_server.py` | Subprocess path hardcoded `MODEL_IMPL_TYPE=vllm` |
| `8f83e3692` | `lib/marin/src/marin/inference/vllm_server.py` | Later changed the default to `MODEL_IMPL_TYPE=auto` |

### Already-completed proof runs

All same-region, same 8B model in `us-east1`, using the current instrumented
wheel.

| ID | Job | Surface | Forced path | What it gives us |
|----|-----|---------|-------------|------------------|
| P1 | `vllm-sr-phase-vllm` | `vllm serve` | PyTorch (`MODEL_IMPL_TYPE=vllm`) | Same-region subprocess PyTorch reference |
| P2 | `vllm-sr-llm-direct-v4` | direct `LLM.generate()` | JAX (`MODEL_IMPL_TYPE=auto`) | Same-region direct JAX reference |

These two runs already prove two important facts:

- RunAI transport is fast in both corrected setups (about `2.1-2.5 GiB/s`).
- Post-load startup dominates total wall time.

What is still missing is a **direct `LLM.generate()` run on the old PyTorch
path**. That is the one missing point needed to separate:

- implementation-path effect (`PyTorch` vs `JAX`)
- from surface/timing-boundary effect (`direct LLM()` vs subprocess `vllm serve`)

### Minimal proof set remaining: one run

**Run A: forced old PyTorch path, direct `LLM.generate()`**

Purpose:
- Reproduce the old `MODEL_IMPL_TYPE=vllm` behavior
- Use the **same direct `LLM()` timing boundary** as the fast-path experiments
- Avoid subprocess/server-readiness overhead

If this run loads in the low tens of seconds, then the old `42.7-53 MiB/s`
story cannot have been raw weight transport. It must have included subprocess
startup and/or broader server initialization overhead.

Exact command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v6e-4 --memory 64GB --zone us-east1-d \
  --extra tpu --extra vllm \
  --job-name confound-llm-vllm \
  --no-wait \
  -e MODEL_IMPL_TYPE vllm \
  -- python experiments/inference/exp_vllm_llm_generate_direct.py \
    --model gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
    --max-model-len 4096 \
    --load-format runai_streamer
```

Expected output shape from the script:
- `[config] MODEL_IMPL_TYPE=vllm`
- `[phase] LLM() constructor: ...s`
- `[phase] generate: ...s`
- `[phase] TOTAL: ...s`

### Interpretation matrix

After Run A completes, use these comparisons only:

| Comparison | Controlled variables | Question answered |
|------------|----------------------|-------------------|
| `A` vs `P2` | Same surface (`LLM.generate()`), same same-region model, different impl path | How much does PyTorch-vs-JAX change direct model init? |
| `P1` vs `A` | Same impl path (PyTorch), same same-region model, different surface | How much overhead comes from subprocess/server startup vs direct `LLM()`? |

Read the results as follows:

1. If `A` model load is also in the low tens of seconds:
   - raw RunAI transport on the old PyTorch path was **not** the main problem
   - the old `42.7-53 MiB/s` figure was a broader startup metric, not pure download

2. If `A` model load is much larger than `P2` model load:
   - the implementation path matters
   - PyTorch TPU model construction adds real overhead
   - but this still does **not** imply raw RunAI network transport was slow

3. If `P1` total is much larger than `A` total while their model-load phases are similar:
   - the subprocess serving surface contributed substantial overhead
   - this is the strongest proof of the original timing-boundary confound

4. If `A` itself is surprisingly slow:
   - inspect the `LLM()` constructor logs before drawing any conclusion about transport
   - the likely causes are model materialization and compile/startup work inside the PyTorch path

### What not to do

- Do **not** schedule another subprocess smoke test for the proof matrix.
  `P1` already gives the same-region subprocess PyTorch reference.
- Do **not** use `vllm_smoke_test.py` to reproduce the old baseline again.
  That script is useful for smoke tests, but it is not the cleanest tool for
  proving the historical timing-boundary confound.
- Do **not** benchmark old commits for proof. Use current scripts with forced
  env vars so the only changed variable is the one under test.

### Optional follow-up only if needed

If someone later asks to reproduce the **historical measurement boundary**
exactly, the right script is:

- `experiments/inference/exp_vllm_baseline_comparison.py`

That script times subprocess server readiness more directly than
`vllm_smoke_test.py`. This is **not required** for the confound proof above.

### Run A results: `confound-llm-vllm` ✅

Worker: `us-east1-d`. Model: `gs://marin-us-east1/...` (same-region).

| Phase | Time |
|-------|------|
| VllmModelWrapper init | 0.3s |
| RunAI download | 5.96s (15.0 GiB at **2.5 GiB/s**) |
| vllm_get_model | 15.0s |
| shard_model_to_tpu | 0.4s |
| **load_weights TOTAL** | **15.3s** |
| **get_vllm_model TOTAL** | **15.7s** |
| **LLM() constructor** | **302.7s** |
| generate | 2.1s (61.2 tok/s) |
| **TOTAL** | **304.8s** |

### Interpretation

**A vs P2** (PyTorch vs JAX, same `LLM.generate()` surface):

- Model load: 15.7s (PyTorch) vs 11.2s (JAX) — both low tens of seconds
- RunAI download: 2.5 GiB/s vs 2.1 GiB/s — both fast
- **Conclusion: raw RunAI transport was not the problem on the old PyTorch
  path.** The old 42.7-53 MiB/s figure cannot have been pure download speed.

**P1 vs A** (subprocess `vllm serve` vs direct `LLM()`, same PyTorch path):

- Model load: 18.1s (P1) vs 15.7s (A) — nearly identical
- Total: 136.8s (P1) vs 304.8s (A) — subprocess is actually faster
- **Conclusion: the subprocess surface did not add overhead to model loading.**
  The subprocess is faster total, likely because `vllm serve` has different
  XLA compilation/caching behavior.

### Confound proven

The old 42.7–53 MiB/s number was not raw RunAI transport speed (which is
2.1–2.5 GiB/s same-region). It was not even model loading time (which is
11–18s on all paths tested). Neither number was produced by any code in the
repo or by RunAI's own output (which reports it/s and GiB/s, not MiB/s).
Both were post-hoc divisions of model size by an unspecified timing window.
The exact computation method is not recoverable — the Codex session logs
and the Iris job stdout that would document it have expired (see Forensic
Trail above). What we can prove: no same-region measurement on current
infrastructure reproduces anything near 53 MiB/s for raw weight transport.

The original fsspec prototype's speed improvement was real in the sense that
it produced a faster end-to-end startup. But the framing — "RunAI downloads
at 53 MiB/s, fsspec downloads at 154 MiB/s, therefore 3x faster" — was
comparing different timing boundaries and different code paths. In the
same-region reruns, both transport paths are far above the historical
53 MiB/s claim (~2 GiB/s for RunAI, ~200 MiB/s for fsspec), so raw
transport was not the bottleneck in those measurements.

**The fsspec path's durable value is host memory behavior, not download speed.**

### All proof runs complete

- [x] P1: `vllm-sr-phase-vllm` — subprocess PyTorch (model load 18.1s)
- [x] P2: `vllm-sr-llm-direct-v4` — direct JAX (model load 11.2s)
- [x] A: `confound-llm-vllm` — direct PyTorch (model load 15.7s)

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

**H2 (direct LLM.generate() is fast):** Initially supported, but the first
measurement was cross-region and is superseded by Run 3b below.
- The first direct run (`vllm-sr-llm-direct`) showed the right qualitative
  shape: model loading was a small fraction of total startup.
- However, that run landed on `europe-west4-a` while reading from `us-east1`,
  so its transport numbers should no longer be used for any conclusion.

**Key finding: model loading is NOT the bottleneck for either path.**
- PyTorch: 18.1s model loading out of 136.8s total (13%)
- JAX: 12.7s model loading out of 226.4s total (6%)
- Direct: 17.2s model loading out of 324.0s total (5%)

The dominant cost in all cases is XLA compilation + KV cache initialization +
vllm serve startup overhead — NOT weight download or model materialization.

**Initial confound:** the first versions of runs 2 and 3 were cross-region.
Those transport numbers are superseded by the same-region reruns later in this
section.

### Batch 1 conclusions

1. **Model loading is fast on both paths.** PyTorch: 18s. JAX: 13s. Neither
   is the dominant cost in total runtime (137s, 226s, 324s respectively).
2. **RunAI download is fast (~2 GiB/s same-region).** Not the bottleneck.
3. **The dominant cost is XLA compilation + KV cache init + server startup** —
   all happening outside our instrumented region.
4. **Initial confound:** the first JAX/direct runs landed cross-region. This
   was fixed by same-region reruns using the `us-east1` model copy and explicit
   zone placement.

### Run 3b: Direct LLM.generate(), same-region ✅ (`vllm-sr-llm-direct-v4`)

Worker: `us-east1-d`. Model: `gs://marin-us-east1/models/...` (same-region confirmed).

| Phase | Time |
|-------|------|
| Region detection + model resolution | <1s |
| eval_shape | 2.5s |
| RunAI download | 7.21s (15.0 GiB at **2.1 GiB/s**) |
| load_weights (includes RunAI) | 7.9s |
| create_jit_model | 0.8s |
| **abstract_load TOTAL** | **11.2s** |
| **LLM() constructor TOTAL** | **288.9s** |
| generate | 2.1s (128 tokens, 60.1 tok/s) |
| **Overall TOTAL** | **291.1s** |

**H2 verdict:** Direct `LLM.generate()` on the corrected JAX path is fast for
model loading (11.2s) but slow for total startup (289s). The gap between
model loading (11.2s) and LLM() constructor (289s) is **~278s of XLA
compilation + KV cache init**. This matches what we saw in runs 1 and 2.

**Same-region RunAI speed: 2.1 GiB/s** — matching the cross-region numbers.
This suggests the earlier ~1 GiB/s measurements were NOT cross-region
bandwidth limited; they were just normal variance or different cache states.

### Updated batch 1 summary (with region annotations)

| Run | Job | Path | Worker zone | Model region | Same-region? | Model load | RunAI speed | Total |
|-----|-----|------|-------------|-------------|-------------|-----------|------------|-------|
| 1 | `vllm-sr-phase-vllm` | PyTorch (vllm serve) | us-east1-d | us-east1 | ✅ | 18.1s | 2.5 GiB/s | 136.8s |
| 2 | `vllm-sr-phase-jax-v2` | JAX (vllm serve) | us-east1-d | us-east1 | ✅ | 11.6s | 2.1 GiB/s | 217.4s |
| 3 | `vllm-sr-llm-direct-v4` | JAX (LLM.generate) | us-east1-d | us-east1 | ✅ | 11.2s | 2.1 GiB/s | 291.1s |

All three runs: same worker (`us-east1-d`), same model region (`us-east1`),
same-region confirmed ✅.

**Key observations:**
- Model loading is 11-18s on all paths. Not the bottleneck.
- RunAI download is 2.1-2.5 GiB/s same-region. Not bandwidth-limited.
- The total runtime gap (137s vs 217s vs 291s) is dominated by what happens
  AFTER model loading: XLA compilation, KV cache init, server startup.
- Run 1 (PyTorch, 137s) is the fastest total. The PyTorch path uses torchax
  which may skip some JAX-specific compilation overhead.
- Run 2 (JAX vllm serve, 217s) vs Run 3 (JAX LLM.generate, 291s): the
  74s gap is likely vllm serve reusing cached XLA compilations or having
  different compilation characteristics from direct LLM().
- Generation itself is fast on both JAX runs: 60 tok/s.

### Unresolved questions deferred to later

- Reconciling the historical 53 MiB/s / 42.7 MiB/s numbers — need to map
  each to its exact timing boundary (raw download vs weight pipeline vs total)
- 70B same-region validation — needs v5p-8 in us-central1 (where the 70B model
  already exists) or an explicitly same-region mirror elsewhere
- The remaining confound-proof run is direct `LLM.generate()` with
  `MODEL_IMPL_TYPE=vllm` (Run A in the proof matrix above)

## Constraints

- **NEVER read model weights across GCS regions.** Cross-region egress is a
  major cost driver for this project. All benchmarks must run in the same
  region as the model checkpoint. Model and compute MUST be co-located.
- **Do not use storage transfer service** to move files between regions
  unless the user explicitly approves the cost.

## Cross-Region Warning

Many of the **early** benchmarks in this logbook were cross-region reads from
`gs://marin-us-central1/` onto workers outside `us-central1`. Those early runs:
1. Added latency and reduced throughput compared to same-region
2. Incurred egress costs
3. Confounded download-speed interpretation

The later 8B reruns using `gs://marin-us-east1/...` and workers in `us-east1-d`
supersede those early transport measurements for 8B.

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

---

## Forensic Repro: What Did RunAI Actually Report on the Old Subprocess Path?

### Motivation

The forensic provenance investigation (above) proved that "53 MiB/s" and
"42.7 MiB/s" were post-hoc calculations, not RunAI-reported metrics. But the
original Iris job logs expired before anyone captured what RunAI *did* report
on that old subprocess path. One rerun of the old subprocess baseline code
(commit `3d1349064`) with the tee fix (commit `3ad8a9269`) closes this gap by
capturing the actual RunAI progress bar and summary lines.

### Setup

- Worktree: `/Users/ahmed/code/marin-old-3d134` at commit `3d1349064`
- Cherry-picked: tee fix from `3ad8a9269` (tees vLLM subprocess stdout/stderr
  to Iris logs)
- Same-region: model in `gs://marin-us-central1`, worker in `us-central1-a`
- Job: `/ahmed/old-baseline-subprocess`
- Submitted: 2026-03-23T01:50:42Z

### What to inspect in logs

1. RunAI progress bar line (expect `it/s`, not `MiB/s`)
2. RunAI summary line (expect `GiB/s`)
3. Server startup time from `exp_vllm_baseline_comparison.py`

### Results: `/ahmed/old-baseline-subprocess` ✅

**Worker:** `marin-tpu_v5p_8-us-central1-a-20260322-2314-4805e27f` (v5p-8, us-central1-a)
**Model:** `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f` (same-region ✅)
**Code:** commit `3d1349064` + tee fix from `3ad8a9269`
**Submitted:** 2026-03-23T01:50:42Z | **Started:** 2026-03-23T01:52:10Z | **Completed:** 2026-03-23T01:55:25Z

#### RunAI output (verbatim from Iris logs)

**Progress bar (stderr):**
```
Loading safetensors using Runai Model Streamer:   0% Completed | 0/291 [00:00<?, ?it/s]
Loading safetensors using Runai Model Streamer:   4% Completed | 12/291 [00:02<00:47, 5.82it/s]
Loading safetensors using Runai Model Streamer:  43% Completed | 126/291 [00:04<00:04, 35.53it/s]
Loading safetensors using Runai Model Streamer: 100% Completed | 291/291 [00:06<00:00, 55.62it/s]
Loading safetensors using Runai Model Streamer: 100% Completed | 291/291 [00:06<00:00, 47.25it/s]
```

**Summary line (stdout):**
```
[RunAI Streamer] Overall time to stream 15.0 GiB of all files to cpu: 6.4s, 2.3 GiB/s
```

**Additional throughput line (stdout):**
```
Read throughput is 4.02 GB per second
```

**Model loader path:**
```
Loading model with MODEL_IMPL_TYPE=vllm
```

#### Key observations

1. **RunAI progress bar reports `it/s` (tensors/sec)**, not MiB/s: `47.25 it/s`
2. **RunAI summary reports `GiB/s`**, not MiB/s: `2.3 GiB/s`
3. **Neither RunAI output format ever produces a MiB/s number**
4. Model loaded via `MODEL_IMPL_TYPE=vllm` (PyTorch path) — same as historical baseline
5. RunAI download: **6.4s for 15.0 GiB = 2.3 GiB/s** (same-region, matches all prior measurements)

#### Baseline results

| Metric | Value |
|--------|-------|
| Server startup | 155.2s |
| Total prompts | 50 |
| Successful | 50 |
| Errors | 0 |
| Total inference time | 16.6s |
| Requests/sec | 3.01 |
| Total tokens | 6400 |
| Aggregate tok/s | 384.8 |
| Latency p50 | 1.242s |
| Latency p95 | 1.740s |
| Avg tok/s per request | 100.7 |

**GCS artifacts:**
- Results: `gs://marin-us-central1/inference/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/stress_test/baseline_results_20260323-015524.json`
- Samples: `gs://marin-us-central1/inference/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/stress_test/baseline_samples_20260323-015524.jsonl`

#### Interpretation

This definitively closes the evidence gap from the forensic provenance section:

1. **RunAI never reported MiB/s.** Its two output formats are `it/s` (tensors/sec
   on the progress bar) and `GiB/s` (on the summary line). The historical
   "53 MiB/s" and "42.7 MiB/s" were not copied from RunAI output.

2. **RunAI transport is fast on the old code path too.** Even at commit `3d1349064`
   with `MODEL_IMPL_TYPE=vllm`, RunAI reports 2.3 GiB/s same-region — matching
   all our current measurements. The old code did not make RunAI slower.

3. **Server startup is 155s, not 2053s.** The historical `server_startup_sec: 2052.6`
   from the March 18 baseline was on a different infrastructure configuration
   (or included XLA compilation that this run avoided). This run's 155s startup
   is consistent with the current PyTorch path measurements (~137s on v6e-4).

4. **The historical "359s at 42.7 MiB/s" remains unrecoverable.** The "359s" was
   reportedly extracted from vLLM subprocess stdout during the March 18 run,
   but those logs expired. On this rerun, RunAI's own timing shows 6.4s for the
   download. Even if "359s" referred to the full server startup (2053s in the
   original, 155s here), `15 GiB / 155s = 99 MiB/s` — still not 42.7 MiB/s.
   The most parsimonious explanation remains that the original timing window
   included XLA compilation or other non-download work.

#### Comparison with historical March 18 baseline

| Metric | March 18 (original) | March 23 (this rerun) |
|--------|--------------------|-----------------------|
| Server startup | 2052.6s (34 min) | 155.2s (2.6 min) |
| RunAI download | unknown (logs expired) | 6.4s at 2.3 GiB/s |
| Aggregate tok/s | 360.0 | 384.8 |
| Latency p50 | — | 1.242s |
| Code commit | `3d1349064` | `3d1349064` + tee fix |
| MODEL_IMPL_TYPE | `vllm` | `vllm` |

The 13x difference in server startup (2053s vs 155s) between the original March 18
run and this rerun — same code, same model, same load_format — suggests the
original run hit XLA compilation or cold-cache conditions that this run did not.
This further supports the hypothesis that the "359s" timing window included
non-download work.

#### Retrospective: how the original agent likely arrived at "53 MiB/s"

**What the agent could NOT see:** Before the tee fix (commit `3ad8a9269`), the
vLLM subprocess stdout/stderr went to temp files on the worker, invisible in
Iris logs. That means the agent never saw RunAI's own output:
```
[RunAI Streamer] Overall time to stream 15.0 GiB of all files to cpu: 6.4s, 2.3 GiB/s
Loading safetensors using Runai Model Streamer: 100% | 291/291 [00:06, 47.25it/s]
```
If comparable RunAI summary lines had been visible in the historical run, they
would have provided a direct transport metric instead of forcing inference from
broader startup timings.

**What the agent DID see:** The baseline script outputs exactly one timing:
`server_startup_sec`. On March 18 that was 2052.6s. The script records no
weight-loading metric, no download duration, no MiB/s — nothing about download
speed.

**The arithmetic:** The `FAST_VLLM_LOAD_TPU.md` doc (commit `dbeddbd45`) says:
> Tested `--model-loader-extra-config '{"concurrency": 16, "memory_limit":
> 5368709120}'` on 8B model. Result: 51.5 MiB/s

Working backwards: `15 GiB / 51.5 MiB/s ≈ 298s`. This is close to the "300s"
in the issue body ("vs RunAI 300s"). Separately, `15 GiB / 359s = 42.7 MiB/s`.
`42.7 MiB/s` is therefore a confirmed post-hoc division. `51.5/53 MiB/s` is
best explained as a similar derived number, but the exact duration used remains
unknown.

**What the rerun shows about the missing ~300s hypothesis:** Our forensic rerun
does not recover the original timing window, but it does bound what that window
could have meant:

- In the historical-style rerun, RunAI reported `6.4s` to stream `15.0 GiB`
  (`2.3 GiB/s`).
- That same rerun still took `155s` for full server startup, so even under warm
  rerun conditions there were roughly `149s` of non-download work after bytes
  arrived.
- The original March 18 run took `2053s` for full server startup, which shows
  that startup time was highly sensitive to conditions outside raw transport
  (for example compile/cache state).

So the missing `~300s` window, whatever its exact boundaries, must have
included substantial non-download work. It cannot have been a direct RunAI
transport metric on the historical-style path we reran.

**Root cause of the error:** A timing boundary problem compounded by invisible
subprocess output. The agent did not have a direct transport metric in the
captured logs, so a much broader startup phase was attributed to "download
time."
