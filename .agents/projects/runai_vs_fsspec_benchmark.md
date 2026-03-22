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

### Where the "53 MiB/s" came from

From `FAST_VLLM_LOAD_TPU.md` (commit `dbeddbd45`), appendix:

> Tested `--model-loader-extra-config '{"concurrency": 16, "memory_limit":
> 5368709120}'` on 8B model. Result: 51.5 MiB/s

A prior agent (Codex) ran `vllm serve` with RunAI and observed 51.5 MiB/s.
But this was likely not pure network transport — it was the RunAI progress bar
rate during `vllm_get_model()`, which includes CPU-side model construction.
The number was then used as if it were a raw download speed to project
70B loading time: 131 GiB / 53 MiB/s = 41 min.

Our new same-region phase-timed measurements show:
- RunAI raw download: 7.08s for 15 GiB = **2.1 GiB/s**
- `vllm_get_model()` total: 17.0s (includes download + construction)
- If you divided model size by `vllm_get_model()` time: 15 GiB / 17s = 900 MiB/s

Even the broadest measurement (full `get_vllm_model` including JIT setup)
gives 900 MiB/s, not 53 MiB/s. The original 53 MiB/s was likely measured
under different conditions (different infrastructure, different RunAI version,
cold GCS cache, or cross-region reads). We cannot reproduce it.

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

---

## Historical Metrics (from issue #3768 and comments)

These are the numbers as originally reported. Metric boundaries were not
clear at the time and are now understood to have been confounded (see
post-mortem above).

### Original issue body (filed ~March 12-15)

| Claimed metric | Value | What it actually measured |
|----------------|-------|--------------------------|
| "RunAI downloads via single-threaded HTTP" | 53 MiB/s | RunAI progress bar rate during `vllm_get_model()` on PyTorch path — includes model construction, not just download |
| "RunAI concurrency (up to 16) no effect" | 51.5 MiB/s | Same — concurrency parameter didn't help because download was already fast; the 51.5 MiB/s reflected total `vllm_get_model` throughput |
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

## What Still Needs Proof: The Confound Proof Matrix

The post-mortem explains what we think went wrong. To prove it, we need
three runs that isolate the variables. Do NOT benchmark old commits —
use current scripts and force old behavior explicitly.

### Archaeological reference (read-only, do not run)

| Commit | What to look at | Purpose |
|--------|----------------|---------|
| `dbeddbd45` | `vllm_inprocess.py`, `exp_vllm_inprocess_direct.py` | Original fast path implementation |
| `3f1420e8d` | `vllm_server.py` line 624/651 | Where `MODEL_IMPL_TYPE=vllm` was hardcoded |
| `8f83e3692` | `vllm_server.py` | Later fix to `MODEL_IMPL_TYPE=auto` |

### Proof runs needed

All same-region (`gs://marin-us-east1/...`, zone `us-east1-d`), all using
current instrumented wheel (`marin-4abb68f4`).

**Run A: Forced old PyTorch path, direct LLM.generate()**

Reproduces the old `MODEL_IMPL_TYPE=vllm` behavior but with direct
`LLM.generate()` (not subprocess). Same timing boundary as our fast path.

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v6e-4 --memory 64GB --zone us-east1-d \
  --extra tpu --extra vllm \
  --job-name confound-llm-vllm \
  --no-wait \
  -e MODEL_IMPL_TYPE vllm \
  -- python experiments/inference/exp_vllm_llm_generate_direct.py \
    --model gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
    --load-format runai_streamer
```

**Run B: Corrected JAX path, direct LLM.generate()**

Already done as `vllm-sr-llm-direct-v4`. Isolates PyTorch vs JAX path
with the same timing boundary as Run A.

**Run C: Historical-style subprocess baseline**

Reproduces the original measurement shape: subprocess `vllm serve` with
`MODEL_IMPL_TYPE=vllm`. Measures server startup time (the timing boundary
that produced the original "53 MiB/s" number).

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v6e-4 --memory 64GB --zone us-east1-d \
  --extra tpu --extra vllm --extra eval \
  --job-name confound-subprocess-vllm \
  --no-wait \
  -e MARIN_VLLM_MODE native \
  -e MODEL_IMPL_TYPE vllm \
  -- python -m marin.inference.vllm_smoke_test \
    --model gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
    --mode native --max-model-len 4096 \
    --load-format runai_streamer \
    --local
```

### What the proof matrix will show

| Comparison | What it isolates |
|-----------|-----------------|
| A vs B | PyTorch vs JAX path, same timing boundary (LLM.generate) |
| B vs C | LLM.generate vs subprocess, same JAX... wait, C forces vllm. So: |
| A vs C | Direct LLM() vs subprocess `vllm serve`, both PyTorch path |
| Run 1 vs A | vllm serve vs LLM.generate, both PyTorch path |

If A and B both show model init in low tens of seconds but C is much
larger, the confound is proven: the old "53 MiB/s" was subprocess
startup overhead, not raw weight transport.

### Completed runs (can serve as Run B)

| Run | Job | Result |
|-----|-----|--------|
| B | `vllm-sr-llm-direct-v4` | ✅ model load 11.2s, RunAI 2.1 GiB/s, total 291s |

### Remaining runs

- [ ] Run A: `confound-llm-vllm` — direct LLM.generate with MODEL_IMPL_TYPE=vllm
- [ ] Run C: `confound-subprocess-vllm` — subprocess vllm serve with MODEL_IMPL_TYPE=vllm

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
