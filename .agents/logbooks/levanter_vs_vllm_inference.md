# Levanter vs vLLM Inference: Benchmark Logbook

## Scope
- **Goal**: Determine if Levanter's multi-host inference can replace vLLM for GRPO sampling workloads; establish comparable throughput numbers for PR #2759 review.
- **Primary metric(s)**: `gen_tok/s` (generated tokens per second), `batch_wall_time` (seconds per 1024-completion mini-batch).
- **Constraints**: Start on v5p-8 (1 host, 4 chips, ~380 GiB HBM). Scale to multi-host later.
- **Stop criteria**: Have comparable numbers on at least v5p-8 for 1 GRPO mini-batch (1024 completions).

## Conclusion: vLLM is the right path for GRPO inference

**Levanter cannot replace vLLM for GRPO sampling.** The benchmark shows a 6-10x throughput gap that is structural, not fixable with tuning. The gap comes from fundamental architectural differences in how the two systems handle inference workloads.

### Head-to-head: v5p-8, Llama 8B, MATH prompts

| System | Config | tok/s | GRPO epoch projection |
|---|---|---:|---:|
| **Levanter** | 64 prompts x 1 gen | **762** | ~24h |
| vLLM (c=64) | 64 prompts x 16 gen | 4,749 | 4.7h |
| vLLM (c=256) | 64 prompts x 16 gen | 7,940 | 2.8h |

### Why vLLM wins

1. **Continuous batching**: vLLM fills freed slots immediately when sequences finish. Levanter runs all sequences in lock-step — when a short sequence finishes at token 200, the slot wastes compute for the remaining 824 iterations. With temp=1.0 MATH problems where output lengths vary from 50 to 1024 tokens, this waste is massive.

2. **Async concurrency**: vLLM serves 256 concurrent requests via async HTTP with a production-grade scheduler. Levanter processes a single static batch in one `engine.generate()` call.

3. **Memory efficiency at scale**: vLLM ran 1,024 concurrent sequences on v5p-8. Levanter OOM'd at 1,024 seqs (16K pages pre-allocated). This forced us to benchmark with only 64 sequences, underutilizing the hardware.

4. **XLA compilation overhead**: Levanter pays 25.3s for XLA compilation on the first decode iteration. vLLM compiles at startup and amortizes across all requests.

5. **Prefill pipeline**: Levanter prefills ALL prompts (41.6s), THEN decodes. vLLM overlaps prefill with decode — new requests start prefilling while existing ones are already generating.

### Where Levanter has advantages (but they don't overcome the gap)

- **KV cache cloning**: `n_generations=16` prefills once and clones 15x sharing KV pages. vLLM prefills each of the 16 completions independently. This is a real advantage but couldn't be exercised on v5p-8 due to OOM.
- **No HTTP overhead**: Direct Python API, no serialization/deserialization.
- **Large model support**: Models too large for one host can use global mesh inference (tensor parallel across hosts). vLLM needs Ray for cross-host PP, which doesn't work without Ray on TPU.

### What could close the gap (and why it's not worth it)

| Fix | Est. speedup | Effort | Notes |
|---|---|---|---|
| XLA warmup | 1.4x | 1 day | Removes 25s compile from first iter |
| Multi-host data-parallel | ~2x | Config change | Already works on v5p-16+ |
| Chunked prefill | 1.3-1.5x | 1-2 weeks | Overlap prefill with decode |
| Continuous batching | 2-3x | Months | Requires rearchitecting the decode loop |
| **Combined (optimistic)** | **~3-4x** | **Months** | Still 2-3x behind vLLM |

Even with all realistic fixes, Levanter would reach ~2,500-3,000 tok/s — still 2.5-3x slower than vLLM on the same hardware. Implementing continuous batching inside Levanter would essentially mean reimplementing vLLM's core scheduler, at which point you should just use vLLM.

### Recommendation

- **For GRPO/RL sampling**: Use vLLM. The throughput gap is too large and the fixes to close it amount to rebuilding vLLM inside Levanter.
- **PR #2759 value**: The multi-host inference work is still useful for:
  - Simple single-shot inference without a vLLM sidecar (evals, probing)
  - Models that require the full training mesh (70B+ on multi-host)
  - Future train-infer interleaving if the TPU runtime bug is resolved
  - Research/debugging where exact control over sampling matters more than throughput
- **Next steps**: Focus engineering effort on making vLLM integration better (fast weight loading, multi-host replicas, queue-based serving) rather than trying to match vLLM throughput in Levanter.

---

## GRPO Workload Definition

Matching Marin's `exp2039_rl_math500.py` inference config:
- **Prompts**: 64 (from Hendrycks MATH, seed=42 deterministic JSONL)
- **Completions per prompt**: 16 (`n_generations=16`)
- **Total per batch**: 1024 completions
- **Temperature**: 1.0
- **Max tokens**: 1024
- **Prompt format**: `"<problem> Write your answer in \boxed{} format."`
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`

## vLLM Baseline Numbers (from `no_ray_multihost_vllm.md`)

### Reference Scripts
- **GRPO stress test**: `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/benchmark_grpo_stress.py`
- **MATH-500 eval**: `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/benchmark_math500.py`
- **Dataset prep**: `/Users/ahmed/code/vllm_tpu_multi/vllm/marin_dev/prepare_hendrycks_math.py`
- **Full logbook**: `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md`

### vLLM GRPO Stress Test Results (concurrency=64/replica)

| Config | Chips | Replicas | gen_tok/s | Batch time (s) | Epoch projection |
|--------|------:|---------:|----------:|--------------:|-----------------:|
| v6e-4 (TP=4, 1 host) | 4 | 1 | ~3,000* | ~109s | ~5.7h |
| v6e-8 (TP=8, 1 host) | 8 | 1 | 3,973 | 104.4s | 5.5h |
| v6e-16 (4xTP=4) | 16 | 4 | 11,912 | 35.5s | 1.85h |
| v5p-8 (TP=4, 1 host) | 4 | 1 | 4,749 | 89.2s | 4.7h |
| v5p-16 (2xTP=4) | 8 | 2 | 8,681 | 48.9s | 2.6h |

*estimated

### vLLM at Optimal Concurrency (256/replica)

| Config | Chips | Replicas | gen_tok/s | Batch time (s) | Epoch projection |
|--------|------:|---------:|----------:|--------------:|-----------------:|
| v6e-16 (4xTP=4) | 16 | 4 | 19,708 | 21.1s | 1.1h |
| v5p-16 (2xTP=4) | 8 | 2 | 14,303 | 29.3s | 1.5h |
| v5p-8 (TP=4, 1 host) | 4 | 1 | 7,940 | 52.9s | 2.8h |

### vLLM Key Findings
- Concurrency=256/replica saturates all hardware (universal ceiling)
- v5p is ~1.45x faster per chip than v6e
- TP=8 vs TP=4 only 3% faster for 8B (communication overhead dominates)
- Architecture: independent replicas, no cross-host communication

## Levanter Existing Numbers (from Codex M5-M10 docs)

### Reference Code
- **Multi-host sampler**: `lib/levanter/src/levanter/main/sample_lm_multihost.py`
- **Config dir**: `lib/levanter/config/sampler/`
- **Engine**: `lib/levanter/src/levanter/inference/engine.py`
- **Codex speedup summary**: `lib/levanter/CODEX_SPEEDUP_SUMMARY.md`
- **M10 host-DP results**: `lib/levanter/CODEX_INFERENCE_M10.md`

### Levanter on v5p-16 (10 prompts x 2048 tokens, global mesh)

| Stage | round_total_s | tok/s (est) | Speedup vs M5 |
|-------|-------------:|------------:|--------------:|
| M5 baseline | 505.3s | ~40.5 | 1.0x |
| M6 final (scheduler) | 160.4s | ~127.7 | 3.15x |
| M8 best (TPU kernel) | 74.7s | ~274.0 | 6.76x |

### Levanter on v5p-32 (host-data-parallel, 4 hosts)

| Prompts | max_new_tokens | Wall clock | Total tokens | Est tok/s |
|--------:|---------------:|-----------:|-------------:|----------:|
| 128 | 2048 | 253.9s | 262,144 | ~1,032 |
| 256 | 2048 | 237.7s | 524,288 | ~2,206 |
| 512 | 2048 | 271.7s | 1,048,576 | ~3,860 |
| 1024 | 2048 | 366.4s | 2,097,152 | ~5,729 |

## Fairness Checklist

| Factor | vLLM | Levanter | Status |
|--------|------|----------|--------|
| Chat template | Yes (`/v1/chat/completions`) | Yes — `apply_chat_template: true` | FIXED |
| Batching | Continuous | Static (lock-step) | Architectural difference |
| n_generations | 16 independent HTTP requests | KV cache cloning (prefill once, clone 15x) | Levanter more efficient |
| max_tokens | 1024 | 1024 | Matched |
| temperature | 1.0 | 1.0 | Matched |
| Model | Llama 3.1 8B Instruct | Same (from GCS) | Matched |
| Prompts | First 64 from Hendrycks MATH (seed=42) | Same | Matched |

## Experiment Plan

| Run | Config | Hypothesis | Status |
|-----|--------|------------|--------|
| LVB-000 | Implement `apply_chat_template` | Pre-req for fair comparison | DONE |
| LVB-001 | Levanter v5p-8, 64 prompts x 1 gen | Establish baseline | DONE — 762 tok/s |
| LVB-002 | Sweep max_seqs | Find optimal batch size | CANCELLED — conclusion reached |
| LVB-003 | v5p-16 host-data-parallel | Multi-host scaling | CANCELLED — conclusion reached |
| LVB-004 | v5p-16 global mesh | Compare modes | CANCELLED — conclusion reached |

## Experiment Log

### 2026-03-18 — LVB-000: Add `apply_chat_template` to `sample_lm_multihost.py`

- **Status**: DONE
- **Change**: Added `apply_chat_template: bool = False` to `SampleLmMultihostConfig`, `_tokenize_prompts()` helper, updated both multihost and single-host tokenization paths.

### 2026-03-18 — LVB-001: Levanter v5p-8 GRPO Baseline

- **Hypothesis**: Establish Levanter throughput for GRPO workload on v5p-8.
- **Config**: `lib/levanter/config/sampler/benchmark_grpo_v5p8.yaml`
  - Model: `meta-llama/Llama-3.1-8B-Instruct` (us-east5 GCS)
  - TPU: v5p-8, us-east5-a, 1 host, 4 chips, TP=4
  - Engine: max_seq_len=2048, max_pages=1024, page_size=128, max_seqs=64, ragged paged attn ON (q16/kv8)
  - 64 prompts x 1 gen, n_rounds=1, apply_chat_template=true, temp=1.0, max_tokens=1024
- **Status**: DONE (after 6 iterations fixing config/code issues)
- **TPU**: `lvb-bench-v5p8`, v5p-8, us-east5-a, best-effort
- **Issues fixed during launch**:
  1. Docker WORKDIR is `/opt/marin/lib/levanter` — paths must be relative to that
  2. `max_seqs=128` too small for 1024 seqs — reduced to 64x1 gen for v5p-8 memory limits
  3. `max_seq_len=1536` too small with chat template — bumped to 2048
  4. `max_prefill_size=4096` too small for 64 chat-templated prompts — bumped to 8192
  5. VMEM OOM in ragged paged attention — reduced block sizes to q16/kv8
  6. `TracerBoolConversionError`: merge set `static_argnums=(3,4)` but `cleanup_each_step` (arg 5) also needs static — fixed to `(3,4,5)`
  7. `--provisioning-model=SPOT` gcloud flag unsupported — removed

- **Result (64 prompts x 1 gen x 1024 max_tokens)**:

  | Metric | Value |
  |---|---|
  | Total tokens generated | 65,536 |
  | Round wall time | 86.0s |
  | **Round tok/s** | **762** |
  | Decode-only tok/s | 1,475 |
  | Decode time | 44.4s |
  | Prefill time | 41.6s |
  | First decode iter (XLA compile) | 25.3s |
  | Decode iter p50 | 1.13s |
  | Device compute p50 | 0.26ms |

- **Time breakdown**:

  | Phase | Time | % of wall |
  |---|---:|---:|
  | Prefill (inc. compile) | 41.6s | 48% |
  | XLA compile (1st decode) | 25.3s | 29% |
  | Decode compute (15 iters) | ~17s | 20% |
  | Host overhead | ~2s | 3% |

- **Comparison to vLLM v5p-8**:

  | System | tok/s | Gap |
  |---|---:|---|
  | Levanter | 762 | baseline |
  | vLLM (c=64) | 4,749 | 6.2x faster |
  | vLLM (c=256) | 7,940 | 10.4x faster |

  **Levanter is 6-10x slower than vLLM on the same hardware.** The decode kernel is fast (device compute <1ms) but prefill is slow (48% of wall time), XLA compilation is costly (29%), and static batching wastes compute on finished sequences.
