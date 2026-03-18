# Levanter vs vLLM Inference: Benchmark Logbook

## Scope
- **Goal**: Determine if Levanter's multi-host inference can replace vLLM for GRPO sampling workloads; establish comparable throughput numbers for PR #2759 review.
- **Primary metric(s)**: `gen_tok/s` (generated tokens per second), `batch_wall_time` (seconds per 1024-completion mini-batch).
- **Constraints**: Start on v5p-8 (1 host, 4 chips, ~380 GiB HBM). Scale to multi-host later.
- **Stop criteria**: Have comparable numbers on at least v5p-8 for 1 GRPO mini-batch (1024 completions).

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
| v6e-16 (4×TP=4) | 16 | 4 | 11,912 | 35.5s | 1.85h |
| v5p-8 (TP=4, 1 host) | 4 | 1 | 4,749 | 89.2s | 4.7h |
| v5p-16 (2×TP=4) | 8 | 2 | 8,681 | 48.9s | 2.6h |

*estimated

### vLLM at Optimal Concurrency (256/replica)

| Config | Chips | Replicas | gen_tok/s | Batch time (s) | Epoch projection |
|--------|------:|---------:|----------:|--------------:|-----------------:|
| v6e-16 (4×TP=4) | 16 | 4 | 19,708 | 21.1s | 1.1h |
| v5p-16 (2×TP=4) | 8 | 2 | 14,303 | 29.3s | 1.5h |
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

### Levanter on v5p-16 (10 prompts × 2048 tokens, global mesh)

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

**Note**: These are NOT directly comparable to vLLM — different workload (max_tokens=2048 vs 1024), different hardware, and wall clock includes container overhead. The benchmark below will establish a proper comparison.

## Fairness Checklist

Differences between vLLM and Levanter benchmarks that must be controlled:

| Factor | vLLM | Levanter | Status |
|--------|------|----------|--------|
| Chat template | Yes (`/v1/chat/completions` applies Llama 3.1 chat template) | Yes — `apply_chat_template: true` in config | FIXED |
| Batching | Continuous (freed slots immediately reused) | Static (all seqs run in lock-step) | Inherent architectural difference — this is what we're measuring |
| n_generations | 16 independent HTTP requests per prompt | KV cache cloning (prefill once, clone 15x) | Levanter is more efficient here |
| Sampling | Per-request PRNG from server | Per-clone PRNG via `jax.random.fold_in` | Equivalent |
| max_tokens | 1024 | 1024 | Matched |
| temperature | 1.0 | 1.0 | Matched |
| Model | Llama 3.1 8B Instruct | Same (from GCS) | Matched |
| Prompts | First 64 from Hendrycks MATH (seed=42 JSONL) | Same | Matched |

### Chat Template Fix (pre-requisite for LVB-001)

`sample_lm_multihost.py` line 685 raw-tokenizes prompts:
```python
prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
```

vLLM's `/v1/chat/completions` applies the Llama 3.1 Instruct chat template, wrapping each prompt in:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

**Fix**: Add `apply_chat_template: bool` to `SampleLmMultihostConfig` and use `tokenizer.apply_chat_template()` in `_broadcast_prompt_payload` when enabled. The infrastructure already exists:
- `lib/levanter/src/levanter/inference/openai.py:572` does exactly this
- `lib/levanter/src/levanter/data/text/formats.py` has `ChatProcessor`
- HuggingFace tokenizer for Llama 3.1 Instruct has a built-in chat template

~10 lines of code. Must be done before running LVB-001.

## Experiment Plan

| Run | Config | Hypothesis |
|-----|--------|------------|
| LVB-000 | Implement `apply_chat_template` in `sample_lm_multihost.py` | Pre-requisite for fair comparison |
| LVB-001 | Levanter v5p-8, 64 prompts × 16 gen, max_tokens=1024, temp=1.0, chat template ON | Establish single-host Levanter baseline for GRPO workload |
| LVB-002 | Same but sweep max_seqs (64, 128, 256) | Find optimal batch size for v5p-8 |
| LVB-003 | Levanter v5p-16, host-data-parallel, same workload | Compare multi-host scaling |
| LVB-004 | Levanter v5p-16, global mesh, same workload | Compare global mesh vs host-DP |

## Experiment Log

### 2026-03-18 — LVB-000: Add `apply_chat_template` to `sample_lm_multihost.py`

- **Motivation**: vLLM benchmark uses `/v1/chat/completions` which applies Llama 3.1 Instruct chat template. Levanter raw-tokenizes prompts. Without this fix, prompt tokenization differs and output distributions won't match.
- **Change**: Add `apply_chat_template: bool = False` to `SampleLmMultihostConfig`. When enabled, `_broadcast_prompt_payload` wraps each prompt as `[{"role": "user", "content": prompt}]` and calls `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`.
- **Files**: `lib/levanter/src/levanter/main/sample_lm_multihost.py`
- **Status**: DONE
- **Result**: Added `apply_chat_template: bool` to `SampleLmMultihostConfig`, `_tokenize_prompts()` helper, updated both multihost and single-host tokenization paths. Benchmark config updated with `apply_chat_template: true`.

### 2026-03-18 — LVB-001: Levanter v5p-8 GRPO Baseline

- **Hypothesis**: Establish Levanter throughput for 1 GRPO mini-batch (64 prompts × 16 gen = 1024 completions, max_tokens=1024, temp=1.0) on v5p-8. Direct comparison to vLLM v5p-8 baseline (4,749 tok/s at c=64, 7,940 tok/s at c=256).
- **Config**: `lib/levanter/config/sampler/benchmark_grpo_v5p8.yaml`
  - Model: `meta-llama/Llama-3.1-8B-Instruct` from GCS (us-east5 bucket)
  - TPU: v5p-8, us-east5-a, 1 host, 4 chips, TP=4
  - Engine: max_seq_len=1536, max_pages=4608, page_size=128, max_seqs=128, ragged paged attn ON (q32/kv16)
  - Prompts: 64 Hendrycks MATH problems (deterministic, first 64 from seed=42 JSONL)
  - n_generations=16, n_rounds=1, apply_chat_template=true
- **Command**:
  ```bash
  python infra/launch.py --foreground --zone us-east5-a \
    --tpu_name lvb-bench-v5p8 --tpu_type v5p-8 --capacity_type best-effort \
    -- uv run lib/levanter/src/levanter/main/sample_lm_multihost.py \
    --config_path lib/levanter/config/sampler/benchmark_grpo_v5p8.yaml
  ```
- **Expected**: Given M8 results (~275 tok/s for 10×2048 on v5p-16 global mesh), and the GRPO workload being larger (1024 completions × 1024 tokens), Levanter will likely be significantly slower than vLLM (expected ~500-2000 tok/s vs vLLM's 4,749-7,940 tok/s). The gap comes from vLLM having continuous batching and optimized scheduling.
- **Status**: DONE (after 6 iterations fixing config/code issues)
- **TPU**: `lvb-bench-v5p8`, v5p-8, us-east5-a, best-effort
- **W&B run**: https://wandb.ai/marin-community/levanter-multihost-inference/runs/io53vmrs (helpful-meadow-16)
- **Issues fixed during launch**:
  1. Docker path: commands must be relative to WORKDIR `/opt/marin/lib/levanter`
  2. `max_seqs` too small (128 for 1024 seqs) — reduced to 64 prompts × 1 gen for v5p-8
  3. `max_seq_len` too small (1536) — bumped to 2048 for chat template overhead
  4. `max_prefill_size` too small (4096) — bumped to 8192 for 64 prompts with chat template
  5. VMEM OOM in ragged paged attention kernel — reduced block sizes to q16/kv8
  6. `TracerBoolConversionError`: merge from main set `static_argnums=(3, 4)` but `cleanup_each_step` (arg 5) also needs to be static → fixed to `(3, 4, 5)`
  7. `--provisioning-model=SPOT` flag not supported by local gcloud → removed
- **Result (64 prompts × 1 gen × 1024 max_tokens, temp=1.0, chat template ON)**:

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
  | Decode iter p90 | 2.01s |
  | Device compute p50 | 0.26ms |

- **Comparison to vLLM v5p-8**:

  | System | Config | tok/s | Batch time |
  |---|---|---:|---:|
  | **Levanter** | 64 prompts × 1 gen | **762** | 86.0s |
  | vLLM (c=64) | 64 prompts × 16 gen | 4,749 | 89.2s |
  | vLLM (c=256) | 64 prompts × 16 gen | 7,940 | 52.9s |

  **Levanter is 6.2-10.4x slower than vLLM** on the same hardware. The decode kernel is fast (device compute <1ms) but prefill is slow (41.6s = 48% of wall time) and the static batching means no continuous batching benefits.

- **Note**: This is 64 sequences, not the full 1024 GRPO batch. v5p-8 OOMs at max_seqs=1024 (16K pages needed). The full GRPO comparison requires either multi-host (v5p-16+) or running 16 sequential rounds of 64 seqs.
