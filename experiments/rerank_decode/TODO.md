# Rerank Decode on TPU — Plan

## 1. Modify Levanter InferenceEngine

File: `lib/levanter/src/levanter/inference/engine.py`

- Add `_run_scoring` kernel — same as `_prefill_kernel` (line 395) but returns raw logits instead of sampling
- Add `score()` method on `InferenceEngine` that:
  1. Prefills the prompt (caches KV)
  2. Clones the sequence N times via `clone_sequence` (line 281) — shares prefix pages
  3. Feeds completion tokens through `model.decode` batched, without sampling
  4. Computes `log_softmax(logits)` at each position, indexes actual completion tokens, sums

## 2. Write a `LevanterScorer` in `experiments/rerank_decode/scorer.py`

Implements the `Scorer` interface, backed by the new `InferenceEngine.score()`.

## 3. Single ExecutorStep on v4-16

- Proposal model: in-process `vllm.LLM`
- Scoring model: Levanter `InferenceEngine`
- One `ExecutorStep` with `ResourceConfig.with_tpu("v4-16")`
- Both Llama and Qwen models supported by Levanter (`LlamaLMHeadModel`, `QwenLMHeadModel`)
- Isoflop models use `LlamaConfig` — also compatible

## 4. Submit

```bash
uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait -- python experiments/rerank_decode/sweep_math500.py
```

## Constraints

- VLLMLogprobScorer is too slow: recomputes full prefix every scoring call, APC does not work with prompt_logprobs
- KVCacheScorer is PyTorch/CUDA-only, single device, doesn't work on TPU
- Levanter InferenceEngine currently only generates tokens, does not support scoring — needs the modifications above
- No changes needed to `model.decode`, `PageCache`, `PageTable`, or `GenState`
