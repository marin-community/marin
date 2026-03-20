# Rerank Decode: Scorer Implementation Notes

## Current: VLLMLogprobScorer

Sends `prompt + completion` to a vLLM server with `echo=True, logprobs=1`.
Sums log-probabilities over all tokens to produce a score per candidate.

**Known limitation**: `echo=True` requests prompt logprobs, which disables
vLLM's automatic prefix caching (APC). Since all N candidates share the
same prompt prefix, this means the scoring model redundantly recomputes
the prompt KV cache for every candidate in every request.

## Future: Speculative-Decoding-Based Scorer

vLLM's speculative decoding (`vllm/spec_decode/`) already implements the
core primitive we need: a target model doing a single batched forward pass
to verify draft tokens, with the prompt prefix KV computed once and shared.

The flow in spec decode is:

1. `SpecDecodeWorker` orchestrates the draft → verify loop.
2. `DraftModelRunner` proposes a sequence of tokens.
3. `TargetModelRunner` does one forward pass over all proposed tokens,
   producing logprobs at every position. The shared prompt prefix KV is
   computed once via paged attention.
4. `RejectionSampler` applies per-token accept/reject:
   accept with prob `min(1, P_target / P_draft)`.

To implement best-of-N reranking, only step 4 changes:

- **Step 2 (draft)**: Generate N independent completions of `chunk_size`
  tokens each (instead of 1 sequence). The draft model already supports
  batched generation, so this is mostly plumbing `num_samples` through.
- **Step 3 (verify)**: No change. The target model scores all N candidates
  in one batched forward pass with shared prefix KV.
- **Step 4 (select)**: Replace `RejectionSampler` with a `BestOfNSelector`
  that sums logprobs per candidate and returns the argmax. No per-token
  accept/reject — just pick the best completion.

### Concrete implementation steps

1. Subclass `SpecDecodeWorker` (or add a mode flag) to support N>1 drafts.
2. Replace `RejectionSampler` with a selector that does
   `argmax_i sum(log P_target(tokens_i))` over the N candidates.
3. Expose `num_samples` as a parameter alongside `num_speculative_tokens`
   (which maps to `chunk_size`).
4. Wrap this in a `Scorer` subclass so it plugs into `rerank()` via the
   same interface as `VLLMLogprobScorer`.

### Simpler intermediate step

Before modifying vLLM internals, use `vllm.LLM` offline API directly:

```python
from vllm import LLM, SamplingParams

scorer_llm = LLM(model=scoring_model)

# All N candidates batched together — vLLM's paged attention
# deduplicates the shared prompt prefix KV within a single batch.
candidates = [prompt + completion_i for completion_i in completions]
outputs = scorer_llm.generate(
    candidates,
    SamplingParams(max_tokens=1, prompt_logprobs=1),
)
# Sum prompt_logprobs over the completion token positions only.
```

This doesn't fix cross-request APC, but within a single batched call the
prefix KV blocks are shared automatically. This may be sufficient if all
N candidates are scored together (which they are in `rerank()`).
