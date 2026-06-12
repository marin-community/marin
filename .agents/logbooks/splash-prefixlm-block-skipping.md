# Splash Prefix-LM Block Skipping: Research Logbook

## Scope
- Goal: Extend Splash Attention support from static causal/local masks plus runtime segment IDs to prefix-LM masks with data-dependent prefix lengths under packing, while preserving or improving block skipping on TPU.
- Primary metrics: Correctness versus dense/reference attention; steady-state attention time on v4-8 and v5p-8 for sequence lengths 8192 and 16384; compile-including time for first-call regressions.
- Constraints: Keep existing causal, full, sliding-window, and segment-ID Splash paths working. Avoid materializing large dense masks on the common static path. Start with v4-8 if v5p availability is poor.

## Baseline
- Date: 2026-06-11
- Code refs:
  - `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`
  - `lib/levanter/src/levanter/layers/attention.py`
  - `lib/levanter/src/levanter/layers/attention_mask.py`
  - `lib/levanter/src/levanter/grug/attention/_core.py`
- Baseline numbers: TBD. PR #6314 merged the shared Splash mask lowering helper and passed a v4-8 Iris smoke test with existing causal/sliding/segment-ID masks.

## Experiment Log
### 2026-06-11 00:00 - Kickoff on merged Splash helper
- Hypothesis: Prefix-LM can be added as a structured `AttentionMask` field and lowered to Splash without a major speed hit by keeping static mask lowering for uniform prefix lengths, then adding a dynamic per-example path for packed/data-dependent prefix lengths.
- Command: `git switch -c codex/splash-prefixlm-block-skipping origin/main`
- Config: Local code inspection on merged `origin/main`.
- Result: Current JAX Splash dynamic-mask path accepts `[heads, q, kv]` masks and still computes `block_mask`/`data_next` to skip fully empty blocks. It does not directly accept a batch axis. Existing Levanter Splash invocation vmaps over batch and shares one kernel object/mask-info across examples.
- Interpretation: Uniform prefix-LM can use a static/sliceable Splash mask. Packed data-dependent prefix-LM needs either per-example kernel/mask-info vmapping or a batch-to-head folding strategy; using only `segment_ids` cannot skip segment-empty blocks because Splash applies `segment_ids` inside visited blocks.
- Next action: Prototype a structured prefix-LM mask surface and static lowering tests first, then test whether vmapping a dynamic-mask `SplashAttentionKernel` over per-example mask-info is accepted by `shard_map`/Pallas.
