# 535B-A23B long-context cooldown

**Status:** 2026-08-24

## TL;DR

The 535B-A23B campaign plans a detached 1–2 day cooldown around day 10–20, followed by context extensions from 4K to 8K near 50% of pretraining and from 8K to 65K near 95%. A targeted 262K phase may follow at the end. The early cooldown should measure whether the full-scale MoE remains stable when the number of independent documents per routing shard falls. The campaign plan is tracked in [#8435](https://github.com/marin-community/marin/issues/8435); [#8374](https://github.com/marin-community/marin/issues/8374) is the most complete blocker list.

The 65K training recipe has prior evidence. Two 67B-A2B cooldowns completed 211B-token legs at 65K, with final Paloma macro losses of 2.2772 and 2.224, and a later 1T-token cooldown also completed. A 262K TPU context-parallel run has since finished, while a `qk157` variant remains in progress. These runs establish that the context transition can train, but they do not settle hero-scale MoE dropping or the GB200 implementation.

Haliax and standard Levanter already provide logical-axis sharding and a simple query-sharded, all-gather-K/V form of context parallelism. Grug uses its own fixed mesh and raw partition specifications. Its production FA4/CuTe path requires the sequence axis to remain unsharded. A prototype Transformer Engine benchmark exists on `codex/research/grug-context-parallel-attention`, but it has not been integrated into the Grug model or trainer.

The immediate decision is whether the early 65K cooldown fits without context parallelism. If it does, the cooldown can proceed after a short exact-shape probe and a routing-drop measurement. The 262K phase remains gated on GB200 context-parallel backward, Grug integration, EP+CP composition, and an evaluation that tests whether the model uses context beyond 32K.

## Campaign intent

The [535B-A23B campaign](https://github.com/marin-community/marin/issues/8435) trains a 535.3B-total, 22.76B-active MoE on 704 GB200s. Pretraining starts at 4K context with EP64. The [current W&B run](https://wandb.ai/marin-community/marin_moe/runs/hero-12d8b6f0-dee637) reports `state=running`.

The planned context schedule is:

1. Fork an early checkpoint into a 1–2 day cooldown around day 10–20. Use it for early RL work and to measure full-scale token dropping.
2. Move the main run from 4K to 8K near 50% of training if the early transition is healthy.
3. Move from 8K to 65K near 95%.
4. Consider a 262K final phase using results from the 67B-A2B 65K-to-262K experiments.

Starting at 4K gives each routing shard more independent sequences. Tokens from the same document tend to route coherently, so longer sequences reduce the effective sample size for expert balancing. [#8374](https://github.com/marin-community/marin/issues/8374) identifies this as the unresolved model-quality risk: at fixed tokens per step, longer contexts can increase capacity overflow even when the token count is unchanged.

## Evidence available

### Context transition

Small-scale experiments found stable 4K-to-16K and 4K-to-32K transitions with YaRN-style QK scaling. Coefficients near 0.1 are the current starting point. The 32K experiments also found higher gradient variance when the batch contained fewer independent documents. See [#6170](https://github.com/marin-community/marin/issues/6170) and [#6194](https://github.com/marin-community/marin/issues/6194).

The 67B-A2B cooldowns provide the closest training precedent:

- The first 8K-to-65K cooldown preserved 67.1M tokens per step by reducing batch size from 8,192 to 1,024. A short sweep selected `qk_mult=1.5703`; the 211B-token leg finished at Paloma macro loss 2.2772. See [#6811](https://github.com/marin-community/marin/issues/6811).
- A second matched cooldown from a later checkpoint finished at Paloma macro loss 2.224. The final cooldown trained the last 1T tokens at 65K. See [#6044](https://github.com/marin-community/marin/issues/6044).
- A [262K TPU CP4 run](https://wandb.ai/marin-community/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k) has finished. A [`qk157` variant](https://wandb.ai/marin-community/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk157) is still running.

The last item supersedes the statement in [#8374](https://github.com/marin-community/marin/issues/8374) that nothing above 65K had trained. The issue remains the best blocker inventory, but its run-status section needs an update.

### GB200 and MoE

An earlier d6144 FSDP benchmark measured a 1.8% throughput reduction at 8K and a 35% reduction at 65K relative to 4K. That shape had eight global-attention layers; the production EP hero has twelve, so the 65K result is a lower-confidence cost estimate for the current model. See [the benchmark result](https://github.com/marin-community/marin/issues/7201#issuecomment-5097482159) and [#8374](https://github.com/marin-community/marin/issues/8374).

A d2048 checkpoint trained with EP64 successfully changed to dropless FSDP chunk-1 for a [65K extension](https://wandb.ai/marin-community/marin_moe/runs/mhep-ctxext-d2048-ep64-to-fsdp-chunk1-20260810-seq65536). This is evidence that an EP-checkpoint-to-FSDP cooldown can work at smaller scale. It does not qualify that topology change at 535B.

The intended hero-scale EP64 measurement was specified in [#8062](https://github.com/marin-community/marin/issues/8062). Those 65K jobs were interrupted by the GB200 illegal-instruction failures tracked in [#7956](https://github.com/marin-community/marin/issues/7956), so the required drop-rate and loss comparison is still missing.

### Long-document data

The in-flight branch [`held/long-context-cooldown-mixture`](https://github.com/marin-community/marin/tree/held/long-context-cooldown-mixture) partitions the June 67B Datakit mixture into documents at or below 64K and documents above 64K. It preserves each quality/domain cell's total weight while exposing a relative multiplier on long-document mass. The branch contains a [runnable 67B cooldown launcher](https://github.com/marin-community/marin/blob/8976cfe80d994f93cd401bca6ca9beab6b0702df/experiments/june_tpu_67b_a2b/moe/moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T_long_context.py).

This branch establishes data plumbing and a controlled long-document skew for the 67B mixture. The 535B Harrier mixture still needs an explicit decision about whether to preserve its domain/quality mass in the same way and which skew to use.

## Context parallelism

Context parallelism shards the sequence dimension across a mesh axis. Each rank owns a subset of queries. Attention still requires keys and values from every sequence shard.

A simple implementation all-gathers K/V to each query rank. Ring attention instead circulates K/V blocks and accumulates the softmax online, reducing peak K/V memory. Causal attention also needs a balanced token layout; a contiguous split gives later sequence shards more work than earlier shards.

### Present in main

- Haliax maps named logical axes to JAX mesh axes and produces the corresponding `PartitionSpec` and `NamedSharding`. See [`haliax.partitioning`](https://github.com/marin-community/marin/blob/main/lib/haliax/src/haliax/partitioning.py).
- Standard Levanter documents a basic configuration with a `context` mesh axis, `position: context`, and unsharded `key_position`. This is the all-gather-K/V variant. See [Mesh Parallelism](https://github.com/marin-community/marin/blob/main/lib/levanter/docs/explanation/Mesh-Parallelism.md#add-context-sequence-parallelism) and the [multi-device parity test](https://github.com/marin-community/marin/blob/main/lib/levanter/tests/test_attention.py#L274-L348).
- Levanter's Transformer Engine path already constructs packed-sequence descriptors. See [`_te_flash_attention`](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/layers/attention.py#L503-L643).

### Missing from Grug main

- Grug's compact mesh contains only `(replica_dcn, data, expert, model)`. See [`compact_grug_mesh`](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/grug/sharding.py#L144-L174).
- The production FA4/CuTe wrapper rejects sequence-sharded Q/K/V. See [`_assert_sequence_axis_unsharded`](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/grug/attention/_fa4_cute.py#L203-L217).
- Grug has no production Transformer Engine context-parallel backend in its attention dispatcher.
- The MoE path has no defined policy for activations sharded over both expert/data axes and a context axis.

### In-flight prototype

David Hall's [`codex/research/grug-context-parallel-attention`](https://github.com/marin-community/marin/tree/codex/research/grug-context-parallel-attention) branch benchmarks Transformer Engine ring and all-gather strategies. It constructs context-sharded Q/K/V, applies striped causal load balancing to packed-sequence metadata, passes the CP arguments to `fused_attn`, and compiles forward plus backward. The branch also includes an Iris launcher for exact-shape multi-node runs. See the [benchmark](https://github.com/marin-community/marin/blob/b2c249240d1bbb447ac20a6d567240da0d059a54/lib/levanter/scripts/bench/bench_grug_context_parallel_attention.py) and [launcher](https://github.com/marin-community/marin/blob/b2c249240d1bbb447ac20a6d567240da0d059a54/experiments/grug/moe_hero_fsdp/context_parallel_attention_benchmark.py).

The prototype is not called by the Grug model or trainer. [#8141](https://github.com/marin-community/marin/issues/8141) reports that Transformer Engine 2.17.1 fails during cuDNN backward graph construction. Building a newer Transformer Engine also failed because its CUDA root did not contain `nvvm/bin/cicc` at the expected path.

## Open gates

The early 65K cooldown needs:

1. An exact hero-shape memory and throughput probe at 65K.
2. A measured drop fraction and loss under the selected EP transport and capacity factor.
3. A checkpoint fork that leaves the main run and its checkpoint lineage unchanged.
4. A decision on the long-document mixture and QK scaling sweep.
5. A capability evaluation in addition to Paloma loss.

The 262K phase also needs:

1. Transformer Engine CP4 forward and backward to pass on four GB200s.
2. A `context` axis in the Grug mesh and sequence-aware activation and metadata partition specs.
3. Integration of ring or all-gather attention into the Grug attention dispatcher.
4. A defined composition of context parallelism with EP, or a recorded decision to use FSDP for the 262K phase.
5. A 64-GPU exact-shape qualification before a hero checkpoint is used.

The acceptance criteria and current runtime failures are tracked in [#8374](https://github.com/marin-community/marin/issues/8374) and [#8141](https://github.com/marin-community/marin/issues/8141).

## Evaluation gap

Paloma loss can show that the context transition trains without a regression. It does not establish retrieval or reasoning over the additional positions. The existing `long_context_32k` bundle stops at 32K, and RULER has not landed; see [#7638](https://github.com/marin-community/marin/issues/7638). A 65K checkpoint should pass a retrieval-style evaluation before the campaign spends on a 262K phase.

## References

- [#8435: 535B-A23B hero campaign](https://github.com/marin-community/marin/issues/8435)
- [#8374: blockers to 262K Grug training](https://github.com/marin-community/marin/issues/8374)
- [#8141: GB200 context-parallel attention experiment](https://github.com/marin-community/marin/issues/8141)
- [#8062: FSDP versus EP experimental design](https://github.com/marin-community/marin/issues/8062)
- [#7956: intermittent GB200 illegal-instruction failures](https://github.com/marin-community/marin/issues/7956)
- [#6811: first 67B-A2B 65K cooldown](https://github.com/marin-community/marin/issues/6811)
- [#6044: later 67B-A2B 65K cooldowns](https://github.com/marin-community/marin/issues/6044)
- [#6170: small-scale context-extension stability](https://github.com/marin-community/marin/issues/6170)
- [#6194: context extension with reduced document diversity](https://github.com/marin-community/marin/issues/6194)
- [#7638: RULER integration](https://github.com/marin-community/marin/issues/7638)
- [`codex/research/grug-context-parallel-attention`](https://github.com/marin-community/marin/tree/codex/research/grug-context-parallel-attention)
- [`held/long-context-cooldown-mixture`](https://github.com/marin-community/marin/tree/held/long-context-cooldown-mixture)
