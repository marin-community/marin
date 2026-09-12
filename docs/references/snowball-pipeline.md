# Snowball model and pipeline stages

`experiments.june_tpu_67b_a2b.moe.rl_model.JuneSnowballConfig` exposes the June
training transformer through Levanter's `LmHeadModel` interface. It uses the
published Snowball HF configuration and tensor layout, with explicit causal
segment masks and optional per-example position IDs. `capacity_factor` controls
capacity in the backend selected by `moe_implementation` and must be finite and
positive. For `ragged_all_to_all`, capacity is based on local token count ×
experts selected per token, multiplied by this factor, then divided and rounded
across local expert chunks. Padding consumes capacity. Other backends retain
their existing capacity rules.

The adapter accepts Levanter's structured causal `AttentionMask` with segment
IDs; explicit dense masks are rejected. Tokens can attend only to earlier or
current positions within the same segment, subject to each layer's window.
Position IDs have the same named batch/position axes as tokens. The adapter
preserves supplied positions, including packed-sequence resets; it does not
infer resets from segment IDs. Stage methods take raw `[batch, sequence]`
segment and position arrays.

`split_june_pipeline_model(model, num_stages)` partitions contiguous blocks
without copying their arrays. The first stage owns the embedding modules; the
last owns the final normalization and output projection. Layer offsets preserve
the original short/long attention schedule, including the final long layer.
The stage interface provides `embed`, `run_blocks`, `finish`, and `get_lm_head`;
an execution scheduler must place and transport these stage computations.
This module does not provide a scheduler, optimizer, or checkpoint lifecycle.

Each stage exports and loads only its owned HF keys, using global layer indices.
The union of stage exports equals the full model export. Loading an owned tensor
subset into an abstract stage template avoids constructing the full model.
`trainable_filter` excludes the fixed per-expert QB biases used in top-k routing.
Partition the model with this filter, cast only the trainable partition, and
combine it with the unchanged FP32 frozen partition. Filtering gradients alone
does not prevent an indiscriminate model-wide cast from rounding these biases.

`run_blocks_with_stats` returns hidden states and routing counts. Each token-to-selected-expert pair is one assignment, including padded tokens.
Backend counters count assignments clipped before transport (`sender`) or after
transport (`receiver`); ragged transport reports its chunk clipping as sender
drops. Stage counters sum backend totals across that stage's layers, without
combining other pipeline stages. Counters also report the maximum drops
in any layer, and the last global layer index with drops (`-1` when none).
The caller decides whether drops are acceptable. With expert parallelism,
telemetry rejects microbatches above 2²⁴ total assignments across all model layers
so downstream floating-point auxiliary transport can represent counts exactly.

The June model retains optimization barriers after RMS variance, embedding
gather, and router sigmoid. These prevent demonstrated forward/AD rounding
changes caused by fusion. The model-only tests cover HF mappings, masks and
positions, forward/gradient agreement, stage ownership, and uneven partitions.
The wider GRPO migration additionally required a scorer-only compute-weight
cast boundary; its end-to-end numerical result should not be attributed to
these model changes alone. See the [numerical investigation](https://marina.oa.dev/echo/wiki/363).
