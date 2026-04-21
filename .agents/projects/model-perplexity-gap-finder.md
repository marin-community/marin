# Model Perplexity Gap Finder

## Problem

Levanter's current analysis path compares models only after they have been
tokenized with a single shared tokenizer. The existing compare-viz entrypoint in
[`lib/levanter/src/levanter/main/viz_logprobs.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/main/viz_logprobs.py#L34)
loads one tokenizer from `config.data.the_tokenizer` and uses one `LmConfig` for
both models
([`viz_logprobs.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/main/viz_logprobs.py#L54),
[`viz_logprobs.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/main/viz_logprobs.py#L123)).
That is fine for "same tokenizer, two checkpoints", but it cannot answer
"where is Marin worse than Llama 3.1?" once the models use different tokenizers.

Levanter already has the right aggregation idea for corpus slices: tagged eval
datasets with hierarchical rollups and per-tag `bpb`
([`lib/levanter/src/levanter/eval.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/eval.py#L199),
[`eval.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/eval.py#L387)).
Marin already defaults validation to Paloma plus uncheatable eval, but only in a
tokenizer-specific cached form
([`experiments/defaults.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/experiments/defaults.py#L297)).

For this feature we want a different path:

- take raw text corpora in the usual InputName-driven Marin style
- tokenize on the fly for each model separately
- compare models on a tokenizer-independent unit
- report both dataset-level gaps and byte-pattern / word-part gaps

No backward compatibility work is needed. Existing cached tokenization, `eval_lm`,
and `viz_logprobs` behavior should stay unchanged.

## Goals

- Compare two Levanter-loadable LMs, where each model may have its own tokenizer,
  its own `LmConfig`, and either an HF or native Levanter checkpoint.
- Score raw text documents directly and normalize results as bits per byte.
- Attribute loss deltas onto byte spans so reports can surface tokenization-free
  "word part" effects such as whitespace runs, punctuation clusters, or short
  literal byte spans.
- Reuse Marin's existing raw-dataset conventions and default to raw Paloma plus
  raw uncheatable eval.
- Produce a persisted report that is readable without W&B.

Non-goals:

- replacing `LmDataConfig` or the cache-based training/eval path
- supporting non-text dataset formats in v1
- unsupervised topic discovery or clustering
- exact token-to-token alignment across two tokenizers

## Proposed Solution

### Core approach

Introduce a new raw-text analysis path in Levanter that scores both models on the
same raw UTF-8 documents, but tokenizes each document independently per model.
Each model's per-token next-token loss is projected back onto the original
document bytes through tokenizer offset mappings. Once both models live on the
same byte axis, every report becomes an aggregation over byte-attributed losses.

This keeps the core invariant simple:

1. raw document bytes are the shared evaluation unit
2. model A and model B may tokenize differently
3. both models' losses are attributed onto those same bytes

### Config shape

Levanter gets a dedicated entrypoint and config rather than extending
`VizLmConfig`.

```python
@dataclass
class GapFinderModelConfig:
    checkpoint_path: str
    model: LmConfig | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str | None = None
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF


@dataclass
class GapFinderConfig:
    model_a: GapFinderModelConfig
    model_b: GapFinderModelConfig
    datasets: dict[str, DatasetComponent]
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_path: str = "gap-finder"
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
```

Marin gets a thin wrapper config that accepts raw datasets, converts them into
`DatasetComponent` values with `UrlDatasetSourceConfig` /
`HfDatasetSourceConfig`, then submits the Levanter job on Iris.

### Raw scoring loop

The raw path should not go through `LmDataConfig.validation_sets()` because that
method is cache- and tokenizer-oriented
([`lib/levanter/src/levanter/data/text/datasets.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/data/text/datasets.py#L817),
[`datasets.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/data/text/datasets.py#L826)).
Instead, the new entrypoint should iterate raw shards via
`DatasetComponent.source.get_shard_source("validation")`, read `text` from
`TextLmDatasetFormat`, tokenize batches on the host, and feed padded arrays into
the model.

The forward pass should reuse the standard next-token loss path rather than
custom logits math:

```python
@hax.named_jit(axis_resources=compute_axis_mapping)
def compute_token_losses(model: LmHeadModel, batch: LmExample):
    model = inference_mode(model, True)
    model = mp.cast_to_compute(model)
    per_pos_loss = model.compute_next_token_loss(
        batch,
        reduction=None,
        reduction_axis=(),
    ).array
    target_ids = jnp.roll(batch.tokens.array, -1, axis=-1)
    return per_pos_loss, batch.loss_weight.array, target_ids
```

### Byte attribution

For each raw document:

1. tokenize with offsets using the model's HF tokenizer
2. add BOS/EOS manually when the tokenizer would not insert them itself
3. score padded chunks up to `max_eval_length`
4. shift losses onto target-token spans, mirroring Levanter eval's target-id
   handling
5. spread each token's loss uniformly across its covered byte span

Uniform byte spreading is the simplest stable attribution rule. It preserves
document-level `bpb`, avoids token-to-token alignment, and lets us aggregate by
arbitrary byte-derived patterns later.

### Report structure

The report should include:

- dataset / subcorpus gap table (`model_a_bpb`, `model_b_bpb`, `gap_bpb`)
- hierarchical rollups for names like `paloma/...`
- top documents by positive and negative delta
- pattern-bucket gap table, with buckets such as:
  - `whitespace/single_space`
  - `whitespace/multi_space`
  - `whitespace/newline`
  - `whitespace/tab_or_cr`
  - `text/url`
  - `text/number`
  - `text/punctuation`
  - `text/non_ascii`
  - `text/word`
- top literal byte spans / short substrings with the largest deltas

Persist both JSON and HTML so downstream scripts can consume the data while
humans can inspect a single rendered report.

## Implementation Outline

1. Add a Levanter raw-text gap finder entrypoint, config types, model-loading
   helpers, and HTML/JSON report writer.
2. Add host-side raw text iteration, tokenizer-with-offset handling, and
   byte-attributed loss aggregation for text datasets.
3. Add a Marin wrapper plus helpers for raw evaluation components and default raw
   Paloma/uncheatable dataset wiring.
4. Add focused tests for byte attribution, bucket aggregation, and a tiny
   end-to-end Levanter run.
5. Add an experiment script that compares `marin-community/marin-8b-base` and
   `meta-llama/Meta-Llama-3.1-8B` on Iris v5p-8 in `us-central1`.

## Notes

- V1 should explicitly support `TextLmDatasetFormat` only. Chat/template-aware
  data can be added later once there is a clear raw-byte contract.
- Existing tagged eval code in
  [`lib/levanter/src/levanter/eval.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/eval.py#L538)
  is still the right model for hierarchical corpus aggregation; the new path just
  computes those aggregates from raw byte-attributed records instead of from a
  shared-tokenizer dataset.
- The existing `byte_length_of_token()` helper
  ([`lib/levanter/src/levanter/utils/hf_utils.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/levanter/src/levanter/utils/hf_utils.py#L23))
  remains useful for sanity checks, but offset-based byte attribution is the
  source of truth for mixed-tokenizer comparison.
- `save_logprobs.py`
  ([`lib/marin/src/marin/evaluation/save_logprobs.py`](/Users/dlwh/.codex/worktrees/a2ab/marin/lib/marin/src/marin/evaluation/save_logprobs.py#L85))
  is a useful reference for how to gather per-token outputs on TPU, but the gap
  finder should not serialize full token streams for both models by default.
- The default raw validation helper should mirror the current tokenized helper's
  dataset coverage so the new tool can be dropped into existing analysis flows.

## Future Work

- support `ChatLmDatasetFormat` and template-rendered raw comparisons
- add optional W&B artifact logging for the HTML report and summary JSON
- richer byte-pattern discovery beyond the fixed interpretable buckets
- support approximate context-preserving chunk transitions for very long
  documents instead of dropping the first-token loss in each chunk
