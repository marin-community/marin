# DPO Plan for Levanter (JAX/Haliax/Equinox)

This document is an implementation plan for adding Direct Preference Optimization (DPO) training to Levanter while reusing as much of the existing data + training stack as possible.

The guiding constraints:

- Reuse the existing cache/build pipeline (`ShardedDataSource` -> `BatchProcessor` -> `TreeCache`).
- Reuse the existing training loop (`Trainer`, `TrainerState`, `DataLoader`) with minimal changes.
- Reuse chat templating + masking logic (`ChatProcessor`, `{% generation %}` assistant masking).
- Keep the implementation modular: DPO is a new *objective* and a new *dataset format* (preference pairs), not a new model.

---

## Section 1: Levanter Prelude (Relevant Pieces + How They Interact)

This section summarizes the core Levanter components that matter for DPO: how raw data becomes token caches, how caches become training examples, how batching/sharding works, and how pretraining vs SFT is “the same thing” in Levanter.

### 1.1 Raw Data Sources: `ShardedDataSource`

Levanter’s ingestion story starts with a *sharded, seekable iterator abstraction*:

- `lib/levanter/src/levanter/data/sharded_datasource.py`
  - `ShardedDataSource[T]`:
    - `shard_names`: list of shard identifiers (file names, HF shards, etc.)
    - `open_shard_at_row(shard_name, row) -> Iterator[T]`: sequential iteration with a “skip to row” hook.
  - Concrete implementations:
    - `UrlDataSource` reads `.jsonl`, `.json`, `.parquet`, `.txt` via `fsspec`.
    - `WrappedHFDataSource` streams HF datasets via `datasets.load_dataset(..., streaming=True)` (often only one “data” shard unless iterable dataset supports sharding).

Why it matters for DPO:

- DPO datasets are typically JSONL or HF datasets that contain *preference pairs* (chosen/rejected).
- We want to keep the same “shardable preprocessing” model (distributed or local) so preference pairs can be cached exactly like other LM datasets.

### 1.2 Preprocessing: `BatchProcessor` and Deterministic, Resumable Caches

Levanter’s preprocessing is designed around:

- A stateless batch transform (`BatchProcessor`) that maps *raw examples* -> *tokenized/processed rows*.
- A persistent on-disk cache (`TreeCache`) that stores the processed rows as ragged arrays (token sequences) backed by Tensorstore.

Key code:

- `lib/levanter/src/levanter/data/_preprocessor.py`
  - `BatchProcessor`: callable on `Sequence[input]` returning batched outputs
  - `output_exemplar`: defines the schema of what’s written to cache
  - `metadata`: identifies transformations that change cache semantics (tokenizer, chat template, etc.)
- `lib/levanter/src/levanter/store/cache.py`
  - `build_or_load_cache(...)`: build if missing, otherwise load
  - `TreeCache`: read-only dataset abstraction over the cached store
  - Metadata ledger tracks processor metadata and shard completion state.
- `lib/levanter/src/levanter/store/tree_store.py`
  - `TreeStore`: stores a PyTree of `JaggedArrayStore` objects

Why it matters for DPO:

- We want to tokenize preference pairs once and store them in `TreeCache`, like SFT and pretraining.
- We want assistant-only masking to be cache-derived, not recomputed at runtime.
- We want preference data to support packing/truncation consistently with the rest of Levanter.

### 1.3 Dataset “Formats”: How Levanter Chooses Preprocessing and Example Construction

Levanter uses a small registry of “dataset formats” to decide:

1) which `BatchProcessor` to use when building caches, and
2) which `AsyncDataset` wrapper to use to yield model-ready training examples.

Key code:

- `lib/levanter/src/levanter/data/text.py`
  - `LmDatasetFormatBase` + concrete formats
    - `TextLmDatasetFormat` (`type: text`)
    - `ChatLmDatasetFormat` (`type: chat`)
  - `preprocessor_for_format(...)` selects `BatchTokenizer` or `ChatProcessor`
  - `dataset_for_format(...)` selects `CausalLmDataset` or `MultiturnChatDataset`

Important observation:

- Levanter’s `train_lm.py` is “objective-agnostic” at the data layer; it wants an `AsyncDataset` whose elements match the loss function signature.
- Today, `train_lm.py` assumes elements are `LmExample`, but the `Trainer` itself is generic over “example types”.

Why it matters for DPO:

- We will introduce a *new format* (preference pairs) that tokenizes two chat transcripts per row.
- We will introduce a *new example type* (a pair of `LmExample`s or equivalent tensors) for the DPO loss.

### 1.4 `LmExample`: The Unifying Representation for “What to Predict”

`LmExample` is the key unifier across pretraining and SFT.

Key code:

- `lib/levanter/src/levanter/models/lm_model.py`
  - `class LmExample(eqx.Module)` has:
    - `tokens: NamedArray` (typically axes: `(Batch, Pos)` or just `(Pos)`)
    - `loss_weight: NamedArray` (same axes as `tokens`)
    - `attn_mask: AttentionMask` (causal; optionally includes segment ids)
  - `LmExample.causal(...)` builds:
    - a causal loss mask (excludes last position)
    - optional ignore-id masking
    - optional segment-id based attention blocking

How pretraining vs SFT differ:

- Pretraining (`type: text`):
  - loss_weight is “all 1s except last token (and ignore tokens)”.
- Chat SFT (`type: chat`):
  - cache stores `assistant_masks` from the chat template
  - dataset converts `assistant_masks` -> `loss_weight` such that only assistant tokens contribute to loss
  - i.e. still next-token LM loss, but masked

Why it matters for DPO:

- DPO requires computing log-probabilities of the (chosen/rejected) *responses* given the prompt.
- Levanter already has the notion “these tokens count toward the objective” via `loss_weight`.
- So for DPO we can:
  - represent chosen transcript as an `LmExample` with assistant-only `loss_weight`
  - represent rejected transcript similarly
  - compute masked log-likelihood sums using existing loss primitives

### 1.5 Packing and Segment IDs

Levanter has built-in packing for variable-length docs:

- `GreedyPrepackedDataset` in `lib/levanter/src/levanter/data/packing.py`
  - Packs multiple documents into a single fixed-length sequence
  - Generates `segment_ids` so attention is blocked across packed boundaries
- Chat SFT uses this via:
  - `MultiturnChatDataset` in `lib/levanter/src/levanter/data/text.py`

Why it matters for DPO:

- DPO pairs are variable length, and naive padding wastes compute.
- We can reuse `GreedyPrepackedDataset` to pack multiple preference pairs per training example.
- But: DPO’s loss is computed *per pair*. If we pack multiple pairs into one “example”, we must compute the DPO loss per segment/pair (not one giant logistic loss over all packed tokens).

This will influence the dataset and loss design in Section 2/3.

### 1.6 Data Loading and Sharding: `DataLoader`

Levanter’s `DataLoader` is:

- batch-size scheduled
- sharding-aware (loads only data needed for local devices)
- async-prefetched
- CPU-side, and materializes per-step Global Device Arrays (GDAs)

Key code:

- `lib/levanter/src/levanter/data/loader.py`
  - `DataLoader(AsyncDataset[Ex], batch_size, mesh, axis_resources, ...)`
  - It calls `dataset.get_batch([...])` with global indices computed from the batch schedule + sharding.
  - It stacks per-example pytrees into a batch pytree using `stack_tree` (NamedArray-aware).

Why it matters for DPO:

- As long as our DPO dataset yields a pytree of arrays/NamedArrays (no Python objects per example),
  `DataLoader` will “just work”.
- DPO examples should be a clean `eqx.Module` / pytree like:
  - `DpoExample(chosen: LmExample, rejected: LmExample)`

### 1.7 Training: `Trainer`, `TrainerState`, and Loss Functions

Levanter’s trainer is generic:

- `lib/levanter/src/levanter/trainer.py`
  - `Trainer(loss_fn, optimizer)` runs `loss_fn(model, batch, ...)` inside JIT and computes gradients.
  - `Trainer.data_loader(...)` constructs `DataLoader`.
- `lib/levanter/src/levanter/trainer_state.py`
  - `TrainerState` supports trainable parameter filtering (`is_trainable`) and model averaging.

Important for DPO:

- We can keep the same `Trainer` and provide a new loss function for DPO.
- The loss function can compute:
  - policy logp(chosen) and logp(rejected)
  - reference logp(chosen) and logp(rejected)
  - then the DPO logistic loss

Crucial pitfall to avoid:

- Do not include the reference model in the differentiated parameter tree unless we stop gradients properly.
- Otherwise, `eqx.filter_value_and_grad` will compute gradients for reference params and blow up memory/compute.

We’ll address reference model handling in Section 3.

---

## Section 2: DPO Data Format and Dataset Construction (Chosen/Rejected Pairs)

This section describes how to add a DPO-friendly dataset format and how to turn preference-pair rows into model-ready batches while reusing Levanter’s existing chat templating and caching stack.

### 2.1 Input Example and Required Semantics

The raw preference row (example provided):

```json
{
  "chosen": [
    {"role": "user", "content": "how can i develop a habit of drawing daily", "name": null, "tool_calls": null, "tool_call_id": null},
    {"role": "assistant", "content": "Developing a daily habit of drawing can be challenging ...", "name": null, "tool_calls": null, "tool_call_id": null}
  ],
  "rejected": [
    {"role": "user", "content": "how can i develop a habit of drawing daily", "name": null, "tool_calls": null, "tool_call_id": null},
    {"role": "assistant", "content": "As an AI language model, I cannot personally develop habits for you ...", "name": null, "tool_calls": null, "tool_call_id": null}
  ],
  "hash": "84a0fcab17a7...",
  "prompt": "how can i develop a habit of drawing daily",
  "score_chosen": 8.5,
  "score_rejected": 8.5
}
```

Core assumptions for DPO:

- Each row contains a *pair* of sequences with the same prompt/context.
- The two sequences differ in (at least) the preferred completion.
- We want to compute the model’s log-likelihood only over the assistant tokens (or only the final assistant response; see note below).

Practical note:

- Most chat templates mark *all* assistant messages with `{% generation %}` tags.
- If the chosen/rejected transcripts share earlier assistant turns, those contributions cancel out in the logp difference, so it’s usually fine to include them.
- If a dataset might contain earlier turns that differ between chosen and rejected, consider an optional “prompt prefix consistency check” during preprocessing (debug-only).

### 2.2 Reuse the Existing Chat Template + Assistant Masking

We should keep the chat-template source of truth identical to SFT:

- Use `transformers` `tokenizer.apply_chat_template(...)`
- Require `{% generation %}` tags when `mask_user_turns=True`
- Cache `assistant_masks` for both chosen and rejected

Existing code we will reuse:

- `ChatProcessor` in `lib/levanter/src/levanter/data/text.py`
  - It:
    - reads a message list field (default `"messages"`)
    - optionally injects a system prompt
    - calls `apply_chat_template(..., return_assistant_tokens_mask=True)`
    - returns:
      - `input_ids: np.ndarray`
      - `assistant_masks: np.ndarray`

For preference pairs, we can compose two `ChatProcessor` instances:

- one pointing at the `chosen` field
- one pointing at the `rejected` field

This avoids re-implementing template logic and makes DPO “just like SFT” from the tokenizer’s perspective.

### 2.3 New Dataset Format: `type: preference_chat` (Plan)

We should add a new dataset format that lives alongside `text` and `chat` in `lib/levanter/src/levanter/data/text.py`.

Proposed API:

```python
@LmDatasetFormatBase.register_subclass("preference_chat")
@dataclass(frozen=True)
class PreferenceChatLmDatasetFormat(LmDatasetFormatBase):
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool = True
    mask_user_turns: bool = True
    # Optional: slice_strategy: Literal["left", "right", "raise", "drop"] = "left"
```

Notes:

- Keep naming consistent with existing `ChatLmDatasetFormat` where possible.
- `chat_template` override semantics should match chat format:
  - if provided, overrides tokenizer template
  - otherwise uses `tokenizer.chat_template`
- The `chat_template_kwargs` mechanism should work per-example just like `ChatProcessor` supports today.

YAML usage example:

```yaml
data:
  id: some/preference-dataset
  format:
    type: preference_chat
    chosen_field: chosen
    rejected_field: rejected
    pack: true
    mask_user_turns: true
    # Optional: slice_strategy: drop
  tokenizer: stanford-crfm/marin-tokenizer
  cache_dir: gs://.../tokenized/preference
```

### 2.4 New Preprocessor: `PreferenceChatProcessor` (Built from `ChatProcessor`)

Add a new `BatchProcessor` that emits *four* jagged sequences per input row:

- `chosen_input_ids`
- `chosen_assistant_masks`
- `rejected_input_ids`
- `rejected_assistant_masks`

Implementation sketch (reuse-focused):

```python
class PreferenceChatProcessor(BatchProcessor[dict, dict]):
    def __init__(..., chosen_field="chosen", rejected_field="rejected", ...):
        self._chosen = ChatProcessor(tokenizer, messages_field=chosen_field, ...)
        self._rejected = ChatProcessor(tokenizer, messages_field=rejected_field, ...)

    def __call__(self, batch):
        chosen_rows = self._chosen(batch)
        rejected_rows = self._rejected(batch)
        out = []
        for c, r in zip(chosen_rows, rejected_rows, strict=True):
            out.append({
                "chosen_input_ids": c["input_ids"],
                "chosen_assistant_masks": c["assistant_masks"],
                "rejected_input_ids": r["input_ids"],
                "rejected_assistant_masks": r["assistant_masks"],
            })
        return out
```

Why this is good:

- It reuses all of the complicated chat-template logic, system prompt injection, and per-example kwargs.
- Cache metadata naturally includes the chat template string via `ChatProcessor.metadata`.
- It keeps the cache schema extremely simple.

Where it plugs in:

- Extend `preprocessor_for_format(...)` in `lib/levanter/src/levanter/data/text.py`:
  - add a `case PreferenceChatLmDatasetFormat(...)` branch returning `PreferenceChatProcessor`.

### 2.5 Cache Layout for Preference Pairs

The cache directory will contain a tree of jagged arrays similar to chat caches, but doubled:

```
cache_dir/
  train/
    chosen_input_ids/...
    chosen_assistant_masks/...
    rejected_input_ids/...
    rejected_assistant_masks/...
    shard_ledger.json
  validation/
    ...
```

Benefits:

- All downstream datasets can assume these four fields exist.
- Packing/truncation can be performed uniformly across all four leaves using `GreedyPrepackedDataset` because:
  - each mask leaf has the same per-row length as its corresponding token leaf
  - the packing algorithm already supports PyTrees of jagged arrays with shared pack ranges

### 2.6 Dataset: `PreferencePairDataset` -> `DpoExample`

We need an `AsyncDataset` that yields something the loss function can consume.

Define a simple pytree type:

```python
class DpoExample(eqx.Module):
    chosen: LmExample
    rejected: LmExample
```

Then construct it by:

1) Using `GreedyPrepackedDataset` to turn jagged cache rows into fixed-length arrays.
2) Converting masks -> `loss_weight` exactly like SFT (roll by -1).
3) Creating two `LmExample.causal(...)` objects, one for chosen and one for rejected.

Implementation sketch:

```python
class PreferencePairDataset(MappedAsyncDataset[tuple[dict, dict], DpoExample]):
    def __init__(self, cache: TreeCache[dict], Pos: Axis, *, pack: bool, mask_user_turns: bool):
        max_segments = 64 if pack else 1
        packed = GreedyPrepackedDataset(cache.store.tree, Pos.size, max_segments_per_example=max_segments, slice_strategy="left")

        @eqx.filter_jit
        def _to_dpo_example(e):
            data, seg_ids = e

            def build_one(prefix):
                tokens = hax.named(data[f"{prefix}_input_ids"], Pos)
                seg = hax.named(seg_ids[f"{prefix}_input_ids"], Pos)
                if mask_user_turns:
                    am = data[f"{prefix}_assistant_masks"]
                    lw = hax.named(jnp.roll(am, -1, axis=-1), Pos)
                else:
                    lw = None
                return LmExample.causal(tokens=tokens, loss_weight=lw, segment_ids=seg)

            return DpoExample(chosen=build_one("chosen"), rejected=build_one("rejected"))

        super().__init__(packed, _to_dpo_example)
```

#### Packing vs no packing

We should implement packing support, but also make it easy to disable:

- `pack: false`:
  - `max_segments_per_example=1`
  - each training example corresponds to exactly one preference pair
  - simplest loss implementation (no per-segment accounting)
- `pack: true`:
  - packs many preference pairs into one fixed-length example
  - requires per-segment DPO reduction in the loss function (Section 3.3)

Given the complexity tradeoff, plan this as:

1) v1: ship with `pack: false` as the default (safe correctness baseline)
2) v2: enable `pack: true` with correct per-segment accounting

(You can still keep the code path unified by always using `GreedyPrepackedDataset`; just set `max_segments_per_example`.)

### 2.7 Shuffling, Mixtures, Validation

We can reuse `SingleDatasetLMConfigBase` and `LMMixtureDatasetConfig` almost unchanged:

- they already handle:
  - cache build/load
  - shuffle (`PermutationDataset` / `EraShufflingDataset`)
  - mixtures (`MixtureDataset`) with constant or scheduled weights

But we should enforce a DPO-specific constraint in the DPO training entrypoint:

- For DPO training, all configured datasets must be `type: preference_chat`.
- Mixture is allowed only across preference datasets.

Implementation approach:

- Add a helper validator in the DPO training script:
  - walk `config.data.sources` and ensure `source.format` is `PreferenceChatLmDatasetFormat`
  - fail fast with a helpful error

---

## Section 3: DPO Training Implementation Plan (Two Options)

This section covers the DPO loss, how to compute it efficiently with Levanter’s model/loss utilities, and two integration approaches:

1) a dedicated DPO entrypoint (`train_dpo.py`) and
2) integrating into `train_lm.py` behind an “objective” switch.

### 3.1 DPO Objective Refresher (What We Need to Compute)

Given a prompt `x` and two responses `y+` (chosen) and `y-` (rejected):

- Policy model: `pi_theta`
- Reference model: `pi_ref` (frozen)

Define:

- `logp_pi(y|x) = sum_{t in response tokens} log pi_theta(t | prefix)`
- `logp_ref(y|x) = sum_{t in response tokens} log pi_ref(t | prefix)`

DPO per-example loss (canonical form):

```
delta_pi  = logp_pi(y+|x)  - logp_pi(y-|x)
delta_ref = logp_ref(y+|x) - logp_ref(y-|x)
logit = beta * (delta_pi - delta_ref)
loss = -log(sigmoid(logit)) = softplus(-logit)
```

How this maps onto Levanter:

- `logp_pi(y|x)` can be computed via per-token next-token NLL with `loss_weight` selecting assistant tokens:
  - `nll_per_tok = model.compute_next_token_loss(example, reduction=None)`
  - since NLL is `-logp`, `logp_per_tok = -nll_per_tok`
  - `logp_sum = sum(logp_per_tok)` over the `Pos` axis
- Reference model uses the same computation, but must not receive gradients.

### 3.2 Implementing `logp_sum` Reusing Existing Loss Machinery

We should **not** write custom “gather logsoftmax” code if we can avoid it; Levanter already has correct, fused implementations:

- `LmHeadModel.compute_next_token_loss(...)` -> `maybe_fused_next_token_loss(...)`
- Handles:
  - shifting targets by one token
  - excluding last token loss
  - applying `loss_weight`
  - optional logsumexp penalty (z-loss)

For DPO:

- Use `reduction=None` to get per-token NLL.
- Use `loss_weight` coming from assistant masks (like SFT) to restrict to assistant tokens.
- Then:
  - `logp_sum = -hax.sum(nll_per_tok, axis=Pos)`

Potential subtlety:

- Ensure `loss_weight` is aligned to the predicted token positions.
- Existing chat dataset does:
  - `mask = assistant_masks`
  - `mask = jnp.roll(mask, -1)` (shift so loss at position i corresponds to token i+1)
  - then `LmExample.causal(...)` multiplies by the “not last token” mask
- Reuse that exact approach for DPO examples.

### 3.3 Packing-Compatible Per-Pair Reduction (If `pack: true`)

If we pack multiple preference pairs into a single sequence, we must compute:

- `logp_sum` per *segment/pair*, not per packed sequence.

Levanter already has segment-aware utilities in `lib/levanter/src/levanter/data/packing.py`, but:

- `per_segment_loss(...)` currently assumes a single example (1D Pos) and uses `jnp.unique` to find segment ids.
- In training, we’ll have a `Batch` axis, so we need a batched segment reducer.

Plan:

1) Add a new helper (either in `packing.py` or a new small module) that computes per-segment sums and supports a batch axis.
2) Use it to compute:
   - `logp_sum_pi_chosen[Batch, Segments]`
   - `logp_sum_pi_rejected[Batch, Segments]`
   - same for reference
3) Compute DPO loss per (Batch, Segment) and average over valid segments.

Proposed helper API (batch-friendly):

```python
def per_segment_sum_batched(
    *,
    values: hax.NamedArray,        # axes: (Batch, Pos)
    segment_ids: hax.NamedArray,   # axes: (Batch, Pos), int32, -1 for padding
    Segments: hax.Axis,            # max segments per example
    Pos: hax.Axis,
) -> tuple[hax.NamedArray, hax.NamedArray]:
    # returns (unique_segment_ids, sums)
    # unique_segment_ids: (Batch, Segments)
    # sums: (Batch, Segments)
```

Implementation approach:

- Use `jax.vmap` over the batch axis around the existing 1D logic:
  - `jnp.unique(..., size=Segments.size, fill_value=-1)` per example
  - build a `(Segments, Pos)` mask and dot with values

This is intentionally simple and predictable; it will compile and work correctly for small `Segments` (e.g. 1..64).

Then DPO reduction:

```python
valid = unique_segment_ids != -1
loss_seg = softplus(-beta * ((logpC - logpR) - (logpC_ref - logpR_ref)))
loss = sum(loss_seg * valid) / sum(valid)
```

### 3.4 Reference Model Handling (No Grad, No Optimizer State)

We need `pi_ref` for DPO.

Requirements:

- Reference model is frozen (no gradients, no optimizer state).
- It should run in eval mode (dropout disabled).
- It should use the same tokenizer and vocab axis.

Two realistic choices:

#### Option 1 (Preferred): Keep reference model out of the differentiated parameter tree

- Policy model is `state.model` as usual.
- Reference model is loaded once and captured in the loss function closure.
- Since the reference does not depend on the policy parameters, gradients flow only through policy computations.

Pros:

- Avoids accidental gradient computation through reference parameters.
- Avoids optimizer state for reference.
- Minimizes backward-pass memory.

Cons:

- The reference model is still a full model copy in device memory.
- Changes to reference model require recompilation (but reference is fixed anyway).

#### Option 2: Put reference model in state but explicitly stop-grad (not recommended for v1)

- Store both `policy` and `reference` in a wrapper model.
- Set `is_trainable` mask to mark reference params non-trainable.
- Also apply `jax.lax.stop_gradient` on reference outputs to ensure no grad is computed.

This is riskier because it’s easy to accidentally cause reference grads or checkpoint weirdness.

Recommendation:

- Implement Option 1 first.

### 3.5 Loss Function Structure (Efficient Forward Passes)

Naively, DPO requires 4 forward passes per batch:

- policy(chosen), policy(rejected), ref(chosen), ref(rejected)

We can reduce this to 2 forward passes per batch by concatenating chosen+rejected:

Plan:

1) Build a “pair axis”:
   - `Pair = hax.Axis("pair", 2)`
2) Stack chosen and rejected `LmExample`s along `Pair`.
3) Run one forward for the policy and one forward for the reference.
4) Split results back into chosen/rejected.

Implementation sketch (conceptual):

```python
def dpo_loss(policy_model, dpo_batch: DpoExample, *, key):
    Pair = hax.Axis("pair", 2)
    # stack tokens / loss_weight / attn_mask along Pair
    both = stack_lm_examples(Pair, dpo_batch.chosen, dpo_batch.rejected)

    # policy logps
    nll_pol = policy_model.compute_next_token_loss(both, reduction=None, key=key)   # axes: (Pair, Batch, Pos)
    logp_pol = -hax.sum(nll_pol, axis=Pos)                                          # axes: (Pair, Batch)

    # reference logps (eval mode, key=None)
    nll_ref = ref_model.compute_next_token_loss(both_ref, reduction=None, key=None)
    logp_ref = -hax.sum(nll_ref, axis=Pos)

    # split
    logp_pol_chosen = logp_pol[{Pair: 0}]
    logp_pol_rejected = logp_pol[{Pair: 1}]
    ...
```

Important notes:

- `stack_lm_examples(...)` needs to correctly stack `AttentionMask` / segment ids.
  - This can be done with `jax.tree_map` and `hax.stack` for NamedArrays.
  - Alternatively (simpler v1): do two forwards and skip stacking.
- Microbatching (`TrainerConfig.microbatch_size`) should still work as long as the batch axis retains its configured name.

### 3.6 Metrics to Log (Minimal but Useful)

At minimum:

- `train/dpo_loss`
- `train/dpo_margin_policy = mean(delta_pi)` (optional)
- `train/dpo_margin_ref = mean(delta_ref)` (optional)
- `train/dpo_accuracy = mean(delta_pi - delta_ref > 0)` or `mean(logit > 0)`

These are cheap and help sanity check learning.

### 3.7 Plan A: Add a Dedicated `train_dpo.py` Entrypoint (Recommended)

This is the clean, modular approach: DPO is not “language modeling”, it’s a separate objective that happens to use the same LM architecture.

#### File additions/changes

- Add dataset format + processor + dataset:
  - `lib/levanter/src/levanter/data/text.py`
    - `PreferenceChatLmDatasetFormat`
    - `PreferenceChatProcessor`
    - `PreferencePairDataset` (yields `DpoExample`)
    - Update `preprocessor_for_format(...)` and `dataset_for_format(...)` to include the new format.
- Add a new training entrypoint:
  - `lib/levanter/src/levanter/main/train_dpo.py`
    - `TrainDpoConfig` dataclass
    - model + tokenizer initialization (reuse patterns from `train_lm.py`)
    - reference model loading (HF or Levanter checkpoint)
    - DPO loss function
    - standard Trainer hooks / checkpointing
- Add docs:
  - `lib/levanter/docs/reference/Data-Formats.md`: document `preference_chat` format + fields
  - Potentially a new doc page: `lib/levanter/docs/Fine-Tuning-DPO.md`
- Add tests:
  - `lib/levanter/tests/test_dpo_processor.py`
  - `lib/levanter/tests/test_dpo_loss.py`

#### `TrainDpoConfig` (proposed)

```python
@dataclass
class TrainDpoConfig:
    data: SingleDatasetLMConfig | LMMixtureDatasetConfig
    trainer: TrainerConfig
    model: LmConfig
    optimizer: OptimizerConfig

    reference_model_path: str
    reference_is_hf: bool = True

    beta: float = 0.1
    train_seq_len: int | None = None

    # optional knobs
    label_smoothing: float = 0.0
    pack: bool | None = None  # if set, override dataset format pack
```

We should also copy useful fields from `TrainLmConfig`:

- `initialize_from_hf`, `use_hf_model_config`
- HF checkpoint save callback options (`hf_save_path`, etc.)

#### Correctness guardrails

- Validate all dataset formats are `preference_chat` in `main(...)`:
  - fail fast if user accidentally mixes in `text`/`chat`
- Validate tokenizer has a chat template or config provides one
- For `mask_user_turns=True`, require `{% generation %}` in template (already enforced by `ChatProcessor`)

### 3.8 Plan B: Integrate DPO Into `train_lm.py` Behind an Objective Switch

This is more invasive but provides one entrypoint.

#### Proposed config change

- Extend `TrainLmConfig` in `lib/levanter/src/levanter/main/train_lm.py`:

```python
objective: Literal["lm", "dpo"] = "lm"

dpo:
  beta: 0.1
  reference_model_path: ...
  reference_is_hf: true
```

#### Code behavior

- If `objective == "lm"`:
  - preserve existing behavior exactly
- If `objective == "dpo"`:
  - require dataset format is `preference_chat`
  - build `PreferencePairDataset` instead of standard LM dataset
  - swap `loss_function` implementation to DPO
  - load reference model

#### Pros / cons

Pros:

- One entrypoint for “LM-ish training”.
- Reuses all `train_lm.py` features (eval hooks, HF save hooks, mixture weight logging).

Cons:

- `train_lm.py` becomes significantly more complex and less readable.
- Harder to reason about which configs are valid for which objective.
- Easy to accidentally allow mixed datasets/objectives (nonsense runs).

Recommendation:

- Prefer Plan A for cleanliness, then optionally add Plan B if there’s strong UX demand.

### 3.9 Testing Plan (Minimal, High Signal)

#### Test 1: Processor produces aligned outputs

- Build a tiny batch with two rows where chosen/rejected share the same user prompt but different assistant strings.
- Run `PreferenceChatProcessor` and assert:
  - `chosen_input_ids` and `chosen_assistant_masks` lengths match
  - same for rejected
  - assistant masks contain at least one `1`

#### Test 2: Dataset produces `DpoExample` with correct masking semantics

- Build a temporary cache using `SerialCacheWriter` (no Ray) with the processor output.
- Construct `PreferencePairDataset(..., pack=False, Pos=Axis("position", 64))`.
- Get one example and assert:
  - `chosen.tokens.axes == (Pos,)` (or includes batch depending on wrapping)
  - `chosen.loss_weight` has zeros in user tokens and ones on assistant tokens (roughly; template-dependent)

#### Test 3: DPO loss monotonicity on a toy model

- Use a tiny randomly initialized model and construct a single DPO batch.
- Manually tweak chosen vs rejected tokens so chosen has higher logp under the model (e.g., identical sequences except one token that is made “easier”).
- Assert:
  - DPO loss decreases when `delta_pi` increases (holding ref fixed).

(This can be done with a small “fake model” that returns deterministic logits if needed.)

### 3.10 Documentation Updates

- Extend `lib/levanter/docs/reference/Data-Formats.md` with a new section:
  - `preference_chat` format
  - required fields
  - chat template requirements (`{% generation %}`)
  - packing behavior
- Add a training guide:
  - how to run `python -m levanter.main.train_dpo --config_path ...`
  - example config for a known preference dataset

---

## Appendix: Implementation Checklist (Suggested Order)

1) Add `PreferenceChatLmDatasetFormat` and `PreferenceChatProcessor` (reuse `ChatProcessor` twice).
2) Add cache construction support via `preprocessor_for_format`.
3) Add `PreferencePairDataset` that yields `DpoExample` (start with `pack: false`).
4) Implement `train_dpo.py` with:
   - policy model initialization (reuse `train_lm.py` patterns)
   - reference model loading (HF path)
   - DPO loss function (naive 4-forward version)
5) Add basic tests (processor + dataset + loss sanity).
6) Optimize loss to 2 forwards via stacking chosen/rejected.
7) Add packing support:
   - implement batched per-segment sum reducer
   - compute DPO loss per segment/pair
8) Add docs and example config(s).

---

## Status Update (2026-01-22)

### What’s Implemented

**Levanter**
- Added `preference_chat` data format, `PreferenceChatProcessor`, `PreferencePairDataset`, and `DpoExample`, plus registry wiring in `preprocessor_for_format` / `dataset_for_format`.
- Implemented `train_dpo.py` with `TrainDpoConfig`, DPO loss, reference model loading, and strict dataset validation:
  - **Rejects** packed preference datasets.
  - **Defaults** to `slice_strategy: "raise"` (errors on over-length sequences) unless you opt into slicing
    or `slice_strategy: "drop"` on the `preference_chat` format.
- Added validation splitting from the training cache (`validation_split_fraction`, default `0.1`) and eval hooks that log validation loss/metrics.
- Added DPO unit tests (`test_dpo_processor.py`, `test_dpo_loss.py`).
- Tokenization now **skips invalid rows** with empty chosen/rejected conversations and logs a loud warning for DPO.

**Marin**
- Added DPO training plumbing in `lib/marin` (`TrainDpoOnPodConfig`, `run_levanter_train_dpo`) and exported it.
- Added `SimpleDPOConfig` + `default_dpo` in `experiments/defaults.py`, including validation split plumbing.
- Added experiment `experiments/exp2101_dpo_ultrafeedback.py` with explicit `validation_split_fraction=0.1`.
- Added Levanter DPO config `lib/levanter/config/dpo_ultrafeedback_llama3_8b.yaml` (all **us-central1** paths, `trainer.ray.auto_start_cluster: false`).

### Validation Performed
- `uv run python infra/pre-commit.py --all-files` passes (ruff/black/pyrefly).
- Levanter tests run by user: **802 passed, 42 skipped, 21 deselected**.

### Remaining / Follow‑Ups
- Replace `<hash>` placeholders in `config/dpo_ultrafeedback_llama3_8b.yaml` with actual GCS output paths.
- Decide whether to support **packing** for DPO (currently rejected) and implement if needed.
- Add doc updates in `lib/levanter/docs/reference/Data-Formats.md` and a DPO training guide (see Section 3.10).
- Optionally wire `test_prefs` as an explicit validation set instead of splitting train.
- Re-run tokenization and DPO training now that invalid preference rows are skipped with warnings.

### Gotcha: vmap + auto-sharding during model init (root cause & fix)

**Symptom (seen with `config/dpo_tiny_gpt2.yaml` on a 4‑device mesh):**
```
ValueError: Sharding ... spec=PartitionSpec('data', None, 'model', None) implies that array axis 0 is partitioned 4 times,
but the dimension size is 5 (full shape: (5, 32, 3, 4, 8))
```

**Why this happens (the subtle Haliax/JAX interaction):**
1) `Gpt2Transformer.init` builds a stacked set of blocks using `Stacked.init(...)`, which internally calls `haliax.vmap(module.init, Block)`.
2) Inside each block init we call `hax.random.truncated_normal(...)`, which returns a `NamedArray` and immediately calls `haliax.auto_sharded(...)`.
3) `haliax.vmap` temporarily wraps `NamedArray` outputs in a **_PassiveNamedArray** so it can insert the new **layer axis** after the vmap finishes.
   - During the vmap body, the underlying JAX array has an extra leading **batch dimension** (size = `num_layers`), **but the NamedArray axes do not include that axis yet**.
4) `auto_sharded` uses the **NamedArray axes** to compute a `PartitionSpec` and then calls `jax.device_put` with that spec.
   - The spec is built as if the array has shape `(embed, qkv, heads, head_size)`.
   - The **real array** at that moment has shape `(layers, embed, qkv, heads, head_size)`.
5) Because the mapping includes `embed -> data` (data axis size 4), JAX tries to shard **axis 0** of the real array (the hidden vmap batch axis, size 5).
   - This yields the “axis 0 is partitioned 4 times but size is 5” error.

**Key insight:** The failure is not in DPO itself—it is a generic Haliax sharding + `vmap` timing issue. The `NamedArray` metadata lags the real array shape during the vmap body, so sharding occurs one axis too early.

**Fix applied:**
- **Skip `device_put` inside `auto_sharded` if the underlying array is a vmap‑batched tracer** (detected via `.batch_dim`).
  This avoids attempting to shard while the vmap batch axis is still “invisible” to `NamedArray` axes.
- **Apply `auto_sharded` after `haliax.vmap` completes** in `Stacked.init`, so the `layers` axis is present in the `NamedArray` and sharding can safely align with the true array shape.

This preserves the correctness of NamedArray semantics and avoids any silent shape mismatches. It also keeps the model initialization path consistent across DPO and non‑DPO configs.

**Regression test added (DPO suite):**
- `test_vmapped_init_with_sharding_handles_layer_axis` spawns a fresh 4‑CPU JAX process, builds a 4‑axis mesh, runs `haliax.vmap` over a `truncated_normal` init, and asserts it succeeds.
  This mirrors the exact failure mode seen in `dpo_tiny_gpt2.yaml` and ensures the fix stays in place.

### Gotcha: Dropout key not forwarded in DPO logp computation

**Symptom (during `train_dpo.py` on configs with non‑zero dropout, e.g. GPT‑2 embed_pdrop):**
```
RuntimeError: Dropout requires a key when running in non-deterministic mode.
```
This occurs inside `_logp_sum` when the policy model computes next‑token loss:
`loss_function -> _logp_sum(policy) -> compute_next_token_loss -> embeddings.dropout`.

**Why this happens (subtle but easy to miss):**
- The DPO loss function *already* receives a PRNG key from the trainer.
- We **split** it for chosen/rejected (`key_chosen`, `key_rejected`) and pass those into `_logp_sum`.
- However, `_logp_sum` **ignored its `key` argument**, calling:
  `model.compute_next_token_loss(..., key=None)`.
- When dropout is active (`embed_pdrop > 0` or similar), Haliax’s dropout layer **requires a key** unless the model is in inference mode.
- The reference model is explicitly wrapped in `inference_mode(..., True)` so it is fine with `key=None`.
  The **policy model is in training mode**, so passing `None` causes the runtime error.

**Fix applied:**
- Thread the key through in `_logp_sum`:
  - `compute_next_token_loss(..., key=key)` so dropout receives a real PRNG key.
- Reference logps still pass `key=None` (because reference model stays in inference mode).

**Regression test added:**
- `test_logp_sum_passes_key_for_dropout` builds a tiny GPT‑2 with `embed_pdrop=0.1`,
  constructs a small `LmExample`, and calls `_logp_sum(..., key=PRNGKey)`.
  This would have thrown the dropout error previously and now validates the fix.

### Gotcha: `pspec_for` + filtered `NamedArray` (array=None) during JAXPR shape plumbing

**Symptom (seen while running `train_dpo.py` via `launch.py`, during JAXPR capture):**
```
ValueError: Shape of underlying array () does not match number of axes ('layers', 'embed'). None
```
The traceback points at `haliax.partitioning.pspec_for` during `named_jit` output sharding inference.

**Why this happens (the Equinox/Haliax interaction):**
1) `named_jit` calls `eqx.filter_eval_shape` so it can infer output sharding from the **shape‑only** result.
2) `eqx.filter_eval_shape` produces a PyTree of `ShapeDtypeStruct` leaves for arrays.
3) Equinox utilities like `eqx.filter(..., lambda _: False)` are used internally to zero out
   “non‑checkpointed” or “static” leaves when combining/merging model state.
4) When `eqx.filter` hits a `NamedArray`, it preserves the **metadata** (axis names) but replaces the **array** with
   `None`, yielding `NamedArray(???('layers','embed'), None)`.
5) `pspec_for` previously unconditionally did `node.axes`, which calls `NamedArray.axes`. That method
   reads `jnp.shape(self.array)` and **blows up when `array` is None**, producing the error above.

This is subtle because it only happens during *shape‑only* plumbing (JAXPR capture or state initialization);
the real arrays are fine. But the partitioning code was too strict about assuming every `NamedArray`
actually had a backing array.

**Fix applied:**
- In `pspec_for`, detect `NamedArray` leaves with `array` that is **not array‑like** (i.e., `None`) and
  return `None` for their partition spec. This means “no explicit sharding” for those filtered leaves,
  which is correct for shape‑only placeholders.

**Regression test added:**
- `test_pspec_for_handles_filtered_namedarray` constructs a `NamedArray`, runs `eqx.filter(..., lambda _: False)`
  to produce the `array=None` case, and asserts `pspec_for(...)` returns `None` without error.
  This recreates the exact failure path seen during DPO runs and prevents regressions.

### Gotcha: `partition_for_grad_overwrite` produced `NamedArray(array=None)` and corrupted the model

**Symptom (during DPO eval hooks after step 1):**
```
ValueError: eval_model contains NamedArray placeholders with array=None
```
The error triggered inside `TrainerState.eval_model` when evaluation hooks tried to read the model.
Training appeared to run for one step and then crashed as soon as eval began.

**Root cause (subtle Equinox + Haliax interaction):**
1) `TrainerState.take_step` always calls `partition_for_grad_overwrite(...)` to split gradient‑overwrite state.
2) `partition_for_grad_overwrite` uses `eqx.partition` with `is_leaf` **only** for `OverwriteWithGradient`.
3) `NamedArray` is a PyTree (array leaf + axis_names aux), so `eqx.partition` **recurses into it**.
4) Because the predicate is false for `NamedArray`, the “overwrite” tree is reconstructed as
   `NamedArray(array=None, axis_names=...)` — a placeholder that looks non‑None.
5) `apply_updates` treats any non‑None overwrite as authoritative and **replaces** the real parameter with the
   placeholder. That silently corrupts the model.
6) The model still “works” for the remainder of that step, but the next use (eval hook) catches the missing arrays.

**Fix applied:**
- Treat `NamedArray` as a **leaf** in `partition_for_grad_overwrite`, alongside `OverwriteWithGradient`.
  This ensures that non‑overwrite parameters get `None` in the overwrite tree (instead of a `NamedArray(None)`),
  so `apply_updates` keeps the real model weights.

**Regression test added (DPO suite):**
- `test_partition_for_grad_overwrite_preserves_namedarrays` builds a tiny `NamedArray`, runs
  `partition_for_grad_overwrite`, then calls `apply_updates`. The test asserts the resulting
  `NamedArray` still has a real backing array and equals the original values.

### Gotcha: HF checkpoint save tried to serialize `DpoModel`

**Symptom (during `train_dpo.py` HF save at step 859 on 2026-01-23):**
```
AttributeError: 'DpoModel' object has no attribute 'config'
```
The traceback originates in `hf_checkpoints._build_hf_config_dict`, which expects `model.config`.

**Why this happens:**
- The HF save callback expects a `ModelWithHfSerializationMixin` (i.e., an `LmHeadModel`) so it can read
  `model.config` and `model.Vocab` and build a HF-compatible state dict.
- DPO training stores a wrapper `DpoModel(policy, reference)` in the trainer state.
- The callback uses `step.eval_model`, which is the wrapper, not the policy model.
- Even if we added `config` to `DpoModel`, saving would still be incorrect because the state dict would include
  *both* policy and reference parameters (with extra prefixes) and would not match HF naming.

**Fix applied:**
- The DPO HF save hook now unwraps and saves **only** the policy model:
  `step.eval_model.policy`.
- This provides the expected `config`/`Vocab` interface and avoids serializing the reference weights.

---

## Minimal Plan: Track Chosen/Rejected Rewards (Levanter)

Goal: log per-batch **chosen_reward** and **rejected_reward** in Levanter DPO, analogous to Tinker’s metrics,
with minimal code changes.

1) **Compute rewards inside `loss_function` in `lib/levanter/src/levanter/main/train_dpo.py`.**  
   Use the already-computed log-prob sums:
   - `chosen_reward = beta * (logp_pi_chosen - logp_ref_chosen)`
   - `rejected_reward = beta * (logp_pi_rejected - logp_ref_rejected)`

2) **Add the metrics to the dict returned by `loss_function`.**  
   Keep naming consistent with existing DPO metrics (recommended):
   - `dpo_chosen_reward`
   - `dpo_rejected_reward`

3) **Return the merged metric dict.**  
   No other code or config changes required.

Minimal code sketch (drop-in change in `loss_function`):

```python
loss, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=config.beta)

chosen_reward = (logp_pi_chosen - logp_ref_chosen) * config.beta
rejected_reward = (logp_pi_rejected - logp_ref_rejected) * config.beta

if isinstance(chosen_reward, hax.NamedArray):
    metrics["dpo_chosen_reward"] = Metric.from_value(hax.mean(chosen_reward).scalar(), ReductionType.MEAN)
    metrics["dpo_rejected_reward"] = Metric.from_value(hax.mean(rejected_reward).scalar(), ReductionType.MEAN)
else:
    metrics["dpo_chosen_reward"] = Metric.from_value(jnp.mean(chosen_reward), ReductionType.MEAN)
    metrics["dpo_rejected_reward"] = Metric.from_value(jnp.mean(rejected_reward), ReductionType.MEAN)

return loss, metrics
```
