# MoE doc-init-tokens: Research Logbook

## Scope
- Goal: Determine whether (a) packing whole documents, (b) zeroing loss on the first
  N tokens of each document, or (c) tanh-positional loss weighting improves MoE
  pretraining vs. the standard concat-and-split + uniform loss recipe.
- Primary metric: `eval/paloma/macro_loss` and the new `eval/uncheatable_eval/...`
  variants (see "Eval surface" below).
- Constraints: gates from `experiments/grug/moe/agent.md` (effective speedup at
  d512+d768; scaling-law projection at d1024+d1280).

## Baseline (V1)
- Code refs: `experiments/grug/moe/launch.py` (`baseline_moe`), `train.py`,
  `model.py`, `heuristic.py`.
- Numbers: see baseline table in `experiments/grug/moe/README.md`.

## Phase 1 (current): Eval-in-loop prototype
Wire up the new eval surface against the V1 baseline first; defer V2/V3/V4
training-side changes until the eval plumbing is verified.

### Eval surface
1. **Regular uncheatable eval** — already produced by the existing
   `TaggedEvaluator` over `eval/uncheatable_eval/*` validation sets. No code
   change needed beyond ensuring those datasets are wired in.
2. **Last-500-tokens uncheatable eval** — same packed validation data, but
   `loss_weight` is masked to 1 only on the last 500 tokens of each segment.
   Reported under `eval/uncheatable_eval_last500/...`.
3. **`mmlu_sl_verb`** (PR #5168) — 0-shot and 5-shot via Levanter's
   `lm_eval_harness` callback. Not via `evaluate_levanter_lm_evaluation_harness`
   (which requires HF export).

### Validation packing
- All validation components must use `pack=True` (greedy whole-doc packing,
  segment_ids per packed example).
- Implementation: helper that sets `component.pack=True` on
  validation-only components after `add_validation_sets_to_mixture`.
- Safe to mutate: validation-only components have `train_weight=0` and are
  not used by the training mixture.

### Plumbing required for phase 1
- `experiments/grug/moe/lm_eval_adapter.py` — `GrugLmHeadAdapter(eqx.Module)`:
  wraps grug `Transformer`, exposes `activations(NamedArray) → NamedArray` and
  `get_lm_head() → NamedArray["Embed","Vocab"]` plus `Vocab`/`Embed`/`Pos` axes.
  This is what `levanter.eval_harness._eval_loglikelihood` reads (see
  `lib/levanter/src/levanter/eval_harness.py:262-285`).
- `experiments/grug/moe/eval_helpers.py` — `LastNTokensDataset` mapping
  `AsyncDataset[GrugLmExample] → AsyncDataset[GrugLmExample]` with loss_weight
  re-masked to last N tokens of each segment using a scatter-max over
  segment_ids.
- `train.py` — extend `GrugEvalConfig` with `eval_harness:
  LmEvalHarnessConfig | None`, `eval_harness_steps: int`,
  `last_n_eval_tokens: int | None`, and `last_n_eval_dataset_filter`
  (which datasets to augment). Wire a `cb_lm_eval_harness_grug` callback
  that wraps `state.eval_model` in `GrugLmHeadAdapter` then calls
  `levanter.eval_harness._actually_run_eval_harness`.
- `launch.py` — `pack=True` on validation components,
  `mmlu_sl_verb` 0/5-shot in `eval_harness.task_spec`,
  `last_n_eval_tokens=500`.

## Variants — DEFERRED, do later

These are the ablation variants. Phase 1 (eval surface) is gating; once
phase 1 is solid, come back here.

### V2: Train-side `pack=True` (whole-document packing)
- **What changes**: training data components for the nemotron mix get
  `pack=True`. Currently they default to `pack=False` ⇒ `CausalLmDataset` ⇒
  concat-and-split. With `pack=True` ⇒ `PackedTokenDataset` ⇒
  `GreedyPrepackedDataset` packs whole documents greedily into seq_len with
  segment_ids.
- **Where**: helper in `experiments/grug/moe/data_helpers.py` that mutates
  the training-set components on `nemotron_mix_block_shuffle` (or analogous)
  to set `pack=True`.
- **Subtlety**: max_segments_per_example default is 64. Verify nemotron mix
  packing density at seq_len=4096 — average doc length matters for whether
  64 is enough.
- **Subtlety**: padding fraction. Whole-doc packing wastes some tokens at
  the tail of each packed example. Track `data/stats/train/.../padding_fraction`.

### V3: Zero out loss on first N tokens of each new doc
- **What changes**: keeps concat-and-split (`pack=False`). In the training
  loss, mask `loss_weight = 0` for tokens whose
  `position_within_doc < N`.
- **Where**: in `_make_train_step` (or in the dataset wrapper, cleaner). The
  `GrugLmExample.causal()` already builds `segment_ids` from EOS when
  `eos_id` is provided and `block_cross_document_attention=True`. We add a
  post-processing step that zeroes the first N positions of each segment.
- **Default N**: 4 (init-tokens literature uses 4–8; conservative end).
- **Tokens-counted-toward-loss accounting**: zeroing reduces effective
  training tokens by ~`N * num_docs / total_tokens`. For nemotron at
  seq_len=4096 and avg doc length D, that's ≈ `N/D`. Need to either
  compensate via more steps, or accept fewer effective tokens.

### V4: tanh(position_within_doc / 10) loss weighting
- **What changes**: keeps concat-and-split. Multiply loss_weight by
  `tanh(position_within_doc / 10)`. So token 0 gets weight ~0, token 10 gets
  ~0.76, token 30+ gets ≈1.
- **Where**: same place as V3. Compute position_within_doc from
  segment_ids (`pos - first_pos_in_segment`), then `loss_weight *= tanh(p/10)`.
- **Subtlety**: this is a soft V3 — the ramp is smoother and there's no hard
  cutoff. Differentiable in the position parameter (if we ever want to
  hyperparam-sweep the divisor `10`).
- **Token accounting**: total effective loss weight is `sum(tanh(p/10))` per
  segment, ≈ `D - 10` for long docs. So similar to V3 with N≈10 in the
  asymptotic limit.

### Shared helper for V3/V4
Write `position_in_segment(segment_ids: jax.Array) → jax.Array`:
```python
def position_in_segment(segment_ids):
    # segment_ids: (seq_len,), monotonically non-decreasing.
    # Returns: position within current segment (0-indexed).
    seq_len = segment_ids.shape[0]
    is_new_segment = jnp.concatenate(
        [jnp.array([True]), jnp.diff(segment_ids) != 0]
    )
    # cumulative count since last True
    # = i - last_true_index_at_or_before_i
    indices = jnp.arange(seq_len)
    last_seg_start = jnp.maximum.accumulate(jnp.where(is_new_segment, indices, -1))
    return indices - last_seg_start
```

Same primitive backs the last-500 eval mask: for the eval, use
`position_from_end_of_segment` instead.

### Compute scale plan
- Phase 1 (eval prototype): single d512 V1 run, just to verify wiring.
- Gate 1 (V2/V3/V4): d512 (2.19e17) + d768 (1.70e18). Effective speedup
  required at both scales.
- Gate 2 (V2/V3/V4): d1024 (9.00e18) + d1280 (2.83e19). Plus all four
  scales must combine into a scaling-law projection beating baseline at 1e21
  and 1e23.

## Experiment Log

### 2026-04-29 — phase 1 plan locked in
- Hypothesis: eval surface (last-500 + mmlu_sl_verb) is plumbing-only;
  V1 macro_loss numbers should match the existing baseline.
- Next action: implement adapter + callback + last-N dataset, then launch
  d512 V1 with new evals. Compare macro_loss to existing
  `moe-v16-compute-opt-d512-2.19e+17` baseline.
