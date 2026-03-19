# WebP Tokenizer Project: JPEG-Parallel Experiment Stream

## Goal

Run the same core investigation we did for JPEG on WebP:

> what tokenizer properties make AR next-token prediction stable and useful in a non-symbolic domain?

Primary endpoint remains whole-sequence metrics (`bits/image`) plus prefix-corruption amplification.

## High-Level Hypotheses

1. Raw WebP bytes should be hard for AR modeling due to strong codec context dependence.
2. Pre-entropy symbolic streams should improve over raw bytes.
3. More context-independent token semantics should improve corruption stability, even if compression worsens.
4. Forced state resets should reduce corruption amplification in context-adaptive streams.

## Codec Scope

Start with WebP lossy (VP8-in-RIFF) only for the first stream. Add lossless (VP8L) as a second stream only after lossy reaches baseline-complete.

## Representation Ladder (Lossy WebP)

Keep the ladder parallel to JPEG naming:

1. `webp_bytes_full`
- full RIFF WebP bytes (including container/chunk headers)

2. `webp_payload_bytes`
- image payload bytes with container framing removed where possible

3. `webp_entropy_events`
- decoded entropy events (tokenized event ids + amplitude/payload components)

4. `webp_symbols_exact`
- exact syntax-symbol stream prior to entropy coding (event vocabulary can be large)

5. `webp_coeff_k{K}`
- block residual/transform coefficient stream with bounded `K` (e.g., `K=4,8,16,all`)

## Phase 0: Local Feasibility (No TPU)

### Deliverables

- `scripts/webp_tokenizer/inspect_representations.py`
- `docs/reports/webp-tokenizer-phase0.md`

### Checks

- deterministic canonicalization and deterministic libwebp encode settings
- sequence-length distributions per representation
- observed/configured vocab sizes
- decode-validity sanity where applicable

### Decision Gate

Proceed only after at least 3 representations have:
- deterministic generation
- stable decode/roundtrip behavior
- tractable sequence lengths for `SWA=4096` training

## Phase 1: Token Stores + Baseline Training

### Deliverables

- token-store builders under `scripts/webp_tokenizer/`
- launch wiring in `experiments/jpeg_tokenizer/base/launch.py` (or a small `experiments/webp_tokenizer/base/` fork if cleaner)
- report section in `docs/reports/webp-tokenizer-phase1.md`

### Initial Runs (whole-image objective only)

- `webp_bytes_full_whole_swa4096`
- `webp_symbols_exact_whole_swa4096`
- `webp_coeff_k8_whole_swa4096`
- then `webp_coeff_kall_whole_swa4096` if sequence budget allows

### Evaluation Rules

- report only whole-sequence/whole-image metrics for conclusions
- avoid reporting mean next-token loss as final evidence

## Phase 2: Prefix-Corruption Harness (WebP)

### Deliverables

- `scripts/webp_tokenizer/evaluate_representation_perturbation.py`
  (can share utility logic with JPEG script)
- output: `summary.md` + `perturbation_eval.json`

### Metrics

- `delta_total`, `delta_immediate`, `delta_h{...}`, `delta_tail_h{...}` (bits/image)
- decoded semantic corruption metrics where we can decode event-space tails

### Required Controls

- perturbation-type control (e.g., marker/context token vs value token), analogous to synthetic `perturb_kind`
- length-normalized deltas (e.g., bits per 1k future tokens) in addition to raw `delta_total`

## Phase 3: Mechanism Tease-Apart Inside WebP

Isolate context dependence vs compression tradeoff using paired streams:

1. `shared_semantics` vs `context_explicit`
- shared tokens interpreted by decoder state
- same stream but with explicit context disambiguation tokens/fields

2. reset sweep
- forced periodic context resets (`8`, `32`, `128`, none)
- measure semantic tail hamming and `delta_total`

This is the WebP-side analogue of the synthetic mechanism check.

## Phase 4: Cross-Codec Comparison

Once WebP lossy baselines are stable, compare against JPEG on matched conditions:

- matched training budget/model config
- matched eval split and whole-image metric definition
- compare:
  - clean `bits/image`
  - corruption amplification
  - sensitivity to reset frequency

## Minimal Execution Order

1. Build Phase 0 inspector and produce first WebP phase0 report.
2. Implement 3 token-store builders (`bytes_full`, `symbols_exact`, `coeff_k8`).
3. Run smoke + short trials.
4. Run whole-image evaluator.
5. Run perturbation harness with perturb-kind control.
6. Add reset/context-explicit ablations.

## Cost/Safety Guardrails

- keep early artifacts local or low-volume GCS (< few hundred GB until representation viability is clear)
- prioritize single-region runs/storage
- avoid large transfer churn until one strong WebP ladder is selected

## Success Criteria (First WebP Pass)

- at least one non-byte WebP representation clearly outperforms raw bytes on whole-image loss
- at least one experiment cleanly shows context dependence increases semantic corruption amplification
- reset ablation gives interpretable directional effect
