# Neural Tokenizer Side Plan: AR Compatibility

## Goal

Extend the JPEG-tokenizer findings to **neural image tokenizers** and answer:

> Which neural tokenizer properties make autoregressive next-token prediction stable and efficient?

This is the neural side of the same thesis used in the JPEG thread:

1. sequence length/locality,
2. noise tolerance,
3. state complexity (token meaning constancy),
4. rollout stability.


## Hard Reporting Rules

- Final conclusions must use **whole-sequence / whole-image metrics** only.
- Primary comparison metric is **mean bits/image** (with bits/pixel as secondary normalization).
- Mean next-token loss is diagnostic only; never use it as final evidence.
- Every comparison must report matched training budget conditions (tokens/step and wall-clock context).


## Scope and Non-Goals

### In scope

- Frozen neural tokenizer benchmarking under a shared AR model/eval harness.
- Controlled locality/context-dependence probes on neural token streams.
- Perturbation and rollout stability analysis with the same rubric used for JPEG.

### Out of scope (for this phase)

- Building a new SOTA image tokenizer.
- Large-scale generative quality benchmarking.
- Massive model scaling before tokenizer-axis effects are clear.


## Tokenizer Taxonomy for This Thread

Use architecture classes rather than brand names as the main scientific units:

1. **Patch-local discrete tokenizer**
   - encoder receptive field is local, token for a patch mostly reflects that patch.
   - expected: better axis (3), potentially longer sequence.

2. **Contextual discrete tokenizer**
   - token assignment depends on broader/global context.
   - expected: better compression/reconstruction tradeoff, but higher axis (3) burden.

3. **Hierarchical / residual discrete tokenizer (RVQ-style or multi-level)**
   - multiple token streams per patch/scale.
   - expected: possible length-quality gains, possible state complexity increase.

4. **Lookup-free / scalar-quantized discrete tokenizer**
   - avoids learned codebook lookup semantics, often factorized/binary/scalar forms.
   - expected: lower codebook aliasing, unknown context sensitivity (to be measured).


## Candidate Selection Policy

For each class, pick one runnable open model/checkpoint that satisfies:

- deterministic encode/decode API,
- practical license for internal experiments,
- feasible inference cost on `marin-eu-west4(-a)`,
- no cross-region data movement requirements.

If a class has no clean runnable checkpoint, substitute with a minimal in-repo trained proxy and log the substitution explicitly.


## Experimental Design

## Phase N0: Harness + Adapters (No Claim Phase)

Deliverables:

- `experiments/neural_tokenizer_ar/base/` scaffold (copy-first, same style as JPEG thread).
- adapter interface:
  - `encode(image) -> token_ids`
  - `decode(token_ids) -> image`
  - tokenizer metadata: `vocab_size`, `seq_len` policy, patch grid, levels/streams.
- token-store builders to `gs://marin-eu-west4/jpeg_tokenizer/token_store/...`-style paths (region-local).

Checks:

- deterministic encode/decode checksum tests,
- token length/vocab summary script,
- small smoke runs for each tokenizer class.


## Phase N1: Frozen Tokenizer AR Head-to-Head

Matrix (minimum):

- 1 patch-local tokenizer,
- 1 contextual tokenizer,
- 1 hierarchical or lookup-free tokenizer.

For each tokenizer:

- build train/validation token stores (Imagenette first),
- run smoke (`32` steps),
- run trial (`2000` steps) with `tokexplore/` step naming and W&B under `marin-community/tokexplore`,
- run whole-image exact eval script to produce:
  - bits/image,
  - bits/pixel,
  - tokens/image.

Primary output:

- table ordered by whole-image bits/image at matched trial budget.


## Phase N2: Mechanism Probes (Same as JPEG Axes)

1. **Context-dependence probe (axis 3)**
   - keep one local spatial region fixed, vary surrounding context, re-encode.
   - metric: local token flip/edit rate.
   - report center/off-center positions.

2. **Perturbation amplification (axis 2)**
   - single-token corruption at controlled positions/fractions.
   - metric: delta bits/image at `h1`, `h64-tail`, and long tail.

3. **Rollout stability (axis 4)**
   - prefix + free-run continuation + decode.
   - metrics:
     - decode validity,
     - degradation slope vs rollout length,
     - bits/image drift under rollout.

4. **Length/locality accounting (axis 1)**
   - tokens/image,
   - effective local dependency radius proxy from perturbation tails.


## Phase N3: Controlled Paired Ablation

If frozen-tokenizer comparisons are suggestive but noisy, run one controlled pair:

- local tokenizer and contextual tokenizer with matched codebook size and stride,
- train both on same image corpus and reconstruction objective class,
- rerun N1/N2 metrics.

Purpose:

- isolate context dependence from unrelated architecture differences.


## Acceptance Gates

Advance phases only when:

- N0: deterministic adapter/tests pass for all selected classes.
- N1: all selected classes have terminal trial + exact whole-image eval outputs.
- N2: all four rubric axes have measurements for at least two contrasting classes.
- N3 optional: only if N1/N2 cannot cleanly separate hypotheses.


## Decision Criteria (Neural Side)

Use the existing AR rubric interpretation:

- Tokenizer is AR-friendly only if it passes axes (2), (3), (4).
- Axis (1) may be borderline only when whole-image bits/image remains top-tier.
- If tokenizer is compact/reconstructive but fails (2)-(4), classify as reconstruction-friendly but AR-hostile.


## Implementation Plan in Repo

Create:

- `experiments/neural_tokenizer_ar/base/adapter.py`
- `experiments/neural_tokenizer_ar/base/data.py`
- `experiments/neural_tokenizer_ar/base/model.py`
- `experiments/neural_tokenizer_ar/base/train.py`
- `experiments/neural_tokenizer_ar/base/launch.py`
- `scripts/neural_tokenizer/build_token_store.py`
- `scripts/neural_tokenizer/evaluate_representation_head2head.py`
- `scripts/neural_tokenizer/evaluate_context_dependence.py`
- `scripts/neural_tokenizer/evaluate_representation_perturbation.py`

Re-use JPEG evaluator logic where possible (same metric definitions, same output schemas).


## Ops and Cost Controls

- Preferred execution region/zone: `marin-eu-west4` / `marin-eu-west4-a`.
- Store artifacts under `gs://marin-eu-west4/`.
- Keep initial datasets and artifacts well below 1 TB.
- Avoid unnecessary cross-region transfers; rematerialize region-local token stores when needed.
- Use Iris/Ray non-blocking submission and monitor to terminal state.


## First Concrete Run Order

1. Implement adapters and token-store builder for first two classes:
   - patch-local,
   - contextual.
2. Build token stores on Imagenette validation/train.
3. Launch smokes.
4. Launch trials.
5. Run exact whole-image eval.
6. Run context-dependence + perturbation probes.
7. Decide whether third tokenizer class is needed before scaling.


## Expected Outcomes

Strong confirmation pattern:

- lower context-dependence score (axis 3) correlates with better perturbation resilience (axis 2) and better whole-image bits/image at fixed AR budget.

Potential contradiction to resolve:

- contextual tokenizer wins bits/image despite worse axis (3); if observed, test whether gains come from reduced sequence length or objective mismatch.


## Exit Condition for Neural Side

Stop when we can make a stable, evidence-backed statement of the form:

> At this scale and budget, neural tokenizers with more context-stable token semantics are better/worse for AR NTP, with explicit tradeoffs in length and reconstruction.

If evidence remains mixed, stop with a bounded uncertainty statement and a minimal follow-up experiment set.
