# Codex Overnight Plan: Tier B Tension Data Scale-Up + `bloomv2_m2` + M2 LoRA DPO

Date: 2026-04-21

## Problem

The current overnight pilot framing is too small. The confusing part is that the
"40" in the current selection is **40 retained tension points**, not 40 final
preference pairs.

Those 40 points come from:

- source: [m2_seed_slice.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/stage4_output/m2_seed_slice.json:1)
- selection rule: oracle-feasible, large `oracle JSR - M1 JSR`, and
  `weakest_marginal_score >= 3`
- coverage: 40 points across 38 statement pairs

If we only materialize `1 chosen x 2 rejecteds x 3 variants` for a tiny subset,
we get a toy dataset and learn very little. That was a de-risking choice, not a
real ceiling.

The overnight job should instead:

1. use **all 40 retained points**
2. generate **prompt-matched train variants** around each tension corner
3. generate multiple chosens and multiple rejecteds per variant
4. assemble **up to ~6k preference pairs**
5. merge them with BloomV2 into a new dataset
6. upload the merged dataset to GCS
7. launch an M2 LoRA DPO run from SFT using the same training recipe as M1

## Goals

- Build a real `D_tension` slice overnight, not a tiny demo.
- Preserve the eval atlas as eval-only; do not train on the exact held-out atlas prompts.
- Minimize API cost by using Batch API everywhere possible.
- Reuse cached work when it is still correct.
- Keep the original BloomV2 dataset untouched.
- Produce a new merged dataset `bloomv2_m2` on GCS.
- Submit an M2 LoRA DPO run with the same effective config as the reference M1 run.

## Non-goals

- Re-ranking the entire 452-point candidate pool overnight.
- Doing a fresh atlas sweep or a new judge study.
- Rewriting the DPO stack.
- Pushing changes to `origin` during the overnight run.

## Core decisions

### 1. The right unit is 40 retained points, not 40 pairs

The overnight plan starts from the already-reviewed seed:

- `40` retained tension points
- `38` statement pairs
- all on the oracle-feasible slice

This is the cleanest overnight starting point because the expensive target
selection has already happened.

### 2. Scale comes from variants x chosens x rejecteds

The correct multiplication is:

- `40` points
- `10` train variants per point
- `3` chosen responses per variant
- `5` rejected responses per variant

This yields:

```text
40 x 10 x 3 x 5 = 6000 preference pairs
```

So the "40" is just the number of seed tension corners. The actual overnight
output target is **up to ~6k pairs**.

### 3. Training data must be prompt-matched

Do not pair an oracle chosen written for one prompt with an M1 rejected produced
for a different prompt. The overnight pipeline must:

- generate train variants near each retained tension corner
- generate chosens on those variant prompts
- run M1 on those same variant prompts
- build pairs within the same `(point_id, variant_idx)` group

### 4. Batch API everywhere we can use it

All OpenAI calls should go through Batch API with `reasoning_effort=none`:

- variant generation: `gpt-4.1`
- chosen generation: `gpt-5.1`
- chosen judging: `gpt-5.1`
- rejected judging: `gpt-5.1`

The only non-Batch heavy stage is M1 variant inference, which should run on TPU.

## Proposed solution

### Data recipe

Build `D_tension` from the 40 retained points as follows:

```text
seed point
  -> 10 nearby train variants
  -> 5 oracle/teacher chosen candidates per variant
  -> select top 3 chosens per variant
  -> 10 M1 responses per variant
  -> select bottom 5 rejecteds per variant
  -> cross product within variant
  -> up to 15 pairs per variant
```

Then merge:

```text
bloomv2_m2 = bloomv2_base + D_tension
```

### Rejected selection rule

Rejected samples are not "all M1 samples." They must fail the joint criterion.

Keep a sample as a rejected candidate only if:

- `joint_satisfied == false`
- failed-side score is meaningfully bad, default `<= 5`
- it is not a near-duplicate of another rejected kept for the same variant

Then select the bottom `5` rejecteds per variant by failed-side severity, with
preference for the systematically sacrificed side when that is obvious.

### Chosen selection rule

For each variant:

- generate `5` chosen candidates
- judge all of them on both paired rubrics
- keep the top `3` with `min(A, B) >= 7`

If fewer than `3` pass, keep all passing ones.

## Execution outline

1. Load the 40-point seed from
   [m2_seed_slice.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/stage4_output/m2_seed_slice.json:1).
2. Generate up to `10` train variants per point with `gpt-4.1` Batch API, reusing already-generated pilot variants where available.
3. Generate up to `5` chosen candidates per variant with `gpt-5.1` Batch API.
4. Run M1 on all variant prompts at `N=10` on TPU to get variant-matched rejected candidates.
5. Judge chosen candidates and M1 samples against the paired rubrics with `gpt-5.1` Batch API.
6. Select top `3` chosens and bottom `5` rejecteds per variant.
7. Assemble up to `6000` prompt-matched preference pairs.
8. Merge those pairs with BloomV2 into `bloomv2_m2`.
9. Upload `bloomv2_m2` to `gs://marin-us-central1/preference/bloomv2_m2/`.
10. Submit the M2 LoRA DPO run from SFT with the matched M1 recipe.

## Detailed overnight phases

### Phase 1: Freeze the seed

Use:

- [m2_seed_slice.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/stage4_output/m2_seed_slice.json:1)
- [m2_target_review.md](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/stage4_output/m2_target_review.md:1)

Do not expand beyond the retained 40 points overnight unless the whole Tier B
pipeline finishes early.

### Phase 2: Generate train variants

Use `gen_train_variants.py`, but make sure it supports:

- `--n-variants 10`
- reuse of already-generated variants
- output keyed by retained point ID and `variant_idx`

Target:

- total variants: `400`
- reuse existing variants if already present
- generate only the delta through Batch API

### Phase 3: Generate chosen candidates

Use `gen_chosens.py` in Batch mode:

- `5` chosen candidates per variant
- total target draws: `400 x 5 = 2000`

This should be the main teacher signal for the overnight run.

### Phase 4: Generate M1 variant responses

Run M1 on the exact variant prompts:

- model target: the same M1 checkpoint used in the current stress-testing pipeline
- `N=10` per variant
- total target generations: `400 x 10 = 4000`

This creates a large rejected candidate pool with no extra frontier API spend.

### Phase 5: Judge everything

Judge both chosens and M1 variant samples on the paired rubrics:

- chosens: `2000 x 2 rubric scores`
- rejecteds: `4000 x 2 rubric scores`

Use Batch API and store the raw judged records so later selection is reproducible.

### Phase 6: Select and assemble pairs

Per variant:

- keep top `3` chosens
- keep bottom `5` rejecteds
- cross product within variant only

Target output:

- max: `400 x 15 = 6000`
- realistic expected yield: `3k-6k`, depending on pass rates and rejected cleanliness

### Phase 7: Merge with BloomV2 and upload

Create a new dataset only:

- base stays untouched:
  `gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`
- new merged dataset:
  `gs://marin-us-central1/preference/bloomv2_m2/`

Recommended split:

- keep the original BloomV2 train/val split unchanged
- split `D_tension` by `variant_idx`
  - `variant_idx = 0` -> val
  - `variant_idx = 1..9` -> train

Then dedupe by pair hash within split before upload.

## DPO LoRA run

### Training intent

Train M2 from SFT, not from M1.

The correct comparison is:

- `M1`: broad BloomV2 DPO from SFT
- `M2`: broad BloomV2 + tension data DPO from SFT

### Config requirement

The overnight agent should match the reference M1 recipe as closely as possible:

- base model: `marin-community/marin-8b-instruct`
- LoRA DPO
- `beta = 0.1`
- `lr = 1e-5`
- `train_batch_size = 64`
- `seed = 0`
- `v5p-8`
- one epoch over the merged dataset

### Important provenance note

The current workspace has staged drafts in:

- [dpo_bloomv2_m2.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/projects/m2_submission_drafts/dpo_bloomv2_m2.py:1)
- [m2_from_sft_beta0p1_lr1e5.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/projects/m2_submission_drafts/m2_from_sft_beta0p1_lr1e5.py:1)

Those drafts encode the intended M1-matched recipe, but the exact live W&B run
config has **not** been re-fetched in this workspace due auth/provenance limits.

So the overnight agent should:

1. use the staged draft as the local source of truth
2. if W&B auth is available in that session, fetch the run config and diff it
   before submit
3. if no W&B auth is available, proceed with the staged draft and report that
   the run matched the local draft, not a live W&B diff

This is better than claiming false certainty.

## Cost-conscious defaults

Use these defaults unless they obviously break yield:

- Batch API for all OpenAI calls
- `reasoning_effort=none`
- reuse already-generated variants and chosens
- cap chosens at `3` per variant
- cap rejecteds at `5` per variant

Expected scale:

- variants: `400`
- chosen draws: `2000`
- M1 variant generations: `4000`
- final pairs: `3k-6k`

This is large enough to materially perturb BloomV2 while staying cheap enough
to run overnight.

## Notes

- The original 40-point eval atlas should remain held out for evaluation.
- The training set should consist of nearby train variants, not the exact atlas prompts.
- Meta statements such as `no_agenda` should not be excluded by hard rule at this
  stage; if they survive the retained seed and variant construction cleanly, keep them.
- The overnight objective is not perfect coverage of the whole 452-point pool. It
  is one strong, clean Tier B run that actually has enough mass to matter.

## Failure handling

- If variant generation yields materially fewer than `400` usable variants, halt
  and report the shortfall.
- If chosen pass rates are too low to keep `>= 2` chosens on most variants, halt
  and report; the prompts or rubrics may be bad.
- If M1 TPU inference wedges for hours, halt and preserve all earlier artifacts.
- If upload to `bloomv2_m2` fails, do not mutate any existing dataset path.
- If the submission branch is dirty, do not force checkout; report and stop.

## Morning deliverables

- `bloomv2_m2` on GCS with BloomV2 + Tier B tension pairs
- raw variant / chosen / judged / rejected artifacts saved locally
- a concrete final pair count and yield report
- either:
  - an active M2 LoRA DPO job ID, or
  - a precise failure report showing what blocked submission

## Future work

- Expand beyond the current 40 retained points into the next-ranked feasible pool.
- Oversample `D_tension` if the one-epoch merged run washes it out.
- Add a second training condition that keeps total budget fixed for a clean paper ablation.
- Move from overnight prompt construction to a reusable `D_tension` pipeline under `experiments/posttrain/`.
