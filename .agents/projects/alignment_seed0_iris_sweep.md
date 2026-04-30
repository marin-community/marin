# APLN-001: IRIS Seed-0 Alignment Eval Sweep

## Goal

Run Marin's Iris-based vLLM inference plus Bloom-compatible GPT-4.1 judging for
the remaining seed-0 Bloom v2 standard-mode targets, using the same eval split,
sampling settings, and prompt-collapsed metric already validated in
`validate_bloom.md`.

This plan intentionally follows the existing Marin pattern:

1. Run target-model inference on Iris with local vLLM serving.
2. Write inference outputs to `gs://marin-us-central1/eval/...`.
3. Run `experiments/posttrain/run_bloom_judge.py` against those inference
   outputs with `openai/gpt-4.1-2025-04-14`.
4. Compare prompt-collapsed stats against Bloom's natural-mode seed-0 runs.

## Scope

Already completed in Marin:

- `beta0.1_lr7.5e-7_seed0`

Pending targets for this sweep:

- `marin-8b-instruct` (SFT baseline)
- `beta0.01_lr5e-7_seed0`
- `beta0.01_lr7.5e-7_seed0`
- `beta0.1_lr5e-7_seed0`

Out of scope for this sweep:

- seed 1 / seed 2 runs
- opposite-mode prompts
- changing judge prompts or judge aggregation
- porting `run_bloom_judge.py` into `evaluate.py`

## Fixed Inputs

- Eval prompts: `gs://marin-us-central1/alignment/gpt-4.1-eval-split`
- Spec path: `experiments/posttrain/specs/openai_model_spec.jsonl`
- Inference config:
  - `prompt_format=BLOOM`
  - `temperature=0.7`
  - `max_tokens=1500`
  - `n=3`
  - `inference_batch_size=256`
  - `tensor_parallel_size=4`
  - `tpu_type=v5p-8`
- Judge config:
  - `judge_model=openai/gpt-4.1-2025-04-14`
  - `temperature=0.0`
  - `max_tokens=4000`
  - `concurrency=128`
  - `use_source_rubric=true`
  - `require_source_rubric=true`

## Target Registry

| Key | Bloom model path | Bloom standard inference run | Bloom standard judge run | Marin GCS model path | Status |
|-----|------------------|------------------------------|--------------------------|----------------------|--------|
| `sft` | `/lfs/skampere3/0/ahmedah/models/marin/marin-8b-instruct` | `.../_lfs_skampere3_0_ahmedah_models_marin_marin-8b-instruct/run_20260324_121413_839fb25afc85` | `.../_lfs_skampere3_0_ahmedah_models_marin_marin-8b-instruct/run_20260324_152421_7c0a6d282da6` | `gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c` | Ready |
| `beta001_lr5e7_seed0` | `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.01_lr5e-7_seed0/step-849` | `.../beta0-01_lr5e-7_seed0_step-849/run_20260325_121828_dfa23de5ee49` | `.../beta0-01_lr5e-7_seed0_step-849/run_20260325_183808_5ef636ab1ebc` | not found under `gs://marin-us-central1/checkpoints/dpo/` | Needs staging |
| `beta001_lr75e7_seed0` | `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.01_lr7.5e-7_seed0/step-849` | `.../beta0-01_lr7-5e-7_seed0_step-849/run_20260325_122803_65a494d75d3a` | `.../beta0-01_lr7-5e-7_seed0_step-849/run_20260325_192827_81d1d40ed0cb` | `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849` | Ready |
| `beta01_lr5e7_seed0` | `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.1_lr5e-7_seed0/step-849` | `.../beta0-1_lr5e-7_seed0_step-849/run_20260325_123739_0cfcfcf7fd09` | `.../beta0-1_lr5e-7_seed0_step-849/run_20260325_195043_8929a88217b8` | not found under `gs://marin-us-central1/checkpoints/dpo/` | Needs staging |
| `beta01_lr75e7_seed0` | `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.1_lr7.5e-7_seed0/step-849` | `.../beta0-1_lr7-5e-7_seed0_step-849/run_20260325_124714_27e8b7bb559f` | n/a, already mirrored in Marin | `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849` | Completed template |

## Planned Marin Output Prefixes

| Key | Marin eval prefix |
|-----|-------------------|
| `sft` | `gs://marin-us-central1/eval/marin_8b_instruct_bloom_speceval/` |
| `beta001_lr5e7_seed0` | `gs://marin-us-central1/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/` |
| `beta001_lr75e7_seed0` | `gs://marin-us-central1/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/` |
| `beta01_lr5e7_seed0` | `gs://marin-us-central1/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/` |

Expected layout per target:

- inference: `gs://marin-us-central1/eval/<prefix>/inference-<hash>`
- judge output: `gs://marin-us-central1/eval/<prefix>/judge-gpt41/`

## Main Blocker

The `lr=5e-7` seed-0 v2 DPO HF exports are not visible under
`gs://marin-us-central1/checkpoints/dpo/` with the same naming pattern used by
the existing `lr=7.5e-7` runs.

Before submitting those two Iris inference jobs, we need one of:

- the exact existing GCS HF-export paths, if they already exist under a
  different name, or
- a new same-region export/upload of:
  - `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.01_lr5e-7_seed0/step-849`
  - `/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.1_lr5e-7_seed0/step-849`

## Implementation Plan

### Phase 1: Parameterize the current Marin inference entry point

Replace the single-target assumptions in
`experiments/posttrain/eval_llama3_8b_alignment.py` with a small target
registry and a `--target` CLI flag.

Suggested shape:

```python
TARGETS = {
    "sft": {
        "name": "marin_8b_instruct_bloom_speceval",
        "model": "gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c",
        "description": "Inference only: Marin 8B Instruct SFT on Bloom eval prompts",
    },
    "beta001_lr75e7_seed0": {
        "name": "marin_dpo_beta001_lr75e7_seed0_bloom_speceval",
        "model": (
            "gs://marin-us-central1/checkpoints/dpo/"
            "bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849"
        ),
        "description": "Inference only: Marin DPO beta0.01 lr7.5e-7 seed0",
    },
    "beta001_lr5e7_seed0": {
        "name": "marin_dpo_beta001_lr5e7_seed0_bloom_speceval",
        "model": "<fill-after-gcs-staging>",
        "description": "Inference only: Marin DPO beta0.01 lr5e-7 seed0",
    },
    "beta01_lr5e7_seed0": {
        "name": "marin_dpo_beta01_lr5e7_seed0_bloom_speceval",
        "model": "<fill-after-gcs-staging>",
        "description": "Inference only: Marin DPO beta0.1 lr5e-7 seed0",
    },
}
```

Keep the rest of the inference settings identical to the already-successful
`beta0.1_lr7.5e-7_seed0` Marin run.

### Phase 2: Submit inference jobs on Iris

Run each target as inference-only first, exactly as the current Marin workflow
does. Judge only after checking that inference completed cleanly.

Command template:

```bash
uv run iris --controller-url http://127.0.0.1:10000 job run \
  --no-wait \
  --job-name eval-<target-key>-iris-v1 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-central1 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py --target <target-key>
```

Recommended submission order:

1. `sft`
2. `beta001_lr75e7_seed0`
3. `beta001_lr5e7_seed0` after GCS staging
4. `beta01_lr5e7_seed0` after GCS staging

Rationale:

- `sft` and `beta001_lr75e7_seed0` are immediately runnable from existing GCS
  model paths.
- They validate the generalized script before spending effort on staging the
  two `lr=5e-7` checkpoints.

### Phase 3: Validate inference before judging

For each completed inference run:

- confirm `2,576 x 3 = 7,728` output rows
- confirm `0` empty responses
- inspect length / EOS behavior against Bloom's standard seed-0 run
- check whether tokenizer metadata issues recur on any newly staged checkpoint

Minimum artifact check:

```bash
gcloud storage ls gs://marin-us-central1/eval/<prefix>/inference-*/
```

### Phase 4: Run Bloom-compatible GPT-4.1 judging

Do not switch to the `evaluate.py` judge step for this sweep. Keep using
`experiments/posttrain/run_bloom_judge.py`, since that is the path already
validated against Bloom prompt parity.

Command template:

```bash
uv run iris --controller-url http://127.0.0.1:10000 job run \
  --no-wait \
  --job-name judge-<target-key>-gpt41-v1 \
  --cpu 8 --memory 32GB --disk 20GB \
  --region us-central1 \
  --env OPENAI_API_KEY=$OPENAI_API_KEY \
  -- python experiments/posttrain/run_bloom_judge.py \
    --inference-path gs://marin-us-central1/eval/<prefix>/inference-<hash> \
    --spec-path experiments/posttrain/specs/openai_model_spec.jsonl \
    --output-path gs://marin-us-central1/eval/<prefix>/judge-gpt41/ \
    --judge-model openai/gpt-4.1-2025-04-14 \
    --max-tokens 4000 \
    --concurrency 128
```

Expected judge behavior:

- around `7,698` judged rows
- around `30` skipped rows for missing rubric
- `0` API errors after retries

### Phase 5: Compare against Bloom and update the logbook

For each target, record:

- Marin inference artifact path
- Marin judge artifact path
- prompt-collapsed overall mean, std, SEM, CI95, compliance
- Bloom standard seed-0 reference mean from the corresponding judged run
- any TPU vs GPU behavior drift notes

Expected Bloom reference means from the standard seed-0 runs:

- `sft`: `7.94`
- `beta0.01_lr5e-7_seed0`: `8.698`
- `beta0.01_lr7.5e-7_seed0`: `8.654`
- `beta0.1_lr5e-7_seed0`: `8.338`
- completed template `beta0.1_lr7.5e-7_seed0`: `8.386`

## Success Criteria

- The four pending targets complete Marin inference on Iris with Bloom-matched
  sampling settings.
- Each target completes Bloom-compatible GPT-4.1 judging into GCS.
- Prompt-collapsed stats are recomputed from Marin judged outputs.
- The logbook records exact commands, output paths, and Bloom-vs-Marin deltas.

## Immediate Next Actions

1. Find or stage GCS HF exports for the two `lr=5e-7` seed-0 v2 checkpoints.
2. Parameterize `experiments/posttrain/eval_llama3_8b_alignment.py` with a
   target registry instead of the single hard-coded model.
3. Submit `sft` and `beta001_lr75e7_seed0` inference jobs on Iris first.
4. After inference completes, launch `run_bloom_judge.py` for those two targets.
5. Use those runs to confirm the generalized Marin workflow before adding the
   staged `lr=5e-7` checkpoints.
