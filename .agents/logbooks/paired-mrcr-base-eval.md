# Paired MRCR base evaluation: Research logbook

## Scope

- Goal: measure whether June 67B context-extension checkpoints use retained conversation context when predicting the same final assistant response.
- Primary metrics: token-micro context gain in NLL and BPB, defined as `query_only - full_context`; positive values mean retained context improves likelihood.
- Constraints: score only response-body tokens, preserve paired source IDs and tokenizer-specific bins, restore parameters only, and keep checkpoint/data/output I/O in `us-central2`.
- Experiment issue: [#8701](https://github.com/marin-community/marin/issues/8701)
- Prior d512 MRCR experiment: [#7181](https://github.com/marin-community/marin/issues/7181)
- Implementation branch: `codex/paired-mrcr-base-eval`

## Baseline

- Date: 2026-08-25
- Dataset: OpenAI MRCR revision `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`.
- Evaluation set: 299 paired two-shot examples at a 262,144-token cap, covering 2/4/8 needles and four evidence-distance bands.
- Scored target: 115,834 response tokens and 561,555 response bytes in each condition.
- Hardware: TPU v4-64 in `us-central2`; batch/data/context/expert mesh `4/1/8/4`.
- Memory recipe: expert hidden weights sharded over context, hidden states staged in host RAM, model parameters released before scoring, and XLA cross-entropy evaluated in 512-token chunks.

Context gain is `query_only - full_context`, so positive NLL or BPB gain is beneficial.

| Checkpoint | Inference qk | Full NLL | Query-only NLL | NLL gain (95% paired bootstrap CI) | Full BPB | Query-only BPB | BPB gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step 141000, pre-extension (8K) | 1.57 | 1.65895 | 1.46442 | -0.19453 [-0.21900, -0.16930] | 0.49369 | 0.43580 | -0.05789 |
| step 156000, 64K source | 1.57 | 1.01333 | 1.41720 | +0.40387 [+0.36399, +0.44537] | 0.30156 | 0.42174 | +0.12019 |
| step 157000, 262K extension | 1.57 | 0.21904 | 1.40763 | +1.18860 [+1.14205, +1.23597] | 0.06518 | 0.41890 | +0.35371 |
| step 157000, deployable arm | 1.75 | 0.14639 | 1.41494 | +1.26855 [+1.22432, +1.31342] | 0.04356 | 0.42107 | +0.37751 |

The fixed-qk=1.57 trajectory changes from -0.05789 BPB before extension to +0.35371 BPB at the permanent final checkpoint, a +0.41161 BPB swing. Query-only NLL stays near 1.41 from step 156000 onward, while full-context NLL falls from 1.01333 to 0.21904. The checkpoint change primarily improves use of retained context on this evaluation.

The deployable qk=1.75 arm adds +0.02379 BPB gain and +0.07995 NLL gain relative to qk=1.57 at step 157000. These are descriptive point differences; no between-run paired interval was computed.

Macro NLL gain follows the same trajectory: -0.19691 at step 141000, +0.41089 at step 156000, +1.23319 at step 157000/qk1.57, and +1.31014 at step 157000/qk1.75.

Confidence is exploratory. Each checkpoint received one evaluation pass over the same 299 paired two-shot source examples: 598 full-context/query-only condition sequences per checkpoint. The four-checkpoint trajectory therefore contains 1,196 paired checkpoint-example comparisons and 2,392 condition sequences. The within-checkpoint NLL intervals use 10,000 deterministic paired-bootstrap samples stratified by needle count and evidence-distance band. Aggregate BPB point estimates were reconstructed from the emitted per-cell loss, BPB, and scored-token totals; the current artifact does not emit an aggregate BPB bootstrap interval. One-shot, no-prefix, and the full 60+16 matrix have not run.

## Experiment log

### 2026-08-25 - Pre-extension 262K probe

- Hypothesis: an 8K-trained checkpoint will not reliably benefit from 262K retained context.
- Command:

  ```bash
  .venv/bin/iris --cluster=marin job run --no-wait --enable-extra-resources \
    --region us-central2 --priority production --no-preemptible --max-retries 100 \
    --cpu 4 --memory 16GB --disk 32GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e GIT_COMMIT 8758e4a7be \
    -- env MRCR_MATRIX_SELECTION=aggregate_262k_probe MRCR_EVAL_TPU=v4-64-cp8-ep4 \
    python -m experiments.grug.moe.eval_mrcr_context --max_concurrent 1
  ```

- Result: -0.19453 NLL gain and -0.05789 BPB gain. Full context hurt response likelihood.
- Artifact: `gs://marin-us-central2/eval/mrcr/step-141000-pre-extension-qk157/two_shot/cap-262144-v464cp8ep4-0010a9`
- W&B: [step141000/qk1.57](https://wandb.ai/marin-community/marin_moe/runs/mrcr-67b-step141000-qk157-cap262144-two_shot-v464cp8ep4)
- Interpretation: retained 262K context is harmful before context extension on this sample.

### 2026-08-25 - 64K source checkpoint

- Hypothesis: the 64K stage should reverse the pre-extension regression at 262K if the aggregate recipe teaches context use that extrapolates beyond its training length.
- Command:

  ```bash
  .venv/bin/iris --cluster=marin job run --no-wait --enable-extra-resources \
    --region us-central2 --priority production --no-preemptible --max-retries 100 \
    --cpu 4 --memory 16GB --disk 32GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e GIT_COMMIT acd0f6e41c \
    -- env MRCR_MATRIX_SELECTION=aggregate_262k_extension MRCR_EVAL_TPU=v4-64-cp8-ep4 \
    python -m experiments.grug.moe.eval_mrcr_context --max_concurrent 1
  ```

- Result: +0.40387 NLL gain and +0.12019 BPB gain.
- Artifact: `gs://marin-us-central2/eval/mrcr/step-156000-source-qk157/two_shot/cap-262144-v464cp8ep4-c11e76`
- W&B: [step156000/qk1.57](https://wandb.ai/marin-community/marin_moe/runs/mrcr-67b-step156000-qk157-cap262144-two_shot-v464cp8ep4)
- Interpretation: the 64K checkpoint turns the negative pre-extension result into a positive 262K context gain at fixed inference qk=1.57.

### 2026-08-26 - Permanent final checkpoints

- Hypothesis: the 262K extension should improve context gain beyond the 64K stage; qk=1.75 may add a smaller inference/deployable-arm effect.
- Command:

  ```bash
  .venv/bin/iris --cluster=marin job run --no-wait --enable-extra-resources \
    --region us-central2 --priority production --no-preemptible --max-retries 100 \
    --cpu 4 --memory 16GB --disk 32GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e GIT_COMMIT 63027b4e72 \
    -- env MRCR_MATRIX_SELECTION=aggregate_262k_final_pair MRCR_EVAL_TPU=v4-64-cp8-ep4 \
    python -m experiments.grug.moe.eval_mrcr_context --max_concurrent 1
  ```

- Result: qk1.57 reached +1.18860 NLL/+0.35371 BPB; qk1.75 reached +1.26855 NLL/+0.37751 BPB.
- Artifacts:
  - `gs://marin-us-central2/eval/mrcr/qk157-step157000/two_shot/cap-262144-v464cp8ep4-f76755`
  - `gs://marin-us-central2/eval/mrcr/qk175-step157000/two_shot/cap-262144-v464cp8ep4-0212a0`
- W&B:
  - [step157000/qk1.57](https://wandb.ai/marin-community/marin_moe/runs/mrcr-67b-step157000-qk157-cap262144-two_shot-v464cp8ep4)
  - [step157000/qk1.75](https://wandb.ai/marin-community/marin_moe/runs/mrcr-67b-step157000-qk175-cap262144-two_shot-v464cp8ep4)
- Interpretation: fixed-qk adaptation accounts for most of the final gain. The qk1.75 deployable arm improves the point estimate further.
- Operational note: the qk1.75 cell completed after two worker-reconcile preemptions. Both final artifacts are atomic and record evaluator commit `63027b4e72`, CP base `db7ffddd339dd4db71fbb83ae2555abe3522c894`, and the pinned dataset revision.

## Next actions

1. Add aggregate BPB gain and paired-bootstrap BPB intervals directly to the evaluator artifact.
2. Run the one-shot and two-shot-no-prefix sensitivities at bounded lengths before expanding the full matrix.
3. Use the existing partial-summary surface for unavailable intermediate checkpoints; do not substitute a different checkpoint silently.
