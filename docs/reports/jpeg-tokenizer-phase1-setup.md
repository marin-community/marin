# JPEG Tokenizer Phase 1 Setup

This note records the first runnable training setup for the `K=4` coefficient baseline.

## Prepared Artifacts

- Token store:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- GCS mirror:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- Store contents:
  - train: `9469` examples
  - validation: `3925` examples
  - sequence length: `4096`
  - vocab size: `4095`

## Code Paths

- Store reader and `LmDataConfig` wiring:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/data.py`
- Store builder:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/scripts/jpeg_tokenizer/build_coeff_token_store.py`
- Runnable launch step:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/launch.py`

## Launch Conventions

- Executor smoke step name:
  `tokexplore/jpeg-tokenizer-k4-smoke`
- Executor step name:
  `tokexplore/jpeg-tokenizer-k4-trial`
- W&B target:
  `marin-community/tokexplore`
- W&B smoke group:
  `tokexplore-jpeg-tokenizer-k4-smoke`
- W&B group:
  `tokexplore-jpeg-tokenizer-k4`
- Default token store path:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- TPU target:
  `v6e-8`

## Validation

- The file-backed store round-trips through the Levanter causal LM path in tests.
- A one-step CPU smoke test runs successfully from the file-backed store.
- The launch module now resolves the token store at runtime rather than import time, so the code remains importable in clean checkouts.
- The TPU smoke run `tokexplore/jpeg-tokenizer-k4-smoke` completed successfully on `marin-eu-west4-a` with eval loss improving from `8.508` to `4.694`.

## Completed Baseline

- Ray job:
  `ray-run-dlwh-launch-20260308-085237`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k4-trial`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.476` at startup, `4.376` at step `1000`, `4.417` at step `2000`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-trial-603f95/checkpoints/step-2000`
- Wall time:
  `2778.14s`

The `K=4` coefficient baseline is now proven on the production Ray + TPU path. The previous tracker/config-artifact
serialization warning has since been fixed locally by switching artifact dumping to YAML over the already-materialized
hyperparameter dict. The remaining known nuisance is a W&B `BrokenPipeError` during shutdown after the run had already
synced successfully.

## Next Step

The `K=8` smoke has now also completed successfully:

- Ray job:
  `ray-run-dlwh-launch-20260308-094432`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-smoke`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.549 -> 4.446 -> 4.124`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-smoke-c2bc8c/checkpoints/step-96`

This establishes that `8192`-token coefficient sequences are viable on `v6e-8` at batch size `128`.

## Next Step

The `K=8` longer trial is now active:

- Ray job:
  `ray-run-dlwh-launch-20260308-095441`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial`
- Current status:
  `RUNNING`
- Recovery note:
  the run survived one TPU slice preemption, resumed on a replacement worker, and continued training under the same
  W&B run ID.

## Next Step

Let the `K=8` trial finish, then compare its terminal eval against `K=4` and decide whether the next bounded rung
should be a `K=16` smoke or a width/depth adjustment at `K=8`.

## K16 Staging

The `K=16` coefficient rung has now been staged:

- Token store:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`
- Sequence length:
  `16384`
- Store size:
  about `839 MiB`
- Smoke step:
  `tokexplore/jpeg-tokenizer-k16-smoke`
- Smoke Ray job:
  `ray-run-dlwh-launch-20260308-101439`
- Smoke output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795`

The `K=16` smoke has passed packaging, executor launch, and Fray TPU dispatch on `v6e-8`. The remaining question is
whether trainer startup and the first eval complete cleanly at batch size `64`.

The `K=8` trial is still active and has already progressed well beyond initial bring-up:

- Latest observed progress:
  about step `498/2000`
- Latest observed train loss:
  `4.08`
- Reliability note:
  the run has survived two TPU preemptions and continued retrying under the same Ray submission.

In parallel, executor-side log noise from config serialization has been fixed locally in
`marin.utilities.json_encoder`, so newly launched runs should no longer emit large warning blocks for dataclass
configs and `PartitionSpec` values.

## K16 Smoke Result

The `K=16` smoke has now completed successfully:

- Ray job:
  `ray-run-dlwh-launch-20260308-101439`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k16-smoke`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.620 -> 3.612 -> 3.435`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795/checkpoints/step-64`

This is enough to say that `16384`-token coefficient sequences are operationally viable on `v6e-8` at batch size `64`.

## K8 Reliability Note

The original `K=8` trial has now been interrupted multiple times by TPU preemptions and has restarted from scratch after
at least one resume because no checkpoint had landed before the node died. The next `K=8` trial should therefore use a
much shorter checkpoint interval instead of the original 10-minute save cadence.

## K8 Retry Status

The retried `K=8` baseline with 2-minute checkpointing is now behaving as intended:

- Ray job:
  `ray-run-dlwh-launch-20260308-103217`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial-r2`
- Latest observed progress:
  step `429/2000` with train loss `4.18`
- Confirmed checkpoints:
  steps `79`, `218`, and `358`
- Checkpoint path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints`

This resolves the main reliability concern from the original `K=8` trial: useful progress is now landing in GCS well
before the next likely TPU preemption.

## K16 Trial Staging

The next rung is now prepared as a full baseline run:

- Executor step:
  `tokexplore/jpeg-tokenizer-k16-trial`
- W&B group:
  `tokexplore-jpeg-tokenizer-k16`
- Sequence length:
  `16384`
- Batch size:
  `64`
- Train steps:
  `2000`
- Eval cadence:
  every `1000` steps
- Checkpoint policy:
  every `2` minutes, keep every `500` steps

This keeps the `K=16` trial on the same comparison shape as the completed `K=4` baseline and the active `K=8` retry.

## Active Monitoring

The current live state is:

- `K=8` retry job:
  `ray-run-dlwh-launch-20260308-103217`
- `K=8` W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial-r2`
- Latest observed `K=8` progress:
  step `772/2000` with train loss `3.61`
- Latest observed `K=8` durable checkpoints:
  `step-639` and `step-777`

This is the first `K=8` run that is clearly surviving the preemptible environment without losing the useful part of the
training curve.

- `K=16` trial job:
  `ray-run-dlwh-launch-20260308-104918`
- `K=16` output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-trial-a39c10`
- Current `K=16` status:
  controller `RUNNING`, executor launched, Fray dispatch completed, trainer/W&B startup not yet observed

At this point the likely blocker for `K=16` is TPU scheduling latency rather than a launch-surface bug, so the right
action is to leave it queued and keep monitoring rather than resubmitting.

## Resume Bug And Mitigation

The next failure was not another infra-only preemption issue. The active `K=8` retry successfully:

- reached `step-1000`
- wrote checkpoint `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints/step-1000`
- recorded eval loss `3.428`

But after the following TPU preemption, the replacement worker:

- discovered `step-1000`
- logged `Loading checkpoint from .../step-1000`
- then immediately fell through to `Starting from scratch`

That exposed a bug in [train.py](/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/grug/base/train.py): the grug
resume path was catching any downstream `FileNotFoundError` from `load_checkpoint(...)` and treating it as "no
checkpoint exists", even after checkpoint discovery had already succeeded. The local fix now resolves checkpoint
discovery first and only soft-fails when no checkpoint is present at all; restore-time file errors now surface instead
of silently resetting the run.

Validation for the fix passed with:

- `uv run --with pytest python -m pytest -o addopts='' tests/test_grug_variant_contracts.py -k 'resume_missing_checkpoint_data_raises or grug_base_run_emits_expected_metrics_with_json_tracker'`

To keep the cluster focused on the half-finished baseline while fixing the resume path, the queued `K=16` trial
`ray-run-dlwh-launch-20260308-104918` was intentionally stopped. The next correct move is to relaunch the `K=8` retry
from fixed code so it can resume from `step-1000` instead of wasting TPU time restarting from zero.

## Fixed K8 Relaunch

That relaunch has now happened:

- old buggy retry stop requested:
  `ray-run-dlwh-launch-20260308-103217`
- new fixed retry job:
  `ray-run-dlwh-launch-20260308-110836`
- code commit:
  `198523e81`
- reused output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd`

The new blocker is not another resume bug. The current scheduler message is:

- `No available node types can fulfill resource requests ... TPU-v6e-8-head`

So the fixed retry is now waiting on TPU capacity rather than failing in the training code. That is the right place to
pause: `K=16` stays stopped, the fixed `K=8` retry stays queued, and no more submission churn is warranted until the
cluster can actually allocate a slice.

## Completed K8 Baseline

That fixed retry has now finished successfully.

- Ray job:
  `ray-run-dlwh-launch-20260308-180311`
- Final status:
  `SUCCEEDED`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints/step-2000`
- Final eval loss:
  `3.253`

The important detail is that this was not just a healthy startup or a midpoint resume. The run made it all the way
through `2000` steps, wrote the terminal checkpoint, and exited cleanly at the Ray level. A shutdown-time
`BrokenPipeError` appeared only inside an ignored `atexit` callback and did not affect the job outcome.

That makes the `K=8` rung a real baseline for the next comparison:

- `K=4` trial final eval loss:
  `4.417`
- `K=8` trial-r2 final eval loss:
  `3.253`

So the first coefficient-fidelity increase looks materially useful on Imagenette, and the next reasonable question is
whether `K=16` continues that trend enough to justify doubling sequence length again.

## Relaunched K16 Trial

With `K=8` complete, the next step is back to the staged `K=16` baseline run.

- New `K=16` submission:
  `ray-run-dlwh-launch-20260309-003238`
- Executor step:
  `tokexplore/jpeg-tokenizer-k16-trial`
- Token store:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`

At the first post-submit check, the controller state is:

- status:
  `PENDING`
- message:
  `Job has not started yet. It may be waiting for the runtime environment to be set up.`

That is consistent with the earlier TPU allocation and runtime-env delays on `marin-eu-west4-a`; there is no fresh
code-level failure signal yet. The right next action is simply to monitor this submission through executor launch,
trainer startup, and the first checkpoint/eval boundary.

## Completed K16 Baseline

That monitoring pass is now complete too.

- Ray job:
  `ray-run-dlwh-launch-20260309-003238`
- Final status:
  `SUCCEEDED`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-trial-a39c10/checkpoints/step-2000`
- Final eval loss:
  `2.668`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k16-trial`

The executor-side result was also clean: `tokexplore/jpeg-tokenizer-k16-trial_166a0b35` finished successfully, the
full `2000`-step training loop completed, and the final checkpoint was written before exit. As with the successful
`K=8` finish, wandb emitted a shutdown-time `BrokenPipeError` inside an ignored `atexit` callback, but that did not
change the Ray job outcome.

At this point the first coefficient ladder on Imagenette has real terminal baselines:

- `K=4` final eval loss:
  `4.417`
- `K=8` final eval loss:
  `3.253`
- `K=16` final eval loss:
  `2.668`

So the core Phase 1 outcome is now concrete:

- increasing retained coefficients continues to help predictive loss on this setup
- the gain from `K=8` to `K=16` is still material, though smaller than the gain from `K=4` to `K=8`
- the implementation and resume/checkpoint path are stable enough that further work can focus on experimental choice,
  not launcher hardening

The next sensible decision is no longer "can we make `K=16` run?" but "what comparison gives the most information per
unit cost?" The main candidates are:

- analyze the `K=4/K=8/K=16` tradeoff explicitly
- introduce one non-coefficient baseline (`bytes` or `symbols`)
- move to a broader robustness corpus once the comparison target is chosen

## Matched-Budget K4 Rerun

The next step after the completed `K=16` baseline is now in flight: rerun `K=4` at the same token budget per step as
`K=8` and `K=16`.

That matters because the original `K=4` baseline was not compute-matched:

- original `K=4` trial:
  `seq_len=4096`, `batch_size=512`, `tokens/step=2,097,152`
- `K=8` trial:
  `seq_len=8192`, `batch_size=128`, `tokens/step=1,048,576`
- `K=16` trial:
  `seq_len=16384`, `batch_size=64`, `tokens/step=1,048,576`

So the matched rerun uses:

- executor step:
  `tokexplore/jpeg-tokenizer-k4-trial-matched`
- run id:
  `jpeg-tokenizer-k4-trial-matched`
- sequence length:
  `4096`
- batch size:
  `256`
- tokens per step:
  `1,048,576`

The code for that launch surface was committed as:

- `d6c58996b`
  `Add matched-budget K4 JPEG trial`

And the new Ray submission is:

- `ray-run-dlwh-launch-20260309-045325`

At the first check, the controller state is still just:

- status:
  `PENDING`
- message:
  `Job has not started yet. It may be waiting for the runtime environment to be set up.`

That is normal for these Ray submissions on `marin-eu-west4-a`; it does not indicate a fresh launch bug. Once this run
finishes, we will have the comparison that actually matters:

- `K=4` at matched token budget
- `K=8` at matched token budget
- `K=16` at matched token budget

## Matched-Budget K4 Result

That rerun is now complete too.

- Ray job:
  `ray-run-dlwh-launch-20260309-045325`
- Final status:
  `SUCCEEDED`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-trial-matched-13c1dd/checkpoints/step-2000`
- Final eval loss:
  `4.340`

So the fair ladder, with the same `1,048,576` tokens per step, is:

- `K=4` matched:
  `4.340`
- `K=8`:
  `3.253`
- `K=16`:
  `2.668`

That keeps the original qualitative result intact: increasing retained coefficients continues to help even after the
token-budget mismatch is removed.

## Sequence-Level Sweep

The next concern was methodological rather than infrastructural: lower mean token loss at larger `K` might just mean
the extra coefficients are easier to predict, not that the representation is actually better.

To answer that, I added a separate evaluator that computes:

- total NLL per image / bits per image for the whole coefficient sequence
- shared-prefix losses on identical target subsets:
  - first `4` coefficients per block for `K=4/K=8/K=16`
  - first `8` coefficients per block for `K=8/K=16`

### Launch Path

The evaluator was first submitted as a plain Ray job, which failed correctly on the CPU head because
`TrainerConfig(require_accelerator=True)` found no accelerator:

- failed submission:
  `ray-run-dlwh-evaluate_coefficient_sweep-20260309-052440`

A TPU-reserved Ray retry then stayed capacity-bound on `marin-eu-west4-a`:

- stopped retry:
  `ray-run-dlwh-evaluate_coefficient_sweep-20260309-052657`

The final successful path used a temporary dev TPU:

- TPU name:
  `dlwh-jpeg-seqeval-0527`
- cluster config:
  `infra/marin-eu-west4-a.yaml`

During that run, the evaluator found one real bug: the last validation batch can be smaller than the `8`-way data
sharding, so `device_put(...)` failed on the tail batch. The fix was to pad the last batch locally, then trim the
extra outputs before aggregation. That is now covered by a targeted regression test.

### Final Output

- output directory:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-coeff-sequence-eval-253e4dc31-r4`

The whole-sequence bits/image numbers are:

- `K=4` matched:
  `25430.55`
- `K=8`:
  `44845.61`
- `K=16`:
  `75156.11`

Those rise with `K`, which is expected because the representation itself is longer. So they are useful as description
lengths, but not as a direct cross-`K` quality metric.

The useful comparison is on shared prefixes:

- `prefix_4` mean bits/image:
  - `K=4`: `25430.55`
  - `K=8`: `24672.92`
  - `K=16`: `24282.01`
- `prefix_8` mean bits/image:
  - `K=8`: `44845.61`
  - `K=16`: `44167.21`

This is the important result from the sweep:

- larger `K` is not only winning because it appends easier tail coefficients
- larger `K` also improves the model's probability assignment on the shared early coefficients
- so the representation-quality story survives the "maybe the extra tokens are just easier" objection

The cleanest current interpretation is:

- `K=16` is the strongest rung in this Imagenette coefficient ladder
- the benefit is real on shared targets, not just on the full longer stream
- the next comparison should probably not be `K=32`, but a different representation family or a broader corpus

## Prefix-Only Context Ablation

There was still one obvious objection to the shared-prefix result above:

- maybe `K=16` beats `K=8` on `prefix_8` only because it gets to condition on coefficients `9..16`
- maybe `K=8` beats `K=4` on `prefix_4` only because it gets to condition on coefficients `5..8`

That is a different claim from "the low-frequency prefix is better modeled in isolation."

To check that directly, I added a hostile-context ablation to the evaluator:

- keep scoring the same shared-prefix targets
- replace all non-prefix source tokens in the context with the zero-coefficient token

This is intentionally out-of-distribution for the trained model, so it is not a clean likelihood estimate. But it is
good enough to answer the specific qualitative question: does the larger-`K` advantage survive once the extra tail
coefficients are removed from context?

### Output

- output directory:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-coeff-sequence-eval-ablate-0179b3388`

The ablated results are:

- `prefix_4_context_prefix_only` mean bits/image:
  - `K=4`: `25430.55`
  - `K=8`: `27249.77`
  - `K=16`: `28640.63`
- `prefix_8_context_prefix_only` mean bits/image:
  - `K=8`: `44845.61`
  - `K=16`: `51906.86`

So the sign flips once the extra tail coefficients are removed from context.

That gives the more precise interpretation of the earlier sweep:

- larger `K` helps because the extra retained coefficients are useful autoregressive context for later low-frequency targets
- larger `K` does **not** appear to produce a model that predicts the shared low-frequency prefix better in isolation
- the real tradeoff exposed by the coefficient ladder is therefore context utility versus sequence length, not simply
  "more retained coefficients always make the core low-frequency representation better"

This result is much more in line with the initial intuition: the first few coefficients per block are largely a
low-frequency summary, and the reason larger `K` helps on shared-prefix targets is that the added mid-frequency tokens
carry information about nearby local structure that improves the next block's prefix prediction.

## Byte And Exact-Coefficient Baselines

The next comparison should not mutate the existing reference coefficient ladder in place. There are now two separate
baselines staged alongside it:

- full canonical JPEG bytes
- exact libjpeg-backed `K=8` coefficients

The important distinction for the byte baseline is that it uses the entire canonical JPEG file after deterministic
re-encoding, not just the entropy payload. In other words, the byte stream includes normal JPEG structure such as
markers, tables, headers, and entropy-coded data, with source metadata stripped during canonicalization.

### Byte Smoke

The byte-window smoke run completed successfully:

- Ray job:
  `ray-run-dlwh-launch-20260309-061949`
- W&B:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-bytes-w8192-smoke`
- eval loss:
  `5.657 -> 4.610 -> 4.546`
- final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-w8192-smoke-c5f441/checkpoints/step-96`

That is enough to promote the full byte trial:

- Ray job:
  `ray-run-dlwh-launch-20260309-073427`

I do not yet have the terminal result for the full byte trial in this note.

### Exact Libjpeg K=8 Coefficients

The new libjpeg path uses `jpeglib.read_dct(...)` on the canonical JPEG bytes and extracts the exact quantized luma
blocks from libjpeg rather than recomputing a floating-point DCT over the canonical luma plane.

I kept this as a separate coefficient source instead of rewriting the old `K=8` reference rung, so the earlier results
remain comparable.

On a small `16`-image train slice:

- exact and reference `K=8` sequences had the same shape: `(16, 8192)`
- no rows were perfectly identical
- token equality was still `94.688%`
- every mismatch was only `±1`

That is the expected qualitative outcome: the reference implementation is close, but not identical, to libjpeg's real
quantized coefficient stream.

The full exact `K=8` token store is now mirrored at:

- `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`

with:

- train examples: `9469`
- validation examples: `3925`
- sequence length: `8192`
- vocab size: `4095`
- size: `420.76 MiB`

The first smoke submission for this exact baseline failed immediately:

- Ray job:
  `ray-run-dlwh-launch-20260309-074034`
- failure:
  `ModuleNotFoundError: No module named 'jpeglib'`

This was not a modeling problem. The job was submitted from the older committed snapshot, so the Ray runtime env did
not include the new dependency yet.

## Current Recommendation

The next two comparisons are now well-defined:

- let the full byte trial finish and compare it against the existing reference `K=8` rung
- relaunch the exact libjpeg `K=8` smoke from the updated commit, then promote it if startup is clean

That gets us the first real answer to the representation question outside the original floating-point reference
coefficient ladder.

## Exact Libjpeg Runtime Fix

The exact-libjpeg launch failure was a runtime-env issue, not a modeling issue. Fray was building the worker
environment from the exported `marin` package environment, so the root-level `jpeglib` dependency never reached the
training worker unless it was passed explicitly on the job request.

The fix was:

- add `pip_packages` threading from the JPEG launch config into `dispatch_grug_training_run(...)`
- set `pip_packages=("jpeglib>=1.0.2",)` on the exact-libjpeg smoke/trial steps
- lazy-import `jpeglib` inside the exact coefficient extraction helper so byte/reference-coefficient runs do not
  require the package at module import time

After that change, the relaunched exact smoke succeeded:

- Ray job:
  `ray-run-dlwh-launch-20260309-080827`
- Executor step:
  `tokexplore/jpeg-tokenizer-k8-libjpeg-smoke_1fb75879`
- Final status:
  `SUCCEEDED`

This is enough to treat the exact JPEG coefficient baseline as operational on the production Ray + TPU path. The next
step is the full `K=8` exact-libjpeg trial.

## Exact Libjpeg K=8 Trial Result

The full exact-libjpeg `K=8` baseline has now completed successfully.

- Ray job:
  `ray-run-dlwh-launch-20260309-081237`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-libjpeg-trial`
- Output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-trial-cdae37`
- Runtime confirmation:
  `Building environment with ['jpeglib>=1.0.2'], extras ['tpu']`
- Startup eval loss:
  `8.544`
- Final eval loss at step `2000`:
  `3.263`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-trial-cdae37/checkpoints/step-2000`
- Terminal status:
  `SUCCEEDED`

This lands essentially on top of the earlier reference `K=8` coefficient baseline (`3.253`). So for the current
question, the floating-point reference coefficient path was already close enough to the exact JPEG coefficient stream
that it did not materially change the learning result.
