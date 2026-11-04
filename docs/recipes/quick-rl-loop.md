# Quick RL loop

Use this loop to improve task selection and the RL recipe: curate tasks, publish
a fixed selection, train from the pinned SFT model, and compare evaluation before
and after training.

The current entry points are the
[Snowball selection recipe](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/recipes/snowball-100k.json)
and
[`train_quick_loop.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/inventory/train_quick_loop.py).
The selection defines the 100k training allocation. The launcher defaults to
25 synchronous RLOO-N updates on 40 H100s, starting from Datakit SFT revision
`fe666478454e79fe6027a0bab0388d9abd68f51a` and its matching tokenizer. It uses
128 prompt groups per update, four samples per group, a 24,576-token context
and an 8,192-token response limit.

## Make or update tasks

Declare new tasks through the
[dataset-owned curation pipeline](../references/task-curation.md#dataset-owned-invocation).
Pin their source revision, converter and grader dependencies. The training pool
comes from FULL curation's admitted `final/*.parquet` files; QUICK conversions
and SAMPLE panels do not qualify.

Use the [curation campaign procedure](../references/task-curation.md#run-a-campaign)
to review and verify the sources. For example, inside an Iris coordinator with
the pipeline dependencies and review credentials available:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver \
  --mode full --source SOURCE_NAME \
  --review-mode chat --model-revision "$REVIEW_MODEL_REVISION" \
  --review-cache "$CURATION_CACHE" \
  --max-workers 64 --coordinator-memory 16g --concurrent-sources 10 \
  --normalized-shards 32 --image-backend gvisor \
  --worker-image "$WORKER_IMAGE" \
  --report-path "$CAMPAIGN/full.json" \
  --selection experiments/post_training/task_curation/recipes/snowball-100k.json
```

Repeat `--source` for every source in the selection recipe, using its invocation
name from
[`runnable_sources()`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py).
For example, recipe ID `MarinSkyRL:eurus2_code` is invoked as
`--source eurus2_code`. The CLI names differ from the stable IDs in the recipe.
Set `REVIEW_MODEL_REVISION` to the pinned reviewer, `WORKER_IMAGE` to a
digest-pinned worker image, and `CURATION_CACHE` and `CAMPAIGN` to retained,
region-local storage prefixes. Build the declared
[grader environments](../references/task-curation.md#run-a-campaign) first.
The command prints a plan; add `--run` to execute it. Successful source artifacts
and exact review requests remain cached across campaigns.

## Publish a selection

If a completed FULL campaign already has the required source artifacts, reuse
its `corpus.json` without rerunning curation:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver \
  --corpus "$CAMPAIGN/corpus.json" \
  --selection experiments/post_training/task_curation/recipes/snowball-100k.json
```

Add `--run` to publish. The printed `task_selection` path is the artifact root
to pass to training. Its `summary.json` records the recipe, source artifacts,
selected counts and evaluation IDs. Check the source and family quotas there
before launching. A source shortfall fails instead of borrowing from another
source. Source or selection changes require a new recipe version; preserve
published versions for comparisons.

The artifact contains `train.parquet`, `validation.parquet` and
`full_validation.parquet`. The fixed validation panel samples 20 tasks per
family from the original 910-task development set. All 910 are excluded from
training by ID, public semantics and exact conversation context. This reused
development set is a hillclimbing signal; use a separate benchmark for final
claims. See [selection semantics](../references/task-curation.md#final-task-selection)
for deduplication and task serialization.

## Train and evaluate

Load the storage, tracking, image-grader and judge credentials required by the
[RL launcher](../references/rl-launching.md#experiment-entry-point) into the
submission environment. Use the published selection path and a fresh run version
to print a three-update plan:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version "$RUN_VERSION" --steps 3
```

Set `SELECTION_URI` to the printed artifact root, including its selection name
and version. Add `--run` to submit the graph through the maintained artifact
entry point. After the short run succeeds, use a fresh `RUN_VERSION` and
`--steps 25` for the normal iteration. Each run starts from the pinned SFT model.
The selected data stays cached; training does not curate or resample it.

The launcher binds `train.parquet` and `validation.parquet` from that artifact,
and the `rl-data/protocol-probes` artifact as a second validation source. The
probes are eight tasks per agent protocol check (required-file submission, tool
execution, thinking disabled, URL loading), built by
[`protocol_probes.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/inventory/protocol_probes.py).
Their file and tool tasks run on a ShellSim machine with four turns, set through
`task_sessions.task_spec.max_turns_by_dataset` and `task_machine_by_dataset`, and
the answer file is graded in the worker from a verifyit-only environment lock
that the graph builds. The launcher enables vLLM's Hermes tool-call parser; without
it, vLLM rejects every request that offers tools. Evaluation attempts the fixed panel and the probes before
training and after the last update, and saves task-level verifier outcomes. Under the run artifact's `exports/`,
`dumped_evals/global_step_0_evals/` holds the baseline and
`dumped_evals/global_step_<steps>_evals/` holds the final evaluation. Each contains
per-dataset JSONL responses and `aggregated_results.jsonl`; compare its
`avg_score` and `pass_at_1` metrics alongside verifier coverage. Each row of
`protocol-probes.jsonl` keeps its protocol assessment under
`verification.diagnostics.protocol`, keyed by check, with a label such as
`no_invocation` or `missing_file` and separate protocol-compliance,
submission-completeness and answer-correctness results. Aggregate rates over
those rows; a check whose evidence cannot decide it carries `not_evaluable`
instead. The thinking check applies only to launches that set
`enable_thinking: false` in `generator.chat_template_kwargs`.

To see the probe labels for a small model before a launch, run them through the
trainer's CPU harness. It evaluates the three probes that need no grading
machine on a local Hugging Face snapshot and prints their label counts:

```bash
cd vendor/skyrl && uv sync --frozen --extra cpu --no-group dev
PROTOCOL_PROBES_MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision> \
  PYTHONPATH=skyrl-train:skyrl-gym uv run --frozen --extra cpu --no-group dev \
  pytest skyrl-train/tests/cpu/test_protocol_probes_cpu.py -s
```

Qwen3-0.6B with thinking off answers the thinking probe but narrates the
command and URL probes without calling a tool, which the labels report as
`no_invocation` and `not_attempted` with wrong answers.

To run all four probes against the same model served by vLLM on one H100,
including the answer file graded in the local sandbox, submit
`uv run python -m experiments.post_training.inventory.protocol_probes_smoke
--version <label>-dev --run`. It trains one dry step so the probes are
evaluated before and after it, and writes the same metrics and dumps. To
evaluate a Snowball checkpoint without training, `uv run python -m
experiments.post_training.inventory.protocol_probes_eval --model snowball-v0.1
--version <label>-dev --run` serves it with vLLM on one 8xH100 node through
SkyRL's evaluation-only entrypoint, in about eight minutes once the model is
staged. Snowball v0.1 calls the tool on every tool and URL probe and answers
from the real result; the dumped rows carry the parsed conversation under
`messages`, so a wrong answer can be traced to the message the grader read.

The final update saves a checkpoint; HF export is opt-in with `--export-hf` and
adds a separate export job after training. Local graders require the privileged
container profile; image graders and host judges use their configured backends.

## Compare the iteration

Require `SUCCESS` from the Iris coordinator submitted by `--run` and its training
child, plus successful completion of the Marin training artifact. With `--export-hf`,
also require the export child to succeed. Its `terminal.json` records the run result;
the returned `SkyRLRun` supplies `iris_job_id` and `checkpoint_root` for follow-up inspection.
`hf_model_uri` is populated only when export is requested. Check
optimizer-update counts, finite metrics, initial/final attempted and scored
counts and native checkpoint metadata; check the completed HF export only when requested.
A missing verifier result is an evaluation error, not a wrong answer; report coverage with scores.

Record the source revision, selection identity, initializer, run version and
elapsed submission-to-completion time with each comparison. Hold the validation
panel fixed and change one recipe choice at a time. Reuse the published data for
training-only changes; produce a new selection version when changing tasks or
sampling. Update this page and the checked-in recipe when adopting a better
configuration, and keep trial results in the experiment issue or report.
