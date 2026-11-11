---
name: run-snowball-rl
description: Choose, run and read a Snowball RL preset through experiments/post_training/snowball_rl.py, compare it with pinned controls and record the experiment. Use for inference, trainer, numerics, task-selection or RL-algorithm measurements, stored-answer inspection and gate reports. New RL launch files and other experiments belong to launch-rl.
---

# Run Snowball RL

Use `experiments/post_training/snowball_rl.py`. The [recipe](../../../docs/recipes/snowball-rl.md)
owns the preset table, flag contracts, measurement windows, saved files and comparison math.
Use `use-iris` for job inspection and `use-fsutil` for object-store reads.

## Plan and submit

1. Choose the preset from the recipe's [choice table](../../../docs/recipes/snowball-rl.md#which-experiment-to-run).
   When it names two, run the second after the first passes.
2. Track or stage new source under `experiments/`, `lib/` and `vendor/`. The source tree hash enters
   the run name. Custom tasks follow the
   [selection handoff](../../../docs/references/task-curation.md#train-on-a-snowball-selection).
   They need an immutable recipe, the same development holdout, the full EvalLite in
   `exclusions`, and no AI-judged tasks.
3. Print the plan:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
   ```

   Check inputs, allocation, objective/constraint names, budget estimate and output paths.
   `--version dev` satisfies the launcher; step versions use each preset's `revision`.
4. Before submission, tell the person the preset, estimated budget and replicate count. Use their
   experiment issue or prepare one through `file-issue`. Submit train-long or multiple 40-H100
   replicates only when the person asked. Set `WANDB_API_KEY`; `WANDB_ENTITY` chooses the team,
   defaulting to `marin-community`.
5. Add `--run` to submit. Add `--wait` for the final report; use a background process for these
   multi-minute jobs. Record the original process and job ID so a yielded tool call does not
   cause a second submission.

Without `--wait`, submission exits 0 when accepted. With it, exits are 0 for PASS or a completed
ungated run, 1 for FAIL, 2 for INVALID, 3 without a required verdict, and 4 for pre-execution
refusal. Repeating the command without `--run` reads saved status from the same source tree;
a source edit selects a new run name. Follow the native attempt, Iris job, W&B URL and Grafana
link in status.

## Change an input

A code or wheel change needs no flag. `--set KEY=VALUE` changes a SkyRL setting; a new key
requires `+KEY=VALUE`. Preset-owned data, placement, evaluation, retention and measurement keys
are refused with a reason. Data presets accept no `--set`.

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev --run
uv run python -m experiments.post_training.snowball_rl --preset data-short \
  --tasks NAME@VERSION --version dev --run
```

Screen a task selection with data-short, then use data-medium for training quality. Use
learning-medium for task-sampling changes. `--init NAME@VERSION` on learning or train-long
selects a published model or kept HF export. Learning-medium and data-medium retain a durable
final export for controls and candidates. Train-long keeps checkpoints and
HF exports on 40 H100, with no verdict. Its original command resumes its own latest checkpoint.

`--require 'KEY<=NUMBER'` or `KEY>=NUMBER` bounds every replicate. `KEY<=control` and
`KEY>=control` compare a mean with calibrated controls independently of the built-in guards.
Use a numeric zero-mismatch bound for a claim of exact numerics:

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-medium \
  --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
```

Requirements can name computed step metrics or panel slices such as
`eval_lite/aaii_category/coding/mean`. Preflight refuses unavailable or mistyped keys.
`--replicates N` makes 1–5 copies with one seed. Saved typed inputs support CPU re-scoring:

```bash
uv run python -m experiments.post_training.snowball_rl_gate \
  --preset learning-medium --candidate RUN_NAME --require 'eval_lite/aaii_category/coding/mean>=control'
```

Promoting a measured setting follows the recipe's
[Change a preset's default](../../../docs/recipes/snowball-rl.md#change-a-presets-default)
procedure.

## Inspect saved answers

Read existing answers before launching another run. Controls are named by preset, `commit`
and revision in `snowball_rl_gate.CONTROLS`; use their configured output paths. Candidate status
prints its output. The recipe's [file layout](../../../docs/recipes/snowball-rl.md#tasks-and-evaluation)
locates primary answers and speed-short's separate quality answers. Each native JSONL row carries
its TaskSpec ID, prompt, output, tokens and verification.

Learning/data medium evaluates at update 0 and 25; train-long uses the 100-task panel every ten
updates. Speed-short's primary answers are fixed at 4,096 tokens. Data-short samples 256 tasks
and generates four answers per task. Named panels use greedy natural replies up to 8,192 tokens.

## Read and record the result

Read objectives (control → run, delta%, noise%), then constraints/invariants, then verdict and label.
Learning objectives include gain per H100-hour from native elapsed time; absent duration makes
that value unavailable. Each JSON comparison is stored once with its role. Reported telemetry
shows directionless changes; the change label uses objectives and built-in guards only.
Zero-spread changes are unavailable. A failed infrastructure invariant can produce FAIL with
`no change` as the metric label.

- PASS: all constraints and invariants hold within the calibrated tolerance.
- FAIL: a constraint or invariant failed; read its row.
- INVALID: saved evidence cannot support the comparison; read the reason and inspect the native job.

PASS covers the preset's measured window, tasks and layout; long-run gains, other layouts and
unrepresented task families need separate evidence.

Confirm the native receipt records success and the Iris job finished before reporting success.
Prepare the experiment issue update with the change, exact source, inputs, run/attempt IDs,
W&B/Grafana links, objective values, constraints, verdict and report path. Have the person review
posting text as required by their publication rules.

## Never

- Stop or restart an Iris cluster, or cancel/resubmit a job without authorization.
- Collect controls or edit pins, a preset revision or `GATE_REVISION` unless asked in the current thread.
- Edit a preset, guard or tolerance to make a candidate pass.
- Set `generator.enforce_eager=true` without a request; CUDA graphs affect speed measurement.
- Claim a speed gain from speed-short alone, or a learning gain from a speed preset.
