---
name: run-snowball-rl
description: Choose, run, and read a Snowball RL preset through `experiments/post_training/snowball_rl.py` (speed-short, speed-medium, speed-long, data-short, data-medium, learning-medium, train-long) and its pass, fail, or invalid verdict against pinned controls. Use to measure or regression-check a vLLM, inference-speed, trainer, numerics, task-selection, task-sampling, evaluation, or RL-algorithm change on Snowball, to read Snowball answers for chain-of-thought inspection, and to explain a Snowball gate report. New RL launch files and other RL experiments belong to launch-rl.
---

# Run Snowball RL

`experiments/post_training/snowball_rl.py` is the entry point for Snowball RL experiments. A preset fixes
the model, tasks, GPU layout, evaluation, and measurement window. Gated presets compare the run with
pinned control runs of the same preset and report `pass`, `fail`, or `invalid` with a change label. The
[recipe guide](../../../docs/recipes/snowball-rl.md) defines every preset, guard, invariant, and known gap;
this skill chooses the preset for a use case and reads the result. New RL launch files and role geometry
belong to `launch-rl`; job inspection and logs belong to `use-iris`; object-store reads belong to
`use-fsutil`.

## Before a run

- Stage new source files: untracked files under `experiments/`, `lib/`, or `vendor/` are refused. The
  run name contains the working tree's hash, so changed code gets a new run and never reuses an old
  result.
- Set credentials before the dry run, which checks them: `TOGETHER_API_KEY` whenever the training
  selection has judge-graded tasks (every preset except speed-short with the default selection), and
  `WANDB_API_KEY` at submission. `WANDB_ENTITY` selects the W&B team; unset uses `marin-community`.
- Print the plan without `--run`:

  ```bash
  uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
  ```

  The output lists the inputs, allocation, `budget_estimate` (minutes and H100-hours), guards,
  reported metrics, known gaps, and each run's `output` path. Invalid settings fail here, before any
  GPU work. `--version dev` satisfies the launcher interface; Snowball steps use the fixed
  `PRESET_REVISION`.
- Before `--run`, tell the person the preset, `budget_estimate`, and replicate count. Submit
  train-long (112 H100 by default, hours to days) or more than one replicate of a 40-H100 preset only
  when the person asked for it.
- `--run` submits and exits 0 once the submission is accepted. Read the result later by running the
  same command without `--run`, from the same source tree: any edit to a tracked file changes the run
  name, and the command then shows `state: not built` for a new run.
- `--run --wait` polls until the coordinator finishes: about 10 minutes for speed-short and data-short,
  35–120 minutes for the medium presets, and hours to days for train-long, so run it in the
  background. It exits 0 for `pass` or a finished train-long, 1 for `fail`, 2 for `invalid`, and 3
  when a submission error, a failed coordinator, or missing evidence prevents a verdict.

## Choose the preset

When a row names two presets, run the second only after the first passes.

### Inference and speed

| change | presets |
|---|---|
| vLLM wheel, kernel, or engine option | speed-short, then speed-medium |
| trainer kernel, parallelism, recompute, or exact numerics | speed-medium, then learning-medium |
| fp8 experts or KV-cache precision | speed-medium, then learning-medium; ask the person to weigh the speed gain against the learning-medium result |
| weight synchronization | speed-medium |
| startup latency | speed-medium; read the `startup/*` rows of the Reported table, which never change the verdict |
| asynchronous loop, grading, or admission throughput | speed-long |

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev --run
uv run python -m experiments.post_training.snowball_rl --preset speed-medium \
  --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
```

- A code or wheel change needs no flag; the run name follows the source tree. An engine option is
  `--set generator.KEY=VALUE`, or `--set +generator.engine_init_kwargs.KEY=VALUE` for a vLLM argument
  the base config lacks. speed-short accepts only `generator.*` keys outside
  `generator.sampling_params`.
- The `--require ...<=0` bound above holds every replicate to zero mismatch, for a change that claims
  exact numerics.
- speed-long has no control pin, so a candidate prints `invalid` and exits 2 before any GPU work. The
  message includes the command that collects four control replicates. Report this to the person;
  collect controls only when they ask (see "Never").
- Every preset reports `timing/grading/*` and `timing/setup/*`. They can change the label to
  `regression` or `trade-off`; they do not change the verdict.

### Chain-of-thought inspection

Read answers that already exist before launching a run. Each pinned control keeps its answers under
`<MARIN_PREFIX>/checkpoints/snowball-rl/control/<preset>-<main_commit>-r<N>/<revision>/`, with
`main_commit` and `revision` from `snowball_rl_gate.CONTROLS`. Otherwise run the preset whose answers
you need and read under the run's `output` path:

- `exports/dumped_evals/<session>/<data source>.jsonl`, where `<session>` is `global_step_<N>_evals`
  for an evaluation after update N, or `eval_only` for data-short. Each line is one answer with
  `task_id`, `input_prompt`, `output_response`, `score`, `stop_reason`, and `verification`.
- learning-medium and data-medium write the 910-task panel after update 25, and also at update 0 with
  `--init`; train-long writes the 100-task panel at update 0 and every 10 updates.
- data-short writes four answers from the pinned SFT model to each of 256 tasks sampled from its
  `--tasks` selection, in 10–15 minutes on 8 H100. `--tasks snowball-100k@2026.11.04` samples the
  default training tasks.
- Do not read speed-short answers as reasoning: every reply is forced to exactly 4,096 tokens, so the
  text continues after the model's end of sequence.

Panel answers use the training sampling settings (gap `eval-uses-training-sampling` in the guide).
`--require` accepts any key the gate computes for control r0: a step-file metric over the measured
window, or a panel slice. It refuses other keys before launch.

### Data: tasks and curriculum

1. Build an immutable selection through [task curation](../../../docs/references/task-curation.md). It
   must name the same held-out input as the default selection.
2. Screen it, then train on it:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset data-short \
     --tasks NAME@VERSION --version dev --run
   uv run python -m experiments.post_training.snowball_rl --preset data-medium \
     --tasks NAME@VERSION --version dev --run
   ```

- data-short measures the fraction of tasks with both correct and incorrect answers: the tasks that
  give RLOO-N a learning signal.
- data-medium trains learning-medium's recipe on the selection and uses learning-medium's controls.
- Data presets refuse `--set`. A task-sampling change runs on learning-medium. `data.shuffle` is in the
  base config; `data.sampling.*` is not, so it takes `+`, for example
  `--preset learning-medium --set +data.sampling.kind=learnability`.

### Evaluation

- learning-medium and data-medium evaluate a fixed 910-task panel, and train-long a fixed 100-task
  panel. A different panel changes the preset, which needs a new `PRESET_REVISION` and new controls.
- No preset evaluates weights without training. `--preset learning-medium --init NAME@VERSION`
  evaluates the initial weights at update 0 and the trained weights after update 25; the report's
  "initial weights" Breakdown table holds the update-0 scores.
- Gate on a slice of the panel:

  ```bash
  uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
    --require 'panel-910/aaii_category/coding/mean>=control' --version dev --run
  ```

  Slice keys are `panel-910/family/<family>/mean`, `panel-910/aaii_category/<category>/mean`, and
  `panel-910/difficulty/<easy|learnable|hard>/mean`; the `slice` column of a finished report's
  Breakdown tables lists the exact names.
- Re-score one finished replicate on CPU with added requirements or baselines, without GPU work:

  ```bash
  uv run python -m experiments.post_training.snowball_rl_gate \
    --preset learning-medium --candidate RUN_NAME --require 'KEY>=control'
  ```

  `RUN_NAME` is one replicate's `name` from the status block; the command scores that replicate alone.

### Algorithm and RL

```bash
uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
  --set trainer.algorithm.advantage_estimator=grpo \
  --set trainer.algorithm.group_advantage_min_size=null --version dev --run
```

- `--set KEY=VALUE` changes loss, estimator, optimizer, schedule, reward shaping, task sampling, or
  batch shape; keys absent from the composed config need `+KEY=VALUE`. Placement, data bindings,
  retention, evaluations, and measurement settings are refused with the owner to change instead.
- `--replicates N` (1–5) narrows the tolerance.
- `--baseline NAME@VERSION` adds a table comparing the run with earlier finished runs of the same
  preset and `PRESET_REVISION`; the verdict still uses the controls. `NAME@VERSION` is a run's `name`
  and `version` from its status block; give `--baseline` once per replicate.
- `--init NAME@VERSION` on learning-medium or train-long starts from another model or a finished
  train-long export.
- train-long is the release loop: checkpoints, periodic panel evaluation, and an HF export, with no
  verdict. `--layout p40` runs it on 40 H100.

## Read the result

- The status block, which every command prints, gives per run `state`, `iris_job_id`, `wandb_entity`,
  `latest_completed_update`, and `output`, plus `wandb_url` and `grafana_links` once available, then
  the `verdict`, `label`, `reason`, and report path. `rl_path_changes_since_control` lists RL source
  changed since the controls' commit; it does not affect the verdict.
- `gate_report.md` beside the report holds the Guards, Reported, and Invariants tables, a "Vs
  baseline" table when `--baseline` is given, and Breakdown tables by panel slice.
  [Compare with controls](../../../docs/recipes/snowball-rl.md#compare-with-controls) defines the
  tolerances and labels.
- `fail`: a guard fell below its tolerance, or a candidate invariant failed (planned updates, finite
  losses, panel coverage of at least 99%, grader errors of at most 1%, fixed reply length, or a numeric
  `--require`). The Guards and Invariants tables show which.
- `invalid`: the gate could not compare the run, and `reason` names the cause, such as no control pin,
  a different preset revision or panel, a missing guard or `--require` metric, or a grader-error count
  that straddles 1%. Fix the cause and rerun; do not reinterpret the numbers.
- When a run's state is failed and no report exists, the status block shows `verdict: fail`,
  `reason: run crashed`, and the run's `iris_job_id`; inspect that job with `use-iris`.
- Before reporting success, confirm that `terminal.json` under the run's `output` records the run as
  succeeded and that the Iris job finished.

## Never

- Cancel, stop, or resubmit a job without authorization in the current thread. Never stop or restart
  an Iris cluster.
- Collect controls (`--control`), or edit `snowball_rl_gate.CONTROLS`, `PRESET_REVISION`, or
  `GATE_REVISION`, unless the person asks for that change in the current thread. A control set costs
  four or five full runs of the preset.
- Edit a preset, tolerance, or guard so that a candidate passes.
- Set `generator.enforce_eager=true` unless the person asks for eager execution; it disables CUDA
  graphs and makes the speed measurement meaningless.
- Claim a speed gain from speed-short alone, or a learning gain from a speed preset.
