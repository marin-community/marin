---
name: run-snowball-rl
description: Choose, run, and read a Snowball RL preset through `experiments/post_training/snowball_rl.py` (speed-short, speed-medium, speed-long, data-short, data-medium, learning-medium, learning-long, data-long, train-long) and its pass, fail, or invalid verdict against pinned controls, and record each experiment in a GitHub experiment issue. Use to measure or regression-check a vLLM, inference-speed, trainer, numerics, task-selection, task-sampling, evaluation, or RL-algorithm change on Snowball, to read Snowball answers for chain-of-thought inspection, and to explain a Snowball gate report. New RL launch files and other RL experiments belong to launch-rl.
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
- Selections must contain no AI-judge-graded tasks. The default training selection is
  `snowball-100k` at `DEFAULT_SELECTION_VERSION` in `snowball_rl_presets.py`; the dry run prints the
  exact version. Set `WANDB_API_KEY` at submission; `WANDB_ENTITY` selects the W&B team, falling back
  to `marin-community`.
- Print the plan without `--run`:

  ```bash
  uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
  ```

  A valid plan lists the inputs, allocation, `budget_estimate` (minutes and H100-hours), guards,
  reported metrics, known gaps, and each run's `output` path. Invalid settings fail here, before any
  GPU work. `--version dev` satisfies the launcher interface; Snowball steps use the fixed
  `PRESET_REVISION`.
- Before `--run`, tell the person the preset, `budget_estimate`, and replicate count, and open the
  experiment's issue ([Record the experiment](#record-the-experiment)). Submit
  train-long (40 H100, hours to days) or more than one replicate of a 40-H100 preset only
  when the person asked for it.
- `--run` submits and exits 0 once the submission is accepted. Read the result later by running the
  same command without `--run`, from the same source tree: any edit to a tracked file changes the run
  name, and the command then shows `state: not built` for a new run.
- `--run --wait` polls until the coordinator finishes: about 15 minutes for speed-short and data-short,
  35–120 minutes for medium presets, about 4 h for learning-long and data-long, and hours to days for train-long, so run it in the
  background. It exits 0 for `pass` or a finished ungated run, 1 for `fail`, 2 for `invalid`, 3
  without a required verdict, and 4 for a refusal before execution (bad flags, refused `--set`
  or missing controls). Status and re-score use the same verdict codes. Stdout includes YAML and
  Markdown; read the printed `report` path for the machine-readable `gate_report.json`.

## Choose the preset

Use the recipe's [preset-choice table](../../../docs/recipes/snowball-rl.md#which-experiment-to-run).
When it names two presets, run the second after the first passes.

### Inference and speed

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
- A preset without a matching control pin prints its refusal and exits 4 before any GPU work. The
  message includes the command that collects its control replicates. Report this to the person;
  collect controls only when they ask (see "Never").
- speed-short measures throughput, then evaluates the full test panel in the same engine job.
  The recipe describes the [engine settings](../../../docs/recipes/snowball-rl.md#tasks-and-evaluation).
  Read `test_panel/panel_scored_fraction` and `test_panel/grader_error_fraction` in the Invariants
  table to identify quality-stage failures.
- Every preset reports `timing/grading/*` and `timing/setup/*`. They can change the label to
  `regression` or `trade-off`; they do not change the verdict.

### Chain-of-thought inspection

Read answers that already exist before launching a run. Each pinned control keeps its answers under
`<MARIN_PREFIX>/checkpoints/snowball-rl/control/<preset>-<main_commit>-r<N>/<revision>/`, with
`main_commit` and `revision` from `snowball_rl_gate.CONTROLS`. Otherwise run the preset whose answers
you need and read under the run's `output` path:

- `exports/dumped_evals/<session>/<data source>.jsonl`, where `<session>` is `global_step_<N>_evals`
  for an evaluation after update N, or `eval_only` for speed-short and data-short. Each line is one
  answer with
  `task_id`, `input_prompt`, `output_response`, `score`, `stop_reason`, and `verification`.
- learning-long and data-long write the 100-task `test_panel_100` at update 0 and every 25 updates,
  and the 1,500-task `test_panel` after update 100 under
  `exports/test_panel/dumped_evals/global_step_100_evals/`. The primary answers use
  `exports/dumped_evals/`.
- learning-medium and data-medium write the 1,500-task `test_panel` at update 0 and after update 25;
  train-long writes the 100-task `test_panel_100` at update 0 and every 10 updates.
- speed-short's quality answers are under
  `exports/test_panel/dumped_evals/eval_only/<data source>.jsonl`; its primary throughput answers are
  forced to 4,096 tokens.
- data-short writes four answers from the pinned SFT model to each of 256 tasks sampled from its
  `--tasks` selection, in 10–15 minutes on 8 H100. Use
  `--tasks snowball-100k@<version from the dry run>` to sample the default training tasks.

Named panels use greedy decoding and natural replies up to `EVAL_REPLY_TOKENS` (8,192).
`--require` accepts any key the gate computes for control r0: a step-file metric over the measured
window, or a panel slice. It refuses other keys before launch.

### Data: tasks and curriculum

1. Build an immutable selection through [task curation](../../../docs/references/task-curation.md#train-on-a-snowball-selection). It
   must name the same held-out input as the default selection.
2. Screen it, then train on it:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset data-short \
     --tasks NAME@VERSION --version dev --run
   uv run python -m experiments.post_training.snowball_rl --preset data-medium \
     --tasks NAME@VERSION --version dev --run
   ```

- data-short measures the fraction of tasks with both correct and incorrect answers to screen a
  candidate selection.
- Each data preset trains the learning preset of the same length on `--tasks` and uses its controls.
- Data presets refuse `--set`. A task-sampling change runs on learning-medium. `data.shuffle` is in the
  base config; `data.sampling.*` is not, so it takes `+`, for example
  `--preset learning-medium --set +data.sampling.kind=learnability`.

### Evaluation

- `DEFAULT_TEST_PANEL_NAME` and `DEFAULT_TEST_PANEL_VERSION` in `snowball_rl_presets.py` name the
  default evaluation artifact; the dry run prints its version. It has 1,500 full tasks and a
  100-task fast subset, with source parts in its recipe and manifest. learning-medium and
  data-medium use the full panel; train-long uses the fast subset.
  Its recipe pins the initializer pass-rate artifact used for difficulty and reference-score rows.
- `--preset learning-medium --init NAME@VERSION` selects the initial weights. The report's
  "initial weights" Breakdown table holds the update-0 scores; long presets report gain on `test_panel_100`.
- Gate on a slice of the panel:

  ```bash
  uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
    --require 'test_panel/aaii_category/coding/mean>=control' --version dev --run
  ```

  Slice keys include `test_panel/family/<family>/mean`, `test_panel/aaii_category/<category>/mean`
  and `test_panel/difficulty/<easy|learnable|hard>/mean`. The `slice` column of a finished report's
  Breakdown tables lists the exact names.
- Learning and data controls keep their final HF weights. Candidates keep them with
  `--keep-weights`:

  ```bash
  uv run python -m experiments.post_training.snowball_rl --preset learning-medium \
    --keep-weights --version dev --run
  ```

- Run a gate re-score of one finished replicate on CPU with added requirements or baselines:

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
- `--init NAME@VERSION` on a learning or train-long preset starts from a published model artifact or a
  kept HF export.
- train-long is the release loop: checkpoints, periodic panel evaluation, and an HF export, with no
  verdict. It runs on 40 H100.

Learning presets score their greedy primary panel at update 0 and after training.
The run's own update-0 dump supplies its baseline; the pass-rate artifact supplies initializer
statistics by slice. `--baseline` compares separate completed runs in the gate report.
Every preset uses FLASH_ATTN, prefix caching on and batch invariance off.

For dashboard publication, retries and pinned controls, follow the
[publication rules](../../../docs/recipes/snowball-rl.md#dashboard-publication).

## Record the experiment

One experiment is one change under test, with every run and preset used to judge it. It has one
GitHub issue: use the issue the person names, or open one with `file-issue` (kind `experiment`) in
`marin-community/marin` before the first `--run`.

- Description: the change under test (branch and commit, and each `--set`, `--tasks`, or `--init`
  value) and the presets planned.
- Hypothesis or Goal: the guard or reported metric the change should move, and in which direction.
- Links: the command, and each run's `name`, `iris_job_id`, `wandb_url`, `grafana_links` once the status block shows them. Pass `--issue N` to record the issue.

After each verdict, add a comment with the preset, `verdict`, `label`, each guard's change against its
tolerance, and the `gate_report.md` path, then update the TL;DR and Status. A later preset for the
same change, such as speed-medium after speed-short, goes in the same issue.

## Read the result

Read the objectives first: each row gives the control and candidate values, percentage change,
noise percentage and direction. Learning presets also report panel score gain per H100-hour using the
update-0 and final scores and native elapsed time saved in the receipt. Long presets use
`test_panel_100` for this gain. A missing native duration makes that rate unavailable.
Read the constraints next, then the verdict and label. The status block and `gate_report.md` use
this order. The report also has the full Guards, Reported, Invariants and Breakdown tables, plus
Vs baseline when requested. The status includes native attempt IDs, job and W&B links, completed
update, output path, the optional experiment issue and `rl_progress_url` after publication. The dashboard
keeps the native run/attempt key; its attempt dropdown marks failed attempts. See
[Compare with controls](../../../docs/recipes/snowball-rl.md#compare-with-controls) for tolerances.

- `fail`: a guard fell below its tolerance, or a candidate invariant failed (planned updates, finite
  losses, panel coverage of at least 99%, grader errors of at most 1%, fixed reply length, or a numeric
  `--require`). The Guards and Invariants tables show which.
- `invalid`: the gate could not compare the run, and `reason` names the cause, such as
  a different preset revision or panel, a missing guard or `--require` metric, or a grader-error count
  that straddles 1%. Fix the cause and rerun; do not reinterpret the numbers.
- When a run's state is failed and no report exists, the status block shows `verdict: invalid`,
  a reason naming the native failure, and the run's `iris_job_id`; inspect that job with `use-iris`.
- Before reporting success, confirm that `terminal.json` under the run's `output` records the run as
  succeeded and that the Iris job finished.

Evaluation identity compares the stored TaskSpec strings and the evaluation request: merged sampling,
sample count, context/reply budget, prompt length, chat template and non-`VLLM_*` task environment.
Engine settings, batch invariance, `VLLM_*` environment and vLLM version are recorded as separate
attempt `engine` facts. They can vary between candidates and controls.

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
