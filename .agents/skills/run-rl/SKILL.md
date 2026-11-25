---
name: run-rl
description: Choose, run and read a Snowball RL preset through experiments/post_training/snowball_rl.py, compare it with pinned controls and record the experiment. Use for inference, trainer, numerics, task-selection or RL-algorithm measurements, stored-answer inspection and gate reports. New RL launch files and other experiments belong to launch-rl.
---

# Run the RL runner

Use `experiments/post_training/snowball_rl.py`. The [recipe](../../../docs/recipes/rl-experiments.md)
owns the preset table, flag contracts, measurement windows, saved files and comparison math.
Use `use-iris` for job inspection and `use-fsutil` for object-store reads.

Map requests for a short loop, quick loop or long loop to the corresponding presets.

- **Short loop** (also called **quick loop**) uses gated presets to compare one change against control runs. Use `speed-*`, `data-*` or `learning-medium`.
- **Long loop** uses ungated training to build a model and retain its best eligible checkpoint. Use `train-long` for EvalLite or `train-long-harbor` to add the Harbor cohorts.

## Plan and submit

1. Choose the preset from the recipe's [choice table](../../../docs/recipes/rl-experiments.md#which-experiment-to-run).
   When it names two, run the second after the first passes.
2. Track or stage new source under `experiments/`, `lib/` and `vendor/`. The source tree hash enters
   the run name. Custom tasks follow the
   [selection handoff](../../../docs/references/task-curation.md#train-on-a-snowball-selection).
   They need an immutable recipe, the same development panel, the full EvalLite in
   `exclusions`, and no AI-judged tasks.
3. A default plan uses the declared control pin. Gated execution compares stored control receipts
   with the current preset defaults. Controls are collected after merge. `--control`
   and train-long need no pin. Print the plan:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev --plan
   ```

   Check inputs, allocation, guards, budget estimate and output paths.
   `--version dev` satisfies the launcher. The preset version fingerprints the composed configuration.
   Use the same plan and inputs to read status for a saved run.
4. Before submission, tell the person the preset, estimated budget and replicate count. Submit
   either train-long preset or multiple 40-H100 replicates only when the person asked. Set `WANDB_API_KEY`.
   `WANDB_ENTITY` chooses the team, defaulting to `marin-community`.
   For `train-long-harbor`, check the recipe's [Daytona requirements](../../../docs/recipes/rl-experiments.md#harbor-evaluation).
5. Pass `--issue ISSUE_NUMBER` if the experimenter supplied an issue number. If the context leaves
   tracking undecided, ask whether to open a new issue, use an existing one, or run without an issue.
   Replace `--plan` with `--run` to submit and add `--wait` for the final report.
   Use a background process and record its process and job IDs so a yielded tool call
   does not cause a second submission.
   Record the finishing status's `rl_progress_url` with its result. The launcher publishes the
   running and terminal attempt with the caller's Marina login. Follow the recipe's
   [publication procedure](../../../docs/recipes/rl-experiments.md#dashboard-publication)
   for retries and pinned controls.
   Use `--marina-url ORIGIN` when publication targets a local Marina stack.
   Repeat the original `--run --wait` invocation with the same source and inputs to resume
   either train-long preset from its latest retained checkpoint.

Follow the recipe's [input and flag contracts](../../../docs/recipes/rl-experiments.md#change-one-input),
[comparison rules](../../../docs/recipes/rl-experiments.md#compare-with-controls) and
[default-promotion procedure](../../../docs/recipes/rl-experiments.md#change-a-presets-default).
For an exact-numerics claim, require zero mismatch:

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-medium \
  --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
```

## Record the experiment

Open an issue through `file-issue` after the plan and control-pin check pass and the experimenter
approves the issue text.
In **Description**, explain what changes and what the preset measures. In **Hypothesis or Goal**,
state the expected result and the measure that would confirm it. Put the relevant PR or source in
**Links**. For an existing issue, pass its number with `--issue`. Use the same issue for later
presets that measure the same change. Use `publish --preset PRESET --run-output PATH --attempt ID --issue NUMBER`
to associate a stored attempt with its issue.

## Inspect saved answers

Read existing answers before launching another run. Controls are named by preset, `commit`
and artifact version in `snowball_rl.SNOWBALL.control_pins`. Use their configured output paths. Candidate status
prints its output. Read the configured export root from the native resolved launch to locate answers.
[EvalLite evaluations](../../../docs/recipes/rl-experiments.md#tasks-and-evaluation) are greedy; data-short
evaluates sampled candidate training tasks. Each native JSONL row carries
its TaskSpec ID, prompt, output, tokens and verification.

## Read and record the result

Read the recipe's [report contract](../../../docs/recipes/rl-experiments.md#compare-with-controls):
the verdict, effect, reason and report path, followed by the Markdown tables. Inspect the reason for INVALID.

Confirm the native receipt records success and the Iris job finished before reporting success.
For a linked issue, post one comment per result after the person's posting approval. Use these fields
in order:

- **Result** — verdict or status, key metrics, guards and report path.
- **Preset** — name, version, source and changed inputs.
- **Run / attempt** — IDs, Iris and Grafana links.
- **W&B** — exact run link.
- **Marina** — attempt link.
- **Cost** — elapsed time and measured GPU-hours, or label the estimate.

## Never

- Stop or restart an Iris cluster, or cancel/resubmit unrelated jobs without authorization.
- Collect controls or edit pins or `GATE_REVISION` unless asked in the current thread.
- Edit a preset, guard or tolerance to make a candidate pass.
- Set `generator.enforce_eager=true` without a request; CUDA graphs affect speed measurement.
- Claim a speed gain from speed-short alone, or a learning gain from a speed preset.
