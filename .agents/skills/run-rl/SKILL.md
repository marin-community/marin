---
name: run-rl
description: Choose, run and read a Snowball RL preset through experiments/post_training/snowball_rl.py, compare it with pinned controls and record the experiment. Use for inference, trainer, numerics, task-selection or RL-algorithm measurements, stored-answer inspection and gate reports. New RL launch files and other experiments belong to launch-rl.
---

# Run the RL runner

Use `experiments/post_training/snowball_rl.py`. The [recipe](../../../docs/recipes/rl-experiments.md)
owns the preset table, flag contracts, measurement windows, saved files and comparison math.
Use `use-iris` for job inspection and `use-fsutil` for object-store reads.

## Plan and submit

1. Choose the preset from the recipe's [choice table](../../../docs/recipes/rl-experiments.md#which-experiment-to-run).
   When it names two, run the second after the first passes.
2. Track or stage new source under `experiments/`, `lib/` and `vendor/`. The source tree hash enters
   the run name. Custom tasks follow the
   [selection handoff](../../../docs/references/task-curation.md#train-on-a-snowball-selection).
   They need an immutable recipe, the same development holdout, the full EvalLite in
   `exclusions`, and no AI-judged tasks.
3. A default plan uses the declared control pin. Gated execution and planning with `--require`
   need a pin at the preset's current revision. Controls are collected after merge. `--control`
   and train-long need no pin. Print the plan:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev --plan
   ```

   Check inputs, allocation, guards, budget estimate and output paths.
   `--version dev` satisfies the launcher; step versions use each preset's `revision`.
   Use the same plan and inputs to read status for a saved run.
4. Before submission, tell the person the preset, estimated budget and replicate count. Submit
   train-long or multiple 40-H100 replicates only when the person asked. Set `WANDB_API_KEY`.
   `WANDB_ENTITY` chooses the team, defaulting to `marin-community`.
5. Pass the optional `--issue ISSUE_NUMBER` when the experimenter links the run to an issue.
   Replace `--plan` with `--run` to submit and add `--wait` for the final report.
   Use a background process and record its process and job IDs so a yielded tool call
   does not cause a second submission.
   Record the finishing status's `rl_progress_url` with the verdict; the launcher publishes using
   the caller's Marina login. Follow the recipe's [publication procedure](../../../docs/recipes/rl-experiments.md#dashboard-publication)
   for retries and pinned controls.
   Use `--url ORIGIN` when publication targets a local Marina stack.
   Repeat the original `--run --wait` invocation with the same source and inputs to resume
   train-long from its latest retained checkpoint.

Follow the recipe's [input and flag contracts](../../../docs/recipes/rl-experiments.md#change-one-input),
[comparison rules](../../../docs/recipes/rl-experiments.md#compare-with-controls) and
[default-promotion procedure](../../../docs/recipes/rl-experiments.md#change-a-presets-default).
For an exact-numerics claim, require zero mismatch:

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-medium \
  --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
```

## Record the experiment

Open an issue through `file-issue` before the first run only when the experimenter asks for one.
In **Description**, explain what changes and what the preset measures. In **Hypothesis or Goal**,
state the expected result and the measure that would confirm it. Put the relevant PR or source in
**Links**. For an existing issue, pass its number with `--issue`. Use the same issue for later
presets that measure the same change.

## Inspect saved answers

Read existing answers before launching another run. Controls are named by preset, `commit`
and revision in `snowball_rl.SNOWBALL.controls`; use their configured output paths. Candidate status
prints its output. Read the configured export root from the native resolved launch to locate answers.
[EvalLite evaluations](../../../docs/recipes/rl-experiments.md#tasks-and-evaluation) are greedy; data-short
evaluates sampled candidate training tasks. Each native JSONL row carries
its TaskSpec ID, prompt, output, tokens and verification.

## Read and record the result

Read the recipe's [report contract](../../../docs/recipes/rl-experiments.md#compare-with-controls):
the verdict, label, reason and report path, followed by the Markdown tables. Inspect the reason for INVALID.

Confirm the native receipt records success and the Iris job finished before reporting success.
For a linked issue, post one comment per result after the person's posting approval. Use these fields
in order:

- **Preset** — name, version, source and changed inputs.
- **Run / attempt** — IDs, Iris and Grafana links.
- **W&B** — exact run link.
- **Marina** — attempt link.
- **Verdict and key metrics** — verdict, objectives, constraints and report path.
- **Cost** — elapsed time and measured GPU-hours, or label the estimate.

## Never

- Stop or restart an Iris cluster, or cancel/resubmit a job without authorization.
- Collect controls or edit pins, a preset revision or `GATE_REVISION` unless asked in the current thread.
- Edit a preset, guard or tolerance to make a candidate pass.
- Set `generator.enforce_eager=true` without a request; CUDA graphs affect speed measurement.
- Claim a speed gain from speed-short alone, or a learning gain from a speed preset.
