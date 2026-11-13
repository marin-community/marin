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
3. A gated preset without a pin at its current revision refuses every command, including the plan print,
   with exit 4. Controls are collected after merge; `--control` and train-long need no pin. Print the plan:

   ```bash
   uv run python -m experiments.post_training.snowball_rl --preset speed-short --version dev
   ```

   Check inputs, allocation, guards, budget estimate and output paths.
   `--version dev` satisfies the launcher; step versions use each preset's `revision`.
4. Before submission, tell the person the preset, estimated budget and replicate count. Use their
   experiment issue or prepare one through `file-issue`. Submit train-long or multiple 40-H100
   replicates only when the person asked. Set `WANDB_API_KEY`; `WANDB_ENTITY` chooses the team,
   defaulting to `marin-community`.
5. Add `--run` to submit. Add `--wait` for the final report; use a background process for these
   multi-minute jobs. Record the original process and job ID so a yielded tool call does not
   cause a second submission.

Follow the recipe's [input and flag contracts](../../../docs/recipes/snowball-rl.md#change-one-input),
[comparison rules](../../../docs/recipes/snowball-rl.md#compare-with-controls) and
[default-promotion procedure](../../../docs/recipes/snowball-rl.md#change-a-presets-default).
For an exact-numerics claim, require zero mismatch:

```bash
uv run python -m experiments.post_training.snowball_rl --preset speed-medium \
  --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
```

## Inspect saved answers

Read existing answers before launching another run. Controls are named by preset, `commit`
and revision in `snowball_rl_gate.CONTROLS`; use their configured output paths. Candidate status
prints its output. Read the configured export root from the native resolved launch to locate answers.
[EvalLite evaluations](../../../docs/recipes/snowball-rl.md#tasks-and-evaluation) are greedy; data-short
evaluates sampled candidate training tasks. Each native JSONL row carries
its TaskSpec ID, prompt, output, tokens and verification.

## Read and record the result

Read the recipe's [report contract](../../../docs/recipes/snowball-rl.md#compare-with-controls):
the verdict, label, reason and report path, followed by the Markdown tables. Inspect the reason for INVALID.

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
