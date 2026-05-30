---
name: wandb-reporting
description: "Use W&B runs, reports, and artifacts consistently for experiments, benchmarks, and task results with dense numeric output."
---

# Skill: W&B Reporting

Use W&B when scalar series, plots, large comparison tables, or raw artifacts are
too dense for issue comments or logbooks. Keep GitHub as the narrative layer and
W&B as the data/report layer.

## Project Policy

- Choose project scope by the type signature of the work.
- Use the `marin` project for pretraining runs.
- Use a new project for materially different work, such as kernel development
  or a new RL variant.
- Runs requiring explicit run-to-run comparison must share a W&B project.
- Decide scope early; you cannot reliably move or copy runs across projects
  later.

## Run Naming and Metadata

- Use the same experiment/task ID in W&B run names, logbook entries, and issue
  comments.
- Record git commit, branch, command, config path, hardware, device count,
  critical env vars, and dataset/checkpoint refs in W&B config or summary.
- Group related sweeps with a stable group name.
- Prefer artifacts for raw CSV/JSON outputs that feed published tables.

## Reporting

- Link W&B runs/reports from the coordinating issue and logbook.
- Summarize only the decision-relevant numbers in GitHub; link W&B for dense
  tables and plots.
- Before publishing a claim, verify expected row counts, key uniqueness, and any
  de-duplication or aggregation logic.
- Keep report titles and sections aligned with the issue/logbook labels.

## Completion Checklist

- Relevant runs are in the intended project.
- Run names map back to issue/logbook experiment IDs.
- Primary comparison table or chart is linked from the issue.
- Raw artifacts needed to reproduce the table are uploaded or linked.
- Claims in GitHub match the final W&B values.
