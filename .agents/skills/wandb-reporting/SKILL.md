---
name: wandb-reporting
description: "Use W&B runs, reports, and artifacts consistently for experiments, benchmarks, and task results with dense numeric output."
---

# W&B Reporting

Use W&B for scalar series, plots, large comparison tables, or raw artifacts too
dense for GitHub. GitHub/logbooks carry narrative; W&B carries data and
reports. Choose the project before creating runs because runs cannot reliably
be moved later:

- default pretraining project: `marin`;
- materially different work (for example kernels or a new RL variant): a new
  project;
- all runs compared directly: one shared project.

Use the same experiment/task ID in run names, logbooks, and issue comments;
stable group names for sweeps; and artifacts for raw CSV/JSON feeding tables.
Keep artifacts under 10 MB or store larger files in the experiment directory
and link them. Before publishing, verify row counts, key uniqueness,
deduplication/aggregation, and that GitHub claims match final W&B values.

Link relevant runs/reports and the primary table/chart from the issue and
logbook; summarize only decision-relevant numbers in those narrative layers.
