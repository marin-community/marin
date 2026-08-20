---
name: organize-experiments
description: Curate the experiment report index at docs/reports/index.md.
---

# Organize Experiment Reports

Run the harvester, then move new entries from `## Uncategorized` into the
canonical category in `docs/reports/index.md`:

```sh
uv run scripts/pm/itemize_experiment_issues.py
```

For each entry, merge W&B/Data Browser resources into an existing experiment
record, remove its duplicate placeholder, and leave genuinely unmappable work
under `## Uncategorized` with a reason. Never delete existing sections or
conclusions, duplicate canonical entries, or change badge styling
(`[![#NNN](https://img.shields.io/...)]`). Prefer direct `wandb.ai` links and
`marin.community` Data Browser links; preserve access tokens already present.

Check duplicates, Markdown, title casing, and that Uncategorized is empty when
possible:

```sh
rg "^- " docs/reports/index.md
```

Ask for guidance when an entry does not fit a known category. See
`archive-experiments` for retiring source experiments.
