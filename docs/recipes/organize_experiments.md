# Recipe: Organize Experiment Reports

## Overview
Use this recipe to curate `docs/reports/index.md` after new experiment issues are harvested. The goal is to keep the report index tidy by folding fresh entries into the right sections, refreshing links, and leaving `## Uncategorized` empty.

## Prerequisites
- Local checkout of the marin repository with write access.
- Ability to run `uv` commands.
- Familiarity with the existing experiment categories in `docs/reports/index.md`.

## Guidelines for Humans

### Standard Workflow
1. Run `uv run --extra pm scripts/pm/itemize_experiment_issues.py` to append new experiments.
2. Open the diff for `docs/reports/index.md` and identify the additions in `## Uncategorized`.
3. For each experiment:
   - Match it to an existing section (e.g., `Training and Performance`, `Data Experiments`).
   - Merge any new resources (WandB links, Data Browser URLs) into the canonical entry in that section.
   - Remove the placeholder entry from `## Uncategorized`.
4. If an experiment truly does not fit, leave it under `## Uncategorized` and add a note explaining why.
5. Proofread for duplicate bullets, broken Markdown, and consistent title casing.
6. Commit the curated report, open a PR, or, if an interactive agent, signal to the user to look.

### Link Hygiene
- Prefer direct `https://wandb.ai/...` links when available; keep legacy `api.wandb.ai` links only if the direct share link does not exist.
- Use `https://marin.community` Data Browser URLs when the script surfaces them.
- Preserve access tokens embedded in links.

## Rules for Agents
- NEVER delete existing sections or conclusions.
- Keep badge styling consistent: `[![#NNN](https://img.shields.io/...)]` next to each experiment title.
- When merging new content, update the canonical entry instead of duplicating it.
- Leave `## Uncategorized` empty whenever possible; a single placeholder sentence is fine.
- Ask for guidance if an experiment does not map cleanly to known categories.

## Validation
- `rg "^- " docs/reports/index.md` to confirm bullets exist only under curated sections, not under `## Uncategorized`.
- Re-run `uv run --extra pm scripts/pm/itemize_experiment_issues.py` if unsure that all experiments were captured.
- Optional: `markdownlint docs/reports/index.md` to catch formatting drift.

## See Also
- `scripts/pm/itemize_experiment_issues.py` for source data generation.
- Existing recipes in `docs/recipes` for editing conventions.
