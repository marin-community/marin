---
name: autodoc
description: Generate agent-optimized documentation for Marin libraries weekly.
schedule_cron: "0 0 9 * * 1"
schedule_tz: America/New_York
---

# autodoc

Generate the budgeted, two-axis agent documentation tree for Marin and verify it
against the probe suite before publishing.

## Doc taxonomy

Generated under `docs/agent/`, each file capped at ~1000 tokens:

- `MAP.md` — monorepo index (which sub-project, dependency edges, top entry points).
- `<project>/overview.md` — 30-second orientation + intent router.
- `<project>/ops.md` — how to **use** the project (entry points, happy path).
- `<project>/architecture.md` — how to **understand and change** it (mental model
  + where in the source to edit).

## Workflow

1. Generate the doc tree:
   ```bash
   uv run --script scripts/agent_docs/main.py --verbose
   ```

2. Score the generated docs against the 10-probe suite (a weak coder writes a
   script from docs alone; a judge reviews it against source-verified ground
   truth):
   ```bash
   uv run --script scripts/agent_docs/eval.py --docs-dir docs/agent --output-dir /tmp/autodoc-eval
   ```
   Inspect `/tmp/autodoc-eval/summary.json`. Treat a regression in mean rubric
   accuracy or mean quality versus the last run as a gate — investigate the
   failing probes' docs before opening the PR.

3. If docs improved (or held), commit and open/update a PR:
   - Branch: `autodoc/weekly-update`
   - Title: `docs: update agent-optimized documentation`
   - Label: `agent-generated`

## Troubleshooting

- `--package <name>` isolates a single package; `--dry-run` skips LLM calls;
  `--stats` verifies package discovery.
- `eval.py --list` prints the probes with no API calls; `eval.py --probe P8 ...`
  runs a single probe (P8 is the fuzzy-dedup regression anchor).

## Cost

- Doc generation: large packages use haiku for file summaries, sonnet for
  aggregation; small packages use sonnet directly. Full run (~100 packages):
  ~$5-10.
- Eval: 10 coder + 10 judge calls per run (haiku + sonnet) — a few dollars.
