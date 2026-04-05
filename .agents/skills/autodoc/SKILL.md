---
name: autodoc
description: Generate agent-optimized documentation for Marin libraries weekly.
schedule_cron: "0 0 9 * * 1"
schedule_tz: America/New_York
---

# autodoc

Generate package-level reference cards for AI coding agents working on Marin.

## Workflow

1. Run the generator script:
   ```bash
   uv run --script scripts/agent_docs/main.py --verbose
   ```

2. Review the output in `docs/agent/`:
   - `packages/*.md` — per-package reference cards (~2-4KB each)
   - `MAP.md` — package index (regenerated from package docs)

3. If docs were updated, commit and open/update a PR:
   - Branch: `autodoc/weekly-update`
   - Title: `docs: update agent-optimized documentation`
   - Label: `agent-generated`

## Troubleshooting

- If the script fails on a specific package, run with `--package <name>` to isolate.
- Use `--dry-run` to see what would be regenerated without LLM calls.
- Use `--stats` to verify package discovery sees the expected packages/functions.

## Cost

Large packages use haiku for file summaries, sonnet for aggregation.
Small packages use sonnet directly.

- Full run (100 packages): ~$5-10
