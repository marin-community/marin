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
   ./scripts/generate_agent_docs.py --verbose
   ```

2. Review the output in `docs/agent/`:
   - `MAP.md` — package index (auto-loaded into every agent conversation)
   - `packages/*.md` — per-package reference cards (~2-4KB each)

3. If docs were updated, commit and open/update a PR:
   - Branch: `autodoc/weekly-update`
   - Title: `docs: update agent-optimized documentation`
   - Label: `agent-generated`

## Troubleshooting

- If the script fails on a specific package, run with `--package <name>` to isolate.
- Use `--dry-run` to see what would be regenerated without LLM calls.
- Use `--stats` to verify package discovery sees the expected packages/functions.
- Use `--full` to force a complete regeneration (ignores cache).

## Cost

Large packages use haiku for file summaries, sonnet for aggregation.
Small packages use sonnet directly.

- Full run (100 packages): ~$5-10
- Incremental (typical weekly): ~$0.50-2
