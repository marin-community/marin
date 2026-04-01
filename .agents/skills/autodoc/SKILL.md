---
name: autodoc
description: Generate agent-optimized documentation for Marin libraries weekly.
schedule_cron: "0 0 9 * * 1"
schedule_tz: America/New_York
---

# autodoc

Generate three-tier agent-optimized documentation for all Marin workspace libraries.

## Workflow

1. Run the generator script:
   ```bash
   ./scripts/generate_agent_docs.py --verbose
   ```

2. Review the output in `docs/agent/`:
   - `MAP.md` — module index (Tier 1, should be <4KB)
   - `modules/*.md` — per-module reference cards (Tier 2, each <8KB)
   - `api/*.yaml` — function-level structured docs (Tier 3)

3. If docs were updated, commit and open/update a PR:
   - Branch: `autodoc/weekly-update`
   - Title: `docs: update agent-optimized documentation`
   - Label: `agent-generated`

## Troubleshooting

- If the script fails on a specific module, run with `--module <name>` to isolate.
- Use `--dry-run` to see what would be regenerated without LLM calls.
- Use `--stats` to verify the graph builder sees the expected modules/functions.
- Use `--full` to force a complete regeneration (ignores cache).

## Cost

- Full run: ~$1-3 (Sonnet)
- Incremental (typical weekly): ~$0.10-0.50
