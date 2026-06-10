# Agent documentation (generated)

This directory holds the **machine-generated agent-doc taxonomy** for the Marin
monorepo. Do not hand-edit these files — they are regenerated from source by the
autodoc pipeline, and edits will be overwritten.

## Layout

```
docs/agent/
  MAP.md                     monorepo index across sub-projects (~1000 tokens)
  <subproject>/overview.md   30-second orientation + intent router
  <subproject>/ops.md        how to USE the sub-project
  <subproject>/architecture.md   how to UNDERSTAND and CHANGE it
  packages/<package>.md      per-package API reference
```

Every taxonomy file is held to a ~1000-token budget so an agent can load the
docs it needs without burning context.

## Regenerating

```bash
# Two-axis taxonomy (overview/ops/architecture per sub-project + MAP.md):
uv run --script scripts/agent_docs/generate_taxonomy.py --output-dir docs/agent

# Per-package API reference + the package list:
uv run --script scripts/agent_docs/main.py --stats
```

The generation prompts, probe-based eval harness, and the experiment that tunes
all of this live under `scripts/agent_docs/`. See the `autodoc` skill
(`.agents/skills/autodoc/SKILL.md`) for the full workflow.
