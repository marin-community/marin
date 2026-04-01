# Marin Agent-Optimized Documentation

## What this is

A doc generator that produces package-level reference cards for AI agents
working on Marin. Two tiers, all markdown:

1. **MAP.md** (~4KB) — package index, auto-loaded into every agent conversation
   via `@docs/agent/MAP.md` in `AGENTS.md`. Agents scan this to decide which
   package doc to read.

2. **Package docs** (~2-4KB each) — per-package reference cards with API surface,
   `file:line` references, and gotchas. Agents read on demand via `Read` tool.

The flow: **MAP.md → package doc → source code**. Two hops.

## Output layout

```
docs/agent/
  MAP.md                                              # Auto-loaded via @AGENTS.md
  packages/
    marin.processing.classification.deduplication.md   # Read on demand
    iris.cluster.controller.md
    rigging.distributed_lock.md
    ...
```

## Running

```bash
./scripts/generate_agent_docs.py                    # incremental update
./scripts/generate_agent_docs.py --full             # regenerate everything
./scripts/generate_agent_docs.py --dry-run          # show what would change
./scripts/generate_agent_docs.py --package <name>   # single package
./scripts/generate_agent_docs.py --stats            # package stats only
```

Dependencies are isolated via uv inline script metadata — no changes to
`pyproject.toml`. Uses `tree-sitter` for Python + Rust parsing and `claude`
CLI for LLM generation.

## How it works

1. **Package discovery** (`scripts/agent_docs/packages.py`): tree-sitter parses
   all `.py` and `.rs` files under `lib/`. Groups by directory into packages
   (e.g. `marin.processing.classification.deduplication`). Resolves cross-package
   import edges.

2. **Package doc generator** (`scripts/agent_docs/tier2.py`): two paths based on
   package size:
   - **Small packages** (<30K chars source): direct LLM call with source → doc
   - **Large packages**: per-file summaries (haiku) → aggregation (sonnet)

3. **MAP generator** (`scripts/agent_docs/tier1.py`): feeds all package docs to
   a single LLM call, produces MAP.md.

4. **Caching** (`scripts/agent_docs/cache.py`): content-addressed by hash of
   public item source code. Changed source → regenerate that package + dependents.

## Weekly schedule

The `autodoc` skill (`.agents/skills/autodoc/SKILL.md`) runs weekly on Monday
9am ET. It regenerates stale package docs and opens a PR.
