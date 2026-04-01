# Marin Agent-Optimized Documentation

## What this is

A doc generator that produces module-level reference cards for AI agents
working on Marin. Two tiers, all markdown:

1. **MAP.md** (~4KB) — module index, auto-loaded into every agent conversation
   via `@docs/agent/MAP.md` in `AGENTS.md`. Agents scan this to decide which
   module doc to read.

2. **Module docs** (~2-8KB each) — per-module reference cards with API surface,
   `file:line` references, key abstractions, and gotchas. Agents read these on
   demand via `Read` tool.

The flow: **MAP.md → module doc → source code**. Two hops.

## How agents use it

1. Agent gets a task (e.g. "add a new data filter step")
2. MAP.md is already in context (auto-loaded via `@` reference in AGENTS.md)
3. Agent identifies relevant modules: `marin.execution`, `marin.transform`
4. Agent reads those 2 module docs (~8KB each, via `Read` tool)
5. Module docs have `file:line` for every public function → agent reads source

Total context: ~4KB (map) + ~16KB (2 modules) = **~20KB** instead of scanning
hundreds of source files.

## Output layout

```
docs/agent/
  MAP.md                         # Auto-loaded via @AGENTS.md
  modules/
    marin.execution.md           # Read on demand
    rigging.distributed_lock.md
    levanter.models.md
    ...
```

## Running

```bash
./scripts/generate_agent_docs.py                         # incremental update
./scripts/generate_agent_docs.py --full                   # regenerate everything
./scripts/generate_agent_docs.py --dry-run                # show what would change
./scripts/generate_agent_docs.py --module marin.execution # single module
./scripts/generate_agent_docs.py --stats                  # graph stats only
```

Dependencies are isolated via uv inline script metadata — no changes to
`pyproject.toml`. Uses `tree-sitter` for Python + Rust parsing and `claude`
CLI for LLM generation.

## How it works

1. **Graph builder** (`scripts/agent_docs/graph.py`): tree-sitter parses all
   `.py` files under `lib/` and `.rs` files under `rust/`. Extracts module
   structure, function/class signatures, import edges.

2. **Module doc generator** (`scripts/agent_docs/tier2.py`): for each stale
   module, feeds raw source of all public items to `claude --print --model sonnet`.
   LLM produces the markdown reference card directly from source — no
   intermediate format.

3. **MAP generator** (`scripts/agent_docs/tier1.py`): feeds all module docs to
   a single LLM call, produces MAP.md with one line per module + path to its
   doc file.

4. **Caching** (`scripts/agent_docs/cache.py`): content-addressed by hash of
   public item source code. Changed source → regenerate that module + dependents.

## Weekly schedule

The `autodoc` skill (`.agents/skills/autodoc/SKILL.md`) runs weekly on Monday
9am ET. It regenerates stale module docs and opens a PR.
