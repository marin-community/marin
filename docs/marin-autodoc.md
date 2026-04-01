# Marin Agent-Optimized Documentation: Execution Plan

## Context

Agents working on Marin today either stuff raw source into context (expensive,
noisy) or rely on human-facing MkDocs tutorials. This project builds a
CI-integrated doc generator that produces a **three-tier documentation
hierarchy** in `docs/agent/`, designed for selective loading by agents based on
task scope. Docs are regenerated weekly via a scheduled skill, using the
`claude` CLI for LLM generation.

---

## Output Layout

```
docs/agent/
  MAP.md                        # Tier 1: module index (~4KB, always loaded)
  modules/
    marin.md                    # Tier 2: per-module reference cards
    marin.execution.md
    levanter.md
    haliax.md
    fray.md
    rigging.md
    iris.md
    zephyr.md
    ...
  api/
    marin.execution.yaml        # Tier 3: function-level structured docs
    marin.processing.yaml
    ...
  .cache.json                   # Content-addressed cache index
```

---

## File Layout

All generator code lives in `scripts/agent_docs/`. The entry point uses **uv
inline script metadata** (`# /// script`) to declare its own dependencies,
keeping the main workspace clean:

```
scripts/
  generate_agent_docs.py        # Entry point (uv script, ~30 LOC)
  agent_docs/
    __init__.py
    graph.py                    # tree-sitter dependency graph builder (~250 LOC)
    tier3.py                    # Function-level YAML generator (~300 LOC)
    tier2.py                    # Module summary generator (~100 LOC)
    tier1.py                    # MAP.md generator (~50 LOC)
    cache.py                    # Content-addressed caching (~100 LOC)
    prompts.py                  # LLM prompt templates (~100 LOC)
    claude_cli.py               # Subprocess wrapper for claude CLI (~60 LOC)
```

Total: ~990 LOC across 8 files.

### Dependency Isolation

The entry point declares deps via uv script syntax — no changes to
`pyproject.toml`:

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tree-sitter==0.24.0",
#     "tree-sitter-python==0.23.6",
#     "tree-sitter-rust==0.23.2",
#     "pyyaml>=6.0",
# ]
# ///
```

This gives us tree-sitter for multi-language parsing (Python + Rust) without
polluting the workspace. Versions are pinned for reproducibility.

---

## Component Details

### 1. `graph.py` — Tree-Sitter Dependency Graph Builder

Uses **tree-sitter** with Python and Rust grammars to parse all source files
under `lib/` and `rust/`. Tree-sitter advantages over `ast`:
- Parses Python + Rust with the same framework
- Handles malformed/partial code gracefully (no `SyntaxError` on WIP files)
- Gives concrete syntax trees with byte ranges (precise source extraction)

**Inputs**: workspace member paths from `pyproject.toml` `[tool.uv.workspace]`
for Python, plus `rust/` directory for Rust crates.

**Extracts per Python file** (via `tree_sitter_python`):
- Module-level imports → module dependency edges
- Top-level function defs with full signatures
- Top-level class defs with method signatures
- `__all__` if present (for public API filtering)

**Extracts per Rust file** (via `tree_sitter_rust`):
- `pub fn` and `pub struct` definitions with signatures
- `use` statements → crate dependency edges
- `impl` blocks with their public methods
- `#[pyfunction]` / `#[pyclass]` attributes (PyO3 exports — these are the
  Python-visible API from `rust/dupekit/`)

**Public API heuristic**:
- Python: if `__all__` exists, use it. Otherwise, non-underscore top-level names.
- Rust: `pub` items only. PyO3-annotated items get flagged as cross-language API.

**Output data structures**:

```python
@dataclass
class FunctionInfo:
    name: str
    qualified_name: str          # e.g. "marin.execution.executor.executor_main"
    signature: str               # e.g. "(steps: list[StepSpec], **kwargs) -> None"
    source: str                  # raw source text
    source_hash: str             # sha256 of source
    file_path: str               # relative path
    line_number: int
    language: str                # "python" or "rust"
    calls: list[str]             # qualified names of callees (best-effort)
    is_public: bool

@dataclass
class ModuleInfo:
    name: str                    # e.g. "marin.execution" or "dupekit"
    functions: list[FunctionInfo]
    classes: list[ClassInfo]     # similar to FunctionInfo
    imports_from: list[str]      # other modules this imports from
    file_paths: list[str]
    language: str                # "python" or "rust"

@dataclass
class RepoGraph:
    modules: dict[str, ModuleInfo]
    libs: list[str]              # ["marin", "levanter", ..., "dupekit"]
```

**Call graph extraction**: best-effort from tree-sitter `call_expression` nodes.
Resolves attribute calls against imports. Unresolved calls are silently
dropped — the call graph is advisory, not authoritative.

**Module grouping**: group by top-level package + first subpackage. E.g.,
`marin.execution.executor` and `marin.execution.step_spec` both belong to
module `marin.execution`. Rust crates are one module per crate (`dupekit`).
This keeps Tier 2 docs at a manageable granularity (~25-30 modules total).

### 2. `cache.py` — Content-Addressed Caching

Mirrors the `StepSpec` hashing philosophy from
`lib/marin/src/marin/execution/step_spec.py`.

**Cache file**: `docs/agent/.cache.json`

```json
{
  "marin.execution.executor.executor_main": {
    "source_hash": "a1b2c3d4",
    "dep_doc_hashes": ["e5f6g7h8", "i9j0k1l2"],
    "doc_hash": "m3n4o5p6",
    "tier": 3
  },
  "marin.execution": {
    "input_hash": "q7r8s9t0",
    "doc_hash": "u1v2w3x4",
    "tier": 2
  }
}
```

**Invalidation**: a function's doc is stale if its `source_hash` changed OR any
of its `dep_doc_hashes` changed. A module's Tier 2 doc is stale if any of its
constituent Tier 3 docs changed. Tier 1 is stale if any Tier 2 changed.

**Propagation**: walk the dependency graph upward from changed nodes. A changed
callee invalidates its callers' docs (since caller docs incorporate callee
summaries).

### 3. `claude_cli.py` — Claude CLI Wrapper

Thin subprocess wrapper around `claude --print --bare`:

```python
def generate(prompt: str, *, model: str = "sonnet", max_budget_usd: float = 0.50) -> str:
    result = subprocess.run(
        ["claude", "--print", "--model", model, "--max-budget-usd", str(max_budget_usd),
         "--system-prompt", "You are a precise documentation generator. Output only what is requested."],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr}")
    return strip_markdown_fences(result.stdout)
```

**Flags used**:
- `--print`: non-interactive, print and exit
- `--system-prompt`: override CLAUDE.md context with a minimal prompt
- `--model sonnet`: use Sonnet for cost efficiency (Tier 1 final pass can use opus)
- `--max-budget-usd 0.50`: per-call cost cap as safety rail

**Note**: `--bare` cannot be used because it disables OAuth/keychain auth.
Response text is post-processed with `strip_markdown_fences()` since the LLM
often wraps YAML output in ```yaml fences despite being told not to.

For Tier 3 (function docs), we can batch multiple small functions into one call
to reduce overhead. Target: ~10-20 functions per call where functions are in the
same module.

### 4. `tier3.py` — Function-Level Doc Generator

**Algorithm**:
1. Topo-sort the call graph (leaves first).
2. For each public function/class:
   a. Check cache — skip if source + dep hashes unchanged.
   b. Build prompt: function source + already-generated callee docs + module context.
   c. Call `claude_cli.generate()`.
   d. Parse output as YAML, validate structure.
   e. Write to `docs/agent/api/{module}.yaml` (append/update).
   f. Update cache.

**Batching**: group functions by module. For each module, send one prompt
containing all its uncached public functions (up to ~20 at a time). The prompt
asks for YAML output with one entry per function.

**Prompt template** (in `prompts.py`):

```
You are generating structured API documentation for an AI agent.
For each function below, produce a YAML entry with these fields:
- signature, summary (one sentence), params (name: description), returns,
  depends_on (list of called functions), defined_in (file:line).

Do NOT add examples or prose. Be precise about types and behavior.

## Already-documented callees (for context):
{callee_docs_yaml}

## Functions to document:
{function_sources}

Output valid YAML only. No markdown fences.
```

**Output format** (`docs/agent/api/marin.execution.yaml`):

```yaml
marin.execution.executor.executor_main:
  signature: "(steps: list[StepSpec], **kwargs) -> None"
  summary: "Entry point for running a DAG of steps."
  params:
    steps: "List of StepSpec objects defining the DAG"
  returns: "None"
  depends_on: [StepSpec, StepStatus]
  defined_in: "lib/marin/src/marin/execution/executor.py:142"
```

### 5. `tier2.py` — Module Summary Generator

For each module, feeds all its Tier 3 YAML entries to the LLM.

**Prompt asks for**:
- Purpose (2-3 sentences)
- Public API surface (function/class name + one-liner, grouped logically)
- Internal dependency graph (what other modules this calls)
- Key abstractions (3-5 most important types)
- Gotchas (things an agent would get wrong)

**Constraint**: output must be < 8KB. Prompt includes the target size.

**Output**: `docs/agent/modules/{module}.md`

### 6. `tier1.py` — MAP.md Generator

Single LLM call with all Tier 2 summaries concatenated.

**Prompt asks for**:
- Module index (one line per module with purpose)
- Dependency edges (flat list: `A -> B`)
- Entry points (5-10 most common functions with signatures)
- Conventions (naming, config style, artifact paths)

**Constraint**: output must be < 4KB.

**Output**: `docs/agent/MAP.md`

### 7. `generate_agent_docs.py` — Entry Point

```python
#!/usr/bin/env -S uv run
"""Generate agent-optimized documentation for Marin."""

from agent_docs.graph import build_repo_graph
from agent_docs.cache import load_cache, save_cache, compute_stale_set
from agent_docs.tier3 import generate_tier3
from agent_docs.tier2 import generate_tier2
from agent_docs.tier1 import generate_tier1

def main():
    graph = build_repo_graph()
    cache = load_cache()
    stale = compute_stale_set(graph, cache)

    # Bottom-up generation
    updated_t3 = generate_tier3(graph, cache, stale)
    updated_t2 = generate_tier2(graph, cache, stale, updated_t3)
    generate_tier1(graph, cache, updated_t2)

    save_cache(cache)
```

CLI flags:
- `--full`: ignore cache, regenerate everything
- `--module <name>`: regenerate only a specific module (for debugging)
- `--dry-run`: print what would be regenerated without calling LLM
- `--model <model>`: override default model (default: sonnet)

---

## Scheduled Skill

Create `.agents/skills/autodoc/SKILL.md`:

```yaml
---
name: autodoc
description: Generate agent-optimized documentation for Marin libraries weekly.
schedule_cron: "0 0 9 * * 1"
schedule_tz: America/New_York
---
```

The skill runs `uv run scripts/generate_agent_docs.py`, commits changed docs
to a branch, and opens/updates a PR.

---

## Scope Boundaries

**Include**:
- All `.py` files under `lib/{marin,levanter,haliax,fray,rigging,iris,zephyr}/src/`
- All `.rs` files under `rust/dupekit/src/`

**Exclude**:
- `experiments/` — one-off scripts, self-documenting
- `tests/` — no documentation value
- `*_test.py`, `test_*.py` — same
- Vendored code (`dolma`, `chatnoir-resiliparse`) — skip or one-time manual
- `__pycache__`, `*.pyc`

---

## Implementation Order

### Phase 1: Graph builder + cache (day 1)
1. `scripts/agent_docs/__init__.py`
2. `scripts/agent_docs/graph.py` — tree-sitter parser, `RepoGraph` construction
3. `scripts/agent_docs/cache.py` — load/save/invalidation logic
4. Test: run graph builder, verify module count (~20-30), function count (~3K),
   cross-lib import edges look correct. Print stats.

### Phase 2: Claude CLI wrapper + Tier 3 (day 2)
1. `scripts/agent_docs/claude_cli.py` — subprocess wrapper
2. `scripts/agent_docs/prompts.py` — prompt templates
3. `scripts/agent_docs/tier3.py` — batched function doc generation
4. Test: generate docs for one small module (e.g., `rigging` with 5 files),
   verify YAML is valid and parseable.

### Phase 3: Tier 2 + Tier 1 (day 3)
1. `scripts/agent_docs/tier2.py` — module summary generation
2. `scripts/agent_docs/tier1.py` — MAP.md generation
3. Test: full pipeline on `rigging` + `fray` (small libs). Verify all three
   tiers are generated and consistent.

### Phase 4: Full run + skill setup (day 4)
1. `scripts/generate_agent_docs.py` — entry point with CLI flags
2. Full generation across all 7 libs
3. `.agents/skills/autodoc/SKILL.md` — scheduled skill
4. Review output quality, tune prompts as needed

### Phase 5: Integration (day 5)
1. Wire `docs/agent/MAP.md` into agent loading (update CLAUDE.md or AGENTS.md
   to reference it)
2. Prompt tuning based on real agent usage
3. Add `--dry-run` stats output for monitoring

---

## Verification

1. **Graph correctness**: `--dry-run` prints module count, function count,
   cross-lib edges. Spot-check against manual `grep` for known functions like
   `executor_main`, `default_train`.
2. **Cache correctness**: run twice with no source changes → zero LLM calls on
   second run. Change one function → only that function + its callers
   regenerate.
3. **Output quality**: manually review Tier 2 docs for 2-3 modules against
   actual source. Check that gotchas are non-trivial and API signatures match.
4. **Size constraints**: `wc -c docs/agent/MAP.md` < 4096; each module doc
   < 8192.
5. **YAML validity**: `python -c "import yaml; yaml.safe_load(open(f))"` for
   all Tier 3 files.
6. **End-to-end**: give an agent a task, observe it loading MAP.md → module
   doc → specific function doc. Measure context used vs raw source approach.

---

## Cost Estimate

- Full generation (~3K functions, ~25 modules): ~$1-3 with Sonnet
- Weekly incremental (typical: 50-200 changed functions): ~$0.10-0.50
- Tier 1 regeneration (single call): < $0.01
