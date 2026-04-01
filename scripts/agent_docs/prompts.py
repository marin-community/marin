# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for agent-optimized documentation generation.

The prompt design is informed by autodoc variation experiments (R5). Key findings:
- Heavy prose explaining WHY defaults matter is the only format where weaker
  models (haiku) correctly propagate parameter defaults (V3, 80% accuracy).
- Fenced code blocks with full import paths are the only format where haiku
  gets import paths right (V4, correct on IMPORT_PATH criterion).
- Neither alone achieved 100%. PACKAGE_PROMPT combines both: a conceptual
  overview that explains domain knowledge encoded in defaults, followed by
  fenced code blocks showing exact import paths and full signatures.
"""

PACKAGE_PROMPT = """\
You are generating a sub-package reference card for an AI coding agent working \
on the Marin monorepo. The agent will read this doc to understand the package's \
API before reading source code. It needs a mental model of HOW the package \
works, exact import paths, and the domain knowledge behind default values.

Given the source code below for package `{package_name}`, produce a markdown \
document with EXACTLY these sections:

## Overview

1-2 paragraphs explaining:
- What this package does and the processing pipeline / data flow it implements.
- What domain knowledge is encoded in default parameter values and WHY those \
values were chosen. For example, if a function defaults to `num_perms=286`, \
explain that 286 permutations with 14 bands of 20 rows targets a ~0.8 Jaccard \
threshold — don't just say "286 is the default". The agent needs to understand \
whether to keep or change defaults.
- How the pieces fit together: which function do you call first, what does it \
return, and what do you feed that into?

## API

For each public function or class, provide a fenced Python code block showing \
the full import statement and complete call signature with types and defaults:

```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

def dedup_fuzzy_document(
    source: str,
    output: str,
    *,
    num_perms: int = 286,
    ngram_size: int = 5,
    num_bands: int = 14,
    seed: int = 42,
    max_parallelism: int = 128,
) -> str: ...
```
One sentence describing what it does and when to use it. `file_path:line_number`

Group by logical function (e.g., "Configuration", "Execution", "I/O"), not \
alphabetically. For classes, show the constructor signature and list key \
public methods as separate code blocks.

Skip private/internal items (underscore-prefixed). If the package has too many \
public symbols, keep the 15-20 most important and note what was omitted.

## Gotchas

Things an agent would get wrong without being told:
- Parameter constraints and valid ranges
- Non-obvious behaviors (e.g., function returns a path, not data)
- Common mistakes with argument ordering or types
- Implicit dependencies on other packages or runtime state

2-5 bullet points. Be specific — name the function and parameter.

Rules:
- Total output MUST be under 4KB.
- The Overview MUST explain WHY defaults have their values, not just list them.
- Code blocks MUST show full import paths from the top-level package \
(e.g., `from marin.processing.classification.deduplication.fuzzy import ...`, \
NOT `from marin.processing import ...`).
- Every API entry MUST include `file_path:line_number` so the agent can jump \
to source.
- If a function is a thin wrapper, say what it wraps instead of redescribing it.

{callee_context}

## Source code for package `{package_name}`:

{sources}
"""

MAP_PROMPT = """\
You are generating a package map for an AI coding agent working on the Marin \
monorepo. This file is loaded into EVERY agent conversation. It must be small \
and scannable — the agent uses it to decide which package doc to read next.

Given the package summaries below, produce a markdown document with EXACTLY \
these sections:

## Package Index
Group by library. For each package, one line in this exact format:
`package_name` — purpose sentence. → `docs/agent/packages/package_name.md`

Libraries: marin, levanter, haliax, fray, rigging, iris, zephyr, dupekit.

## Dependency Edges
Cross-library import edges only. Format: `A -> B` (A imports from B).
One line per edge. Skip intra-library edges.

## Entry Points
The 10-15 functions an experiment script or pipeline is most likely to call.
Format: `qualified.name(sig)` — one-line description.
Focus on marin.run, marin.execution, marin.training, marin.processing, \
levanter.main.

## Conventions
Bullet points covering:
- Config style (draccus dataclasses, composition)
- Artifact paths (MARIN_PREFIX, output path construction, version hashing)
- Execution patterns (StepSpec, RemoteCallable, Fray dispatch)

Rules:
- Total output MUST be under 4KB.
- No examples, no code blocks, no prose paragraphs.
- Every package line must include the path to its package doc file.

## Package Summaries:

{package_summaries}
"""

MERGE_PROMPT = """\
You are merging multiple partial package reference cards into one cohesive \
document for package `{package_name}`. Each partial doc covers a subset of the \
package's API.

Combine them into a single document with the same sections as the originals \
(Overview, API, Gotchas). Deduplicate and merge — do not repeat entries. \
The Overview should be a unified narrative, not concatenated paragraphs. \
Keep total output under 4KB. Preserve all `file_path:line` references and \
fenced code blocks exactly.

## Partial docs to merge:

{partial_docs}
"""
