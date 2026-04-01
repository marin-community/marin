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

Two-phase architecture for large packages:
- FILE_SUMMARY_PROMPT: per-file structured summary (phase 1)
- PACKAGE_PROMPT: aggregation from file summaries OR direct from source (phase 2)
- MERGE_PROMPT: merging partial docs (fallback for very large aggregations)
"""

FILE_SUMMARY_PROMPT = """\
You are summarizing a single source file for a downstream aggregator that will \
produce a package reference card. Your output is structured data, not prose.

Summarize `{file_path}` from package `{package_name}`.

Output format — follow EXACTLY:

FILE: {file_path}
PURPOSE: 1-2 sentences describing what this file provides.

ITEMS:
- `{package_name}.ItemName` — `signature_with_types_and_defaults` — One-line description.

GOTCHAS:
- Any non-obvious constraints, implicit dependencies, or common mistakes. Skip if none.

Rules:
- Include ONLY public items (no underscore-prefixed names).
- Signatures must include full type annotations and default values.
- For classes, show the constructor signature AND list key public methods as sub-items.
- Total output MUST be under 1500 chars. Be ruthlessly concise.
- Do NOT include import statements, code blocks, or markdown formatting.
- Do NOT explain obvious things. The aggregator can read source.

## Source code:

{sources}
"""

PACKAGE_PROMPT = """\
You are generating a sub-package reference card for an AI coding agent working \
on the Marin monorepo. The agent reads this doc to understand the package API \
before reading source code. It needs a mental model of HOW the package works, \
exact import paths, and the domain knowledge behind default values.

{input_description}

Produce a markdown document with EXACTLY these sections:

## Overview

1-2 paragraphs explaining:
- What this package does and the processing pipeline / data flow it implements.
- What domain knowledge is encoded in default parameter values and WHY those \
values were chosen. The agent needs to understand whether to keep or change defaults.
- How the pieces fit together: which function do you call first, what does it \
return, and what do you feed that into?

Be a SUMMARY, not a regurgitation. The agent can always read source code — \
your job is to provide the mental model that source code alone does not convey.

## API

For each public function or class, provide a fenced Python code block showing \
the full import statement and complete call signature with types and defaults:

```python
from {package_name}.module import function_name

def function_name(
    arg: Type = default,
) -> ReturnType: ...
```
One sentence describing what it does. `file_path:line_number`

Group by logical function, not alphabetically. For classes, show the constructor \
signature and list key public methods as separate code blocks.

Keep the 15-20 most important items. Note what was omitted if anything.

## Gotchas

2-5 bullet points. Things an agent would get wrong without being told:
- Parameter constraints, valid ranges
- Non-obvious behaviors (returns a path, not data)
- Common mistakes with argument ordering or types

Rules:
- Total output MUST be under 3KB. Aim for 2KB.
- Code blocks MUST show full import paths from the top-level package.
- Every API entry MUST include `file_path:line_number`.
- If a function is a thin wrapper, say what it wraps instead of redescribing it.
- Do NOT repeat information already obvious from signatures.

{callee_context}

## {input_section_header}

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
