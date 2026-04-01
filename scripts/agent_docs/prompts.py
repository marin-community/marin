# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for agent-optimized documentation generation."""

MODULE_PROMPT = """\
You are generating a module reference card for an AI coding agent working on \
the Marin monorepo. The agent will read this doc to understand the module's API \
before reading source code. It needs to know WHAT exists, WHERE it is, and \
what will trip it up — not HOW things work internally.

Given the source code below for module `{module_name}`, produce a markdown \
document with EXACTLY these sections:

## Purpose
2-3 sentences: what this module does and when an agent would touch it.

## Public API
For each public function/class, one line in this exact format:
- `name(signature)` — one-sentence description. `file_path:line`

Group logically (e.g., "Configuration", "Execution", "Utilities").
For classes, list public methods indented under the class entry.
Skip private/internal items (underscore-prefixed).

## Dependencies
Which other Marin modules this imports from, as a flat list:
- `module_name` — what it uses from there.

Only list cross-module dependencies within the monorepo (marin, levanter, \
haliax, fray, rigging, iris, zephyr, dupekit). Skip stdlib and third-party.

## Key Abstractions
The 3-5 most important types/classes and what they represent. One line each.

## Gotchas
Things an agent would get wrong without being told. Non-obvious behaviors, \
implicit contracts, common mistakes. 2-5 bullet points. Be specific — name \
the function or parameter that bites you.

Rules:
- Total output MUST be under 8KB.
- No examples, no code blocks, no tutorials.
- Be specific and actionable, not vague.
- Every API entry MUST include `file_path:line_number` so the agent can jump \
to source.
- If a function is a thin wrapper, say what it wraps instead of redescribing it.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""

MAP_PROMPT = """\
You are generating a module map for an AI coding agent working on the Marin \
monorepo. This file is loaded into EVERY agent conversation. It must be small \
and scannable — the agent uses it to decide which module doc to read next.

Given the module summaries below, produce a markdown document with EXACTLY \
these sections:

## Module Index
Group by library. For each module, one line in this exact format:
`module_name` — purpose sentence. → `docs/agent/modules/module_name.md`

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
- Every module line must include the path to its module doc file.

## Module Summaries:

{module_summaries}
"""
