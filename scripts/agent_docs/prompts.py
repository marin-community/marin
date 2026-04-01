# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for agent-optimized documentation generation."""

TIER3_PROMPT = """\
You are generating structured API documentation for an AI coding agent.
For each function/class below, produce a YAML entry with these exact fields:
- signature: the full signature string
- summary: one sentence describing what it does (be precise about behavior, not vague)
- params: mapping of param name to description (skip self/cls)
- returns: what the function returns
- depends_on: list of other functions/classes this calls or uses from other modules
- defined_in: "file_path:line_number"

Rules:
- Output valid YAML only. No markdown fences, no commentary.
- Be precise about types and behavior. Name concrete types, not "object" or "value".
- For classes, document the class itself and list its public methods with signatures.
- If a function is a thin wrapper, say so and name what it wraps.
- Do NOT add examples, prose, or usage notes.

{callee_context}

## Functions/classes to document:

{sources}

Output valid YAML only. Each top-level key is the qualified name.
"""

TIER2_PROMPT = """\
You are generating a module reference card for an AI coding agent.
Given the structured API docs below for module `{module_name}`, produce a markdown \
document with EXACTLY these sections:

## Purpose
2-3 sentences describing what this module does and when an agent would use it.

## Public API
For each public function/class, one line: `name(signature)` — one-sentence description.
Group logically (e.g., "Configuration", "Execution", "Utilities"). Skip private/internal items.

## Dependencies
Which other modules this imports from, as a flat list: `module_name` — what it uses from there.

## Key Abstractions
The 3-5 most important types/classes and what they represent. One line each.

## Gotchas
Things an agent would get wrong without being told. Non-obvious behaviors, implicit \
contracts, common mistakes. 2-5 bullet points.

Rules:
- Keep the total output under 8KB.
- No examples, no code blocks, no tutorials.
- Be specific and actionable, not vague.

## Module API docs:

{api_docs}
"""

TIER1_PROMPT = """\
You are generating a module map for an AI coding agent working on the Marin monorepo.
Given the module summaries below, produce a markdown document with EXACTLY these sections:

## Module Index
One line per module: `module_name` — purpose (what it does, not what it contains).
Group by library (marin, levanter, haliax, fray, rigging, iris, zephyr, dupekit).

## Dependency Edges
Flat list of import edges: `A -> B` (A imports from B). Only cross-library edges.

## Entry Points
The 10-15 most common functions an experiment script or pipeline would call.
Format: `qualified.name(signature)` — one-line description.
Focus on functions in marin.run, marin.execution, marin.training, marin.processing, \
levanter.main, and experiment defaults.

## Conventions
- Config style (draccus dataclasses, how they compose)
- Artifact paths (MARIN_PREFIX, how output paths are constructed)
- Naming patterns (step names, version hashing)
- Common patterns (RemoteCallable for Fray dispatch, StepSpec for DAG nodes)

Rules:
- Keep total output under 4KB.
- No examples, no code blocks.
- Be precise and agent-actionable.

## Module Summaries:

{module_summaries}
"""
