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

Budgeted two-axis taxonomy (experiment v2):
- OVERVIEW_PROMPT: per sub-project 30-second orientation + intent router.
- OPS_PROMPT: "how to USE project X" — entry points and the happy-path call.
- ARCHITECTURE_PROMPT: "how to UNDERSTAND and CHANGE project X" — mental model
  plus file-level "where would I change this" pointers.
Each is budgeted to ~1000 tokens.

Eval harness prompts:
- CODER_PROMPT: the weak, tool-disabled coder writing a script from a doc bundle.
- JUDGE_PROMPT: the judge scoring that script against embedded ground truth.
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

OVERVIEW_PROMPT = """\
You are generating the 30-second orientation card for sub-project \
`{project_name}` in the Marin monorepo. An AI coding agent reads this FIRST to \
decide which deeper doc to open next. It is a router, not a reference.

{input_description}

Produce a markdown document with EXACTLY these sections:

## What it is
1-2 sentences: what `{project_name}` is and the one problem it solves.

## Files that matter
The 3-5 files or modules an agent must know to be productive here, one line \
each: `path/to/file.py` — what lives there. Pick load-bearing entry points, \
not every file.

## Where to go next
Two lines that route by intent:
- USE / "how do I run or call it?" -> `{project_name}/ops.md`
- UNDERSTAND or CHANGE / "how does it work, where do I edit?" -> \
`{project_name}/architecture.md`

Rules:
- Total output MUST be under ~1000 tokens (~4000 characters).
- No code blocks, no API dumps — this only orients and routes.
- Name real files from the sources; do not invent paths or symbols.

## Sources:

{sources}
"""

OPS_PROMPT = """\
You are generating the "how to USE `{project_name}`" doc for an AI coding agent \
on the Marin monorepo. It answers "how do I run/call this?" — entry points, the \
happy-path call, required vs default parameters, and where artifacts land.

{input_description}

This sub-project contains MANY packages and tasks; an agent may need any of \
them, so prioritize BREADTH of coverage over a single deep example. Produce a \
markdown document with EXACTLY these sections:

## Entry point index
The 12-20 most important callable entry points across the WHOLE sub-project — \
the functions/classes/CLIs an agent is most likely to need for real tasks. \
Span the sub-project's major areas (do not over-cover one area). One line each:

`full.import.path.symbol(key_arg=default, required_arg)` — one-line purpose.

Include the few load-bearing arguments inline with their default VALUES; mark \
required args. Cover the long tail tersely rather than omitting whole areas.

## Happy path
ONE short fenced code block for the single most common end-to-end workflow, with \
full import paths and real argument names.

## Critical defaults
Only the handful of non-obvious required params or tuned defaults (with exact \
values) that an agent would otherwise get wrong. Skip the obvious ones.

## Where artifacts land
1-3 lines: output paths / return values — does a call return data, a path, or \
write files? Name the path-construction convention if there is one.

Rules:
- Total output MUST be under ~1000 tokens (~4000 characters).
- Code blocks MUST show full import paths from the top-level package.
- Every required parameter and every non-obvious default MUST appear with its \
exact value.
- Use ONLY names — functions, classes, parameters, and dataclass fields — that \
appear VERBATIM in the digest below. NEVER invent a parameter or field name. If \
a detail is not in the digest, omit it rather than guessing.

## Sources:

{sources}
"""

ARCHITECTURE_PROMPT = """\
You are generating the "how to UNDERSTAND and CHANGE `{project_name}`" doc for \
an AI coding agent on the Marin monorepo. It answers "how does this work, and \
where would I make a given change?" — the mental model plus file-level pointers.

{input_description}

Produce a markdown document with EXACTLY these sections:

## Mental model
1-2 prose paragraphs: the components, how data flows between them, and WHY the \
design is the way it is. Prose, not bullets — the point is to convey reasoning \
that signatures alone cannot. Source code is always available; give the model \
the model.

## Key invariants
2-5 bullets: properties the code relies on that an editor must not break \
(determinism, ordering, all-or-nothing admission, state-machine transitions). \
State the invariant and, in one phrase, where it is enforced.

## Where to change things
The load-bearing section. For 3-6 likely changes, give a file-path-level \
pointer: "To change X, edit `path/to/file.py` (function/class `name`)." These \
must be real paths and symbols from the sources.

Rules:
- Total output MUST be under ~1000 tokens (~4000 characters).
- The "Where to change things" pointers MUST cite real `path:symbol` from the \
sources; do not invent.
- Use ONLY names that appear VERBATIM in the digest; never invent symbols, \
parameters, or fields. Omit a detail rather than guess.
- Prefer the mental model over an API dump — `ops.md` carries the call details.

## Sources:

{sources}
"""

CODER_PROMPT = """\
You are a Python engineer working on the Marin monorepo. You have NO tools: you \
cannot run code, read files, or search the repo. You must rely ONLY on the \
documentation below.

## Documentation

{bundle}

---

## Task

{task}

Output ONLY the script — a single Python file's contents, no prose, no \
explanation, no markdown fences.
"""

JUDGE_PROMPT = """\
You are a senior engineer reviewing a script written by a weaker model that had \
ONLY documentation to work from. Judge the script against the ground truth \
below. You are reviewing the TEXT of the script — do not execute it.

## Task the script was meant to accomplish

{task}

## Ground truth: anchors (the KEY APIs a correct answer must use)

These are the load-bearing symbols for THIS task, with their real file paths. \
They are checkpoints, NOT an exhaustive list of every allowed API — a correct \
script will also use many other real helper and standard-library APIs that are \
not listed, and that is completely fine.

{anchors}

{forbidden_block}
## How to score each anchor

For each anchor SYMBOL, score 1 if the script uses it correctly — right name, \
right import path where applicable, right required params and default values; \
score 0 if it is missing-but-required, or used in a way that contradicts the \
ground truth. An anchor the task did not require the script to call directly \
scores 1 unless the script contradicts it.

## What counts as a hallucination (STRICT, high bar)

Put something in "hallucinated" ONLY if it is one of:
1. a term in the forbidden list above; OR
2. a use that DIRECTLY CONTRADICTS an anchor — e.g. calling an anchor function \
with a parameter or dataclass field that does not exist on it, or with a wrong \
default value; OR
3. an obviously fabricated method or field on a class named in the anchors.

Do NOT flag an API merely because it is not in the anchor list. Unlisted but \
plausible helpers, imports, and standard-library calls are NOT hallucinations. \
When unsure, do not flag it.

## Output

Output ONLY a JSON object, no other text, with EXACTLY these keys:
- "anchors": object mapping each anchor symbol (exact string) to 0 or 1.
- "hallucinated": array of strings — ONLY items meeting the strict bar above. \
Empty array if none.
- "quality": integer 1-5 — holistic: would this actually work, is it idiomatic, \
does it use the right entry points? (5 = ship it; 1 = wrong approach).
- "notes": one-paragraph justification of the quality score.

## Script to review

```python
{script}
```
"""

SHORTEN_PROMPT = """\
The markdown document below is {tokens} tokens but MUST be at most {budget} \
tokens. Shorten it to fit while preserving the highest-value content for an AI \
coding agent: exact import paths, required parameters and their default VALUES, \
and `file:symbol` pointers. Cut prose, redundancy, and the least-critical \
entries first. Keep the same section headings. Output ONLY the shortened \
markdown — no commentary.

## Document

{document}
"""

TAXONOMY_MAP_PROMPT = """\
You are generating the top-level map for the Marin monorepo's agent docs. This \
file is loaded into EVERY agent conversation, so it must be tiny and scannable: \
the agent reads it to decide which sub-project doc to open next.

Given the per sub-project overviews below, produce a markdown document with \
EXACTLY these sections:

## Sub-project Index
One line per sub-project, in this exact format:
`project` — one-sentence purpose. → `docs/agent/<project>/overview.md`

## Dependency Edges
Cross-sub-project edges only, one per line: `A -> B` (A depends on B). Skip \
intra-project edges.

## Where to look
Two lines reminding the agent of the two-axis split:
- "How do I use X?" -> `docs/agent/<project>/ops.md`
- "How does X work / where do I change it?" -> `docs/agent/<project>/architecture.md`

Rules:
- Total output MUST be under ~1000 tokens (~4000 characters).
- No code blocks, no prose paragraphs.

## Sub-project overviews:

{overviews}
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
