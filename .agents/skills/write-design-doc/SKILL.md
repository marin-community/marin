---
name: write-design-doc
description: Produce a 1-page design proposal when the user explicitly asks for one. Do not use to answer or evaluate an idea inline.
---

# Design Doc Workflow

Use only when the user explicitly asks for a design doc, one-pager, or proposal.
Do not use for questions, walkthroughs, informal architecture reviews, or idea
evaluation. This skill owns the design artifacts and their publication.

## Artifacts and framing

Create '.agents/projects/<slug>/' with:

- design.md: the settled one-page proposal;
- research.md: the complete prior-work digest;
- spec.md: the public contract derived after design stabilizes.

Infer a short lowercase underscore slug and state the path. Stop only for a
collision. Ask questions only when missing rationale would change the design.
Confirm with the user after research, draft, spec, and before publish when a
decision is not obvious.

## Workflow

1. Frame the problem, motivation, and scope.

2. Research. Use background-research in design-doc mode (low or medium effort);
   an isolated Explore agent is preferred. Save in-repo findings, relevant prior
   art, surprises, and unknowns with file/line references to research.md. Link
   it from a 3–5 sentence Background section in design.md.

3. Interrogate. Ask 3–6 targeted questions in one batch about out-of-scope work, the smallest
   regression test, tradeoffs, and review questions. Do not ask questions that
   research can answer. Assume call sites are updated unless persisted data or
   an externally consumed API requires compatibility work.

4. Draft. Read .agents/projects/design-template.md and write design.md from the
   start; it is the source of truth. Target about 1000 words, cite real paths,
   and use one 10–30 line snippet only when prose is less clear. Keep Open
   Questions non-empty; if there are no unknowns, ask what feedback the user wants.
   Omit a compatibility section unless migration matters.
   Do not write spec.md until design.md is settled.

5. Spec. Always write spec.md, even for a tiny change. It pins what reviewers
   approve: typed public Python signatures with contract text covering behavior,
   edge cases, and ordering guarantees; complete proto definitions; file paths;
   persisted schema-registry CREATE statements, on-disk layout, file naming, and
   JSON/proto envelopes; every new error and its trigger plus changed error paths;
   and explicit out-of-scope items. It contains no algorithm
   pseudocode, sequenced implementation plan, or file-by-file how-to.

6. Stress-test. Before publishing, give both files to a Plan agent with this
   prompt: Review the design and spec for underspecified areas, weak motivation,
   missing tradeoffs, divergent implementations, and mismatched intent/contracts.
   Incorporate clear fixes into both files; ask the user only about ambiguous or
   load-bearing choices, then summarize incorporated and deferred feedback.

7. Publish. Use branch design/<slug> and .agents/skills/commit/SKILL.md for one
   commit adding design.md, research.md, and spec.md. Create the design label if
   missing with gh label create design --description "Design doc / 1-pager for review".
   Follow writing-style/pull-requests.md
   and ai-writing-donts.md. Use an imperative PR title, link all three files
   with absolute branch-rooted URLs, omit a stock discussion footer, and add
   labels design and agent-generated.
   Then send a two-line code-review Discord message with
   python scripts/ops/discord.py --channel code-review containing the PR title,
   URL, and framing sentence. Feedback lives on the PR; implementation is out
   of scope.

## Linking and source rules

Use SHA-pinned GitHub permalinks for load-bearing code citations, with the SHA
from git rev-parse main. In PR bodies, use branch-rooted URLs for sibling files;
relative paths do not resolve there.

The canonical template is .agents/projects/design-template.md. Do not use other
projects in that directory as style examples. If the user skips a phase, still
produce a non-empty Open Questions section, spec.md, and the Plan stress-test.
