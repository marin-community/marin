# Marin lint sub-prompts

One Markdown file per detector category. These are sub-prompts; the entry
point is `infra/lint.md`, which `@`-includes every file in this directory
along with the severity / confidence rubric and output format.

This is not a static rule catalog — each detector is a curated prompt
that pairs an AGENTS.md style rule with anchor examples lifted from real
PR review comments.

## How to run the detector

The runner is whatever agent harness is available. Two recipes:

**Inside Claude Code or another interactive agent:**
> "Run the Marin lint at `@infra/lint.md` against `git diff main...HEAD`."

The `@`-include in `lint.md` pulls every sub-prompt automatically.

**Headless / CI:**
```bash
git diff main...HEAD | claude -p "$(cat infra/lint.md)" --add-dir .
```

Output format (ruff-compatible, defined in `infra/lint.md`):
```
<path>:<line>: M-<category> [<severity>] (<confidence>) <message>
```

Detector is *advisory*. It never blocks a push. A pre-push hook can surface
findings as a warning, and CI can post them as PR comments, but
acceptance/dismissal is a human decision.

## Prompt file format

Each sub-prompt is plain Markdown with this structure:

```markdown
# <category> — detector prompt

## AGENTS.md anchor
<one line citing the rule(s) this detector enforces, e.g.
"AGENTS.md § API Design, § Types & Data Structures">

## What to look for
<2-3 sentences. "Flag code that ...">

## Anchor examples
5-8 examples:
- Reviewer quote (≤200 chars)
- Code shape it was responding to (1-2 lines)
- One sentence on why it's a real concern

## False-positive guidance
3-5 bullets on when the pattern is acceptable.

## Suggested confidence floor
One sentence on when to raise the bar.
```

Stay under ~500 words per file. `lint.md` `@`-includes all of them; the
total system-prompt budget is ~6K tokens.

Each detector should cite the AGENTS.md section it enforces by name, not
re-paraphrase the rule. AGENTS.md is the single source of truth; if it
changes, the detector inherits the change automatically. Detectors exist
to add anchor examples and false-positive guidance the bare rule doesn't.

## Cross-detector overlap

Several categories touch adjacent surface — bool→enum sits between
`api-shape` and `types`, silent fallback sits between `defensive` and
`config-explicitness`, vestigial qualifiers sit between `naming` and
`dead-code`. `infra/lint.md` § "Cross-detector precedence" assigns one
canonical home per overlapping shape so the runtime doesn't emit duplicate
findings. When adding a new category, update that section.

## Adding a new category

1. Write `infra/lint/<category>.md` following the format above. Cite the
   AGENTS.md rule by section; anchor with quoted reviewer language; supply
   false-positive guidance.
2. Add a corresponding `@infra/lint/<category>.md` line to `infra/lint.md`
   under the "Detector library" section so the include picks it up.
3. Assign a default severity (`nit` / `warn`) in the severity rubric in
   `lint.md`.
4. If the new category overlaps an existing one, add a tiebreaker bullet
   under "Cross-detector precedence" in `lint.md`.

## Versioning

Prompts are checked into the repo; their git history *is* the version
log. A reviewer who wants to understand why a category fired the way it
did reads the current prompt; a reviewer debugging a false positive reads
the git blame to see how the prompt evolved.

## What this is not

- **Not a static rule catalog.** Adding patterns means writing prose, not
  AST visitors. The detector is an LLM call, not a regex.
- **Not a replacement for human review.** It catches recurring nits so
  humans spend review time on the harder calls (architecture,
  correctness, perf).
- **Not a blocker.** Advisory by design; both pre-push and CI surface
  findings as suggestions, not failures. Hard-failing on fuzzy LLM
  judgments is how teams stop trusting the tool.
