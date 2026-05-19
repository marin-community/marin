# Marin lint sub-prompts

One Markdown file per detector category. These are sub-prompts; the entry
point is `infra/lint.md`, which `@`-includes every file in this directory
along with the severity / confidence rubric and output format.

This is not a static rule catalog — each detector is a curated prompt
distilled from real review comments in this repo's history.

## Where the prompts came from

Distilled offline from `pr_reviews.db` (a local export of historical PR
review comments). The methodology:

1. **Filter to humans.** Bot authors (`chatgpt-codex-connector[bot]`,
   `claude[bot]`, `Copilot`, `github-advanced-security[bot]`,
   `ravwojdyla-agent`) are excluded; they over-flag patterns existing tools
   already catch and would dominate the corpus by raw count (~48%).
2. **Carve the corpus by category.** Each category gets keyword-filtered
   SQL queries to surface the relevant subset (~5-80 comments per category).
3. **Distill with a Haiku sub-agent per category.** Each agent reads its
   slice and produces:
   - A "what to look for" intent paragraph.
   - 5-8 anchor examples (quoted reviewer language + inferred code shape).
   - False-positive guidance (when the pattern is acceptable).
   - A confidence-floor recommendation.

The distillation is offline + monthly-ish; runtime is just `prompt + diff
→ findings`. No retrieval, no embeddings, no vector DB.

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
<path>:<line>:<col>: M-<category> [<severity>] (<confidence>) <message>
```

Detector is *advisory*. It never blocks a push. A pre-push hook can surface
findings as a warning, and CI can post them as PR comments, but
acceptance/dismissal is a human decision.

## Prompt file format

Each sub-prompt is plain Markdown with this structure:

```markdown
# <category> — detector prompt

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

Cross-prompt overlap is intentional — `types` and `api-shape` both flag
bool→enum from different angles; `defensive` and `config-explicitness`
both touch silent fallbacks. At runtime the model sees all relevant frames
and picks one.

## Adding a new category

1. Write `infra/lint/<category>.md` following the format above.
2. Anchor with quoted reviewer language from real corpus comments. Don't
   invent patterns the corpus doesn't show — false-positive cost is high.
3. Add a corresponding `@infra/lint/<category>.md` line to `infra/lint.md`
   under the "Detector library" section so the include picks it up.
4. Update the severity rubric in `lint.md` to assign a default severity
   (`nit` / `warn` / `block`) for the new category.

## Re-distilling from updated corpus

When `pr_reviews.db` is refreshed (monthly cadence is fine):

```bash
./scripts/distill_lint_prompts.py --category <name>  # re-run one category
./scripts/distill_lint_prompts.py --all              # re-run all categories
```

The distillation script fans out one Haiku sub-agent per category, each
with the same instructions used to bootstrap this library. Review the
diffs before merging — distillations vary slightly across runs. *(Script
not yet written; see `infra/lint.md` history for the agent template that
seeded the current library.)*

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
