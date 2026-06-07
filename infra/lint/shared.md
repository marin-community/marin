# Marin lint rules

Catalog of patterns reviewers in this repo recurrently flag. Each rule has a
short code (`ml-...`), the condition, why it's bad, when it's nevertheless
acceptable, and a bad-pattern example. Rules are *advisory* — surface findings
to the author, never block.

This file is the shared harness for the rule catalog in `infra/lint/`: the lane
files (`complexity.md`, `interfaces.md`, `robustness.md`, `cruft.md`,
`prose.md`) hold the rules; this file holds the detector usage, suppression,
confidence, overlap precedence, output format, and self-evaluation that govern
every lane. Each rule states the convention it enforces in its own "Why it's
bad" section. See "Detector usage" below for how the catalog is run against a
diff.

## Audience

- **Reviewer / agent**: scan a diff and emit findings in the format described
  under "Output format". See "Detector usage" for input selection.
- **Author**: search `infra/lint/` for the code from a finding (`ml-...`) — it
  lives in the lane file for its category — to see the rule, why it matters, and
  when it's OK to ignore.

This is not a security review (see `/security-review`), a correctness checker,
or a formatter (ruff / Black already exist; stay out of whitespace, import
order, line length).

---

## Detector usage

For agents running this catalog against a diff:

### Lanes & composition

`./infra/pre-commit.py --review` runs the catalog as a fan-out, not a single
pass. One headless agent runs per lane; each receives this harness, its own lane
file, and the branch diff, and emits findings in the "Output format" below. The
complexity lane additionally receives a `COMPLEXITY LEADS` block of advisory
static metrics. One lane is holistic: the **meta** lane (`meta.md`) reasons over
the whole change rather than scanning hunks, may read beyond the diff to confirm a
finding, and runs only on larger diffs (>100 changed lines). A final **composer**
agent then merges the lanes' outputs into one list.

Consequences for a lane agent: apply only your lane's rules, and resolve
overlap precedence *within* your lane. Cross-lane duplicates (the same line
flagged by two lanes) and cross-lane precedence are the composer's job — do not
second-guess other lanes. The composer never culls a finding on disagreement; it
only collapses true duplicates and drops precedence losers.

"Stay in scope" has two axes. *Rule scope* — the `ml-` codes you own — is narrow
for **every** lane, meta included: never emit another lane's code. *Diff scope* —
how much of the change you may read — is narrow for the five local lanes (the
changed hunks plus enough context to judge intent) and wide for the meta lane (the
whole diff, the call graph, sibling files). The meta lane fires only where a
hunk-scoped pass structurally cannot see the problem; if a single hunk would let a
local lane catch it, the meta lane defers.

`--lint-lane <name>` runs a single lane; `--no-lint-compose` skips the composer
and concatenates lanes (deduped) instead of reasoning over them.

### Inputs

Pick the diff that applies, typically the current code versus the merge base with `main`, unless the user explicitly requests a tighter set.

- Feature branch: `git diff main...HEAD -- '*.py' '*.proto'`.
- Pre-commit / pre-push: `git diff --cached -- '*.py' '*.proto'`.
- A specific PR: `gh pr diff <number> -- '*.py' '*.proto'`.
- A named file or two: read the file in full.

If the diff is empty, emit nothing and stop. If it is larger than one pass, drive a second pass file-by-file via `git diff main...HEAD --name-only -- '*.py' '*.proto'`. Do not sample or truncate.

Scan added/modified hunks plus enough surrounding context to judge intent (usually the enclosing function/class). Do not flag pre-existing code in unchanged regions. Migrations, `__init__.py` exports, proto definitions, and test fixtures all count. Generated stubs (e.g. `*_pb2.py`, `*_pb2_grpc.py`) are out of scope — skip them.

Security findings (auth, injection, secrets) are out of scope — they belong in `/security-review`.

### Suppression markers

A finding is suppressed when the line it cites carries a trailing
`# noqa: <code>` comment that names the rule — ruff's suppression convention.
`<code>` is the rule's `ml-...` code, or for `ml-local-import` the equivalent
ruff code `PLC0415`. A suppressed line is an author-approved exception: do not
emit a finding for it.

### Confidence

Every finding has a confidence in `[0.0, 1.0]`:

- `≥0.9` — example is near-verbatim; the reviewer comment writes itself.
- `0.7–0.9` — pattern fits the rule's intent; some context uncertainty.
- `<0.7` — do not emit.

Do not pad. Empty output is correct. False positives are the failure mode that erodes trust.

### Overlap precedence

Several rules touch adjacent surface.

If a single line legitimately violates two unrelated rules (e.g. `Any` return and a `_v2` suffix), emit two findings.

When rules overlap on one line, pick the more specific one and emit it alone:

- A comment that is both wrong *and* paraphrases the code → `ml-stale-inline-comment` (the staleness is the bigger problem). Plain restatement of correct code → `ml-restating-comment`.
- A flag-gated duplicate path → `ml-flag-gated-parallel-path` over `ml-parallel-source-impl` (the flag is the more specific shape) if that rule applies; otherwise `ml-parallel-source-impl`.
- A `_v2` / `_legacy` suffix on a function whose contrast no longer exists → `ml-vestigial-qualifier`, not `ml-misleading-name`.
- A function that does too much: boolean *knobs* in the signature drive the split → `ml-monolithic-function`; an ordinary signature wrapping a body that does too many distinct things → `ml-overloaded-function`. A class/module mixing unrelated responsibility clusters → `ml-god-class`.
- A whole-diff meta finding (e.g. `ml-shotgun-surgery`, `ml-echo-across-files`) and a local finding naming one instance of the same issue cite different lines → keep both: the meta finding names the aggregate, the local names the instance. The meta lane should already have deferred anything a single hunk exposes.

### Output format

One finding per line:

```
<path>:<line>: <code> (<confidence>) <message>
```

- `<path>` — repo-relative, forward slashes.
- `<line>` — 1-indexed in the file as it exists post-change.
- `<code>` — the `ml-...` code from the catalog.
- `<confidence>` — two decimals, e.g. `0.82`.
- `<message>` — ≤200 chars. State the concern; do not propose a fix.

Worked examples:

```
lib/iris/src/iris/cluster/worker/reconcile.py:284: ml-try-except-fallback (0.90) silent fallback contradicts docstring's MISSING contract; will mask cache bugs
lib/iris/src/iris/cluster/controller/transitions.py:1673: ml-bool-return-status (0.85) error: str | None encodes two unrelated transactions in one method
lib/marin/src/marin/processing/tokenize/tokenize_utils.py:1: ml-utils-module (0.75) module name uses generic _utils suffix
lib/iris/src/iris/cluster/worker/task_attempt.py:107: ml-speculative-abstraction (0.80) sentinel exists only to satisfy one import
```

If the diff is empty or has no Python files, emit nothing — no "no findings" message, no preamble, no summary, no JSON, no Markdown, no fenced code blocks. One finding per line, no blank lines between them. Do not echo the input.

### Self-evaluation

- **Precision over recall.** A reviewer who sees a false positive once trusts the tool less. When uncertain, suppress.
- **Calibration.** If you wouldn't bet $1 on a finding being valid, score it below 0.7.
- **Stay in scope.** Only the rules in your lane's file. Don't moonlight as a security / perf / style reviewer for things outside the catalog, or as another lane. The meta lane is the one exception on *diff* scope — it may read the whole change and beyond — but it still emits only meta codes.
- **Anchor in real shapes.** A reader at the cited line should immediately see why you flagged it. If you're reaching, suppress.
