---
name: review-pr-holistic
description: Review an explicitly identified Marin pull request as a whole change, asking whether it fits repository patterns, whether a better abstraction exists, and what breaks outside the diff; use for large or agent-authored PRs, after or alongside review-pr.
allowed-tools: Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(gh api:*), Bash(git diff:*), Bash(git grep:*), Bash(git log:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git show:*), Bash(rg:*), Bash(uv run pytest:*), Bash(uv run infra/codehealth/log_stats.py:*)
---

# Review a pull request as a whole

`review-pr` reports certain, line-anchored defects. This skill answers the
questions a senior reviewer asks after reading the entire change: does it fit
the repository, is there a smaller shape, and what does it break outside the
diff. It reads beyond the diff and reports design findings. With `--comment`,
it maintains one overview comment and may post a short follow-up with new
findings. Otherwise it is read-only.

Run it at the exact head named by the caller. Review agent-generated pull
requests; they are the population where structural findings concentrate.

## Load the context

1. Read the pull-request title, body, issue comments, inline comments, and
   submitted reviews. Note every claim the body makes (verified, calibrated,
   network-denied, exercised end to end) and every principle it states.
2. Confirm the checked-out commit and the pull request's current head both
   match the requested head. If either differs, report the stale review and
   post nothing.
3. Fetch all issue comments with pagination. The overview to update is the
   earliest comment authored by `loom-oa-dev[bot]` with a well-formed
   `<!-- marin-holistic-review:<sha> -->` marker containing 40 hex characters.
   Ignore markers from other authors and follow-up comments. If older runs
   left duplicate overviews, read them too; the most recently updated
   overview supplies the previous findings and last reviewed sha. A rebase
   or force-push does not change which comment to edit. When the last
   reviewed sha still names a commit on the PR branch, diff it against the
   head. Stop without posting or editing when that delta is under 100 changed
   lines, touches only fixtures, generated files, or documentation, or
   consists only of commits whose subject starts with `Deslop`. If the sha
   is unavailable or no longer on the branch, do a full review and replace
   the same overview. With no overview, perform a first review. On re-review,
   carry earlier finding numbers forward, mark each resolved, open, or
   changed, and assign new numbers above the largest previously used number.
4. Read root `AGENTS.md`, every `AGENTS.md` that scopes a changed path, and
   `TESTING.md` when tests change.
5. Read the complete diff, then the unchanged code it depends on: base classes
   of every new dataclass or config instance, the callers of every changed
   signature, the module each new helper could have reused, and every test
   marked `docker`, `slow`, `integration`, `cluster`, or `manual` that the
   diff adds or edits.
6. Learn the system as it is before the change, for the walkthrough. Read
   the base version of each important changed file
   (`git show <merge-base>:<path>`), the callers of each changed interface
   (`git grep -n <symbol> <merge-base>`), and the recent history of the
   touched paths (`git log --oneline -15 <merge-base> -- <path>`). Read each
   issue and pull request that the body or the commits link.
7. List open pull requests that touch the same paths. `gh pr list --search`
   matches titles and bodies only, so list the open PRs with their files and
   filter on paths:
   `gh pr list --state open --limit 100 --json number,title,files`, then keep
   PRs whose `files[].path` intersects the changed paths.

## Answer the questions

Answer each question for the whole change. A finding must name a file and
line, the concrete consequence, and the concrete replacement. Omit anything
without a named replacement.

1. **Fit.** For each new helper, type, flag, module, or dependency, name the
   existing repository concept that could carry it. Search `lib/*/src`, the
   nearest `AGENTS.md`, and sibling modules. Cite the existing `def` or class
   by path. Respect dependency direction: `iris`/`haliax` to
   `levanter`/`zephyr` to `marin`.
2. **Better abstraction.** Name one construct that deletes surface the PR
   added and cite the lines it deletes. The recurring shapes are an enum plus
   a wrapper that re-derives what the type already decides, an always-truthy
   wrapper around an optional value, a nested conditional that a lookup table
   replaces, an alias or re-export, a helper duplicated with one branch
   changed, boolean dispatch where separate classes fit, a compatibility shim
   where the callers are all in-repo, and parameters threaded through layers
   that only forward them.
3. **Consequences outside the diff.** For every new config or dataclass
   instance, read the base defaults and ask whether a critical parameter
   (memory, timeout, user, network, version, region) is satisfied by a silent
   default whose assumption this instance violates. For every status, error,
   or result mapping, enumerate the source states and check that no failure
   is encoded as a legitimate value (zero reward, empty result, `None`, a
   default) that a caller cannot distinguish from a real outcome. For every
   new environment read, check whether planning or dry-run paths reach it.
4. **Hidden tests.** For every test under a skipped marker, decide whether it
   would pass. Read its parametrization, fixtures, and the code it calls.
   `uv run pytest --collect-only -q <path>` catches collection errors; the
   rest is reasoning. Report tests that raise before asserting. Run a changed
   test file when it needs no cluster, container, or credential; a run that
   disproves a suspected finding removes it from the packet.
5. **Claims versus evidence.** For each claim and principle noted from the
   body, cite the code, test, or artifact that proves it, or state that
   nothing does. Flag identity or version bumps that rest on unproven claims
   and code that contradicts a principle the body states.
6. **Related pull requests.** Check whether each open PR touching the same
   paths deletes, duplicates, or depends on this change. Report only a
   conflict, dependency, or overlap that requires action. Recommend folding
   when one deletes what the other adds, and name the merge order. Omit
   independent PRs and notes that either merge order works.
7. **Test value.** Apply the TESTING.md gate to every added or changed test:
   "this test fails when ___ behavior is wrong". List tests to drop with the
   reason: projection of a fixture, guard mirroring, prose assertion, private
   state, or implementation copied into the expectation. Name redundant
   matrices that one parametrized test replaces.
8. **Scope.** Separate what the title promises from what the diff does. Flag
   evidence files nothing reads, scripts inside library packages, one-shot
   machinery in reusable code, and unrelated changes riding along.

Validate every finding against the code. Do not report style preferences,
speculative failures, or pre-existing problems the PR did not touch. A finding
that a hunk-scoped reviewer or the lint catalog would report belongs to them;
report it here only when the whole-change view changes its conclusion.

## Review overview

Write a walkthrough followed by the findings. A reader should
understand the problem, the important before/after behavior, and the review
decision without opening the diff. Return this overview without `--comment`;
with it, create or replace the comment according to Posting below.

```markdown
🤖 Holistic review of <sha>.

<One or two sentences: what the diff does, the verdict, and the decision or
fix the reviewer should focus on.>

<A walkthrough sized to the change: brief prose for a self-contained change,
or the fuller system, motivation, before/after, design, and limits outline
when needed.>

**Findings**

- **1. Blocker:** `<file>:<line>` <consequence>. <replacement>.
- **2. Should fix:** `<file>:<line>` <consequence>. <replacement>.
- **3. Drop test:** `<test name>` <TESTING.md reason>.
- **4. Related PR:** #<n>: <required action and merge order>.

Earlier findings: <resolved item numbers and a short reason, on re-review>.

<!-- marin-holistic-review:<sha> -->
```

The example shows possible item types, not required slots. Omit empty
categories and the earlier-findings line on a first review. With no open
findings, omit the Findings label and write `No findings.` Retain any
resolved-item status on a re-review.

- Scale the walkthrough to what the reviewer must understand. A
  self-contained change usually needs only a few short paragraphs before
  the findings. A large change spanning subsystems, interfaces, or protocols
  can use the full System, Why, Current state, How, and Limitations outline:
  show the components, explain the trigger and prior behavior, then the
  design decisions, evidence, and limits. Choose sections for their value,
  not to fill every slot or meet a word count. Review depth stays the same.
- Lead with what the diff does, the verdict, and what needs attention; flag
  a title/body mismatch when it affects the review. Explain interfaces and
  decisions before implementation details. Link the issue, incident, or
  evidence instead of retelling its history, and avoid repeating the PR body.
- Use bold run-in labels for short sections, such as `**Before.**` or
  `**Design.**` Reserve Markdown headings for substantial sections where
  they help navigation. Omit labels that add nothing, empty categories, and
  category subheadings for only one or two findings.
- Include a component diagram when it helps explain the system. Show the
  system after the change with real code names, at most about ten boxes and
  three side by side. Unstyled boxes existed before; use `changed` and
  `added` classes for affected components. Use `flowchart TB`, `subgraph`,
  node, edge, and `classDef` lines, with quoted labels containing punctuation.
  A large architecture change may also need the old connections for
  comparison; a simple change may need no diagram.
- Use focused 3–12-line code or diff snippets when the exact code helps
  assess a decision or finding. Put the path in the first line. A large
  change may need several snippets; otherwise link to the code. Show each
  piece once and avoid snippets that repeat what the prose already explains.
- State the evidence once, distinguishing local results from claims in the
  PR body. Include only limits that affect a review or merge decision. Do
  not duplicate numbered findings or add review-process status such as
  "no review threads yet".
- Keep each finding to the consequence, evidence, and concrete action. Use
  inline labels for short finding lists. Use bullets with explicit
  finding numbers so Markdown cannot renumber gaps left by resolved items.
  Preserve numbers across re-reviews; mark changed items, keep open findings
  in place, and summarize resolved numbers once below the list. Do not repeat
  open findings in a separate carried-items section.

Number all items, including tests to drop and related PRs, so every item
carries one category in the telemetry event. A blocker makes the change
wrong or unsafe as merged: a hidden test that raises, a resource default
that fails the workload, a failure encoded as a result, a principle the body
states and the code violates, a security boundary the code crosses, or an
open PR stacked on this one that deletes what it adds. A later PR that
supersedes this one is a related-PR note with a merge order; it is never a
blocker. Everything else is should-fix.

Tag each item with its category in the telemetry event below: `fit`,
`abstraction`, `consequence`, `hidden-test`, `evidence`, `related-pr`,
`slop-test`, or `scope`.

## Telemetry

Emit one best-effort stats event before returning, including clean runs.
Never retry or surface a telemetry failure. Run from the repository root:

```bash
cat <<'EOF' | uv run infra/codehealth/log_stats.py
{
  "tool": "review-pr-holistic",
  "invocation": {
    "trigger": "local",
    "agent_cli": "claude",
    "pr_number": <PR>,
    "agent_exit_code": 0,
    "timed_out": false
  },
  "findings": [
    ["<file>", <line>, "<category>", 1.0, "<first 200 chars of the item>"]
  ]
}
EOF
```

## Posting

With `--comment`, refresh the paginated issue comments and re-read the PR
head immediately before writing. If the head no longer matches the requested
head, write nothing and report the stale review; the launcher cannot cancel
a Loom session already running. If a bot-authored overview marker already
names this head or a later commit on the branch, write nothing.
Keep the stale-head and small-delta gates from Load the context.

Maintain one overview, including on re-review:

- If none exists, create it with `gh pr comment <PR> --body-file <file>`.
- Otherwise replace the original overview by its exact issue-comment
  ID, including its walkthrough, current findings, and head marker:
  `gh api --method PATCH repos/marin-community/marin/issues/comments/<id> -F body=@<file>`.
  Do not use `--edit-last`: the bot also posts deslop replies and follow-ups.
  If the edit fails, report the failure; do not create a duplicate as a
  fallback. Leave historical duplicate comments alone and use the original
  overview for future updates.

On re-review, an optional follow-up starts with `🤖` and contains only new or
materially changed actionable findings, including follow-up lint findings.
Link to the overview and reuse its finding numbers. Do not repeat the
walkthrough, unchanged open findings, or resolved-item bookkeeping. If there
are no such findings, edit only the overview. End a follow-up with
`<!-- marin-holistic-followup:<sha> -->`, never the overview marker. Before
posting it, recheck the head and skip it if a follow-up by `loom-oa-dev[bot]`
already has that head's marker. Do not post inline comments.

## Deslop follow-through

A deslop session launched from a posted packet on an `agent-generated` pull
request may apply only these items: tests to drop, aliases and re-exports,
duplicated helpers, nested conditionals replaced by lookups, stale docstrings
and comments, and checked-in evidence nothing reads. It then runs the
`noslop` skill over the branch diff. It never applies a blocker or any item
that changes behavior; those stay with the author.

The session verifies the head repository is this repository, applies the
items, runs `./infra/pre-commit.py --changed-files --fix` and the narrow
affected tests, and pushes one commit whose subject starts with `Deslop` and
names the reviewed sha. It replies once with `🤖`, lists what it removed, and
ends the reply with `<!-- marin-deslop:<sha> -->`. If any test fails it pushes
nothing and posts the findings instead. One deslop commit per reviewed head;
a head whose last commit is itself a deslop commit is never deslopped again.
On a pull request without `agent-generated`, the session posts committable
suggestions and pushes nothing.
