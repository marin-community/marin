---
name: review-pr-holistic
description: Review an explicitly identified Marin pull request as a whole change, asking whether it fits repository patterns, whether a better abstraction exists, and what breaks outside the diff; use for large or agent-authored PRs, after or alongside review-pr.
allowed-tools: Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(gh api:*), Bash(git diff:*), Bash(git log:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git show:*), Bash(rg:*), Bash(uv run pytest:*), Bash(uv run infra/codehealth/log_stats.py:*)
---

# Review a pull request as a whole

`review-pr` reports certain, line-anchored defects. This skill answers the
questions a senior reviewer asks after reading the entire change: does it fit
the repository, is there a smaller shape, and what does it break outside the
diff. It reads beyond the diff and reports design findings. It is read-only
except for the one comment posted when `--comment` is requested.

Run it at the exact head named by the caller. Review agent-generated pull
requests; they are the population where structural findings concentrate.

## Load the context

1. Read the pull-request title, body, issue comments, inline comments, and
   submitted reviews. Note every claim the body makes (verified, calibrated,
   network-denied, exercised end to end) and every principle it states.
2. Confirm the checked-out commit and the pull request's current head both
   match the requested head. If either differs, report the stale review and
   post nothing.
3. Find the newest `<!-- marin-holistic-review:<sha> -->` marker in issue
   comments authored by `loom-oa-dev[bot]`; ignore markers from any other
   author, and ignore a marker whose sha is not 40 hex characters naming a
   commit on the PR branch. When one exists, diff that sha against the head. Stop without
   comment when the delta is under 100 changed lines, touches only fixtures,
   generated files, or documentation, or consists only of commits whose
   subject starts with `Deslop`. Otherwise this is a re-review: carry the
   earlier numbered items forward and report each as resolved, open, or
   changed before any new finding.
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
6. **Related pull requests.** For each open PR touching the same paths, state
   whether it deletes, duplicates, or depends on this change. Recommend
   folding when one deletes what the other adds, and name the merge order.
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

## Review packet

Write the review as one packet: a walkthrough of the change, then the
findings. A reader who did not see the work must understand the problem, the
system before the change, the change, and its limits from the packet alone,
without the diff. The packet is the terminal output without `--comment` and
the posted comment body with it.

````markdown
🤖 Holistic review of <sha>.

> **In short:** <Two or three sentences: what the diff does, the verdict, and
> the one thing the reviewer must decide or examine.>

### Why
<The problem or goal, and its trigger: an issue, an incident, a parent
project, or a failed run. Quote the evidence.>

### Current state
```mermaid
flowchart TB
  subgraph <system>["<System name>"]
    a["<component>"]:::changed --> b["<component>"]
  end
  classDef changed fill:#fde7c8,stroke:#e8710a,stroke-width:2px
  classDef added fill:#d7f0dd,stroke:#1e8e3e,stroke-width:2px
```
<What the diagram cannot show: entry points, data flow, callers, and the
prerequisite work the PR builds on. End with what is wrong or missing.>

### How
**<A bold sentence that states one decision.>** <The snippet that shows it and
the reason for it.>

<Mechanical changes, one line each. The evidence that the change works.>

### Limitations
- **<Label>:** <one unsupported case, risk, or gap in the evidence>

### Review

**Blockers**
1. `<file>:<line>` <consequence>. <replacement>.

**Should fix**
2. ...

**Tests to drop**
3. `<test name>`: <TESTING.md reason>

**Related PRs**
4. #<n>: <relationship and merge order>

**Carried from <previous sha>** (re-reviews only)
- <item number>: resolved | open | changed, <one line>

<!-- marin-holistic-review:<sha> -->
````

Write the walkthrough sections with these rules:

- **Budget.** "In short" at most 60 words, Why at most 80, Current state at
  most 150, How at most 250, and Limitations at most five items of 30 words
  each. Code and the diagram do not count. Give each section space in
  proportion to what a reviewer must understand, not to the size of the diff.
- **In short.** Tell what the diff does, not what the title says, and name
  the difference when they disagree. When item 1 recommends a split, start
  with it.
- **System diagram.** Every packet has one component diagram, at the start
  of Current state. Draw each system as a subgraph, each component that the
  PR touches as a box with its real code name, and each call or data flow as
  an edge with a short label. Mark changed components with `changed` and new
  ones with `added`. Use at most about 10 boxes and at most three side by
  side. Use only `flowchart TB`, `subgraph`, node, edge, and `classDef` lines,
  and quote labels that contain punctuation, so that GitHub renders it. Add a
  second diagram in How only when the PR changes how the components connect.
- **Snippets.** Each snippet shows a decision, a behavior to check, or the
  problem. Use 3 to 12 lines, with the path as a comment on the first line.
  Show each piece of code once: use one `diff` snippet for a before/after
  change. Do not show imports, configuration lists, or code that the prose
  already describes.
- **How.** Put interfaces and decisions first, the implementation second, and
  tests and docs last. A reader who reads only the bold sentences must get the
  full design. When a decision changed after a review comment, name the
  reviewer. Do not repeat a claim from the body as a fact without evidence;
  write "the PR states".
- **Limitations.** Report open review concerns that are still valid at the
  head with the reviewer and `path:line`, also from resolved threads. Do not
  repeat a numbered finding.

On a re-review, write the full packet again at the new head.

Number items continuously across sections, including tests to drop and
related PRs, so every item carries one category in the telemetry event. A
blocker is a finding that makes the change wrong or unsafe as merged: a
hidden test that raises, a resource default that fails the workload, a
failure encoded as a result, a principle the body states and the code
violates, a security boundary the code crosses, or an open PR stacked on this
one that deletes what it adds. A later PR that supersedes this one is a
related-PR note with a merge order; it is never a blocker. Everything else is
should-fix. When no question yields a finding, the Review section is
`No findings.`; the walkthrough and the marker stay.

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

With `--comment`, re-read the PR head immediately before posting. If it no
longer matches the requested head, post nothing and report the stale review;
the launcher cannot cancel a Loom session already running, and a stale packet
would trigger deslop against a newer head. Otherwise post the packet as one
top-level comment with `gh pr comment`. Do not post inline comments; the
packet is read as a unit. Post nothing when step 2 reported a stale head or
step 3 reported a small delta.

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
