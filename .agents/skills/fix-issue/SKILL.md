---
name: fix-issue
description: End-to-end workflow to fix a GitHub issue in marin-community/marin.
---

# Skill: Fix GitHub Issue

Fix the Github issue indicated by the user.

## Background

Read first:

@AGENTS.md

Complete tasks in order; the task list is at the end of this document. If you
cannot complete a task, write a Github comment with your last status and why.

## Writing Style for All GitHub Comments

**Be terse.** Every sentence must convey new information:

- No preamble, filler, or editorializing ("I've thoroughly investigated...").
- No repeating the issue body.
- No restating what code does when a link suffices.
- Max 3-4 sentences of prose per comment section. Use links and code, not words.
- Annotate code links, don't narrate them. Bad: "The transfer happens in
  arrow_flight.py where the implementation copies data from TPU to CPU".
  Good: `arrow_flight.py#L384` - TPU→CPU copy.

## Research

Use `gh` to fetch the issue. Read the codebase for all relevant source files.
Post a single comment with two sections: **Research** and **Proposed Fix**.

Before writing code, run a duplicate-work preflight:

- Check open PRs for the same issue or subsystem (`gh pr list --state open --search "<issue-id or keyword>"`).
- Check open issues for the same root cause (`gh issue list --state open --search "<keyword>"`).
- If overlapping work exists, do not open a parallel implementation PR. Add a
  short issue comment summarizing what you found, and either hand off to the
  existing PR or scope your change to non-overlapping follow-up work.

### Research section

Title: `# Research`

- 2-3 sentences: what's broken/missing and why (root cause, not symptoms).
- A `**Relevant code**` list: max 5 links, each with a short (<10 word) annotation.

Example:

> `jax.device_get()` produces non-contiguous memory layout, limiting TPU→CPU
> transfer to ~1GB/s (expect 4-7GB/s). No parallelization across shards.
>
> **Relevant code**
> - [`arrow_flight.py#L384`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/rl/weight_transfer/arrow_flight.py#L384) - TPU→CPU copy
> - [`jax.py#L394-L402`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/rl/weight_transfer/jax.py#L394-L402) - transfer impl
> - [`base.py#L32-L35`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/rl/weight_transfer/base.py#L32-L35) - transfer mode defs

### Proposed Fix section

Title: `# Proposed Fix`

There are 2 cases:

* **Bug fix**: Show the call chain that triggers the error, then state the fix
  in one sentence. Include a code snippet only if the fix is non-obvious.

  > `env.run()` returns a tuple; `foo()` assumes an object and calls `.abc()`.
  > Fix: unwrap the tuple in `RolloutWorker._step()` before passing to `foo()`.

* **Design change**: The design doc replaces the `# Proposed Fix` section.
  Write it per `.agents/skills/write-design-doc/` directly in the issue comment
  (agents cannot commit files before the PR). The 3-4 sentence prose cap
  applies to `# Research`; the design doc follows its own length guidelines.
  Keep everything in one comment.

In both cases: find the minimally disruptive fix. Do not over-engineer.

## Implementation

Implement changes in a branch named `agent/{YYYYMMDD}-fix-{issue-id}`. You may
create branches on github and submit PRs from them.

## Testing

* Write the test _before_ the fix, then validate the fix works.
* Use an existing test file in `tests/` if appropriate.
* Never use mocks.
* Keep tests simple and minimal. Do not test obvious behavior like "object has an attr".
* Run `./infra/pre-commit.py --all-files --fix` and resolve all reported issues.
* Run the default local test suite for the touched package and ensure it passes
  before uploading. Use `uv run pytest` for root or mixed changes; do not
  override the configured marker expression. Slow, integration, live-cluster,
  Docker, and manual tests run in dedicated CI jobs unless the task explicitly
  requires one.

## Uploading

When all tests pass, upload your branch and open a PR following
`.agents/skills/commit/SKILL.md` **exactly** — use the plain-text format it
specifies (no markdown headers, bullet lists, or `## Summary` sections;
violations are rejected). Attach a comment to the Github issue summarizing the fix.

## Verify CI Status

After opening the PR, verify CI checks pass:

* Monitor with `gh pr view <number> --json statusCheckRollup`.
* Wait for unit tests; if they fail, investigate and push fixes.
* Do not consider the PR complete until all relevant checks pass.

Key checks: **unit tests** (must pass), **lint-and-format** (must pass),
**build-docs** (should pass if you modified documentation).

# Tasks

The tasks for this skill:

- [ ] Fetch issue information
- [ ] Research codebase for all relevant source files
- [ ] Run duplicate-work preflight against open PRs/issues
- [ ] Formulate fix and post Research + Proposed Fix comment
- [ ] Create branch for the changes
- [ ] Write a test case as needed for changes
- [ ] Implement changes until all tests pass
- [ ] Run `./infra/pre-commit.py --all-files --fix` and resolve all issues
- [ ] Upload branch to github
- [ ] Open pull request
- [ ] Verify CI checks pass and address any failures (by polling gh pr view)
- [ ] Update comment with final status

If at any point you are unable to proceed, you must add a comment to the Github
issue with your last status.

# RULES

0. Never credit yourself in commits or comments.
