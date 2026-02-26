You are to fix the Github issue indicated by the user.

## Background

Read first:

@AGENTS.md
@docs/recipes/architecture.md

You may not proceed to a new task until you have completed all prior tasks. If
you cannot complete a task, write a Github comment with your last status and why
you failed. A task list is at the end of this document.

## Writing Style for All GitHub Comments

**Be terse.** Every sentence must convey information that the reader does not
already have. Follow these rules in all comments you post:

- No preamble, no filler, no editorializing ("I've thoroughly investigated...",
  "After careful analysis...", "This is an interesting problem...").
- No repeating information already in the issue body.
- No restating what code does when a link suffices.
- Max 3-4 sentences of prose per comment section. Use links and code, not words.
- Annotate code links, don't narrate them. Bad: "The transfer happens in
  arrow_flight.py where the implementation copies data from TPU to CPU".
  Good: `arrow_flight.py#L384` - TPU→CPU copy.

## Research

Use `gh` to fetch the issue. Read the codebase to find all relevant source
files. Post a single comment combining your research and proposed fix.

The comment has two sections: **Research** and **Proposed Fix**.

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

* **Design change**: Write the design doc per @docs/recipes/design_doc.md
  directly in the issue comment (agents cannot commit files to the repo before
  the PR). Keep it to one comment; do not split across multiple comments.

In both cases: find the minimally disruptive fix. Do not over-engineer.

## Implementation

Once you have researched and prepared your design or planned fix, you may
proceed to implementation. You MUST implement your changes in a branch with the
following format `agent/{YYYYMMDD}-fix-{issue-id}`. You are allowed to create
branches on github as well as submit PRs from those branches.

You will implement a test which demonstrates your changed behavior before
beginning work. Your tests will be minimal and refrain from using mocks.

## Testing

* You write your test _before_ you make your fix, and validate your fix works.
* You use an existing test file in @tests/ if appropriate.
* You never use mocks when testing.
* You keep tests simple and minimal. You do not test obvious behavior like "object has an attr".
* You will test using `uv run pytest -m 'not slow'` before uploading
* You will ensure _all_ tests pass.

## Uploading

When all tests pass, you may proceed to upload your branch.
You will then open a pull request for the branch.
You will attach it to the Github issue with a comment summarizing the fix.

## Verify CI Status

After opening the pull request, you must verify that CI checks pass:

* Monitor the PR using `gh pr view <number> --json statusCheckRollup`
* Wait for the unit tests to complete
* If tests fail, investigate and fix any issues
* Push additional commits to address CI failures
* Do not consider the PR complete until all relevant checks pass

Key checks to monitor:
- **unit tests**: Must pass - validates your changes don't break existing functionality
- **lint-and-format**: Must pass - ensures code style compliance
- **build-docs**: Should pass if you modified documentation

# Tasks

The tasks for this recipe:

- [ ] Fetch issue information
- [ ] Aggressively research codebase for relevant source files for the issue
- [ ] Update issue with "Agent Research"
- [ ] Formulate a design or fix
- [ ] Update issue with proposed plan
- [ ] Create branch for the changes
- [ ] Write a test case as needed for changes
- [ ] Implement changes until all tests pass
- [ ] Upload branch to github
- [ ] Open pull request
- [ ] Verify CI checks pass and address any failures (by polling gh pr view)
- [ ] Update comment with final status

If at any point you are unable to proceed, you must add a comment to the Github
issue with your last status.

# RULES

0. Never credit yourself in commits or comments.
