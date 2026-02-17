# Recipe: Agent-Authored PRs

## Overview

Agent PRs follow a **specification-driven** workflow: the specification is the
primary review artifact, the code is validated against it. This keeps human
review focused on _what should happen_ and lets agents (and automated review)
verify _that it happened correctly_.

## When a Specification Is Required

Any PR that adds or modifies **more than ~500 lines of code** must include a
specification. Smaller PRs may include one if the change is architecturally
significant or touches many modules.

The specification can live in one of two places:

| Location | When to use |
|---|---|
| GitHub PR comment (first comment after description) | Default. Keeps spec co-located with review discussion. |
| `docs/design/<topic>.md` or `.agents/projects/<topic>.md` | When the spec will be referenced by future work or multiple PRs. Link from the PR description. |

## What a Specification Contains

A specification is a prompt: given sufficient context, any competent agent or
engineer reading it would produce substantially the same set of changes.

**Required elements:**

1. **Problem** — What is broken or missing. Reference files and line numbers.
2. **Approach** — The concrete plan. Which modules change, what gets added/removed,
   key data structures or interfaces introduced.
3. **Key code** — Short snippets (10-30 lines) for non-obvious logic: new
   protocols, algorithms, data schemas. Not boilerplate.
4. **Tests** — What is tested and how. Name the test files and describe the
   scenarios. Tests must be extensive and cover the changed code in sufficient detail as to
   provide confidence that the changes are correct. Tests should be behavioral and end-to-end.
   Tests should _not_ validate facially correct behavior or duplicate existing functionality.
   Don't test that a type is a type, a constant is a value, or that 1 + 1 = 2.

**Optional elements:**

- Non-goals — Scope boundaries.
- Future work — What is deliberately deferred.
- Alternatives considered — Why this approach over others.

Follow the format in [design_doc.md](design_doc.md) for longer specifications
that live as standalone documents.

### Example (abbreviated)

```markdown
## Problem

(What's the reason for this PR? What are we trying to accomplish? Why is it important?)

`ExecutorStepInfo` (executor.py:324) has no timing fields. Developers cannot
identify pipeline bottlenecks without manual instrumentation.

## Approach

(Describe the way the problem has been solved in sufficient detail as to ensure
another competent engineer would be able to recover your code given the description.)

Extend `StatusEvent` with `execution_start_time`/`execution_end_time` fields.
Wrap step functions in `_launch_step()` to record timestamps. Compute durations
in `get_infos()` from existing status events.

## Key changes

(What changes in the code base? Provide psuedo-code samples to describe
high-level changes or important details.)

```python
class StatusEvent
  execution_start_time: Timestamp
  execution_end_time: Timestamp
...

def launch_step:
  event = StatusEvent(... start_time = TimeStamp.now())
  ...
  event.execution_end_time = TimeStamp.now()
```

## Testing

Write out the set of test cases and their intentions. Why is this the correct
set of tests, why do we believe it covers the complete change and will identify
bugs and regressions?

```

## Review Process

### Human review: specification-first

Human reviewers focus on the **specification**, not the diff:

1. **Is the problem statement accurate?** Does this need solving?
2. **Is the approach sound?** Will it work? Are there better alternatives?
3. **Is the scope right?** Too much? Too little? Unintended side effects?
4. **Are the files listed complete?** Any surprising additions or omissions?

### Agent review: specification compliance

Automated/agent review validates that the implementation matches the
specification:

1. **Coverage** — Every change in the diff should be traceable to the spec. Flag where the spec is ambiguous. Is this the only reasonable output from the spec?
2. **Approach fidelity** — The implementation should follow the described
   approach. Flag significant deviations (different data structures, different
   control flow, missing components).
3. **Test coverage** — The described test scenarios should exist and test what
   they claim.
4. **Correctness** — Bugs, logic errors, resource leaks, race conditions in the
   implementation itself.

The standard [PR review prompt](github-pr-review.md) includes specification
validation when a spec is present.

## Writing Good Specifications

A specification passes the **reproduction test**: could a different agent, given
only the spec and access to the repo, produce substantially the same PR?

Common failures:

| Problem | Fix |
|---|---|
| "Missing test case for Y" | Which test file, what scenarios, expected behavior |
| Code uses approach X, spec says Y | Either update spec or re-implement |

## Case Studies

These existing design docs demonstrate the spec format at different scales:

- `.agents/projects/grugformer.md` —
  Project-level working agreement defining principles and conventions for an
  ongoing body of work.

## See Also

- [github-pr-review.md](github-pr-review.md) — PR review prompt (includes spec validation)
- `AGENTS.md` — Coding guidelines
