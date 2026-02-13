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
2. **Approach** — The concrete plan. Which files change, what gets added/removed,
   key data structures or interfaces introduced.
3. **Files touched** — Explicit list of files created, modified, or deleted.
   Every file in the diff must be inferable from this list.
4. **Key code** — Short snippets (10-30 lines) for non-obvious logic: new
   protocols, algorithms, data schemas. Not boilerplate.
5. **Tests** — What is tested and how. Name the test files and describe the
   scenarios.

**Optional elements:**

- Non-goals — Scope boundaries.
- Future work — What is deliberately deferred.
- Alternatives considered — Why this approach over others.

**Target length:** 100-300 lines. Shorter is better if it's unambiguous.

Follow the format in [design_doc.md](design_doc.md) for longer specifications
that live as standalone documents.

### Example (abbreviated)

```markdown
## Problem

`ExecutorStepInfo` (executor.py:324) has no timing fields. Developers cannot
identify pipeline bottlenecks without manual instrumentation.

## Approach

Extend `StatusEvent` with `execution_start_time`/`execution_end_time` fields.
Wrap step functions in `_launch_step()` to record timestamps. Compute durations
in `get_infos()` from existing status events.

## Files touched

- `lib/marin/src/marin/execution/executor_step_status.py` — Add timing fields to StatusEvent
- `lib/marin/src/marin/execution/executor.py` — Add `_create_timed_wrapper()`, update `_launch_step()` and `get_infos()`
- `tests/test_executor_timing.py` — New: timing capture, persistence, failure cases

## Key code

```python
def _create_timed_wrapper(self, original_fn, output_path):
    def timed_wrapper(config):
        start = time.time()
        result = original_fn(config)
        append_status(status_path, STATUS_SUCCESS,
                     execution_start_time=start, execution_end_time=time.time())
        return result
    return timed_wrapper
```

## Tests

- `test_step_timing` — Verify start/end captured for successful steps
- `test_timing_on_failure` — Verify timing captured when step raises
- `test_timing_in_executor_json` — Verify timing appears in `.executor_info` output
```

## Review Process

### Human review: specification-first

Human reviewers focus on the **specification**, not the diff:

1. **Is the problem statement accurate?** Does this need solving?
2. **Is the approach sound?** Will it work? Are there better alternatives?
3. **Is the scope right?** Too much? Too little? Unintended side effects?
4. **Are the files listed complete?** Any surprising additions or omissions?

If the spec is approved, implementation review is lighter — verify the code
matches the spec and flag only correctness issues.

If the spec is wrong, reject early. Don't line-by-line review code that
implements the wrong thing.

### Agent review: specification compliance

Automated/agent review validates that the implementation matches the
specification:

1. **File coverage** — Every file in the diff should be traceable to the spec's
   "Files touched" list. Flag unexpected files.
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
| "Refactor the module" | Name the specific functions/classes, describe the target structure |
| "Add error handling" | Which errors, where, what recovery behavior |
| "Update tests" | Which test file, what scenarios, expected behavior |
| Spec says 4 files, diff has 12 | Spec missed downstream call-site updates — list them |
| Code uses approach X, spec says Y | Either update spec or re-implement |

## Case Studies

These existing design docs demonstrate the spec format at different scales:

- [`docs/design/rl-unified-interface.md`](../design/rl-unified-interface.md) —
  Large-scale interface redesign. Defines protocols, dataclasses, and usage
  examples. Multi-phase implementation plan.
- [`docs/recipes/design_doc.md`](design_doc.md) — The design doc template
  itself, with an embedded example (step execution timing).
- [`.agents/projects/grugformer.md`](../../.agents/projects/grugformer.md) —
  Project-level working agreement defining principles and conventions for an
  ongoing body of work.

## See Also

- [design_doc.md](design_doc.md) — Template for standalone design documents
- [github-pr-review.md](github-pr-review.md) — PR review prompt (includes spec validation)
- `AGENTS.md` — Coding guidelines
