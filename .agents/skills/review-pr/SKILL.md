---
name: review-pr
description: Independently review an identified Marin pull request for correctness, design fit, and consequences outside the diff. Supports correctness comments and holistic overviews.
allowed-tools: Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(gh api:*), Bash(git diff:*), Bash(git grep:*), Bash(git log:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git show:*), Bash(rg:*), Bash(uv run pytest:*), Bash(uv run infra/codehealth/log_stats.py:*), mcp__github_inline_comment__create_inline_comment
---

# Review a pull request

Review the requested pull request at the exact head named by the caller. The
review is read-only except for comments posted when `--comment` is requested.

## Review standard

A reviewable Marin pull request:

- Implements the behavior claimed by its title and description without an
  introduced correctness, security, state-transition, or error-handling defect.
- Follows the root and path-scoped `AGENTS.md` or `CLAUDE.md` instructions. If
  tests change, it also follows root `TESTING.md` and any scoped testing guide.
- Keeps public interfaces, invariants, dependency direction, configuration,
  and types coherent across every changed call site. Marin does not preserve
  backward compatibility unless the request explicitly requires it.
- Uses tests for behavior rather than implementation details, incidental prose,
  private state, tautologies, or mocks below an I/O boundary.
- Keeps abstraction proportional to demonstrated reuse. Flag a maintainability
  problem only when the change creates a concrete structural obstacle, hidden
  assumption, or distant coupling that materially constrains nearby work.
- Keeps documentation synchronized with behavior. The pull-request title and
  body must follow `docs/dev-guide/writing-style/pull-requests.md`; the body is
  the squash-merge commit message, not a template, file inventory, or test log.

Report only actionable defects introduced by the pull request. Validate every
finding against the diff, relevant surrounding code, and applicable
instructions. Do not report style preferences, speculative failures, harmless
duplication in `experiments/grug`, or pre-existing problems. If the evidence is
uncertain, omit the finding.

## Choose the review mode

For a normal review, assess correctness and the whole design together. Scale
the surrounding-code investigation to the size and risk of the change.
Return one set of actionable findings; do not manufacture a second pass.
For a substantial design change, use the holistic reference's review questions
to examine callers, defaults, and consequences outside the diff.

When automation or the user requests a particular reporting contract, load
only its reference:

- [Correctness review](references/correctness.md): certain defects, inline
  comments, and the correctness-review idempotency marker.
- [Holistic review](references/holistic.md): whole-change walkthrough,
  cross-boundary design findings, and a maintained overview with stable
  finding numbers. Its re-review gates and mechanical cleanup handoff apply
  only to this mode.

With `--comment`, choose the requested contract; default to correctness
comments when no contract is named. Without `--comment`, return findings in
the conversation and emit the selected mode's telemetry event; a normal review
uses the correctness event.
Keep the mode-specific telemetry names and comment markers: automation and
historical dashboards use those protocol identifiers.

Reviewing never authorizes code edits. Use `simplify-code` only when the task
explicitly asks for cleanup; `manage-pr` owns author-side fixes and publication.
