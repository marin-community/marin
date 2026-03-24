---
name: github-pr-review
description: Review a pull request for correctness and behavioral regressions. Use when asked to review a PR in marin-community/marin.
---

# Skill: GitHub PR Review

Review this PR for correctness and behavioral regressions.

Be concise and high-signal. No emoji. No fluff.

## Review Format

1. A 1-2 line description of what the PR does and whether it fulfills its stated objectives.
2. **Specification check** (if a specification exists as a PR comment, linked issue, design doc, or `.agents/projects/` file):
   - Every change in the diff should be traceable to the spec.
   - The implementation should follow the described approach. Flag significant deviations.
   - Described test scenarios should exist and test what they claim. Tests should
     exercise meaningful behavior, not merely restate production logic.
   - If no specification exists and the PR exceeds ~500 lines of code changes,
     note that a specification is expected (see `.agents/skills/pull-request/`).
3. Tight bullets (if any) for:
   - Bugs, logic errors, resource leaks, or race conditions
   - Violations of `AGENTS.md` or coding guidelines
   - Functionality that diverges from the PR description or linked issue

## Scope

- Only report high-confidence findings that cause bugs, correctness issues,
  guideline violations, or spec divergence.
- If a specification exists (issue description, design doc, acceptance criteria,
  or inline requirements), verify the code adheres to it and flag concrete
  mismatches.
- Ignore formatting, import order, lint/style preferences, naming opinions,
  missing docstrings/comments, and generic best-practice advice.

## Human vs Agent Review

Human reviewers focus on the specification: is the problem accurate, is the
approach sound, is the scope right?

Agent review (this skill) validates the implementation matches the spec and is
correct. These are complementary — do not duplicate human-level concerns.

## Grug Variants

In `experiments/grug`, duplication is often intentional for high-velocity
research iteration. Do not flag copy/paste or DRY concerns if
behavior/contracts are correct. Only call out duplication when it causes a
concrete correctness issue, regression risk, or divergence from stated
objectives.

## Output Contract

- Return exactly one final review.
- Keep output compact and high-signal.
- Do not include progress narration, process notes, or extra sections.
