---
name: review-pr
description: Review an explicitly identified Marin pull request against the repository's correctness and maintainability expectations.
allowed-tools: Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh api:*), Bash(git diff:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git show:*), Bash(rg:*), Bash(uv run infra/codehealth/log_stats.py:*), mcp__github_inline_comment__create_inline_comment
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
  body must follow `.agents/skills/writing-style/pull-requests.md`; the body is
  the squash-merge commit message, not a template, file inventory, or test log.

Report only actionable defects introduced by the pull request. Validate every
finding against the diff, relevant surrounding code, and applicable
instructions. Do not report style preferences, speculative failures, harmless
duplication in `experiments/grug`, or pre-existing problems. If the evidence is
uncertain, omit the finding.

## Workflow

1. Inspect the pull request metadata, issue comments, inline comments, and
   current head. Stop without commenting when the pull request is closed, a
   draft, an automated dependency update, or too trivial to benefit from code
   review. Still review agent-generated pull requests.

2. Check both issue and inline comments for
   `<!-- marin-correctness-review -->`. Stop when a prior automated correctness
   review is present unless a maintainer explicitly requested another review.

3. Confirm that the checked-out commit and the pull request's current head both
   match the requested head. If either differs, report the stale review and post
   nothing.

4. Read the root instructions and every instruction file that scopes a changed
   path. Read the root and scoped testing guides when tests change. Inspect the
   complete diff and enough surrounding code to validate behavior and call-site
   consistency.

5. Review the code against the standard above. For each finding, record its
   changed file and line, category (`bug`, `instruction-following`, or
   `maintainability`), concrete impact, and the evidence that makes it certain.
   Separately record title or description problems.

6. Emit one best-effort stats event before returning, including clean and
   non-commenting reviews. Never retry or surface a telemetry failure. Run from
   the repository root:

   ```bash
   cat <<'EOF' | uv run infra/codehealth/log_stats.py
   {
     "tool": "review-pr",
     "invocation": {
       "trigger": "local",
       "agent_cli": "codex",
       "pr_number": <PR>,
       "agent_exit_code": 0,
       "timed_out": false
     },
     "findings": [
       ["<file>", <line>, "<category>", 1.0, "<first 200 chars of issue description>"]
     ]
   }
   EOF
   ```

   Use an empty `findings` array when the code review is clean.

7. Print the findings or `No issues found.` and list pull-request-description
   problems separately. Without `--comment`, stop here.

8. With `--comment`, post one top-level comment for all title or description
   problems. Begin with `🤖`, name each concrete problem and fix, and end with
   `<!-- marin-correctness-review -->`.

9. If the code review is clean and step 8 did not post a description comment,
   post exactly this top-level comment. Otherwise stop after the description
   comment.

   ```markdown
   🤖 Code review

   No issues found.

   <!-- marin-correctness-review -->
   ```

10. Otherwise, post one inline comment per unique code finding. Begin each body
    with `🤖`, explain the defect and its impact, and end with
    `<!-- marin-correctness-review -->`. Include a committable suggestion only
    when that suggestion completely fixes a small, self-contained issue. Use a
    full-SHA GitHub link with surrounding context when citing repository
    instructions or related code.

Use `gh` for GitHub reads and top-level comments. Post inline comments with the
GitHub inline-comment tool and `confirmed: true`.
