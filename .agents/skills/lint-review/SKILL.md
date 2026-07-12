---
name: lint-review
description: In CI, run the infra/lint catalog review over a PR.
allowed-tools: Bash(./infra/pre-commit.py:*), Bash(gh pr comment:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh api:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git status:*), mcp__github_inline_comment__create_inline_comment
---

# Skill: Lint-catalog review on a PR

Run the `infra/lint/` catalog review (`./infra/pre-commit.py --review`)
over a pull request's branch diff and surface every finding — as `file:line`
inline review comments where the finding's line is available, and as a
single fallback comment for the rest.

## Your contract

You are running the review and reporting its output. You are **read-only except
for posting comments**: never edit, stage, commit, push, or "fix" anything, and
never run a state-changing `git`/`gh` command. The review's own lane agents are
already locked read-only.

Report the findings **faithfully**. The `--review` run (its lanes + composer) is
the authority on what is a finding: post each surviving finding **verbatim** —
one comment per finding. Do **not** drop, merge, reword the substance of, soften,
re-judge, or invent findings. Silently losing a real finding is the one
unforgivable error; so is fabricating one.

## Steps

1. **Idempotency guard (only with `--comment`).** Check whether this skill has
   already posted on the PR: look for the marker `<!-- marin-lint-review -->` in
   both issue comments (`gh pr view <PR> --json comments`) and inline review
   comments (`gh api repos/{owner}/{repo}/pulls/<PR>/comments --paginate`). If the
   marker is present, stop now — the PR already has a lint pass and we do not want
   duplicate comments. Otherwise continue.

2. **Run the review.** From the repo root:

   ```bash
   ./infra/pre-commit.py --review
   ```

   The command writes its raw per-arm prompts/outputs and the
   combined findings under `/tmp/marin-linter/<branch>/<timestamp>-<uniq>/` (path printed at the
   end); read it if a run looks wrong.

3. **Collect the findings.** Each finding the command emits on stdout is one line
   in the canonical catalog format:

   ```
   <path>:<line>: ml-<code> (<confidence>) <message>
   ```

   e.g. `lib/iris/src/iris/foo.py:42: ml-cruft-dead-branch (0.85) Unreachable else after early return`.

4. **No findings vs. failed run.** Distinguish two zero-finding cases:

   - **Clean** — the command exited 0 and printed `Lint review: no findings.` (or
     emitted no finding lines). Post **nothing**: the green job check is the "lint
     pass clean" signal, and a second "all clear" comment would only duplicate the
     high-level review. State "Lint review: no findings." to the terminal and stop.
   - **Failed to run** — the command exited non-zero or printed that every lane
     failed / the agent was not found / the merge-base could not be resolved. Do
     **not** report this as clean. State plainly in your final output that the lint
     review could not run and why; post no comments. (A broken run is a job-log
     signal, not a PR comment.)

   (Without `--comment`: just print the findings, if any, to the terminal and stop
   here regardless.)

5. **Post inline comments.** With `--comment` and findings present, for **each**
   finding post one inline comment with
   `mcp__github_inline_comment__create_inline_comment`, the
   finding's `path` and `line`, and a body of exactly this shape:

   ```
   `ml-<code>` · confidence <confidence>

   <message>

   <!-- marin-lint-review -->
   ```

   The `<message>` is copied verbatim from the finding. Post one comment per
   finding; never post two comments for the same finding.

6. **Handle un-anchorable findings.** The inline-comment tool rejects a line that
   is not part of the PR diff (it raises a validation error). A finding can land
   on such a line — e.g. the holistic `meta` lane anchors on context outside the
   added hunks. When a post fails for that reason, **do not abort**: record that
   finding and keep going through the rest.

7. **Fallback summary.** After attempting every inline comment, if any findings
   could not be placed inline, post **one** issue comment with `gh pr comment <PR>`
   so none are dropped. Format:

   ```
   Lint review:

   These infra/lint findings anchor on lines not in the PR diff, so they could not
   be attached inline.

   - <path>:<line>: ml-<code> (<confidence>) <message>
   - ...

   <!-- marin-lint-review -->
   ```

   List every un-anchorable finding verbatim. If every finding was placed inline,
   do not post this comment.

## Notes

- Use the `gh` CLI for GitHub; do not web-fetch.
- The review reads the **branch diff against the merge base with `origin/main`**,
  covering committed and uncommitted work. CI checks out the PR head and fetches
  `origin/main` before invoking you, so the merge base resolves.
