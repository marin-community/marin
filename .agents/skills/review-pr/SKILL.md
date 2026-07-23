---
name: review-pr
description: Multi-agent correctness review of a pull request.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(uv run infra/codehealth/log_stats.py:*), mcp__github_inline_comment__create_inline_comment
---

Provide a code review for the given pull request.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional. Do not test tools or make exploratory calls.
- Only call a tool if it is required to complete the task.

Follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The PR is closed
   - The PR is a draft
   - The PR does not need code review (e.g. automated PR, trivial obviously-correct change)
   - Claude has already commented on this PR (check `gh pr view <PR> --comments`) AND a re-review was not explicitly requested. When a maintainer explicitly requests a re-review, always proceed even if a prior review exists.

   If any condition is true, stop. Note: still review Claude-generated PRs.

2. Launch a haiku agent to return file paths (not contents) for all relevant CLAUDE.md and AGENTS.md files:
   - The root CLAUDE.md and AGENTS.md files, if they exist
   - Any CLAUDE.md or AGENTS.md files in directories (and parent directories) containing files modified by the PR

3. Launch an opus agent to view the PR and return a summary of the changes. The
   same agent also checks the PR title and description against
   `.agents/skills/writing-style/pull-requests.md` and returns any problems it
   finds:

   - a title over 72 characters, a non-imperative title, or a conventional-commit
     prefix such as `feat:` or `fix:`;
   - a body whose length comes from diff narration, repetition, or background a
     future reader does not need;
   - a "Testing" / "Validation" / "Test plan" section, or "how I verified it" narration;
   - a templated What/Change/Scope/Testing heading scaffold, or empty boilerplate
     headings (a `## Summary` that restates the title, a `## Changes` that just
     lists the touched files) — markdown is fine when it makes the change clearer,
     the problem is structure that carries no information a reviewer needs;
   - checkboxes, emoji, agent/provider attribution, session URLs, or a filler
     opener ("This PR…", "Summary of changes:");
   - a file, symbol, or test inventory that repeats information visible in the
     diff;
   - verdict or advocacy language that substitutes emphasis for evidence, such
     as `why this is correct`, `cleaner`, `provably`, or all-caps claims;
   - a body that buries what-the-change-does under boilerplate instead of leading
     with it.

   A terse, plain body for a small change is correct — do not flag mere brevity or
   the absence of markdown. Flag descriptions that read like a filled-in form or
   implementation report rather than a commit message.

4. Launch 4 agents in parallel to independently review the changes. Each returns a list of issues; each issue includes a description and the reason it was flagged (e.g. "CLAUDE.md adherence", "bug").

   Agents 1 + 2: CLAUDE.md/AGENTS.md compliance opus agents. Audit changes for compliance. When evaluating a file, only consider CLAUDE.md/AGENTS.md files that share its path or are parents. If the PR adds or changes tests, read root `TESTING.md` plus the relevant module-specific testing docs, and check for low-value/slop tests or local testing-policy violations.

   Agents 3 + 4: Opus bug agents (parallel). Scan for obvious bugs, security issues, and incorrect logic within the changed code. Focus only on the diff without reading extra context. Flag only significant bugs you can validate from the diff alone; ignore nitpicks and likely false positives.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
   - The code will definitely produce wrong results regardless of inputs (clear logic errors)
   - Clear, unambiguous CLAUDE.md or AGENTS.md violations where you can quote the exact rule being broken

   Do NOT flag:
   - Code style or quality concerns
   - Potential issues that depend on specific inputs or state
   - Subjective suggestions or improvements

   If you are not certain an issue is real, do not flag it. False positives erode trust.

   Tell each subagent the PR title and description for author-intent context.

   **Marin-specific:** In `experiments/grug`, duplication is often intentional for high-velocity research iteration. Do not flag copy/paste or DRY concerns if behavior/contracts are correct.

5. For each issue from agents 3 and 4, launch a parallel subagent to validate it. Give the subagent the PR title, description, and issue description. It must confirm with high confidence that the issue is real — e.g. for "variable is not defined", verify that in the code; for a CLAUDE.md issue, verify the rule is scoped to this file and actually violated. Use Opus subagents throughout — for both bugs/logic and CLAUDE.md violations.

6. Filter out any issues not validated in step 5. The remainder is the high-signal review list.

7. Emit a stats event for this review (best-effort — never retry, never
   surface failures to the user). This step runs unconditionally, *before*
   any of the early-stop branches below, so we capture no-finding runs and
   non-`--comment` runs in the dashboard. Run from the repo root:

   ```bash
   cat <<'EOF' | uv run infra/codehealth/log_stats.py
   {
     "tool": "review-pr",
     "invocation": {
       "trigger": "local",
       "agent_cli": "claude",
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

   - `<category>` is one of: `bug`, `claude-md-adherence`.
   - One `findings` row per validated issue. Pass `"findings": []` if there
     were none — the empty row in the `invocations` table is the
     "tool ran with no signal" datapoint we want. `finding_count` is derived
     from the `findings` array length by `log_stats.py`.

8. Output a summary of the review findings to the terminal:
   - If issues were found, list each issue with a brief description.
   - If no issues were found, state: "No issues found. Checked for bugs and CLAUDE.md compliance."
   - Separately, report any PR-description problems from step 3.

   If `--comment` argument was NOT provided, stop here. Do not post any GitHub comments.

   If `--comment` IS provided and step 3 found PR-description problems, post **one**
   top-level comment with `gh pr comment` (prefixed `🤖`, not inline) naming the
   specific problems and the concrete fix (e.g. "drop the Testing section; lead
   with what changed and why"). This is independent of the code review — post it
   whether or not code issues were found, but skip it when the description is fine.

   If `--comment` argument IS provided and NO code issues were found, post the
   no-issues summary comment using `gh pr comment` and stop.

   If `--comment` argument IS provided and code issues were found, continue to step 9.

9. Draft the list of comments you plan to leave. For your own review only — do not post it anywhere.

10. Post inline comments for each issue using `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`. For each comment:
    - Provide a brief description of the issue
    - For small, self-contained fixes, include a committable suggestion block
    - For larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block
    - Never post a committable suggestion UNLESS committing the suggestion fixes the issue entirely. If follow up steps are required, do not leave a committable suggestion.

    **IMPORTANT: Only post ONE comment per unique issue. Do not post duplicate comments.**

Use this list when evaluating issues in Steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch (do not run the linter to verify)
- General code quality concerns (e.g., lack of test coverage, general security issues) unless explicitly required in CLAUDE.md
- Issues mentioned in CLAUDE.md but explicitly silenced in the code (e.g., via a lint ignore comment)

Notes:

- Use gh CLI to interact with GitHub (e.g., fetch pull requests, create comments). Do not use web fetch.
- Create a todo list before starting.
- You must cite and link each issue in inline comments (e.g., if referring to a CLAUDE.md, include a link to it ideally with line number).
- For changed tests, use root `TESTING.md` and the relevant module-specific `AGENTS.md`/`TESTING.md`/testing docs as the review checklist. Flag only concrete violations; do not use them to request broad coverage improvements.
- If no issues are found and `--comment` argument is provided, post a comment with the following format:

---

## Code review

No issues found. Checked for bugs and CLAUDE.md compliance.

---

- When linking to code in inline comments, follow the following format precisely, otherwise the Markdown preview won't render correctly: https://github.com/marin-community/marin/blob/c21d3c10bc8e898b7ac1a2d745bdc9bc4e423afe/package.json#L10-L15
  - Requires full git sha
  - You must provide the full sha. Commands like `https://github.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` will not work, since your comment will be directly rendered in Markdown.
  - Repo name must match the repo you're code reviewing
  - # sign after the file name
  - Line range format is L[start]-L[end]
  - Provide at least 1 line of context before and after, centered on the line you are commenting about (eg. if you are commenting about lines 5-6, you should link to `L4-7`)
