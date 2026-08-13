# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift cleanup: spawns parallel sub-agents per subproject, merges results into one PR."""

import datetime
import json
import logging
import secrets
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.ci.claude_runner import (
    NO_SELF_CREDIT_SETTINGS,
    ClaudeRunStatus,
    report_rate_limit,
    run_claude,
)

logger = logging.getLogger(__name__)

SUBPROJECTS = ["lib/marin/src/marin", "lib/iris/src/iris", "lib/zephyr/src/zephyr", "lib/levanter/src/levanter"]

SCOUT_PROMPT = """\
You are a Nightshift Scout Agent assigned to: {subproject}

Your random seed is: {haiku_seed}

## Your Mission

Browse `{subproject}` for a cleanup whose value is obvious from the diff. It is
normal and preferable to report no change when there is no clear improvement.
Eligible changes fix a concrete bug or inefficiency, delete code proven unused,
or centralize non-trivial policy whose copies can drift. Look for things like:

- Dead code: unused functions, stale TODO/FIXME (>90 days via `git blame`),
  commented-out blocks
- Duplicated non-trivial logic that should use an existing helper from elsewhere
  in the repo
- Copy-pasted policy or algorithms whose copies can drift
- Overly defensive error handling (try/except that swallows exceptions,
  redundant None checks)
- Parameter sprawl, stringly-typed code, leaky abstractions
- Unnecessary work: redundant computations, repeated file reads, N+1 patterns
- Test issues: tests with no assertions, `time.sleep()` in tests, mocks of
  internal functions

Read `AGENTS.md` (especially its Testing section) and
`.agents/skills/commit/SKILL.md` for project conventions, and follow them.

## Rules of Engagement

- Only make changes with a concrete payoff that a reviewer can see directly in
  the diff: corrected behavior, less work at runtime, deleted dead code, or one
  authoritative implementation of non-trivial logic.
- Do NOT refactor working code for style alone.
- Do NOT extract a helper merely to replace short, linear, locally readable
  code, even when that code is repeated. A configurable helper that takes
  callbacks, resources, flags, or similar arguments to reconstruct a caller's
  obvious steps is not a cleanup.
- A refactor must reduce cognitive load at its call sites and centralize a real
  invariant or policy. If its value needs a long explanation, report no change.
- Do not bundle an opportunistic behavior change into a refactor. Preserve
  behavior or make the behavior fix the explicit purpose of the change.
- Run `./infra/pre-commit.py --all-files --fix` before committing.
- Run relevant tests: `uv run pytest -x` on any test files you modified or
  that test modules you changed.
- If you find issues but the fix is non-trivial, file a GitHub issue instead
  of making a risky change.
- Keep changes focused: one coherent improvement per subproject.
- Never write tautological, trivial, or "slop" tests. A test must fail when the
  behavior is wrong, not merely when the implementation changes. Do not add a
  test for a thin wrapper around a library call, for a one-off script, or just
  to have a test. If a change does not warrant a meaningful test, add none.
- Never credit yourself in commit messages: no `Co-Authored-By: Claude` or
  "Generated with Claude Code" trailer.

## Output

Do NOT create a PR or push to a remote. Work locally only.

If you made improvements, commit them to the current branch with a message like:
  `[nightshift] <subproject>: <concise description>`

Then write a JSON summary to `{result_file}`:
```json
{{
  "subproject": "{subproject}",
  "status": "changed",
  "summary": "<one-paragraph description of what was cleaned and why>",
  "value": "<the concrete bug, runtime cost, dead code, or non-trivial policy addressed>",
  "files_changed": ["<list of files changed>"]
}}
```

If you find nothing worth changing, write:
```json
{{
  "subproject": "{subproject}",
  "status": "no_change",
  "summary": "<brief note on what you checked>"
}}
```
"""

MERGE_PROMPT = """\
You are the Nightshift Merge Agent.

## Context

Parallel scout agents searched these subprojects for cleanup opportunities.
Each scout worked in its own git worktree and committed changes independently.
Their results:

{scout_results}

## Worktrees

The scout worktrees with their commits are at these paths:
{worktree_info}

## Conventions

Read `AGENTS.md` and `.agents/skills/commit/SKILL.md`, and follow them for the
PR. Never credit yourself: no `Co-Authored-By: Claude` or "Generated with Claude
Code" trailer in commits, and no self-attribution in the PR description.

## Your Mission

1. Inspect every scout diff independently. A `status: "changed"` result is not
   evidence that the change is worth landing. Reject changes that wrap short,
   obvious local code in a configurable helper, increase indirection without
   centralizing non-trivial policy, or need a long explanation to establish
   their value. Prefer fewer changes or no PR over speculative cleanup.

2. Cherry-pick only the scout commits whose concrete payoff is obvious from the
   diff into the current branch (`nightshift/cleanup-{date}`). Skip any that
   conflict or fail tests after merging.

3. Run `./infra/pre-commit.py --all-files --fix` and `uv run pytest -x` on all
   affected test files to verify the combined changes are clean.

4. Before pushing, rebase on origin/main:
   ```
   git fetch origin main
   git rebase origin/main
   ```

5. Open a PR:
   - Title: an imperative sentence of at most 72 characters that names the
     concrete outcome; do not use a date or "multi-cleanup" as the outcome
   - Body: a compact explanation of the accepted changes and their concrete
     payoff; omit generation metadata, epigraphs, diff inventories, and test
     narration
   - Labels: `agent-generated`, `nightshift`

6. Enable automerge:
   ```
   gh pr merge --auto --squash <PR_NUMBER>
   ```

7. Pick a reviewer by finding who recently touched the changed files. Use
   GitHub login names (NOT emails):
   ```
   gh api '/repos/{{owner}}/{{repo}}/commits?path=<changed_file>&per_page=20' \\
     --jq '.[].author.login' | grep -v '\\[bot\\]' | sort | uniq -c | sort -rn | head -5
   ```
   Pick the top human contributor and request their review:
   ```
   gh pr edit <PR_NUMBER> --add-reviewer <login>
   ```

If no scouts produced changes, exit cleanly — no branch, no PR.
"""


def setup_scout_worktree(subproject: str, date: str, repo_root: Path) -> Path:
    """Create a fresh worktree for a scout. Must run sequentially — `git worktree add`
    serializes on repo metadata and races when called from multiple threads."""
    worktree_name = f"nightshift-scout-{subproject.replace('/', '-')}"
    worktree_path = repo_root / ".claude" / "nightshift-worktrees" / worktree_name
    branch_name = f"nightshift/scout-{subproject.replace('/', '-')}-{date}"

    subprocess.run(["git", "worktree", "remove", "--force", str(worktree_path)], capture_output=True, cwd=repo_root)
    subprocess.run(["git", "branch", "-D", branch_name], capture_output=True, cwd=repo_root)

    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/main"],
        check=True,
        cwd=repo_root,
    )
    return worktree_path


def run_scout(subproject: str, worktree_path: Path) -> tuple[str, dict, str]:
    """Run a single scout agent in a pre-created git worktree."""
    with tempfile.NamedTemporaryFile(suffix=".json", prefix=f"nightshift-{worktree_path.name}-", delete=False) as f:
        result_file = f.name
    haiku_seed = secrets.token_hex(4)

    prompt = SCOUT_PROMPT.format(
        subproject=subproject,
        haiku_seed=haiku_seed,
        result_file=result_file,
    )

    logger.info("Starting scout for %s in %s", subproject, worktree_path)
    agent_result = run_claude(
        prompt,
        [
            "--model=opus",
            "--dangerously-skip-permissions",
            *NO_SELF_CREDIT_SETTINGS,
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "400",
        ],
        cwd=worktree_path,
    )
    if agent_result.status == ClaudeRunStatus.RATE_LIMITED:
        Path(result_file).unlink(missing_ok=True)
        report_rate_limit()
        return (
            subproject,
            {"subproject": subproject, "status": "rate_limited", "summary": "Claude rate limited"},
            str(worktree_path),
        )
    logger.info("%s", agent_result.output)

    result = {"subproject": subproject, "status": "error", "summary": "Scout did not produce a result file"}
    if Path(result_file).exists():
        try:
            result = json.loads(Path(result_file).read_text())
        except json.JSONDecodeError:
            result = {"subproject": subproject, "status": "error", "summary": "Scout produced invalid JSON"}
        Path(result_file).unlink()

    return subproject, result, str(worktree_path)


def run_merge(date: str, scout_results: list[dict], worktree_info: list[tuple[str, str]]) -> None:
    """Run the merge agent to combine scout results into a single PR."""
    results_text = "\n".join(
        f"### {r['subproject']}\n- Status: {r['status']}\n- Summary: {r.get('summary', 'N/A')}\n"
        f"- Value: {r.get('value', 'N/A')}"
        for r in scout_results
    )
    worktree_text = "\n".join(f"- `{subproject}`: `{path}`" for subproject, path in worktree_info)

    prompt = MERGE_PROMPT.format(
        scout_results=results_text,
        worktree_info=worktree_text,
        date=date,
    )

    result = run_claude(
        prompt,
        [
            "--model=opus",
            "--dangerously-skip-permissions",
            *NO_SELF_CREDIT_SETTINGS,
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "200",
        ],
    )
    if result.status == ClaudeRunStatus.RATE_LIMITED:
        report_rate_limit()
        return
    logger.info("%s", result.output)


def cleanup_worktrees(repo_root: Path) -> None:
    """Remove all nightshift scout worktrees."""
    worktrees_dir = repo_root / ".claude" / "nightshift-worktrees"
    if not worktrees_dir.exists():
        return
    for child in worktrees_dir.iterdir():
        subprocess.run(["git", "worktree", "remove", "--force", str(child)], capture_output=True, cwd=repo_root)
    # Fall back to shutil.rmtree if git worktree remove left anything behind
    if worktrees_dir.exists():
        shutil.rmtree(worktrees_dir, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    date = datetime.date.today().strftime("%Y%m%d")
    repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())

    # Fetch once — scouts reuse this ref
    subprocess.run(["git", "fetch", "origin", "main"], check=True, cwd=repo_root, capture_output=True)

    merge_branch = f"nightshift/cleanup-{date}"
    subprocess.run(["git", "branch", "-D", merge_branch], capture_output=True, cwd=repo_root)
    subprocess.run(["git", "checkout", "-b", merge_branch, "origin/main"], check=True, cwd=repo_root)

    scout_results: list[dict] = []
    worktree_info: list[tuple[str, str]] = []

    # Create worktrees sequentially, then run claude in parallel. `git worktree add`
    # serializes on repo metadata, so concurrent creation races and some invocations
    # fail with exit 255.
    subproject_worktrees = [(sp, setup_scout_worktree(sp, date, repo_root)) for sp in SUBPROJECTS]

    with ThreadPoolExecutor(max_workers=len(SUBPROJECTS)) as pool:
        futures = {pool.submit(run_scout, sp, wt): sp for sp, wt in subproject_worktrees}
        for future in as_completed(futures):
            sp = futures[future]
            try:
                subproject, result, worktree_path = future.result()
            except Exception:
                logger.exception("Scout %s failed", sp)
                scout_results.append({"subproject": sp, "status": "error", "summary": "Scout raised an exception"})
                continue
            scout_results.append(result)
            if result.get("status") == "changed":
                worktree_info.append((subproject, worktree_path))
            logger.info("Scout %s: %s", subproject, result.get("status"))

    failed = [result for result in scout_results if result.get("status") == "error"]
    if failed:
        cleanup_worktrees(repo_root)
        raise RuntimeError(f"{len(failed)} cleanup scouts failed")

    changed = [r for r in scout_results if r.get("status") == "changed"]
    if not changed:
        logger.info("No scouts found changes. Exiting cleanly.")
        cleanup_worktrees(repo_root)
        return

    try:
        run_merge(date, scout_results, worktree_info)
    finally:
        cleanup_worktrees(repo_root)


if __name__ == "__main__":
    main()
