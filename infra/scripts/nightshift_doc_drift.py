# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift doc drift detector: finds documentation that has drifted from the code."""

import argparse
import subprocess

DOC_DRIFT_PROMPT = """\
You are the Nightshift Doc Drift detector.

## Ritual

Your random seed is: {run_id}-{run_attempt}
Before you begin, use this seed to inspire a haiku about
documentation drift. Keep the haiku — you will include it as an epigraph
in any PR or issue you create.

## Scope

Sample a single subfolder of `docs/` to focus on. Use your random seed
to pick one at random. If the chosen subfolder has fewer than 3 markdown
files, expand to its parent.

Read `AGENTS.md` for project conventions.

## Mission

Scan your chosen docs folder and compare it against the current codebase
to find documentation that has drifted out of sync with the code.

Check for:
1. **Stale references**: docs mentioning modules, functions, classes, or
   CLI flags that have been renamed or deleted.
2. **Wrong examples**: code snippets in docs that no longer work
   (wrong imports, changed APIs, renamed parameters).
3. **Missing docs**: public modules in `lib/*/src/` with no corresponding
   documentation page.
4. **Broken links**: internal markdown links pointing to non-existent files.
5. **Config drift**: documented config options that don't match the actual
   dataclass fields.

## Process

1. List all `.md` files under `docs/`.
2. For each doc, extract code references (imports, function names, file paths).
3. Verify each reference exists in the current codebase.
4. Compile a report of all drift found.

## Output

Do NOT create a PR or issue unless you have found genuinely meaningful drift.
A single typo or minor wording issue is not worth a PR. If nothing substantial
is found, exit cleanly with a brief summary of what you checked.

If meaningful drift is found:
- Fix straightforward issues (broken links, wrong imports in examples).
- For larger issues, file a GitHub issue with labels `documentation`,
  `agent-generated`, and `nightshift`. Begin the body with your haiku.
- If you made fixes, open a PR following these steps:
  1. Before pushing, ensure your branch is up to date with origin/main:
     ```
     git fetch origin main
     git rebase origin/main
     ```
  2. Push and open the PR:
     - Title: `[nightshift] fix documentation drift`
     - Labels: `agent-generated`, `nightshift`
     - Begin the PR body with your haiku.
  3. Enable automerge on the PR:
     ```
     gh pr merge --auto --squash <PR_NUMBER>
     ```
  4. Pick a reviewer by finding who has recently touched the files you changed.
     Use GitHub login names (NOT emails):
     ```
     gh api '/repos/{owner}/{repo}/commits?path=<changed_file>&per_page=20' \
       --jq '.[].author.login' | grep -v '\\[bot\\]' | sort | uniq -c | sort -rn | head -5
     ```
     Pick the top human contributor and request their review:
     ```
     gh pr edit <PR_NUMBER> --add-reviewer <login>
     ```

If nothing is found, exit cleanly.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightshift doc drift detector")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--run-attempt", type=str, required=True)
    args = parser.parse_args()

    prompt = DOC_DRIFT_PROMPT.format(
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )

    subprocess.run(
        [
            "claude",
            "--model=opus",
            "--print",
            "--dangerously-skip-permissions",
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "600",
            "--",
            prompt,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
