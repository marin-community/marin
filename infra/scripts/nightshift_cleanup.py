# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift cleanup agent: picks target folders and runs code cleanup."""

import argparse
import json
import os
import random
import secrets
import subprocess
from pathlib import Path

BASE_PATHS = [
    "lib/marin/src/marin",
    "lib/levanter/src/levanter",
    "lib/iris/src/iris",
    "lib/haliax/src/haliax",
    "lib/zephyr/src/zephyr",
    "lib/fray/src/fray",
    "experiments",
    "tests",
]

MIN_PY_FILES = 3
MAX_DEPTH = 3


def find_candidate_directories() -> list[str]:
    """Find subdirectories with at least MIN_PY_FILES .py files, up to MAX_DEPTH deep."""
    candidates = []
    for base in BASE_PATHS:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for dirpath in _walk_max_depth(base_path, MAX_DEPTH):
            py_count = sum(1 for f in dirpath.iterdir() if f.is_file() and f.suffix == ".py")
            if py_count >= MIN_PY_FILES:
                candidates.append(str(dirpath))
    return candidates


def _walk_max_depth(root: Path, max_depth: int) -> list[Path]:
    """Collect all directories under root up to max_depth levels deep (inclusive of root)."""
    results = [root]
    frontier = [root]
    for _ in range(max_depth):
        next_frontier = []
        for d in frontier:
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        results.append(child)
                        next_frontier.append(child)
            except PermissionError:
                continue
        frontier = next_frontier
    return results


def plan(args: argparse.Namespace) -> None:
    """Pick target folders and output a GitHub Actions matrix."""
    agent_count = args.agent_count

    if args.target_folder:
        selected = [args.target_folder] * agent_count
    else:
        candidates = find_candidate_directories()
        if not candidates:
            raise SystemExit("No candidate directories found")
        selected = random.sample(candidates, min(agent_count, len(candidates)))

    includes = []
    for i, folder in enumerate(selected, 1):
        includes.append(
            {
                "agent_id": i,
                "folder": folder,
                "haiku_seed": secrets.token_hex(4),
            }
        )

    matrix = {"include": includes}
    matrix_json = json.dumps(matrix)

    github_output = os.environ["GITHUB_OUTPUT"]
    with open(github_output, "a") as f:
        f.write(f"matrix={matrix_json}\n")

    print("Selected folders:")
    print(json.dumps(matrix, indent=2))


CLEANUP_PROMPT = """\
You are Nightshift Cleanup Agent #{agent_id}.

Your random seed is: {haiku_seed}
Use this seed to compose a haiku that will serve as the epigraph
for your PR description. The haiku should relate to code maintenance.

## Your Mission

Scan the folder `{folder}` for code cleanup opportunities.
Read `docs/dev-guide/coding-standards.md` for the full set of rules.
Read `AGENTS.md` for project conventions.

## What to Look For

Scan for issues across three dimensions. Within each, items are in priority order.

### 1. Code Reuse

- **Duplicated utilities**: search for existing helpers in `lib/*/src/` that could
  replace newly written or existing hand-rolled code. Flag any function that
  duplicates existing functionality and suggest the existing one instead.
- **Inline logic that should use an existing utility**: hand-rolled string
  manipulation, manual path handling, custom environment checks, ad-hoc type
  guards, and similar patterns that a codebase utility already handles.

### 2. Code Quality

- **Dead code**: unused imports, unreferenced functions, commented-out blocks,
  stale `TODO`/`FIXME` (>90 days via `git blame`)
- **Redundant state**: state that duplicates existing state, cached values that
  could be derived, observers/effects that could be direct calls
- **Parameter sprawl**: functions with >5 parameters, or new parameters added
  instead of generalizing or restructuring existing ones
- **Copy-paste with slight variation**: near-duplicate code blocks that should be
  unified with a shared abstraction
- **Leaky abstractions**: exposing internal details that should be encapsulated,
  or breaking existing abstraction boundaries
- **Stringly-typed code**: using raw strings where constants, enums, or typed
  alternatives already exist in the codebase
- **LLM anti-patterns**: over-protective try/except, defensive None checks,
  verbose docstrings restating the code, unnecessary abstractions
- **Test quality**: tests with no assertions, `time.sleep()` in tests,
  mocks of internal functions, `@pytest.mark.skip` without linked issue
- **Import hygiene**: mid-function imports (except cycle-breaking), `TYPE_CHECKING`
  guards, wrong dependency direction

### 3. Efficiency

- **Unnecessary work**: redundant computations, repeated file reads, duplicate
  network/API calls, N+1 query patterns
- **Missed concurrency**: independent operations run sequentially when they could
  run in parallel
- **Hot-path bloat**: blocking work added to startup or per-request hot paths
- **Recurring no-op updates**: state updates inside polling loops or event handlers
  that fire unconditionally — add a change-detection guard so downstream consumers
  aren't notified when nothing changed
- **Unnecessary existence checks**: pre-checking file/resource existence before
  operating (TOCTOU anti-pattern) — operate directly and handle the error
- **Memory**: unbounded data structures, missing cleanup, listener leaks
- **Overly broad operations**: reading entire files when only a portion is needed,
  loading all items when filtering for one

## Rules of Engagement

- Only make changes you are confident are correct improvements.
- Do NOT refactor working code for style alone — focus on genuine issues.
- Do NOT create a PR unless you have identified genuinely high-value fixes.
  Cosmetic-only or marginal improvements do not warrant a PR. If nothing
  meaningful is found, exit cleanly instead.
- Do NOT touch files outside `{folder}` unless fixing an import.
- Run `./infra/pre-commit.py --all-files --fix` before committing.
- Run `uv run pytest -m 'not slow'` on any test files you modified or that
  test modules you changed.
- If you find issues but the fix is non-trivial, file a GitHub issue instead
  of making a risky change.
- Keep each PR focused: one concern per PR. If you find multiple independent
  issues, pick the most impactful one.

## Output

Create a branch named `nightshift/cleanup-{agent_id}-$(date +%Y%m%d)`
and open a PR with:
- Title: `[nightshift] <concise description of cleanup>`
- Body: your haiku, then a summary of what was cleaned and why
- Labels: `agent-generated`, `nightshift`

If you find nothing worth changing, do NOT create a PR. Instead, exit cleanly
with a message saying the folder is in good shape.
"""


def run(args: argparse.Namespace) -> None:
    """Run a cleanup agent on the specified folder."""
    prompt = CLEANUP_PROMPT.format(
        agent_id=args.agent_id,
        folder=args.folder,
        haiku_seed=args.haiku_seed,
    )

    subprocess.run(
        [
            "claude",
            "--model=opus",
            "--print",
            "--dangerously-skip-permissions",
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "80",
            "--",
            prompt,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightshift cleanup agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Pick target folders and output matrix")
    plan_parser.add_argument("--agent-count", type=int, default=5)
    plan_parser.add_argument("--target-folder", type=str, default=None)

    run_parser = subparsers.add_parser("run", help="Run cleanup on a folder")
    run_parser.add_argument("--agent-id", type=int, required=True)
    run_parser.add_argument("--folder", type=str, required=True)
    run_parser.add_argument("--haiku-seed", type=str, required=True)

    args = parser.parse_args()

    if args.command == "plan":
        plan(args)
    elif args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
