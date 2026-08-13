# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift CI test audit: inspect recent CI logs for slow or unstable tests."""

import datetime as dt
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from scripts.ci.claude_runner import ClaudeRunStatus, report_rate_limit, run_claude

logger = logging.getLogger(__name__)

WORKFLOWS = (
    ".github/workflows/unified-unit.yaml",
    ".github/workflows/marin-integration.yaml",
)
MAX_RUNS_PER_WORKFLOW = 5
MAX_CANDIDATES = 8
MAX_CANDIDATES_PER_FILE = 2
MIN_FAILURE_RUNS = 2
MIN_SLOW_SECONDS = 60.0
# Require a test to land in the per-workflow `--durations` slow window in at least
# this many distinct runs before treating it as actionable. Single-observation
# slow hits are dominated by JIT warm-up cost and cold imports; they are not
# evidence of a real perf regression.
MIN_SLOW_RUNS = 2

# Suppress Claude Code's default "Co-Authored-By: Claude" / "Generated with
# Claude Code" trailers on the commits and PRs the agent creates. AGENTS.md
# forbids self-credit, and a prose instruction alone does not reliably override
# the harness default — this setting does.
NO_SELF_CREDIT_SETTINGS = ("--settings", '{"attribution":{"commit":"","pr":""}}')

DURATION_RE = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)s\s+(?:setup|call|teardown)\s+(?P<test>\S+::.+)$")
FAILURE_RE = re.compile(r"(?:FAILED|ERROR)\s+(?P<test>\S+::.+?)(?:\s+-\s|$)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Every job runs pytest against a single package, whose pyproject.toml becomes pytest's
# rootdir, so they all report node ids as `tests/...` no matter where the suite lives.
# The job is what tells them apart, and its name is the name of its log file.
JOB_TEST_PREFIXES = (
    ("marin-rigging", "lib/rigging/"),
    ("marin-haliax", "lib/haliax/"),
    ("marin-iris", "lib/iris/"),
    ("marin-fray", "lib/fray/"),
    ("marin-levanter", "lib/levanter/"),
    ("marin-zephyr", "lib/zephyr/"),
    ("levanter-torch", "lib/levanter/"),
    ("levanter-tpu", "lib/levanter/"),
    ("iris-e2e-smoke", "lib/iris/"),
)


class GitHubApiError(RuntimeError):
    """Raised when the GitHub REST API returns an error."""


def strip_ansi(text: str) -> str:
    """Remove terminal escape sequences from a log line."""
    return ANSI_RE.sub("", text)


def parse_duration_line(line: str) -> tuple[str, float] | None:
    """Extract a pytest duration record from one log line."""
    match = DURATION_RE.search(strip_ansi(line))
    if match is None:
        return None
    return match.group("test"), float(match.group("seconds"))


def parse_failure_line(line: str) -> str | None:
    """Extract a pytest failure test id from one log line."""
    match = FAILURE_RE.search(strip_ansi(line))
    if match is None:
        return None
    return match.group("test")


def subproject_prefix(log_name: str) -> str:
    """Repo-relative prefix that turns one job's rootdir-relative node ids into repo paths."""
    for job, prefix in JOB_TEST_PREFIXES:
        if job in log_name:
            return prefix
    return ""


def canonicalize_test_name(test_name: str, prefix: str = "") -> str:
    """Normalize test ids across jobs so dedupe and aggregation are stable."""
    test_name = test_name.strip()
    if "::" not in test_name:
        return test_name
    file_path, sep, remainder = test_name.partition("::")
    if prefix and file_path.startswith("tests/"):
        file_path = f"{prefix}{file_path}"
    return f"{file_path}{sep}{remainder}"


def github_api(
    repo: str,
    path: str,
    *,
    token: str,
    accept: str = "application/vnd.github+json",
    parse_json: bool = True,
) -> Any:
    """Call the GitHub REST API."""
    url = f"https://api.github.com/repos/{repo}{path}"
    req = request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with request.urlopen(req) as response:
            payload = response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubApiError(f"GitHub API request failed for {path}: {exc.code} {detail}") from exc
    if not parse_json:
        return payload
    return json.loads(payload)


def repo_root() -> Path:
    """Return the git repository root."""
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def checkout_branch(root: Path, branch_name: str) -> None:
    """Reset a local branch to origin/main."""
    subprocess.run(["git", "fetch", "origin", "main"], check=True, cwd=root, capture_output=True)
    subprocess.run(["git", "checkout", "-B", branch_name, "origin/main"], check=True, cwd=root)


def workflow_runs(repo: str, token: str, workflow_file: str) -> list[dict[str, Any]]:
    """Fetch recent completed push runs for one workflow file."""
    encoded = parse.quote(workflow_file, safe="")
    payload = github_api(
        repo,
        (
            f"/actions/workflows/{encoded}/runs"
            f"?per_page={MAX_RUNS_PER_WORKFLOW}&branch=main&status=completed&event=push"
        ),
        token=token,
    )
    return list(payload.get("workflow_runs", []))


def download_logs(repo: str, token: str, run_id: int, destination: Path) -> Path:
    """Download and extract one GitHub Actions log archive."""
    payload = github_api(
        repo,
        f"/actions/runs/{run_id}/logs",
        token=token,
        accept="application/vnd.github+json",
        parse_json=False,
    )
    zip_path = destination / f"{run_id}.zip"
    zip_path.write_bytes(payload)
    extract_dir = destination / str(run_id)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def test_file_key(test_name: str) -> str:
    """Return the file portion of a canonicalized pytest node id."""
    return canonicalize_test_name(test_name).partition("::")[0]


def collect_evidence(log_dir: Path, workflow_name: str, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse pytest slow/failure evidence from one extracted log archive."""
    evidence: dict[str, dict[str, Any]] = {}
    run_id = int(run["id"])
    run_url = str(run["html_url"])
    seen_file_hashes: set[str] = set()
    seen_slow_tests: set[str] = set()
    seen_failed_tests: set[str] = set()
    for log_file in sorted(log_dir.rglob("*.txt")):
        log_text = log_file.read_text(errors="replace")
        log_hash = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
        if log_hash in seen_file_hashes:
            continue
        seen_file_hashes.add(log_hash)
        prefix = subproject_prefix(str(log_file))
        for line_number, raw_line in enumerate(log_text.splitlines(), start=1):
            duration = parse_duration_line(raw_line)
            if duration is not None:
                name, seconds = duration
                key = canonicalize_test_name(name, prefix)
                slow_observation_key = f"{run_id}:{key}"
                record = evidence.setdefault(
                    key,
                    {
                        "test": key,
                        "workflows": set(),
                        "slow_hits": 0,
                        "max_seconds": 0.0,
                        "failure_runs": set(),
                        "slow_examples": [],
                        "failure_examples": [],
                        "run_ids": set(),
                    },
                )
                record["workflows"].add(workflow_name)
                record["run_ids"].add(run_id)
                record["max_seconds"] = max(record["max_seconds"], seconds)
                if slow_observation_key not in seen_slow_tests:
                    seen_slow_tests.add(slow_observation_key)
                    record["slow_hits"] += 1
                    record["slow_examples"].append(
                        {
                            "seconds": seconds,
                            "log_file": str(log_file.relative_to(log_dir)),
                            "line": line_number,
                            "run_url": run_url,
                        }
                    )

            failure = parse_failure_line(raw_line)
            if failure is None:
                continue
            key = canonicalize_test_name(failure, prefix)
            failure_observation_key = f"{run_id}:{key}"
            record = evidence.setdefault(
                key,
                {
                    "test": key,
                    "workflows": set(),
                    "slow_hits": 0,
                    "max_seconds": 0.0,
                    "failure_runs": set(),
                    "slow_examples": [],
                    "failure_examples": [],
                    "run_ids": set(),
                },
            )
            record["workflows"].add(workflow_name)
            record["run_ids"].add(run_id)
            if failure_observation_key not in seen_failed_tests:
                seen_failed_tests.add(failure_observation_key)
                record["failure_runs"].add(run_id)
                record["failure_examples"].append(
                    {
                        "log_file": str(log_file.relative_to(log_dir)),
                        "line": line_number,
                        "run_url": run_url,
                    }
                )
    return evidence


def merge_evidence(dest: dict[str, dict[str, Any]], src: dict[str, dict[str, Any]]) -> None:
    """Merge parsed evidence from one run into the global accumulator."""
    for key, incoming in src.items():
        current = dest.setdefault(
            key,
            {
                "test": key,
                "workflows": set(),
                "slow_hits": 0,
                "max_seconds": 0.0,
                "failure_runs": set(),
                "slow_examples": [],
                "failure_examples": [],
                "run_ids": set(),
            },
        )
        current["workflows"].update(incoming["workflows"])
        current["slow_hits"] += incoming["slow_hits"]
        current["max_seconds"] = max(current["max_seconds"], incoming["max_seconds"])
        current["failure_runs"].update(incoming["failure_runs"])
        current["slow_examples"].extend(incoming["slow_examples"])
        current["failure_examples"].extend(incoming["failure_examples"])
        current["run_ids"].update(incoming["run_ids"])


def candidate_kind(record: dict[str, Any]) -> list[str]:
    """Classify a test candidate from the aggregated evidence."""
    kinds: list[str] = []
    if record["max_seconds"] >= MIN_SLOW_SECONDS and len(record["run_ids"]) >= MIN_SLOW_RUNS:
        kinds.append("slow")
    if len(record["failure_runs"]) >= MIN_FAILURE_RUNS:
        kinds.append("unstable")
    return kinds


def ranked_candidates(evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Select and rank actionable test candidates."""
    candidates: list[dict[str, Any]] = []
    for record in evidence.values():
        kinds = candidate_kind(record)
        if not kinds:
            continue
        candidates.append(
            {
                "test": record["test"],
                "kinds": kinds,
                "workflows": sorted(record["workflows"]),
                "max_seconds": round(record["max_seconds"], 2),
                "slow_hits": record["slow_hits"],
                "failure_run_count": len(record["failure_runs"]),
                "run_count": len(record["run_ids"]),
                "slow_examples": record["slow_examples"][:3],
                "failure_examples": record["failure_examples"][:3],
            }
        )
    candidates.sort(
        key=lambda item: (
            item["failure_run_count"],
            item["max_seconds"],
            item["slow_hits"],
            item["run_count"],
            item["test"],
        ),
        reverse=True,
    )
    return candidates


def select_candidates(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diversify selected candidates across test files."""
    selected: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}
    for candidate in ranked:
        file_key = test_file_key(candidate["test"])
        if per_file_counts.get(file_key, 0) >= MAX_CANDIDATES_PER_FILE:
            continue
        selected.append(candidate)
        per_file_counts[file_key] = per_file_counts.get(file_key, 0) + 1
        if len(selected) >= MAX_CANDIDATES:
            break
    return selected


def build_prompt(
    *,
    date: str,
    haiku_seed: str,
    candidate_file: Path,
    log_root: Path,
    repo: str,
    candidates: list[dict[str, Any]],
) -> str:
    """Build the agent prompt for one CI-test audit run."""
    return f"""\
You are the Nightshift CI Test Audit agent.

Your random seed is: {haiku_seed}
Use this seed to compose a haiku about test maintenance. Include it as the
epigraph of any PR you create.

## Context

You are working in `{repo}` on {date}. A wrapper already inspected recent CI log
archives on `main` and aggregated candidate tests.

Candidate summary JSON: `{candidate_file}`
Downloaded logs root: `{log_root}`

Candidates:
{json.dumps(candidates, indent=2)}

## Mission

Pick the highest-leverage candidate or small coherent pair of candidates and
determine:
1. Should this test exist?
2. If yes, can it be made faster and/or less flaky without weakening coverage?
3. If no, should it be removed, moved out of CI, or replaced with a better test?

Treat `unstable` as a hypothesis from log evidence, not a proven flake. Confirm
against the code and test intent before changing behavior.

Read `AGENTS.md` (especially its Testing section) and
`.agents/skills/commit/SKILL.md` for project conventions, and follow them.

## Rules of Engagement

- The only artifact you may produce is a PR with a real code change. Do NOT
  open standalone GitHub issues from this workflow — if no concrete fix is
  justified, exit cleanly. The same candidates will surface again on future
  runs, and that is fine.
- Prefer a focused in-repo improvement when the fix is straightforward and
  low-risk. Examples of acceptable fixes: removing a redundant compile,
  hoisting a shared fixture, deleting a parametrize cell whose coverage is
  already provided by a sibling, replacing a real-network setup with a
  pre-recorded fixture.
- Do not weaken assertions or mark a useful test `slow` just to hide a problem.
- Do not remove a test unless you can defend why its coverage is redundant,
  invalid, or better expressed elsewhere.
- Never write tautological, trivial, or "slop" tests. A test must fail when the
  behavior is wrong, not merely when the implementation changes. Do not add a
  test for a thin wrapper around a library call, for a one-off script, or just
  to have a test. If a change does not warrant a meaningful test, add none.
- Never credit yourself. Do not add a `Co-Authored-By: Claude` or "Generated
  with Claude Code" trailer to commits, and do not self-attribute in the PR
  description.
- If you modify code or tests, run `./infra/pre-commit.py --all-files --fix`
  and run the relevant `uv run pytest ...` targets.

## Output

- If you make code changes:
  1. Create or use branch `nightshift/ci-tests-{date.replace('-', '')}`.
  2. Push and open a PR titled `[nightshift] investigate slow/flaky CI tests`.
  3. Add labels `agent-generated` and `nightshift`.
  4. Begin the PR body with your haiku.
  5. Enable automerge with squash.
- Otherwise, exit cleanly and explain in plain text why no fix was justified.
  Do not open an issue. Do not create an empty PR. Do not edit unrelated files.
"""


def run_agent(prompt: str, root: Path) -> None:
    """Invoke Claude Code with the generated prompt."""
    result = run_claude(
        prompt,
        [
            "--model=opus",
            "--dangerously-skip-permissions",
            *NO_SELF_CREDIT_SETTINGS,
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "400",
        ],
        cwd=root,
    )
    if result.status == ClaudeRunStatus.RATE_LIMITED:
        report_rate_limit()
        return
    logger.info("%s", result.output)


def infer_repo() -> str:
    """Resolve the GitHub repository slug."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    remote = remote.removesuffix(".git")
    if remote.startswith("git@github.com:"):
        return remote.split(":", maxsplit=1)[1]
    if "github.com/" in remote:
        return remote.split("github.com/", maxsplit=1)[1]
    raise RuntimeError("Unable to infer GitHub repository; set GITHUB_REPOSITORY.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ["GH_TOKEN"]
    repo = infer_repo()
    today = dt.date.today()
    date = today.isoformat()
    root = repo_root()

    combined_evidence: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="nightshift-ci-tests-") as temp_dir:
        temp_root = Path(temp_dir)
        logs_root = temp_root / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        for workflow_file in WORKFLOWS:
            runs = workflow_runs(repo, token, workflow_file)
            logger.info("Inspecting %d runs for %s", len(runs), workflow_file)
            for run in runs:
                if run.get("conclusion") == "cancelled":
                    continue
                run_id = int(run["id"])
                workflow_name = str(run["name"])
                try:
                    extracted = download_logs(repo, token, run_id, logs_root)
                except (GitHubApiError, zipfile.BadZipFile) as exc:
                    logger.warning("Skipping run %s for %s: %s", run_id, workflow_file, exc)
                    continue
                merge_evidence(combined_evidence, collect_evidence(extracted, workflow_name, run))

        candidates = select_candidates(ranked_candidates(combined_evidence))

        if not candidates:
            logger.info("No actionable CI test candidates. Exiting cleanly.")
            return

        checkout_branch(root, f"nightshift/ci-tests-{today.strftime('%Y%m%d')}")
        candidate_file = temp_root / "candidates.json"
        candidate_file.write_text(
            json.dumps(
                {
                    "generated_at": dt.datetime.now(dt.UTC).isoformat(),
                    "repo": repo,
                    "candidates": candidates,
                },
                indent=2,
            )
        )

        prompt = build_prompt(
            date=date,
            haiku_seed=secrets.token_hex(4),
            candidate_file=candidate_file,
            log_root=logs_root,
            repo=repo,
            candidates=candidates,
        )
        run_agent(prompt, root)


if __name__ == "__main__":
    main()
