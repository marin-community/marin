# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift CI test audit: inspect recent CI logs for slow or unstable tests."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from typing import Any
from urllib import error, parse, request
import zipfile
import hashlib

logger = logging.getLogger(__name__)

WORKFLOWS = (
    ".github/workflows/marin-unit-tests.yaml",
    ".github/workflows/iris-unit-tests.yaml",
    ".github/workflows/levanter-tests.yaml",
    ".github/workflows/fray-unit-tests.yaml",
    ".github/workflows/zephyr-unit-tests.yaml",
    ".github/workflows/marin-itest.yaml",
)
MAX_RUNS_PER_WORKFLOW = 5
MAX_CANDIDATES = 8
MAX_CANDIDATES_PER_FILE = 2
MIN_FAILURE_RUNS = 2
MIN_SLOW_SECONDS = 8.0
COOLDOWN_DAYS = 30

DURATION_RE = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)s\s+(?:setup|call|teardown)\s+(?P<test>[^\s].+::[^\s]+)")
FAILURE_RE = re.compile(r"(?:FAILED|ERROR)\s+(?P<test>[\w./-]+::[^\s\[]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TEST_MARKER_RE = re.compile(r"<!--\s*nightshift-ci-test:\s*(?P<key>.+?)\s*-->")
COOLDOWN_MARKER_RE = re.compile(r"<!--\s*nightshift-ci-cooldown-until:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*-->")
TEST_PATH_PREFIX_RE = re.compile(r"^(?:lib/[^/]+/)?tests/")


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


def canonicalize_test_name(test_name: str) -> str:
    """Normalize test ids across workflows so dedupe and aggregation are stable."""
    test_name = test_name.strip()
    if "::" not in test_name:
        return test_name
    file_path, sep, remainder = test_name.partition("::")
    normalized_path = TEST_PATH_PREFIX_RE.sub("tests/", file_path)
    return f"{normalized_path}{sep}{remainder}"


def parse_iso_date(value: str | None) -> dt.date | None:
    """Parse an ISO date string if present."""
    if not value:
        return None
    return dt.date.fromisoformat(value)


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


def paginated_issues(repo: str, token: str, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    """Fetch all issues and PRs with the requested labels."""
    page = 1
    issues: list[dict[str, Any]] = []
    encoded_labels = parse.quote(",".join(labels))
    while True:
        payload = github_api(
            repo,
            f"/issues?state=all&per_page=100&page={page}&labels={encoded_labels}&sort=updated&direction=desc",
            token=token,
        )
        if not payload:
            return issues
        issues.extend(payload)
        page += 1


def collect_cooldowns(items: list[dict[str, Any]]) -> dict[str, tuple[dt.date | None, str, str]]:
    """Map test ids to cooldowns from prior Nightshift issues/PRs."""
    cooldowns: dict[str, tuple[dt.date | None, str, str]] = {}
    for item in items:
        body = item.get("body") or ""
        keys = TEST_MARKER_RE.findall(body)
        if not keys:
            continue
        cooldown_match = COOLDOWN_MARKER_RE.search(body)
        cooldown_until = parse_iso_date(cooldown_match.group("date")) if cooldown_match else None
        if item.get("state") == "open":
            cooldown_until = dt.date.max
        elif cooldown_until is None and item.get("closed_at"):
            closed_date = dt.date.fromisoformat(str(item["closed_at"])[:10])
            cooldown_until = closed_date + dt.timedelta(days=COOLDOWN_DAYS)

        artifact_url = str(item.get("html_url", ""))
        artifact_title = str(item.get("title", ""))
        for key in keys:
            existing = cooldowns.get(key)
            if existing is None or (existing[0] or dt.date.min) < (cooldown_until or dt.date.min):
                cooldowns[key] = (cooldown_until, artifact_url, artifact_title)
    return cooldowns


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


def test_key(test_name: str) -> str:
    """Normalize test ids used for cooldown markers and aggregation."""
    return canonicalize_test_name(test_name)


def test_file_key(test_name: str) -> str:
    """Return the file portion of a canonicalized pytest node id."""
    return test_key(test_name).partition("::")[0]


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
        for line_number, raw_line in enumerate(log_text.splitlines(), start=1):
            duration = parse_duration_line(raw_line)
            if duration is not None:
                name, seconds = duration
                key = test_key(name)
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
            key = test_key(failure)
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
    if record["max_seconds"] >= MIN_SLOW_SECONDS:
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


def select_candidates(
    ranked: list[dict[str, Any]],
    cooldowns: dict[str, tuple[dt.date | None, str, str]],
    today: dt.date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter cooldowns before truncation and diversify across test files."""
    fresh_ranked, skipped = filter_recent_candidates(ranked, cooldowns, today)
    selected: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}

    for candidate in fresh_ranked:
        file_key = test_file_key(candidate["test"])
        if per_file_counts.get(file_key, 0) >= MAX_CANDIDATES_PER_FILE:
            continue
        selected.append(candidate)
        per_file_counts[file_key] = per_file_counts.get(file_key, 0) + 1
        if len(selected) >= MAX_CANDIDATES:
            break

    return selected, skipped


def filter_recent_candidates(
    candidates: list[dict[str, Any]],
    cooldowns: dict[str, tuple[dt.date | None, str, str]],
    today: dt.date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into fresh and skipped-by-cooldown sets."""
    fresh: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in candidates:
        cooldown = cooldowns.get(candidate["test"])
        if cooldown is None:
            fresh.append(candidate)
            continue
        until, artifact_url, artifact_title = cooldown
        if until is not None and until >= today:
            skipped.append(
                {
                    "test": candidate["test"],
                    "cooldown_until": until.isoformat(),
                    "artifact_url": artifact_url,
                    "artifact_title": artifact_title,
                }
            )
            continue
        fresh.append(candidate)
    return fresh, skipped


def build_prompt(
    *,
    date: str,
    haiku_seed: str,
    candidate_file: Path,
    log_root: Path,
    repo: str,
    fresh_candidates: list[dict[str, Any]],
    skipped_candidates: list[dict[str, Any]],
) -> str:
    """Build the agent prompt for one CI-test audit run."""
    return f"""\
You are the Nightshift CI Test Audit agent.

Your random seed is: {haiku_seed}
Use this seed to compose a haiku about test maintenance. Include it as the
epigraph of any PR or issue you create.

## Context

You are working in `{repo}` on {date}. A wrapper already inspected recent CI log
archives on `main`, aggregated candidate tests, and filtered out tests that
already have an open or recently closed Nightshift investigation.

Candidate summary JSON: `{candidate_file}`
Downloaded logs root: `{log_root}`

Fresh candidates:
{json.dumps(fresh_candidates, indent=2)}

Skipped due to active/recent investigations:
{json.dumps(skipped_candidates, indent=2)}

## Mission

Pick the highest-leverage candidate or small coherent pair of candidates and
determine:
1. Should this test exist?
2. If yes, can it be made faster and/or less flaky without weakening coverage?
3. If no, should it be removed, moved out of CI, or replaced with a better test?

Treat `unstable` as a hypothesis from log evidence, not a proven flake. Confirm
against the code and test intent before changing behavior.

Read `AGENTS.md` for project conventions.

## Rules of Engagement

- Prefer a focused in-repo improvement over opening a new issue when the fix is
  straightforward and low-risk.
- Do not weaken assertions or mark a useful test `slow` just to hide a problem.
- Do not remove a test unless you can defend why its coverage is redundant,
  invalid, or better expressed elsewhere.
- If you modify code or tests, run `./infra/pre-commit.py --all-files --fix`
  and run the relevant `uv run pytest ...` targets.
- If the investigation is real but the fix is too large or risky, open a GitHub
  issue instead of forcing a partial change.
- Do not open duplicate artifacts for tests listed in the skipped section.

## Required dedupe markers

If you open a PR or issue, include these hidden markers in the body for every
test you investigated:

<!-- nightshift-ci-test: path/to/test_file.py::test_name -->
<!-- nightshift-ci-cooldown-until: YYYY-MM-DD -->

Use a cooldown of at least 30 days from today so future Nightshift runs do not
re-investigate the same test immediately.

## Output

- If you make code changes:
  1. Create or use branch `nightshift/ci-tests-{date.replace('-', '')}`.
  2. Push and open a PR titled `[nightshift] investigate slow/flaky CI tests`.
  3. Add labels `agent-generated` and `nightshift`.
  4. Begin the PR body with your haiku and include the required hidden markers.
  5. Enable automerge with squash.
- If code changes are not justified but follow-up is:
  1. Open one GitHub issue titled `[nightshift] investigate CI test performance/stability`.
  2. Add labels `agent-generated` and `nightshift`.
  3. Begin the issue body with your haiku and include the required hidden markers.
- If no justified action remains after inspection, exit cleanly and explain why.
"""


def run_agent(prompt: str, root: Path) -> None:
    """Invoke Claude Code with the generated prompt."""
    subprocess.run(
        [
            "claude",
            "--model=opus",
            "--print",
            "--dangerously-skip-permissions",
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "400",
            "--",
            prompt,
        ],
        check=True,
        cwd=root,
    )


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

    prior_items = paginated_issues(repo, token, ("nightshift", "agent-generated"))
    cooldowns = collect_cooldowns(prior_items)
    logger.info("Loaded %d prior nightshift cooldown markers", len(cooldowns))

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

        candidates = ranked_candidates(combined_evidence)
        fresh_candidates, skipped_candidates = select_candidates(candidates, cooldowns, today)

        if not fresh_candidates:
            logger.info("No fresh CI test candidates after cooldown filtering. Exiting cleanly.")
            return

        checkout_branch(root, f"nightshift/ci-tests-{today.strftime('%Y%m%d')}")
        candidate_file = temp_root / "candidates.json"
        candidate_file.write_text(
            json.dumps(
                {
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "repo": repo,
                    "candidates": fresh_candidates,
                    "skipped": skipped_candidates,
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
            fresh_candidates=fresh_candidates,
            skipped_candidates=skipped_candidates,
        )
        run_agent(prompt, root)


if __name__ == "__main__":
    main()
