# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Workflow inventory + policy auditor for Marin's .github/workflows/.

Subcommands:
- audit: parse workflows and check naming, extension, SHA pinning, and required-context coverage.
- required-contexts: fetch branch-protection required status checks via `gh api`.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

# Actions that may be referenced by semver tag rather than SHA.
TRUSTED_TAG_PINNED_ACTIONS = frozenset(
    [
        "actions/checkout",
        "actions/setup-python",
        "actions/setup-node",
        "actions/cache",
        "actions/upload-artifact",
        "actions/download-artifact",
        "actions/create-github-app-token",
        "astral-sh/setup-uv",
        "google-github-actions/auth",
        "google-github-actions/setup-gcloud",
        "docker/setup-buildx-action",
        "docker/login-action",
        "docker/build-push-action",
        "github/codeql-action/init",
        "github/codeql-action/analyze",
    ]
)

ALLOWED_DOMAINS = frozenset(
    [
        "Iris",
        "Zephyr",
        "Marin",
        "Levanter",
        "Haliax",
        "Fray",
        "Dupekit",
        "Ops",
    ]
)

# A 40-character hex string is an unambiguous commit SHA.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class WorkflowJob:
    job_id: str
    job_name: str | None
    matrix_context: str | None  # short string describing matrix shape, e.g. "os, python" or None
    required_context: str | None  # the status context name as it appears in branch protection, or None


@dataclass(frozen=True)
class WorkflowRecord:
    path: Path
    workflow_name: str  # the top-level `name:` field, or path stem if missing
    jobs: tuple[WorkflowJob, ...]
    third_party_actions: tuple[str, ...]  # full action refs like "peaceiris/actions-gh-pages@v4"


def _matrix_context(strategy: dict) -> str | None:
    """Return a comma-separated key list for a matrix strategy, or None if absent."""
    matrix = strategy.get("matrix") if strategy else None
    if not matrix:
        return None
    # Skip special keys like 'include' and 'exclude' — they describe overrides, not axes.
    keys = [k for k in matrix if k not in ("include", "exclude")]
    return ", ".join(keys) if keys else None


def _collect_action_refs(steps: list) -> list[str]:
    """Extract all `uses: action@ref` strings from a job's steps."""
    refs = []
    for step in steps or []:
        uses = step.get("uses") if isinstance(step, dict) else None
        if uses and "@" in uses:
            refs.append(uses)
    return refs


def _parse_job(job_id: str, job_data: dict) -> WorkflowJob:
    strategy = job_data.get("strategy")
    matrix_ctx = _matrix_context(strategy) if strategy else None
    return WorkflowJob(
        job_id=job_id,
        job_name=job_data.get("name"),
        matrix_context=matrix_ctx,
        required_context=None,  # populated externally when branch-protection data is available
    )


def workflow_records(workflows_dir: Path) -> tuple[WorkflowRecord, ...]:
    """Parse workflow YAML files and return inventory.

    Reads all files ending in .yaml or .yml inside workflows_dir (non-recursive).
    """
    records = []
    for workflow_path in sorted(workflows_dir.iterdir()):
        if workflow_path.suffix not in (".yaml", ".yml"):
            continue
        raw = yaml.safe_load(workflow_path.read_text())
        if not isinstance(raw, dict):
            continue

        workflow_name = raw.get("name") or workflow_path.stem
        jobs_data = raw.get("jobs") or {}

        jobs = []
        third_party: list[str] = []
        for job_id, job_data in jobs_data.items():
            if not isinstance(job_data, dict):
                continue
            jobs.append(_parse_job(job_id, job_data))
            steps = job_data.get("steps") or []
            third_party.extend(_collect_action_refs(steps))

        records.append(
            WorkflowRecord(
                path=workflow_path,
                workflow_name=workflow_name,
                jobs=tuple(jobs),
                third_party_actions=tuple(third_party),
            )
        )
    return tuple(records)


def required_status_contexts(repo: str, branch: str) -> tuple[str, ...]:
    """Return branch-protection required status contexts via `gh api`.

    Tries the newer `checks` shape first, falls back to the older `contexts` list.
    """
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/branches/{branch}/protection"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    checks_data = data.get("required_status_checks", {})

    # Newer API returns a list of {context, app_id} objects under "checks".
    checks_list = checks_data.get("checks")
    if checks_list is not None:
        return tuple(chk["context"] for chk in checks_list)

    # Older API returns a plain list of strings under "contexts".
    return tuple(checks_data.get("contexts", []))


# ---------------------------------------------------------------------------
# Policy checks — pure functions that return lists of failure strings
# ---------------------------------------------------------------------------


def _check_file_extensions(records: tuple[WorkflowRecord, ...]) -> list[str]:
    failures = []
    for rec in records:
        if rec.path.suffix != ".yaml":
            failures.append(f"{rec.path.name}: file extension is '{rec.path.suffix}'; expected '.yaml'")
    return failures


def _check_workflow_names(records: tuple[WorkflowRecord, ...]) -> list[str]:
    failures = []
    for rec in records:
        name = rec.workflow_name
        has_valid_domain = any(name.startswith(f"{domain} - ") for domain in ALLOWED_DOMAINS)
        if not has_valid_domain:
            failures.append(
                f"{rec.path.name}: workflow name '{name}' does not start with '<Domain> - '"
                f" (allowed: {', '.join(sorted(ALLOWED_DOMAINS))})"
            )
    return failures


def _check_job_ids(records: tuple[WorkflowRecord, ...]) -> list[str]:
    failures = []
    for rec in records:
        for job in rec.jobs:
            if not _KEBAB_RE.match(job.job_id):
                failures.append(
                    f"{rec.path.name}: job id '{job.job_id}' is not lowercase kebab-case"
                    f" (must match ^[a-z][a-z0-9-]*$)"
                )
    return failures


def _is_sha_pinned(ref: str) -> bool:
    return bool(_SHA_RE.match(ref))


def _check_action_pinning(records: tuple[WorkflowRecord, ...]) -> list[str]:
    """Verify non-trusted third-party actions are pinned to a 40-char hex SHA."""
    failures = []
    for rec in records:
        for action_ref in rec.third_party_actions:
            action_path, _, ref = action_ref.partition("@")
            if action_path in TRUSTED_TAG_PINNED_ACTIONS:
                continue
            if not _is_sha_pinned(ref):
                failures.append(
                    f"{rec.path.name}: action '{action_ref}' is not trusted-tag-pinned"
                    f" and not SHA-pinned (40-char hex ref required)"
                )
    return failures


def _check_required_contexts(
    records: tuple[WorkflowRecord, ...],
    contexts: tuple[str, ...],
) -> list[str]:
    """Report required branch-protection contexts that have no matching workflow job."""
    # Build a set of all (job_id, job_name) pairs across all workflows.
    known_ids: set[str] = set()
    known_names: set[str] = set()
    for rec in records:
        for job in rec.jobs:
            known_ids.add(job.job_id)
            if job.job_name:
                known_names.add(job.job_name)

    failures = []
    for ctx in contexts:
        if ctx not in known_ids and ctx not in known_names:
            failures.append(f"required context '{ctx}' has no matching job id or job name in any workflow")
    return failures


def policy_failures(
    records: tuple[WorkflowRecord, ...],
    required_contexts: tuple[str, ...] | None = None,
) -> list[str]:
    """Return all policy failures across the given workflow records.

    Args:
        records: Parsed workflow records from workflow_records().
        required_contexts: Optional branch-protection required status contexts.
            When provided, check #5 (context coverage) is also run.

    Returns:
        A list of human-readable failure strings. Empty list means clean audit.
    """
    failures: list[str] = []
    failures.extend(_check_file_extensions(records))
    failures.extend(_check_workflow_names(records))
    failures.extend(_check_job_ids(records))
    failures.extend(_check_action_pinning(records))
    if required_contexts is not None:
        failures.extend(_check_required_contexts(records, required_contexts))
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Workflow inventory and policy auditor for Marin's .github/workflows/."""


@cli.command("audit")
@click.option(
    "--workflows-dir",
    default=".github/workflows",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing GitHub Actions workflow files.",
)
@click.option(
    "--repo",
    default=None,
    metavar="OWNER/REPO",
    help="GitHub repo (e.g. marin-community/marin). When set, fetches required status" " contexts and checks coverage.",
)
@click.option("--branch", default="main", show_default=True, help="Branch to inspect.")
def audit_command(workflows_dir: Path, repo: str | None, branch: str) -> None:
    """Parse workflows and check naming, extension, SHA pinning, and context coverage."""
    records = workflow_records(workflows_dir)

    contexts: tuple[str, ...] | None = None
    if repo:
        contexts = required_status_contexts(repo, branch)

    failures = policy_failures(records, required_contexts=contexts)

    # Summary always goes to stdout.
    total_workflows = len(records)
    total_jobs = sum(len(r.jobs) for r in records)
    click.echo(f"Audited {total_workflows} workflow(s), {total_jobs} job(s)." f" Found {len(failures)} failure(s).")

    if failures:
        for msg in failures:
            click.echo(f"FAIL: {msg}", err=True)
        sys.exit(1)


@cli.command("required-contexts")
@click.option("--repo", required=True, metavar="OWNER/REPO", help="GitHub repo slug.")
@click.option("--branch", default="main", show_default=True, help="Branch to inspect.")
def required_contexts_command(repo: str, branch: str) -> None:
    """Print required branch-protection status contexts, one per line."""
    contexts = required_status_contexts(repo, branch)
    for ctx in contexts:
        click.echo(ctx)


if __name__ == "__main__":
    cli()
